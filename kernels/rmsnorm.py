"""
Cute-Snippets: RMSNorm / LayerNorm Example

RMSNorm/LayerNorm implementation using CuTe tiled primitives.

Refactored to use HTile's TileFusion-inspired abstractions:
    - ElementwiseTileConfig: Tile primitive configuration (threads, cluster, tiler)
    - ElementwiseKernelContext: Dataflow setup (partition, predicate, smem, cluster)

The RMSNorm class now focuses on the algorithm-specific dataflow:
    load x (+ residual) → reduce (sum_sq / mean+var) → normalize → apply weight/bias → store
"""

import sys
import torch
import cutlass
from cutlass import const_expr, Float32
import cutlass.cute as cute
from typing import Type, Optional, Tuple
import cuda.bindings.driver as cuda

from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from htile import (
    ElementwiseTileConfig,
    ElementwiseKernelContext,
    row_reduce,
    make_fake_tensor,
    torch2cute_dtype_map,
    expand,
)

import math


class RMSNorm:
    def __init__(
        self, dtype: Type[cutlass.Numeric], N: int, is_layernorm: bool
    ):
        self.dtype = dtype
        self.N = N
        self.is_layernorm = is_layernorm
        self.delay_w_load = False
        self.reload_from = (
            None if N <= (16384 if is_layernorm else 8192) else "smem"
        )

        stage = 2 if is_layernorm else 1

        # ---- TileFusion-style: Tile Primitive Configuration ----
        self.config = ElementwiseTileConfig(
            dtype=dtype,
            N=N,
            cluster_n=None,  # auto
            stage=stage,
        )

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: Optional[cute.Tensor],
        mB: Optional[cute.Tensor],
        mRes: Optional[cute.Tensor],
        mO: cute.Tensor,
        mResO: Optional[cute.Tensor],
        mRstd: Optional[cute.Tensor],
        mMean: Optional[cute.Tensor],
        eps: Float32,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype

        largest_dtype_width = const_expr(
            max(
                *(
                    t.element_type.width
                    for t in (mX, mW, mB, mRes, mO, mResO)
                    if t is not None
                )
            )
        )

        vec_size = math.gcd(self.N, 128 // largest_dtype_width)
        threads_per_row = self.config.threads_per_row()
        num_threads = self.config.num_threads()

        num_block_N = cute.ceil_div(
            self.N // vec_size, threads_per_row * self.config.cluster_n
        )
        tiled_copy = self.config.make_vector_copy(vec_size).tiled_copy_2d()
        tiler_mn = (
            num_threads // threads_per_row,
            vec_size * threads_per_row * num_block_N,
        )

        mW, mB = [
            (
                expand(mT, dim=0, size=tiler_mn[0])
                if const_expr(mT is not None)
                else None
            )
            for mT in (mW, mB)
        ]

        mRstd, mMean = [
            (
                expand(mT, dim=1, size=self.N)
                if const_expr(mT is not None)
                else None
            )
            for mT in (mRstd, mMean)
        ]

        self.kernel(
            mX,
            mW,
            mB,
            mRes,
            mO,
            mResO,
            mRstd,
            mMean,
            eps,
            tiler_mn,
            tiled_copy,
            threads_per_row,
        ).launch(
            grid=[
                cute.ceil_div(mX.shape[0], tiler_mn[0]),
                self.config.cluster_n,
                1,
            ],
            block=[num_threads, 1, 1],
            cluster=(
                [1, self.config.cluster_n, 1]
                if const_expr(self.config.cluster_n > 1)
                else None
            ),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: Optional[cute.Tensor],
        mB: Optional[cute.Tensor],
        mRes: Optional[cute.Tensor],
        mO: cute.Tensor,
        mResO: Optional[cute.Tensor],
        mRstd: Optional[cute.Tensor],
        mMean: Optional[cute.Tensor],
        eps: Float32,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        """
        Device: RMSNorm kernel using CuTe primitives.
        """
        # ---- TileFusion-style: Dataflow Setup (common infrastructure) ----
        # extra_smem_tensors=1 if residual is present (for sRes)
        has_res = const_expr(mRes is not None)
        ctx = ElementwiseKernelContext(self.config)
        ctx.setup(
            mX,
            tiler_mn,
            tiled_copy,
            extra_smem_tensors=1 if has_res else 0,
        )

        # ---- Partition additional tensors (RMSNorm-specific) ----
        # Residual smem tensor (if allocated)
        if const_expr(has_res):
            sRes = ctx.extra_sX[0]
            gRes = ctx.partition_global(mRes, tiler_mn)
            tXgRes = ctx.partition_source(gRes)
            tXsRes = ctx.partition_smem_dest(sRes)
            tXrRes = ctx.make_fragment(tXgRes)

        # Output
        gO = ctx.partition_global(mO, tiler_mn)
        tXgO = ctx.partition_dest(gO)
        tXrO = ctx.make_fragment(tXgO)

        # Residual output
        if const_expr(mResO is not None):
            gResO = ctx.partition_global(mResO, tiler_mn)
            tXgResO = ctx.partition_dest(gResO)

        # Rstd and Mean (partitioned as dest for writing)
        if const_expr(mRstd is not None):
            gRstd = ctx.partition_global(mRstd, tiler_mn)
            tXrRstd = ctx.partition_dest(gRstd)

        if const_expr(mMean is not None):
            gMean = ctx.partition_global(mMean, tiler_mn)
            tXrMean = ctx.partition_dest(gMean)

        # Weight and Bias (tiled at row 0, not bidx)
        if const_expr(mW is not None):
            gW = cute.local_tile(mW, tiler_mn, (0, ctx.cluster_y))
            tXgW = ctx.thr_copy.partition_S(gW)
            tXrW = ctx.make_fragment(tXgW)

        if const_expr(mB is not None):
            gB = cute.local_tile(mB, tiler_mn, (0, ctx.cluster_y))
            tXgB = ctx.thr_copy.partition_S(gB)
            tXrB = ctx.make_fragment(tXgB)

        # ---- TileFusion-style: Dataflow (algorithm-specific) ----

        vector_copy = ctx.vector_copy_fn
        row = ctx.row
        shape = ctx.shape

        # Load X (and Res) from global → shared (async)
        if row < shape[0]:
            vector_copy(ctx.tXgX, ctx.tXsX, is_async=True)
            if const_expr(has_res):
                vector_copy(tXgRes, tXsRes, is_async=True)

        cute.arch.cp_async_commit_group()

        # Optionally load weight/bias while waiting for async copy
        if const_expr(not self.delay_w_load):
            if const_expr(mW is not None):
                vector_copy(tXgW, tXrW)
            if const_expr(mB is not None):
                vector_copy(tXgB, tXrB)

        cute.arch.cp_async_wait_group(0)
        cute.autovec_copy(ctx.tXsX, ctx.tXrX)

        x = ctx.tXrX.load().to(cute.Float32)
        if const_expr(has_res):
            cute.autovec_copy(tXsRes, tXrRes)
            x += tXrRes.load().to(cute.Float32)

        # Store residual output if needed
        if const_expr(mResO is not None):
            tXrResO = ctx.make_fragment(tXgResO)
            tXrResO.store(x.to(tXrResO.element_type))
            if row < shape[0]:
                vector_copy(tXrResO, tXgResO)

        # ---- Reduction: compute mean and rstd ----
        mean, rstd = None, None

        if const_expr(self.is_layernorm):
            pass  # TODO: LayerNorm reduction (mean + variance)
        else:
            # RMSNorm: compute sum of squares directly
            mean = const_expr(0.0)
            sum_sq_x = row_reduce(
                x * x,
                cute.ReductionOp.ADD,
                threads_per_row,
                ctx.reduction_buffer[None, None, 0],
                ctx.mbar_ptr,
                init_val=0.0,
                hook_fn=(
                    cute.arch.cluster_wait
                    if const_expr(self.config.cluster_n > 1)
                    else None
                ),
            )
            rstd = cute.math.rsqrt(sum_sq_x / shape[1] + eps, fastmath=True)

        # Write rstd to global memory
        if const_expr(mRstd is not None):
            if (
                ctx.tXcX[0][1] == 0
                and row < shape[0]
                and (
                    self.config.cluster_n == 1
                    or cute.arch.block_idx_in_cluster() == 0
                )
            ):
                tXrRstd[0] = rstd

        # Delayed weight/bias load
        if const_expr(self.delay_w_load):
            if const_expr(mW is not None):
                vector_copy(tXgW, tXrW)
            if const_expr(mB is not None):
                vector_copy(tXgB, tXrB)

        # Reload from smem/gmem if needed (for large N)
        if const_expr(self.reload_from == "smem" or self.reload_from == "gmem"):
            if const_expr(self.reload_from == "smem"):
                cute.autovec_copy(ctx.tXsX, ctx.tXrX)
                if const_expr(has_res):
                    cute.autovec_copy(tXsRes, tXrRes)
            else:
                vector_copy(ctx.tXgX, ctx.tXrX)
                if const_expr(has_res):
                    vector_copy(tXgRes, tXrRes)

            x = ctx.tXrX.load().to(cute.Float32)
            if const_expr(has_res):
                x += tXrRes.load().to(cute.Float32)

        # ---- Normalize and apply weight/bias ----
        x_hat = (x - mean) * rstd if const_expr(self.is_layernorm) else x * rstd
        y = x_hat

        if const_expr(mW is not None):
            y *= tXrW.load().to(cute.Float32)
        if const_expr(mB is not None):
            y += tXrB.load().to(cute.Float32)

        # Store: R → G
        tXrO.store(y.to(tXrO.element_type))
        if row < shape[0]:
            vector_copy(tXrO, tXgO)


@torch.library.custom_op(
    "cute_snippets::_rmsnorm_fwd",
    mutates_args=["out", "rstd", "mean", "residual_out"],
    device_types="cuda",
    # We need to specify the schema manually since we're mutating an optional tensor
    schema="(Tensor x, Tensor? weight, Tensor(a2!) out, Tensor? bias, Tensor(a4!)? rstd, Tensor(a5!)? mean, Tensor? residual, Tensor(a7!)? residual_out, float eps=1e-6, bool is_layernorm=False) -> ()",
)
def _rmsnorm_fwd(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    out: torch.Tensor,
    bias: Optional[torch.Tensor],
    rstd: Optional[torch.Tensor],
    mean: Optional[torch.Tensor],
    residual: Optional[torch.Tensor],
    residual_out: Optional[torch.Tensor],
    eps: float = 1e-6,
    is_layernorm: bool = False,
) -> None:
    """
    RMSNorm/LayerNorm forward pass.
    Args:
        x: Input tensor of shape (batch_size, sequence_length, hidden_size)
        weight: Weight tensor of shape (hidden_size,)
        out: Output tensor of shape (batch_size, sequence_length, hidden_size)
        bias: Bias tensor of shape (hidden_size,)
        rstd: Running standard deviation tensor of shape (hidden_size,)
        mean: Running mean tensor of shape (hidden_size,)
        residual: Residual tensor of shape (batch_size, sequence_length, hidden_size)
        residual_out: Residual output tensor of shape (batch_size, sequence_length, hidden_size)
        eps: Epsilon value for numerical stability
        is_layernorm: Whether to use layer normalization
    Returns:
        Normalized output tensor of same shape as x
    """

    supported_types = {torch.float16, torch.bfloat16, torch.float32}
    assert (
        x.dtype in supported_types
    ), "x must be a float16, bfloat16, or float32 tensor"

    if weight is not None:
        assert weight.dtype == x.dtype, "weight must have the same dtype as x"
    if residual is not None:
        assert (
            residual.dtype == x.dtype
        ), "residual must have the same dtype as x"

    _, N = x.shape
    dtype, out_dtype, weight_dtype, bias_dtype, res_dtype, res_out_dtype = [
        torch2cute_dtype_map[t.dtype] if t is not None else None
        for t in (x, out, weight, bias, residual, residual_out)
    ]

    compile_key = (
        dtype,
        out_dtype,
        res_dtype,
        weight_dtype,
        bias_dtype,
        res_out_dtype,
        N,
        rstd is not None,
        mean is not None,
        is_layernorm,
    )

    if compile_key not in _rmsnorm_fwd.compile_cache:
        batch_sym = cute.sym_int()
        all_dtypes = [
            dtype,
            out_dtype,
            weight_dtype,
            bias_dtype,
            res_dtype,
            res_out_dtype,
        ]
        div = math.gcd(
            N, *(128 // dt.width for dt in all_dtypes if dt is not None)
        )

        x_cute, out_cute, res_cute, res_out_cute = [
            make_fake_tensor(dt, (batch_sym, N), div)
            for dt in [dtype, out_dtype, res_dtype, res_out_dtype]
        ]
        weight_cute, bias_cute = [
            make_fake_tensor(dt, (N,), div) for dt in [weight_dtype, bias_dtype]
        ]
        rstd_cute = (
            make_fake_tensor(Float32, (batch_sym,))
            if rstd is not None
            else None
        )
        mean_cute = (
            make_fake_tensor(Float32, (batch_sym,))
            if mean is not None
            else None
        )

        _rmsnorm_fwd.compile_cache[compile_key] = cute.compile(
            RMSNorm(dtype, N, is_layernorm),
            x_cute,
            weight_cute,
            bias_cute,
            res_cute,
            out_cute,
            res_out_cute,
            rstd_cute,
            mean_cute,
            Float32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _rmsnorm_fwd.compile_cache[compile_key](
        x, weight, bias, residual, out, residual_out, rstd, mean, eps
    )


_rmsnorm_fwd.compile_cache = {}


def rmsnorm_fwd(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    residual_dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
    store_rstd: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    # Need to wrap to handle the case where residual_out is a alias of x, which makes torch.library
    # and torch.compile unhappy. Also allocate memory for out and residual_out if they are None
    # so that _layer_norm_fwd_impl doesn't have to return them.
    out_dtype = x.dtype if out_dtype is None else out_dtype
    out = torch.empty_like(x, dtype=out_dtype)
    rstd = (
        torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
        if store_rstd
        else None
    )
    if residual is not None:
        residual_dtype = residual.dtype
    if residual is not None or (
        residual_dtype is not None and residual_dtype != x.dtype
    ):
        residual_out = torch.empty_like(
            x, dtype=residual_dtype if residual_dtype is not None else x.dtype
        )
    else:
        residual_out = None
    _rmsnorm_fwd(
        x, weight, out, bias, rstd, None, residual, residual_out, eps, False
    )
    # residual_out is None if residual is None and residual_dtype == input_dtype and dropout_p == 0.0
    if residual_out is None:
        residual_out = x
    return out, residual_out, rstd
