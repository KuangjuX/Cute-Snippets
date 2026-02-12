"""
Based on the QuACK implementation: https://github.com/Dao-AILab/quack/blob/main/quack/softmax.py

Cute-Snippets: Softmax Example

Softmax implementation using CuTe tiled primitives.
Computes softmax along the last dimension of the input tensor.

Refactored to use HTile's TileFusion-inspired abstractions:
    - ElementwiseTileConfig: Tile primitive configuration
    - ElementwiseKernelContext: Dataflow setup (partition, predicate, smem, cluster)
    - ElementwiseStorer: R→G data movement

The Softmax class now focuses purely on the algorithm-specific dataflow:
    load → max_reduce → exp → sum_reduce → normalize → store

"""

import math
import torch
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass import Float32, const_expr

from typing import Optional, Type

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import htile.copy as hcopy
from htile import (
    row_reduce,
    make_fake_tensor,
    torch2cute_dtype_map,
    ElementwiseTileConfig,
    ElementwiseKernelContext,
)


class Softmax:

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        N: int,
        cluster_n: Optional[int],
        online_softmax: bool = False,
    ):
        """
        Args:
            dtype: Data type for computation
            N: Size of the last dimension
            cluster_n: Number of blocks in cluster (1 for no cluster). Use None for auto.
            online_softmax: Whether to use online softmax algorithm
        """
        self.dtype = dtype
        self.N = N
        self.online_softmax = online_softmax

        stage = 2 if not online_softmax else 1

        # ---- TileFusion-style: Tile Primitive Configuration ----
        self.config = ElementwiseTileConfig(
            dtype=dtype,
            N=N,
            cluster_n=cluster_n,
            stage=stage,
        )

    @cute.jit
    def __call__(self, mX: cute.Tensor, mO: cute.Tensor, stream: cuda.CUstream):
        assert (
            mX.element_type == self.dtype
        ), f"Input tensor element type {mX.element_type} does not match dtype {self.dtype}"

        largest_dtype_width = const_expr(
            max(t.element_type.width for t in (mX, mO))
        )
        vec_size = 128 // largest_dtype_width
        threads_per_row = self.config.threads_per_row()
        num_threads = self.config.num_threads()

        num_block_N = cute.ceil_div(
            self.N // vec_size, threads_per_row * self.config.cluster_n
        )
        tiler_mn = (
            num_threads // threads_per_row,
            vec_size * threads_per_row * num_block_N,
        )

        tiled_copy = self.config.make_vector_copy(vec_size).tiled_copy_2d()

        self.kernel(mX, mO, tiler_mn, tiled_copy, threads_per_row).launch(
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
        mO: cute.Tensor,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        # ---- TileFusion-style: Dataflow Setup (common infrastructure) ----
        ctx = ElementwiseKernelContext(self.config)
        ctx.setup(mX, tiler_mn, tiled_copy)

        # Partition output tensor
        gO = ctx.partition_global(mO, tiler_mn)
        tXgO = ctx.partition_dest(gO)
        tXrO = ctx.make_fragment(tXgO)

        # ---- TileFusion-style: Dataflow (algorithm-specific) ----

        # Load: G → S → R
        ctx.load_to_smem_and_regs()

        # Fill OOB values with -inf for softmax correctness
        is_even_N = const_expr(
            ctx.shape[1] == tiler_mn[1] * self.config.cluster_n
        )
        if const_expr(not is_even_N):
            hcopy.fill_oob(ctx.tXsX, ctx.tXpX, -ctx.sX.element_type.inf)
            # Reload from smem after fill
            cute.autovec_copy(ctx.tXsX, ctx.tXrX)

        x = ctx.tXrX.load().to(cute.Float32)

        # Compute: max reduction
        max_x = row_reduce(
            x,
            cute.ReductionOp.MAX,
            threads_per_row,
            ctx.reduction_buffer[None, None, 0],
            ctx.mbar_ptr + 0 if const_expr(self.config.cluster_n > 1) else None,
            init_val=-Float32.inf,
            hook_fn=(
                cute.arch.cluster_wait
                if const_expr(self.config.cluster_n > 1)
                else None
            ),
        )

        # Compute: exp and sum reduction
        log2_e = math.log2(math.e)
        exp_x = cute.math.exp2(x * log2_e - (max_x * log2_e), fastmath=True)
        denom = row_reduce(
            exp_x,
            cute.ReductionOp.ADD,
            threads_per_row,
            ctx.reduction_buffer[None, None, 1],
            ctx.mbar_ptr + 1 if const_expr(self.config.cluster_n > 1) else None,
            init_val=0.0,
        )

        # Compute: normalize
        y = exp_x * cute.arch.rcp_approx(denom)

        # Store: R → G
        tXrO.store(y.to(tXrO.element_type))
        ctx.store_from_regs(tXrO, tXgO)


@torch.library.custom_op("cute_snippets::_softmax_fwd", mutates_args={"out"})
def _softmax_fwd(x: torch.Tensor, out: torch.Tensor) -> None:

    assert x.dim() == 2, "x must be a 2D tensor"
    assert x.is_cuda, "x must be a CUDA tensor"
    assert x.dtype in [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ], "x must be a float16, bfloat16, or float32 tensor"

    N = x.size(1)
    dtype, out_dtype = [torch2cute_dtype_map[t.dtype] for t in (x, out)]
    compile_key = (dtype, out_dtype, N)

    if compile_key not in _softmax_fwd.compile_cache:
        batch_sym = cute.sym_int()
        div = math.gcd(128 // dtype.width, N)
        x_cute, out_cute = [
            make_fake_tensor(dt, (batch_sym, N), div)
            for dt in [dtype, out_dtype]
        ]
        softmax_op = Softmax(dtype, N, cluster_n=None)
        _softmax_fwd.compile_cache[compile_key] = cute.compile(
            softmax_op,
            x_cute,
            out_cute,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    _softmax_fwd.compile_cache[compile_key](x, out)


_softmax_fwd.compile_cache = {}


def softmax_fwd(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    _softmax_fwd(x, out)
    return out
