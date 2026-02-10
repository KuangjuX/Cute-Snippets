"""
Based on the QuACK implementation: https://github.com/Dao-AILab/quack/blob/main/quack/softmax.py

Cute-Snippets: Softmax Example

Softmax implementation using CuTe tiled primitives.
Computes softmax along the last dimension of the input tensor.

"""

import math
import torch
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass import Int64, Float32, const_expr, Int32, Float16, BFloat16
from cutlass.cute.runtime import from_dlpack

from typing import Optional, Type
from functools import partial

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import htile.copy as hcopy
from htile import VectorCopy, row_reduce, ReduceLayout, make_fake_tensor


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
        if cluster_n is None:
            cluster_n = self._set_cluster_n()
        self.cluster_n = cluster_n
        self.stage = 2 if not online_softmax else 1
        self.online_softmax = online_softmax

        self.reduce_layout = ReduceLayout(
            threads=self._num_threads(),
            threads_per_row=self._threads_per_row(),
            cluster_n=self.cluster_n,
            stage=self.stage,
        )

        # Initialize vector_copy in __init__
        vec_size = 128 // dtype.width
        threads_per_row = self._threads_per_row()
        num_threads = self._num_threads()
        self.vector_copy = VectorCopy(
            dtype=self.dtype,
            threads_per_row=threads_per_row,
            num_threads=num_threads,
            num_copy_elems=vec_size,
        )

    def _num_threads(self):
        return 128 if self.N <= 16384 else 256

    def _threads_per_row(self):
        N = self.N
        for limit, threads in [
            (64, 8),
            (128, 16),
            (3072, 32),
            (6144, 64),
            (16384, 128),
        ]:
            if N <= limit:
                return threads
        return 256

    def _allocate_reduction_buffer_and_mbar(
        self,
        smem: cutlass.utils.SmemAllocator,
        tv_layout: cute.Shape,
        is_persistent: bool = False,
    ) -> tuple[cute.Tensor, cute.Tensor]:

        reduction_buffer = smem.allocate_tensor(
            cutlass.Float32,
            self.reduce_layout.make_layout(),
            byte_alignment=8,
        )

        if const_expr(self.cluster_n > 1):
            mbar_ptr = smem.allocate_array(
                Int64,
                num_elems=self.stage if not is_persistent else self.stage * 2,
            )
        else:
            mbar_ptr = None

        return reduction_buffer, mbar_ptr

    # def _get_reduction_buffer_layout(
    #     self, tv_layout: cute.Layout, cluster_n: int
    # ):
    #     num_warps = self._num_threads() // cute.arch.WARP_SIZE
    #     threads_per_row = self._threads_per_row()

    #     # 模式 A (Rank 1): 如果 shape[0] 只是一个简单的数字，说明该 Block 只负责处理单行。
    #     # 此时所有 Warp 都参与这一行的归约，因此 warps_per_row = num_warps。
    #     # 模式 B (Nested Rank): 如果 shape[0] 是嵌套的（例如之前提到的 ((V, M), ...)），
    #     # tv_layout.shape[0][0] 提取的是负责行方向分片的线程部分。
    #     # 通过这个值计算出在单行数据上协作的 Warp 数量。
    #     warps_per_row = threads_per_row // cute.arch.WARP_SIZE

    #     # 第一维 num_warps // warps_per_row:
    #     # 代表这个 Block 同时处理的逻辑行数。

    #     # 第二维 (warps_per_row, cluster_n):
    #     # 这是一个嵌套维度，专门用于跨 Warp 和跨集群归约。
    #     # warps_per_row 用于存放不同 Warp 的中间结果。
    #     # cluster_n 用于存放集群中不同 Block 的结果（如果开启了 SM90+ 的 Cluster 特性）。

    #     # 第三维 self.stage:
    #     # 代表归约的阶段数。在 Softmax 前向计算中，通常需要 2 个阶段（第一阶段找 Max，第二阶段求 Sum）；
    #     # 如果是 Online Softmax，则可能是 1 个阶段，但存储的是 (max, sum) 对。

    #     # order=(1, 0, 2) 的物理意义
    #     # 物理存储顺序: 这里的 order 决定了哪个维度的索引在内存地址上变化最快。

    #     # 优化逻辑: 1 放在最前面（变化最快），意味着同一行内不同 Warp/Cluster 的归约结果在内存中是连续存储的。
    #     # 这有利于后续执行 block_reduce 或 cluster_reduce 时，线程能够以连续的地址合并访问（Coalesced Access）这些中间值。
    #     return cute.make_ordered_layout(
    #         (
    #             num_warps // warps_per_row,
    #             (warps_per_row, cluster_n),
    #             self.stage,
    #         ),
    #         order=(1, 0, 2),
    #     )

    def _set_cluster_n(self):
        N = self.N
        if const_expr(self.dtype.width == 16):
            thresholds = [
                (16 * 1024, 1),
                (32 * 1024, 2),
                (64 * 1024, 4),
                (128 * 1024, 8),
            ]
        else:
            thresholds = [
                (32 * 1024, 1),
                (64 * 1024, 2),
                (128 * 1024, 4),
                (256 * 1024, 8),
            ]
        for limit, cluster_n in thresholds:
            if N <= limit:
                return cluster_n
        return 16

    @cute.jit
    def _initialize_cluster(
        self,
        tidx: Int32,
        mbar_ptr: cute.Pointer,
        num_warps: int,
        is_persistent: bool = False,
    ):

        if const_expr(self.cluster_n > 1):
            if tidx < self.stage:
                cute.arch.mbarrier_init(mbar_ptr + tidx, 1)
                if const_expr(is_persistent):
                    cute.arch.mbarrier_init(
                        mbar_ptr + tidx + self.stage, num_warps * self.cluster_n
                    )
            cute.arch.mbarrier_init_fence()
            # Cluster arrive after barrier init
            cute.arch.cluster_arrive_relaxed()

    @cute.jit
    def __call__(self, mX: cute.Tensor, mO: cute.Tensor, stream: cuda.CUstream):
        assert (
            mX.element_type == self.dtype
        ), f"Input tensor element type {mX.element_type} does not match dtype {self.dtype}"

        largest_dtype_width = const_expr(
            max(t.element_type.width for t in (mX, mO))
        )
        vec_size = 128 // largest_dtype_width
        threads_per_row = self._threads_per_row()
        num_threads = self._num_threads()

        num_block_N = cute.ceil_div(
            self.N // vec_size, threads_per_row * self.cluster_n
        )
        tiler_mn = (
            self._num_threads() // threads_per_row,
            vec_size * threads_per_row * num_block_N,
        )

        tiled_copy = self.vector_copy.tiled_copy_2d()

        # cute.printf("tiler_mn: {}\n", tiler_mn)
        # cute.printf("blockX: {}\n", cute.ceil_div(mX.shape[0], tiler_mn[0]))

        self.kernel(mX, mO, tiler_mn, tiled_copy, threads_per_row).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=(
                [1, self.cluster_n, 1]
                if const_expr(self.cluster_n > 1)
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

        tv_layout = tiled_copy.layout_tv_tiled

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cluster_y = (
            const_expr(0)
            if const_expr(self.cluster_n == 1)
            else cute.arch.block_idx()[1]
        )

        num_warps = self._num_threads() // cute.arch.WARP_SIZE

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)

        gX, gO, cX = [
            cute.local_tile(mT, tiler_mn, (bidx, cluster_y))
            for mT in (mX, mO, idX)
        ]

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )

        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(
            smem, tv_layout
        )

        thr_copy_X = tiled_copy.get_slice(tidx)

        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)

        tXgO = thr_copy_X.partition_D(gO)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        tXrX, tXrO = [cute.make_fragment_like(thr) for thr in (tXgX, tXgO)]

        # if tidx == 0:
        #     cute.printf("tXrX: {}\n", tXrX)
        #     cute.printf("tXrO: {}\n", tXrO)

        # Handle non-even boundary memory safe access
        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = (
            None
            if is_even_N
            else self.vector_copy.predicate_k(
                thr_copy_X.partition_S(cX),
                shape[1],
            )
        )

        vector_copy = partial(hcopy.vector_copy, pred=tXpX)

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        if tXcX[0][0] < shape[0]:
            vector_copy(tXgX, tXsX, is_async=True)

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(cute.Float32)

        max_x = row_reduce(
            x,
            cute.ReductionOp.MAX,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr + 0 if const_expr(self.cluster_n > 1) else None,
            init_val=-Float32.inf,
            hook_fn=(
                cute.arch.cluster_wait
                if const_expr(self.cluster_n > 1)
                else None
            ),
        )

        log2_e = math.log2(math.e)
        exp_x = cute.math.exp2(x * log2_e - (max_x * log2_e), fastmath=True)
        denom = row_reduce(
            exp_x,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 1],
            mbar_ptr + 1 if const_expr(self.cluster_n > 1) else None,
            init_val=0.0,
        )

        y = exp_x * cute.arch.rcp_approx(denom)
        tXrO.store(y.to(tXrO.element_type))
        if tXcX[0][0] < shape[0]:
            vector_copy(tXrO, tXgO)


torch2cute_dtype_map = {
    torch.float16: Float16,
    torch.bfloat16: BFloat16,
    torch.float32: Float32,
    torch.int32: Int32,
    torch.int64: Int64,
}


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
