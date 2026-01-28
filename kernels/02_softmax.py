import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass import Int64, Float32, const_expr

from typing import Type

from htile.copy import VectorCopy, vector_copy

class Softmax():

    def __init__(self, dtype: Type[cutlass.Numeric], N: int, cluster_n: int):
        self.dtype = dtype
        self.N = N
        self.cluster_n = cluster_n

    def _num_threads(self):
        return 128 if self.N <= 16384 else 256

    def _threads_per_row(self):
        N = self.N 
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
            if N <= limit:
                return threads
        return 256


    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype, f"Input tensor element type {mX.element_type} does not match dtype {self.dtype}"

        largest_dtype_width = const_expr(max(t.element_type.width for t in (mX, mO)))
        vec_size = 128 // largest_dtype_width
        threads_per_row = self._threads_per_row()
        num_threads = self._num_threads()

        num_block_N = cute.ceil_div(self.N // vec_size, threads_per_row * self.cluster_n)
        tiler_mn = (self._num_threads() // threads_per_row, vec_size * threads_per_row * num_block_N)
        
        self.vector_copy = VectorCopy(
            dtype=self.dtype,
            threads_per_row=threads_per_row,
            num_threads=num_threads,
            num_copy_elems=vec_size,
        )
        tiled_copy = self.vector_copy.tiled_copy_2d()

        self.kernel(mX, mO, tiler_mn, tiled_copy, threads_per_row).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None,
            stream=stream,
        )

    
    @cute.kernel
    def kernel(
        self, 
        mX: cute.Tensor,
        mO: cute.Tensor, 
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
    ):

        tv_layout = tiled_copy.layout_tv_tiled

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cluster_y = const_expr(0) if const_expr(self.cluster_n == 1) else cute.arch.block_idx()[1]

        num_warps = self._num_threads() // cute.arch.WARP_SIZE

        shape = mX.shape 
        idX = cute.make_identify_tensor(shape)

        gX, gO, cX = [cute.local_tile(mT, tiler_mn, (bidx, cluster_y)) for mT in (mX, mO, idX)]

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16
        )

        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        thr_copy_X = tiled_copy.get_slice(tidx)

        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)

        tXgO = thr_copy_X.partition_D(gO)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        tXrX, tXrO = [cute.make_fragment_like(thr) for thr in (tXgX, tXgO)]

        if tXcX[0][0] < shape[0]:
            vector_copy(tXgX, tXsX, is_async=True)

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

        cute.autovec_copy(tXsX, tXrX)
    