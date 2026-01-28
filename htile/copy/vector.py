import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cutlass_dsl import dsl_user_op
from cutlass import const_expr

from typing import Type, Optional

class VectorCopy():
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        threads_per_row: int,
        num_threads: int,
        num_copy_elems: int = 1
    ):
        self.dtype = dtype
        self.threads_per_row = threads_per_row
        self.num_threads = num_threads
        self.num_copy_elems = num_copy_elems
        self.num_copy_bits = num_copy_elems * self.threads_per_row

    def _threads_per_row(self):
        return self.threads_per_row

    def _num_threads(self):
        return self.num_threads

    def _num_copy_elems(self):
        return self.num_copy_elems

    def tiled_copy_2d(self, is_async: bool = False):
        copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
        copy_atom = cute.make_copy_atom(copy_op, self.dtype, num_bits_per_copy=self.num_copy_bits)

        thr_layout = cute.make_ordered_layout(
            (self.num_threads // self.threads_per_row, self.threads_per_row),
            order=(1, 0)
        )

        val_layout = cute.make_layout((1, self.num_copy_elems))

        return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)



@dsl_user_op
def vector_copy(
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    pred: Optional[cute.Tensor] = None,
    is_async: bool = False,
    loc=None,
    ip=None,
    **kwargs,
):
    num_copy_elems = src.shape[0][0]
    dtype = src.element_type

    num_copy_bits = const_expr(min(128, num_copy_elems * dtype.width))
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)

    cute.copy(copy_atom, src, dst, pred=pred, loc=loc, ip=ip, **kwargs)