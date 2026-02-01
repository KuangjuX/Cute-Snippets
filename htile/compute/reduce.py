import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, const_expr

from typing import Optional, Callable

import operator


@cute.jit
def row_reduce(
    x: cute.TensorSSA | cute.Numeric,
    op: cute.ReductionOp,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: Optional[cute.Tensor] = None,
    mbar_ptr: Optional[cute.Pointer] = None,
    phase: Optional[Int32] = None,
    init_val: cute.Numeric = 0.0,
    hook_fn: Optional[Callable] = None,
):
    if const_expr(isinstance(x, cute.TensorSSA)):
        val = x.reduce(op, init_val=init_val, reduction_profile=0)
    else:
        val = x

    warp_op = {
        cute.ReductionOp.ADD: operator.add,
        cute.ReductionOp.MAX: (
            cute.arch.fmax if const_expr(x.dtype == Float32) else max
        ),
        cute.ReductionOp.MIN: min,
        cute.ReductionOp.MUL: operator.mul,
    }[op]

    val = cute.arch.warp_reduction(
        val,
        warp_op,
        threads_in_group=min(threads_per_row, cute.arch.WARP_SIZE),
    )

    if const_expr(hook_fn is not None):
        hook_fn()

    if const_expr(reduction_buffer is not None):
        warps_per_row, cluster_n = reduction_buffer.shape[1]
        assert (
            cluster_n == 1 or mbar_ptr is not None
        ), "mbar_ptr is required for cluster_n > 1"

        if const_expr(warps_per_row > 1 or cluster_n > 1):
            # This is a block-level reduction
