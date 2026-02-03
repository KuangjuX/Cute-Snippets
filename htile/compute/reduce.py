from multiprocessing import reduction
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, const_expr

from typing import Optional, Callable, Type

import htile.utils as utils

import operator


class ReduceLayout:
    """
    Reduce buffer for warp-level and cluster-level reductions.
    It has shape (num_warps // warps_per_row, (warps_per_row, cluster_n), stage)
    """

    def __init__(
        self,
        threads: int,
        threads_per_row: int,
        cluster_n: int,
        stage: int,
    ):
        self.warps_per_row = threads_per_row // cute.arch.WARP_SIZE
        self.num_warps = threads // cute.arch.WARP_SIZE
        self.cluster_n = cluster_n
        self.stage = stage

    @cute.jit
    def make_layout(self) -> cute.Layout:
        """
        Returns the reduction buffer layout.

        Returns:
            The CuTe layout for the reduction buffer with shape
            (num_warps // warps_per_row, (warps_per_row, cluster_n), stage).
        """
        reduce_layout = cute.make_ordered_layout(
            (
                self.num_warps // self.warps_per_row,
                (self.warps_per_row, self.cluster_n),
                self.stage,
            ),
            order=(1, 0, 2),
        )
        return reduce_layout


@cute.jit
def block_reduce(
    val: cute.Numeric,
    op: Callable,
    reduction_buffer: cute.Tensor,
    init_val: cute.Numeric = 0.0,
) -> cute.Numeric:
    """
    Block reduction for warp-level reductions.
    reduction_buffer has shape (num_waprs / warp_per_row, warp_per_row)
    """
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    warps_per_row = cute.size(reduction_buffer.shape[1])
    row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row

    if lane_idx == 0:
        reduction_buffer[row_idx, col_idx] = val

    cute.arch.barrier()

    block_reduce_val = init_val
    if lane_idx < warps_per_row:
        block_reduce_val = reduction_buffer[row_idx, lane_idx]

    return cute.arch.warp_reduction(block_reduce_val, op)


@cute.jit
def cluster_reduce(
    val: cute.Numeric,
    op: Callable,
    reduction_buffer: cute.Tensor,
    mbar_ptr: cute.Pointer,
    phase: Int32,
    init_val: cute.Numeric = 0.0,
) -> cute.Numeric:
    """
    Cluster reduction for cluster-level reductions.
    """
    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    rows_per_block, (warps_per_row, cluster_n) = reduction_buffer.shape
    row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row

    if warp_idx == 0:
        with cute.arch.elect_one():
            num_warps = rows_per_block * warps_per_row
            cute.arch.mbarrier_arrive_and_expect_tx(
                mbar_ptr,
                num_warps
                * cluster_n
                * reduction_buffer.element_type.width
                // 8,
            )

    if lane_idx < cluster_n:
        utils.store_shared_remote(
            val,
            reduction_buffer[row_idx, lane_idx],
            mbar_ptr,
            cta_rank_in_cluster,
        )

    cute.arch.mbarrier_wait(mbar_ptr, phase=phase if phase is not None else 0)
    block_reduce_val = init_val

    num_iter = cute.ceil_div(warps_per_row * cluster_n, cute.arch.WARP_SIZE)

    for i in cutlass.range_constexpr(num_iter):
        idx = lane_idx + i * cute.arch.WARP_SIZE
        if idx < cute.size(reduction_buffer, mode=[1]):
            block_reduce_val = op(
                block_reduce_val, reduction_buffer[row_idx, idx]
            )
    return cute.arch.warp_reduction(block_reduce_val, op)


@cute.jit
def block_or_cluster_reduce(
    val: cute.Numeric,
    op: Callable,
    reduction_buffer: cute.Tensor,
    mbar_ptr: Optional[cute.Pointer],
    phase: Optional[Int32] = None,
    init_val: cute.Numeric = 0.0,
) -> cute.Numeric:
    """
    Block or cluster reduction for warp-level or cluster-level reductions.
    """
    # block_reduce handles reduction across warps within a block
    # When cluster_n > 1, cluster synchronization is handled by hook_fn in row_reduce
    return block_reduce(val, op, reduction_buffer, init_val)


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

    # if const_expr(threads_per_row <= cute.arch.WARP_SIZE):
    #     return val

    # if const_expr(hook_fn is not None):
    #     hook_fn()

    # if const_expr(reduction_buffer is not None):
    #     warps_per_row, cluster_n = reduction_buffer.shape[1]
    #     assert (
    #         cluster_n == 1 or mbar_ptr is not None
    #     ), "mbar_ptr is required for cluster_n > 1"

    #     if const_expr(warps_per_row > 1 or cluster_n > 1):
    #         # This is a block-level reduction
    #         val = block_or_cluster_reduce(
    #             val, warp_op, reduction_buffer, mbar_ptr, phase, init_val
    #         )
    return val
