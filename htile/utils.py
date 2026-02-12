# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

from typing import Optional
import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr, Float16, BFloat16, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


torch2cute_dtype_map = {
    torch.float16: Float16,
    torch.bfloat16: BFloat16,
    torch.float32: Float32,
    torch.int32: Int32,
    torch.int64: Int64,
}


@cute.jit
def expand(a: cute.Tensor, dim: int, size) -> cute.Tensor:
    """Broadcast-expand a CuTe tensor by inserting a new dimension with stride=0.

    This is the CuTe equivalent of ``torch.unsqueeze(dim).expand(size)``
    — it inserts a new axis at position *dim* whose size is *size* and
    whose stride is **0**, so every index along that axis aliases the
    same underlying memory (zero-copy broadcast).

    Args:
        a: Source CuTe tensor.
        dim: Position at which to insert the new dimension
             (0 ≤ dim ≤ rank(a)).
        size: Extent of the new dimension (Int32, int, or sym value).

    Returns:
        A new CuTe tensor that shares ``a``'s data pointer but has one
        extra dimension.

    Example::

        # mW has shape (N,), stride (1,)
        mW2d = expand(mW, dim=0, size=M)
        # → shape (M, N), stride (0, 1)
        # mW2d[i, j] == mW[j]  for all i  (broadcast along rows)
    """
    shape = (*a.shape[:dim], size, *a.shape[dim:])
    stride = (*a.layout.stride[:dim], 0, *a.layout.stride[dim:])
    return cute.make_tensor(a.iterator, cute.make_layout(shape, stride=stride))


def make_fake_tensor(
    dtype, shape, divisibility=1, leading_dim=-1
) -> Optional[cute.Tensor]:
    if leading_dim < 0:
        leading_dim = len(shape) + leading_dim
    if dtype is None:
        return None
    stride = tuple(
        cute.sym_int64(divisibility=divisibility) if i != leading_dim else 1
        for i in range(len(shape))
    )
    return cute.runtime.make_fake_tensor(
        dtype,
        shape,
        stride=stride,
        assumed_align=divisibility * dtype.width // 8,
    )


@dsl_user_op
def elem_pointer(
    x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None
) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@dsl_user_op
def set_block_rank(
    smem_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    """Map the given smem pointer to the address at another CTA rank in the cluster."""
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def store_shared_remote(
    val: float | Float32 | Int32 | cutlass.Int64,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: cute.typing.Int,
    *,
    loc=None,
    ip=None,
) -> None:
    remote_smem_ptr_i32 = set_block_rank(
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    remote_mbar_ptr_i32 = set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    if const_expr(isinstance(val, float)):
        val = Float32(val)
    assert isinstance(
        val, (Float32, Int32, cutlass.Int64)
    ), "val must be Float32, Int32, or Int64"
    suffix = {Float32: "f32", Int32: "s32", cutlass.Int64: "s64"}[type(val)]
    constraint = {Float32: "f", Int32: "r", cutlass.Int64: "l"}[type(val)]
    llvm.inline_asm(
        None,
        [
            remote_smem_ptr_i32,
            val.ir_value(loc=loc, ip=ip),
            remote_mbar_ptr_i32,
        ],
        f"st.async.shared::cluster.mbarrier::complete_tx::bytes.{suffix} [$0], $1, [$2];",
        f"r,{constraint},r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
