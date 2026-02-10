import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cutlass_dsl import dsl_user_op
from cutlass import const_expr, Int32

from typing import Type, Optional


class VectorCopy:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        threads_per_row: int,
        num_threads: int,
        num_copy_elems: int = 1,
    ):
        self.dtype = dtype
        self.threads_per_row = threads_per_row
        self.num_threads = num_threads
        self.num_copy_elems = num_copy_elems
        self.num_copy_bits = num_copy_elems * dtype.width

    def _threads_per_row(self):
        return self.threads_per_row

    def _num_threads(self):
        return self.num_threads

    def _num_copy_elems(self):
        return self.num_copy_elems

    def tiled_copy_2d(self, is_async: bool = False):
        copy_op = (
            cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
        )
        copy_atom = cute.make_copy_atom(
            copy_op, self.dtype, num_bits_per_copy=self.num_copy_bits
        )

        thr_layout = cute.make_ordered_layout(
            (self.num_threads // self.threads_per_row, self.threads_per_row),
            order=(1, 0),
        )

        val_layout = cute.make_layout((1, self.num_copy_elems))

        return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

    @cute.jit
    # @staticmethod
    def predicate_k(self, tAcA: cute.Tensor, limit: Int32) -> cute.Tensor:
        """
        为 K 维度（通常是矩阵的列维度）生成断言张量（Predicate Tensor），用于处理非对齐内存访问。

        参数：
            tAcA：线程持有的坐标张量（Identity Tensor 的 partition 结果）。存储了
            逻辑坐标（m，n）。
            limit：边界限制值，通常为矩阵的总列数 N。

        返回：
            tApA：一个布尔类型的断言张量，形状与输入张量匹配，用于控制加载/存储指令。
        """

        # 1. 创建断言张量的布局（Layout——
        # 目标：创建一个形状与划分后的数据张量一致的布尔 Fragment。
        # 关键点：这里的 Stride 设计为 (cute.size(tAcA, mode=[2]), 0, 1)。
        # 这种设计以为在内存向量化维度（mode 0 的子维度）上步长为 0.
        # 使得个向量化指令（如 16 个元素）共享同一个断言结果，提高掩码计算效率。

        # mode 指的是布局中的一个轴，mode=0，最外层的第一个维度
        # 代码中通过 cute.size(tAcA, mode=[...]) 提取特定层级的长度，重新构造了一个
        # 专用的断言张量 `tApA`。
        # 这里新的张量去掉了向量化维度 mode=[0, 0]，在一个向量化拷贝指令中
        # 这 16 个元素在 K 维度的越界情况通常是完全一致的。因此，不需要为每个元素存储一个
        # 布尔值，只需为整组向量存储一个布尔值即可。

        # 在 stride 中间的 0，对应 M 维度的步长。这意味着无论 M 索引如何变化，物理上都会指向
        # 同一个位置，这实现了广播，进一步节省了寄存器空间（共享同一套列边界检查逻辑，
        # 列边界 N 对所有行都是一样的）。
        tApA = cute.make_fragment(
            cute.make_layout(
                cute.size(tAcA, mode=[0, 1]),
                cute.size(tAcA, mode=[1]),
                cute.size(tAcA, mode=[2]),
            ),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),
        )

        # 2. 遍历线程负责的所有逻辑块
        # rest_v 对应向量化后的剩余迭代维度
        for rest_v in cutlass.range_constexpr(tApA.shape[0]):
            # rest_k 对应 k 方向（列方向）的迭代维度
            for rest_k in cutlass.range_constexpr(tApA.shape[2]):

                # 3. 提取列索引并进行辩解比较
                # tAcA[(0, rest_v), 0, rest_k] 获取当前位置的坐标元组(row, col)
                # [1] 提取坐标中的第二个分量，即列索引（column index / k-index）
                # cute.elem_less 会生成，如果 col_idx < limit 则为 True，否则为 False
                # 这确保了当 N 不是 Tile 大小整数倍的时，越界的内存方位会被掩码屏蔽
                tApA[rest_v, 0, rest_k] = cute.elem_less(
                    tAcA[(0, rest_v), 0, rest_k][1], limit
                )

        return tApA


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
    copy_atom = cute.make_copy_atom(
        copy_op, dtype, num_bits_per_copy=num_copy_bits
    )

    cute.copy(copy_atom, src, dst, pred=pred, loc=loc, ip=ip, **kwargs)
