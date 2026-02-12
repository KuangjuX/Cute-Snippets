"""
htile.config — Elementwise Tile Configuration

Inspired by TileFusion's separation of "Tile Primitive Configuration" from
"Dataflow over Memory Hierarchies", this module encapsulates the common
configuration logic shared across elementwise kernels (softmax, rmsnorm,
layernorm, cross-entropy, etc.).

TileFusion Mapping
------------------
    TileFusion Concept          →  HTile Equivalent
    ─────────────────────────────────────────────────
    GlobalTile shape/layout     →  tiler_mn, tiled_copy
    WarpLayout                  →  thr_layout (derived from threads_per_row)
    TileIterator                →  cute.local_tile(mT, tiler_mn, coord)
    Loader / Storer             →  See htile.dataflow
    RegTile                     →  VectorRegTile / make_fragment_like

This class owns:
    - Thread / warp / cluster configuration (heuristic-based)
    - VectorCopy and TiledCopy construction
    - tiler_mn computation
    - ReduceLayout construction
    - Shared memory allocation for reduction buffers and mbarriers
    - Cluster initialization (mbarrier_init + cluster_arrive)
"""

from typing import Type, Optional

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Int32, Int64

from htile.copy import VectorCopy
from htile.compute.reduce import ReduceLayout


class ElementwiseTileConfig:
    """Configuration for elementwise (row-wise) tile kernels.

    Encapsulates the shared configuration logic between softmax, rmsnorm,
    layernorm, and similar kernels that process data row-by-row with
    optional cluster-level parallelism.

    Args:
        dtype: Primary data type for computation.
        N: Size of the last (reduction) dimension.
        cluster_n: Number of blocks in cluster. Use None for auto.
        stage: Number of reduction stages (e.g. 2 for softmax: max + sum,
               1 for rmsnorm: sum_sq only).
    """

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        N: int,
        cluster_n: Optional[int] = None,
        stage: int = 1,
    ):
        self.dtype = dtype
        self.N = N
        self.stage = stage

        if cluster_n is None:
            cluster_n = self._auto_cluster_n()
        self.cluster_n = cluster_n

        self.reduce_layout = ReduceLayout(
            threads=self.num_threads(),
            threads_per_row=self.threads_per_row(),
            cluster_n=self.cluster_n,
            stage=self.stage,
        )

    # ------------------------------------------------------------------
    # Thread / Warp / Cluster heuristics
    # ------------------------------------------------------------------

    def num_threads(self) -> int:
        """Number of threads per CTA."""
        return 128 if self.N <= 16384 else 256

    def threads_per_row(self) -> int:
        """Number of threads collaborating on a single row."""
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

    def _auto_cluster_n(self) -> int:
        """Automatically determine cluster size based on dtype and N."""
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

    # ------------------------------------------------------------------
    # VectorCopy / TiledCopy / Tiler
    # ------------------------------------------------------------------

    def make_vector_copy(self, vec_size: Optional[int] = None) -> VectorCopy:
        """Create a VectorCopy instance for this configuration.

        Args:
            vec_size: Override the vectorization width. If None, uses
                      128 // dtype.width.
        """
        if vec_size is None:
            vec_size = 128 // self.dtype.width
        return VectorCopy(
            dtype=self.dtype,
            threads_per_row=self.threads_per_row(),
            num_threads=self.num_threads(),
            num_copy_elems=vec_size,
        )

    def make_tiler_mn(self, vec_size: Optional[int] = None) -> tuple:
        """Compute the tiler shape for cute.local_tile.

        Returns:
            (tile_M, tile_N) where:
                tile_M = num_threads // threads_per_row
                tile_N = vec_size * threads_per_row * num_block_N
        """
        if vec_size is None:
            vec_size = 128 // self.dtype.width
        threads_per_row = self.threads_per_row()
        num_threads = self.num_threads()
        num_block_N = cute.ceil_div(
            self.N // vec_size, threads_per_row * self.cluster_n
        )
        return (
            num_threads // threads_per_row,
            vec_size * threads_per_row * num_block_N,
        )

    # ------------------------------------------------------------------
    # Shared memory allocation
    # ------------------------------------------------------------------

    @cute.jit
    def allocate_reduction_buffer_and_mbar(
        self,
        smem: cutlass.utils.SmemAllocator,
        is_persistent: bool = False,
    ) -> tuple:
        """Allocate reduction buffer and mbarrier in shared memory.

        Args:
            smem: Shared memory allocator.
            is_persistent: Whether to allocate extra mbarriers for
                           persistent kernels.

        Returns:
            (reduction_buffer, mbar_ptr) tuple. mbar_ptr is None when
            cluster_n == 1.
        """
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

    # ------------------------------------------------------------------
    # Cluster initialization
    # ------------------------------------------------------------------

    @cute.jit
    def initialize_cluster(
        self,
        tidx: Int32,
        mbar_ptr: cute.Pointer,
        num_warps: int,
        is_persistent: bool = False,
    ):
        """Initialize mbarriers and synchronize cluster.

        Must be called at the beginning of the kernel when cluster_n > 1.

        Args:
            tidx: Thread index within the CTA.
            mbar_ptr: Pointer to mbarrier array in shared memory.
            num_warps: Number of warps per CTA.
            is_persistent: Whether this is a persistent kernel.
        """
        if const_expr(self.cluster_n > 1):
            if tidx < self.stage:
                cute.arch.mbarrier_init(mbar_ptr + tidx, 1)
                if const_expr(is_persistent):
                    cute.arch.mbarrier_init(
                        mbar_ptr + tidx + self.stage,
                        num_warps * self.cluster_n,
                    )
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()

    # ------------------------------------------------------------------
    # Launch helpers
    # ------------------------------------------------------------------

    def grid_shape(self, batch_dim: int) -> list:
        """Compute grid dimensions for kernel launch.

        Args:
            batch_dim: Size of the batch (M) dimension.

        Returns:
            [grid_x, grid_y, grid_z] list.
        """
        tiler_mn = self.make_tiler_mn()
        return [cute.ceil_div(batch_dim, tiler_mn[0]), self.cluster_n, 1]

    def block_shape(self) -> list:
        """Compute block dimensions for kernel launch."""
        return [self.num_threads(), 1, 1]

    def cluster_shape(self) -> Optional[list]:
        """Compute cluster dimensions for kernel launch.

        Returns:
            [1, cluster_n, 1] if cluster_n > 1, else None.
        """
        if const_expr(self.cluster_n > 1):
            return [1, self.cluster_n, 1]
        return None
