"""
htile.dataflow — Dataflow Primitives for Memory Hierarchy Traversal

Inspired by TileFusion's Loader/Storer abstraction, this module provides
composable building blocks for moving data between memory hierarchies
(Global → Shared → Register and Register → Global) in elementwise kernels.

TileFusion Mapping
------------------
    TileFusion Concept          →  HTile Equivalent
    ─────────────────────────────────────────────────
    GlobalToRegLoader           →  ElementwiseLoader (G→S→R pipeline)
    RegToGlobalStorer           →  ElementwiseStorer (R→G pipeline)
    TileIterator + partition    →  ElementwiseKernelContext.setup()

The key design principle (from TileFusion):
    "Rigorously separate data flow across the memory hierarchy from the
     configuration of individual macro kernels."

ElementwiseKernelContext orchestrates the full setup sequence that is
identical across softmax, rmsnorm, layernorm, etc.:
    1. Compute thread/block indices and cluster position
    2. Allocate shared memory (data tile + reduction buffer + mbarrier)
    3. Partition global tiles via TiledCopy
    4. Generate boundary predicates
    5. Create register fragments
    6. Initialize cluster barriers
"""

from typing import Optional
from functools import partial

import cutlass
import cutlass.cute as cute
from cutlass import const_expr

import htile.copy as hcopy
from htile.config import ElementwiseTileConfig


class ElementwiseLoader:
    """Loader: Global Memory → Shared Memory → Registers.

    Encapsulates the common load pipeline:
        1. vector_copy(G→S, async) with optional predicate
        2. cp_async_commit_group + cp_async_wait_group
        3. autovec_copy(S→R)

    Corresponds to TileFusion's GlobalToRegLoader / GlobalToSharedLoader.
    """

    @staticmethod
    @cute.jit
    def load(
        tXgX: cute.Tensor,
        tXsX: cute.Tensor,
        tXrX: cute.Tensor,
        pred: Optional[cute.Tensor] = None,
        row_in_bounds: bool = True,
    ):
        """Load data from global memory through shared memory into registers.

        Args:
            tXgX: Partitioned global source tensor.
            tXsX: Partitioned shared memory destination tensor.
            tXrX: Register fragment to load into.
            pred: Optional predicate tensor for boundary handling.
            row_in_bounds: Whether the current row is within bounds.
        """
        vector_copy = partial(hcopy.vector_copy, pred=pred)

        if row_in_bounds:
            vector_copy(tXgX, tXsX, is_async=True)

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

        cute.autovec_copy(tXsX, tXrX)

    @staticmethod
    @cute.jit
    def load_direct(
        tXgX: cute.Tensor,
        tXrX: cute.Tensor,
        pred: Optional[cute.Tensor] = None,
        row_in_bounds: bool = True,
    ):
        """Load data directly from global memory into registers (no smem).

        Args:
            tXgX: Partitioned global source tensor.
            tXrX: Register fragment to load into.
            pred: Optional predicate tensor for boundary handling.
            row_in_bounds: Whether the current row is within bounds.
        """
        vector_copy = partial(hcopy.vector_copy, pred=pred)
        if row_in_bounds:
            vector_copy(tXgX, tXrX)


class ElementwiseStorer:
    """Storer: Registers → Global Memory.

    Encapsulates the common store pipeline:
        1. tXrO.store(result)
        2. vector_copy(R→G) with optional predicate and row guard

    Corresponds to TileFusion's RegToGlobalStorer.
    """

    @staticmethod
    @cute.jit
    def store(
        tXrO: cute.Tensor,
        tXgO: cute.Tensor,
        pred: Optional[cute.Tensor] = None,
        row_in_bounds: bool = True,
    ):
        """Store data from registers to global memory.

        Args:
            tXrO: Register fragment containing the result.
            tXgO: Partitioned global destination tensor.
            pred: Optional predicate tensor for boundary handling.
            row_in_bounds: Whether the current row is within bounds.
        """
        vector_copy = partial(hcopy.vector_copy, pred=pred)
        if row_in_bounds:
            vector_copy(tXrO, tXgO)


class ElementwiseKernelContext:
    """Kernel-level context that orchestrates tile setup inside a @cute.kernel.

    This class encapsulates the boilerplate that is identical across
    elementwise kernels:
        - Thread/block/cluster index extraction
        - Shared memory allocation (data tile + reduction buffer + mbar)
        - Global tile partitioning via TiledCopy
        - Boundary predicate generation
        - Register fragment creation
        - Cluster initialization

    After calling setup(), the kernel only needs to focus on the
    computation-specific dataflow (the "algorithm").

    Usage in a @cute.kernel::

        ctx = ElementwiseKernelContext(config)
        ctx.setup(mX, mO, tiler_mn, tiled_copy, threads_per_row)

        # Now use ctx.tXrX, ctx.tXgO, ctx.reduction_buffer, etc.
        # for the kernel-specific computation.
    """

    def __init__(self, config: ElementwiseTileConfig):
        self.config = config

    @cute.jit
    def setup(
        self,
        mX: cute.Tensor,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        extra_smem_tensors: int = 0,
    ):
        """Perform the common kernel setup sequence.

        This method:
            1. Extracts tidx, bidx, cluster_y
            2. Allocates smem for the input tile and reduction buffer
            3. Partitions global/smem tiles via tiled_copy
            4. Generates boundary predicates
            5. Creates register fragments
            6. Initializes cluster barriers

        Args:
            mX: Global input tensor (used for shape and element_type).
            tiler_mn: Tile shape for cute.local_tile.
            tiled_copy: The TiledCopy object.
            extra_smem_tensors: Number of additional smem tensors to
                allocate with the same layout as sX (e.g. for residual).

        Sets the following attributes on self:
            tidx, bidx, cluster_y: Thread/block indices
            smem: SmemAllocator (for further allocations by the kernel)
            sX: Shared memory tensor for input
            extra_sX: List of additional smem tensors (if extra_smem_tensors > 0)
            reduction_buffer, mbar_ptr: Reduction infrastructure
            thr_copy: Thread-level copy slice
            tXgX, tXsX: Partitioned global/smem input tensors
            tXrX: Register fragment for input
            tXcX: Coordinate tensor for boundary checks
            tXpX: Predicate tensor (None if N is evenly tiled)
            row_in_bounds: Whether current row is within bounds
            shape: Shape of the global input tensor
            vector_copy_fn: Partial vector_copy with predicate bound
        """
        self.tidx, _, _ = cute.arch.thread_idx()
        self.bidx, _, _ = cute.arch.block_idx()
        self.cluster_y = (
            const_expr(0)
            if const_expr(self.config.cluster_n == 1)
            else cute.arch.block_idx()[1]
        )

        self.shape = mX.shape

        # ---- Shared memory allocation ----
        self.smem = cutlass.utils.SmemAllocator()
        self.sX = self.smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )

        self.extra_sX = []
        for _ in cutlass.range_constexpr(extra_smem_tensors):
            s = self.smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
            self.extra_sX.append(s)

        self.reduction_buffer, self.mbar_ptr = (
            self.config.allocate_reduction_buffer_and_mbar(self.smem)
        )

        # ---- Identity tensor for coordinate tracking ----
        idX = cute.make_identity_tensor(self.shape)

        # ---- Tile partitioning (TileFusion's TileIterator equivalent) ----
        self.gX = cute.local_tile(mX, tiler_mn, (self.bidx, self.cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (self.bidx, self.cluster_y))

        self.thr_copy = tiled_copy.get_slice(self.tidx)

        self.tXgX = self.thr_copy.partition_S(self.gX)
        self.tXsX = self.thr_copy.partition_D(self.sX)
        self.tXcX = self.thr_copy.partition_S(cX)[(0, None), None, None]

        self.tXrX = cute.make_fragment_like(self.tXgX)

        # ---- Boundary predicates ----
        is_even_N = const_expr(
            self.shape[1] == tiler_mn[1] * self.config.cluster_n
        )
        self.tXpX = (
            None
            if is_even_N
            else self.config.make_vector_copy().predicate_k(
                self.thr_copy.partition_S(cX),
                self.shape[1],
            )
        )

        self.vector_copy_fn = partial(hcopy.vector_copy, pred=self.tXpX)

        # ---- Row bounds check ----
        self.row = self.tXcX[0][0]
        self.row_in_bounds = self.row < self.shape[0]

        # ---- Cluster initialization ----
        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        self.config.initialize_cluster(self.tidx, self.mbar_ptr, num_warps)

    @cute.jit
    def partition_global(
        self, mT: cute.Tensor, tiler_mn: cute.Shape
    ) -> cute.Tensor:
        """Partition an additional global tensor using the same tiling scheme.

        Useful for output tensors, weight tensors, etc.

        Args:
            mT: Global tensor to partition.
            tiler_mn: Tile shape.

        Returns:
            Partitioned tensor (via partition_S for source or partition_D for dest).
        """
        gT = cute.local_tile(mT, tiler_mn, (self.bidx, self.cluster_y))
        return gT

    @cute.jit
    def partition_source(self, gT: cute.Tensor) -> cute.Tensor:
        """Partition a local-tiled tensor as source using the thread copy slice."""
        return self.thr_copy.partition_S(gT)

    @cute.jit
    def partition_dest(self, gT: cute.Tensor) -> cute.Tensor:
        """Partition a local-tiled tensor as destination using the thread copy slice."""
        return self.thr_copy.partition_D(gT)

    @cute.jit
    def partition_smem_dest(self, sT: cute.Tensor) -> cute.Tensor:
        """Partition a shared memory tensor as destination."""
        return self.thr_copy.partition_D(sT)

    @cute.jit
    def make_fragment(self, partitioned: cute.Tensor) -> cute.Tensor:
        """Create a register fragment matching a partitioned tensor."""
        return cute.make_fragment_like(partitioned)

    @cute.jit
    def load_to_smem_and_regs(self):
        """Execute the standard G→S→R load for the primary input (self.tXgX → sX → tXrX)."""
        ElementwiseLoader.load(
            self.tXgX,
            self.tXsX,
            self.tXrX,
            pred=self.tXpX,
            row_in_bounds=self.row_in_bounds,
        )

    @cute.jit
    def store_from_regs(self, tXrO: cute.Tensor, tXgO: cute.Tensor):
        """Execute the standard R→G store."""
        ElementwiseStorer.store(
            tXrO,
            tXgO,
            pred=self.tXpX,
            row_in_bounds=self.row_in_bounds,
        )
