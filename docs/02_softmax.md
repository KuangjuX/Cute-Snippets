# Softmax Technical Document

<p align="center">
  <img src="../media/00_layout/tiled_tv_layout.svg" alt="tiled_tv_layout.svg" style="max-width: 800px; width: 100%; border: 1px solid #888; box-shadow: 2px 2px 12px #ccc;" />
</p>


## Technical Deep Dive: tensor Layouts and Boundary Handling

The Softmax implementation leverages CuTe's hierarchical layouts to achieve high-performance memory access while maintaining safety for non-aligned matrix dimensions.

**1. Hierarchical Layout Decomposition**

A `TiledCopy` partitions the global coordinate tensor into a local layout $tXcX$ structured as `((V, M), 1, K)`:

- Mode 0: `(V, M)` (Atom & Intra-Tile Rows):

    - **V (Vectorized Atom)**: Represents hardware-contiguous elements (e.g., 16 elements for a 128-bit load). These are handled as a single unit by cp.async.
    - **M (Row Iteration)**: Represents additional rows assigned to the same thread for better instruction-level parallelism.

- **Mode 1**: `(1)` (N-Atom Placeholder): Indicates the Copy Atom width aligns with the thread's local territory.
- **Mode 2**: `(K)` (Global Strided Iteration): Represents the strided "jumps" a thread makes to cover the full width ($N$) of the Tile.

**2. The `predicate_k` Strategy**

To handle matrices where width $N$ is not a multiple of the Tile width, QuACK generates a **Predicate Tensor** to mask out-of-bounds accesses.

**Stride-0 Broadcasting and Vector Skipping**

The `predicate_k` function optimizes register usage through two key techniques:

1. **Vector Skipping**: By using `mode=[0, 1]`, the algorithm ignores the inner vectorization dimension $V$. Since 16 elements in a 128-bit load cross the boundary together, only one bit is needed per vector.

2. **Row Broadcasting**: By setting the $M$ dimension stride to 0, the column-boundary check is applied to every row handled by the thread without consuming extra space.

### 3. Predicate Tensor Layout Visualization

The predicate tensor `tApA` is created with a layout that efficiently represents boundary checks for vectorized memory accesses:

<p align="center">
  <img src="../media/00_layout/tApA_layout.svg" alt="tApA_layout.svg" style="max-width: 1000px; width: 100%; border: 1px solid #888; box-shadow: 2px 2px 12px #ccc;" />
</p>

**Understanding the Visualization:**

The visualization shows a 2D layout of the predicate tensor `tApA` with shape `(rows, cols)` where:
- **Rows (Vertical dimension)**: Represent different vectorized groups or row iterations
  - Each row corresponds to a different vectorized group or row iteration handled by the thread
  - The number of rows = `size_mode[0,1] * size_mode[1]`, representing different vectorization groups or row groupings
  
- **Columns (Horizontal dimension)**: Represent the K dimension (matrix column direction)
  - Each column corresponds to a vectorized group's position in the K dimension
  - The number of columns = `size_mode[2]`, representing the number of vectorized groups needed to cover the tile width
  - Each cell contains a boolean predicate value that controls whether the corresponding vectorized memory access is valid (within matrix boundary N)

**Layout Structure:**

The `tApA` layout is created with:
- **Shape**: `(size_mode[0,1], size_mode[1], size_mode[2])` 
- **Stride**: `(size_mode[2], 0, 1)`

The stride pattern `(size_mode[2], 0, 1)` enables:
- **Vector Skipping** (`stride[0] = size_mode[2]`): One predicate value per vectorized group (e.g., 16 elements), skipping the inner vectorization dimension
- **Row Broadcasting** (`stride[1] = 0`): All rows share the same predicate values, saving register space since column boundaries are the same for all rows
- **Sequential K Access** (`stride[2] = 1`): Sequential access along the K dimension for efficient iteration

**Practical Example:**

In the visualization above, each cell (indexed 0-255) represents:
- A boolean predicate value (True/False) that controls whether a vectorized memory access is valid
- The row indicates which vectorized group or row iteration this predicate belongs to
- The column indicates the position in the K dimension (matrix column direction)
- At runtime, these positions are filled with True/False values based on whether `column_index < N` (matrix width)

This design allows the kernel to safely handle non-aligned matrix dimensions while minimizing register usage through vector skipping and row broadcasting.

## 4. Reduction Buffer Layout Visualization

The reduction buffer is used to store intermediate reduction results during the softmax computation. It is organized to support efficient warp-level and cluster-level reductions:

<p align="center">
  <img src="../media/00_layout/reduction_buffer_layout.svg" alt="reduction_buffer_layout.svg" style="max-width: 800px; width: 100%; border: 1px solid #888; box-shadow: 2px 2px 12px #ccc;" />
</p>

**Understanding the Visualization:**

The visualization shows a 2D flattened view of the reduction buffer layout with shape `(rows, cols)` where:
- **Rows (Vertical dimension)**: Represent warps within a row group
  - Each row corresponds to a different warp in the same row group
  - The number of rows = `warps_per_row * cluster_n`, representing all warps that work together on the same row
  - In the example above: 2 rows indicate either 2 warps per row with 1 cluster, or 1 warp per row with 2 clusters
  
- **Columns (Horizontal dimension)**: Represent warp groups across reduction stages
  - Each column corresponds to a different warp group or reduction stage
  - The number of columns = `num_warp_groups * stage`, representing all warp groups across all reduction stages
  - In the example above: 8 columns indicate 4 warp groups × 2 stages (double buffering)

**Layout Structure:**

The reduction buffer layout is created with:
- **Original 3D Shape**: `(num_warps // warps_per_row, (warps_per_row, cluster_n), stage)`
  - Dimension 0: Number of warp groups (how many groups of warps work on different rows)
  - Dimension 1: `(warps_per_row, cluster_n)` - Warps per row and cluster dimension
  - Dimension 2: `stage` - Reduction stage (1 or 2 for double buffering)
- **Order**: `(1, 0, 2)` - This reorders dimensions to optimize memory access patterns

**Memory Access Pattern:**

The stride pattern `(1, total_warps_in_row)` in the 2D view reveals:
- **Row stride = 1**: Sequential access within a row group (warps access consecutive memory locations)
- **Column stride = total_warps_in_row**: Jumping between warp groups or stages (skipping by the number of warps in a row group)

**Practical Example:**

In the visualization above:
- The numbers 0-15 represent memory locations in the reduction buffer
- **Row 0** (even numbers: 0, 2, 4, 6, 8, 10, 12, 14): First warp in each row group accesses these locations
- **Row 1** (odd numbers: 1, 3, 5, 7, 9, 11, 13, 15): Second warp in each row group accesses these locations
- The pattern shows that warps within the same row group access interleaved memory locations (stride=1 within row, stride=2 across columns)
- This interleaving allows efficient parallel reduction operations where multiple warps can work simultaneously without memory conflicts

**Why This Layout Design:**

1. **Warp-Level Parallelism**: Each warp can independently perform reductions on its assigned data
2. **Cluster Support**: The cluster dimension allows multiple clusters to work on different parts of the reduction
3. **Double Buffering**: When `stage=2`, the layout supports overlapping computation and memory operations
4. **Memory Coalescing**: The order `(1, 0, 2)` ensures that warps accessing consecutive memory locations (dimension 1) get the best memory bandwidth utilization

This layout design enables efficient hierarchical reduction: threads reduce within a warp, warps reduce within a row group, and row groups reduce across the entire tile.