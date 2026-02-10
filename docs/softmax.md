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

## Reduction Buffer Layout and Memory Mapping

The **Reduction Buffer** serves as a high-speed scratchpad in smem for inter-warp and inter-block data exachange. It is primarily used during the row-reduction phase to synchronize partial results (such as `max_x` and `sum_exp_x`) across different execution units.

<p align="center">
  <img src="../media/00_layout/reduction_buffer_layout.svg" alt="reduction_buffer_layout.svg" style="max-width: 800px; width: 100%; border: 1px solid #888; box-shadow: 2px 2px 12px #ccc;" />
</p>

The buffer's physical organization is dynamically computed based on the thread-value mapping (`tv_layout`) and the hardware cluster configuration.

The resulting layout is a 3D structure that optimizes for bank-conflict-free access and coalesced writes:

- **Dimension 0**: `num_warps // warps_per_row` **(Logical Rows)**:
  - Represents the number of independent rows being processed concurrently within a single Thread Block.

- **Dimension 1**: `(warps_per_row, cluster_n) (Reduction Peers)`:

  - `warps_per_row`: Stores partial results from different warps working on the same row.

  - `cluster_n`: Allocates space for cross-block synchronization when using NVIDIA SM90+ Cluster features (Distributed Shared Memory).

- **Dimension 2**: `self.stage` **(Pipeline Stages)**:
  - Supports multi-stage reductions. For standard Softmax, this is typically `stage=2` (one for Max, one for Sum). For Online Softmax, it is often `stage=1` but stores fused `(max, sum)` pairs.


The physical memory mapping is strictly ordered as `(1, 0, 2)`:

- **Coalesced Access**: By setting the reduction peer dimension (mode 1) as the fastest-changing dimension in memory, the hardware ensures that when multiple warps write their partial results simultaneously, the accesses are coalesced and memory bank conflicts are minimized.

- **Logical Isolation**: Stage-wise data (`mode 2`) is placed as the slowest-changing dimension to ensure clear separation between the Max and Sum reduction phases.


The relationship between the compute layout ($T$) and the reduction buffer ($V$) can be visualized as follows:

```
[ Logical Matrix Row ]
      |
      +--- Processed by [Warp 0, Warp 1, ..., Warp N]
                              |
                              V
[ Reduction Buffer (Smem) ]
+-------------------------------------------------------------+
| Stage 0 (Max) | Row 0: [W0_res][W1_res]...[Wn_res]          | <-- Physical Order (1, 0)
|               | Row 1: [W0_res][W1_res]...[Wn_res]          |
+-------------------------------------------------------------+
| Stage 1 (Sum) | Row 0: [W0_res][W1_res]...[Wn_res]          |
|               | Row 1: [W0_res][W1_res]...[Wn_res]          |
+-------------------------------------------------------------+
```

This structured layout allows the `block_reduce` and `cluster_reduce` functions to perform final aggregations using optimized warp-shuffle instructions after a single synchronized Smem load.

## Performance Evaluation

```
========================================================================================================================
Performance Summary Table
========================================================================================================================
[M, N]               Ours                                torch.compile                       Liger Kernel                       
-----------------------------------------------------------------------------------------------------------------------------
                     Latency(ms) / BW (GB/s)             Latency(ms) / BW (GB/s)             Latency(ms) / BW (GB/s)            
-----------------------------------------------------------------------------------------------------------------------------
[32768, 1024]        0.0485 / 2765.00                    0.0588 / 2283.00                    0.0479 / 2799.00                   
[32768, 2048]        0.0931 / 2884.00                    0.2485 / 1080.00                    0.0909 / 2953.00                   
[32768, 4096]        0.1833 / 2930.00                    0.3942 / 1362.00                    0.1804 / 2976.00                   
[32768, 6144]        0.2725 / 2955.00                    0.5409 / 1489.00                    0.2699 / 2984.00                   
[16384, 8192]        0.1839 / 2920.00                    0.3776 / 1422.00                    0.1800 / 2982.00                   
[8192, 16384]        0.1829 / 2936.00                    0.3576 / 1501.00                    0.1785 / 3007.00                   
[4096, 16384]        0.0939 / 2857.00                    0.1843 / 1457.00                    0.0908 / 2955.00                   
[4096, 32768]        0.1822 / 2946.00                    0.3635 / 1477.00                    0.1908 / 2814.00                   
[4096, 65536]        0.3618 / 2968.00                    0.7451 / 1441.00                    0.5815 / 1847.00                   
[4096, 65536]        0.3611 / 2974.00                    0.7438 / 1444.00                    0.5819 / 1845.00                   
[4096, 131072]       0.7169 / 2996.00                    1.4992 / 1432.00                    unsupported                        
[4096, 8192]         0.0490 / 2739.00                    0.1001 / 1341.00                    0.0459 / 2921.00                   
[8192, 8192]         0.0939 / 2858.00                    0.1923 / 1396.00                    0.0909 / 2954.00                   
[16384, 16384]       0.3622 / 2964.00                    0.7030 / 1527.00                    0.3597 / 2985.00                   
=============================================================================================================================
```