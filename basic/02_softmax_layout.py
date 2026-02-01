import cutlass
from cutlass import cute
from cute_viz import (
    render_tiled_copy_svg,
    render_layout_svg,
    render_tv_layout_svg,
)
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import htile.copy as hcopy
from htile.copy import VectorCopy


@cute.jit
def visualize_layouts():
    # Based on kernels/02_softmax.py configuration
    M, N = 64, 2048
    # Use cluster_n=2 for better visualization of reduction buffer layout
    cluster_n = 2
    dtype = cutlass.Float16

    # Calculate parameters same as in Softmax.__call__
    largest_dtype_width = 16  # Float16 width
    vec_size = 128 // largest_dtype_width  # 8

    # threads_per_row calculation (same as _threads_per_row method)
    # Use 16 for better visualization (creates more warps for reduction buffer)
    threads_per_row = 16  # For N=2048

    # num_threads calculation (same as _num_threads method)
    num_threads = 128  # For N <= 16384

    # Calculate tiler_mn (same as in __call__)
    num_block_N = cute.ceil_div(
        N // vec_size, threads_per_row * cluster_n
    )  # ceil_div(256, 16) = 16
    tiler_mn = (
        num_threads // threads_per_row,  # 8
        vec_size * threads_per_row * num_block_N,  # 8 * 16 * 16 = 2048
    )

    # Create VectorCopy and tiled_copy (same as in Softmax.__call__)
    vector_copy = VectorCopy(
        dtype=dtype,
        threads_per_row=threads_per_row,
        num_threads=num_threads,
        num_copy_elems=vec_size,
    )
    tiled_copy = vector_copy.tiled_copy_2d()

    # Get tv_layout (same as in kernel)
    tv_layout = tiled_copy.layout_tv_tiled

    # Render tv_layout
    # Use a smaller tiler_mn for visualization to avoid huge canvas
    # The layout structure is the same, just with fewer elements
    # Original: tiler_mn = (8, 2048), use (8, 128) for better visualization
    tiler_mn_viz = (
        tiler_mn[0],
        tiler_mn[1],
    )  # Keep M dimension, reduce N dimension
    render_tiled_copy_svg(tiled_copy, tiler_mn_viz, "tiled_tv_layout.svg")
    cute.printf("tv_layout: {}\n", tv_layout)
    cute.printf(
        "Note: Visualization uses tiler_mn={} (reduced from {}) for better canvas size\n",
        tiler_mn_viz,
        tiler_mn,
    )

    # Create identity tensor and tile it (same as in kernel)
    shape = (M, N)
    idX = cute.make_identity_tensor(shape)

    # Simulate local_tile with bidx=0, cluster_y=0
    bidx = 0
    cluster_y = 0
    cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

    # Get thread slice and partition (same as in kernel)
    # For visualization, we'll use tidx=0 as an example
    tidx = 0
    thr_copy_X = tiled_copy.get_slice(tidx)

    # Calculate tXcX (same as in kernel)
    # First get the full partition, then slice it
    tXcX_full = thr_copy_X.partition_S(cX)
    tXcX = tXcX_full[(0, None), None, None]

    # Print tXcX layout information for debugging
    cute.printf("\n=== tXcX Layout Information ===\n")
    cute.printf("tXcX_full: {}\n", tXcX_full)
    cute.printf("tXcX: {}\n", tXcX)
    cute.printf("tXcX shape: {}\n", tXcX.shape)
    cute.printf("tXcX stride: {}\n", tXcX.stride)

    # Manually reconstruct the layout from shape and stride for visualization
    # From output: shape=(1,1,16), stride=(0,0,128@1)
    # The stride (0,0,128@1) means: first two dims have stride 0, third has stride 128
    # Let's create a layout with shape (1,1,16) and stride (0,0,128)
    tXcX_shape_recon = (4, 1, 64)
    tXcX_stride_recon = (2, 0, 32)
    tXcX_layout_recon = cute.make_layout(
        tXcX_shape_recon, stride=tXcX_stride_recon
    )

    # Visualize the reconstructed layout
    render_layout_svg(tXcX_layout_recon, "tXcX_layout.svg")
    cute.printf("tXcX layout visualization saved to tXcX_layout.svg\n")
    cute.printf("Reconstructed layout: {}\n", tXcX_layout_recon)

    # Create tApA layout (predicate tensor layout) similar to predicate_k function
    # tAcA is tXcX_full (the full partition before slicing)
    tAcA = tXcX_full

    # Calculate sizes for each mode
    # mode=[0,1] means the combined size of mode 0 and mode 1
    # mode=[1] means the size of mode 1
    # mode=[2] means the size of mode 2
    tAcA_size_mode01 = cute.size(tAcA, mode=[0, 1])
    tAcA_size_mode1 = cute.size(tAcA, mode=[1])
    tAcA_size_mode2 = cute.size(tAcA, mode=[2])

    # Create tApA layout with shape (size_mode01, size_mode1, size_mode2)
    # and stride (size_mode2, 0, 1)
    # This is the same as in predicate_k function
    tApA_shape = (tAcA_size_mode01, tAcA_size_mode1, tAcA_size_mode2)
    tApA_stride = (tAcA_size_mode2, 0, 1)
    tApA_layout = cute.make_layout(tApA_shape, stride=tApA_stride)

    # Print tApA layout information
    cute.printf("\n=== tApA Layout Information ===\n")
    cute.printf("tAcA (tXcX_full) shape: {}\n", tAcA.shape)
    cute.printf("tAcA (tXcX_full) full info: {}\n", tAcA)
    cute.printf("tAcA size mode[0,1]: {}\n", tAcA_size_mode01)
    cute.printf("tAcA size mode[1]: {}\n", tAcA_size_mode1)
    cute.printf("tAcA size mode[2]: {}\n", tAcA_size_mode2)
    cute.printf("tApA shape: {}\n", tApA_shape)
    cute.printf("tApA stride: {}\n", tApA_stride)
    cute.printf("tApA layout: {}\n", tApA_layout)

    # Explanation of tApA layout:
    # - Shape: (size_mode01, size_mode1, size_mode2) = (1, 1, 16) for a single thread
    # - Stride: (size_mode2, 0, 1) = (16, 0, 1)
    # - The stride[1]=0 means broadcasting: all rows share the same predicate values
    # - The stride[0]=16 means vector skipping: one predicate per vectorized group
    # - Since shape[0]=1 and shape[1]=1, we only see the K dimension (16 elements)
    # - This is correct! For a single thread, tApA is effectively 1D with 16 predicate values

    cute.printf("tApA total size: {}\n", cute.size(tApA_layout))
    cute.printf("\n=== tApA Layout Explanation ===\n")
    cute.printf("tApA has shape (1, 1, 16) for a single thread.\n")
    cute.printf(
        "  - Dimension 0 (size=1): Vector/row grouping (collapsed for single thread)\n"
    )
    cute.printf(
        "  - Dimension 1 (size=1): Row dimension with stride=0 (broadcasting)\n"
    )
    cute.printf(
        "  - Dimension 2 (size=16): K dimension with stride=1 (16 predicate values)\n"
    )
    cute.printf(
        "  - The visualization shows [0,0,:] which is the 16 predicate values.\n"
    )
    cute.printf("  - Stride=(16,0,1) design allows:\n")
    cute.printf(
        "    * Vector skipping: stride[0]=16 skips vectorized elements\n"
    )
    cute.printf(
        "    * Row broadcasting: stride[1]=0 shares predicates across rows\n"
    )
    cute.printf(
        "    * K iteration: stride[2]=1 for sequential K dimension access\n"
    )

    # Visualize tApA layout
    # Note: tApA has shape (1, 1, 16), which means it's effectively a 1D array
    # of 16 predicate values. The visualization shows [0,0,:] which is correct.
    #
    # To better understand the layout structure, we can create a 2D view
    # by reshaping to show the K dimension more clearly.
    # Since shape[0]=1 and shape[1]=1, we can flatten to 1D for visualization
    # or create a 2D view that shows the stride pattern.

    # Create a 2D view: (1, 16) to show the K dimension more clearly
    # This preserves the stride information while making it easier to visualize
    tApA_2d_shape = (tAcA_size_mode01 * tAcA_size_mode1, tAcA_size_mode2)
    tApA_2d_stride = (
        tAcA_size_mode2,
        1,
    )  # stride[0] = 16 (from original stride[0]), stride[1] = 1
    tApA_2d_layout = cute.make_layout(tApA_2d_shape, stride=tApA_2d_stride)

    # Visualize the 2D layout (which is effectively 1x16)
    render_layout_svg(tApA_2d_layout, "tApA_layout.svg")
    cute.printf(
        "tApA layout visualization (2D view: 1x16) saved to tApA_layout.svg\n"
    )
    cute.printf(
        "tApA 2D layout shape: {}, stride: {}\n", tApA_2d_shape, tApA_2d_stride
    )
    cute.printf("\n=== Understanding the tApA Visualization ===\n")
    cute.printf("The visualization shows a 1x16 array of predicate values.\n")
    cute.printf(
        "Each value (index 0-15) is a boolean that controls whether a vectorized\n"
    )
    cute.printf(
        "memory access in the K dimension is valid (within matrix boundary N).\n"
    )
    cute.printf("The stride pattern (16,1) means:\n")
    cute.printf(
        "  - stride[0]=16: Vector skipping (one predicate per 16-element vector)\n"
    )
    cute.printf("  - stride[1]=1: Sequential K dimension access\n")

    # Also visualize the original 3D layout for reference
    render_layout_svg(tApA_layout, "tApA_layout_3d.svg")
    cute.printf(
        "tApA layout visualization (3D view: 1x1x16) saved to tApA_layout_3d.svg\n"
    )
    cute.printf("\n=== Understanding the 3D Layout Visualization [0,:,:] ===\n")
    cute.printf("The visualization shows [0,:,:] which means:\n")
    cute.printf("  - [0] in dimension 0: Only one value (size=1)\n")
    cute.printf(
        "  - [:] in dimension 1: All values, but only 1 (size=1, stride=0 for broadcasting)\n"
    )
    cute.printf("  - [:] in dimension 2: All 16 values (size=16, stride=1)\n")
    cute.printf(
        "So [0,:,:] is actually [0,0,:] showing the 16 predicate values.\n"
    )
    cute.printf(
        "This is the same as the 2D view, just with an extra broadcasting dimension.\n"
    )

    # Visualize reduction_buffer layout
    # Based on _get_reduction_buffer_layout method
    # Calculate parameters same as in the method
    num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
    warps_per_row = (
        num_warps
        if cute.rank(tv_layout.shape[0]) == 1
        else max(tv_layout.shape[0][0] // cute.arch.WARP_SIZE, 1)
    )

    # For visualization, we'll use stage=2 (double buffering) to show the structure better
    # In practice, this could be 1 or 2 for double buffering
    stage = 2

    # Create reduction_buffer layout
    # Shape: (num_warps // warps_per_row, (warps_per_row, cluster_n), stage)
    # Order: (1, 0, 2) - this reorders the dimensions
    reduction_buffer_shape = (
        num_warps // warps_per_row,
        (warps_per_row, cluster_n),
        stage,
    )
    reduction_buffer_layout = cute.make_ordered_layout(
        reduction_buffer_shape, order=(1, 0, 2)
    )

    # Print reduction_buffer layout information
    cute.printf("\n=== Reduction Buffer Layout Information ===\n")
    cute.printf("tv_layout shape[0]: {}\n", tv_layout.shape[0])
    cute.printf("tv_layout rank: {}\n", cute.rank(tv_layout.shape[0]))
    cute.printf("num_warps: {}\n", num_warps)
    cute.printf("warps_per_row: {}\n", warps_per_row)
    cute.printf("cluster_n: {}\n", cluster_n)
    cute.printf("stage: {}\n", stage)
    cute.printf("reduction_buffer_shape: {}\n", reduction_buffer_shape)
    cute.printf("reduction_buffer_layout: {}\n", reduction_buffer_layout)
    cute.printf(
        "reduction_buffer_layout shape: {}\n", reduction_buffer_layout.shape
    )
    cute.printf(
        "reduction_buffer_layout stride: {}\n", reduction_buffer_layout.stride
    )

    # Explanation of reduction_buffer layout:
    # - Dimension 0: num_warps // warps_per_row (number of warp groups)
    # - Dimension 1: (warps_per_row, cluster_n) (warps per row, cluster dimension)
    # - Dimension 2: stage (reduction stage, typically 1 or 2 for double buffering)
    # - Order (1, 0, 2) means: dimension 1 becomes the fastest changing, then 0, then 2
    cute.printf("\n=== Reduction Buffer Layout Explanation ===\n")
    cute.printf(
        "The reduction buffer is used to store intermediate reduction results.\n"
    )
    cute.printf("Layout structure:\n")
    cute.printf(
        "  - Dimension 0 (size={}): Number of warp groups (num_warps // warps_per_row)\n",
        num_warps // warps_per_row,
    )
    cute.printf(
        "  - Dimension 1 (size=({}, {})): Warps per row and cluster dimension\n",
        warps_per_row,
        cluster_n,
    )
    cute.printf(
        "  - Dimension 2 (size={}): Reduction stage (1 or 2 for double buffering)\n",
        stage,
    )
    cute.printf("  - Order (1, 0, 2): Dimension 1 is fastest, then 0, then 2\n")
    cute.printf(
        "    This ordering optimizes memory access patterns for reduction operations.\n"
    )

    # Visualize reduction_buffer layout
    # Create a 2D flattened view for better visualization
    # The 3D shape is (num_warps // warps_per_row, (warps_per_row, cluster_n), stage)
    # After order (1, 0, 2), dimension 1 becomes fastest, then 0, then 2
    # We'll flatten to show: rows = (warps_per_row * cluster_n), cols = (num_warps // warps_per_row * stage)
    num_warp_groups = num_warps // warps_per_row
    total_warps_in_row = warps_per_row * cluster_n

    # Create a 2D view that shows the structure more clearly
    # Rows: warps_per_row * cluster_n (showing all warps in a row group)
    # Cols: num_warp_groups * stage (showing all warp groups across all stages)
    reduction_buffer_2d_shape = (total_warps_in_row, num_warp_groups * stage)

    # Calculate stride based on order (1, 0, 2)
    # After reordering, the memory layout is:
    # - Fastest: dimension 1 (warps_per_row, cluster_n) - stride 1
    # - Medium: dimension 0 (num_warp_groups) - stride (warps_per_row * cluster_n)
    # - Slowest: dimension 2 (stage) - stride (num_warp_groups * warps_per_row * cluster_n)
    # So for 2D view (rows, cols):
    # - Row stride: 1 (sequential within row group)
    # - Col stride: total_warps_in_row (jump to next warp group or stage)
    reduction_buffer_2d_stride = (1, total_warps_in_row)
    reduction_buffer_2d_layout = cute.make_layout(
        reduction_buffer_2d_shape, stride=reduction_buffer_2d_stride
    )

    render_layout_svg(reduction_buffer_2d_layout, "reduction_buffer_layout.svg")
    cute.printf(
        "reduction_buffer layout visualization (2D view) saved to reduction_buffer_layout.svg\n"
    )
    cute.printf(
        "2D view shape: {} (rows={} warps per row group, cols={} warp groups × {} stages)\n",
        reduction_buffer_2d_shape,
        total_warps_in_row,
        num_warp_groups,
        stage,
    )
    cute.printf(
        "Note: The original layout is 3D with shape {} and order (1, 0, 2).\n",
        reduction_buffer_shape,
    )
    cute.printf(
        "      The 2D view shows: rows = warps in a row group, cols = warp groups × stages.\n"
    )


def main():
    print("Visualizing layouts based on kernels/02_softmax.py configuration...")
    print("M=64, N=2048, dtype=Float16, threads_per_row=16, num_threads=128")
    print()

    visualize_layouts()

    print("\nVisualization files saved:")
    print("  - tiled_tv_layout.svg (tv_layout visualization)")
    print(
        "  - tXcX_layout.svg (tXcX layout visualization - manually reconstructed)"
    )
    print("  - tApA_layout.svg (tApA predicate tensor layout - flattened)")
    print("  - tApA_layout_3d.svg (tApA predicate tensor layout - 3D view)")
    print(
        "  - reduction_buffer_layout.svg (reduction buffer layout visualization)"
    )
    print(
        "\nNote: tXcX layout was manually reconstructed from shape and stride information."
    )
    print(
        "      tApA layout is created using the same logic as predicate_k function."
    )
    print(
        "      reduction_buffer layout is created using _get_reduction_buffer_layout method."
    )


if __name__ == "__main__":
    main()
