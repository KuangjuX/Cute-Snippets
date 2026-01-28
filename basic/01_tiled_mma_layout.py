import cutlass
from cutlass import cute
from cute_viz import render_tiled_mma_svg 

@cute.jit
def main():
    mma_atom = cute.nvgpu.warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (16, 8, 8))

    # atom_layout_mnk = cute.make_layout((2, 2, 1))
    
    # permutations_mnk = cute.make_layout((1, 1, 1))

    # tiled_mma = cute.make_tiled_mma(
    #     mma_atom,
    #     atom_layout_mnk,
    #     permutations_mnk
    # )

    tiled_mma = cute.make_tiled_mma(mma_atom)


    svg_filename = "tiled_mma_layout.svg"
    render_tiled_mma_svg(tiled_mma, (16, 8, 8), svg_filename)
    print(f"Tiled MMA layout visualization saved to {svg_filename}")

if __name__ == "__main__":
    main()