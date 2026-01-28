import cutlass
from cutlass import cute
from cute_viz import render_tiled_copy_svg

@cute.jit
def main():
    copy_op = cute.nvgpu.CopyUniversalOp()
    g2s_atom = cute.make_copy_atom(copy_op, cutlass.Float16, num_bits_per_copy=64)

    thr_layout = cute.make_layout((8, 4), stride=(4, 1))
    val_layout = cute.make_layout((1, 4))
    tiled_g2s_copy = cute.make_tiled_copy_tv(
        g2s_atom,
        thr_layout,
        val_layout
    )


    src_tiled_layout = tiled_g2s_copy.layout_src_tv_tiled
    dst_tiled_layout = tiled_g2s_copy.layout_dst_tv_tiled
    render_tiled_copy_svg(tiled_g2s_copy, (8, 16), "tiled_tv_layout.svg")
    print("Tiled TV layout saved to tiled_tv_layout.svg")
    cute.printf("src_tiled_layout: {}\n", src_tiled_layout)
    cute.printf("dst_tiled_layout: {}\n", dst_tiled_layout)

if __name__ == "__main__":
    main()
