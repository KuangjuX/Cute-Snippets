# Copyright (c) 2025 - 2026 Cute-Snippets Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Cute-Snippets: Tiled Copy Example

The tiling and warp/thread layouts can be customized.

Run:
    python kernels/00_tiled_copy.py
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.utils as utils
import cuda.bindings.driver as cuda

class TiledCopyExample:
    def __init__(self, m: int, n: int, tile_m: int, tile_n: int):
        self.m = m
        self.n = n
        self.tile_m = tile_m
        self.tile_n = tile_n

        self.threads_per_cta = 32  # 1 warp (32 threads) per CTA

        self.smem_layout = None
        self.smem_cosize = None
        self.shared_storage = None

    def _setup_smem_layout(self):
        """Setup shared memory layout for the tile."""
        self.smem_layout = cute.make_layout(
            (self.tile_m, self.tile_n),
            stride=(self.tile_n, 1)
        )
        self.smem_cosize = cute.cosize(self.smem_layout)

    @cute.jit
    def __call__(self, gA: cute.Tensor, gB: cute.Tensor, stream):
        """Host: Prepare copy atoms, layouts, tiled copies, and launch the kernel."""
        self._setup_smem_layout()

        copy_op = cute.nvgpu.CopyUniversalOp()
        g2s_atom = cute.make_copy_atom(copy_op, cutlass.Float16, num_bits_per_copy=64)
        s2g_atom = cute.make_copy_atom(copy_op, cutlass.Float16, num_bits_per_copy=64)

        smem_cosize = self.smem_cosize

        @cute.struct
        class SharedStorage:
            smem: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float16, smem_cosize], 128
            ]
        self.shared_storage = SharedStorage

        thr_layout = cute.make_layout((8, 4), stride=(4, 1))
        val_layout = cute.make_layout((1, 4), stride=(4, 1))
        tiled_g2s_copy = cute.make_tiled_copy_tv(g2s_atom, thr_layout, val_layout)
        tiled_s2g_copy = cute.make_tiled_copy_tv(s2g_atom, thr_layout, val_layout)

        self.kernel(
            gA,
            gB,
            tiled_g2s_copy,
            tiled_s2g_copy
        ).launch(
            grid=(1, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        tiled_g2s_copy: cute.TiledCopy,
        tiled_s2g_copy: cute.TiledCopy,
    ):
        """
        Device: Tiled copy (Global <-> Shared) using CuTe primitives.
        """
        tidx, _, _ = cute.arch.thread_idx()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA = storage.smem.get_tensor(
            cute.make_layout((self.tile_m, self.tile_n), stride=(self.tile_n, 1))
        )
        tile_shape = (self.tile_m, self.tile_n)

        for i in range(self.m // self.tile_m):
            for j in range(self.n // self.tile_n):
                local_gA = cute.local_tile(gA, tile_shape, coord=(i, j))
                local_gB = cute.local_tile(gB, tile_shape, coord=(i, j))

                # Per-thread slices for this CTA
                thr_g2s_copy = tiled_g2s_copy.get_slice(tidx)
                thr_s2g_copy = tiled_s2g_copy.get_slice(tidx)

                tAgA = thr_g2s_copy.partition_S(local_gA)
                tAsA = thr_g2s_copy.partition_D(sA)
                tBsB = thr_s2g_copy.partition_S(sA)
                tBgB = thr_s2g_copy.partition_D(local_gB)

                # Global -> Shared (TMA load)
                cute.copy(thr_g2s_copy, tAgA, tAsA)

                # Shared -> Global (TMA store)
                cute.copy(thr_s2g_copy, tBsB, tBgB)

def test_tiled_copy():
    """Test tiled copy using CuTe-Snippets (tiled copy primitives)."""
    print("\n" + "=" * 60)
    print("Test: Cute-Snippets Tiled Copy")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return True

    device = torch.cuda.get_device_properties(0)
    if device.major < 9:
        print(f"Hopper (SM90+) required, skipping on SM{device.major}{device.minor}")
        return True

    try:
        m, n = 256, 128
        tile_m, tile_n = 128, 64

        # Create input/output tensors
        input_tensor = torch.randn(m, n, dtype=torch.float16, device="cuda")
        output_tensor = torch.zeros(m, n, dtype=torch.float16, device="cuda")

        print(f"Input shape: {input_tensor.shape}")
        print(f"Input (first 4): {input_tensor.flatten()[:4]}")

        # Convert to CuTe tensors
        input_cute = from_dlpack(input_tensor, assumed_align=16)
        input_cute.element_type = cutlass.Float16

        output_cute = from_dlpack(output_tensor, assumed_align=16)
        output_cute.element_type = cutlass.Float16

        # Create CUDA stream
        err, stream = cuda.cuStreamCreate(0)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Failed to create stream: {err}")

        print("Running tiled copy kernel...")

        example = TiledCopyExample(m, n, tile_m, tile_n)
        example(input_cute, output_cute, stream)

        # Sync and cleanup
        cuda.cuStreamSynchronize(stream)
        cuda.cuStreamDestroy(stream)

        # ---- Verification ----
        output_tensor_cpu = output_tensor.cpu()
        input_tensor_cpu = input_tensor.cpu()
        print(f"Output (first 4): {output_tensor_cpu.flatten()[:4]}")
        max_diff = torch.max(torch.abs(input_tensor_cpu - output_tensor_cpu)).item()
        print(f"Max difference: {max_diff}")

        if max_diff < 1e-3:
            print("✓ Cute-Snippets tiled copy PASSED")
            return True
        else:
            print(f"✗ Cute-Snippets tiled copy FAILED: max_diff={max_diff}")
            return False

    except Exception as e:
        print(f"✗ Cute-Snippets tiled copy FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_tiled_copy()
