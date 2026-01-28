# Copyright (c) 2025 - 2026 Cute-Snippets Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Cute-Snippets: Tiled MMA Example

Matrix multiplication using CuTe tiled MMA primitives.
Computes C = A * B + C where A is M x K, B is K x N, C is M x N.

Run:
    python kernels/01_tiled_mma.py
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.utils as utils
import cuda.bindings.driver as cuda

class TiledMMAExample:
    def __init__(self, m: int, n: int, k: int, tile_m: int, tile_n: int, tile_k: int):
        self.m = m
        self.n = n
        self.k = k
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tile_k = tile_k

        self.threads_per_cta = 32  # 1 warp (32 threads) per CTA

        self.smem_layout_A = None
        self.smem_layout_B = None
        self.smem_cosize_A = None
        self.smem_cosize_B = None
        self.shared_storage = None

    def _setup_smem_layouts(self):
        """Setup shared memory layouts for tiles A and B."""
        padding = 8
        self.smem_layout_A = cute.make_layout(
            (self.tile_m, self.tile_k),
            stride=(self.tile_k + padding, 1)
        )
        self.smem_layout_B = cute.make_layout(
            (self.tile_k, self.tile_n),
            stride=(self.tile_n + padding, 1)
        )
        self.smem_cosize_A = cute.cosize(self.smem_layout_A)
        self.smem_cosize_B = cute.cosize(self.smem_layout_B)

    @cute.jit
    def __call__(self, gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, stream):
        """Host: Prepare copy atoms, MMA atom, layouts, tiled copies, and launch the kernel."""
        self._setup_smem_layouts()

        # Copy atoms for loading A and B from global to shared memory
        copy_op = cute.nvgpu.CopyUniversalOp()
        g2s_atom_A = cute.make_copy_atom(copy_op, cutlass.Float16, num_bits_per_copy=64)
        g2s_atom_B = cute.make_copy_atom(copy_op, cutlass.Float16, num_bits_per_copy=64)
        r2g_atom_C = cute.make_copy_atom(copy_op, cutlass.Float32, num_bits_per_copy=64)


        # MMA atom for computation
        mma_atom = cute.nvgpu.warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (16, 8, 8))
        tiled_mma = cute.make_tiled_mma(mma_atom)
        
        # ldmatrix_op = cute.nvgpu.warp.LdMatrix16x16x8bOp(num_matrices=1)
        ldmatrix_op = cute.nvgpu.CopyUniversalOp()
        s2r_atom = cute.make_copy_atom(ldmatrix_op, cutlass.Float16)
        s2r_copy_A = cute.make_tiled_copy_A(s2r_atom, tiled_mma)
        s2r_copy_B = cute.make_tiled_copy_B(s2r_atom, tiled_mma)

        smem_cosize_A = self.smem_cosize_A
        smem_cosize_B = self.smem_cosize_B

        @cute.struct
        class SharedStorage:
            smem_A: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float16, smem_cosize_A], 128
            ]
            smem_B: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float16, smem_cosize_B], 128
            ]
        self.shared_storage = SharedStorage

        # Thread and value layouts for tiled copy
        thr_layout = cute.make_layout((8, 4), stride=(4, 1))
        val_layout = cute.make_layout((1, 4), stride=(4, 1))
        tiled_g2s_copy_A = cute.make_tiled_copy_tv(g2s_atom_A, thr_layout, val_layout)
        tiled_g2s_copy_B = cute.make_tiled_copy_tv(g2s_atom_B, thr_layout, val_layout)
        tiled_r2g_copy_C = cute.make_tiled_copy_C(r2g_atom_C, tiled_mma)

        self.kernel(
            gA,
            gB,
            gC,
            tiled_g2s_copy_A,
            tiled_g2s_copy_B,
            tiled_r2g_copy_C,
            s2r_copy_A, 
            s2r_copy_B,
            tiled_mma
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
        gC: cute.Tensor,
        tiled_g2s_copy_A: cute.TiledCopy,
        tiled_g2s_copy_B: cute.TiledCopy,
        tiled_r2g_copy_C: cute.TiledCopy,
        s2r_copy_A: cute.TiledCopy,
        s2r_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ):
        """
        Device: Tiled MMA (Matrix Multiply-Accumulate) using CuTe primitives.
        Computes C = A * B + C
        """
        tidx, _, _ = cute.arch.thread_idx()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # swizzle = cute.Swizzle(3, 3, 3)
        
        # Shared memory tensors for A and B tiles
        padding = 8
        sA = storage.smem_A.get_tensor(
            cute.make_layout((self.tile_m, self.tile_k), stride=(self.tile_k + padding, 1)),
        )
        sB = storage.smem_B.get_tensor(
            cute.make_layout((self.tile_k, self.tile_n), stride=(self.tile_n + padding, 1)),
        )

        # Thread-local slices for copy operations
        thr_g2s_copy_A = tiled_g2s_copy_A.get_slice(tidx)
        thr_g2s_copy_B = tiled_g2s_copy_B.get_slice(tidx)

        thr_s2r_copy_A = s2r_copy_A.get_slice(tidx)
        thr_s2r_copy_B = s2r_copy_B.get_slice(tidx)
        
        # Thread-local slice for MMA operation
        thr_mma = tiled_mma.get_slice(tidx)

        # Tile shapes
        tile_shape_MK = (self.tile_m, self.tile_k)
        tile_shape_KN = (self.tile_k, self.tile_n)
        tile_shape_MN = (self.tile_m, self.tile_n)

        # Iterate over output tiles
        for i in range(self.m // self.tile_m):
            for j in range(self.n // self.tile_n):
                # Get output tile
                local_gC = cute.local_tile(gC, tile_shape_MN, coord=(i, j))
                
                # Partition C for MMA (accumulator)
                tCrC = thr_mma.partition_C(local_gC)
                rC = cute.make_fragment_like(tCrC)

                # # Initialize accumulator
                # cute.clear(rC)

                # rC = cute.zeros_like(tCrC, dtype=cutlass.Float32)

                # K-loop: accumulate over K dimension
                for k_idx in range(self.k // self.tile_k):
                    # Load A tile (M x K)
                    local_gA = cute.local_tile(gA, tile_shape_MK, coord=(i, k_idx))
                    tAgA = thr_g2s_copy_A.partition_S(local_gA)
                    tAsA = thr_g2s_copy_A.partition_D(sA)
                    cute.copy(thr_g2s_copy_A, tAgA, tAsA)

                    # Load B tile (K x N)
                    local_gB = cute.local_tile(gB, tile_shape_KN, coord=(k_idx, j))
                    tBgB = thr_g2s_copy_B.partition_S(local_gB)
                    tBsB = thr_g2s_copy_B.partition_D(sB)
                    cute.copy(thr_g2s_copy_B, tBgB, tBsB)

                    # Synchronize to ensure data is loaded before computation
                    cute.arch.sync_threads()

                    tAsA = thr_s2r_copy_A.partition_S(sA)
                    tBsB = thr_s2r_copy_B.partition_S(sB)

                    tArA = cute.make_fragment_like(tAsA)
                    tBrB = cute.make_fragment_like(tBsB)

                    cute.copy(thr_s2r_copy_A, tAsA, tArA)
                    cute.copy(thr_s2r_copy_B, tBsB, tBrB)

                    if tidx == 0:
                        cute.printf("tArA: {}\n", tArA)
                        cute.printf("tBrB: {}\n", tBrB)

                    # Perform MMA: rC = A * B + rC
                    # cute.gemm(thr_mma, rC, tArA, tBrB, rC)

                    # Synchronize before next iteration
                    # cute.arch.sync_threads()

                # Write back result
                # cute.copy(tiled_r2g_copy_C, rC, tCrC)

def test_tiled_mma():
    """Test tiled MMA using CuTe-Snippets (tiled MMA primitives)."""
    print("\n" + "=" * 60)
    print("Test: Cute-Snippets Tiled MMA")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return True

    device = torch.cuda.get_device_properties(0)
    if device.major < 9:
        print(f"Hopper (SM90+) required, skipping on SM{device.major}{device.minor}")
        return True

    try:
        m, n, k = 256, 128, 64
        tile_m, tile_n, tile_k = 128, 64, 32

        # Create input tensors
        A = torch.randn(m, k, dtype=torch.float16, device="cuda")
        B = torch.randn(k, n, dtype=torch.float16, device="cuda")
        C = torch.randn(m, n, dtype=torch.float32, device="cuda")
        C_ref = C.clone()

        print(f"Matrix shapes: A={A.shape}, B={B.shape}, C={C.shape}")
        print(f"A (first 4): {A.flatten()[:4]}")
        print(f"B (first 4): {B.flatten()[:4]}")
        print(f"C (first 4): {C.flatten()[:4]}")

        # Convert to CuTe tensors
        A_cute = from_dlpack(A, assumed_align=16)
        A_cute.element_type = cutlass.Float16

        B_cute = from_dlpack(B, assumed_align=16)
        B_cute.element_type = cutlass.Float16

        C_cute = from_dlpack(C, assumed_align=16)
        C_cute.element_type = cutlass.Float32

        # Create CUDA stream
        err, stream = cuda.cuStreamCreate(0)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Failed to create stream: {err}")

        print("Running tiled MMA kernel...")

        example = TiledMMAExample(m, n, k, tile_m, tile_n, tile_k)
        example(A_cute, B_cute, C_cute, stream)

        # Sync and cleanup
        cuda.cuStreamSynchronize(stream)
        cuda.cuStreamDestroy(stream)

        # ---- Verification ----
        # Reference computation: C_ref = A @ B + C_ref
        C_ref = C_ref + torch.matmul(A.float(), B.float())
        
        C_cpu = C.cpu()
        C_ref_cpu = C_ref.cpu()
        
        print(f"Output C (first 4): {C_cpu.flatten()[:4]}")
        print(f"Reference C (first 4): {C_ref_cpu.flatten()[:4]}")
        
        max_diff = torch.max(torch.abs(C_cpu - C_ref_cpu)).item()
        print(f"Max difference: {max_diff}")

        if max_diff < 1e-2:
            print("✓ Cute-Snippets tiled MMA PASSED")
            return True
        else:
            print(f"✗ Cute-Snippets tiled MMA FAILED: max_diff={max_diff}")
            return False

    except Exception as e:
        print(f"✗ Cute-Snippets tiled MMA FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_tiled_mma()
