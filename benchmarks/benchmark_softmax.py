"""
Benchmark comparing PyTorch's built-in softmax with the custom CuTe kernel.

This script benchmarks both implementations across different input sizes
and provides performance metrics including throughput and latency.
"""

import argparse
import sys
from pathlib import Path
from typing import Type

import torch
import torch.nn.functional as F

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import softmax_fwd function
from kernels.softmax import softmax_fwd

import cutlass
import cutlass.torch as cutlass_torch


try:
    from liger_kernel.transformers.functional import liger_softmax
except ImportError:
    liger_softmax = None


def verify_softmax_correctness(
    cute_out: torch.Tensor,
    pytorch_out: torch.Tensor,
    rtol: float = 1e-2,
    atol: float = 1e-3,
):
    """
    Verify correctness of softmax_fwd compared to PyTorch's softmax.
    Exit if verification fails.
    """
    # Move to CPU and convert to float32 for comparison
    cute_cpu = cute_out.cpu().to(torch.float32)
    pytorch_cpu = pytorch_out.cpu().to(torch.float32)

    # Check if outputs are close
    is_close = torch.allclose(cute_cpu, pytorch_cpu, rtol=rtol, atol=atol)

    if not is_close:
        max_diff = torch.max(torch.abs(cute_cpu - pytorch_cpu)).item()
        mean_diff = torch.mean(torch.abs(cute_cpu - pytorch_cpu)).item()
        print("\n✗ Correctness check FAILED!")
        print(f"Max difference: {max_diff:.6e}")
        print(f"Mean difference: {mean_diff:.6e}")
        print(f"Tolerance: rtol={rtol}, atol={atol}")
        sys.exit(1)

    print("✓ Correctness check PASSED")


def benchmark_function(fn, warmup_iterations=20, iterations=100):
    """
    Benchmark a function using PyTorch CUDA Events.

    Args:
        fn: Function to benchmark (should be a callable that performs CUDA operations)
        warmup_iterations: Number of warmup iterations
        iterations: Number of benchmark iterations

    Returns:
        Average execution time in milliseconds
    """
    # Warmup
    for _ in range(warmup_iterations):
        fn()

    # Synchronize before timing
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iterations):
        fn()
    end_event.record()

    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)

    return elapsed_time_ms / iterations


def run_softmax(
    M,
    N,
    dtype: Type[cutlass.Numeric],
    warmup_iterations=10,
    iterations=1000,
):
    if not torch.cuda.is_available():
        raise RuntimeError("Ampere GPU is required to run this example!")

    print(f"Tensor dimensions: [{M}, {N}]")
    print(f"Input and Output Data type: {dtype}")

    torch_dtype = cutlass_torch.dtype(dtype)

    device = "cuda"
    x = 0.1 * torch.randn(M, N, device=device, dtype=torch_dtype)

    print("Input tensor shapes:")
    print(f"x: {x.shape}, dtype: {x.dtype}")

    # Verify correctness before benchmarking
    print("\nVerifying Correctness...")
    cute_out = softmax_fwd(x)
    pytorch_out = F.softmax(x, dim=-1)
    verify_softmax_correctness(cute_out, pytorch_out)

    # Benchmark CuTe softmax
    print("\n[CuTe Softmax]")
    fn_cute = lambda: softmax_fwd(x)
    avg_time_cute = benchmark_function(fn_cute, warmup_iterations, iterations)
    mem_bw_cute = round(
        2 * x.numel() * dtype.width // 8 / (avg_time_cute / 1000) / 1e9
    )
    print(f"Kernel execution time: {avg_time_cute:.4f} ms")
    print(f"Mem throughput: {mem_bw_cute:.2f} GB/s")

    # Benchmark PyTorch compiled softmax
    print("\n[PyTorch Compiled Softmax]")
    compiled_func_ref = torch.compile(lambda x: F.softmax(x, dim=-1))
    fn_pytorch = lambda: compiled_func_ref(x)
    avg_time_pytorch = benchmark_function(
        fn_pytorch, warmup_iterations, iterations
    )
    mem_bw_pytorch = round(
        2 * x.numel() * dtype.width // 8 / (avg_time_pytorch / 1000) / 1e9
    )
    print(f"Kernel execution time: {avg_time_pytorch:.4f} ms")
    print(f"Mem throughput: {mem_bw_pytorch:.2f} GB/s")

    # Benchmark Liger softmax if available
    avg_time_liger = None
    mem_bw_liger = None
    liger_error = None
    if liger_softmax is not None:
        print("\n[Liger Softmax]")
        try:
            fn_liger = lambda: liger_softmax(x)
            # Test run to catch unsupported shapes before benchmarking
            fn_liger()
            avg_time_liger = benchmark_function(
                fn_liger, warmup_iterations, iterations
            )
            mem_bw_liger = round(
                2 * x.numel() * dtype.width // 8 / (avg_time_liger / 1000) / 1e9
            )
            print(f"Kernel execution time: {avg_time_liger:.4f} ms")
            print(f"Mem throughput: {mem_bw_liger:.2f} GB/s")
        except RuntimeError as e:
            liger_error = str(e)
            print(f"⚠ Liger Softmax unsupported for this shape: {e}")

    return {
        "M": M,
        "N": N,
        "cute": {"latency_ms": avg_time_cute, "mem_bw_gbs": mem_bw_cute},
        "pytorch": {
            "latency_ms": avg_time_pytorch,
            "mem_bw_gbs": mem_bw_pytorch,
        },
        "liger": {
            "latency_ms": avg_time_liger,
            "mem_bw_gbs": mem_bw_liger,
            "error": liger_error,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark softmax forward and backward passes"
    )
    # parser.add_argument("--M", default=4096, type=int)
    # parser.add_argument("--N", default=16384, type=int)
    parser.add_argument(
        "--dtype",
        type=cutlass.dtype,
        choices=[cutlass.BFloat16, cutlass.Float16, cutlass.Float32],
        default=cutlass.Float16,
    )
    parser.add_argument("--warmup_iterations", default=20, type=int)
    parser.add_argument("--iterations", default=100, type=int)

    args = parser.parse_args()
    torch.manual_seed(0)

    # ----------------------------------------------------------------
    # Shape groups designed to exercise every kernel code path:
    #
    #   threads_per_row thresholds:
    #     N ≤ 64   → 8,   N ≤ 128  → 16,  N ≤ 3072  → 32,
    #     N ≤ 6144 → 64,  N ≤ 16384 → 128, N > 16384 → 256
    #
    #   num_threads: N ≤ 16384 → 128,  N > 16384 → 256
    #
    #   cluster_n (fp16, width=16):
    #     N ≤ 16K → 1,  N ≤ 32K → 2,  N ≤ 64K → 4,
    #     N ≤ 128K → 8, N > 128K → 16
    # ----------------------------------------------------------------

    MN_pairs = [
        # --- Small N: intra-warp reduction only (threads_per_row ≤ WARP_SIZE) ---
        # (32768, 64),  # threads_per_row=8
        # (32768, 128),  # threads_per_row=16
        # (32768, 256),  # threads_per_row=32
        # --- Medium N: multi-warp block reduction ---
        (32768, 1024),  # threads_per_row=32
        (32768, 2048),  # threads_per_row=32
        (32768, 4096),  # threads_per_row=64
        (32768, 6144),  # threads_per_row=64, boundary
        # --- Large N: many warps, single cluster (cluster_n=1) ---
        (16384, 8192),  # threads_per_row=128, num_threads=128
        (8192, 16384),  # threads_per_row=128, num_threads=128, boundary
        # --- Very large N: 256 threads, cluster_n=1→2 transition ---
        (4096, 16384),  # cluster_n=1 (fp16)
        (4096, 32768),  # cluster_n=2 (fp16)
        # --- Cluster scaling ---
        (4096, 65536),  # cluster_n=4 (fp16)
        (4096, 65536),  # cluster_n=4 (fp16)
        (4096, 131072),  # cluster_n=8 (fp16)
        # --- Typical Transformer shapes ---
        (4096, 8192),  # common attention head dim
        (8192, 8192),
        (16384, 16384),  # large square
        # --- Non-power-of-2 N: OOB / boundary handling ---
        # (8192, 1000),  # not aligned to any tile
        # (4096, 12288),  # 3 * 4096, common in FFN
        # (4096, 14336),  # LLaMA FFN intermediate size
    ]

    results = []
    for M, N in MN_pairs:
        print("\n" + "-" * 80)
        result = run_softmax(
            M,
            N,
            dtype=args.dtype,
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations,
        )
        results.append(result)

    # Print summary table
    print("\n" + "=" * 120)
    print("Performance Summary Table")
    print("=" * 120)

    has_liger = liger_softmax is not None

    if has_liger:
        header = f"{'[M, N]':<20} {'Ours':<35} {'torch.compile':<35} {'Liger Kernel':<35}"
        sub_header = f"{'':<20} {'Latency(ms) / BW (GB/s)':<35} {'Latency(ms) / BW (GB/s)':<35} {'Latency(ms) / BW (GB/s)':<35}"
        sep_width = 125
    else:
        header = f"{'[M, N]':<20} {'Ours':<35} {'torch.compile':<35}"
        sub_header = f"{'':<20} {'Latency(ms) / BW (GB/s)':<35} {'Latency(ms) / BW (GB/s)':<35}"
        sep_width = 90

    print(header)
    print("-" * sep_width)
    print(sub_header)
    print("-" * sep_width)

    for r in results:
        mn_str = f"[{r['M']}, {r['N']}]"
        cute_str = (
            f"{r['cute']['latency_ms']:.4f} / {r['cute']['mem_bw_gbs']:.2f}"
        )
        pytorch_str = f"{r['pytorch']['latency_ms']:.4f} / {r['pytorch']['mem_bw_gbs']:.2f}"
        if has_liger:
            if r["liger"].get("error"):
                liger_str = "unsupported"
            elif r["liger"]["latency_ms"] is not None:
                liger_str = f"{r['liger']['latency_ms']:.4f} / {r['liger']['mem_bw_gbs']:.2f}"
            else:
                liger_str = "N/A"
            print(
                f"{mn_str:<20} {cute_str:<35} {pytorch_str:<35} {liger_str:<35}"
            )
        else:
            print(f"{mn_str:<20} {cute_str:<35} {pytorch_str:<35}")

    print("=" * sep_width)
