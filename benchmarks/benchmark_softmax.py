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
    if liger_softmax is not None:
        print("\n[Liger Softmax]")
        fn_liger = lambda: liger_softmax(x)
        avg_time_liger = benchmark_function(
            fn_liger, warmup_iterations, iterations
        )
        mem_bw_liger = round(
            2 * x.numel() * dtype.width // 8 / (avg_time_liger / 1000) / 1e9
        )
        print(f"Kernel execution time: {avg_time_liger:.4f} ms")
        print(f"Mem throughput: {mem_bw_liger:.2f} GB/s")

    return {
        "M": M,
        "N": N,
        "cute": {"latency_ms": avg_time_cute, "mem_bw_gbs": mem_bw_cute},
        "pytorch": {
            "latency_ms": avg_time_pytorch,
            "mem_bw_gbs": mem_bw_pytorch,
        },
        "liger": {"latency_ms": avg_time_liger, "mem_bw_gbs": mem_bw_liger},
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

    MN_pairs = [
        (4096, 8192),
        (4096, 16384),
        (8192, 16384),
        (16384, 16384),
    ]

    results = []
    for M, N in MN_pairs:
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

    if liger_softmax is not None:
        header = f"{'[M, N]':<15} {'Ours':<30} {'torch.compile':<30} {'Liger Kernel':<30}"
        print(header)
        print("-" * 120)
        sub_header = f"{'':<15} {'Latency(ms) / Memory Bandwidth (GB/s)':<30} {'Latency(ms) / Memory Bandwidth (GB/s)':<30} {'Latency(ms) / Memory Bandwidth (GB/s)':<30}"
        print(sub_header)
        print("-" * 120)

        for r in results:
            mn_str = f"[{r['M']}, {r['N']}]"
            cute_str = (
                f"{r['cute']['latency_ms']:.4f} / {r['cute']['mem_bw_gbs']:.2f}"
            )
            pytorch_str = f"{r['pytorch']['latency_ms']:.4f} / {r['pytorch']['mem_bw_gbs']:.2f}"
            liger_str = (
                f"{r['liger']['latency_ms']:.4f} / {r['liger']['mem_bw_gbs']:.2f}"
                if r["liger"]["latency_ms"] is not None
                else "N/A"
            )
            print(
                f"{mn_str:<15} {cute_str:<30} {pytorch_str:<30} {liger_str:<30}"
            )
    else:
        header = f"{'[M, N]':<15} {'Ours':<30} {'torch.compile':<30}"
        print(header)
        print("-" * 90)
        sub_header = f"{'':<15} {'Latency(ms) / Memory Bandwidth (GB/s)':<30} {'Latency(ms) / Memory Bandwidth (GB/s)':<30}"
        print(sub_header)
        print("-" * 90)

        for r in results:
            mn_str = f"[{r['M']}, {r['N']}]"
            cute_str = (
                f"{r['cute']['latency_ms']:.4f} / {r['cute']['mem_bw_gbs']:.2f}"
            )
            pytorch_str = f"{r['pytorch']['latency_ms']:.4f} / {r['pytorch']['mem_bw_gbs']:.2f}"
            print(f"{mn_str:<15} {cute_str:<30} {pytorch_str:<30}")

    print("=" * 120)

    # MN_pairs = [(32768, 256), (32768, 512), (32768, 1024), (32768, 2048), (32768, 4096), (32768, 8192), (32768, 16384), (32768, 32768), (32768, 65536), (16384, 131072), (8192, 262144)]
    # # MN_pairs = [(32768, 1024)]
    # results = []
    # for M, N in MN_pairs:
    #     res = run_softmax(
    #         M,
    #         N,
    #         dtype=args.dtype,
    #         warmup_iterations=args.warmup_iterations,
    #         iterations=args.iterations,
    #     )
    #     results.append(res)
    # # print(results)
    # print([x for x, _ in results])
