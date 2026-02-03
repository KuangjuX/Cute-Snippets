"""
Benchmark comparing PyTorch's built-in softmax with the custom CuTe kernel.

This script benchmarks both implementations across different input sizes
and provides performance metrics including throughput and latency.
"""

import torch
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import softmax_fwd function
from kernels.softmax import softmax_fwd


def benchmark_pytorch_softmax(
    x: torch.Tensor, num_warmup: int = 10, num_iterations: int = 100
):
    """Benchmark PyTorch's built-in softmax implementation."""
    # Warmup
    for _ in range(num_warmup):
        _ = torch.softmax(x, dim=-1)

    # Synchronize before timing
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iterations):
        out = torch.softmax(x, dim=-1)
    end_event.record()

    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)

    return out, elapsed_time_ms / num_iterations


def benchmark_cute_softmax(
    x: torch.Tensor,
    num_warmup: int = 10,
    num_iterations: int = 100,
):
    """Benchmark the custom CuTe softmax kernel using softmax_fwd."""
    # Warmup
    for _ in range(num_warmup):
        _ = softmax_fwd(x)

    # Synchronize before timing
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iterations):
        out = softmax_fwd(x)
    end_event.record()

    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)

    return out, elapsed_time_ms / num_iterations


def verify_correctness(
    pytorch_out: torch.Tensor,
    cute_out: torch.Tensor,
    rtol: float = 1e-2,
    atol: float = 1e-3,
):
    """Verify that both implementations produce similar results."""
    pytorch_out = pytorch_out.cpu().to(torch.float32)
    cute_out = cute_out.cpu().to(torch.float32)

    max_diff = torch.max(torch.abs(pytorch_out - cute_out)).item()
    mean_diff = torch.mean(torch.abs(pytorch_out - cute_out)).item()

    is_close = torch.allclose(pytorch_out, cute_out, rtol=rtol, atol=atol)

    return is_close, max_diff, mean_diff


def format_throughput(num_elements: int, time_ms: float):
    """Format throughput in elements per second."""
    time_s = time_ms / 1000.0
    throughput = num_elements / time_s
    if throughput >= 1e9:
        return f"{throughput / 1e9:.2f} G elements/s"
    elif throughput >= 1e6:
        return f"{throughput / 1e6:.2f} M elements/s"
    else:
        return f"{throughput / 1e3:.2f} K elements/s"


def run_benchmark(
    M: int,
    N: int,
    dtype: torch.dtype = torch.float16,
    num_warmup: int = 10,
    num_iterations: int = 100,
    verify: bool = True,
):
    """Run benchmark for a specific input size."""
    print(f"\n{'='*80}")
    print(f"Benchmark: M={M}, N={N}, dtype={dtype}")
    print(f"{'='*80}")

    # Create input tensor
    x = torch.randn(M, N, dtype=dtype, device="cuda")

    # Benchmark PyTorch
    print("\n[PyTorch Softmax]")
    pytorch_out, pytorch_time_ms = benchmark_pytorch_softmax(
        x, num_warmup=num_warmup, num_iterations=num_iterations
    )
    print(f"  Latency: {pytorch_time_ms:.4f} ms")
    print(f"  Throughput: {format_throughput(M * N, pytorch_time_ms)}")

    # Benchmark CuTe kernel
    print("\n[CuTe Softmax Kernel]")
    cute_out, cute_time_ms = benchmark_cute_softmax(
        x, num_warmup=num_warmup, num_iterations=num_iterations
    )
    print(f"  Latency: {cute_time_ms:.4f} ms")
    print(f"  Throughput: {format_throughput(M * N, cute_time_ms)}")

    # Compare performance
    speedup = pytorch_time_ms / cute_time_ms
    print(f"\n[Comparison]")
    print(
        f"  Speedup: {speedup:.2f}x ({'CuTe' if speedup > 1 else 'PyTorch'} is faster)"
    )

    # Verify correctness
    if verify:
        print(f"\n[Correctness Check]")
        is_close, max_diff, mean_diff = verify_correctness(
            pytorch_out, cute_out
        )
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        if is_close:
            print(f"  ✓ Results match (rtol=1e-2, atol=1e-3)")
        else:
            print(f"  ✗ Results differ (rtol=1e-2, atol=1e-3)")

    return {
        "M": M,
        "N": N,
        "pytorch_time_ms": pytorch_time_ms,
        "cute_time_ms": cute_time_ms,
        "speedup": speedup,
    }


def main():
    """Main benchmark function."""
    print("=" * 80)
    print("Softmax Benchmark: PyTorch vs CuTe Kernel")
    print("=" * 80)

    if not torch.cuda.is_available():
        print(
            "ERROR: CUDA not available. This benchmark requires a CUDA-capable GPU."
        )
        return

    device = torch.cuda.get_device_properties(0)
    print(f"\nGPU: {device.name}")
    print(f"Compute Capability: {device.major}.{device.minor}")

    if device.major < 9:
        print(
            f"\nWARNING: Hopper (SM90+) recommended for optimal performance. "
            f"Current: SM{device.major}{device.minor}"
        )

    # Benchmark configurations
    configs = [
        (64, 2048),
        (128, 2048),
        (256, 2048),
        (512, 2048),
        (64, 4096),
        (128, 4096),
        (256, 4096),
        (512, 4096),
        (64, 8192),
        (128, 8192),
        (256, 8192),
        (512, 8192),
    ]

    results = []

    for M, N in configs:
        try:
            result = run_benchmark(
                M=M,
                N=N,
                dtype=torch.float16,
                num_warmup=10,
                num_iterations=100,
                verify=True,
            )
            results.append(result)
        except Exception as e:
            print(f"\nERROR: Failed to benchmark M={M}, N={N}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(
        f"{'M':<8} {'N':<8} {'PyTorch (ms)':<15} {'CuTe (ms)':<15} {'Speedup':<10}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['M']:<8} {r['N']:<8} {r['pytorch_time_ms']:<15.4f} "
            f"{r['cute_time_ms']:<15.4f} {r['speedup']:<10.2f}x"
        )

    avg_speedup = (
        sum(r["speedup"] for r in results) / len(results) if results else 0
    )
    print(f"\nAverage Speedup: {avg_speedup:.2f}x")


if __name__ == "__main__":
    main()
