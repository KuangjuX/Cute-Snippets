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

# Try to import liger_kernel
try:
    from liger_kernel.transformers import LigerSoftmax

    LIGER_KERNEL_AVAILABLE = True
except ImportError:
    LIGER_KERNEL_AVAILABLE = False
    print(
        "WARNING: liger_kernel not available. Install with: pip install liger-kernel"
    )


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


def benchmark_liger_softmax(
    x: torch.Tensor,
    num_warmup: int = 10,
    num_iterations: int = 100,
):
    """Benchmark Liger-Kernel's softmax implementation."""
    if not LIGER_KERNEL_AVAILABLE:
        raise ImportError("liger_kernel is not available")

    # Create LigerSoftmax instance
    liger_softmax = LigerSoftmax()

    # Warmup
    for _ in range(num_warmup):
        _ = liger_softmax(x)

    # Synchronize before timing
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iterations):
        out = liger_softmax(x)
    end_event.record()

    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)

    return out, elapsed_time_ms / num_iterations


def verify_correctness(
    pytorch_out: torch.Tensor,
    cute_out: torch.Tensor,
    liger_out: torch.Tensor = None,
    rtol: float = 1e-2,
    atol: float = 1e-3,
):
    """Verify that implementations produce similar results."""
    pytorch_out = pytorch_out.cpu().to(torch.float32)
    cute_out = cute_out.cpu().to(torch.float32)

    results = {}

    # Compare CuTe with PyTorch
    max_diff_cute = torch.max(torch.abs(pytorch_out - cute_out)).item()
    mean_diff_cute = torch.mean(torch.abs(pytorch_out - cute_out)).item()
    is_close_cute = torch.allclose(pytorch_out, cute_out, rtol=rtol, atol=atol)
    results["cute"] = {
        "is_close": is_close_cute,
        "max_diff": max_diff_cute,
        "mean_diff": mean_diff_cute,
    }

    # Compare Liger with PyTorch if available
    if liger_out is not None:
        liger_out = liger_out.cpu().to(torch.float32)
        max_diff_liger = torch.max(torch.abs(pytorch_out - liger_out)).item()
        mean_diff_liger = torch.mean(torch.abs(pytorch_out - liger_out)).item()
        is_close_liger = torch.allclose(
            pytorch_out, liger_out, rtol=rtol, atol=atol
        )
        results["liger"] = {
            "is_close": is_close_liger,
            "max_diff": max_diff_liger,
            "mean_diff": mean_diff_liger,
        }

    return results


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

    # Benchmark Liger-Kernel if available
    liger_out = None
    liger_time_ms = None
    if LIGER_KERNEL_AVAILABLE:
        try:
            print("\n[Liger-Kernel Softmax]")
            liger_out, liger_time_ms = benchmark_liger_softmax(
                x, num_warmup=num_warmup, num_iterations=num_iterations
            )
            print(f"  Latency: {liger_time_ms:.4f} ms")
            print(f"  Throughput: {format_throughput(M * N, liger_time_ms)}")
        except Exception as e:
            print(
                f"  ERROR: Failed to benchmark Liger-Kernel: {e}"
            )  # noqa: BLE001

    # Compare performance
    # print("\n[Comparison]")
    cute_speedup = pytorch_time_ms / cute_time_ms
    # print(
    #     f"  CuTe vs PyTorch: {cute_speedup:.2f}x ({'CuTe' if cute_speedup > 1 else 'PyTorch'} is faster)"
    # )

    # if liger_time_ms is not None:
    #     cute_vs_liger = liger_time_ms / cute_time_ms
    #     print(
    #         f"  CuTe vs Liger: {cute_vs_liger:.2f}x ({'CuTe' if cute_vs_liger > 1 else 'Liger'} is faster)"
    #     )

    # Verify correctness
    if verify:
        print("\n[Correctness Check]")
        results = verify_correctness(pytorch_out, cute_out, liger_out)

        # Verify correctness
        cute_result = results["cute"]
        print(f"  CuTe - Max difference: {cute_result['max_diff']:.6f}")
        print(f"  CuTe - Mean difference: {cute_result['mean_diff']:.6f}")
        if cute_result["is_close"]:
            print("  ✓ CuTe results match (rtol=1e-2, atol=1e-3)")
        else:
            print("  ✗ CuTe results differ (rtol=1e-2, atol=1e-3)")

    # Liger results
    # if "liger" in results:
    # liger_result = results["liger"]
    # print(f"  Liger - Max difference: {liger_result['max_diff']:.6f}")
    # print(f"  Liger - Mean difference: {liger_result['mean_diff']:.6f}")
    # if liger_result["is_close"]:
    #     print("  ✓ Liger results match (rtol=1e-2, atol=1e-3)")
    # else:
    #     print("  ✗ Liger results differ (rtol=1e-2, atol=1e-3)")

    result = {
        "M": M,
        "N": N,
        "pytorch_time_ms": pytorch_time_ms,
        "cute_time_ms": cute_time_ms,
        "cute_speedup": cute_speedup,
    }

    if liger_time_ms is not None:
        result["liger_time_ms"] = liger_time_ms
        result["cute_vs_liger"] = liger_time_ms / cute_time_ms

    return result


def main():
    """Main benchmark function."""
    print("=" * 80)
    print("Softmax Benchmark: PyTorch vs CuTe Kernel vs Liger-Kernel")
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
        (64, 16384),
        (128, 16384),
        (256, 16384),
        (512, 16384),
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
        except Exception as e:  # noqa: BLE001
            print(f"\nERROR: Failed to benchmark M={M}, N={N}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")

    if LIGER_KERNEL_AVAILABLE and any("liger_time_ms" in r for r in results):
        print(
            f"{'M':<8} {'N':<8} {'PyTorch (ms)':<15} {'CuTe (ms)':<15} {'Liger (ms)':<15} "
            f"{'CuTe vs PT':<12} {'CuTe vs Liger':<15}"
        )
        print("-" * 95)
        for r in results:
            if "liger_time_ms" in r:
                print(
                    f"{r['M']:<8} {r['N']:<8} {r['pytorch_time_ms']:<15.4f} "
                    f"{r['cute_time_ms']:<15.4f} {r['liger_time_ms']:<15.4f} "
                    f"{r['cute_speedup']:<12.2f}x {r['cute_vs_liger']:<15.2f}x"
                )
            else:
                print(
                    f"{r['M']:<8} {r['N']:<8} {r['pytorch_time_ms']:<15.4f} "
                    f"{r['cute_time_ms']:<15.4f} {'N/A':<15} "
                    f"{r['cute_speedup']:<12.2f}x {'N/A':<15}"
                )
    else:
        print(
            f"{'M':<8} {'N':<8} {'PyTorch (ms)':<15} {'CuTe (ms)':<15} {'Speedup':<10}"
        )
        print("-" * 80)
        for r in results:
            print(
                f"{r['M']:<8} {r['N']:<8} {r['pytorch_time_ms']:<15.4f} "
                f"{r['cute_time_ms']:<15.4f} {r['cute_speedup']:<10.2f}x"
            )

    avg_cute_speedup = (
        sum(r["cute_speedup"] for r in results) / len(results) if results else 0
    )
    print(f"\nAverage CuTe Speedup vs PyTorch: {avg_cute_speedup:.2f}x")

    liger_results = [r for r in results if "cute_vs_liger" in r]
    if liger_results:
        avg_cute_vs_liger = sum(
            r["cute_vs_liger"] for r in liger_results
        ) / len(liger_results)
        print(
            f"Average CuTe vs Liger: {avg_cute_vs_liger:.2f}x ({'CuTe' if avg_cute_vs_liger > 1 else 'Liger'} is faster on average)"
        )


if __name__ == "__main__":
    main()
