"""
Benchmark comparing PyTorch's RMSNorm with the custom CuTe kernel.

This script benchmarks both implementations across different input sizes
and provides performance metrics including throughput and latency.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import rmsnorm_fwd function
from kernels.rmsnorm import rmsnorm_fwd

import cutlass
import cutlass.torch as cutlass_torch

try:
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
except ImportError:
    LigerRMSNorm = None


def rmsnorm_ref(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Reference RMSNorm implementation using PyTorch.
    """
    if residual is not None:
        x = x + residual

    # Compute RMS: sqrt(mean(x^2) + eps)
    x_sq = x * x
    mean_sq = x_sq.mean(dim=-1, keepdim=True)
    rms = torch.sqrt(mean_sq + eps)
    out = x / rms

    # Apply weight and bias
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias

    return out


def verify_rmsnorm_correctness(
    cute_out: torch.Tensor,
    pytorch_out: torch.Tensor,
    rtol: float = 1e-2,
    atol: float = 1e-3,
):
    """
    Verify correctness of rmsnorm_fwd compared to PyTorch's RMSNorm.
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


def run_rmsnorm(
    M,
    N,
    dtype: cutlass.Numeric,
    weight_dtype: Optional[cutlass.Numeric] = None,
    bias_dtype: Optional[cutlass.Numeric] = None,
    residual_dtype: Optional[cutlass.Numeric] = None,
    has_weight: bool = True,
    has_bias: bool = False,
    has_residual: bool = False,
    warmup_iterations=10,
    iterations=1000,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required to run this benchmark!")

    print(f"Tensor dimensions: [{M}, {N}]")
    print(f"Input and Output Data type: {dtype}")

    torch_dtype = cutlass_torch.dtype(dtype)
    weight_torch_dtype = (
        cutlass_torch.dtype(weight_dtype)
        if weight_dtype is not None
        else torch_dtype
    )
    bias_torch_dtype = (
        cutlass_torch.dtype(bias_dtype)
        if bias_dtype is not None
        else torch_dtype
    )
    residual_torch_dtype = (
        cutlass_torch.dtype(residual_dtype)
        if residual_dtype is not None
        else torch_dtype
    )

    device = "cuda"
    x = 0.1 * torch.randn(M, N, device=device, dtype=torch_dtype)
    weight = (
        torch.randn(N, device=device, dtype=weight_torch_dtype)
        if has_weight
        else None
    )
    bias = (
        torch.randn(N, device=device, dtype=bias_torch_dtype)
        if has_bias
        else None
    )
    residual = (
        torch.randn(M, N, device=device, dtype=residual_torch_dtype)
        if has_residual
        else None
    )

    print("Input tensor shapes:")
    print(f"x: {x.shape}, dtype: {x.dtype}")
    if weight is not None:
        print(f"weight: {weight.shape}, dtype: {weight.dtype}")
    if bias is not None:
        print(f"bias: {bias.shape}, dtype: {bias.dtype}")
    if residual is not None:
        print(f"residual: {residual.shape}, dtype: {residual.dtype}")

    eps = 1e-6

    # Verify correctness before benchmarking
    print("\nVerifying Correctness...")
    cute_out, _, _ = rmsnorm_fwd(
        x,
        weight=weight,
        bias=bias,
        residual=residual,
        eps=eps,
        store_rstd=False,
    )
    pytorch_out = rmsnorm_ref(
        x, weight=weight, bias=bias, residual=residual, eps=eps
    )
    verify_rmsnorm_correctness(cute_out, pytorch_out)

    # Calculate memory bytes for bandwidth calculation
    dtype_width = dtype.width // 8
    weight_dtype_width = (
        (weight_dtype.width // 8) if weight_dtype is not None else dtype_width
    )
    bias_dtype_width = (
        (bias_dtype.width // 8) if bias_dtype is not None else dtype_width
    )
    residual_dtype_width = (
        (residual_dtype.width // 8)
        if residual_dtype is not None
        else dtype_width
    )

    mem_bytes = 2 * x.numel() * dtype_width  # x read, out write
    if weight is not None:
        mem_bytes += weight.numel() * weight_dtype_width  # weight read
    if bias is not None:
        mem_bytes += bias.numel() * bias_dtype_width  # bias read
    if residual is not None:
        mem_bytes += (
            2 * residual.numel() * residual_dtype_width
        )  # residual read, residual_out write

    # Benchmark CuTe RMSNorm
    print("\n[CuTe RMSNorm]")
    fn_cute = lambda: rmsnorm_fwd(
        x, weight=weight, bias=bias, residual=residual, eps=eps
    )
    avg_time_cute = benchmark_function(fn_cute, warmup_iterations, iterations)
    mem_bw_cute = round(mem_bytes / (avg_time_cute / 1000) / 1e9)
    print(f"Kernel execution time: {avg_time_cute:.4f} ms")
    print(f"Mem throughput: {mem_bw_cute:.2f} GB/s")

    # Benchmark PyTorch compiled RMSNorm
    print("\n[PyTorch Compiled RMSNorm]")
    compiled_func_ref = torch.compile(rmsnorm_ref)
    fn_pytorch = lambda: compiled_func_ref(
        x, weight=weight, bias=bias, residual=residual, eps=eps
    )
    avg_time_pytorch = benchmark_function(
        fn_pytorch, warmup_iterations, iterations
    )
    mem_bw_pytorch = round(mem_bytes / (avg_time_pytorch / 1000) / 1e9)
    print(f"Kernel execution time: {avg_time_pytorch:.4f} ms")
    print(f"Mem throughput: {mem_bw_pytorch:.2f} GB/s")

    # Benchmark Liger RMSNorm if available
    avg_time_liger = None
    mem_bw_liger = None
    liger_error = None
    if LigerRMSNorm is not None:
        print("\n[Liger RMSNorm]")
        try:
            # Liger RMSNorm only supports weight, not bias or residual
            # We'll test it only when bias and residual are not used
            if has_bias or has_residual:
                liger_error = "Liger RMSNorm does not support bias or residual"
                print(f"⚠ {liger_error}")
            else:
                # Create Liger RMSNorm module
                # Liger RMSNorm only supports weight
                # Try different initialization methods for compatibility with different versions
                try:
                    # Try with elementwise_affine parameter (newer versions)
                    liger_norm = LigerRMSNorm(
                        hidden_size=N,
                        eps=eps,
                        elementwise_affine=has_weight,
                    ).to(device)
                except TypeError:
                    # Fallback: try without elementwise_affine (older versions)
                    # Liger RMSNorm always creates weight by default
                    liger_norm = LigerRMSNorm(
                        hidden_size=N,
                        eps=eps,
                    ).to(device)

                # Set weight if provided
                if has_weight and weight is not None:
                    if (
                        hasattr(liger_norm, "weight")
                        and liger_norm.weight is not None
                    ):
                        liger_norm.weight.data = weight.clone()
                # Note: If has_weight is False, Liger RMSNorm might still have a weight
                # parameter, but we'll use it as-is (initialized to ones/zeros)

                # Test run to catch unsupported shapes before benchmarking
                _ = liger_norm(x)

                fn_liger = lambda: liger_norm(x)
                avg_time_liger = benchmark_function(
                    fn_liger, warmup_iterations, iterations
                )
                mem_bw_liger = round(mem_bytes / (avg_time_liger / 1000) / 1e9)
                print(f"Kernel execution time: {avg_time_liger:.4f} ms")
                print(f"Mem throughput: {mem_bw_liger:.2f} GB/s")
        except (RuntimeError, ValueError, TypeError) as e:
            liger_error = str(e)
            print(f"⚠ Liger RMSNorm error: {e}")

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
        description="Benchmark RMSNorm forward pass"
    )
    parser.add_argument(
        "--dtype",
        type=cutlass.dtype,
        choices=[cutlass.BFloat16, cutlass.Float16, cutlass.Float32],
        default=cutlass.Float16,
    )
    parser.add_argument(
        "--weight_dtype",
        type=cutlass.dtype,
        choices=[cutlass.BFloat16, cutlass.Float16, cutlass.Float32],
        default=None,
        help="Weight dtype (defaults to input dtype)",
    )
    parser.add_argument(
        "--bias_dtype",
        type=cutlass.dtype,
        choices=[cutlass.BFloat16, cutlass.Float16, cutlass.Float32],
        default=None,
        help="Bias dtype (defaults to input dtype)",
    )
    parser.add_argument(
        "--residual_dtype",
        type=cutlass.dtype,
        choices=[cutlass.BFloat16, cutlass.Float16, cutlass.Float32],
        default=None,
        help="Residual dtype (defaults to input dtype)",
    )
    parser.add_argument(
        "--has_weight",
        action="store_true",
        default=True,
        help="Include weight tensor",
    )
    parser.add_argument(
        "--no_weight",
        dest="has_weight",
        action="store_false",
        help="Exclude weight tensor",
    )
    parser.add_argument(
        "--has_bias",
        action="store_true",
        default=False,
        help="Include bias tensor",
    )
    parser.add_argument(
        "--has_residual",
        action="store_true",
        default=False,
        help="Include residual tensor",
    )
    parser.add_argument("--warmup_iterations", default=20, type=int)
    parser.add_argument("--iterations", default=100, type=int)

    args = parser.parse_args()
    torch.manual_seed(0)

    # ----------------------------------------------------------------
    # Shape groups designed to exercise different kernel code paths:
    #
    #   Similar to softmax, RMSNorm uses ElementwiseTileConfig which
    #   has similar thresholds for threads_per_row and cluster_n
    # ----------------------------------------------------------------

    MN_pairs = [
        # --- Small N: intra-warp reduction only ---
        (32768, 256),
        (32768, 512),
        # --- Medium N: multi-warp block reduction ---
        (32768, 1024),
        (32768, 2048),
        (32768, 4096),
        (32768, 6144),
        # --- Large N: many warps, single cluster (cluster_n=1) ---
        (16384, 8192),
        (8192, 16384),
        # --- Very large N: cluster_n=1→2 transition ---
        (4096, 16384),
        (4096, 32768),  # cluster_n=2 (fp16)
        # --- Cluster scaling ---
        (4096, 65536),  # cluster_n=4 (fp16)
        (4096, 131072),  # cluster_n=8 (fp16)
        # --- Typical Transformer shapes ---
        (4096, 8192),  # common attention head dim
        (8192, 8192),
        (16384, 16384),  # large square
    ]

    results = []
    for M, N in MN_pairs:
        print("\n" + "-" * 80)
        result = run_rmsnorm(
            M,
            N,
            dtype=args.dtype,
            weight_dtype=args.weight_dtype or args.dtype,
            bias_dtype=args.bias_dtype or args.dtype,
            residual_dtype=args.residual_dtype or args.dtype,
            has_weight=args.has_weight,
            has_bias=args.has_bias,
            has_residual=args.has_residual,
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations,
        )
        results.append(result)

    # Print summary table
    print("\n" + "=" * 125)
    print("Performance Summary Table")
    print("=" * 125)

    has_liger = LigerRMSNorm is not None

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
                liger_str = r["liger"]["error"][:33]  # Truncate long errors
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
