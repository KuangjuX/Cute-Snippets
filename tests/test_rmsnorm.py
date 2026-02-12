"""
Test for RMSNorm kernel implementation.
"""

import torch

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from kernels.rmsnorm import rmsnorm_fwd


def run_rmsnorm_case(M: int, N: int, has_weight: bool = True, has_bias: bool = False, has_residual: bool = False) -> bool:
    x = torch.randn(M, N, dtype=torch.float16, device="cuda")
    weight = torch.randn(N, dtype=torch.float16, device="cuda") if has_weight else None
    bias = torch.randn(N, dtype=torch.float16, device="cuda") if has_bias else None
    residual = torch.randn(M, N, dtype=torch.float16, device="cuda") if has_residual else None
    
    out, residual_out, rstd = rmsnorm_fwd(
        x, weight=weight, bias=bias, residual=residual, eps=1e-6, store_rstd=True
    )

    # Check results
    out_cpu = out.cpu()
    rstd_cpu = rstd.cpu() if rstd is not None else None
    
    # RMSNorm reference implementation
    x_fp32 = x.cpu().to(torch.float32)
    if residual is not None:
        residual_fp32 = residual.cpu().to(torch.float32)
        x_fp32 = x_fp32 + residual_fp32
    
    # Compute RMS: sqrt(mean(x^2) + eps)
    x_sq = x_fp32 * x_fp32
    mean_sq = x_sq.mean(dim=1, keepdim=True)
    rms = torch.sqrt(mean_sq + 1e-6)
    ref = x_fp32 / rms
    
    # Apply weight and bias
    if weight is not None:
        weight_fp32 = weight.cpu().to(torch.float32)
        ref = ref * weight_fp32
    if bias is not None:
        bias_fp32 = bias.cpu().to(torch.float32)
        ref = ref + bias_fp32
    
    ref = ref.to(torch.float16)
    
    # Check output
    max_diff = torch.max(torch.abs(ref - out_cpu)).item()
    print(f"Max difference: {max_diff}")
    
    # Check rstd if available
    if rstd_cpu is not None:
        rstd_ref = 1.0 / rms.squeeze(1)  # rstd is 1 / rms
        rstd_diff = torch.max(torch.abs(rstd_ref - rstd_cpu)).item()
        print(f"Rstd max difference: {rstd_diff}")
        if rstd_diff > 1e-2:
            return False
    
    return max_diff < 1e-2


def test_rmsnorm():
    """Test RMSNorm kernel using rmsnorm_fwd function."""
    print("\n" + "=" * 60)
    print("Test: Cute-Snippets RMSNorm")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return True

    device = torch.cuda.get_device_properties(0)
    if device.major < 9:
        print(
            f"Hopper (SM90+) required, skipping on SM{device.major}{device.minor}"
        )
        return True

    try:
        test_cases = [
            (128, 256, True, False, False),   # Basic RMSNorm with weight
            (256, 2048, True, False, False),  # Larger N
            (64, 8192, True, True, False),   # With bias
            (32, 32768, True, False, True),  # With residual, larger N triggers cluster auto-selection
            (128, 1024, True, True, True),   # Full features: weight, bias, residual
        ]

        all_ok = True
        for case in test_cases:
            M, N = case[0], case[1]
            has_weight = case[2] if len(case) > 2 else True
            has_bias = case[3] if len(case) > 3 else False
            has_residual = case[4] if len(case) > 4 else False
            
            features = []
            if has_weight:
                features.append("weight")
            if has_bias:
                features.append("bias")
            if has_residual:
                features.append("residual")
            features_str = ", ".join(features) if features else "none"
            
            print(f"\nRunning RMSNorm kernel: M={M}, N={N}, features=[{features_str}]")
            case_ok = run_rmsnorm_case(M, N, has_weight, has_bias, has_residual)
            if case_ok:
                print("✓ Case PASSED")
            else:
                print("✗ Case FAILED")
                all_ok = False

        if all_ok:
            print("✓ Cute-Snippets RMSNorm PASSED")
            return True
        print("✗ Cute-Snippets RMSNorm FAILED")
        return False

    except Exception as e:
        print(f"✗ Cute-Snippets RMSNorm FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_rmsnorm()
