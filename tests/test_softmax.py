"""
Test for Softmax kernel implementation.
"""

import torch

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from kernels.softmax import softmax_fwd


def run_softmax_case(M: int, N: int) -> bool:
    x = torch.randn(M, N, dtype=torch.float16, device="cuda")
    out = softmax_fwd(x)

    # Check results
    out_cpu = out.cpu()
    # Safe softmax reference: subtract max to avoid overflow
    x_fp32 = x.cpu().to(torch.float32)
    x_shifted = x_fp32 - x_fp32.max(dim=1, keepdim=True).values
    ref = torch.exp(x_shifted)
    ref = ref / ref.sum(dim=1, keepdim=True)
    ref = ref.to(torch.float16)

    max_diff = torch.max(torch.abs(ref - out_cpu)).item()
    print(f"Max difference: {max_diff}")
    return max_diff < 1e-2


def test_softmax():
    """Test Softmax kernel using softmax_fwd function."""
    print("\n" + "=" * 60)
    print("Test: Cute-Snippets Softmax")
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
            (128, 256),
            (256, 2048),
            (64, 8192),
            (32, 32768),  # larger N triggers cluster auto-selection
        ]

        all_ok = True
        for M, N in test_cases:
            print(f"\nRunning softmax kernel: M={M}, N={N}")
            case_ok = run_softmax_case(M, N)
            if case_ok:
                print("✓ Case PASSED")
            else:
                print("✗ Case FAILED")
                all_ok = False

        if all_ok:
            print("✓ Cute-Snippets Softmax PASSED")
            return True
        print("✗ Cute-Snippets Softmax FAILED")
        return False

    except Exception as e:
        print(f"✗ Cute-Snippets Softmax FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_softmax()
