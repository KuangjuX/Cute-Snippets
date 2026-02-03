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
        M, N = 64, 2048

        x = torch.randn(M, N, dtype=torch.float16, device="cuda")

        print("Running softmax kernel...")
        out = softmax_fwd(x)

        # Check results
        out_cpu = out.cpu()
        # Safe softmax reference: subtract max to avoid overflow
        x_fp32 = x.cpu().to(torch.float32)
        x_shifted = x_fp32 - x_fp32.max(dim=1, keepdim=True).values
        ref = torch.exp(x_shifted)

        # print(f"ref.sum: {ref.sum(dim=1)}")
        ref = ref / ref.sum(dim=1, keepdim=True)
        ref = ref.to(torch.float16)

        max_diff = torch.max(torch.abs(ref - out_cpu)).item()
        print(f"Output: {out_cpu.flatten()}")
        print(f"Softmax ref: {ref.flatten()}")
        print(f"Max difference: {max_diff}")

        if max_diff < 1e-2:
            print("✓ Cute-Snippets Softmax PASSED")
            return True
        else:
            print(f"✗ Cute-Snippets Softmax FAILED: max_diff={max_diff}")
            return False

    except Exception as e:
        print(f"✗ Cute-Snippets Softmax FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_softmax()
