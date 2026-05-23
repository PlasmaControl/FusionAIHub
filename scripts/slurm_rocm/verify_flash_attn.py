"""Smoke test for flash-attention 2 on Frontier (MI250X / gfx90a)."""
import sys

import torch

try:
    import flash_attn
    from flash_attn import flash_attn_func
except ImportError as e:
    sys.exit(f"flash_attn not importable: {e}")

assert torch.cuda.is_available(), "no GPU visible to torch"
assert torch.version.hip is not None, "torch is not a ROCm build"

arch = torch.cuda.get_device_properties(0).gcnArchName
assert "gfx90a" in arch, f"unexpected gcn arch: {arch}"

q = k = v = torch.randn(2, 8, 16, 64, device="cuda", dtype=torch.float16)
out = flash_attn_func(q, k, v, causal=True)
assert out.shape == q.shape

print(
    f"flash_attn {flash_attn.__version__} OK on "
    f"{torch.cuda.get_device_name(0)} ({arch})"
)
