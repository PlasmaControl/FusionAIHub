"""Measure REAL backbone peak VRAM vs sequence length + batch.

Production backbone is d=1024, 48 layers, 8 heads, grad-checkpointed
nn.MultiheadAttention blocks. This isolates the SEQ-DEPENDENT memory (activations
+ attention + backbone params/grads/Adam) so we can ground the token-budget
memory estimate in a measurement instead of a formula. Head/tokenizer/data memory
(~constant in token count) is added separately via the ~64 GB folded-24 anchor.

Candidate sequence lengths (from the token sweep):
  799  = folded-24 production;  1087 = fold-96;  2239 = fold-384;  2287 = per-channel.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tokamak_foundation_model.e2e.backbone import SharedBackbone

GB = 1024 ** 3
D, H, L = 1024, 8, 48
CONFIGS = [(799, 16), (1087, 16), (2239, 16), (2287, 16),
           (1087, 8), (2239, 8), (2287, 8), (2287, 4)]


def main():
    dev = torch.device("cuda")
    print(f"[vram] backbone probe d={D} L={L} H={H} grad_ckpt=True bf16 "
          f"(real backbone params+grads+Adam+activations+attention)", flush=True)
    for seq, B in CONFIGS:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = SharedBackbone(D, H, L, grad_checkpoint=True).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        try:
            for _ in range(3):
                tok = torch.randn(B, seq, D, device=dev)
                si = torch.zeros(B, device=dev)
                to = torch.zeros(B, device=dev)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(tok, si, to)
                    loss = out.float().pow(2).mean()
                loss.backward()
                opt.step()
            peak = torch.cuda.max_memory_allocated() / GB
            print(f"[vram]   seq={seq:>5} B={B:>2}: backbone peak = {peak:6.2f} GB", flush=True)
        except RuntimeError as e:
            print(f"[vram]   seq={seq:>5} B={B:>2}: OOM/ERR {str(e)[:70]}", flush=True)
        del model, opt
        torch.cuda.empty_cache()
    print("[vram] DONE", flush=True)


if __name__ == "__main__":
    main()
