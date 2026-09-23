"""The KV cache on a trained checkpoint: the same logits, the same rollout, less time.

`tests/ignite/test_kv_cache.py` checks exactness on tiny configs; this checks a real
checkpoint on one shot's frame codes. It prints, per modality, how far the cached logits
of the first predicted frame are from the uncached ones, then times a short rollout both
ways (same generator seed) and a long one cached.

Measured 2026-09-22 on the pinned v4 checkpoint (V100S, fp32, shot 185786): max |dlogit|
<= 4e-5 at logit scale 10-28, total variation <= 6e-6, argmax equal in every modality;
3 frames x 10 passes 42.0 s uncached, 4.7 s cached, the same tokens; 80 frames cached
106.1 s at a 16.6 GiB peak.

    python scripts/evaluation/ignite_kv_cache_check.py \
        --ckpt <IGNITE_v4>/ignite_dynamics_prod_v4_mskfull_step3200.pt \
        --codes <frame_codes>/185786.pt [--frames 3 --passes 10 --long 80]
"""

from __future__ import annotations

import argparse
import time

import numpy as np  # noqa: F401  (before torch, so the env's libstdc++ is the one loaded)
import torch

from tokamak_foundation_model.ignite import eval_dynamics


def _logits(model, codes, act, k0):
    """The first predicted frame's logits, uncached and from a prefilled cache."""
    bb = model.backbone
    with torch.no_grad():
        full = bb.tok.logits_last(bb.encode(codes, act)[:, -1:])
        seed = {n: c[:, :k0] for n, c in codes.items()}
        kv = bb.prefill(seed, act[:, :k0], k0 + 1)
        new = bb.step(kv, {n: c[:, k0] for n, c in codes.items()}, act[:, k0])
        return full, bb.tok.logits_last(new.unsqueeze(1))


def _rollout(model, seed, act, n, kv, device):
    torch.cuda.synchronize()
    t = time.time()
    gen = torch.Generator(device).manual_seed(3)
    out = model.rollout(seed, act, n_predict=n, generator=gen, kv_cache=kv)
    torch.cuda.synchronize()
    return out, time.time() - t


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt", required=True, help="dynamics checkpoint (.pt)")
    p.add_argument("--codes", required=True, help="one shot's frame-code cache (.pt)")
    p.add_argument("--k0", type=int, default=20, help="seed frames")
    p.add_argument("--frames", type=int, default=3, help="short rollout, both ways")
    p.add_argument("--passes", type=int, default=10, help="decode passes per frame")
    p.add_argument(
        "--long", type=int, default=80, help="long rollout, cached (0: skip)"
    )
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    model, cfg, step = eval_dynamics.load_model(args.ckpt, args.device)
    cfg.maskgit_decode_steps = args.passes
    cache = torch.load(args.codes, map_location="cpu", weights_only=False)
    names = [m.name for m in cfg.modalities]
    k0, dev = args.k0, args.device

    def codes(n):
        return {m: cache["codes"][m][:n].long().unsqueeze(0).to(dev) for m in names}

    def actuators(n):
        return cache["actuators"][:n].float().unsqueeze(0).to(dev)

    print(f"step {step}, {args.codes}, k0 {k0}")
    full, cached = _logits(model, codes(k0 + 1), actuators(k0 + 1), k0)
    for m in names:
        p1, p2 = full[m].float().softmax(-1), cached[m].float().softmax(-1)
        print(
            f"{m:22s} max|dlogit| {(full[m] - cached[m]).abs().max():.1e}"
            f"  logit scale {full[m].abs().max():5.1f}"
            f"  max TV {0.5 * (p1 - p2).abs().sum(-1).max():.1e}"
            f"  argmax equal {bool((full[m].argmax(-1) == cached[m].argmax(-1)).all())}"
        )

    seed = codes(k0)
    horizon = max(args.frames, args.long)
    act = actuators(k0 + horizon)
    a, ta = _rollout(model, seed, act, args.frames, False, dev)
    b, tb = _rollout(model, seed, act, args.frames, True, dev)
    agree = min(float((a[m] == b[m]).float().mean()) for m in names)
    print(
        f"{args.frames} frames x {args.passes} passes: uncached {ta:.1f} s, "
        f"cached {tb:.1f} s; token agreement (worst modality) {agree:.4f}"
    )
    if args.long:
        torch.cuda.reset_peak_memory_stats()
        _, tl = _rollout(model, seed, act, args.long, True, dev)
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"{args.long} frames cached: {tl:.1f} s, peak {peak:.1f} GiB")


if __name__ == "__main__":
    main()
