"""Merge gate: strictly load an e2e checkpoint and run one forward pass.

Usage (login node, pixi frontier env)::

    python scripts/evaluation/smoke_load_checkpoint.py            # CPU-load + GPU forward if available
    python scripts/evaluation/smoke_load_checkpoint.py --no-forward

Passes iff ``load_state_dict(strict=True)`` accepts every key and the forward
pass produces one prediction per diagnostic with the expected shapes
(spectrogram time axis truncated to a multiple of T_p, e.g. ece 98→96).
"""

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "src"))
sys.path.insert(0, str(_HERE.parent))

import torch  # noqa: E402

from tfm_eval.ckpt import DEFAULT_CKPT, build_model_from_ckpt, load_ckpt  # noqa: E402


def synthetic_batch(diagnostics, actuators, batch, device):
    gen = torch.Generator().manual_seed(0)

    def rand(*shape):
        return torch.randn(*shape, generator=gen).to(device)

    diag = {}
    for cfg in diagnostics:
        if cfg.kind in ("slow_ts", "fast_ts"):
            diag[cfg.name] = rand(batch, cfg.n_channels, cfg.window_samples)
        elif cfg.kind == "spectrogram":
            diag[cfg.name] = rand(
                batch, cfg.n_channels, cfg.freq_bins, cfg.window_samples
            )
        elif cfg.kind == "video":
            diag[cfg.name] = rand(
                batch, cfg.n_channels, cfg.window_samples, cfg.height, cfg.width
            )
        else:
            raise ValueError(f"unknown kind {cfg.kind!r}")
    acts = {
        cfg.name: rand(batch, cfg.n_channels, cfg.window_samples)
        for cfg in actuators
    }
    step = torch.zeros(batch, dtype=torch.long, device=device)
    time_s = torch.full((batch,), 1.5, device=device)
    return diag, acts, step, time_s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--no-forward", action="store_true")
    ap.add_argument("--expect-tokens", type=int, default=1178)
    args = ap.parse_args()

    t0 = time.time()
    ckpt = load_ckpt(args.checkpoint)
    print(
        f"[{time.time()-t0:6.1f}s] loaded ckpt: step={ckpt.get('step')} "
        f"val_loss={ckpt.get('val_loss')} best={ckpt.get('best_val_loss')} "
        f"@ step {ckpt.get('best_step')}"
    )
    a = ckpt.get("args", {})
    print(
        f"          d_model={a.get('d_model')} n_layers={a.get('n_layers')} "
        f"n_heads={a.get('n_heads')} use_spectro={a.get('use_spectro')} "
        f"use_video={a.get('use_video')}"
    )

    model, diagnostics, actuators = build_model_from_ckpt(ckpt)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[{time.time()-t0:6.1f}s] STRICT LOAD OK  params={n_params/1e6:.2f}M  "
        f"n_total_tokens={model.n_total_tokens}"
    )
    if model.n_total_tokens != args.expect_tokens:
        print(
            f"WARNING: n_total_tokens={model.n_total_tokens} != expected "
            f"{args.expect_tokens}"
        )

    if args.no_forward:
        print("PASS (load only)")
        return 0

    model = model.to(args.device)
    diag, acts, step, time_s = synthetic_batch(
        diagnostics, actuators, args.batch, args.device
    )
    with torch.no_grad():
        t1 = time.time()
        preds = model(diag, acts, step, time_s)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.time() - t1

    diag_names = {c.name for c in diagnostics}
    assert set(preds.keys()) == diag_names, (
        f"prediction keys {set(preds.keys())} != diagnostics {diag_names}"
    )
    print(f"[{time.time()-t0:6.1f}s] forward OK on {args.device} ({dt:.2f}s):")
    for cfg in diagnostics:
        shape = tuple(preds[cfg.name].shape)
        note = ""
        if cfg.kind == "spectrogram":
            t_p = cfg.spectrogram_patch_size[1]
            trunc_t = (cfg.window_samples // t_p) * t_p
            assert shape[-1] == trunc_t, (
                f"{cfg.name}: time dim {shape[-1]} != trunc_t {trunc_t}"
            )
            note = f"  (trunc_t {cfg.window_samples}->{trunc_t} ok)"
        print(f"    {cfg.name:24s} {cfg.kind:12s} {shape}{note}")
        assert torch.isfinite(preds[cfg.name]).all(), f"{cfg.name}: non-finite output"
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
