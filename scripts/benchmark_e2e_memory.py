"""Memory + timing benchmark for the integrated TS (+ optional video)
foundation model.

Closes Step 5 item 5 of the Phase C plan. Reports, for each
configuration:

* parameter count
* total backbone tokens, broken into diag prefix + actuators
* peak GPU memory on the same forward + backward + optimizer.step
  cadence the trainers actually run
* median step wall time over a small number of measured passes

Configurations tested by default:

1. **TS-only baseline.** ~398 tokens. Mirrors Phase A Stage 1.
2. **TS + tangtv.** 398 + 300 = 698 tokens. Mirrors what
   ``train_e2e_stage1.py --use_video tangtv`` would build.

Each is run at the same batch size (default 128, matching Phase A
Stage 2b's training batch). If TS-only fits comfortably the script
also retries at batch 256 to bracket the headroom.

Synthetic input. The benchmark is about peak memory and step
throughput, not correctness; constructing the data loader on a
benchmark node is unnecessary overhead.

Usage::

    pixi run python scripts/benchmark_e2e_memory.py --batch_size 128
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Tuple

import torch

from tokamak_foundation_model.e2e.model import (
    ActuatorConfig,
    DiagnosticConfig,
    E2EFoundationModel,
)


# ── Modality registries (mirrors train_e2e_stage1.py) ──────────────────


SLOW_TS_MODALITIES: List[Tuple[str, int]] = [
    ("ts_core_density", 44),
    ("ts_core_temp", 44),
    ("ts_tangential_density", 10),
    ("ts_tangential_temp", 10),
    ("cer_ti", 48),
    ("cer_rot", 48),
    ("mse", 69),
]
FAST_TS_MODALITIES: List[Tuple[str, int, int]] = [
    ("filterscopes", 8, 50),
]
ACTUATOR_MODALITIES: List[Tuple[str, int]] = [
    ("pin", 8),
    ("beam_voltage", 8),
    ("ech_power", 12),
    ("ech_tor_angle", 12),
    ("ech_pol_angle", 12),
    ("ech_polarization", 12),
    ("gas_flow", 11),
    ("gas_raw", 11),
    ("rmp", 12),
]
VIDEO_MODALITIES: List[Tuple[str, int, int, Tuple[int, int], Tuple[int, int, int]]] = [
    ("tangtv", 7, 3, (120, 360), (3, 12, 12)),
]
SLOW_FS = 100.0
FAST_FS = 10_000.0
CHUNK_DURATION_S = 0.05


def build_configs(
    use_video: List[str],
) -> Tuple[List[DiagnosticConfig], List[ActuatorConfig]]:
    slow_samples = round(CHUNK_DURATION_S * SLOW_FS)
    fast_samples = round(CHUNK_DURATION_S * FAST_FS)
    diags: List[DiagnosticConfig] = [
        DiagnosticConfig(name, "slow_ts", c, slow_samples)
        for name, c in SLOW_TS_MODALITIES
    ] + [
        DiagnosticConfig(name, "fast_ts", c, fast_samples, p)
        for name, c, p in FAST_TS_MODALITIES
    ]
    if use_video:
        registry = {entry[0]: entry for entry in VIDEO_MODALITIES}
        for cam in use_video:
            (_, n_chan, n_frames, (h, w), patch) = registry[cam]
            diags.append(
                DiagnosticConfig(
                    name=cam, kind="video",
                    n_channels=n_chan, window_samples=n_frames,
                    height=h, width=w, video_patch_size=patch,
                )
            )
    acts = [
        ActuatorConfig(n, c, fast_samples, n_tokens=5)
        for n, c in ACTUATOR_MODALITIES
    ]
    return diags, acts


def make_synthetic_batch(
    model: E2EFoundationModel,
    batch_size: int,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    diag_inputs: Dict[str, torch.Tensor] = {}
    for cfg in model.diagnostics:
        if cfg.kind == "video":
            shape = (
                batch_size, cfg.n_channels, cfg.window_samples,
                cfg.height, cfg.width,
            )
        else:
            shape = (batch_size, cfg.n_channels, cfg.window_samples)
        diag_inputs[cfg.name] = torch.randn(shape, device=device)
        if cfg.kind == "video":
            # Realistic mix: ~half the cameras present per batch in
            # production data; here we mark all valid so the heaviest
            # path runs.
            diag_inputs[f"{cfg.name}_valid"] = torch.ones(
                batch_size, dtype=torch.long, device=device
            )
    act_inputs: Dict[str, torch.Tensor] = {
        cfg.name: torch.randn(
            (batch_size, cfg.n_channels, cfg.window_samples), device=device
        )
        for cfg in model.actuators
    }
    step_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
    time_offset = torch.zeros(batch_size, device=device)
    return diag_inputs, act_inputs, step_idx, time_offset


def benchmark_one(
    use_video: List[str],
    batch_size: int,
    device: torch.device,
    d_model: int = 256,
    n_layers: int = 8,
    n_heads: int = 8,
    n_warmup: int = 2,
    n_measured: int = 3,
) -> Dict[str, float]:
    diags, acts = build_configs(use_video)
    model = E2EFoundationModel(
        diagnostics=diags, actuators=acts,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    diag_inputs, act_inputs, step_idx, time_offset = make_synthetic_batch(
        model, batch_size, device
    )

    def one_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        out = model(diag_inputs, act_inputs, step_idx, time_offset)
        loss = sum(t.abs().mean() for t in out.values())
        loss.backward()
        optimizer.step()

    # Warmup — exercises the cuDNN algo selection / cache.
    for _ in range(n_warmup):
        one_step()
    torch.cuda.synchronize()

    # Reset stats AFTER warmup so the reported peak is what steady-state
    # training would actually allocate.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    times: List[float] = []
    for _ in range(n_measured):
        torch.cuda.synchronize()
        t0 = time.time()
        one_step()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # Free the model + optimizer state before returning so the next
    # configuration starts from a clean GPU.
    del model, optimizer, diag_inputs, act_inputs, step_idx, time_offset
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return {
        "params": float(n_params),
        "median_step_s": float(sorted(times)[len(times) // 2]),
        "min_step_s": float(min(times)),
        "max_step_s": float(max(times)),
        "peak_gb": float(peak_gb),
    }


def report(label: str, batch: int, result: Dict[str, float]) -> None:
    print(
        f"  {label:30s}  "
        f"batch={batch:4d}  "
        f"params={result['params'] / 1e6:6.2f}M  "
        f"peak={result['peak_gb']:5.2f} GB  "
        f"step={result['median_step_s']:.3f} s "
        f"(min {result['min_step_s']:.3f}, max {result['max_step_s']:.3f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--also_batch_256", action="store_true",
        help="If both configs fit at the requested batch, retry at "
        "batch=256 to bracket headroom.",
    )
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — this benchmark requires a GPU.")
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU: {gpu_name}  total memory: {total_gb:.1f} GB")
    print(f"Backbone: d_model={args.d_model} n_layers={args.n_layers} "
          f"n_heads={args.n_heads}  loss=AdamW + sum(|out|)")
    print()

    runs = [
        ("TS-only (Phase A)", []),
        ("TS + tangtv (Phase C)", ["tangtv"]),
    ]

    print("Per-config metrics:")
    fits_at_default: Dict[str, bool] = {}
    for label, use_video in runs:
        try:
            result = benchmark_one(
                use_video=use_video,
                batch_size=args.batch_size,
                device=device,
                d_model=args.d_model,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
            )
            report(label, args.batch_size, result)
            fits_at_default[label] = True
        except torch.cuda.OutOfMemoryError as e:
            print(f"  {label}: OOM at batch={args.batch_size}: {e}")
            fits_at_default[label] = False
            torch.cuda.empty_cache()

    # Also report tokens from a tiny rebuild (cheap, no forward).
    print()
    print("Token counts:")
    for label, use_video in runs:
        diags, acts = build_configs(use_video)
        m = E2EFoundationModel(
            diagnostics=diags, actuators=acts,
            d_model=args.d_model, n_heads=args.n_heads,
            n_layers=args.n_layers,
        )
        n_diag = m.n_diag_tokens
        n_total = m.n_total_tokens
        print(
            f"  {label:30s}  total={n_total:4d}  "
            f"diag={n_diag:4d}  actuator={n_total - n_diag:4d}"
        )
        del m

    if args.also_batch_256 and all(fits_at_default.values()):
        print()
        print("Bracketing at batch=256:")
        for label, use_video in runs:
            try:
                result = benchmark_one(
                    use_video=use_video, batch_size=256, device=device,
                    d_model=args.d_model, n_layers=args.n_layers,
                    n_heads=args.n_heads,
                )
                report(label, 256, result)
            except torch.cuda.OutOfMemoryError as e:
                print(f"  {label}: OOM at batch=256: {e}")
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()