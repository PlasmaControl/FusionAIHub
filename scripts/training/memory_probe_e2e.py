"""Memory-ceiling probe for the e2e model at scaled-up sizes.

Constructs ``E2EFoundationModel`` at a configurable size, generates synthetic
inputs matching each modality's expected shape, and runs one forward +
backward under bf16 autocast. Prints peak memory and param count.

Use to find the largest model that fits on one MI250X GCD under various
combinations of `attn_impl` and `gradient_checkpoint`. Reports both the
single-step ("stage 1") and K-step rollout ("stage 2") cases.

Typical usage (inside a 1-GCD SLURM allocation):

    python scripts/training/memory_probe_e2e.py \\
        --d_model 1024 --n_layers 24 --n_heads 16 \\
        --batch_size 4 --K_rollout 1 \\
        --attn_impl sdpa --gradient_checkpoint
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

# Resolve train_e2e_stage1 without installing as a package.
sys.path.insert(0, str(Path(__file__).parent))

from tokamak_foundation_model.e2e.model import E2EFoundationModel  # noqa: E402
from train_e2e_stage1 import (  # type: ignore  # noqa: E402
    SPECTROGRAM_MODALITIES,
    VIDEO_MODALITIES,
    build_configs,
)


def make_synthetic_inputs(
    diagnostics, actuators, batch: int, device: torch.device, dtype: torch.dtype,
):
    """Random tensors matching each modality's expected (channels, *spatial, samples).

    Mirrors the layout the real tokenizers expect: see the SlowTimeSeriesTokenizer,
    FastTimeSeriesTokenizer, VideoTokenizer, SpectrogramTokenizer ctors and the
    forward signatures in tokenizers.py.
    """
    diag_in: dict[str, torch.Tensor] = {}
    for d in diagnostics:
        if d.kind in ("slow_ts", "fast_ts"):
            diag_in[d.name] = torch.randn(
                batch, d.n_channels, d.window_samples, device=device, dtype=dtype
            )
        elif d.kind == "video":
            assert d.height is not None and d.width is not None
            # VideoTokenizer's patch_embed is a Conv3d expecting
            # (B, n_channels, T, H, W). For tangtv n_channels=2.
            diag_in[d.name] = torch.randn(
                batch, d.n_channels, d.window_samples, d.height, d.width,
                device=device, dtype=dtype,
            )
        elif d.kind == "spectrogram":
            assert d.freq_bins is not None
            diag_in[d.name] = torch.randn(
                batch, d.n_channels, d.freq_bins, d.window_samples,
                device=device, dtype=dtype,
            )
        else:
            raise ValueError(d.kind)
    act_in = {
        a.name: torch.randn(
            batch, a.n_channels, a.window_samples, device=device, dtype=dtype
        )
        for a in actuators
    }
    return diag_in, act_in


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--d_model", type=int, default=1024)
    p.add_argument("--n_layers", type=int, default=24)
    p.add_argument("--n_heads", type=int, default=16)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--chunk_duration_s", type=float, default=0.05)
    p.add_argument(
        "--use_video", nargs="*",
        default=["tangtv"],
        choices=[e[0] for e in VIDEO_MODALITIES],
    )
    p.add_argument(
        "--use_spectro", nargs="*",
        default=["ece", "co2", "bes"],
        choices=[e[0] for e in SPECTROGRAM_MODALITIES],
    )
    p.add_argument(
        "--attn_impl", choices=["standard", "sdpa", "flash"], default="standard",
    )
    p.add_argument("--gradient_checkpoint", action="store_true")
    p.add_argument(
        "--K_rollout", type=int, default=1,
        help="Simulate K-step rollout: repeat forward K times, backprop "
             "through the chain (matches stage-2 memory pattern).",
    )
    p.add_argument("--no_amp", action="store_true",
                   help="Disable bf16 autocast (debug only).")
    args = p.parse_args()

    assert torch.cuda.is_available(), "No CUDA/HIP device visible"
    device = torch.device("cuda")
    dtype = torch.float32  # inputs in fp32; autocast handles bf16 internally
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"config: d_model={args.d_model} n_layers={args.n_layers} "
          f"n_heads={args.n_heads}  attn_impl={args.attn_impl}  "
          f"grad_ckpt={args.gradient_checkpoint}  K_rollout={args.K_rollout}")

    diagnostics, actuators = build_configs(
        args.chunk_duration_s,
        use_video=args.use_video,
        use_spectro=args.use_spectro,
    )
    print(f"diagnostics: {[d.name for d in diagnostics]}")
    print(f"actuators  : {[a.name for a in actuators]}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_pre_model = torch.cuda.memory_allocated() / 1e9

    model = E2EFoundationModel(
        diagnostics=diagnostics, actuators=actuators,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio, dropout=args.dropout,
        attn_impl=args.attn_impl,
        gradient_checkpoint=args.gradient_checkpoint,
    ).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    n_total_tokens = model.n_total_tokens

    mem_after_model = torch.cuda.memory_allocated() / 1e9
    print()
    print(f"params       : {n_params/1e6:.1f}M")
    print(f"n_total_tokens: {n_total_tokens}")
    print(f"weight mem   : {mem_after_model - mem_pre_model:.2f} GB "
          f"(should be ~{n_params * 4 / 1e9:.2f} GB at fp32)")

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)

    diag_in, act_in = make_synthetic_inputs(
        diagnostics, actuators, args.batch_size, device, dtype,
    )
    step_index = torch.zeros(args.batch_size, dtype=torch.long, device=device)
    time_offset_s = torch.zeros(args.batch_size, dtype=dtype, device=device)

    # Reset peak so we measure only the forward+backward window
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_at_start = torch.cuda.memory_allocated() / 1e9
    t0 = time.perf_counter()

    ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
           if not args.no_amp else
           torch.amp.autocast(device_type="cuda", enabled=False))

    try:
        optim.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=device)
        with ctx:
            # K-step rollout: forward K times, accumulating loss. Each forward
            # holds activations needed for backward, matching stage 2's pattern.
            for k in range(args.K_rollout):
                outputs = model(diag_in, act_in, step_index + k, time_offset_s)
                # model returns Dict[str, Tensor] (per-modality reconstructions).
                # Cheap proxy loss — sum of squared outputs across all
                # modalities. We only care about making backprop happen, not
                # the loss value.
                for v in outputs.values():
                    loss = loss + (v.float() ** 2).mean()
        loss.backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        reserved = torch.cuda.max_memory_reserved() / 1e9
        print()
        print(f"forward+backward time: {elapsed:.2f} s")
        print(f"peak alloc           : {peak:.2f} GB")
        print(f"peak reserved        : {reserved:.2f} GB")
        print(f"loss                 : {loss.item():.4f}  (sanity)")
        print()
        print("SUCCESS — model + step fit on this GCD.")
    except torch.cuda.OutOfMemoryError as e:
        peak = torch.cuda.max_memory_allocated() / 1e9
        reserved = torch.cuda.max_memory_reserved() / 1e9
        print()
        print(f"OOM during forward+backward.")
        print(f"peak alloc at OOM    : {peak:.2f} GB")
        print(f"peak reserved at OOM : {reserved:.2f} GB")
        print(f"error: {e}")
        sys.exit(1)
    finally:
        # Clean up before exit so SLURM reports a sensible final state.
        del diag_in, act_in, optim, model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
