"""Diagnostics for whether the video AE is information-bound or has a
training bug.

Three checks per the user's prompt:

1. Does gradient reach the stem at init? Cross-attention with 32 queries
   over ~8100 keys may divide gradient by ~8100 if softmax starts near
   uniform, leaving the stem with near-zero learning signal.

2. Is the decoder output simply the per-(batch, channel, frame) spatial
   mean? If yes the ConvT cascade can't escape "predict the local mean"
   from a 4x8 latent grid, and the bottleneck size is irrelevant.

3. If we replace the upsampling output head with a stem-resolution
   reconstruction (decode tokens to a 30x90 latent rather than to
   120x360), can the same 32 tokens reconstruct that? If yes, the
   bottleneck is fine and the upsampling decoder is the bottleneck.

Read-only on the running 2724175 job.
"""

from __future__ import annotations

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokamak_foundation_model.e2e.tokenizers.video import VideoTokenizer
from tokamak_foundation_model.e2e.output_heads import VideoOutputHead


def standardize(x: torch.Tensor) -> torch.Tensor:
    mu = x.mean(dim=(2, 3, 4), keepdim=True)
    sd = x.std(dim=(2, 3, 4), keepdim=True).clamp(min=1.0)
    return (x - mu) / sd


def make_input(B: int = 8) -> torch.Tensor:
    """Try to load real tangtv windows; fall back to synthetic Gaussian."""
    try:
        from pathlib import Path

        from tokamak_foundation_model.data.data_loader import (
            TokamakH5Dataset, collate_fn,
        )
        from torch.utils.data import DataLoader

        files = sorted(
            Path("/scratch/gpfs/EKOLEMEN/foundation_model").glob(
                "*_processed.h5"
            )
        )
        x_batches = []
        for f in files[:80]:
            with h5py.File(f, "r") as h:
                if (
                    "tangtv" not in h
                    or "ydata" not in h["tangtv"]
                    or h["tangtv"]["ydata"].size == 0
                ):
                    continue
                if h["tangtv"]["ydata"].ndim != 4:
                    continue
            ds = TokamakH5Dataset(
                hdf5_path=f,
                chunk_duration_s=0.05,
                prediction_mode=True,
                prediction_horizon_s=0.05,
                input_signals=["tangtv"],
                target_signals=["tangtv"],
            )
            for i in range(min(2, len(ds))):
                sample = ds[len(ds) // 2 + i]
                if sample["inputs"]["tangtv_valid"] == 1:
                    x_batches.append(sample["inputs"]["tangtv"])
                    if len(x_batches) >= B:
                        break
            if len(x_batches) >= B:
                break
        if len(x_batches) >= B:
            x = torch.stack(x_batches[:B])
            return x.float()
    except Exception as e:
        print(f"  (real data load failed: {e}; using synthetic)")
    return torch.randn(B, 7, 3, 120, 360)


def diagnostic_1_grad_flow(
    tok: VideoTokenizer, head: VideoOutputHead, x_norm: torch.Tensor
) -> None:
    """Check grad norms at every layer after one backward pass."""
    print("\n=== DIAGNOSTIC 1 — grad flow at init ===")
    target = x_norm.permute(0, 2, 1, 3, 4)
    tokens = tok(x_norm)
    recon = head(tokens)
    loss = (recon - target).abs().mean()
    print(f"  loss at init = {loss.item():.4f}")
    loss.backward()

    pairs = [
        ("stem[0] (Conv 7→64)", tok.stem[0].weight.grad),
        ("stem[3] (Conv 64→128)", tok.stem[3].weight.grad),
        ("kv_proj", tok.kv_proj.weight.grad),
        ("queries (param)", tok.queries.grad),
        ("spatial_pe", tok.spatial_pe.grad),
        ("temporal_pe", tok.temporal_pe.grad),
        ("cross_attn.in_proj", tok.cross_attn.in_proj_weight.grad),
        ("cross_attn.out_proj", tok.cross_attn.out_proj.weight.grad),
        ("ffn[0]", tok.ffn[0].weight.grad),
        ("ffn[3]", tok.ffn[3].weight.grad),
        ("modality_emb", tok.modality_emb.grad),
        ("missing_token", tok.missing_token.grad),
        ("head.channel_reduce[0]", head.channel_reduce[0].weight.grad),
        ("head.decoder[0] (ConvT)", head.decoder[0].weight.grad),
        ("head.final", head.final.weight.grad),
    ]
    longest = max(len(name) for name, _ in pairs)
    print(f"  {'layer'.ljust(longest)}   grad.norm()       grad.abs().max()")
    print(f"  {'-' * longest}   --------------    -----------------")
    for name, g in pairs:
        if g is None:
            print(f"  {name.ljust(longest)}   (no grad)")
            continue
        gn = g.norm().item()
        gmax = g.abs().max().item()
        print(f"  {name.ljust(longest)}   {gn:14.6e}    {gmax:14.6e}")

    # Reference scale.
    queries_grad = tok.queries.grad.norm().item()
    stem0_grad = tok.stem[0].weight.grad.norm().item()
    print(
        f"\n  stem[0] grad / queries grad = "
        f"{stem0_grad / max(queries_grad, 1e-30):.3e}"
    )
    if stem0_grad < 1e-6:
        print("  → stem grad is < 1e-6: gradient is dying in cross-attention.")
    elif stem0_grad / max(queries_grad, 1e-30) < 1e-3:
        print(
            "  → stem grad < 0.1% of queries grad: cross-attention is "
            "diluting gradient heavily."
        )
    else:
        print("  → stem grad looks healthy at init.")


def diagnostic_2_recon_vs_spatial_mean(
    tok: VideoTokenizer, head: VideoOutputHead, x_norm: torch.Tensor
) -> None:
    """Is the decoder output ≈ per-(B, T, C) spatial mean?"""
    print("\n=== DIAGNOSTIC 2 — recon vs per-(B, T, C) spatial mean ===")
    with torch.no_grad():
        target = x_norm.permute(0, 2, 1, 3, 4)
        tokens = tok(x_norm)
        recon = head(tokens)
        spatial_mean_target = target.mean(dim=(3, 4), keepdim=True)
        spatial_mean_target_full = spatial_mean_target.expand_as(target)
        recon_var = (recon - recon.mean(dim=(3, 4), keepdim=True)).var(
            dim=(3, 4)
        )
        target_var = (target - target.mean(dim=(3, 4), keepdim=True)).var(
            dim=(3, 4)
        )
        var_ratio = recon_var.mean().item() / max(
            target_var.mean().item(), 1e-30
        )
        mae_recon_vs_target = (recon - target).abs().mean().item()
        mae_recon_vs_spatial_mean = (
            recon - spatial_mean_target_full
        ).abs().mean().item()
        print(f"  per-pixel spatial variance of recon  : {recon_var.mean().item():.4f}")
        print(f"  per-pixel spatial variance of target : {target_var.mean().item():.4f}")
        print(f"  variance ratio (recon / target)      : {var_ratio:.4f}")
        print(f"  MAE(recon, target)                   : {mae_recon_vs_target:.4f}")
        print(f"  MAE(recon, target.spatial_mean)      : {mae_recon_vs_spatial_mean:.4f}")
        if var_ratio < 0.05:
            print(
                "  → recon spatial variance < 5% of target's: decoder is "
                "outputting near-uniform-per-(B,T,C) — i.e. spatial mean."
            )
        else:
            print(
                f"  → recon carries some spatial variance ({var_ratio*100:.1f}%); "
                "decoder is doing something beyond spatial mean."
            )


# ── Diagnostic 3: stem-resolution head + brief training ─────────────────


class StemResolutionHead(nn.Module):
    """Decode tokens to a (n_frames, n_channels, h_out, w_out) tensor.

    h_out, w_out match the stem output (default 30x90). No bilinear
    upsampling — if this head can reconstruct the stem-resolution latent
    well, the bottleneck is not the issue; the upsampling decoder is.
    """

    def __init__(
        self,
        n_queries: int = 32,
        d_model: int = 256,
        n_channels: int = 7,
        n_frames: int = 3,
        out_hw: tuple[int, int] = (30, 90),
        grid_hw: tuple[int, int] = (4, 8),
    ) -> None:
        super().__init__()
        gh, gw = grid_hw
        assert gh * gw == n_queries
        self.gh, self.gw = gh, gw
        self.d_model = d_model
        self.n_frames = n_frames
        self.n_channels = n_channels
        self.out_hw = out_hw
        # 1x1 reduce, then ConvTranspose to out_hw via stride-2 stages
        self.reduce = nn.Sequential(
            nn.Conv2d(d_model, 128, 1),
            nn.GroupNorm(16, 128),
            nn.GELU(),
        )
        # 4x8 -> 8x16 -> 16x32 -> 32x64 then bilinear to (30, 90)
        # is overkill spatially. Cleaner: keep the 4x8 latent and expand
        # via three ConvTranspose stages then a small bilinear to 30x90
        self.up = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
        )
        self.final = nn.Conv2d(32, n_channels * n_frames, 3, padding=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        x = tokens.transpose(1, 2).reshape(B, self.d_model, self.gh, self.gw)
        x = self.reduce(x)
        x = self.up(x)                                          # (B, 32, 16, 32)
        x = F.interpolate(x, size=self.out_hw, mode="bilinear",
                          align_corners=False)                  # (B, 32, h_out, w_out)
        x = self.final(x)                                       # (B, F*C, h_out, w_out)
        return x.reshape(
            B, self.n_frames, self.n_channels, *self.out_hw
        )


def diagnostic_3_stem_resolution_train(
    tok: VideoTokenizer, x_norm: torch.Tensor, n_steps: int = 200
) -> None:
    """Train tokenizer + stem-resolution head end-to-end on the SAME
    fixed batch for ``n_steps``. If MAE drops to a small fraction of init,
    32 tokens carry enough info for stem-resolution reconstruction.
    """
    print("\n=== DIAGNOSTIC 3 — stem-resolution overfit on a fixed batch ===")
    head_sr = StemResolutionHead(n_queries=tok.n_queries, grid_hw=(4, 8))

    # Stem-resolution target: average input down to the stem output H, W
    # = (30, 90). We use the standardized input directly.
    target = x_norm.permute(0, 2, 1, 3, 4)                    # (B, T, C, H, W)
    target_lo = F.adaptive_avg_pool2d(
        target.reshape(-1, 1, *target.shape[-2:]), output_size=(30, 90)
    ).reshape(*target.shape[:3], 30, 90)

    opt = torch.optim.AdamW(
        list(tok.parameters()) + list(head_sr.parameters()), lr=1e-3
    )
    init_mae = None
    for step in range(n_steps):
        tokens = tok(x_norm)
        recon = head_sr(tokens)
        loss = (recon - target_lo).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 0:
            init_mae = loss.item()
        if step % 50 == 0 or step == n_steps - 1:
            spatial_mean = target_lo.mean(dim=(3, 4), keepdim=True).expand_as(
                target_lo
            )
            mean_baseline = (target_lo - spatial_mean).abs().mean().item()
            print(
                f"  step {step:4d}  MAE={loss.item():.4f}  "
                f"mean_baseline={mean_baseline:.4f}  "
                f"ratio={loss.item() / mean_baseline:.3f}"
            )
    print(
        f"  init MAE / final MAE = "
        f"{init_mae / max(loss.item(), 1e-30):.2f}x reduction"
    )


def main() -> None:
    torch.manual_seed(0)
    print("Loading inputs…")
    x = make_input(B=8)
    print(f"  input shape: {tuple(x.shape)}")

    x_norm = standardize(x)

    tok = VideoTokenizer(
        n_channels=7, n_frames=3, n_queries=32,
        d_stem=128, d_model=256, spatial_size=(120, 360),
    )
    head = VideoOutputHead(
        n_queries=32, d_model=256, n_channels=7, n_frames=3,
        output_size=(120, 360), grid_hw=(4, 8),
    )

    diagnostic_1_grad_flow(tok, head, x_norm)
    # Re-init for diagnostic 2 (zero grads).
    torch.manual_seed(0)
    tok = VideoTokenizer(
        n_channels=7, n_frames=3, n_queries=32,
        d_stem=128, d_model=256, spatial_size=(120, 360),
    )
    head = VideoOutputHead(
        n_queries=32, d_model=256, n_channels=7, n_frames=3,
        output_size=(120, 360), grid_hw=(4, 8),
    )
    diagnostic_2_recon_vs_spatial_mean(tok, head, x_norm)

    torch.manual_seed(0)
    tok = VideoTokenizer(
        n_channels=7, n_frames=3, n_queries=32,
        d_stem=128, d_model=256, spatial_size=(120, 360),
    )
    diagnostic_3_stem_resolution_train(tok, x_norm, n_steps=200)


if __name__ == "__main__":
    main()