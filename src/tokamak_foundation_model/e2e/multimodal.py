"""Shared multimodal helpers for the E2E trainers.

Pure data + pure functions used by ``train_e2e_stage1.py``,
``train_e2e_stage2_delta.py``, and ``train_e2e_stage2_extended.py``.
Factored out so all three trainers register the same modalities and slice
targets the same way; before this module existed, the registries and
splitters lived as duplicates inside the per-stage files and drifted.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from tokamak_foundation_model.e2e.model import DiagnosticConfig


# ── Modality registries ──────────────────────────────────────────────────

# Per-camera video modality registry. Mirrors train_e2e_stage1.py.
# Empty --use_video default reproduces TS-only behaviour byte-for-byte.
VIDEO_MODALITIES: List[
    Tuple[str, int, int, Tuple[int, int], Tuple[int, int, int]]
] = [
    ("tangtv", 7, 3, (120, 360), (3, 12, 12)),
]

# Spectrogram modality registry. STFT shape fixed by the data loader
# (n_fft=1024, hop=256, fs=500 kHz) → freq_bins=512, time_frames=98 per
# 50 ms window.
SPECTRO_FREQ_BINS = 512
SPECTRO_TIME_FRAMES = 98
# Patch sizes — original (kept after considering the (4F, T/4) variant).
# Aspect-ratio shifts alone don't change the per-patch compression
# bottleneck (~40× for ECE). To improve fine-pattern reconstruction
# (harmonics + transients), we instead bumped the spectrogram refine-block
# stack depth from 4 → 12 in spectrogram.py / output_heads.py — see
# memory/project-session-pause-20260519.md.
SPECTROGRAM_MODALITIES: List[Tuple[str, int, Tuple[int, int]]] = [
    ("ece", 40, (32, 8)),
    ("co2",  4, (64, 8)),
    ("bes", 16, (32, 8)),
    ("mhr",  6, (32, 8)),
]


# ── Diagnostic-list extension ────────────────────────────────────────────


def append_multimodal_diagnostics(
    diagnostics: List[DiagnosticConfig],
    use_video: Optional[List[str]],
    use_spectro: Optional[List[str]],
    spectro_patch_f: Optional[int] = None,
    spectro_patch_t: Optional[int] = None,
) -> List[DiagnosticConfig]:
    """Append spectrogram then video DiagnosticConfigs to ``diagnostics``.

    Order inside the diagnostic prefix is locked at
    ``[slow_ts | fast_ts | spectrogram | video | actuators]`` so the
    rollout's diagnostic-prefix slice (``rollout.py``) stays contiguous
    (Guard G1). Returns a new list; callers append actuators afterwards.

    ``spectro_patch_f`` / ``spectro_patch_t`` override the registry patch
    shape for ALL spectro modalities when given (else the per-modality
    registry default is used). Set ``spectro_patch_f=SPECTRO_FREQ_BINS`` for a
    full-frequency patch (one token spans the whole spectrum); this changes the
    encoder/decoder kernel shape and so is a from-scratch architecture change.
    """
    out = list(diagnostics)
    if use_spectro:
        registry = {entry[0]: entry for entry in SPECTROGRAM_MODALITIES}
        for spec_name in use_spectro:
            if spec_name not in registry:
                raise SystemExit(
                    f"--use_spectro {spec_name!r}: unknown modality; known: "
                    f"{sorted(registry.keys())}"
                )
            (_, n_ch, patch_size) = registry[spec_name]
            if spectro_patch_f is not None or spectro_patch_t is not None:
                pf, pt = patch_size
                patch_size = (spectro_patch_f or pf, spectro_patch_t or pt)
            out.append(
                DiagnosticConfig(
                    name=spec_name, kind="spectrogram",
                    n_channels=n_ch, window_samples=SPECTRO_TIME_FRAMES,
                    freq_bins=SPECTRO_FREQ_BINS,
                    spectrogram_patch_size=patch_size,
                )
            )
    if use_video:
        registry = {entry[0]: entry for entry in VIDEO_MODALITIES}
        for cam_name in use_video:
            if cam_name not in registry:
                raise SystemExit(
                    f"--use_video {cam_name!r}: unknown camera; known: "
                    f"{sorted(registry.keys())}"
                )
            (_, n_ch, n_frames, (h, w), patch_size) = registry[cam_name]
            out.append(
                DiagnosticConfig(
                    name=cam_name, kind="video", n_channels=n_ch,
                    window_samples=n_frames, height=h, width=w,
                    video_patch_size=patch_size,
                )
            )
    return out


# ── Per-batch (B, C) z-score for video ───────────────────────────────────


def video_standardize_per_bc(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-(B, C) z-score over (T, H, W). Returns ``(x_norm, mu, sd)``.

    ``sd.clamp(min=1.0)`` keeps off-channels (zero-filled) finite. Same
    convention as train_e2e_stage1.py / standalone video AE.
    """
    mu = x.mean(dim=(2, 3, 4), keepdim=True)
    sd = x.std(dim=(2, 3, 4), keepdim=True).clamp(min=1.0)
    return (x - mu) / sd, mu, sd


# ── Per-modality loss gates ──────────────────────────────────────────────


def video_loss_gate(
    name: str, batch: Dict, device: torch.device,
) -> torch.Tensor:
    """Per-element loss gate combining camera-validity scalar with the
    per-channel availability mask. Shape ``(B, C, 1, 1, 1)`` broadcasts
    cleanly over ``(B, C, T, H, W)``. Per-shot, not per-step."""
    chan = batch["targets"][f"{name}_channel_mask"].to(
        device, non_blocking=True
    ).float()
    valid = batch["targets"][f"{name}_valid"].to(
        device, non_blocking=True
    ).float()
    return valid[:, None, None, None, None] * chan[:, :, None, None, None]


def spectro_loss_gate(
    name: str, batch: Dict, device: torch.device,
) -> torch.Tensor:
    """Per-sample loss gate from per-modality presence ``<name>_valid``.

    Spectrograms have no per-channel runtime availability mask; the
    gate is just a per-batch scalar broadcast over ``(B, C, F, T)``.
    """
    valid = batch["targets"][f"{name}_valid"].to(
        device, non_blocking=True
    ).float()
    return valid[:, None, None, None]                # (B, 1, 1, 1)


# ── Per-step target splitters ────────────────────────────────────────────


def split_video_target_by_step(
    target: torch.Tensor, k_steps: int, n_per_step: int,
) -> List[torch.Tensor]:
    """Split (B, C, K * n_per_step, H, W) into K windows of (B, C, n_per_step, H, W).

    Pairs with the K-window emission added to ``data_loader._getitem_prediction``.
    """
    expected = k_steps * n_per_step
    if target.shape[2] < expected:
        raise ValueError(
            f"video target T={target.shape[2]} < expected K*n={expected}"
        )
    return [
        target[:, :, k * n_per_step : (k + 1) * n_per_step].contiguous()
        for k in range(k_steps)
    ]


def split_spectro_target_by_step(
    target: torch.Tensor, k_steps: int, trunc_t: int,
) -> List[torch.Tensor]:
    """Split (B, C, F, T) into K windows of ``trunc_t`` frames each.

    ``trunc_t`` must equal the spectrogram tokenizer's truncated time
    length — i.e. ``(DiagnosticConfig.window_samples // T_p) * T_p``,
    typically 96 for the standard 98-frame, T_p=8 config. The
    spectrogram head emits exactly ``trunc_t`` frames per step, so the
    target is sliced to the same length to match shapes for the
    masked-MAE loss. Frames past ``K * trunc_t`` are discarded — STFT
    over the full extended (input+prediction) window with
    ``center=True`` doesn't produce a frame count that divides cleanly
    by K, so a handful of trailing frames are dropped (typically <2%
    of the window).
    """
    needed = k_steps * trunc_t
    if target.shape[3] < needed:
        raise ValueError(
            f"spectro target T={target.shape[3]} < K * trunc_t = {needed}"
        )
    return [
        target[:, :, :, k * trunc_t : (k + 1) * trunc_t].contiguous()
        for k in range(k_steps)
    ]


def spectro_trunc_t(cfg: DiagnosticConfig) -> int:
    """Return the per-step time-axis truncation for a spectrogram cfg.

    Mirrors ``SpectrogramTokenizer.trunc_t`` so trainer-side target
    slicing and the head's ``patch_unembed`` output stay in lockstep.
    """
    assert cfg.kind == "spectrogram" and cfg.spectrogram_patch_size is not None
    _, T_p = cfg.spectrogram_patch_size
    return (cfg.window_samples // T_p) * T_p