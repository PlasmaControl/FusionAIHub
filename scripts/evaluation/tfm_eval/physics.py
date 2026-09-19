"""Per-window physical scalars from prediction-mode batches.

For UMAP/PCA coloring (A5), physics probes (A8), and Study B dose-response
readouts. Each scalar is computed from the z-space batch tensors by
inverting to physical units elementwise (``InversePreprocessor.inverse``)
FIRST and aggregating after — the log transforms make aggregate-then-invert
wrong.

Masking: prediction mode drops elementwise ``_mask`` keys, but the dataset
zero-fills invalid positions in z-space, so ``z != 0`` recovers the mask
(exact-zero true values are measure-zero). Windows with ``*_valid == 0``
yield NaN.

Physical units are the dataset's post-resample space (see
``inverse_preprocess`` docstring); good for ordering/regression targets,
not for absolute calibration claims.
"""

from __future__ import annotations

from typing import Dict

import torch

from tfm_eval.inverse_preprocess import InversePreprocessor

# (scalar_name, signal_name, batch side, channel aggregation)
#   mean — average over channels & time (profile-average quantities)
#   sum  — time-mean per channel, then sum over channels (additive powers)
#   absmean — mean |value| (signed coil currents)
DIAG_SCALARS = [
    ("ne_core", "ts_core_density", "mean"),
    ("te_core", "ts_core_temp", "mean"),
    ("ne_tang", "ts_tangential_density", "mean"),
    ("te_tang", "ts_tangential_temp", "mean"),
    ("ti", "cer_ti", "mean"),
    ("rotation", "cer_rot", "mean"),
    ("d_alpha", "filterscopes", "mean"),
]
ACT_SCALARS = [
    ("pin_total", "pin", "sum"),
    ("tin_total", "tin", "sum"),
    ("ech_power_total", "ech_power", "sum"),
    ("gas_flow_total", "gas_flow", "sum"),
    ("beam_voltage", "beam_voltage", "mean"),
    ("rmp_abs", "rmp", "absmean"),
]


def _aggregate(
    phys: torch.Tensor, zmask: torch.Tensor, how: str
) -> torch.Tensor:
    """(B, C, T) physical + bool mask → (B,) scalar; NaN when fully masked."""
    m = zmask.float()
    if how == "absmean":
        phys = phys.abs()
    if how == "sum":
        # Per-channel masked time-mean, then sum channels that had any data.
        num = (phys * m).sum(dim=-1)
        den = m.sum(dim=-1)
        ch_mean = num / den.clamp_min(1.0)
        ch_any = den > 0
        out = (ch_mean * ch_any.float()).sum(dim=-1)
        out[~ch_any.any(dim=-1)] = float("nan")
        return out
    num = (phys * m).flatten(1).sum(1)
    den = m.flatten(1).sum(1)
    out = num / den.clamp_min(1.0)
    out[den == 0] = float("nan")
    return out


@torch.no_grad()
def window_physics_scalars(
    batch: Dict, inv: InversePreprocessor
) -> Dict[str, torch.Tensor]:
    """``{scalar_name: (B,) float tensor}`` from one CPU prediction batch.

    Diagnostics read from ``batch['inputs']`` (state at t), actuators from
    ``batch['targets']`` (the command driving t → t+1, matching what the
    model is conditioned on).
    """
    out: Dict[str, torch.Tensor] = {}
    for scalar, signal, how in DIAG_SCALARS:
        if signal in batch["inputs"]:
            out[scalar] = _one(batch["inputs"], signal, how, inv)
    for scalar, signal, how in ACT_SCALARS:
        # Not every checkpoint conditions on every actuator (e.g. no tin
        # in the stage-1 9-actuator set) — skip absent ones.
        if signal in batch["targets"]:
            out[scalar] = _one(batch["targets"], signal, how, inv)
    return out


def _one(
    side: Dict, signal: str, how: str, inv: InversePreprocessor
) -> torch.Tensor:
    z = side[signal].float()
    finite = torch.isfinite(z)
    z = torch.where(finite, z, torch.zeros_like(z))
    phys = inv.inverse(signal, z)
    zmask = (z != 0) & finite
    scalar = _aggregate(phys, zmask, how)
    valid = side.get(f"{signal}_valid")
    if valid is not None:
        scalar = torch.where(
            valid.to(scalar.device) > 0,
            scalar,
            torch.full_like(scalar, float("nan")),
        )
    return scalar
