"""Post-hoc ELM gate for the confinement networks: QH or WPQH -> H inside an ELM train.

The networks call part of ELMy H-mode QH or WPQH (on the ELMy-QH overlap shots, 69 % of the time
with four or more elm-inst2 onsets in 200 ms). A train of ELMs is direct evidence against QH, so
the gate moves a millisecond's QH and WPQH probability to H when the networks call it QH (WPQH)
and the ELM pipeline's onsets (``elm_fdp_label.py``, decoding ``elm-inst2-sized``) form a train
around it:

* an onset's size is the filterscope peak in ``[t, t + 3 ms]`` over the median of
  ``[t - 10, t - 1 ms]`` (the max over the ELM pipeline's filterscopes, ``inputs/<shot>.npz``
  ``fmax``); an onset counts when its size is at least ``Rule.size``;
* QH -> H where at least ``qh_count`` counted onsets fall in the centred ``qh_window_ms`` window
  and the millisecond lies between two counted onsets at most ``max_gap_ms`` apart (inside the
  train, not in the quiet time after it);
* WPQH -> H the same way with ``wp_count`` per ``wp_window_ms``: only a sustained train, so the
  isolated breakthrough ELMs of WPQH phases (Jalal's and Hiro's WPQH ELM set) leave WPQH alone.

The defaults' scores (release networks): ELMy time of the overlap shots inside an ELM train
called QH or WPQH 0.70 -> 0.01; held-out curated per-ms macro F1 0.9172 -> 0.9178 (QH recall
0.936 -> 0.934, WPQH 0.817 unchanged); QH or WPQH calls inside Hiro's WPQH phases flipped to H
1.4 %. conf_elmyh_2026-11-07/README.txt (residual/gate_v2.csv) has the variants.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

H, QH, WPQH = 1, 2, 3  # zerod.CLASSES: L, H, QH, WP
DECODING = "elm-inst2-sized"  # the onsets the rule was tuned on


@dataclass(frozen=True)
class Rule:
    size: float = 1.3
    max_gap_ms: float = 50.0
    qh_count: int = 4
    qh_window_ms: int = 200
    wp_count: int = 25
    wp_window_ms: int = 500

    def describe(self) -> str:
        return (f"QH -> H inside a train of >= {self.qh_count} elm-inst2 onsets of size >= {self.size:g} per "
                f"{self.qh_window_ms} ms, WPQH -> H inside a train of >= {self.wp_count} per {self.wp_window_ms} ms "
                f"(inside: between two such onsets at most {self.max_gap_ms:g} ms apart); size: filterscope peak "
                "in [t, t + 3 ms] over the median of [t - 10, t - 1 ms]")

    def as_dict(self) -> dict:
        return asdict(self)


RULE = Rule()


def onset_sizes(onsets_ms, fmax, grid0_ms: float, dt_ms: float) -> np.ndarray:
    """Each onset's size (NaN when its windows fall off the record or the baseline is not positive)."""
    fs = np.fmax.reduce(np.atleast_2d(np.asarray(fmax, float)), axis=0)
    on = np.asarray(onsets_ms, float)
    pre, gap, post = int(10 / dt_ms), int(1 / dt_ms), int(3 / dt_ms)
    out = np.full(on.size, np.nan)
    for k, i in enumerate(np.rint((on - grid0_ms) / dt_ms).astype(int)):
        if i - pre < 0 or i + post > fs.size:
            continue
        base, peak = fs[i - pre:i - gap], fs[i:i + post]
        if np.isfinite(base).any() and np.isfinite(peak).any() and (b := np.nanmedian(base)) > 0:
            out[k] = np.nanmax(peak) / b
    return out


def _count(on: np.ndarray, t: np.ndarray, window: int) -> np.ndarray:
    return np.searchsorted(on, t + window // 2) - np.searchsorted(on, t - window // 2)


def _inside(on: np.ndarray, t: np.ndarray, max_gap: float) -> np.ndarray:
    if on.size < 2:
        return np.zeros(t.size, bool)
    k = np.searchsorted(on, t, side="right")
    prev = np.where(k > 0, on[np.maximum(k - 1, 0)], -np.inf)
    nxt = np.where(k < on.size, on[np.minimum(k, on.size - 1)], np.inf)
    return nxt - prev <= max_gap


def masks(onsets_ms, sizes, n: int, t_first_ms: float = 0.0, hop_ms: float = 1.0,
          rule: Rule = RULE) -> tuple[np.ndarray, np.ndarray]:
    """(qh, wp): where on the n-bin confinement grid the gate may flip a QH, a WPQH call."""
    t = t_first_ms + hop_ms * np.arange(n)
    on = np.sort(np.asarray(onsets_ms, float)[np.asarray(sizes, float) >= rule.size])
    inside = _inside(on, t, rule.max_gap_ms)
    return (inside & (_count(on, t, rule.qh_window_ms) >= rule.qh_count),
            inside & (_count(on, t, rule.wp_window_ms) >= rule.wp_count))


def flips(prob: np.ndarray, qh: np.ndarray, wp: np.ndarray) -> np.ndarray:
    """The bins the gate flips: called QH where ``qh``, or WPQH where ``wp``."""
    call = prob.argmax(0)
    return ((call == QH) & qh) | ((call == WPQH) & wp)


def move(prob: np.ndarray, flip: np.ndarray) -> np.ndarray:
    """A copy of ``prob`` (4 x n) with the QH and WPQH probability of the ``flip`` bins moved to H
    (H then holds 1 - p_L >= p_QH, p_WPQH >= p_L: the call becomes H)."""
    out = np.array(prob, copy=True)
    out[H, flip] += out[QH, flip] + out[WPQH, flip]
    out[QH, flip] = 0
    out[WPQH, flip] = 0
    return out


def shot_masks(elm_root: Path, shot: int, n: int, t_first_ms: float = 0.0, hop_ms: float = 1.0,
               rule: Rule = RULE) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """(qh, wp, status) from the ELM pipeline's ``work/<shot>.npz`` (onsets) and ``inputs/<shot>.npz``
    (filterscopes). Status ``on``, or, with no masks: ``no_elm`` (either file missing),
    ``elm_decoding`` (onsets not from ``DECODING``), ``elm_unreadable``."""
    work, inputs = Path(elm_root) / "work" / f"{shot}.npz", Path(elm_root) / "inputs" / f"{shot}.npz"
    if not (work.exists() and inputs.exists()):
        return None, None, "no_elm"
    try:
        with np.load(work) as z:
            if "onsets_ms" not in z.files or str(z["decoding"]) != DECODING:
                return None, None, "elm_decoding"
            on = z["onsets_ms"].astype(float)
        with np.load(inputs) as z:
            sizes = onset_sizes(on, z["fmax"], float(z["grid0_ms"]), float(z["dt_ms"]))
    except Exception:  # noqa: BLE001 - a file being rewritten or cut short: run without the gate
        return None, None, "elm_unreadable"
    qh, wp = masks(on, sizes, n, t_first_ms, hop_ms, rule)
    return qh, wp, "on"
