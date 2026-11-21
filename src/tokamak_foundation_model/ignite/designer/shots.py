"""Measured actuators of a tokenized shot, in physical units, as IGNITE sees them."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional

import numpy as np

from . import machine as mc


@dataclass
class Shot:
    """One tokenized shot.

    ``act_norm`` (F, 88) is exactly what IGNITE is given (the float16 dataset-wide
    normalised actuators); ``act`` is the same in physical units (``act_norm * sd + mu``),
    so an unedited channel round-trips to the identical model input.
    """

    shot: int
    token_set: mc.TokenSet
    act_norm: np.ndarray
    act: np.ndarray
    t_end: Optional[float]
    ip_peak_ma: Optional[float]

    @property
    def n_frames(self) -> int:
        return self.act.shape[0]

    @property
    def times(self) -> np.ndarray:
        return mc.time_of(np.arange(self.n_frames))

    def group(self, name: str) -> np.ndarray:
        return self.act[:, mc.columns(name)]


def locate(shot: int) -> Optional[mc.TokenSet]:
    """The token set holding ``shot`` (its frame codes and actuators), or None."""
    for ts in mc.TOKEN_SETS:
        if (ts.frame_codes / f"{shot}.pt").exists() and (ts.act_global / f"{shot}.pt").exists():
            return ts
    return None


@lru_cache(maxsize=64)
def load(shot: int) -> Shot:
    """Load a tokenized shot's actuators. Raises if the shot is not tokenized."""
    import torch

    shot = int(shot)
    ts = locate(shot)
    if ts is None:
        raise FileNotFoundError(
            f"shot {shot} is not tokenized (no frame codes and act_global entry in "
            f"{[t.name for t in mc.TOKEN_SETS]})"
        )
    entry = torch.load(ts.act_global / f"{shot}.pt", map_location="cpu", weights_only=False)
    norm = entry["actuators_global"].float().numpy()
    mu, sd = mc.act_stats()
    # The physical value IGNITE's input stands for; channels without spread keep mu.
    act = (norm * sd + mu).astype(np.float32)
    t_end = ip_peak = None
    end_file = ts.plasma_end / f"{shot}.json"
    if end_file.exists():
        info = json.loads(end_file.read_text())
        t_end = info.get("t_end")
        ip_peak = info.get("ip_peak_a")
        ip_peak = None if ip_peak is None else abs(float(ip_peak)) / 1e6
    return Shot(shot, ts, norm, act, None if t_end is None else float(t_end), ip_peak)


def _window(s: Shot, t0: float, t1: float) -> slice:
    k0 = max(0, int(np.floor((t0 - mc.T0) / mc.FRAME_S + 1e-9)))
    k1 = min(s.n_frames, int(np.ceil((t1 - mc.T0) / mc.FRAME_S - 1e-9)))
    return slice(k0, max(k0, k1))


def describe(shot: int, t0: float = 2.0, t1: Optional[float] = None,
             step_s: float = 0.25) -> Dict:
    """Summary of a shot's measured actuators for an agent, in display units.

    Per channel that was used (any frame above a small threshold): the mean over
    ``[t0, t1)`` and a coarse waveform (``step_s`` averages). Beams also report
    their torque per MW, which gives the injection direction (sign) in this shot.
    """
    s = load(shot)
    t1 = t1 if t1 is not None else (s.t_end if s.t_end else mc.time_of(s.n_frames))
    win = _window(s, t0, t1)
    times = s.times
    grid = np.arange(t0, t1, step_s)
    out = {
        "shot": s.shot,
        "token_set": s.token_set.name,
        "plasma_end_s": s.t_end,
        "ip_peak_MA": s.ip_peak_ma,
        "frames": {"t_first_s": float(times[0]), "t_last_s": float(times[-1] + mc.FRAME_S),
                   "dt_s": mc.FRAME_S},
        "window_s": [float(t0), float(t1)],
        "waveform_times_s": [round(float(t), 3) for t in grid],
        "groups": {},
    }
    for group in mc.GROUPS:
        unit, scale = mc.DISPLAY[group]
        vals = s.group(group) * scale
        used = {}
        for i, name in enumerate(mc.CHANNEL_NAMES[group]):
            x = vals[:, i]
            if np.nanmax(np.abs(x[win])) < _USED[group]:
                continue
            coarse = [float(np.nanmean(x[_window(s, g, g + step_s)])) for g in grid]
            used[name] = {"mean": round(float(np.nanmean(x[win])), 4),
                          "max": round(float(np.nanmax(x[win])), 4),
                          "min": round(float(np.nanmin(x[win])), 4),
                          "waveform": [round(v, 3) for v in coarse]}
        if used:
            out["groups"][group] = {"unit": unit, "channels": used}
    pinj = s.group("pinj")[win] * 1e-6
    tinj = s.group("tinj")[win]
    rate = {}
    for i, b in enumerate(mc.BEAMS):
        on = pinj[:, i] > 0.3
        if on.sum() >= 3:
            rate[b] = round(float(np.median(tinj[on, i] / pinj[on, i])), 3)
    out["beams"] = {
        "torque_per_MW_Nm": rate,
        "direction": {b: ("co" if r > 0 else "counter") for b, r in rate.items()},
        "total_power_MW_mean": round(float(pinj.sum(1).mean()), 3),
        "net_torque_Nm_mean": round(float(tinj.sum(1).mean()), 3),
        "note": "torque sign: + co-current (the plasma current direction), - counter",
    }
    out["ech_total_MW_mean"] = round(float((s.group("ech_power")[win] * 1e-6).sum(1).mean()), 3)
    out["gas_total_TorrL_s_mean"] = round(float(s.group("gas_flow")[win].sum(1).mean()), 3)
    return out


# A channel counts as used when it exceeds this (display units) in the window.
_USED = {"ech_power": 0.02, "pinj": 0.1, "beam_voltage": 5.0, "tinj": 0.1, "gas_flow": 0.5,
         "gas_raw": 0.05, "rmp": 0.1, "i_coil": 0.1}
