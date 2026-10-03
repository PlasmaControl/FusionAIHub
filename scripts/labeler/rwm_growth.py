#!/usr/bin/env python
"""How the n = 1 magnetics and beta_N behave around Jeremy Hanson's RWM onsets.

The label's growth window (`labeler.rwm.labels.GROWTH_MS`) is justified here, from the
data and not from the wall time alone: for every listed onset the 5 ms mean of the n = 1
RMS is compared with its value a lag earlier, the same ratio is taken at random flat-top
times of the same shots (no onset within 150 ms either side) as a control, and the largest
trailing 20 ms growth rate around the onset is read off. beta_N is followed from 100 ms
before the onset to 40 ms after. Writes `outputs/labeler/rwm/growth.json` and a
per-onset table under `$LABELER_ROOT/round4/rwm/`.

    python scripts/labeler/rwm_growth.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.config import Paths
from labeler.events import rwm
from labeler.rwm import data, features, labels

LAGS_MS = (10, 20, 40, 80, 150)
CONTROLS_PER_SHOT = 300
CONTROL_EXCLUSION_MS = 150.0
SEED = 0


def _quartiles(values) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    q1, q2, q3 = np.percentile(values, [25, 50, 75])
    return {"n": len(values), "q1": q1, "median": q2, "q3": q3}


def _log_ratio(t, y, times, lag):
    """ln of the 5 ms mean n = 1 RMS at each time over its value `lag` ms earlier."""
    floor = features.LOG_FLOOR_G
    times = np.asarray(times, dtype=float)
    now = features.trailing_mean(t, y, times, features.RMS_WINDOW_MS)
    before = features.trailing_mean(t, y, times - lag, features.RMS_WINDOW_MS)
    return np.log(np.maximum(now, floor) / np.maximum(before, floor))


def _controls(signals, onsets_ms, rng):
    window = features.flattop_window(*signals["ip"], 0.5)
    if window is None:
        return np.array([])
    lo, hi = window[0] + 200.0, window[1]
    times = rng.uniform(lo, hi, CONTROLS_PER_SHOT) if hi > lo else np.array([])
    near = np.zeros(len(times), dtype=bool)
    for onset in onsets_ms:
        near |= np.abs(times - onset) < CONTROL_EXCLUSION_MS
    return times[~near]


def main() -> None:
    paths = Paths.from_env()
    listed = rwm.onset_table(paths)
    table = pd.DataFrame(
        [
            (int(shot), t, int(ntor))
            for (shot, ntor), group in listed.groupby(["shot", "ntor"])
            for t in labels.merge_close(group.t_ms)
        ],
        columns=["shot", "t_ms", "ntor"],
    )
    out_dir = paths.root / "round4" / "rwm"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows, ratios = [], {lag: [] for lag in LAGS_MS}
    controls = {lag: [] for lag in LAGS_MS}
    control_slopes = []
    for shot, group in table.groupby("shot"):
        try:
            signals = data.load_signals(int(shot), paths)
        except Exception as error:  # noqa: BLE001 - recorded, not hidden
            print(f"{shot}: {error}")
            continue
        t, y = signals["n1rms"]
        tb, betan = signals["betan"]
        all_onsets = group.t_ms.to_numpy()
        times = _controls(signals, all_onsets, rng)
        slope_times = np.arange(-150.0, 30.0, 2.0)
        for lag in LAGS_MS:
            controls[lag].extend(_log_ratio(t, y, times, lag))
        control_slopes.extend(
            features.trailing_log_slope(
                t, y, times, features.GROWTH_WINDOW_MS, features.LOG_FLOOR_G
            )
        )
        for onset, ntor in zip(group.t_ms, group.ntor):
            slope = features.trailing_log_slope(
                t,
                y,
                onset + slope_times,
                features.GROWTH_WINDOW_MS,
                features.LOG_FLOOR_G,
            )
            best = int(np.nanargmax(slope)) if np.isfinite(slope).any() else None
            row = {
                "shot": int(shot),
                "onset_ms": float(onset),
                "ntor": int(ntor),
                "max_growth_per_s": float(slope[best]) if best is not None else np.nan,
                "max_growth_at_ms": float(slope_times[best])
                if best is not None
                else np.nan,
                "slope_at_onset_per_s": float(
                    features.trailing_log_slope(
                        t, y, [onset], features.GROWTH_WINDOW_MS, features.LOG_FLOOR_G
                    )[0]
                ),
                "betan_m100": float(np.interp(onset - 100, tb, betan)),
                "betan_0": float(np.interp(onset, tb, betan)),
                "betan_p40": float(np.interp(onset + 40, tb, betan)),
            }
            for lag in LAGS_MS:
                row[f"log_ratio_{lag}"] = float(_log_ratio(t, y, [onset], lag)[0])
                if ntor == 1:
                    ratios[lag].append(row[f"log_ratio_{lag}"])
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "growth_onsets.csv", index=False)
    n1 = frame[frame.ntor == 1]
    result = {
        "script": "scripts/labeler/rwm_growth.py",
        "onsets": {
            "listed": len(listed),
            "n1": len(n1),
            "n2": int((frame.ntor == 2).sum()),
            "shots": int(frame.shot.nunique()),
        },
        "growth_window_ms": labels.GROWTH_MS,
        "trailing_window_ms": features.GROWTH_WINDOW_MS,
        "controls": {
            "per_shot": CONTROLS_PER_SHOT,
            "exclusion_ms": CONTROL_EXCLUSION_MS,
            "seed": SEED,
            "n": len(control_slopes),
        },
        "max_growth_per_s_n1": _quartiles(n1.max_growth_per_s),
        "efold_ms_n1": _quartiles(
            1000.0 / n1.max_growth_per_s[n1.max_growth_per_s > 0]
        ),
        "max_growth_at_ms_n1": _quartiles(n1.max_growth_at_ms),
        "slope_at_onset_per_s_n1": _quartiles(n1.slope_at_onset_per_s),
        "control_slope_per_s": _quartiles(control_slopes),
        "log_ratio_n1": {},
        "betan": {
            "m100": _quartiles(n1.betan_m100),
            "at_onset": _quartiles(n1.betan_0),
            "p40": _quartiles(n1.betan_p40),
            "fraction_dropping_20pct_by_p40": float(
                np.mean(n1.betan_p40 < 0.8 * n1.betan_m100)
            ),
        },
    }
    for lag in LAGS_MS:
        control = np.asarray(controls[lag])
        control = control[np.isfinite(control)]
        onset = np.asarray(ratios[lag])
        onset = onset[np.isfinite(onset)]
        p90 = float(np.percentile(control, 90))
        result["log_ratio_n1"][str(lag)] = {
            "onset": _quartiles(onset),
            "control": _quartiles(control),
            "control_p90": p90,
            "fraction_of_onsets_above_control_p90": float(np.mean(onset > p90)),
        }
    result = json.loads(json.dumps(result, default=float))
    target = REPO / "outputs" / "labeler" / "rwm" / "growth.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
