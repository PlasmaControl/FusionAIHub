"""Does the n = 1 radial field separate Hanson's RWM onsets from comparison shots?

Round five's sensor probe (R2). The sensors are PTDATA ``ONSBRADIAL``
(``rwm_sensor_fetch.py``), the cached ``DUSBRADIAL`` on shots outside its reported
corrupt range 176030-176912 (`labeler.tearing.rule.valid_lock_shot`), and the cached
N1RMS for reference. Each trace is |y| binned to 1 ms and averaged over a 5 ms
trailing window. At a time t:

- ``level`` is the largest 5 ms mean in [t, t + 20 ms], native units;
- ``jump`` is ln(level / baseline), the baseline the median 5 ms mean over
  [t - 120, t - 20 ms]; a zero or missing baseline gives no value.

The positives are the merged n = 1 Hanson onsets (`labels.merge_close`, as
``rwm_growth.py``). The primary controls are the matched comparison shots
(``matched_to`` in ``shots.csv``) at their Hanson shot's onset times, inside the
comparison shot's own |Ip| >= 0.5 peak window. The secondary controls are flat-top
times on the n = 1 Hanson shots at least 170 ms from any onset (50 per shot,
seed 0). AUC is the chance an onset's value exceeds a control's (ties half), with a
95% shot-bootstrap interval (2,000 resamples of shots within each group). The bar
for a usable RWM sensor is ONSBRADIAL ``jump`` AUC above 0.65 against the matched
controls, fixed before the run.

    pixi run --frozen -e labeler python scripts/labeler/rwm_sensor_separation.py

Writes ``outputs/labeler/rwm/sensor_separation.json`` and the per-time values to
``$LABELER_ROOT/round5/rwm/sensor/separation_points.csv``. Never fetches.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from labeler.config import Paths, git_sha
from labeler.events import rwm
from labeler.events.verify import NoDataError
from labeler.rwm import data, features, labels
from labeler.tearing.rule import valid_lock_shot
from labeler.timebase import decimate_to_step

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/labeler/rwm/sensor_separation.json"
SENSORS = ("onsbradial", "dusbradial", "n1rms")
STATISTICS = ("jump", "level")
AFTER_MS = 20.0
BASELINE_MS = (-120.0, -20.0)
CONTROLS_PER_SHOT = 50
CONTROL_EXCLUSION_MS = 170.0
BOOTSTRAP = 2000
SEED = 0
BAR = 0.65


def sensor_dir(paths: Paths) -> Path:
    return paths.root / "round5/rwm/sensor/onsbradial"


def traces(shot: int, paths: Paths) -> tuple[dict, dict]:
    """`{sensor: (t_ms, |y| on 1 ms)}` and the cached signals (for Ip)."""
    try:
        signals = data.load_signals(shot, paths)
    except NoDataError:
        signals = {}
    out = {}
    record = sensor_dir(paths) / f"{shot}.npz"
    if record.is_file():
        with np.load(record) as npz:
            t, y = decimate_to_step(npz["t_ms"], np.abs(npz["onsbradial"]), 1.0)
        out["onsbradial"] = (t, y[0])
    if "dusbradial" in signals and valid_lock_shot(shot):
        t, y = signals["dusbradial"]
        out["dusbradial"] = (t, np.abs(y))
    if "n1rms" in signals:
        out["n1rms"] = signals["n1rms"]
    return out, signals


def values(t, y, times) -> dict[str, np.ndarray]:
    """`level` and `jump` at each time (see the module docstring)."""
    times = np.asarray(times, dtype=float)
    if not len(times):
        return {"level": np.array([]), "jump": np.array([])}
    mean = features.trailing_mean(t, y, t, features.RMS_WINDOW_MS)
    level, base = np.full(len(times), np.nan), np.full(len(times), np.nan)
    for i, at in enumerate(times):
        after = (t >= at) & (t <= at + AFTER_MS) & np.isfinite(mean)
        before = (
            (t >= at + BASELINE_MS[0]) & (t < at + BASELINE_MS[1]) & np.isfinite(mean)
        )
        if after.any():
            level[i] = mean[after].max()
        if before.any():
            base[i] = np.median(mean[before])
    with np.errstate(invalid="ignore", divide="ignore"):
        jump = np.where((base > 0) & (level > 0), np.log(level / base), np.nan)
    return {"level": level, "jump": jump}


def auc(positive, negative) -> float:
    """P(positive > negative), ties half."""
    positive, negative = np.asarray(positive), np.asarray(negative)
    if not len(positive) or not len(negative):
        return float("nan")
    ranks = pd.Series(np.concatenate([positive, negative])).rank().to_numpy()
    u = ranks[: len(positive)].sum() - len(positive) * (len(positive) + 1) / 2
    return float(u / (len(positive) * len(negative)))


def compare(points: pd.DataFrame, rng) -> dict:
    """AUC of onsets over controls with a shot-bootstrap interval."""
    points = points[np.isfinite(points.value)]
    positive = points[points.kind == "onset"]
    negative = points[points.kind != "onset"]
    out = {
        "onset_points": len(positive),
        "onset_shots": int(positive.shot.nunique()),
        "control_points": len(negative),
        "control_shots": int(negative.shot.nunique()),
        "auc": auc(positive.value, negative.value),
    }
    if not len(positive) or not len(negative):
        return out
    pos = {s: g.value.to_numpy() for s, g in positive.groupby("shot")}
    neg = {s: g.value.to_numpy() for s, g in negative.groupby("shot")}
    draws = []
    for _ in range(BOOTSTRAP):
        a = rng.choice(list(pos), len(pos))
        b = rng.choice(list(neg), len(neg))
        draws.append(
            auc(
                np.concatenate([pos[s] for s in a]), np.concatenate([neg[s] for s in b])
            )
        )
    out["ci95"] = [float(q) for q in np.nanpercentile(draws, [2.5, 97.5])]
    out["onset_median"] = float(positive.value.median())
    out["control_median"] = float(negative.value.median())
    return out


def main() -> int:
    paths = Paths.from_env()
    study = pd.read_csv(paths.root / "round4/rwm/shots.csv")
    study = study[study.usable.astype(bool)]
    listed = rwm.onset_table(paths)
    onsets = {
        int(shot): list(labels.merge_close(group.t_ms))
        for shot, group in listed[listed.ntor == 1].groupby("shot")
    }
    every = {int(s): g.t_ms.to_numpy() for s, g in listed.groupby("shot")}
    rng = np.random.default_rng(SEED)
    rows = []
    coverage = {name: {"hanson": 0, "comparison": 0} for name in SENSORS}
    for row in study.itertuples():
        shot = int(row.shot)
        signals, cached = traces(shot, paths)
        group = "hanson" if row.role == "hanson" else "comparison"
        for name in signals:
            coverage[name][group] += 1
        window = features.flattop_window(*cached["ip"], 0.5) if "ip" in cached else None
        times = []
        if row.role == "hanson":
            times += [("onset", t) for t in onsets.get(shot, [])]
            if shot in onsets and window is not None:
                lo, hi = window[0] + 200.0, window[1] - 30.0
                draws = rng.uniform(lo, hi, CONTROLS_PER_SHOT) if hi > lo else []
                times += [
                    ("within_shot", t)
                    for t in draws
                    if np.all(np.abs(every[shot] - t) > CONTROL_EXCLUSION_MS)
                ]
        elif pd.notna(row.matched_to) and window is not None:
            times += [
                ("matched", t)
                for t in onsets.get(int(row.matched_to), [])
                if window[0] <= t <= window[1]
            ]
        for name, (t, y) in signals.items():
            found = values(t, y, [at for _, at in times])
            for i, (kind, at) in enumerate(times):
                for statistic in STATISTICS:
                    rows.append(
                        {
                            "shot": shot,
                            "campaign": int(row.campaign),
                            "role": row.role,
                            "kind": kind,
                            "t_ms": float(at),
                            "sensor": name,
                            "statistic": statistic,
                            "value": float(found[statistic][i]),
                        }
                    )
    points = pd.DataFrame(rows)
    out_dir = paths.root / "round5/rwm/sensor"
    out_dir.mkdir(parents=True, exist_ok=True)
    points.to_csv(out_dir / "separation_points.csv", index=False)
    results = []
    for (name, statistic), frame in points.groupby(["sensor", "statistic"]):
        for control in ("matched", "within_shot"):
            for campaign in ("all", 2014, 2018):
                part = frame[frame.kind.isin(["onset", control])]
                if campaign != "all":
                    part = part[part.campaign == campaign]
                results.append(
                    {
                        "sensor": name,
                        "statistic": statistic,
                        "controls": control,
                        "campaign": campaign,
                        **compare(part, np.random.default_rng(SEED)),
                    }
                )
    primary = next(
        r
        for r in results
        if (r["sensor"], r["statistic"], r["controls"], r["campaign"])
        == ("onsbradial", "jump", "matched", "all")
    )
    fetched = sorted(int(p.stem) for p in sensor_dir(paths).glob("*.npz"))
    markers = sorted(sensor_dir(paths).glob("*.missing.json"))
    missing = sorted(int(p.name.split(".")[0]) for p in markers)
    # Why a shot has no record (`rwm_sensor_fetch.py`): absent from the PTDATA
    # pointname index, or a marker from before that check ("unchecked").
    reasons = Counter(
        json.loads(p.read_text()).get("reason", "unchecked") for p in markers
    )
    record = {
        "script": "scripts/labeler/rwm_sensor_separation.py",
        "git_sha": git_sha(),
        "study_shots": len(study),
        "onsbradial_fetch": {
            "shots_with_record": len(fetched),
            "shots_without": len(missing),
            "without_by_reason": dict(reasons),
            "without_by_campaign": {
                str(c): int(study[study.shot.isin(missing)].campaign.eq(c).sum())
                for c in (2014, 2018)
            },
            "by_campaign_with_record": {
                str(c): int(study[study.shot.isin(fetched)].campaign.eq(c).sum())
                for c in (2014, 2018)
            },
            "directory": str(sensor_dir(paths)),
        },
        "coverage_shots": coverage,
        "definition": {
            "level": "max 5 ms trailing mean of |y| in [t, t+20 ms], native units",
            "jump": "ln(level / median 5 ms mean in [t-120, t-20 ms])",
            "onsets": "merged n=1 Hanson onsets",
            "matched": "matched comparison shots at their Hanson shot's onset "
            "times inside their own |Ip| >= 0.5 peak window",
            "within_shot": "n=1 Hanson flat-top times >= 170 ms from any onset, "
            "50 draws per shot, seed 0",
            "auc": "P(onset > control), ties half; 95% shot bootstrap, 2000 draws",
        },
        "bar": {"sensor": "onsbradial", "statistic": "jump", "controls": "matched"},
        "bar_auc": BAR,
        "primary": primary,
        "passes": bool(np.isfinite(primary["auc"]) and primary["auc"] > BAR),
        "results": results,
        "points": str(out_dir / "separation_points.csv"),
    }
    OUT.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("onsbradial_fetch", "primary", "passes")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
