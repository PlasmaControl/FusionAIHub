#!/usr/bin/env python
"""Score the detachment states against divertor Thomson Te near the outer target.

The processed divertor Thomson Te (`\\ELECTRONS::TSTE_DIV`, 14 or 16 chords at
R = 1.485 m, 20 ms, fetched by `detach_fetch_round4.py --stage dts`) is a temperature
measurement that none of the three indicators uses, so it checks them where the
label cannot: the cold-target criterion that the literature itself uses.

* Chords: those 1.5 to 5 cm above the outer shelf (Z = -1.25 m). The lowest chord
  (0.7 cm) is left out for stray light (Eldon 2017) and Chen 2026's own chords on
  shot 201081 are Z = -1.222 and -1.205 m. A chord counts at a time only while it
  lies on the SOL side of the separatrix and in the near SOL, 1.000 < psiN <= 1.05,
  by the multi-slice EFIT map on disk (the strike point moves off the chord on
  other shots); samples outside 0.1-50 eV are rejected (Eldon 2017).
* Bands: attached is Te >= 10 eV, detached Te <= 5 eV (Eldon 2017's detachment
  threshold; the Te cliff jumps between about 1-3 eV and 8-20 eV and 3-8 eV values
  are rare). Chen 2026 reports 20 eV to 3 eV across the cliffs of shot 201081.
  The bands were fixed from those digests before scoring.
* A bin's Te is the median of the selected chords' valid samples inside it.

Reported per primary state, per label tier and per indicator vote: how many bins and
shots carry a Te, the Te quantiles, the share inside the band, and a threshold-free
AUROC of -Te for the detached against the attached bins (pooled and within shots,
with shot-bootstrap intervals; the number of shots behind each within-shot figure is
beside it). The tiers are `certain` (TangTV + Afrac agreement), `tangtv_only`
(TangTV alone, silver), both together (the exported attached and detached bins),
`tangtv_only_lmode` (the TangTV detached vote on a known L-mode bin, exported as
uncertain) and the TangTV vote alone; a paired shot bootstrap says whether the
second vote improves the agreement with Te, and is "not estimable" where the
certain tier sits on fewer than `MIN_PAIRED_SHOTS` shots with a Te. The same
numbers are given by regime (H, L, probable L, probable H, unknown; the regime
source of the label model), for the TangTV vote before the gate
(`tangtv_vote_before_gate`, every TangTV vote on the upper shelf, with no L-mode
gate) and for the exported states (after the gate), and by DZ band for the TangTV detached votes: the
gate itself was set a priori (the DZ cutoffs come from an H-mode shot; only the
detached vote is gated, because L-mode inner-SOL leakage biases DZ upward, so a
low DZ stays trustworthy) and is only reported here, never tuned on Te. The
before-gate rows are the check against Te; the after-gate shares are post-
selection (the gate was added after this check flagged the L-mode bins). One row
leaves the anchor shot 201081 out (it sets the DZ cliff). Shot 201081's two cliff
times are read from the data and set beside the published ones. No threshold is
fitted here. Every shot bootstrap uses the fresh generator of
`detach_benchmark.boot_rng`, so a population has one interval; a population on
fewer than `detach_benchmark.MIN_INTERVAL_SHOTS` shots has none ("n/a (N shots)").

    pixi run --frozen -e labelmaker python scripts/labeler/detach_te_check.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import detach_benchmark as bench
import numpy as np
import pandas as pd
from detach_json import dumps

from labeler.events.detachment import core, signals
from labeler.events.detachment import thresholds as th

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ["LABELER_ROOT"]) / "round4/detach"
OUT = REPO / "docs/labeler/results/detachment_te_check.json"
CHORD_MIN_ABOVE_SHELF_M = 0.015
CHORD_MAX_ABOVE_SHELF_M = 0.05
PSI_N_MIN, PSI_N_MAX = 1.0, 1.05
TE_VALID_EV = (0.1, 50.0)
TE_ATTACHED_MIN_EV = 10.0
TE_DETACHED_MAX_EV = 5.0
PUBLISHED_CLIFFS_MS = {201081: (2650.0, 4450.0)}
MIN_SHOT_BINS = 5
#: A paired shot bootstrap on fewer shots than this has no estimate.
MIN_PAIRED_SHOTS = 10
ANCHOR_SHOT = 201081
#: TangTV detached votes are described by front height in these bands (DZ).
DZ_BANDS = ((0.5, 0.65), (0.65, 0.8), (0.8, 1.2))
#: Known H and L (curated intervals or a D-alpha detector), the two a-priori proxies
#: for a shot whose regime no source gives (`signals.probable_regimes`: ELM-free and
#: median P_in under 2 MW is probable L; ELMy is probable H, report only) and the
#: bins that are none of these.
REGIMES = ("H", "L", "probable_L", "probable_H", "unknown")


def load_dts(shot: int):
    path = ROOT / "dts" / f"{shot}.npz"
    if not path.is_file():
        return None
    with np.load(path) as f:
        if "te" not in f.files:
            return None
        return {k: f[k] for k in f.files if k != "status"}


def target_te(shot: int, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bin Te (eV, NaN if none) and the number of chord samples behind it."""
    dts = load_dts(shot)
    maps = signals.load_flux_map(shot)
    nan = np.full(len(starts), np.nan)
    if dts is None or maps is None:
        return nan, np.zeros(len(starts), int)
    te, t, r, z = dts["te"], dts["t_ms"], dts["r"], dts["z"]
    above = z - th.SHELF_Z
    chords = np.flatnonzero(
        (above >= CHORD_MIN_ABOVE_SHELF_M) & (above <= CHORD_MAX_ABOVE_SHELF_M)
    )
    if not len(chords):
        return nan, np.zeros(len(starts), int)
    psi = signals.flux_at_positions(maps, t, np.stack([r[chords], z[chords]], axis=1))
    chord_te = te[chords].astype(float)
    ok = (
        np.isfinite(chord_te)
        & (chord_te >= TE_VALID_EV[0])
        & (chord_te <= TE_VALID_EV[1])
        & (psi > PSI_N_MIN)
        & (psi <= PSI_N_MAX)
    )
    chord_te = np.where(ok, chord_te, np.nan)
    width = core.BIN_MS
    index = np.floor((t - starts[0]) / width).astype(int)
    keep = (index >= 0) & (index < len(starts))
    value = np.full(len(starts), np.nan)
    count = np.zeros(len(starts), int)
    for b in np.unique(index[keep]):
        samples = chord_te[:, keep & (index == b)].ravel()
        samples = samples[np.isfinite(samples)]
        if len(samples):
            value[b], count[b] = np.median(samples), len(samples)
    return value, count


def attach_te(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["te_ev"] = np.nan
    frame["te_n"] = 0
    for shot, idx in frame.groupby("shot").indices.items():
        starts = frame.start_ms.to_numpy(float)[idx]
        grid = np.arange(starts.min(), starts.max() + core.BIN_MS, core.BIN_MS)
        value, count = target_te(int(shot), grid)
        place = np.rint((starts - grid[0]) / core.BIN_MS).astype(int)
        frame.iloc[idx, frame.columns.get_loc("te_ev")] = value[place]
        frame.iloc[idx, frame.columns.get_loc("te_n")] = count[place]
    return frame


def summarise(te: np.ndarray, band) -> dict:
    te = te[np.isfinite(te)]
    if not len(te):
        return {"n_bins": 0}
    return {
        "n_bins": len(te),
        "te_ev_quantiles_10_50_90": [float(v) for v in np.percentile(te, [10, 50, 90])],
        "n_in_band": int(np.sum(band(te))),
        "share_in_band": float(np.mean(band(te))),
        "share_between_bands": float(
            np.mean((te > TE_DETACHED_MAX_EV) & (te < TE_ATTACHED_MIN_EV))
        ),
    }


def score_states(frame, state_column, mask=None) -> dict:
    """Te summaries of the attached/detached/MARFE bins and the AUROC of -Te."""
    sel = frame[np.isfinite(frame.te_ev)]
    if mask is not None:
        sel = sel[mask[np.isfinite(frame.te_ev).to_numpy()]]
    out = {}
    for code, name, band in (
        (core.ATTACHED, "attached", lambda x: x >= TE_ATTACHED_MIN_EV),
        (core.DETACHED, "detached", lambda x: x <= TE_DETACHED_MAX_EV),
        (core.MARFE, "marfe", lambda x: x <= TE_DETACHED_MAX_EV),
    ):
        rows = sel[sel[state_column] == code]
        out[name] = {
            **summarise(rows.te_ev.to_numpy(float), band),
            "n_shots": int(rows.shot.nunique()),
            "shot_ids": sorted(int(s) for s in rows.shot.unique()),
        }
    two = sel[sel[state_column].isin((core.ATTACHED, core.DETACHED))]
    pooled = bench.auroc_boot(
        -two.te_ev.to_numpy(float),
        (two[state_column] == core.DETACHED).to_numpy(),
        two.shot.to_numpy(),
    )
    per_shot = []
    for _, rows in two.groupby("shot"):
        positive = (rows[state_column] == core.DETACHED).to_numpy()
        if min(positive.sum(), (~positive).sum()) >= MIN_SHOT_BINS:
            per_shot.append(bench.auroc(-rows.te_ev.to_numpy(float), positive))
    finite = np.asarray([v for v in per_shot if np.isfinite(v)])
    out["auroc_neg_te_detached_vs_attached"] = {
        "pooled": pooled,
        "within_shot": {
            "min_class_bins": MIN_SHOT_BINS,
            "n_shots": len(finite),
            "mean": float(finite.mean()) if len(finite) else None,
            "mean_ci95": bench.mean_boot(finite),
        },
    }
    return out


def te_subset(frame: pd.DataFrame, state: pd.Series, mask) -> tuple:
    """(score, positive, shots) for the attached/detached bins of `state` in `mask`
    that carry a Te: the score is -Te and the positive class is detached."""
    keep = (
        np.asarray(mask, bool)
        & np.isfinite(frame.te_ev).to_numpy()
        & state.isin((core.ATTACHED, core.DETACHED)).to_numpy()
    )
    rows = frame[keep]
    return (
        -rows.te_ev.to_numpy(float),
        (state[keep] == core.DETACHED).to_numpy(),
        rows.shot.to_numpy(),
    )


def paired_difference(a: tuple, b: tuple) -> dict:
    """AUROC(-Te) of subset `a` minus that of subset `b`, one shot bootstrap that
    resamples the shots of both subsets together (so shared shots stay paired).
    Replicates where either subset has a single class are dropped and counted."""
    rng = bench.boot_rng()
    shots = np.unique(np.concatenate([a[2], b[2]]))
    if not len(shots):
        return {"difference": None, "ci95": [None, None], "n_shots": 0}
    index = [{s: np.flatnonzero(x[2] == s) for s in shots} for x in (a, b)]
    full = bench.auroc(a[0], a[1]) - bench.auroc(b[0], b[1])
    draws = []
    for _ in range(bench.REPLICATES):
        pick = shots[rng.integers(0, len(shots), len(shots))]
        values = []
        for x, ind in zip((a, b), index, strict=True):
            rows = np.concatenate([ind[s] for s in pick])
            values.append(bench.auroc(x[0][rows], x[1][rows]))
        if np.isfinite(values).all():
            draws.append(values[0] - values[1])
    return {
        "difference": float(full) if np.isfinite(full) else None,
        "ci95": bench.interval(draws),
        "n_shots": len(shots),
        "n_shots_first_subset": len(np.unique(a[2])),
        "n_shots_second_subset": len(np.unique(b[2])),
        "valid_replicates": len(draws),
        "dropped_replicates": bench.REPLICATES - len(draws),
    }


def second_vote_verdict(difference: dict) -> str:
    """Plain reading of a paired difference: better, worse, not distinguishable, or
    not estimable (fewer than `MIN_PAIRED_SHOTS` shots behind the first subset, or an
    interval that is not finite)."""
    lo, hi = difference["ci95"]
    few = difference.get("n_shots_first_subset", 0) < MIN_PAIRED_SHOTS
    if difference["difference"] is None or lo is None or not np.isfinite(lo) or few:
        return "not estimable"
    if lo > 0:
        return "better"
    if hi < 0:
        return "worse"
    return "not distinguishable"


def reading_text(difference: dict) -> str:
    """The reading as a sentence fragment: 'not estimable on 6 shots' and so on."""
    reading = second_vote_verdict(difference)
    if reading == "not estimable":
        return f"not estimable on {difference.get('n_shots_first_subset', 0)} shots"
    return reading


def tangtv_before_gate(frame: pd.DataFrame) -> pd.Series:
    """The TangTV vote where it is valid on the upper shelf, else abstain: the vote
    before the regime gate."""
    return frame.tangtv_vote.where(
        frame.tangtv_valid.astype(bool) & frame.tangtv_tier.eq("upper_shelf"),
        core.ABSTAIN,
    )


def tier_scores(frame: pd.DataFrame) -> dict:
    """Te agreement per label tier, TangTV alone, and the second vote's effect."""
    tangtv = tangtv_before_gate(frame)
    frame = frame.assign(tangtv_before_gate=tangtv)
    certain = frame.tier.eq("certain").to_numpy()
    silver = frame.tier.eq("tangtv_only").to_numpy()
    lmode = frame.tier.eq("tangtv_only_lmode").to_numpy()
    conflict = frame.tier.eq("conflict").to_numpy() & tangtv.isin((1, 2)).to_numpy()
    everywhere = np.ones(len(frame), bool)
    out = {
        "certain": score_states(frame, "state_rule", mask=certain),
        "tangtv_only": score_states(frame, "state_rule", mask=silver),
        "certain_or_tangtv_only": score_states(frame, "state_rule"),
        "tangtv_vote_before_gate": score_states(frame, "tangtv_before_gate"),
        "tangtv_vote_before_gate_without_201081": score_states(
            frame, "tangtv_before_gate", mask=(frame.shot != ANCHOR_SHOT).to_numpy()
        ),
        "tangtv_vote_before_gate_in_conflict_bins": score_states(
            frame, "tangtv_before_gate", mask=conflict
        ),
        "tangtv_vote_before_gate_in_gated_bins": score_states(
            frame, "tangtv_before_gate", mask=lmode
        ),
    }
    state = frame.state_rule
    a = te_subset(frame, state, certain)
    b = te_subset(frame, tangtv, everywhere)
    c = te_subset(frame, tangtv, silver | conflict)
    difference = paired_difference(a, b)
    against_rest = paired_difference(a, c)
    out["second_vote_effect"] = {
        "question": "does requiring an agreeing second indicator improve the "
        "agreement of the TangTV state with divertor Thomson Te?",
        "certain_minus_tangtv_before_gate": {
            **difference,
            "reading": second_vote_verdict(difference),
            "reading_text": reading_text(difference),
            "note": "AUROC(-Te) of the certain bins minus that of every TangTV "
            "attached/detached vote on the upper shelf that has a Te; the certain "
            "bins are a subset of them",
        },
        "certain_minus_tangtv_where_second_vote_missing_or_clashing": {
            **against_rest,
            "reading": second_vote_verdict(against_rest),
            "reading_text": reading_text(against_rest),
            "note": "against the TangTV votes that are tangtv_only or in conflict",
        },
    }
    return out


def gated_by_shot(frame: pd.DataFrame) -> dict:
    """What the gate withholds, shot by shot: the TangTV detached votes (before the
    gate) on known L-mode and probable-L bins, with those at DZ >= 0.8 (the cold
    ones) and the Te in band (<= 5 eV) among the bins that have a Te."""
    vote = tangtv_before_gate(frame).eq(core.DETACHED).to_numpy()
    out = {}
    for regime in ("L", "probable_L"):
        rows = frame[vote & frame.regime.eq(regime).to_numpy()]
        per_shot = {}
        for shot, group in rows.groupby("shot"):
            te = group.te_ev[np.isfinite(group.te_ev)]
            per_shot[str(int(shot))] = {
                "n_bins": len(group),
                "n_dz_at_least_0_8": int((group.tangtv_value >= 0.8).sum()),
                "n_dz_below_0_8": int((group.tangtv_value < 0.8).sum()),
                "n_with_te": len(te),
                "n_te_at_most_5_ev": int((te <= TE_DETACHED_MAX_EV).sum()),
            }
        keys = (
            "n_bins",
            "n_dz_at_least_0_8",
            "n_dz_below_0_8",
            "n_with_te",
            "n_te_at_most_5_ev",
        )
        out[regime] = {
            "per_shot": per_shot,
            "total": {k: sum(v[k] for v in per_shot.values()) for k in keys},
            "n_shots": len(per_shot),
        }
    return out


def regime_scores(frame: pd.DataFrame) -> dict:
    """Te agreement by regime, before the gate (the TangTV vote with no L-mode gate)
    and after it (the exported attached/detached states), and by DZ band.

    The gate was set a priori from the H-mode origin of the DZ cutoffs; this block
    describes what it removed and does not select anything. `probable_H` is
    report-only (no gate acts on it).
    """
    tangtv = tangtv_before_gate(frame)
    frame = frame.assign(tangtv_before_gate=tangtv)
    out = {
        "note": "regime from the regime source of the label model (the confinement "
        "suggestion table: curated Gill and Butt intervals where they exist, else "
        "the dalpha_lh detector; then the D-alpha H-mode detector; else unknown), "
        "then the a-priori proxies for a shot with no regime (`probable_L`: ELM "
        "coverage known, no ELM share in the window, median P_in under 2 MW; "
        "`probable_H`: any ELM share, report only). The gate acts on L and "
        "probable_L only; it is a priori (the DZ cutoffs come from an H-mode shot) "
        "and is reported here, not tuned on Te. Rows of `tangtv_vote_before_gate` "
        "are the check against Te; rows of `exported_state` are post-selection",
        "bins_by_regime": {r: int(frame.regime.eq(r).sum()) for r in REGIMES},
        "shots_by_regime": {
            r: int(frame.loc[frame.regime.eq(r), "shot"].nunique()) for r in REGIMES
        },
        "tangtv_vote_before_gate": {},
        "exported_state": {},
        "tangtv_detached_by_dz_band": {},
        "gated_detached_votes_by_shot": gated_by_shot(frame),
    }
    for regime in REGIMES:
        mask = frame.regime.eq(regime).to_numpy()
        out["tangtv_vote_before_gate"][regime] = score_states(
            frame, "tangtv_before_gate", mask
        )
        out["exported_state"][regime] = score_states(frame, "state_rule", mask)
        bands = {}
        for lo, hi in DZ_BANDS:
            rows = frame[
                mask
                & frame.tangtv_before_gate.eq(core.DETACHED).to_numpy()
                & np.isfinite(frame.te_ev).to_numpy()
                & (frame.tangtv_value >= lo).to_numpy()
                & (frame.tangtv_value < hi).to_numpy()
            ]
            bands[f"{lo:g}-{hi:g}"] = {
                **summarise(
                    rows.te_ev.to_numpy(float), lambda x: x <= TE_DETACHED_MAX_EV
                ),
                "n_shots": int(rows.shot.nunique()),
            }
        out["tangtv_detached_by_dz_band"][regime] = bands
    return out


def cliffs(frame: pd.DataFrame, shot: int) -> dict | None:
    """Te-cliff times of one shot read from the data: the first and last bin in
    its flat top where the Te median of 3 bins crosses the 5-10 eV gap."""
    rows = frame[(frame.shot == shot) & np.isfinite(frame.te_ev)]
    if rows.empty:
        return None
    t = rows.start_ms.to_numpy(float) + core.BIN_MS / 2
    te = rows.te_ev.rolling(3, center=True, min_periods=1).median().to_numpy(float)
    cold = te <= TE_DETACHED_MAX_EV
    warm = te >= TE_ATTACHED_MIN_EV
    first_cold = float(t[cold][0]) if cold.any() else None
    last_cold = float(t[cold][-1]) if cold.any() else None
    return {
        "first_cold_ms": first_cold,
        "last_cold_ms": last_cold,
        "warm_before_first_cold_share": float(np.mean(warm[t < first_cold]))
        if first_cold
        else None,
        "published_cliffs_ms": list(PUBLISHED_CLIFFS_MS.get(shot, ())),
        "te_ev_at_2000_ms_and_3500_ms_and_5000_ms": [
            float(
                np.nanmedian(
                    rows.te_ev[(rows.start_ms >= x - 100) & (rows.start_ms <= x + 100)]
                )
            )
            for x in (2000.0, 3500.0, 5000.0)
        ],
    }


def main() -> int:
    frame = pd.read_csv(ROOT / "labels_bins.csv.gz")
    frame = attach_te(frame)
    fetched = sorted(int(p.stem) for p in (ROOT / "dts").glob("*.npz"))
    usable = sorted(
        int(s) for s in frame.loc[np.isfinite(frame.te_ev), "shot"].unique()
    )
    record = {
        "bin_ms": core.BIN_MS,
        "method": __doc__.split("\n\n")[1:5],
        "bands_ev": {
            "attached_min": TE_ATTACHED_MIN_EV,
            "detached_max": TE_DETACHED_MAX_EV,
            "valid": list(TE_VALID_EV),
        },
        "chord_selection": {
            "above_shelf_m": [CHORD_MIN_ABOVE_SHELF_M, CHORD_MAX_ABOVE_SHELF_M],
            "shelf_z_m": th.SHELF_Z,
            "psi_n_window": [PSI_N_MIN, PSI_N_MAX],
        },
        "shots_with_dts_fetched": fetched,
        "shots_with_te_in_assessed_bins": usable,
        "bins_with_te": int(np.isfinite(frame.te_ev).sum()),
        "bins_assessed": len(frame),
        "primary_state": score_states(frame, "state_rule"),
        "by_tier": tier_scores(frame),
        "by_regime": regime_scores(frame),
        "indicator_votes_population": "assessed bins of the exported label set "
        "(labels_bins.csv.gz), valid votes only",
        "indicator_votes": {
            name: score_states(
                frame.assign(
                    vote=frame[f"{name}_vote"].where(frame[f"{name}_valid"], 0)
                ),
                "vote",
            )
            for name in ("afrac", "prad", "tangtv")
        },
        "cliff_check": {str(s): cliffs(frame, s) for s in PUBLISHED_CLIFFS_MS},
    }
    OUT.write_text(dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("bins_with_te", "bins_assessed")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
