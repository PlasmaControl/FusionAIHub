#!/usr/bin/env python
"""Does the Afrac proxy measure detachment or the probe it happened to read?

Opus re-review 5 (C1) found that the earlier Afrac read 0.79 near the separatrix and
0.10 further out for the same TangTV state: it measured the selected probe's flux
position. The proxy was rebuilt (`src/labeler/events/detachment/afrac.py`): each
probe has its own attached reference, the probe nearest the separatrix inside
`AFRAC_PSI_WINDOW` is read, and bins in a known L-mode stretch abstain. This script
scores the rebuilt proxy and writes `docs/labeler/results/detachment_afrac_check.json`:

* the median Afrac and vote shares against the selected probe's psiN, before (the
  earlier labels, kept in `$LABELER_ROOT/round4/detach/fix_round5_before`) and
  after, and with the TangTV state held fixed;
* the proxy against the TangTV state (threshold-free AUROC of -Afrac for TangTV
  detached against attached, upper shelf) and against divertor Thomson Te (AUROC of
  -Te for the Afrac detached against attached votes, and AUROC of -Afrac for cold
  (<= 5 eV) against warm (>= 10 eV) Te), with 1000-replicate shot bootstraps;
* the flux window swept to 0.015 and 0.02 (`detach_bins.py --afrac-window`);
* the L/H gate: how many bins and shots the regime is known on and from what source;
* the decision to keep Afrac in the vote: it stays unless its AUROC against Te is
  below `KEEP_MIN_AUROC` (0.65, the owner's bar for round 5).

    for w in 0.015 0.02; do
      python scripts/labeler/detach_bins.py --shots-file shots_afrac_window.txt \\
          --redo --afrac-window $w --out-dir $LABELER_ROOT/round4/detach/bins_afw$w
    done
    python scripts/labeler/detach_afrac_check.py

The shots are the 37 on which the earlier label set had any assessed bin. No
threshold is fitted here and the cohort test split holds none of them.
"""

from __future__ import annotations

import os
from pathlib import Path

import detach_benchmark as bench
import detach_label as dl
import detach_te_check as tc
import numpy as np
import pandas as pd
from detach_json import dumps

from labeler.events.detachment import core
from labeler.events.detachment import thresholds as th

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ["LABELER_ROOT"]) / "round4/detach"
OUT = REPO / "docs/labeler/results/detachment_afrac_check.json"
BEFORE = ROOT / "fix_round5_before/labels_bins.csv.gz"
SHOTS = ROOT / "shots_afrac_window.txt"
WINDOW_DIRS = {
    th.AFRAC_PSI_WINDOW: ROOT / "bins",
    0.015: ROOT / "bins_afw0.015",
    0.02: ROOT / "bins_afw0.02",
}
#: Afrac is kept in the vote only if its AUROC against Te reaches this.
KEEP_MIN_AUROC = 0.65
PSI_EDGES = (0.98, 0.995, 1.0, 1.005, 1.01, 1.02, 1.03, 1.05, 1.10)
TE_COLD_MAX_EV = tc.TE_DETACHED_MAX_EV
TE_WARM_MIN_EV = tc.TE_ATTACHED_MIN_EV
#: Shot 201081's phases from its published cliff times (2650 and 4450 ms), each
#: trimmed by 150 ms on both sides so a bin is clear of the cliff.
PHASE_SHOT = 201081
PHASES_MS = {
    "attached_early": (1900.0, 2500.0),
    "detached": (2800.0, 4300.0),
    "reattached": (4600.0, 5400.0),
}
STRIKE_REAL_R_M = (0.8, 2.5)


def load_bins(directory: Path, shots: set[int]) -> pd.DataFrame:
    frames = [
        dl.load_one(directory / f"{s}.npz")
        for s in sorted(shots)
        if (directory / f"{s}.npz").is_file()
    ]
    return pd.concat(frames, ignore_index=True)


def clean(value):
    """A JSON-safe float (None for NaN)."""
    value = float(value)
    return value if np.isfinite(value) else None


def psin_table(frame: pd.DataFrame, state: int | None = None) -> dict:
    """Median Afrac and Afrac vote shares by the selected probe's psiN bin.

    With `state`, only upper-shelf bins that TangTV votes `state` (attached or
    detached): the TangTV state is then held fixed, so any change with psiN is the
    proxy's dependence on the probe position, not on the plasma.
    """
    valid = frame.afrac_valid.astype(bool) & np.isfinite(frame.aux_jsat_selected_psin)
    if state is not None:
        valid &= (
            frame.tangtv_tier.eq("upper_shelf")
            & frame.tangtv_valid.astype(bool)
            & frame.tangtv_vote.eq(state)
        )
    rows = frame[valid]
    cut = pd.cut(rows.aux_jsat_selected_psin, list(PSI_EDGES))
    out = {}
    for interval, group in rows.groupby(cut, observed=True):
        out[f"({interval.left:g}, {interval.right:g}]"] = {
            "n_bins": len(group),
            "n_shots": int(group.shot.nunique()),
            "median_afrac": clean(group.afrac_value.median()),
            "share_attached_vote": float((group.afrac_vote == core.ATTACHED).mean()),
            "share_detached_vote": float((group.afrac_vote == core.DETACHED).mean()),
        }
    return out


def psin_dependence(frame: pd.DataFrame, state: int) -> dict:
    """Spearman correlation of Afrac with the selected probe's psiN, TangTV state
    fixed, and the medians on the two sides of the separatrix."""
    rows = frame[
        frame.afrac_valid.astype(bool)
        & np.isfinite(frame.aux_jsat_selected_psin)
        & frame.tangtv_tier.eq("upper_shelf")
        & frame.tangtv_valid.astype(bool)
        & frame.tangtv_vote.eq(state)
    ]
    inner = rows[rows.aux_jsat_selected_psin <= 1.005]
    outer = rows[rows.aux_jsat_selected_psin > 1.01]
    return {
        "tangtv_state": core.STATE_NAMES[state],
        "n_bins": len(rows),
        "n_shots": int(rows.shot.nunique()),
        "spearman_afrac_vs_psin": clean(
            bench.spearman(
                rows.afrac_value.to_numpy(float),
                rows.aux_jsat_selected_psin.to_numpy(float),
            )
        )
        if len(rows) > 2
        else None,
        "median_afrac_psin_at_most_1.005": clean(inner.afrac_value.median())
        if len(inner)
        else None,
        "n_bins_psin_at_most_1.005": len(inner),
        "median_afrac_psin_above_1.01": clean(outer.afrac_value.median())
        if len(outer)
        else None,
        "n_bins_psin_above_1.01": len(outer),
    }


def vs_tangtv(frame: pd.DataFrame, rng) -> dict:
    """The proxy against the TangTV state on the upper shelf (both valid)."""
    both = (
        frame.afrac_valid.astype(bool)
        & frame.tangtv_valid.astype(bool)
        & frame.tangtv_tier.eq("upper_shelf")
        & np.isfinite(frame.afrac_value)
        & frame.tangtv_vote.isin((core.ATTACHED, core.DETACHED))
    )
    rows = frame[both]
    out = {
        "population": "upper-shelf bins where Afrac is valid and TangTV votes "
        "attached or detached",
        "auroc_neg_afrac_tangtv_detached_vs_attached": bench.auroc_boot(
            -rows.afrac_value.to_numpy(float),
            rows.tangtv_vote.eq(core.DETACHED).to_numpy(),
            rows.shot.to_numpy(),
            rng,
        ),
    }
    cast = rows[rows.afrac_vote.isin((core.ATTACHED, core.DETACHED))]
    table = pd.crosstab(
        cast.tangtv_vote.map({1: "tangtv_attached", 2: "tangtv_detached"}),
        cast.afrac_vote.map({1: "afrac_attached", 2: "afrac_detached"}),
    )
    out["vote_table"] = {
        str(r): {str(c): int(table.loc[r, c]) for c in table.columns}
        for r in table.index
    }
    out["vote_agreement"] = {
        "bins": len(cast),
        "shots": int(cast.shot.nunique()),
        "agreement": clean((cast.afrac_vote == cast.tangtv_vote).mean())
        if len(cast)
        else None,
        "kappa": clean(
            dl.cohen_kappa(cast.afrac_vote.to_numpy(), cast.tangtv_vote.to_numpy())
        )
        if len(cast)
        else None,
    }
    return out


def vs_te(frame: pd.DataFrame, rng) -> dict:
    """The proxy against divertor Thomson Te (chords and bands of `detach_te_check`)."""
    withte = tc.attach_te(frame[["shot", "start_ms"]].copy())
    frame = frame.assign(te_ev=withte.te_ev.to_numpy())
    voted = frame.assign(
        vote=frame.afrac_vote.where(frame.afrac_valid.astype(bool), core.ABSTAIN)
    )
    vote_based = tc.score_states(voted, "vote", rng)
    scored = frame[
        frame.afrac_valid.astype(bool)
        & np.isfinite(frame.afrac_value)
        & np.isfinite(frame.te_ev)
        & ((frame.te_ev <= TE_COLD_MAX_EV) | (frame.te_ev >= TE_WARM_MIN_EV))
    ]
    value_based = bench.auroc_boot(
        -scored.afrac_value.to_numpy(float),
        (scored.te_ev <= TE_COLD_MAX_EV).to_numpy(),
        scored.shot.to_numpy(),
        rng,
    )
    return {
        "bands_ev": {"cold_max": TE_COLD_MAX_EV, "warm_min": TE_WARM_MIN_EV},
        "votes_attached_and_detached": {
            "attached": vote_based["attached"],
            "detached": vote_based["detached"],
            "auroc_neg_te_detached_vs_attached": vote_based[
                "auroc_neg_te_detached_vs_attached"
            ],
        },
        "value_cold_vs_warm_te": {
            "note": "threshold-free in Afrac: AUROC of -Afrac for Te <= 5 eV against "
            "Te >= 10 eV, every valid Afrac bin with such a Te",
            **value_based,
            "cold_shots": int(
                scored.loc[scored.te_ev <= TE_COLD_MAX_EV, "shot"].nunique()
            ),
        },
    }


def phases(frame: pd.DataFrame) -> dict:
    rows = frame[(frame.shot == PHASE_SHOT) & frame.afrac_valid.astype(bool)]
    out = {}
    for name, (lo, hi) in PHASES_MS.items():
        window = rows[(rows.start_ms >= lo) & (rows.start_ms < hi)]
        out[name] = {
            "time_ms": [lo, hi],
            "n_bins": len(window),
            "median_afrac": clean(window.afrac_value.median()) if len(window) else None,
            "probes": sorted(int(p) for p in window.aux_jsat_selected_probe.unique()),
        }
    return out


def window_row(frame: pd.DataFrame, rng) -> dict:
    valid = frame.afrac_valid.astype(bool)
    return {
        "bins_valid": int(valid.sum()),
        "shots_valid": int(frame.loc[valid, "shot"].nunique()),
        "reasons_invalid": {
            str(k): int(v)
            for k, v in frame.loc[~valid, "afrac_reason"].value_counts().items()
        },
        "vs_tangtv": vs_tangtv(frame, rng),
        "vs_te": vs_te(frame, rng),
        "shot_201081_phase_medians": phases(frame),
    }


def regime_block(frame: pd.DataFrame) -> dict:
    """Where the L/H gate knows the regime and what it removed."""
    out = {
        "rule": "bins in a known L-mode stretch abstain (reason l_mode) and enter no "
        "reference; a bin whose regime is unknown is not gated",
        "source_order": [
            "confinement suggestion table (category 2 low = L; 1, 3, 4 = H)",
            (
                "D-alpha H-mode detector (L = measured time outside every H or "
                "uncertain span)"
            ),
            "unknown",
        ],
        "per_shot": {},
    }
    for shot, rows in frame.groupby("shot"):
        out["per_shot"][str(int(shot))] = {
            "bins": len(rows),
            "bins_L": int((rows.regime == "L").sum()),
            "bins_H": int((rows.regime == "H").sum()),
            "bins_unknown": int((rows.regime == "unknown").sum()),
            "source": sorted(str(s) for s in rows.regime_source.unique()),
            "afrac_l_mode_abstentions": int((rows.afrac_reason == "l_mode").sum()),
            "afrac_valid": int(rows.afrac_valid.astype(bool).sum()),
        }
    known = frame.regime.isin(("L", "H"))
    out["totals"] = {
        "bins": len(frame),
        "bins_regime_known": int(known.sum()),
        "bins_L": int((frame.regime == "L").sum()),
        "bins_H": int((frame.regime == "H").sum()),
        "bins_unknown": int((frame.regime == "unknown").sum()),
        "shots": int(frame.shot.nunique()),
        "shots_with_known_regime": int(frame.loc[known, "shot"].nunique()),
        "afrac_l_mode_abstentions": int((frame.afrac_reason == "l_mode").sum()),
        "by_source": {
            str(k): int(v) for k, v in frame.regime_source.value_counts().items()
        },
    }
    return out


def sentinel_strikes(frame: pd.DataFrame) -> dict:
    """Bins whose exported outer strike point is not a real EFIT position (M1)."""
    r = frame.aux_jsat_strike_r_m
    bad = np.isfinite(r) & ((r < STRIKE_REAL_R_M[0]) | (r > STRIKE_REAL_R_M[1]))
    return {
        "bins_with_a_strike_point": int(np.isfinite(r).sum()),
        "bins_with_a_sentinel_strike_point": int(bad.sum()),
        "shots": sorted(int(s) for s in frame.loc[bad, "shot"].unique()),
    }


def main() -> int:
    shots = {int(s) for s in SHOTS.read_text().split()}
    rng = np.random.default_rng(0)
    before = pd.read_csv(BEFORE)
    before = before[before.shot.isin(shots)].reset_index(drop=True)
    windows = {w: load_bins(d, shots) for w, d in WINDOW_DIRS.items()}
    primary = windows[th.AFRAC_PSI_WINDOW]

    rows = {f"{w:g}": window_row(frame, rng) for w, frame in windows.items()}
    key = f"{th.AFRAC_PSI_WINDOW:g}"
    te_vote = rows[key]["vs_te"]["votes_attached_and_detached"][
        "auroc_neg_te_detached_vs_attached"
    ]["pooled"]
    te_value = rows[key]["vs_te"]["value_cold_vs_warm_te"]
    statistic = min(te_vote["value"], te_value["value"])
    keep = bool(np.isfinite(statistic) and statistic >= KEEP_MIN_AUROC)

    record = {
        "script": "scripts/labeler/detach_afrac_check.py",
        "method": __doc__.split("\n\n")[0:2],
        "shots": sorted(shots),
        "n_shots": len(shots),
        "primary_window_psin": th.AFRAC_PSI_WINDOW,
        "reference_quantile": th.AFRAC_REFERENCE_QUANTILE,
        "reference_min_bins": th.AFRAC_REFERENCE_MIN_BINS,
        "votes_cutoffs": {
            "attached_min": th.AFRAC_ATTACHED_MIN,
            "detached_max": th.AFRAC_DETACHED_MAX,
        },
        "cause_of_the_earlier_failure": (
            "The earlier proxy read the peak-current probe among those with psiN in "
            "(1.000, 1.05] and at least 5 mm outboard of the outer strike point, "
            "against one whole-shot reference. The strike-point probe, which shows "
            "the rollover, was excluded, and which probe was read followed "
            "millimetre-scale strike-point motion; probe spacing (1.1 to 4.4 cm) is "
            "wider than the current peak, so a fixed-psiN interpolation was not an "
            "alternative. The proxy then varied with the read probe's flux position, "
            "not with the plasma state."
        ),
        "before_after": {
            "before": {
                "source": "$LABELER_ROOT/round4/detach/fix_round5_before/"
                "labels_bins.csv.gz (the round-4 labels, assessed bins)",
                "valid_bins": int(before.afrac_valid.astype(bool).sum()),
                "valid_shots": int(
                    before.loc[before.afrac_valid.astype(bool), "shot"].nunique()
                ),
                "by_psin": psin_table(before),
                "by_psin_tangtv_attached": psin_table(before, core.ATTACHED),
                "by_psin_tangtv_detached": psin_table(before, core.DETACHED),
                "dependence_tangtv_attached": psin_dependence(before, core.ATTACHED),
                "dependence_tangtv_detached": psin_dependence(before, core.DETACHED),
                "sentinel_strikes": sentinel_strikes(before),
            },
            "after": {
                "source": "$LABELER_ROOT/round4/detach/bins (every extracted bin of "
                "the same 37 shots)",
                "valid_bins": rows[key]["bins_valid"],
                "valid_shots": rows[key]["shots_valid"],
                "by_psin": psin_table(primary),
                "by_psin_tangtv_attached": psin_table(primary, core.ATTACHED),
                "by_psin_tangtv_detached": psin_table(primary, core.DETACHED),
                "dependence_tangtv_attached": psin_dependence(primary, core.ATTACHED),
                "dependence_tangtv_detached": psin_dependence(primary, core.DETACHED),
                "sentinel_strikes": sentinel_strikes(primary),
            },
        },
        "windows": rows,
        "regime_gate": regime_block(primary),
        "decision": {
            "rule": f"Afrac stays in the vote unless its AUROC against Te is below "
            f"{KEEP_MIN_AUROC} (the lower of the vote-based and the value-based "
            "point estimates at the primary window)",
            "keep_min_auroc": KEEP_MIN_AUROC,
            "auroc_vote_based": te_vote["value"],
            "auroc_value_based": te_value["value"],
            "statistic": statistic,
            "keep_afrac_in_vote": keep,
        },
    }
    OUT.write_text(dumps(record, indent=1) + "\n")
    print(
        dumps(
            {
                "decision": record["decision"],
                "valid_bins_by_window": {
                    w: (r["bins_valid"], r["shots_valid"]) for w, r in rows.items()
                },
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
