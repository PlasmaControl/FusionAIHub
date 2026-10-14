"""Freeze the m/n model once: fit on the dev corpus tables without any gold shot (one-off).

Inputs (dev/mhd, read-only): the corpus track census `dev/mhd/corpus/out/tracks_census.csv`,
its frames `frames_all.csv`, the track-coherent features
`$LABELER_ROOT/mhd/fusion/tracks_feat_cut0.95.npz` and the parked EFIT02 and ZIPFIT files.
The ECE locators are recomputed here from the features (`views.locators`). The label model is
fit by EM on every modelled track (rules 1-4 sets clamped, dev's `resonance.corpus_posterior`)
and the resonance law on the anchors 2/1, 3/2, 4/3; both go to `freeze.json` with the sha256
of the fit's shot list.

The check applies the frozen model to the same corpus tracks and to the eight case shots'
tracks (dev/mhd/cases, features in `feat_cases/`) and compares the statuses and labels with
dev/mhd/fusion/out/resonance_corpus.csv and cases_res.csv. The case shots' dev posterior was
fit without the three corpus case shots, so small differences there are expected.

    python -m labeler.mhd.freeze --check-only | --write

`--check-only` prints the check against the frozen file and writes nothing: the file's sha
keys every cache and table. Refitting and overwriting `freeze.json` needs `--write`; with
neither flag the command refuses.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Paths, git_sha
from . import labelmodel as LM
from . import resonance as RE
from . import rules, views
from .equilibrium import efit02, rot_profile, rotation

REPO = Path(__file__).resolve().parents[3]
DEV = REPO / "dev" / "mhd"
FREEZE = Path(__file__).with_name("freeze.json")
CUT = views.CUT
GOLD = "gold/gold_shots.csv"
TAB_COLUMNS = [
    "shot",
    "family_id",
    "n",
    "frames",
    "t0",
    "t1",
    "f",
    "qmin",
    "q95",
    "par",
    "harm",
    "anchor4",
    "list4",
]


def load() -> dict:
    """The frozen model: params, law, constants."""
    return json.loads(FREEZE.read_text())


def sha() -> str:
    """The freeze file's sha256 (cache keys and metas)."""
    return hashlib.sha256(FREEZE.read_bytes()).hexdigest()[:16]


def keep_rows(tab: pd.DataFrame) -> np.ndarray:
    """dev's modelled tracks: not a harmonic, n 1-3, not a line or harmonic by rules 1-4."""
    return (
        (~tab.harm.astype(bool))
        & tab.n.isin([1, 2, 3])
        & ~tab.anchor4.isin(["harmonic", "line"])
    ).values


def corpus(paths: Paths):
    """The dev corpus tracks the fit uses: (tab, F) in the features file's order."""
    F = dict(np.load(paths.root / "mhd" / "fusion" / f"tracks_feat_cut{CUT:g}.npz"))
    cen = pd.read_csv(DEV / "corpus" / "out" / "tracks_census.csv", usecols=TAB_COLUMNS)
    key = pd.DataFrame({"shot": F["shot"], "family_id": F["family_id"]})
    tab = key.merge(cen, on=["shot", "family_id"], how="left")
    keep = keep_rows(tab)
    idx = np.flatnonzero(keep)
    tab = tab.iloc[idx].reset_index(drop=True)
    F = {
        k: v[idx]
        for k, v in F.items()
        if getattr(v, "ndim", 0) >= 1 and len(v) == len(keep)
    }
    return tab, F


def gold_shots(paths: Paths) -> set[int]:
    return {int(s) for s in pd.read_csv(paths.label_tables / GOLD).shot}


def fit(paths: Paths) -> dict:
    tab, F = corpus(paths)
    gold = gold_shots(paths)
    dropped = sorted(set(tab.shot) & gold)
    if dropped:
        keep = ~tab.shot.isin(gold).values
        tab = tab[keep].reset_index(drop=True)
        F = {k: v[keep] for k, v in F.items()}
    tab = pd.concat([tab, views.locators(F)], axis=1)
    shots = sorted(int(s) for s in tab.shot.unique())
    eqs = {s: efit02(paths, s) for s in shots}
    rots_u = {s: rot_profile(paths, s) for s in shots}
    lm, _ = LM.fit(tab, F, eqs, rots_u)
    frames = pd.read_csv(
        DEV / "corpus" / "out" / "frames_all.csv",
        usecols=["shot", "family_id", "t_ms", "f0"],
    )
    sl = RE.slices(tab, frames, eqs, RE.load_rotation(paths, shots))
    law = RE.fit_law(sl, tab)
    listed = "\n".join(str(s) for s in shots).encode()
    return {
        "method": "mhd_modes",
        "version": "v2",
        "made": dt.datetime.now().astimezone().date().isoformat(),
        "git_sha": git_sha(),
        "fit": {
            "source": "dev/mhd/corpus/out/tracks_census.csv + "
            f"$LABELER_ROOT/mhd/fusion/tracks_feat_cut{CUT:g}.npz (ECE locators recomputed)",
            "shots": len(shots),
            "shots_sha256": hashlib.sha256(listed).hexdigest(),
            "shot_list": "the sorted shot numbers, one per line, no trailing newline",
            "gold_shots_dropped": dropped,
            "tracks": len(tab),
            "anchors": {
                c: int((tab.anchor4 == c).sum()) for c in ("2/1", "3/2", "4/3", "3/1")
            },
            "em_iterations": LM.ITERS,
            "law_slices": law["N"],
        },
        "constants": {
            "cut": CUT,
            "locator": "L_jump, else L_amp",
            "thr": views.THR,
            "views": LM.ALL2,
            "weights": LM.WEIGHT,
            "q_margin": LM.Q_MARGIN,
            "sig_q": LM.SIG_Q,
            "floor_q": LM.FLOOR_Q,
            "rot_range_khz": LM.ROT_RANGE,
            "r_star": RE.R_STAR,
            "p_conf": RE.P_CONF,
            "q_tol": RE.Q_TOL,
            "dt_rot_ms": RE.DT_ROT,
            "dt_f_ms": RE.DT_F,
            "max_slices": RE.MAX_SLICES,
            "eq_max_dt_ms": rules.EQ_MAX_DT,
            "margin": rules.MARGIN,
        },
        "params": lm.params(),
        "law": {k: law[k] for k in ("a", "b", "s_f", "s_q", "frac_corot")},
    }


def apply(tab, F, frames, paths: Paths, frozen: dict) -> pd.DataFrame:
    """Statuses of `tab`'s tracks under the frozen model."""
    shots = sorted(int(s) for s in tab.shot.unique())
    eqs = {s: efit02(paths, s) for s in shots}
    rots_u = {s: rot_profile(paths, s) for s in shots}
    P = LM.posterior(tab, F, eqs, rots_u, frozen["params"])
    sl = RE.slices(tab, frames, eqs, {s: rotation(paths, s) for s in shots})
    R, A, fn, top = RE.distances(sl, tab, frozen["law"])
    label, status, reason, _, p, p_res = RE.decide(P, R, A, fn, top, frozen["law"])
    out = tab[["shot", "family_id"]].copy()
    out["label"], out["status"], out["reason"] = label, status, reason
    out["p"], out["p_res"] = p, p_res
    out["map"] = np.array(LM.NAMES)[P.argmax(axis=1)]
    return out


def cases(paths: Paths):
    """The case shots' modelled tracks (dev/mhd/cases) with their features and locators."""
    cen = pd.read_csv(DEV / "cases" / "out" / "tracks.csv", usecols=TAB_COLUMNS)
    rows = []
    for p in sorted(
        glob.glob(str(paths.root / "mhd" / "fusion" / "feat_cases" / "*.npz"))
    ):
        shot = int(os.path.basename(p)[:-4])
        rows += [dict(r, shot=shot) for r in views.track_features(dict(np.load(p)))]
    key = {(r["shot"], r["family_id"]): r for r in rows}
    cen = cen[[(s, f) in key for s, f in zip(cen.shot, cen.family_id)]]
    cen = cen[keep_rows(cen)].reset_index(drop=True)
    F = views.stack([key[(s, f)] for s, f in zip(cen.shot, cen.family_id)])
    return pd.concat([cen, views.locators(F)], axis=1), F


def compare(mine: pd.DataFrame, dev: pd.DataFrame) -> dict:
    m = mine.merge(
        dev[["shot", "family_id", "status", "label"]],
        on=["shot", "family_id"],
        suffixes=("", "_dev"),
    )
    same = m.status == m.status_dev
    out = {
        "shots": int(m.shot.nunique()),
        "tracks": len(m),
        "status_agree": int(same.sum()),
        "status_agree_frac": round(float(same.mean()), 4),
        "label_agree": int((m.label == m.label_dev).sum()),
        "dev_status_counts": m.status_dev.value_counts().to_dict(),
        "frozen_status_counts": m.status.value_counts().to_dict(),
    }
    off = m[~same]
    if len(off):
        out["disagreements"] = (
            off.groupby(["status_dev", "status"])
            .size()
            .rename("n")
            .reset_index()
            .apply(lambda r: f"{r.status_dev} -> {r.status}: {r.n}", axis=1)
            .tolist()
        )
    return out


def check(paths: Paths, frozen: dict) -> dict:
    res = {}
    tab, F = corpus(paths)
    tab = pd.concat([tab, views.locators(F)], axis=1)
    audit = pd.read_csv(
        DEV / "fusion" / "out" / f"audit_cut{CUT:g}.csv",
        usecols=["shot", "family_id", "L_jump", "L_amp", "eceP_new"],
    )
    a = tab[["shot", "family_id", "L_jump", "L_amp", "eceP_new"]].merge(
        audit, on=["shot", "family_id"], suffixes=("", "_dev")
    )
    res["locators_vs_audit"] = {
        c: round(
            float(np.mean(np.isclose(a[c], a[f"{c}_dev"], atol=1e-6, equal_nan=True))),
            4,
        )
        for c in ("L_jump", "L_amp", "eceP_new")
    }
    frames = pd.read_csv(
        DEV / "corpus" / "out" / "frames_all.csv",
        usecols=["shot", "family_id", "t_ms", "f0"],
    )
    mine = apply(tab, F, frames, paths, frozen)
    dev = pd.read_csv(
        DEV / "fusion" / "out" / "resonance_corpus.csv",
        keep_default_na=False,
        usecols=["shot", "family_id", "status", "label"],
    )
    res["corpus"] = compare(mine, dev)
    ctab, cF = cases(paths)
    cframes = pd.read_csv(
        DEV / "cases" / "out" / "frames_all.csv",
        usecols=["shot", "family_id", "t_ms", "f0"],
    )
    cmine = apply(ctab, cF, cframes, paths, frozen)
    cdev = pd.read_csv(
        DEV / "fusion" / "out" / "cases_res.csv",
        keep_default_na=False,
        usecols=["shot", "family_id", "status", "label"],
    )
    res["cases"] = compare(cmine, cdev)
    res["cases_by_shot"] = {
        str(s): compare(g, cdev[cdev.shot == s])["status_agree_frac"]
        for s, g in cmine.groupby("shot")
    }
    return res


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true", help="check, write nothing")
    mode.add_argument(
        "--write",
        action="store_true",
        help="refit and overwrite freeze.json (its sha keys every cache and table)",
    )
    args = parser.parse_args(argv)
    paths = Paths.from_env()
    if args.check_only:  # read-only: freeze.json and its sha stay as they are
        print(json.dumps({"sha": sha(), "check": check(paths, load())}, indent=1))
        return 0
    frozen = fit(paths)
    frozen["check"] = check(paths, frozen)
    FREEZE.write_text(json.dumps(frozen, indent=1) + "\n")
    print(json.dumps({k: v for k, v in frozen.items() if k != "params"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
