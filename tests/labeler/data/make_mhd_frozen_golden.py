"""Make tests/labeler/data/mhd_frozen_golden.npz - once per freeze (labeler.mhd.freeze).

Five of dev's case tracks on 199607 (dev/mhd/cases), one per resonance status, each short
(at most 8 frames) and labelled the same by dev and by the frozen model, with what the
frozen model reads to label them again:
- the five tracks and the rules 1-4 anchors that overlap them (the rotation-order view
  reads those), their census columns and track-coherent views (the case features in
  $LABELER_ROOT/mhd/fusion/feat_cases, as `freeze.cases` sums them);
- the five tracks' frames;
- the EFIT02 slices the model reads for them (each row's mid-time slice, every slice
  inside a picked track) and the ZIPFIT times (each row's nearest, both ends; per EFIT02
  slice the nearest of ZIPFIT01 and 02), as stored on disk.
Expected: the frozen posterior, 90% set, distances and status computed on the WHOLE shot
(all its modelled tracks and inputs), and dev's own cases_res.csv row for each track. The
script checks that the cut-down inputs give the whole-shot numbers before it writes.

Dev's case posterior (dev/mhd/fusion/out/cases_v2_cut0.95.csv, `plm_*`) was fit by EM on the
corpus without the case shots, freeze.json on every corpus shot (none is gold), so the two
differ slightly. The script refits dev's rows with the ported EM (`labelmodel.fit`), checks
that this reproduces dev's posterior on every case track, and stores those parameters too:
the test then checks the port against dev's own numbers and the shipped freeze against its
snapshot.

    PYTHONPATH=src python tests/labeler/data/make_mhd_frozen_golden.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from labeler.config import Paths
from labeler.mhd import equilibrium as EQ
from labeler.mhd import freeze as FZ
from labeler.mhd import labelmodel as LM
from labeler.mhd import resonance as RE
from labeler.mhd import views

SHOT = 199607
MAX_FRAMES = 8
OUT = Path(__file__).with_name("mhd_frozen_golden.npz")
TAB = ["shot", "family_id", "n", "frames", "t0", "t1", "f", "qmin", "q95"]
TEXT = ["par", "harm", "anchor4", "list4"]
VIEWS = ["mir_c", "mir_g", "ece_c", "ece_g", "te", "rho", "R", "side"]
VIEWS += ["co2_c", "co2_g", "co2_n"]
EFIT = ["times", "qpsi", "fpol", "psirz", "ssimag", "ssibry", "rmaxis"]


class Npz(dict):
    """A dict that `equilibrium.Equilibrium` reads like an opened npz."""

    @property
    def files(self):
        return list(self)


def label(tab, F, frames, eq, rot_u, rot_r, frozen):
    P = LM.posterior(tab, F, {SHOT: eq}, {SHOT: rot_u}, frozen["params"])
    sl = RE.slices(tab, frames, {SHOT: eq}, {SHOT: rot_r})
    R, A, fn, top = RE.distances(sl, tab, frozen["law"])
    got = RE.decide(P, R, A, fn, top, frozen["law"])
    mp, pm, s90 = LM.credible(P, np.array([LM.NAMES] * len(tab)))
    keys = ("label", "status", "reason", "cand", "p", "p_res")
    return {"P": P, "R": R, "map": mp, "pmax": pm, "set90": np.array(s90)} | dict(
        zip(keys, got)
    )


def overlaps(tab, i):
    """Row positions of the rules 1-4 anchors overlapping row i (`rotation_refs`)."""
    r = tab.iloc[i]
    out = []
    for j, a in tab.iterrows():
        if j == i or a.anchor4 not in LM.SURFQ:
            continue
        ov = min(r.t1, a.t1) - max(r.t0, a.t0)
        if ov > 0.5 * min(r.t1 - r.t0, a.t1 - a.t0):
            out.append(j)
    return out


def text(values):
    """A fixed-width unicode array (np.load reads it without pickle)."""
    return np.array([str(v) for v in values], dtype=str)


def nearest(t, x):
    return int(np.argmin(np.abs(np.asarray(t, float) - x)))


def dev_case_fit(paths, case_shots):
    """dev/mhd/fusion/cases.py's fit_v2 label model: EM on the corpus without the case
    shots; its parameters."""
    tab, F = FZ.corpus(paths)
    keep = ~tab.shot.isin(case_shots).values
    tab = tab[keep].reset_index(drop=True)
    F = {k: v[keep] for k, v in F.items()}
    tab = pd.concat([tab, views.locators(F)], axis=1)
    eqs, rots = LM.load_inputs(paths, sorted(tab.shot.unique()))
    lm, _ = LM.fit(tab, F, eqs, rots)
    return lm.params(), int((~keep).sum())


def main():
    paths = Paths.from_env()
    frozen = FZ.load()
    ctab, cF = FZ.cases(paths)
    case_shots = sorted(int(s) for s in ctab.shot.unique())
    dev_params, dropped = dev_case_fit(paths, case_shots)
    ceqs, crots = LM.load_inputs(paths, case_shots)
    P_dev = LM.posterior(ctab, cF, ceqs, crots, dev_params)
    plm = pd.read_csv(FZ.DEV / "fusion" / "out" / "cases_v2_cut0.95.csv")
    plm = ctab[["shot", "family_id"]].merge(plm, on=["shot", "family_id"], how="left")
    plm = plm[[f"plm_{c}" for c in LM.NAMES]].to_numpy(float)
    dev_err = float(np.max(np.abs(P_dev - plm)))
    print(
        f"dev's case fit refit ({dropped} corpus tracks of the case shots dropped): "
        f"max |P - dev plm| over {len(ctab)} case tracks {dev_err:.2e}"
    )
    assert dev_err < 1e-9, dev_err
    keep = (ctab.shot == SHOT).values
    tab = ctab[keep].reset_index(drop=True)
    F = {k: v[keep] for k, v in cF.items()}
    frames = pd.read_csv(
        FZ.DEV / "cases" / "out" / "frames_all.csv",
        usecols=["shot", "family_id", "t_ms", "f0"],
    )
    frames = frames[frames.shot == SHOT].reset_index(drop=True)
    eq, rot_u, rot_r = (
        EQ.efit02(paths, SHOT),
        EQ.rot_profile(paths, SHOT),
        EQ.rotation(paths, SHOT),
    )
    whole = label(tab, F, frames, eq, rot_u, rot_r, frozen)
    dev = pd.read_csv(
        FZ.DEV / "fusion" / "out" / "cases_res.csv", keep_default_na=False
    )
    dev = dev[dev.shot == SHOT].set_index("family_id").loc[tab.family_id]
    agree = (
        (whole["status"] == dev.status.values)
        & (whole["label"] == dev.label.values)
        & (whole["cand"] == dev.cand.values)
        & (whole["map"] == dev.map_lm.values)
        & (tab.frames.values <= MAX_FRAMES)
    )
    picks = []
    for status in RE.STATUSES:
        cand = np.flatnonzero(agree & (whole["status"] == status))
        with_refs = [i for i in cand if overlaps(tab, i)]
        picks.append(int((with_refs or list(cand))[0]))
    rows = sorted(set(picks).union(*(overlaps(tab, i) for i in picks)))
    sub = tab.iloc[rows].reset_index(drop=True)
    at = [rows.index(i) for i in picks]

    # the EFIT02 slices and ZIPFIT times the picked tracks read
    ks = {nearest(eq.t, (r.t0 + r.t1) / 2) for r in sub.itertuples()}
    for i in picks:
        r = tab.iloc[i]
        inside = np.flatnonzero((eq.t >= r.t0) & (eq.t <= r.t1))
        assert len(inside) <= RE.MAX_SLICES
        ks |= set(inside.tolist())
    ks = sorted(ks)
    with np.load(EQ.efit02_path(paths, SHOT)) as d:
        efit = Npz({k: d[k][ks] for k in EFIT} | {"r": d["r"], "z": d["z"]})
    small_eq = EQ.Equilibrium(efit)
    w, rho, tt = rot_u
    ju = sorted(
        {0, len(tt) - 1} | {nearest(tt, (r.t0 + r.t1) / 2) for r in sub.itertuples()}
    )
    small_u = (w[ju], rho, tt[ju])
    sources, sign = rot_r
    small_r = []
    for ws, rs, ts in sources:
        js = sorted({nearest(ts, eq.t[k]) for k in ks})
        small_r.append((ws[js], rs, ts[js]))
    small_r = (small_r, sign)
    sub_F = {k: F[k][rows] for k in VIEWS}
    sub_frames = frames[frames.family_id.isin(tab.family_id.values[picks])]
    sub_frames = sub_frames.reset_index(drop=True)
    small = label(sub, sub_F, sub_frames, small_eq, small_u, small_r, frozen)
    small_dev = LM.posterior(sub, sub_F, {SHOT: small_eq}, {SHOT: small_u}, dev_params)
    dev_rows = np.flatnonzero(keep)[picks]
    assert np.max(np.abs(small_dev[at] - plm[dev_rows])) < 1e-9
    for k in ("P", "R"):
        err = np.nanmax(np.abs(small[k][at] - whole[k][picks]))
        assert err < 1e-12 or np.isnan(err), (k, err)
    for k in ("status", "label", "cand", "set90"):
        assert (small[k][at] == whole[k][picks]).all(), k

    out = {f"tab_{c}": sub[c].to_numpy() for c in TAB}
    out |= {
        f"tab_{c}": text(sub[c].astype(object).where(sub[c].notna(), "")) for c in TEXT
    }
    out |= {f"F_{k}": v for k, v in sub_F.items()}
    out |= {
        f"frames_{c}": sub_frames[c].to_numpy() for c in ("family_id", "t_ms", "f0")
    }
    out |= {f"eq_{k}": v for k, v in efit.items()}
    out |= {"rotu_w": small_u[0], "rotu_rho": small_u[1], "rotu_t": small_u[2]}
    for s, (ws, rs, ts) in enumerate(small_r[0]):
        out |= {f"rotr{s}_w": ws, f"rotr{s}_rho": rs, f"rotr{s}_t": ts}
    out |= {"rotr_sign": np.float64(sign), "rotr_n": np.int64(len(small_r[0]))}
    out |= {"at": np.array(at)}
    for k in ("P", "R", "pmax", "p", "p_res"):
        out[f"want_{k}"] = np.asarray(whole[k][picks], float)
    for k in ("map", "set90", "status", "label", "cand", "reason"):
        out[f"want_{k}"] = text(whole[k][picks])
    d = dev.iloc[picks]
    out |= {
        "dev_status": text(d.status),
        "dev_label": text(d.label),
        "dev_cand": text(d.cand),
        "dev_map": text(d.map_lm),
        "dev_pmax": d.pmax_lm.values.astype(float),
        "dev_p": d.p.values.astype(float),
        "dev_p_res": d.p_res.values.astype(float),
        "dev_R": d[[f"r_{c}" for c in LM.NAMES]]
        .replace("", np.nan)
        .values.astype(float),
        "dev_P": plm[dev_rows],
        "dev_params": np.array(json.dumps(dev_params)),
        "freeze_sha": np.array(FZ.sha()),
    }
    assert not [k for k, v in out.items() if np.asarray(v).dtype == object]
    np.savez_compressed(OUT, **out)
    print(
        f"wrote {OUT} ({OUT.stat().st_size} bytes): rows {len(sub)}, picks "
        f"{tab.family_id.values[picks].tolist()} {whole['status'][picks].tolist()}, "
        f"EFIT02 slices {len(ks)}, "
        f"max |pmax - dev| {np.max(np.abs(whole['pmax'][picks] - d.pmax_lm.values)):.2e}, "
        f"max |p_res - dev| {np.max(np.abs(whole['p_res'][picks] - d.p_res.values)):.2e}, "
        f"max |R - dev| {np.nanmax(np.abs(out['want_R'] - out['dev_R'])):.2e}"
    )


if __name__ == "__main__":
    main()
