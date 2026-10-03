#!/usr/bin/env python
"""TangTV front height from the raw camera frames, for shots without an inversion.

    python scripts/labeler/detach_tv_surrogate.py --shots-file shots_fetch.txt

The TangTV indicator (`labeler.events.detachment.tangtv`) needs the height ZE of the
C-III emission front, which the owner's tomographic inversion gives; only 41 shots
have an inversion on disk. The corpus holds the raw frames of thousands of shots (the
`tangtv` group, channel 2, the lower-divertor perpendicular view: its frames are the
`VID` of the plasma_tv `.sav` files to correlation 1.00, resampled to 50 fps). The
inversion is a linear operator on the frame, so ZE is regressed on the frame, the way
Chen 2026 trains its weighted-emission models (`plasma_tv`, ridge on pixels).

This script fits that regression itself on the inverted shots, so it can be scored
honestly: ZE (from `outer_leg_ze` on the inversion) is regressed on the black-level
subtracted, square-rooted, 6 x 6 block-averaged frame plus the X-point radius, by
ridge in the dual form (numpy only), and scored by leave-one-shot-out CV on the shelf
shots (the only ones the indicator is valid on). The CV is run twice: on the `.sav`
frames the model is trained on, and on the corpus frames at the inverted frames'
times, which is the path the deployed model takes. The deployed model, fitted on all
the training shots, predicts ZE for every fetched shot that has corpus frames and no
inversion; `detach_bins.py` then uses it as the TangTV source on those shots (the
bins record `tangtv_source`). No shot of the cohort's test split is in the fit.

Writes `docs/labeler/results/detachment_tangtv_surrogate.json` and, under
`$LABELER_ROOT/round4/detach/tv_surrogate/`, one `<shot>.npz` per predicted shot
(`times_ms`, `ze`) and `loso.npz` (the CV predictions).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import readsav
from scipy.linalg import solve

from labeler.events.detachment import core, signals, tangtv

REPO = Path(__file__).resolve().parents[2]
RESULT = REPO / "docs" / "labeler" / "results" / "detachment_tangtv_surrogate.json"
COHORT = REPO / "data" / "events" / "catalog" / "cohort.csv"
CORPUS = signals.CORPUS
#: Camera black level (counts) and the block size of the feature map (pixels).
BLACK = 16.0
BLOCK = 6
#: The lower-divertor perpendicular view, as a channel of the corpus `tangtv` group.
CHANNEL = 2
#: Weight of the X-point radius among the pixel features.
RX_SCALE = 10.0
ALPHAS = (1.0, 10.0, 100.0, 1000.0)
BOOTSTRAPS = 1000


def root() -> Path:
    return Path(os.environ["LABELER_ROOT"]) / "round4" / "detach"


def features(frames: np.ndarray, rx: np.ndarray) -> np.ndarray:
    """`(n, 240, 720)` frames and `(n,)` X-point radii to `(n, 40 * 120 + 1)`."""
    x = np.maximum(np.asarray(frames, dtype=np.float32) - BLACK, 0.0)
    n, h, w = x.shape
    x = x.reshape(n, h // BLOCK, BLOCK, w // BLOCK, BLOCK).mean(axis=(2, 4))
    x = np.sqrt(x).reshape(n, -1)
    # ZE_Norm, Chen 2026: per-frame mean 0 and standard deviation 1.
    scale = x.std(axis=1, keepdims=True)
    x = (x - x.mean(axis=1, keepdims=True)) / np.maximum(scale, 1e-6)
    return np.concatenate([x, (RX_SCALE * rx)[:, None].astype(np.float32)], axis=1)


def nearest(t_ms: np.ndarray, at_ms: np.ndarray) -> np.ndarray:
    """Index of the sample of `t_ms` (sorted) nearest each of `at_ms`."""
    k = np.clip(np.searchsorted(t_ms, at_ms), 1, len(t_ms) - 1)
    return np.where(np.abs(t_ms[k - 1] - at_ms) <= np.abs(t_ms[k] - at_ms), k - 1, k)


def geometry(shot: int, at_ms: np.ndarray):
    """EFIT geometry at the times, with the shelf gate: `(rx, zx, zs, valid)`."""
    cache, source = signals.tangtv_geometry(shot)
    need = ("rvsod", "zvsod", "rxpt1", "zxpt1")
    if any(k not in cache for k in need):
        return None
    et = cache["rxpt1"][0]
    near = nearest(et, at_ms)
    rv, zv, rx, zx = (np.asarray(cache[k][1], float)[near] for k in need)
    valid, _ = tangtv.shelf_gate(rv, zv, rx, zx)
    valid &= source == "EFIT02"
    valid &= np.abs(et[near] - at_ms) <= 40.0
    return rx, zx, zv, valid


def corpus_frames(shot: int):
    """`(t_ms, frames[T, 240, 720])` of channel 2, or None for a stub or a dead camera."""
    path = CORPUS / f"{shot}_processed.h5"
    if not path.is_file():
        return None
    with h5py.File(path, "r") as f:
        if "tangtv" not in f or f["tangtv"]["ydata"].shape[-1] <= 1:
            return None
        t = np.asarray(f["tangtv"]["xdata"][:], dtype=float) * 1000.0
        frames = np.asarray(f["tangtv"]["ydata"][CHANNEL], dtype=np.float32)
    live = np.isfinite(frames.reshape(len(frames), -1)[:, ::997]).all(axis=1)
    if not live.any():
        return None
    return t[live], frames[live]


def load_training(shots: list[int]) -> dict[int, dict]:
    """Per inverted shot: the sav frames at the inverted times, ZE, geometry, votes."""
    out = {}
    for shot in shots:
        inv = np.load(root() / "inversions" / f"{shot}.npz")
        record = None
        source = Path(str(inv["source"]))
        for path in (source, source.with_name(source.stem + "_raw.sav")):
            if path.is_file():
                rec = readsav(str(path))["emission_structure"][0]
                if "VID" in rec.dtype.names:
                    record = rec
                    break
        if record is None:
            continue
        ft = np.asarray(inv["times_ms"], dtype=float)
        geo = geometry(shot, ft)
        if geo is None:
            continue
        rx, zx, zs, gate = geo
        index = np.asarray(record["FRAMES"]).astype(int)
        if len(index) != len(ft):
            continue
        ze = tangtv.outer_leg_ze(
            inv["frames"].astype(np.float32), inv["radii"], inv["elevation"], rx
        )
        dz = tangtv.front_dz(ze, zx, zs)
        ok = gate & np.isfinite(dz) & (dz >= -0.25)
        if ok.sum() < 20:
            continue
        sav_frames = np.asarray(record["VID"])[index[ok]]
        out[shot] = {
            "t": ft[ok],
            "ze": ze[ok],
            "zx": zx[ok],
            "zs": zs[ok],
            "rx": rx[ok],
            "rv": np.interp(ft[ok], *signals.tangtv_geometry(shot)[0]["rvsod"]),
            "brightness": np.mean(sav_frames, axis=(1, 2)),
            "x_sav": features(sav_frames, rx[ok]),
            "corpus": None,
        }
        got = corpus_frames(shot)
        if got is not None:
            tc, fc = got
            if np.abs(tc[nearest(tc, ft[ok])] - ft[ok]).max() <= 20.0:
                out[shot]["x_corpus"] = features(fc[nearest(tc, ft[ok])], rx[ok])
                out[shot]["corpus"] = True
        print("train", shot, int(ok.sum()), "corpus" if out[shot]["corpus"] else "")
    return out


class Ridge:
    """Dual-form ridge with an intercept: `fit(X, y, alpha)`, `predict(X)`."""

    def fit(self, x: np.ndarray, y: np.ndarray, alpha: float) -> Ridge:
        x = np.asarray(x, dtype=np.float64)
        self.mean = x.mean(axis=0)
        self.y0 = float(y.mean())
        xc = x - self.mean
        gram = xc @ xc.T
        gram[np.diag_indices_from(gram)] += alpha
        self.dual = solve(gram, y - self.y0, assume_a="pos")
        self.xc = xc
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return (x - self.mean) @ (self.xc.T @ self.dual) + self.y0


def votes_of(ze, zx, zs) -> np.ndarray:
    return tangtv.dz_vote(tangtv.front_dz(ze, zx, zs))


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    return float(_kappa(a, b))


def _kappa(a, b):
    labels = np.union1d(a, b)
    if len(a) == 0:
        return float("nan")
    po = np.mean(a == b)
    pe = sum(np.mean(a == k) * np.mean(b == k) for k in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def scores(rows: pd.DataFrame) -> dict:
    """Frame-level metrics of predicted vs inversion ZE, pooled over the rows."""
    err = (rows.ze_pred - rows.ze).to_numpy()
    leg = (rows.zx - rows.zs).to_numpy()
    both = (rows.vote_true > 0) & (rows.vote_pred > 0)
    return {
        "n_frames": len(rows),
        "ze_mae_cm": float(np.mean(np.abs(err)) * 100),
        "dz_mae": float(np.mean(np.abs(err) / leg)),
        "frames_voting_in_both": int(both.sum()),
        "vote_agreement": float(
            np.mean(rows.vote_true[both].to_numpy() == rows.vote_pred[both].to_numpy())
        ),
        "vote_kappa": kappa(
            rows.vote_true[both].to_numpy(), rows.vote_pred[both].to_numpy()
        ),
        "attached_recall": recall(rows, both, core.ATTACHED),
        "detached_recall": recall(rows, both, core.DETACHED),
        "marfe_recall": recall(rows, both, core.MARFE),
    }


def recall(rows, both, state) -> float | None:
    hit = both & (rows.vote_true == state)
    if not hit.any():
        return None
    return float(np.mean(rows.vote_pred[hit].to_numpy() == state))


def bootstrap(rows: pd.DataFrame, rng, n=BOOTSTRAPS) -> dict:
    """95% shot-bootstrap intervals of ZE MAE, vote agreement and kappa."""
    by_shot = {s: g for s, g in rows.groupby("shot")}
    keys = list(by_shot)
    stats = {"ze_mae_cm": [], "vote_agreement": [], "vote_kappa": []}
    for _ in range(n):
        pick = rng.choice(len(keys), size=len(keys))
        sample = pd.concat([by_shot[keys[i]] for i in pick], ignore_index=True)
        s = scores(sample)
        for k, values in stats.items():
            values.append(s[k])
    return {
        k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
        for k, v in stats.items()
    }


def loso(data: dict[int, dict], alphas) -> dict[float, dict[str, pd.DataFrame]]:
    """Leave-one-shot-out predictions for each alpha, from the sav and corpus frames.

    The Gram matrix of all training frames is formed once; each fold slices it,
    centres it on its own training frames and solves one system per alpha, so the
    held-out shot never enters a mean, a kernel or a fit.
    """
    shots = list(data)
    x_all = np.concatenate([data[s]["x_sav"] for s in shots]).astype(np.float64)
    y_all = np.concatenate([data[s]["ze"] for s in shots])
    owner = np.concatenate([np.full(len(data[s]["ze"]), s) for s in shots])
    gram = x_all @ x_all.T
    out = {a: {"x_sav": [], "x_corpus": []} for a in alphas}
    for held in shots:
        tr = np.flatnonzero(owner != held)
        k = gram[np.ix_(tr, tr)]
        row, tot = k.mean(axis=1), k.mean()
        centred = k - row[:, None] - row[None, :] + tot
        x_tr, y_tr = x_all[tr], y_all[tr]
        y0 = float(y_tr.mean())
        d = data[held]
        keys = [key for key in ("x_sav", "x_corpus") if key in d]
        cross = {key: d[key].astype(np.float64) @ x_tr.T for key in keys}
        for alpha in alphas:
            system = centred.copy()
            system[np.diag_indices_from(system)] += alpha
            dual = solve(system, y_tr - y0, assume_a="pos")
            for key in keys:
                c = cross[key]
                kc = c - c.mean(axis=1)[:, None] - row[None, :] + tot
                pred = kc @ dual + y0
                out[alpha][key].append(
                    pd.DataFrame(
                        {
                            "shot": held,
                            "t": d["t"],
                            "ze": d["ze"],
                            "ze_pred": pred,
                            "zx": d["zx"],
                            "zs": d["zs"],
                            "vote_true": votes_of(d["ze"], d["zx"], d["zs"]),
                            "vote_pred": votes_of(pred, d["zx"], d["zs"]),
                        }
                    )
                )
        print("fold", held, flush=True)
    return {
        a: {key: pd.concat(v, ignore_index=True) for key, v in keys.items() if v}
        for a, keys in out.items()
    }


def frame_check(data: dict[int, dict]) -> dict:
    """The corpus and sav frames of the same inverted times, feature by feature."""
    corr = []
    for d in data.values():
        if d["corpus"]:
            a, b = d["x_sav"][:, :-1], d["x_corpus"][:, :-1]
            corr.extend(
                float(np.corrcoef(a[i], b[i])[0, 1]) for i in range(0, len(a), 10)
            )
    return {
        "shots": int(sum(1 for d in data.values() if d["corpus"])),
        "feature_correlation_median": float(np.median(corr)) if corr else None,
        "feature_correlation_min": float(np.min(corr)) if corr else None,
    }


_MODEL: Ridge | None = None
_DOMAIN = {}
_VERIFIED_CAMERA = set()


def domain(data):
    """Envelope and brightness ranges from training inversion frames only."""
    vectors = {
        "leg": np.concatenate([d["zx"] - d["zs"] for d in data.values()]),
        "rv": np.concatenate([d["rv"] for d in data.values()]),
        "rx": np.concatenate([d["rx"] for d in data.values()]),
        "brightness": np.concatenate([d["brightness"] for d in data.values()]),
    }
    return {k: [float(np.min(v)), float(np.max(v))] for k, v in vectors.items()}


def select_alpha(data):
    """Three shot-grouped inner folds; at most 20 frames/shot for selection."""
    shots = sorted(data)
    reduced = {
        s: {
            k: v[np.linspace(0, len(d["ze"]) - 1, min(20, len(d["ze"]))).astype(int)]
            for k, v in d.items()
            if k in ("x_sav", "ze")
        }
        for s, d in data.items()
    }
    errors = {a: [] for a in ALPHAS}
    for held in np.array_split(shots, 3):
        train = [s for s in shots if s not in held]
        x = np.concatenate([reduced[s]["x_sav"] for s in train])
        y = np.concatenate([reduced[s]["ze"] for s in train])
        xe = np.concatenate([reduced[s]["x_sav"] for s in held])
        ye = np.concatenate([reduced[s]["ze"] for s in held])
        for alpha in ALPHAS:
            errors[alpha].append(
                float(np.mean(np.abs(Ridge().fit(x, y, alpha).predict(xe) - ye)))
            )
    return min(errors, key=lambda a: np.mean(errors[a]))


def nested_loso(data):
    """Outer held shot excluded from alpha selection, scaling and final fit."""
    rows = {"x_sav": [], "x_corpus": []}
    selection = {}
    for held, d in data.items():
        train = {s: v for s, v in data.items() if s != held}
        alpha = select_alpha(train)
        selection[str(held)] = {"alpha": alpha, "training_shots": sorted(train)}
        model = Ridge().fit(
            np.concatenate([v["x_sav"] for v in train.values()]),
            np.concatenate([v["ze"] for v in train.values()]),
            alpha,
        )
        for key, output in rows.items():
            if key not in d:
                continue
            pred = model.predict(d[key])
            output.append(
                pd.DataFrame(
                    {
                        "shot": held,
                        "t": d["t"],
                        "ze": d["ze"],
                        "ze_pred": pred,
                        "zx": d["zx"],
                        "zs": d["zs"],
                        "vote_true": votes_of(d["ze"], d["zx"], d["zs"]),
                        "vote_pred": votes_of(pred, d["zx"], d["zs"]),
                    }
                )
            )
        print("outer", held, "inner alpha", alpha, flush=True)
    return {k: pd.concat(v, ignore_index=True) for k, v in rows.items() if v}, selection


def predict_one(shot: int) -> tuple[int, str]:
    """Predict one shot's front heights with the module's model and park them."""
    got = corpus_frames(shot)
    if got is None:
        return shot, "no_frames"
    t, frames = got
    geo = geometry(shot, t)
    if geo is None:
        return shot, "no_efit"
    rx, zx, zs, gate = geo
    gcache, _ = signals.tangtv_geometry(shot)
    rv = np.interp(t, *gcache["rvsod"])
    brightness = frames.mean(axis=(1, 2))
    in_envelope = gate.copy()
    for key, vector in (("leg", zx - zs), ("rv", rv), ("rx", rx)):
        lo, hi = _DOMAIN[key]
        in_envelope &= (vector >= lo) & (vector <= hi)
    lo, hi = _DOMAIN["brightness"]
    exposure_ok = (
        (brightness >= lo)
        & (brightness <= hi)
        & ((frames >= 255).mean(axis=(1, 2)) < 0.01)
    )
    # No camera/filter/exposure metadata accompanies corpus frames. Only shots
    # with same-time SAV/corpus comparison have verified cam240perp provenance.
    camera_ok = shot in _VERIFIED_CAMERA
    gate = in_envelope & exposure_ok & camera_ok
    ze = np.full(len(t), np.nan)
    if gate.any():
        ze[gate] = _MODEL.predict(features(frames[gate], rx[gate]))
    np.savez_compressed(
        root() / "tv_surrogate" / f"{shot}.npz",
        times_ms=t,
        ze=ze,
        valid=gate,
        in_envelope=in_envelope,
        exposure_ok=exposure_ok,
        camera_ok=np.full(len(t), camera_ok),
    )
    return shot, f"ok {int(gate.sum())} gated frames"


def predict_shots(
    model: Ridge, shots: list[int], skip: set[int], workers: int = 6
) -> dict:
    """Predict every shot not in `skip`, `workers` processes sharing the model."""
    import multiprocessing as mp

    global _MODEL, _DOMAIN, _VERIFIED_CAMERA
    _MODEL = model
    _DOMAIN = model.domain
    _VERIFIED_CAMERA = set(model.camera_verified)
    (root() / "tv_surrogate").mkdir(parents=True, exist_ok=True)
    todo = [s for s in shots if s not in skip]
    status = {}
    with mp.get_context("fork").Pool(workers) as pool:
        for shot, state in pool.imap_unordered(predict_one, todo, chunksize=1):
            status[shot] = state
            print("predict", shot, state, flush=True)
    return status


def predict_only(args) -> None:
    """Fit the deployed model with the CV's recorded alpha and predict the shots."""
    record = json.loads(Path(args.out).read_text())
    test = set(pd.read_csv(COHORT).query("split == 'test'").shot.astype(int))
    inverted = sorted(
        int(p.stem) for p in (root() / "inversions").glob("*.npz") if int(p.stem)
    )
    data = load_training([s for s in inverted if s not in test])
    if sorted(data) != record["training_shots"]:
        raise SystemExit("the training shots differ from the recorded CV run")
    x = np.concatenate([d["x_sav"] for d in data.values()])
    y = np.concatenate([d["ze"] for d in data.values()])
    model = Ridge().fit(x, y, record["alpha"])
    model.domain = domain(data)
    model.camera_verified = [s for s, d in data.items() if d["corpus"]]
    shots = [int(s) for s in Path(args.shots_file).read_text().split()]
    record["predicted"] = predict_shots(model, shots, skip=set(inverted))
    Path(args.out).write_text(json.dumps(record, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shots-file", required=True)
    parser.add_argument("--out", default=str(RESULT))
    parser.add_argument("--no-predict", action="store_true")
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="skip the CV: refit with the alpha recorded in --out and predict",
    )
    args = parser.parse_args()
    if args.predict_only:
        predict_only(args)
        return
    rng = np.random.default_rng(0)
    test = set(pd.read_csv(COHORT).query("split == 'test'").shot.astype(int))
    inverted = sorted(
        int(p.stem) for p in (root() / "inversions").glob("*.npz") if int(p.stem)
    )
    train_shots = [s for s in inverted if s not in test]
    data = load_training(train_shots)
    n_frames = sum(len(d["ze"]) for d in data.values())
    print(len(data), "training shots", n_frames, "frames")
    result = {
        "training_shots": sorted(data),
        "excluded_test_cohort": sorted(set(inverted) & test),
        "n_frames": n_frames,
        "features": f"sqrt of (frame - {BLACK}) block-averaged {BLOCK}x{BLOCK}, "
        f"plus {RX_SCALE} x X-point R",
        "frame_check": frame_check(data),
        "alphas": {},
    }
    folds, selection = nested_loso(data)
    best = select_alpha(data)
    best_rows, corpus_rows = folds["x_sav"], folds["x_corpus"]
    result["alpha"] = best
    result["selection"] = (
        "nested LOSO, three shot-grouped inner folds; 20 frames/shot for inner selection"
    )
    result["outer_folds"] = selection
    result["domain"] = domain(data)
    result["camera_verified_shots"] = [s for s, d in data.items() if d["corpus"]]
    result["deployment_note"] = (
        "Unverified camera/filter metadata abstains; brightness and saturation filter exposure changes."
    )
    result["loso_sav"] = {**scores(best_rows), "ci95": bootstrap(best_rows, rng)}
    result["loso_corpus"] = {
        **scores(corpus_rows),
        "ci95": bootstrap(corpus_rows, rng),
        "shots": sorted(int(s) for s in corpus_rows.shot.unique()),
    }
    out_dir = root() / "tv_surrogate"
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_rows.to_csv(out_dir / "loso_corpus.csv.gz", index=False)
    best_rows.to_csv(out_dir / "loso_sav.csv.gz", index=False)
    if not args.no_predict:
        x = np.concatenate([d["x_sav"] for d in data.values()])
        y = np.concatenate([d["ze"] for d in data.values()])
        model = Ridge().fit(x, y, best)
        model.domain = domain(data)
        model.camera_verified = [s for s, d in data.items() if d["corpus"]]
        shots = [int(s) for s in Path(args.shots_file).read_text().split()]
        result["predicted"] = predict_shots(model, shots, skip=set(inverted))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(
        json.dumps(
            {k: result[k] for k in ("alpha", "loso_sav", "loso_corpus")}, indent=1
        )
    )


if __name__ == "__main__":
    main()
