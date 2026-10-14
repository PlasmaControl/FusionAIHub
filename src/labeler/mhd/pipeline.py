"""One shot from its corpus record to its labelled mode tracks, cached per shot.

`run_shot` does, over the shot's window (`window`):
1. `tracks.frames`: Mirnov peaks, n, tracks and the per-frame measurements;
2. `rules`: gates, harmonic lock, the track table and rules 1-4;
3. for the modelled tracks (not a line or harmonic, n 1-3, q_min and q95 known):
   `views` (cross-spectra summed over the track, ECE locators), the frozen label model's
   posterior inside the rules 1-4 set (`labelmodel.posterior`) and the resonance gate
   (`resonance.decide`) with the frozen law.
The equilibrium is `equilibrium.equilibrium`'s: EFIT02, else the EFIT02ER or EFIT01 fallback
file; the info record names its tree (`inputs.tree`, None without one) and file.
The result is a track table, the frames (time, frequency, track) and an info record, written
under `$LABELER_ROOT/mhd/labeler/<CACHE_VERSION>/` as `<shot>.tracks.parquet`,
`<shot>.frames.parquet`, `<shot>.feat.npz` (the track-level views) and `<shot>.json`. The
cache is reused while its key holds: the window, the input files' mtime and size (corpus
record, ZIPFIT), the equilibrium file used (path, mtime, size, tree), the Thomson geometry
used, the freeze file's sha and `PIPELINE_VERSION`; `stale` names the fields that no longer
hold.

`load(shot, paths)` reads a cached shot (None when absent): what other streams should use.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Paths
from . import equilibrium as EQ
from . import freeze as FZ
from . import labelmodel as LM
from . import resonance as RE
from . import rules, tracks, views

log = logging.getLogger(__name__)

#: 3 (mhd_modes v4, v5): the equilibrium is EFIT02, else the EFIT02ER or EFIT01 fallback,
#: each with >= 2 slices. 2 (v3): a Mirnov record NaN over part of the window is used where
#: it is finite. The v5 cache began as a copy of v4's (same pipeline: a shot whose key still
#: holds is not rerun; the one-slice EFIT02 shots' new fallback files change theirs). The
#: v2, v3 and v4 caches (`.../labeler/v2/` ...) stay on disk; RWM reads v4's.
PIPELINE_VERSION = 3
CACHE_VERSION = "v5"
#: Statuses whose classes in the 90% credible set are uncertain for a child (decision M3).
UNSURE = ("uncertain", "no rotation", "q_min above")
#: frames() reasons that still mean the Mirnov was read: no mode, not no data.
SEEN_NOTHING = ("no peaks", "no n=1-3 tracks")
TRACK_COLUMNS = [
    "shot",
    "family_id",
    "n",
    "frames",
    "t0",
    "t1",
    "f",
    "fiqr",
    "amp",
    "coh",
    "par",
    "ngated",
    "locked",
    "harm",
    "fund_id",
    "line",
    "qmin",
    "q95",
    "anchor4",
    "list4",
    "modelled",
    "why",
    "map",
    "pmax",
    "set90",
    "label",
    "status",
    "reason",
    "cand",
    "p",
    "p_res",
    "fn",
    "fn_max",
    "name",
] + [f"p_{c}" for c in LM.NAMES]


@dataclass
class Shot:
    shot: int
    info: dict
    tracks: pd.DataFrame
    frames: pd.DataFrame

    @property
    def mirnov(self) -> bool:
        """The Mirnov record was read (modes or none); False: not observable."""
        return (
            self.info.get("status") == "ok" or self.info.get("status") in SEEN_NOTHING
        )


def cache_dir(paths: Paths) -> Path:
    return paths.root / "mhd" / "labeler" / CACHE_VERSION


def _files(paths: Paths, shot: int) -> dict:
    base = cache_dir(paths) / str(int(shot))
    return {
        "info": base.with_suffix(".json"),
        "tracks": Path(f"{base}.tracks.parquet"),
        "frames": Path(f"{base}.frames.parquet"),
        "feat": Path(f"{base}.feat.npz"),
    }


def window(shot: int, paths: Paths) -> tuple[tuple[float, float] | None, str]:
    """Decision M7: the catalog's rule-4 window (`plasma_window`), else
    `frames.targets.window_for`'s Ip window on the corpus or raw cache, else the same rule on
    the feature store's Ip; `(None, reason)` when none has one."""
    from ..events.catalog.window import assessed_window, restrike_end
    from ..frames.targets import window_for

    try:
        got, source = window_for(int(shot), paths, None)
        return (float(got[0]), float(got[1])), source
    except ValueError as error:
        reason = str(error)
    path = paths.features_file(int(shot))
    try:
        import h5py

        with h5py.File(path, "r") as f:
            t_ms = np.asarray(f["ip/xdata"][:], float) * 1000.0
            ip_a = np.asarray(f["ip/ydata"][:], float).reshape(-1)
    except (OSError, KeyError):
        return None, reason
    got = assessed_window(t_ms, ip_a)
    if got is None:
        return None, reason
    end = restrike_end(t_ms, ip_a, got)
    if end is not None and end > got[0]:
        got = (got[0], end)
    return (float(got[0]), float(got[1])), "features ip"


def key(shot: int, paths: Paths, win) -> dict:
    geo, used = EQ.ts_geometry(paths, int(shot))
    geo_hash = (
        hashlib.sha256(geo[["ch", "R", "Z"]].to_numpy().tobytes()).hexdigest()[:12]
        if geo is not None
        else None
    )
    return {
        "window": [round(float(win[0]), 3), round(float(win[1]), 3)],
        "corpus": EQ.stamp(tracks.corpus_path(paths, shot)),
        "equilibrium": _eq_key(paths, shot),
        "rot": EQ.stamp(EQ.rot_path(paths, shot)),
        "ts_geom": [used, geo_hash],
        "freeze": FZ.sha(),
        "pipeline": PIPELINE_VERSION,
    }


def stale(got: Shot, k: dict) -> list[str]:
    """The key fields in which the cached shot differs from `k` (`key` now); [] when its
    key holds."""
    old = got.info.get("key")
    if not isinstance(old, dict):
        return ["(no key)"]
    return [f for f in sorted(set(old) | set(k)) if old.get(f) != k.get(f)]


def _eq_key(paths: Paths, shot: int) -> list | None:
    """The equilibrium file the shot uses: `[path, mtime_ns, size, tree]`, None without."""
    got = EQ.equilibrium_file(paths, shot)
    return None if got is None else [*got[0], got[1]]


def tree(info: dict) -> str | None:
    """The shot's equilibrium tree from its info record (v2/v3 caches: EFIT02 or none)."""
    inputs = info.get("inputs", {})
    if "tree" in inputs:
        return inputs["tree"]
    return "efit02" if inputs.get("efit02") else None


def eq_times(info: dict) -> list:
    """The equilibrium's slice times (ms) from the info record."""
    return info.get("eq_t", info.get("efit02_t", []))


def load(shot: int, paths: Paths) -> Shot | None:
    """The cached shot, whatever its key; None when it was never run."""
    f = _files(paths, shot)
    if not f["info"].is_file():
        return None
    info = json.loads(f["info"].read_text())
    T = pd.read_parquet(f["tracks"]) if f["tracks"].is_file() else _empty_tracks()
    fr = pd.read_parquet(f["frames"]) if f["frames"].is_file() else _empty_frames()
    return Shot(int(shot), info, T, fr)


def _empty_tracks() -> pd.DataFrame:
    return pd.DataFrame(columns=TRACK_COLUMNS)


def _empty_frames() -> pd.DataFrame:
    return pd.DataFrame(columns=["shot", "family_id", "t_ms", "f0", "n"])


def run_shot(shot: int, paths: Paths, win, source: str, *, force=False) -> Shot:
    """The shot's tracks over `win`, from the cache when its key holds."""
    shot = int(shot)
    k = key(shot, paths, win)
    if not force:
        got = load(shot, paths)
        if got is not None and got.info.get("key") == k:
            return got
    t0 = time.time()
    got = compute(shot, paths, win, source)
    got.info["key"] = k
    got.info["seconds"] = round(time.time() - t0, 1)
    save(got, paths)
    return got


def save(got: Shot, paths: Paths) -> None:
    f = _files(paths, got.shot)
    f["info"].parent.mkdir(parents=True, exist_ok=True)
    got.tracks.to_parquet(f["tracks"], index=False)
    got.frames.to_parquet(f["frames"], index=False)
    feat = got.info.pop("_feat", None)
    if feat is not None:
        np.savez_compressed(f["feat"], **feat)
    else:
        f["feat"].unlink(missing_ok=True)  # an earlier run's views are not this run's
    tmp = f["info"].with_suffix(".json.tmp")
    tmp.write_text(json.dumps(got.info, indent=1, default=_json) + "\n")
    tmp.replace(f["info"])


def _json(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def compute(shot: int, paths: Paths, win, source: str) -> Shot:
    eq = EQ.equilibrium(paths, shot)
    rot_u = EQ.rot_profile(paths, shot)
    rot_r = EQ.rotation(paths, shot)
    info = {
        "shot": shot,
        "window": [float(win[0]), float(win[1])],
        "window_source": source,
        "inputs": {
            "tree": eq.tree if eq is not None else None,
            "equilibrium": eq.path if eq is not None else None,
            "ece_freq": eq is not None and eq.freq is not None,
            "rotation": bool(rot_r[0]),
        },
        "eq_t": np.round(eq.t, 2).tolist() if eq is not None else [],
        "freeze": FZ.sha(),
    }
    df, reason, got = tracks.frames(shot, paths, win, eq)
    info.update(got)
    info["status"] = reason
    if df is None:
        return Shot(shot, info, _empty_tracks(), _empty_frames())
    df = rules.harmonic_lock(rules.prepare(df))
    T = rules.census(rules.track_table(df))
    T["line"] = [rules.is_line(r) for r in T.itertuples()]
    keep = FZ.keep_rows(T)
    T["why"] = np.select(
        [T.line.values, T.harm.values, ~keep],
        ["line", "harmonic", "harmonic or line by rules 1-4"],
        "",
    ).astype(object)
    if eq is None:
        T.loc[keep, "why"] = "no equilibrium"
    else:
        noq = keep & ~(np.isfinite(T.qmin) & np.isfinite(T.q95)).values
        T.loc[noq, "why"] = "no equilibrium slice within 30 ms"
    T["modelled"] = T.why == ""
    for c in ("map", "set90", "label", "status", "reason", "cand", "name"):
        T[c] = ""
    for c in ["pmax", "p", "p_res", "fn", "fn_max"] + [f"p_{c}" for c in LM.NAMES]:
        T[c] = np.nan
    feat = None
    if T.modelled.any():
        try:
            feat = _model(shot, paths, eq, rot_u, rot_r, T, df)
        except Exception as error:  # noqa: BLE001 - one shot's failure is its own
            log.warning("shot %s: the model failed: %s", shot, error)
            T.loc[T.modelled, "why"] = f"model failed: {type(error).__name__}: {error}"
            T["modelled"] = False
            info["model_error"] = f"{type(error).__name__}: {error}"
    _name_harmonics(T)
    info["tracks"] = len(T)
    info["modelled"] = int(T.modelled.sum())
    info["status_counts"] = T[T.modelled].status.value_counts().to_dict()
    if feat is not None:
        info["_feat"] = feat
    frames = df[["shot", "family_id", "t_ms", "f0", "n"]].reset_index(drop=True)
    return Shot(shot, info, T[TRACK_COLUMNS].reset_index(drop=True), frames)


def _model(shot, paths, eq, rot_u, rot_r, T, df) -> dict | None:
    """Fill the modelled rows of T in place; return the track-level views (for the cache)."""
    tab = T[T.modelled].reset_index(drop=True)
    fr = df[df.family_id.isin(tab.family_id)][["shot", "family_id", "t_ms", "f0"]]
    geo, _ = EQ.ts_geometry(paths, shot)
    d, why = views.frame_features(
        tracks.corpus_path(paths, shot), fr.reset_index(drop=True), eq, geo
    )
    if d is None:
        T.loc[T.modelled, "why"] = f"no views: {why}"
        T["modelled"] = False
        return None
    rows = {r["family_id"]: r for r in views.track_features(d)}
    F = views.stack([rows[int(f)] for f in tab.family_id])
    tab = pd.concat([tab, views.locators(F)], axis=1)
    frozen = FZ.load()
    P = LM.posterior(tab, F, {shot: eq}, {shot: rot_u}, frozen["params"])
    sl = RE.slices(tab, df, {shot: eq}, {shot: rot_r})
    R, A, fn, top = RE.distances(sl, tab, frozen["law"])
    label, status, reason, cand, p, p_res = RE.decide(P, R, A, fn, top, frozen["law"])
    names = np.array([LM.NAMES] * len(tab))
    mp, pm, s90 = LM.credible(P, names)
    idx = T.index[T.modelled]
    T.loc[idx, "map"], T.loc[idx, "pmax"], T.loc[idx, "set90"] = mp, pm, s90
    T.loc[idx, "label"], T.loc[idx, "status"], T.loc[idx, "reason"] = (
        label,
        status,
        reason,
    )
    T.loc[idx, "cand"], T.loc[idx, "p"], T.loc[idx, "p_res"] = cand, p, p_res
    T.loc[idx, "fn"], T.loc[idx, "fn_max"] = fn, top
    for j, c in enumerate(LM.NAMES):
        T.loc[idx, f"p_{c}"] = P[:, j]
    out = {"family_id": tab.family_id.values, "P": P, "R": R}
    out.update(
        {
            k: v
            for k, v in F.items()
            if k
            in ("ece_c", "ece_g", "mir_c", "mir_g", "co2_c", "co2_g", "rho", "side")
        }
    )
    out.update(
        {c: tab[c].values.astype(float) for c in ("L_jump", "L_amp", "eceP_new")}
    )
    return out


def _name_harmonics(T: pd.DataFrame) -> None:
    """`name`: a labelled track's m/n, and a harmonic's k m/k n after its labelled
    fundamental (2/1 -> 4/2, 6/3); empty otherwise."""
    lab = dict(zip(T.family_id, T.label))
    names = T.label.astype(object).copy()
    for i, r in T[T.harm.astype(bool)].iterrows():
        got = rules.harmonic_name(r.n, lab.get(r.fund_id, ""))
        names[i] = got or ""
    T["name"] = names


def eq_cells(shot_eq_t, lo: int, hi: int, max_dt=rules.EQ_MAX_DT) -> np.ndarray:
    """Whole-ms cells [lo, hi) with an equilibrium slice within `max_dt`."""
    mid = np.arange(lo, hi) + 0.5
    t = np.sort(np.asarray(shot_eq_t, float))
    if not len(t) or not len(mid):
        return np.zeros(len(mid), bool)
    j = np.clip(np.searchsorted(t, mid), 1, max(1, len(t) - 1))
    near = np.minimum(
        np.abs(t[j - 1] - mid), np.abs(t[np.minimum(j, len(t) - 1)] - mid)
    )
    return near <= max_dt
