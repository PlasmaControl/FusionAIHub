"""The learned tokenizers of stage 1 (plan 1c): the arms, their examples and the linear recoders.

`store` encodes the fronts of the signals that a recoder can sit behind (every `band_dct` and `linear` signal
and each `frame_dct` signal with more than one sample per 5 ms frame: 24 signals on cache v1) over the shots of
1b's fit (training) and over `val_select` shots (validation), each in the domain of its setting in an
assignment (default ``<stage1>/assignments/v1.json``), and writes ``<stage1>/store/v1/train`` and ``.../val``.
`linear --arm NAME` fits the linear recoder of one arm (`klt.*` per stream, `pca.*` concatenated) on the
training store and writes ``<stage1>/recoders/NAME``; it prints, per signal, the share of the validation
examples' energy that is left after keeping the first 1, 2, 4, 8 and 16 values per stream. `arms` lists the 24
arms and which recoders exist. `joint --signals a,b,c,d` measures, on the stores, what one basis over all streams
of several signals leaves out beside each signal's own concatenated basis and one per-stream basis pooled over all
their streams, at the same number of values in all. `train --arm NAME` trains the network of one network arm on
the stores (on one GPU, or on the CPU with `--cpu`) and writes ``<stage1>/recoders/<NAME>.s<seed>[.<variant>]``:
the weights of the best validation, the log, the provenance and, last, ``result.json`` with the collapse check;
it exits 3 if the network collapsed (rule 7 as ruling C33 reads it: `rule7_usage`, on the distinct validation
examples, no token's first value is used; `result.json` holds the plan's literal reading beside it). `fluct` measures what the trend path leaves out of the 50 kHz channels (the band above
a signal's K cosine values, by frequency; no tokenizer is built) and writes ``<stage1>/compare/fluct.csv``. Torch is loaded at import by the modules this script uses; `train` defers the CUDA
device choice until after its refusals (`CUDA_VISIBLE_DEVICES` is set before any CUDA call).

Units. "Front values" are what a front codec's ``encode_domain`` returns (`store.Norm` takes them to "example
units": per-stream standardized), and "recoder values" are what a recoder returns. This file does nothing at
import time except define (`store` starts spawned workers, which import it again) and imports no torch of its own.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import inspect
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from tokamak_foundation_model.ignite.mast import cache, paths, registry, tasks, transform
from tokamak_foundation_model.ignite.mast.tokenizers import assign, code_sha256, fit, grid
from tokamak_foundation_model.ignite.mast.tokenizers import learned
from tokamak_foundation_model.ignite.mast.tokenizers.frames import frame_len
from tokamak_foundation_model.ignite.mast.tokenizers.learned import compare, fluct, identity, recode, store

FIT_VERSION = "v1"                        # 1b's fit, whose shots are the training shots
STORE_VERSION = "v1"
DEFAULT_ASSIGNMENT = "v1"                 # the assignment whose domains the fronts are encoded in
SIZES = (1, 2, 4, 8, 16)                  # values per stream at which `linear` reports what is left over
MAGNETICS = ("magnetics-b_field_pol_probe_ccbv_field", "magnetics-b_field_pol_probe_obr_field",
             "magnetics-b_field_pol_probe_obv_field", "magnetics-flux_loop_flux")
BANDS = tasks.SPECTROGRAM_SIGNALS
SHORT = {"ccbv": MAGNETICS[0], "obr": MAGNETICS[1], "obv": MAGNETICS[2], "flux": MAGNETICS[3],
         "cc": BANDS[0], "omv": BANDS[1]}
HEADROOM_SCRIPT = Path(__file__).resolve().with_name("run_headroom.py")
_VARIANT = re.compile(r"[A-Za-z0-9_-]+")
DEFAULT_LR, DEFAULT_STEPS = 3e-4, 20000   # `train.TrainConfig`'s (a test holds them equal; `train.py` is imported only after the refusals, with the CUDA choice)
RUN_FILES = ("result.json", "recoder.json", "best.pt", "train_log.csv", "norm.npz", "table.npz", "ranges.npz",
             "config.json", "trained_by.json")        # what a run writes, `result.json` first and so first to go
STORE_HASHES = ("assignment_sha256", "fit_sha256", "stats_sha256")     # what a validation store must share with its training store
RULE7_MIN_ENTROPY = inspect.signature(compare.usage).parameters["min_entropy"].default   # rule 7's 0.3, read from where `compare.usage` defines it
BANDCAT_TOKENS = 54                       # tokens of 8 = 432 values: v1's 3 x (32 + 7 x 16) per Mirnov signal (ruling C4)


class Refused(Exception):
    """A refusal that is not an error of the code: `main` prints it and exits with status 2."""


def _prepare(path: Path) -> Path:
    """`paths.prepare_write_root(path)`; `Refused` if it refuses the directory (repository, TokaMind, TokaMark)."""
    try:
        return paths.prepare_write_root(path)
    except ValueError as e:
        raise Refused(str(e)) from e


def recoders_root() -> Path:
    """``<stage1>/recoders``, where `run_tokenizer.py` reads the recoders an assignment names."""
    return paths.stage1_dir() / "recoders"


# ------------------------------------------------------------------------------------------- arms

@dataclass(frozen=True)
class Arm:
    """One tokenizer of the comparison: the `signals` it covers, whether it takes all streams of a frame at once
    (`concat`) or one stream at a time, whether it is a network (`net`) or a linear recoder, and its
    ``n_tok * n_val`` recoder values."""

    name: str
    signals: tuple[str, ...]
    concat: bool
    net: bool
    n_tok: int
    n_val: int


def _tokens(name: str, values: int, n_val: int) -> int:
    if values % n_val:
        raise ValueError(f"arm {name}: {values} values do not fill whole tokens of {n_val}")
    return values // n_val


def arms(series: Sequence[str], profiles: Sequence[str], maps: Sequence[str],
         streams: Mapping[str, int]) -> dict[str, Arm]:
    """The 24 arms of plan 1c, by name. `series`, `profiles` and `maps` are signal names; `streams[name]` is
    the number of streams ``S`` of a signal (channels; probes times 8 bands; 1 for a profile or map).

    Linear (11): `klt.series`, `klt.mag` (per stream, 1 x 16), `pca.<ccbv|obr|obv|flux>` (one signal,
    concatenated, all ``S * 16`` values, tokens of 4 for ccbv and of 6 for the others: ruling C5),
    `klt.band`, `klt.band.<cc|omv>` (per stream, 1 x 32), `pca.band.<cc|omv>` (concatenated, all ``S * 32``
    values in tokens of 4). Networks (13): `net.series`, `net.mag` (1 x 16), `net.<array>` (concatenated, ``S * 8``
    values, tokens of 8 for ccbv and of 6 for the others), `net.band`, `net.band.<cc|omv>` (1 x 32),
    `net.bandcat.<cc|omv>` (concatenated, 54 tokens of 8: ruling C4), `net.profile` (24 x 4) and `net.map` (48 x 2).
    """
    out: dict[str, Arm] = {}

    def add(name: str, signals: Sequence[str], concat: bool, net: bool, n_tok: int, n_val: int) -> None:
        out[name] = Arm(name, tuple(signals), concat, net, n_tok, n_val)

    array_tokens = (("ccbv", 4, 8), ("obr", 6, 6), ("obv", 6, 6), ("flux", 6, 6))     # short, pca n_val, net n_val
    add("klt.series", series, False, False, 1, 16)
    add("klt.mag", MAGNETICS, False, False, 1, 16)
    for short, n_val, _ in array_tokens:
        sig = SHORT[short]
        add(f"pca.{short}", [sig], True, False, _tokens(f"pca.{short}", streams[sig] * 16, n_val), n_val)
    add("klt.band", BANDS, False, False, 1, 32)
    for short in ("cc", "omv"):
        add(f"klt.band.{short}", [SHORT[short]], False, False, 1, 32)
    for short in ("cc", "omv"):
        sig = SHORT[short]
        add(f"pca.band.{short}", [sig], True, False, _tokens(f"pca.band.{short}", streams[sig] * 32, 4), 4)
    add("net.series", series, False, True, 1, 16)
    add("net.mag", MAGNETICS, False, True, 1, 16)
    for short, _, n_val in array_tokens:
        sig = SHORT[short]
        add(f"net.{short}", [sig], True, True, _tokens(f"net.{short}", streams[sig] * 8, n_val), n_val)
    add("net.band", BANDS, False, True, 1, 32)
    for short in ("cc", "omv"):
        add(f"net.band.{short}", [SHORT[short]], False, True, 1, 32)
    for short in ("cc", "omv"):
        add(f"net.bandcat.{short}", [SHORT[short]], True, True, BANDCAT_TOKENS, 8)
    add("net.profile", profiles, False, True, 24, 4)
    add("net.map", maps, False, True, 48, 2)
    return out


def net_name(arm: str, seed: int, variant: str | None = None) -> str:
    """The recoder directory of a trained network: ``<arm>.s<seed>``, with ``.<variant>`` appended if given."""
    return f"{arm}.s{seed}" + (f".{variant}" if variant else "")


def _n_streams(spec: tasks.SignalSpec, kind: str) -> int:
    """Streams of a frame: a channel of a `frame_dct` signal, a (probe, band) of a `band_dct` one, one else."""
    if kind == "band_dct":
        return int(spec.values_shape[0]) * identity.N_BANDS
    if kind == "frame_dct":
        return int(np.prod(spec.values_shape, dtype=np.int64))
    return 1


def catalog(specs: Sequence[tasks.SignalSpec]) -> tuple[list[str], list[str], list[str], dict[str, int]]:
    """``(series, profiles, maps, streams)`` of the signals: the `frame_dct` signals with more than one sample per
    5 ms frame, the linear signals of the profile family, those of the map family, and the streams of every signal."""
    series, profiles, maps, streams = [], [], [], {}
    for spec in specs:
        kind = assign.codec_kind(spec)
        streams[spec.name] = _n_streams(spec, kind)
        if kind == "frame_dct" and frame_len(spec.dt) > 1:
            series.append(spec.name)
        elif kind == "linear":
            (maps if registry.family_of(spec) == "map" else profiles).append(spec.name)
    return series, profiles, maps, streams


def _front_names(specs: Sequence[tasks.SignalSpec]) -> list[str]:
    """The signals that have a store: band signals, linear signals and series signals, in the registry's order."""
    series, profiles, maps, _ = catalog(specs)
    keep = {*series, *profiles, *maps, *BANDS}
    return [s.name for s in specs if s.name in keep]


# ------------------------------------------------------------------------------------- the measure

def left_over(recoder: recode.Recoder, e: np.ndarray, ident: np.ndarray, sizes: Sequence[int]) -> list[float]:
    """The share of the energy of examples `e` that a recoder's first values leave out, for each of `sizes`.

    `e` is ``(M, d_in)`` in example units and `ident` int64 ``(M,)`` (the rows of the stream table; -1 for none).
    For each size, the mean square of ``e - decode(first size values of encode(e))`` over the mean square of `e`
    (1 keeps nothing of it, 0 all). A NaN entry of `e` (a missing stream of a concatenated example) is encoded as
    0, as the recoded codec does, and counts in neither sum. NaN if `e` has no finite entry.
    """
    e = np.asarray(e)
    ok = np.isfinite(e)
    e0 = np.where(ok, e, 0.0)
    v = recoder.encode(e0, ident)
    total = float((e0.astype(np.float64) ** 2).sum())
    out = []
    for size in sizes:
        cut = np.zeros_like(v)
        cut[:, :size] = v[:, :size]
        miss = np.where(ok, e0 - recoder.decode(cut, ident), 0.0)
        out.append(float((miss ** 2).sum()) / total if total > 0 else float("nan"))
    return out


# --------------------------------------------------------------------------------------- the store

def _stats_sha(cache_root: Path, fit_meta: Mapping[str, object]) -> str:
    """The one hash of ``stats.npz``, the manifest and the fit's meta (`ValueError` naming "statistics" if they differ)."""
    manifest = json.loads((Path(cache_root) / "manifest.json").read_text())
    shas = {"stats.npz": transform.stats_sha256(Path(cache_root) / "stats.npz"), "manifest.json": manifest["stats_sha256"],
            "fit.json": fit_meta["stats_sha256"]}
    if len(set(shas.values())) != 1:
        raise ValueError("the statistics are not one set: " + ", ".join(f"{k} {str(v)[:12]}" for k, v in shas.items()))
    return shas["stats.npz"]


def run_store(args: argparse.Namespace) -> int:
    """Write the training and validation stores. Prints one line per signal: frames, S, D, skipped shots (or
    "no data"). Exit 2 for a refusal, before any write."""
    if args.n_val < 1 or args.workers < 1:
        raise Refused("--n-val and --workers are at least 1")
    stage1 = paths.stage1_dir()
    out = stage1 / "store" / STORE_VERSION
    dirs = {"train": out / "train", "val": out / "val"}
    held = [d for d in dirs.values() if (d / "store.json").exists()]
    if held and not args.force:
        raise Refused(f"{held[0] / 'store.json'} exists: a store is written once; --force writes it again")
    assignment_path = Path(args.assignment) if args.assignment else stage1 / "assignments" / f"{DEFAULT_ASSIGNMENT}.json"
    fit_dir = stage1 / "fit" / FIT_VERSION
    if not assignment_path.is_file():
        raise Refused(f"there is no assignment at {assignment_path}")
    if not (fit_dir / "fit.npz").exists():
        raise Refused(f"there is no fit at {fit_dir}: run `run_tokenizer.py fit` first")
    root = Path(args.cache_dir) if args.cache_dir else paths.cache_dir()
    try:
        assignment = assign.load_assignment(assignment_path)
        fits, fit_meta = fit.load_fit(fit_dir)
        stats_sha = _stats_sha(root, fit_meta)
        stats = transform.load_stats(root / "stats.npz")
        specs = registry.all_signals()
        train = [int(s) for s in fit_meta["shots"]]
    except (OSError, KeyError, ValueError) as e:
        raise Refused(f"the inputs cannot be read ({type(e).__name__}: {e})") from e
    val = cache.pick_evenly(tasks.split_shots("val_select"), args.n_val)
    gone = [s for s in (*train, *val) if not (root / "shots" / f"{s}.h5").exists()]
    if gone:
        raise Refused(f"{len(gone)} shots have no cache file in {root}; the first is {gone[0]}")

    fronts, info, no_fit = {}, {}, []
    wanted = set(_front_names(specs))
    for spec in specs:
        if spec.name not in wanted:
            continue
        kind = assign.codec_kind(spec)
        setting = assignment.get(spec.name)
        if setting is None or setting.codec != kind:
            raise Refused(f"the setting of {spec.name} must be a plain {kind!r} codec to build its front, got "
                          f"{None if setting is None else setting.codec!r}")
        if spec.name not in fits or fits[spec.name].n_examples == 0:
            no_fit.append(spec.name)
            continue
        vpt = int(setting.k[1]) if kind == "linear" else 1
        fronts[spec.name] = store.front_codec(spec, stats[spec.name], fits[spec.name], kind, setting.domain, vpt)
        s, _ = store.stream_shape(fronts[spec.name])
        info[spec.name] = {"kind": kind, "domain": setting.domain, "n_streams": int(s), "values_per_token": vpt}
    for d in dirs.values():
        _prepare(d)                                         # after every refusal above
    for name in no_fit:
        print(f"{name}: no data (not fitted: no training shot has it)", flush=True)
    made = {}
    for key, shots in (("train", train), ("val", val)):
        made[key] = store.build_store(root, shots, fronts, workers=args.workers)
    for name, front in fronts.items():
        s, d = store.stream_shape(front)
        tr, va = made["train"][name], made["val"][name]
        n_data = int(np.isfinite(tr.values).any(axis=(1, 2)).sum()) if len(tr.values) else 0
        if n_data == 0:
            print(f"{name}: no data (skipped shots: train {tr.n_skipped}, val {va.n_skipped})", flush=True)
        else:
            print(f"{name}: train frames {len(tr.values)} ({n_data} with data), val frames {len(va.values)}, "
                  f"S {s}, D {d}, skipped shots: train {tr.n_skipped}, val {va.n_skipped}", flush=True)
    common = {"date": datetime.now(timezone.utc).isoformat(), "assignment": str(assignment_path),
              "assignment_sha256": assign.assignment_sha256(assignment), "fit_sha256": fit.fit_sha256(fit_dir),
              "stats_sha256": stats_sha, "learned_code_sha256": learned.code_sha256(), "cache_dir": str(root),
              "signals": info}
    for key, shots in (("train", train), ("val", val)):
        if (dirs[key] / "store.json").exists():
            (dirs[key] / "store.json").unlink()             # --force: the old store is gone before the new one is written
        store.save_store(made[key], dirs[key], {**common, "split": "fit shots" if key == "train" else "val_select",
                                                "shots": shots})
    return 0


# --------------------------------------------------------------------------------------- linear

def _rows(e: np.ndarray, concat: bool) -> tuple[np.ndarray, np.ndarray]:
    """The examples of a store in example units ``(M, S, D)``: ``(rows, stream)``.

    Per stream: every stream with all D values finite, ``rows (R, D)`` and `stream` int64 ``(R,)`` its index
    in the signal. Concatenated: one row per frame with some data, ``(R, S * D)`` with NaN for a missing
    stream, and `stream` all -1 (no identity).
    """
    m, s, d = e.shape
    if concat:
        rows = e.reshape(m, s * d)
        rows = rows[np.isfinite(rows).any(axis=1)]
        return rows, np.full(len(rows), -1, dtype=np.int64)
    flat = e.reshape(m * s, d)
    keep = np.isfinite(flat).all(axis=1)
    return flat[keep], np.tile(np.arange(s, dtype=np.int64), m)[keep]


def _value_ranges(recoder: recode.Recoder, e: np.ndarray, ident: np.ndarray, chunk: int = 20000) -> tuple:
    """``(lo, hi, width)`` of the recoder values of a store's examples `e` ``(M, S, D)`` (example units), each
    ``(S, n_values)`` per stream and ``(1, n_values)`` concatenated, in recoder values (`grid.fit_ranges`)."""
    m, s, d = e.shape
    concat = bool(recoder.concat)
    out = np.full((m, 1 if concat else s, recoder.n_values), np.nan, dtype=np.float32)
    for i in range(0, m, chunk):
        part = e[i:i + chunk]
        finite = np.isfinite(part).all(-1)                       # (n, S)
        if concat:
            frame = finite.any(-1)
            rows = np.where(finite[..., None], part, 0.0)[frame].reshape(-1, s * d)
            if len(rows):
                out[i:i + chunk][frame, 0] = recoder.encode(rows, np.full(len(rows), -1, dtype=np.int64))
        elif finite.any():
            out[i:i + chunk][finite] = recoder.encode(part[finite], np.broadcast_to(ident, finite.shape)[finite])
    return grid.fit_ranges(out)


def run_linear(args: argparse.Namespace) -> int:
    """Fit one linear arm on the training store and write ``<stage1>/recoders/<arm>``; print, per signal, `left_over`
    on the validation store at 1, 2, 4, 8 and 16 values per stream (concatenated: S times those). Exit 2 for an
    unknown arm, a network arm, an existing directory, or a store that is missing or does not fit the arm."""
    stage1 = paths.stage1_dir()
    series, profiles, maps, streams = catalog(registry.all_signals())
    all_arms = arms(series, profiles, maps, streams)
    if args.arm not in all_arms:
        raise Refused(f"unknown arm {args.arm!r}; `arms` lists them")
    arm = all_arms[args.arm]
    if arm.net:
        raise Refused(f"{arm.name} is a network arm: it is trained, not fitted by `linear`")
    target = recoders_root() / arm.name
    if (target / recode.RECODER_JSON).exists():
        raise Refused(f"{target} exists: a recoder is written once")
    dirs = {"train": stage1 / "store" / STORE_VERSION / "train", "val": stage1 / "store" / STORE_VERSION / "val"}
    for d in dirs.values():
        if not (d / "store.json").is_file():
            raise Refused(f"there is no store at {d}: run `store` first")
    try:
        train, train_meta = store.load_store(dirs["train"], list(arm.signals))
        val, _ = store.load_store(dirs["val"], list(arm.signals))
    except (OSError, KeyError, ValueError) as e:
        raise Refused(f"the stores do not hold the signals of {arm.name} ({type(e).__name__}: {e})") from e
    info = {n: train_meta["signals"][n] for n in arm.signals}
    n_values = arm.n_tok * arm.n_val
    if arm.concat and len(arm.signals) != 1:
        raise Refused(f"{arm.name} is concatenated over {len(arm.signals)} signals; plan 1c has one per such arm")
    for n in arm.signals:
        s, d = train[n].values.shape[1:]
        if s != streams[n] or s != info[n]["n_streams"]:
            raise Refused(f"the store holds {n} with {s} streams, the registry has {streams[n]}")
        if (s * d if arm.concat else d) != n_values:
            raise Refused(f"{arm.name} keeps {n_values} values, but {n} has {s * d if arm.concat else d} per example "
                          "in the store: a linear arm keeps them all")
    root = Path(args.cache_dir) if args.cache_dir else Path(train_meta["cache_dir"])
    try:
        table = identity.build_table(registry.load_registry(root / "registry.csv"),
                                     [(n, info[n]["kind"], info[n]["n_streams"]) for n in arm.signals])
    except (OSError, ValueError) as e:
        raise Refused(f"the stream table cannot be made from {root / 'registry.csv'} ({type(e).__name__}: {e})") from e
    norms = {n: store.fit_norm(train[n].values) for n in arm.signals}
    e_train = {n: norms[n].apply(train[n].values) for n in arm.signals}
    rows = np.concatenate([_rows(e_train[n], arm.concat)[0] for n in arm.signals])
    if len(rows) < 2:
        raise Refused(f"{arm.name}: the training store has {len(rows)} examples; a basis needs at least 2")
    _prepare(target)                                        # after every refusal above
    recoder = recode.fit_linear(rows, n_values, concat=arm.concat, n_val=arm.n_val)
    ranges = {n: _value_ranges(recoder, e_train[n], table.index_of(n, info[n]["n_streams"])) for n in arm.signals}
    meta = {"signals": info, "arm": arm.name, "date": datetime.now(timezone.utc).isoformat(),
            "n_train_examples": int(len(rows)), "store": {k: v for k, v in train_meta.items() if k != "signals"}}
    recode.save_recoder(target, recoder, norms, table, meta, ranges)
    for n in arm.signals:
        s = info[n]["n_streams"]
        r, stream = _rows(norms[n].apply(val[n].values), arm.concat)
        if len(r) == 0:
            print(f"{arm.name} {n}: no data in the validation store", flush=True)
            continue
        ident = stream if arm.concat else table.index_of(n, s)[stream]
        sizes = [min(k * (s if arm.concat else 1), n_values) for k in SIZES]
        left = left_over(recoder, r, ident, sizes)
        print(f"{arm.name} {n}: {len(r)} validation examples; left over "
              + ", ".join(f"{k}: {x:.4f}" for k, x in zip(sizes, left)), flush=True)
    print(f"wrote {target}", flush=True)
    return 0


# ------------------------------------------------------------------------------------------ joint

def _frame_keys(shots: np.ndarray) -> np.ndarray:
    """An int64 key per frame of a store, ``shot * 2**20 + the frame's index within its shot`` (a store is written
    shot by shot, frame after frame from the first sample), so that the frames of two stores can be matched. This
    assumes that the signals of a shot start at one time: measured 2026-11-23 on 41 shots of cache v1 spread evenly
    over its files, the four magnetics signals have one ``t0`` in every one of them."""
    n = len(shots)
    start = np.maximum.accumulate(np.where(np.r_[True, shots[1:] != shots[:-1]], np.arange(n), 0)) if n else shots
    index = np.arange(n) - start
    if n and index.max() >= 1 << 20:
        raise ValueError("a shot has 2**20 frames or more")
    return shots.astype(np.int64) * (1 << 20) + index


def _common_frames(examples: Mapping[str, np.ndarray], stores: Mapping[str, store.Store]) -> np.ndarray:
    """The frames of every signal side by side: ``(R, sum of S * D)`` over the frames that every signal has with some
    finite stream (a NaN stays NaN), in example units, signals in the order of `examples` (each ``(M, S, D)``)."""
    keys = {n: _frame_keys(stores[n].shots) for n in examples}
    common = None
    for n, e in examples.items():
        k = keys[n][np.isfinite(e).any(axis=(1, 2))]
        common = k if common is None else np.intersect1d(common, k)
    blocks = []
    for n, e in examples.items():
        order = np.argsort(keys[n])
        at = order[np.searchsorted(keys[n][order], common)]
        blocks.append(e[at].reshape(len(common), -1))
    return np.concatenate(blocks, axis=1)


def run_joint(args: argparse.Namespace) -> int:
    """For `--sizes` k, the share of the validation examples' energy (example units) left out by keeping k values per
    stream in all, in three ways: the joint basis of all streams of all the listed signals (one example is a frame
    of every signal, fitted on the training frames in which every listed signal is present), each signal's own
    concatenated basis (the fit of `linear --arm pca.<signal>`, ``k * S`` values of each signal), and one per-stream
    basis pooled over all their streams (k values of each stream). The validation examples are the frames in which
    every listed signal is present, meaning it has at least one finite stream in the frame; a missing stream is
    encoded as 0 and counts in neither sum, as `linear` and ruling C21 do
    (`compare.left_over_sums` with ``masked``: a probe that is dead in every frame leaves no frame in which all
    streams are finite, measured 2026-11-23 on the stage-1c trial store: 0 of 167 validation frames). The three
    measures see the same observed streams (a pooled basis, which takes one stream at a time, the streams with all
    their values). Prints one line per size. Exit 2 for a refusal (before any file is read for the fits)."""
    names = []
    for token in (t for t in args.signals.split(",") if t):
        full = SHORT.get(token, token)
        if full in names:
            raise Refused(f"--signals lists {full} twice")
        names.append(full)
    if len(names) < 2:
        raise Refused(f"--signals names at least two signals (short names {sorted(SHORT)} or full names), got {args.signals!r}")
    try:
        sizes = [int(v) for v in args.sizes.split(",") if v]
    except ValueError as e:
        raise Refused(f"--sizes must be integers, comma-separated, got {args.sizes!r}") from e
    if not sizes or min(sizes) < 1:
        raise Refused(f"--sizes must be one or more integers of at least 1, got {args.sizes!r}")
    stage1 = paths.stage1_dir()
    dirs = {"train": stage1 / "store" / STORE_VERSION / "train", "val": stage1 / "store" / STORE_VERSION / "val"}
    for d in dirs.values():
        if not (d / "store.json").is_file():
            raise Refused(f"there is no store at {d}: run `store` first")
    try:
        train, _ = store.load_store(dirs["train"], names)
        val, _ = store.load_store(dirs["val"], names)
    except (OSError, KeyError, ValueError) as e:
        raise Refused(f"the stores do not hold {names} ({type(e).__name__}: {e})") from e
    depths = {n: train[n].values.shape[2] for n in names}
    if len(set(depths.values())) != 1:
        raise Refused(f"the signals have different numbers of values per stream, {depths}: a pooled basis needs one")
    d = depths[names[0]]
    if max(sizes) > d:
        raise Refused(f"--sizes asks for {max(sizes)} values per stream, and a stream has {d}")
    n_streams = {n: train[n].values.shape[1] for n in names}
    total_streams = sum(n_streams.values())

    norms = {n: store.fit_norm(train[n].values) for n in names}
    e_train = {n: norms[n].apply(train[n].values) for n in names}
    e_val = {n: norms[n].apply(val[n].values) for n in names}
    joint_train = _common_frames(e_train, train)
    joint_val = _common_frames(e_val, val)
    pooled_val = joint_val.reshape(-1, d)
    pooled_val = pooled_val[np.isfinite(pooled_val).all(axis=1)]
    print(f"joint {', '.join(names)}: {total_streams} streams of {d} values; training frames {len(joint_train)}, "
          f"validation frames {len(joint_val)} with {len(pooled_val)} streams observed of {len(joint_val) * total_streams}",
          flush=True)
    if len(joint_train) < 2:
        raise Refused("fewer than 2 training frames have data in every listed signal")
    if len(pooled_val) == 0:
        print("no data: no validation frame has data in every listed signal", flush=True)
        return 0

    each_total = [k * total_streams for k in sizes]
    joint_residual, joint_total = compare.left_over_sums(joint_train, joint_val, each_total, masked=True)
    own_residual, own_total = np.zeros(len(sizes)), 0.0
    at = 0
    for n in names:
        width = n_streams[n] * d
        residual, total = compare.left_over_sums(_rows(e_train[n], True)[0], joint_val[:, at:at + width],
                                                 [k * n_streams[n] for k in sizes], masked=True)
        own_residual += residual
        own_total += total
        at += width
    pooled_train = np.concatenate([_rows(e_train[n], False)[0] for n in names])
    pooled_residual, pooled_total = compare.left_over_sums(pooled_train, pooled_val, sizes)
    for i, k in enumerate(sizes):
        print(f"{k} values per stream ({k * total_streams} in all): left over joint {joint_residual[i] / joint_total:.4f}, "
              f"own concatenated {own_residual[i] / own_total:.4f}, pooled per stream {pooled_residual[i] / pooled_total:.4f}",
              flush=True)
    return 0


# ----------------------------------------------------------------------------------------- arms

def run_arms(args: argparse.Namespace) -> int:
    """One line per arm (name, signals, streams, concatenated, linear or network, tokens x values) and the recoder
    directories that exist for it (a network's are ``<arm>.s<seed>[.<variant>]``). Exit 0."""
    series, profiles, maps, streams = catalog(registry.all_signals())
    table = arms(series, profiles, maps, streams)
    root = recoders_root()
    held = sorted(p.name for p in root.iterdir() if (p / recode.RECODER_JSON).is_file()) if root.is_dir() else []
    n_present = 0
    for arm in table.values():
        mine = [h for h in held if (h.startswith(f"{arm.name}.s") if arm.net else h == arm.name)]
        n_present += bool(mine)
        print(f"{arm.name:<17} {len(arm.signals):>2} signals, {sum(streams[n] for n in arm.signals):>3} streams, "
              f"{'concatenated' if arm.concat else 'per stream  '}, {'network' if arm.net else 'linear '}, "
              f"{arm.n_tok} x {arm.n_val} values; " + (f"present: {', '.join(mine)}" if mine else "not trained"))
    print(f"{len(table)} arms, {n_present} present under {root}", flush=True)
    return 0


# ----------------------------------------------------------------------------------------- train

def _run_headroom():
    """`scripts/mast/run_headroom.py` loaded by path (its GPU and lock helpers; `py.sh` runs Python with `-I`, so the
    scripts directory is not importable). The module name is `run_tokenizer.py`'s, so one process loads it once.
    Loading it imports torch (through `harness`) but does not initialize CUDA, so `CUDA_VISIBLE_DEVICES` can still be
    set after it, as `run_headroom.py` documents; `train` loads it for `--cpu` only after its refusals."""
    name = "run_headroom_for_tokenizer"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HEADROOM_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclasses look their module up here
        spec.loader.exec_module(module)
    return sys.modules[name]


def _refuse_train_options(args: argparse.Namespace) -> None:
    """`Refused` for options that do not fit, before anything is read."""
    if (args.gpu is None) == (not args.cpu):
        raise Refused("give exactly one of --gpu N (a GPU without a compute process) and --cpu")
    if args.share_gpu and args.gpu is None:
        raise Refused("--share-gpu belongs to --gpu")
    if args.variant is not None and not _VARIANT.fullmatch(args.variant):
        raise Refused(f"--variant {args.variant!r} is not a name (letters, digits, - and _)")
    if (args.lr != DEFAULT_LR or args.steps != DEFAULT_STEPS) and args.variant is None:
        raise Refused("--lr and --steps other than the defaults name another run: give it a --variant")
    if args.steps < 1 or not (math.isfinite(args.lr) and args.lr > 0) or args.seed < 0:
        raise Refused("--steps is at least 1, --lr is positive and --seed is not negative")


def _unfinished(run_dir: Path, force: bool) -> None:
    """`Refused` if `run_dir` holds a finished run (a `result.json`) and `force` is not given."""
    if (run_dir / "result.json").exists() and not force:
        raise Refused(f"{run_dir / 'result.json'} exists: a trained network is not overwritten; --force trains it "
                      "again, or give another --seed or --variant")


def rule7_usage(values: np.ndarray, rows: np.ndarray) -> dict:
    """Rule 7 for one signal, read as ruling C33 reads it, and the plan's literal reading beside it.

    `values` is ``(M, n_tok, n_val)``, a network's bounded values (unitless, in [-1, 1]) on the signal's M
    validation examples. `rows` is ``(M, d)``, the encoder's input of the same examples (example units and the
    stream's row in the identity table, as one float row): two examples are one example only when their values
    and their identity are the same. Examples with a NaN anywhere in `values` are left out of everything.

    Returns, from `compare.usage` on all finite examples (the plan's reading, unchanged): `collapsed_any_token`
    (some token's first value is under 0.3 normalised entropy), `first_value_entropy` and `active_values`, with
    `n_examples`. Then, on the distinct examples (an exact duplicate row counted once): `n_distinct_examples`;
    `first_value_entropy_by_token`, a list over the tokens; `active_tokens`, how many of them reach
    `RULE7_MIN_ENTROPY` (reported, it decides nothing); and `collapsed`, True when none does (C33 b: the verdict).
    `share_most_common_row` is the share, over all finite examples, of the most common row. For a one-token arm
    with no repeated row `collapsed` is `collapsed_any_token`. `compare.usage`'s `ValueError` (no finite
    example) is not caught."""
    v = np.asarray(values, dtype=np.float64)
    r = np.asarray(rows)
    if v.ndim != 3 or r.ndim != 2 or len(r) != len(v):
        raise ValueError(f"values must be (M, n_tok, n_val) and rows (M, d) of the same M, got {v.shape} and {r.shape}")
    plan = compare.usage(v)                                   # all finite examples; ValueError if none is left
    finite = np.isfinite(v).all(axis=(1, 2))
    _, first, counts = np.unique(r[finite], axis=0, return_index=True, return_counts=True)
    distinct = v[finite][first]
    by_token = [compare.usage(distinct[:, t:t + 1, :])["first_value_entropy"] for t in range(v.shape[1])]
    active = sum(h >= RULE7_MIN_ENTROPY for h in by_token)
    return {"first_value_entropy": plan["first_value_entropy"], "active_values": plan["active_values"],
            "collapsed_any_token": plan["collapsed"], "n_examples": plan["n_examples"],
            "n_distinct_examples": int(len(first)), "share_most_common_row": float(counts.max() / counts.sum()),
            "first_value_entropy_by_token": [float(h) for h in by_token], "active_tokens": int(active),
            "collapsed": bool(active == 0)}


def run_train(args: argparse.Namespace) -> int:
    """Train the network of one arm on the training store, validate on the validation store, write the run into
    ``<stage1>/recoders/<arm>.s<seed>[.<variant>]`` and print its last line. Exit 2 for a refusal (the options, the
    arm, a finished run, a busy GPU: before any store is read), 1 for a loss that is not finite, 3 if the network
    collapsed (`rule7_usage`: some signal has no token whose first value is used, ruling C33), else 0. Written: `config.json` and `trained_by.json` before the first step,
    `train.lock` while it trains, `train_log.csv` and `best.pt` during it, then `recoder.json`, `norm.npz` and
    `table.npz` (a network's values have the even grid: no ranges) and last `result.json`."""
    _refuse_train_options(args)
    series, profiles, maps, streams = catalog(registry.all_signals())
    all_arms = arms(series, profiles, maps, streams)
    if args.arm not in all_arms:
        raise Refused(f"unknown arm {args.arm!r}; `arms` lists them")
    arm = all_arms[args.arm]
    if not arm.net:
        raise Refused(f"{arm.name} is a linear arm: it is fitted by `linear`, not trained")
    stage1 = paths.stage1_dir()
    name = net_name(arm.name, args.seed, args.variant)
    run_dir = recoders_root() / name
    _unfinished(run_dir, args.force)
    dirs = {"train": stage1 / "store" / STORE_VERSION / "train", "val": stage1 / "store" / STORE_VERSION / "val"}
    for d in dirs.values():
        if not (d / "store.json").is_file():
            raise Refused(f"there is no store at {d}: run `store` first")
    device = "cpu"
    if args.gpu is not None:
        rh = _run_headroom()
        if rh._cuda_initialized():
            raise Refused("CUDA is already initialized in this process, so --gpu cannot select the GPU")
        looked_up = rh._look_up_gpu(args.gpu)
        if looked_up is None:
            return 2                                            # it printed why
        gpu_uuid, gpu_pids = looked_up
        if rh._refuse_busy_gpu(args.gpu, gpu_pids, args.share_gpu):
            return 2
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid           # read when CUDA starts, which it has not: the GPU is cuda:0
        device = "cuda:0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        rh = _run_headroom()
    from tokamak_foundation_model.ignite.mast.tokenizers.learned import nets, train as trainer   # torch, from here on

    try:
        train, train_meta = store.load_store(dirs["train"], list(arm.signals))
        val, val_meta = store.load_store(dirs["val"], list(arm.signals))
    except (OSError, KeyError, ValueError) as e:
        raise Refused(f"the stores do not hold the signals of {arm.name} ({type(e).__name__}: {e})") from e
    try:                                                        # every store-meta value the run needs, before any write
        info = {n: train_meta["signals"][n] for n in arm.signals}
        specs = [(n, info[n]["kind"], info[n]["n_streams"]) for n in arm.signals]
        cache_dir = train_meta["cache_dir"]
        hashes = {k: train_meta[k] for k in STORE_HASHES}
        val_hashes = {k: val_meta[k] for k in STORE_HASHES}
    except KeyError as e:
        raise Refused(f"the store's store.json has no {e.args[0]!r} (a store from an older `store`?): "
                      "write the stores again with `store --force`") from e
    for k in STORE_HASHES:
        if hashes[k] != val_hashes[k]:
            raise Refused(f"the validation store's {k} ({val_hashes[k][:8]}) is not the training store's "
                          f"({hashes[k][:8]}): the two stores are not of one assignment and fit")
    if arm.concat and len(arm.signals) != 1:
        raise Refused(f"{arm.name} is concatenated over {len(arm.signals)} signals; plan 1c has one per such arm")
    for n in arm.signals:
        s = train[n].values.shape[1]
        if s != streams[n] or s != info[n]["n_streams"]:
            raise Refused(f"the store holds {n} with {s} streams, the registry has {streams[n]}")
    depths = {n: train[n].values.shape[2] for n in arm.signals}
    if not arm.concat and len(set(depths.values())) != 1:
        raise Refused(f"{arm.name} takes one stream at a time, and its signals have {depths} values per stream")
    d_in = train[arm.signals[0]].values.shape[1] * depths[arm.signals[0]] if arm.concat else depths[arm.signals[0]]
    root = Path(cache_dir)
    try:
        table = identity.build_table(registry.load_registry(root / "registry.csv"), specs)
    except (OSError, ValueError) as e:
        raise Refused(f"the stream table cannot be made from {root / 'registry.csv'} ({type(e).__name__}: {e})") from e
    norms = {n: store.fit_norm(train[n].values) for n in arm.signals}
    train_ex = trainer.examples(train, norms, table, concat=arm.concat)
    val_ex = trainer.examples(val, norms, table, concat=arm.concat)
    if len(train_ex[0]) == 0 or len(val_ex[0]) == 0:
        raise Refused(f"{arm.name}: {len(train_ex[0])} training and {len(val_ex[0])} validation examples; both need some")
    cfg = trainer.TrainConfig(steps=args.steps, lr=args.lr, seed=args.seed)
    net_config = {"d_in": int(d_in), "n_values": int(arm.n_tok * arm.n_val), "n_streams": 0 if arm.concat else len(table),
                  "d_model": int(cfg.d_model), "depth": int(cfg.depth), "p_drop_ident": float(cfg.p_drop_ident),
                  "concat": bool(arm.concat), "n_tok": int(arm.n_tok), "n_val": int(arm.n_val)}

    _prepare(run_dir)                                           # after every refusal above
    try:
        lock = rh._take_lock(run_dir)
    except rh.Refused as e:
        raise Refused(str(e)) from e
    try:
        _unfinished(run_dir, args.force)                        # another training may have finished since the first look
        for f in RUN_FILES:                                     # the files of an unfinished run, or of the one --force replaces
            (run_dir / f).unlink(missing_ok=True)
        rh._write_json_atomic(run_dir / "config.json", {
            "arm": arm.name, "seed": args.seed, "variant": args.variant, "train": dataclasses.asdict(cfg), "net": net_config})
        rh._write_json_atomic(run_dir / "trained_by.json", {
            "learned_code_sha256": learned.code_sha256(), "tokenizer_code_sha256": code_sha256(),
            **hashes, "argv": list(args.argv),
            "date": datetime.now(timezone.utc).isoformat()})
        try:
            result = trainer.train(net_config, cfg, train_ex, val_ex, table, run_dir, device,
                                   log=lambda line: print(line, flush=True))
        except FloatingPointError as e:
            print(f"failed: {e}; {run_dir} holds the log up to it and no result.json", file=sys.stderr)
            return 1
        recoder = nets.load_net_recoder(run_dir, table)         # the best weights, as a later reader loads them
        usage = {}
        for n in arm.signals:                                   # rule 7: one verdict per signal, on the validation examples
            e, _, ident = trainer.examples({n: val[n]}, {n: norms[n]}, table, concat=arm.concat)
            if len(e) == 0:
                usage[n] = {"n_examples": 0, "collapsed": None}   # no data is not a verdict
                print(f"{name} {n}: no validation examples, no verdict", flush=True)
                continue
            values = recoder.encode(e, ident)
            rows = np.concatenate([e.astype(np.float64), ident[:, None].astype(np.float64)], axis=1)   # the example and its identity
            usage[n] = u = rule7_usage(values.reshape(len(values), arm.n_tok, arm.n_val), rows)
            print(f"{name} {n}: {'COLLAPSED' if u['collapsed'] else 'not collapsed'} (C33), "
                  f"{'collapsed' if u['collapsed_any_token'] else 'not collapsed'} on any token (plan's reading), "
                  f"active tokens {u['active_tokens']} of {arm.n_tok}, active values {u['active_values']} of {arm.n_val}, "
                  f"{u['n_distinct_examples']} distinct of {u['n_examples']} examples, "
                  f"most common row {u['share_most_common_row']:.1%}", flush=True)
        collapsed = any(u["collapsed"] is True for u in usage.values())
        meta = {"signals": info, "arm": arm.name, "seed": args.seed, "variant": args.variant,
                "date": datetime.now(timezone.utc).isoformat(), "n_train_examples": int(len(train_ex[0])),
                "n_val_examples": int(len(val_ex[0])), "store": {k: v for k, v in train_meta.items() if k != "signals"}}
        recode.save_recoder(run_dir, recoder, norms, table, meta)
        plateau = trainer.plateau(pd.read_csv(run_dir / trainer.LOG_CSV))
        rh._write_json_atomic(run_dir / "result.json", {**result, "plateau": plateau, "collapsed": collapsed,
                                                        "usage": usage})
    finally:
        rh._release_lock(lock)
    unchecked = [n for n, u in usage.items() if u["collapsed"] is None]
    note = f" ({len(unchecked)} of {len(usage)} signals unchecked: no validation examples)" if unchecked else ""
    differ = [n for n, u in usage.items() if u["collapsed"] is not None and u["collapsed"] != u["collapsed_any_token"]]
    tail = f" (rule 7 readings differ on: {', '.join(differ)}; C33 decides)" if differ else ""
    print(f"{name}: best val_mse {result['best_val_mse']:.4g} at step {result['best_step']} of {result['steps']}, "
          f"plateau {'yes' if plateau else 'no'}, collapsed {'yes' if collapsed else 'no'}{note}, {result['seconds']:.0f} s{tail}",
          flush=True)
    return 3 if collapsed else 0


# ------------------------------------------------------------------------------------- fluct

FLUCT_SAMPLES = 250                       # samples of a 5 ms frame at 50 kHz
FLUCT_COLUMNS = ("signal", "k_trend", "n_frames", "band_to_trend", "share_below_5khz", "persistence")


def run_fluct(args: argparse.Namespace) -> int:
    """For every series signal with 250 samples per 5 ms frame (`frames.frame_len` of its sampling interval; four on
    cache v1), the power above the trend path, by frequency, over `--n-shots` ``val_select`` shots spread evenly
    (`fluct.summarize`; the samples are the front codec's domain, the K of the trend is the signal's in
    ``<stage1>/assignments/v1.json``). Writes ``<stage1>/compare/fluct.csv`` atomically (`signal`, `k_trend`,
    `n_frames`, `band_to_trend`, `share_below_5khz`, `persistence`, `bin_00` to `bin_24`; a signal without a frame has
    `n_frames` 0 and empty cells) and prints one line per signal without the bin columns, "no data" for one without a
    frame. Exit 2 for a refusal, before any file is read (a missing assignment, fit or cache manifest) or written."""
    if args.n_shots < 1 or args.workers < 1:
        raise Refused("--n-shots and --workers are at least 1")
    stage1 = paths.stage1_dir()
    assignment_path = stage1 / "assignments" / f"{DEFAULT_ASSIGNMENT}.json"
    fit_dir = stage1 / "fit" / FIT_VERSION
    root = Path(args.cache_dir) if args.cache_dir else paths.cache_dir()
    if not assignment_path.is_file():
        raise Refused(f"there is no assignment at {assignment_path}")
    if not (root / "manifest.json").is_file():
        raise Refused(f"there is no cache manifest at {root / 'manifest.json'}")
    if not (fit_dir / "fit.npz").exists():
        raise Refused(f"there is no fit at {fit_dir}: run `run_tokenizer.py fit` first")
    specs = [s for s in registry.all_signals() if assign.codec_kind(s) == "frame_dct" and frame_len(s.dt) == FLUCT_SAMPLES]
    if not specs:
        raise Refused(f"no series signal has {FLUCT_SAMPLES} samples per frame")
    try:
        assignment = assign.load_assignment(assignment_path)
        fits, fit_meta = fit.load_fit(fit_dir)
        stats_sha = _stats_sha(root, fit_meta)
        stats = transform.load_stats(root / "stats.npz")
    except (OSError, KeyError, ValueError) as e:
        raise Refused(f"the inputs cannot be read ({type(e).__name__}: {e})") from e
    shots = cache.pick_evenly(tasks.split_shots("val_select"), args.n_shots)
    gone = [s for s in shots if not (root / "shots" / f"{s}.h5").exists()]
    if gone:
        raise Refused(f"{len(gone)} shots have no cache file in {root}; the first is {gone[0]}")
    fronts, no_fit = {}, []
    for spec in specs:
        setting = assignment.get(spec.name)
        if setting is None or setting.codec != "frame_dct":
            raise Refused(f"the setting of {spec.name} must be a plain 'frame_dct' codec (its K is the trend path), got "
                          f"{None if setting is None else setting.codec!r}")
        if spec.name not in fits or fits[spec.name].n_examples == 0:
            no_fit.append(spec)
            continue
        try:
            fronts[spec.name] = store.front_codec(spec, stats[spec.name], fits[spec.name], "frame_dct", setting.domain)
        except (KeyError, ValueError) as e:
            raise Refused(f"the front codec of {spec.name} cannot be built ({type(e).__name__}: {e})") from e
    out_dir = _prepare(stage1 / "compare")                  # after every refusal above
    print(f"{len(specs)} signals with {FLUCT_SAMPLES} samples per frame, over {len(shots)} val_select shots", flush=True)
    rows = []
    for spec in specs:
        k = int(assignment[spec.name].k[0])
        if spec.name in fronts:
            s = fluct.summarize((u for u in transform.map_shots(
                fluct.shot_domain, shots, (str(root), stats_sha, fronts[spec.name]), args.workers) if u is not None),
                                FLUCT_SAMPLES, k)
        else:
            s = fluct.summarize([], FLUCT_SAMPLES, k)         # not fitted: no training shot has it
        rows.append({"signal": spec.name, "k_trend": k, "n_frames": s["n_frames"], "band_to_trend": s["band_to_trend"],
                     "share_below_5khz": s["share_below_5khz"], "persistence": s["persistence"],
                     **{f"bin_{i:02d}": float(v) for i, v in enumerate(s["share_by_bin"])}})
        if s["n_frames"] == 0:
            print(f"{spec.name}: no data (K {k})", flush=True)
        else:
            print(f"{spec.name}: K {k}, {s['n_frames']} channel-frames, band to trend {s['band_to_trend']:.3g}, "
                  f"share below 5 kHz {s['share_below_5khz']:.3f}, persistence {s['persistence']:.3f}", flush=True)
    target = out_dir / "fluct.csv"
    tmp = target.with_name(target.name + ".tmp")
    pd.DataFrame(rows, columns=list(FLUCT_COLUMNS) + [f"bin_{i:02d}" for i in range(fluct.N_FLUCT_BINS)]).to_csv(tmp, index=False)
    os.replace(tmp, target)
    print(f"wrote {target}", flush=True)
    return 0


# -------------------------------------------------------------------------------------------- main

def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("store", help="encode the fronts of the training and validation shots")
    p.add_argument("--assignment", default=None, help="the assignment file whose domains are used (default <stage1>/assignments/v1.json)")
    p.add_argument("--n-val", type=int, default=100, help="val_select shots, spread evenly (default 100)")
    p.add_argument("--workers", type=int, default=64, help="reader processes (default 64)")
    p.add_argument("--cache-dir", default=None, help="the cache (default <root>/cache/v1)")
    p.add_argument("--force", action="store_true", help="write the store again over an existing one")
    p.set_defaults(run=run_store)

    p = sub.add_parser("linear", help="fit a linear arm on the training store")
    p.add_argument("--arm", required=True, help="a linear arm (`arms` lists them)")
    p.add_argument("--cache-dir", default=None, help="the cache whose registry.csv names the streams (default: the store's)")
    p.set_defaults(run=run_linear)

    p = sub.add_parser("joint", help="the left-over of one basis over the streams of several signals, against the own ones; a frame counts when every signal has a finite stream in it, and a missing stream is encoded as 0 and counts in neither sum")
    p.add_argument("--signals", required=True, help="signals, short (ccbv,obr,obv,flux,cc,omv) or full names, comma-separated")
    p.add_argument("--sizes", default="1,2,4,8", help="values per stream in all (default 1,2,4,8)")
    p.set_defaults(run=run_joint)

    p = sub.add_parser("arms", help="list the arms and which recoders exist")
    p.set_defaults(run=run_arms)

    p = sub.add_parser("train", help="train the network of one network arm on the stores")
    p.add_argument("--arm", required=True, help="a network arm (`arms` lists them)")
    p.add_argument("--seed", type=int, default=0, help="seed of the weights, the batches and the net's draws (default 0)")
    p.add_argument("--gpu", type=int, default=None, help="train on this GPU (one without a compute process, unless --share-gpu)")
    p.add_argument("--share-gpu", action="store_true", help="use a GPU that has compute processes")
    p.add_argument("--cpu", action="store_true", help="train on the CPU (no GPU is selected)")
    p.add_argument("--lr", type=float, default=DEFAULT_LR, help=f"learning rate (default {DEFAULT_LR}; another needs --variant)")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS, help=f"optimizer steps (default {DEFAULT_STEPS}; another needs --variant)")
    p.add_argument("--variant", default=None, help="a name that is appended to the run directory (<arm>.s<seed>.<variant>)")
    p.add_argument("--force", action="store_true", help="train again over a finished run (its files are removed first)")
    p.set_defaults(run=run_train)

    p = sub.add_parser("fluct", help="the power above the trend path of the 50 kHz channels, by frequency (a statistic; no tokenizer)")
    p.add_argument("--n-shots", type=int, default=100, help="val_select shots, spread evenly (default 100)")
    p.add_argument("--workers", type=int, default=64, help="reader processes (default 64)")
    p.add_argument("--cache-dir", default=None, help="the cache (default <root>/cache/v1)")
    p.set_defaults(run=run_fluct)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.argv = list(sys.argv[1:] if argv is None else argv)      # what `train` records in trained_by.json
    try:
        return args.run(args)
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
