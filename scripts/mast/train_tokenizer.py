"""The learned tokenizers of stage 1 (plan 1c): the arms, their examples and the linear recoders.

`store` encodes the fronts of the signals that a recoder can sit behind (every `band_dct` and `linear` signal
and each `frame_dct` signal with more than one sample per 5 ms frame: 24 signals on cache v1) over the shots of
1b's fit (training) and over `val_select` shots (validation), each in the domain of its setting in an
assignment (default ``<stage1>/assignments/v1.json``), and writes ``<stage1>/store/v1/train`` and ``.../val``.
`linear --arm NAME` fits the linear recoder of one arm (`klt.*` per stream, `pca.*` concatenated) on the
training store and writes ``<stage1>/recoders/NAME``; it prints, per signal, the share of the validation
examples' energy that is left after keeping the first 1, 2, 4, 8 and 16 values per stream. `arms` lists the 24
arms and which recoders exist. Networks are trained by later sub-commands of this file.

Units. "Front values" are what a front codec's ``encode_domain`` returns (`store.Norm` takes them to "example
units": per-stream standardized), and "recoder values" are what a recoder returns. This file does nothing at
import time except define (`store` starts spawned workers, which import it again) and imports no torch of its own.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from tokamak_foundation_model.ignite.mast import cache, paths, registry, tasks, transform
from tokamak_foundation_model.ignite.mast.tokenizers import assign, fit, grid
from tokamak_foundation_model.ignite.mast.tokenizers import learned
from tokamak_foundation_model.ignite.mast.tokenizers.frames import frame_len
from tokamak_foundation_model.ignite.mast.tokenizers.learned import identity, recode, store

FIT_VERSION = "v1"                        # 1b's fit, whose shots are the training shots
STORE_VERSION = "v1"
DEFAULT_ASSIGNMENT = "v1"                 # the assignment whose domains the fronts are encoded in
SIZES = (1, 2, 4, 8, 16)                  # values per stream at which `linear` reports what is left over
MAGNETICS = ("magnetics-b_field_pol_probe_ccbv_field", "magnetics-b_field_pol_probe_obr_field",
             "magnetics-b_field_pol_probe_obv_field", "magnetics-flux_loop_flux")
BANDS = tasks.SPECTROGRAM_SIGNALS
SHORT = {"ccbv": MAGNETICS[0], "obr": MAGNETICS[1], "obv": MAGNETICS[2], "flux": MAGNETICS[3],
         "cc": BANDS[0], "omv": BANDS[1]}
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

    p = sub.add_parser("arms", help="list the arms and which recoders exist")
    p.set_defaults(run=run_arms)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.run(args)
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
