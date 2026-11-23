"""Ladders of the fixed tokenizers: the passes over the rungs, the errors read from the cache, and the choice of K.

A ladder scores every rung of a signal's ladder (`assign.rungs`) as the floor of a tokenizer, the NRMSE of
a perfect predictor through it, on TokaMark's evaluator; `passes` lists what is scored together (one
setting for every output signal of a task, so one scoring pass of the task serves all its signals) and
`run_ladder` collects the numbers into a table. Rule 4 has no task to score: for a signal no task
predicts the error is read from the cache (`shot_error`, TokaMark units, per shot over the valid
samples). The spectral contrast of rule 5 is read from the cache too (`band_contrast`). The choice of
K is `choose` (rules 2 and 3), `choose_band` (rule 5) and `build_assignment`, all pure functions of
tables of numbers.

Units. "Physical" is the unit of the data. "Model units" are the cache's (`transform.to_model`).
"TokaMark units" are ``(x - SignalSpec.mean) / SignalSpec.std``. A floor, an error or a target is an
NRMSE in TokaMark units; a table of floors is one row per rung in rising size. NaN in a table is "not
measured" and is never read as 0. The cache is read only through `Codec.roundtrip`, so a codec of a later
plan, given the same `Codec` methods, works here unchanged.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from tokamak_foundation_model.ignite.mast import cache, transform
from tokamak_foundation_model.ignite.mast.tasks import SignalSpec, TaskSpec
from tokamak_foundation_model.ignite.mast.tokenizers import assign, fit, frames
from tokamak_foundation_model.ignite.mast.tokenizers.assign import Setting
from tokamak_foundation_model.ignite.mast.tokenizers.codecs import Codec

LADDER_TASKS = ("task_1-2", "task_1-3", "task_2-1", "task_3-1", "task_3-2", "task_4-5")
MAX_COST = 0.01                 # the pass-through cost (relative) a rung may have under rule 3
FIRST_BIN = 32                  # the spectral contrast counts the bins from here (31 kHz) up
_COST_EPS = 1e-12               # float fuzz of `passed / raw - 1 <= MAX_COST`, not a tolerance


# ---------------------------------------------------------------------------------------- passes

def values_per_frame(spec: SignalSpec, setting: Setting) -> int:
    """Values kept per 5 ms frame by `setting` for `spec`, an integer count (no unit).

    `frame_dct`: ``prod(values_shape) * K``. `linear`: the number of components, ``k[0]``. `band_dct`:
    ``P * sum(k)`` with `P` the probes (``values_shape[0]``) and `k` the values of each band. The names
    of plan 1c count too: `recoded` (one tokenizer per channel, `k` the values kept) counts like the
    signal's own kind (`assign.codec_kind`), `recoded_concat` (all channels in one) is ``k[0]``. Any
    other codec name raises `ValueError`.
    """
    codec = setting.codec
    if codec == "recoded_concat":
        return int(setting.k[0])
    if codec == "recoded":
        codec = assign.codec_kind(spec)
    if codec == "frame_dct":
        return math.prod(spec.values_shape) * int(setting.k[0])
    if codec == "linear":
        return int(setting.k[0])
    if codec == "band_dct":
        return int(spec.values_shape[0]) * int(sum(setting.k))
    raise ValueError(f"unknown codec {setting.codec!r} for {spec.name}")


def passes(task: TaskSpec, compress: Mapping[str, str]) -> list[tuple[str, int, dict[str, Setting]]]:
    """The scoring passes of a task: ``(domain, rung index, {output signal: Setting})``.

    `compress` maps a signal name to its ``"none"`` or ``"asinh"`` (`ChannelStats.compress`). The domains
    are the union of the outputs' `assign.domains` in first-seen order; in one domain there is a pass for
    each rung of the longest ladder. A signal without that domain keeps the setting of its first domain,
    and a signal whose ladder is shorter stays on its last rung. A pass whose settings equal an earlier
    pass's is left out. For a task with spectrogram outputs these are the passes of step a of rule 5.
    """
    ladders: dict[str, dict[str, tuple[Setting, ...]]] = {}
    first: dict[str, str] = {}
    order: list[str] = []
    for s in task.outputs:
        ds = assign.domains(s, compress[s.name])
        first[s.name] = ds[0]
        ladders[s.name] = {d: assign.rungs(s, d) for d in ds}
        order.extend(d for d in ds if d not in order)
    out: list[tuple[str, int, dict[str, Setting]]] = []
    for d in order:
        for i in range(max(len(by[d]) for by in ladders.values() if d in by)):
            settings = {}
            for name, by in ladders.items():
                rungs = by.get(d) or by[first[name]]
                settings[name] = rungs[min(i, len(rungs) - 1)]
            if all(settings != earlier for _, _, earlier in out):
                out.append((d, i, settings))
    return out


def band_passes(task: TaskSpec, k0: int) -> list[tuple[str, int, dict[str, Setting]]]:
    """Step b of rule 5: the passes ``("band-b", rung index, settings)`` with the lowest band at `k0` values
    and bands 1 to 7 at each common K of `assign.BAND_RUNGS`, for every output signal of `task`."""
    return [("band-b", i, {s.name: setting for s in task.outputs})
            for i, setting in enumerate(assign.band_rungs(k0))]


def signal_ladder(spec: SignalSpec, compress: str) -> list[Setting]:
    """Every rung of every domain of `spec` (`assign.domains` order, rising size in each); `compress` is
    its ``"none"`` or ``"asinh"``."""
    return [st for d in assign.domains(spec, compress) for st in assign.rungs(spec, d)]


def run_ladder(task: TaskSpec, passes: Sequence[tuple[str, int, Mapping[str, Setting]]],
               build: Callable[[dict[str, Setting]], dict[str, Codec]],
               score: Callable[[str, Mapping[str, Codec], str], Mapping[str, float] | None]) -> pd.DataFrame:
    """Score every pass; one row per distinct (signal, setting) of the passes, in the order of the passes.

    `build(settings)` gives the codecs of one pass. `score(kind, codecs, name)` returns the NRMSE in
    TokaMark units per signal, of kind ``"floor"`` (the true targets through the codecs) or ``"pass"``
    (the prediction of a direct model through them), or None if that kind is not measured;
    ``score("raw", {}, "raw")`` is asked once, for the direct model unchanged. The pass `name` is
    ``ladder-<domain>-<rung index>``. Columns: `signal`, `domain` (of the setting), `k` (K for a series,
    the components for a profile or map, the lowest band's values for a spectrogram), `values_per_frame`
    (`values_per_frame`), `floor`, `raw`, `passed`; NaN for what was not measured.
    """
    raw = score("raw", {}, "raw")
    by_name = {s.name: s for s in task.outputs}

    def pick(table: Mapping[str, float] | None, name: str) -> float:
        return math.nan if table is None or name not in table else float(table[name])

    rows, seen = [], set()
    for domain, i, settings in passes:
        name = f"ladder-{domain}-{i}"
        codecs = build(dict(settings))
        floors, passed = score("floor", codecs, name), score("pass", codecs, name)
        for signal in by_name:
            setting = settings[signal]
            if (signal, setting) in seen:
                continue
            seen.add((signal, setting))
            rows.append({"signal": signal, "domain": setting.domain, "k": int(setting.k[0]),
                         "values_per_frame": values_per_frame(by_name[signal], setting),
                         "floor": pick(floors, signal), "raw": pick(raw, signal), "passed": pick(passed, signal)})
    return pd.DataFrame(rows, columns=["signal", "domain", "k", "values_per_frame", "floor", "raw", "passed"])


# --------------------------------------------------------------------------- errors from the cache

def channel_errors(codec: Codec, spec: SignalSpec, z_model: np.ndarray,
                   mode: str = "unquantized") -> tuple[np.ndarray, np.ndarray]:
    """Per channel, the squared round-trip error and the number of valid samples of one shot.

    `z_model` is the cache's ``(*values_shape, T)`` in model units, NaN where not valid. The values go
    through `Codec.roundtrip` (so through `Codec.read`), and the error is taken in TokaMark units, at the
    samples where the target and the round-trip are both finite. A series counts whole frames only (the
    samples after the last whole frame are left out); a codec whose `domain` is ``"band"`` (a spectrogram,
    the class is not looked at) keeps all its columns. Returns ``(sum of squared errors, n valid samples)``,
    both ``(C,)`` with ``C = prod(values_shape)`` the channels flattened row-major: float64 in TokaMark
    units squared, and int64.
    """
    x = transform.to_physical(np.asarray(z_model), codec.stats)               # (*values_shape, T) physical
    n = x.shape[-1]
    if codec.domain != "band":
        n -= n % frames.frame_len(spec.dt)
    x = x[..., :n]
    channels = math.prod(spec.values_shape)
    if n == 0:
        return np.zeros(channels), np.zeros(channels, dtype=np.int64)
    rt = codec.roundtrip(x[None], mode)[0]
    diff = ((rt - spec.mean) / spec.std - (x - spec.mean) / spec.std).reshape(channels, n)
    ok = np.isfinite(x).reshape(channels, n) & np.isfinite(diff)
    return np.square(np.where(ok, diff, 0.0)).sum(axis=1), ok.sum(axis=1).astype(np.int64)


def shot_error(codec: Codec, spec: SignalSpec, z_model: np.ndarray, mode: str = "unquantized") -> float | None:
    """The RMSE of the round-trip over the valid samples of one shot, in TokaMark units.

    `z_model` is the cache's ``(*values_shape, T)`` in model units. ``sqrt(sum(sq) / sum(n))`` of
    `channel_errors`; None if there is no valid sample (no data is not an error of 0).
    """
    sq, n = channel_errors(codec, spec, z_model, mode)
    return None if n.sum() == 0 else float(math.sqrt(sq.sum() / n.sum()))


def band_contrast(codec: Codec, z_model: np.ndarray, first_bin: int = FIRST_BIN) -> tuple[float, float] | None:
    """The spectral contrast ratio of one shot's spectrogram after the round-trip, and its weight.

    `z_model` is the cache's ``(P, n_bins, T)`` in model units; `codec` is any codec whose `domain` is
    ``"band"`` (only `Codec` methods are used). The true frame-mean spectra (mean of linear power over
    each frame's valid columns) and their round-trip are converted to model units, the bins from
    `first_bin` on go to `gate.spectral_contrast_ratio` as ``(1, P, bins, frames)``, over the frames where
    both are finite everywhere. Returns ``(ratio, weight)``, the ratio of decoded to true contrast
    (dimensionless, 1 is exact) and ``weight = spectral_contrast_gt * P * frames`` (model units); None if
    no frame is valid or the true contrast is 0.
    """
    from tokamak_foundation_model.ignite.gate import spectral_contrast_ratio

    x = transform.to_physical(np.asarray(z_model), codec.stats)               # (P, n_bins, T) linear power
    p, _, t = x.shape
    f = frames.frame_index(t, codec.dt)
    _, first, lens = np.unique(f, return_index=True, return_counts=True)     # frames are contiguous
    ok = np.isfinite(x)
    counts = np.add.reduceat(ok, first, axis=-1)
    sums = np.add.reduceat(np.where(ok, x, 0.0), first, axis=-1)
    truth = np.divide(sums, counts, out=np.full(sums.shape, np.nan), where=counts > 0)   # (P, n_bins, F)
    decoded = codec.roundtrip(x[None])[0][..., first]                          # the same spectrum in a frame
    truth, decoded = (transform.to_model(a, codec.stats)[:, first_bin:, :] for a in (truth, decoded))
    keep = np.isfinite(truth).all(axis=(0, 1)) & np.isfinite(decoded).all(axis=(0, 1))
    if not keep.any():
        return None
    got = spectral_contrast_ratio(decoded[None][..., keep], truth[None][..., keep])
    ratio, gt = got["spectral_contrast_ratio"], got["spectral_contrast_gt"]
    if not (math.isfinite(ratio) and gt > 0):
        return None
    return float(ratio), float(gt * p * int(keep.sum()))


def pooled_contrast(per_shot: Sequence[tuple[float, float] | None]) -> float:
    """The weighted mean of the ratios of `band_contrast` over shots (dimensionless); NaN if none had data."""
    got = [r for r in per_shot if r is not None]
    weight = sum(w for _, w in got)
    return math.nan if not got or weight <= 0 else float(sum(r * w for r, w in got) / weight)


# ---------------------------------------------------------------- workers of the cache-side ladders

@functools.lru_cache(maxsize=4)
def _fit_and_stats(fit_dir: str, cache_root: str):
    return fit.load_fit(Path(fit_dir))[0], transform.load_stats(Path(cache_root) / "stats.npz")


@functools.lru_cache(maxsize=64)
def _rung_codecs(fit_dir: str, cache_root: str, spec: SignalSpec, band_low: int | None) -> list[Codec] | None:
    """The codecs of every rung of `signal_ladder(spec)` (or of step b at `band_low`), None if `spec` was
    not fitted. Built once in a process and kept."""
    fits, stats = _fit_and_stats(fit_dir, cache_root)
    f = fits[spec.name]
    if f.n_examples == 0:
        return None
    settings = (assign.band_rungs(band_low) if band_low is not None
                else signal_ladder(spec, stats[spec.name].compress))
    return [fit.build_codec(spec, stats[spec.name], f, st) for st in settings]


def input_errors_job(job: tuple) -> dict[str, np.ndarray | None]:
    """One shot of the input ladder, for `transform.map_shots`: ``job = (shot, cache_root, stats_sha256,
    fit_dir, specs)`` with `specs` a tuple of `SignalSpec`. Returns, per signal name, the `shot_error` (TokaMark
    units) of every setting of `signal_ladder`, ``(n_settings,)`` float64 with NaN for a shot without valid
    samples; None if the shot does not have the signal or the signal was not fitted."""
    shot, cache_root, sha, fit_dir, specs = job
    got = cache.read_shot(Path(cache_root) / "shots" / f"{shot}.h5", [s.name for s in specs], stats_sha256=sha)
    out: dict[str, np.ndarray | None] = {}
    for spec in specs:
        c = got[spec.name]
        rungs = None if c is None else _rung_codecs(str(fit_dir), str(cache_root), spec, None)
        if rungs is None:
            out[spec.name] = None
            continue
        errors = [shot_error(codec, spec, c.values) for codec in rungs]
        out[spec.name] = np.array([math.nan if e is None else e for e in errors])
    return out


def band_contrast_job(job: tuple) -> list[tuple[float, float] | None] | None:
    """One shot of the contrast of step b, for `transform.map_shots`: ``job = (shot, cache_root,
    stats_sha256, fit_dir, spec, k0)``. Returns `band_contrast` of every rung of ``assign.band_rungs(k0)``,
    or None if the shot does not have the signal or it was not fitted."""
    shot, cache_root, sha, fit_dir, spec, k0 = job
    c = cache.read_shot(Path(cache_root) / "shots" / f"{shot}.h5", [spec.name], stats_sha256=sha)[spec.name]
    rungs = None if c is None else _rung_codecs(str(fit_dir), str(cache_root), spec, k0)
    return None if rungs is None else [band_contrast(codec, c.values) for codec in rungs]


# ------------------------------------------------------------------------------ the choice of K

def choose(rows: pd.DataFrame, target: float, domains: Sequence[str],
           max_cost: float = MAX_COST) -> tuple[str, int, str]:
    """Rules 2 and 3 for one signal: ``(domain, rung index in that domain's rows, note)``.

    `rows` are the signal's rows of a ladder table (columns `signal`, `domain`, `floor`, `raw`,
    `passed`, rungs in rising size); `target` and the floors are NRMSE in TokaMark units; `domains` is
    the order of preference. A domain meets the target at `assign.choose_rung`; of the domains that
    meet it the smallest index wins, then the earlier domain. If none meets it, the cost of a rung is
    ``passed / raw - 1`` (relative); the smallest index with a cost at or under `max_cost` wins, then
    the earlier domain; if no cost is measured or none is that small, the last rung of the domain whose
    last floor is lowest. The note starts with ``meets <target> with <floor>`` or ``misses <target>:
    floor <floor>`` (four decimals), then ``pass-through cost <x.x%>`` or ``no pass-through measurement``
    for the chosen rung. `ValueError` naming "no ladder" if there is no row in any of `domains`.
    """
    per = [(d, rows[rows["domain"] == d].reset_index(drop=True)) for d in domains]
    per = [(d, sub) for d, sub in per if len(sub)]
    if not per:
        who = f" for {rows['signal'].iloc[0]}" if len(rows) else ""
        raise ValueError(f"no ladder{who}: no row in the domains {list(domains)}")
    met = []
    for order, (d, sub) in enumerate(per):
        i, ok = assign.choose_rung(sub["floor"].tolist(), target)
        if ok:
            met.append((i, order, d))

    def cost(sub: pd.DataFrame) -> pd.Series:
        raw = sub["raw"].where(sub["raw"] > 0)
        return sub["passed"] / raw - 1.0

    if met:
        i, order, domain = min(met)
        meets = True
    else:
        meets = False
        cheap = [(int(np.flatnonzero((cost(sub) <= max_cost + _COST_EPS).to_numpy())[0]), order, d)
                 for order, (d, sub) in enumerate(per) if (cost(sub) <= max_cost + _COST_EPS).any()]
        if cheap:
            i, order, domain = min(cheap)
        else:
            last = [(math.inf if math.isnan(sub["floor"].iloc[-1]) else float(sub["floor"].iloc[-1]), order, d)
                    for order, (d, sub) in enumerate(per)]
            _, order, domain = min(last)
            i = len(per[order][1]) - 1
    sub = per[order][1]
    floor_here = float(sub["floor"].iloc[i])
    head = f"meets {target:.4f} with {floor_here:.4f}" if meets else f"misses {target:.4f}: floor {floor_here:.4f}"
    c = cost(sub).iloc[i]
    tail = "no pass-through measurement" if math.isnan(c) else f"pass-through cost {100.0 * c:.1f}%"
    return domain, int(i), f"{head}; {tail}"


def _band_steps(rows: pd.DataFrame, signals: Sequence[str]) -> tuple[dict[str, tuple[pd.DataFrame, pd.DataFrame]], list[str]]:
    """Per usable signal the step-a and step-b rows (a sorted by `k0`, b by `k`), and the signals left out
    (named in `signals` or in `rows`, without rows of step a or of step b)."""
    names = list(dict.fromkeys([*rows["signal"], *signals]))
    usable, left = {}, []
    for name in names:
        g = rows[rows["signal"] == name]
        a = g[g["step"] == "a"].sort_values("k0", kind="stable").reset_index(drop=True)
        b = g[g["step"] == "b"].sort_values("k", kind="stable").reset_index(drop=True)
        if len(a) and len(b):
            usable[name] = (a, b)
        else:
            left.append(name)
    return usable, left


def _frame_oracle(name: str, a: pd.DataFrame) -> float:
    ref = a[a["k0"] == assign.BAND_WHOLE]["floor"]
    if not len(ref):
        raise ValueError(f"{name}: no step-a row with k0 = {assign.BAND_WHOLE}, the frame oracle")
    return float(ref.iloc[0])


def choose_low_band(rows: pd.DataFrame, tolerance: float = 0.02) -> int:
    """Step a of rule 5: the lowest band's number of values, from the rows of step a only.

    `rows` have columns `signal`, `step`, `k0`, `floor` (NRMSE in TokaMark units); step ``"a"`` has one row
    per `k0` of `assign.LOW_BAND_RUNGS`. Per signal the index is `assign.choose_within` the floors at the
    frame oracle (`k0` = 32); the largest index over the signals gives `k0`. `ValueError` naming "no
    ladder" if no signal has a row of step a.
    """
    best = None
    for name in dict.fromkeys(rows["signal"]):
        g = rows[(rows["signal"] == name) & (rows["step"] == "a")].sort_values("k0", kind="stable").reset_index(drop=True)
        if not len(g):
            continue
        i = assign.choose_within(g["floor"].tolist(), _frame_oracle(name, g), tolerance)
        if best is None or i > best[0]:
            best = (i, int(g["k0"].iloc[i]))
    if best is None:
        raise ValueError("no ladder: no row of step a")
    return best[1]


def choose_band(rows: pd.DataFrame, tolerance: float = 0.02, min_contrast: float = 0.9,
                signals: Sequence[str] = ()) -> tuple[tuple[int, ...], str]:
    """Rule 5 for the spectrograms: the values per band ``(k0, k, k, k, k, k, k, k)`` and a note.

    `rows` have columns `signal`, `step`, `k0`, `k`, `floor` (NRMSE in TokaMark units) and `contrast`
    (dimensionless, 1 is exact): step ``"a"`` has one row per `k0`, step ``"b"`` one row per common `k`
    of bands 1 to 7. Per signal the step-a index is `assign.choose_within` the floors at `k0` = 32 (the
    frame oracle); the largest index over the signals gives `k0`. Per signal the step-b index is the larger
    of `assign.choose_within` the signal's step-a floor at the chosen `k0` and the first index whose
    `contrast` is at least `min_contrast` (the last index if none is); the largest over the signals gives
    `k`. A signal without rows of both steps is left out of the choice and named in the note, as is each of
    `signals` that has no rows at all. The note gives each signal's frame oracle, its floors and contrast
    at the choice (four decimals). `ValueError` naming "no ladder" if no signal is left.
    """
    usable, left = _band_steps(rows, signals)
    if not usable:
        raise ValueError(f"no ladder for the spectrograms {left or '(no rows)'}: rows of step a and step b are needed")
    k0 = choose_low_band(rows[rows["signal"].isin(usable)], tolerance)
    pick: dict[str, tuple[int, int, bool]] = {}              # name -> (index, K at it, contrast decides)
    for name, (a, b) in usable.items():
        reference = float(a[a["k0"] == k0]["floor"].iloc[0])
        by_floor = assign.choose_within(b["floor"].tolist(), reference, tolerance)
        enough = np.flatnonzero((b["contrast"] >= min_contrast).to_numpy())
        by_contrast = int(enough[0]) if len(enough) else len(b) - 1
        i = max(by_floor, by_contrast)
        pick[name] = (i, int(b["k"].iloc[i]), by_contrast > by_floor)
    k = max(v[1] for v in pick.values())
    parts = []
    for name, (a, b) in usable.items():
        fa, fb, contrast = (float(a[a["k0"] == k0]["floor"].iloc[0]), float(b[b["k"] == k]["floor"].iloc[0]),
                            float(b[b["k"] == k]["contrast"].iloc[0]))
        deciding = ", contrast sets K" if pick[name][2] else ""
        parts.append(f"{name}: frame oracle {_frame_oracle(name, a):.4f}, step a floor {fa:.4f} at k0 {k0}, "
                     f"step b floor {fb:.4f} at k {k}, contrast {contrast:.4f}{deciding}")
    if left:
        parts.append("left out, no rows: " + ", ".join(left))
    return (k0,) + (k,) * (assign.N_BANDS - 1), "; ".join(parts)


def build_assignment(signals: Sequence[SignalSpec], compress: Mapping[str, str], ladders: pd.DataFrame,
                     targets: Mapping[str, float | None]) -> tuple[dict[str, Setting], dict[str, str]]:
    """The assignment of every signal, and one note per signal.

    `ladders` is the concatenation of the ladder tables (columns `signal`, `domain`, `k`, `floor`, `raw`,
    `passed`, and `step`, `k0`, `contrast` for the spectrograms); `targets` the NRMSE target of each signal in
    TokaMark units (None for a spectrogram). A `frame_dct` signal with one sample per frame needs no ladder
    and gets ``Setting("frame_dct", (1,))`` ("K = 1 is exact"); the spectrograms share the result of
    `choose_band`; every other signal goes through `choose` over its `assign.domains`. `ValueError` names a
    signal that needs a ladder and has no rows, or has no target.
    """
    out: dict[str, Setting] = {}
    notes: dict[str, str] = {}
    bands = [s for s in signals if assign.codec_kind(s) == "band_dct"]
    for s in signals:
        kind = assign.codec_kind(s)
        if kind == "band_dct":
            continue
        if kind == "frame_dct" and frames.frame_len(s.dt) == 1:
            out[s.name], notes[s.name] = Setting("frame_dct", (1,)), "one sample per frame: K = 1 is exact"
            continue
        rows = ladders[ladders["signal"] == s.name]
        if not len(rows):
            raise ValueError(f"signal {s.name} has no rows in the ladders")
        if targets.get(s.name) is None:
            raise ValueError(f"signal {s.name} has no target")
        domain, i, note = choose(rows, float(targets[s.name]), assign.domains(s, compress[s.name]))
        setting = assign.rungs(s, domain)[i]
        stored = rows[rows["domain"] == domain].reset_index(drop=True)["k"].iloc[i]
        if int(stored) != int(setting.k[0]):
            raise ValueError(f"signal {s.name}: row {i} of {domain} is k = {int(stored)} but the rung is "
                             f"{setting.k[0]}; the ladder file was made for other rungs")
        out[s.name], notes[s.name] = setting, note
    if bands:
        k, note = choose_band(ladders[ladders["signal"].isin([s.name for s in bands])],
                              signals=[s.name for s in bands])
        for s in bands:
            out[s.name], notes[s.name] = Setting("band_dct", k, "band"), note
    return {s.name: out[s.name] for s in signals}, {s.name: notes[s.name] for s in signals}
