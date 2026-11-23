"""The stage-1 gate: one verdict for every (task, output signal) pair, from stored floors and the cache.

A tokenizer setting passes the gate for a signal when (1) it reads back the true targets well enough
(its fidelity target, a fifth of the best published score of the task), and (2) it does not lose to the
signal's own mean: a setting that keeps more values per frame than the reference assignment must score a
floor below the own-mean floor, or the extra values buy nothing. `verdict` is that rule, `gate_rows`
applies it to a table of numbers, `to_markdown` prints the table.

The floors are not computed here. They are the per-signal NRMSE (TokaMark units) that
``scripts/mast/run_tokenizer.py score`` stored for each task, in three read-outs (`unquantized`, `ranged`,
`nearest`). The own-mean floor, the ``ref`` read-out, is the `unquantized` score of the reference
assignment: ``run_tokenizer.py assign --reference --name ref`` writes it to ``<stage1>/assignments/ref.json``
and ``run_tokenizer.py score --assignment <stage1>/assignments/ref.json --mode unquantized`` stores it.
A floor that is missing or stale is None, and a verdict that needs it is `not scored`; a missing number is
never compared as a number and never printed as 0.

The cache-side columns are measured here from the cached values of a split's shots: how many of the bins
of each value's grid the encoded values reach (`bin_usage`: a value that sits in one bin carries no
information), the channel with the largest round-trip error (`channel_errors`, defined in `ladder` and
used here unchanged, so that the sample rule exists once), and, for spectrograms, the spectral contrast
(`ladder.band_contrast`). `cache_side_job` is the worker of `transform.map_shots` for that pass.

Units. "Physical" is the unit of the data. "Model units" are the cache's (`transform.to_model`). "TokaMark
units" are ``(x - SignalSpec.mean) / SignalSpec.std``. A floor, an error or a target is an NRMSE or RMSE in
TokaMark units; ``values_per_frame`` is a count of values per 5 ms frame.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from tokamak_foundation_model.ignite.mast import cache, tasks, transform
from tokamak_foundation_model.ignite.mast.tasks import SignalSpec
from tokamak_foundation_model.ignite.mast.tokenizers import assign, fit, frames
from tokamak_foundation_model.ignite.mast.tokenizers.assign import Setting
from tokamak_foundation_model.ignite.mast.tokenizers.codecs import Codec
# `channel_errors` is `ladder`'s, used here and reachable as `gate.channel_errors`: the sample rule exists once.
from tokamak_foundation_model.ignite.mast.tokenizers.ladder import (
    band_contrast, channel_errors, pooled_contrast, values_per_frame)

READS = ("unquantized", "ranged", "nearest", "ref")      # the floors the gate reads; "ref" is the own-mean one
FIDELITY = 0.2                                           # a setting must read back to this part of the best published score
COLUMNS = ("task", "signal", "codec", "k", "domain", "values_per_frame", "target", "unquantized", "ranged",
           "nearest", "own_mean", "bins_used_min", "worst_channel", "worst_channel_error", "contrast", "verdict")
_SCORE_COLUMNS = ("unquantized", "ranged", "nearest", "own_mean")
_SIDE_COLUMNS = ("bins_used_min", "worst_channel", "worst_channel_error", "contrast")
_INTEGER_COLUMNS = ("k", "values_per_frame", "worst_channel")


def _num(v) -> float | None:
    """`v` as a float, or None for None and NaN: a missing number is never compared as a number."""
    return None if v is None or (isinstance(v, float | np.floating) and np.isnan(v)) else float(v)


# ------------------------------------------------------------------------------------ bin usage

def _used_bins(index: np.ndarray, n_bins: int) -> np.ndarray:
    """Which bins each position reaches: `index` ``(N, n_tok, n_val)`` integer bin indices (-1 missing) to a
    bool array ``(n_tok, n_val, n_bins)``."""
    used = np.zeros(index.shape[1:] + (n_bins,), dtype=bool)
    m, t, v = np.nonzero(index >= 0)
    used[t, v, index[m, t, v]] = True
    return used


def _usage(used: np.ndarray) -> np.ndarray:
    """The fraction of bins used, ``(n_tok, n_val)`` float64, from `_used_bins`'s ``(n_tok, n_val, n_bins)``;
    NaN where no bin is used (no example)."""
    frac = used.sum(axis=-1) / used.shape[-1]
    return np.where(used.any(axis=-1), frac, np.nan)


def bin_usage(index: np.ndarray, n_bins: int) -> np.ndarray:
    """The fraction of the `n_bins` bins that each position uses.

    `index` is ``(N, n_tok, n_val)`` integer: the bin of each of `N` examples at each position, -1 for a
    missing one. Returns ``(n_tok, n_val)`` float64 in ``[1 / n_bins, 1]``, NaN for a position without an
    example (no data is not a usage of 0).
    """
    return _usage(_used_bins(np.asarray(index), n_bins))


# ----------------------------------------------------------------------------------- cache side

def empty_side(n_skipped: int = 0) -> dict:
    """The cache-side record of a signal with no shot: ``n_shots`` 0, ``n_skipped`` as counted, None elsewhere."""
    return {"n_shots": 0, "n_skipped": int(n_skipped), "bins_used_min": None, "worst_channel": None,
            "worst_channel_error": None, "contrast": None}


def _kept_positions(codec: Codec) -> np.ndarray | None:
    """Which ``(n_tok, n_val)`` positions a band codec keeps (bool), from its public ``k`` (one value per band,
    tiled over the probes, token order ``probe * n_bands + band``); None for any other codec."""
    if codec.domain != "band":
        return None
    return np.arange(codec.n_val)[None, :] < np.tile(codec.k, int(codec.values_shape[0]))[:, None]


def shot_side(codec: Codec, spec: SignalSpec, z_model: np.ndarray) -> dict:
    """What one shot adds to the cache-side record of a signal.

    `z_model` is the cache's ``(*values_shape, T)`` in model units, NaN where not valid. Returns
    ``sq`` and ``n`` (`channel_errors`: per channel, TokaMark units squared and a count, both ``(C,)``),
    ``used`` (bool ``(n_tok, n_val, n_bins)``, the bins the shot's encoded values reach, or None without
    grids) and ``contrast`` (``band_contrast``'s ``(ratio, weight)`` or None; only for a band codec).
    The encoded values are not read as tokens: the grid only counts which bin each falls in. A band codec's
    dropped positions (values from ``k[band]`` on, which it encodes as 0) are marked as no example.
    """
    sq, n = channel_errors(codec, spec, z_model)
    used = None
    if codec.grids is not None:
        x = transform.to_physical(np.asarray(z_model), codec.stats)           # (*values_shape, T) physical
        if codec.domain != "band":
            x = x[..., : x.shape[-1] - x.shape[-1] % frames.frame_len(spec.dt)]
        if x.shape[-1]:
            index = codec.grids.index(codec.encode(x[None])[0])               # (F, n_tok, n_val), -1 missing
            keep = _kept_positions(codec)
            if keep is not None:
                index = np.where(keep[None], index, -1)
            used = _used_bins(index, codec.grids.n_bins)
        else:
            used = np.zeros((codec.n_tok, codec.n_val, codec.grids.n_bins), dtype=bool)
    contrast = band_contrast(codec, z_model) if codec.domain == "band" else None
    return {"sq": sq, "n": n, "used": used, "contrast": contrast}


def combine_sides(codec: Codec, parts: Sequence[dict], n_skipped: int) -> dict:
    """The cache-side record from the `shot_side` of every shot that has the signal (`n_skipped` shots do not).

    Keys: ``n_shots``, ``n_skipped``, ``bins_used_min`` (the smallest `bin_usage` over the positions that have
    an example, all shots together; None without grids or without an example), ``worst_channel`` (flat index
    of the channel with the largest RMSE) and ``worst_channel_error`` (its RMSE over the valid samples of all
    shots, TokaMark units; None if no channel has a valid sample), ``contrast`` (the weighted mean of the
    shots' spectral contrast ratio, dimensionless; None unless the codec is a band codec with data).
    """
    if not parts:
        return empty_side(n_skipped)
    out = empty_side(n_skipped)
    out["n_shots"] = len(parts)
    sq, n = (np.sum([p[key] for p in parts], axis=0) for key in ("sq", "n"))
    if n.sum() > 0:
        rmse = np.where(n > 0, np.sqrt(sq / np.maximum(n, 1)), np.nan)
        out["worst_channel"] = int(np.nanargmax(rmse))
        out["worst_channel_error"] = float(rmse[out["worst_channel"]])
    if codec.grids is not None and parts[0]["used"] is not None:
        usage = _usage(np.any([p["used"] for p in parts], axis=0))
        if np.isfinite(usage).any():
            out["bins_used_min"] = float(np.nanmin(usage))
    if codec.domain == "band":
        pooled = pooled_contrast([p["contrast"] for p in parts])
        out["contrast"] = None if math.isnan(pooled) else pooled
    return out


def cache_side(codec: Codec, spec: SignalSpec, shots_values: Iterable[np.ndarray | None]) -> dict:
    """The cache-side record of one signal over shots: see `combine_sides` for the keys.

    `shots_values` yields each shot's cached ``(*values_shape, T)`` in model units, or None for a shot
    that does not have the signal (counted in ``n_skipped``). With no shot at all: ``n_shots`` 0 and None
    elsewhere.
    """
    parts, skipped = [], 0
    for z in shots_values:
        if z is None:
            skipped += 1
        else:
            parts.append(shot_side(codec, spec, z))
    return combine_sides(codec, parts, skipped)


@functools.lru_cache(maxsize=4)
def _fit_and_stats(fit_dir: str, cache_root: str):
    return fit.load_fit(Path(fit_dir))[0], transform.load_stats(Path(cache_root) / "stats.npz")


@functools.lru_cache(maxsize=64)
def _codec_of(fit_dir: str, cache_root: str, spec: SignalSpec, setting: Setting) -> Codec | None:
    """The codec of `setting` for `spec` from the fit, or None if the signal has no fitted data. Built once
    in a process and kept."""
    fits, stats = _fit_and_stats(fit_dir, cache_root)
    f = fits.get(spec.name)
    return None if f is None or f.n_examples == 0 else fit.build_codec(spec, stats[spec.name], f, setting)


def cache_side_job(job: tuple) -> dict[str, dict | None]:
    """One shot of the cache-side pass, for `transform.map_shots`: ``job = (shot, cache_root, stats_sha256,
    fit_dir, specs, settings)`` with `specs` a tuple of `SignalSpec` and `settings` the `Setting` of each,
    in the same order. Returns, per signal name, `shot_side` of the shot's cached values (model units), or
    None if the shot does not have the signal or the signal was not fitted."""
    shot, cache_root, sha, fit_dir, specs, settings = job
    got = cache.read_shot(Path(cache_root) / "shots" / f"{shot}.h5", [s.name for s in specs], stats_sha256=sha)
    out: dict[str, dict | None] = {}
    for spec, setting in zip(specs, settings, strict=True):
        c = got[spec.name]
        codec = None if c is None else _codec_of(str(fit_dir), str(cache_root), spec, setting)
        out[spec.name] = None if codec is None else shot_side(codec, spec, c.values)
    return out


# ------------------------------------------------------------------------------------- the verdict

def verdict(*, values: int, reference_values: int, floor: float | None, ranged: float | None,
            own_mean: float | None, target: float | None) -> str:
    """The verdict of one (task, signal) pair. `values` and `reference_values` are counts of values per frame
    of the setting and of the reference assignment; `floor` (the `unquantized` read-out), `ranged`,
    `own_mean` (the reference's floor) and `target` are NRMSE in TokaMark units, None if missing.

    In this order: ``not scored`` if `floor` or `ranged` is None, or if `values` is above `reference_values`
    and `own_mean` is None; ``blocks: does not beat its own mean (<floor> against <own_mean>)`` if `values`
    is above `reference_values` and `floor` is not below `own_mean` (equal is not better);
    ``pass (no fidelity target)`` if `target` is None; ``pass`` if `ranged` is at most `target`; else
    ``pass, misses the fidelity target (<ranged> against <target>)``. Numbers have four decimals.
    """
    floor, ranged, own_mean, target = (_num(v) for v in (floor, ranged, own_mean, target))
    more = values > reference_values
    if floor is None or ranged is None or (more and own_mean is None):
        return "not scored"
    if more and not floor < own_mean:
        return f"blocks: does not beat its own mean ({floor:.4f} against {own_mean:.4f})"
    if target is None:
        return "pass (no fidelity target)"
    if ranged <= target:
        return "pass"
    return f"pass, misses the fidelity target ({ranged:.4f} against {target:.4f})"


def _k_cell(setting: Setting) -> int | str:
    """The ``k`` column: the K of `frame_dct`, the number of values of `linear` (an int), and every band's K
    joined by ``/`` for `band_dct`."""
    return "/".join(str(int(v)) for v in setting.k) if setting.codec == "band_dct" else int(setting.k[0])


def gate_rows(signals_by_task: Mapping[str, Sequence[SignalSpec]], assignment: Mapping[str, Setting],
              floors: Mapping[tuple[str, str], Mapping[str, float] | None], best_published: Mapping[str, float],
              cache_side: Mapping[str, dict]) -> pd.DataFrame:
    """One row per (task, output signal), with the columns `COLUMNS`.

    `floors[(task, read)]` is the per-signal NRMSE (TokaMark units) of the task's score for `read` in
    `READS`, or None if that score is not stored or is stale; a key that is absent counts as None.
    `cache_side[signal]` is a record of `cache_side` (an absent signal has no data). `target` is
    ``FIDELITY * best_published[task]`` (an NRMSE), NaN for the spectrograms; `own_mean` is the ``ref``
    floor. A missing number is NaN in the table and None for `verdict`. `ValueError` for a signal with no
    setting in `assignment`.
    """
    rows = []
    for task, specs in signals_by_task.items():
        stored = {read: floors.get((task, read)) for read in READS}
        for spec in specs:
            if spec.name not in assignment:
                raise ValueError(f"the assignment has no setting for signal {spec.name} (task {task})")
            setting = assignment[spec.name]
            n = values_per_frame(spec, setting)
            ref = values_per_frame(spec, assign.reference_assignment([spec])[spec.name])
            target = None if spec.name in tasks.SPECTROGRAM_SIGNALS else FIDELITY * best_published[task]
            got = {read: _num((stored[read] or {}).get(spec.name)) for read in READS}
            side = cache_side.get(spec.name) or {}
            rows.append({
                "task": task, "signal": spec.name, "codec": setting.codec, "k": _k_cell(setting),
                "domain": setting.domain, "values_per_frame": n, "target": _nan(target),
                "unquantized": _nan(got["unquantized"]), "ranged": _nan(got["ranged"]),
                "nearest": _nan(got["nearest"]), "own_mean": _nan(got["ref"]),
                **{c: _nan(side.get(c)) for c in _SIDE_COLUMNS},
                "verdict": verdict(values=n, reference_values=ref, floor=got["unquantized"], ranged=got["ranged"],
                                   own_mean=got["ref"], target=target)})
    return pd.DataFrame(rows, columns=list(COLUMNS))


def _nan(v) -> float:
    """`v` as a float, NaN for None."""
    return math.nan if v is None else float(v)


# ------------------------------------------------------------------------------------- the table

def _cell(column: str, v) -> str:
    """One cell: four decimals, integer columns as integers, a missing number as `not scored` (the score
    columns), `no target` (the target) or `no data` (the cache-side columns); never as 0."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ("not scored" if column in _SCORE_COLUMNS else "no target" if column == "target"
                else "no data" if column in _SIDE_COLUMNS else "")
    if isinstance(v, str):
        return v
    return str(int(v)) if column in _INTEGER_COLUMNS else f"{float(v):.4f}"


def to_markdown(table: pd.DataFrame) -> str:
    """`table` (the columns of `gate_rows`) as a Markdown table, one line per row (see `_cell`)."""
    columns = list(table.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(" --- " for _ in columns) + "|"]
    for row in table.to_dict("records"):
        lines.append("| " + " | ".join(_cell(c, row[c]) for c in columns) + " |")
    return "\n".join(lines)
