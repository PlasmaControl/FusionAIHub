"""One shot, end to end: masks, tracks, transients, heuristics, text.

Every other module here does one thing to one array. This is the only place
that knows the ORDER - which channels a shot can spend the GPU on, what is
computed while the probabilities are still in memory, which failure costs a
step and which costs the shot - and it exists so that `run.py` and the SLURM
script it will be driven from cannot drift on any of it. `process_shot` is
the whole of a shot's work; `run.py`'s `events` stage is a loop around it.

**Guarded step by step.** A step that raises records `skipped[step]` and the
shot goes on: a corpus without `bes` (67% of shots), a `co2` channel whose
digitiser gapped, a heuristic that meets a record it cannot read - none of
them is a reason to lose the seven other channels or the sawtooth train.
`error` is set for two things only: an I/O failure on the corpus file, because
then there is nothing to process, and a failed write, because then what was
processed is not on disk. A skip is therefore ORDINARY;
`skipped` is a record of what was not looked at, not a degraded run, which
is why `ShotResult.status` stays `ok` while it fills up.

**The probabilities never reach disk.** `masks` stores a boolean mask at
`PROB_THRESHOLD` and two summaries of it, so `tracks.tracks_for_block` read
back off a stored block has no probabilities to weigh - its `conf` and
`mean_prob` come back NaN by construction (task L4's note). Every descriptor
this module writes is therefore computed HERE, on the float32 map the
network returned, before it is packed; the `*_for_block` helpers are for
offline re-analysis of a file whose probabilities are already gone.

**Independent transient and ELM references.** TokEye keeps one mask
reference: the magnetics wide block with most transient activity, ties by
name. Those rows are class-agnostic `transient` points. The ELM clock reads
filterscopes 0-7 directly, preferring the first finite channel by index.
`ShotResult.elm_reference` identifies that filterscope. The clock's points
and quiet intervals share its finite coverage and run even without masks.

**What the corpus cannot serve.** `ip` is not a corpus group (it is archive-
and fdp-served; see `features/namespace.py`), so `actuator_intervals` gets
no `ip` and never claims `nbi_counter`, and `qh_candidates` gets an empty
Ip flat-top and therefore claims nothing. Both are recorded as skips rather
than left to look like an absence of the phenomenon. `betan` is not a corpus
group either, so an L-H transition's `attrs["betan"]` is `None`.
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ..ae.labels import PROB_THRESHOLD
from ..ae.transform import STD_EPS
from ..config import Paths
from ..labels.store import append_index
from . import (
    channels,
    coverage,
    databases,
    heuristics,
    masks,
    schema,
    text_weak,
    tracks,
    transients,
)
from .lexicon import Lexicon
from .schema import Event
from .unet import CHECKPOINT_SHA256

#: How the spectrogram is standardised before the network sees it.
#: `record` is the z-score over the whole record, which is what
#: `ae_dataset.py` fed the network in training and is therefore the default.
NORMS = ("record", "plasma")

#: The diagnostics used to select the class-agnostic mask reference; `mirnov` is `mhr`'s stand-in on the 14% of shots
#: that have no `mhr`.
MAGNETICS_DIAGS = ("mhr", "mirnov")

#: The identity of a masks-file block for `events_index` and for the index
#: keys of `tracks.cooccurrence`.
CO_KEY = "{diag}:{channel}:{pass_name}"

#: Canonical actuator feature -> `(corpus group, scale to canonical units,
#: how to reduce the channels)`. `heuristics._ACTUATORS` names features in
#: `features/namespace.py`'s terms and thresholds them in its units, so this
#: is where a corpus group becomes one:
#:
#: * the eight beams are summed and taken from W to kW, and the eight
#:   torques summed as they are (`resolve_corpus.SCALE_TO_CANONICAL`);
#: * `rmp` has no namespace entry and `heuristics.RMP_ON_KA` expects kA,
#:   while the corpus group is coil current in AMPS - MEASURED +/-1170 A on
#:   shot 198658, the same family and magnitude as `i_coil`'s +/-1162 - so
#:   1e-3, and all twelve coils are kept because "the coils are on" is a
#:   claim about the set of them;
#: * `gas` is namespace's `gas_raw#0` (PTDATA `gasa`) and is channel 0
#:   ALONE, not the eleven valves' loudest. MEASURED on 198658: channel 3
#:   idles at 0.675-0.876 V for the whole 105 s record, so a maximum over
#:   the group is over `GAS_ON_V` everywhere and `gas_on` comes back as one
#:   interval covering the record; channel 0 runs 0.062-6.42 V and is over
#:   the threshold for 1.1% of it, which is a gas puff.
ACTUATOR_GROUPS: dict[str, tuple[str, float, str]] = {
    "pinj_total": ("pinj", 1e-3, "sum"),
    "tinj_total": ("tinj", 1.0, "sum"),
    "ech_power_total": ("ech_power", 1.0, "sum"),
    "rmp": ("rmp", 1e-3, "all"),
    "gas": ("gas_raw", 1.0, "first"),
}

#: Why `nbi_counter` is never claimed off the corpus alone.
NO_IP = (
    "`ip` is not a corpus group (archive- and fdp-served; see "
    "features/namespace.py), and counter-injection is a comparison of the "
    "torque's sign with the current's"
)

#: Why `qh_proxy` claims nothing off the corpus alone. Same missing signal,
#: a different consequence: the proxy is an intersection and one of the
#: three sets it intersects cannot be computed.
NO_FLATTOP = (
    "`ip` is not a corpus group (archive- and fdp-served; see "
    "features/namespace.py), so there is no flat-top interval to intersect "
    "an EHO with and the proxy claims nothing"
)


@dataclass
class ShotResult:
    """What one shot's run produced, and what it did not.

    `skipped` maps a step to why it did not happen - `"channel bes:26"` ->
    `"group absent"`, `"sawtooth"` -> `"KeyError: no group 'ece'"` - and is
    the answer to "was there no ELM here, or did nobody look". `error` is
    non-empty for two failures only: the corpus file could not be read, or
    the results could not be written.
    """

    shot: int
    n_blocks: int = 0
    n_tracks: int = 0
    n_elms: int = 0
    n_sawteeth: int = 0
    n_lh: int = 0
    n_text: int = 0
    n_events: int = 0
    by_source: dict[str, int] = field(default_factory=dict)
    elm_reference: str = ""
    elapsed_s: float = 0.0
    skipped: dict[str, str] = field(default_factory=dict)
    error: str = ""
    #: One record per curated table that NAMED this shot - see
    #: `databases_block`. They are written to the shot's sources file with
    #: every other source's; this field is what the run JSON reports.
    #: Empty is the ordinary case and means no table names the shot, which
    #: is not the same as a table finding nothing in it.
    database_sources: list[dict] = field(default_factory=list)

    @property
    def status(self) -> str:
        """`error` or `ok`; a skip is not a failed shot.

        Deliberately two values and not three. Every real shot skips
        something - `bes` is absent on 67% of them, `nbi_counter` on all of
        them - so a `partial` status would be every shot's status and would
        say nothing. What did not run is in `skipped`, per step, with its
        reason.
        """
        return "error" if self.error else "ok"

    def as_row(self) -> dict:
        """The summary row `run.py` logs and writes into the run's JSON."""
        return {
            "shot": int(self.shot),
            "status": self.status,
            "n_blocks": self.n_blocks,
            "n_tracks": self.n_tracks,
            "n_elms": self.n_elms,
            "n_sawteeth": self.n_sawteeth,
            "n_lh": self.n_lh,
            "n_text": self.n_text,
            "n_events": self.n_events,
            "by_source": dict(sorted(self.by_source.items())),
            "elm_reference": self.elm_reference,
            "seconds": round(self.elapsed_s, 2),
            "skipped": dict(sorted(self.skipped.items())),
            "error": self.error,
        }

    def line(self) -> str:
        """One line per shot, for the terminal."""
        if self.error:
            return f"{self.shot}: ERROR {self.error}"
        by_source = " ".join(
            f"{s}={n}" for s, n in sorted(self.by_source.items())
        )
        return (
            f"{self.shot}: {self.n_blocks} blocks, {self.n_events} events "
            f"[{by_source}] {len(self.skipped)} skipped "
            f"in {self.elapsed_s:.1f}s"
        )


@dataclass(eq=False)
class _BlockRun:
    """One `(diag, channel, pass)` after the big arrays have been let go.

    `eq=False` because it holds arrays. What survives the block is what a
    later step needs and nothing else: the seven storable keys (packed, a
    couple of MB), the tracks measured on the probabilities, and the column
    activity trace - so a shot's peak memory is one block's `(512, T)`
    float32 maps and not the whole plan's.
    """

    diag: str
    channel: int
    pass_name: str
    prefix: str
    arrays: dict[str, Any]
    tracks: list[tracks.Track]
    activity: np.ndarray
    bursts: list[transients.Burst]
    t_s: np.ndarray
    t_cov: tuple[float, float]

    @property
    def co_key(self) -> str:
        return CO_KEY.format(diag=self.diag, channel=self.channel,
                             pass_name=self.pass_name)


def _cause(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:200]


def databases_block(shot: int, paths: Paths) -> tuple[list[Event], list[dict]]:
    """This shot's curated-table rows, and which tables named it.

    One function so the events stage's hook is one guarded call. What makes
    it different from every other step: a table that does not list this
    shot contributes NOTHING - no event row and no source record. Nobody
    looked, and `status="ok", n_events=0` against every table for every one
    of 16,909 corpus shots would be a coverage claim no author made.

    Cheap enough to sit in the per-shot path (a CSV read, cached per
    process), but it is knowledge about a shot we may hold no signals for,
    so `run.py events --databases-only` ingests a table over any shot list
    without the corpus or the network - that, and not this call site, is
    how a table reaches a shot whose corpus file is missing.

    The records it returns go to `events/<shot>_sources.parquet` beside
    every other source's, through `coverage.source_records`: `status="ok"`,
    `reason=""` (an ok row carries none - `schema._source_row` enforces
    that), `n_events`, and coverage NaN, which is the signal. They are also
    kept on `ShotResult.database_sources` for the run JSON.
    """
    return databases.events_for_shot(shot, root=paths.label_tables)


def _read_group(corpus_file, diag: str, *, stop: int | None = None):
    """`(t_s, y)` of a whole corpus group, exactly as it is on disk.

    Non-finite samples are NOT stripped, unlike `masks.read_waveform`: the
    heuristics all turn a non-finite bin into "no change" (`sawtooth_events`
    via `envelope`, `lh_transitions` via `_usable_span`), a filterscope
    record's NaN head and tail are part of what they are written to handle,
    and stripping on channel 0's behalf would move every other channel's
    samples. This is also what `scripts/labelmaker/sawtooth_reference_check.py`
    reads, so the pipeline's sawtooth count is the one that script pins.

    `stop` reads only the first `stop` channels - D-alpha is filterscopes
    0-7 and the other 96 channels are NaN on every shot, and the group is
    104 rows wide.
    """
    with h5py.File(corpus_file, "r", locking=False) as f:
        if diag not in f or "ydata" not in f[diag]:
            raise KeyError(f"no group {diag!r}")
        dset = f[diag]["ydata"]
        if dset.ndim != 2 or dset.shape[-1] < channels.MIN_SAMPLES:
            raise ValueError(f"{diag}: absent on this shot (ydata is {dset.shape})")
        if stop is not None and dset.shape[0] < stop:
            raise ValueError(
                f"{diag}: {dset.shape[0]} channels, {stop} wanted"
            )
        y = np.asarray(dset[:stop, :] if stop else dset[:, :], dtype=np.float32)
        t_s = np.asarray(f[diag]["xdata"][:], dtype=np.float64)
    if t_s.size != y.shape[1]:
        raise ValueError(
            f"{diag}: xdata has {t_s.size} samples, ydata {y.shape[1]}"
        )
    return t_s, y


def _actuator_features(corpus_file, skipped: dict[str, str]):
    """The canonical actuator features this corpus file can serve.

    `{name: (t_s, y)}` in `heuristics.actuator_intervals`' terms. A group
    that is absent is left out and recorded, because a shot without RMP
    coils is the common case and not an error.
    """
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (diag, scale, reduce) in ACTUATOR_GROUPS.items():
        try:
            t_s, y = _read_group(corpus_file, diag,
                                 stop=1 if reduce == "first" else None)
        except (KeyError, ValueError, OSError) as exc:
            skipped[f"actuator {name}"] = _cause(exc)
            continue
        y = np.asarray(y, dtype=np.float64) * float(scale)
        if reduce == "sum":
            y = y.sum(axis=0)
        elif reduce == "first":
            y = y[0]
        out[name] = (t_s, y)
    return out


def _norm_window(corpus_file, diags: Sequence[str]) -> tuple[float, float] | None:
    """The intersection of the fast groups' coverage, or None.

    `--norm plasma` wants the statistics of the part of the record where
    there is a plasma, and the honest way to say that is with `ip` - which
    is not in the corpus (`NO_IP`). This is the available proxy: the window
    every planned diagnostic's digitiser was running in, which excludes the
    seconds of pre-shot and post-shot record that the fast groups carry (
    `mirnov` on 198658 is live for 6 s of a 16.4 s record) and does NOT
    exclude a diagnostic that ran through a fizzle. It is a LIMITATION and
    named as one: the pilot (task 12) decides between the two norms by
    measurement, and this is the version of `plasma` it gets to judge.

    Reads one channel row per diagnostic (`channels.coverage`), and only
    when `plasma` was asked for.
    """
    spans = [channels.coverage(corpus_file, diag) for diag in sorted(set(diags))]
    lo = max((t0 for t0, _ in spans if np.isfinite(t0)), default=None)
    hi = min((t1 for _, t1 in spans if np.isfinite(t1)), default=None)
    if lo is None or hi is None or not hi > lo:
        return None
    return (float(lo), float(hi))


def _standardise_within(raw: np.ndarray, t_s: np.ndarray,
                        window: tuple[float, float]):
    """Re-apply `ae.transform.standardise`'s z-score over `window`'s columns.

    The same formula `prep` applied, over a subset of the columns, and the
    returned `(mean, std)` replace `meta`'s so that `masks.unstandardise`
    still recovers `raw` exactly. `None` when the window holds under two
    columns, which the caller reports as a fall back to `record`.
    """
    inside = np.flatnonzero((t_s >= window[0]) & (t_s <= window[1]))
    if inside.size < 2:
        return None
    sub = raw[:, inside]
    mean, std = float(sub.mean()), float(sub.std())
    spec = ((raw - mean) / (std + STD_EPS)).astype(np.float32)
    return spec, mean, std, float(t_s[inside[0]]), float(t_s[inside[-1]])


#: The sentence `prep_block` carries out when `--norm plasma` had no window
#: to standardise inside. `_one_block` and the batch driver both record it
#: under `skipped[f"norm {key}:{pass}"]`, so it is written once.
NO_NORM_WINDOW = (
    "no usable coverage intersection; standardised over the whole record "
    "instead"
)


@dataclass(eq=False)
class PreparedBlock:
    """The CPU half of one `(channel, pass)`, ready for the network.

    The unit `events/driver.py` moves between processes, which is why it is
    a dataclass here rather than four locals inside `_one_block`: a prep
    worker returns one of these and the parent runs `infer_block` and
    `describe_block` on it. Two `(512, T)` float32 arrays - 131 MB each on
    a `mirnov` wide pass - so a driver holding `prefetch` of them holds
    `prefetch` x 262 MB, and `spectrogram` is dropped the moment the
    network has read it.

    `norm_note` is `NO_NORM_WINDOW` when `--norm plasma` found no window and
    empty otherwise: a stage function has no `skipped` dict to write into,
    so the reason travels with the block and its caller records it.
    """

    spec: Any
    pass_name: str
    spectrogram: np.ndarray | None
    raw: np.ndarray
    t_s: np.ndarray
    meta: dict
    fs_hz: float
    t_cov: tuple[float, float]
    norm_note: str = ""

    @property
    def key(self) -> str:
        """`"mhr:0:wide"` - how this block is named in `skipped`."""
        return f"{self.spec.key}:{self.pass_name}"

    @property
    def n_tiles(self) -> int:
        """The tiles the network will be run on, counted before they exist."""
        return masks.n_tiles(self.meta["n_cols"])


def prep_block(
    y, fs_hz: float, t0_s: float, t1_s: float, spec, pass_name: str, *,
    norm: str = "record", window=None,
) -> PreparedBlock:
    """The CPU part of one `(channel, pass)`: STFT, standardise, time axis.

    Everything `_one_block` does before the network and nothing else, so
    that a driver can run it somewhere other than where the network is
    (`events/driver.py` runs it in a pool of prep processes while the GPU
    is busy with the block before). No torch, no device, no model.

    The pre-standardisation log-power is carried out beside the spectrogram
    rather than recovered later with `unstandardise`: under `--norm plasma`
    the spectrogram and the statistics in `meta` are BOTH replaced, and
    inverting the second z-score would return `raw` only to float32
    rounding - which is exactly the identity the driver has to keep.
    """
    decim = 1 if pass_name == "wide" else masks.ZOOM_DECIM
    spectrogram, meta = masks.prep(y, fs_hz=fs_hz, decim=decim)
    t_s = masks.col_times_s(meta["n_cols"], fs_hz, decim, t0_s)
    raw = masks.unstandardise(spectrogram, meta)
    meta["norm"] = "record"
    note = ""
    if norm == "plasma":
        got = None if window is None else _standardise_within(raw, t_s, window)
        if got is None:
            note = NO_NORM_WINDOW
        else:
            spectrogram, meta["spec_mean"], meta["spec_std"] = got[:3]
            meta["norm"] = "plasma"
            meta["norm_t0_s"], meta["norm_t1_s"] = got[3], got[4]
    return PreparedBlock(
        spec=spec, pass_name=pass_name, spectrogram=spectrogram, raw=raw,
        t_s=t_s, meta=meta, fs_hz=float(fs_hz),
        t_cov=(float(t0_s), float(t1_s)), norm_note=note,
    )


def infer_block(prepared: PreparedBlock, *, model, device, tile_batch: int,
                amp: bool) -> np.ndarray:
    """The GPU part: `(2, 512, T)` float32 probabilities for one block.

    A thin call, named so that a driver's `infer_s` measures the same thing
    `_one_block` spends on the network and nothing else. Tiles are pooled to
    `tile_batch` WITHIN the block: `masks.infer` is the one definition of
    how a spectrogram reaches the network, and pooling tiles from two blocks
    into one forward pass would change the batch a tile is run in - the one
    thing that could make a driver's output differ from `process_shot`'s in
    the last bits.
    """
    return masks.infer(model, prepared.spectrogram, device, batch=tile_batch,
                       amp=amp)


def describe_block(prepared: PreparedBlock, probs, *,
                   unet_sha256: str) -> _BlockRun:
    """The CPU part after the network: pack, track, and the activity trace.

    Everything that needs the float32 probability maps happens here -
    `block_arrays` (which packs them), the tracks and the column activity -
    so that the maps themselves are unreachable by the time the next channel
    is read. On the widest group that is 144 MB per block that would
    otherwise be held until the file is written.
    """
    spec, pass_name = prepared.spec, prepared.pass_name
    raw, t_s = prepared.raw, prepared.t_s
    block = masks.MaskBlock(
        diag=spec.diag, channel=spec.channel, pass_name=pass_name,
        coh=probs[0], tra=probs[1], raw_logpow=raw, t_s=t_s,
        meta=prepared.meta,
    )
    arrays = masks.block_arrays(block, unet_sha256=unet_sha256)
    coh_mask = probs[0] >= PROB_THRESHOLD
    freq_khz = masks.freq_axis_khz(prepared.fs_hz, int(prepared.meta["decim"]))
    found = [
        tracks.descriptors(group, prob=probs[0], raw_logpow=raw,
                           freq_khz=freq_khz, t_s=t_s)
        for group in tracks.merge(tracks.components(coh_mask))
    ]
    activity = transients.column_activity(probs[1])
    return _BlockRun(
        diag=spec.diag, channel=spec.channel, pass_name=pass_name,
        prefix=block.prefix, arrays=arrays, tracks=found,
        activity=activity,
        bursts=transients.extract_bursts(activity),
        t_s=t_s, t_cov=prepared.t_cov,
    )


def _one_block(
    y, fs_hz: float, t0_s: float, t1_s: float, spec, pass_name: str, *,
    model, device, tile_batch: int, amp: bool, norm: str,
    window, unet_sha256: str, skipped: dict[str, str],
) -> _BlockRun:
    """One `(channel, pass)`: transform, infer, describe, then let go.

    The three stages above, composed, and the ONE place the composition is
    written: `events/driver.py` runs the same three on three different
    schedules, and a step that lived here rather than in a stage function
    would be a step the driver silently did not do.
    """
    prepared = prep_block(y, fs_hz, t0_s, t1_s, spec, pass_name, norm=norm,
                          window=window)
    if prepared.norm_note:
        skipped[f"norm {prepared.key}"] = prepared.norm_note
    probs = infer_block(prepared, model=model, device=device,
                        tile_batch=tile_batch, amp=amp)
    # As soon as the network has read it: 131 MB on the widest block, and
    # `describe_block` wants `raw`, not the standardised copy.
    prepared.spectrogram = None
    return describe_block(prepared, probs, unet_sha256=unet_sha256)


def _cooccurrence(runs: Sequence[_BlockRun]) -> dict[tuple[str, int], list[str]]:
    """`(block key, track index)` -> the other blocks' tracks that are it.

    Within one diagnostic only. A mode on the magnetics and on the ECE at
    the same time and frequency is evidence about what the mode IS, and it
    is `annotate/priors.py`'s to weigh; what this records is the cheaper
    half of it, one detector seeing the same thing on two of its own
    channels, which is what tells a mode from a channel's own artefact.
    """
    out: dict[tuple[str, int], list[str]] = {}
    by_diag: dict[str, dict[str, Sequence[tracks.Track]]] = {}
    for run in runs:
        by_diag.setdefault(run.diag, {})[run.co_key] = run.tracks
    for by_channel in by_diag.values():
        for group in tracks.cooccurrence(by_channel):
            for key, i in group:
                out[(key, i)] = sorted(
                    f"{k}#{j}" for k, j in group if (k, j) != (key, i)
                )
    return out


def _elm_reference(runs: Sequence[_BlockRun]) -> _BlockRun | None:
    """The one transient mask reference: most transient activity, wide.

    Magnetics first and the wide pass first - an ELM is broadband and the
    zoom pass has thrown the top three quarters of the band away - and the
    tie broken by block name so that two identical channels always give the
    same answer. Falls back to any block at all rather than to nothing: a
    shot whose only planned channel was an ECE one still has ELMs in it.
    """
    for pool in (
        [r for r in runs if r.diag in MAGNETICS_DIAGS and r.pass_name == "wide"],
        [r for r in runs if r.diag in MAGNETICS_DIAGS],
        [r for r in runs if r.pass_name == "wide"],
        list(runs),
    ):
        if pool:
            # `min` of `(-activity, prefix)`: most activity first, and the
            # name breaks the tie two identical channels always produce.
            # An empty trace scores 0.0, which sorts behind every real one.
            return min(
                pool,
                key=lambda r: (
                    -float(np.mean(r.activity)) if r.activity.size else 0.0,
                    r.prefix,
                ),
            )
    return None


def plan_shot(
    corpus_file,
    *,
    plan: Sequence[channels.ChannelSpec] = channels.ROUND1_PLAN,
    norm: str = "record",
) -> tuple[list[channels.ChannelSpec], dict[str, str], tuple[float, float] | None]:
    """`(specs, skipped, window)`: what this shot can run, before any of it.

    The header-only half of a shot - `channels.plan_for` plus, under
    `--norm plasma`, the coverage intersection `prep_block` standardises
    inside - separated out because it is the first thing a batch driver
    hands to a prep worker and the last thing that should happen in the
    process that holds the GPU.

    Raises only what `plan_for` raises, which is the one failure that ends a
    shot: a corpus file that cannot be opened. Everything else is recorded -
    the planner's own reasons under `channel <key>`, a coverage read that
    failed under `norm`, exactly as `process_shot` records them.
    """
    specs, reasons = channels.plan_for(corpus_file, plan)
    skipped = {f"channel {key}": why for key, why in reasons.items()}
    window = None
    if norm == "plasma" and specs:
        try:
            window = _norm_window(corpus_file, [s.diag for s in specs])
        except Exception as exc:  # noqa: BLE001 - per-step isolation
            skipped["norm"] = _cause(exc)
    return specs, skipped, window


def process_shot(
    shot: int,
    paths: Paths | None = None,
    *,
    model,
    device: str = "cpu",
    corpus_dir=None,
    plan: Sequence[channels.ChannelSpec] = channels.ROUND1_PLAN,
    passes: Sequence[str] = masks.PASS_NAMES,
    tile_batch: int = 32,
    amp: bool = False,
    norm: str = "record",
    write: bool = True,
    lexicon: Lexicon | None = None,
    run_id: str = "manual",
    unet_sha256: str = CHECKPOINT_SHA256,
) -> ShotResult:
    """Everything one shot produces: `masks/<shot>_masks.npz`, its events,
    and the index rows for them.

    `write=False` computes all of it and stores none of it, which is what a
    pilot measuring a threshold wants and what keeps a test off the disk.
    `lexicon=None` leaves the text out entirely; with one, the shot's own
    logbook entries are read FROM THE SUBSET (`text_weak.build_logs_subset`
    is the caller's to run once per run, in the parent, over the whole shot
    list - a per-shot build would stream 616 MB per shot).

    The masks and the events are both written with `merge=True`, so a re-run
    of one channel replaces that channel and leaves the rest of the file
    alone, and every source that RAN is named in the write even when it
    found nothing - "the tracker ran and saw no mode" is a different claim
    from "the tracker has not run".

    That claim is also written out on its own, as
    `events/<shot>_sources.parquet`: one row per `(source, diag, channel,
    pass)` that ran or was skipped, with the coverage it ran over and how
    many events it produced. An events file cannot carry it - a detector
    that found nothing writes no row to one - and without it a query for a
    shot's ELMs cannot tell an ELM-free shot from an unprocessed one.
    Coverage there is per source and per quantity, computed over FINITE
    samples: the gas valves' 105 s axis is not the NBI digitiser's
    coverage, and the L-H detector covers only where the D-alpha, the line
    density AND the injected power were all measured.
    """
    started = time.monotonic()
    res = ShotResult(shot=int(shot))
    paths = Paths.from_env() if paths is None else paths
    if str(norm) not in NORMS:
        raise ValueError(f"norm must be one of {NORMS}; got {norm!r}")
    passes = tuple(passes)
    bad = [p for p in passes if p not in masks.PASS_NAMES]
    if bad:
        raise ValueError(f"passes must be from {masks.PASS_NAMES}; got {bad}")
    corpus_file = (
        Path(corpus_dir) / f"{int(shot)}_processed.h5" if corpus_dir is not None
        else paths.corpus_file(shot)
    )

    def finish() -> ShotResult:
        res.elapsed_s = time.monotonic() - started
        return res

    try:
        specs, planned_skips, window = plan_shot(corpus_file, plan=plan,
                                                 norm=norm)
    except Exception as exc:  # noqa: BLE001 - the one run-ending failure
        res.error = _cause(exc)
        return finish()
    res.skipped.update(planned_skips)

    # ---------------------------------------------------------- the masks
    runs: list[_BlockRun] = []
    for spec in specs:
        try:
            y, fs_hz, t0_s, t1_s = masks.read_waveform(
                corpus_file, spec.diag, spec.channel
            )
        except Exception as exc:  # noqa: BLE001 - per-channel isolation
            res.skipped[f"read {spec.key}"] = _cause(exc)
            continue
        for pass_name in passes:
            try:
                runs.append(_one_block(
                    y, fs_hz, t0_s, t1_s, spec, pass_name, model=model,
                    device=device, tile_batch=tile_batch, amp=amp, norm=norm,
                    window=window, unet_sha256=unet_sha256,
                    skipped=res.skipped,
                ))
            except Exception as exc:  # noqa: BLE001 - per-pass isolation
                res.skipped[f"mask {spec.key}:{pass_name}"] = _cause(exc)
        del y
    res.n_blocks = len(runs)

    finish_shot(res, paths, corpus_file, runs, unet_sha256=unet_sha256,
                lexicon=lexicon, run_id=run_id, write=write)
    return finish()


def finish_shot(
    res: ShotResult,
    paths: Paths,
    corpus_file,
    runs: Sequence[_BlockRun],
    *,
    unet_sha256: str = CHECKPOINT_SHA256,
    lexicon: Lexicon | None = None,
    run_id: str = "manual",
    write: bool = True,
) -> ShotResult:
    """Everything a shot does AFTER its mask blocks, on `res` in place.

    The tracks and their co-occurrences, the ELM clock, the three
    heuristics, the QH proxy, the text, and the two writes - in this order,
    each guarded, because the order is the thing this module exists to own.
    Split out of `process_shot` so that `events/driver.py`, which produces
    the same `_BlockRun`s on a different schedule, finishes a shot through
    the same code rather than through a copy of it: a heuristic added here
    is a heuristic the batch runs get.

    `runs` are this shot's blocks in PLAN ORDER. The order reaches disk -
    `_elm_reference` breaks a tie by block name, `write_masks` writes the
    keys in the order given - so a caller that reorders them writes a
    different file.
    """
    shot = int(res.shot)
    events: list[Event] = []
    sources: set[str] = set()
    # `(source, diag, channel, pass_name)` -> the coverage it ran over.
    # EVERY key that ran, event or no event: `events/<shot>_sources.parquet`
    # is what tells "the tracker looked and saw nothing" from "the tracker
    # never ran", and an events file states neither.
    ran: dict[tuple[str, str, int, str], tuple[float, float]] = {}

    # --------------------------------------------------------- the tracks
    if runs:
        sources.add(tracks.SOURCE)
        partners = _cooccurrence(runs)
        for run in runs:
            try:
                rows = tracks.tracks_to_events(
                    run.tracks, shot=shot, diag=run.diag, channel=run.channel,
                    pass_name=run.pass_name, t_cov=run.t_cov,
                    unet_sha256=unet_sha256,
                )
            except Exception as exc:  # noqa: BLE001 - independent diagnostic blocks
                # An invalid padded track must not lose the D-alpha clock
                # or other diagnostics. This block is unknown, not quiet.
                key = f"track {run.diag}:{run.channel}:{run.pass_name}"
                res.skipped[key] = _cause(exc)
                continue
            res.n_tracks += len(rows)
            ran[(tracks.SOURCE, run.diag, run.channel, run.pass_name)] = (
                run.t_cov
            )
            events.extend(
                replace(e, attrs={
                    **e.attrs,
                    "cooccurrent_with": partners.get((run.co_key, i), []),
                })
                for i, e in enumerate(rows)
            )

    # ----------------------------------------- class-agnostic transients
    reference = _elm_reference(runs)
    if reference is None:
        res.skipped[transients.SOURCE] = "no mask block to read a transient trace off"
    else:
        try:
            elm_times = transients.elm_events(reference.activity, reference.t_s)
            events.extend(transients.transients_to_events(
                elm_times, reference.bursts, shot=shot, diag=reference.diag,
                channel=reference.channel, pass_name=reference.pass_name,
                t_s=reference.t_s, t_cov=reference.t_cov,
                unet_sha256=unet_sha256, activity=reference.activity,
            ))
            sources.add(transients.SOURCE)
            key = (reference.diag, reference.channel, reference.pass_name)
            ran[(transients.SOURCE, *key)] = reference.t_cov
        except Exception as exc:  # noqa: BLE001 - per-step isolation
            res.skipped[transients.SOURCE] = _cause(exc)

    # ------------------------------------------------------ the ELM clock
    elm_free = np.zeros((0, 2), dtype=np.float64)
    try:
        dalpha_t_s, dalpha_y = _read_group(
            corpus_file, "filterscopes", stop=heuristics.N_DALPHA_CHANNELS
        )
        # Prefer channel 0, falling back by channel index only when absent.
        # Selecting by number of peaks would favour the noisiest detector.
        channel = next((i for i, y in enumerate(dalpha_y)
                        if (np.isfinite(y[:-1]) & np.isfinite(y[1:])).any()), -1)
        if channel < 0:
            raise ValueError("no finite D-alpha channel in filterscopes 0-7")
        elm_cov = coverage.finite_span(dalpha_t_s, dalpha_y[channel])
        found = transients.elm_clock_events(
            dalpha_y[channel], dalpha_t_s, shot=shot, channel=channel,
        )
        events.extend(found)
        res.n_elms = sum(e.phenomenon == transients.ELM_PHENOMENON for e in found)
        res.elm_reference = f"filterscopes_{channel:02d}"
        elm_free = np.array([
            [e.t0_s, e.t1_s] for e in found
            if e.phenomenon == transients.FREE_PHENOMENON
        ], dtype=np.float64).reshape(-1, 2)
        sources.add(transients.ELM_SOURCE)
        ran[(transients.ELM_SOURCE, "filterscopes", channel, "")] = elm_cov
    except Exception as exc:  # noqa: BLE001 - per-step isolation
        res.skipped["elm_clock"] = _cause(exc)

    # ------------------------------------------------------- the sawteeth
    try:
        ece_t_s, ece_y = _read_group(corpus_file, "ece")
        ece_cov = coverage.finite_span(ece_t_s, ece_y)
        found = heuristics.sawtooth_events(
            ece_y, ece_t_s, shot=shot, t_cov=ece_cov,
        )
        del ece_y
        events.extend(found)
        res.n_sawteeth = len(found)
        sources.add(heuristics.SAWTOOTH_SOURCE)
        ran[(heuristics.SAWTOOTH_SOURCE, "ece", -1, "")] = ece_cov
    except Exception as exc:  # noqa: BLE001 - per-step isolation
        res.skipped["sawtooth"] = _cause(exc)

    # --------------------------------------------------- the L-H detector
    try:
        dalpha_t_s, dalpha_y = _read_group(
            corpus_file, "filterscopes", stop=heuristics.N_DALPHA_CHANNELS
        )
        ne_t_s, ne_y = _read_group(corpus_file, "co2", stop=1)
        pinj_t_s, pinj_y = _read_group(corpus_file, "pinj")
        lh_cov = coverage.intersect([
            coverage.finite_span(dalpha_t_s, dalpha_y),
            coverage.finite_span(ne_t_s, ne_y[0]),
            coverage.finite_span(pinj_t_s, pinj_y),
        ])
        found = heuristics.lh_transitions(
            dalpha_t_s, dalpha_y,
            ne_t_s=ne_t_s, ne_y=ne_y[0],
            # `betan` is not a corpus group; the transition's `attrs` say
            # `None` for it rather than a number nobody measured.
            betan_t_s=None, betan_y=None,
            pinj_t_s=pinj_t_s,
            pinj_y=np.asarray(pinj_y, dtype=np.float64).sum(axis=0) * 1e-3,
            shot=shot,
            # Three inputs, one answer: the transition is claimed where
            # ALL THREE were measured, not over the D-alpha alone.
            t_cov=lh_cov,
        )
        events.extend(found)
        res.n_lh = len(found)
        sources.add(heuristics.LH_SOURCE)
        ran[(heuristics.LH_SOURCE, "filterscopes", -1, "")] = lh_cov
    except Exception as exc:  # noqa: BLE001 - per-step isolation
        res.skipped["lh"] = _cause(exc)

    # -------------------------------------------------------- the actuators
    nbi_on: list[tuple[float, float]] = []
    features = _actuator_features(corpus_file, res.skipped)
    res.skipped["nbi_counter"] = NO_IP
    if features:
        # PER FEATURE, over finite samples: the gas recorder's -10 to
        # 94.86 s axis is not the NBI digitiser's coverage (task Lfix-C1).
        spans = coverage.feature_spans(features)
        ran.update(
            ((heuristics.ACTUATOR_SOURCE, name, -1, ""), span)
            for name, span in spans.items()
        )
        try:
            found = heuristics.actuator_intervals(
                features, shot=shot, t_cov=spans,
            )
            events.extend(found)
            nbi_on = [(e.t0_s, e.t1_s) for e in found if e.phenomenon == "nbi_on"]
            sources.add(heuristics.ACTUATOR_SOURCE)
        except Exception as exc:  # noqa: BLE001 - per-step isolation
            res.skipped["actuator"] = _cause(exc)

    # --------------------------------------------------------- the QH proxy
    try:
        # No Ip flat-top to intersect with, so this claims nothing today; it
        # runs anyway so that the source is declared and a shot's row can
        # say "the proxy looked" rather than nothing at all.
        res.skipped["qh_flattop"] = NO_FLATTOP
        events.extend(heuristics.qh_candidates(
            [t for run in runs for t in run.tracks], elm_free, nbi_on, [],
            shot=shot,
            t_cov=reference.t_cov if reference else coverage.UNKNOWN,
        ))
        sources.add(heuristics.QH_SOURCE)
        # `qh_candidates` stamps its rows `pass_name="zoom"`, so that is
        # the key its completion record has to carry.
        ran[(heuristics.QH_SOURCE, "", -1, "zoom")] = (
            reference.t_cov if reference else coverage.UNKNOWN
        )
    except Exception as exc:  # noqa: BLE001 - per-step isolation
        res.skipped["qh"] = _cause(exc)

    # ------------------------------------------------------------- the text
    if lexicon is None:
        res.skipped["text"] = "no lexicon passed"
    elif text_weak.load_log_record(shot, paths=paths) is None:
        # Read off the SUBSET, so a shot the logbook has no record of costs
        # nothing and cannot reach the 616 MB source through `text_events`.
        res.skipped["text"] = "no logbook record in the subset"
    else:
        try:
            found = text_weak.text_events(shot, lexicon, paths=paths)
            events.extend(found)
            res.n_text = len(found)
            sources.add("text")
            ran[("text", "", -1, "")] = text_weak.shot_span_s(shot, paths=paths)
        except Exception as exc:  # noqa: BLE001 - per-step isolation
            res.skipped["text"] = _cause(exc)

    # --------------------------------------------- the curated label tables
    try:
        found, res.database_sources = databases_block(shot, paths)
        events.extend(found)
        sources |= {e.source for e in found}
        # Through the same `ran` map as every detector, so the records
        # reach `schema.write_sources` with the rest: one key per table
        # that NAMED this shot, keyed `(source, "", -1, "")` because a
        # curated list reads no diagnostic, and coverage `UNKNOWN` - NaN,
        # not empty - because nobody recorded which interval was examined.
        # A table that does not name the shot adds no key here, so it
        # writes no row, which is the whole point.
        for record in res.database_sources:
            ran[(
                str(record["source"]), str(record["diag"]),
                int(record["channel"]), str(record["pass_name"]),
            )] = (float(record["t_cov0_s"]), float(record["t_cov1_s"]))
    except Exception as exc:  # noqa: BLE001 - per-step isolation
        res.skipped["database"] = _cause(exc)

    res.n_events = len(events)
    for event in events:
        res.by_source[event.source] = res.by_source.get(event.source, 0) + 1

    if write:
        try:
            masks.write_masks(
                paths.masks_file(shot), shot, [r.arrays for r in runs],
                merge=True,
            )
            events_file = paths.events_file(shot)
            schema.write_events(events_file, shot, events, run_id=run_id,
                                merge=True, sources=sorted(sources))
            schema.write_sources(
                paths.sources_file(shot), shot,
                coverage.source_records(
                    shot, ran=ran, skipped=res.skipped, events=events,
                ),
                run_id=run_id, merge=True,
            )
            append_index(paths.events_index, schema.index_rows(events_file),
                         keys=["shot", "source", "phenomenon"], replace_shots=[shot])
        except Exception as exc:  # noqa: BLE001 - see below
            # The one failure other than the corpus file that sets `error`.
            # Everything above is a claim this shot could not make; a failed
            # write means the claims it DID make are not on disk, so the
            # shot is not done and a re-run has to redo it - which is what
            # `status == "error"` tells a run summary.
            res.error = f"write failed: {_cause(exc)}"
    return res


def summarise(rows: Sequence[dict]) -> dict[str, Any]:
    """A run's totals, from `ShotResult.as_row()` dicts.

    Rows rather than `ShotResult`s because a shot that hit `run.py`'s
    per-shot SIGALRM has no result to summarise and must still be counted:
    the caller's error row has the same three keys this reads.
    """
    by_source: dict[str, int] = {}
    skips: dict[str, int] = {}
    counts: dict[str, int] = {}
    for row in rows:
        for source, n in (row.get("by_source") or {}).items():
            by_source[source] = by_source.get(source, 0) + int(n)
        for step in row.get("skipped") or {}:
            skips[step] = skips.get(step, 0) + 1
        status = str(row.get("status", "error"))
        counts[status] = counts.get(status, 0) + 1
    return {
        "counts": dict(sorted(counts.items())),
        "events_by_source": dict(sorted(by_source.items())),
        "shots_skipping": dict(sorted(skips.items())),
        "n_blocks": sum(int(r.get("n_blocks", 0)) for r in rows),
        "n_events": sum(int(r.get("n_events", 0)) for r in rows),
        "seconds": round(sum(float(r.get("seconds", 0.0)) for r in rows), 2),
    }


__all__ = [
    "PreparedBlock",
    "ShotResult",
    "describe_block",
    "finish_shot",
    "infer_block",
    "plan_shot",
    "prep_block",
    "process_shot",
    "summarise",
]
