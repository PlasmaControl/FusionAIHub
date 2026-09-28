"""One JPEG per shot: the AE rows, the owner's label and the model's frames.

    python -m labeler.ae.xpower.gallery [--workers N] [--shots S ...] [--models DIR]

draws every AE180 shot from its review store into
`$LABELER_ROOT/gallery/alfven_eigenmode/ae_xpower-v1/reviewed/<shot>.jpg` (the
owner has saved it) or `unreviewed/<shot>.jpg` (not yet: the strip is then the
source table's), and writes `index.csv` beside the folders. `extend` draws its
shots into `extension/` with the same `draw`. A version with a label snapshot (v2)
takes the owner's labels from its snapshot, not the live file; `--version` must
be the models directory's name and the checkpoint's version.

A picture is three cross-power rows, 0-250 kHz, on the review page's colour
scale (inferno over -3..27 dB above each bin's quiet median) with a dashed line
at 80 kHz, where the model's band starts; a strip of the owner's frames
(present red, uncertain amber, not observable grey); and the model's P(AE) per
10 ms frame with its threshold, its present frames shaded, and the frames
TokEye marks as MHD ticked along the bottom.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from ...config import Paths, atomic_path
from ...events.catalog.states import NOT_OBSERVABLE, PRESENT, UNCERTAIN
from ...events.review import labels
from ...scoring.frames import FRAME_MS
from . import (
    EVENT,
    LABEL_SNAPSHOTS,
    VERSION,
    check_bound,
    event_dir,
    gallery_dir,
    model_dir,
    read_snapshot,
    tokeye_masks,
)
from .data import (
    CROSS_ROWS,
    clean_path,
    mhd_frames,
    seldnet_split,
    store_rows,
    targets,
    window_frames,
)
from .evaluate import chosen_model
from .train import f1_of, frame_cells, load, probabilities, read_split

STATE_COLOURS = {
    PRESENT: "#d62728",
    UNCERTAIN: "#ffb000",
    NOT_OBSERVABLE: "#9a9a9a",
}
INDEX_COLUMNS = (
    "shot",
    "group",
    "split",
    "file",
    "window_start_ms",
    "window_end_ms",
    "model_present_frames",
    "reference_present_frames",
    "f1_vs_owner",
    "threshold",
    "candidate",
    "version",
)
MARGIN_MS = 50.0
BAND_LINE_KHZ = 80.0


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """`(start, stop)` of each run of True."""
    edges = np.flatnonzero(np.diff(np.r_[0, flags.astype(np.int8), 0]))
    return list(zip(edges[::2], edges[1::2]))


def draw(
    path,
    *,
    title: str,
    grid,
    values: np.ndarray,
    y0: float,
    dy: float,
    first: int,
    prob: np.ndarray,
    threshold: float,
    reference: np.ndarray | None = None,
    reference_name: str = "owner",
    mhd: np.ndarray | None = None,
) -> None:
    """One shot's picture, JPEG. `values` is `(3, n_y, n)` bytes on `grid`;
    `prob`, `reference` and `mhd` are per frame from frame `first`."""
    n = len(prob)
    t0, t1 = first * FRAME_MS - MARGIN_MS, (first + n) * FRAME_MS + MARGIN_MS
    fig = Figure(figsize=(16, 10), dpi=100, layout="constrained")
    axes = fig.subplots(
        5, 1, sharex=True, gridspec_kw={"height_ratios": [3, 3, 3, 0.5, 1.4]}
    )
    extent = (
        grid.t0_ms,
        grid.t0_ms + grid.n * grid.dt_ms,
        y0 - dy / 2,
        y0 + (values.shape[1] - 0.5) * dy,
    )
    for ax, name, image in zip(axes[:3], CROSS_ROWS, values):
        ax.imshow(
            image,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="inferno",
            vmin=0,
            vmax=255,
            interpolation="nearest",
        )
        ax.axhline(BAND_LINE_KHZ, color="white", lw=0.8, ls="--")
        ax.set_ylim(0, 250)
        ax.set_ylabel(f"{name.replace('x', ' × ')}\nkHz")
    edges = (first + np.arange(n + 1)) * FRAME_MS
    strip = axes[3]
    if reference is not None:
        for state, colour in STATE_COLOURS.items():
            for a, b in _runs(reference == state):
                strip.axvspan(edges[a], edges[b], color=colour, lw=0)
        strip.set_ylabel(reference_name, rotation=0, ha="right", va="center")
    else:
        strip.text(
            0.5,
            0.5,
            "no label: nobody has reviewed this shot",
            transform=strip.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="#666666",
        )
    strip.set_yticks([])
    model = axes[4]
    centres = edges[:-1] + FRAME_MS / 2
    for a, b in _runs(prob >= threshold):
        model.axvspan(
            edges[a], edges[b], color=STATE_COLOURS[PRESENT], alpha=0.25, lw=0
        )
    model.plot(centres, prob, color="black", lw=0.8)
    model.axhline(threshold, color="black", lw=0.6, ls=":")
    if mhd is not None and mhd.any():
        model.plot(centres[mhd], np.full(mhd.sum(), 0.03), "|", color="#1f77b4", ms=6)
    model.set_ylim(0, 1)
    model.set_ylabel("P(AE)")
    model.set_xlabel("time (ms)")
    model.set_xlim(t0, t1)
    if t1 > 2000:
        for ax in axes:
            colour = "white" if ax in axes[:3] else "black"
            ax.axvline(2000, color=colour, lw=0.8, ls="--", label="scored: 0-2 s")
            ax.text(
                2000,
                0.98,
                "scored: 0-2 s",
                transform=ax.get_xaxis_transform(),
                ha="right",
                va="top",
                color=colour,
                fontsize=8,
            )
    keys = (
        ("present", STATE_COLOURS[PRESENT]),
        ("uncertain", STATE_COLOURS[UNCERTAIN]),
        ("not observable", STATE_COLOURS[NOT_OBSERVABLE]),
        ("MHD frame (TokEye)", "#1f77b4"),
    )
    handles = [Patch(color=colour, label=name) for name, colour in keys]
    fig.legend(handles=handles, loc="outside lower center", fontsize=8, ncols=4)
    fig.suptitle(title)
    with atomic_path(path) as tmp:
        fig.savefig(tmp, format="jpeg", pil_kwargs={"quality": 85})


_WORKER: dict = {}


def _init(
    model_file: str, root: str, label_tables: str, corpus: str, version: str = VERSION
) -> None:
    import torch

    torch.set_num_threads(1)
    model, blob = load(model_file)
    paths = Paths(root=Path(root), label_tables=Path(label_tables), corpus=Path(corpus))
    split = read_split(Path(model_file).parent / "split.csv")
    _WORKER.update(
        version=version,
        model=model,
        blob=blob,
        paths=paths,
        split=split,
        # A snapshot version's owner labels are its snapshot, never the live file.
        live=(
            read_snapshot(paths, version)[1]
            if version in LABEL_SNAPSHOTS
            else labels.read_saved(event_dir(paths))
        ),
        source=labels.read_source(event_dir(paths)),
    )


def picture(shot: int) -> dict:
    """Draw one AE180 shot; its `index.csv` row."""
    w = _WORKER
    paths, blob = w["paths"], w["blob"]
    version = w["version"]
    reviewed = shot in w["live"]
    label = w["live"].get(shot) or w["source"].get(shot)
    if label is None:
        raise KeyError(f"{shot} has neither a saved nor a source label")
    first, n = window_frames(label.window)
    store = paths.spectrogram_file(EVENT, shot)
    prob, _ = probabilities(
        w["model"], store_rows(store), first, n, band=blob["band_khz"]
    )
    grid, values, y0, dy = store_rows(store, level=8)
    reference = targets(label, first, n)
    mhd = mhd_frames(clean_path(tokeye_masks(paths), shot), first, n)
    split = w["split"].get(shot, "unreviewed" if not reviewed else "after training")
    cells = frame_cells(prob, reference, blob["threshold"])
    f1 = f1_of(cells) if reviewed else float("nan")
    group = "reviewed" if reviewed else "unreviewed"
    file = gallery_dir(paths, version) / group / f"{shot}.jpg"
    title = f"{shot}   AE, ae_xpower {version} ({blob['candidate']}), split {split}" + (
        f", F1 vs owner {f1:.2f}"
        if reviewed
        else ", not reviewed: strip is the source table"
    )
    draw(
        file,
        title=title,
        grid=grid,
        values=values,
        y0=y0,
        dy=dy,
        first=first,
        prob=prob,
        threshold=blob["threshold"],
        reference=reference,
        reference_name="owner" if reviewed else "source",
        mhd=mhd,
    )
    other_group = "unreviewed" if reviewed else "reviewed"
    (gallery_dir(paths, version) / other_group / f"{shot}.jpg").unlink(missing_ok=True)
    return {
        "shot": shot,
        "group": group,
        "split": split,
        "file": str(file.relative_to(gallery_dir(paths, version))),
        "window_start_ms": label.window[0],
        "window_end_ms": label.window[1],
        "model_present_frames": int((prob >= blob["threshold"]).sum()),
        "reference_present_frames": int((reference == PRESENT).sum()),
        "f1_vs_owner": "" if not reviewed else round(f1, 4),
        "threshold": blob["threshold"],
        "candidate": blob["candidate"],
        "version": version,
    }


def write_index(path, rows: Sequence[dict]) -> None:
    """`index.csv`, merged with what is there: a shot drawn again replaces its row.

    Reviewed/unreviewed share a row; extension keeps its own row for that shot.
    """

    def key(row):
        group = row["group"]
        family = "gallery" if group in ("reviewed", "unreviewed") else group
        return int(row["shot"]), family

    path = Path(path)
    old = {}
    if path.is_file():
        with path.open() as f:
            old = {key(r): r for r in csv.DictReader(f)}
    for row in rows:
        old[key(row)] = row
    with atomic_path(path) as tmp, open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        for identity in sorted(old):
            writer.writerow({c: old[identity].get(c, "") for c in INDEX_COLUMNS})


def run_all(work, shots, workers: int, init, initargs):
    """`(shot, row or the exception)` for each shot, in order; one bad shot does
    not stop the rest. Workers are spawned, not forked, so torch starts clean."""
    if workers <= 1:
        init(*initargs)
        for shot in shots:
            try:
                yield shot, work(shot)
            except Exception as error:  # noqa: BLE001
                yield shot, error
        return
    context = get_context("spawn")
    with ProcessPoolExecutor(
        workers, mp_context=context, initializer=init, initargs=initargs
    ) as pool:
        for shot, future in [(s, pool.submit(work, s)) for s in shots]:
            try:
                yield shot, future.result()
            except Exception as error:  # noqa: BLE001
                yield shot, error


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--shots", type=int, nargs="*", help="default: every AE180 shot")
    p.add_argument(
        "--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    )
    p.add_argument(
        "--models", type=Path, help="default $LABELER_ROOT/models/ae_xpower/v1"
    )
    p.add_argument("--version", default=VERSION)
    args = p.parse_args(argv)
    version = args.version
    paths = Paths.from_env()
    models = args.models or model_dir(paths, version)
    try:
        check_bound(version, models)
        model_file = chosen_model(models)
        check_bound(version, models, load(model_file)[1], model_file)
    except (OSError, ValueError, KeyError) as error:
        p.error(str(error))
    shots = args.shots or sorted(seldnet_split(tokeye_masks(paths)))
    init = (
        str(model_file),
        str(paths.root),
        str(paths.label_tables),
        str(paths.corpus),
        version,
    )
    rows, failed = [], []
    for shot, outcome in run_all(picture, shots, args.workers, _init, init):
        if isinstance(outcome, Exception):
            failed.append(shot)
            print(f"{shot}: {type(outcome).__name__}: {outcome}", flush=True)
        else:
            rows.append(outcome)
    write_index(gallery_dir(paths, version) / "index.csv", rows)
    print(
        f"drew {len(rows)} shots into {gallery_dir(paths, version)}; "
        f"{len(failed)} failed"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
