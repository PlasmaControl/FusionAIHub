"""Points of interest: one per AE region the segmentation model draws.

    python -m labeler.ae.seg.poi [--shots S ...] [--from-corpus] [--workers N]
                                 [--no-pictures] [--models DIR]

For each shot (default: every AE180 shot, from its review store; with
`--from-corpus`, the shots given, from the corpus's CO2 chords, as the
extension reads them) the model's mask at its threshold, 80-250 kHz: each
8-connected region of at least `MIN_POI_PIXELS` pixels is a point of interest.
A shot viewer shows each as a box on the spectrogram with its label, "AE", and
its peak time and frequency.

Writes `$LABELER_ROOT/poi/alfven_eigenmode/ae_seg-v1/poi.csv` (`POI_COLUMNS`,
one row per region, merged over runs: a shot drawn again replaces its rows)
and, unless `--no-pictures`, a JPEG per shot in
`$LABELER_ROOT/gallery/alfven_eigenmode/ae_seg-v1/`: the R0 x V1 row with the
regions outlined (the 30 largest numbered on crowded shots), and below it the
pseudo-mask with any current region review applied. Every point is a suggestion:
nobody has reviewed it.
The last two table columns flag peaks inside the scored [0, 2000) ms window
and the v1 rule-4 Ip window (start inclusive, end exclusive), from
`catalog/population.csv` or, for shots it does not list, the same rule measured
into `$LABELER_ROOT/ae/ip/ip.jsonl` (`python -m labeler.events.catalog.window
--log`). A missing window leaves `in_plasma` blank; no point is dropped.
`meta.json` counts those flags, names each window file with its hash and shots,
and records the model's G3 verdict.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import ndimage

from ...config import Paths, atomic_path, git_sha, sha256_of
from ...events.catalog.window import read_log
from ...events.review.rows import Grid, pool
from ...events.verify import corpus_signal
from ..xpower import tokeye_masks
from ..xpower.data import BAND_KHZ, band_slice, raw_rows, seldnet_split, store_rows
from ..xpower.gallery import BAND_LINE_KHZ, run_all
from . import EVENT, METHOD, VERSION, model_dir, poi_dir, regions
from .pseudo import EIGHT, IGNORE, LEVEL, PseudoMask
from .train import load, predict

MIN_POI_PIXELS = 20
POI_COLUMNS = (
    "shot",
    "event",
    "method",
    "region",
    "t_start_ms",
    "t_end_ms",
    "f_lo_khz",
    "f_hi_khz",
    "t_peak_ms",
    "f_peak_khz",
    "pixels",
    "mean_prob",
    "max_prob",
    "source",
    "in_scored_window",
    "in_plasma",
)


def gallery_dir(paths: Paths) -> Path:
    return paths.root / "gallery" / EVENT / f"{METHOD}-{VERSION}"


def ae_pixels(prob: np.ndarray, threshold: float, y0: float, dy: float) -> np.ndarray:
    """The pixels the model calls AE: P(AE) at its threshold, 80-250 kHz."""
    on = np.zeros_like(prob, dtype=bool)
    band = band_slice(y0, dy, prob.shape[0], BAND_KHZ)
    on[band] = prob[band] >= threshold
    return on


def points(
    shot: int, prob: np.ndarray, threshold: float, grid: Grid, y0: float, dy: float
) -> tuple[list[dict], np.ndarray]:
    """The shot's points of interest, and the labelled regions they come from."""
    on = ae_pixels(prob, threshold, y0, dy)
    labelled, count = ndimage.label(on, structure=EIGHT)
    sizes = np.bincount(labelled.ravel(), minlength=count + 1)
    keep = np.flatnonzero(sizes >= MIN_POI_PIXELS)
    keep = keep[keep > 0]
    out = np.zeros_like(labelled)
    found = []
    for number, k in enumerate(keep, start=1):
        region = labelled == k
        out[region] = number
        rows, cols = np.nonzero(region)
        values = prob[rows, cols]
        peak = int(np.argmax(values))
        found.append(
            {
                "shot": int(shot),
                "event": EVENT,
                "method": f"{METHOD}-{VERSION}",
                "region": number,
                "t_start_ms": round(grid.t0_ms + cols.min() * grid.dt_ms, 1),
                "t_end_ms": round(grid.t0_ms + (cols.max() + 1) * grid.dt_ms, 1),
                "f_lo_khz": round(y0 + (rows.min() - 0.5) * dy, 2),
                "f_hi_khz": round(y0 + (rows.max() + 0.5) * dy, 2),
                "t_peak_ms": round(grid.t0_ms + (cols[peak] + 0.5) * grid.dt_ms, 1),
                "f_peak_khz": round(y0 + rows[peak] * dy, 2),
                "pixels": int(region.sum()),
                "mean_prob": round(float(values.mean()), 4),
                "max_prob": round(float(values.max()), 4),
            }
        )
    return found, out


def draw(
    path,
    *,
    title,
    grid: Grid,
    row,
    y0,
    dy,
    labelled,
    found,
    pseudo=None,
    pseudo_review=None,
):
    """R0 x V1 with the regions outlined and numbered; the pseudo-mask below."""
    panels = 2 if pseudo is not None else 1
    fig = Figure(figsize=(16, 4 + 3 * panels), dpi=100, layout="constrained")
    axes = np.atleast_1d(fig.subplots(panels, 1, sharex=True))
    extent = (
        grid.t0_ms,
        grid.t0_ms + grid.n * grid.dt_ms,
        y0 - dy / 2,
        y0 + (row.shape[0] - 0.5) * dy,
    )
    t = grid.t0_ms + (np.arange(grid.n) + 0.5) * grid.dt_ms
    f = y0 + np.arange(row.shape[0]) * dy
    for ax in axes:
        ax.imshow(
            row,
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
        ax.set_ylabel("R0 × V1\nkHz")
        if extent[1] > 2000:
            ax.axvline(2000, color="white", lw=0.8, ls="--", label="scored: 0-2 s")
            ax.text(
                2000,
                0.98,
                "scored: 0-2 s",
                transform=ax.get_xaxis_transform(),
                ha="right",
                va="top",
                color="white",
                fontsize=8,
            )
    if labelled.any():
        axes[0].contour(
            t, f, labelled > 0, levels=[0.5], colors="#00e5ff", linewidths=1
        )
    numbered = sorted(found, key=lambda p: (-p["pixels"], p["region"]))[:30]
    for p in numbered:
        axes[0].annotate(
            str(p["region"]),
            (p["t_start_ms"], p["f_hi_khz"]),
            color="#00e5ff",
            fontsize=8,
            va="bottom",
        )
    crowded = " (the 30 largest numbered; all in poi.csv)" if len(found) > 30 else ""
    axes[0].set_title(
        f"{len(found)} regions the model draws{crowded}", fontsize=9, loc="left"
    )
    if pseudo is not None:
        shown = np.ma.masked_where(pseudo != 1, np.ones_like(pseudo, dtype=float))
        axes[1].imshow(
            shown,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="cool",
            alpha=0.7,
            interpolation="nearest",
        )
        review_title = (
            "pseudo-mask (TokEye inside the owner's AE frames; regions not reviewed)"
        )
        if pseudo_review is not None:
            k, n = pseudo_review
            review_title = (
                f"pseudo-mask after the owner's region review ({k} of {n} rejected)"
            )
        axes[1].set_title(review_title, fontsize=9, loc="left")
    axes[-1].set_xlabel("time (ms)")
    fig.suptitle(title)
    with atomic_path(path) as tmp:
        fig.savefig(tmp, format="jpeg", pil_kwargs={"quality": 85})


_WORKER: dict = {}


def _init(model_file: str, root: str, label_tables: str, corpus: str, flags: dict):
    import torch

    torch.set_num_threads(1)
    model, blob = load(model_file)
    paths = Paths(root=Path(root), label_tables=Path(label_tables), corpus=Path(corpus))
    decisions = regions.read_decisions(Path(label_tables) / EVENT)
    _WORKER.update(model=model, blob=blob, paths=paths, decisions=decisions, **flags)


def shot_rows(paths: Paths, shot: int, from_corpus: bool):
    """`(grid, (3, n_y, n) uint8, y0, dy)` at store level 8."""
    if not from_corpus:
        return store_rows(paths.spectrogram_file(EVENT, shot), LEVEL)
    co2 = corpus_signal(shot, "co2", corpus=paths.corpus)
    grid, values, y0, dy = raw_rows(co2.x, co2.y)
    coarse = Grid(grid.t0_ms, grid.dt_ms * LEVEL, -(-grid.n // LEVEL))
    return coarse, pool(values, LEVEL, "image"), y0, dy


def shot_points(shot: int) -> list[dict]:
    w = _WORKER
    paths, blob = w["paths"], w["blob"]
    grid, values, y0, dy = shot_rows(paths, shot, w["from_corpus"])
    prob = predict(w["model"], values)
    found, labelled = points(shot, prob, blob["threshold"], grid, y0, dy)
    source = "corpus" if w["from_corpus"] else "store"
    for p in found:
        p["source"] = source
    if w["pictures"]:
        file = regions.pseudo_file(paths, shot)
        mask = None
        review = None
        if file.is_file() and not w["from_corpus"]:
            pm = PseudoMask.load(file)
            sha = regions.file_sha256(file)
            decision = w["decisions"].get(shot)
            mask = regions.reviewed_mask(pm, decision, sha)
            if decision and decision.get("pseudo_sha256") == sha:
                _, count = regions.label_regions(pm.mask)
                review = (len(decision["rejected"]), count)
            mask = np.where(mask == IGNORE, 0, mask)
        draw(
            gallery_dir(paths) / f"{shot}.jpg",
            title=f"{shot}   AE regions, {METHOD} {VERSION}, threshold "
            f"{blob['threshold']}, not reviewed",
            grid=grid,
            row=values[0],
            y0=y0,
            dy=dy,
            labelled=labelled,
            found=found,
            pseudo=mask,
            pseudo_review=review,
        )
    return found


def ae_ip_log(paths: Paths) -> Path:
    """The rule-4 Ip windows measured for AE shots that v1's population lacks."""
    return paths.root / "ae" / "ip" / "ip.jsonl"


def window_sources(paths: Paths) -> dict[int, tuple[float, float, Path]]:
    """Each shot's v1 rule-4 Ip window and the file it came from.

    `catalog/population.csv` (the windows the xpower extension uses, without its
    cut) wins; the AE Ip log only fills shots the population does not list, so a
    population row without a window stays without one.
    """
    found: dict[int, tuple[float, float, Path]] = {}
    log = ae_ip_log(paths)
    if log.is_file():
        frame = read_log(log)
        for row in frame[frame.status == "ok"].itertuples(index=False):
            found[int(row.shot)] = (
                float(row.window_start_ms),
                float(row.window_end_ms),
                log,
            )
    file = paths.catalog / "population.csv"
    if file.is_file():
        for row in pd.read_csv(file).itertuples(index=False):
            found.pop(int(row.shot), None)
            if (
                pd.notna(row.window_start_ms)
                and pd.notna(row.window_end_ms)
                and row.window_end_ms > row.window_start_ms
            ):
                found[int(row.shot)] = (
                    float(row.window_start_ms),
                    float(row.window_end_ms),
                    file,
                )
    return found


def plasma_windows(paths: Paths) -> dict[int, tuple[float, float]]:
    """Each shot's v1 rule-4 Ip window, from `window_sources`."""
    return {shot: (a, b) for shot, (a, b, _) in window_sources(paths).items()}


def write_points(
    path: Path,
    shots,
    rows: list[dict],
    *,
    windows: dict[int, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """`poi.csv`, merged: the shots in `shots` replace their old rows."""
    new = pd.DataFrame(rows, columns=list(POI_COLUMNS))
    if path.is_file():
        old = pd.read_csv(path)
        new = pd.concat([old[~old.shot.isin(list(shots))], new], ignore_index=True)
    new = new.sort_values(["shot", "region"], kind="stable").reset_index(drop=True)
    # Also annotate retained legacy rows without changing their point values.
    new["in_scored_window"] = (new.t_peak_ms >= 0) & (new.t_peak_ms < 2000)
    if windows is not None:
        new["in_plasma"] = [
            bool(windows[r.shot][0] <= r.t_peak_ms < windows[r.shot][1])
            if r.shot in windows
            else ""
            for r in new.itertuples(index=False)
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as tmp:
        new.to_csv(tmp, index=False)
    return new


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--shots", type=int, nargs="*", help="default: every AE180 shot")
    p.add_argument("--from-corpus", action="store_true", help="rows from the corpus")
    p.add_argument("--no-pictures", action="store_true")
    p.add_argument("--models", type=Path, help="default $LABELER_ROOT/models/ae_seg/v1")
    p.add_argument(
        "--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    )
    args = p.parse_args(argv)
    if args.from_corpus and not args.shots:
        p.error("--from-corpus needs --shots")
    paths = Paths.from_env()
    model_file = (args.models or model_dir(paths)) / "model.pt"
    model_hash = sha256_of(model_file)
    _, blob = load(model_file)
    shots = args.shots or sorted(seldnet_split(tokeye_masks(paths)))
    flags = {"from_corpus": args.from_corpus, "pictures": not args.no_pictures}
    init = (
        str(model_file),
        str(paths.root),
        str(paths.label_tables),
        str(paths.corpus),
        flags,
    )
    rows, done, failed = [], [], []
    for shot, outcome in run_all(shot_points, shots, args.workers, _init, init):
        if isinstance(outcome, Exception):
            failed.append(shot)
            print(f"{shot}: {type(outcome).__name__}: {outcome}", flush=True)
        else:
            done.append(shot)
            rows += outcome
    if sha256_of(model_file) != model_hash:
        p.error(f"{model_file}: model changed while drawing points")
    sources = window_sources(paths)
    table = write_points(
        poi_dir(paths) / "poi.csv",
        done,
        rows,
        windows={shot: (a, b) for shot, (a, b, _) in sources.items()},
    )
    used: dict[Path, list[int]] = {}
    for shot in sorted(done):
        if shot in sources:
            used.setdefault(sources[shot][2], []).append(shot)
    completed = table[table.shot.isin(done)]
    evaluation_file = model_file.parent / "evaluation.json"
    evaluation = (
        json.loads(evaluation_file.read_text()) if evaluation_file.is_file() else {}
    )
    record = {
        "model": str(model_file),
        "model_sha256": model_hash,
        "threshold": blob["threshold"],
        "MIN_POI_PIXELS": MIN_POI_PIXELS,
        "shots": sorted(done),
        "pictures": not args.no_pictures,
        "points_before_2s": int(sum(row["t_peak_ms"] < 2000 for row in rows)),
        "points_after_2s": int(sum(row["t_peak_ms"] >= 2000 for row in rows)),
        "points_in_plasma": int(completed.in_plasma.eq(True).sum()),
        "points_outside_plasma": int(completed.in_plasma.eq(False).sum()),
        "points_unknown_plasma": int((completed.in_plasma == "").sum()),
        "plasma_window_sources": {
            str(file): {"sha256": sha256_of(file), "shots": shots}
            for file, shots in used.items()
        },
        "plasma_window_rule": "v1 rule 4; start <= t_peak_ms < end",
        "G3": evaluation.get("bar", {}).get("G3"),
        "scope": "shots completed in this run; after includes peaks at 2000 ms",
        "git_sha": git_sha(),
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    meta_file = poi_dir(paths) / "meta.json"
    previous = (
        json.loads(meta_file.read_text()).get("runs", []) if meta_file.exists() else []
    )
    with atomic_path(meta_file) as tmp:
        tmp.write_text(
            json.dumps({**record, "runs": [*previous, record]}, indent=1) + "\n"
        )
    print(
        f"{len(rows)} points on {len(done)} shots ({len(table)} in the table); "
        f"{len(failed)} failed"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
