"""Points of interest: one per AE region the segmentation model draws.

    python -m labeler.ae.seg.poi [--version V] [--shots S ...] [--from-corpus]
                                 [--workers N] [--no-pictures] [--models DIR]

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

**SegNet v2** (`--version v2`) draws over the band its blob records, 0-250 kHz
(`train.blob_band`), so a region below 80 kHz is a point too. Its shots default
to those of TokEye's whole-shot masks (`ae/masks-full`), and it writes to
`poi/alfven_eigenmode/ae_seg-v2/` and `gallery/alfven_eigenmode/ae_seg-v2/`.
Its `in_scored_window` flags a peak inside the shot's window in ae_xpower v3's
label snapshot (start inclusive, end exclusive), blank for a shot the snapshot
does not label; `meta.json` counts the points inside, outside and without that
window (`points_in_scored_window` and the like) beside the split at 2 s, and its
pictures draw no 0-2 s line. A model is drawn only as the version its blob
records (`train.blob_version`).

**SegNet v3** (`--version v3`) is drawn as SegNet v2 is, from its own model
(`models/ae_seg/v3`), into `poi/alfven_eigenmode/ae_seg-v3/` and
`gallery/alfven_eigenmode/ae_seg-v3/`.
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
from ..xpower import SCORED_UNTIL_MS, read_snapshot, tokeye_masks
from ..xpower.data import BAND_KHZ, band_slice, raw_rows, seldnet_split, store_rows
from ..xpower.gallery import run_all
from . import EVENT, METHOD, SEG_VERSIONS, VERSION, model_dir, poi_dir, regions
from .pseudo import EIGHT, IGNORE, LEVEL, PseudoMask
from .train import blob_band, blob_version, load, predict

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


def gallery_dir(paths: Paths, version: str = VERSION) -> Path:
    return paths.root / "gallery" / EVENT / f"{METHOD}-{version}"


def ae_pixels(
    prob: np.ndarray, threshold: float, y0: float, dy: float, band=BAND_KHZ
) -> np.ndarray:
    """The pixels the model calls AE: P(AE) at its threshold, in `band` (kHz;
    80-250 kHz, v1's, by default)."""
    on = np.zeros_like(prob, dtype=bool)
    rows = band_slice(y0, dy, prob.shape[0], band)
    on[rows] = prob[rows] >= threshold
    return on


def points(
    shot: int,
    prob: np.ndarray,
    threshold: float,
    grid: Grid,
    y0: float,
    dy: float,
    *,
    band=BAND_KHZ,
    method: str = f"{METHOD}-{VERSION}",
) -> tuple[list[dict], np.ndarray]:
    """The shot's points of interest in `band`, each named by `method`, and the
    labelled regions they come from."""
    on = ae_pixels(prob, threshold, y0, dy, band)
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
                "method": method,
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
    scored_until_ms: float | None = SCORED_UNTIL_MS,
):
    """R0 x V1 with the regions outlined and numbered; the pseudo-mask below. The
    scored window's end is marked at `scored_until_ms` when the picture runs past
    it; None, a whole-window version's, marks none."""
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
        ax.set_ylim(0, 250)
        ax.set_ylabel("R0 × V1\nkHz")
        if scored_until_ms is not None and extent[1] > scored_until_ms:
            caption = f"scored: 0-{scored_until_ms / 1000:g} s"
            ax.axvline(scored_until_ms, color="white", lw=0.8, ls="--", label=caption)
            ax.text(
                scored_until_ms,
                0.98,
                caption,
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
    paths, blob, version = w["paths"], w["blob"], w["version"]
    grid, values, y0, dy = shot_rows(paths, shot, w["from_corpus"])
    prob = predict(w["model"], values)
    found, labelled = points(
        shot,
        prob,
        blob["threshold"],
        grid,
        y0,
        dy,
        band=blob_band(blob),
        method=f"{METHOD}-{version}",
    )
    source = "corpus" if w["from_corpus"] else "store"
    for p in found:
        p["source"] = source
    if w["pictures"]:
        file = regions.pseudo_file(paths, shot, version)
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
            own = SEG_VERSIONS[version].pseudo
            clicked, _ = regions.clicked_mask(paths, shot, decision, own)
            if clicked is not None:  # made on the review page's masks, as training
                on = PseudoMask.load(clicked)
                mask = regions.transfer(mask, pm, on, decision)
                _, count = regions.label_regions(on.mask)
                review = (len(decision["rejected"]), count)
            mask = np.where(mask == IGNORE, 0, mask)
        whole = SEG_VERSIONS[version].whole_window
        draw(
            gallery_dir(paths, version) / f"{shot}.jpg",
            title=f"{shot}   AE regions, {METHOD} {version}, threshold "
            f"{blob['threshold']}, not reviewed",
            grid=grid,
            row=values[0],
            y0=y0,
            dy=dy,
            labelled=labelled,
            found=found,
            pseudo=mask,
            pseudo_review=review,
            scored_until_ms=None if whole else SCORED_UNTIL_MS,
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
    scored: dict[int, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """`poi.csv`, merged: the shots in `shots` replace their old rows. A peak is
    in the scored window at 0 <= t < 2000 ms (v1's) or, given `scored`, inside
    its shot's `(start, end)`, blank for a shot `scored` lacks."""
    new = pd.DataFrame(rows, columns=list(POI_COLUMNS))
    if path.is_file():
        old = pd.read_csv(path)
        new = pd.concat([old[~old.shot.isin(list(shots))], new], ignore_index=True)
    new = new.sort_values(["shot", "region"], kind="stable").reset_index(drop=True)
    # Also annotate retained legacy rows without changing their point values.
    if scored is None:
        new["in_scored_window"] = (new.t_peak_ms >= 0) & (new.t_peak_ms < 2000)
    else:
        new["in_scored_window"] = [
            bool(scored[r.shot][0] <= r.t_peak_ms < scored[r.shot][1])
            if r.shot in scored
            else ""
            for r in new.itertuples(index=False)
        ]
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
    p.add_argument(
        "--version",
        choices=sorted(SEG_VERSIONS),
        default=VERSION,
        help="the SegNet version the model is (default v1); v2 and v3 draw over "
        "0-250 kHz",
    )
    p.add_argument("--shots", type=int, nargs="*", help="default: every AE180 shot")
    p.add_argument("--from-corpus", action="store_true", help="rows from the corpus")
    p.add_argument("--no-pictures", action="store_true")
    p.add_argument(
        "--models", type=Path, help="default $LABELER_ROOT/models/ae_seg/<version>"
    )
    p.add_argument(
        "--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    )
    args = p.parse_args(argv)
    if args.from_corpus and not args.shots:
        p.error("--from-corpus needs --shots")
    paths = Paths.from_env()
    spec = SEG_VERSIONS[args.version]
    model_file = (args.models or model_dir(paths, args.version)) / "model.pt"
    model_hash = sha256_of(model_file)
    _, blob = load(model_file)
    found = blob_version(blob)
    if found != args.version:
        p.error(f"{model_file}: a SegNet {found} model; draw it with --version {found}")
    scored = None
    if spec.whole_window:
        # A whole-window version's scored window is the shot's labelled one.
        try:
            _, saved = read_snapshot(paths, spec.labels)
        except (OSError, ValueError) as error:
            p.error(f"label snapshot {spec.labels}: {type(error).__name__}: {error}")
        scored = {shot: tuple(label.window) for shot, label in saved.items()}
    shots = args.shots or sorted(seldnet_split(tokeye_masks(paths, spec.ae_version)))
    flags = {
        "from_corpus": args.from_corpus,
        "pictures": not args.no_pictures,
        "version": args.version,
    }
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
        poi_dir(paths, args.version) / "poi.csv",
        done,
        rows,
        windows={shot: (a, b) for shot, (a, b, _) in sources.items()},
        scored=scored,
    )
    used: dict[Path, list[int]] = {}
    for shot in sorted(done):
        if shot in sources:
            used.setdefault(sources[shot][2], []).append(shot)
    completed = table[table.shot.isin(done)]
    # A whole-window version's scored window is the shot's own, not 0-2 s.
    inside = completed.in_scored_window
    scored_counts = (
        {
            "points_in_scored_window": int(inside.eq(True).sum()),
            "points_outside_scored_window": int(inside.eq(False).sum()),
            "points_unknown_scored_window": int((inside == "").sum()),
        }
        if spec.whole_window
        else {}
    )
    evaluation_file = model_file.parent / "evaluation.json"
    evaluation = (
        json.loads(evaluation_file.read_text()) if evaluation_file.is_file() else {}
    )
    record = {
        "model": str(model_file),
        "model_sha256": model_hash,
        "version": args.version,
        "band_khz": [float(b) for b in blob_band(blob)],
        "scored_window": (
            f"the shot's {spec.labels}-snapshot window"
            if spec.whole_window
            else "0-2 s"
        ),
        "threshold": blob["threshold"],
        "MIN_POI_PIXELS": MIN_POI_PIXELS,
        "shots": sorted(done),
        "pictures": not args.no_pictures,
        "points_before_2s": int(sum(row["t_peak_ms"] < 2000 for row in rows)),
        "points_after_2s": int(sum(row["t_peak_ms"] >= 2000 for row in rows)),
        **scored_counts,
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
    meta_file = poi_dir(paths, args.version) / "meta.json"
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
