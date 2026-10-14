#!/usr/bin/env python
"""Label a list of shots with confine-ours from what is already on disk, no fetch.

``confinement_ours.py apply`` labels the roster, whose density and EFIT traces
``confinement_zerod_fetch.py`` pulled from MDSplus. A shot outside the roster (the gold
set) has none, so this script builds the same seven-channel 1 ms grid from the files the
corpus, the raw cache and the feature store hold, then applies the five fold networks
with ``apply``'s own segmentation (beam-on bins, 21-bin mode filter, 20 ms minimum,
confidence floor 0.7, tiers). Channels:

* ``da_*``, ``pinj``  as for the roster: ``zerod.assemble`` reads the corpus or raw cache
* ``dens``   row 2 of the ``co2`` group (the V2 chord, DENUF), 1 ms mean; it is the same
  line integral as ``\\BCI::DENV2F`` the roster uses (``check`` measures how close)
* ``betan``  the ``betan`` group of the feature store, else of the raw cache (EFIT)
* ``wmhd``   the ``wmhd`` group of the raw cache; most shots have none

A record ``confinement_zerod_fetch.py`` already made for the shot is used as it stands.
A shot still lacking density, beta_N or W_MHD is skipped (``--missing skip``, the
default) because the network was trained with all three on 836 of 839 roster grids and
``check`` finds that masking W_MHD changes the class of about a fifth of the beam-on
bins; ``--missing mask`` runs it anyway. ``check`` also tries estimating W_MHD from
beta_N and I_p (``WMHD_PER_BETAN_MA``): that agrees with the fetched-input labels less
than masking does, so ``label`` never uses it.

Stages::

    label  --shots FILE --name STEM [--unlabelled [--include SHOT... --include-reason T]]
           [--missing skip|mask]
           [--input-check FILE] [--out-dir DIR]
           -> data/events/confinement/extend_confine_ours/<STEM>.csv + <STEM>.meta.json
              (shot, category, t_start, t_end, confidence, attrs: the review page's
              table reader takes no other column, so the roster's predicted,
              source, tier and extrapolated go in the `attrs` JSON)
    check  --sample N --out FILE --scratch DIR
           -> agreement of the labels made from disk inputs with those from the
              fetched ones, on N roster shots outside the curated set
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import confinement_ours as ours  # puts src/ on sys.path

from labeler.confinement import zerod

FEATURES = ours.LABELER / "features"
#: The ``co2`` group's rows are R0, V1, V2, V3 (``configs/shot_design/signals.yaml``).
CO2_V2_ROW = 2
#: Records in the fetch's own format, built from files on disk (outputs, not inputs).
ZEROD_DISK = ours.WORK / "zerod_disk"
SPARSE_SOURCES = ("dens", "betan", "wmhd")
#: W_MHD / (beta_N * I_p[MA]) in J. W_MHD = 3/2 <p> V and <p> = beta_N I_p B / (200 mu0 a),
#: so the ratio varies only with the field, the minor radius and the volume. Median of
#: the roster shots' fetched EFIT01 traces (111 shots from 180000 up, one ratio each:
#: interquartile range 330-368 kJ, 8 % to 12 % spread within an era).
WMHD_PER_BETAN_MA = 357e3
LABELLED_KINDS = ("saved", "format", "extend", "source")


def _read_group(path: Path, group: str, row: int = 0):
    """Time (ms) and one row of a corpus-style group (time in s), or None."""
    import h5py

    if not path.exists():
        return None
    with h5py.File(path, "r") as f:
        if group not in f or f[group]["ydata"].shape[-1] <= 1:
            return None
        y = f[group]["ydata"]
        if y.ndim < 2 or row >= y.shape[0]:
            return None
        return (
            f[group]["xdata"][:].astype(np.float64) * 1000.0,
            y[row].astype(np.float32),
        )


def _usable(t_ms: np.ndarray, y: np.ndarray) -> bool:
    """At least two finite samples, over half of them, and a trace that varies."""
    finite = np.isfinite(y) & np.isfinite(t_ms)
    if finite.sum() < 2 or finite.mean() <= 0.5:
        return False
    # float64 on a subsample: a float32 square overflows on the largest traces
    sample = y[finite][:: max(1, int(finite.sum()) // 20000)].astype(np.float64)
    return bool(np.std(sample) > 0)


def read_ip(shot: int, raw_dir: Path, corpus: Path, features: Path):
    """Time (ms) and |I_p| in MA from the corpus, the raw cache or the feature store,
    or None."""
    for path in (
        corpus / f"{shot}_processed.h5",
        raw_dir / f"{shot}_processed.h5",
        features / f"{shot}_features.h5",
    ):
        got = _read_group(path, "ip")
        if got is not None and _usable(*got):
            t_ms, y = got
            scale = 1e-6 if np.nanmedian(np.abs(y)) > 1e4 else 1.0  # A to MA
            return t_ms, np.abs(y) * scale
    return None


def estimate_wmhd(
    betan_t_ms: np.ndarray, betan: np.ndarray, ip_t_ms: np.ndarray, ip_ma: np.ndarray
) -> np.ndarray:
    """W_MHD (J) at beta_N's times as ``WMHD_PER_BETAN_MA * beta_N * I_p``."""
    ok = np.isfinite(ip_ma) & np.isfinite(ip_t_ms)
    return (
        WMHD_PER_BETAN_MA * betan * np.interp(betan_t_ms, ip_t_ms[ok], ip_ma[ok])
    ).astype(np.float32)


def disk_record(
    shot: int,
    raw_dir: Path,
    corpus: Path,
    features: Path,
    *,
    estimate: bool = False,
):
    """The fetch's per-shot arrays (``dens``, ``betan``, ``wmhd``) read from disk, and
    where each came from; a channel with no usable record is left out. With
    ``estimate``, a shot with beta_N and I_p but no W_MHD gets the estimate of
    ``estimate_wmhd``."""
    arrays: dict[str, np.ndarray] = {}
    source: dict[str, str] = {}
    for tag, base in (("corpus", corpus), ("raw", raw_dir)):
        got = _read_group(base / f"{shot}_processed.h5", "co2", CO2_V2_ROW)
        if got is not None and _usable(*got):
            t_ms, y = got
            n = int(np.floor(t_ms[np.isfinite(t_ms)].max())) + 1
            mean, _, _ = zerod.bin_stats(t_ms, y, n)
            arrays["dens"], arrays["dens_t0_ms"] = mean, np.array(0.0)
            source["dens"] = f"{tag}:co2 V2 (DENUF), 1 ms mean"
            break
    sources = {
        "betan": (
            ("features", features / f"{shot}_features.h5"),
            ("raw", raw_dir / f"{shot}_processed.h5"),
        ),
        "wmhd": (("raw", raw_dir / f"{shot}_processed.h5"),),
    }
    for name, where in sources.items():
        for tag, path in where:
            got = _read_group(path, name)
            if got is not None and _usable(*got):
                arrays[f"{name}_t_ms"], arrays[name] = got
                source[name] = f"{tag}:{name}"
                break
    if estimate and "wmhd" not in arrays and "betan" in arrays:
        ip = read_ip(shot, raw_dir, corpus, features)
        if ip is not None:
            arrays["wmhd_t_ms"] = arrays["betan_t_ms"]
            arrays["wmhd"] = estimate_wmhd(arrays["betan_t_ms"], arrays["betan"], *ip)
            source["wmhd"] = (
                f"estimated: {WMHD_PER_BETAN_MA / 1e3:.0f} kJ x beta_N x I_p[MA]"
            )
    return arrays, source


def assemble_disk(
    shot: int,
    zerod_dir: Path,
    *,
    raw_dir: Path = ours.RAW,
    corpus: Path = zerod.CORPUS,
    features: Path = FEATURES,
    estimate: bool = False,
    fetched_dir: Path | None = None,
) -> dict | None:
    """The shot's grid from disk inputs (``zerod.assemble`` on a record in the fetch's
    format), with ``inputs`` naming each sparse channel's source or ``masked``; None
    without D-alpha. A record the fetch already made in ``fetched_dir`` is used as it
    stands, as ``apply`` does."""
    if fetched_dir is not None and (fetched_dir / f"{shot}.npz").exists():
        result = zerod.assemble(shot, raw_dir, fetched_dir, corpus)
        if result is not None:
            result["inputs"] = {
                n: "masked" if n in result["lacks"] else "fetched"
                for n in SPARSE_SOURCES
            }
            result["inputs"]["pinj"] = (
                "masked" if "pinj" in result["lacks"] else "corpus/raw/fetched"
            )
        return result
    arrays, source = disk_record(shot, raw_dir, corpus, features, estimate=estimate)
    zerod_dir.mkdir(parents=True, exist_ok=True)
    target = zerod_dir / f"{shot}.npz"
    target.unlink(missing_ok=True)
    if arrays:
        np.savez(target, **arrays)
    result = zerod.assemble(shot, raw_dir, zerod_dir, corpus)
    if result is None:
        return None
    inputs = {name: source.get(name, "masked") for name in SPARSE_SOURCES}
    inputs["pinj"] = "corpus/raw"
    for name in result["lacks"]:  # a record that left the channel empty after all
        inputs[name] = "masked"
    result["inputs"] = inputs
    return result


def missing_reason(result: dict) -> str | None:
    """Why a shot cannot be read as the network was trained, or None: one of density,
    beta_N and W_MHD has no record (the roster's grids lack one on 3 shots of 839)."""
    lacking = [n for n in SPARSE_SOURCES if result["inputs"][n] == "masked"]
    if not lacking:
        return None
    return (
        f"no {', '.join(lacking)} record on disk (the network was trained with "
        "them; `confinement_zerod_fetch.py` would supply them)"
    )


def segment_grid(grid: np.ndarray, probs: np.ndarray):
    """The segments of one shot and why there are none: ``apply``'s own steps."""
    power = grid[zerod.CHANNELS.index("pinj")] * zerod.SCALES["pinj"]
    if not np.isfinite(power).any():
        return [], "no beam power record"
    active = ours.bridge_gaps(
        np.nan_to_num(power, nan=0.0) >= ours.BEAM_ON_W, ours.BEAM_GAP_MS
    )
    if active.sum() < ours.MIN_SEGMENT_MS:
        return [], "beam power never reaches the threshold"
    found = ours.segment_shot(probs, active)
    if not found:
        return [], (
            f"no segment of {ours.MIN_SEGMENT_MS} ms or more while the beam is on"
        )
    return found, None


def predict(x: np.ndarray, models, device, fold: int | None = None) -> np.ndarray:
    """Class probabilities of one shot: the fold network that never saw it (a curated
    shot), else the mean of all five."""
    from labeler.confinement import unet

    if fold is not None:
        return unet.predict_shot(models[fold], x, device)
    return np.mean([unet.predict_shot(m, x, device) for m in models], axis=0)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_shot_list(path: Path) -> list[int]:
    return sorted({int(s) for s in pd.read_csv(path).shot})


def already_labelled(exclude: Path) -> set[int]:
    """Shots some confinement table the review page opens already labels (a saved,
    format, extend or source table), apart from ``exclude``."""
    from labeler.config import Paths
    from labeler.events.review import silver

    paths = Paths.from_env()
    out: set[int] = set()
    for kind, path in silver._tables(paths, "confinement"):
        if (
            kind in LABELLED_KINDS
            and path.exists()
            and path.resolve() != exclude.resolve()
        ):
            out |= {int(s) for s in pd.read_csv(path, usecols=["shot"]).shot}
    return out


ATTR_COLUMNS = ours.ATTR_COLUMNS
to_extend_table = ours.to_extend_table


def summarise(frame: pd.DataFrame) -> dict:
    """Class mix of the labelled time, and how much of it the floor made uncertain."""
    seconds = (frame.t_end - frame.t_start) / 1000
    names = {1: "H", 2: "L", 3: "QH", 4: "WPQH", 5: "uncertain"}
    under = frame.category == ours.UNCERTAIN
    return {
        "segments": len(frame),
        "label_seconds": round(float(seconds.sum()), 2),
        "segments_under_floor": int(under.sum()),
        "fraction_segments_under_floor": round(float(under.mean()), 3)
        if len(frame)
        else None,
        "fraction_seconds_under_floor": round(
            float(seconds[under].sum() / seconds.sum()), 3
        )
        if len(frame)
        else None,
        "by_category": {
            names[c]: {
                "segments": int((frame.category == c).sum()),
                "seconds": round(float(seconds[frame.category == c].sum()), 2),
            }
            for c in names
        },
        "by_predicted_class": {
            names[c]: {
                "segments": int((frame.predicted == c).sum()),
                "seconds": round(float(seconds[frame.predicted == c].sum()), 2),
            }
            for c in (1, 2, 3, 4)
        },
    }


def select_shots(
    listed: list[int], have: set[int] | None, include: list[int] | None = None
) -> list[int]:
    """The listed shots to run: those no other table labels (all of them when ``have``
    is None), plus the listed shots named in ``include`` whatever ``have`` says."""
    keep = {s for s in listed if have is None or s not in have}
    keep |= set(include or ()) & set(listed)
    return sorted(keep)


def input_summary(inputs: dict[int, dict]) -> dict:
    """Where the density, beta_N and W_MHD of the run's shots came from: a count of
    each source per channel, the shots with any channel not fetched, and a sentence."""
    sources = {n: {} for n in SPARSE_SOURCES}
    for record in inputs.values():
        for name in SPARSE_SOURCES:
            sources[name][record[name]] = sources[name].get(record[name], 0) + 1
    other = sorted(
        shot
        for shot, record in inputs.items()
        if any(record[n] != "fetched" for n in SPARSE_SOURCES)
    )
    if not other:
        statement = (
            f"all {len(inputs)} shots read fetched records "
            "(`confinement_zerod_fetch.py`: \\BCI::DENV2F, EFIT01 beta_N and W_MHD) "
            "for density, beta_N and W_MHD, as the roster does; no surrogate input "
            "(CO2 V2 density, feature-store beta_N, masked or estimated W_MHD) was used"
        )
    else:
        statement = (
            f"{len(other)} of {len(inputs)} shots have a channel that is not a "
            "fetched record (see `sources` and `per_shot[...].inputs`); "
            "`confinement_ours_shots.py check` measures how far disk inputs move "
            "the labels"
        )
    return {
        "sources": sources,
        "shots_not_fetched": other,
        "statement": statement,
    }


def label(args: argparse.Namespace) -> None:
    import torch

    device = torch.device(
        args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    models = ours.mean_models(device, "run_day")
    if len(models) < ours.FOLDS:
        raise SystemExit(f"{len(models)} of {ours.FOLDS} fold models found")
    ids, _, ys, _ = ours.load_curated()
    test_of, _, _ = ours.fold_roles(ids, ys, "run_day")
    last_curated = int(ours.bw.curated_intervals().shot.max())
    out_csv = args.out_dir / f"{args.name}.csv"
    on_list = read_shot_list(args.shots)
    listed = len(on_list)
    strangers = sorted(set(args.include or ()) - set(on_list))
    if strangers:
        raise SystemExit(f"--include shots not in {args.shots}: {strangers}")
    shots = select_shots(
        on_list, already_labelled(out_csv) if args.unlabelled else None, args.include
    )
    rows, missing, inputs = [], {}, {}
    stats = {"curated_held_out": 0, "ensemble": 0}
    for shot in shots:
        result = assemble_disk(
            shot, ZEROD_DISK, fetched_dir=ours.ZEROD if args.use_fetched else None
        )
        if result is None:
            missing[shot] = "no D-alpha record"
            continue
        inputs[shot] = result["inputs"]
        why = missing_reason(result)
        if why is not None and args.missing == "skip":
            missing[shot] = why
            continue
        grid = result["grid"]
        fold = test_of.get(shot)
        probs = predict(zerod.to_input(grid), models, device, fold)
        stats["curated_held_out" if fold is not None else "ensemble"] += 1
        found, why = segment_grid(grid, probs)
        if why is not None:
            missing[shot] = why
        rows += ours.roster_rows(
            shot, found, curated=fold is not None, last_curated=last_curated
        )
    frame = pd.DataFrame(rows, columns=ours.ROSTER_COLUMNS)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = to_extend_table(frame)
    table.to_csv(out_csv, index=False)
    per_shot = {}
    for shot in sorted(inputs):
        part = frame[frame.shot == shot]
        seconds = (part.t_end - part.t_start) / 1000
        per_shot[str(shot)] = {
            "segments": len(part),
            "label_seconds": round(float(seconds.sum()), 2),
            "seconds_by_category": {
                str(c): round(float(seconds[part.category == c].sum()), 2)
                for c in sorted(part.category.unique())
            },
            "inputs": inputs.get(shot),
        }
    summary_inputs = input_summary(inputs)
    surrogate = bool(summary_inputs["shots_not_fetched"])
    checkpoints = {
        f"fold{k}": {
            "path": str(ours.ours_dir("run_day") / f"fold{k}_model.pt"),
            "sha256": sha256(ours.ours_dir("run_day") / f"fold{k}_model.pt"),
        }
        for k in range(ours.FOLDS)
    }
    meta = {
        "category": "confinement",
        "categories": {
            "1": "high (H)",
            "2": "low (L)",
            "3": "qh (QH)",
            "4": "wpqh (WPQH)",
            "5": "uncertain (segment below the confidence floor)",
        },
        "columns": list(table.columns),
        "attrs_keys": ATTR_COLUMNS,
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "made_by": "scripts/labeler/confinement_ours_shots.py label",
        "script_sha256": sha256(Path(__file__)),
        "git_sha": ours.git_sha(),
        "producer": "confine-ours",
        "status": "experimental",
        "model": "1D U-Net (U-Time style) over D-alpha, line density, beta_N, W_MHD "
        "and injected power; 5 folds, each holding whole run days out; "
        "docs/labeler/confinement_ours.md",
        "checkpoints": checkpoints,
        "shot_list": {
            "file": str(args.shots),
            "shots_listed": listed,
            "shots_run": len(shots),
            "filter": "shots no saved, format, extend or source confinement table "
            "labels (the roster's roster.csv and the curated regimes source)"
            if args.unlabelled
            else "none",
            "shots": shots,
            **(
                {
                    "included": sorted(args.include),
                    "included_because": args.include_reason,
                }
                if args.include
                else {}
            ),
        },
        "inputs": {
            "no_fetch": "this script fetches nothing from MDSplus; a channel comes "
            "from a record `confinement_zerod_fetch.py` already made, else from the "
            "corpus, the raw cache or the feature store",
            "used": summary_inputs,
            "da_and_pinj": "as for the roster (zerod.assemble: corpus, else raw cache)",
            "dens": "a fetched \\BCI::DENV2F record if there is one, else row 2 of the "
            "co2 group (V2 chord, DENUF, the same line integral unfiltered), 1 ms mean",
            "betan": "a fetched EFIT01 record, else the feature store's betan group, "
            "else the raw cache's",
            "wmhd": "a fetched EFIT01 record, else the raw cache's wmhd group; there "
            "is no other source on disk",
            "missing_channel": "skip: a shot lacking a record of density, beta_N or "
            "W_MHD is not labelled (the network read all three on 836 of 839 roster "
            "grids; `check` shows masking W_MHD changes the class of about a "
            "fifth of the beam-on bins and an estimate from beta_N and I_p is worse "
            "than masking it)"
            if args.missing == "skip"
            else "mask: a shot lacking a record of density, beta_N or W_MHD is run "
            "with that channel masked; `check` shows how far that moves the "
            "labels",
            "check": "`confinement_ours_shots.py check` compares labels from disk inputs "
            "with those from fetched ones",
        },
        "input_check": json.loads(args.input_check.read_text())
        if args.input_check
        else summary_inputs,
        "segmentation": {
            "bins_ms": 1.0,
            "active": f"total beam power >= {ours.BEAM_ON_W:.0f} W, gaps of at most "
            f"{ours.BEAM_GAP_MS} ms bridged",
            "mode_filter_bins": ours.MODE_BINS,
            "min_segment_ms": ours.MIN_SEGMENT_MS,
            "confidence": "mean probability of the segment's class over its bins",
            "confidence_floor": f"a segment under {ours.CONFIDENCE_FLOOR} is category "
            f"{ours.UNCERTAIN} (uncertain); `predicted` keeps its class",
        },
        "columns_meaning": {
            "attrs": "a JSON object with the roster's four extra columns, predicted, "
            "source, tier and extrapolated (the review page's table reader accepts "
            "no column beyond `attrs`, so roster.csv's own extra columns make it "
            "skip that table); the keys are explained below",
            "category": "1 high, 2 low, 3 qh, 4 wpqh, 5 uncertain (under the floor)",
            "predicted": "the segment's class, before the confidence floor",
            "source": "held_out: a curated shot, read by the fold model that never "
            "saw it; ensemble: any other shot, the mean of the five fold models",
            "tier": "unreviewed: a QH or WPQH segment on a shot outside the curated "
            "set (the curated set may not contain ELM-free and other quiescent "
            "H-modes the network could read as QH (unverified)); model: the rest",
            "extrapolated": f"true for a shot past the last curated one "
            f"({last_curated}); the network never saw those campaigns",
        },
        "shots_labelled": int(frame.shot.nunique()),
        "shots_missing": {
            "count": len(missing),
            "reasons": {str(k): v for k, v in sorted(missing.items())},
        },
        "prediction_source": {
            "curated shots": "the fold model that never saw the shot",
            "other shots": "mean of the five fold models",
            **stats,
        },
        "summary": summarise(frame),
        "per_shot": per_shot,
        "caveats": [
            (
                "Unreviewed: nothing here has been read by a person; the labels are "
                "for the review page to open the shots on."
            ),
            (
                "Active (beam-on) bins only; time without beam power is not "
                "labelled, and not low."
            ),
            (
                "Segments under the 0.7 floor are category 5; on the roster 31 % of "
                "segments were, here see `summary.fraction_segments_under_floor`."
            ),
            (
                "Inputs differ from the roster's where no fetched record exists (see "
                "`inputs` and `per_shot[...].inputs`): density from the unfiltered "
                "CO2 V2 trace (95 % of beam-on bins keep their class against the "
                "fetched density on 120 roster shots), beta_N from the feature store "
                "(99 %)."
                if surrogate
                else "Every shot's density, beta_N and W_MHD are fetched records, as "
                "for the roster; no surrogate input."
            ),
            (
                "Shots past 196493 are flagged extrapolated; no fold held a campaign "
                "out."
            ),
        ],
    }
    out_csv.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    s = meta["summary"]
    print(
        f"{s['segments']} segments on {frame.shot.nunique()} of {len(shots)} shots "
        f"({listed} listed); {len(missing)} missing; "
        f"{s['segments_under_floor']} under the floor; {s['label_seconds']} s"
    )
    for shot, why in sorted(missing.items()):
        print(f"  missing {shot}: {why}")


def _labels(probs: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Per-bin class after the mode filter, -1 where the beam is off."""
    return np.where(active, ours.mode_filter(probs.argmax(0), ours.MODE_BINS), -1)


def _era(shot: int) -> str:
    for lo, name in ((190000, "190000+"), (180000, "180000s"), (170000, "170000s")):
        if shot >= lo:
            return name
    return "before 170000"


VARIANTS = {
    "disk": "dens from co2 V2, betan from features/raw, W_MHD from raw else masked",
    "disk_wmhd_estimated": "the same with W_MHD estimated from beta_N and I_p "
    "where the raw cache has none",
    "dens_from_co2_only": "fetched inputs with only the density from co2 V2",
    "betan_from_disk_only": "fetched inputs with only beta_N from features/raw",
    "wmhd_estimated_only": "fetched inputs with only W_MHD replaced by the estimate "
    "(from the fetched beta_N)",
    "wmhd_masked_only": "fetched inputs with only W_MHD masked",
    "all_masked": "fetched inputs with density, beta_N and W_MHD all masked",
}


def check(args: argparse.Namespace) -> None:
    import torch

    device = torch.device(
        args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    models = ours.mean_models(device, "run_day")
    ids, _, ys, _ = ours.load_curated()
    test_of, _, _ = ours.fold_roles(ids, ys, "run_day")
    roster = sorted(int(s) for s in pd.read_csv(ours.ROSTER).shot)
    rng = np.random.default_rng(ours.SEED)
    pool = [
        s for s in roster if s not in test_of and (ours.GRIDS / f"{s}.npz").exists()
    ]
    rng.shuffle(pool)
    args.scratch.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="check_", dir=args.scratch))
    chan = zerod.CHANNELS
    row = {n: chan.index(n) for n in SPARSE_SOURCES}
    kept = []
    totals = {v: {"bins": 0, "same": 0, "hl_same": 0} for v in VARIANTS}
    by_era: dict[str, dict] = {}
    floor_time = {v: [0.0, 0.0] for v in ("fetched", *VARIANTS)}
    for shot in pool:
        if len(kept) >= args.sample:
            break
        full, _ = zerod.load_grid(ours.GRIDS / f"{shot}.npz")
        if not all(np.isfinite(full[r]).any() for r in row.values()):
            continue
        disk = assemble_disk(shot, scratch)
        est = assemble_disk(shot, scratch, estimate=True)
        if (
            disk is None
            or est is None
            or "masked"
            in (
                disk["inputs"]["dens"],
                disk["inputs"]["betan"],
            )
        ):
            continue
        ip = read_ip(shot, ours.RAW, zerod.CORPUS, FEATURES)
        if ip is None:
            continue
        with np.load(ours.WORK / "zerod" / f"{shot}.npz") as d:
            fetched_betan = (d["betan_t_ms"], d["betan"])
        grids = {
            "fetched": full,
            "disk": disk["grid"],
            "disk_wmhd_estimated": est["grid"],
        }
        for name in ("dens_from_co2_only", "betan_from_disk_only"):
            grids[name] = full.copy()
        grids["dens_from_co2_only"][row["dens"]] = disk["grid"][row["dens"]]
        grids["betan_from_disk_only"][row["betan"]] = disk["grid"][row["betan"]]
        grids["wmhd_estimated_only"] = full.copy()
        w = estimate_wmhd(*fetched_betan, *ip)
        grids["wmhd_estimated_only"][row["wmhd"]] = (
            zerod.interpolate_sparse(fetched_betan[0], w, full.shape[1])
            / zerod.SCALES["wmhd"]
        )
        grids["wmhd_masked_only"] = full.copy()
        grids["wmhd_masked_only"][row["wmhd"]] = np.nan
        grids["all_masked"] = full.copy()
        grids["all_masked"][list(row.values())] = np.nan
        power = full[chan.index("pinj")] * zerod.SCALES["pinj"]
        active = ours.bridge_gaps(
            np.nan_to_num(power, nan=0.0) >= ours.BEAM_ON_W, ours.BEAM_GAP_MS
        )
        if active.sum() < ours.MIN_SEGMENT_MS:
            continue
        probs = {}
        for name, grid in grids.items():
            n = min(grid.shape[1], full.shape[1])
            p = predict(zerod.to_input(grid[:, :n]), models, device)
            probs[name] = np.pad(p, ((0, 0), (0, full.shape[1] - n)))
        base = _labels(probs["fetched"], active)
        for name in grids:
            segs = ours.segment_shot(probs[name], active)
            span = sum(hi - lo for _, lo, hi, _ in segs)
            low = sum(hi - lo for _, lo, hi, cf in segs if cf < ours.CONFIDENCE_FLOOR)
            floor_time[name][0] += low
            floor_time[name][1] += span
        for name in VARIANTS:
            other = _labels(probs[name], active)
            same = int((other[active] == base[active]).sum())
            t = totals[name]
            t["bins"] += int(active.sum())
            t["same"] += same
            t["hl_same"] += int(((other[active] == 0) == (base[active] == 0)).sum())
            if name in ("disk", "disk_wmhd_estimated"):
                era = by_era.setdefault(
                    f"{name} {_era(shot)}", {"shots": 0, "bins": 0, "same": 0}
                )
                era["shots"] += 1
                era["bins"] += int(active.sum())
                era["same"] += same
        kept.append(shot)
    summary = {
        "what": "labels of N roster shots outside the curated set (five-network mean) "
        "from the fetched traces, against the same from disk inputs",
        "shots": len(kept),
        "variants": VARIANTS,
        "wmhd_estimate": f"{WMHD_PER_BETAN_MA / 1e3:.0f} kJ x beta_N x I_p[MA]",
        "per_bin_agreement_with_fetched": {
            v: {
                "same_class": round(t["same"] / max(t["bins"], 1), 4),
                "same_l_or_h_family": round(t["hl_same"] / max(t["bins"], 1), 4),
                "active_bins": t["bins"],
            }
            for v, t in totals.items()
        },
        "agreement_by_era": {
            k: {
                "shots": v["shots"],
                "same_class": round(v["same"] / max(v["bins"], 1), 4),
            }
            for k, v in sorted(by_era.items())
        },
        "fraction_of_segment_time_under_floor": {
            v: round(t[0] / max(t[1], 1), 4) for v, t in floor_time.items()
        },
        "note": "agreement is per beam-on 1 ms bin after the mode filter; "
        "same_l_or_h_family counts L against any H-class (H, QH, WPQH)",
    }
    shutil.rmtree(scratch)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    lb = sub.add_parser("label")
    lb.add_argument("--shots", type=Path, required=True, help="CSV with a shot column")
    lb.add_argument("--name", required=True, help="output stem in extend_confine_ours")
    lb.add_argument(
        "--unlabelled",
        action="store_true",
        help="only shots no other confinement table labels",
    )
    lb.add_argument(
        "--include",
        type=int,
        nargs="+",
        help="listed shots to label although another table already does (with "
        "--unlabelled)",
    )
    lb.add_argument(
        "--include-reason", default="", help="why, for the meta (with --include)"
    )
    lb.add_argument(
        "--missing",
        choices=("skip", "mask"),
        default="skip",
        help="a shot with no record of density, beta_N or W_MHD: skip it, or run it "
        "with the channel masked (`check` measures what masking costs)",
    )
    lb.add_argument(
        "--no-fetched",
        dest="use_fetched",
        action="store_false",
        help="ignore records `confinement_zerod_fetch.py` made",
    )
    lb.add_argument(
        "--out-dir",
        type=Path,
        default=ours.EXTEND,
        help="where the table and its meta go (default: extend_confine_ours)",
    )
    lb.add_argument("--input-check", type=Path, help="JSON written by `check`")
    lb.add_argument("--device", help="default: cuda:0 if there is one, else cpu")
    ck = sub.add_parser("check")
    ck.add_argument("--sample", type=int, default=100)
    ck.add_argument("--out", type=Path, required=True)
    ck.add_argument(
        "--scratch", type=Path, required=True, help="directory for a temporary folder"
    )
    ck.add_argument("--device", help="default: cuda:0 if there is one, else cpu")
    args = ap.parse_args(argv)
    {"label": label, "check": check}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
