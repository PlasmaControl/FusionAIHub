#!/usr/bin/env python
"""ELM presence on a 1 ms grid for every DIII-D plasma shot FDP serves (the ELM twin of
``ae_fdp_label.py``, whose candidates, logs, status rules, grid format and coverage it shares).

The inputs are those of ``elm-ours`` (``labeler.elm.inputs``): the filterscopes
``\\SPECTROSCOPY::FS02``-``FS04`` and the fast CO2 chords ``\\BCI::DENV2F`` and ``DENV3F``.
They are fetched as byte ranges of FDP's MDSplus tree files (``elm_mds_ranges.py``), about
40 MB a shot instead of 250-400 MB of whole trees.

Subcommands, each resumable (a shot already done is skipped):

``stage`` (login node, fdp env under ``fdp -D d3d run``): for each candidate with an FDP view,
    the record bytes the five nodes need into ``stage/<shot>/`` (sparse tree files), in worker
    processes (MDSplus finds trees through process-wide ``<tree>_path`` variables). Every
    ``--batch`` staged shots it writes ``batches/<k>.csv`` and, with ``--submit``, submits the
    extract job (and, with ``--models``, the infer job after it). It pauses while ``stage/``
    holds more than ``--budget-gb``. Stops at the first refused token.
``extract`` (CPU job, fdp env, no network): ``stage/<shot>/`` -> ``inputs/<shot>.npz``, the two
    records on the 0.1 ms grid from -50 ms (FS02-04 maximum and DENV2F/DENV3F mean per cell,
    as ``labeler.elm.inputs.cells``; ~2 MB). The staged folder is then deleted. The inputs are
    kept, so a later model reruns every shot without fetching.
``infer`` (GPU job, stellarai env): ``inputs/<shot>.npz`` -> ``work/<shot>.npz`` (event
    probability per ms) and the 1 ms calls in ``results/<job>.{shots,spans}.csv``, with the
    models of ``--models`` (a run folder in the ``labeler.elm.serve.load_models`` layout:
    ``fold<k>/model.pt``, ``fold<k>/fold.json`` with its threshold, ``run.json``). One shot at
    a time on its whole record: the network's GroupNorm sees the whole input, as in serving.
``publish`` (login node): the label folder ``--out``, as ``ae_fdp_label.py publish`` (the
    deployment for the all-shot root, ``dev/edge_localized_mode/<root name>`` for any other).
Coverage: ``ae_fdp_label.py coverage`` reads the published folder, not this root.

**1 ms calls.** Millisecond ``k`` (``[k, k + 1)`` ms) is present when the models' mean event
probability is at least the mean of their thresholds. The assessed window ``[window_start_ms,
window_end_ms)`` runs from the first to the last millisecond in ``[0, plasma end)`` whose ten
grid cells all hold samples of both records; inside it a frame that is not in a span is absent
(a gap in the record included), outside it unassessed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ae_fdp_label as ae  # the shared pipeline code
import elm_mds_ranges as R

REPO = ae.REPO
sys.path.insert(0, str(REPO / "src"))
LABELER_ROOT = ae.LABELER_ROOT
ALL_SHOT_ROOT = LABELER_ROOT / "elm_fdp_1ms"
ROOT = Path(os.environ.get("ELM_FDP_ROOT", ALL_SHOT_ROOT))
OUT = ae.DEPLOYMENT / "edge_localized_mode" / "elm_fdp_1ms"
CANDIDATES = LABELER_ROOT / "ae_fdp_1ms/candidates.csv"

FS_TAGS = [r"\SPECTROSCOPY::FS02", r"\SPECTROSCOPY::FS03", r"\SPECTROSCOPY::FS04"]
DENSITY_TAGS = [r"\BCI::DENV2F", r"\BCI::DENV3F"]
TAGS = {"spectroscopy": FS_TAGS, "bci": DENSITY_TAGS}
# elm-ours saw 50 kHz filterscopes and 100 kHz chords; under 20 kHz an ELM is a sample or two. 20 kHz
# records (the 150000s: 19980.5 Hz, 0.05005 ms) still put a sample in every 0.1 ms cell, so they pass.
MAX_FS_DT_MS = 0.0505
MAX_DENSITY_DT_MS = 0.0505
KEEP_STAGED = {"read_failed", "extract_error", "no_staged_file"}  # restaged or looked at; the rest is final
#: Stage findings a rerun would repeat (the extract and infer ones are in ae.RETRY's complement).
#: ptdata_record: a record is a PTDATA call (old shots), which the MDSplus files cannot serve;
#: such shots need the PTDATA view (not built yet).
FINAL_STAGE = {"no_view", "no_fs", "no_density", "ptdata_record"}


def _set_root(root: Path) -> None:
    global ROOT
    ROOT = Path(root)
    ae.ROOT = ROOT  # the shared log, status and result readers


def stage_dir_of(shot: int) -> Path:
    return ROOT / "stage" / str(shot)


def held_gb() -> float:
    """Bytes really on disk under stage/ (the datafiles are sparse).

    Extract jobs delete shot folders while this runs, so any entry may vanish mid-listing: os.walk
    raised NotADirectoryError on one (2026-11-03) and stopped the staging. Every error is skipped."""
    total = 0
    pending = [str(ROOT / "stage")]
    while pending:
        try:
            with os.scandir(pending.pop()) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            pending.append(e.path)
                        else:
                            total += e.stat(follow_symlinks=False).st_blocks * 512
                    except OSError:
                        pass
        except OSError:
            pass
    return total / 1e9


# ------------------------------------------------------------------ stage (login node)


def stage_one(shot: int, root: str, whole: bool) -> dict:
    """One shot's sparse trees (a worker process: its own MDSplus tree paths)."""
    _set_root(Path(root))
    dest = stage_dir_of(shot)
    if (dest / "staged.json").exists() and not whole:
        info = json.loads((dest / "staged.json").read_text())
        return {"shot": shot, "status": "staged", "bytes": info["bytes"], "s": 0.0}
    part = dest.with_name(f".{shot}.part")
    shutil.rmtree(part, ignore_errors=True)
    t0 = time.monotonic()
    try:
        info = R.stage_nodes(shot, part, TAGS, whole=whole)
    except R.AuthError:
        shutil.rmtree(part, ignore_errors=True)
        raise
    except Exception as error:  # noqa: BLE001 - network faults: retried next run
        shutil.rmtree(part, ignore_errors=True)
        return {"shot": shot, "status": "failed", "error": f"{type(error).__name__}: {error}"[-300:]}
    miss = info["missing"]
    if info.get("ptdata"):
        shutil.rmtree(part, ignore_errors=True)
        return {"shot": shot, "status": "ptdata_record", "error": "; ".join(info["ptdata"])[:300]}
    for tags, status in ((FS_TAGS, "no_fs"), (DENSITY_TAGS, "no_density")):
        if any(t in miss for t in tags):
            shutil.rmtree(part, ignore_errors=True)
            return {"shot": shot, "status": status,
                    "error": "; ".join(f"{t}: {miss[t]}" for t in tags if t in miss)[:300]}
    shutil.rmtree(dest, ignore_errors=True)  # a sparse staging being replaced by whole files
    os.replace(part, dest)
    return {"shot": shot, "status": "staged", "bytes": info["bytes"], "whole": info["whole"],
            "s": round(time.monotonic() - t0, 1)}


def submit_batch(k: int, rows: list[tuple[int, float]], models: str | None) -> list[str]:
    (ROOT / "batches").mkdir(parents=True, exist_ok=True)
    path = ROOT / "batches" / f"b{k:04d}.csv"
    ae.write_list(path, rows)
    env = {"LIST": str(path), "ELM_FDP_ROOT": str(ROOT)}
    ext = ae.sbatch(REPO / "scripts/labeler/elm_fdp_extract.sbatch", env)
    ids = [ext]
    note = f"batch {k}: {len(rows)} shots -> extract {ext}"
    if models:
        inf = ae.sbatch(REPO / "scripts/labeler/elm_fdp_infer.sbatch", {**env, "MODELS": str(models)},
                        dependency=f"afterany:{ext}")
        ids.append(inf)
        note += f", infer {inf}"
    ae.log("jobs", batch=k, list=str(path), shots=len(rows), jobs=ids)
    print(note, flush=True)
    return ids


def cmd_stage(args) -> None:
    cands = ae.read_list(Path(args.candidates))
    if args.list:
        want = {s for s, _ in ae.read_list(Path(args.list))}
        cands = [c for c in cands if c[0] in want]
    if args.shots:
        want = set(args.shots)
        cands = [c for c in cands if c[0] in want]
    staged = ae.last_status("stage")
    extracted = ae.last_status("extract")
    inputs_dir = ROOT / "inputs"
    have = {int(p.stem) for p in inputs_dir.glob("*.npz")} if inputs_dir.exists() else set()

    def needed(shot: int) -> bool:
        if shot in have:
            return False
        st = staged.get(shot, {}).get("status")
        ex = extracted.get(shot, {}).get("status")
        if args.whole:  # the shots whose sparse read failed
            return ex == "read_failed"
        if st in FINAL_STAGE:
            return False
        if ex is not None and ex not in ae.RETRY:
            return False  # extracted to a final finding (a rate screen, no record in the window)
        return True

    todo = [c for c in cands if needed(c[0])]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} shots to stage of {len(cands)} candidates; {args.procs} processes; "
          f"batch {args.batch}; budget {args.budget_gb} GB{'; whole datafiles' if args.whole else ''}",
          flush=True)
    (ROOT / "stage").mkdir(parents=True, exist_ok=True)
    (ROOT / "batches").mkdir(parents=True, exist_ok=True)
    views = ae.Views()
    end_of = dict(todo)
    k = len(list((ROOT / "batches").glob("b*.csv")))
    batch: list[tuple[int, float]] = []
    counts: dict[str, int] = {}
    t_start, nbytes = time.monotonic(), 0

    def flush(force: bool = False) -> None:
        nonlocal batch, k
        if batch and (force or len(batch) >= args.batch):
            if args.submit:
                submit_batch(k, batch, args.models)
            else:
                ae.write_list(ROOT / "batches" / f"b{k:04d}.csv", batch)
                print(f"batch {k}: {len(batch)} shots written (not submitted)", flush=True)
            k += 1
            batch = []
            if args.publish:  # the batches whose infer jobs have finished; never stops staging
                try:
                    cmd_publish(argparse.Namespace(candidates=args.candidates, models=args.models, out=args.out))
                except Exception as error:  # noqa: BLE001
                    print("publish failed:", error, flush=True)

    def record(row: dict) -> None:
        nonlocal nbytes
        ae.log("stage", **row)
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        if row["status"] == "staged":
            batch.append((row["shot"], end_of[row["shot"]]))
            nbytes += row.get("bytes", 0)
        done = sum(counts.values())
        if done % 50 == 0:
            rate = done / max(time.monotonic() - t_start, 1)
            print(f"{done}/{len(todo)} {counts} | {nbytes / 1e9:.1f} GB fetched | "
                  f"{rate * 3600:.0f} shots/h", flush=True)
        flush()

    pending: dict = {}
    try:
        with ProcessPoolExecutor(args.procs) as pool:
            for shot, _ in todo:
                try:
                    has = views.has(shot)
                except ae.AuthError as error:
                    raise R.AuthError(str(error)) from None
                except Exception as error:  # noqa: BLE001 - a listing that keeps failing
                    record({"shot": shot, "status": "failed", "error": f"view listing: {error}"[-300:]})
                    continue
                if not has:
                    record({"shot": shot, "status": "no_view"})
                    continue
                while len(pending) >= args.procs:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for fut in done:
                        pending.pop(fut)
                        record(fut.result())  # an AuthError propagates from here
                while held_gb() > args.budget_gb:
                    time.sleep(60)
                pending[pool.submit(stage_one, shot, str(ROOT), args.whole)] = shot
            for fut in list(pending):
                pending.pop(fut)
                record(fut.result())
    except R.AuthError as error:
        ae.log("stage", event="auth_error", error=str(error)[:300])
        print("FDP REFUSED, stopping (an expired login: run `fdp login`; then rerun):", error, flush=True)
        flush(force=True)
        raise SystemExit(3) from None
    flush(force=True)
    print(f"staging done: {counts} in {(time.monotonic() - t_start) / 60:.1f} min, "
          f"{nbytes / 1e9:.1f} GB fetched", flush=True)


# ------------------------------------------------------------------ extract (CPU job)


def _reduce(records: list[tuple[np.ndarray, np.ndarray]], n: int, how: str):
    """Each record on the grid on its own timebase; a cell counts where every record has samples."""
    from labeler.elm import inputs

    rows, counts = [], []
    for y, t in records:
        r, c = inputs.block_reduce(t, y[None, :], n, how)
        rows.append(r[0])
        counts.append(c)
    count = np.min(np.stack(counts), axis=0)
    return np.stack(rows), np.minimum(count, 255).astype(np.uint8)


def extract_one(job: tuple) -> dict:
    """``_extract_one``; then the stage folder goes unless the shot will be restaged (or ``keep``)."""
    shot, end_ms, keep, root = job
    row = _extract_one(shot, end_ms, root)
    dest = stage_dir_of(shot)
    if not keep and row["status"] not in KEEP_STAGED and ae.inside(dest, ROOT / "stage"):
        shutil.rmtree(dest, ignore_errors=True)
    return row


def _extract_one(shot: int, end_ms: float, root: str) -> dict:
    _set_root(Path(root))
    from labeler.elm import inputs

    dest = stage_dir_of(shot)
    row = {"shot": shot}
    if not (dest / "staged.json").exists():
        return {**row, "status": "no_staged_file"}
    try:
        got = R.read_nodes(shot, dest, TAGS)
        bad = {t: v[0] for t, v in got.items() if len(v) == 1}
        if bad:
            # NODATA is the node's own finding (record length 0 in the fully fetched
            # .characteristics), not a missed byte range
            broken = [v for v in map(str, bad.values())
                      if ("read failed" in v or "tree open failed" in v) and "NODATA" not in v]
            if broken:
                status = "read_failed"
            else:
                status = "no_fs" if any(t in FS_TAGS for t in bad) else "no_density"
            return {**row, "status": status, "error": "; ".join(f"{t}: {v}" for t, v in bad.items())[:300]}

        def record(tag):
            y, t = np.asarray(got[tag][0], np.float64).ravel(), np.asarray(got[tag][1], np.float64).ravel()
            return y, t

        fs, den = [record(t) for t in FS_TAGS], [record(t) for t in DENSITY_TAGS]
        for name, recs in (("fs", fs), ("density", den)):
            for y, t in recs:
                if y.size < 2 or y.shape != t.shape:
                    return {**row, "status": "bad_record", "error": f"{name}: data {y.shape}, times {t.shape}"}
        fs_dt = max(float(np.median(np.diff(t))) for _, t in fs)
        den_dt = max(float(np.median(np.diff(t))) for _, t in den)
        if fs_dt > MAX_FS_DT_MS:
            return {**row, "status": "fs_rate_low", "fs_hz": round(1e3 / fs_dt, 1)}
        if den_dt > MAX_DENSITY_DT_MS:
            return {**row, "status": "density_rate_low", "density_hz": round(1e3 / den_dt, 1)}
        last = min(float(t[-1]) for _, t in fs + den)
        n = int(np.floor(last - inputs.GRID0_MS)) * inputs.CELLS_PER_MS  # inputs.grid_length
        if n <= 0:
            return {**row, "status": "no_record_in_window"}
        fmax, fcount = _reduce(fs, n, "max")
        dmean, dcount = _reduce(den, n, "mean")
        c0 = int(-inputs.GRID0_MS) * inputs.CELLS_PER_MS  # the cell of t = 0
        c1 = min(n, c0 + int(np.floor(end_ms)) * inputs.CELLS_PER_MS)
        if c1 <= c0 or not ((fcount[c0:c1] > 0) & (dcount[c0:c1] > 0)).any():
            return {**row, "status": "no_record_in_window", "record_end_ms": last}
        out = ROOT / "inputs" / f"{shot}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{shot}.{os.getpid()}.npz.tmp")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, fmax=fmax.astype(np.float32), fcount=fcount,
                                dmean=dmean.astype(np.float32), dcount=dcount, plasma_end_ms=float(end_ms),
                                grid0_ms=inputs.GRID0_MS, dt_ms=inputs.DT_MS, fs_hz=1e3 / fs_dt,
                                density_hz=1e3 / den_dt, tags=np.array(FS_TAGS + DENSITY_TAGS))
        os.replace(tmp, out)
        return {**row, "status": "ok", "fs_hz": round(1e3 / fs_dt, 1), "density_hz": round(1e3 / den_dt, 1),
                "n_cells": n, "bytes": out.stat().st_size}
    except Exception as error:  # noqa: BLE001 - recorded, retried next run
        return {**row, "status": "extract_error", "error": f"{type(error).__name__}: {error}"[:300]}


def cmd_extract(args) -> None:
    from multiprocessing import Pool

    rows = ae.read_list(Path(args.list))
    done = ae.last_status("extract")

    def needed(shot: int) -> bool:
        if (ROOT / "inputs" / f"{shot}.npz").exists():
            return False
        return done.get(shot, {}).get("status") in (None, "ok", *ae.RETRY, *args.redo)

    jobs = [(s, e, args.keep_inputs, str(ROOT)) for s, e in rows if needed(s)]
    print(f"extract {len(jobs)} of {len(rows)} shots with {args.workers} workers at {ae.now()}", flush=True)
    t0 = time.monotonic()
    counts: dict[str, int] = {}
    with Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(extract_one, jobs, chunksize=1), 1):
            ae.log("extract", job=os.environ.get("SLURM_JOB_ID", "local"), **row)
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            if i % 100 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} {counts} {(time.monotonic() - t0):.0f} s", flush=True)
    print(f"extract done {counts} in {time.monotonic() - t0:.0f} s", flush=True)


# ------------------------------------------------------------------ infer (GPU job)


def stamp() -> str:
    return datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")


def window_calls(prob: np.ndarray, valid_ms: np.ndarray, end_ms: float, threshold: float):
    """(w0, w1, call) in ms of the shot's clock: ``call[k - w0]`` for k in [w0, w1). ``prob`` and
    ``valid_ms`` are per ms index i = k - GRID0_MS."""
    from labeler.elm import inputs

    off = int(-inputs.GRID0_MS)
    n = min(prob.size, valid_ms.size)
    k1 = min(int(np.floor(end_ms)), n - off)
    ok = valid_ms[off:off + max(k1, 0)]
    idx = np.flatnonzero(ok)
    if not idx.size:
        return 0, 0, np.zeros(0, bool)
    w0, w1 = int(idx[0]), int(idx[-1]) + 1
    call = (prob[off + w0:off + w1] >= threshold) & ok[w0:w1]
    return w0, w1, call


def cmd_infer(args) -> None:
    import torch
    from torch.utils.data import DataLoader, Dataset

    from labeler.elm import inputs, net, serve

    in_dir, work, res_dir = ROOT / "inputs", ROOT / "work", ROOT / "results"
    work.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    if args.list:
        shots = [s for s, _ in ae.read_list(Path(args.list))]
    else:
        shots = sorted(int(p.stem) for p in in_dir.glob("*.npz"))
    todo = [s for s in shots if (in_dir / f"{s}.npz").exists() and not (work / f"{s}.npz").exists()]
    todo = todo[args.part::args.parts]
    tag = os.environ.get("SLURM_JOB_ID", f"local{os.getpid()}")
    if args.parts > 1:
        tag = f"{tag}p{args.part}"
    print(f"infer {len(todo)} of {len(shots)} listed shots, job {tag}, {args.workers} workers at {ae.now()}",
          flush=True)
    if not todo:
        return

    class Inputs(Dataset):
        def __len__(self):
            return len(todo)

        def __getitem__(self, i):
            shot = todo[i]
            try:
                with np.load(in_dir / f"{shot}.npz") as z:
                    x = inputs.channels_from_cells(z["fmax"].astype(np.float64), z["fcount"],
                                                   z["dmean"].astype(np.float64), z["dcount"])
                    meta = {k: float(z[k]) for k in ("plasma_end_ms", "fs_hz", "density_hz")}
            except Exception as error:  # noqa: BLE001
                return {"shot": shot, "skip": f"input unreadable: {type(error).__name__}: {error}"}
            n = x.shape[1]
            buf = np.zeros((x.shape[0], net.pad_to(n)), np.float32)
            buf[:, :n] = x
            valid_ms = x[inputs.VALID, : n // inputs.CELLS_PER_MS * inputs.CELLS_PER_MS]
            valid_ms = valid_ms.reshape(-1, inputs.CELLS_PER_MS).all(axis=1)
            return {"shot": shot, "x": torch.from_numpy(buf), "n_ms": n // inputs.CELLS_PER_MS,
                    "valid_ms": torch.from_numpy(valid_ms), **meta}  # the loader would make it a tensor anyway

    device = torch.device(args.device)
    models, thresholds = serve.load_models(Path(args.models), device)
    threshold = float(np.mean(thresholds))
    loader = DataLoader(Inputs(), batch_size=None, num_workers=args.workers,
                        prefetch_factor=4 if args.workers else None, pin_memory=device.type == "cuda")
    spans_csv, shots_csv = res_dir / f"{tag}.spans.csv", res_dir / f"{tag}.shots.csv"
    new = not shots_csv.exists()
    f_spans, f_shots = open(spans_csv, "a"), open(shots_csv, "a")
    if new:
        f_spans.write("shot,t_start_ms,t_end_ms\n")
        f_shots.write("shot,status,window_start_ms,window_end_ms,present_ms,n_spans,plasma_end_ms,"
                      "fs_hz,density_hz,mean_prob,job,error\n")
    print("LOOP_START", stamp(), flush=True)
    t_loop, ran, warm = time.perf_counter(), 0, False
    with torch.inference_mode():
        for item in loader:
            shot = int(item["shot"])
            if "skip" in item:
                f_shots.write(f"{shot},infer_failed,,,,,,,,,{tag},\"{ae.clean(item['skip'])}\"\n")
                continue
            try:
                xb = item["x"].to(device, non_blocking=True)[None]
                prob = torch.stack([torch.sigmoid(m(xb))[0, 0] for m in models]).mean(0)
                prob = prob[: item["n_ms"]].float().cpu().numpy()
            except (RuntimeError, ValueError) as error:
                f_shots.write(f"{shot},infer_failed,,,,,,,,,{tag},"
                              f"\"{ae.clean(f'{type(error).__name__}: {error}')}\"\n")
                continue
            end = item["plasma_end_ms"]
            w0, w1, call = window_calls(prob, item["valid_ms"].numpy(), end, threshold)
            spans = [(w0 + a, w0 + b) for a, b in ae.runs(call)]
            off = int(-inputs.GRID0_MS)
            mean_prob = float(prob[off + w0:off + w1].mean()) if w1 > w0 else float("nan")
            tmp = work / f".{shot}.{os.getpid()}.npz.tmp"
            with open(tmp, "wb") as fh:
                np.savez(fh, prob=prob.astype(np.float16), t_first_ms=inputs.GRID0_MS, hop_ms=1.0,
                         threshold=threshold, window_ms=np.array([w0, w1]),
                         spans_ms=np.array(spans, dtype=np.int32).reshape(-1, 2),
                         plasma_end_ms=end, models=str(args.models), job=tag)
            os.replace(tmp, work / f"{shot}.npz")
            for a, b in spans:
                f_spans.write(f"{shot},{a},{b}\n")
            present = int(call.sum())
            f_shots.write(f"{shot},labeled,{w0},{w1},{present},{len(spans)},{end:g},{item['fs_hz']:.1f},"
                          f"{item['density_hz']:.1f},{mean_prob:.4f},{tag},\n")
            f_spans.flush()
            f_shots.flush()
            ran += 1
            if not warm:
                print("LOOP_WARM", stamp(), flush=True)
                warm = True
            if ran % 500 == 0 or ran <= 3:
                print(shot, f"{w0}-{w1} ms |", len(spans), "spans |", present, "ms present", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("LOOP_END", stamp(), flush=True)
    f_spans.close()
    f_shots.close()
    wall = time.perf_counter() - t_loop
    print(f"{ran} shots in {wall:.1f} s ({wall / max(ran, 1):.3f} s a shot); threshold {threshold:.3f}",
          flush=True)


# ------------------------------------------------------------------ publish (login node)


def _git_sha() -> str:
    import subprocess

    r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                       check=False)
    return r.stdout.strip() or "unknown"


def _evaluation(models: Path | None) -> dict | None:
    """The tier gate of the run (``elm_tiers/evaluation.json`` next to ``models/<role>``)."""
    f = models.parent.parent / "evaluation.json" if models else None
    if not f or not f.exists():
        return None
    e = json.loads(f.read_text())
    return {"file": str(f), "at": e.get("at"), "gate": e.get("gate"), "gate_passed": e.get("gate_passed"),
            "sets": e.get("sets"), "scores": e.get("scores", {}).get(models.name)}


def _readme(meta: dict, model: str) -> str:
    c, ev = meta["counts"], meta.get("evaluation") or {}
    sc = ev.get("scores") or {}
    gate = ""
    if sc.get("silver_owner_oof") and sc.get("gold_owner"):
        s, g = sc["silver_owner_oof"], sc["gold_owner"]
        gate = (f"\nGate (1 ms frames): owner-reviewed silver out of fold ({ev['sets']['silver_owner_oof']} shots) "
                f"P {s['precision']:.3f} R {s['recall']:.3f};\ngold with an owner review "
                f"({ev['sets']['gold_owner']} shots) P {g['precision']:.3f} R {g['recall']:.3f}.\n")
    return f"""# ELM activity, 1 ms grid, every FDP plasma shot

Model output of `{model}` (the ELM label-tiers student: trained on silver and teacher-refined bronze
shots), thresholded to present/absent on a 1 ms grid. Present means an ELMy period, not single ELMs.
Unreviewed. Built {meta['published']} by `{meta['made_by']}` (git {meta['git_sha']}).
{gate}
Candidates {sum(c.values())}; labeled {c.get('labeled', 0)}; spans {meta['n_spans']};
present {meta['present_ms'] / 3.6e6:.1f} h of {meta['assessed_ms'] / 3.6e6:.1f} h assessed.
Status counts and caveats: `meta.json`.

```python
import numpy as np, pandas as pd
spans = pd.read_parquet("spans.parquet")          # shot, t_start_ms, t_end_ms  ([start, end) ms)
shots = pd.read_parquet("shots.parquet").set_index("shot")
def elm_1ms(shot):
    \"\"\"(t_ms, label) on the assessed window: 1 present, 0 absent.\"\"\"
    w0, w1 = shots.loc[shot, ["window_start_ms", "window_end_ms"]].astype(int)
    lab = np.zeros(w1 - w0, dtype=np.uint8)
    for a, b in spans.loc[spans.shot == shot, ["t_start_ms", "t_end_ms"]].itertuples(index=False):
        lab[a - w0:b - w0] = 1
    return np.arange(w0, w1), lab
```

The same labels dense, one file a shot holding only the 0/1 list (1 present) as packed bits, 8 frames
a byte, first frame in the top bit; its start and end are in `{ae.GRID_DIR}/index.csv`:

```python
grid = pd.read_csv("{ae.GRID_DIR}/index.csv").set_index("shot")             # shot, start_ms, end_ms, job
n = grid.end_ms[shot] - grid.start_ms[shot]
elm = np.unpackbits(np.fromfile(f"{ae.GRID_DIR}/{{shot // 1000}}/{{shot}}.bin", np.uint8), count=n)  # e.g. 200/200123.bin
t_ms = grid.start_ms[shot] + np.arange(n)                            # frame [t, t + 1) ms; ends at end_ms
```

`valid.csv` lists every shot number of the d3drdb snapshot's range with `valid` 1 when the shot has
ELM labels and 0 otherwise (no plasma, no usable record, or not run yet; `shots.parquet` `status`
says which for the plasma candidates).

Frame `k` is `[k, k + 1)` ms. Outside `[window_start_ms, window_end_ms)` nothing was assessed.
Shots whose `status` is not `labeled` have no window: see `meta.json` `statuses`.
"""


def cmd_publish(args) -> None:
    import pandas as pd

    cands = pd.read_csv(args.candidates)
    lab, spans, status_of = ae.shot_statuses()
    labeled = set(lab.index.astype(int))
    rows = []
    for c in cands.itertuples(index=False):
        st, note = status_of(int(c.shot))
        r = {"shot": int(c.shot), "status": st, "plasma_end_ms": int(c.end_ms), "note": note}
        if st == "labeled":
            L = lab.loc[int(c.shot)]
            r.update(window_start_ms=int(L.window_start_ms), window_end_ms=int(L.window_end_ms),
                     present_ms=int(L.present_ms), n_spans=int(L.n_spans), fs_hz=float(L.fs_hz),
                     density_hz=float(L.density_hz), mean_prob=float(L.mean_prob))
        rows.append(r)
    table = pd.DataFrame(rows)
    for col in ("window_start_ms", "window_end_ms", "present_ms", "n_spans"):
        if col not in table:
            table[col] = pd.NA
        table[col] = table[col].astype("Int32")
    table["shot"] = table.shot.astype("int32")
    counts = table.status.value_counts().to_dict()
    first, last = ae.snapshot_range(cands)
    valid = pd.DataFrame({"shot": np.arange(first, last + 1, dtype=np.int32)})
    valid["valid"] = valid.shot.isin(labeled).astype(np.uint8)
    models = Path(args.models) if args.models else None
    run = json.loads((models / "run.json").read_text()) if models and (models / "run.json").exists() else {}
    thresholds = []
    if models:
        for f in sorted(models.glob("fold*/fold.json")):
            thresholds.append(float(json.loads(f.read_text())["threshold"]))
    meta = {
        "event": "edge_localized_mode",
        "published": ae.now(),
        "grid": "1 ms frames [k, k + 1) ms of the shot clock; spans [t_start_ms, t_end_ms)",
        "model": {"run_dir": str(models) if models else None, "run_name": run.get("name"),
                  "folds": len(thresholds), "fold_thresholds": thresholds,
                  "threshold": float(np.mean(thresholds)) if thresholds else None,
                  "rule": "present when the folds' mean event probability >= the mean threshold"},
        "inputs": {"filterscopes": FS_TAGS, "density": DENSITY_TAGS,
                   "source": "FDP MDSplus trees (views/shots/<shot>/v1/mdsplus), byte ranges",
                   "grid": "0.1 ms cells from -50 ms: filterscope maximum, density mean (labeler.elm.inputs)",
                   "screens": {"fs_rate_low": f"filterscope sample interval > {MAX_FS_DT_MS} ms",
                               "density_rate_low": f"density sample interval > {MAX_DENSITY_DT_MS} ms"}},
        "counts": counts, "n_spans": int(len(spans)),
        "present_ms": int(lab.present_ms.sum()) if len(lab) else 0,
        "assessed_ms": int((lab.window_end_ms - lab.window_start_ms).sum()) if len(lab) else 0,
        "made_by": "scripts/labeler/elm_fdp_label.py", "git_sha": _git_sha(),
        "evaluation": _evaluation(models),
        "statuses": {
            "labeled": "run; its window and spans are in the tables",
            "pending": "not run yet (staging or jobs still to come)",
            "no_view": "FDP has no per-shot view for it",
            "ptdata_record": "the filterscope or density node is a PTDATA call (shots before ~131000): "
                             "the FDP tree files hold no samples, so it is not run",
            "no_fs": "no filterscope record (no tree file, no node or an empty node)",
            "no_density": "no DENV2F/DENV3F record (no tree file, no node or an empty node)",
            "fs_rate_low": f"filterscopes sampled slower than every {MAX_FS_DT_MS} ms",
            "density_rate_low": f"density sampled slower than every {MAX_DENSITY_DT_MS} ms",
            "no_record_in_window": "no samples in [0, plasma end]",
            "failed": "the download failed after retries (retried on the next run)",
            "other": "read_failed, bad_record, extract_error, infer_failed, no_staged_file: the note column says why",
        },
        "caveats": [
            ("Model output, unreviewed: not ground truth. See evaluation for the gate it passed (1 ms frames, "
             "precision and recall >= 0.75 on owner-reviewed silver shots out of fold and on the gold shots "
             "with an owner review; only 7 gold shots have one)."),
            ("Present means an ELMy period (the stretch with ELMs, the quiet time between them included), "
             "not single ELMs: the review labels it learned from mark ELMy stretches as crowd spans."),
            "The model reads the D-alpha filterscopes FS02-04 and the DENV2F/3F density only.",
            ("Trained on silver shots 132710-204937 and refined bronze shots 132240-204944; outside that range "
             "the output is untested."),
            "Inside the window, a millisecond with an incomplete input record reads absent.",
            ("Training shots had 50 or 100 kHz filterscopes. Shots with 20 kHz ones (the 150000s) are labeled "
             "too but untested; shots.parquet fs_hz says which."),
            "The window end is the d3drdb pulse_length; shots after the snapshot's last shot are not candidates.",
        ],
        "files": {
            "valid.csv": (f"shot, valid: every shot number {first}-{last} (the d3drdb snapshot's range), valid 1 "
                          "when the shot has ELM labels (status labeled), else 0; shots.parquet's status says why "
                          "for the plasma candidates"),
            "shots.parquet": "one row per plasma candidate: status, assessed window, present ms, spans, rates",
            "spans.parquet": "shot, t_start_ms, t_end_ms: the present stretches",
            ae.GRID_DIR + "/": "<shot // 1000>/<shot>.bin: one bit a 1 ms frame of the assessed window "
                               "(np.packbits, first frame in the top bit); index.csv gives start_ms, end_ms",
        },
    }
    readme = _readme(meta, run.get("name") or (str(models) if models else "unknown"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, write in (("valid.csv", lambda p: valid.to_csv(p, index=False)),
                        ("spans.parquet", lambda p: spans.to_parquet(p, index=False)),
                        ("shots.parquet", lambda p: table.to_parquet(p, index=False)),
                        ("meta.json", lambda p: Path(p).write_text(json.dumps(meta, indent=1))),
                        ("README.md", lambda p: Path(p).write_text(readme))):
        tmp = out / f".{name}.tmp"
        write(tmp)
        os.replace(tmp, out / name)
    n_grid = ae.write_grid(out, lab, spans)
    print(f"published to {out}: {counts}; {len(spans)} spans; {n_grid} grid files written", flush=True)


# ------------------------------------------------------------------ main


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=ROOT, help="the cache root (default %(default)s)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--candidates", default=str(CANDIDATES))
    s.add_argument("--list", default=None, help="only these shots (a shot,end_ms CSV)")
    s.add_argument("--shots", type=int, nargs="*")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--procs", type=int, default=16)
    s.add_argument("--batch", type=int, default=2000)
    s.add_argument("--budget-gb", type=float, default=300)
    s.add_argument("--submit", action="store_true")
    s.add_argument("--models", default=None, help="a model run folder: also submit infer per batch")
    s.add_argument("--publish", action="store_true",
                   help="publish to --out after each batch (login node; with --models)")
    s.add_argument("--out", default=None, help=ae.OUT_HELP)
    s.add_argument("--whole", action="store_true",
                   help="restage the shots whose sparse read failed, with whole datafiles")
    e = sub.add_parser("extract")
    e.add_argument("--list", required=True)
    e.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    e.add_argument("--keep-inputs", action="store_true", help="keep the staged tree files")
    e.add_argument("--redo", nargs="*", default=[],
                   help="also extract listed shots whose last status is one of these (after a screen changed)")
    i = sub.add_parser("infer")
    i.add_argument("--models", required=True)
    i.add_argument("--list", default=None)
    i.add_argument("--workers", type=int, default=6)
    i.add_argument("--parts", type=int, default=1)
    i.add_argument("--part", type=int, default=0)
    i.add_argument("--device", default="cuda")
    u = sub.add_parser("publish")
    u.add_argument("--candidates", default=str(CANDIDATES))
    u.add_argument("--models", default=None, help="the run folder the labels came from (meta.json)")
    u.add_argument("--out", default=None, help=ae.OUT_HELP)
    args = p.parse_args(argv)
    _set_root(args.root)
    if args.cmd in ("stage", "publish"):
        args.out = ae.publish_out(args.out, ROOT, ALL_SHOT_ROOT, OUT)
    {"stage": cmd_stage, "extract": cmd_extract, "infer": cmd_infer, "publish": cmd_publish}[args.cmd](args)


if __name__ == "__main__":
    main()
