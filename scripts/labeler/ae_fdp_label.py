#!/usr/bin/env python
"""AE activity on a 1 ms grid for every DIII-D plasma shot FDP serves, from the stable AE model.

The model is ``d3d_ae_activity_seldnet`` (``configs/labeler/best_models.yaml``; stable and
latest in ``docs/labeler/events/alfven_eigenmode/README.md``), run with the GPU chain of the
silver run (``labeler.ae.gpu``: 500 kHz resample, 80-250 kHz STFT, clip, standardise,
1,024-column windows with 128 of context, sigmoid, 5-column moving average).

Seven subcommands, each resumable (a shot already done is skipped):

``candidates`` (login node, an env with pandas): the plasma shots of the d3drdb snapshot
    (``SUMMARIES``: pulse_length > 0.1 s and |ipmax| > 100 kA) -> ``candidates.csv``
    (shot, end_ms = floor(pulse_length) in ms, pbeam), newest first. ``pulse_length`` is the
    plasma end: on the 422 silver shots with an Ip window it equals the Ip >= 50 kA end
    within 10 ms for 93 % (97 % within 50 ms).
``stage`` (login node, fdp env under ``fdp -D d3d run``): for each candidate in FDP's shot
    views, ``pelican object get <view>/ptdata/<shot>.BCI`` (the CO2 chords' PTDATA file,
    ~170 MB) into ``stage/``. Every ``--batch`` staged shots it writes ``batches/<k>.csv`` and,
    with ``--submit``, submits the extract job and the infer job after it. It pauses while
    ``stage/`` and ``co2/`` hold more than ``--budget-gb``. Stops at the first auth error.
``extract`` (CPU job, fdp env, no network): ``stage/<shot>.BCI`` -> ``co2/<shot>.npz``: the
    four chords DENR0UF, DENV1UF, DENV2UF, DENV3UF read with GA's ptdata reader, cut to
    ``[-10, end + 10]`` ms exactly as ``labeler.ae.gpu.read_co2`` cuts them; then deletes the
    staged file (only files under ``<root>/stage``).
``infer`` (GPU job, stellarai env): ``co2/<shot>.npz`` -> ``work/<shot>.npz`` (the STFT
    column probabilities, float32) and the 1 ms calls appended to ``results/<job>.*.csv``; then
    deletes the co2 file (only files under ``<root>/co2``).
``publish`` (login node, fdp env): the results and logs -> ``--out`` (``/projects``):
    ``spans.parquet`` (present spans), ``shots.parquet`` (one row per candidate, with its
    status and assessed window), ``meta.json`` and ``README.md``.
``coverage`` (login node, fdp env): the public listing of processed shot numbers (tokamaksi's
    ``coverage/processed.txt``) from the statuses of one or more roots; re-runs the
    ``coverage.ipynb`` beside it.
``view`` (login node, fdp env under ``fdp -D d3d run``): the CO2 spectrogram of a few shots for
    ``examples/labeler/fdp_labels_view.ipynb`` -> ``$LABELER_ROOT/ae_view/<shot>.npy``.

**1 ms calls.** Frame ``k`` covers ``[k, k + 1)`` ms and is present when at least half of
its STFT columns (0.256 ms apart, so 3-4 a frame) have a smoothed probability >= 0.5: the
10 ms rule of ``ae_gold_extend.py`` at 1 ms. Frames inside a labeled shot's
``[window_start_ms, window_end_ms)`` that are not in a span are absent; frames outside it
are unassessed.

Input data lives only under ``<root>`` (``$LABELER_ROOT/ae_fdp_1ms``) and only until its
shot is labeled; ``--keep-inputs`` keeps it (the equivalence test reads ``D3D/fdp_stage``,
which is never deleted: deletes are confined to ``<root>/stage`` and ``<root>/co2``).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LABELER_ROOT = Path(os.environ.get("LABELER_ROOT",
                                   "/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler"))
ROOT = Path(os.environ.get("AE_FDP_ROOT", LABELER_ROOT / "ae_fdp_1ms"))
OUT = Path("/projects/EKOLEMEN/fusionaihub_labels/alfven_eigenmode/ae_seldnet_fdp_1ms")
D3DRDB = Path("/scratch/gpfs/EKOLEMEN/D3D/text/d3drdb_20261006T135944Z")
SLURM_LOGS = LABELER_ROOT / "runs/slurm"

CHORDS = ("DENR0UF", "DENV1UF", "DENV2UF", "DENV3UF")  # labeler.events.verify.CO2_CHORDS
MARGIN_MS = 10.0          # labeler.ae.full.MARGIN_MS
MIN_FS_HZ = 499_000.0     # the 80-250 kHz band needs the 500 kHz record the chain resamples to
MIN_END_MS = 50.0
FRAME_MS = 1
THRESHOLD, MIN_SHARE = 0.5, 0.5
MIN_PULSE_S, MIN_IP_A = 0.1, 1e5

PELICAN = "pelican://osg-htc.org:443"
VIEW = "/fdp-d3d/views/shots/{prefix}/{shot}/v1"
# From scripts/data_fetching_fdp/stage_shots_fdp.py, without osdf1.chic: on 2026-11-03 it hung on every
# request from 14:00 and then failed AE reads with "too many open files" (30 s each) from 17:20.
# dtn-pas.cinc, mghpcc and newy32aoa served the same views in under a second.
CACHES = ("https://osdf-uw-cache.svc.osg-htc.org:8443,"
          "https://osdf-cache-01.gwave.ics.psu.edu:8443,"
          "https://dtn-pas.cinc.nrp.internet2.edu:8443,"
          "https://mghpcc-cache.nationalresearchplatform.org:8443,"
          "https://osdf1.newy32aoa.nrp.internet2.edu:8443")
ATTEMPTS = 6
# Whole words only: a bare "401" also matched shot 192401 in a timeout's message (2026-11-02)
AUTH_MARKERS = re.compile(
    r"\b(?:401|403|unauthorized|forbidden|token|permission denied"
    # FDP's side is down (its origin's TLS certificate lapsed on 2026-11-01): every
    # shot would fail the same way, so stop instead of marking them all failed
    r"|x509|failed to verify certificate)\b", re.I)


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def log(name: str, **row) -> None:
    """Append one JSON line to <root>/logs/<name>.jsonl (one writer per file at a time)."""
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "logs" / f"{name}.jsonl", "a") as fh:
        fh.write(json.dumps({"t": now(), **row}) + "\n")


def last_status(name: str) -> dict[int, dict]:
    """The last logged row per shot of <root>/logs/<name>.jsonl."""
    path = ROOT / "logs" / f"{name}.jsonl"
    out: dict[int, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut by a killed writer
            if "shot" in row:
                out[int(row["shot"])] = row
    return out


def read_list(path: Path) -> list[tuple[int, float]]:
    """(shot, end_ms) rows of a candidates or batch CSV."""
    rows = []
    lines = Path(path).read_text().splitlines()
    head = lines[0].split(",")
    i_shot, i_end = head.index("shot"), head.index("end_ms")
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) > max(i_shot, i_end):
            rows.append((int(parts[i_shot]), float(parts[i_end])))
    return rows


def write_list(path: Path, rows) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("shot,end_ms\n" + "".join(f"{s},{e:g}\n" for s, e in rows))
    os.replace(tmp, path)


def inside(path: Path, parent: Path) -> bool:
    try:
        return Path(path).resolve().is_relative_to(Path(parent).resolve())
    except OSError:
        return False


# ------------------------------------------------------------------ candidates


def cmd_candidates(args) -> None:
    import pandas as pd

    s = pd.read_parquet(D3DRDB / "SUMMARIES.parquet", columns=["shot", "pulse_length", "ipmax", "pbeam"])
    s = s.drop_duplicates("shot")
    keep = s[(s.pulse_length > MIN_PULSE_S) & (s.ipmax.abs() > MIN_IP_A)].copy()
    keep["end_ms"] = np.floor(keep.pulse_length * 1000.0).astype(int)
    keep = keep.sort_values("shot", ascending=False)
    ROOT.mkdir(parents=True, exist_ok=True)
    out = ROOT / "candidates.csv"
    keep[["shot", "end_ms", "pulse_length", "ipmax", "pbeam"]].to_csv(out, index=False)
    print(f"{len(keep)} candidates of {len(s)} SUMMARIES rows ({keep.shot.min()}-{keep.shot.max()}) -> {out}")


# ------------------------------------------------------------------ stage (login node)


class AuthError(RuntimeError):
    pass


def pelican(*args: str, pinned: bool = False, timeout: int = 1800) -> str:
    if args[0] == "get" and pinned:
        args = ("get", "-c", CACHES, *args[1:])
    out = subprocess.run(["pelican", "object", *args], capture_output=True, text=True, timeout=timeout)
    if out.returncode:
        err = out.stderr.strip()[-400:]
        if AUTH_MARKERS.search(err):
            raise AuthError(f"pelican object {args[0]}: {err}")
        raise RuntimeError(f"pelican object {' '.join(args)}: {err}")
    return out.stdout


def names_in(listing: str) -> set[str]:
    """The last path component of each line of a `pelican object ls` listing."""
    names = set()
    for line in listing.splitlines():
        parts = line.split()
        if parts:
            names.add(parts[0].rstrip("/").rsplit("/", 1)[-1])
    return names


class Views:
    """Which shots have an FDP view, one listing per 100 shots."""

    def __init__(self):
        self.cache: dict[int, set[int]] = {}

    def has(self, shot: int) -> bool:
        prefix = shot // 100
        if prefix not in self.cache:
            for attempt in range(ATTEMPTS):
                try:
                    names = names_in(pelican("ls", f"{PELICAN}/fdp-d3d/views/shots/{prefix}"))
                    break
                except AuthError:
                    raise
                except (RuntimeError, subprocess.TimeoutExpired) as error:
                    if "no such file" in str(error).lower() or "not found" in str(error).lower():
                        names = set()
                        break
                    if attempt == ATTEMPTS - 1:
                        raise
                    time.sleep(10 * (attempt + 1))
            self.cache[prefix] = {int(n) for n in names if n.isdigit()}
        return shot in self.cache[prefix]


def stage_one(shot: int, views: Views) -> dict:
    final = ROOT / "stage" / f"{shot}.BCI"
    if final.exists():
        return {"shot": shot, "status": "staged", "bytes": final.stat().st_size, "s": 0.0}
    try:
        if not views.has(shot):
            return {"shot": shot, "status": "no_view"}
    except AuthError:
        raise
    except Exception as error:  # noqa: BLE001 - a listing that keeps failing: retried next run
        return {"shot": shot, "status": "failed", "error": f"view listing: {error}"[-300:]}
    base = VIEW.format(prefix=shot // 100, shot=shot)
    part = final.with_suffix(".BCI.part")
    t0 = time.monotonic()
    err = ""
    for attempt in range(ATTEMPTS):
        try:
            part.unlink(missing_ok=True)
            # every attempt pinned to CACHES: the director also lists osdf1.chic (2026-11-03)
            pelican("get", f"{PELICAN}{base}/ptdata/{shot}.BCI", str(part), pinned=True)
            if part.exists() and part.stat().st_size > 0:
                os.replace(part, final)
                return {"shot": shot, "status": "staged", "bytes": final.stat().st_size,
                        "s": round(time.monotonic() - t0, 1)}
            err = "empty download"
        except AuthError:
            part.unlink(missing_ok=True)
            raise
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            err = str(error)[-300:]
        if attempt == 0:
            # A missing file fails the same way every time: look once before retrying.
            try:
                if f"{shot}.BCI" not in names_in(pelican("ls", f"{PELICAN}{base}/ptdata")):
                    part.unlink(missing_ok=True)
                    return {"shot": shot, "status": "no_bci"}
            except AuthError:
                raise
            except (RuntimeError, subprocess.TimeoutExpired):
                pass
        time.sleep(10 * (attempt + 1))
    part.unlink(missing_ok=True)
    return {"shot": shot, "status": "failed", "error": err}


def held_gb() -> float:
    total = 0
    for sub in ("stage", "co2"):
        d = ROOT / sub
        if d.exists():
            with os.scandir(d) as it:
                for e in it:
                    try:
                        total += e.stat().st_size if e.is_file() else 0
                    except FileNotFoundError:  # a .part renamed, or a file deleted, while listing
                        pass
    return total / 1e9


SUBMIT_WAIT_S, SUBMIT_TRIES = 60, 360


def sbatch(script: Path, env: dict, dependency: str | None = None) -> str:
    """Submit; a full per-user submit limit is waited out (up to 6 h), not raised.

    Jobs of 30 min or less land in QOS test/gpu-test, which queue 5 jobs per user. With the GPUs
    full, AE and ELM infer jobs reached 5 and the next sbatch stopped the ELM staging (2026-11-03).
    """
    cmd = ["sbatch", "--parsable"]
    if dependency:
        cmd.append(f"--dependency={dependency}")
    cmd.append(str(script))
    for attempt in range(SUBMIT_TRIES):
        out = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **env}, check=False)
        if out.returncode == 0:
            return out.stdout.strip().split(";")[0]
        if "MaxSubmitJob" not in out.stderr or attempt == SUBMIT_TRIES - 1:
            raise subprocess.CalledProcessError(out.returncode, cmd, out.stdout, out.stderr)
        if attempt == 0:
            print(f"sbatch {script.name}: submit limit reached, retrying every {SUBMIT_WAIT_S} s", flush=True)
        time.sleep(SUBMIT_WAIT_S)
    raise AssertionError("unreachable")


def submit_batch(k: int, rows: list[tuple[int, float]]) -> tuple[str, str]:
    (ROOT / "batches").mkdir(parents=True, exist_ok=True)
    path = ROOT / "batches" / f"b{k:04d}.csv"
    write_list(path, rows)
    env = {"LIST": str(path), "AE_FDP_ROOT": str(ROOT)}
    ext = sbatch(REPO / "scripts/labeler/ae_fdp_extract.sbatch", env)
    inf = sbatch(REPO / "scripts/labeler/ae_fdp_infer.sbatch", env, dependency=f"afterany:{ext}")
    log("jobs", batch=k, list=str(path), shots=len(rows), extract=ext, infer=inf)
    print(f"batch {k}: {len(rows)} shots -> extract {ext}, infer {inf}", flush=True)
    return ext, inf


def jobs_running(ids: list[str]) -> list[str]:
    if not ids:
        return []
    out = subprocess.run(["squeue", "-h", "-o", "%i", "-j", ",".join(ids)], capture_output=True, text=True)
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def cmd_stage(args) -> None:
    cands = read_list(Path(args.candidates))
    if args.shots:
        want = set(args.shots)
        cands = [c for c in cands if c[0] in want]
    staged_before = last_status("stage")
    extracted = last_status("extract")
    labeled = {int(p.stem) for p in (ROOT / "work").glob("*.npz")} if (ROOT / "work").exists() else set()

    def needed(shot: int) -> bool:
        if shot in labeled:
            return False
        st = staged_before.get(shot, {}).get("status")
        if st in ("no_view", "no_bci"):
            return False
        # Staged and then extracted (to a co2 file or to a final finding): nothing left to fetch.
        return not (st == "staged" and extracted.get(shot, {}).get("status") not in (None, "no_staged_file"))

    todo = [c for c in cands if needed(c[0])]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} shots to stage of {len(cands)} candidates; {args.threads} threads; "
          f"batch {args.batch}; budget {args.budget_gb} GB", flush=True)
    (ROOT / "stage").mkdir(parents=True, exist_ok=True)
    views = Views()
    batch: list[tuple[int, float]] = []
    k = len(list((ROOT / "batches").glob("b*.csv"))) if (ROOT / "batches").exists() else 0
    jobs: list[str] = []
    counts: dict[str, int] = {}
    t_start = time.monotonic()
    nbytes = 0

    def flush(force: bool = False) -> None:
        nonlocal batch, k
        if batch and (force or len(batch) >= args.batch):
            if args.submit:
                jobs.extend(submit_batch(k, batch))
            else:
                (ROOT / "batches").mkdir(parents=True, exist_ok=True)
                write_list(ROOT / "batches" / f"b{k:04d}.csv", batch)
                print(f"batch {k}: {len(batch)} shots written (not submitted)", flush=True)
            k += 1
            batch = []
            if args.publish:
                try:
                    cmd_publish(argparse.Namespace(out=args.out, quiet=True))
                except Exception as error:  # noqa: BLE001 - a failed publish never stops staging
                    print("publish failed:", error, flush=True)

    end_of = dict(todo)
    with ThreadPoolExecutor(args.threads) as pool:
        it = iter(todo)
        pending = {}
        try:
            while True:
                while len(pending) < args.threads:
                    while held_gb() > args.budget_gb:
                        time.sleep(60)
                    try:
                        shot, _ = next(it)
                    except StopIteration:
                        break
                    pending[pool.submit(stage_one, shot, views)] = shot
                if not pending:
                    break
                for fut in [f for f in pending if f.done()] or [next(iter(pending))]:
                    shot = pending.pop(fut)
                    row = fut.result()  # an AuthError propagates from here
                    log("stage", **row)
                    counts[row["status"]] = counts.get(row["status"], 0) + 1
                    if row["status"] == "staged":
                        batch.append((shot, end_of[shot]))
                        nbytes += row.get("bytes", 0)
                    done = sum(counts.values())
                    if done % 50 == 0:
                        rate = done / max(time.monotonic() - t_start, 1)
                        print(f"{done}/{len(todo)} {counts} | {nbytes / 1e9:.0f} GB | {rate * 3600:.0f} shots/h "
                              f"| held {held_gb():.0f} GB", flush=True)
                    flush()
        except AuthError as error:
            log("stage", event="auth_error", error=str(error))
            print("FDP REFUSED, stopping (an expired login: run `fdp login`; a certificate error: FDP's "
                  "server, wait for it; then rerun):", error, flush=True)
            for f in pending:
                f.cancel()
            flush(force=True)
            raise SystemExit(3)
    flush(force=True)
    print(f"staging done: {counts} in {(time.monotonic() - t_start) / 60:.1f} min", flush=True)
    if args.submit and args.wait:
        while running := jobs_running(jobs):
            print(f"waiting on {len(running)} jobs", flush=True)
            time.sleep(120)
    if args.publish:
        cmd_publish(argparse.Namespace(out=args.out, quiet=False))


# ------------------------------------------------------------------ extract (CPU job)


def bci_path(shot: int, stage_dir: Path) -> Path:
    flat = stage_dir / f"{shot}.BCI"
    return flat if flat.exists() else stage_dir / str(shot) / "ptdata" / f"{shot}.BCI"


def extract_one(job: tuple) -> dict:
    shot, end_ms, stage_dir, keep = job
    t0 = time.monotonic()
    path = bci_path(shot, Path(stage_dir))
    row = {"shot": shot}
    if not path.exists():
        return {**row, "status": "no_staged_file"}
    try:
        import ptdata

        reader = ptdata.PtDataReader(sys_d3_paths=[str(path.parent)])
        params = ptdata.ExtractionParams()
        rows, times, units = [], None, set()
        for name in CHORDS:
            try:
                res = reader.fetch(name, shot, params, ".BCI")
            except Exception as error:  # noqa: BLE001 - ptdata raises its own
                return {**row, "status": "no_co2_points", "error": f"{name}: {type(error).__name__}: {error}"[:300]}
            data = np.asarray(res.data)
            x = np.asarray(res.times, dtype=np.float64)
            units.add(str(getattr(res, "units", "")).strip())
            if data.ndim != 1 or data.size < 2 or x.shape != data.shape:
                return {**row, "status": "bad_record", "error": f"{name}: data {data.shape}, times {x.shape}"}
            if times is None:
                times = x
            elif x.shape != times.shape or not np.array_equal(x, times):
                return {**row, "status": "chord_timebase_mismatch", "error": name}
            rows.append(data)
        if "vo" in {u.lower() for u in units}:
            return {**row, "status": "uncalibrated", "units": sorted(units)}
        end = min(float(times[-1]), float(end_ms))
        keep_mask = (times >= -MARGIN_MS) & (times <= end + MARGIN_MS)
        if end < MIN_END_MS or not keep_mask.any():
            return {**row, "status": "no_record_in_window", "record_ms": [float(times[0]), float(times[-1])]}
        lo = int(keep_mask.argmax())
        hi = keep_mask.size - int(keep_mask[::-1].argmax())
        if keep_mask[lo:hi].all():
            chords = np.stack([r[lo:hi] for r in rows]).astype(np.float32)
            t = times[lo:hi]
        else:
            chords = np.stack([r[keep_mask] for r in rows]).astype(np.float32)
            t = times[keep_mask]
        if not np.all(np.diff(t) > 0):
            return {**row, "status": "timebase_not_increasing"}
        fs = (t.size - 1) / (t[-1] - t[0]) * 1000.0
        if fs < MIN_FS_HZ:
            return {**row, "status": "rate_below_500khz", "fs_hz": round(fs, 1)}
        if not np.isfinite(chords).all():
            return {**row, "status": "nonfinite", "fs_hz": round(fs, 1)}
        dead = int(sum(float(np.std(c)) == 0.0 for c in chords))
        (Path(ROOT) / "co2").mkdir(parents=True, exist_ok=True)
        out = Path(ROOT) / "co2" / f"{shot}.npz"
        tmp = out.with_name(f".{shot}.{os.getpid()}.npz.tmp")
        with open(tmp, "wb") as fh:
            np.savez(fh, shot=shot, chords=chords, n_samples=t.size, first_ms=float(t[0]),
                     last_ms=float(t[-1]), end_ms=end, plasma_end_ms=float(end_ms), fs_hz=fs,
                     record_first_ms=float(times[0]), record_last_ms=float(times[-1]),
                     units=",".join(sorted(units)), dead_chords=dead)
        os.replace(tmp, out)
        if not keep and inside(path, Path(ROOT) / "stage"):
            path.unlink()
        return {**row, "status": "ok", "fs_hz": round(fs, 1), "end_ms": end, "dead_chords": dead,
                "units": sorted(units), "s": round(time.monotonic() - t0, 2)}
    except Exception as error:  # noqa: BLE001 - one bad shot never stops the job
        return {**row, "status": "extract_error", "error": f"{type(error).__name__}: {error}"[:300]}


def cmd_extract(args) -> None:
    from multiprocessing import Pool

    rows = read_list(Path(args.list))
    stage_dir = Path(args.stage_dir)
    done = last_status("extract")

    def needed(shot: int) -> bool:
        if (Path(ROOT) / "co2" / f"{shot}.npz").exists() or (Path(ROOT) / "work" / f"{shot}.npz").exists():
            return False
        # A final finding (no chords, low rate, ...) is not redone; an error or a missing file is.
        return done.get(shot, {}).get("status") in (None, "ok", "extract_error", "no_staged_file")

    jobs = [(s, e, str(stage_dir), args.keep_inputs) for s, e in rows if needed(s)]
    print(f"extract {len(jobs)} of {len(rows)} shots with {args.workers} workers at {now()}", flush=True)
    t0 = time.monotonic()
    counts: dict[str, int] = {}
    with Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(extract_one, jobs, chunksize=1), 1):
            log("extract", job=os.environ.get("SLURM_JOB_ID", "local"), **row)
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            if i % 100 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} {counts} {(time.monotonic() - t0):.0f} s", flush=True)
    print(f"extract done {counts} in {time.monotonic() - t0:.0f} s", flush=True)


# ------------------------------------------------------------------ infer (GPU job)


def frames_1ms(t_ms: np.ndarray, prob: np.ndarray, end_ms: float):
    """1 ms calls over [0, end): (call bool, has-columns bool) per frame."""
    n = int(math.floor(end_ms / FRAME_MS))
    k = np.floor(t_ms / FRAME_MS).astype(np.int64)
    ok = (k >= 0) & (k < n)
    cnt = np.bincount(k[ok], minlength=n)
    hits = np.bincount(k[ok], weights=(prob[ok] >= THRESHOLD).astype(np.float64), minlength=n)
    has = cnt > 0
    share = hits / np.maximum(cnt, 1)
    return (share >= MIN_SHARE) & has, has


def clean(text: str) -> str:
    """A message safe inside one quoted CSV field."""
    return str(text).replace('"', "'").replace("\n", " ")[:200]


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) index runs of True."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def cmd_infer(args) -> None:
    import torch
    from torch.utils.data import DataLoader, Dataset

    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "scripts/labeler"))
    from labeler.ae import full, gpu, lines
    from labeler.models.d3d_ae_activity_seldnet import spec as S
    import ae_gold_extend as G

    co2_dir, work = Path(ROOT) / "co2", Path(ROOT) / "work"
    work.mkdir(parents=True, exist_ok=True)
    res_dir = Path(ROOT) / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    if args.list:
        shots = [s for s, _ in read_list(Path(args.list))]
    else:
        shots = sorted(int(p.stem) for p in co2_dir.glob("*.npz"))
    todo = [s for s in shots if (co2_dir / f"{s}.npz").exists() and not (work / f"{s}.npz").exists()]
    todo = todo[args.part::args.parts]  # --parts processes share one GPU, each a disjoint share
    tag = os.environ.get("SLURM_JOB_ID", f"local{os.getpid()}")
    if args.parts > 1:
        tag = f"{tag}p{args.part}"
    print(f"infer {len(todo)} of {len(shots)} listed shots, job {tag}, {args.workers} workers at {now()}", flush=True)
    if not todo:
        return

    class Co2Files(Dataset):
        def __len__(self):
            return len(todo)

        def __getitem__(self, i):
            shot, t0 = todo[i], time.perf_counter()
            try:
                with np.load(co2_dir / f"{shot}.npz") as z:
                    item = {k: z[k].item() if z[k].ndim == 0 else z[k] for k in z.files}
            except Exception as error:  # noqa: BLE001
                return {"shot": shot, "skip": f"co2 file unreadable: {type(error).__name__}: {error}"}
            item["chords"] = torch.from_numpy(np.ascontiguousarray(item["chords"], dtype=np.float32))
            item["read_s"] = time.perf_counter() - t0
            return item

    device = torch.device(args.device)
    loader = DataLoader(Co2Files(), batch_size=None, collate_fn=G.as_is, num_workers=args.workers,
                        prefetch_factor=2 if args.workers else None, pin_memory=device.type == "cuda",
                        in_order=False)
    items = iter(loader)
    module, blob = G.load(device, wrap=False)
    run_on = f"{G.GPU_CHAIN}; network {G.device_note(device)}; {G.ENV_NOTE}"
    if args.checkpoint:  # another AeSeldNet checkpoint (ae_train / ae_tiers format), same chain
        from labeler.ae.model import AeSeldNet, AeSeldNetConfig

        blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        module = AeSeldNet(AeSeldNetConfig.from_dict(blob["config"]))
        module.load_state_dict(blob["state_dict"])
        module = module.to(device).eval()
        run_on += f"; checkpoint {args.checkpoint}"
    if args.notch:
        run_on += "; fixed-frequency lines notched (labeler.ae.lines.notch_shot_torch)"
    spec_dir = Path(args.spec_dir) if args.spec_dir else None
    if spec_dir:
        spec_dir.mkdir(parents=True, exist_ok=True)
    writer, pending = ThreadPoolExecutor(max_workers=8), []  # GPFS writes scale with streams

    def save_spec(shot, spec16, meta):
        """The meta file goes last: its presence marks a complete spec (ae_tiers.spec_shots)."""
        tmp = spec_dir / f".{shot}.{os.getpid()}.npy.tmp"
        with open(tmp, "wb") as fh:
            np.save(fh, spec16)
        os.replace(tmp, spec_dir / f"{shot}.npy")
        tmp = spec_dir / f".{shot}.{os.getpid()}.meta.npz.tmp"
        with open(tmp, "wb") as fh:
            np.savez(fh, **meta)
        os.replace(tmp, spec_dir / f"{shot}.meta.npz")
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    spans_csv, shots_csv = res_dir / f"{tag}.spans.csv", res_dir / f"{tag}.shots.csv"
    new = not shots_csv.exists()
    fs_spans, fs_shots = open(spans_csv, "a"), open(shots_csv, "a")
    if new:
        fs_spans.write("shot,t_start_ms,t_end_ms\n")
        fs_shots.write("shot,status,window_start_ms,window_end_ms,present_ms,n_spans,plasma_end_ms,"
                       "fs_native_hz,dead_chords,mean_prob,job,error\n")
    print("LOOP_START", G.stamp(), flush=True)
    t_loop, warm, ran = time.perf_counter(), False, 0
    total: dict[str, float] = {}
    for _ in range(len(todo)):
        t = time.perf_counter()
        item = next(items)
        took = {"wait": time.perf_counter() - t}
        shot = int(item["shot"])
        if "skip" in item:
            fs_shots.write(f"{shot},infer_failed,,,,,,,,,{tag},\"{clean(item['skip'])}\"\n")
            continue
        try:
            co2 = {"shot": shot, "chords": item["chords"], "n_samples": int(item["n_samples"]),
                   "first_ms": float(item["first_ms"]), "last_ms": float(item["last_ms"]),
                   "end_ms": float(item["end_ms"])}
            t = time.perf_counter()
            sig, t0_ms, fs_hz = gpu.input_signals(co2, device=device)
            spec = gpu.model_input(sig)
            net_in = spec
            if args.notch:  # (chords, bins, columns) -> (columns, chords, bins) and back
                net_in = lines.notch_shot_torch(spec.permute(2, 0, 1))[0].permute(1, 2, 0)
            out = gpu.run_windows(module, net_in.transpose(1, 2).contiguous(),
                                  window=S.WINDOW_FRAMES, context=S.CONTEXT_FRAMES)
            logit = out[:, 0].float().cpu().numpy()
            # (columns, chords, bins) float16: a training window is one contiguous block
            spec16 = spec.permute(2, 0, 1).to(torch.float16).cpu().numpy() if spec_dir else None
            sync()
            took["gpu"] = time.perf_counter() - t
        except (ValueError, RuntimeError) as error:
            fs_shots.write(f"{shot},infer_failed,,,,,,,,,{tag},\"{clean(f'{type(error).__name__}: {error}')}\"\n")
            continue
        t = time.perf_counter()
        n = int(sig.shape[1])
        del sig, spec, net_in, out
        t_ms = full.frame_times(n, t0_ms, fs_hz)
        if spec_dir is not None:
            meta = dict(t_ms=t_ms, end_ms=float(item["end_ms"]), plasma_end_ms=float(item["plasma_end_ms"]),
                        fs_hz=fs_hz, fs_native_hz=float(item["fs_hz"]), dead_chords=int(item["dead_chords"]),
                        layout="columns,chords,bins float16; labeler.ae.gpu.model_input")
            while len(pending) >= 16:  # ~65 MB a shot, at most 16 in flight
                pending.pop(0).result()
            pending.append(writer.submit(save_spec, shot, spec16, meta))
            del spec16
        prob = S.moving_average(1.0 / (1.0 + np.exp(-logit.astype(np.float64))))
        call, has = frames_1ms(t_ms, prob, co2["end_ms"])
        idx = np.flatnonzero(has)
        w0, w1 = (int(idx[0]), int(idx[-1]) + 1) if idx.size else (0, 0)
        call[:w0] = False
        call[w1:] = False
        spans = runs(call)
        hop = float(np.median(np.diff(t_ms))) if t_ms.size > 1 else 0.0
        tmp = work / f".{shot}.{os.getpid()}.npz.tmp"
        with open(tmp, "wb") as fh:
            np.savez(fh, t_first_ms=float(t_ms[0]), hop_ms=hop, n_columns=t_ms.size,
                     prob=prob.astype(np.float32), end_ms=co2["end_ms"], plasma_end_ms=float(item["plasma_end_ms"]),
                     fs_hz=fs_hz, fs_native_hz=float(item["fs_hz"]), window_ms=np.array([w0, w1]),
                     spans_ms=np.array(spans, dtype=np.int32).reshape(-1, 2), run_on=run_on, job=tag,
                     notch=bool(args.notch))
        os.replace(tmp, work / f"{shot}.npz")
        for a, b in spans:
            fs_spans.write(f"{shot},{a},{b}\n")
        present = int(sum(b - a for a, b in spans))
        mean_prob = float(prob[(t_ms >= w0) & (t_ms < w1)].mean()) if w1 > w0 else float("nan")
        fs_shots.write(f"{shot},labeled,{w0},{w1},{present},{len(spans)},{float(item['plasma_end_ms']):g},"
                       f"{float(item['fs_hz']):.1f},{int(item['dead_chords'])},{mean_prob:.4f},{tag},\n")
        fs_spans.flush()
        fs_shots.flush()
        src = co2_dir / f"{shot}.npz"
        if not args.keep_inputs and inside(src, co2_dir):
            src.unlink(missing_ok=True)
        took["post"] = time.perf_counter() - t
        took["read"] = float(item["read_s"])
        for key, v in took.items():
            total[key] = total.get(key, 0.0) + v
        ran += 1
        if not warm:
            print("LOOP_WARM", G.stamp(), flush=True)
            warm = True
        if ran % 100 == 0 or ran <= 5:
            print(shot, w1, "ms |", len(spans), "spans |", " ".join(f"{k} {v:.3f}" for k, v in took.items()),
                  flush=True)
    sync()
    for f in pending:
        f.result()
    writer.shutdown(wait=True)
    print("LOOP_END", G.stamp(), flush=True)
    fs_spans.close()
    fs_shots.close()
    wall = time.perf_counter() - t_loop
    print(f"{ran} shots in {wall:.1f} s ({wall / max(ran, 1):.3f} s a shot); stage totals "
          + " ".join(f"{k} {v:.1f}" for k, v in total.items()), flush=True)


# ------------------------------------------------------------------ publish (login node)


def read_results(root: Path | None = None):
    import pandas as pd

    res_dir = Path(root or ROOT) / "results"
    shots, spans = [], []
    for f in sorted(res_dir.glob("*.shots.csv")):
        shots.append(pd.read_csv(f, on_bad_lines="skip", dtype={"error": str, "job": str}))
    for f in sorted(res_dir.glob("*.spans.csv")):
        d = pd.read_csv(f, on_bad_lines="skip")
        d["job"] = f.name.split(".")[0]
        spans.append(d)
    shots = pd.concat(shots, ignore_index=True) if shots else pd.DataFrame(
        columns=["shot", "status", "job", "error"])
    spans = pd.concat(spans, ignore_index=True) if spans else pd.DataFrame(
        columns=["shot", "t_start_ms", "t_end_ms", "job"])
    return shots, spans


def labeled_results(root: Path | None = None):
    """(shots, lab, spans) of the run under ``root`` (ROOT): one results row per shot, the labeled
    ones indexed by shot, and their spans."""
    shots, spans = read_results(root)
    # The latest labeled row per shot wins; an infer failure counts only where no label exists.
    shots["labeled"] = shots.status.eq("labeled")
    shots = shots.sort_values(["shot", "labeled"]).drop_duplicates("shot", keep="last")
    lab = shots[shots.labeled].set_index("shot")
    spans = spans.merge(lab[["job"]].reset_index(), on=["shot", "job"], how="inner")
    spans = spans[["shot", "t_start_ms", "t_end_ms"]].astype("int32").sort_values(["shot", "t_start_ms"])
    return shots, lab, spans


def shot_statuses():
    """(lab, spans, status_of) for the run under ROOT: the labeled rows, their spans, and
    shot -> (status, note) as ``shots.parquet`` records it ("pending" = not reached yet)."""
    stage = last_status("stage")
    extract = last_status("extract")
    shots, lab, spans = labeled_results()

    failed = dict(zip(shots.shot[~shots.labeled].astype(int), shots.error[~shots.labeled].astype(str)))
    labeled = set(lab.index.astype(int))

    def status_of(shot: int) -> tuple[str, str]:
        if shot in labeled:
            return "labeled", ""
        if shot in failed:
            return "infer_failed", failed[shot]
        e = extract.get(shot)
        if e and e["status"] != "ok":
            return e["status"], str(e.get("error", e.get("fs_hz", "")))
        s = stage.get(shot)
        if s is None:
            return "pending", ""
        if s["status"] == "staged":
            return "pending", "staged"
        return s["status"], str(s.get("error", ""))[:200]

    return lab, spans, status_of


#: Statuses a rerun would retry (a passing fault); every other status is final.
#: read_failed is elm_fdp_label.py's (a sparse read that a whole-datafile restage may fix);
#: coverage reads both pipelines' roots with this set.
RETRY = {"pending", "failed", "infer_failed", "extract_error", "no_staged_file", "read_failed"}


def snapshot_range(cands) -> tuple[int, int]:
    """The d3drdb snapshot's shot range, stretched to the newest candidate."""
    import pandas as pd

    s = pd.read_parquet(D3DRDB / "SUMMARIES.parquet", columns=["shot"]).shot
    return int(s.min()), max(int(s.max()), int(cands.shot.max()))


def cmd_publish(args) -> None:
    import pandas as pd

    sys.path.insert(0, str(REPO / "src"))
    cands = pd.read_csv(Path(ROOT) / "candidates.csv")
    lab, spans, status_of = shot_statuses()
    labeled = set(lab.index.astype(int))
    notched: set[int] = set()
    if getattr(args, "notch_root", None):  # the relabel with the line notch replaces those shots' labels
        _, lab2, spans2 = labeled_results(Path(args.notch_root))
        notched = set(lab2.index.astype(int)) & labeled
        lab = pd.concat([lab.drop(index=sorted(notched)), lab2.loc[sorted(notched)]]).sort_index()
        spans = pd.concat([spans[~spans.shot.isin(notched)], spans2[spans2.shot.isin(notched)]],
                          ignore_index=True).sort_values(["shot", "t_start_ms"], ignore_index=True)
    out_rows = []
    for c in cands.itertuples(index=False):
        st, note = status_of(int(c.shot))
        r = {"shot": int(c.shot), "status": st, "plasma_end_ms": int(c.end_ms),
             "pbeam": float(c.pbeam) if pd.notna(c.pbeam) else np.nan, "note": note,
             "notch": int(c.shot) in notched}
        if st == "labeled":
            L = lab.loc[int(c.shot)]
            r.update(window_start_ms=int(L.window_start_ms), window_end_ms=int(L.window_end_ms),
                     present_ms=int(L.present_ms), n_spans=int(L.n_spans),
                     fs_native_hz=float(L.fs_native_hz), dead_chords=int(L.dead_chords),
                     mean_prob=float(L.mean_prob))
        out_rows.append(r)
    table = pd.DataFrame(out_rows)
    for col, dt in (("window_start_ms", "Int32"), ("window_end_ms", "Int32"), ("present_ms", "Int32"),
                    ("n_spans", "Int32"), ("dead_chords", "Int8")):
        if col not in table:
            table[col] = pd.NA
        table[col] = table[col].astype(dt)
    table["shot"] = table.shot.astype("int32")
    counts = table.status.value_counts().to_dict()

    # valid.csv: every shot number of the d3drdb snapshot's range, 1 when it has AE labels
    first, last = snapshot_range(cands)
    valid = pd.DataFrame({"shot": np.arange(first, last + 1, dtype=np.int32)})
    valid["valid"] = valid.shot.isin(labeled).astype(np.uint8)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = meta_json(table, spans, counts)
    meta["files"]["valid.csv"] = (
        f"shot, valid: every shot number {first}-{last} (the d3drdb snapshot's range), valid 1 when the shot "
        "has AE labels (status labeled), else 0: not a plasma shot (no d3drdb row, pulse_length <= 0.1 s or "
        "|ipmax| <= 100 kA), no usable CO2 record, or not run yet; shots.parquet's status says which for the "
        "plasma candidates")
    for name, write in (("valid.csv", lambda p: valid.to_csv(p, index=False)),
                        ("spans.parquet", lambda p: spans.to_parquet(p, index=False)),
                        ("shots.parquet", lambda p: table.to_parquet(p, index=False)),
                        ("meta.json", lambda p: Path(p).write_text(json.dumps(meta, indent=1))),
                        ("README.md", lambda p: Path(p).write_text(readme(meta)))):
        tmp = out / f".{name}.tmp"
        write(tmp)
        os.replace(tmp, out / name)
    n_grid = write_grid(out, lab, spans)
    if not getattr(args, "quiet", False):
        print(f"published to {out}: {counts}; {len(spans)} spans; {n_grid} grid files written")
    else:
        print(f"published {counts.get('labeled', 0)} labeled shots, {len(spans)} spans", flush=True)


def cmd_view(args) -> None:
    """The CO2 spectrogram of a few shots for the viewer notebook, whose inputs were deleted after
    labeling: stage the .BCI, extract the chords, run the model's own input chain on the CPU and
    keep only ``<root>/<shot>.npy``, ``(columns, 348)`` float16 averaged over the chords (80-250 kHz),
    with ``<root>/<shot>.meta.npz`` holding ``t_ms``. The .BCI and the chords are deleted. Login node,
    fdp env under ``fdp -D d3d run``; one shot is a fetch of ~170 MB and a few seconds of CPU."""
    import pandas as pd
    import torch

    sys.path.insert(0, str(REPO / "src"))
    from labeler.ae import full, gpu

    cands = pd.read_csv(LABELER_ROOT / "ae_fdp_1ms" / "candidates.csv").set_index("shot")
    (ROOT / "stage").mkdir(parents=True, exist_ok=True)
    views = Views()
    for shot in args.shots:
        if (ROOT / f"{shot}.meta.npz").exists():
            print(f"{shot}: already in {ROOT}")
            continue
        if shot not in cands.index:
            print(f"{shot}: not a plasma shot of the d3drdb snapshot")
            continue
        t_start = time.monotonic()
        got = stage_one(shot, views)  # an AuthError stops here: log in and rerun
        if got["status"] != "staged":
            print(f"{shot}: {got['status']} {got.get('error', '')}")
            continue
        got = extract_one((shot, float(cands.end_ms[shot]), str(ROOT / "stage"), False))
        if got["status"] != "ok":
            print(f"{shot}: {got['status']} {got.get('error', '')}")
            bci_path(shot, ROOT / "stage").unlink(missing_ok=True)
            continue
        co2_file = ROOT / "co2" / f"{shot}.npz"
        with np.load(co2_file) as z:
            co2 = {k: z[k].item() if z[k].ndim == 0 else z[k] for k in z.files}
        co2_file.unlink()
        with torch.no_grad():
            sig, t0_ms, fs_hz = gpu.input_signals(co2, device=torch.device("cpu"))
            spec = gpu.model_input(sig).mean(dim=0).T.to(torch.float16).numpy()  # (columns, bins)
        np.save(ROOT / f"{shot}.npy", spec)
        np.savez(ROOT / f"{shot}.meta.npz", t_ms=full.frame_times(int(sig.shape[1]), t0_ms, fs_hz),
                 layout="columns,bins float16, chords averaged; labeler.ae.gpu.model_input")
        print(f"{shot}: {spec.shape[0]} columns in {time.monotonic() - t_start:.0f} s -> {ROOT}")


def cmd_coverage(args) -> None:
    """The public listing (the tokamaksi repo's ``coverage/processed.txt``): every shot number
    from ``--first`` to the snapshot's last, as inclusive ranges of processed shots. Processed
    = a final status in any of ``--roots`` (labeled, or skipped for no FDP view, no CO2 file
    or an unusable record), or not a plasma candidate at all (skipped on purpose: no d3drdb
    row, pulse_length <= 0.1 s or |ipmax| <= 100 kA). Pending and passing faults are not."""
    global ROOT
    import pandas as pd

    cands = pd.read_csv(args.candidates)
    first, last = args.first, snapshot_range(cands)[1]
    done = set()
    for root in args.roots:
        ROOT = Path(root)
        _, _, status_of = shot_statuses()
        done |= {int(c) for c in cands.shot if status_of(int(c))[0] not in RETRY}
    shots = np.arange(first, last + 1)
    ok = ~np.isin(shots, cands.shot.to_numpy()) | np.isin(shots, np.fromiter(done, int, len(done)))
    edge = np.flatnonzero(np.diff(np.r_[0, ok.astype(np.int8), 0]))
    runs = [(int(shots[a]), int(shots[b - 1])) for a, b in zip(edge[::2], edge[1::2], strict=True)]
    lines = [
        "# DIII-D shots processed by the automatic labeling pass, as inclusive ranges first-last.",
        "# Processed includes shot numbers skipped on purpose (no plasma, or no usable data).",
        f"# range {first}-{last}",
        f"# processed {int(ok.sum())}",
        f"# updated {time.strftime('%Y-%m-%d')}",
    ] + [f"{a}-{b}" if b > a else f"{a}" for a, b in runs]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, out)
    print(f"{out}: {int(ok.sum())} of {len(shots)} shots ({first}-{last}) processed "
          f"({100 * ok.mean():.1f} %), {len(runs)} ranges; plasma candidates done {len(done)}")
    nb = out.with_name("coverage.ipynb")
    if nb.exists():
        run_notebook(nb)
        print(f"{nb}: re-run")


def run_notebook(path: Path) -> None:
    """Re-run a notebook in place, leaving no interpreter version, path or cell metadata in it."""
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()
    nb.metadata["language_info"] = {"name": "python"}
    for cell in nb.cells:
        cell.metadata = {}
        for o in cell.get("outputs", []):
            o.pop("metadata", None)
    nbformat.write(nb, path)


GRID_DIR = "grid_1ms"


def write_grid(out: Path, lab, spans) -> int:
    """grid_1ms/<shot // 1000>/<shot>.bin for every labeled shot: only the labels, one bit a 1 ms
    frame (1 present), packed 8 to a byte with the first frame in the top bit (np.packbits), bit i
    the frame [start_ms + i, start_ms + i + 1), the last byte padded with zeros. grid_1ms/index.csv
    gives each shot's start_ms and end_ms (the assessed window; end_ms - start_ms bits) and the
    infer job the file came from. Only new or relabeled shots are written, so the repeated
    publishes during staging stay light."""
    import pandas as pd

    gdir = out / GRID_DIR
    gdir.mkdir(parents=True, exist_ok=True)
    index = gdir / "index.csv"
    done = {}
    if index.exists():
        old = pd.read_csv(index, dtype={"shot": int, "start_ms": int, "end_ms": int, "job": str})
        done = {int(r.shot): (int(r.start_ms), int(r.end_ms), r.job) for r in old.itertuples(index=False)}
    by_shot = {int(k): g[["t_start_ms", "t_end_ms"]].to_numpy() for k, g in spans.groupby("shot")}
    n = 0
    for shot, row in lab.iterrows():
        shot, job = int(shot), str(row.job)
        f = gdir / str(shot // 1000) / f"{shot}.bin"
        w0, w1 = int(row.window_start_ms), int(row.window_end_ms)
        if done.get(shot) == (w0, w1, job) and f.exists():
            continue
        a = np.zeros(max(w1 - w0, 0), np.uint8)
        for t0, t1 in by_shot.get(shot, ()):
            a[max(int(t0) - w0, 0):max(int(t1) - w0, 0)] = 1
        f.parent.mkdir(exist_ok=True)
        tmp = f.with_name(f".{f.name}.tmp")
        tmp.write_bytes(np.packbits(a).tobytes())
        os.replace(tmp, f)
        f.with_suffix(".csv").unlink(missing_ok=True)  # the text grid of earlier publishes
        done[shot] = (w0, w1, job)
        n += 1
    tmp = gdir / ".index.csv.tmp"
    pd.DataFrame([(k, *v) for k, v in sorted(done.items())], columns=["shot", "start_ms", "end_ms", "job"]).to_csv(
        tmp, index=False)
    os.replace(tmp, index)
    return n


def model_block() -> dict:
    """The model behind the labels: the stable model, or the checkpoint in $CKPT (the ae_tiers
    student; the infer jobs read the same variable)."""
    import hashlib

    ck = os.environ.get("CKPT")
    if not ck:
        ckpt = LABELER_ROOT / "models/d3d_ae_activity_seldnet/ae_seldnet_threeway_sce.pt"
        return {
            "name": "d3d_ae_activity_seldnet", "version": "2026_09_06",
            "card": "src/labeler/models/d3d_ae_activity_seldnet/README.md",
            "checkpoint": str(ckpt),
            "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest() if ckpt.exists() else None,
            "role": "the stable AE model (configs/labeler/best_models.yaml)",
            "scores": "AUROC 0.982, AUPRC 0.992, F1 0.944 on 10 ms frames of 19 validation shots against "
                      "the owner-reviewed labels (README Models); not scored at 1 ms",
            "training": "180 hand-annotated shots 170659-178879 (120 fitted, 60 validation)",
        }
    ck = Path(ck)
    info = json.loads((ck.parent / "training.json").read_text()) if (ck.parent / "training.json").exists() else {}
    return {
        "name": f"ae_tiers {ck.parent.name}", "version": info.get("made_at"),
        "card": "scripts/labeler/ae_tiers.py (gold / silver / bronze / database)",
        "checkpoint": str(ck), "checkpoint_sha256": hashlib.sha256(ck.read_bytes()).hexdigest(),
        "role": "the AE label-tiers student: trained on silver and teacher-refined bronze shots",
        "scores": "gold metrics and the release gate: <ae_tiers root>/eval/*.json; not scored at 1 ms",
        "training": f"{info.get('n_train_silver')} silver + {info.get('n_bronze')} bronze shots "
                    f"(silver fold {info.get('fold')} of {info.get('folds')} held out)",
    }


def meta_json(table, spans, counts) -> dict:
    sys.path.insert(0, str(REPO / "src"))
    model = model_block()
    student = bool(os.environ.get("CKPT"))
    try:
        git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        git = None
    lab = table[table.status == "labeled"]
    assessed = int((lab.window_end_ms - lab.window_start_ms).sum()) if len(lab) else 0
    notched = lab.shot[lab.notch] if "notch" in lab else lab.shot[:0]
    meta = {
        "name": "ae_seldnet_fdp_1ms",
        "category": "alfven_eigenmode",
        "made_at": now(),
        "made_by": "scripts/labeler/ae_fdp_label.py",
        "git_sha": git,
        "model": model,
        "files": {
            "spans.parquet": "present spans: shot (int32), t_start_ms, t_end_ms (int32, half-open [start, end) on "
                             "the 1 ms grid). Every 1 ms frame of a labeled shot's window not in a span is absent.",
            "shots.parquet": "one row per candidate shot: status, plasma_end_ms (d3drdb pulse_length), pbeam, notch "
                             "(true when the line notch was applied, method.line_notch), and for labeled shots "
                             "window_start_ms, window_end_ms (the assessed window, half-open), present_ms, n_spans, "
                             "fs_native_hz, dead_chords, mean_prob",
            f"{GRID_DIR}/<shot // 1000>/<shot>.bin": "the same labels dense, one file a labeled shot: only the "
                                                    "labels, one bit a 1 ms frame (1 present), 8 to a byte, first "
                                                    "frame in the top bit (numpy packbits, big bit order), bit i the "
                                                    "frame [start_ms + i, start_ms + i + 1), last byte zero-padded",
            f"{GRID_DIR}/index.csv": "shot, start_ms, end_ms (the grid's first frame start and last frame end; "
                                     "end_ms - start_ms bits), job (the infer job the file came from)",
        },
        "method": {
            "inputs": "CO2 interferometer chords DENR0UF, DENV1UF, DENV2UF, DENV3UF (PTDATA, the shot's .BCI file "
                      "from FDP's per-shot view, read with GA's ptdata reader), cut to [-10, end + 10] ms",
            "chain": "labeler.ae.gpu: resample to 500 kHz, STFT hann 1024 hop 128 (0.256 ms columns), log, clip, "
                     "per-chord standardisation over the window, 80-250 kHz band, AeSeldNet in 1,024-column "
                     "windows with 128 of context, sigmoid, 5-column centred moving average",
            "frame_call": "1 ms frame [k, k+1) present when at least half of its STFT columns (3-4) have "
                          "probability >= 0.5, otherwise absent",
            "threshold": THRESHOLD,
            "window": "[0, end) with end = floor(pulse_length) from the d3drdb SUMMARIES snapshot "
                      f"({D3DRDB.name}), or the CO2 record end if earlier; frames outside it are unassessed",
            "candidates": f"d3drdb SUMMARIES rows with pulse_length > {MIN_PULSE_S} s and |ipmax| > {MIN_IP_A:.0f} A",
        },
        "counts": {"candidates": int(len(table)), **{k: int(v) for k, v in counts.items()},
                   "spans": int(len(spans)), "present_ms": int(lab.present_ms.sum()) if len(lab) else 0,
                   "assessed_ms": assessed},
        "statuses": {
            "labeled": "run; its window and spans are in the tables",
            "pending": "not run yet (staging or jobs still to come)",
            "no_view": "FDP has no per-shot view for it",
            "no_bci": "its FDP view has no .BCI PTDATA file",
            "no_co2_points": "the .BCI file lacks one of the four chords",
            "rate_below_500khz": "the CO2 record is sampled below 500 kHz, so the 80-250 kHz band is not resolved",
            "no_record_in_window": "no CO2 samples in [0, plasma end]",
            "nonfinite": "NaN or inf in the CO2 record",
            "failed": "the download failed after retries (retried on the next run)",
            "other": "extract_error, bad_record, chord_timebase_mismatch, timebase_not_increasing, uncalibrated, "
                     "infer_failed: the note column says why",
        },
        "caveats": [
            ("Model output, unreviewed: not ground truth. The student was selected on held-out silver shots and "
             "checked on gold at 10 ms, never at 1 ms." if student else
             "Model output, unreviewed: not ground truth. The model over-calls against the raw Heidbrink annotation "
             "(precision 0.26 on its validation shots) and was scored on 10 ms frames, never on 1 ms."),
            "Absent means absent within the 80-250 kHz band: AE below 80 kHz (BAE, some RSAE) reads absent.",
            "The model reads CO2 only (no beams, ECH or Ip): a quiet CO2 record reads absent.",
            ("Trained on silver shots 132240-204937 and bronze shots 185601-204999. Far from those eras (other CO2 "
             "digitisers, plasma scenarios, diagnostics changes) the output is untested." if student else
             "Trained on 2017-2019 shots (170659-178879). Far from that era (other CO2 digitisers, plasma "
             "scenarios, diagnostics changes) the output is untested."),
            "The staged PTDATA reader agreed with the corpus on modern shots, but on shots 144300 and 146918 it "
            "returned implausible CO2 values where GA's TDI reads zero (scripts/data_fetching_fdp/mds_reader.py); "
            "older shots have no such check here.",
            "The window end is the d3drdb pulse_length, not the Ip >= 50 kA end the silver table uses (equal within "
            "10 ms on 93 % of 422 silver shots; within 50 ms on 97 %).",
            "Shots after the snapshot's last shot (209026) have no pulse_length and are not candidates.",
        ],
    }
    if len(notched):
        meta["method"]["line_notch"] = (
            f"{len(notched)} shots ({int(notched.min())}-{int(notched.max())}; shots.parquet notch) were labeled "
            "with fixed-frequency pickup lines notched out of the model input first (labeler.ae.lines: narrow "
            "runs standing out of the per-chord median spectrum over the whole shot in some chord, retested per "
            "500 ms block in every chord, interpolated over frame by frame). Before 145000 such lines are common "
            "and the network reads them as AE: AE called with the beams off for 50 ms or more fell from .3-.7 to "
            ".01-.14 of that time by era on 15 shots an era, while gold F1 (.857) and silver held-out F1 "
            "(.913 -> .912) did not move (dev/ae_notch, 2026-11-06)")
        meta["counts"]["notched"] = len(notched)
        meta["caveats"].append(
            "On shots with notch true, a real mode that holds one frequency where a line was found is notched "
            "with it; lines the detector misses (broad banding, a dead chord's carrier tone) can still read as AE.")
    return meta


def readme(meta: dict) -> str:
    c = meta["counts"]
    notch_note = (f"\nShots with `notch` true in `shots.parquet` ({c['notched']}) were labeled with pickup lines "
                  "notched out of the input first (`meta.json` method.line_notch).") if c.get("notched") else ""
    return f"""# AE activity, 1 ms grid, every FDP plasma shot

Model output of `{meta['model']['name']}` ({meta['model']['role']}), thresholded to
present/absent on a 1 ms grid. Unreviewed. Built {meta['made_at']} by `{meta['made_by']}` (git {meta['git_sha']}).

Candidates {c['candidates']}; labeled {c.get('labeled', 0)}; spans {c['spans']};
present {c['present_ms'] / 3.6e6:.1f} h of {c['assessed_ms'] / 3.6e6:.1f} h assessed.{notch_note}
Status counts and caveats: `meta.json`.

```python
import numpy as np, pandas as pd
spans = pd.read_parquet("spans.parquet")          # shot, t_start_ms, t_end_ms  ([start, end) ms)
shots = pd.read_parquet("shots.parquet").set_index("shot")
def ae_1ms(shot):
    \"\"\"(t_ms, label) on the assessed window: 1 present, 0 absent.\"\"\"
    w0, w1 = shots.loc[shot, ["window_start_ms", "window_end_ms"]].astype(int)
    lab = np.zeros(w1 - w0, dtype=np.uint8)
    for a, b in spans.loc[spans.shot == shot, ["t_start_ms", "t_end_ms"]].itertuples(index=False):
        lab[a - w0:b - w0] = 1
    return np.arange(w0, w1), lab
```

The same labels dense, one file a shot holding only the 0/1 list (1 present) as packed bits, 8 frames
a byte, first frame in the top bit; its start and end are in `{GRID_DIR}/index.csv`:

```python
grid = pd.read_csv("{GRID_DIR}/index.csv").set_index("shot")             # shot, start_ms, end_ms, job
n = grid.end_ms[shot] - grid.start_ms[shot]
ae = np.unpackbits(np.fromfile(f"{GRID_DIR}/{{shot // 1000}}/{{shot}}.bin", np.uint8), count=n)  # e.g. 200/200123.bin
t_ms = grid.start_ms[shot] + np.arange(n)                            # frame [t, t + 1) ms; ends at end_ms
```

`valid.csv` lists every shot number of the d3drdb snapshot's range with `valid` 1 when the shot has
AE labels and 0 otherwise (no plasma, no usable CO2 record, or not run yet; `shots.parquet` `status`
says which for the plasma candidates).

Frame `k` is `[k, k + 1)` ms. Outside `[window_start_ms, window_end_ms)` nothing was assessed.
Shots whose `status` is not `labeled` have no window: see `meta.json` `statuses`.
"""


# ------------------------------------------------------------------ main


def main(argv=None) -> None:
    global ROOT
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=ROOT, help="the cache root (default %(default)s)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("candidates")
    s = sub.add_parser("stage")
    s.add_argument("--candidates", default=None)
    s.add_argument("--shots", type=int, nargs="*")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--threads", type=int, default=16)
    s.add_argument("--batch", type=int, default=3000)
    s.add_argument("--budget-gb", type=float, default=1500)
    s.add_argument("--submit", action="store_true")
    s.add_argument("--wait", action="store_true")
    s.add_argument("--publish", action="store_true")
    s.add_argument("--out", type=Path, default=OUT)
    e = sub.add_parser("extract")
    e.add_argument("--list", required=True)
    e.add_argument("--stage-dir", default=None)
    e.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    e.add_argument("--keep-inputs", action="store_true")
    i = sub.add_parser("infer")
    i.add_argument("--list", default=None)
    i.add_argument("--device", default="cuda")
    i.add_argument("--workers", type=int, default=max(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")) - 1, 0))
    i.add_argument("--keep-inputs", action="store_true")
    i.add_argument("--part", type=int, default=0)
    i.add_argument("--parts", type=int, default=1, help="processes sharing the GPU (the sbatch runs PARTS)")
    i.add_argument("--spec-dir", default=None,
                   help="also save each shot's model input here (ae_tiers: silver, gold, bronze)")
    i.add_argument("--notch", action="store_true",
                   help="notch fixed-frequency pickup lines in the model input first (labeler.ae.lines)")
    i.add_argument("--checkpoint", default=None,
                   help="an AeSeldNet checkpoint to run instead of the stable model (the ae_tiers student)")
    u = sub.add_parser("publish")
    u.add_argument("--out", type=Path, default=OUT)
    u.add_argument("--notch-root", type=Path, default=None,
                   help="a relabel run with --notch (NOTCH=1) whose labeled shots replace this root's")
    c = sub.add_parser("coverage")
    c.add_argument("--out", type=Path, required=True, help="the listing (tokamaksi's coverage/processed.txt)")
    c.add_argument("--roots", nargs="+", required=True, help="run roots whose statuses count")
    c.add_argument("--candidates", default=str(LABELER_ROOT / "ae_fdp_1ms" / "candidates.csv"))
    c.add_argument("--first", type=int, default=100000)
    v = sub.add_parser("view", help="CO2 spectrograms of a few shots for the viewer notebook")
    v.add_argument("shots", type=int, nargs="+")
    v.add_argument("--dir", type=Path, default=LABELER_ROOT / "ae_view", help="where they go (default %(default)s)")
    args = p.parse_args(argv)
    ROOT = args.dir if args.cmd == "view" else args.root
    if args.cmd == "candidates":
        cmd_candidates(args)
    elif args.cmd == "stage":
        args.candidates = args.candidates or str(ROOT / "candidates.csv")
        cmd_stage(args)
    elif args.cmd == "extract":
        args.stage_dir = args.stage_dir or str(ROOT / "stage")
        cmd_extract(args)
    elif args.cmd == "infer":
        cmd_infer(args)
    elif args.cmd == "coverage":
        cmd_coverage(args)
    elif args.cmd == "view":
        cmd_view(args)
    else:
        cmd_publish(args)


if __name__ == "__main__":
    main()
