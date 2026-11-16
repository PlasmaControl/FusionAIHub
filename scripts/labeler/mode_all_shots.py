#!/usr/bin/env python
"""MHD mode labels (toroidal n, with m/n) for every DIII-D plasma shot 185000-204999, on StellarAI.

The labeller is omnimode's redesign (PlasmaControl/integratedmode, branch ``labeller-redesign``),
variant ``ermap-magrot`` with the TokEye 1.0 front end: Mirnov, Ip, the "er" EFIT ladder and CER
rotation; no ECE, no CO2. This script fetches those inputs over FDP, in omnimode's formats
(``labeler.modes.fetch``), and runs omnimode's own commands on them; nothing of the labeller is
reimplemented. Runbook: ``docs/labeler/mode_fdp_allshots.md``.

The shot list (``shots``) is the AE/ELM candidates (``ae_fdp_1ms/candidates.csv``) in
185000-204999 minus omnimode's sealed lists (``protocol/heldout_v1.txt``, ``heldout_v2.txt``,
``quarantine_v2.txt``, and ``seal.held_out()``/``pools.quarantined()``), filtered inside the
omnimode venv. Only counts are printed; no sealed shot is ever in a list, a log or a file here.

Subcommands, each resumable:

``shots`` (login node): ``<root>/shots.txt``.
``pilot`` (login node): ``<root>/batches/pilot.txt``, a few listed shots that the full corpus
    (``--reference``, StellarAI's ``D3D/signals``) also holds, for ``compare``.
``stage`` (login node, fdp env under ``fdp -D d3d run``): for each listed shot with an FDP view,
    the PTDATA files holding the 14 Mirnov probes and Ip and the trees EFIT02ER, EFIT02, EFITRT2,
    ZIPFIT01 and ZIPFIT02 (``pelican object get``) into ``stage/<shot>/``. Every ``--batch``
    staged shots it writes ``batches/bNNNN.txt`` and, with ``--submit``, submits the chain
    extract (CPU) -> TokEye masks (GPU) -> labels (CPU array), and frees the Mirnov copies of
    labelled shots (``clean``). It pauses while ``stage/`` and ``corpus/`` hold more than
    ``--budget-gb``. Stops at the first refused token (exit 3).
``extract`` (CPU job, fdp env, no network): ``stage/<shot>/`` -> ``corpus/<shot>_processed.h5``
    (mirnov + ip) and ``omnimode/inputs/{efit02er,efit02,efitrt2,rot}/<shot>.npz``; the stage
    folder is then deleted.
``submit`` (login node): the chain for one list (``--from masks`` skips the extract job).
``clean`` (login node): delete the corpus file of every labelled shot (it is about 0.5 GB); a
    shot whose label is retryable (no TokEye mask, a read or model error) keeps it for ``redo``.
``redo`` (login node): the retryable shots that still have their corpus file, into
    ``batches/redo_<time>.txt``; with ``--submit``, masks then labels with ``--force``.
``status`` (login node): counts of every stage.
``compare`` (login node, light): the fetched Mirnov against the full corpus on the shots both hold.
``publish`` (login node): submits the CPU job that writes the databases of
    ``docs/labeler/fdp_labels.md`` (``mode_fdp_label.py publish --labeller redesign``) into
    ``<root>/publish/``; copy them to ``/projects/EKOLEMEN/tokamaksi/diiid/dev`` after.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ae_fdp_label as ae  # the shared log, list, pelican and sbatch helpers

REPO = ae.REPO
sys.path.insert(0, str(REPO / "src"))
from labeler.modes import fetch as F

LABELER_ROOT = ae.LABELER_ROOT
ROOT = Path(os.environ.get("MODE_FDP_ROOT", LABELER_ROOT / "mode_fdp"))
OMNIMODE_REPO = Path(
    os.environ.get("OMNIMODE_REPO", "/scratch/gpfs/EKOLEMEN/nc1514/integratedmode")
)
CANDIDATES = LABELER_ROOT / "ae_fdp_1ms/candidates.csv"
REFERENCE = Path("/scratch/gpfs/EKOLEMEN/D3D/signals")
FIRST, LAST = 185000, 204999
VARIANT = "ermap-magrot"
SEALED_FILES = ("heldout_v1.txt", "heldout_v2.txt", "quarantine_v2.txt")
WEIGHTS_NAME = "big_tf_unet_251210.pt"
RUN = "omnimode_ermap_magrot_fdp_1ms"
JOBS = {
    name: REPO / "scripts" / "labeler" / f"mode_fdp_{name}.sbatch"
    for name in ("extract", "masks", "labels", "publish")
}
#: Stage findings a rerun would repeat.
FINAL_STAGE = {"no_view", "no_mirnov", "no_ip"}
#: Extract findings worth another try (the rest are the shot's own: no probe, a wrong rate).
RETRY_EXTRACT = {"no_staged_file", "extract_error"}
#: A labelled shot whose info record says this is retried by ``redo`` and keeps its corpus file.
RETRY_LABEL = ("no TokEye mask", "TokEye peaks failed", "read error")
SEAL_PY = r"""
import sys
from pathlib import Path
from omnimode.protocol import pools, seal
src, dst, names = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
listed = [int(x) for x in Path(src).read_text().split()]
sealed = set(seal.held_out()) | set(pools.quarantined())
for name in names:
    sealed |= set(seal.read_shots(seal.protocol_dir() / name))
keep = [s for s in listed if s not in sealed]
tmp = dst.with_name(dst.name + ".tmp")
tmp.write_text("".join(f"{s}\n" for s in keep))
tmp.replace(dst)
print(len(listed), len(keep))
"""
STORE_PY = f"""
from omnimode.config import Paths
from omnimode.frontend import config as FC
from omnimode.modes import pipeline
cfg = FC.frontend(FC.DEFAULT_FRONTEND)
pipeline.check_config(cfg, "{VARIANT}")
print(pipeline.cache_dir(Paths.from_env(), None if pipeline.is_port(cfg) else cfg, variant="{VARIANT}"))
"""


def _set_root(root: Path) -> None:
    global ROOT
    ROOT = Path(root)
    ae.ROOT = ROOT  # ae.log and ae.last_status write and read <root>/logs


_set_root(ROOT)


def omnimode_env() -> dict[str, str]:
    """omnimode's paths for this root; ``mode_fdp_env.sh`` sets the same for the jobs. The
    catalog, raw-cache and feature roots point at empty folders, so every window comes from
    the fetched Ip and nothing is read from another run."""
    data = ROOT / "omnimode"
    none = data / "none"
    return {
        "OMNIMODE_DATA": str(data),
        "OMNIMODE_CORPUS": str(ROOT / "corpus"),
        "OMNIMODE_RAW_CACHE": str(none / "raw_cache"),
        "OMNIMODE_FEATURES": str(none / "features"),
        "OMNIMODE_LABELER_ROOT": str(none / "labeler"),
        "OMNIMODE_TOKEYE_WEIGHTS": str(ROOT / "tokeye" / WEIGHTS_NAME),
    }


def run_omnimode(argv: list[str]) -> str:
    """Run the omnimode venv's python (``-I``) in its checkout; its stdout. Its stderr is shown
    on a failure (omnimode's seal errors never name a sealed shot)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("OMNIMODE_")}
    env |= omnimode_env()
    py = OMNIMODE_REPO / ".venv" / "bin" / "python"
    out = subprocess.run(
        [str(py), "-I", *argv],
        cwd=OMNIMODE_REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode:
        raise SystemExit(f"omnimode ({py}) failed:\n{out.stderr.strip()[-2000:]}")
    return out.stdout.strip()


_STORE: list[Path] = []


def store_dir() -> Path:
    """The label store omnimode's batch writes for this root and variant."""
    if not _STORE:
        _STORE.append(Path(run_omnimode(["-c", STORE_PY]).splitlines()[-1]))
    return _STORE[0]


def read_shots(path) -> list[int]:
    """A list file: one shot a line, ``#`` comments (omnimode's ``seal.read_shots`` format)."""
    out = []
    for line in Path(path).read_text().splitlines():
        text = line.split("#", 1)[0].strip()
        if text:
            out.append(int(text))
    return out


def write_shots(path: Path, shots) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(f"{int(s)}\n" for s in shots))
    os.replace(tmp, path)


def listed() -> list[int]:
    path = ROOT / "shots.txt"
    if not path.exists():
        raise SystemExit(f"{path} is missing: run `shots` first")
    return read_shots(path)


def label_info(shot: int, store: Path) -> dict | None:
    path = store / f"{int(shot)}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def retryable(info: dict) -> bool:
    status = str(info.get("status", ""))
    return bool(info.get("model_error")) or status.startswith(RETRY_LABEL)


def disk_gb(*folders: Path) -> float:
    """Bytes on disk under `folders`; an entry deleted while listing is skipped."""
    total, pending = 0, [str(f) for f in folders]
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


# ------------------------------------------------------------------ shots, pilot


def cmd_shots(args) -> None:
    cands = sorted(
        {s for s, _ in ae.read_list(Path(args.candidates)) if FIRST <= s <= LAST}
    )
    ROOT.mkdir(parents=True, exist_ok=True)
    src = ROOT / ".candidates_in_range.txt"
    write_shots(src, cands)
    try:
        out = run_omnimode(
            ["-c", SEAL_PY, str(src), str(ROOT / "shots.txt"), *SEALED_FILES]
        )
    finally:
        src.unlink(missing_ok=True)
    n_in, n_out = map(int, out.split()[-2:])
    print(
        f"{n_in} candidate shots in {FIRST}-{LAST}; {n_out} listed after omnimode's sealed "
        f"lists -> {ROOT / 'shots.txt'}"
    )


def cmd_pilot(args) -> None:
    have = [
        s for s in listed() if (Path(args.reference) / f"{s}_processed.h5").exists()
    ]
    if not have:
        raise SystemExit(f"no listed shot has a file in {args.reference}")
    picks = [
        have[int(i)] for i in np.linspace(0, len(have) - 1, min(args.n, len(have)))
    ]
    path = ROOT / "batches" / "pilot.txt"
    write_shots(path, sorted(set(picks)))
    print(
        f"{len(set(picks))} pilot shots of {len(have)} listed shots in {args.reference} -> {path}"
    )


# ------------------------------------------------------------------ stage (login node)


def listing(base: str) -> dict[str, int]:
    """Relative path -> size of every object under a view."""
    files = {}
    for line in ae.pelican("ls", "-l", "-r", f"{ae.PELICAN}{base}").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith(base) and parts[1].isdigit():
            files[parts[0][len(base) + 1 :]] = int(parts[1])
    return files


def wanted(
    files: dict[str, int], index: dict, shot: int
) -> tuple[dict[str, str], list[str]]:
    """`({view path: local path}, missing points)`: the PTDATA files holding the Mirnov probes
    and Ip (through the shot's ``meta/index.json``), and each tree of `F.TREES` from one layout
    (``mdsplus/shots`` first), placed under ``mdsplus/shots/`` where MDSplus will look."""

    def ext(value: str) -> str:
        value = str(value).upper()
        return value if value.startswith(".") else "." + value

    pt = {k.lower(): ext(v) for k, v in (index.get("pointname_ext") or {}).items()}
    names = [*F.POINTS.values(), F.IP_POINT]
    missing = [n for n in names if n.lower() not in pt]
    exts = {pt[n.lower()] for n in names if n.lower() in pt}
    out = {
        rel: rel
        for rel in files
        if rel.startswith("ptdata/")
        and "." in rel
        and ext(rel.rsplit(".", 1)[1]) in exts
    }
    for tree in F.TREES:
        layouts: dict[str, dict[str, str]] = {}
        for rel in files:
            if not rel.startswith("mdsplus/") or "/" not in rel:
                continue
            folder, leaf = rel.rsplit("/", 1)
            stem, _, kind = leaf.rpartition(".")
            if stem.lower() == f"{tree}_{shot}" and kind in (
                "tree",
                "characteristics",
                "datafile",
            ):
                layouts.setdefault(folder, {})[kind] = rel
        for folder in sorted(layouts, key=lambda d: (d != "mdsplus/shots", d)):
            got = layouts[folder]
            if "tree" in got and "characteristics" in got:
                for kind, rel in got.items():
                    out[rel] = f"mdsplus/shots/{tree}_{shot}.{kind}"
                break
    return out, missing


def get(base: str, rel: str, dest: Path, size: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    err = ""
    for attempt in range(ae.ATTEMPTS):
        try:
            dest.unlink(missing_ok=True)
            ae.pelican("get", f"{ae.PELICAN}{base}/{rel}", str(dest), pinned=True)
            got = dest.stat().st_size if dest.exists() else 0
            if got == size:
                return
            err = f"{got} of {size} bytes"
        except ae.AuthError:
            raise
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            err = str(error)[-300:]
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"{rel}: {err}")


def stage_one(shot: int, streams: int) -> dict:
    final = ROOT / "stage" / str(shot)
    if (final / "meta" / "stage.json").exists():
        info = json.loads((final / "meta" / "stage.json").read_text())
        return {"shot": shot, "status": "staged", "bytes": info["bytes"], "s": 0.0}
    base = ae.VIEW.format(prefix=shot // 100, shot=shot)
    part = ROOT / "stage" / f".{shot}.part"
    shutil.rmtree(part, ignore_errors=True)
    t0 = time.monotonic()
    try:
        files = listing(base)
        if "meta/index.json" not in files:
            return {"shot": shot, "status": "no_view"}
        get(
            base,
            "meta/index.json",
            part / "meta" / "index.json",
            files["meta/index.json"],
        )
        index = json.loads((part / "meta" / "index.json").read_text())
        todo, missing = wanted(files, index, shot)
        if all(F.POINTS[row] in missing for row in F.TOROIDAL):
            shutil.rmtree(part, ignore_errors=True)
            return {"shot": shot, "status": "no_mirnov", "missing": missing}
        if F.IP_POINT in missing:
            shutil.rmtree(part, ignore_errors=True)
            return {"shot": shot, "status": "no_ip"}
        with ThreadPoolExecutor(streams) as pool:
            list(
                pool.map(
                    lambda kv: get(base, kv[0], part / kv[1], files[kv[0]]),
                    todo.items(),
                )
            )
        nbytes = sum(files[rel] for rel in todo)
        trees = sorted(
            {
                Path(p).name.rsplit("_", 1)[0]
                for p in todo.values()
                if p.startswith("mdsplus/")
            }
        )
        (part / "meta" / "stage.json").write_text(
            json.dumps(
                {
                    "shot": shot,
                    "view": base,
                    "files": todo,
                    "bytes": nbytes,
                    "trees": trees,
                    "missing": missing,
                },
                indent=1,
            )
        )
        shutil.rmtree(final, ignore_errors=True)
        os.replace(part, final)
        return {
            "shot": shot,
            "status": "staged",
            "bytes": nbytes,
            "trees": trees,
            "missing": missing,
            "s": round(time.monotonic() - t0, 1),
        }
    except ae.AuthError:
        shutil.rmtree(part, ignore_errors=True)
        raise
    except Exception as error:  # noqa: BLE001 - network faults: retried next run
        shutil.rmtree(part, ignore_errors=True)
        return {
            "shot": shot,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}"[-300:],
        }


def sbatch(script: Path, env: dict, dependency: str | None = None, extra=()) -> str:
    """``ae.sbatch`` with extra sbatch options (an array); a full submit limit is waited out."""
    cmd = ["sbatch", "--parsable", *extra]
    if dependency:
        cmd.append(f"--dependency={dependency}")
    cmd.append(str(script))
    for attempt in range(ae.SUBMIT_TRIES):
        out = subprocess.run(
            cmd, capture_output=True, text=True, env={**os.environ, **env}, check=False
        )
        if out.returncode == 0:
            return out.stdout.strip().split(";")[0]
        if "MaxSubmitJob" not in out.stderr or attempt == ae.SUBMIT_TRIES - 1:
            raise subprocess.CalledProcessError(
                out.returncode, cmd, out.stdout, out.stderr
            )
        if attempt == 0:
            print(
                f"sbatch {script.name}: submit limit reached, retrying every "
                f"{ae.SUBMIT_WAIT_S} s",
                flush=True,
            )
        time.sleep(ae.SUBMIT_WAIT_S)
    raise AssertionError("unreachable")


def submit_chain(
    path: Path, n: int, *, per_task: int, start: str = "extract", force: bool = False
) -> list[str]:
    """extract -> masks -> labels for the list `path` (`start` "masks" skips extract). Each
    step waits for the one before with afterany: a shot a step could not do is recorded by the
    next (no corpus file, "no TokEye mask"), never a stuck chain. The labels array is whole
    and new for its list (omnimode: never resubmit a subset of an array)."""
    env = {
        "LIST": str(path),
        "MODE_FDP_ROOT": str(ROOT),
        "OMNIMODE_REPO": str(OMNIMODE_REPO),
        "LABEL_ARGS": "--force" if force else "",
    }
    ids, dep = [], None
    if start == "extract":
        ids.append(sbatch(JOBS["extract"], env))
        dep = f"afterany:{ids[-1]}"
    ids.append(sbatch(JOBS["masks"], env, dep))
    tasks = max(1, math.ceil(n / per_task))
    ids.append(
        sbatch(
            JOBS["labels"], env, f"afterany:{ids[-1]}", extra=[f"--array=0-{tasks - 1}"]
        )
    )
    ae.log(
        "jobs",
        list=str(path),
        shots=n,
        jobs=ids,
        label_tasks=tasks,
        start=start,
        force=force,
    )
    print(
        f"{path.name}: {n} shots -> jobs {' -> '.join(ids)} ({tasks} label tasks)",
        flush=True,
    )
    return ids


def cmd_stage(args) -> None:
    shots = listed()
    if args.list:
        want = set(read_shots(args.list))
        shots = [s for s in shots if s in want]
    store = store_dir()
    staged = ae.last_status("stage")
    extracted = ae.last_status("extract")
    batched = {
        s for path in (ROOT / "batches").glob("b[0-9]*.txt") for s in read_shots(path)
    }

    def needed(shot: int) -> bool:
        if F.corpus_path(ROOT / "corpus", shot).exists():
            return False
        ex = extracted.get(shot, {}).get("status")
        if ex in RETRY_EXTRACT:
            # its label (if any) is a no-window record, which omnimode recomputes once the
            # corpus file is there (modes.run.label_shot)
            return True
        if (store / f"{shot}.json").exists():
            return False
        if staged.get(shot, {}).get("status") in FINAL_STAGE:
            return False
        # in a batch and not yet extracted: its jobs are pending (a batch whose jobs were
        # lost is resubmitted whole with `submit --list`), never staged into a second batch
        return ex is None and shot not in batched

    todo = [s for s in shots if needed(s)]
    if args.limit:
        todo = todo[: args.limit]
    print(
        f"{len(todo)} shots to stage of {len(shots)} listed; {args.parallel} in flight; batch "
        f"{args.batch}; budget {args.budget_gb} GB; submit {args.submit}",
        flush=True,
    )
    (ROOT / "stage").mkdir(parents=True, exist_ok=True)
    (ROOT / "batches").mkdir(parents=True, exist_ok=True)
    (ROOT / "omnimode" / "inputs").mkdir(parents=True, exist_ok=True)
    views = ae.Views()
    k = len(list((ROOT / "batches").glob("b[0-9]*.txt")))
    batch: list[int] = []
    counts: Counter = Counter()
    t_start, nbytes = time.monotonic(), 0

    def flush(force: bool = False) -> None:
        nonlocal batch, k
        if batch and (force or len(batch) >= args.batch):
            path = ROOT / "batches" / f"b{k:04d}.txt"
            write_shots(path, sorted(batch))
            if args.submit:
                submit_chain(path, len(batch), per_task=args.per_task)
            else:
                print(
                    f"{path.name}: {len(batch)} shots written (not submitted)",
                    flush=True,
                )
            k += 1
            batch = []
            clean(store, keep_mirnov=args.keep_mirnov)

    def record(row: dict) -> None:
        nonlocal nbytes
        ae.log("stage", **row)
        counts[row["status"]] += 1
        if row["status"] == "staged":
            batch.append(row["shot"])
            nbytes += row.get("bytes", 0)
        done = sum(counts.values())
        if done % 20 == 0:
            rate = done / max(time.monotonic() - t_start, 1)
            print(
                f"{done}/{len(todo)} {dict(counts)} | {nbytes / 1e9:.1f} GB | "
                f"{rate * 3600:.0f} shots/h",
                flush=True,
            )
        flush()

    pending: dict = {}
    try:
        with ThreadPoolExecutor(args.parallel) as pool:
            for shot in todo:
                try:
                    has = views.has(shot)
                except ae.AuthError:
                    raise
                except Exception as error:  # noqa: BLE001 - a listing that keeps failing
                    record(
                        {
                            "shot": shot,
                            "status": "failed",
                            "error": f"view listing: {error}"[-300:],
                        }
                    )
                    continue
                if not has:
                    record({"shot": shot, "status": "no_view"})
                    continue
                while len(pending) >= args.parallel:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for fut in done:
                        pending.pop(fut)
                        record(fut.result())  # an AuthError propagates from here
                while disk_gb(ROOT / "stage", ROOT / "corpus") > args.budget_gb:
                    clean(store, keep_mirnov=args.keep_mirnov)
                    time.sleep(120)
                pending[pool.submit(stage_one, shot, args.streams)] = shot
            for fut in list(pending):
                pending.pop(fut)
                record(fut.result())
    except ae.AuthError as error:
        ae.log("stage", event="auth_error", error=str(error)[:300])
        print(
            "FDP REFUSED, stopping (an expired login: run `fdp login`, then rerun):",
            error,
            flush=True,
        )
        flush(force=True)
        raise SystemExit(3) from None
    flush(force=True)
    print(
        f"staging done: {dict(counts)} in {(time.monotonic() - t_start) / 60:.1f} min, "
        f"{nbytes / 1e9:.1f} GB",
        flush=True,
    )


# ------------------------------------------------------------------ extract (CPU job)


def extract_one(job: tuple) -> dict:
    """One shot in a fresh worker process (``maxtasksperchild=1``: the PTDATA engine keeps its
    first configuration); the stage folder goes unless the shot will be restaged or `keep`."""
    shot, root, keep = job
    _set_root(Path(root))
    dest = ROOT / "stage" / str(shot)
    if not (dest / "meta" / "stage.json").exists():
        return {"shot": shot, "status": "no_staged_file"}
    row = _extract(shot, dest)
    if (
        not keep
        and row["status"] not in RETRY_EXTRACT
        and ae.inside(dest, ROOT / "stage")
    ):
        shutil.rmtree(dest, ignore_errors=True)
    return row


def _extract(shot: int, dest: Path) -> dict:
    F.use_local_stage(dest)
    t0 = time.monotonic()
    row: dict = {"shot": shot}
    try:
        records, errors = {}, {}
        for r, name in F.POINTS.items():
            try:
                t, y = F.read_ptdata(name, shot)
            except Exception as error:  # noqa: BLE001 - a dead or absent probe is a NaN row
                errors[name] = f"{type(error).__name__}: {error}"[:160]
                continue
            if t.size >= 2 and t.shape == y.shape and np.isfinite(y).any():
                records[r] = (t, y)
            else:
                errors[name] = f"data {y.shape}, times {t.shape}"
        try:
            times, Y, rows = F.assemble(records)
        except F.RateError as error:
            return {
                **row,
                "status": "mirnov_rate",
                "error": str(error),
                "probe_errors": errors,
            }
        except ValueError as error:
            return {
                **row,
                "status": "no_mirnov",
                "error": str(error),
                "probe_errors": errors,
            }
        records.clear()
        try:
            ip_t, ip_a = F.read_ptdata(F.IP_POINT, shot)
        except Exception as error:  # noqa: BLE001
            return {
                **row,
                "status": "no_ip",
                "error": f"{type(error).__name__}: {error}"[:300],
            }
        inputs = ROOT / "omnimode" / "inputs"
        eq = {}
        for tree in F.EQ_TREES:
            try:
                F.save_npz(inputs / tree / f"{shot}.npz", F.read_eq(tree, shot))
                eq[tree] = "ok"
            except Exception as error:  # noqa: BLE001 - a tree that did not run: next rung
                eq[tree] = f"{type(error).__name__}: {error}"[:160]
        rot, rot_errors = F.read_rot(shot)
        if rot:
            F.save_npz(inputs / "rot" / f"{shot}.npz", rot)
        # the corpus file last: it marks the shot extracted (`clean` then frees its stage folder)
        nbytes = F.write_corpus(
            F.corpus_path(ROOT / "corpus", shot), times, Y, rows, ip_t, ip_a
        )
        span = [round(float(times[0]), 3), round(float(times[-1]), 3)]
        del Y, times
        return {
            **row,
            "status": "ok",
            "probes": rows,
            "probe_errors": errors,
            "span_ms": span,
            "mb": round(nbytes / 1e6, 1),
            "eq": eq,
            "rot": sorted(k for k in rot if k.startswith("w")),
            "rot_errors": rot_errors,
            "s": round(time.monotonic() - t0, 1),
        }
    except Exception as error:  # noqa: BLE001 - one bad shot never stops the job
        return {
            **row,
            "status": "extract_error",
            "error": f"{type(error).__name__}: {error}"[:300],
        }


def cmd_extract(args) -> None:
    from multiprocessing import Pool

    shots = read_shots(args.list)
    jobs = [
        (s, str(ROOT), args.keep_stage)
        for s in shots
        if not F.corpus_path(ROOT / "corpus", s).exists()
    ]
    (ROOT / "corpus").mkdir(parents=True, exist_ok=True)
    print(
        f"extract {len(jobs)} of {len(shots)} shots with {args.workers} workers at {ae.now()}",
        flush=True,
    )
    t0, counts = time.monotonic(), Counter()
    with Pool(args.workers, maxtasksperchild=1) as pool:
        for i, row in enumerate(pool.imap_unordered(extract_one, jobs, chunksize=1), 1):
            ae.log("extract", job=os.environ.get("SLURM_JOB_ID", "local"), **row)
            counts[row["status"]] += 1
            if i % 20 == 0 or i == len(jobs):
                print(
                    f"{i}/{len(jobs)} {dict(counts)} {time.monotonic() - t0:.0f} s",
                    flush=True,
                )
    print(f"extract done {dict(counts)} in {time.monotonic() - t0:.0f} s", flush=True)


# ------------------------------------------------------------------ submit, clean, redo, status


def cmd_submit(args) -> None:
    path = Path(args.list).resolve()
    submit_chain(
        path,
        len(read_shots(path)),
        per_task=args.per_task,
        start=args.start,
        force=args.force,
    )


def clean(store: Path, *, keep_mirnov: bool = False) -> dict:
    """Delete the corpus file of each labelled shot whose label is final (not `retryable`)."""
    corpus = ROOT / "corpus"
    counts = Counter()
    gb = 0.0
    for path in sorted(corpus.glob("*_processed.h5")) if corpus.exists() else []:
        info = label_info(int(path.name.split("_")[0]), store)
        if info is None:
            counts["waiting"] += 1
        elif retryable(info):
            counts["kept_for_redo"] += 1
        elif keep_mirnov:
            counts["kept"] += 1
        else:
            try:
                size = path.stat().st_size
                path.unlink()
            except FileNotFoundError:
                continue
            gb += size / 1e9
            counts["deleted"] += 1
    # a stage folder whose shot is extracted or labelled (an extract job killed before it
    # deleted the folder): nothing will read it again
    stage = ROOT / "stage"
    for folder in sorted(stage.iterdir()) if stage.exists() else []:
        if not folder.name.isdigit() or not ae.inside(folder, stage):
            continue
        shot = int(folder.name)
        if F.corpus_path(corpus, shot).exists() or (store / f"{shot}.json").exists():
            gb += disk_gb(folder)
            shutil.rmtree(folder, ignore_errors=True)
            counts["stage_deleted"] += 1
    if counts.get("deleted") or counts.get("stage_deleted"):
        ae.log("clean", **counts, gb=round(gb, 1))
    return {**counts, "gb": round(gb, 1)}


def cmd_clean(args) -> None:
    print(clean(store_dir(), keep_mirnov=args.keep_mirnov))


def cmd_redo(args) -> None:
    store = store_dir()
    redo = [
        s
        for s in listed()
        if F.corpus_path(ROOT / "corpus", s).exists()
        and (info := label_info(s, store)) is not None
        and retryable(info)
    ]
    if not redo:
        print("no retryable shot with its corpus file")
        return
    path = ROOT / "batches" / f"redo_{datetime.now().astimezone():%Y%m%d_%H%M%S}.txt"
    write_shots(path, redo)
    print(f"{len(redo)} retryable shots -> {path}")
    if args.submit:
        submit_chain(path, len(redo), per_task=args.per_task, start="masks", force=True)


def cmd_status(args) -> None:
    shots = listed()
    store = store_dir()
    stage_rows = ae.last_status("stage").values()
    extract_rows = ae.last_status("extract").values()
    staged = Counter(r.get("status") for r in stage_rows)
    extracted = Counter(r.get("status") for r in extract_rows)
    trees = Counter(t for r in stage_rows for t in r.get("trees") or ())
    eq = Counter(
        t for r in extract_rows for t, v in (r.get("eq") or {}).items() if v == "ok"
    )
    rot = Counter(w for r in extract_rows for w in r.get("rot") or ())
    labels = Counter()
    for s in shots:
        info = label_info(s, store)
        labels["(none yet)" if info is None else str(info.get("status"))] += 1
    masks = ROOT / "omnimode" / "frontend" / "masks"
    n_masks = sum(1 for _ in masks.rglob("*.npz")) if masks.exists() else 0
    corpus = (
        list((ROOT / "corpus").glob("*_processed.h5"))
        if (ROOT / "corpus").exists()
        else []
    )
    print(f"listed {len(shots)}")
    print(f"stage   {dict(staged)}; shots staged with each tree {dict(trees)}")
    print(
        f"extract {dict(extracted)}; equilibria read {dict(eq)}; rotation {dict(rot)}"
    )
    print(f"masks   {n_masks} files")
    print(f"labels  {dict(labels.most_common())}")
    print(
        f"on disk: stage {disk_gb(ROOT / 'stage'):.1f} GB, corpus {len(corpus)} files "
        f"{disk_gb(ROOT / 'corpus'):.1f} GB, omnimode {disk_gb(ROOT / 'omnimode'):.1f} GB"
    )


# ------------------------------------------------------------------ compare (pilot check)


def compare_shot(ours: Path, ref: Path, window_s: tuple[float, float]) -> list[dict]:
    """Each fetched row against the full corpus over `window_s`: correlation, rms of the
    difference over the reference's rms, and the sample offset that aligns them best."""
    import h5py

    out = []
    with h5py.File(ours, "r") as a, h5py.File(ref, "r") as b:
        xa, xb = a["mirnov/xdata"], b["mirnov/xdata"]
        fa = (xa.shape[0] - 1) / (float(xa[-1]) - float(xa[0]))
        fb = (xb.shape[0] - 1) / (float(xb[-1]) - float(xb[0]))
        ia = round((window_s[0] - float(xa[0])) * fa)
        ib = round((window_s[0] - float(xb[0])) * fb)
        n = int((window_s[1] - window_s[0]) * fb)
        if ia < 0 or ib < 0 or ia + n > xa.shape[0] or ib + n > xb.shape[0]:
            return [{"row": None, "note": f"window {window_s} outside a record"}]
        for row in F.ROWS:
            ya = np.asarray(
                a["mirnov/ydata"][row, ia - 5 : ia + n + 5], dtype=np.float64
            )
            yb = np.asarray(b["mirnov/ydata"][row, ib : ib + n], dtype=np.float64)
            if not (np.isfinite(yb).all() and np.isfinite(ya).all()) or yb.std() == 0:
                out.append({"row": row, "point": F.POINTS[row], "note": "NaN or flat"})
                continue
            best = max(range(11), key=lambda k: np.corrcoef(ya[k : k + n], yb)[0, 1])
            seg = ya[best : best + n]
            out.append(
                {
                    "row": row,
                    "point": F.POINTS[row],
                    "offset": best - 5,
                    "corr": round(float(np.corrcoef(seg, yb)[0, 1]), 5),
                    "rel_rms": round(float(np.std(seg - yb) / np.std(yb)), 5),
                    "fs_ours": round(fa),
                    "fs_ref": round(fb),
                }
            )
    return out


def cmd_compare(args) -> None:
    shots = read_shots(args.list) if args.list else listed()
    pairs = [
        (
            s,
            F.corpus_path(ROOT / "corpus", s),
            Path(args.reference) / f"{s}_processed.h5",
        )
        for s in shots
    ]
    pairs = [p for p in pairs if p[1].exists() and p[2].exists()]
    if not pairs:
        raise SystemExit("no shot has both a fetched and a reference corpus file")
    worst = 1.0
    for shot, ours, ref in pairs[: args.n]:
        rows = compare_shot(ours, ref, (args.t0, args.t0 + args.seconds))
        good = [r for r in rows if "corr" in r]
        worst = min([worst, *(r["corr"] for r in good)])
        print(
            f"{shot}: {len(good)} rows compared; min corr "
            f"{min((r['corr'] for r in good), default=float('nan')):.5f}; max rel rms "
            f"{max((r['rel_rms'] for r in good), default=float('nan')):.5f}; offsets "
            f"{sorted({r['offset'] for r in good})}; other {[r for r in rows if 'corr' not in r]}"
        )
    print(f"worst correlation {worst:.5f} over {min(len(pairs), args.n)} shots")


# ------------------------------------------------------------------ publish


def cmd_publish(args) -> None:
    store = store_dir()
    env = {
        "STORE": str(store),
        "SHOTS": str(ROOT / "shots.txt"),
        "RUN": args.run,
        "OUT_ROOT": str(ROOT / "publish"),
        "MODE_FDP_ROOT": str(ROOT),
    }
    job = sbatch(JOBS["publish"], env)
    print(
        f"publish job {job}: {store} -> {ROOT / 'publish'}/<event>/{args.run}; then\n"
        f"  rsync -a {ROOT / 'publish'}/ {ae.DEV}/"
    )


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root", type=Path, default=None, help=f"run root (default {ROOT})")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shots")
    s.add_argument("--candidates", type=Path, default=CANDIDATES)
    s = sub.add_parser("pilot")
    s.add_argument("--n", type=int, default=4)
    s.add_argument("--reference", type=Path, default=REFERENCE)

    def chain(q):
        q.add_argument(
            "--per-task", type=int, default=20, help="shots a label array task"
        )

    s = sub.add_parser("stage")
    s.add_argument("--list", type=Path, help="only these shots (of shots.txt)")
    s.add_argument("--limit", type=int)
    s.add_argument("--parallel", type=int, default=6, help="shots in flight")
    s.add_argument("--streams", type=int, default=3, help="pelican transfers a shot")
    s.add_argument("--batch", type=int, default=200)
    s.add_argument("--budget-gb", type=float, default=4000.0)
    s.add_argument("--submit", action="store_true")
    s.add_argument(
        "--keep-mirnov", action="store_true", help="never delete a corpus file"
    )
    chain(s)
    s = sub.add_parser("extract")
    s.add_argument("--list", type=Path, required=True)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--keep-stage", action="store_true")
    s = sub.add_parser("submit")
    s.add_argument("--list", type=Path, required=True)
    s.add_argument("--start", choices=("extract", "masks"), default="extract")
    s.add_argument(
        "--force", action="store_true", help="relabel shots that have labels"
    )
    chain(s)
    s = sub.add_parser("clean")
    s.add_argument("--keep-mirnov", action="store_true")
    s = sub.add_parser("redo")
    s.add_argument("--submit", action="store_true")
    chain(s)
    sub.add_parser("status")
    s = sub.add_parser("compare")
    s.add_argument("--list", type=Path)
    s.add_argument("--reference", type=Path, default=REFERENCE)
    s.add_argument("--n", type=int, default=4)
    s.add_argument("--t0", type=float, default=1.0, help="window start (s)")
    s.add_argument("--seconds", type=float, default=0.5)
    s = sub.add_parser("publish")
    s.add_argument("--run", default=RUN)
    args = p.parse_args(argv)
    if args.root is not None:
        _set_root(args.root)
    {
        "shots": cmd_shots,
        "pilot": cmd_pilot,
        "stage": cmd_stage,
        "extract": cmd_extract,
        "submit": cmd_submit,
        "clean": cmd_clean,
        "redo": cmd_redo,
        "status": cmd_status,
        "compare": cmd_compare,
        "publish": cmd_publish,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
