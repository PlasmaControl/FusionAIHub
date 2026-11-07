#!/usr/bin/env python
"""Confinement mode (L, H, QH, WPQH) on a 1 ms grid for every DIII-D plasma shot FDP serves, by
``confine-ours`` (``confinement_ours.py``, variant ``base3``: five run-day folds, three seeds each).
The confinement twin of ``elm_fdp_label.py``: the same candidates, logs, batches and status rules
(``ae_fdp_label.py``), the same byte-range reads of FDP's MDSplus files (``elm_mds_ranges.py``).

The inputs are those of ``labeler.confinement.zerod``: D-alpha (the first live chord of
``\\SPECTROSCOPY::FS01``-``FS08``, at its native rate), ``\\BCI::DENV2F``, ``\\EFIT01`` beta_N and
W_MHD (from FDP's code archive) and the eight beams' ``\\NB::TOP.NB<b>:PINJ_<b>``.

Subcommands, each resumable (a shot already done is skipped):

``stage`` (login node, fdp env under ``fdp -D d3d run``): per candidate with an FDP view, into
    ``stage/<shot>/``: (1) the filterscopes' record lengths; a PTDATA call (shots before ~131000)
    is ``ptdata_record``; (2) the beams' power: a shot whose total stays under 200 kW (gaps of
    50 ms bridged) for all but 20 ms before the plasma end is ``no_beam`` and nothing more is
    fetched, as confine-ours labels beam-on time only; (3) FS01, FS02, ... until one is live
    (over half its samples finite, not constant: ``zerod.read_d_alpha``'s rule); (4) DENV2F
    and EFIT01, which a shot may lack (the network reads a mask then, as in training). Every
    ``--batch`` staged shots it writes ``batches/<k>.csv`` and, with ``--submit``, submits the
    extract job. Stops at the first refused token.
``extract`` (CPU job, fdp env, no network): ``stage/<shot>/`` -> ``inputs/<shot>.npz``, the
    seven-channel 1 ms grid of ``zerod.assemble`` (its transforms, from these records). The
    staged folder is then deleted.
``infer`` (GPU job, stellarai env; the networks are tiny, so one batched job covers every
    extracted shot): ``inputs/`` -> ``work/<shot>.npz`` (class probabilities, float16) and
    ``results/<job>.{shots,spans}.csv``, as ``confinement_ours.py apply`` does: the mean softmax
    of the fifteen networks (a curated shot: the three of the fold that held it out), the
    beam-on mask (total power >= 200 kW, gaps <= 50 ms bridged) cut at the plasma end, the ELM gate
    (``labeler.confinement.elm_gate``: QH or WPQH -> H inside a train of the ELM pipeline's onsets,
    ``--elm-root``; ``--no-elm-gate`` turns it off), the mode filter (21 bins), segments of 20 ms or
    more, and category 5 (uncertain) under a mean class probability of 0.7. ``work/<shot>.npz``
    ``prob`` stays the networks' output; ``elm_flip`` marks the bins the gate flipped.
``publish`` (login node): the label folder ``--out`` (the deployment for the all-shot root,
    ``dev/confinement/<root name>`` for any other).

**1 ms codes.** Millisecond ``k`` is ``[k, k + 1)`` ms of the shot clock. Inside the assessed
window ``[window_start_ms, window_end_ms)`` (first to last beam-on millisecond) a millisecond is
1 H, 2 L, 3 QH, 4 WPQH, 5 uncertain, or 0 when no segment covers it (a beam-off stretch of more
than 50 ms, or a run too short to keep).
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
ALL_SHOT_ROOT = LABELER_ROOT / "conf_fdp_1ms"
ROOT = Path(os.environ.get("CONF_FDP_ROOT", ALL_SHOT_ROOT))
OUT = ae.DEPLOYMENT / "confinement" / "confine_ours_fdp_1ms"
CANDIDATES = LABELER_ROOT / "ae_fdp_1ms/candidates.csv"
ELM_ROOT = LABELER_ROOT / "elm_inst2_fdp_1ms"  # the ELM pipeline's run whose onsets gate QH and WPQH
VARIANT = "base3"

FS_TAGS = [rf"\SPECTROSCOPY::FS{i:02d}" for i in range(1, 9)]  # the corpus' filterscopes rows 0-7
DENSITY_TAG = r"\BCI::DENV2F"
EFIT_TAGS = {"betan": r"\EFIT01::TOP.RESULTS.AEQDSK:BETAN", "wmhd": r"\EFIT01::TOP.RESULTS.AEQDSK:WMHD"}
BEAMS = ("15L", "15R", "21L", "21R", "30L", "30R", "33L", "33R")  # zerod.BEAMS
NB_TAGS = [rf"\NB::TOP.NB{b}:PINJ_{b}" for b in BEAMS]
# The beams' sum in kW, same 0.1 ms clock (equal to the per-beam sum to 1e-7 on 157097 and 185786); the
# only power record of shots without the per-beam nodes (TreeNNF before ~148000). Per-beam stays first,
# as training read it.
NB_TOTAL_TAG = r"\NB::TOP:PINJ"
NB_TOTAL_W = 1e3
# The beam gate of confinement_ours.py (BEAM_ON_W, BEAM_GAP_MS, MIN_SEGMENT_MS); the infer step uses
# the module's own values and checks these against them.
BEAM_ON_W, BEAM_GAP_MS, MIN_ON_MS = 2e5, 50, 20
# Training read D-alpha at 10 kHz (the corpus' resampling, 146 shots) or 20-50 kHz (native, 270)
MAX_FS_DT_MS = 0.1005
KEEP_STAGED = {"read_failed", "extract_error", "no_staged_file"}  # restaged or looked at; the rest is final
FINAL_STAGE = {"no_view", "no_fs", "ptdata_record", "no_beam", "no_beam_record"}
#: The infer step's findings for a shot it did not label (final, not faults).
INFER_FINDINGS = {"no_beam", "no_beam_record", "no_segment"}
CATEGORIES = {1: "H", 2: "L", 3: "QH", 4: "WPQH", 5: "uncertain"}


def _set_root(root: Path) -> None:
    global ROOT
    ROOT = Path(root)
    ae.ROOT = ROOT  # the shared log and result readers


def stage_dir_of(shot: int) -> Path:
    return ROOT / "stage" / str(shot)


def held_gb() -> float:
    """Bytes really on disk under stage/ (the datafiles are sparse); entries vanishing mid-listing
    (extract jobs delete folders) are skipped."""
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


def bridge_gaps(active: np.ndarray, gap: int) -> np.ndarray:
    """confinement_ours.bridge_gaps (the stage step runs without importing that script)."""
    out = active.copy()
    edges = np.flatnonzero(np.diff(np.r_[False, active, False].astype(np.int8)))
    for end, start in zip(edges[1:-1:2], edges[2::2], strict=False):
        if start - end <= gap:
            out[end:start] = True
    return out


def beam_power(got: dict, n: int | None = None, total: bool = False) -> tuple[np.ndarray | None, list[str]]:
    """The beams' total power (W) as 1 ms bin means on [0, n) ms (n: to the last beam sample), NaN
    where no beam has a sample, as ``zerod.read_beam_power`` and ``bin_stats`` give it; None when no
    beam has a record. Also the beams that could not be read. ``total``: from NB_TOTAL_TAG (kW)."""
    from labeler.confinement import zerod

    rows, notes = [], []
    for tag in [NB_TOTAL_TAG] if total else NB_TAGS:
        v = got.get(tag, ("not read",))
        if len(v) == 1:
            notes.append(f"{tag.rsplit(':', 1)[1]}: {v[0]}"[:120])
            continue
        y, t = np.asarray(v[0], np.float32).ravel(), np.asarray(v[1], np.float64).ravel()
        if y.size < 2 or y.shape != t.shape:
            notes.append(f"{tag.rsplit(':', 1)[1]}: data {y.shape}, times {t.shape}")
            continue
        rows.append((t, y * NB_TOTAL_W if total else y))
    if not rows:
        return None, notes
    if n is None:
        n = int(np.floor(max(float(np.nanmax(t)) for t, _ in rows))) + 1
    means = np.stack([zerod.bin_stats(t, y, n)[0] for t, y in rows])
    total = np.nansum(means, axis=0).astype(np.float32)
    total[np.isnan(means).all(axis=0)] = np.nan
    return total, notes


def beam_on(power: np.ndarray, end_ms: float) -> np.ndarray:
    """confine-ours' active mask, cut at the plasma end (frames k < floor(end_ms), as the AE and ELM
    windows)."""
    active = bridge_gaps(np.nan_to_num(power, nan=0.0) >= BEAM_ON_W, BEAM_GAP_MS)
    active[max(int(np.floor(end_ms)), 0):] = False
    return active


def live(y: np.ndarray) -> bool:
    """``zerod.read_d_alpha``'s rule: over half the samples finite, and not constant."""
    finite = np.isfinite(y)
    if not finite.size:
        return False
    sample = y[finite][:: max(1, int(finite.sum()) // 20000)].astype(np.float64)
    return bool(finite.mean() > 0.5 and sample.size and np.std(sample) > 0)


# ------------------------------------------------------------------ stage (login node)


def fs_records(st: R.SparseTrees, shot: int) -> list[tuple[str, int]]:
    """FS01-FS08 that hold a record (NCI record length > 0), in order, with that length."""
    st.open_files("spectroscopy")
    T = R._open(st, "spectroscopy", shot)
    out = []
    for tag in FS_TAGS:
        try:
            rlen = int(T.getNode(tag).rlength)
        except Exception:  # noqa: BLE001, S112 - a chord this era does not have
            continue
        if rlen > 0:
            out.append((tag, rlen))
    return out


def _stage(shot: int, part: Path, end_ms: float, whole: bool) -> dict:
    st = R.SparseTrees(shot, part)
    try:
        order = fs_records(st, shot)
    except R.Missing as error:
        return {"status": "no_fs", "error": f"no tree file {error}"}
    if not order:
        return {"status": "no_fs", "error": "no filterscope record"}
    if order[0][1] < R.SMALL_RECORD:  # an expression: a PTDATA call before ~131000
        info = R.stage_nodes(shot, part, {"spectroscopy": [order[0][0]]}, whole=whole, trees=st)
        if info["ptdata"]:
            return {"status": "ptdata_record", "error": "; ".join(info["ptdata"])[:300]}
    # the beams: no beam-on time, nothing else to fetch
    info = R.stage_nodes(shot, part, {"nb": NB_TAGS}, whole=whole, trees=st)
    power, notes = beam_power(R.read_nodes(shot, part, {"nb": NB_TAGS}))
    beam_source = "beams"
    if power is None:  # no per-beam record: the NB total
        more = R.stage_nodes(shot, part, {"nb": [NB_TOTAL_TAG]}, whole=whole, trees=st)
        power, total_notes = beam_power(R.read_nodes(shot, part, {"nb": [NB_TOTAL_TAG]}), total=True)
        info["ptdata"] += more["ptdata"]
        notes += total_notes
        beam_source = "nb_total"
        if power is None and not info["ptdata"] and any("NODATA" in n for n in total_notes):
            # the NB total node exists but was never written: no beam fired (d3drdb pbeam empty)
            return {"status": "no_beam", "beam_on_ms": 0, "error": "; ".join(total_notes)[:300]}
    if power is None:
        why = "; ".join(info["ptdata"]) if info["ptdata"] else "; ".join(notes)
        return {"status": "no_beam_record", "error": why[:300]}
    on_ms = int(beam_on(power, end_ms).sum())
    if on_ms < MIN_ON_MS:
        return {"status": "no_beam", "beam_on_ms": on_ms, "beam_max_w": float(np.nanmax(power))}
    # the first live filterscope
    chosen, tried = None, []
    for tag, _ in order:
        info = R.stage_nodes(shot, part, {"spectroscopy": [tag]}, whole=whole, trees=st)
        if info["ptdata"]:
            return {"status": "ptdata_record", "error": "; ".join(info["ptdata"])[:300]}
        v = R.read_nodes(shot, part, {"spectroscopy": [tag]})[tag]
        name = tag.rsplit("::", 1)[1]
        if len(v) == 1:
            tried.append(f"{name}: {v[0]}"[:100])
            continue
        y = np.asarray(v[0], np.float32).ravel()
        if live(y):
            t = np.asarray(v[1], np.float64).ravel()
            chosen = {"fs": tag, "fs_hz": round(1e3 / float(np.median(np.diff(t))), 1) if t.size > 1 else 0.0}
            break
        tried.append(f"{name}: not live")
    if chosen is None:
        return {"status": "no_fs", "error": ("none live: " + "; ".join(tried))[:300]}
    # density and EFIT01: optional
    info = R.stage_nodes(shot, part, {"bci": [DENSITY_TAG], "efit01": list(EFIT_TAGS.values())},
                         whole=whole, trees=st)
    chosen.update(beam_on_ms=on_ms, beam_source=beam_source, end_ms=end_ms, absent=info["missing"],
                  ptdata=info["ptdata"])
    (part / "conf.json").write_text(json.dumps(chosen))
    return {"status": "staged", "bytes": st.bytes, "whole": sorted(st.whole), **chosen}


def stage_one(shot: int, end_ms: float, root: str, whole: bool) -> dict:
    """One shot's sparse trees (a worker process: its own MDSplus tree paths)."""
    # `fdp run` points default_tree_path at pelican://; MDSplus then probes it on every spectroscopy
    # open (8-9 s each, 2026-11-06). Only the staged copies are opened here (<tree>_path), and without
    # it the reads were identical on 203064, 141349 and 184451 at 3 s a shot instead of 20-37 s.
    os.environ.pop("default_tree_path", None)
    _set_root(Path(root))
    dest = stage_dir_of(shot)
    if (dest / "conf.json").exists() and not whole:
        info = json.loads((dest / "staged.json").read_text())
        return {"shot": shot, "status": "staged", "bytes": info["bytes"], "s": 0.0}
    part = dest.with_name(f".{shot}.part")
    shutil.rmtree(part, ignore_errors=True)
    t0 = time.monotonic()
    try:
        row = _stage(shot, part, end_ms, whole)
    except R.AuthError:
        shutil.rmtree(part, ignore_errors=True)
        raise
    except Exception as error:  # noqa: BLE001 - network faults: retried next run
        shutil.rmtree(part, ignore_errors=True)
        return {"shot": shot, "status": "failed", "error": f"{type(error).__name__}: {error}"[-300:]}
    row = {"shot": shot, **row, "s": round(time.monotonic() - t0, 1)}
    if row["status"] != "staged":
        shutil.rmtree(part, ignore_errors=True)
        return row
    shutil.rmtree(dest, ignore_errors=True)  # a sparse staging being replaced by whole files
    os.replace(part, dest)
    return row


def submit_batch(k: int, rows: list[tuple[int, float]]) -> str:
    (ROOT / "batches").mkdir(parents=True, exist_ok=True)
    path = ROOT / "batches" / f"b{k:04d}.csv"
    ae.write_list(path, rows)
    ext = ae.sbatch(REPO / "scripts/labeler/conf_fdp_extract.sbatch", {"LIST": str(path), "CONF_FDP_ROOT": str(ROOT)})
    ae.log("jobs", batch=k, list=str(path), shots=len(rows), jobs=[ext])
    print(f"batch {k}: {len(rows)} shots -> extract {ext}", flush=True)
    return ext


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
            return False  # extracted to a final finding
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
                submit_batch(k, batch)
            else:
                ae.write_list(ROOT / "batches" / f"b{k:04d}.csv", batch)
                print(f"batch {k}: {len(batch)} shots written (not submitted)", flush=True)
            k += 1
            batch = []

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
            for shot, end_ms in todo:
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
                pending[pool.submit(stage_one, shot, end_ms, str(ROOT), args.whole)] = shot
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


def build_grid(fs: tuple, dens: tuple | None, efit: dict, power: np.ndarray | None) -> dict:
    """``zerod.assemble`` from records in hand: D-alpha ``(t_ms, y)``, density ``(t_ms, y)`` or
    None, ``{"betan": (t_ms, y), ...}``, and the beams' total power on 1 ms bins from 0 ms."""
    from confinement_zerod_fetch import bin_mean  # the training fetch's density binning

    from labeler.confinement import zerod

    t_ms, y = fs
    n = int(np.floor(t_ms[-1] / zerod.BIN_MS)) + 1
    if n <= 0:
        return {"grid": None}
    mean, peak, std = zerod.bin_stats(t_ms, y, n)
    ref = float(np.nanpercentile(y[np.isfinite(y) & (y > 0)], 95)) if (y > 0).any() else 1.0
    ref = max(ref, 1e-30)
    grid = np.full((len(zerod.CHANNELS), n), np.nan, dtype=np.float32)
    grid[0] = np.log10(np.maximum(mean, 0) / ref + 0.01)
    grid[1] = np.log10(np.maximum(peak, 0) / ref + 0.01)
    grid[2] = np.minimum(std / ref, 3.0)
    if dens is not None:
        t0, d = bin_mean(*dens)
        lo = round(float(t0) / zerod.BIN_MS)
        place = np.full(n, np.nan, dtype=np.float32)
        src = slice(max(0, -lo), min(len(d), n - lo))
        if src.stop > src.start:
            place[lo + src.start: lo + src.stop] = d[src]
        grid[3] = place / zerod.SCALES["dens"]
    for name, row in (("betan", 4), ("wmhd", 5)):
        if name in efit:
            grid[row] = zerod.interpolate_sparse(efit[name][0], efit[name][1], n) / zerod.SCALES[name]
    if power is not None:
        p = np.full(n, np.nan, dtype=np.float32)
        m = min(n, power.size)
        p[:m] = power[:m]
        grid[6] = p / zerod.SCALES["pinj"]
    lacks = [c for i, c in enumerate(zerod.CHANNELS) if c in zerod.SPARSE and not np.isfinite(grid[i]).any()]
    return {"grid": grid, "da_reference": ref, "lacks": lacks}


def extract_one(job: tuple) -> dict:
    """``_extract_one``; then the stage folder goes unless the shot will be restaged (or ``keep``)."""
    shot, end_ms, keep, root = job
    row = _extract_one(shot, end_ms, root)
    dest = stage_dir_of(shot)
    if not keep and row["status"] not in KEEP_STAGED and ae.inside(dest, ROOT / "stage"):
        shutil.rmtree(dest, ignore_errors=True)
    return row


def _broken(v: tuple) -> bool:
    """A read that failed for want of bytes, not a node's own NODATA (record length 0)."""
    text = str(v[0])
    return ("read failed" in text or "tree open failed" in text) and "NODATA" not in text


def _extract_one(shot: int, end_ms: float, root: str) -> dict:
    _set_root(Path(root))
    dest = stage_dir_of(shot)
    row = {"shot": shot}
    if not (dest / "conf.json").exists():
        return {**row, "status": "no_staged_file"}
    try:
        conf = json.loads((dest / "conf.json").read_text())
        whole = set(json.loads((dest / "staged.json").read_text())["whole"])
        fs_tag = conf["fs"]
        beam_source = conf.get("beam_source", "beams")
        nb_tags = [NB_TOTAL_TAG] if beam_source == "nb_total" else NB_TAGS
        got = R.read_nodes(shot, dest, {"spectroscopy": [fs_tag], "bci": [DENSITY_TAG],
                                        "efit01": list(EFIT_TAGS.values()), "nb": nb_tags})
        tree_of = {fs_tag: "spectroscopy", DENSITY_TAG: "bci", **{t: "efit01" for t in EFIT_TAGS.values()},
                   **{t: "nb" for t in nb_tags}}
        # a read that failed for want of bytes: restage with whole datafiles (--whole); after that an
        # optional channel that still fails is left out (masked) and the D-alpha chord is final
        broken = [t for t, v in got.items() if len(v) == 1 and _broken(v) and tree_of[t] not in whole]
        if broken:
            return {**row, "status": "read_failed",
                    "error": "; ".join(f"{t}: {got[t][0]}" for t in broken)[:300]}
        notes = {t: str(v[0])[:120] for t, v in got.items() if len(v) == 1}
        if fs_tag in notes:
            return {**row, "status": "fs_unreadable", "error": f"{fs_tag}: {notes[fs_tag]}"[:300]}

        def arrays(tag):
            v = got[tag]
            if len(v) == 1:
                return None
            y, t = np.asarray(v[0], np.float32).ravel(), np.asarray(v[1], np.float64).ravel()
            return (t, y) if y.size >= 2 and y.shape == t.shape else None

        fs = arrays(fs_tag)
        if fs is None:
            return {**row, "status": "bad_record", "error": f"{fs_tag}: data and times differ"}
        fs_dt = float(np.median(np.diff(fs[0])))
        if fs_dt > MAX_FS_DT_MS:
            return {**row, "status": "fs_rate_low", "fs_hz": round(1e3 / fs_dt, 1)}
        efit = {}
        for name, tag in EFIT_TAGS.items():
            a = arrays(tag)
            if a is not None:
                efit[name] = a
        power, _ = beam_power({t: got[t] for t in nb_tags}, total=beam_source == "nb_total")
        if power is None:
            return {**row, "status": "no_beam_record", "error": "; ".join(notes.get(t, "") for t in nb_tags)[:300]}
        built = build_grid(fs, arrays(DENSITY_TAG), efit, power)
        grid = built["grid"]
        if grid is None or not np.isfinite(grid[0, : max(int(np.floor(end_ms)), 0)]).any():
            return {**row, "status": "no_record_in_window", "record_end_ms": float(fs[0][-1])}
        out = ROOT / "inputs" / f"{shot}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{shot}.{os.getpid()}.npz.tmp")
        meta = {"da_source": "fdp:" + fs_tag.rsplit("::", 1)[1], "da_reference": built["da_reference"],
                "lacks": built["lacks"], "fs_hz": round(1e3 / fs_dt, 1), "plasma_end_ms": float(end_ms),
                "beam_source": beam_source,
                "absent": {**conf.get("absent", {}), **{t: v for t, v in notes.items() if t not in nb_tags}}}
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, grid=grid, meta=np.array(json.dumps(meta)))
        os.replace(tmp, out)
        return {**row, "status": "ok", "n_ms": int(grid.shape[1]), "fs_hz": meta["fs_hz"], "lacks": built["lacks"],
                "da_source": meta["da_source"], "beam_source": beam_source, "bytes": out.stat().st_size}
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


def _as_is(item):
    """The loader's collate: keep the numpy arrays (the batches are built per padded length)."""
    return item


def cmd_infer(args) -> None:
    import confinement_ours as co
    import torch
    from torch.utils.data import DataLoader, Dataset

    from labeler.confinement import elm_gate, unet, zerod

    assert (co.BEAM_ON_W, co.BEAM_GAP_MS, co.MIN_SEGMENT_MS) == (BEAM_ON_W, BEAM_GAP_MS, MIN_ON_MS)
    torch.backends.cudnn.allow_tf32 = False  # the networks were trained and scored in float32 on CPU
    torch.backends.cuda.matmul.allow_tf32 = False
    variant = co.VARIANTS[VARIANT]
    in_dir, work, res_dir = ROOT / "inputs", ROOT / "work", ROOT / "results"
    work.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    if args.list:
        shots = [s for s, _ in ae.read_list(Path(args.list))]
    else:
        shots = sorted(int(p.stem) for p in in_dir.glob("*.npz"))
    todo = [s for s in shots if (in_dir / f"{s}.npz").exists() and (args.redo or not (work / f"{s}.npz").exists())]
    tag = os.environ.get("SLURM_JOB_ID", f"local{os.getpid()}")
    print(f"infer {len(todo)} of {len(shots)} listed shots, job {tag}, {args.workers} workers at {ae.now()}",
          flush=True)
    if not todo:
        return
    device = torch.device(args.device)
    folds = co.fold_networks(device, VARIANT)  # 5 folds x 3 seeds
    if len(folds) != co.FOLDS or any(len(f) != variant.seeds for f in folds):
        raise SystemExit(f"expected {co.FOLDS} folds of {variant.seeds} networks in {variant.directory}")
    nets = [m for f in folds for m in f]
    test_of = co.recorded_test_folds(variant)  # curated shot -> the fold that held it out
    first_trained, last_trained = min(test_of), max(test_of)

    class Inputs(Dataset):
        def __len__(self):
            return len(todo)

        def __getitem__(self, i):
            shot = todo[i]
            try:
                with np.load(in_dir / f"{shot}.npz") as z:
                    grid = z["grid"]
                    meta = json.loads(str(z["meta"]))
                x = co.network_input(grid, shot, variant)
            except Exception as error:  # noqa: BLE001
                return {"shot": shot, "skip": f"input unreadable: {type(error).__name__}: {error}"}
            power = grid[zerod.CHANNELS.index("pinj")] * zerod.SCALES["pinj"]
            gate = elm_gate.shot_masks(Path(args.elm_root), shot, x.shape[1]) if args.elm_gate else (None, None, "off")
            return {"shot": shot, "x": x, "power": power, "meta": meta, "gate": gate}

    loader = DataLoader(Inputs(), batch_size=None, num_workers=args.workers, collate_fn=_as_is,
                        prefetch_factor=8 if args.workers else None)
    spans_csv, shots_csv = res_dir / f"{tag}.spans.csv", res_dir / f"{tag}.shots.csv"
    new = not shots_csv.exists()
    f_spans, f_shots = open(spans_csv, "a"), open(shots_csv, "a")
    if new:
        f_spans.write("shot,t_start_ms,t_end_ms,category,predicted,confidence,tier\n")
        f_shots.write("shot,status,window_start_ms,window_end_ms,h_ms,l_ms,qh_ms,wpqh_ms,uncertain_ms,n_spans,"
                      "n_ms,plasma_end_ms,beam_on_ms,source,extrapolated,da_source,fs_hz,lacks,beam_source,job,error,"
                      "elm_gate,gated_ms\n")

    def finish(item: dict, prob: np.ndarray) -> None:
        shot, meta, power = item["shot"], item["meta"], item["power"]
        end = float(meta["plasma_end_ms"])
        curated = shot in test_of
        extrapolated = shot < first_trained or shot > last_trained
        qh, wp, gate = item["gate"]
        common = (f"{item['x'].shape[1]},{end:g},{{on}},{'held_out' if curated else 'ensemble'},"
                  f"{int(extrapolated)},{meta['da_source']},{meta['fs_hz']},{'|'.join(meta['lacks'])},"
                  f"{meta.get('beam_source', 'beams')},{tag}")
        if not np.isfinite(power).any():
            f_shots.write(f"{shot},no_beam_record,,,,,,,,,{common.format(on=0)},,{gate},\n")
            return
        active = beam_on(power, end)
        on = int(active.sum())
        if on < co.MIN_SEGMENT_MS:
            f_shots.write(f"{shot},no_beam,,,,,,,,,{common.format(on=on)},,{gate},\n")
            return
        flip = elm_gate.flips(prob, qh, wp) if gate == "on" else np.zeros(prob.shape[1], bool)
        found = co.segment_shot(elm_gate.move(prob, flip), active)
        idx = np.flatnonzero(active)
        w0, w1 = int(idx[0]), int(idx[-1]) + 1
        rows = co.roster_rows(shot, found, curated=curated, last_curated=last_trained)
        codes = np.zeros(w1 - w0, np.uint8)
        for r in rows:
            codes[int(r["t_start"]) - w0:int(r["t_end"]) - w0] = r["category"]
            f_spans.write(f"{shot},{int(r['t_start'])},{int(r['t_end'])},{r['category']},{r['predicted']},"
                          f"{r['confidence']:.3f},{r['tier']}\n")
        ms = [int((codes == c).sum()) for c in (1, 2, 3, 4, 5)]
        status = "labeled" if rows else "no_segment"
        tmp = work / f".{shot}.{os.getpid()}.npz.tmp"
        with open(tmp, "wb") as fh:
            np.savez(fh, prob=prob.astype(np.float16), t_first_ms=0.0, hop_ms=1.0, window_ms=np.array([w0, w1]),
                     codes=codes, plasma_end_ms=end, source="held_out" if curated else "ensemble", job=tag,
                     elm_flip=flip, elm_gate=gate)
        os.replace(tmp, work / f"{shot}.npz")
        f_shots.write(f"{shot},{status},{w0},{w1},{','.join(map(str, ms))},{len(rows)},{common.format(on=on)},,{gate},"
                      f"{int((flip & active).sum())}\n")

    buckets: dict[int, list] = {}

    def run(items: list) -> None:
        length = items[0]["x"].shape[1] + (-items[0]["x"].shape[1]) % unet.STRIDE
        xb = np.zeros((len(items), items[0]["x"].shape[0], length), np.float32)
        for i, it in enumerate(items):
            xb[i, :, : it["x"].shape[1]] = it["x"]
        xt = torch.from_numpy(xb).to(device, non_blocking=True)
        probs = torch.stack([torch.softmax(m(xt), 1) for m in nets]).cpu().numpy()  # (nets, B, 4, length)
        for i, it in enumerate(items):
            n = it["x"].shape[1]
            k = test_of.get(it["shot"])
            sel = probs[k * variant.seeds:(k + 1) * variant.seeds, i] if k is not None else probs[:, i]
            finish(it, sel.mean(0)[:, :n])

    print("LOOP_START", stamp(), flush=True)
    t_loop, ran, warm = time.perf_counter(), 0, False
    with torch.inference_mode():
        for item in loader:
            if "skip" in item:
                f_shots.write(f"{item['shot']},infer_failed,,,,,,,,,,,,,,,,,,{tag},\"{ae.clean(item['skip'])}\",,\n")
                continue
            n = item["x"].shape[1]
            key = n + (-n) % unet.STRIDE  # equal padded length: each shot sees exactly predict_shot's padding
            buckets.setdefault(key, []).append(item)
            if len(buckets[key]) >= args.batch:
                run(buckets.pop(key))
                ran += args.batch
                if not warm:
                    print("LOOP_WARM", stamp(), flush=True)
                    warm = True
                f_spans.flush()
                f_shots.flush()
        for key in list(buckets):
            items = buckets.pop(key)
            run(items)
            ran += len(items)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("LOOP_END", stamp(), flush=True)
    f_spans.close()
    f_shots.close()
    wall = time.perf_counter() - t_loop
    print(f"{ran} shots in {wall:.1f} s ({wall / max(ran, 1):.4f} s a shot), {len(nets)} networks", flush=True)


# ------------------------------------------------------------------ publish (login node)


def statuses():
    """(lab, spans, status_of) for the run under ROOT, as ``ae.shot_statuses`` with this pipeline's
    columns and its final infer findings."""
    import pandas as pd

    stage = ae.last_status("stage")
    extract = ae.last_status("extract")
    shots, spans = ae.read_results()
    shots["done"] = shots.status.isin({"labeled", *INFER_FINDINGS})
    shots = shots.sort_values(["shot", "done"]).drop_duplicates("shot", keep="last")
    lab = shots[shots.status.eq("labeled")].set_index("shot")
    found = dict(zip(shots.shot[shots.done & ~shots.status.eq("labeled")].astype(int),
                     shots.status[shots.done & ~shots.status.eq("labeled")]))
    failed = dict(zip(shots.shot[~shots.done].astype(int), shots.error[~shots.done].astype(str)))
    spans = spans.merge(lab[["job"]].reset_index(), on=["shot", "job"], how="inner")
    spans = spans.drop(columns="job").sort_values(["shot", "t_start_ms"])
    for c in ("shot", "t_start_ms", "t_end_ms"):
        spans[c] = spans[c].astype("int32")
    for c in ("category", "predicted"):
        spans[c] = spans[c].astype("uint8")
    spans["confidence"] = spans.confidence.astype("float32")
    labeled = set(lab.index.astype(int))

    def status_of(shot: int) -> tuple[str, str]:
        if shot in labeled:
            return "labeled", ""
        if shot in found:
            return found[shot], ""
        if shot in failed:
            return "infer_failed", failed[shot]
        e = extract.get(shot)
        if e and e["status"] != "ok":
            return e["status"], str(e.get("error", e.get("fs_hz", "")))
        if e and e["status"] == "ok":
            return "pending", "extracted"
        s = stage.get(shot)
        if s is None:
            return "pending", ""
        if s["status"] == "staged":
            return "pending", "staged"
        return s["status"], str(s.get("error", s.get("beam_on_ms", "")))[:200]

    return lab, spans, status_of, pd


GRID_DIR = "grid_1ms"


def write_grid(out: Path, lab, spans) -> int:
    """grid_1ms/<shot // 1000>/<shot>.bin: the window's 1 ms codes (0 none, 1 H, 2 L, 3 QH, 4 WPQH,
    5 uncertain), two to a byte, the first millisecond in the high four bits; index.csv gives each
    shot's start_ms and end_ms (end_ms - start_ms codes) and the infer job. Only new or relabeled
    shots are written."""
    import pandas as pd

    gdir = out / GRID_DIR
    gdir.mkdir(parents=True, exist_ok=True)
    index = gdir / "index.csv"
    done = {}
    if index.exists():
        old = pd.read_csv(index, dtype={"shot": int, "start_ms": int, "end_ms": int, "job": str})
        done = {int(r.shot): (int(r.start_ms), int(r.end_ms), r.job) for r in old.itertuples(index=False)}
    by_shot = {int(k): g[["t_start_ms", "t_end_ms", "category"]].to_numpy() for k, g in spans.groupby("shot")}
    n = 0
    for shot, row in lab.iterrows():
        shot, job = int(shot), str(row.job)
        f = gdir / str(shot // 1000) / f"{shot}.bin"
        w0, w1 = int(row.window_start_ms), int(row.window_end_ms)
        if done.get(shot) == (w0, w1, job) and f.exists():
            continue
        codes = np.zeros(max(w1 - w0, 0) + (max(w1 - w0, 0) % 2), np.uint8)
        for t0, t1, cat in by_shot.get(shot, ()):
            codes[max(int(t0) - w0, 0):max(int(t1) - w0, 0)] = cat
        f.parent.mkdir(exist_ok=True)
        tmp = f.with_name(f".{f.name}.tmp")
        tmp.write_bytes(((codes[0::2] << 4) | codes[1::2]).astype(np.uint8).tobytes())
        os.replace(tmp, f)
        done[shot] = (w0, w1, job)
        n += 1
    tmp = gdir / ".index.csv.tmp"
    pd.DataFrame([(k, *v) for k, v in sorted(done.items())], columns=["shot", "start_ms", "end_ms", "job"]).to_csv(
        tmp, index=False)
    os.replace(tmp, index)
    return n


def _readme(meta: dict) -> str:
    c = meta["counts"]
    h = {k: v / 3.6e6 for k, v in meta["ms_by_category"].items()}
    return f"""# Confinement mode, 1 ms grid, every FDP plasma shot

Model output of `confine-ours` (variant `{VARIANT}`: a 1D U-Net over D-alpha, line density, beta_N,
W_MHD and beam power; five run-day folds of three seeds), on beam-on time only. Unreviewed.
Built {meta['published']} by `{meta['made_by']}` (git {meta['git_sha']}).
Out of sample on the 416 curated shots (run-day folds): per-ms macro F1 0.910 (L 0.924, H 0.968,
QH 0.914, WPQH 0.836), segment F1@25 0.832 (docs/labeler/confinement_ours.md).

Candidates {sum(c.values())}; labeled {c.get('labeled', 0)}; spans {meta['n_spans']}.
Hours: H {h['1']:.0f}, L {h['2']:.0f}, QH {h['3']:.0f}, WPQH {h['4']:.0f}, uncertain {h['5']:.0f}.
Status counts and caveats: `meta.json`.

QH and WPQH calls inside ELM trains (elm-inst2 onsets; `src/labeler/confinement/elm_gate.py`) are
turned into H before segmenting: `shots.parquet` `elm_gate` (on, no_elm, ...) and `gated_ms`.

Categories: 1 H, 2 L, 3 QH, 4 WPQH, 5 uncertain (the segment's mean class probability is under
0.7; `predicted` keeps its class). 0 in the grid: not in a segment.

```python
import numpy as np, pandas as pd
spans = pd.read_parquet("spans.parquet")   # shot, t_start_ms, t_end_ms, category, predicted, confidence, tier
shots = pd.read_parquet("shots.parquet").set_index("shot")
def confinement_1ms(shot):
    \"\"\"(t_ms, code) on the assessed window: 0 none, 1 H, 2 L, 3 QH, 4 WPQH, 5 uncertain.\"\"\"
    w0, w1 = shots.loc[shot, ["window_start_ms", "window_end_ms"]].astype(int)
    code = np.zeros(w1 - w0, dtype=np.uint8)
    for a, b, c in spans.loc[spans.shot == shot, ["t_start_ms", "t_end_ms", "category"]].itertuples(index=False):
        code[a - w0:b - w0] = c
    return np.arange(w0, w1), code
```

The same codes dense, one file a shot, two milliseconds a byte (the first in the high four bits);
start and end are in `{GRID_DIR}/index.csv`:

```python
grid = pd.read_csv("{GRID_DIR}/index.csv").set_index("shot")             # shot, start_ms, end_ms, job
n = grid.end_ms[shot] - grid.start_ms[shot]
b = np.fromfile(f"{GRID_DIR}/{{shot // 1000}}/{{shot}}.bin", np.uint8)         # e.g. 185/185786.bin
code = np.stack([b >> 4, b & 15], axis=1).ravel()[:n]
t_ms = grid.start_ms[shot] + np.arange(n)                            # frame [t, t + 1) ms
```

`valid.csv` lists every shot number of the d3drdb snapshot's range with `valid` 1 when the shot has
confinement labels and 0 otherwise (`shots.parquet` `status` says why for the plasma candidates).
Frame `k` is `[k, k + 1)` ms. Outside `[window_start_ms, window_end_ms)` nothing was assessed.
"""


def cmd_publish(args) -> None:
    import confinement_ours as co

    from labeler.confinement import elm_gate

    lab, spans, status_of, pd = statuses()
    cands = pd.read_csv(args.candidates)
    extract = ae.last_status("extract")
    rows = []
    for c in cands.itertuples(index=False):
        st, note = status_of(int(c.shot))
        r = {"shot": int(c.shot), "status": st, "plasma_end_ms": int(c.end_ms), "note": note}
        if st == "labeled":
            L = lab.loc[int(c.shot)]
            r.update(window_start_ms=int(L.window_start_ms), window_end_ms=int(L.window_end_ms),
                     h_ms=int(L.h_ms), l_ms=int(L.l_ms), qh_ms=int(L.qh_ms), wpqh_ms=int(L.wpqh_ms),
                     uncertain_ms=int(L.uncertain_ms), n_spans=int(L.n_spans), beam_on_ms=int(L.beam_on_ms),
                     source=str(L.source), extrapolated=bool(L.extrapolated), da_source=str(L.da_source),
                     fs_hz=float(L.fs_hz), lacks="" if pd.isna(L.lacks) else str(L.lacks),
                     beam_source=str(L.beam_source),
                     elm_gate=str(L.get("elm_gate")) if pd.notna(L.get("elm_gate")) else "",
                     gated_ms=int(L.gated_ms) if pd.notna(L.get("gated_ms")) else pd.NA)
        elif int(c.shot) in extract and extract[int(c.shot)].get("status") == "ok":
            e = extract[int(c.shot)]
            r.update(da_source=e.get("da_source"), fs_hz=e.get("fs_hz"), lacks="|".join(e.get("lacks", [])),
                     beam_source=e.get("beam_source"))
        rows.append(r)
    table = pd.DataFrame(rows)
    for col in ("window_start_ms", "window_end_ms", "h_ms", "l_ms", "qh_ms", "wpqh_ms", "uncertain_ms",
                "n_spans", "beam_on_ms", "gated_ms"):
        if col not in table:
            table[col] = pd.NA
        table[col] = table[col].astype("Int32")
    table["shot"] = table.shot.astype("int32")
    counts = table.status.value_counts().to_dict()
    first, last = ae.snapshot_range(cands)
    valid = pd.DataFrame({"shot": np.arange(first, last + 1, dtype=np.int32)})
    valid["valid"] = valid.shot.isin(set(lab.index.astype(int))).astype(np.uint8)
    variant = co.VARIANTS[VARIANT]
    test_of = co.recorded_test_folds(variant)
    gates = {str(k): int(v) for k, v in table.elm_gate.value_counts().items()} if "elm_gate" in table else {}
    ms_by = {str(k): int((spans.t_end_ms - spans.t_start_ms)[spans.category == k].sum()) for k in CATEGORIES}
    meta = {
        "event": "confinement",
        "published": ae.now(),
        "grid": "1 ms frames [k, k + 1) ms of the shot clock; spans [t_start_ms, t_end_ms)",
        "categories": {"0": "grid only: inside the window but in no segment", "1": "H (high)", "2": "L (low)",
                       "3": "QH (quiescent H)", "4": "WPQH (wide-pedestal QH)",
                       "5": "uncertain: a segment whose mean class probability is under 0.7 (predicted keeps its class)"},
        "model": {"name": "confine-ours", "variant": VARIANT, "networks": str(variant.directory),
                  "folds": co.FOLDS, "seeds_per_fold": variant.seeds,
                  "curated_shots": len(test_of), "trained_shot_range": [min(test_of), max(test_of)],
                  "rule": ("mean softmax of the fifteen networks (a curated shot: the three of the fold that held "
                           "it out); beam-on mask; ELM gate (elm_gate); mode filter of 21 ms; segments of 20 ms or "
                           "more (a shorter run joins the segment before it); category 5 under a mean class "
                           "probability of 0.7"),
                  "scores": "docs/labeler/confinement_ours.md (scores_base3.json)"},
        "elm_gate": {"rule": elm_gate.RULE.describe(), "params": elm_gate.RULE.as_dict(),
                     "onsets": (f"{ELM_ROOT.name} work/<shot>.npz onsets_ms (decoding {elm_gate.DECODING}); "
                                "filterscopes from its inputs/<shot>.npz fmax"),
                     "code": "src/labeler/confinement/elm_gate.py", "counts": gates,
                     "statuses": {"on": "gated (shots.parquet gated_ms: beam-on ms flipped to H)",
                                  "no_elm": "no ELM labels for the shot: the networks' calls as they are",
                                  "elm_decoding": "the shot's ELM labels predate elm-inst2-sized: not gated",
                                  "elm_unreadable": "the ELM files did not read: not gated",
                                  "off": "run with --no-elm-gate",
                                  "": "labelled before the gate existed"}},
        "inputs": {"d_alpha": "the first live of " + ", ".join(t.rsplit("::", 1)[1] for t in FS_TAGS)
                              + " (over half its samples finite, not constant), at its native rate",
                   "density": DENSITY_TAG, "efit": list(EFIT_TAGS.values()), "beams": NB_TAGS,
                   "beams_fallback": (f"{NB_TOTAL_TAG} x {NB_TOTAL_W:g} (kW to W) when no per-beam node has a "
                                      "record (shots before ~148000; shots.parquet beam_source nb_total)"),
                   "source": "FDP MDSplus trees by byte ranges (EFIT01 from FDP's code archive)",
                   "grid": "labeler.confinement.zerod: 1 ms bins of D-alpha (mean, max, std over its 95th "
                           "percentile), density / 1e14, beta_N / 1.5, W_MHD / 5e5 J, beam power / 3e6 W"},
        "assessed": (f"beam-on time: total beam power >= {BEAM_ON_W:.0f} W in 1 ms means, gaps of at most "
                     f"{BEAM_GAP_MS} ms bridged, before the plasma end (d3drdb pulse_length); window = first to "
                     "last such millisecond"),
        "counts": counts, "n_spans": len(spans), "ms_by_category": ms_by,
        "made_by": "scripts/labeler/conf_fdp_label.py", "git_sha": ae_git_sha(),
        "statuses": {
            "labeled": "run; its window and spans are in the tables",
            "pending": "not run yet",
            "no_view": "FDP has no per-shot view for it",
            "ptdata_record": "the filterscopes (or the beams) are PTDATA calls (shots before ~131000): not run",
            "no_fs": "no filterscope record, or none of FS01-FS08 live",
            "no_beam_record": "no beam power record",
            "no_beam": (f"the beams' total power is under {BEAM_ON_W:.0f} W for all but {MIN_ON_MS} ms, or the "
                        "NB nodes were never written (no beam fired)"),
            "no_segment": "beam-on time exists but no run of one class reaches 20 ms",
            "fs_rate_low": f"D-alpha sampled slower than every {MAX_FS_DT_MS} ms",
            "no_record_in_window": "no D-alpha samples in [0, plasma end]",
            "failed": "the download failed after retries (retried on the next run)",
            "other": "read_failed, fs_unreadable, bad_record, extract_error, infer_failed: the note says why",
        },
        "caveats": [
            "Model output, unreviewed: not ground truth.",
            ("Beam-on time only: L-mode before the beams or in a beam-off phase is not labelled; a shot "
             "without beams has no labels."),
            (f"Trained on curated shots {min(test_of)}-{max(test_of)}; shots outside that range are "
             "flagged extrapolated (shots.parquet): untested there."),
            ("QH and WPQH segments on shots outside the curated set are on the unreviewed tier "
             "(spans.parquet tier): the curated set may not contain ELM-free and other quiescent H-modes "
             "the network could read as QH or WPQH."),
            ("The density channel (DENV2F) is uncalibrated, and the BCI tree is absent before ~141000: those "
             "shots run with the channel masked (lacks), which training rarely saw."),
            ("D-alpha is read at its native rate (20-50 kHz); training mixed native records (270 shots) "
             "with the corpus' 10 kHz resampling (146)."),
            "Curated shots are labelled by the fold that held them out, every other shot by all fifteen networks.",
            ("The ELM gate turns QH and WPQH calls into H inside trains of elm-inst2 onsets (meta elm_gate). "
             "It is only as good as those onsets: it can fire on a small periodic D-alpha modulation in a "
             "quiescent phase (onset sizes 1.4-1.6, e.g. 163522), it turns QH calls into H where breakthrough "
             "ELMs come at ~20 Hz, and WPQH into H during a long burst of them (e.g. 175523), more so where "
             "the networks call a WPQH phase QH (in the wpqh_elm_hiro WPQH windows 5% of QH calls flip, "
             "0.7% of WPQH calls); isolated "
             "breakthrough ELMs do not trigger it. Shots without ELM labels are not gated (shots.parquet elm_gate)."),
            "The window end is the d3drdb pulse_length; shots after the snapshot's last shot are not candidates.",
        ],
        "files": {
            "valid.csv": f"shot, valid: every shot number {first}-{last}, 1 when the shot is labeled",
            "shots.parquet": "one row per plasma candidate: status, window, ms per category, source, inputs",
            "spans.parquet": "shot, t_start_ms, t_end_ms, category, predicted, confidence, tier",
            GRID_DIR + "/": "<shot // 1000>/<shot>.bin: the window's codes, 4 bits a ms, first in the high bits; "
                            "index.csv gives start_ms, end_ms",
        },
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, write in (("valid.csv", lambda p: valid.to_csv(p, index=False)),
                        ("spans.parquet", lambda p: spans.to_parquet(p, index=False)),
                        ("shots.parquet", lambda p: table.to_parquet(p, index=False)),
                        ("meta.json", lambda p: Path(p).write_text(json.dumps(meta, indent=1))),
                        ("README.md", lambda p: Path(p).write_text(_readme(meta)))):
        tmp = out / f".{name}.tmp"
        write(tmp)
        os.replace(tmp, out / name)
    n_grid = write_grid(out, lab, spans)
    print(f"published to {out}: {counts}; {len(spans)} spans; {n_grid} grid files written", flush=True)


def ae_git_sha() -> str:
    import subprocess

    r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                       check=False)
    return r.stdout.strip() or "unknown"


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
    s.add_argument("--budget-gb", type=float, default=200)
    s.add_argument("--submit", action="store_true", help="submit the extract job of each batch")
    s.add_argument("--whole", action="store_true",
                   help="restage the shots whose sparse read failed, with whole datafiles")
    e = sub.add_parser("extract")
    e.add_argument("--list", required=True)
    e.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    e.add_argument("--keep-inputs", action="store_true", help="keep the staged tree files")
    e.add_argument("--redo", nargs="*", default=[],
                   help="also extract listed shots whose last status is one of these (after a screen changed)")
    i = sub.add_parser("infer")
    i.add_argument("--list", default=None)
    i.add_argument("--workers", type=int, default=8)
    i.add_argument("--batch", type=int, default=64, help="shots of one padded length a forward pass")
    i.add_argument("--device", default="cuda")
    i.add_argument("--redo", action="store_true", help="also shots that already have work/<shot>.npz")
    i.add_argument("--elm-root", default=str(ELM_ROOT), help="the ELM pipeline's run (default %(default)s)")
    i.add_argument("--no-elm-gate", dest="elm_gate", action="store_false",
                   help="label from the networks alone (no QH/WPQH -> H in ELM trains)")
    u = sub.add_parser("publish")
    u.add_argument("--candidates", default=str(CANDIDATES))
    u.add_argument("--out", default=None, help=ae.OUT_HELP)
    args = p.parse_args(argv)
    _set_root(args.root)
    if args.cmd == "publish":
        args.out = ae.publish_out(args.out, ROOT, ALL_SHOT_ROOT, OUT)
    {"stage": cmd_stage, "extract": cmd_extract, "infer": cmd_infer, "publish": cmd_publish}[args.cmd](args)


if __name__ == "__main__":
    main()
