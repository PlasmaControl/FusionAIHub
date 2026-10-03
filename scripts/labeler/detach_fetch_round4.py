#!/usr/bin/env python
"""Fetch the round-four detachment inputs: beam power and divertor Thomson Te.

Run on the login node under the fdp wrapper, one worker, `timeout` around it:

    timeout 3600 pixi run --frozen -e labelmaker fdp run python \\
        scripts/labeler/detach_fetch_round4.py --stage all

* `bms`: the total injected neutral-beam power, PTDATA `BMSPINJ`, for every cached
  shot whose corpus `pinj` group is a stub or absent (the `\\NB::PINJ*` and per-beam
  `\\D3D::...PINJ_<beam>` nodes hold no data on those shots). The record is a total
  in megawatts stored as volts (1 V = 1 MW); it is parked 1 ms block-mean in the
  shot's cache file as `pinj_bms`. The calibration check fetches the same node for
  shots whose corpus beams are real and records the ratio in
  `$LABELER_ROOT/round4/detach/nbi_calibration.json`.
* `dts`: processed divertor Thomson Te, `\\ELECTRONS::TSTE_DIV` with its error, the
  density and the chord positions `TSR_DIV`/`TSZ_DIV` (16 chords at fixed R, Z),
  parked per shot in `$LABELER_ROOT/round4/detach/dts/<shot>.npz`. This is the
  independent, localised temperature check of the label.

An authentication or login error sets the shared stop flag and ends the run with
status 3: the owner's login has lapsed and nothing more may be fetched.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from detach_fetch import block_mean, corpus_is_stub, is_auth_error
from detach_json import dumps

from labeler.features import resolve_fdp

REPO = Path(__file__).resolve().parents[2]
NBI_RECORD = REPO / "docs/labeler/results/detachment_nbi_calibration.json"
ROOT = Path(os.environ["LABELER_ROOT"]) / "round4/detach"
STOP = ROOT / "fetch_auth_stop"
LOG = ROOT / "fetch_round4_log.jsonl"
#: Shots whose corpus beams are real, to calibrate BMSPINJ against their summed beams.
CALIBRATION_SHOTS = (189057, 189093, 190109, 195952, 199166, 199172, 199351, 200977)
DTS_NODES = {
    "te": "TSTE_DIV",
    "te_err": "TSTE_E_DIV",
    "ne": "TSNE_DIV",
    "r": "TSR_DIV",
    "z": "TSZ_DIV",
    "t_ms": "TSTIME_DIV",
}
BMS_BLOCK_MS = 1.0


def fetch(kind: str, name: str, shot: int, tree: str = "ELECTRONS"):
    if STOP.exists():
        raise RuntimeError("authentication stop flag")
    try:
        if kind == "ptdata":
            return resolve_fdp._fetch_ptdata(name, shot)
        return resolve_fdp._fetch_mds(name, tree, shot)
    except Exception as error:
        if is_auth_error(str(error)):
            STOP.touch()
        raise


def error_text(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"[:200]


def fetch_bms(shot: int) -> tuple[np.ndarray, np.ndarray]:
    rec = fetch("ptdata", "BMSPINJ", shot)
    t = np.asarray(rec["times"], dtype=float)
    y = np.asarray(rec["data"], dtype=float)
    if len(t) < 2 or len(t) != len(y):
        raise ValueError("not a usable time series")
    t, y = block_mean(t, y, BMS_BLOCK_MS)
    return t, y.astype(np.float32)


def stage_bms(shots, pace: float) -> list[dict]:
    rows = []
    calibration = {}
    cal_path = ROOT / "nbi_calibration.json"
    if cal_path.exists():
        calibration = json.loads(cal_path.read_text())
    for shot in shots:
        path = ROOT / "cache" / f"{shot}.npz"
        row = {"shot": shot, "stage": "bms"}
        if not path.is_file():
            rows.append({**row, "status": "no_cache"})
            continue
        with np.load(path) as f:
            arrays = {k: f[k] for k in f.files if k != "status"}
            status = json.loads(str(f["status"]))
        if "pinj_bms__y" in arrays:
            rows.append({**row, "status": "cached"})
            continue
        try:
            t, y = fetch_bms(shot)
            arrays["pinj_bms__t"], arrays["pinj_bms__y"] = t, y
            status["pinj_bms"] = f"ok {len(y)} MW (BMSPINJ, volts at 1 V = 1 MW)"
            tmp = path.with_name(f".{path.name}.bms.npz")
            np.savez(tmp, status=json.dumps(status), **arrays)
            tmp.replace(path)
            row["status"] = "ok"
            row["max_mw"] = float(np.nanmax(y))
        except Exception as error:  # noqa: BLE001  an absent node is data
            row["status"] = "error " + error_text(error)
            row["auth"] = STOP.exists() or is_auth_error(row["status"])
        rows.append(row)
        record_row(row)
        if row.get("auth"):
            return rows
        time.sleep(pace)
    for shot in CALIBRATION_SHOTS:
        if str(shot) in calibration:
            continue
        row = {"shot": shot, "stage": "bms_calibration"}
        try:
            if corpus_is_stub(shot, "pinj"):
                raise ValueError("corpus beams are not real on this shot")
            import h5py

            t, y = fetch_bms(shot)
            with h5py.File(
                Path("/scratch/gpfs/EKOLEMEN/foundation_model")
                / f"{shot}_processed.h5",
                "r",
            ) as f:
                tc = np.asarray(f["pinj"]["xdata"][:], float) * 1000.0
                yc = np.asarray(f["pinj"]["ydata"][:], float).sum(axis=0)
            ratios = []
            for centre in np.arange(1000.0, 5001.0, 250.0):
                a = np.nanmean(y[(t > centre - 50) & (t < centre + 50)])
                b = np.nanmean(yc[(tc > centre - 50) & (tc < centre + 50)])
                if np.isfinite(a) and a > 0.5:
                    ratios.append(float(b / a / 1e6))
            calibration[str(shot)] = {
                "n_windows": len(ratios),
                "corpus_over_bms_median": float(np.median(ratios)) if ratios else None,
                "corpus_over_bms_range": [min(ratios), max(ratios)] if ratios else None,
                "bms_max_mw": float(np.nanmax(y)),
                "corpus_max_mw": float(np.nanmax(yc) / 1e6),
            }
            row["status"] = "ok"
        except Exception as error:  # noqa: BLE001
            row["status"] = "error " + error_text(error)
            row["auth"] = STOP.exists() or is_auth_error(row["status"])
        rows.append(row)
        record_row(row)
        if row.get("auth"):
            break
        time.sleep(pace)
    cal_path.write_text(json.dumps(calibration, indent=1))
    return rows


def record_row(row: dict) -> None:
    with LOG.open("a") as log:
        log.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def stage_dts(shots, pace: float) -> list[dict]:
    out_dir = ROOT / "dts"
    out_dir.mkdir(exist_ok=True)
    rows = []
    for shot in shots:
        path = out_dir / f"{shot}.npz"
        row = {"shot": shot, "stage": "dts"}
        if path.is_file():
            rows.append({**row, "status": "cached"})
            continue
        arrays, status = {}, {}
        for key, node in DTS_NODES.items():
            try:
                rec = fetch("mds", rf"\ELECTRONS::{node}", shot)
                arrays[key] = np.asarray(rec["data"], dtype=np.float32)
                status[key] = f"ok {np.asarray(rec['data']).shape}"
            except Exception as error:  # noqa: BLE001  an absent node is data
                status[key] = "error " + error_text(error)
                if STOP.exists() or is_auth_error(status[key]):
                    row["auth"] = True
                    break
            time.sleep(pace)
        row["status"] = status
        if not row.get("auth") and "te" in arrays and "t_ms" in arrays:
            tmp = path.with_name(f".{path.name}.tmp.npz")
            np.savez_compressed(tmp, status=json.dumps(status), **arrays)
            tmp.replace(path)
        elif not row.get("auth"):
            # park the absence so a rerun does not refetch a shot with no data
            tmp = path.with_name(f".{path.name}.tmp.npz")
            np.savez_compressed(tmp, status=json.dumps(status))
            tmp.replace(path)
        rows.append(row)
        record_row(row)
        if row.get("auth"):
            return rows
    return rows


def write_nbi_record() -> None:
    """Commit the BMSPINJ calibration and the shots it supplied (small, reviewable)."""
    calibration = json.loads((ROOT / "nbi_calibration.json").read_text())
    supplied = {}
    for line in LOG.read_text().splitlines():
        row = json.loads(line)
        if row.get("stage") == "bms" and str(row.get("status", "")).startswith("ok"):
            supplied[str(row["shot"])] = {"max_mw": row.get("max_mw")}
    ratios = [
        v["corpus_over_bms_median"]
        for v in calibration.values()
        if v.get("corpus_over_bms_median")
    ]
    record = {
        "node": "PTDATA BMSPINJ",
        "unit": "volts, 1 V = 1 MW total injected neutral-beam power",
        "why": (
            "The corpus `pinj` group is a stub on these shots and the "
            "`\\NB::PINJ*` and per-beam `\\D3D::...PINJ_<beam>` nodes hold no data."
        ),
        "calibration_against_corpus_beams": calibration,
        "corpus_over_bms_median_range": [min(ratios), max(ratios)] if ratios else None,
        "n_shots_supplied": len(supplied),
        "shots_supplied": supplied,
    }
    NBI_RECORD.write_text(dumps(record, indent=1) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stage", choices=("bms", "dts", "all"), default="all")
    parser.add_argument("--pace", type=float, default=1.0)
    parser.add_argument(
        "--dts-shots-file", type=Path, default=ROOT / "shots_inversion.txt"
    )
    args = parser.parse_args()
    if args.pace < 1:
        raise SystemExit("pace must be at least one second")
    if STOP.exists():
        raise SystemExit("AUTH ERROR: shared stop flag exists; fetching stopped")
    rows = []
    if args.stage in ("bms", "all"):
        stubs = sorted(
            int(p.stem)
            for p in (ROOT / "cache").glob("*.npz")
            if corpus_is_stub(int(p.stem), "pinj")
        )
        rows += stage_bms(stubs, args.pace) or []
    if args.stage in ("dts", "all") and not any(r.get("auth") for r in rows):
        shots = sorted(set(map(int, args.dts_shots_file.read_text().split())))
        rows += stage_dts(shots, args.pace) or []
    if any(r.get("auth") for r in rows):
        print("AUTH ERROR: stopping all fetching", flush=True)
        return 3
    if args.stage in ("bms", "all") and (ROOT / "nbi_calibration.json").exists():
        write_nbi_record()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
