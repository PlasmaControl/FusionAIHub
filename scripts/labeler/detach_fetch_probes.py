#!/usr/bin/env python
"""Processed lower-divertor Langmuir profiles, including per-shot R/Z positions."""

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ["LABELER_ROOT"]) / "round4/detach"
AUTH_STOP = ROOT / "fetch_auth_stop"


def fetch(shot):
    import MDSplus

    dest = ROOT / "processed_probes" / f"{shot}.npz"
    if dest.exists():
        return {"shot": shot, "status": "cached"}
    arrays, status = {}, {}
    try:
        tree = MDSplus.Tree("langmuir", shot, "readonly")
        # Probe 1..20 are the two lower outer shelves; read their positions in
        # each shot, never reuse a different campaign's coordinates.
        for i in range(1, 21):
            if AUTH_STOP.exists():
                return {"shot": shot, "auth": True}
            node = rf"\LANGMUIR::TOP.PROBE_{i:03d}"
            try:
                r, z = [float(tree.getNode(node + ":" + k).data()) for k in ("R", "Z")]
                if not (0.8 < r < 2 and -1.5 < z < -1.15):
                    status[str(i)] = "missing_position"
                    continue
                j = tree.getNode(node + ":JSAT")
                y = np.asarray(j.data(), float)
                t = np.asarray(tree.getNode(node + ":TIME").data(), float)
                if len(t) != len(y) or len(t) < 2:
                    status[str(i)] = "no_samples"
                    continue
                arrays[f"p{i}_jsat"] = y
                arrays[f"p{i}_t_ms"] = t
                arrays[f"p{i}_rz"] = [r, z]
                status[str(i)] = f"ok {len(t)} {j.units}"
            except Exception as error:  # noqa: BLE001  diagnostic absence is recorded
                msg = f"{type(error).__name__}: {error}"[:200]
                status[str(i)] = msg
                if any(
                    k in msg.lower()
                    for k in (
                        "auth",
                        "login",
                        "token",
                        "credential",
                        "401",
                        "403",
                        "permission",
                    )
                ):
                    AUTH_STOP.touch()
                    return {"shot": shot, "auth": True, "status": status}
        np.savez_compressed(dest, status=json.dumps(status), **arrays)
        row = {"shot": shot, "n_probes": len(arrays) // 3, "status": status}
    except Exception as error:  # noqa: BLE001  diagnostic absence is recorded
        msg = f"{type(error).__name__}: {error}"[:200]
        row = {"shot": shot, "error": msg}
        if any(
            k in msg.lower()
            for k in (
                "auth",
                "login",
                "token",
                "credential",
                "401",
                "403",
                "permission",
            )
        ):
            AUTH_STOP.touch()
            row["auth"] = True
    time.sleep(1)
    return row


def main():
    (ROOT / "processed_probes").mkdir(exist_ok=True)
    shots = list(map(int, (ROOT / "shots_fetch.txt").read_text().split()))
    with mp.Pool(2) as pool, (ROOT / "processed_probes_log.jsonl").open("a") as log:
        for row in pool.imap_unordered(fetch, shots):
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(
                row["shot"], row.get("n_probes", row.get("error", "cached")), flush=True
            )
            if row.get("auth"):
                pool.terminate()
                raise SystemExit(3)


if __name__ == "__main__":
    main()
