#!/usr/bin/env python
"""Refresh only radiation votes on all cached grids, reusing per-shot inputs.

This runs after the complete extraction when only the Prad validity rule changes.
It preserves all other indicator/geometry arrays and records changed-validity
counts in a JSONL log. Future full extraction uses exactly the same library gate.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from labeler.events.detachment import core, prad, signals

ROOT = Path(os.environ["LABELER_ROOT"]) / "round4/detach"


def refresh(shot):
    cache = signals.load_cache(shot)
    power = signals.heating_power(shot, cache)
    p_t, p_in = (None, None) if power is None else power[:2]
    elm_t, flags = signals.elm_mask(shot, cache) or (None, None)
    rt, ry = cache.get("prad_divl", (None, None))
    changes = {}
    for directory in ("bins", "bins_w20", "bins_w50", "bins_w100"):
        path = ROOT / directory / f"{shot}.npz"
        if not path.exists():
            continue
        with np.load(path) as f:
            out = {k: f[k] for k in f.files}
        starts = out["start_ms"]
        width = core.BIN_MS if directory == "bins" else float(directory[6:])
        edges = np.r_[starts, starts[-1] + width]
        indicator = prad.prad_indicator(edges, rt, ry, p_t, p_in, elm_t, flags)
        changes[directory] = {
            "changed_validity_bins": int((out["prad_valid"] != indicator.valid).sum()),
            "negative_radiation_bins": int(
                (indicator.reason == "negative_radiation").sum()
            ),
        }
        out["prad_value"] = indicator.value.astype(np.float32)
        out["prad_valid"] = indicator.valid
        out["prad_reason"] = indicator.reason.astype(str)
        out["prad_vote"] = indicator.vote
        out["aux_prad_elm_window_known"] = prad.elm_window_known(edges, elm_t, flags)
        out["aux_prad_divl_native_w"] = np.full(len(starts), np.nan, np.float32)
        if rt is not None:
            out["aux_prad_divl_native_w"] = core.bin_mean(
                rt, ry, edges, keep=~core.elm_at(rt, elm_t, flags)
            )[0].astype(np.float32)
        pending = path.with_name(f".{path.name}.pending.npz")
        np.savez_compressed(pending, **out)
        pending.replace(path)
    return {"shot": shot, "status": "ok", "changes": changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    shots = sorted(int(p.stem) for p in (ROOT / "bins").glob("*.npz"))
    log = ROOT / "logs/round3_radiation_refresh.jsonl"
    with mp.Pool(args.workers) as pool, log.open("w") as handle:
        for i, result in enumerate(pool.imap_unordered(refresh, shots), 1):
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            print(i, len(shots), result, flush=True)


if __name__ == "__main__":
    main()
