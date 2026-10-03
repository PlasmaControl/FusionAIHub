#!/usr/bin/env python
"""Survey which corpus shots hold real detachment-relevant records.

A corpus group is a stub when its time axis has one sample (the absent-signal
sentinel); group presence alone says nothing. For every `<shot>_processed.h5`
this records, for the groups the detachment indicators and the review video
panel could use, the length of the time axis and (for the movies and the
bolometer) whether the data are live: finite, non-constant. One CSV row per
shot, written to `$LABELER_ROOT/round4/detach/survey/corpus_survey.csv`.

    python scripts/labeler/detach_survey.py [--workers 8] [--limit N]
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

CORPUS = Path("/scratch/gpfs/EKOLEMEN/foundation_model")
#: tangtv channel order (scripts/data_preparation/scan_video_channels.py).
TANGTV_CHANNELS = (
    "ch0_lodiv_par_int",
    "ch1_lodiv_par_std",
    "ch2_lodiv_perp",
    "ch3_updiv225_perp",
    "ch4_updiv0_perp",
    "ch5_updiv225_par",
    "ch6_updiv0_par",
)
LENGTH_GROUPS = (
    "bolo",
    "irtv",
    "langmuir",
    "filterscopes",
    "gas_flow",
    "co2",
    "pinj",
    "ts_tangential_temp",
    "ts_tangential_density",
    "ts_core_density",
    "neutron_rate",
)


def _live(values: np.ndarray) -> bool:
    """Finite somewhere and not constant."""
    finite = values[np.isfinite(values)]
    return bool(finite.size > 0 and float(finite.max()) > float(finite.min()))


def survey_one(path: Path) -> dict:
    """One shot's row."""
    row: dict = {"shot": int(path.name.split("_")[0])}
    try:
        with h5py.File(path, "r") as handle:
            for group in LENGTH_GROUPS:
                if group not in handle:
                    row[f"n_{group}"] = 0
                    continue
                row[f"n_{group}"] = int(handle[group]["xdata"].shape[0])
            if "bolo" in handle and row["n_bolo"] > 1:
                data = handle["bolo"]["ydata"]
                idx = np.linspace(0, data.shape[1] - 1, 8).astype(int)
                sample = data[:, idx]
                row["bolo_live_channels"] = int(
                    sum(_live(sample[c]) for c in range(sample.shape[0]))
                )
                x = handle["bolo"]["xdata"]
                row["bolo_t0"], row["bolo_t1"] = float(x[0]), float(x[-1])
            if "tangtv" in handle:
                x = handle["tangtv"]["xdata"]
                row["n_tangtv"] = int(x.shape[0])
                if x.shape[0] > 1:
                    row["tangtv_t0"], row["tangtv_t1"] = float(x[0]), float(x[-1])
                    data = handle["tangtv"]["ydata"]
                    idx = np.linspace(x.shape[0] // 5, 4 * x.shape[0] // 5, 4)
                    idx = idx.astype(int)
                    for c, name in enumerate(TANGTV_CHANNELS):
                        live = False
                        for i in idx:
                            if _live(np.asarray(data[c, i])):
                                live = True
                                break
                        row[f"tangtv_{name}"] = live
            else:
                row["n_tangtv"] = 0
    except Exception as error:  # noqa: BLE001 - record and continue
        row["error"] = f"{type(error).__name__}: {error}"
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    root = Path(os.environ["LABELER_ROOT"])
    out = args.out or root / "round4/detach/survey/corpus_survey.csv"
    files = sorted(CORPUS.glob("*_processed.h5"))
    if args.limit:
        files = files[:: max(1, len(files) // args.limit)][: args.limit]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(survey_one, files, chunksize=8)):
            rows.append(row)
            if (i + 1) % 1000 == 0:
                print(f"{i + 1}/{len(files)}", flush=True)
    frame = pd.DataFrame(rows).sort_values("shot")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"wrote {out}: {len(frame)} shots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
