#!/usr/bin/env python
"""Inspect only existing ONSBRADIAL cache entries on zero-DUSBRADIAL 2014 shots."""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from labeler.config import Paths
from labeler.events.raw import cache_path

REPO = Path(__file__).resolve().parents[2]


def main():
    if os.environ.get("LABELER_NO_FETCH") != "1":
        raise RuntimeError("this cache-only audit requires LABELER_NO_FETCH=1")
    paths = Paths.from_env()
    source = REPO / "outputs/labeler/rwm/shots.json"
    slices_path = paths.root / "round4/rwm/slices.parquet"
    slices = pd.read_parquet(slices_path)
    zeros = (
        slices[slices.campaign == 2014]
        .groupby("shot")
        .lock_v.apply(
            lambda values: values.notna().any() and values.dropna().eq(0).all()
        )
    )
    shots = [int(shot) for shot in zeros[zeros].index]
    expected = json.loads(source.read_text())["input_audit"][
        "dusbradial_zero_2014_shots"
    ]
    if len(shots) != expected:
        raise ValueError("zero-DUSBRADIAL shot count differs from the input audit")
    rows = []
    for shot in shots:
        path = cache_path(shot, paths=paths)
        role = str(slices.loc[slices.shot == shot, "role"].iloc[0])
        row = {"shot": shot, "role": role, "path": str(path), "cached_onsbradial": []}
        if path.is_file():
            with h5py.File(path, "r") as cache:
                names = []

                def locate(name, item, names=names):
                    locator = " ".join(
                        str(value)
                        for key, value in item.attrs.items()
                        if key in ("locator", "locators", "point", "signal")
                    )
                    if "onsbradial" in (name + " " + locator).lower():
                        if isinstance(item, h5py.Group) and "ydata" in item:
                            names.append(name + "/ydata")
                        elif name.endswith("ydata"):
                            names.append(name)

                cache.visititems(locate)
                for name in sorted(set(names)):
                    item = cache[name]
                    if isinstance(item, h5py.Dataset) and name.endswith("ydata"):
                        values = np.asarray(item)
                        row["cached_onsbradial"].append(
                            {
                                "dataset": name,
                                "shape": list(values.shape),
                                "finite": int(np.isfinite(values).sum()),
                                "nonzero": int(
                                    (np.isfinite(values) & (values != 0)).sum()
                                ),
                            }
                        )
        rows.append(row)
    record = {
        "script": "scripts/labeler/rwm_cached_onsbradial.py",
        "source": str(source.relative_to(REPO))
        + "#/input_audit/dusbradial_zero_2014_shots",
        "slices": str(slices_path),
        "LABELER_NO_FETCH": "1",
        "network_calls": 0,
        "shots_checked": len(rows),
        "hanson_shots_checked": sum(r["role"] == "hanson" for r in rows),
        "shots_with_cached_onsbradial": sum(bool(r["cached_onsbradial"]) for r in rows),
        "rows": rows,
        "next_input": (
            "PTDATA ONSBRADIAL, reported disruption-py fallback for DUSBRADIAL"
        ),
        "fallback_verified_here": False,
    }
    out = REPO / "outputs/labeler/rwm/cached_sensor_audit.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"checked {len(rows)} cached shots; wrote {out}")


if __name__ == "__main__":
    main()
