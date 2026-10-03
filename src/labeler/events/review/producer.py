"""Read-only detachment label/vote strips from the producer's final bin table.

Interpretation is frozen from producer-owned records. No vote is computed here. The
final labels_bins.csv.gz interface uses 50 ms bins; zero means not assessed.
Test rows are never suggestions for this training-facing review page.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import pandas as pd

from ..panels.detachment import indicator_path
from . import recipe

BIN_MS = 50.0


def source_path(paths) -> Path:
    return Path(
        os.environ.get(
            "LABELER_DETACHMENT_LABELS",
            str(paths.root / "round4/detach/labels_bins.csv.gz"),
        )
    )


@lru_cache(maxsize=2)
def _read(path, _mtime, _size, _inode):
    return pd.read_csv(path, keep_default_na=False, low_memory=False)


def _verify_snapshot(frame, bins):
    """Reject labels from an incomplete or different producer vote snapshot."""
    starts = np.asarray(bins["start_ms"], dtype=float)
    if not np.isfinite(starts).all() or np.any(np.diff(starts) != BIN_MS):
        raise ValueError("producer NPZ clock must be increasing in 50 ms bins")
    if frame.empty:
        # The producer also applies a per-shot eligibility gate. No published
        # rows is distinct from missing assessed rows within a published shot.
        return
    assessed = (
        np.sum(
            [
                bins[f"{name}_valid"]
                for name in (
                    "afrac",
                    "prad",
                    "tangtv",
                )
            ],
            axis=0,
        )
        >= 2
    )
    labelled = frame[pd.to_numeric(frame.state_lm, errors="coerce") > 0]
    if set(labelled.start_ms) != set(starts[assessed]):
        raise ValueError("producer labels incomplete for assessed vote bins")
    indexed = frame.set_index("start_ms")
    for name in ("afrac", "prad", "tangtv"):
        for suffix in ("valid", "vote", "reason"):
            field = f"{name}_{suffix}"
            if not np.array_equal(
                indexed.loc[starts[assessed], field], bins[field][assessed]
            ):
                raise ValueError(f"producer snapshot disagrees on {field}")
        field = f"{name}_value"
        if field in indexed and field in bins:
            values = pd.to_numeric(
                indexed.loc[starts[assessed], field], errors="coerce"
            )
            if not np.allclose(
                values, bins[field][assessed], rtol=1e-5, atol=1e-7, equal_nan=True
            ):
                raise ValueError(f"producer snapshot disagrees on {field}")
    if not np.array_equal(
        indexed.loc[starts[assessed], "tangtv_source"], bins["tangtv_source"][assessed]
    ):
        raise ValueError("producer snapshot disagrees on tangtv_source")


def load(shot, paths, window_ms=None):
    """JSON-ready strips, with validity distinct from valid abstention (-1)."""
    path = source_path(paths)
    interpretation = recipe.load(path)
    result = {
        "source": str(path),
        "bin_width_ms": BIN_MS,
        "definitions": interpretation["record"].get("definitions", {}),
        "recipe": interpretation,
        "source_suppressed": False,
        "bin_start_ms": [],
        "bin_end_ms": [],
        "state_lm": [],
        "state_rule": [],
        "votes": {},
        "tangtv_source": [],
        "confidence": [],
        "label_available": False,
        "note": "Unverified primary producer suggestion, with rule comparison.",
    }
    if not path.is_file():
        result["source_suppressed"] = True
        result["reason"] = "Producer label table unavailable"
        return result
    try:
        stat = path.stat()
        frame = _read(path, stat.st_mtime_ns, stat.st_size, stat.st_ino)
        frame = frame[pd.to_numeric(frame.shot, errors="coerce") == int(shot)]
        result["label_available"] = not frame.empty
        if frame.empty:
            result["note"] = "No producer label published for this shot; votes only."
        bins_path = indicator_path(shot, paths)
        if frame.empty and not (bins_path.is_file() and bins_path.suffix == ".npz"):
            result["reason"] = "Shot has no producer label bins"
            return result
        if "split" in frame and frame.split.astype(str).str.lower().eq("test").any():
            result["source_suppressed"] = True
            result["reason"] = "Producer blind test shot excluded from suggestions"
            return result
        if (
            "holdout" in frame
            and frame.holdout.astype(str).str.lower().isin(["true", "1", "yes"]).any()
        ):
            result["source_suppressed"] = True
            result["reason"] = "Producer blind test shot excluded from suggestions"
            return result
        if bins_path.is_file() and bins_path.suffix == ".npz":
            with np.load(bins_path, allow_pickle=False) as bins:
                if "start_ms" in bins:
                    if not set(frame.start_ms).issubset(set(bins["start_ms"])):
                        raise ValueError("producer labels and vote bin clocks disagree")
                    _verify_snapshot(frame, bins)
                    # labels_bins contains assessed bins only. Votes/validity
                    # span every producer bin, including unassessed time.
                    frame = frame.set_index("start_ms").reindex(bins["start_ms"])
                    for field in bins.files:
                        if field.endswith(("_vote", "_valid", "_reason")) or (
                            field == "tangtv_source"
                        ):
                            frame[field] = bins[field]
                    for field in ("state_lm", "state_rule"):
                        frame[field] = frame[field].fillna(0)
                    frame = frame.reset_index()
                    result["vote_source"] = str(bins_path)
        starts = np.asarray(frame.start_ms, dtype=float)
        if not np.isfinite(starts).all() or np.any(np.diff(starts) <= 0):
            raise ValueError("producer clock must be finite and increasing")
        ends = starts + BIN_MS
        if np.any(starts[1:] < ends[:-1]):
            raise ValueError("producer bins overlap")
        keep = np.ones(len(frame), dtype=bool)
        if window_ms:
            keep = (starts < window_ms[1]) & (ends > window_ms[0])
            starts = np.maximum(starts, window_ms[0])
            ends = np.minimum(ends, window_ms[1])
        result["bin_start_ms"] = starts[keep].tolist()
        result["bin_end_ms"] = ends[keep].tolist()
        for name in ("state_lm", "state_rule"):
            values = np.asarray(frame[name], dtype=float)
            if not np.isin(values, [0, 1, 2, 3, 4]).all():
                raise ValueError(f"invalid producer state in {name}")
            result[name] = values[keep].astype(int).tolist()
        for name in ("afrac", "prad", "tangtv"):
            valid = frame[f"{name}_valid"].astype(str).str.lower()
            if not valid.isin(["true", "false", "1", "0"]).all():
                raise ValueError(f"invalid producer validity in {name}")
            valid = valid.isin(["true", "1"]).to_numpy()
            votes = np.asarray(frame[f"{name}_vote"], dtype=float)
            if not np.isin(votes, [-1, 1, 2, 3]).all():
                raise ValueError(f"invalid producer vote in {name}")
            if np.any(votes[~valid] != -1):
                raise ValueError(f"producer vote on invalid {name} bin")
            result["votes"][name] = {
                "valid": valid[keep].tolist(),
                "vote": votes[keep].astype(int).tolist(),
                "reason": frame[f"{name}_reason"].astype(str).to_numpy()[keep].tolist(),
            }
        result["tangtv_source"] = (
            frame.tangtv_source.astype(str).to_numpy()[keep].tolist()
        )
        result["evidence"] = {
            field: frame[field].fillna("").to_numpy()[keep].tolist()
            for field in frame.columns
            if field in ("tier", "afrac_method") or "marfe_" in field
        }
        confidence = pd.to_numeric(frame.confidence, errors="coerce").to_numpy()
        result["confidence"] = [
            float(value) if np.isfinite(value) else None for value in confidence[keep]
        ]
        return result
    except (
        OSError,
        ValueError,
        KeyError,
        AttributeError,
        BadZipFile,
        EOFError,
    ) as error:
        result.update(
            bin_start_ms=[],
            bin_end_ms=[],
            state_lm=[],
            state_rule=[],
            votes={},
            label_available=False,
            source_suppressed=True,
        )
        result["reason"] = f"Producer label table unreadable: {error}"
        return result
