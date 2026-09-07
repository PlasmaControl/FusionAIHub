"""Reading and writing `<shot>_labels.h5`.

Corpus layout, so an IGNITE loader can read a label like any other signal:
`xdata` float64 seconds, `ydata` float32 `(C, T)`. Each label gets two
companions - `<label>_spread` `(2, T)` with the ensemble min and max, and
`<label>_valid` `(1, T)` uint8, one where every input was present and inside
the training domain. Probabilities are stored, never thresholded labels:
thresholding is the consumer's decision and it is not recoverable once lost.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np

from .. import __version__
from ..config import atomic_path, git_sha
from .schema import LabelSpec

#: Every label group has two of these beside it; they are not labels.
COMPANION_SUFFIXES = ("_spread", "_valid")

#: What identifies a row of `labels_index.parquet`.
INDEX_KEYS = ("shot", "slug", "label")


@dataclass(frozen=True)
class LabelArray:
    x: np.ndarray
    y: np.ndarray
    attrs: dict[str, str] = field(default_factory=dict)


def _put(parent, name: str, x: np.ndarray, y: np.ndarray, dtype) -> h5py.Group:
    g = parent.create_group(name)
    g.create_dataset("xdata", data=np.asarray(x, dtype=np.float64))
    g.create_dataset("ydata", data=np.asarray(y, dtype=dtype))
    return g


def write_labels(
    path,
    shot: int,
    t: np.ndarray,
    decoded: Mapping[str, object],
    specs: Sequence[LabelSpec],
    valid: np.ndarray,
    *,
    run_id: str,
    features_sha256: str,
    merge: bool = True,
) -> None:
    """Write one model's labels for one shot, atomically.

    With `merge`, groups written by other models are preserved, so a second
    model's run adds to the same per-shot file instead of replacing it. The
    model's own group is always rewritten: re-running a model means its old
    numbers are stale.
    """
    path = Path(path)
    slug = specs[0].slug
    if any(spec.slug != slug for spec in specs):
        raise ValueError(
            f"every spec must share one slug; got {sorted({s.slug for s in specs})}"
        )
    now = datetime.now(UTC).isoformat(timespec="seconds")
    with atomic_path(path) as tmp, h5py.File(tmp, "w") as f:
        if merge and path.exists():
            with h5py.File(path, "r") as old:
                for key, value in old.attrs.items():
                    f.attrs[key] = value
                for group in old:
                    if group != slug:
                        old.copy(group, f, name=group)
        f.attrs["shot"] = int(shot)
        f.attrs["labelmaker_version"] = __version__
        f.attrs["git_sha"] = git_sha()
        f.attrs["written_at"] = now
        f.attrs["run_id"] = run_id
        f.attrs["features_sha256"] = features_sha256
        model_group = f.create_group(slug)
        valid_u8 = np.asarray(valid, dtype=bool).astype(np.uint8)[None, :]
        for spec in specs:
            dec = decoded[spec.name]
            g = _put(model_group, spec.name, t, np.asarray(dec.mean)[None, :],
                     np.float32)
            for key, value in spec.attrs:
                g.attrs[key] = str(value)
            g.attrs["task"] = spec.task
            g.attrs["activation"] = spec.activation
            g.attrs["units"] = spec.units
            g.attrs["classes"] = list(spec.classes)
            g.attrs["card_id"] = spec.card_id
            g.attrs["slug"] = spec.slug
            g.attrs["time_step_ms"] = float(spec.time_step_ms)
            g.attrs["ensemble_n"] = int(spec.ensemble_n)
            g.attrs["artifact_sha256"] = spec.artifact_sha256
            _put(
                model_group, f"{spec.name}_spread", t,
                np.stack([np.asarray(dec.lo), np.asarray(dec.hi)]), np.float32,
            )
            _put(model_group, f"{spec.name}_valid", t, valid_u8, np.uint8)


def read_label(path, slug: str, label: str) -> LabelArray:
    """One label series out of a label file, or KeyError."""
    with h5py.File(path, "r") as f:
        key = f"{slug}/{label}"
        if key not in f:
            raise KeyError(f"{key} not in {path}")
        g = f[key]
        return LabelArray(
            x=np.asarray(g["xdata"], dtype=np.float64),
            y=np.asarray(g["ydata"], dtype=np.float64),
            attrs={k: v for k, v in g.attrs.items()},
        )


def labelled(path) -> set[str]:
    """`"<slug>/<label>"` for every real label in the file (not companions)."""
    if not Path(path).exists():
        return set()
    out = set()
    with h5py.File(path, "r") as f:
        for slug in f:
            for name in f[slug]:
                if name.endswith(COMPANION_SUFFIXES):
                    continue
                out.add(f"{slug}/{name}")
    return out


def index_rows(path) -> list[dict]:
    """One summary row per label, for `labels_index.parquet`."""
    rows: list[dict] = []
    with h5py.File(path, "r") as f:
        shot = int(f.attrs["shot"])
        run_id = str(f.attrs.get("run_id", ""))
        written_at = str(f.attrs.get("written_at", ""))
        for slug in sorted(f):
            for label in sorted(f[slug]):
                if label.endswith(COMPANION_SUFFIXES):
                    continue
                g = f[slug][label]
                y = np.asarray(g["ydata"], dtype=np.float64)[0]
                v = np.asarray(f[slug][f"{label}_valid"]["ydata"])[0].astype(bool)
                rows.append(
                    {
                        "shot": shot,
                        "slug": slug,
                        "label": label,
                        "card_id": str(g.attrs.get("card_id", "")),
                        "task": str(g.attrs.get("task", "")),
                        "artifact_sha256": str(g.attrs.get("artifact_sha256", "")),
                        "n_total": int(y.size),
                        "n_valid": int(v.sum()),
                        "mean_valid": float(y[v].mean()) if v.any() else float("nan"),
                        "max_valid": float(y[v].max()) if v.any() else float("nan"),
                        "run_id": run_id,
                        "written_at": written_at,
                    }
                )
    return rows


def append_index(index_path, rows, *, keys=INDEX_KEYS) -> None:
    """Merge rows into the parquet index, replacing any (shot, slug, label).

    Rewritten whole and renamed into place: the index is small (one row per
    shot per label) and an interrupted run must never leave it truncated.

    `keys` is the identity of a row - which rows a re-run replaces and how the
    file is sorted. The default is the label index's; the events index passes
    `["shot", "source", "phenomenon"]` and gets the same merge for free.
    """
    import pandas as pd

    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(keys)
    new = pd.DataFrame(rows)
    if new.empty:
        return
    if index_path.exists():
        old = pd.read_parquet(index_path)
        merged = pd.concat([old, new], ignore_index=True)
        merged = merged.drop_duplicates(subset=keys, keep="last")
    else:
        merged = new
    merged = merged.sort_values(keys).reset_index(drop=True)
    tmp = index_path.with_name(index_path.name + ".tmp")
    merged.to_parquet(tmp, index=False)
    tmp.replace(index_path)
