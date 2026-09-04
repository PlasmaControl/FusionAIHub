"""What one label is.

A `LabelSpec` is everything a consumer needs to interpret a stored series
without opening the model card: the task, the activation already applied,
the class names, the model's time step, and the digest of the weights that
produced it. It is written into the HDF5 group's attributes, so a label file
is self-describing even when moved.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class LabelSpec:
    name: str
    task: str
    activation: str
    units: str
    classes: tuple[str, ...]
    slug: str
    card_id: str
    time_step_ms: float
    ensemble_n: int
    artifact_sha256: str


def group_path(slug: str, label: str) -> str:
    """Where a label lives inside a label file."""
    return f"{slug}/{label}"


def artifact_digest(sha_by_name: Mapping[str, str]) -> str:
    """One digest standing for a whole set of weight files.

    Order-independent: the per-file digests are sorted by file name before
    hashing, so re-listing the ensemble cannot change the result.
    """
    h = hashlib.sha256()
    for name in sorted(sha_by_name):
        h.update(name.encode())
        h.update(sha_by_name[name].encode())
    return h.hexdigest()


def specs_for(adapter, artifact_sha256: str) -> tuple[LabelSpec, ...]:
    """One `LabelSpec` per output field of a model."""
    return tuple(
        LabelSpec(
            name=f.name,
            task=f.task,
            activation=f.activation,
            units=f.units,
            classes=tuple(f.classes),
            slug=adapter.slug,
            card_id=adapter.card_id,
            time_step_ms=float(adapter.time_step_ms),
            ensemble_n=int(adapter.ensemble_n),
            artifact_sha256=artifact_sha256,
        )
        for f in adapter.output_spec.fields
    )
