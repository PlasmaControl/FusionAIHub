"""The links table: one row per (shot, paper) verified by the context rule.

It holds the links that rule verifies, not every paper that names a shot.

Columns: `shot, source, record_id, doi, title, year, venue,
context, match_type, verified_by`. `source` is `osti` or `arxiv`; `match_type`
is `exact` or `range`; `verified_by` is `auto` (the context rule) or `human`
(the hand check). Only verified links are rows; a hit whose text lacks context
stays in the audit table beside the cache, never here.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass

import pandas as pd

from ..config import atomic_path
from ..events.catalog.points import validate_csv_fields
from ..events.databases import DatabaseError
from .context import mentions

PAPER_COLUMNS = (
    "shot",
    "source",
    "record_id",
    "doi",
    "title",
    "year",
    "venue",
    "context",
    "match_type",
    "verified_by",
)
SOURCES = ("osti", "arxiv")
MATCH_TYPES = ("exact", "range")
VERIFIERS = ("auto", "human")


@dataclass(frozen=True)
class Record:
    """One paper: where it came from and how to cite it."""

    source: str
    record_id: str
    doi: str = ""
    title: str = ""
    year: int | None = None
    venue: str = ""


def links(record: Record, text: str, shots: Collection[int]) -> list[dict]:
    """One row per shot in `shots` that `text` names in context.

    A shot named several times keeps its first exact mention, else its first
    range.
    """
    best: dict[int, object] = {}
    for m in mentions(text, shots):
        held = best.get(m.shot)
        if held is None or (held.match_type == "range" and m.match_type == "exact"):
            best[m.shot] = m
    return [
        {
            "shot": shot,
            "source": record.source,
            "record_id": record.record_id,
            "doi": record.doi,
            "title": record.title,
            "year": record.year,
            "venue": record.venue,
            "context": m.context,
            "match_type": m.match_type,
            "verified_by": "auto",
        }
        for shot, m in sorted(best.items())
    ]


def validate_papers(frame: pd.DataFrame) -> pd.DataFrame:
    """The table with its types fixed; `DatabaseError` names the first problem."""
    if tuple(frame.columns) != PAPER_COLUMNS:
        raise DatabaseError(f"Expected columns {PAPER_COLUMNS}")
    result = frame.copy()
    shots = pd.to_numeric(result["shot"], errors="coerce")
    if not (shots.notna() & (shots >= 0) & (shots % 1 == 0)).all():
        raise DatabaseError("shot must be a nonnegative integer")
    result["shot"] = shots.astype("int64")
    for column, allowed in (
        ("source", SOURCES),
        ("match_type", MATCH_TYPES),
        ("verified_by", VERIFIERS),
    ):
        bad = sorted(set(result[column]) - set(allowed), key=str)
        if bad:
            raise DatabaseError(f"{column} must be one of {list(allowed)}, not {bad}")
    if (result["record_id"].astype(str).str.strip() == "").any():
        raise DatabaseError("every link names its record")
    if result.duplicated(["shot", "source", "record_id"]).any():
        raise DatabaseError("a (shot, source, record_id) link appears twice")
    result["year"] = pd.to_numeric(result["year"], errors="coerce").astype("Int64")
    return result


def papers_frame(rows: Iterable[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows), columns=list(PAPER_COLUMNS))
    return validate_papers(frame).sort_values(
        ["shot", "source", "record_id"], kind="stable", ignore_index=True
    )


def write_papers(frame: pd.DataFrame, path) -> None:
    """Validate, sort by shot, source and record, and write atomically."""
    frame = papers_frame(frame.to_dict("records"))
    with atomic_path(path) as tmp:
        frame.to_csv(tmp, index=False)


def read_papers(path) -> pd.DataFrame:
    validate_csv_fields(path)
    frame = pd.read_csv(
        path,
        dtype={"record_id": str, "doi": str},
        keep_default_na=False,
        na_values={"year": [""]},
    )
    return validate_papers(frame)
