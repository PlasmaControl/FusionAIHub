"""Read a shot of omnimode's port-v6 labels and apply the owner's display rules.

The files are the ones `labeler.mhd.pipeline` writes (omnimode's `modes.pipeline` is its port
and keeps the columns), so a shot reads as a `labeler.mhd.pipeline.Shot` and every function of
`labeler.mhd.rows` runs on it. The rules, owner 2026-11-07:

1. Harmonics are not labelled: a track with `harm` true (a 4/2 locked at 2f of a 2/1) is
   dropped with its frames. An independent 4/2, `harm` false, is a mode and stays.
2. Scope is |n| = 1, 2, 3 with their m (the port emits no other n; a later redesign adds the
   "n>3" and "n=0" classes). A track outside it is dropped.
3. Only labelled modes are written (`labelled`): status "m/n", the MAP class held at
   P >= 0.9 inside the resonance gate. An abstention (uncertain, no rotation, q_min above) or
   an unmodelled track is not a mode.
4. Amplitudes are uncalibrated; none is carried.

A fixed-frequency line (`line` true) is never a mode and is dropped by `labeler.mhd.rows` itself.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from labeler.mhd import pipeline

#: The toroidal numbers the port emits and this package labels.
SCOPE_N = (1, 2, 3)
#: The frame columns read: `labeler.mhd.rows` and `labeler.modes.cells` use only these, and a
#: frames file holds about 70 (an all-shot publish would otherwise hold them all in memory).
FRAME_COLUMNS = ["family_id", "t_ms"]
#: `m/n`, as the label model names its classes (`labeler.mhd.labelmodel.NAMES`).
CLASS = re.compile(r"(\d+)/(\d+)\Z")


def parse_class(name: str) -> tuple[int, int]:
    """`(m, n)` of a class name: `"2/1"` is `(2, 1)`."""
    found = CLASS.match(str(name))
    if not found:
        raise ValueError(f"{name!r} is not an m/n class name")
    return int(found.group(1)), int(found.group(2))


def event_of(name: str) -> str:
    """The subcategory id of a class: `"2/1"` is `mhd_n1.m2`."""
    m, n = parse_class(name)
    return f"mhd_n{n}.m{m}"


def files(directory, shot: int) -> dict[str, Path]:
    base = Path(directory) / str(int(shot))
    return {
        "info": base.with_suffix(".json"),
        "tracks": Path(f"{base}.tracks.parquet"),
        "frames": Path(f"{base}.frames.parquet"),
    }


def has_labels(directory, shot: int) -> bool:
    """The shot's three files exist."""
    return all(path.is_file() for path in files(directory, shot).values())


def read_shot(directory, shot: int) -> pipeline.Shot | None:
    """The shot's info record, tracks and frames (`FRAME_COLUMNS`); None without the three files.

    Nothing is dropped here: `apply_rules` does that, so a count of what the rules removed is
    possible. The caller decides which shots may be read."""
    paths = files(directory, shot)
    if not all(path.is_file() for path in paths.values()):
        return None
    info = json.loads(paths["info"].read_text())
    return pipeline.Shot(
        int(shot),
        info,
        pd.read_parquet(paths["tracks"]),
        pd.read_parquet(paths["frames"], columns=FRAME_COLUMNS),
    )


def _flag(table: pd.DataFrame, column: str) -> pd.Series:
    """A boolean column with a missing value false; all false when it is absent."""
    if column not in table.columns:
        return pd.Series(False, index=table.index)
    return table[column].astype("boolean").fillna(False).astype(bool)


def _restrict(got: pipeline.Shot, keep: pd.Series) -> pipeline.Shot:
    tracks = got.tracks[keep.to_numpy()].reset_index(drop=True)
    frames = got.frames[got.frames.family_id.isin(tracks.family_id)]
    return pipeline.Shot(got.shot, got.info, tracks, frames.reset_index(drop=True))


def apply_rules(got: pipeline.Shot) -> tuple[pipeline.Shot, dict[str, int]]:
    """Rules 1 and 2: the shot without its harmonic tracks and its tracks outside |n| 1-3.

    n is taken as |n|. Returns the shot and what was dropped: `harmonics` (tracks with
    `harm` true, whatever their n), `out_of_scope` (the other tracks outside |n| 1-3) and
    `kept`. A fixed-frequency line is kept here; `labeler.mhd.rows` leaves it out of every
    event, as it does for v5."""
    tracks = got.tracks.copy()
    if len(tracks):
        tracks["n"] = tracks["n"].astype(int).abs()
    harm = _flag(tracks, "harm")
    scope = tracks["n"].isin(SCOPE_N) if len(tracks) else pd.Series(dtype=bool)
    kept = ~harm & scope
    counts = {
        "harmonics": int(harm.sum()),
        "out_of_scope": int((~harm & ~scope).sum()),
        "kept": int(kept.sum()),
    }
    frames = got.frames
    shot = pipeline.Shot(got.shot, got.info, tracks, frames)
    return _restrict(shot, kept), counts


def labelled(got: pipeline.Shot) -> pipeline.Shot:
    """Rule 3: only the tracks that are a labelled mode, status "m/n" with a class, no line.

    Call after `apply_rules`. An abstention or an unmodelled track is not a mode."""
    tracks = got.tracks
    if not len(tracks):
        return got
    label = tracks["label"].fillna("").astype(str)
    keep = (tracks["status"] == "m/n") & (label != "") & ~_flag(tracks, "line")
    return _restrict(got, keep)


def classes(got: pipeline.Shot) -> list[str]:
    """The classes the labelled modes of a (rule-applied) shot carry, by n then m."""
    tracks = labelled(got).tracks
    names = set(tracks["label"].astype(str)) if len(tracks) else set()
    return sorted(names, key=lambda name: parse_class(name)[::-1])
