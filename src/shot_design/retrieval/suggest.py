"""A suggested actuator configuration from a result list.

Nothing here is learned: for every actuator key the registry knows, it is the median and the
25th-75th percentile of what the successful matched shots ran, over the shots where that
actuator was on, with the source shots attached so every number is traceable. The column read is
the one `flags.rules.actuator_columns()` resolves the key to -- the `peak` stat, the level a
physicist asks the machine for -- because that is the column `QueryState.actuators` and the
operating-limit rules compare against; suggesting a mean and checking a peak would be two
quantities under one name.


Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from ..flags import rules
from ..schema import CandidateActuation
from ..shotdb.store import ShotDB
from .rank import split_stat, units


class ActuatorSuggestion(BaseModel):
    key: str  # "nbi.total", "ech.LUKE" -- the QueryState.actuators key
    column: str  # the database column it was read from
    units: str | None = None
    median: float
    q25: float
    q75: float
    n_on: int  # shots (among those used) where this actuator was on -- shots, never segments
    shots: list[int]


class Suggestion(BaseModel):
    segment: str
    n_considered: int
    n_used: int
    used_shots: list[int]
    dropped: dict[int, str] = Field(default_factory=dict)  # shot -> why it was left out
    keys: list[ActuatorSuggestion]
    candidate: CandidateActuation


def _why_drop(shot_row) -> str | None:
    if str(shot_row.get("verdict")) == "bad":
        return "verdict bad"
    hit = shot_row.get("ip_target_hit")
    if hit is None or (isinstance(hit, float) and np.isnan(hit)):
        return None  # unknown is not a miss
    if not bool(hit):
        return "missed Ip target"
    return None


def suggest(
    result_ids: list[str], db: ShotDB, k: int = 10, successful_only: bool = True
) -> Suggestion:
    """`result_ids` are segment ids ("161172:flat_top") in rank order; the first k that exist in
    the database are considered. `successful_only` drops a shot with a bad verdict or a missed
    Ip target and records why.

    One shot counts once. A result list can hold several segments of the same shot (161172:full
    and 161172:flat_top are two rows), and without this the k-cut would spend its budget on one
    shot and `n_on`/`used_shots` would report segments while calling them shots -- the same shot's
    beam power would then be weighted twice in the median. The first-ranked segment of each shot
    wins, so the surviving id is the one the search ranked highest.
    """
    ids: list[str] = []
    seen: set[int] = set()
    for sid in result_ids:
        if sid not in db.segments.index:
            continue
        shot = int(db.segments.loc[sid, "shot"])
        if shot in seen:
            continue
        seen.add(shot)
        ids.append(sid)
    ids = ids[:k]
    segment = ids[0].split(":", 1)[1] if ids else "flat_top"
    used: list[str] = []
    dropped: dict[int, str] = {}
    for sid in ids:
        shot = int(db.segments.loc[sid, "shot"])
        reason = _why_drop(db.shots.loc[shot]) if successful_only else None
        if reason:
            dropped[shot] = reason
        else:
            used.append(sid)
    sub = db.segments.loc[used]
    shots = sub["shot"].to_numpy(dtype=int) if used else np.zeros(0, dtype=int)
    unit_of = units()
    keys: list[ActuatorSuggestion] = []
    for key, col in rules.actuator_columns().items():
        if col not in sub.columns or not used:
            continue
        vals = sub[col].to_numpy(dtype=float)
        on = np.isfinite(vals) & (vals > 0)
        if not on.any():
            continue
        v = vals[on]
        keys.append(
            ActuatorSuggestion(
                key=key,
                column=col,
                units=unit_of.get(split_stat(col)[0]),
                median=float(np.median(v)),
                q25=float(np.percentile(v, 25)),
                q75=float(np.percentile(v, 75)),
                n_on=int(on.sum()),
                shots=[int(s) for s in shots[on]],
            )
        )
    candidate = CandidateActuation(
        segment=segment,  # type: ignore[arg-type]  (ids come from the database, so it is a SegName)
        actuators={s.key: s.median for s in keys},
        notes=f"median over {len(used)} matched shot(s) where each actuator was on",
    )
    return Suggestion(
        segment=segment,
        n_considered=len(ids),
        n_used=len(used),
        used_shots=[int(s) for s in shots],
        dropped=dropped,
        keys=keys,
        candidate=candidate,
    )
