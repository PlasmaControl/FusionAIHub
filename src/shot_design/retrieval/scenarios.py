"""Step 1 of the two-step flow: one question -> a handful of scenarios.

A scenario is one experiment: the search hits that matched it plus the other database shots of
the same mini-proposal or run, or failing that the same research theme. No language model is
involved; the grouping is deterministic so the landing page works with Ollama down. The model,
when present, only phrases these (ui/chat.py's propose_scenarios tool calls this function).


Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Iterable

from pydantic import BaseModel, Field

from shot_design import config
from shot_design.retrieval import describe, rank
from shot_design.schema import QueryState
from shot_design.shotdb.store import ShotDB

_LABELS = {
    "startup_checkout": "startup / checkout",
    "rmp_elm": "RMP ELM control",
    "qh_mode": "QH mode",
    "neg_tri": "negative triangularity",
    "detachment_divertor": "detachment / divertor",
    "hybrid_high_beta": "high beta / steady state",
    "pedestal": "pedestal",
    "lh_threshold_isotope": "L-H threshold / isotope",
    "fast_ion_ae": "fast ions / Alfvén modes",
    "tearing_mhd": "tearing / MHD",
    "disruption_runaway": "disruptions / runaways",
    "current_drive": "current drive",
    "transport": "transport",
    "control": "control",
}


class Scenario(BaseModel):
    kind: str  # "mp" | "run" | "theme"
    key: str  # mpid, run_id or theme id
    goal: str  # MP title (or run title, or theme label) -- what the experiment set out to do
    shots: list[int]  # matched hits in rank order, then the group's other database shots
    representative_shot: int  # best-ranked shot of the group
    n_shots: int  # len(shots): the whole experiment, not just what the search surfaced
    n_hits: int  # how many of `shots` were search hits
    n_success: int  # verdict == "good"
    verdicts: dict[str, int] = Field(default_factory=dict)
    themes: list[str] = Field(default_factory=list)
    quote: str | None = None  # one operator entry, verbatim, from quote_shot
    quote_shot: int | None = None
    best_rank: int  # 1-based rank of the representative in the search


class ScenarioSet(BaseModel):
    query_text: str
    theme: str | None
    n_hits: int
    scenarios: list[Scenario]


def themes() -> list[dict]:
    return list(config.load_yaml("labels.yaml")["themes"])


def theme_label(theme_id: str) -> str:
    return _LABELS.get(theme_id, theme_id.replace("_", " "))


def _pattern(theme: dict) -> re.Pattern[str]:
    return re.compile("|".join(re.escape(k) for k in theme["keywords"]), re.IGNORECASE)


def theme_counts(titles: Iterable[str | None], theme_list: list[dict]) -> dict[str, int]:
    """Shots per theme, matched on titles only (matching full MP text is too broad)."""
    rows = [t for t in titles if t]
    return {t["id"]: sum(1 for title in rows if _pattern(t).search(title)) for t in theme_list}


def themes_of(title: str | None, theme_list: list[dict]) -> list[str]:
    if not title:
        return []
    return [t["id"] for t in theme_list if _pattern(t).search(title)]


def group_key(mpid: str | None, run_id: str | None, shot_themes: list[str]) -> tuple[str, str]:
    if mpid:
        return ("mp", mpid)
    if run_id:
        return ("run", run_id)
    return ("theme", shot_themes[0] if shot_themes else "other")


def _rest_of_group(db: ShotDB, kind: str, key: str, hits: list[int]) -> list[int]:
    """The database shots of this experiment that the search did not surface, ascending.

    `rank.search` dedups near-duplicates and decays a run day, so the hits are a sample of the
    experiment, not the experiment. An mpid or a run_id names the rest of it; a theme does not.
    A "run" expansion excludes rows that carry an mpid: those belong to their own MP scenario."""
    column = {"mp": "mpid", "run": "run_id"}.get(kind)
    if column is None:
        return []
    already = set(hits)
    mask = db.shots[column] == key
    if kind == "run":
        mask &= db.shots["mpid"].isna()
    same = db.shots.index[mask]
    return sorted(int(s) for s in same if int(s) not in already)


def _order(scenarios: list[Scenario]) -> list[Scenario]:
    """Sort scenarios by number of matching hits, then group size, then best rank."""
    return sorted(scenarios, key=lambda s: (-s.n_hits, -s.n_shots, s.best_rank))


def propose_scenarios(
    text: str | None,
    db: ShotDB,
    theme: str | None = None,
    n: int | None = None,
    n_hits: int | None = None,
) -> ScenarioSet:
    """Search with `text` (or the theme's keywords), group the hits, keep the `n` largest groups
    ordered by number of matching hits, then group size, then best rank. Hits are flat-top
    segments, one per shot after grouping.

    A group is then expanded back to the whole experiment: an "mp" group takes every database shot
    with that `mpid` and a "run" group every database shot with that `run_id`, because `rank.search`
    dedups near-duplicates and decays a run day, so an MP that ran twelve near-identical shots can
    surface as three hits. A "theme" group has no such key and keeps the hits only. `shots` lists
    the matched hits first, in rank order, then the group's other database shots ascending;
    `n_shots` is `len(shots)`, `n_hits` is how many of them were hits, and `n_success` / `verdicts`
    are counted over the whole list. `representative_shot` is still the best-ranked hit."""
    cfg = config.load_yaml("ui.yaml")["landing"]
    n = n or int(cfg["n_scenarios"])
    n_hits = n_hits or int(cfg["n_hits"])
    theme_list = themes()
    if theme is not None:
        match = [t for t in theme_list if t["id"] == theme]
        if not match:
            raise ValueError(f"unknown theme {theme!r}")
        if not text:
            text = " ".join(match[0]["keywords"][:4])
    query_text = (text or "").strip()
    if not query_text:
        return ScenarioSet(query_text="", theme=theme, n_hits=0, scenarios=[])
    found = rank.search(QueryState(text=query_text, n=n_hits), db)
    groups: OrderedDict[tuple[str, str], list[tuple[int, int]]] = OrderedDict()
    seen: set[int] = set()
    records = {}  # one db.get per hit; the title that groups a shot is the title we report
    for i, it in enumerate(found.items, start=1):
        if it.shot in seen:
            continue
        seen.add(it.shot)
        row = db.shots.loc[it.shot]
        mpid = row["mpid"] if isinstance(row["mpid"], str) else None
        run_id = row["run_id"] if isinstance(row["run_id"], str) else None
        records[it.shot] = db.get(it.shot)
        title = records[it.shot].human.mp_title or records[it.shot].human.run_title
        key = group_key(mpid, run_id, themes_of(title, theme_list))
        groups.setdefault(key, []).append((i, it.shot))
    out: list[Scenario] = []
    for (kind, key), members in groups.items():
        members.sort()
        best_rank, rep = members[0]
        hits = [s for _, s in members]
        shots = hits + _rest_of_group(db, kind, key, hits)
        rows = db.shots.loc[shots]
        verdicts: dict[str, int] = {}
        for v in rows["verdict"]:
            label = v if isinstance(v, str) else "missing"  # NaN / None is not a verdict
            verdicts[label] = verdicts.get(label, 0) + 1
        rec = records[rep]
        title = rec.human.mp_title or rec.human.run_title
        goal = title or (theme_label(key) if kind == "theme" else f"run {key}")
        quote, quote_shot = None, None
        for s in shots:  # the representative first, then the others, until one has a quote
            q = describe.best_quote(records.get(s) or db.get(s))
            if q is not None:
                quote, quote_shot = q[1], s
                break
        out.append(
            Scenario(
                kind=kind,
                key=key,
                goal=goal,
                shots=shots,
                representative_shot=rep,
                n_shots=len(shots),
                n_hits=len(hits),
                n_success=verdicts.get("good", 0),
                verdicts=verdicts,
                themes=themes_of(title, theme_list),
                quote=quote,
                quote_shot=quote_shot,
                best_rank=best_rank,
            )
        )
    out = _order(out)
    return ScenarioSet(query_text=query_text, theme=theme, n_hits=len(seen), scenarios=out[:n])
