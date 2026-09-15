"""The 200-prompt evalset: loading it, the frozen split, and the metrics it is run to produce.

What is measured, and why each one is here rather than a single "accuracy":

* **coverage** -- the fraction of prompts that got any result at all. The plan's bar is 95 %. A
  retrieval system that answers nothing is not inaccurate, it is absent, and this separates the
  two failure modes before any ranking question is asked.
* **hard-filter survival** -- how much of the split each prompt's `--where` clauses left standing.
  A top-5 out of six candidates and a top-5 out of three hundred are different answers, and
  without this number they are indistinguishable in the aggregate.
* **channel participation** -- which of `retrieval.channels.CHANNELS` had anything to say. A
  channel that fires on 3 % of prompts is either mis-wired or measuring something very narrow,
  and neither shows up in a score.
* **category -> phenomenon resolution** -- the fraction of prompts whose authored
  `expect_phenomena` are ALL returned by `retrieval.phenomena.resolve`. This is a measurement of
  the LEXICON against the words physicists actually type, not of retrieval: "wide pedestal QH"
  and "radiative divertor" are how the operators write two phenomena whose alias lists do not
  contain those words. The plan's bar is 80 % on qh_mode, elm_rmp and fast_ions.
* **run diversity and duplicate rate** -- ten results from one run day are one experiment shown
  ten times. Diversity is averaged over the prompts that ANSWERED: a prompt that returned nothing
  has no top-10 to be diverse, and dividing by it reports a number smaller than any top-10 ever
  showed. `rerank`'s run-day decay and dedup exist to prevent that, and these two numbers are
  what says whether they work on real queries.
* **the proxy grade** -- a narrow, machine-checkable stand-in for the 20 hand grades, which
  announces itself as a proxy in `ProxyGrade.caveat`. See there.

**The split.** `configs/shot_design/evalsets/split.yaml` cuts the 500 `recommender_v1` shots into
`dev` (~400) and `eval` (~100) by a sha256 of the RUN ID, so no run day straddles the boundary.
That is the only cut that means anything here: two shots of one run day share a session leader, a
mini-proposal, a machine configuration and usually a logbook sentence, so a shot-wise split would
leave a near-copy of every eval shot in dev and every text channel would be scored on its own
training data. The rule is a pure function of the run id (`run_bucket`), so it is reproducible in
any process and from any shot list; `split.yaml` is the frozen result of applying it to the
committed `recommender_v1.yaml`, and a test re-derives it.

Nothing in this module tunes anything. It runs the frozen prompts through `retrieval.rank.search`
exactly as `shot_design query` would and writes down what came back.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .. import config
from ..retrieval import channels as ch_mod
from ..retrieval import phenomena as ph_mod
from ..retrieval import rank as rank_mod
from ..schema import (
    CategoryStats,
    EvalPrompt,
    EvalReport,
    PromptOutcome,
    ProxyGrade,
    Range,
)

__all__ = [
    "EVALSET_NAME",
    "EVAL_BUCKET_MAX",
    "SPLIT_BUCKETS",
    "SPLIT_RULE",
    "Evalset",
    "bars_met",
    "build_split",
    "evalset_dir",
    "failed_bars",
    "load_evalset",
    "load_split",
    "markdown",
    "parse_constraints",
    "parse_phenomena",
    "run",
    "run_bucket",
    "split_of",
    "split_shots",
]

EVALSET_NAME = "reference_shot_prompts.csv"
SPLIT_NAME = "split.yaml"

# The split. 20 % of the hash space, chosen as a round fraction and not as a number that made a
# count come out nicely: on `recommender_v1` it lands on 110 eval shots over 50 run days and 390
# dev shots over 182, which is the "~400/100" the brief asks for.
SPLIT_BUCKETS = 1000
EVAL_BUCKET_MAX = 200
SPLIT_RULE = (
    "eval iff int(sha256(run_id)[:8], 16) % 1000 < 200 -- a run day, never a shot, decides"
)

# The top-k the diversity and duplicate numbers are read off. Ten because that is `shot_design query`'s
# own default; the proxy grade looks at five because the hand rubrics are written about a top-5.
TOP_K = 10
PROXY_K = 5

# The exit bars the plan states, kept here so the markdown can mark them and the report cannot
# quietly acquire new ones.
COVERAGE_BAR = 0.95
RESOLUTION_BAR = 0.80
RESOLUTION_BAR_CATEGORIES = ("qh_mode", "elm_rmp", "fast_ions")


def evalset_dir() -> Path:
    return config.CONFIG_DIR / "evalsets"


# --------------------------------------------------------------------------------- the split


def run_bucket(run_id: str | None) -> int:
    """The run day's bucket in [0, 1000). sha256, not `hash()`: Python salts string hashes per
    process, and a split that moved between runs would not be a split."""
    digest = hashlib.sha256(str(run_id).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % SPLIT_BUCKETS


def split_of(run_id: str | None) -> str:
    return "eval" if run_bucket(run_id) < EVAL_BUCKET_MAX else "dev"


def build_split(shot_list: Path | str) -> dict[str, Any]:
    """Apply the rule to a committed shot-list YAML and return the document `split.yaml` holds.

    The rows carry the run id the selection recorded, so this needs no database and no text: the
    split is decided by the shot list alone and can be recomputed by anyone holding it.
    """
    doc = yaml.safe_load(Path(shot_list).read_text(encoding="utf-8"))
    sides: dict[str, dict[str, set]] = {
        name: {"shots": set(), "run_ids": set()} for name in ("dev", "eval")
    }
    for row in doc.get("shots") or []:
        run_id = str(row.get("run_id"))
        side = sides[split_of(run_id)]
        side["shots"].add(int(row["shot"]))
        side["run_ids"].add(run_id)
    return {
        "source": doc.get("name", str(shot_list)),
        "rule": SPLIT_RULE,
        "buckets": SPLIT_BUCKETS,
        "eval_bucket_max": EVAL_BUCKET_MAX,
        **{
            name: {
                "n_shots": len(side["shots"]),
                "n_run_ids": len(side["run_ids"]),
                "run_ids": sorted(side["run_ids"]),
                "shots": sorted(side["shots"]),
            }
            for name, side in sides.items()
        },
    }


def load_split(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path) if path else evalset_dir() / SPLIT_NAME
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def split_shots(split: str = "eval", path: Path | str | None = None) -> set[int]:
    """The shots on one side of the frozen split. `all` is both, which is not a split and says so
    in the report it lands in."""
    if split not in ("dev", "eval", "all"):
        raise ValueError(f"unknown split {split!r}; it is one of dev, eval, all")
    doc = load_split(path)
    if split == "all":
        return {int(s) for s in doc["dev"]["shots"]} | {int(s) for s in doc["eval"]["shots"]}
    return {int(s) for s in doc[split]["shots"]}


# ------------------------------------------------------------------------------ the evalset


def parse_phenomena(value: str | None) -> list[str]:
    return [p for p in str(value or "").split("|") if p]


def parse_constraints(value: str | None) -> dict[str, Range]:
    """`{"betan_mean":[2.0,null]}` -> `{"betan_mean": Range(lo=2.0, hi=None)}`.

    The same `[lo, hi]` shape `--where COLUMN=LO:HI` parses to, so a prompt row and a command line
    cannot disagree about what a bound means.
    """
    text = str(value or "").strip()
    if not text:
        return {}
    raw = json.loads(text)
    out: dict[str, Range] = {}
    for col, span in raw.items():
        lo, hi = span
        out[str(col)] = Range(lo=None if lo is None else float(lo), hi=None if hi is None else float(hi))
    return out


@dataclass(frozen=True)
class Evalset:
    path: Path
    sha256: str
    prompts: list[EvalPrompt]

    def __len__(self) -> int:
        return len(self.prompts)


def load_evalset(path: Path | str | None = None) -> Evalset:
    """The frozen CSV, with its hash. The hash travels with the prompts into every report so a
    result can never be attributed to the wrong set."""
    p = Path(path) if path else evalset_dir() / EVALSET_NAME
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    with p.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    prompts = [
        EvalPrompt(
            prompt_id=r["prompt_id"],
            category=r["category"],
            prompt=r["prompt"],
            expect_phenomena=parse_phenomena(r["expect_phenomena"]),
            expect_constraints=parse_constraints(r["expect_constraints"]),
            expect_segment=r["expect_segment"] or "flat_top",
            hand_graded=str(r["hand_graded"]).strip() == "1",
            notes=r.get("notes") or "",
        )
        for r in rows
    ]
    return Evalset(p, digest, prompts)


# -------------------------------------------------------------------------------- the run


def _exclusions(db, keep: set[int] | None) -> set[int]:
    """Which of the database's shots this run must not see.

    Excluding the complement, rather than filtering the results afterwards, is what makes the
    split real: the hard filter, the BM25 corpus statistics and the k-NN neighbourhoods are all
    computed on the shots that remain, so an eval-split run never touches a dev shot at any stage.
    """
    if keep is None:
        return set()
    return {int(s) for s in db.shots.index} - keep


def _resolution(prompt: EvalPrompt) -> tuple[list[str], bool | None]:
    resolved = [pid for pid, _ in ph_mod.resolve(prompt.prompt)]
    if not prompt.expect_phenomena:
        return resolved, None
    return resolved, set(prompt.expect_phenomena) <= set(resolved)


def _proxy_hit(prompt: EvalPrompt, shots: list[int], db) -> bool | None:
    """The narrow, machine-checkable half of a hand grade -- see `ProxyGrade`.

    A prompt that expects phenomena passes if ANY of the top 5 shots carries evidence of EVERY
    one of them, in any evidence class (an operator's sentence counts, and is reported as
    text-only elsewhere). A prompt that expects only constraints passes if EVERY one of the top 5
    satisfies them -- that is a hard filter, so anything else would be a bug rather than a miss.
    A prompt that expects neither is not machine-checkable and returns None instead of a free pass.

    `evidence()` is read at `phenomena.configured_label_floor()`, the SAME floor `locate()`
    passes at its own call site, and it is passed explicitly: the proxy grade is about the
    shots a user would have been shown, so a floor that changed what `locate` returns and not
    what this reads would make the two disagree silently.
    """
    top = shots[:PROXY_K]
    if not top:
        return False
    floor = ph_mod.configured_label_floor()
    if prompt.expect_phenomena:
        for shot in top:
            if all(
                ph_mod.has_evidence(
                    ph_mod.evidence(
                        shot, pid, db, prompt.expect_segment, label_floor=floor
                    )
                )
                for pid in prompt.expect_phenomena
            ):
                return True
        return False
    if prompt.expect_constraints:
        mask = db.mask(prompt.expect_segment, prompt.expect_constraints)
        allowed = {int(s) for s in db.segments.loc[mask, "shot"]}
        return all(shot in allowed for shot in top)
    return None


def _outcome(prompt: EvalPrompt, db, keep: set[int] | None, n: int) -> PromptOutcome:
    from ..schema import QueryState

    resolved, expected_resolved = _resolution(prompt)
    state = QueryState(
        text=prompt.prompt,
        segment=prompt.expect_segment,
        constraints=dict(prompt.expect_constraints),
        exclude_shots=_exclusions(db, keep),
        n=n,
    )
    base = PromptOutcome(
        prompt_id=prompt.prompt_id,
        category=prompt.category,
        n_results=0,
        candidates=0,
        segment_rows=0,
        resolved=resolved,
        expected_resolved=expected_resolved,
    )
    try:
        report = rank_mod.search_report(state, db)
        found = rank_mod.search(state, db)
    except KeyError as e:  # a constraint on a column this database does not carry
        return base.model_copy(update={"error": f"unknown column: {e.args[0]}"})
    shots = [int(item.shot) for item in found.items]
    seen: set[int] = set()
    duplicates = 0
    for s in shots:
        if s in seen:
            duplicates += 1
        seen.add(s)
    return base.model_copy(
        update={
            "n_results": len(found.items),
            "candidates": int(report["candidates"]),
            "segment_rows": int(report["segment_rows"]),
            "channels": {name: len(r) for name, r in found.rankings.items()},
            "shots": shots,
            "run_ids": sorted({r.run_id for r in found.items if r.run_id}),
            "duplicate_shots": duplicates,
            "proxy_hit": _proxy_hit(prompt, shots, db) if prompt.hand_graded else None,
        }
    )


def run(
    db,
    evalset: Evalset | None = None,
    *,
    split: str = "eval",
    split_path: Path | str | None = None,
    n: int = TOP_K,
) -> EvalReport:
    """Run every prompt of the frozen evalset against `db`, on one side of the frozen split."""
    evalset = evalset or load_evalset()
    keep: set[int] | None = None
    if split != "all":
        keep = split_shots(split, split_path) & {int(s) for s in db.shots.index}
    outcomes = [_outcome(p, db, keep, n) for p in evalset.prompts]
    return _report(evalset, outcomes, db, split, keep, n)


def _report(evalset, outcomes, db, split, keep, n) -> EvalReport:
    ok = [o for o in outcomes if o.error is None]
    answered = [o for o in outcomes if o.n_results > 0]
    n_prompts = len(outcomes)
    n_results = sum(o.n_results for o in outcomes)
    survival = [
        o.candidates / o.segment_rows for o in ok if o.segment_rows
    ]
    participation = {
        name: (sum(1 for o in ok if o.channels.get(name, 0) > 0) / len(ok)) if ok else 0.0
        for name in ch_mod.CHANNELS
    }
    graded = [o for o in outcomes if o.proxy_hit is not None or _is_graded(evalset, o)]
    checkable = [o for o in graded if o.proxy_hit is not None]
    with_expectation = [o for o in outcomes if o.expected_resolved is not None]
    return EvalReport(
        evalset=evalset.path.name,
        evalset_sha256=evalset.sha256,
        split=split,
        split_shots=len(keep) if keep is not None else len(db.shots),
        n_prompts=n_prompts,
        n_results_total=n_results,
        coverage=_frac(sum(1 for o in outcomes if o.n_results > 0), n_prompts),
        hard_filter_survival=(sum(survival) / len(survival)) if survival else 0.0,
        channel_participation=participation,
        resolution_overall=(
            _frac(sum(1 for o in with_expectation if o.expected_resolved), len(with_expectation))
            if with_expectation
            else None
        ),
        categories=_categories(outcomes),
        mean_run_diversity=_mean_diversity(answered),
        duplicate_rate=_frac(sum(o.duplicate_shots for o in outcomes), n_results),
        proxy=ProxyGrade(
            n_prompts=len(graded),
            n_checkable=len(checkable),
            n_hit=sum(1 for o in checkable if o.proxy_hit),
            fraction=(
                _frac(sum(1 for o in checkable if o.proxy_hit), len(checkable))
                if checkable
                else None
            ),
        ),
        outcomes=outcomes,
        db_dir=str(getattr(db, "db_dir", "")),
        db_manifest_sha=str(getattr(db, "manifest", {}).get("config_sha") or "") or None,
        n_errors=sum(1 for o in outcomes if o.error),
        generated_at=dt.datetime.now(dt.UTC),
    )


def _is_graded(evalset: Evalset, outcome: PromptOutcome) -> bool:
    return any(p.prompt_id == outcome.prompt_id and p.hand_graded for p in evalset.prompts)


def _frac(num: int, den: int) -> float:
    return (num / den) if den else 0.0


def _mean_diversity(answered: list[PromptOutcome]) -> float | None:
    """Distinct run days per answered top-k, or None when nothing answered.

    None rather than 0.0: "no prompt returned anything" and "every prompt returned ten results
    from one run day" are opposite findings and 0.0 cannot tell them apart.
    """
    if not answered:
        return None
    return sum(len(o.run_ids) for o in answered) / len(answered)


def _categories(outcomes: list[PromptOutcome]) -> list[CategoryStats]:
    by: dict[str, list[PromptOutcome]] = {}
    for o in outcomes:
        by.setdefault(o.category, []).append(o)
    out = []
    for name in sorted(by):
        rows = by[name]
        expecting = [o for o in rows if o.expected_resolved is not None]
        out.append(
            CategoryStats(
                category=name,
                n_prompts=len(rows),
                coverage=_frac(sum(1 for o in rows if o.n_results > 0), len(rows)),
                n_with_expectation=len(expecting),
                resolution=(
                    _frac(sum(1 for o in expecting if o.expected_resolved), len(expecting))
                    if expecting
                    else None
                ),
                mean_run_diversity=_mean_diversity([o for o in rows if o.n_results > 0]),
            )
        )
    return out


def failed_bars(report: EvalReport) -> list[str]:
    """Which of the plan's bars this report misses, named.

    Both bars, not just the first. `markdown` has always printed `FAIL` beside a barred category
    whose resolution is under 80 %, but the exit code read only `coverage` -- so a CI job gating
    on `shot_design eval prompts` would have reported "eval passed" on a run whose `fast_ions`
    resolution was 0 %, as long as coverage held. The two are computed here, once, and both the
    exit code and any other caller read this.

    A barred category whose `resolution is None` -- no prompt of it stated an expectation -- is
    NOT a failure. That is a gap in the evalset, not a miss by the lexicon, and scoring it as one
    would make an empty category indistinguishable from a wrong one.
    """
    missed = []
    if report.coverage < COVERAGE_BAR:
        missed.append(f"coverage {report.coverage:.3f} < {COVERAGE_BAR}")
    for c in report.categories:
        barred = c.category in RESOLUTION_BAR_CATEGORIES and c.resolution is not None
        if barred and c.resolution < RESOLUTION_BAR:
            missed.append(f"{c.category} resolution {c.resolution:.3f} < {RESOLUTION_BAR}")
    return missed


def bars_met(report: EvalReport) -> bool:
    return not failed_bars(report)


# ------------------------------------------------------------------------------- rendering


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100.0 * x:.1f} %"


def _num(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def markdown(report: EvalReport) -> str:
    """The report as a table a human reads, with the two plan bars marked and nothing else
    judged. `split` is stated in the first line because a number from `all` is not comparable
    with one from `eval` and nothing downstream can tell them apart otherwise."""
    lines = [
        (
            f"### `{report.evalset}` on split `{report.split}` "
            f"({report.split_shots} shots, {report.n_prompts} prompts)"
        ),
        "",
        f"evalset sha256 `{report.evalset_sha256}` — frozen; database `{report.db_dir}`",
        "",
        "| metric | value | bar |",
        "| --- | --- | --- |",
        (
            f"| coverage (prompts with >= 1 result) | {_pct(report.coverage)} | "
            f">= {_pct(COVERAGE_BAR)} "
            f"{'PASS' if report.coverage >= COVERAGE_BAR else 'FAIL'} |"
        ),
        f"| hard-filter survival (mean) | {_pct(report.hard_filter_survival)} | — |",
        (
            f"| phenomenon resolution (all prompts with an expectation) | "
            f"{_pct(report.resolution_overall)} | — |"
        ),
        (
            f"| mean distinct run days per answered top-{TOP_K} | "
            f"{_num(report.mean_run_diversity)} | — |"
        ),
        f"| duplicate rate (repeated shots / results) | {_pct(report.duplicate_rate)} | — |",
        f"| prompts that raised an error | {report.n_errors} | — |",
        "",
        "| channel | prompts it contributed to |",
        "| --- | --- |",
    ]
    for name, frac in sorted(report.channel_participation.items()):
        lines.append(f"| `{name}` | {_pct(frac)} |")
    lines += [
        "",
        "| category | prompts | coverage | with expectation | resolution | run days / top-10 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for c in report.categories:
        bar = ""
        if c.category in RESOLUTION_BAR_CATEGORIES and c.resolution is not None:
            bar = " PASS" if c.resolution >= RESOLUTION_BAR else " FAIL"
        lines.append(
            f"| `{c.category}` | {c.n_prompts} | {_pct(c.coverage)} | {c.n_with_expectation} | "
            f"{_pct(c.resolution)}{bar} | {_num(c.mean_run_diversity)} |"
        )
    lines += [
        "",
        (
            "`resolution` is computed from the prompt text and the lexicon alone — no database, "
            "no split — so it is **identical on every split**, unlike every other column here."
        ),
    ]
    p = report.proxy
    lines += [
        "",
        (
            f"**Proxy grade** {p.n_hit}/{p.n_checkable} checkable of {p.n_prompts} "
            f"hand-graded ({_pct(p.fraction)}). {p.caveat}"
        ),
    ]
    errors = Counter(o.error for o in report.outcomes if o.error)
    if errors:
        lines += ["", "Errors:"] + [f"- {n}× {msg}" for msg, n in errors.most_common()]
    return "\n".join(lines)
