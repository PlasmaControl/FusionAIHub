"""What the operators wrote, as evidence - and only ever as weak evidence.

DIII-D's logbook is the only place some phenomena are ever named: nobody
runs a detector for a fishbone, but somebody typed "fishbones on this one"
at 3 pm. `lexicon.py` is the list of what they call each thing and the
matcher over it; this module is where the text comes FROM, and the two
outputs it feeds - `weak_labels` (a frame over many shots) and
`text_events` (rows against one).

**Text is never a label by itself** (plan 2). A lexicon hit is a claim that
a human wrote a word, not that the phenomenon happened: the same sentence
may be a plan for the next shot, a complaint about a diagnostic, or a note
about a different shot entirely. So a hit's confidence is capped at
`TEXT_ONLY_CEILING` for a single mention and reaches 1.0 only after four,
and its place in the pipeline is to PRIORITISE - which chunks a human is
asked to annotate, which shots shot_design ranks - never to decide.

TWO SOURCES, and which one a hit came from is the whole of what it means:

* `sql/logs.jsonl` - one JSON record per shot, whose `log_text` is that
  shot's own logbook entries, each headed `### [ROLE] user timestamp` on
  its own line. This is SHOT scope: somebody wrote it about this shot. It
  is 616 MB and 53,179 records, so it is read once into a subset of the
  shots in hand (`build_logs_subset`) and never scanned again.
* the per-shot bundle (`shotsummary/.../shot_<N>.txt`) - whose text before
  the marker line is the RUN's session context, shared by every shot of the
  session, and whose block after the marker is, in this corpus, the
  `SHOT TABLE ROW` and nothing else. A sawtooth named in the session text
  is evidence about the run: attaching it to this shot attaches it to the
  twenty others in the same session too (Appendix C item 12). So the bundle
  is `scope="run"` text and the SPAN (`PULSE-LENGTH`), and nothing else.

Everything above the outputs is reached through three accessors, and
`weak_labels` and `text_events` use no others:

* `shot_entries` - THE shot-scope seam, and the one to swap. `shot_prose`
  is that accessor flattened to a string and not a second source, so a
  swap that changed only `shot_prose` would move the hits and leave
  `n_entries` and `roles` reading the old source.
* `run_context` - the run-scope seam (the bundle's session text).
* `shot_span_s` - the SPAN, which has its own source on purpose.

That is not tidiness. The shot-scope source WAS the bundle until the L7
review, and making the swap found that `text_events` had been taking its
span from the same block as its text - which the new source has none of,
so every event would have silently collapsed to a point at zero with no
test failing. Named accessors are what make the next swap one function.

The matching is `lexicon.py`'s and takes text, not a shot: nothing in this
module decides what a phrase means.
"""
from __future__ import annotations

import copy
import functools
import json
import math
import os
import re
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import Paths
from .lexicon import TEXT_ONLY_CEILING, Hit, Lexicon, hits
from .schema import Event

#: The line that separates a run's session text from this shot's own.
MARKER = "## Shot-specific context (from summary.html)"

#: The header of the `- KEY: value` block at the end of a shot's block.
TABLE_HEADER = "SHOT TABLE ROW"

#: Where a hit's text came from. `run` is the bundle's session text, which
#: every shot of the run shares; `shot` is this shot's own logbook entries.
SCOPES = ("shot", "run")

#: What `text_events` records as the source of its text, so a row can be
#: told from one written before the source was swapped.
TEXT_SOURCE = "logs.jsonl"

#: How much of the first positive sentence a row carries.
SNIPPET_CHARS = 200

#: Column order of a `weak_labels` frame.
COLUMNS = ("shot", "phenomenon", "n_pos", "n_neg", "scope", "snippet")

#: Dtype of every column; `object` is pandas' dtype for python strings.
DTYPES = {
    "shot": "int32",
    "phenomenon": "object",
    "n_pos": "int32",
    "n_neg": "int32",
    "scope": "object",
    "snippet": "object",
}

#: Every line of `logs.jsonl` begins exactly like this. Selecting lines on
#: the raw BYTES is what makes one pass over 616 MB affordable: parsing all
#: 53,179 records to keep 500 of them is a minute of JSON, matching a
#: prefix is seconds of memchr.
_SHOT_PREFIX = re.compile(rb'^\{"shot":\s*(\d+)')

#: The head of one logbook entry inside `log_text`: `### [PHYSICS_OPERATOR]
#: someone 2024-05-17 13:12:07`, alone on its line.
_ENTRY_HEADER = re.compile(
    r"^###\s*\[(?P<role>[A-Za-z_]+)\]\s+(?P<author>\S+)\s+"
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*$",
    re.MULTILINE,
)

#: The five HTML tags that turn up inside `log_text` - the mini-proposal
#: link and some hand formatting. Named one by one on purpose: `<[^>]*>`
#: would also eat `<ne>=4.2e13`, which is a measurement, and the sentences
#: with measurements in them are the ones worth reading.
_HTML_TAG = re.compile(r"</?(?:a|b|pre|font|del)(?:\s[^>]*)?>", re.IGNORECASE)

#: Roles whose entries are a machine dump rather than something a person
#: wrote. `[PCS]` is one per shot - `PCS CHANGES: ... set vertices ...` -
#: and it is where `RWM_GAINMULT` and every other identifier that reads
#: like a phenomenon lives.
MACHINE_ROLES = ("PCS",)


@dataclass(frozen=True)
class LogEntry:
    """One logbook entry: who wrote it, when, and what it says."""

    role: str
    author: str
    time: str
    text: str


def _paths(paths: Paths | None) -> Paths:
    return Paths.from_env() if paths is None else paths


# ------------------------------------------------- the shot's own logbook

def _read_missing(paths: Paths) -> set[int]:
    """The shots `logs_jsonl` was searched for and did not have.

    A plain list of integers, one per line, because that is all it is and a
    person deleting a stale line should not need a parser. A file that is
    not there is an empty set; an unreadable one is treated the same way,
    since forgetting a miss costs one pass and refusing to build costs the
    whole run.
    """
    try:
        text = paths.logs_subset_missing.read_text(encoding="utf-8")
    except OSError:
        return set()
    out = set()
    for line in text.split():
        try:
            out.add(int(line))
        except ValueError:
            continue
    return out


def _write_missing(paths: Paths, shots: set[int]) -> None:
    """Replace the missing-shot sidecar, atomically."""
    out = paths.logs_subset_missing
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(
            "".join(f"{s}\n" for s in sorted(shots)), encoding="utf-8"
        )
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)


def build_logs_subset(shots: Iterable[int], *,
                      paths: Paths | None = None,
                      refresh_missing: bool = False) -> int:
    """Copy the records for `shots` out of `logs_jsonl` into the subset.

    One pass over the 616 MB source, selecting lines by their byte prefix,
    and only for shots the subset does not already hold AND that a previous
    pass did not already fail to find - so a second call for the same shots
    reads nothing and returns 0 whether or not the logbook has a record of
    them. Returns how many records were added.

    REMEMBERING THE MISSES is the whole of `logs_subset_missing`. A shot the
    logbook has no record of can never enter the subset, so before the
    sidecar it stayed wanted forever and every later call re-streamed the
    source end to end: `text_events` builds for one shot at a time, so a
    500-shot loop with 20 record-less shots did 12 GB of reads for nothing,
    silently. `refresh_missing=True` is the one call that pays a pass to ask
    again - the logbook gains records for today's shots, and a miss is a
    fact about the source at the time and not forever. A shot found on such
    a pass is dropped from the sidecar.

    The cache is rewritten WHOLE - the existing complete lines, then the
    new records - into a sibling `.tmp` that is fsynced and renamed over
    the old one, so a reader never sees a half-written file and a build
    killed at any point leaves either the old cache or the new one. The
    `.tmp` carries the pid, because several SLURM array tasks sharing a
    `LABELMAKER_ROOT` build this cache at once and a fixed name lets them
    interleave their writes into one file. (This hand-rolls `atomic_path`
    rather than using it because it needs the fsync, which `atomic_path`
    does not do. Appending in place is what produces the torn trailing line
    `_subset_records` tolerates; a line of the old file that lacks its
    newline is such a tail, is not carried over, and its shot is fetched
    again.) Mirrors `shot_design.shotdb.text.build_logs_subset`, which is where
    the pattern and the failure it fixes were measured.

    A build that finds NOTHING does not touch the subset at all: rewriting
    it would churn the file and invalidate the parsed-subset memo, which is
    keyed on its size and mtime, to say the same thing it already said. The
    one exception is a TORN line: a pass that had to drop one rewrites the
    file whether or not it found anything, because the alternative is a
    repair that waits for an unrelated successful fetch while every fresh
    parse of that version of the file re-emits the warning.

    The misses are pruned against the subset as well as against this call's
    `found`, so a shot that reached the cache by some other route - a
    refresh in another process, a hand-restored file - does not stay listed
    as missing over a fact that is no longer true. The blunt recovery is the
    sidecar itself: DELETING `logs_subset.missing` forgets every recorded
    miss, and the next build re-asks the source about all of them.

    A missing or unreadable `logs_jsonl` RAISES rather than reading as no
    text. A shot the logbook has no record of is ordinary and answers
    `None`; a source that is not there is a misconfigured `Paths`, and the
    failure it would otherwise make - every shot silently textless - is the
    one this module has been burned by once.
    """
    paths = _paths(paths)
    asked = {int(s) for s in shots}
    recorded_missing = _read_missing(paths)
    have = set(_read_subset(paths))
    wanted = asked - have
    if not refresh_missing:
        wanted -= recorded_missing
    if not wanted:
        return 0
    out = paths.logs_subset
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
    found: set[int] = set()
    n = 0
    torn = False
    try:
        with open(tmp, "wb") as dst:
            if out.exists():
                with open(out, "rb") as old:
                    for line in old:
                        if line.endswith(b"\n"):
                            dst.write(line)
                        elif line.strip():
                            # The tail of a build killed mid-write. Dropped
                            # here, and remembered so the rename happens
                            # even if this pass finds nothing to add.
                            torn = True
            with open(paths.logs_jsonl, "rb") as src:
                for line in src:
                    m = _SHOT_PREFIX.match(line)
                    if m and int(m.group(1)) in wanted:
                        dst.write(line.rstrip(b"\n") + b"\n")
                        found.add(int(m.group(1)))
                        n += 1
            dst.flush()
            os.fsync(dst.fileno())
        if n or torn:
            os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    # `have | found` and not `found` alone: a shot already in the subset is
    # not a miss, whoever put it there.
    now_missing = (recorded_missing | (wanted - found)) - found - have
    if now_missing != recorded_missing:
        _write_missing(paths, now_missing)
    return n


def _stat_key(path: Path) -> tuple[int, int]:
    st = path.stat()
    return (st.st_size, st.st_mtime_ns)


@functools.lru_cache(maxsize=2)
def _subset_records(path_str: str, key: tuple[int, int]) -> dict[int, dict]:
    """The parsed subset, memoised on `(path, size, mtime_ns)`.

    Parsed once per VERSION of the file rather than once per shot: the
    access pattern is build-then-read-every-shot, and re-reading a 7 MB
    cache to answer each of 500 questions is 20 ms a shot of nothing. The
    key is what keeps an append-then-read correct anyway.

    An undecodable LAST line is dropped with a warning: that is what a
    build killed mid-write leaves behind. A bad line ANYWHERE else - two
    array tasks that raced on the same cache, a person who hand-edited it,
    a record with no `shot` - is skipped, and all of them together are one
    warning naming the first. Raising was the old policy and it was the
    wrong one: this is a cache, it can be rebuilt from the source, and one
    damaged line must not make every later read fail until somebody deletes
    the file by hand. The shots those lines carried are simply fetched
    again by the next build.

    The mapping is shared between callers - `load_log_record` copies out of
    it, and nothing else may hand it out.
    """
    out: dict[int, dict] = {}
    lines = [ln for ln in Path(path_str).read_bytes().split(b"\n") if ln.strip()]
    skipped: list[int] = []
    for i, line in enumerate(lines):
        try:
            rec = json.loads(line)
            shot = int(rec["shot"])
        except (json.JSONDecodeError, UnicodeDecodeError):
            if i == len(lines) - 1:
                warnings.warn(
                    f"{path_str}: dropping an undecodable trailing line "
                    f"({len(line)} bytes; a build was killed mid-write) - "
                    "its shot is fetched again by the next build",
                    stacklevel=2,
                )
                break
            skipped.append(i + 1)
        except (KeyError, TypeError, ValueError):
            skipped.append(i + 1)
        else:
            out[shot] = rec
    if skipped:
        warnings.warn(
            f"{path_str}: skipping {len(skipped)} unusable line(s), the "
            f"first on line {skipped[0]} - a record is one JSON object with "
            "an integer `shot`. The shots they carried are fetched again by "
            "the next build",
            stacklevel=2,
        )
    return out


def _read_subset(paths: Paths) -> dict[int, dict]:
    path = paths.logs_subset
    if not path.exists():
        return {}
    return _subset_records(str(path), _stat_key(path))


def load_log_record(shot, *, paths: Paths | None = None) -> dict | None:
    """One shot's logbook record from the SUBSET, or `None`.

    Never the 616 MB source: a lookup that could fall back to scanning it
    would do so once per shot, and the 500-shot loop that looks fine in a
    test would take an hour on the real corpus. `build_logs_subset` is the
    only thing that reads the source; a shot nobody built is `None`.

    A copy, because the parsed subset is memoised and shared and two of the
    record's fields (`topics`, `keywords`) are lists.
    """
    rec = _read_subset(_paths(paths)).get(int(shot))
    return None if rec is None else copy.deepcopy(rec)


def log_entries(log_text: str) -> list[LogEntry]:
    """`log_text` split into the entries its headers announce.

    Text before the first header - there usually is none - is dropped
    rather than guessed at: an entry with no header has no author and no
    role, and who wrote a sentence is half of what this module claims.
    """
    if not log_text:
        return []
    heads = list(_ENTRY_HEADER.finditer(log_text))
    out = []
    for i, head in enumerate(heads):
        stop = heads[i + 1].start() if i + 1 < len(heads) else len(log_text)
        out.append(
            LogEntry(
                role=head.group("role"),
                author=head.group("author"),
                time=head.group("ts"),
                text=_HTML_TAG.sub("", log_text[head.end():stop]).strip(),
            )
        )
    return out


# ----------------------------------------------------- the four accessors

def shot_entries(shot, *, paths: Paths | None = None,
                 exclude_roles: tuple[str, ...] = MACHINE_ROLES,
                 ) -> tuple[LogEntry, ...]:
    """This shot's logbook entries, the machine dump left out.

    THE shot-scope text accessor in its structured form - `shot_prose` is
    the same thing flattened, and nothing else in this module reads a
    shot's own text. `()` for a shot with no record, which is ordinary.
    """
    rec = load_log_record(shot, paths=paths)
    if rec is None:
        return ()
    skip = {r.upper() for r in exclude_roles}
    return tuple(
        e for e in log_entries(rec.get("log_text") or "")
        if e.role.upper() not in skip
    )


def shot_prose(shot, *, paths: Paths | None = None,
               exclude_roles: tuple[str, ...] = MACHINE_ROLES) -> str:
    """What people wrote about this shot: the `scope="shot"` text.

    The entry bodies, newline-joined - which is also what keeps a phrase
    from spanning two ENTRIES, since `sentences` breaks on newlines: a
    phrase made of the end of one person's note and the start of another's
    is a sentence nobody wrote.

    `MACHINE_ROLES` are excluded by default, and by default only: a caller
    who wants the PCS dump asks for it with `exclude_roles=()`.
    """
    return "\n".join(
        e.text for e in shot_entries(shot, paths=paths,
                                     exclude_roles=exclude_roles)
    )


def run_context(shot, *, paths: Paths | None = None) -> str:
    """The session text of the run this shot belongs to: `scope="run"`.

    The bundle's text BEFORE the marker. Shared by every shot of the
    session, so what it names is a property of the run and never of this
    shot alone (Appendix C item 12).
    """
    return _split(_read_bundle(shot, paths))[0]


def shot_span_s(shot, *, paths: Paths | None = None) -> tuple[float, float]:
    """`[0, PULSE-LENGTH)` for this shot, or `(0.0, 0.0)`.

    The span has its OWN source, deliberately: it comes from the bundle's
    `SHOT TABLE ROW`, the only place the corpus states a shot's length,
    while the text comes from the logbook, which has no such row. They were
    one call until the L7 review, and swapping the text source under that
    would have collapsed every text event to a point at zero with nothing
    failing.
    """
    return (0.0, _pulse_length_s(shot_table_row(shot_block(shot, paths=paths))))


# ------------------------------------------------------ reading a bundle

def _read_bundle(shot, paths: Paths | None) -> str:
    """One shot's whole text bundle, or "" where there is no file.

    Missing is ordinary: the corpus has bundles for about 13,000 of the
    17,000 shots, and a caller joining text to features must not have to
    care. UTF-8 with `replace`, because these are scraped from HTML and a
    stray byte in one bundle may not stop a 500-shot run.
    """
    path = _paths(paths).text_file(int(shot))
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _split(text: str) -> tuple[str, str]:
    """`(run context, shot block)` either side of `MARKER`.

    No marker means no shot block - not "the whole file is the shot's" -
    because the run's text is the thing that is always there. Both sides
    are stripped, so that a bundle which ENDS at the marker - the
    session-fallback case, where the summary carried nothing for this shot
    - reads as an empty block rather than as a newline.
    """
    head, sep, tail = text.partition(MARKER)
    return (head.strip(), tail.strip()) if sep else (text.strip(), "")


def shot_block(shot, *, paths: Paths | None = None) -> str:
    """The bundle's block after the marker: the TABLE ROW, in this corpus.

    A source accessor for `shot_span_s` and nothing more. It is NOT the
    shot's prose: measured over a 400-bundle sample, the largest block
    after the marker is 379 bytes and none of them carries a sentence -
    they are `SHOT: N`, `- BTOR: ...`, `- PULSE-LENGTH: ...`. What people
    wrote about the shot is in the logbook, and `shot_prose` is that.
    """
    return _split(_read_bundle(shot, paths))[1]


def shot_table_row(text: str) -> dict[str, str]:
    """The `- KEY: value` block under `SHOT TABLE ROW`, as a mapping.

    Values are left as strings, and split on the FIRST colon: a
    `TIME-OF-SHOT` of `10:08` is one value and not a second key. `{}` where
    the block is missing, which is the session-fallback case and the
    "(Shot table key/value mapping not found.)" one.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(TABLE_HEADER):
            break
    else:
        return {}
    out: dict[str, str] = {}
    for line in lines[i + 1:]:
        if not line.strip():
            continue
        if not line.startswith("- "):
            break
        key, sep, value = line[2:].partition(":")
        if sep:
            out[key.strip()] = value.strip()
    return out


def _pulse_length_s(table: Mapping[str, str]) -> float:
    """`PULSE-LENGTH` in seconds, or 0.0 where the table does not say.

    The table row is the only shot duration the text corpus carries, and it
    is a string somebody's HTML scraper produced: "(none)" and "" both
    happen. A text claim is about the whole shot, so a shot whose length is
    unknown gets the point event at 0 - `Event` requires finite times, and
    inventing a span would be inventing a claim about when.
    """
    try:
        value = float(table.get("PULSE-LENGTH", ""))
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) and value > 0.0 else 0.0


# -------------------------------------------------------------- the outputs

def _empty_frame() -> pd.DataFrame:
    """No hits, right columns, right dtypes."""
    return pd.DataFrame({name: pd.Series(dtype=DTYPES[name]) for name in COLUMNS})


def _counts(found: Mapping[str, list[Hit]]) -> list[tuple[str, int, int, str]]:
    """`(phenomenon, n_pos, n_neg, snippet)` per phenomenon, id order."""
    rows = []
    for pid in sorted(found):
        hs = found[pid]
        pos = [h for h in hs if h.polarity == "pos"]
        snippet = pos[0].sentence[:SNIPPET_CHARS] if pos else ""
        rows.append((pid, len(pos), len(hs) - len(pos), snippet))
    return rows


def weak_labels(shots: Iterable[int], lexicon: Lexicon, *,
                paths: Paths | None = None,
                scope: str = "shot") -> pd.DataFrame:
    """One row per (shot, phenomenon) named in the shots' text.

    `scope="shot"` reads each shot's own logbook entries; `scope="run"`
    reads the session text it shares with the rest of its run, which is
    evidence about the RUN and is labelled as such - a consumer that treats
    a run row as a shot row has claimed the same sentence for every shot of
    the session.

    The subset is built for `shots` at SHOT scope only, in one pass for the
    whole list, and a `Paths` pointing at a logbook nobody can read says so
    there rather than by handing back an empty frame. Run scope reads the
    bundles and nothing else, so it neither pays for the logbook nor needs
    to be able to see it - which is also what makes run scope usable on a
    machine that has the bundles and not the 616 MB dump.

    No row at all for a phenomenon nobody mentioned: an absent row is "not
    mentioned", which is not the same claim as `n_pos == 0`.
    """
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}; got {scope!r}")
    paths = _paths(paths)
    shots = [int(s) for s in shots]
    if scope == "shot":
        build_logs_subset(shots, paths=paths)
    rows = []
    for shot in shots:
        text = (shot_prose(shot, paths=paths) if scope == "shot"
                else run_context(shot, paths=paths))
        for phenomenon, n_pos, n_neg, snippet in _counts(hits(text, lexicon)):
            rows.append(
                {
                    "shot": shot,
                    "phenomenon": phenomenon,
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "scope": scope,
                    "snippet": snippet,
                }
            )
    if not rows:
        return _empty_frame()
    return (
        pd.DataFrame(rows, columns=list(COLUMNS))
        .sort_values(["shot", "phenomenon"], kind="stable")
        .reset_index(drop=True)
        .astype(DTYPES)
    )


def text_events(shot, lexicon: Lexicon, *,
                paths: Paths | None = None) -> list[Event]:
    """One `evidence_kind="text"` event per phenomenon this shot's log names.

    The text is `shot_entries` - what people wrote about THIS shot - and
    the span is `shot_span_s`, which reads the bundle's table row: two
    sources and two accessors, because the logbook carries no shot length
    and the bundle carries no prose. `[0, PULSE-LENGTH)`, or a point at 0 where the
    length is not stated; coverage is the same span, so "no `eho` row"
    means nobody wrote it and not that nobody looked.

    The session text is NOT read here. It is the run's (Appendix C item
    12); `weak_labels(..., scope="run")` is where it goes, and a consumer
    that wants to fold it in has to decide what a run-scope claim is worth
    for one of its twenty shots.

    `attrs` says where the row came from: `n_entries` (how many entries
    were read), `roles` (the roles whose own entries carry a positive hit
    for this phenomenon - one operator saying "fishbones" and the chief
    operator saying it are not one claim twice), and `text_source`.

    ONE pass over the entries gives the counts, the snippet and the roles
    together. Matching per entry for the roles and again over the joined
    prose for the counts gives the same answer - `shot_prose` joins entry
    bodies with a newline and `sentences` splits on newlines, so the prose
    hits ARE the union of the per-entry hits, in order - but it is two
    passes and two places that have to go on agreeing.
    """
    paths = _paths(paths)
    build_logs_subset([shot], paths=paths)
    t0_s, t1_s = shot_span_s(shot, paths=paths)
    entries = shot_entries(shot, paths=paths)
    found: dict[str, list[Hit]] = {}
    roles: dict[str, set[str]] = {}
    for entry in entries:
        for pid, hs in hits(entry.text, lexicon).items():
            found.setdefault(pid, []).extend(hs)
            if any(h.polarity == "pos" for h in hs):
                roles.setdefault(pid, set()).add(entry.role)
    out = []
    for phenomenon, n_pos, n_neg, snippet in _counts(found):
        if n_pos < 1:
            continue
        out.append(
            Event(
                shot=int(shot),
                source="text",
                evidence_kind="text",
                phenomenon=phenomenon,
                t0_s=t0_s,
                t1_s=t1_s,
                confidence=min(1.0, TEXT_ONLY_CEILING * n_pos),
                attrs={
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "snippet": snippet,
                    "scope": "shot",
                    "n_entries": len(entries),
                    "roles": sorted(roles.get(phenomenon, ())),
                    "text_source": TEXT_SOURCE,
                },
                t_cov0_s=t0_s,
                t_cov1_s=t1_s,
            )
        )
    return out
