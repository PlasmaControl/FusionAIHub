"""OSTI: probe hits, full texts at three requests a minute, and the links they give.

The probe (`$LABELER_ROOT/literature/osti/osti_probe.py`) asked OSTI's full-text
search for `"<shot>" AND "DIII-D"`, one corpus shot at a time. Each line of its
output is `{"shot", "n", "records": [{osti_id, doi, title, date, journal,
type}]}`, or `{"shot", "error"}`.

- `fetch` downloads the full text of each record not dated before `FIRST_YEAR`
  (no earlier paper can name a shot in range) from its OSTI purl into
  `<cache>/fulltext/<osti_id>.pdf`, one request at a time and at least
  `MIN_INTERVAL_S` apart, and extracts the text with pypdf into `<osti_id>.txt`
  (pages separated by form feeds). Every attempt is a line of
  `<cache>/fulltext/fetched.jsonl`; a rerun skips records whose last attempt
  ended in a `FINAL` status.
- `links` scans every extracted text for every corpus shot in the catalog's
  range and writes `<cache>/links.csv` (each probe hit and each mention, with
  its verdict), `<cache>/papers.csv` (links verified by the context rule, not
  every paper that names a shot, in the release schema), and
  `<cache>/papers.meta.json` (the build inputs, rules, outputs and coverage).
  A text with no printable character (a scanned PDF) counts as none, and a
  paper dated before its shot's year cannot name it: that pair is `predates`.
  The shots' years come from the population's `pool.csv` (`year_starts`).

    pixi run -e labelmaker python -m labeler.literature.osti fetch
    pixi run -e labelmaker python -m labeler.literature.osti links

Copying `papers.csv` and `papers.meta.json` into `data/events/catalog/` is the
owner's call.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import platform
import re
import shlex
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from ..catalog import corpus_shots
from ..config import Paths, atomic_path, git_dirty, git_sha
from ..events.catalog.check import CatalogError
from ..events.catalog.population import FIRST_SHOT, LAST_SHOT, read_pool
from . import context
from .papers import Record, links, papers_frame

FIRST_YEAR = 2021  # the year of FIRST_SHOT, the corpus' first campaign
PURL = "https://www.osti.gov/servlets/purl/{}"
USER_AGENT = "FusionAIHub-literature/0.1 (research; one request at a time)"
# OSTI's full-text servlet (measured 2026-09-26): three requests/minute/host;
# a fourth is dropped unanswered. 21 s keeps any minute to three starts.
MIN_INTERVAL_S = 21.0
TIMEOUT_S = 120
FINAL = ("ok", "not_pdf", "missing", "unreadable")
LINK_COLUMNS = (
    "shot",
    "record_id",
    "in_probe",
    "verdict",
    "paper_year",
    "shot_year",
    "match_type",
    "context",
)


@dataclass(frozen=True)
class Hits:
    """What the probe found for the shots in range."""

    records: dict[str, Record]  # osti_id -> the paper
    by_shot: dict[int, tuple[str, ...]]  # shot -> the osti_ids its query returned
    truncated: tuple[int, ...]  # shots whose query matched more than it returned
    failed: tuple[int, ...]  # shots whose query never answered
    truncated_queries: tuple[dict[str, int], ...] = ()  # last truncation per shot


def _record(raw: dict) -> Record:
    date = raw.get("date") or ""
    return Record(
        source="osti",
        record_id=str(raw["osti_id"]),
        doi=raw.get("doi") or "",
        title=" ".join((raw.get("title") or "").split()),
        year=int(date[:4]) if date[:4].isdigit() else None,
        venue=raw.get("journal") or raw.get("type") or "",
    )


def read_hits(
    paths: Iterable[Path], first: int = FIRST_SHOT, last: int = LAST_SHOT
) -> Hits:
    records, by_shot, truncated, failed = {}, {}, [], []
    truncated_queries = {}
    for path in paths:
        data = path.read() if hasattr(path, "read") else Path(path).read_bytes()
        for line in data.splitlines():
            row = json.loads(line)
            shot = int(row["shot"])
            if not first <= shot <= last:
                continue
            if "error" in row:
                failed.append(shot)
                continue
            found = [_record(r) for r in row.get("records", [])]
            records.update((r.record_id, r) for r in found)
            by_shot[shot] = tuple(r.record_id for r in found)
            if row.get("n", 0) > len(found):
                truncated.append(shot)
                truncated_queries[shot] = {
                    "shot": shot,
                    "n": int(row["n"]),
                    "kept": len(found),
                }
    answered = set(by_shot)
    return Hits(
        records,
        by_shot,
        tuple(sorted(truncated)),
        tuple(sorted(set(failed) - answered)),
        tuple(truncated_queries[s] for s in sorted(truncated_queries)),
    )


def to_fetch(hits: Hits, first_year: int = FIRST_YEAR) -> list[str]:
    """The records worth a full text: those not dated before `first_year`.

    An undated record is fetched: nothing shows that it predates the corpus.
    """
    kept = (r for r in hits.records.values() if r.year is None or r.year >= first_year)
    return [r.record_id for r in kept]


@dataclass(frozen=True)
class Response:
    status: int | None  # HTTP status; None if no answer came
    content_type: str = ""
    body: bytes = b""
    url: str = ""
    error: str = ""


class Fetcher:
    """GETs one URL at a time, starting each at least `min_interval` s after the last.

    Transient failures (no answer, 429, 5xx) are retried with a doubling wait.
    """

    def __init__(
        self,
        *,
        opener: Callable = urllib.request.urlopen,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        min_interval: float = MIN_INTERVAL_S,
        tries: int = 3,
        backoff_s: float = 5.0,
    ):
        self.opener, self.sleep, self.clock = opener, sleep, clock
        self.min_interval, self.tries, self.backoff_s = min_interval, tries, backoff_s
        self._last: float | None = None

    def _wait(self) -> None:
        if self._last is not None:
            gap = self.min_interval - (self.clock() - self._last)
            if gap > 0:
                self.sleep(gap)
        self._last = self.clock()

    def get(self, url: str) -> Response:
        headers = {"User-Agent": USER_AGENT, "Accept": "application/pdf"}
        error = ""
        for attempt in range(self.tries):
            self._wait()
            try:
                request = urllib.request.Request(url, headers=headers)
                with self.opener(request, timeout=TIMEOUT_S) as r:
                    kind = r.headers.get("Content-Type", "")
                    return Response(r.status, kind, r.read(), r.geturl())
            except urllib.error.HTTPError as e:
                if e.code != 429 and e.code < 500:
                    return Response(e.code, url=url, error=str(e))
                error = str(e)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                error = str(e)
            if attempt < self.tries - 1:
                self.sleep(self.backoff_s * 2**attempt)
        return Response(None, url=url, error=error)


def extract_text(pdf: Path) -> tuple[str, int, int]:
    """(text, pages, pages that failed); raises if the file is not a readable PDF."""
    from pypdf import PdfReader  # only fetching needs pypdf

    reader = PdfReader(pdf)
    pages, failed = [], 0
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - one broken page must not lose the rest
            pages.append("")
            failed += 1
    return "\f".join(pages), len(pages), failed


def fetch_record(fetcher: Fetcher, osti_id: str, folder: Path) -> dict:
    """Fetch one record's full text into `folder`; return its log line."""
    response = fetcher.get(PURL.format(osti_id))
    line = {
        "osti_id": osti_id,
        "http_status": response.status,
        "content_type": response.content_type,
        "url": response.url,
        "n_bytes": len(response.body),
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "error": response.error,
    }
    if response.status != 200:
        status = "missing" if response.status in (404, 410) else "error"
        return {**line, "status": status}
    if not response.body.startswith(b"%PDF"):
        return {**line, "status": "not_pdf"}
    pdf = folder / f"{osti_id}.pdf"
    with atomic_path(pdf) as tmp:
        tmp.write_bytes(response.body)
    line["sha256"] = hashlib.sha256(response.body).hexdigest()
    try:
        text, n_pages, failed = extract_text(pdf)
    except Exception as exc:  # noqa: BLE001 - pypdf raises many kinds on a bad file
        return {**line, "status": "unreadable", "error": f"{type(exc).__name__}: {exc}"}
    with atomic_path(folder / f"{osti_id}.txt") as tmp:
        tmp.write_text(text)
    n_chars = len("".join(text.split()))  # 0 for a scanned or image-only PDF
    return {
        **line,
        "status": "ok",
        "n_pages": n_pages,
        "failed_pages": failed,
        "n_chars": n_chars,
    }


def fetch_log(folder) -> dict[str, dict]:
    """The last logged attempt per record."""
    if hasattr(folder, "read"):
        data = folder.read()
    else:
        log = folder / "fetched.jsonl"
        data = log.read_bytes() if log.is_file() else b""
    last = {}
    for line in data.splitlines():
        if line.strip():
            row = json.loads(line)
            last[row["osti_id"]] = row
    return last


def fetch_all(
    osti_ids: Iterable[str], folder: Path, fetcher: Fetcher, *, limit: int | None = None
) -> Counter:
    """Fetch every record not yet final; one log line per attempt."""
    folder.mkdir(parents=True, exist_ok=True)
    done = {k for k, row in fetch_log(folder).items() if row["status"] in FINAL}
    todo = [k for k in sorted(set(osti_ids), key=int) if k not in done][:limit]
    counts = Counter()
    with (folder / "fetched.jsonl").open("a") as log:
        for i, osti_id in enumerate(todo, 1):
            line = fetch_record(fetcher, osti_id, folder)
            log.write(json.dumps(line) + "\n")
            log.flush()
            counts[line["status"]] += 1
            if i % 50 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} {dict(counts)}", flush=True)
    return counts


def year_starts(pool: pd.DataFrame) -> dict[int, int]:
    """Each year's first shot, from the pool shots whose run id is their own.

    A shot's year is its run id's (`pool.csv`, from its text bundle). A bundle that
    fell back to its session's block (`session_fallback`) can hold another run's
    id, so only the other shots date the calendar. Shot numbers only grow, so their
    years must too: a shot dated before an earlier one is a `CatalogError`.
    """
    fallback = pool.reasons.str.contains("session_fallback", regex=False)
    own = pool[pool.year.notna() & ~fallback].sort_values("shot", ignore_index=True)
    years = own.year.astype("int64")
    back = own.index[years.diff() < 0]
    if len(back):
        i = back[0]
        raise CatalogError(
            f"shot {own.shot[i]} is dated {years[i]}, after shot {own.shot[i - 1]}"
            f" was dated {years[i - 1]}"
        )
    return {int(y): int(s) for y, s in own.groupby(years).shot.min().items()}


def shot_year(shot: int, starts: Mapping[int, int]) -> int | None:
    """The year `shot` was run, at the earliest: the last year starting at or before it.

    A shot between one year's last dated shot and the next year's first gets the
    earlier year, so no link is dropped for a year its shot may not have had.
    """
    years = [year for year, first in starts.items() if first <= shot]
    return max(years) if years else None


def build_links(
    hits: Hits,
    texts: Mapping[str, str],
    shots: Iterable[int],
    years: Mapping[int, int | None],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(audit, papers): every probe hit and mention with its verdict; the verified.

    A pair whose paper is dated before its shot's year (`years`) is `predates`: the
    paper cannot name that shot, whatever its text says. Otherwise a mention in
    context is `verified`, and a probe hit without one is `no_context`, or
    `no_text` if its record has no text. A mention the probe did not return (a
    range, or a shot OSTI's index missed) is verified all the same.
    """
    shots = frozenset(shots)
    mentioned = []
    for record_id in sorted(set(texts) & set(hits.records), key=int):
        mentioned += links(hits.records[record_id], texts[record_id], shots)
    probe = {(s, r) for s, ids in hits.by_shot.items() if s in shots for r in ids}
    found = {(row["shot"], row["record_id"]) for row in mentioned}

    def predates(shot: int, record_id: str) -> bool:
        paper, run = hits.records[record_id].year, years.get(shot)
        return paper is not None and run is not None and paper < run

    def row(shot, record_id, verdict, match_type="", context="") -> dict:
        return {
            "shot": shot,
            "record_id": record_id,
            "in_probe": (shot, record_id) in probe,
            "verdict": "predates" if predates(shot, record_id) else verdict,
            "paper_year": hits.records[record_id].year,
            "shot_year": years.get(shot),
            "match_type": match_type,
            "context": context,
        }

    audit = [
        row(m["shot"], m["record_id"], "verified", m["match_type"], m["context"])
        for m in mentioned
    ]
    audit += [
        row(s, r, "no_context" if r in texts else "no_text")
        for s, r in sorted(probe - found, key=lambda sr: (sr[0], int(sr[1])))
    ]
    audit = (
        pd.DataFrame(audit, columns=list(LINK_COLUMNS))
        .astype({"paper_year": "Int64", "shot_year": "Int64"})
        .sort_values(["shot", "record_id"], kind="stable", ignore_index=True)
    )
    verified = [m for m in mentioned if not predates(m["shot"], m["record_id"])]
    return audit, papers_frame(verified)


def _texts(folder: Path, log: Mapping[str, dict]) -> tuple[dict[str, str], list[str]]:
    """(texts, empty): each fetched record's text, and those with no printable text.

    Empty texts are retained here for hashing, then excluded from link verification.
    """
    texts, empty = {}, []
    for k, row in log.items():
        path = folder / f"{k}.txt"
        if row["status"] == "ok" and path.is_file():
            text = path.read_bytes().decode("utf-8")
            texts[k] = text
            if not text.strip():
                empty.append(k)
    return texts, empty


def _summary(
    hits: Hits,
    audit: pd.DataFrame,
    papers: pd.DataFrame,
    *,
    starts: Mapping[int, int],
    empty_texts: int,
) -> dict:
    probe = audit[audit.in_probe]
    exact = papers[papers.match_type == "exact"]
    verified = audit.verdict.eq("verified")
    return {
        "hit_shots": len([s for s, ids in hits.by_shot.items() if ids]),
        "hit_links": len(probe),
        "verdicts": probe.verdict.value_counts().to_dict(),
        "verified_shots": int(papers.shot.nunique()),
        "verified_shots_exact": int(exact.shot.nunique()),
        "verified_links_not_in_probe": int((~audit.in_probe & verified).sum()),
        "predates_links": int(audit.verdict.eq("predates").sum()),
        "empty_texts": empty_texts,
        "year_starts": dict(sorted(starts.items())),
        "truncated_shots": list(hits.truncated),
        "failed_queries": list(hits.failed),
    }


def _coverage(
    hits: Hits,
    log: Mapping[str, dict],
    texts: Mapping[str, str],
    empty: list[str],
    audit: pd.DataFrame,
) -> dict:
    wanted = to_fetch(hits)
    statuses = Counter(
        dict.fromkeys(("ok", "missing", "not_pdf", "error", "unattempted"), 0)
    )

    def status(record_id: str) -> str:
        return log.get(record_id, {}).get("status", "unattempted")

    statuses.update(status(k) for k in wanted)
    empty_ids = set(empty)
    verified = set(audit.loc[audit.verdict.eq("verified"), "shot"])
    missing = audit[audit.verdict.eq("no_text")]
    causes = missing.record_id.map(lambda k: "empty" if k in empty_ids else status(k))
    no_text = {}
    for cause, part in missing.groupby(causes):
        shots = set(part.shot)
        no_text[cause] = {
            "pairs": len(part),
            "shots": len(shots),
            "shots_without_verified_link": len(shots - verified),
        }
    no_context = audit[audit.verdict.eq("no_context")]
    normalised = {k: context.normalise(text) for k, text in texts.items()}
    present = []
    for row in no_context.itertuples():
        number = str(row.shot)
        pattern = rf"(?<!\d){number[:3]} ?{number[3:]}(?!\d)"
        present.append(bool(re.search(pattern, normalised[row.record_id])))

    def counts(part):
        return {"pairs": len(part), "shots": int(part.shot.nunique())}

    in_text = pd.Series(present, index=no_context.index, dtype=bool)
    return {
        "records": {
            "probe": len(hits.records),
            "dated_before_first_year": len(hits.records) - len(wanted),
            "to_fetch": len(wanted),
            "fetch_status": dict(statuses),
            "texts_read": len(texts),
            "texts_printable": len(texts) - len(empty),
            "texts_empty": len(empty),
            "empty_text_ids": sorted(empty, key=int),
        },
        "truncated_queries": list(hits.truncated_queries),
        "no_text": no_text,
        "no_context": counts(no_context)
        | {
            "number_in_text": counts(no_context[in_text]),
            "number_not_in_text": counts(no_context[~in_text]),
        },
    }


def _build_record(
    *,
    cache: Path,
    hit_files: list[Path],
    pool: Path,
    hits: Hits,
    log: Mapping[str, dict],
    texts: Mapping[str, str],
    empty: list[str],
    shots: list[int],
    audit: pd.DataFrame,
    summary: dict,
    argv: list[str],
    corpus: Path,
    input_bytes: Mapping[Path, bytes | None],
    output_bytes: Mapping[str, bytes],
) -> dict:
    """The inputs read and coverage of a links build, for `papers.meta.json`."""

    def file_record(path: Path) -> dict:
        try:
            name = str(path.resolve().relative_to(cache.resolve()))
        except ValueError:
            name = str(path)
        data = input_bytes[path]
        return {
            "path": name,
            "sha256": hashlib.sha256(data).hexdigest() if data is not None else None,
        }

    text_lines = "".join(
        f"{k} {hashlib.sha256(texts[k].encode('utf-8')).hexdigest()}\n"
        for k in sorted(texts, key=int)
    )
    shot_lines = "".join(f"{s}\n" for s in sorted(shots))
    return {
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(full=True),
        "git_dirty": git_dirty(),
        "command": shlex.join(["python", "-m", "labeler.literature.osti", *argv]),
        "rules": {
            "REACH": context.REACH,
            "RANGE_CAP": context.RANGE_CAP,
            "_ARTICLE_SPAN": context._ARTICLE_SPAN,
            "TICK_MIN_RUN": context.TICK_MIN_RUN,
            "TICK_MIN_STEP": context.TICK_MIN_STEP,
            "TICK_PAIR_STEP": context.TICK_PAIR_STEP,
            "POSTAL_REACH": context.POSTAL_REACH,
            "FIRST_SHOT": FIRST_SHOT,
            "LAST_SHOT": LAST_SHOT,
            "FIRST_YEAR": FIRST_YEAR,
        },
        "inputs": {
            "hits": [file_record(p) for p in hit_files],
            "fetch_log": file_record(cache / "fulltext" / "fetched.jsonl"),
            "pool": file_record(pool),
            "probe_script": file_record(cache / "osti_probe.py"),
            "texts": {
                "count": len(texts),
                "sha256": hashlib.sha256(text_lines.encode("utf-8")).hexdigest(),
            },
            "corpus_shots": {
                "directory": str(corpus),
                "count": len(shots),
                "sha256": hashlib.sha256(shot_lines.encode("utf-8")).hexdigest(),
            },
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "pypdf": version("pypdf"),
        },
        "outputs": {
            name: hashlib.sha256(data).hexdigest()
            for name, data in output_bytes.items()
        },
        "summary": summary,
        "coverage": _coverage(hits, log, texts, empty, audit),
    }


def main(argv=None) -> int:
    """Fetch texts, or write `links.csv`, `papers.csv` and `papers.meta.json`."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="python -m labeler.literature.osti")
    parser.add_argument("command", choices=("fetch", "links"))
    parser.add_argument(
        "--cache", type=Path, help="default: $LABELER_ROOT/literature/osti"
    )
    parser.add_argument("--hits", type=Path, nargs="+", help="probe output files")
    parser.add_argument("--limit", type=int, help="fetch: at most this many records")
    parser.add_argument(
        "--pool", type=Path, help="links: default $LABELER_ROOT/catalog/pool.csv"
    )
    args = parser.parse_args(argv)
    try:
        return _run(args, argv, Paths.from_env())
    except CatalogError as error:
        parser.error(str(error))


def _run(args, argv, paths) -> int:
    cache = args.cache or paths.literature / "osti"
    hit_files = args.hits or sorted(cache.glob("osti_phase*.jsonl"))
    if not hit_files:
        raise CatalogError(f"no probe output under {cache}")
    input_bytes = {path: path.read_bytes() for path in hit_files}
    hits = read_hits([io.BytesIO(input_bytes[path]) for path in hit_files])
    folder = cache / "fulltext"
    if args.command == "fetch":
        # pypdf warns of each broken cross-reference it recovers from; the log
        # line's page and character counts are what matter.
        logging.getLogger("pypdf").setLevel(logging.ERROR)
        wanted = to_fetch(hits)
        counts = fetch_all(wanted, folder, Fetcher(), limit=args.limit)
        print(
            json.dumps(
                {
                    "fetched": dict(counts),
                    "records": len(hits.records),
                    f"dated_before_{FIRST_YEAR}": len(hits.records) - len(wanted),
                }
            )
        )
        return 0
    pool = args.pool or paths.catalog / "pool.csv"
    if not pool.is_file():
        raise CatalogError(
            f"no {pool}: links date the shots from the population's pool"
        )
    input_bytes[pool] = pool.read_bytes()
    starts = year_starts(read_pool(io.BytesIO(input_bytes[pool])))
    if not starts or min(starts) < FIRST_YEAR:
        raise CatalogError(
            f"{pool} dates shots from {min(starts, default=None)}: the fetch skipped"
            f" every paper dated before {FIRST_YEAR}"
        )
    shots = [s for s in corpus_shots(paths) if FIRST_SHOT <= s <= LAST_SHOT]
    for path in (folder / "fetched.jsonl", cache / "osti_probe.py"):
        input_bytes[path] = path.read_bytes() if path.is_file() else None
    log = fetch_log(io.BytesIO(input_bytes[folder / "fetched.jsonl"] or b""))
    texts, empty = _texts(folder, log)
    empty_ids = set(empty)
    printable = {k: text for k, text in texts.items() if k not in empty_ids}
    years = {s: shot_year(s, starts) for s in shots}
    audit, papers = build_links(hits, printable, shots, years)
    output_bytes = {
        "links.csv": audit.to_csv(index=False).encode("utf-8"),
        "papers.csv": papers_frame(papers.to_dict("records"))
        .to_csv(index=False)
        .encode("utf-8"),
    }
    for name, data in output_bytes.items():
        with atomic_path(cache / name) as tmp:
            tmp.write_bytes(data)
    summary = _summary(hits, audit, papers, starts=starts, empty_texts=len(empty))
    record = _build_record(
        cache=cache,
        hit_files=hit_files,
        pool=pool,
        hits=hits,
        log=log,
        texts=texts,
        empty=empty,
        shots=shots,
        audit=audit,
        summary=summary,
        argv=argv,
        corpus=paths.corpus,
        input_bytes=input_bytes,
        output_bytes=output_bytes,
    )
    with atomic_path(cache / "papers.meta.json") as tmp:
        tmp.write_text(json.dumps(record, sort_keys=True, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
