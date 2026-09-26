"""OSTI: probe hits, a polite fetcher, text extraction and the links they give.

No test touches the network: the fetcher takes a fake opener, clock and sleep.
"""

from __future__ import annotations

import hashlib
import io
import json
import platform
import shlex
import sys
import urllib.error
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from itertools import pairwise

import pandas as pd
import pytest

from labeler.config import git_dirty, git_sha, sha256_of
from labeler.events.catalog.check import CatalogError
from labeler.events.catalog.population import POOL_COLUMNS
from labeler.literature import context, osti
from labeler.literature.osti import (
    Fetcher,
    build_links,
    extract_text,
    fetch_all,
    fetch_record,
    read_hits,
    shot_year,
    to_fetch,
    year_starts,
)


def _pdf(text: str) -> bytes:
    """A one-page PDF that shows `text`, built by hand."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    bodies = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
    ]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for number, body in enumerate(bodies, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(bodies) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\n" % (len(bodies) + 1)
    out += b"startxref\n%d\n%%%%EOF\n" % xref
    return bytes(out)


class _Answer:
    def __init__(self, body: bytes, status=200, kind="application/pdf", url=""):
        self.body, self.status, self.url = body, status, url
        self.headers = {"Content-Type": kind}

    def read(self):
        return self.body

    def geturl(self):
        return self.url

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Web:
    """A fake opener: answers (or raises) in order; records each request."""

    def __init__(self, *answers):
        self.answers, self.requests = list(answers), []

    def __call__(self, request, timeout):
        self.requests.append(request)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class _Clock:
    def __init__(self):
        self.now, self.slept = 0.0, []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def _http(code):
    return urllib.error.HTTPError("u", code, "x", {}, io.BytesIO())


URL = "https://www.osti.gov/servlets/purl/7"


def _fetcher(web, clock=None):
    clock = clock or _Clock()
    return Fetcher(opener=web, sleep=clock.sleep, clock=clock, backoff_s=5.0)


def test_hits_in_range_with_truncation_and_failures(tmp_path):
    probe = tmp_path / "osti_phase2.jsonl"
    lines = [
        {"shot": 150000, "n": 1, "records": [{"osti_id": "1"}]},  # out of range
        {
            "shot": 189631,
            "n": 3,
            "records": [
                {
                    "osti_id": "7",
                    "title": "A  title",
                    "date": "2023-11-07",
                    "journal": "Nucl. Fusion",
                },
                {"osti_id": "8", "type": "Technical Report"},
            ],
        },
        {"shot": 190000, "error": "timed out"},
        {"shot": 190001, "error": "timed out"},
        {"shot": 190001, "n": 0, "records": []},  # answered on a rerun
    ]
    probe.write_text("".join(json.dumps(line) + "\n" for line in lines))
    hits = read_hits([probe])
    assert sorted(hits.records) == ["7", "8"]
    assert hits.records["7"].title == "A title" and hits.records["7"].year == 2023
    assert hits.records["8"].venue == "Technical Report"
    assert hits.by_shot == {189631: ("7", "8"), 190001: ()}
    assert hits.truncated == (189631,) and hits.failed == (190000,)


def test_records_dated_before_the_corpus_are_not_fetched(tmp_path):
    probe = tmp_path / "osti_phase1.jsonl"
    records = [
        {"osti_id": "1", "date": "2019-05-01"},
        {"osti_id": "2", "date": "2021-01-04"},
        {"osti_id": "3"},  # undated: fetched
    ]
    probe.write_text(json.dumps({"shot": 189631, "n": 3, "records": records}) + "\n")
    assert sorted(to_fetch(read_hits([probe]))) == ["2", "3"]


def test_at_most_three_requests_start_in_any_minute_without_an_email():
    clock = _Clock()
    web = _Web(*(_Answer(b"%PDF") for _ in range(4)))
    starts = []

    def opener(request, timeout):
        starts.append(clock.now)
        return web(request, timeout)

    fetcher = _fetcher(opener, clock)
    for record_id in range(1, 5):
        fetcher.get(osti.PURL.format(record_id))
        clock.now += 0.25
    assert all(b - a >= osti.MIN_INTERVAL_S for a, b in pairwise(starts))
    assert starts[3] - starts[0] >= 60
    headers = " ".join(v for r in web.requests for v in r.headers.values())
    assert "@" not in headers


def test_transient_failures_are_retried_and_a_404_is_not():
    web = _Web(_http(503), urllib.error.URLError("reset"), _Answer(b"%PDF-1.4"))
    assert _fetcher(web).get(URL).status == 200
    web = _Web(_http(404))
    got = _fetcher(web).get(URL)
    assert got.status == 404 and len(web.requests) == 1
    web = _Web(_http(429), _http(429), _http(429))
    assert _fetcher(web).get(URL).status is None


def test_a_pdf_is_kept_and_its_text_extracted(tmp_path):
    web = _Web(_Answer(_pdf("In DIII-D shot 189631 the mode locked.")))
    line = fetch_record(_fetcher(web), "7", tmp_path)
    assert line["status"] == "ok" and line["n_pages"] == 1
    assert line["n_chars"] == len("InDIII-Dshot189631themodelocked.")
    assert "189631" in (tmp_path / "7.txt").read_text()
    text, pages, failed = extract_text(tmp_path / "7.pdf")
    assert (pages, failed) == (1, 0) and "shot 189631" in text


@pytest.mark.parametrize(
    "answer, status",
    [
        (_Answer(b"<html>citation page</html>", kind="text/html"), "not_pdf"),
        (_http(404), "missing"),
        (_http(403), "error"),
        (_Answer(b"%PDF-1.4 but broken"), "unreadable"),
    ],
)
def test_what_is_not_a_readable_pdf_is_logged_as_such(tmp_path, answer, status):
    line = fetch_record(_fetcher(_Web(answer)), "9", tmp_path)
    assert line["status"] == status
    assert not (tmp_path / "9.txt").exists()


def test_a_rerun_skips_final_records_and_retries_errors(tmp_path, capsys):
    web = _Web(_http(403), _Answer(_pdf("shot 189631")))
    counts = fetch_all(["2", "1"], tmp_path, _fetcher(web))  # "1" first
    assert counts == {"error": 1, "ok": 1}
    web = _Web(_Answer(_pdf("shot 189632")))
    assert fetch_all(["1", "2"], tmp_path, _fetcher(web)) == {"ok": 1}
    assert [r.full_url for r in web.requests] == [osti.PURL.format("1")]
    assert len((tmp_path / "fetched.jsonl").read_text().splitlines()) == 3


def test_the_calendar_comes_from_shots_with_their_own_run_id():
    pool = pd.DataFrame(
        {
            "shot": [100, 150, 160, 200, 300],
            "year": pd.array([2021, 2026, None, 2022, 2023], dtype="Int64"),
            "reasons": ["", "ip;session_fallback", "no_bundle", "heating", ""],
        }
    )
    starts = year_starts(pool)  # shot 150's run id is another session's
    assert starts == {2021: 100, 2022: 200, 2023: 300}
    got = [shot_year(s, starts) for s in (99, 150, 250, 999)]
    assert got == [None, 2021, 2022, 2023]
    pool.loc[4, "year"] = 2020
    with pytest.raises(CatalogError, match="shot 300 is dated 2020, after shot 200"):
        year_starts(pool)


def test_links_verify_hits_in_their_own_text(tmp_path):
    probe = tmp_path / "osti_phase1.jsonl"
    probe.write_text(
        json.dumps(
            {
                "shot": 189631,
                "n": 4,
                "records": [
                    {"osti_id": "5", "date": "2019-03-01"},
                    {"osti_id": "7"},
                    {"osti_id": "8"},
                    {"osti_id": "9"},
                ],
            }
        )
        + "\n"
    )
    hits = read_hits([probe])
    texts = {
        "5": "In DIII-D shot 189631 ...",  # in context, but from 2019
        "7": "In DIII-D shot 189631 and discharges 189640-189642 ...",
        "8": "Table 3 lists 189631 grants.",  # no context
    }  # record 9 has no text
    years = dict.fromkeys(range(189600, 189700), 2022)
    audit, papers = build_links(hits, texts, range(189600, 189700), years)
    assert audit[["shot", "record_id", "in_probe", "verdict"]].values.tolist() == [
        [189631, "5", True, "predates"],
        [189631, "7", True, "verified"],
        [189631, "8", True, "no_context"],
        [189631, "9", True, "no_text"],
        [189640, "7", False, "verified"],
        [189641, "7", False, "verified"],
        [189642, "7", False, "verified"],
    ]
    assert audit.paper_year[0] == 2019 and audit.shot_year[0] == 2022
    assert audit.paper_year.isna()[1:].all()  # undated: never predates
    assert papers[["shot", "record_id", "match_type"]].values.tolist() == [
        [189631, "7", "exact"],
        [189640, "7", "exact"],
        [189641, "7", "range"],
        [189642, "7", "exact"],
    ]


def test_the_links_command_writes_both_tables(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "osti"
    folder = cache / "fulltext"
    folder.mkdir(parents=True)
    records = [{"osti_id": "7"}, {"osti_id": "8", "date": "2023-02-01"}]
    (cache / "osti_phase2.jsonl").write_text(
        json.dumps({"shot": 189631, "n": 2, "records": records}) + "\n"
    )
    (folder / "7.txt").write_text("DIII-D shot 189631")
    (folder / "8.txt").write_text(" \f\n ")  # a scanned PDF: no text
    (folder / "fetched.jsonl").write_text(
        "".join(json.dumps({"osti_id": k, "status": "ok"}) + "\n" for k in "78")
    )
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for shot in (189631, 189632, 150000):
        (corpus / f"{shot}_processed.h5").touch()
    monkeypatch.setenv("LABELER_CORPUS", str(corpus))
    pool = tmp_path / "pool.csv"
    rows = {"shot": [189631, 189632], "year": [2022, 2022], "reasons": ["", ""]}
    pd.DataFrame(rows).reindex(columns=list(POOL_COLUMNS)).to_csv(pool, index=False)
    args = ["links", "--cache", str(cache), "--pool", str(pool)]
    assert osti.main(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["verified_shots"] == 1 and summary["hit_links"] == 2
    assert summary["verdicts"] == {"verified": 1, "no_text": 1}
    assert summary["empty_texts"] == 1 and summary["year_starts"] == {"2022": 189631}
    assert pd.read_csv(cache / "papers.csv").shot.tolist() == [189631]
    links = pd.read_csv(cache / "links.csv")
    assert links.verdict.tolist() == ["verified", "no_text"]
    with pytest.raises(SystemExit):  # no pool: the shots cannot be dated
        osti.main(["links", "--cache", str(cache), "--pool", str(tmp_path / "x.csv")])


@pytest.fixture
def links_inputs(tmp_path, monkeypatch):
    cache = tmp_path / "osti cache"
    folder = cache / "fulltext"
    folder.mkdir(parents=True)
    probe = cache / "osti_phase1.jsonl"
    extra_probe = tmp_path / "extra hits.jsonl"
    records = [
        {"osti_id": "1", "date": "2019-01-01"},
        *({"osti_id": str(k)} for k in range(2, 12)),
    ]
    rows = [
        {"shot": 189631, "n": 12, "records": records},
        {
            "shot": 189632,
            "n": 3,
            "records": [{"osti_id": k} for k in ("3", "4", "5")],
        },
        {"shot": 189633, "n": 1, "records": [{"osti_id": "3"}]},
    ]
    probe.write_text("".join(json.dumps(row) + "\n" for row in rows))
    extra_probe.write_text(
        json.dumps({"shot": 189634, "n": 1, "records": [{"osti_id": "3"}]}) + "\n"
    )
    # Deliberately not numeric record order; the digest must sort it.
    raw_texts = {
        "10": "DIII-D discharge 189633; β\n",
        "4": " \f\n ",
        "2": "DIII-D shots 189631 and 189632\n",
    }
    for record_id, text in raw_texts.items():
        (folder / f"{record_id}.txt").write_text(text, encoding="utf-8")
    statuses = [
        ("3", "error"),  # only the later 'missing' status counts
        ("10", "ok"),
        ("2", "ok"),
        ("4", "ok"),
        ("3", "missing"),
        ("11", "missing"),
        ("5", "not_pdf"),
        ("6", "error"),
        ("8", "unreadable"),
        ("9", "ok"),  # logged ok, but the text file is absent
    ]  # record 7 has never been attempted; record 1 predates the corpus
    (folder / "fetched.jsonl").write_text(
        "".join(
            json.dumps({"osti_id": k, "status": status}) + "\n"
            for k, status in statuses
        )
    )
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    shots = [189634, 189632, 189631, 189633]
    for shot in [*shots, 150000, 205000]:
        (corpus / f"{shot}_processed.h5").touch()
    monkeypatch.setenv("LABELER_CORPUS", str(corpus))
    monkeypatch.setenv("LABELER_ROOT", str(tmp_path / "root"))
    pool = tmp_path / "pool.csv"
    rows = {"shot": shots, "year": [2022] * 4, "reasons": [""] * 4}
    pd.DataFrame(rows).reindex(columns=list(POOL_COLUMNS)).to_csv(pool, index=False)
    args = [
        "links",
        "--cache",
        str(cache),
        "--pool",
        str(pool),
        "--hits",
        str(probe),
        str(extra_probe),
    ]
    return cache, pool, [probe, extra_probe], raw_texts, shots, args


@pytest.mark.parametrize("probe_script", [False, True])
def test_links_build_record_inputs_outputs_and_reproducibility(
    links_inputs, monkeypatch, capsys, probe_script
):
    cache, pool, probes, raw_texts, shots, args = links_inputs
    if probe_script:
        (cache / "osti_probe.py").write_text("# fixture probe script\n")
    before = datetime.now(UTC).replace(microsecond=0)
    assert osti.main(args) == 0
    summary = json.loads(capsys.readouterr().out)
    meta_path = cache / "papers.meta.json"
    assert meta_path.is_file()
    raw = meta_path.read_text()
    doc = json.loads(raw)
    assert raw == json.dumps(doc, sort_keys=True, indent=1) + "\n"
    written = datetime.fromisoformat(doc["written_at"])
    assert before <= written <= datetime.now(UTC)
    assert written.utcoffset().total_seconds() == 0 and written.microsecond == 0
    assert doc["git_sha"] == git_sha() and doc["git_dirty"] == git_dirty()
    assert shlex.split(doc["command"]) == [
        "python",
        "-m",
        "labeler.literature.osti",
        *args,
    ]
    assert doc["summary"] == summary
    assert doc["rules"] == {
        "REACH": context.REACH,
        "RANGE_CAP": context.RANGE_CAP,
        "_ARTICLE_SPAN": context._ARTICLE_SPAN,
        "TICK_MIN_RUN": context.TICK_MIN_RUN,
        "TICK_MIN_STEP": context.TICK_MIN_STEP,
        "POSTAL_REACH": context.POSTAL_REACH,
        "FIRST_SHOT": osti.FIRST_SHOT,
        "LAST_SHOT": osti.LAST_SHOT,
        "FIRST_YEAR": osti.FIRST_YEAR,
    }
    text_lines = "".join(
        f"{k} {hashlib.sha256(raw_texts[k].encode('utf-8')).hexdigest()}\n"
        for k in sorted(raw_texts, key=int)
    )
    shot_lines = "".join(f"{s}\n" for s in sorted(shots))
    assert doc["inputs"] == {
        "hits": [
            {"path": "osti_phase1.jsonl", "sha256": sha256_of(probes[0])},
            {"path": str(probes[1]), "sha256": sha256_of(probes[1])},
        ],
        "fetch_log": {
            "path": "fulltext/fetched.jsonl",
            "sha256": sha256_of(cache / "fulltext" / "fetched.jsonl"),
        },
        "pool": {"path": str(pool), "sha256": sha256_of(pool)},
        "probe_script": {
            "path": "osti_probe.py",
            "sha256": sha256_of(cache / "osti_probe.py") if probe_script else None,
        },
        "texts": {
            "count": 3,
            "sha256": hashlib.sha256(text_lines.encode("utf-8")).hexdigest(),
        },
        "corpus_shots": {
            "count": 4,
            "sha256": hashlib.sha256(shot_lines.encode("utf-8")).hexdigest(),
        },
    }
    assert doc["outputs"] == {
        name: sha256_of(cache / name) for name in ("links.csv", "papers.csv")
    }
    assert doc["environment"] == {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "pypdf": version("pypdf"),
    }
    # Force a later timestamp without sleeping, and exercise main(argv=None).
    later = written + timedelta(seconds=1)

    class Later:
        @staticmethod
        def now(tz):
            assert tz is UTC
            return later

    monkeypatch.setattr(osti, "datetime", Later)
    monkeypatch.setattr(sys, "argv", ["osti.py", *args])
    assert osti.main() == 0
    assert json.loads(capsys.readouterr().out) == summary
    second = json.loads(meta_path.read_text())
    assert second.pop("written_at") == later.isoformat(timespec="seconds")
    doc.pop("written_at")
    assert second == doc
    assert not (cache / "papers.meta.json.tmp").exists()


def test_links_build_record_coverage_and_last_status(links_inputs, capsys):
    cache, _, _, _, _, args = links_inputs
    assert osti.main(args) == 0
    doc = json.loads((cache / "papers.meta.json").read_text())
    assert doc["coverage"] == {
        "records": {
            "probe": 11,
            "dated_before_first_year": 1,
            "to_fetch": 10,
            "fetch_status": {
                "ok": 4,
                "missing": 2,
                "not_pdf": 1,
                "error": 1,
                "unreadable": 1,
                "unattempted": 1,
            },
            "texts_read": 3,
            "empty_text_ids": ["4"],
        },
        "truncated_queries": [{"shot": 189631, "n": 12, "kept": 11}],
        "no_text": {
            "missing": {"pairs": 5, "shots": 4, "shots_without_verified_link": 1},
            "empty": {"pairs": 2, "shots": 2, "shots_without_verified_link": 0},
            "not_pdf": {"pairs": 2, "shots": 2, "shots_without_verified_link": 0},
            "error": {"pairs": 1, "shots": 1, "shots_without_verified_link": 0},
            "unreadable": {"pairs": 1, "shots": 1, "shots_without_verified_link": 0},
            "unattempted": {"pairs": 1, "shots": 1, "shots_without_verified_link": 0},
            "ok": {"pairs": 1, "shots": 1, "shots_without_verified_link": 0},
        },
    }
    summary = json.loads(capsys.readouterr().out)
    assert summary["predates_links"] == 1 and summary["empty_texts"] == 1
    assert summary["truncated_shots"] == [189631]
    assert summary["verified_links_not_in_probe"] == 2


def test_links_build_record_before_any_fetch(links_inputs, capsys):
    cache, _, _, _, _, args = links_inputs
    (cache / "fulltext" / "fetched.jsonl").unlink()
    assert osti.main(args) == 0
    doc = json.loads((cache / "papers.meta.json").read_text())
    assert doc["inputs"]["fetch_log"]["sha256"] is None
    assert doc["inputs"]["texts"] == {
        "count": 0,
        "sha256": hashlib.sha256(b"").hexdigest(),
    }
    assert doc["coverage"]["records"]["fetch_status"] == {
        "ok": 0,
        "missing": 0,
        "not_pdf": 0,
        "error": 0,
        "unattempted": 10,
    }
    assert doc["coverage"]["no_text"] == {
        "unattempted": {"pairs": 15, "shots": 4, "shots_without_verified_link": 4}
    }


def test_hits_keep_counts_beside_unchanged_truncated_shots(links_inputs):
    _, _, probes, _, _, _ = links_inputs
    hits = read_hits(probes)
    assert hits.truncated == (189631,)
    assert hits.truncated_queries == ({"shot": 189631, "n": 12, "kept": 11},)


def test_repeated_probe_counts_keep_the_last_truncation(tmp_path):
    probe = tmp_path / "hits.jsonl"
    probe.write_text(
        "".join(
            json.dumps({"shot": 189631, "n": n, "records": [{"osti_id": "1"}]}) + "\n"
            for n in (4, 2, 1)
        )
    )
    hits = read_hits([probe])
    assert hits.truncated == (189631, 189631)  # the existing history is unchanged
    assert hits.truncated_queries == ({"shot": 189631, "n": 2, "kept": 1},)
