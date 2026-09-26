"""OSTI: probe hits, a polite fetcher, text extraction and the links they give.

No test touches the network: the fetcher takes a fake opener, clock and sleep.
"""

from __future__ import annotations

import io
import json
import urllib.error
from itertools import pairwise

import pandas as pd
import pytest

from labeler.events.catalog.check import CatalogError
from labeler.events.catalog.population import POOL_COLUMNS
from labeler.literature import osti
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
