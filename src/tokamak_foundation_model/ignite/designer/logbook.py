"""DIII-D logbook text for the designer: what operators wrote about each shot and run.

Source: the logbook export in ``TEXT_DIR`` (``sql/logs.jsonl``, one record per shot with
run title, mini-proposal, configuration, pre-shot comment and all timestamped log
entries; ``sql/runs/<run>/session.md``, the run summary). It is read once into a SQLite
database with full-text search (FTS5, ranked by BM25), ``INDEX_DIR/logbook.sqlite``:

    python -m tokamak_foundation_model.ignite.designer.logbook

No learned model is involved. The text is read-only; the database can be rebuilt any time.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from . import machine as mc

TEXT_DIR = Path("/scratch/gpfs/EKOLEMEN/ps9551/ignite/foundation_model_text")
DB = mc.INDEX_DIR / "logbook.sqlite"
SHOT_FIELDS = ("run", "run_title", "mpid", "configuration", "shot_type", "precomment",
               "shot_brief", "log_text")


def build(text_dir: Path = TEXT_DIR, db: Path = DB) -> Path:
    """Read the logbook export into ``db`` (replaced if it exists)."""
    t_start = time.time()
    tokenized = set()
    try:
        import numpy as np
        z = np.load(mc.INDEX_DIR / "shots.npz", allow_pickle=True)
        tokenized = {int(s) for s in z["shot"]}
    except FileNotFoundError:
        pass
    tmp = db.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.executescript("""
        CREATE TABLE shots (shot INTEGER PRIMARY KEY, run TEXT, run_title TEXT, mpid TEXT,
            configuration TEXT, shot_type TEXT, precomment TEXT, shot_brief TEXT,
            log_text TEXT, topics TEXT, tokenized INTEGER);
        CREATE TABLE runs (run TEXT PRIMARY KEY, title TEXT, summary TEXT);
        CREATE VIRTUAL TABLE shots_fts USING fts5(run_title, precomment, log_text,
            content='shots', content_rowid='shot');
        CREATE VIRTUAL TABLE runs_fts USING fts5(title, summary, content='runs');
    """)
    titles = {}
    n = 0
    with open(text_dir / "sql" / "logs.jsonl") as fh:
        rows = []
        for line in fh:
            r = json.loads(line)
            shot = int(r["shot"])
            rows.append((shot, *[(r.get(k) or "") for k in SHOT_FIELDS],
                         ",".join(r.get("topics") or []), int(shot in tokenized)))
            titles.setdefault(r.get("run") or "", r.get("run_title") or "")
            if len(rows) >= 5000:
                con.executemany("INSERT OR REPLACE INTO shots VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
                n += len(rows)
                rows = []
        con.executemany("INSERT OR REPLACE INTO shots VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
        n += len(rows)
    runs = []
    for d in sorted((text_dir / "sql" / "runs").iterdir()):
        f = d / "session.md"
        summary = f.read_text(errors="replace") if f.exists() else ""
        runs.append((d.name, titles.get(d.name, ""), summary))
    con.executemany("INSERT OR REPLACE INTO runs VALUES (?,?,?)", runs)
    con.execute("INSERT INTO shots_fts(shots_fts) VALUES ('rebuild')")
    con.execute("INSERT INTO runs_fts(runs_fts) VALUES ('rebuild')")
    con.commit()
    con.close()
    tmp.replace(db)
    print(f"logbook: {n} shots, {len(runs)} runs, {sum(1 for _ in tokenized)} tokenized "
          f"shots known -> {db} ({time.time() - t_start:.0f} s)")
    return db


def _con() -> sqlite3.Connection:
    if not DB.exists():
        raise FileNotFoundError(
            f"{DB} is missing; build it with "
            "`python -m tokamak_foundation_model.ignite.designer.logbook`")
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + f"\n[... {len(text) - n} more characters]"


def shot(shot: int, max_chars: int = 6000, run_chars: int = 4000) -> dict:
    """Everything the logbook holds on ``shot``: its fields, its log entries and the
    summary of its run (each clipped to ``max_chars`` / ``run_chars``)."""
    con = _con()
    r = con.execute("SELECT * FROM shots WHERE shot = ?", (int(shot),)).fetchone()
    if r is None:
        return {"shot": int(shot), "found": False}
    out = {k: r[k] for k in r.keys() if k not in ("log_text",)}
    out["found"] = True
    out["topics"] = [x for x in (r["topics"] or "").split(",") if x]
    out["log_text"] = _clip(r["log_text"] or "", max_chars)
    run = con.execute("SELECT summary FROM runs WHERE run = ?", (r["run"],)).fetchone()
    out["run_summary"] = _clip(run["summary"], run_chars) if run and run["summary"] else ""
    others = con.execute("SELECT shot FROM shots WHERE run = ? ORDER BY shot", (r["run"],))
    out["run_shots"] = [x[0] for x in others]
    return out


def _fts_query(query: str) -> str:
    """A plain query as an FTS5 expression: quoted phrases stay phrases, other words are
    ANDed (a word ending in * is a prefix). FTS5 syntax (OR, NEAR, ...) passes through
    when the query already uses it."""
    if re.search(r"\b(OR|AND|NOT|NEAR)\b", query):
        return query
    parts = re.findall(r'"[^"]+"|\S+', query)
    out = []
    for p in parts:
        if p.startswith('"'):
            out.append(p)
        else:
            w = re.sub(r"[^\w*\-]", "", p)
            if w:
                out.append(f'"{w[:-1]}"*' if w.endswith("*") else f'"{w}"')
    return " ".join(out)


def search(query: str, shots: Optional[Iterable[int]] = None, shot_range=None,
           tokenized_only: bool = False, limit: int = 20) -> dict:
    """Shots whose run title, pre-shot comment or log text match ``query`` (BM25 rank),
    each with a snippet around the match. ``shots`` restricts to a set of shot numbers
    (e.g. measured precedents), ``shot_range`` to ``[first, last]``."""
    con = _con()
    q = _fts_query(query)
    sql = ("SELECT s.shot, s.run, s.run_title, s.configuration, s.precomment, s.tokenized, "
           "snippet(shots_fts, 2, '[', ']', ' ... ', 24) AS snip, bm25(shots_fts) AS score "
           "FROM shots_fts JOIN shots s ON s.shot = shots_fts.rowid WHERE shots_fts MATCH ?")
    args: List = [q]
    if tokenized_only:
        sql += " AND s.tokenized = 1"
    if shot_range:
        sql += " AND s.shot BETWEEN ? AND ?"
        args += [int(shot_range[0]), int(shot_range[1])]
    keep = None if shots is None else {int(x) for x in shots}
    sql += " ORDER BY score"
    rows = con.execute(sql, args)
    out = []
    n_total = 0
    for r in rows:
        if keep is not None and r["shot"] not in keep:
            continue
        n_total += 1
        if len(out) < limit:
            out.append({"shot": r["shot"], "run": r["run"], "run_title": r["run_title"],
                        "configuration": r["configuration"], "precomment": r["precomment"],
                        "tokenized": bool(r["tokenized"]), "snippet": r["snip"],
                        "score": round(-float(r["score"]), 2)})
    return {"query": query, "fts_query": q, "n_matches": n_total, "matches": out}


def search_runs(query: str, limit: int = 10) -> dict:
    """Runs (experiments) whose title or summary match ``query``, with their shots."""
    con = _con()
    q = _fts_query(query)
    rows = con.execute(
        "SELECT r.run, r.title, snippet(runs_fts, 1, '[', ']', ' ... ', 24) AS snip, "
        "bm25(runs_fts) AS score FROM runs_fts JOIN runs r ON r.rowid = runs_fts.rowid "
        "WHERE runs_fts MATCH ? ORDER BY score LIMIT ?", (q, int(limit))).fetchall()
    out = []
    for r in rows:
        shots_ = [x[0] for x in con.execute("SELECT shot FROM shots WHERE run = ? ORDER BY shot",
                                            (r["run"],))]
        n_tok = con.execute("SELECT COUNT(*) FROM shots WHERE run = ? AND tokenized = 1",
                            (r["run"],)).fetchone()[0]
        out.append({"run": r["run"], "title": r["title"], "snippet": r["snip"],
                    "shots": [shots_[0], shots_[-1]] if shots_ else [], "n_shots": len(shots_),
                    "n_tokenized": n_tok})
    return {"query": query, "fts_query": q, "runs": out}


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else TEXT_DIR)
