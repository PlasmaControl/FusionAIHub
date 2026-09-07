"""The shot-log contract: what to run on the GA network, and how to bring the result back.

The text corpus under `$IDEATE_TEXT_ROOT/shotsummary/` is not this project's output. It is the
output of **PlasmaControl/d3dlogfetching** (`main.py` scrapes a run day off the DIII-D logbook into
`data/runs/<run_id>/`; `textprocess.py` turns each of those into one `shot_<N>.txt` per shot), and
that tool can only run on a machine inside the GA fusion network -- not on Stellar, not on Omega.
So a shot with no `shot_<N>.txt` cannot be fixed from here at all. What ideate can do is exactly
two things, and this module is both of them:

* **`missing`** -- say which shots have no bundle and print the exact commands to run over there,
  chunked so a command line is a command line rather than a 200 kB argv.
* **`import`** -- take a `runs/` tree that came back over rsync, put it into the corpus's own
  `raw/<run_id>/` layout, and regenerate the per-shot bundles for it.

## Why the composition is a port and not a reimplementation

`import` has to produce bundles that are **byte-identical** to what `textprocess.py` would have
written. The corpus has 22,950 of them and every reader downstream -- `text.shot_table_row`,
`text.mp_text`, the phenomenon search, the selection rule of `select.py` -- was written against
their exact shape: the section headers, the `- KEY: value` bullets, the single-space indentation
that `normalize_text`'s `[ \\t]+ -> " "` leaves in the metadata JSON, the `(Shot table key/value
mapping not found.)` sentinel. A re-implementation that is merely equivalent in spirit gives the
corpus two populations of bundle and no way to tell which one a given file came from.

So everything from `sanitize_text` down to `compose_per_shot_bundle` below is a **verbatim port**
of `textprocess.py` @ d40d8e9 (PlasmaControl/d3dlogfetching), reformatted for this project's line
length and typed, with no change to what any of it returns. `test_logs.py` re-composes a real run
out of `shotsummary/raw/` and asserts byte equality against the bundles already in the corpus,
which is the only check that can actually say the port held.

Three deliberate departures, none of which touch a byte of output:

* `build_shot_specific_text` no longer calls `extract_shot_summary_block_text`. Upstream computes
  it into a local and never uses it; it is pure cost.
* `pdf_to_text` and `html_to_text` import their libraries lazily and `pdf_to_text` returns `""`
  when pypdf is absent -- upstream's own behaviour via its module-level `try: import` guards.
* The `Session` dataclass is frozen.
* `re.I`/`re.M` are spelled `re.IGNORECASE`/`re.MULTILINE` and one `k in d and d[k]` is `d.get(k)`,
  because this repo's ruff gate rejects the aliases. They are the same objects and the same
  truthiness, so the byte-equality test still covers them -- but a future re-sync against upstream
  will diff on those lines, and that is the whole list of places it will.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from bs4 import BeautifulSoup

# The tool's own artefact names, in the order `RunStorage.save_run` writes them. `import` copies
# exactly these and nothing else: a `runs/` tree that has been rsynced may also carry
# `out_shots/`, `.DS_Store` and whatever else the operator's machine left in it, and none of that
# belongs in the corpus's raw layout.
RUN_FILES = (
    "metadata.json",
    "summary.html",
    "summary.md",
    "miniproposal.md",
    "miniproposal.html",
    "miniproposal.pdf",
)

# A run directory is one that carries `metadata.json` -- `RunStorage.is_run_cached`'s own test.
RUN_MARKER = "metadata.json"

# Shots per `main.py` invocation. The tool takes them as positional argv and looks each one up in
# turn, so the only limit is the command line; 200 keeps a chunk pasteable and its output
# readable, and is the brief's number.
FETCH_CHUNK = 200

# The run id is the run DAY as the logbook names it -- eight digits, optionally with a
# disambiguating letter for a second session that day (`20220630A`). `main.py` gets it from the
# shot lookup, names the directory `data/runs/<run_id>/` after it, and records `{shot: run_id}` in
# `index.json`; `textprocess.py --run-id <run_id>` then names that same directory. So a run id can
# only be known here for a shot the corpus's own `sql/index.json` already places -- for a shot it
# does not, the run id does not exist until `main.py` has run and written its index.


# ------------------------------------------------------------------------------ logs missing


def missing(shots: list[int] | set[int], text_dir: Path) -> list[int]:
    """The shots of `shots` with no `shot_<N>.txt` in `text_dir`, sorted and deduplicated."""
    text_dir = Path(text_dir)
    return sorted({int(s) for s in shots if not (text_dir / f"shot_{int(s)}.txt").exists()})


def run_ids(shots: list[int] | set[int], index_json: Path | None) -> dict[int, str | None]:
    """`{shot: run_id}` from the corpus's `sql/index.json`, None where it does not know the shot.

    None is the honest answer and not a gap to be filled in: the run id of a shot whose logbook
    entry has never been fetched is a fact that lives on the GA side, and `main.py` writes it into
    its own `index.json` as part of fetching. `missing` prints those shots under their own heading
    for that reason.
    """
    out: dict[int, str | None] = {int(s): None for s in sorted({int(s) for s in shots})}
    if not index_json or not Path(index_json).exists():
        return out
    try:
        index = json.loads(Path(index_json).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return out
    for shot in out:
        got = index.get(str(shot))
        if got:
            out[shot] = str(got)
    return out


def fetch_commands(shots: list[int], *, chunk: int = FETCH_CHUNK) -> list[str]:
    """`uv run python main.py <shots...>`, one command per `chunk` shots, in shot order."""
    ordered = sorted({int(s) for s in shots})
    return [
        "uv run python main.py " + " ".join(str(s) for s in ordered[i : i + chunk])
        for i in range(0, len(ordered), chunk)
    ]


def textprocess_commands(
    ids: list[str], *, base_dir: str = "data/runs", outdir: str = "out"
) -> list[str]:
    """`uv run python textprocess.py --base-dir ... --outdir ... --run-id <run_id>`, one per run.

    `--base-dir` is where the `<run_id>/` directories live -- `data/runs` under whatever
    `RunStorage` was pointed at -- and `--outdir` is where `per_shot_txt/` will be created. Both
    are the tool's own flags; only `--run-id` varies per command, which is why there is one line
    per run and not per shot.
    """
    return [
        f"uv run python textprocess.py --base-dir {base_dir} --outdir {outdir} --run-id {run_id}"
        for run_id in sorted({str(r) for r in ids if r})
    ]


def format_missing(
    shots: list[int],
    *,
    n_total: int,
    index_json: Path | None,
    base_dir: str = "data/runs",
    outdir: str = "out",
    chunk: int = FETCH_CHUNK,
) -> str:
    """The whole `logs missing` report: the count, the shots, and the commands to run at GA."""
    lines = [f"{len(shots):,} of {n_total:,} shots have no shot_<N>.txt"]
    if not shots:
        return lines[0]
    lines += ["", " ".join(str(s) for s in shots), ""]
    lines += [
        "These can only be fetched on a machine inside the GA fusion network (VPN on; not the",
        "DIII-D HPC machines). In a PlasmaControl/d3dlogfetching checkout with .env set:",
        "",
    ]
    lines += ["  " + c for c in fetch_commands(shots, chunk=chunk)]
    known = run_ids(shots, index_json)
    placed = sorted({r for r in known.values() if r})
    unplaced = [s for s, r in known.items() if not r]
    lines += ["", "then, once those have written data/runs/<run_id>/ and data/index.json:", ""]
    lines += ["  " + c for c in textprocess_commands(placed, base_dir=base_dir, outdir=outdir)]
    if unplaced:
        lines += [
            "",
            "run id not yet known for: " + " ".join(str(s) for s in unplaced),
            "  (read it out of the index.json main.py writes, then one textprocess.py per run)",
        ]
    lines += [
        "",
        "Finally rsync the runs/ tree back and run:",
        "",
        "  ideate logs import <runs_dir>",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------------------ logs import


@dataclass
class ImportReport:
    """What one `import` did, or would have done under `--dry-run`."""

    runs: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # directories that are not runs
    files_copied: int = 0
    bundles_written: int = 0
    bundles_kept: int = 0  # an existing per-shot file left alone (no --force)
    would_write: list[str] = field(default_factory=list)


def import_runs(
    runs_dir: Path,
    *,
    raw_dir: Path,
    txt_dir: Path,
    dry_run: bool = False,
    force: bool = False,
) -> ImportReport:
    """Copy a synced `runs/` tree into `raw_dir/<run_id>/` and regenerate its per-shot bundles.

    An existing `shot_<N>.txt` is never overwritten without `force`: the corpus's own bundles are
    the archived truth, and a re-composition that differs from one of them is a finding to look
    at, not a file to replace. (`force` is safe to use anyway -- the port is byte-identical, so a
    forced rewrite of an unchanged run rewrites the same bytes; `test_logs.py` pins that.)

    `dry_run` reports without touching the filesystem, including without creating `raw_dir`.
    """
    runs_dir, raw_dir, txt_dir = Path(runs_dir), Path(raw_dir), Path(txt_dir)
    report = ImportReport()
    for run in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        if not (run / RUN_MARKER).exists():
            report.skipped.append(run.name)
            continue
        report.runs.append(run.name)
        dest = raw_dir / run.name
        if not dry_run:
            dest.mkdir(parents=True, exist_ok=True)
        for name in RUN_FILES:
            src = run / name
            if not src.exists():
                continue
            report.files_copied += 1
            if not dry_run:
                shutil.copy2(src, dest / name)
        # Composed from the SOURCE tree, so a dry run needs no copy to have happened first and a
        # real one cannot depend on the order the copies landed in.
        session = load_session(runs_dir, run.name)
        for shot in session.shot_ids:
            out = txt_dir / f"shot_{shot}.txt"
            if out.exists() and not force:
                report.bundles_kept += 1
                continue
            report.bundles_written += 1
            report.would_write.append(out.name)
            if not dry_run:
                txt_dir.mkdir(parents=True, exist_ok=True)
                out.write_text(compose_per_shot_bundle(session, shot), encoding="utf-8")
    return report


# =================================================================================================
# Ported verbatim from PlasmaControl/d3dlogfetching @ d40d8e9, `textprocess.py`.
# Do not "improve" anything below this line: every line of it is load-bearing for byte equality
# with the 22,950 bundles already in the corpus. See the module docstring.
# =================================================================================================


def sanitize_text(text: str) -> str:
    return text.encode("utf-8", "replace").decode("utf-8")


def normalize_text(text: str) -> str:
    t = sanitize_text(text)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("\u00a0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _clean_ws(s: str) -> str:
    s = sanitize_text(s).replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def html_to_text(raw_html: str) -> str:
    from bs4 import BeautifulSoup as _BS

    raw_html = sanitize_text(raw_html)
    soup = _BS(raw_html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    return soup.get_text("\n")


def pdf_to_text(pdf_path: Path, max_pages: int | None = None) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover - upstream's module-level guard, same effect
        return ""
    reader = PdfReader(str(pdf_path))
    pages = reader.pages if max_pages is None else reader.pages[:max_pages]
    out: list[str] = []
    for p in pages:
        try:
            out.append(p.extract_text() or "")
        except Exception:  # noqa: BLE001 -- upstream: a page that will not extract is empty text
            out.append("")
    return sanitize_text("\n".join(out))


def read_best_plan_text(run_dir: Path) -> str:
    txt = ""
    pdf = run_dir / "miniproposal.pdf"
    if pdf.exists():
        miniprop_pdf = normalize_text(pdf_to_text(pdf))
        txt += f"Mini-proposal PDF\n{miniprop_pdf}"
    md = run_dir / "miniproposal.md"
    if md.exists():
        miniprop_md = normalize_text(md.read_text(errors="ignore"))
        txt += f"Mini-proposal MD\n{miniprop_md}"
    html = run_dir / "miniproposal.html"
    if html.exists():
        raw = html.read_text(errors="ignore")
        miniprop_html = normalize_text(html_to_text(raw))
        txt += f"Mini-proposal HTML\n{miniprop_html}"
    return txt


_SHOT_RE = re.compile(r"\b\d{6}\b")
_SHOT_SUMMARY_HREF_RE = re.compile(r"shot_summaries\.php\?ShotNumber=(\d{6})", re.IGNORECASE)
_SUMMARY_SHOT_HDR_HREF = re.compile(r"searchstring=SUMMARIES\.SHOT\b", re.IGNORECASE)

_SECTION_HEADINGS = [
    "RF SUMMARY",
    "DIAGNOSTICS SUMMARY",
    "COMPUTER_OPS SUMMARY",
    "CHIEF_OPERATOR SUMMARY",
    "BEAMS SUMMARY",
]

_HEADER_KEYS = [
    "Shot Range",
    "Run",
    "Session Leader",
    "Assistant Session Leader",
    "Physics Operator",
    "Assistant Physics Operator",
    "Diagnostics Coordinator",
    "Computer Operator",
    "Subject",
]


def _text_of(tag) -> str:
    return _clean_ws(tag.get_text("\n", strip=True))


def find_shot_anchor(soup: BeautifulSoup, shot: int):
    shot_str = str(shot)
    for a in soup.find_all("a", href=True):
        m = _SHOT_SUMMARY_HREF_RE.search(a.get("href", ""))
        if m and m.group(1) == shot_str:
            return a
    return None


def get_shot_tr_block_tags(soup: BeautifulSoup, shot: int, max_tr: int = 200) -> list:
    anchor = find_shot_anchor(soup, shot)
    if anchor is None:
        return []
    header_tr = anchor.find_parent("tr")
    if header_tr is None:
        return []
    shot_str = str(shot)

    trs = [header_tr]
    cur = header_tr
    for _ in range(max_tr):
        cur = cur.find_next_sibling("tr")
        if cur is None:
            break
        next_a = cur.find("a", href=True)
        if next_a:
            m2 = _SHOT_SUMMARY_HREF_RE.search(next_a.get("href", ""))
            if m2 and m2.group(1) != shot_str:
                break
        trs.append(cur)
    return trs


def extract_shot_summary_block_text(soup: BeautifulSoup, shot: int) -> str:
    trs = get_shot_tr_block_tags(soup, shot)
    if not trs:
        return ""
    parts: list[str] = []
    seen = set()
    for tr in trs:
        txt = _text_of(tr)
        if txt and txt not in seen:
            seen.add(txt)
            parts.append(txt)
    return "\n\n".join(parts).strip()


def extract_pre_blocks_from_shot(soup: BeautifulSoup, shot: int) -> list[str]:
    trs = get_shot_tr_block_tags(soup, shot)
    pres: list[str] = []
    seen = set()
    for tr in trs[1:]:
        for pre in tr.find_all("pre"):
            t = _clean_ws(pre.get_text("\n", strip=True))
            if t and t not in seen:
                seen.add(t)
                pres.append(t)
    return pres


def _cell_text_with_breaks_value(cell) -> str:
    return _clean_ws(cell.get_text(",", strip=True))


def _cell_text_with_breaks_name(cell) -> str:
    txt = _clean_ws(cell.get_text("-", strip=True))
    txt = re.sub(" ", "-", txt)
    return txt


def find_main_shots_table_and_header_row(soup: BeautifulSoup):
    for a in soup.find_all("a", href=True):
        if _SUMMARY_SHOT_HDR_HREF.search(a.get("href", "")):
            tr = a.find_parent("tr")
            table = a.find_parent("table")
            if tr is not None and table is not None:
                return table, tr
    for tr in soup.find_all("tr"):
        t = tr.get_text(" ", strip=True).upper()
        if "SHOT" in t and "SHOT_TYPE" in t and "TIME" in t:
            table = tr.find_parent("table")
            if table is not None:
                return table, tr
    return None, None


def extract_shot_table_headers(soup: BeautifulSoup) -> list[str]:
    _, header_tr = find_main_shots_table_and_header_row(soup)
    if header_tr is None:
        return []
    headers: list[str] = []
    for cell in header_tr.find_all(["td", "th"]):
        a = cell.find("a")
        txt = _cell_text_with_breaks_name(a) if a else _cell_text_with_breaks_value(cell)
        if txt:
            headers.append(re.sub(r"\s+", ",", txt).strip())
    return headers


def extract_shot_row_tds_for_shot_table(soup: BeautifulSoup, shot: int):
    table, header_tr = find_main_shots_table_and_header_row(soup)
    if table is None:
        return None
    shot_str = str(shot)

    for a in table.find_all("a", href=True):
        if a.get_text(strip=True) == shot_str:
            tr = a.find_parent("tr")
            if tr is not None and tr is not header_tr:
                return tr

    shot_pat = re.compile(rf"\b{re.escape(shot_str)}\b")
    for tr in table.find_all("tr"):
        if tr is header_tr:
            continue
        if shot_pat.search(tr.get_text(" ", strip=True)):
            return tr
    return None


def extract_shot_table_kv(soup: BeautifulSoup, shot: int) -> dict[str, str]:
    headers = extract_shot_table_headers(soup)
    row_tr = extract_shot_row_tds_for_shot_table(soup, shot)
    if not headers or row_tr is None:
        return {}
    cells = row_tr.find_all("td")
    # Upstream's `[...][0]`, kept: `next(...)` would raise StopIteration instead of IndexError on
    # a row with no <td> at all, and the caller has no handler for either.
    values = [_cell_text_with_breaks_value(td) for td in cells][0]  # noqa: RUF015
    values = values.split(",")
    n = min(len(headers), len(values))
    kv: dict[str, str] = {}
    for i in range(n):
        kv[headers[i]] = values[i]
    if len(values) > n:
        for i in range(n, len(values)):
            kv[f"col_{i}"] = values[i]
    return kv


def parse_header_fields_from_text(summary_text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    lines = [ln.strip() for ln in summary_text.splitlines() if ln.strip()]
    for ln in lines[:300]:
        for k in _HEADER_KEYS:
            if ln.lower().startswith(k.lower()):
                if ":" in ln:
                    _, v = ln.split(":", 1)
                    fields[k] = v.strip()
                else:
                    fields[k] = ln[len(k) :].strip(" -\t")
    if "Subject" not in fields:
        for ln in lines[:200]:
            if ("Role of" in ln) or ("Intrinsic" in ln) or ("Rotation" in ln):
                fields.setdefault("Subject", ln.strip())
                break
    return fields


def slice_section(summary_text: str, heading: str) -> str:
    pattern = re.compile(rf"^{re.escape(heading)}\b.*$", re.MULTILINE | re.IGNORECASE)
    m = pattern.search(summary_text)
    if not m:
        return ""
    start = m.end()
    next_pos = None
    for h in _SECTION_HEADINGS:
        if h.lower() == heading.lower():
            continue
        mm = re.compile(rf"^{re.escape(h)}\b.*$", re.MULTILINE | re.IGNORECASE).search(summary_text, start)
        if mm and (next_pos is None or mm.start() < next_pos):
            next_pos = mm.start()
    return summary_text[start:next_pos].strip() if next_pos else summary_text[start:].strip()


def remove_other_shot_lines(text: str, keep_shot: int) -> str:
    keep = str(keep_shot)
    out_lines: list[str] = []
    for ln in text.splitlines():
        shots = _SHOT_RE.findall(ln)
        if not shots:
            out_lines.append(ln)
            continue
        if all(s == keep for s in shots):
            out_lines.append(ln)
    return "\n".join(out_lines).strip()


@dataclass(frozen=True)
class Session:
    run_id: str
    metadata: dict[str, Any]
    summary_html_raw: str
    summary_text_flat: str
    plan_text: str
    shot_ids: list[int]


def load_session(base_dir: Path, run_id: str) -> Session:
    run_dir = Path(base_dir) / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    meta: dict[str, Any] = {}
    meta_path = run_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(sanitize_text(meta_path.read_text(errors="ignore")))

    summary_html_path = run_dir / "summary.html"
    if not summary_html_path.exists():
        raise FileNotFoundError(f"summary.html not found in {run_dir}")

    summary_html_raw = sanitize_text(summary_html_path.read_text(errors="ignore"))
    summary_text_flat = normalize_text(html_to_text(summary_html_raw))
    plan_text = read_best_plan_text(run_dir)

    shots: list[int] = []
    for key in ["shots", "shot_ids", "shot_list"]:
        if key in meta and isinstance(meta[key], list) and meta[key]:
            try:
                shots = sorted(int(x) for x in meta[key])
                break
            except (TypeError, ValueError):
                pass
    if not shots:
        shots = sorted({int(x) for x in _SHOT_RE.findall(summary_text_flat)})

    return Session(
        run_id=run_id,
        metadata=meta,
        summary_html_raw=summary_html_raw,
        summary_text_flat=summary_text_flat,
        plan_text=plan_text,
        shot_ids=shots,
    )


def build_general_session_text(sess: Session, keep_shot: int) -> str:
    header_fields = parse_header_fields_from_text(sess.summary_text_flat)
    parts: list[str] = [f"RUN_ID: {sess.run_id}"]

    if header_fields:
        parts.append("\nGENERAL SESSION INFO")
        for k, v in header_fields.items():
            parts.append(f"- {k}: {v}")

    if sess.metadata:
        sel = {
            k: sess.metadata[k]
            for k in [
                "run_id",
                "date",
                "shot_range",
                "session_leader",
                "physics_operator",
                "topic",
                "title",
            ]
            if sess.metadata.get(k)
        }
        if sel:
            parts.append("\nMETADATA (selected)\n" + json.dumps(sel, indent=2, ensure_ascii=False))

    parts.append("\nSESSION-WIDE SUMMARIES (filtered to exclude other shots)")
    for heading in _SECTION_HEADINGS:
        chunk = remove_other_shot_lines(
            slice_section(sess.summary_text_flat, heading), keep_shot=keep_shot
        )
        if chunk.strip():
            parts.append(f"\n### {heading}\n{chunk}")

    return normalize_text("\n".join(parts))


def build_planned_text(sess: Session, max_chars: int = 100000) -> str:
    if not sess.plan_text.strip():
        return ""
    t = sess.plan_text.strip()
    if len(t) > max_chars:
        t = t[:max_chars].rstrip() + "\n\n[...truncated plan text...]\n"
    return normalize_text(t)


def format_kv_as_bullets(kv: dict[str, str]) -> str:
    if not kv:
        return "(Shot table key/value mapping not found.)"
    preferred = [
        "SHOT",
        "SHOT_TYPE",
        "TIME OF SHOT",
        "PULSE LENGTH",
        "IP (MA)",
        "BTOR",
        "PBEAM MAX (MW)",
        "PECH MAX (MW)",
        "PLH MAX (MW)",
        "A",
        "R",
        "KAPPA",
        "NEUTRONS (TOTAL)",
        "WTOT MAX (MJ)",
    ]
    kv_upper = {k.upper(): (k, v) for k, v in kv.items()}
    lines: list[str] = []
    used = set()
    for p in preferred:
        if p in kv_upper:
            orig_k, v = kv_upper[p]
            lines.append(f"- {orig_k}: {v}")
            used.add(orig_k)
    for k, v in kv.items():
        if k in used:
            continue
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def build_shot_specific_text(sess: Session, shot: int) -> str:
    from bs4 import BeautifulSoup as _BS

    soup = _BS(sess.summary_html_raw, "html.parser")
    kv = extract_shot_table_kv(soup, shot)
    pre_blocks = extract_pre_blocks_from_shot(soup, shot)
    # Upstream also calls extract_shot_summary_block_text here and never uses the result.

    parts: list[str] = [
        f"SHOT: {shot}",
        "\nSHOT TABLE ROW (name -> value)",
        format_kv_as_bullets(kv),
    ]
    if pre_blocks:
        parts.append("\nIMPORTANT <pre> BLOCKS (e.g., PCS CHANGES)")
        for pb in pre_blocks:
            parts.append("\n" + pb)
    return normalize_text("\n".join(parts))


def compose_per_shot_bundle(sess: Session, shot: int) -> str:
    general = build_general_session_text(sess, keep_shot=shot)
    planned = build_planned_text(sess)
    shot_specific = build_shot_specific_text(sess, shot)
    out: list[str] = [
        "# DIII-D per-shot text bundle",
        f"RUN_ID: {sess.run_id}",
        f"SHOT: {shot}",
        "\n## General session context\n" + general,
    ]
    if planned.strip():
        out.append("\n## Planned context (mini-proposal)\n" + planned)
    out.append("\n## Shot-specific context (from summary.html)\n" + shot_specific)
    return normalize_text("\n\n".join(out))
