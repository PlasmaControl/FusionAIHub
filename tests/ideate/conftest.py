"""Synthetic fixtures: tiny shots in the d3d_fusion_data layout, written with pandas.to_hdf, and
-- at the end of this file -- tiny shots in the FAITH corpus layout, written with h5py.

Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

from ideate import config

BEAMS = ["15l", "15r", "21l", "21r", "30l", "30r", "33l", "33r"]
OUR_SCHEMA = "ideate-raw-v1"  # scripts/fetch_shots.SCHEMA; pinned equal in test_fetch_plan


def stamp_ours(path: Path, shot: int = 0) -> None:
    """The file-level markers scripts/fetch_shots.stamp_file writes (`schema`, `shot`), so a
    fixture file is recognised as our fetcher's by its own attrs -- which is how legacy_raw.is_ours
    decides -- and not by the directory it happens to sit in. Every real file in raw_dir carries
    them; a fixture that models a torn or mislocated one of ours has to as well."""
    with h5py.File(path, "a") as f:
        f.attrs["shot"] = int(shot)
        f.attrs["schema"] = OUR_SCHEMA


@pytest.fixture
def paths(tmp_path: Path, monkeypatch) -> config.Paths:
    root, staged, text = tmp_path / "root", tmp_path / "staged", tmp_path / "text"
    for d in (
        "raw",
        "db",
        "text_cache",
        "ignite_inputs",
        "models",
        "eval",
        "sessions",
        "llm_cache",
    ):
        (root / d).mkdir(parents=True)
    staged.mkdir()
    for d in ("sql", "shotsummary/raw", "shotsummary/processed/per_shot_txt"):
        (text / d).mkdir(parents=True)
    cfg = yaml.safe_load((config.CONFIG_DIR / "paths.yaml").read_text())
    cfg.update(
        data_root=str(root),
        models_dir=str(root / "models"),
        staged_raw_dir=str(staged),
        text_root=str(text),
        qh_database_csv=str(text / "QH_Database.csv"),
        foundation_model_processed_dir=str(tmp_path / "fm"),
        fdp_project_dir=str(tmp_path / "fdp"),
    )
    pfile = tmp_path / "paths.yaml"
    pfile.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("IDEATE_PATHS", str(pfile))
    monkeypatch.delenv("IDEATE_DATA_ROOT", raising=False)
    return config.load_paths()


def write_frame(path: Path, group: str, t_ms, cols: dict, attrs: dict | None = None) -> None:
    """Write one group exactly as the fetcher does (Task 11): float64 ms index, float32 columns."""
    t = np.asarray(t_ms, dtype=np.float64)
    df = pd.DataFrame(
        {k: np.asarray(v, dtype=np.float32) for k, v in cols.items()},
        index=pd.Index(t, name="time_ms"),
    )
    with pd.HDFStore(path, mode="a") as store:
        store.put(group, df, format="fixed")
        a = store.get_storer(group).attrs
        dt = float(np.median(np.diff(t))) if t.size > 1 else float("nan")
        a.sampling_frequency_kHz = (1.0 / dt) if dt and np.isfinite(dt) else float("nan")
        a.start_time_ms, a.end_time_ms = float(t[0]), float(t[-1])
        for k, v in (attrs or {}).items():
            setattr(a, k, v)


def trapezoid(t, t_start=0.0, t_flat0=800.0, t_flat1=4800.0, t_end=5500.0, level=1.2e6):
    """A physical Ip trace in amps (level defaults to 1.2 MA). ipsip is stored in megaamps on
    disk (see signals.yaml's ip.scale comment) -- divide this by 1e6 at the point of writing it
    into an ipsip column; iptipp (already amps, no scale in the registry) writes it straight
    through."""
    y = np.zeros_like(t, dtype=np.float64)
    up = (t >= t_start) & (t < t_flat0)
    y[up] = level * (t[up] - t_start) / (t_flat0 - t_start)
    y[(t >= t_flat0) & (t <= t_flat1)] = level
    dn = (t > t_flat1) & (t <= t_end)
    y[dn] = level * (t_end - t[dn]) / (t_end - t_flat1)
    return y


def _efit_grid():
    return np.arange(100.0, 5800.0, 25.0)  # EFIT slices every 25 ms, as in the real files


@pytest.fixture
def staged_shot_a(paths) -> int:
    """A clean 1.2 MA shot: 3 beams on 1.0-4.5 s, EFIT scalars, gas A puffing."""
    shot, path = 900001, paths.staged_raw_dir / "900001.h5"
    t = np.arange(-500.0, 6500.0, 1.0)  # 1 kHz
    on = (t >= 1000) & (t <= 4500)
    write_frame(
        path, "ip", t, {"ipsip": trapezoid(t) / 1.0e6, "iptipp": trapezoid(t, level=1.25e6)}
    )
    write_frame(path, "mag_b0", t, {"bcoil": np.full_like(t, 1e5), "bt": np.full_like(t, 2.0)})
    pin = {f"pinjf_{b}": np.zeros_like(t) for b in BEAMS}
    for b, lvl in (("15l", 2.0e6), ("30l", 1.5e6), ("33l", 1.0e6)):
        pin[f"pinjf_{b}"][on] = lvl
    write_frame(path, "p_inj", t, pin)
    write_frame(
        path,
        "gas",
        t,
        {
            "gasa": np.where(on, 20.0, 0.0),
            **{g: np.zeros_like(t) for g in ("gasb", "gasc", "gasd", "gase")},
        },
    )
    te = _efit_grid()
    ft = (te >= 800) & (te <= 4800)
    write_frame(path, "beta", te, {"betan": np.where(ft, 1.9, 0.5), "betap": np.full_like(te, 0.8)})
    write_frame(
        path,
        "q_psi",
        te,
        {"q0": np.full_like(te, 1.1), "q95": np.full_like(te, 3.6), "qmin": np.full_like(te, 1.05)},
    )
    write_frame(
        path,
        "mag_geo_para",
        te,
        {
            "kappa": np.full_like(te, 1.78),
            "tritop": np.full_like(te, 0.45),
            "tribot": np.full_like(te, 0.40),
            "aminor": np.full_like(te, 0.58),
            "r0": np.full_like(te, 1.7),
            "volume": np.full_like(te, 19.0),
        },
    )
    write_frame(
        path, "other_profiles", te, {"li": np.full_like(te, 0.9), "wmhd": np.full_like(te, 0.8e6)}
    )
    return shot


@pytest.fixture
def staged_shot_b(paths) -> int:
    """Ip only, plus a (1,1) NaN placeholder `ech` group and a two-block frame (float32 + float64 cols)."""
    shot, path = 900002, paths.staged_raw_dir / "900002.h5"
    t = np.arange(-500.0, 6500.0, 1.0)
    write_frame(
        path,
        "ip",
        t,
        {"ipsip": trapezoid(t, level=0.9e6) / 1.0e6, "iptipp": trapezoid(t, level=0.9e6)},
    )
    pd.DataFrame({"echpwr": [np.nan]}, index=pd.Index([np.nan])).to_hdf(
        path, key="ech", mode="a", format="fixed"
    )
    te = _efit_grid()
    mixed = pd.DataFrame(
        {
            "betan": np.full(te.size, 1.5, dtype=np.float32),
            "betap": np.full(te.size, 0.7, dtype=np.float64),
        },
        index=pd.Index(te),
    )
    mixed.to_hdf(path, key="beta", mode="a", format="fixed")  # pandas writes two blocks here
    return shot


@pytest.fixture
def our_shot_c(paths) -> int:
    """A fetched shot that disrupts at 2.0 s while the PCS target stays high; LUKE marked missing."""
    shot, path = 900003, paths.raw_dir / "900003.h5"
    t = np.arange(-500.0, 6500.0, 1.0)
    ip = trapezoid(t)
    ip[(t > 2000) & (t <= 2020)] = 1.2e6 * (2020 - t[(t > 2000) & (t <= 2020)]) / 20.0
    ip[t > 2020] = 0.0
    write_frame(
        path,
        "ip",
        t,
        {"ipsip": ip / 1.0e6, "iptipp": trapezoid(t, level=1.25e6)},
        attrs={
            "units": "A",
            "source": "toksearch",
            "complete": True,
            "missing_channels": np.array([], dtype="S"),
        },
    )
    write_frame(
        path,
        "ech",
        t,
        {"ecleifpwrc": np.where((t >= 500) & (t <= 1900), 0.8e6, 0.0)},
        attrs={
            "units": "W",
            "source": "toksearch",
            "complete": True,
            "missing_channels": np.array([b"eclukfpwrc"], dtype="S"),
        },
    )
    return shot


@pytest.fixture
def dual_shot_d(paths) -> int:
    """The same shot fetched into both locations, to prove our file actually wins a real
    conflict (every other fixture above puts its shot in exactly one location, so a regression
    that swapped or dropped the precedence order would pass unnoticed). Two cases: `ip/ipsip` is
    present in both files with different values (ours must win), and `gas` exists in our file
    but without `gasa` (must fall through to the staged copy rather than reading None).

    Both of our groups carry `complete=True`, because every group scripts/fetch_shots.py finishes
    writing does -- it is the last attr `write_group` stamps, and a group without it is by
    contract a write that was interrupted or failed. The staged groups deliberately do not: no
    staged file has that attribute at all.
    """
    shot = 900004
    our_file, staged_file = paths.raw_dir / "900004.h5", paths.staged_raw_dir / "900004.h5"
    t = np.arange(0.0, 10.0, 1.0)
    done = {"complete": True}
    write_frame(staged_file, "ip", t, {"ipsip": np.full_like(t, 1.0)})  # 1 MA, staged
    write_frame(our_file, "ip", t, {"ipsip": np.full_like(t, 2.0)}, done)  # 2 MA -- must win
    write_frame(staged_file, "gas", t, {"gasa": np.full_like(t, 5.0)})
    write_frame(our_file, "gas", t, {"gasb": np.full_like(t, 9.0)}, done)
    return shot


@pytest.fixture
def text_fixtures(paths) -> dict:
    """Two logbook records, a run folder with metadata + MP header, and one per-shot text bundle.

    Field population mirrors the real corpus (task-9-report.md): `shot_ok`, `quality_comment`,
    `chief_operator`, `plasma_shot` and `refshot` are null in every one of the 53,179 records of
    sql/logs.jsonl, so they are null here too -- the "bad" verdict on 900003 has to come from the
    CHIEF_OPERATOR line inside `log_text`, which is the only production path there is. The task-9
    brief's correction C2 named `keywords` as a sixth always-null field; fix wave 1 re-measured it
    and it is not -- it is an always-empty list, present and non-null in all 53,179 records.
    """
    log1 = (
        "### [SESSION_LEADER] luce 2015-01-20 10:54:43\nPreshot: QH-mode access at low torque\n"
        "requested ip: 1.20 MA, btor: 2.00 T, pnbi: 4.5 MW, pech: 0 MW\n\n"
        "### [SESSION_LEADER] luce 2015-01-20 11:10:00\nGood shot, ran through to rampdown. Wide pedestal QH.\n\n"
        "### [CHIEF_OPERATOR] byrnep 2015-01-20 11:12:00\n900001 Plasma good. IP min.\n"
    )
    log3 = (
        "### [SESSION_LEADER] luce 2015-01-20 12:00:00\nRepeat 900001 with more gas.\n\n"
        "### [CHIEF_OPERATOR] byrnep 2015-01-20 12:05:00\n900003 Plasma Shot - Dud Trip @ 2.0 sec. Bummer, disrupted.\n"
    )
    recs = [
        {
            "shot": 900001,
            "run": "20150120",
            "shot_brief": "autoload system",
            "shot_type": "plasma",
            "plasma_shot": None,  # null in the real corpus
            "shot_ok": None,  # null in the real corpus
            "quality_comment": None,  # null in the real corpus
            "chief_operator": None,  # null in the real corpus
            "run_title": "Access and control of QH-mode edge regime at low NBI torque",
            "mpid": "2014-21-20",
            "experiment_number": "2014-21-20",
            "configuration": "lower null",
            "refshot": None,
            "mp_step": "1A",
            "precomment": "Step 1A, Ip 1.2 MA",
            "log_text": log1,
            "run_log_text": "",
            # Present and non-null in all 53,179 real records: `topics` lists the roles that wrote
            # entries, `keywords` is always the empty list. Correction C2 called `keywords` null,
            # which fix wave 1 re-measured and disproved -- it is [], not None. They are here
            # because they are the only list-valued fields in a record, and so the ones that made
            # load_log_record's old shallow copy unable to protect the cache.
            "topics": ["CHIEF_OPERATOR", "SESSION_LEADER"],
            "keywords": [],
        },
        {
            "shot": 900003,
            "run": "20150120",
            "shot_brief": "autoload system",
            "shot_type": "plasma",
            "plasma_shot": None,  # null in the real corpus
            "shot_ok": None,  # null in the real corpus
            "quality_comment": None,  # null in the real corpus
            "chief_operator": None,  # null in the real corpus
            "run_title": "Access and control of QH-mode edge regime at low NBI torque",
            "mpid": "2014-21-20",
            "experiment_number": "2014-21-20",
            "configuration": "lower null",
            "refshot": 900001,  # null in the real corpus; kept to exercise the parse
            "mp_step": "1B",
            "precomment": "more gas",
            "log_text": log3,
            "run_log_text": "",
            "topics": ["CHIEF_OPERATOR", "SESSION_LEADER"],
            "keywords": [],
        },
    ]
    (paths.text_root / "sql").mkdir(parents=True, exist_ok=True)
    with open(paths.logs_jsonl, "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in recs)
        fh.write(
            json.dumps({"shot": 123456, "run": "20120101", "log_text": ""}) + "\n"
        )  # never asked for
    paths.shot_index_json.write_text(
        json.dumps({"900001": "20150120", "900002": "20150120", "900003": "20150120"})
    )
    run = paths.shotsummary_raw_dir / "20150120"
    run.mkdir(parents=True)
    (run / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": "20150120",
                "mpid": "2014-21-20",
                "shot_range": "900001 - 900003",
                "title": "Access and control of QH-mode edge regime at low NBI torque",
                "session_leaders": "luce burrell",
            }
        )
    )
    (run / "miniproposal.md").write_text(
        "# Miniproposal\n**MiniProposal No.**: 2014-21-20\n**Subject**: QH-mode access at low NBI torque\n"
    )
    bundle = (
        "# DIII-D per-shot text bundle\n\nRUN_ID: 20150120\n\nSHOT: 900001\n\n"
        "## General session context\n...\n\n"
        "## Planned context (mini-proposal)\nMini-proposal PDF\nSubject: QH-mode access at low NBI torque\n"
        "1. Purpose of Experiment\n"
        + "Access the wide-pedestal QH-mode regime with co-current neutral beams at low torque. "
        * 4
        + "\n\n## Shot logbook\nx\n"
    )
    (paths.per_shot_txt_dir / "shot_900001.txt").write_text(bundle)
    return {"shots": {900001, 900003}}


@pytest.fixture
def write_mp_pdf():
    """Write a minimal one-page PDF whose extracted text is `lines`, for the mp_text pdf tier.

    The pdf tier is 435 of the 447 real mini-proposal-text hits on this project's staged shots and
    the only producer of `mp_purpose`, but nothing in the suite created a PDF before fix wave 1 --
    `raise AssertionError` as the first statement of `_pdf_text` left all 97 tests green. This is a
    hand-built PDF rather than a checked-in binary so the text under test is visible in the test.
    """

    def write(path: Path, lines: list[str]) -> None:
        def esc(s: str) -> str:
            return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")

        body = "BT /F1 11 Tf 12 TL 40 750 Td\n" + "".join(f"({esc(x)}) Tj T*\n" for x in lines)
        stream = (body + "ET").encode("latin-1", "replace")
        objs = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
             b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        ]
        out, offsets = bytearray(b"%PDF-1.4\n"), []
        for i, o in enumerate(objs, start=1):
            offsets.append(len(out))
            out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
        xref = len(out)
        out += b"xref\n0 %d\n" % (len(objs) + 1) + b"0000000000 65535 f \n"
        for off in offsets:
            out += b"%010d 00000 n \n" % off
        out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
            len(objs) + 1,
            xref,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(out))

    return write


# ------------------------------------------------------------------ the FAITH corpus layout

# The corpus files are `<shot>_processed.h5` with one group per diagnostic, each holding `xdata`
# (n,) SECONDS and `ydata` (C, n) -- or (C, n, H, W) for a video diagnostic. Shots here are in
# the 100000 band so they cannot be confused with the 900000 band of the d3d_fusion_data
# fixtures above; the corpus's own shots are 185601-204999.
CORPUS_FULL = 100001  # every shape the reader has to tell apart
CORPUS_TRUNCATED = 100002  # a valid file cut in half: h5py raises OSError on open
CORPUS_ABSENT = 100003  # no file at all
CORPUS_SMALL = 100004  # one present group and one placeholder


def write_corpus_group(path: Path, group: str, x_s, y) -> None:
    """One corpus group, written as the real files carry it: contiguous float32 `xdata`/`ydata`
    and no attributes at all (the real files have none, so nothing may depend on them)."""
    with h5py.File(path, "a") as f:
        g = f.create_group(group)
        g.create_dataset("xdata", data=np.asarray(x_s, dtype=np.float32))
        g.create_dataset("ydata", data=np.asarray(y, dtype=np.float32))


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    """A four-shot corpus: a full file, a truncated one, an absent one and a small one.

    The full file carries, deliberately, one group of each kind the layout has: a plain actuator
    group, a fast group with the trailing all-channel NaN pad sample (the real ones are 2^k+1
    long), a group with a channel that recorded nothing, the `(C, 1)` placeholder an absent
    diagnostic is written as, and a video group whose time axis is the SECOND one.
    """
    d = tmp_path / "corpus"
    d.mkdir()
    full = d / f"{CORPUS_FULL}_processed.h5"
    write_corpus_group(full, "pinj", np.arange(5) * 1.0e-3, np.arange(40).reshape(8, 5))
    mhr = np.arange(18, dtype=np.float64).reshape(2, 9)
    mhr[:, -1] = np.nan
    write_corpus_group(full, "mhr", np.arange(9) * 2.0e-6, mhr)
    gas = np.arange(18, dtype=np.float64).reshape(3, 6)
    gas[1, :] = np.nan  # a channel that recorded nothing
    gas[:, -1] = np.nan  # the pad sample
    write_corpus_group(full, "gas_flow", np.arange(6) * 1.0e-3, gas)
    write_corpus_group(full, "co2", [0.0], np.full((4, 1), np.nan))
    write_corpus_group(full, "tangtv", np.arange(3) * 0.02, np.zeros((7, 3, 2, 4)))

    good = full.read_bytes()
    (d / f"{CORPUS_TRUNCATED}_processed.h5").write_bytes(good[: len(good) // 2])

    small = d / f"{CORPUS_SMALL}_processed.h5"
    write_corpus_group(small, "pinj", np.arange(5) * 1.0e-3, np.arange(40).reshape(8, 5))
    write_corpus_group(small, "co2", [0.0], np.full((4, 1), np.nan))
    return d


# ------------------------------------------------- the per-shot text bundles and census frames
#
# `text_bundle` writes the layout d3dlogfetching's `compose_per_shot_bundle` produces, which is
# the layout of all 22,950 real `shot_<N>.txt` files: the RUN_ID/SHOT header, the general session
# context with its own `- Run:`-style bullets and a `METADATA (selected)` JSON block, the
# mini-proposal, and then the shot-specific block with the `SHOT TABLE ROW` the selection rule
# reads. The JSON is written with SINGLE-space indentation because the tool runs its output
# through `re.sub(r"[ \t]+", " ", ...)`, which collapses `json.dumps(indent=2)` to exactly that.

SHOT_TABLE_MISSING = "(Shot table key/value mapping not found.)"


def text_bundle(
    shot: int,
    *,
    row: dict[str, str] | None,
    title: str | None = "Tearing mode avoidance",
    run_id: str = "20220301",
    pre_blocks: str = "",
    subjects: tuple[str, ...] | None = None,
) -> str:
    """One per-shot text bundle. `row=None` is the session-fallback case: the summary page had
    no shot table row for this shot, so the bundle carries the session's text and nothing of the
    shot's own.

    `subjects` is the mini-proposal block's `Subject:` lines. A real block carries up to two --
    the PDF extraction's (often truncated, sometimes with its glyphs mangled) and the markdown
    export's `**Subject**:` (clean) -- and both are matched for a theme, so the fixture can write
    both. The default is one line repeating the run title, which is what most bundles look like.
    """
    meta = {"run_id": run_id, "shot_range": f"{shot - 4} - {shot + 4}"}
    if title is not None:
        meta["title"] = title
    general = "\n".join(
        [
            f"RUN_ID: {run_id}",
            "",
            "GENERAL SESSION INFO",
            "- Run: ",
            f"- Shot Range: {shot - 4} - {shot + 4}",
            "- Session Leader: ",
            "- Physics Operator: ",
            "",
            "METADATA (selected)",
            json.dumps(meta, indent=1),
            "",
            "SESSION-WIDE SUMMARIES (filtered to exclude other shots)",
        ]
    )
    if row is None:
        table = SHOT_TABLE_MISSING
    else:
        table = "\n".join([f"- SHOT: {shot}"] + [f"- {k}: {v}" for k, v in row.items()])
    specific = "\n".join([f"SHOT: {shot}", "", "SHOT TABLE ROW (name -> value)", table])
    if pre_blocks:
        specific += "\n\nIMPORTANT <pre> BLOCKS (e.g., PCS CHANGES)\n\n" + pre_blocks
    subs = (title or "an experiment",) if subjects is None else tuple(subjects)
    planned = ["\n## Planned context (mini-proposal)\nMini-proposal PDF"]
    planned += [
        f"Subject: {s}" if i == 0 else f"**Subject**: {s}" for i, s in enumerate(subs)
    ]
    planned += [
        "1. Purpose of Experiment",
        "Hypothesis to be tested: that this fixture reads like the real thing.",
    ]
    return "\n\n".join(
        [
            "# DIII-D per-shot text bundle",
            f"RUN_ID: {run_id}",
            f"SHOT: {shot}",
            "\n## General session context\n" + general,
            "\n".join(planned),
            "\n## Shot-specific context (from summary.html)\n" + specific,
        ]
    )


def census_frame(rows, *, openable: bool = True) -> pd.DataFrame:
    """A census table from `(shot, group, t0_s, t1_s, present)` tuples, in census.COLUMNS dtypes.

    Built through `census.table` rather than by hand so that a change to the census's own columns
    breaks the selection tests here instead of silently giving them a table the real one is not.
    """
    from ideate.shotdb import census

    made = [
        {
            "shot": shot,
            "group": group,
            "kind": "signal" if group else "",
            "n_channels": 1 if group else 0,
            "n_samples": 100 if present else 1,
            "t0_s": t0,
            "t1_s": t1,
            "fs_hz": float("nan"),
            "present": present,
            "openable": openable,
            "error": "" if openable else "OSError: truncated",
        }
        for shot, group, t0, t1, present in rows
    ]
    return census.table(made)


# ------------------------------------------- a corpus shot addressed through the signal registry
#
# `corpus_dir` above is for the FILE layer: it models the shapes `CorpusReader` has to tell apart.
# The fixtures below are for the SPEC layer -- `CorpusSignalReader` resolving `signals.yaml` and
# `actuators.yaml` addresses -- so their groups are the real ones the address blocks name and
# their values are chosen so that every reduction has one exact expected answer.

CORPUS_SIGNAL_SHOT = 100010  # every address kind, resolvable
CORPUS_BARE_SHOT = 100011  # a corpus file with none of the addressed groups


def _flat(y: float, n: int = 5) -> np.ndarray:
    return np.full(n, y, dtype=np.float64)


@pytest.fixture
def signal_corpus(paths) -> int:
    """One shot under `paths.foundation_model_processed_dir` carrying every corpus address kind.

    Channel values are constants, distinct per channel, so a wrong channel or a wrong reduction
    is an arithmetic mismatch rather than a near miss:

    * `pinj` 8 beams at (i+1) * 1e5 W          -> sum 3.6e6 W
    * `beam_voltage` 8 sources at (i+1) * 1e3 V -> mean 4.5e3 V
    * `tinj` 8 beams at (i+1) N m               -> sum 36 N m
    * `ech_power` 12 gyrotrons, only the corpus's LEIA channel (index 5) is on
    * `gas_raw` 11 valves, only GASA (index 0) is open
    * `i_coil` 18 coils, C19 (index 0) at +10 A and IU30 (index 6) at -20 A
    * `rmp` the 12 I-coils, two of them in antiphase: a signed sum is 10 A, a sum of |.| is 30 A
    * `filterscopes` 10 channels: 0 recorded nothing, 1..7 read 1..7, 8 and 9 read 1000
      (a mean that includes them is 204.4, one that does not is 4.0)
    * `co2` 4 chords, V2 (index 2) at 5e13 and the others at 1.0
    * `neutron_rate` 4 detectors, NEUTRONSRATE (index 3) at 7e14 and the others at 1.0
    """
    d = Path(paths.foundation_model_processed_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{CORPUS_SIGNAL_SHOT}_processed.h5"
    x = np.arange(5) * 1.0e-3  # seconds -> 0..4 ms
    write_corpus_group(p, "pinj", x, np.stack([_flat((i + 1) * 1.0e5) for i in range(8)]))
    write_corpus_group(p, "beam_voltage", x, np.stack([_flat((i + 1) * 1.0e3) for i in range(8)]))
    write_corpus_group(p, "tinj", x, np.stack([_flat(i + 1.0) for i in range(8)]))
    ech = np.zeros((12, 5))
    ech[5] = 1.0e6  # LEIA, in the corpus's alphabetical channel order
    write_corpus_group(p, "ech_power", x, ech)
    gas = np.zeros((11, 5))
    gas[0] = 2.0  # GASA
    write_corpus_group(p, "gas_raw", x, gas)
    write_corpus_group(p, "gas_flow", x, np.full((11, 5), 3.0))
    coils = np.zeros((18, 5))
    coils[0], coils[6] = 10.0, -20.0  # C19, IU30
    write_corpus_group(p, "i_coil", x, coils)
    rmp = np.zeros((12, 5))
    rmp[0], rmp[1] = -20.0, 10.0  # the I-coil subset, driven in antiphase: |sum| 10, sum |.| 30
    write_corpus_group(p, "rmp", x, rmp)
    fs = np.stack([_flat(float(i)) for i in range(10)])
    fs[0] = np.nan
    fs[8] = fs[9] = 1000.0
    write_corpus_group(p, "filterscopes", x, fs)
    co2 = np.ones((4, 5))
    co2[2] = 5.0e13
    write_corpus_group(p, "co2", x, co2)
    neutrons = np.ones((4, 5))
    neutrons[3] = 7.0e14
    write_corpus_group(p, "neutron_rate", x, neutrons)

    bare = d / f"{CORPUS_BARE_SHOT}_processed.h5"
    write_corpus_group(bare, "mhr", x, np.ones((2, 5)))
    return CORPUS_SIGNAL_SHOT


def write_feature_file(features_dir: Path, shot: int, arrays: dict, missing: dict) -> Path:
    """A `<shot>_features.h5` written by labelmaker's own writer, never by hand.

    `arrays` maps a canonical feature name to `(x_seconds, y (C, T), resolver)`; a hand-written
    file would be this module's guess at that layout rather than the layout `read_feature` reads.
    """
    from labelmaker.features.store import FeatureArray, write_features

    features_dir.mkdir(parents=True, exist_ok=True)
    write_features(
        features_dir / f"{shot}_features.h5",
        shot,
        {
            name: FeatureArray(
                x=np.asarray(x, dtype=np.float64),
                y=np.atleast_2d(np.asarray(y, dtype=np.float64)),
                attrs={"resolver": resolver},
            )
            for name, (x, y, resolver) in arrays.items()
        },
        dict(missing),
        merge=False,
    )
    return features_dir / f"{shot}_features.h5"


@pytest.fixture
def labelmaker_features(tmp_path: Path, monkeypatch) -> Path:
    """`$LABELMAKER_ROOT/features` with one file for `CORPUS_SIGNAL_SHOT`.

    Written to cover all four statuses in one shot: a stored scalar (`bt`), a stored profile
    (`ne_zipfit`, whose core/edge/peak reductions differ), a stored feature with no finite sample
    (`kappa`), a transiently missed one (`qmin` -- fdp is worth another attempt), a permanently
    missed one (`volume`), and one that was never attempted at all (`li`).
    """
    root = tmp_path / "labelmaker"
    monkeypatch.setenv("LABELMAKER_ROOT", str(root))
    x = np.arange(0, 6.0, 0.025)  # the 25 ms grid, in seconds
    n = x.size
    ip = np.clip(np.minimum(x / 0.5, (5.0 - x) / 0.5), 0.0, 1.0) * 1.2e6
    profile = np.stack([np.full(n, 10.0 - 9.0 * r) for r in np.linspace(0.0, 1.0, 33)])
    profile[16] = 12.0  # a peak that is neither the core nor the edge
    write_feature_file(
        root / "features",
        CORPUS_SIGNAL_SHOT,
        {
            "ip": (x, ip, "fdp"),
            "bt": (x, np.full(n, -2.05), "archive"),
            "kappa": (x, np.full(n, np.nan), "archive"),
            "ne_zipfit": (x, profile, "archive"),
        },
        {"qmin": "fdp:TreeFOPENR", "volume": "corpus:SignalAbsent"},
    )
    return root / "features"
