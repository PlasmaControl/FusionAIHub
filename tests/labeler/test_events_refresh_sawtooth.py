"""`events.refresh_sawtooth`: v3's sawtooth rows over existing events files.

The files are written here the way the pipeline wrote them - a v2 sawtooth
train on ECE beside a track, an L-H transition and their sources rows - and
the corpus file is the synthetic shot's ECE alone, so what v3 finds is the
fixture's ten crashes and what it cannot read is the SXR group.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import h5py
import numpy as np
import pandas as pd
import pytest

from labeler.config import Paths
from labeler.events import pipeline, schema
from labeler.events import refresh_sawtooth as rs
from labeler.events.schema import Event

SHOT = 199999
#: v2's train as the files hold it: none of these times is one of the
#: fixture's crashes, so a row left behind is seen.
V2_TIMES_S = (0.05, 0.15, 0.25, 0.35, 0.45)


def _v2(t):
    return Event(
        shot=SHOT,
        source="ece_sawtooth",
        phenomenon="sawtooth",
        t0_s=t,
        t1_s=t,
        confidence=1.0,
        diag="ece",
        attrs={"inversion_channel": 20},
        t_cov0_s=0.0,
        t_cov1_s=0.8,
        evidence_kind="heuristic",
    )


def _others():
    return [
        Event(
            shot=SHOT,
            source="tokeye_track",
            phenomenon="coherent_mode",
            t0_s=0.1,
            t1_s=0.5,
            f0_khz=4.9,
            f1_khz=6.8,
            confidence=0.9,
            diag="mhr",
            channel=4,
            pass_name="wide",
            t_cov0_s=0.0,
            t_cov1_s=0.8,
        ),
        Event(
            shot=SHOT,
            source="dalpha_lh",
            phenomenon="lh_transition",
            t0_s=0.3,
            t1_s=0.3,
            diag="filterscopes",
            attrs={"why": "fixture"},
            t_cov0_s=0.0,
            t_cov1_s=0.8,
            evidence_kind="heuristic",
        ),
    ]


def _record(source, diag, n, channel=-1, pass_name=""):
    return {
        "source": source,
        "status": "ok",
        "reason": "",
        "t_cov0_s": 0.0,
        "t_cov1_s": 0.8,
        "n_events": n,
        "diag": diag,
        "channel": channel,
        "pass_name": pass_name,
    }


@pytest.fixture
def paths(tmp_path, monkeypatch, synth_shot):
    p = Paths(root=tmp_path / "root", corpus=tmp_path / "corpus")
    monkeypatch.setenv("LABELER_ROOT", str(p.root))
    monkeypatch.setenv("LABELER_CORPUS", str(p.corpus))
    p.corpus.mkdir(parents=True)
    noise = np.random.default_rng(3).normal(0.0, 1e-4, synth_shot["ece_y"].shape)
    with h5py.File(p.corpus_file(SHOT), "w") as f:
        g = f.create_group("ece")
        g.create_dataset("xdata", data=synth_shot["ece_t_s"].astype(np.float32))
        g.create_dataset("ydata", data=(synth_shot["ece_y"] + noise).astype(np.float32))
    schema.write_events(
        p.events_file(SHOT),
        SHOT,
        [_v2(t) for t in V2_TIMES_S] + _others(),
        run_id="v2-run",
    )
    schema.write_sources(
        p.sources_file(SHOT),
        SHOT,
        [
            _record("tokeye_track", "mhr", 1, channel=4, pass_name="wide"),
            _record("dalpha_lh", "filterscopes", 1),
            _record("ece_sawtooth", "ece", len(V2_TIMES_S)),
        ],
        run_id="v2-run",
    )
    return p


def _not_sawtooth(frame):
    out = frame[frame.source != "ece_sawtooth"].reset_index(drop=True)
    if "intervals" in out:
        # Null on disk either way: pandas reads an all-null column's nulls as
        # None and a string column's as NaN, and v3's rows make it a string.
        out["intervals"] = (
            out["intervals"].astype(object).where(out["intervals"].notna(), None)
        )
    return out


def test_the_refresh_replaces_only_the_sawtooth_rows(paths, synth_shot):
    events0 = schema.read_events(paths.events_file(SHOT))
    sources0 = schema.read_sources(paths.sources_file(SHOT))
    row = rs.refresh_shot(
        SHOT,
        corpus_file=paths.corpus_file(SHOT),
        events_dir=paths.events,
        run_id="v3-run",
    )
    assert (row["v2"], row["v3"], row["diags"]) == (len(V2_TIMES_S), 10, ["ece"])
    assert set(row["skipped"]) == {"sawtooth sxr"}

    events = schema.read_events(paths.events_file(SHOT))
    sources = schema.read_sources(paths.sources_file(SHOT))
    # Everything else: the same rows, the same content, the same order.
    pd.testing.assert_frame_equal(_not_sawtooth(events), _not_sawtooth(events0))
    pd.testing.assert_frame_equal(_not_sawtooth(sources), _not_sawtooth(sources0))
    saw = events[events.source == "ece_sawtooth"]
    np.testing.assert_allclose(saw.t0_s, synth_shot["crash_times_s"], atol=2e-3)
    assert {json.loads(a)["detector"] for a in saw["attrs"]} == {"v3"}
    assert set(saw.run_id) == {"v3-run"}
    got = {
        r.diag: (r.status, r.n_events, r.run_id)
        for r in sources[sources.source == "ece_sawtooth"].itertuples()
    }
    assert got == {"ece": ("ok", 10, "v3-run"), "sxr": ("skipped", 0, "v3-run")}
    # Replaced, not merged: the same v3 step writes the same rows again.
    rs.refresh_shot(
        SHOT,
        corpus_file=paths.corpus_file(SHOT),
        events_dir=paths.events,
        run_id="v3-again",
    )
    again = schema.read_events(paths.events_file(SHOT))
    assert (again.source == "ece_sawtooth").sum() == 10


def test_a_shot_where_v3_cannot_run_loses_v2s_rows_to_a_skip(paths):
    with h5py.File(paths.corpus_file(SHOT), "a") as f:
        del f["ece"]
    row = rs.refresh_shot(
        SHOT,
        corpus_file=paths.corpus_file(SHOT),
        events_dir=paths.events,
        run_id="v3-run",
    )
    assert row["v3"] == 0 and "no sawtooth diagnostic ran" in row["skipped"]["sawtooth"]
    events = schema.read_events(paths.events_file(SHOT))
    assert "ece_sawtooth" not in set(events.source)
    sources = schema.read_sources(paths.sources_file(SHOT), source="ece_sawtooth")
    assert list(zip(sources.diag, sources.status, strict=True)) == [("ece", "skipped")]


def test_no_corpus_file_leaves_the_shot_as_it_was(paths):
    before = paths.events_file(SHOT).read_bytes()
    paths.corpus_file(SHOT).unlink()
    [row] = rs.run([SHOT], paths=paths, events_dir=paths.events, run_id="r")
    assert row["error"].startswith("no corpus file")
    assert paths.events_file(SHOT).read_bytes() == before


def test_in_place_the_originals_are_backed_up_before_the_first_write(
    paths,
    tmp_path,
    monkeypatch,
):
    originals = {
        f.name: f.read_bytes()
        for f in (paths.events_file(SHOT), paths.sources_file(SHOT))
    }
    backup = tmp_path / "backup"
    real = pipeline.sawtooth_block

    def checked(shot, corpus_file):
        # The step runs only once both copies are there and are the v2 files.
        assert {f.name: f.read_bytes() for f in backup.iterdir()} == originals
        return real(shot, corpus_file)

    monkeypatch.setattr(pipeline, "sawtooth_block", checked)
    assert rs.main(["--backup", str(backup), "--run-id", "v3-run"]) == 0
    assert set(schema.read_events(paths.events_file(SHOT)).run_id) >= {"v3-run"}
    # A second run keeps the first backup: v2's copy is never replaced.
    monkeypatch.setattr(pipeline, "sawtooth_block", real)
    assert rs.main(["--backup", str(backup), "--run-id", "v3-again"]) == 0
    assert {f.name: f.read_bytes() for f in backup.iterdir()} == originals


def test_the_default_backup_is_named_for_v2_and_the_day(paths, capsys):
    assert rs.main(["--shots", str(SHOT)]) == 0
    [backup] = paths.root.glob(f"{rs.BACKUP_PREFIX}*")
    today = datetime.now(UTC).astimezone().date()
    assert backup.name == f"events-backup-sawtooth-v2-{today}"
    assert sorted(f.name for f in backup.iterdir()) == [
        f"{SHOT}_events.parquet",
        f"{SHOT}_sources.parquet",
    ]
    out = capsys.readouterr().out
    assert f"{SHOT}: v2 {len(V2_TIMES_S)} -> v3 10" in out
    totals = json.loads(out.strip().splitlines()[-1])
    assert (totals["v2_events"], totals["v3_events"]) == (len(V2_TIMES_S), 10)
    assert (totals["v2_shots_with_any"], totals["v3_shots_with_any"]) == (1, 1)


def test_out_refreshes_copies_and_leaves_the_root_alone(paths, tmp_path):
    before = {f.name: f.read_bytes() for f in sorted(paths.events.iterdir())}
    out = tmp_path / "pilot"
    assert rs.main(["--out", str(out)]) == 0
    assert {f.name: f.read_bytes() for f in sorted(paths.events.iterdir())} == before
    assert not list(paths.root.glob(f"{rs.BACKUP_PREFIX}*")), "no backup needed"
    refreshed = schema.read_events(out / f"{SHOT}_events.parquet")
    assert (refreshed.source == "ece_sawtooth").sum() == 10


def test_a_shot_with_one_file_is_listed_and_left_alone(paths, capsys):
    lone = paths.events / f"{SHOT + 1}_events.parquet"
    lone.write_bytes(paths.events_file(SHOT).read_bytes())
    assert rs.listed(paths.events) == ([SHOT], {SHOT + 1: "no sources file"})
    before = lone.read_bytes()
    assert rs.main([]) == 0
    assert lone.read_bytes() == before
    assert f"{SHOT + 1}: left alone" in capsys.readouterr().out
