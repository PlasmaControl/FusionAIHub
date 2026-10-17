"""Filling the raw cache ahead of a review: only what neither tier holds."""

from __future__ import annotations

import json

import numpy as np

from labeler.events import raw
from labeler.events.panels import edge_localized_mode as elm
from labeler.events.verify import NoDataError
from labeler.features.store import FeatureArray

from . import editor_tree as tree


def _fake_fdp(calls, missing=()):
    def fdp_signal(shot, exprs, *, tree, via, **kwargs):
        calls.append((shot, exprs[0]))
        if shot in missing:
            raise NoDataError(f"shot {shot}: nothing under {exprs[0]}")
        y = np.ones((len(exprs), 2000), dtype="float32")
        return FeatureArray(x=np.arange(0.0, 100.0, 0.05), y=y, attrs={})

    return fdp_signal


def test_only_what_neither_tier_holds_is_fetched(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    calls = []
    monkeypatch.setattr(raw, "fdp_signal", _fake_fdp(calls, missing={3}))
    monkeypatch.setattr(raw, "RETRY_DELAY_S", 0.0)
    t = tree.times(0.0, 100.0, 10_000)
    tree.write(p.corpus_file(1), {"co2": (t, np.ones((4, len(t))))})
    tree.write(raw.cache_path(2, paths=p), {"pcphd03": (t, np.ones(len(t)))})
    assert raw.fill_cache(1, ("co2", "pcphd03"), p) == {
        "co2": "corpus",
        "pcphd03": "fetched",
    }
    assert raw.fill_cache(2, ("pcphd03",), p) == {"pcphd03": "cache"}
    assert calls == [(1, "PCPHD03")]
    assert raw.raw_signal(1, "pcphd03", paths=p).attrs["tier"] == "cache"
    [where] = raw.fill_cache(3, ("pcphd03",), p).values()
    assert where.startswith("missing: shot 3: nothing under PCPHD03")
    assert calls[1:] == [(3, "PCPHD03")] * 2, "one try and its retry"


def test_main_fills_the_roster_and_counts(tmp_path, monkeypatch, capsys):
    p = tree.paths(tmp_path)
    tree.use_env(monkeypatch, p)
    calls = []
    monkeypatch.setattr(raw, "fdp_signal", _fake_fdp(calls))
    roster = p.label_tables / "high_confinement_mode" / "shots.csv"
    roster.parent.mkdir(parents=True)
    roster.write_text(
        "shot,tier,holdout,reviewers,verified_on,notes\n"
        "5,unverified,false,,,\n4,unverified,false,,,\n"
    )
    assert raw.main(["--event", "high_confinement_mode", "--pace", "0"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines == [
        {"shot": 5, "co2": "fetched", "ece_edge": "fetched"},
        {"shot": 4, "co2": "fetched", "ece_edge": "fetched"},
        {"shots": 2, "fetched": 4},
    ]
    assert raw.main(["--event", "high_confinement_mode", "--limit", "1"]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == '{"shots": 1, "cache": 2}'
    assert len(calls) == 4


def test_ece_edge_is_not_fetched_where_the_corpus_or_the_cache_holds_ece(
    tmp_path, monkeypatch
):
    p = tree.paths(tmp_path)
    calls = []
    monkeypatch.setattr(raw, "fdp_signal", _fake_fdp(calls))
    t = tree.times(0.0, 100.0, 10_000)
    ece = (t, np.ones((48, len(t))))
    tree.write(p.corpus_file(1), {"ece": ece})
    tree.write(raw.cache_path(2, paths=p), {"ece": ece})
    assert raw.fill_cache(1, ("ece_edge",), p) == {
        "ece_edge": "skipped: ece in the corpus"
    }
    assert raw.fill_cache(2, ("ece_edge",), p) == {"ece_edge": "skipped: ece in the cache"}
    assert calls == []
    assert raw.fill_cache(3, ("ece_edge",), p) == {"ece_edge": "fetched"}
    assert [shot for shot, _ in calls] == [3]
    assert raw.fill_cache(3, ("ece_edge",), p) == {"ece_edge": "cache"}


def test_the_new_fetch_specs_are_the_measured_mds_nodes():
    up = raw.FETCH_SPECS["filterscopes_up"]
    assert (up.via, up.tree, up.missing_ok) == ("mds", "SPECTROSCOPY", True)
    assert up.exprs == tuple(
        rf"\SPECTROSCOPY::FS{n}UP" for n in ("13", "12", "11", "00", "01", "02", "03", "04")
    )
    strike = raw.FETCH_SPECS["strike_points"]
    assert (strike.via, strike.tree) == ("mds", "EFIT01")
    assert strike.exprs == (
        r"\EFIT01::TOP.RESULTS.AEQDSK:RVSOUT",
        r"\EFIT01::TOP.RESULTS.AEQDSK:RVSIN",
        r"\EFIT01::TOP.RESULTS.AEQDSK:ZVSOUT",
    )
    # The panel reads the upper chords in this order, and the strike points by row.
    assert tuple(e.rsplit("::", 1)[1] for e in up.exprs) == elm.UPPER_FETCHED


def test_main_fills_the_alfven_roster_with_co2_and_honours_shots(
    tmp_path, monkeypatch, capsys
):
    p = tree.paths(tmp_path)
    tree.use_env(monkeypatch, p)
    calls = []
    monkeypatch.setattr(raw, "fdp_signal", _fake_fdp(calls))
    assert raw.main(["--event", "alfven_eigenmode", "--shots", "9", "--pace", "0"]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[0]) == {
        "shot": 9,
        "co2": "fetched",
    }
    assert [shot for shot, _ in calls] == [9]


def test_main_groups_overrides_the_events_default(tmp_path, monkeypatch, capsys):
    p = tree.paths(tmp_path)
    tree.use_env(monkeypatch, p)
    calls = []
    monkeypatch.setattr(raw, "fdp_signal", _fake_fdp(calls))
    roster = p.label_tables / "confinement" / "shots.csv"
    roster.parent.mkdir(parents=True)
    roster.write_text("shot,tier,holdout,reviewers,verified_on,notes\n7,unverified,false,,,\n")
    assert raw.main(["--event", "confinement", "--groups", "pinj", "--pace", "0"]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[0]) == {
        "shot": 7,
        "pinj": "fetched",
    }
    assert calls == [(7, r"\D3D::TOP.NB.NB15L:PINJ_15L")]


def test_the_mirnov_spec_is_the_corpus_rows_the_readers_use():
    from pathlib import Path

    import yaml

    from labeler.events.panels import neoclassical_tearing_mode as ntm
    from labeler.mhd import tracks

    spec = raw.FETCH_SPECS["mirnov"]
    assert (spec.via, spec.missing_ok, spec.n_rows) == ("ptdata", False, 29)
    assert spec.rows == raw.MIRNOV_ROWS
    # The six toroidal probes the tearing review rows and features read, nothing
    # else: the reference coils (rows 0, 8 of the mode tracks) can run at another rate.
    assert spec.rows == tuple(ntm.PROBES)
    assert set(spec.rows) < set(tracks.ROWS)
    assert not {0, 8} & set(spec.rows)
    # Each point is the one the corpus' own build put at that row.
    modalities = (
        Path(__file__).resolve().parents[2]
        / "src/tokamak_foundation_model/data/config/modalities/modalities.yaml"
    )
    keys = yaml.safe_load(modalities.read_text())["signals"]["mirnov"]["input_key"]
    assert len(keys) == raw.MIRNOV_WIDTH
    assert list(raw.MIRNOV_POINTS) == keys[: len(raw.MIRNOV_POINTS)]
    assert spec.exprs == tuple(keys[row] for row in spec.rows)
    assert keys[15] == "MPI66M322D"


def test_a_fetched_mirnov_reads_back_by_the_corpus_row_index(tmp_path, monkeypatch):
    from labeler.events.panels import neoclassical_tearing_mode as ntm

    p = tree.paths(tmp_path)
    calls = []
    spec = raw.FETCH_SPECS["mirnov"]
    t = np.arange(0.0, 100.0, 0.5)

    def fdp_signal(shot, exprs, *, tree, via, **kwargs):
        calls.append((shot, list(exprs), via))
        y = np.arange(1, len(exprs) + 1, dtype="float32")[:, None] * np.ones(len(t))
        return FeatureArray(x=t, y=y, attrs={})

    monkeypatch.setattr(raw, "fdp_signal", fdp_signal)
    assert raw.fill_cache(7, ("mirnov",), p) == {"mirnov": "fetched"}
    assert calls == [(7, list(spec.exprs), "ptdata")]
    probes = list(ntm.PROBES)
    got = raw.raw_signal(7, "mirnov", channels=probes, paths=p)
    assert got.attrs["tier"] == "cache"
    # The fetch's k-th point landed on the corpus row spec.rows[k]: row 15 is the
    # first fetched point (value 1), row 22 the last (value 6).
    assert [float(row[0]) for row in got.y] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    unfetched = raw.raw_signal(7, "mirnov", channels=[1, 23], paths=p)
    assert np.isnan(unfetched.y).all()
    assert raw.fill_cache(7, ("mirnov",), p) == {"mirnov": "cache"}
    assert len(calls) == 1, "a cached group is not fetched again"

