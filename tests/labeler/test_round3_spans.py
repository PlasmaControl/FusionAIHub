"""Round three's span rules (spec §4): a dead filterscope is not observable, the
ramp-up is uncertain where the detector saw events there, the ELM onsets the
frame models read, and a version's table written beside v1's, a run into a table
drafted under other rules refused."""

from __future__ import annotations

import dataclasses
import json
import re

import h5py
import numpy as np
import pandas as pd
import pytest

from labeler.events import spans, suggestions
from labeler.events.catalog.states import ABSENT, NOT_OBSERVABLE, PRESENT, UNCERTAIN

from . import editor_tree as tree
from .test_ae_sbatch import _help, _runs, _script
from .test_events_spans import SHOT, _elm_train, _ip_ramp, _write_synth

#: `_elm_train`'s time axis: 0.1 ms samples over 0-1000 ms.
ELM_T = tree.times(0.0, 1000.0, 10_000)


def test_a_flat_run_of_dead_ms_or_more_is_a_dead_stretch():
    assert spans.DEAD_MS == 200.0
    y = 1.0 + tree.noise(1, ELM_T, 0.01)[0]
    assert spans.dead_stretches(ELM_T, y) == [], "noise repeats no sample"
    y[(ELM_T >= 100) & (ELM_T <= 350)] = 0.5  # 250 ms
    y[(ELM_T >= 600) & (ELM_T <= 750)] = 0.5  # 150 ms
    y[(ELM_T >= 800) & (ELM_T <= 950)] = np.nan  # a gap: the coverage's, not dead
    assert spans.dead_stretches(ELM_T, y) == [(100.0, 350.0)]
    assert spans.dead_stretches(ELM_T, y.astype(np.float32)) == [(100.0, 350.0)]
    assert spans.dead_stretches(ELM_T, y, min_ms=150.0) == [
        (100.0, 350.0),
        (600.0, 750.0),
    ]
    y[ELM_T >= 700] = 0.25  # to the record's last sample, 999.9 ms
    got = spans.dead_stretches(ELM_T, y)
    assert got == [(100.0, 350.0), (700.0, pytest.approx(999.9))]


def _flatten(p, lo_ms, hi_ms, value=1.0, row=0):
    """Filterscope `row` (0: FS01) held at `value` over `[lo_ms, hi_ms]`, as a
    filterscope that stopped reading holds one value."""
    with h5py.File(p.corpus_file(SHOT), "a") as f:
        y = f["filterscopes/ydata"][...]
        y[row, (ELM_T >= lo_ms) & (ELM_T <= hi_ms)] = value
        f["filterscopes/ydata"][...] = y


def test_a_dead_stretch_of_the_elm_channel_is_not_observable(tmp_path):
    p = tree.paths(tmp_path)
    _elm_train(p)  # ELMs every 20 ms over 200-400 ms, and one at 700
    _flatten(p, 0.0, 1000.0, value=0.5, row=1)  # FS02 dead throughout
    found = spans.detect_elm(SHOT, p)
    assert found.info["channel"] == "FS01" and "dead_ms" not in found.info
    assert found.measured == ((0.0, pytest.approx(999.9)),), "FS02 is not read"
    _flatten(p, 500.0, 800.0)
    found = spans.detect_elm(SHOT, p)
    assert found.measured == (
        (0.0, pytest.approx(500.0)),
        (pytest.approx(800.0), pytest.approx(999.9)),
    )
    assert found.info["dead_ms"] == 300.0
    [(start, _, state)] = found.spans
    assert state == PRESENT and start == pytest.approx(195, abs=1)
    rows = spans.shot_rows(SHOT, (0, 1000), found)
    assert [row[1] for row in rows] == [
        ABSENT,
        PRESENT,
        ABSENT,
        NOT_OBSERVABLE,
        ABSENT,
    ]
    assert rows[3][2:4] == [500, 800] and rows[-1][3] == 1000
    _ip_ramp(p, full_at_ms=10.0, t1_ms=1000.0)
    info = spans.detect_elm(SHOT, p, (0, 650)).info
    assert info["dead_ms"] == 150.0, "only what lies inside the window"
    rule = spans.METHODS["edge_localized_mode"].rule
    assert rule["min_dead_ms"] == spans.DEAD_MS
    assert "min_dead_ms" not in spans.METHODS["sawtooth_oscillation"].rule


def test_a_dead_stretch_can_run_to_the_records_end(tmp_path):
    p = tree.paths(tmp_path)
    _elm_train(p)
    _flatten(p, 750.0, 1000.0)  # to the last sample, 999.9 ms
    found = spans.detect_elm(SHOT, p)
    assert found.measured == ((0.0, pytest.approx(750.0)),)
    assert found.info["dead_ms"] == 249.9
    rows = spans.shot_rows(SHOT, (0, 1000), found)
    assert rows[-1][1:4] == [NOT_OBSERVABLE, 750, 1000]


def _hmode(*marked):
    """The H-mode method's answer: the `marked` spans over 0-1000 ms, L-mode in
    between."""
    return spans.Found(tuple(marked), ((0.0, 1000.0),))


def _hmode_after(t_ms):
    """The H-mode method's answer: L-mode until `t_ms`, H-mode after."""
    return _hmode((t_ms, 1000.0, PRESENT))


def test_elms_in_the_ramp_up_make_it_uncertain_unless_it_was_l_mode(
    tmp_path, monkeypatch
):
    p = tree.paths(tmp_path)
    _elm_train(p)  # ELMs every 20 ms over 200-400 ms, and one at 700
    _ip_ramp(p, full_at_ms=362.5, t1_ms=1000.0)  # the plasma starts at 290 ms
    start, _ = spans.plasma_start(SHOT, p, (0, 1000))
    found = spans.detect_elm(SHOT, p, (0, 1000))  # no H-mode inputs: no gate
    ramp, (lo, _, state) = found.spans
    assert ramp == (0.0, start, UNCERTAIN)
    assert found.info["ramp_events"] == 5, "the ELMs at 200-280 ms"
    assert state == PRESENT and lo == pytest.approx(295, abs=1)
    rows = spans.suggest(spans.METHODS["edge_localized_mode"], SHOT, (0, 1000), p)[0]
    assert [row[1] for row in rows] == [UNCERTAIN, ABSENT, PRESENT, ABSENT]
    monkeypatch.setattr(spans, "detect_hmode", lambda shot, paths: _hmode_after(230))
    found = spans.detect_elm(SHOT, p, (0, 1000))
    assert found.spans[0] == (230.0, start, UNCERTAIN), "not the L-mode to 230 ms"
    assert found.info["ramp_events"] == 3, "200 and 220 ms were L-mode"
    marked = (
        (0.0, 150.0, UNCERTAIN),
        (210.0, 225.0, UNCERTAIN),
        (265.0, 1000.0, PRESENT),
    )
    monkeypatch.setattr(spans, "detect_hmode", lambda shot, paths: _hmode(*marked))
    found = spans.detect_elm(SHOT, p, (0, 1000))  # L-mode 150-210 and 225-265 ms
    assert found.spans[:2] == ((210.0, 225.0, UNCERTAIN), (265.0, start, UNCERTAIN))
    assert found.info["ramp_events"] == 2, "220 and 280 ms; 0-150 ms saw none"
    rows = spans.shot_rows(SHOT, (0, 1000), found)
    assert [row[1:4] for row in rows[:4]] == [
        [ABSENT, 0, 210],
        [UNCERTAIN, 210, 225],
        [ABSENT, 225, 265],
        [UNCERTAIN, 265, int(np.ceil(start))],
    ]
    monkeypatch.setattr(spans, "detect_hmode", lambda shot, paths: _hmode_after(300))
    found = spans.detect_elm(SHOT, p, (0, 1000))
    assert [state for _, _, state in found.spans] == [PRESENT], "L-mode ramp (D63)"
    assert "ramp_events" not in found.info


def test_a_ramp_up_without_elms_is_absent(tmp_path):
    p = tree.paths(tmp_path)
    _elm_train(p)
    _ip_ramp(p, full_at_ms=187.5, t1_ms=1000.0)  # the plasma starts at 150 ms
    found = spans.detect_elm(SHOT, p, (0, 1000))
    assert found.info["start_ms"] == pytest.approx(150, abs=1)
    [(_, _, state)] = found.spans
    assert state == PRESENT and "ramp_events" not in found.info


def test_crashes_in_the_ramp_up_make_it_uncertain(tmp_path, synth_shot):
    p = tree.paths(tmp_path)
    _write_synth(p, synth_shot)
    crashes = np.asarray(synth_shot["crash_times_s"]) * 1000  # 75.9 ms, every 76
    found = spans.detect_sawtooth(SHOT, p, (-500, 800))  # no Ip: starts at 200 ms
    assert found.spans[0] == (-500.0, 200.0, UNCERTAIN)
    assert found.info["ramp_events"] == 2
    _ip_ramp(p)  # the plasma starts at 240 ms
    start, _ = spans.plasma_start(SHOT, p, (100, 800))
    found = spans.detect_sawtooth(SHOT, p, (100, 800))
    ramp, (lo, _, state) = found.spans
    assert ramp == (100.0, start, UNCERTAIN)
    assert found.info["ramp_events"] == 2, "151.9 and 227.9; 75.9 is outside"
    assert state == PRESENT
    assert lo == pytest.approx(crashes[crashes >= start][0] - spans.PAD_MS, abs=1.5)
    assert "ramp" in spans.METHODS["sawtooth_oscillation"].rule["start"]


def test_a_ramp_up_without_crashes_is_absent(tmp_path, synth_shot):
    p = tree.paths(tmp_path)
    _write_synth(p, synth_shot)
    _ip_ramp(p, full_at_ms=10.0)  # the plasma starts before the first crash
    found = spans.detect_sawtooth(SHOT, p, (0, 800))
    [(_, _, state)] = found.spans
    assert state == PRESENT and "ramp_events" not in found.info


def test_the_elm_onsets_are_the_clocks_inside_the_present_spans(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    _elm_train(p)  # ELMs every 20 ms over 200-400 ms, and one at 700
    clock = spans.transients.elm_clock_events
    monkeypatch.setattr(  # the clock's events backwards: the onsets come sorted
        spans.transients, "elm_clock_events", lambda *a, **k: clock(*a, **k)[::-1]
    )
    onsets = spans.elm_onsets(SHOT, p)
    assert isinstance(onsets, spans.Onsets) and isinstance(onsets.times_ms, tuple)
    expected = np.arange(200.0, 401.0, 20.0)  # not the lone ELM at 700 ms
    assert list(onsets.times_ms) == pytest.approx(expected, abs=0.5)
    assert onsets.found == spans.detect_elm(SHOT, p)
    _ip_ramp(p, full_at_ms=362.5, t1_ms=1000.0)  # the plasma starts at 290 ms
    monkeypatch.setattr(spans, "detect_hmode", lambda shot, paths: _hmode_after(330))
    onsets = spans.elm_onsets(SHOT, p, (0, 1000))
    # 200-280 ms are before the plasma's start, and 200-320 ms L-mode
    assert list(onsets.times_ms) == pytest.approx([340, 360, 380, 400], abs=0.5)
    found = spans.detect_elm(SHOT, p, (0, 1000))
    assert onsets.found == found and onsets.found.info == found.info


def test_the_elm_onsets_are_those_the_tables_present_rows_hold(tmp_path, monkeypatch):
    found = spans.Found(
        ((100.0, 300.0, PRESENT), (400.0, 500.0, UNCERTAIN)), ((0.0, 250.0),)
    )
    clock = [90.0, 150.0, 240.0, 260.0, 450.0]
    monkeypatch.setattr(spans, "_elm", lambda shot, paths, window: (clock, found))
    p = tree.paths(tmp_path)
    assert spans.elm_onsets(SHOT, p).times_ms == (150.0, 240.0), "260: not measured"
    assert spans.elm_onsets(SHOT, p, (0, 200)).times_ms == (150.0,), "240: past it"


@pytest.fixture
def two_shots(tmp_path, monkeypatch):
    """A cohort of shots 2 and 3, and an ELM method with one span a shot."""
    p = tree.paths(tmp_path)
    tree.use_env(monkeypatch, p)
    tree.cohort(p, [tree.queue_row(2, 0), tree.queue_row(3, 1)])

    def detect(shot, paths, window=None):
        return spans.Found(((100, 200, PRESENT),), ((0.0, 800.0),))

    method = dataclasses.replace(spans.METHODS["edge_localized_mode"], detect=detect)
    monkeypatch.setitem(spans.METHODS, "edge_localized_mode", method)
    return p


def test_a_version_is_written_beside_v1(two_shots, capsys):
    p = two_shots
    assert spans.main(["--event", "edge_localized_mode"]) == 0
    v1 = suggestions.table_path(p, "edge_localized_mode", "elm_clock", "v1")
    before = [path.read_bytes() for path in (v1, v1.with_suffix(".meta.json"))]
    assert json.loads(before[1])["version"] == spans.VERSION == "v1"
    capsys.readouterr()
    argv = ["--event", "edge_localized_mode", "--version", "v2", "--shots", "3"]
    assert spans.main(argv) == 0
    printed = json.loads(capsys.readouterr().out)
    v2 = suggestions.table_path(p, "edge_localized_mode", "elm_clock", "v2")
    assert v2.name == "edge_localized_mode_suggest_elm_clock_v2.csv"
    assert (printed["table"], printed["shots_run"]) == (str(v2), 1), "3: not in v2"
    assert pd.read_csv(v2).shot.unique().tolist() == [3]
    assert json.loads(v2.with_suffix(".meta.json").read_text())["version"] == "v2"
    after = [path.read_bytes() for path in (v1, v1.with_suffix(".meta.json"))]
    assert after == before, "v1's table and meta, byte for byte"


def _bytes(table):
    return [path.read_bytes() for path in (table, table.with_suffix(".meta.json"))]


def test_a_table_drafted_under_other_rules_is_refused(two_shots, capsys):
    p = two_shots
    method = spans.METHODS["edge_localized_mode"]
    v1 = suggestions.table_path(p, "edge_localized_mode", "elm_clock", "v1")
    old = json.loads(json.dumps(method.rule))  # v1's rule, round two's
    del old["min_dead_ms"], old["start"]["ramp"]
    old["l_mode"] = "less the time the dalpha_lh method saw in L-mode"
    suggestions.write_table(v1, [[2, ABSENT, 0, 1000, ""]], {"rule": old})
    before, meta = _bytes(v1), str(v1.with_suffix(".meta.json"))
    for extra in ([], ["--force"], ["--shots", "3"], ["--gold"]):
        with pytest.raises(SystemExit) as refused:
            spans.main(["--event", "edge_localized_mode", *extra])
        err = capsys.readouterr().err
        assert refused.value.code == 2 and meta in err and "--version" in err, extra
    with pytest.raises(spans.RuleChanged, match="--version"):
        spans.run(method, spans.queue(p), p, windows="cohort")
    assert _bytes(v1) == before, "v1's table and meta, byte for byte"
    assert spans.main(["--event", "edge_localized_mode", "--version", "v2"]) == 0
    v2 = suggestions.table_path(p, "edge_localized_mode", "elm_clock", "v2")
    assert sorted(pd.read_csv(v2).shot.unique()) == [2, 3]
    argv = ["--event", "edge_localized_mode", "--version", "v2", "--force"]
    assert spans.main(argv) == 0, "v2 runs again under its own rules"
    v2.with_suffix(".meta.json").unlink()
    with pytest.raises(spans.RuleChanged):  # no meta: no rules to go on
        spans.run(method, spans.queue(p), p, windows="cohort", version="v2")
    assert _bytes(v1) == before


@pytest.mark.parametrize("version", ["V2", "2", "v", "v2b", "../v2"])
def test_a_version_is_v_and_a_number(two_shots, capsys, version):
    with pytest.raises(SystemExit) as refused:
        spans.main(["--event", "edge_localized_mode", "--version", version])
    assert refused.value.code == 2
    assert "not a version such as v1 or v2" in capsys.readouterr().err


def test_the_spans_job_passes_its_version():
    text = _script("spans.sbatch")
    assert '${VERSION:+--version "$VERSION"}' in text
    [(module, flags)] = _runs(text)
    assert module == "labeler.events.spans" and "--version" in flags
    helped = _help(module)
    for flag in flags:
        assert re.search(rf"(?<![\w-]){flag}\b", helped), flag
