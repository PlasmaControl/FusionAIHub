"""Span suggestions for the ELM, H-mode and sawtooth editors."""

from __future__ import annotations

import dataclasses
import json

import h5py
import numpy as np
import pandas as pd
import pytest

from labeler.events import heuristics, pipeline, spans, suggestions
from labeler.events.catalog.states import ABSENT, NOT_OBSERVABLE, PRESENT, UNCERTAIN
from labeler.events.verify import NoDataError

from . import editor_tree as tree
from .test_events_heuristics import V3_CRASHES_MS, v3_rows

SHOT = 198658
#: The ECE noise `_write_synth` adds: v3 judges each step in its channel's own
#: noise, and the fixture's array has none.
SYNTH_ECE_NOISE = 0.005
#: What `read` says of a group the shot does not have.
NO_SXR = f"NoDataError: shot {SHOT}: no 'sxr' in the corpus or the raw cache"


def test_runs_need_min_count_points_and_are_padded():
    times = [100, 120, 140, 900, 1000, 1150, 1500, 1510]
    got = spans.runs(times, max_gap_ms=200, min_count=3, pad_ms=5)
    assert got == [(95.0, 145.0), (895.0, 1155.0)]
    assert spans.runs([], max_gap_ms=200, min_count=3, pad_ms=5) == []


@pytest.mark.parametrize(
    ("marks", "measured", "expected"),
    [
        ([(300, True), (600, False)], [(0, 1000)], [(300, 600, PRESENT)]),
        ([(300, True)], [(0, 1000)], [(300, 1000, PRESENT)]),
        ([(200, False)], [(0, 1000)], [(0, 200, UNCERTAIN)]),
        (
            [(100, True), (400, False), (700, False)],
            [(0, 1000)],
            [(100, 400, PRESENT), (400, 700, UNCERTAIN)],
        ),
        ([(100, True), (200, True), (300, False)], [(0, 1000)], [(100, 300, PRESENT)]),
        (
            [(300, True), (550, True), (800, False)],
            [(0, 500), (600, 1000)],
            [(300, 500, PRESENT), (600, 800, UNCERTAIN)],
        ),
    ],
)
def test_hmode_spans(marks, measured, expected):
    assert spans.hmode_spans(marks, measured) == expected


def test_rows_are_not_observable_where_the_method_could_not_see():
    assert spans.shot_rows(SHOT, (0, 1000), None) == [
        [SHOT, NOT_OBSERVABLE, 0, 1000, ""]
    ]
    found = spans.Found(((100, 300, PRESENT), (850, 950, PRESENT)), ((50.4, 900.2),))
    got = [row[1:4] for row in spans.shot_rows(SHOT, (0, 1000), found)]
    assert got == [
        [NOT_OBSERVABLE, 0, 50],
        [ABSENT, 50, 100],
        [PRESENT, 100, 300],
        [ABSENT, 300, 850],
        [PRESENT, 850, 901],
        [NOT_OBSERVABLE, 901, 1000],
    ]


def test_the_tearing_mode_editor_gets_its_window_all_absent():
    method = spans.METHODS["neoclassical_tearing_mode"]
    assert (method.name, method.inputs) == ("window", ())
    found = method.detect(SHOT, None)
    assert spans.shot_rows(SHOT, (-20, 4000), found) == [[SHOT, ABSENT, -20, 4000, ""]]


def test_the_coverage_resolutions_are_the_pipelines():
    assert spans.ELM_MIN_GAP_S == pipeline.ELM_MIN_GAP_S
    assert spans.SAWTOOTH_MIN_GAP_S == pipeline.SAWTOOTH_MIN_GAP_S
    assert spans.LH_MIN_GAP_S == pipeline.LH_MIN_GAP_S


def _write_synth(p, s):
    rng = np.random.default_rng(SHOT)
    ece = s["ece_y"] + rng.normal(0.0, SYNTH_ECE_NOISE, s["ece_y"].shape)
    tree.write(
        p.corpus_file(SHOT),
        {
            "filterscopes": (s["dalpha_t_s"] * 1000, s["dalpha_y"]),
            "pinj": (s["pinj_t_s"] * 1000, s["pinj_y"][None] * 1000),  # kW -> W
            "ece": (s["ece_t_s"] * 1000, ece),
        },
    )
    # The density as a fetch leaves it: in the raw cache, not the corpus.
    tree.write(
        p.raw_cache / f"{SHOT}_processed.h5",
        {"co2": (s["ne_t_s"] * 1000, s["ne_y"][None])},
    )


def test_hmode_and_sawteeth_from_the_synthetic_shot(tmp_path, synth_shot):
    p = tree.paths(tmp_path)
    _write_synth(p, synth_shot)
    hmode = spans.detect_hmode(SHOT, p)
    [(start, stop, state)] = hmode.spans
    assert state == PRESENT
    assert start == pytest.approx(300, abs=1) and stop == pytest.approx(600, abs=1)
    assert hmode.measured[0][0] <= 100 and hmode.measured[-1][1] >= 700
    saw = spans.detect_sawtooth(SHOT, p)
    crashes = np.asarray(synth_shot["crash_times_s"]) * 1000
    [(start, stop, state)] = saw.spans
    assert start == pytest.approx(crashes[0] - spans.PAD_MS, abs=1.5)
    assert stop == pytest.approx(crashes[-1] + spans.PAD_MS, abs=1.5)


def _elm_train(p):
    t = tree.times(0.0, 1000.0, 10_000)
    y = np.full((8, len(t)), np.nan)
    y[0] = 1.0 + tree.noise(1, t, 0.002)[0]
    for peak in [*np.arange(200.0, 401.0, 20.0), 700.0]:  # eleven ELMs, one alone
        y[0] += 1.5 * np.exp(-0.5 * ((t - peak) / 0.4) ** 2)
    tree.write(p.corpus_file(SHOT), {"filterscopes": (t, y)})


def test_elm_runs_from_a_dalpha_spike_train(tmp_path):
    p = tree.paths(tmp_path)
    _elm_train(p)
    found = spans.detect_elm(SHOT, p)  # no density or beams: no H-mode gate
    [(start, stop, state)] = found.spans
    assert state == PRESENT
    assert start == pytest.approx(195, abs=1) and stop == pytest.approx(405, abs=1)
    assert found.measured == ((0.0, pytest.approx(999.9)),)


def test_elms_the_hmode_method_saw_in_l_mode_are_dropped(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    _elm_train(p)
    hmode = spans.Found(((300.0, 1000.0, PRESENT),), ((0.0, 1000.0),))
    monkeypatch.setattr(spans, "detect_hmode", lambda shot, paths: hmode)
    found = spans.detect_elm(SHOT, p)
    [(start, stop, _)] = found.spans
    assert start == 300.0 and stop == pytest.approx(405, abs=1)
    assert found.info == {"channel": "FS01", "hmode_gate": "ran"}


def test_the_elm_draft_records_its_channel_and_why_no_gate_ran(tmp_path):
    p = tree.paths(tmp_path)
    _elm_train(p)
    with h5py.File(p.corpus_file(SHOT), "a") as f:  # FS01 dark: FS02 carries it
        y = f["filterscopes/ydata"][...]
        y[1], y[0] = y[0], np.nan
        f["filterscopes/ydata"][...] = y
    _ip_ramp(p, full_at_ms=362.5, t1_ms=1000.0)
    found = spans.detect_elm(SHOT, p, (0, 1000))
    assert found.info["channel"] == "FS02"
    assert found.info["hmode_gate"].startswith("NoDataError: shot 198658: no 'co2'")
    assert found.info["start_ms"] == pytest.approx(290, abs=1)
    assert found.info["start_from"] == "ip"


def test_the_gate_catches_only_missing_inputs(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    _elm_train(p)
    intervals, status = spans.lmode(SHOT, p)
    assert intervals == [] and status.startswith("NoDataError: shot 198658: no 'co2'")

    def broken(shot, paths):
        raise ValueError("a bug in the H-mode method")

    monkeypatch.setattr(spans, "detect_hmode", broken)
    with pytest.raises(ValueError, match="a bug"):
        spans.detect_elm(SHOT, p)


def test_the_sawtooth_draft_records_where_it_started(tmp_path, synth_shot):
    p = tree.paths(tmp_path)
    _write_synth(p, synth_shot)
    ran = {  # the ELM impostor has no heat pulse: ten crashes
        "diagnostics": ["ece"],
        "crashes": {"ece": 10},
        "collapses_ms": [],
        "not_run": {"sxr": NO_SXR},
    }
    assert spans.detect_sawtooth(SHOT, p, (-500, 800)).info == {
        **ran,
        "start_ms": 200.0,
        "start_from": spans.plasma_start(SHOT, p, (-500, 800))[1],
        "ramp_events": 2,  # the crashes at 75.9 and 151.9 ms
    }
    _ip_ramp(p)
    info = spans.detect_sawtooth(SHOT, p, (0, 800)).info
    assert info["start_from"] == "ip" and info["start_ms"] == pytest.approx(240, abs=2)
    assert info["ramp_events"] == 3
    assert spans.detect_sawtooth(SHOT, p).info == ran, "no window: no start"


def test_minus():
    assert spans.minus([(0, 100)], [(50, 60), (10, 20)]) == [
        (0, 10),
        (20, 50),
        (60, 100),
    ]
    assert spans.minus([(0, 10), (20, 30)], [(-5, 25)]) == [(25, 30)]
    assert spans.minus([(0, 10)], []) == [(0, 10)]


def test_a_missing_input_is_a_no_data_error(tmp_path):
    p = tree.paths(tmp_path)
    tree.write(p.corpus_file(SHOT), {"ip": ([0.0, 1.0], [[1.0, 1.0]])})
    with pytest.raises(NoDataError, match="corpus or the raw cache"):
        spans.detect_hmode(SHOT, p)


@pytest.fixture
def cohort_env(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    tree.use_env(monkeypatch, p)
    tree.cohort(
        p,
        [
            tree.queue_row(3, 3),
            tree.queue_row(1, 0, blind=True),
            tree.queue_row(2, 1),
            tree.queue_row(4, 2),
            tree.queue_row(5, 4, window=("", "")),
        ],
    )
    failing = {4}
    windows = {}

    def detect(shot, paths, window=None):
        windows[shot] = window
        if shot in failing:
            raise NoDataError(f"shot {shot}: no 'filterscopes'")
        info = {"start_ms": 10.0 * shot, "start_from": "ip"}
        return spans.Found(((100, 200, PRESENT),), ((0.0, 800.0),), info)

    method = dataclasses.replace(
        spans.METHODS["edge_localized_mode"], detect=detect, inputs=("x",)
    )
    monkeypatch.setitem(spans.METHODS, "edge_localized_mode", method)
    return p, failing, windows


def _run(capsys, *args):
    assert spans.main(["--event", "edge_localized_mode", *args]) == 0
    return json.loads(capsys.readouterr().out)


def test_the_queue_is_suggested_in_order_and_merged_by_shot(cohort_env, capsys):
    p, failing, windows = cohort_env
    assert list(spans.queue(p).shot) == [2, 4, 3, 5]
    first = _run(capsys, "--limit", "2")
    assert (first["shots_run"], first["skipped_run"], first["shots"]) == (2, 1, 2)
    path = suggestions.table_path(p, "edge_localized_mode", "elm_clock", "v1")
    table = pd.read_csv(path)
    four = table[table.shot == 4]
    assert four.iloc[:, :4].values.tolist() == [[4, NOT_OBSERVABLE, 0, 1000]]
    assert four.confidence.isna().all()
    assert table[table.shot == 2].category.tolist() == [0, 1, 0, 3]
    everything = _run(capsys)
    assert (everything["shots_run"], everything["shots"]) == (1, 3), "2 and 4 are done"
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    assert meta["skipped"] == {
        "4": "NoDataError: shot 4: no 'filterscopes'",
        "5": "no catalog window",
    }
    assert meta["method"] == "elm_clock" and meta["rule"]["min_run"] == spans.MIN_RUN
    assert meta["rule"] == spans.METHODS["edge_localized_mode"].rule
    assert windows[2] == (0, 1000), "each shot is detected on its window"
    assert meta["per_shot"] == {
        "2": {"start_ms": 20.0, "start_from": "ip"},
        "3": {"start_ms": 30.0, "start_from": "ip"},
    }, "what each shot's drafts started from; 4 and 5 are under skipped"
    failing.clear()
    again = _run(capsys, "--shots", "4", "--force")
    assert (again["shots_run"], again["skipped_run"]) == (1, 0)
    table = pd.read_csv(path)
    assert table[table.shot == 4].category.tolist() == [0, 1, 0, 3]
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    assert meta["skipped"] == {"5": "no catalog window"}
    assert sorted(meta["per_shot"]) == ["2", "3", "4"], "2 and 3 are kept"
    assert meta["per_shot"]["4"]["start_ms"] == 40.0
    assert sorted(table.shot.unique()) == [2, 3, 4], "the blind shot is never suggested"


def test_a_blind_shot_cannot_be_named(cohort_env, capsys):
    with pytest.raises(SystemExit):
        spans.main(["--event", "edge_localized_mode", "--shots", "1"])
    assert "not in the cohort (or blind): [1]" in capsys.readouterr().err


def test_with_no_cohort_in_the_tables_the_checkouts_is_read(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    own = tmp_path / "checkout" / "data" / "events"
    monkeypatch.setattr(spans, "DEFAULT_LABEL_TABLES", own)
    assert spans.cohort_path(p) == own / "catalog" / "cohort.csv"
    tree.cohort(p, [tree.queue_row(2, 1)])
    assert spans.cohort_path(p) == p.label_tables / "catalog" / "cohort.csv"


def _ip_ramp(p, *, full_at_ms=300.0, t1_ms=800.0):
    """Ip as a fetch leaves it, in the raw cache: a linear ramp to 1 MA, then flat."""
    t = tree.times(0.0, t1_ms, 1_000)
    ip = 1e6 * np.clip(t / full_at_ms, 0.0, 1.0)
    tree.write(p.raw_cache / f"{SHOT}_processed.h5", {"ip": (t, ip[None])})


def test_the_plasma_starts_where_ip_reaches_its_flat_top_fraction():
    t = np.arange(0.0, 1000.0, 0.5)
    ip = -1e6 * np.clip((t - 100) / 400, 0, 1)  # either sign: |Ip|
    ip[t > 900] = 0.0
    ip[t < 20] = -3e6  # a spike outside the window is not the plateau
    start = spans.ramp_start(t, ip, (50, 950))
    assert start == pytest.approx(100 + 400 * spans.RAMP_FRACTION, abs=1)
    assert spans.RAMP_FRACTION == 0.8, "the catalog's flat-top fraction"
    assert spans.ramp_start(t, np.zeros_like(t), (50, 950)) is None


def test_sawteeth_before_the_ramp_are_dropped_before_runs_form(tmp_path, synth_shot):
    p = tree.paths(tmp_path)
    _write_synth(p, synth_shot)
    _ip_ramp(p)  # 0.8 MA at 240 ms
    crashes = np.asarray(synth_shot["crash_times_s"]) * 1000
    assert spans.plasma_start(SHOT, p, (0, 800)) == (pytest.approx(240, abs=2), "ip")
    plasma, _ = spans.plasma_start(SHOT, p, (0, 800))
    found = spans.detect_sawtooth(SHOT, p, (0, 800))
    ramp, (start, stop, state) = found.spans
    assert ramp == (0.0, plasma, UNCERTAIN), "crashes in the ramp-up: uncertain"
    assert found.info["ramp_events"] == np.sum(crashes < plasma) == 3
    first = crashes[crashes >= 240][0]
    assert state == PRESENT
    assert start == pytest.approx(first - spans.PAD_MS, abs=1.5), "not the Ip start"
    assert stop == pytest.approx(crashes[-1] + spans.PAD_MS, abs=1.5)
    [(start, _, _)] = spans.detect_sawtooth(SHOT, p).spans  # no window: as before
    assert start == pytest.approx(crashes[0] - spans.PAD_MS, abs=1.5)
    rows = spans.suggest(spans.METHODS["sawtooth_oscillation"], SHOT, (0, 800), p)[0]
    ceil = int(np.ceil(plasma))
    assert rows[0][1:4] == [UNCERTAIN, 0, ceil]
    assert rows[1][1:4] == [ABSENT, ceil, int(np.floor(found.spans[1][0]))]


def test_events_outside_the_window_do_not_form_runs(tmp_path, synth_shot):
    p = tree.paths(tmp_path)
    _write_synth(p, synth_shot)
    _ip_ramp(p, full_at_ms=10.0)
    crashes = np.asarray(synth_shot["crash_times_s"]) * 1000
    [(_, stop, _)] = spans.detect_sawtooth(SHOT, p, (0, 500)).spans
    assert stop == pytest.approx(crashes[crashes <= 500][-1] + spans.PAD_MS, abs=1.5)


def test_without_ip_the_plasma_starts_a_fixed_delay_into_the_window(
    tmp_path, synth_shot
):
    p = tree.paths(tmp_path)
    _write_synth(p, synth_shot)
    start, source = spans.plasma_start(SHOT, p, (-500, 800))
    assert start == -500 + spans.RAMP_FALLBACK_MS == 200
    assert source.startswith("window start + 700 ms: NoDataError")
    crashes = np.asarray(synth_shot["crash_times_s"]) * 1000
    found = spans.detect_sawtooth(SHOT, p, (-500, 800))
    ramp, (first, _, _) = found.spans
    assert ramp == (-500.0, 200.0, UNCERTAIN)
    assert found.info["ramp_events"] == 2, "the crashes at 75.9 and 151.9 ms"
    assert first == pytest.approx(crashes[crashes >= 200][0] - spans.PAD_MS, abs=1.5)


def test_elms_before_the_ramp_are_dropped(tmp_path):
    p = tree.paths(tmp_path)
    _elm_train(p)
    _ip_ramp(p, full_at_ms=362.5, t1_ms=1000.0)  # 0.8 MA at 290 ms
    plasma, _ = spans.plasma_start(SHOT, p, (0, 1000))
    found = spans.detect_elm(SHOT, p, (0, 1000))
    ramp, (start, stop, state) = found.spans
    assert ramp == (0.0, plasma, UNCERTAIN), "the ELMs at 200-280 ms: uncertain"
    assert found.info["ramp_events"] == 5
    assert state == PRESENT
    assert start == pytest.approx(295, abs=1) and stop == pytest.approx(405, abs=1)


def test_each_method_records_only_the_constants_it_uses():
    rules = {event: method.rule for event, method in spans.METHODS.items()}
    start = rules["sawtooth_oscillation"]["start"]
    assert start["ip_fraction"] == spans.RAMP_FRACTION
    assert start["fallback_ms"] == spans.RAMP_FALLBACK_MS
    assert rules["sawtooth_oscillation"]["max_gap_ms"] == spans.SAWTOOTH_MAX_GAP_MS
    assert rules["edge_localized_mode"]["max_gap_ms"] == spans.ELM_MAX_GAP_MS
    assert rules["edge_localized_mode"]["start"] == start
    assert "start" not in rules["high_confinement_mode"]
    assert "max_gap_ms" not in rules["high_confinement_mode"]
    assert rules["neoclassical_tearing_mode"] == {}
    everything = json.dumps(rules)
    assert "elm_max_gap_ms" not in everything and "sawtooth_max_gap" not in everything
    assert "crash" not in rules["edge_localized_mode"]


def test_the_sawtooth_rule_records_v3():
    method = spans.METHODS["sawtooth_oscillation"]
    assert (method.name, method.inputs) == ("ece_sawtooth", ("ece", "sxr", "ip"))
    assert method.rule["crash"].startswith("v3: heuristics.sawtooth_events_v3")
    constants = method.rule["crash_constants"]
    assert constants == heuristics.SAWTOOTH_V3_CONSTANTS
    assert json.loads(json.dumps(constants)) == constants, "as the meta records it"
    assert constants["z_drop"] == heuristics.Z_DROP
    assert constants["collapse_guard_ms"] == heuristics.COLLAPSE_GUARD_MS
    assert constants["diags"] == ["ece", "sxr"]
    assert constants["sxr_fans"][0] == "SX90RM1F"


# ------------------------------------------------ sawtooth, v3's diagnostics
#
# `read` stands in for the corpus: the ECE array of `v3_rows`, whose crashes
# are `V3_CRASHES_MS` (300.3-1750.3 ms), and a 320-row SXR array over
# 500-2500 ms with a train on SX90RP1F from 802.3 ms, 2 ms after an ECE crash.

SXR_CRASHES_MS = 802.3 + 50.0 * np.arange(30)
#: The SX90RP1F chords `_sxr` lights, and the dropping block in them.
SXR_LIT = range(4, 28)
SXR_BLOCK = (12, 17)


def _sxr(quench_ms=None):
    """`(t_s, y)`: 320 SXR rows. SX90RM1F (192-223) has 7 chords lit and 3 more
    finite over 40 % of the record; SX90RP1F (256-287) carries `v3_rows`' train
    on its chords 4-27 with no heat pulse; the rest are dark. `quench_ms`: all
    of SX90RP1F falls by 90 % there and recovers over 30 ms."""
    t_s, fan = v3_rows(
        n_channels=len(SXR_LIT),
        centre=10.0,
        core=(SXR_BLOCK[0] - SXR_LIT[0], SXR_BLOCK[1] - SXR_LIT[0]),
        pulse=(),
        crashes_ms=SXR_CRASHES_MS,
        drop=0.1,  # half ECE's: where the two merge, ECE's crash is the larger
        t_ms=(500.0, 2500.0),
        seed=1,
    )
    if quench_ms is not None:
        t_ms = t_s * 1e3
        ramp = np.clip((t_ms - quench_ms) / 30.0, 0.0, 1.0)
        fan = fan * np.where(t_ms < quench_ms, 1.0, 0.1 + 0.9 * ramp).astype(np.float32)
    y = np.full((320, t_s.size), np.nan, dtype=np.float32)
    y[192:199] = 1.0
    y[199:202, : int(0.4 * t_s.size)] = 1.0
    y[256 + SXR_LIT[0] : 256 + SXR_LIT[-1] + 1] = fan
    return t_s, y


def _reads(monkeypatch, **groups):
    """`spans.read` over `groups` ({group: (t_s, y)}); the rows each call asked
    for are appended to the list returned."""
    asked = []

    def read(shot, group, paths, channels=None):
        if group not in groups:
            raise NoDataError(
                f"shot {shot}: no {group!r} in the corpus or the raw cache"
            )
        t_s, y = groups[group]
        rows = None if channels is None else list(channels)
        asked.append((group, rows))
        return t_s, y if rows is None else y[rows]

    monkeypatch.setattr(spans, "read", read)
    return asked


def test_v3_drafts_from_the_ece_array_alone(tmp_path, monkeypatch):
    _reads(monkeypatch, ece=v3_rows())
    found = spans.detect_sawtooth(SHOT, tree.paths(tmp_path))
    [(start, stop, state)] = found.spans
    assert state == PRESENT
    assert start == pytest.approx(V3_CRASHES_MS[0] - spans.PAD_MS, abs=1)
    assert stop == pytest.approx(V3_CRASHES_MS[-1] + spans.PAD_MS, abs=1)
    assert found.measured == ((0.0, pytest.approx(1999.9)),)
    assert found.info == {
        "diagnostics": ["ece"],
        "crashes": {"ece": 30},
        "collapses_ms": [],
        "not_run": {"sxr": NO_SXR},
    }


def test_v3_drafts_from_the_first_lit_sxr_fan_alone(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    asked = _reads(monkeypatch, sxr=_sxr())
    name, _, y, chords = spans.sxr_fan(SHOT, p)
    assert name == "SX90RP1F", "SX90RM1F has 7 chords lit over half the record"
    assert chords.tolist() == list(SXR_LIT) and y.shape[0] == len(SXR_LIT)
    assert asked == [("sxr", list(range(192, 224))), ("sxr", list(range(256, 288)))]
    found = spans.detect_sawtooth(SHOT, p)
    [(start, stop, state)] = found.spans
    assert state == PRESENT
    assert start == pytest.approx(SXR_CRASHES_MS[0] - spans.PAD_MS, abs=1)
    assert stop == pytest.approx(SXR_CRASHES_MS[-1] + spans.PAD_MS, abs=1)
    assert found.measured == ((500.0, pytest.approx(2499.9)),)
    no_ece = f"NoDataError: shot {SHOT}: no 'ece' in the corpus or the raw cache"
    assert found.info == {
        "diagnostics": ["sxr"],
        "sxr_fan": "SX90RP1F",
        "crashes": {"sxr": 30},
        "collapses_ms": [],
        "not_run": {"ece": no_ece},
    }
    [crashes], _, _ = spans._sawtooth_crashes(SHOT, p)
    one = crashes.events[0]
    assert (one.diag, one.attrs["fan"]) == ("sxr", "SX90RP1F")
    got = (one.attrs["inversion_channel_lo"], one.attrs["inversion_channel_stop"])
    assert got == SXR_BLOCK, "in the fan's own chord numbers"
    assert "pulse_channel_lo" not in one.attrs


def test_v3_takes_the_union_of_ece_and_sxr_over_both_coverages(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    _reads(monkeypatch, ece=v3_rows(), sxr=_sxr())
    found = spans.detect_sawtooth(SHOT, p)
    [(start, stop, state)] = found.spans
    assert state == PRESENT
    assert start == pytest.approx(V3_CRASHES_MS[0] - spans.PAD_MS, abs=1)
    assert stop == pytest.approx(SXR_CRASHES_MS[-1] + spans.PAD_MS, abs=1)
    assert found.measured == ((0.0, pytest.approx(2499.9)),), "ECE's, then SXR's"
    assert found.info == {
        "diagnostics": ["ece", "sxr"],
        "sxr_fan": "SX90RP1F",
        "crashes": {"ece": 30, "sxr": 30},
        "collapses_ms": [],
    }
    both, _, _ = spans._sawtooth_crashes(SHOT, p)
    union = heuristics.sawtooth_events_v3(both)
    # SXR's crashes 2 ms after ECE's merge into them; its last ten are its own.
    assert [e.diag for e in union] == ["ece"] * 30 + ["sxr"] * 10


def test_a_collapse_on_sxr_drops_ece_crashes_after_it(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    _reads(monkeypatch, ece=v3_rows(), sxr=_sxr(quench_ms=610.3))
    found = spans.detect_sawtooth(SHOT, p)
    assert found.info["collapses_ms"] == [610.5]
    assert found.info["crashes"] == {"ece": 30, "sxr": 30}, "each keeps its own"
    # 650.3-900.3 on ECE are within 300 ms of it: the run breaks there.
    (a, b, _), (c, d, _) = found.spans
    assert a == pytest.approx(300.3 - spans.PAD_MS, abs=1)
    assert b == pytest.approx(600.3 + spans.PAD_MS, abs=1)
    assert c == pytest.approx(950.3 - spans.PAD_MS, abs=1)
    assert d == pytest.approx(SXR_CRASHES_MS[-1] + spans.PAD_MS, abs=1)


def test_a_broken_diagnostic_is_left_out_and_the_other_drafts(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    t_s, y = _sxr()
    _reads(monkeypatch, ece=v3_rows(), sxr=(t_s, y[:200]))  # short of every fan
    found = spans.detect_sawtooth(SHOT, p)
    assert found.info["diagnostics"] == ["ece"]
    assert found.info["not_run"]["sxr"].startswith("IndexError: ")
    [(start, _, _)] = found.spans
    assert start == pytest.approx(V3_CRASHES_MS[0] - spans.PAD_MS, abs=1)
    ece_t, ece_y = v3_rows()
    _reads(monkeypatch, ece=(ece_t, ece_y[:47]), sxr=_sxr())  # not the 48 channels
    found = spans.detect_sawtooth(SHOT, p)
    assert found.info["diagnostics"] == ["sxr"]
    assert found.info["not_run"]["ece"].startswith("ValueError: ")
    _reads(monkeypatch, ece=(ece_t, ece_y[:47]), sxr=(t_s, y[:200]))
    with pytest.raises(NoDataError, match="ece: ValueError: .*; sxr: IndexError: "):
        spans.detect_sawtooth(SHOT, p)


def test_v3_with_neither_diagnostic_is_no_data(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    _reads(monkeypatch)
    with pytest.raises(NoDataError, match="no 'ece'.*; .*no 'sxr'"):
        spans.detect_sawtooth(SHOT, p)
    t_s, y = _sxr()
    y[256:288] = np.nan  # SX90RP1F dark: no fan has 8 chords lit
    _reads(monkeypatch, sxr=(t_s, y))
    with pytest.raises(NoDataError, match="no SXR fan has 8 finite chords"):
        spans.detect_sawtooth(SHOT, p)


GOLD_EVENT = "sawtooth_oscillation"


def _gold_tree(p):
    """A roster of two gold shots and one not, and saved labels for 1 and 3."""
    event_dir = p.label_tables / GOLD_EVENT
    (event_dir / "review").mkdir(parents=True)
    pd.DataFrame(
        {"shot": [1, 2, 3], "tier": ["gold", "gold", "unverified"], "holdout": False}
    ).to_csv(event_dir / "shots.csv", index=False)
    rows = [
        [1, 0, 0, 100, ""],
        [1, 1, 100, 300, ""],
        [1, 0, 300, 1000, ""],
        [3, 1, 0, 1000, ""],
    ]
    pd.DataFrame(
        rows, columns=["shot", "category", "t_start", "t_end", "confidence"]
    ).to_csv(event_dir / "review" / "labels.csv", index=False)
    return event_dir


def _gold_method(seen):
    def detect(shot, paths, window=None):
        seen[shot] = window
        return spans.Found(((100, 200, PRESENT),), ((0.0, 1000.0),))

    return dataclasses.replace(spans.METHODS[GOLD_EVENT], detect=detect)


def test_the_drafts_are_scored_on_the_gold_shots_frame_by_frame(tmp_path):
    p = tree.paths(tmp_path)
    _gold_tree(p)
    seen = {}
    got = spans.gold(_gold_method(seen), p)
    assert seen == {1: (0, 1000)}, "each gold label's own window; shot 3 is not gold"
    assert (got["shots"], got["tp"], got["fp"], got["fn"], got["tn"]) == (
        1,
        10,
        0,
        10,
        80,
    )
    assert (got["precision"], got["recall"]) == (1.0, 0.5)
    assert got["missing"] == {"2": "no gold label"}
    assert got["reference"].endswith("review/labels.csv")


def test_the_gold_score_goes_into_the_meta_and_stays(tmp_path, monkeypatch, capsys):
    p = tree.paths(tmp_path)
    tree.use_env(monkeypatch, p)
    tree.cohort(p, [tree.queue_row(7, 0)])
    _gold_tree(p)
    monkeypatch.setitem(spans.METHODS, GOLD_EVENT, _gold_method({}))
    assert spans.main(["--event", GOLD_EVENT, "--gold"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["gold"]["recall"] == 0.5
    path = suggestions.table_path(p, GOLD_EVENT, "ece_sawtooth", "v1")
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    assert meta["gold"]["shots"] == 1 and meta["gold"]["git_sha"] == meta["git_sha"]
    assert spans.main(["--event", GOLD_EVENT, "--force"]) == 0
    capsys.readouterr()
    again = json.loads(path.with_suffix(".meta.json").read_text())
    assert again["gold"] == meta["gold"], "a rerun without --gold keeps the score"
    other = p.label_tables / "other.csv"
    pd.DataFrame(
        [[1, 1, 0, 1000, ""]],
        columns=["shot", "category", "t_start", "t_end", "confidence"],
    ).to_csv(other, index=False)
    assert spans.main(["--event", GOLD_EVENT, "--gold", str(other)]) == 0
    assert json.loads(capsys.readouterr().out)["gold"]["reference"] == str(other)
