"""The assessed window and the flat-top, from Ip traces drawn by hand."""

from __future__ import annotations

import json

import numpy as np
import pytest

from labeler.config import Paths
from labeler.events.catalog import window
from labeler.events.catalog.window import (
    assessed_window,
    fetch,
    flattop_s,
    read_log,
    summarise,
)
from labeler.events.verify import NoDataError
from labeler.features.store import FeatureArray
from shot_design.shotdb.select import flattop_from_ip


def _trace(stop=4000.0, peak=1e6):
    """Up over 0-500 ms, flat to 2000 ms, down by 2500 ms; samples at x.2 and x.7."""
    t = np.arange(-1000.3, stop, 0.5)
    return t, peak * np.interp(t, [0, 500, 2000, 2500], [0, 1, 1, 0])


def test_the_window_is_rounded_inward_whatever_the_sign():
    t, ip = _trace()  # 50 kA at 25 ms and 2475 ms; samples 25.2 ... 2474.7
    assert assessed_window(t, ip) == (26, 2474)
    assert assessed_window(t, -ip) == (26, 2474)


def test_a_sample_on_a_whole_ms_counts_whichever_way_its_float_rounds():
    t = np.array([5.0, 6.0, 7.0, 8.0])
    for noise in (-1e-7, 0.0, 1e-7):  # float64 steps, float32 seconds
        assert assessed_window(t + noise, [0.0, 6e4, 6e4, 0.0]) == (6, 7)


def test_pickup_away_from_the_plasma_does_not_move_the_window():
    t, ip = _trace()
    ip[(t >= -900) & (t < -892)] = 60e3  # the pre-magnetisation pulse
    ip[(t >= 3000) & (t < 3005)] = -60e3
    assert assessed_window(t, ip) == (26, 2474)


def test_short_gaps_are_bridged_and_long_ones_split():
    t, ip = _trace()
    short = ip.copy()
    short[(t >= 30) & (t < 35)] = 0.0
    short[(t >= 1000) & (t < 1001)] = np.nan
    assert assessed_window(t, short) == (26, 2474)
    long = ip.copy()
    long[(t >= 100) & (t < 120)] = 0.0
    assert assessed_window(t, long) == (121, 2474)  # the longer stretch


def test_no_plasma_no_window():
    t, ip = _trace(peak=40e3)
    assert assessed_window(t, ip) is None
    assert assessed_window(t, np.full_like(t, np.nan)) is None
    assert assessed_window([], []) is None


@pytest.mark.parametrize("sign", [1, -1])
def test_a_quench_ends_at_the_plasmas_own_sign(sign):
    t, _ = _trace()
    ip = 1e6 * np.interp(t, [0, 500, 2000, 2005, 3000], [0, 1, 1, -0.3, 0])
    assert assessed_window(t, sign * ip) == (26, 2003)


@pytest.mark.parametrize("dip", ["one_sample", "six_ms"])
def test_a_brief_dip_does_not_split_the_flattop(dip):
    t, ip = _trace()
    if dip == "one_sample":
        ip[np.argmin(abs(t - 1000.2))] = 0.79e6
    else:
        ip[(t >= 1000) & (t < 1006)] = 0.78e6
    assert flattop_s(t, ip, (26, 2474)) == pytest.approx(1.6995)


def test_the_noisy_flattop_agrees_across_native_rates():
    rng = np.random.default_rng(0)
    measured = []
    for start, dt in [(-1000.03, 0.05), (-1000.3, 0.5)]:
        t = np.arange(start, 4000.0, dt)
        ip = 1e6 * np.interp(t, [0, 500, 2000, 2500], [0, 1, 1, 0])
        ip += rng.normal(0, 4e3, t.size)
        measured.append(flattop_s(t, ip, assessed_window(t, ip)))
    assert measured[0] == pytest.approx(measured[1], abs=0.002)


def test_the_flattop_is_measured_inside_the_window():
    t, ip = _trace()  # at least 0.8 MA from 400.2 to 2099.7 ms
    assert flattop_s(t, ip, (26, 2474)) == pytest.approx(1.6995)
    far_t, far_ip = _trace(stop=40_000.0)  # the same shot, a longer record
    assert flattop_s(far_t, far_ip, (26, 2474)) == pytest.approx(1.6995)
    assert flattop_from_ip(far_t / 1000, far_ip) > 2.0  # the whole record's


def test_a_summary_line_says_what_was_measured():
    t, ip = _trace()
    assert summarise(7, t, ip) == {
        "shot": 7,
        "n": t.size,
        "dt_ms": 0.5,
        "status": "ok",
        "window_start_ms": 26,
        "window_end_ms": 2474,
        "flattop_s": 1.6995,
        "ip_peak_ma": 1.0,
    }
    assert summarise(8, t, ip / 100)["status"] == "no_plasma"


def test_the_spacing_survives_the_raw_caches_float32_seconds():
    t = np.arange(-4000.0, 20_000.0, 0.05)  # 0.05 ms, as from about shot 189,000
    cached = (t / 1000).astype("float32").astype(float) * 1000
    assert np.median(np.diff(cached)) != pytest.approx(0.05, abs=1e-6)
    assert summarise(9, cached, np.full_like(cached, 1e6))["dt_ms"] == 0.05


def _fake_ip(calls):
    t, ip = _trace()

    def raw_signal(shot, group, *, paths):
        assert group == "ip"
        calls.append(shot)
        if shot == 3:
            raise NoDataError("fdp could not answer")
        scale = 0.01 if shot == 2 else 1.0
        return FeatureArray(x=t, y=(ip * scale)[None], attrs={"tier": "fetch"})

    return raw_signal


def test_a_rerun_measures_only_what_is_not_settled(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(window, "raw_signal", _fake_ip(calls))
    log, paths = tmp_path / "catalog" / "ip.jsonl", Paths(root=tmp_path)
    assert fetch([3, 1, 2], log, paths) == {"ok": 1, "no_plasma": 1, "error": 1}
    assert fetch([1, 2, 3], log, paths) == {"error": 1}
    assert calls == [1, 2, 3, 3]
    table = read_log(log)
    assert table[["shot", "status"]].values.tolist() == [
        [1, "ok"],
        [2, "no_plasma"],
        [3, "error"],
    ]
    assert table.loc[0, ["window_start_ms", "window_end_ms"]].tolist() == [26, 2474]
    assert table.loc[2, "error"] == "fdp could not answer"
    assert len(log.read_text().splitlines()) == 4


def test_an_unexpected_failure_is_logged_and_the_run_goes_on(tmp_path, monkeypatch):
    fake = _fake_ip([])

    def raw_signal(shot, group, *, paths):
        if shot == 5:
            raise OSError("No space left on device")
        return fake(shot, group, paths=paths)

    monkeypatch.setattr(window, "raw_signal", raw_signal)
    log = tmp_path / "ip.jsonl"
    assert fetch([5, 1], log, Paths(root=tmp_path)) == {"ok": 1, "error": 1}
    assert read_log(log).loc[1, "error"] == "OSError: No space left on device"


def test_workers_measure_the_same(tmp_path, monkeypatch):
    monkeypatch.setattr(window, "raw_signal", _fake_ip([]))
    log = tmp_path / "ip.jsonl"
    counts = fetch([1, 2, 3, 4], log, Paths(root=tmp_path), workers=2)
    assert counts == {"ok": 2, "no_plasma": 1, "error": 1}
    assert read_log(log)["status"].tolist() == ["ok", "no_plasma", "error", "ok"]


def test_an_empty_log_is_an_empty_table(tmp_path):
    table = read_log(tmp_path / "none.jsonl")
    assert table.empty and list(table.columns) == list(window.LOG_COLUMNS)


def test_the_command_logs_under_the_root(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(window, "raw_signal", _fake_ip([]))
    monkeypatch.setenv("LABELER_ROOT", str(tmp_path))
    shots = tmp_path / "shots.txt"
    shots.write_text("2\n1  # a comment\n\n3\n")
    assert window.main(["--shot-file", str(shots), "--workers", "1"]) == 0
    out = capsys.readouterr().out
    summary = json.loads(out[out.index("{\n") :])  # after the progress line
    assert summary["log"] == str(tmp_path / "catalog" / "ip.jsonl")
    assert summary["status"] == {"ok": 1, "no_plasma": 1, "error": 1}
    assert summary["flattop_at_least_1s"] == 1


@pytest.mark.live
def test_live_window_of_a_measured_shot(tmp_path):
    line = window.measure(200111, Paths(root=tmp_path))
    assert line["dt_ms"] == 0.05 and line["n"] == 480_256
    assert (line["window_start_ms"], line["window_end_ms"]) == (7, 5348)
    assert line["flattop_s"] == pytest.approx(3.534, abs=0.001)
