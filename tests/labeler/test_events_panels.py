"""The per-event panel registry."""

from __future__ import annotations

import numpy as np
import pytest

from labeler.config import Paths
from labeler.events import panels as registry
from labeler.events.panels import alfven_eigenmode as ae
from labeler.features.store import FeatureArray

#: Seconds on disk, like the corpus - deliberately not round thousands, so a
#: bug that forgets the `* 1000.0` conversion cannot pass by coincidence.
_SECONDS = np.array([100.0, 101.0, 102.0, 103.0, 104.0])

#: One array per `_generic.TRACES` entry, each a distinct `(C, T)` shape so a
#: swapped or dropped trace would show up as the wrong channel count too.
_SIGNALS = {
    "ip": np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]),
    "betan": np.array([[10.0, 11.0, 12.0, 13.0, 14.0], [20.0, 21.0, 22.0, 23.0, 24.0]]),
    "pinj_total": np.array(
        [
            [100.0, 200.0, 300.0, 400.0, 500.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [9.0, 8.0, 7.0, 6.0, 5.0],
        ]
    ),
}


def _tmp_paths(tmp_path):
    """A `Paths` rooted entirely under `tmp_path` - never the real corpus."""
    return Paths(
        root=tmp_path / "root",
        corpus=tmp_path / "corpus",
        text_root=tmp_path / "text",
        logs_jsonl=tmp_path / "logs.jsonl",
        label_tables=tmp_path / "events",
        raw_cache=tmp_path / "raw",
    )


def test_generic_panels_converts_units_and_clips_t_range(monkeypatch, tmp_path):
    """The real `_generic.panels()`, not the dispatch-only lambda stub.

    `TRACES` fixes the trace order and labels; the stubbed signal read
    proves the seconds-on-disk -> milliseconds-on-screen conversion and
    `t_range` clip actually run, rather than only checking that *some*
    builder was called.
    """
    calls = []

    def fake_read_feature(path, name):
        calls.append((path, name))
        return FeatureArray(x=_SECONDS.copy(), y=_SIGNALS[name].copy())

    monkeypatch.setattr(registry._generic, "read_feature", fake_read_feature)
    paths = _tmp_paths(tmp_path)

    built = registry._generic.panels(12345, paths=paths)

    assert [panel.title for panel in built] == ["ip", "betan", "pinj_total"]
    assert [panel.ylabel for panel in built] == ["A", "", "kW"]
    assert calls == [
        (paths.features_file(12345), "ip"),
        (paths.features_file(12345), "betan"),
        (paths.features_file(12345), "pinj_total"),
    ]

    expected_ms = _SECONDS * 1000.0
    for panel, name in zip(built, ("ip", "betan", "pinj_total"), strict=True):
        np.testing.assert_allclose(panel.x, expected_ms)
        np.testing.assert_array_equal(panel.y, _SIGNALS[name])

    # t_range is milliseconds, applied after the conversion, so a window
    # that would clip nothing in seconds must clip correctly here.
    clipped = registry._generic.panels(
        12345, t_range=(101_000.0, 103_000.0), paths=paths
    )
    expected_clipped_ms = np.array([101_000.0, 102_000.0, 103_000.0])
    for panel, name in zip(clipped, ("ip", "betan", "pinj_total"), strict=True):
        np.testing.assert_allclose(panel.x, expected_clipped_ms)
        np.testing.assert_array_equal(panel.y, _SIGNALS[name][:, 1:4])


def test_an_unregistered_event_falls_back_to_the_generic_builder(monkeypatch):
    calls = []
    monkeypatch.setattr(
        registry._generic,
        "panels",
        lambda shot, *, t_range=None, paths=None: calls.append(shot) or [],
    )
    assert registry.build("no_such_event", 42) == []
    assert calls == [42]


def test_every_registered_event_has_guidance():
    for event in registry.BUILDERS:
        assert registry.guidance(event).strip(), event


def test_guidance_for_an_unregistered_event_says_it_is_generic():
    assert "generic" in registry.guidance("no_such_event").lower()


def test_the_registry_covers_the_events_with_bespoke_panels():
    assert set(registry.BUILDERS) == {
        "alfven_eigenmode",
        "fishbone",
        "minimum_safety_factor",
        "sawtooth_oscillation",
    }


def test_crosspower_takes_its_rate_from_the_span_not_a_median_diff():
    """A float32 time vector quantises its spacing at t ~ 3 s.

    Successive differences of such a vector are wrong by percents and in a
    biased direction; the span is exact. A 120 kHz tone must land at 120 kHz
    on a time base that starts at 3000 ms, not merely on one starting at 0.
    """
    rate = 1_000_000.0
    n = 200_000
    t_ms = (3000.0 + np.arange(n) / rate * 1000.0).astype("float32")
    tone = np.sin(2 * np.pi * 120_000.0 * np.arange(n) / rate).astype("float32")
    freq_khz, _, power = ae.crosspower(t_ms.astype("float64"), tone, tone)
    peak = freq_khz[np.argmax(power.mean(axis=1))]
    assert abs(peak - 120.0) < 1.0


def test_crosspower_caps_its_time_bins():
    rate = 1_000_000.0
    n = 400_000
    t_ms = np.arange(n) / rate * 1000.0
    noise = np.random.default_rng(0).normal(size=n).astype("float32")
    _, t_out, power = ae.crosspower(t_ms, noise, noise, max_bins=250)
    assert power.shape[1] <= 250
    assert len(t_out) == power.shape[1]


def test_crosspower_averages_power_before_taking_the_log():
    """A geometric mean would be dragged down by the quiet bins in a block.

    That is exactly what suppresses a short burst - the thing the panel
    exists to show. One loud block among quiet ones must survive averaging.
    """
    rate = 1_000_000.0
    n = 200_000
    t_ms = np.arange(n) / rate * 1000.0
    rng = np.random.default_rng(1)
    signal = rng.normal(scale=1e-3, size=n)
    signal[100_000:110_000] += 5.0 * np.sin(
        2 * np.pi * 120_000.0 * np.arange(10_000) / rate
    )
    freq_khz, t_out, power = ae.crosspower(t_ms, signal, signal, max_bins=40)
    band = (freq_khz > 110.0) & (freq_khz < 130.0)
    profile = power[band].mean(axis=0)
    burst = np.argmax(profile)
    assert 95.0 < t_out[burst] < 115.0
    assert profile[burst] > np.median(profile) + 1.0


def test_alfven_panels_are_three_crosspower_heatmaps(monkeypatch):
    """The whole unit seam, on a shot-like time base rather than one at zero.

    `raw_signal` hands back MILLISECONDS and `t_range` is milliseconds, so a
    time base starting at 0 hides both halves of the bug this plan has hit
    twice: a dropped window offset and a second, spurious seconds-to-ms
    conversion both leave the panels sitting at t ~ 0 on such a base.
    """
    rate = 1_000_000.0
    n = 60_000
    start_ms = 2000.0
    fake = FeatureArray(
        x=start_ms + np.arange(n) / rate * 1000.0,
        y=np.random.default_rng(2).normal(size=(4, n)).astype("float32"),
        attrs={"tier": "cache"},
    )
    seen = []

    def fake_raw_signal(shot, group, **kwargs):
        seen.append((shot, group, kwargs))
        return fake

    monkeypatch.setattr(ae, "raw_signal", fake_raw_signal)
    window = (start_ms, start_ms + 60.0)
    built = registry.build("alfven_eigenmode", 178642, t_range=window)
    assert len(built) == 3
    assert {p.kind for p in built} == {"heatmap"}
    assert all(p.bands == [(80.0, 250.0)] for p in built)
    assert all(p.ylabel == "kHz" for p in built)

    # The pairing IS the diagnostic - the reference chord against each
    # vertical one - so the titles pin channel identity, not just the count.
    assert [panel.title for panel in built] == [
        "CO2 crosspower DENR0UF x DENV1UF",
        "CO2 crosspower DENR0UF x DENV2UF",
        "CO2 crosspower DENR0UF x DENV3UF",
    ]

    # Every panel must be drawn inside the window that was fetched.
    end_ms = start_ms + (n - 1) / rate * 1000.0
    for panel in built:
        assert panel.x.min() >= start_ms
        assert panel.x.max() <= end_ms

    # The window has to reach the fetch: without it a reviewer wanting a
    # 60 ms look waits on a whole shot of PTDATA.
    assert seen == [(178642, "co2", {"t_range": window, "paths": None})]


def test_crosspower_rejects_a_degenerate_window():
    """An out-of-range `t_range` slices the signal to nothing.

    A zero span divides to nan and silently draws a blank panel; an empty one
    used to die on an IndexError far from the cause.
    """
    one = np.array([1.0])
    two = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="more than one instant"):
        ae.crosspower(np.array([]), np.array([]), np.array([]))
    with pytest.raises(ValueError, match="more than one instant"):
        ae.crosspower(np.array([2000.0]), one, one)
    with pytest.raises(ValueError, match="more than one instant"):
        ae.crosspower(np.array([2000.0, 2000.0]), two, two)
