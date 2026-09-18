"""The per-event panel registry."""

from __future__ import annotations

import numpy as np

from labeler.config import Paths
from labeler.events import panels as registry
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
