"""The per-event panel registry."""

from __future__ import annotations

from labeler.events import panels as registry


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
