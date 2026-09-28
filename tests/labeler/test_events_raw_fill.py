"""Filling the raw cache ahead of a review: only what neither tier holds."""

from __future__ import annotations

import json

import numpy as np

from labeler.events import raw
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
        {"shot": 5, "co2": "fetched"},
        {"shot": 4, "co2": "fetched"},
        {"shots": 2, "fetched": 2},
    ]
    assert raw.main(["--event", "high_confinement_mode", "--limit", "1"]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == '{"shots": 1, "cache": 1}'
    assert len(calls) == 2
