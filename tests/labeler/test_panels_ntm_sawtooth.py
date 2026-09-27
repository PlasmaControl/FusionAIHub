"""The tearing-mode and sawtooth editors' panels, on synthetic corpus files."""

from __future__ import annotations

import numpy as np
import pytest

from labeler.events import panels
from labeler.events.panels import neoclassical_tearing_mode as ntm
from labeler.events.panels import sawtooth_oscillation as saw
from labeler.events.review import panel_rows
from labeler.events.verify import NoDataError

from . import editor_tree as tree

SHOT = 201235


def _mirnov(p, n: int, *, khz: float = 10.0):
    """A mode cos(wt - n phi) on the MPI66M probes from 50 to 150 ms, in noise."""
    t = tree.times(0.0, 200.0, 500_000)
    y = tree.noise(29, t)
    on = (t > 50) & (t < 150)
    for row, phi in ntm.PROBES.items():
        y[row, on] += np.cos(2 * np.pi * khz * t[on] - n * np.deg2rad(phi))
    tree.write(p.corpus_file(SHOT), {"mirnov": (t, y)})


@pytest.mark.parametrize("n", [2, -3])
def test_the_n_strip_finds_the_mode_number_where_the_mode_is(tmp_path, n):
    p = tree.paths(tmp_path)
    _mirnov(p, n)
    power, strip = panels.build("neoclassical_tearing_mode", SHOT, paths=p)
    assert power.title == "MPI66M322D power" and power.y[-1] <= 30
    line = np.argmin(np.abs(power.y - 10.0))
    inside = (power.x > 60) & (power.x < 140)
    assert power.z[line, inside].mean() > power.z[line, ~inside].mean() + 20
    assert list(strip.y) == list(range(-4, 5)) and (strip.zmin, strip.zmax) == (0, 1)
    best = strip.y[np.argmax(strip.z[:, inside], axis=0)]
    assert (best == n).all() and strip.z[:, inside].max(axis=0).min() > 0.9
    outside = (power.x < 40) | (power.x > 160)
    assert not strip.z[:, outside].any(), "noise columns are not scored"


def test_ntm_rows_and_beta_n(tmp_path):
    p = tree.paths(tmp_path)
    _mirnov(p, 1)
    tree.features(p, SHOT, np.arange(0.0, 200.0, 20.0), np.linspace(1, 2, 10))
    titles = [x.title for x in panels.build("neoclassical_tearing_mode", SHOT, paths=p)]
    assert titles[-1] == "beta_N" and len(titles) == 3
    _grid, rows, _info = panel_rows.build("neoclassical_tearing_mode", SHOT, p)
    assert [row.y_units for row in rows] == ["kHz", "n", "β_N"]


def _sawtooth(p, *, ece=True, sxr=True):
    groups = {}
    if ece:
        t = tree.times(0.0, 100.0, 500_000)
        groups["ece"] = (t, 1.0 + tree.noise(48, t, 0.01))
    if sxr:
        t = tree.times(0.0, 100.0, 10_000)
        y = np.full((320, len(t)), np.nan)  # the SX90RM1F fan is dark
        first = dict(saw.SXR_ARRAYS)["SX90RP1F"]
        chords = np.arange(32)
        y[first : first + 32] = np.exp(-(((chords - 10.5) / 3) ** 2))[:, None]
        groups["sxr"] = (t, y)
    tree.write(p.corpus_file(SHOT), groups)


def test_sawteeth_draw_ece_and_the_brightest_sxr_chords(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    tree.no_fetch(monkeypatch)
    _sawtooth(p)
    built = panels.build("sawtooth_oscillation", SHOT, paths=p)
    assert [x.title for x in built][:4] == [
        f"ECE ch {a}-{a + 3}" for a in (20, 24, 28, 32)
    ]
    sxr = built[4]
    assert sxr.title == "SXR SX90RP1F, the 4 brightest chords"
    assert sxr.legend == [f"SX90RP1F{c}" for c in ("10", "11", "12", "13")]


def test_a_shot_without_ece_or_sxr_draws_the_other(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    tree.no_fetch(monkeypatch)
    _sawtooth(p, ece=False)
    [only] = panels.build("sawtooth_oscillation", SHOT, paths=p)
    assert only.title.startswith("SXR")
    other = tree.paths(tmp_path / "other")
    _sawtooth(other, sxr=False)
    assert len(panels.build("sawtooth_oscillation", SHOT, paths=other)) == 4
    none = tree.paths(tmp_path / "none")
    tree.write(none.corpus_file(SHOT), {"ip": ([0.0, 1.0], [[1.0, 1.0]])})
    with pytest.raises(NoDataError, match="no panels"):
        panel_rows.build("sawtooth_oscillation", SHOT, none)
