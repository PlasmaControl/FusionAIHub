"""The robust y-range on the H-mode D-alpha row and the sawtooth SXR row.

Synthetic corpus files with a plasma window each. The page scales a trace row to
the samples it holds, so one spike after the plasma flattened the row; both rows
now sit inside `_shared.robust_limits` over the window, as the ELM D-alpha rows
do. The sawtooth ECE rows are not clipped: each of their samples is the median
of its 0.05 ms already, and a clip on top would hide a real crash's extreme.
"""

from __future__ import annotations

import numpy as np
import pytest

from labeler.events.panels import _shared
from labeler.events.panels import edge_localized_mode as elm
from labeler.events.panels import high_confinement_mode as hmode
from labeler.events.panels import sawtooth_oscillation as saw

from . import editor_tree as tree

SHOT = 201236
#: D-alpha: a 300 ms record, 3000 samples, whose plasma is 50-200 ms. SXR: 400 ms,
#: 4000 samples, plasma 0-300 ms. Each spike is 50 samples after the plasma
#: (250-255 ms, 340-345 ms): over 1 % of the record, so a range taken over the
#: whole record would take it in, and only the plasma window's clips it.
DALPHA_WINDOW = (50.0, 200.0)
DALPHA_SPIKE = slice(2500, 2550)
SXR_WINDOW = (0.0, 300.0)
SXR_SPIKE = slice(3400, 3450)
#: The SXR chords that crash every 50 ms, and so the four the row draws.
CRASHING = (3, 5, 20, 25)
SXR_TITLE = "SXR SX90RP1F, the 4 chords with the most crash-like drops"


def _dalpha(p, *, spike: bool) -> np.ndarray:
    """FS01-FS03 at 10 kHz, 1 with noise; with `spike`, FS02 reads 50 after the
    plasma. Returns the three channels as the corpus holds them."""
    t = tree.times(0.0, 300.0, 10_000)
    y = (1.0 + tree.noise(3, t, 0.01)).astype(np.float32)
    if spike:
        y[1, DALPHA_SPIKE] = 50.0
    fs = np.full((104, len(t)), np.nan)
    fs[:3] = y
    tree.write(p.corpus_file(SHOT), {"filterscopes": (t, fs)})
    tree.cohort(p, [tree.queue_row(SHOT, 0, window=DALPHA_WINDOW)])
    return y


def _sxr(p, *, spike: bool) -> np.ndarray:
    """SX90RP1F lit, the other fans dark: chords 3, 5, 20 and 25 crash every 50 ms,
    the rest sit still, all near 1; with `spike`, chord 5 reads 50 after the
    plasma. Returns the fan's 32 chords as the corpus holds them."""
    t = tree.times(0.0, 400.0, 10_000)
    fan = 1.0 + tree.noise(saw.CHORDS, t, 1e-4)
    fan[list(CRASHING)] += 0.05 * (t % 50.0) / 50.0
    fan = fan.astype(np.float32)
    if spike:
        fan[5, SXR_SPIKE] = 50.0
    y = np.full((320, len(t)), np.nan)
    first = dict(saw.SXR_ARRAYS)["SX90RP1F"]
    y[first : first + saw.CHORDS] = fan
    tree.write(p.corpus_file(SHOT), {"sxr": (t, y)})
    tree.cohort(p, [tree.queue_row(SHOT, 0, window=SXR_WINDOW)])
    return fan


def _limits(panel, raw, window) -> np.ndarray:
    """`(2, C, 1)`: the `robust_limits` of the raw channels over the window, as the
    float32 the panel holds them, ready to compare with its `(C, T)` values."""
    return _shared.robust_limits(panel.x, raw, window).astype(np.float32)[:, :, None]


def test_clipped_is_shared_and_the_elm_panel_uses_it():
    assert _shared.CLIPPED == ", clipped to its plasma range"
    assert elm.CLIPPED is _shared.CLIPPED


def test_a_dalpha_spike_is_clipped_to_the_plasma_range_and_the_title_says_so(
    tmp_path, monkeypatch
):
    p = tree.paths(tmp_path)
    tree.no_fetch(monkeypatch)
    y = _dalpha(p, spike=True)
    [row] = hmode.dalpha_panel(SHOT, paths=p)
    low, high = _limits(row, y, DALPHA_WINDOW)
    assert row.legend == ["FS01", "FS02", "FS03"] and row.y.shape == y.shape
    assert (row.y >= low).all() and (row.y <= high).all(), "inside robust_limits"
    assert high[1, 0] < 2, "against the spike's 50"
    assert (row.y[1, DALPHA_SPIKE] == high[1]).all(), "the spike sits on the edge"
    rest = np.ones(y.shape, dtype=bool)
    rest[1, DALPHA_SPIKE] = False
    assert np.array_equal(row.y[rest], y[rest]), "nothing else moves"
    assert row.title == "D-alpha filterscopes" + _shared.CLIPPED


def test_a_quiet_dalpha_is_drawn_as_it_is(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    tree.no_fetch(monkeypatch)
    y = _dalpha(p, spike=False)
    [row] = hmode.dalpha_panel(SHOT, paths=p)
    assert np.array_equal(row.y, y)
    assert row.title == "D-alpha filterscopes"


def test_an_sxr_spike_is_clipped_to_the_plasma_range_and_the_title_says_so(
    tmp_path, monkeypatch
):
    p = tree.paths(tmp_path)
    tree.no_fetch(monkeypatch)
    fan = _sxr(p, spike=True)
    [row] = saw.sxr_panels(SHOT, paths=p)
    assert row.legend == [f"SX90RP1F{c + 1:02d}" for c in CRASHING]
    chosen = fan[list(CRASHING)]
    low, high = _limits(row, chosen, SXR_WINDOW)
    assert row.y.shape == chosen.shape
    assert (row.y >= low).all() and (row.y <= high).all(), "inside robust_limits"
    assert high[1, 0] < 2, "against the spike's 50"
    assert (row.y[1, SXR_SPIKE] == high[1]).all(), "the spike sits on the edge"
    rest = np.ones(chosen.shape, dtype=bool)
    rest[1, SXR_SPIKE] = False
    assert np.array_equal(row.y[rest], chosen[rest]), "nothing else moves"
    assert row.title == SXR_TITLE + _shared.CLIPPED
    # Clipped over the record, as the chords are chosen: a view cuts the same row.
    [view] = saw.sxr_panels(SHOT, paths=p, t_range=(330.0, 350.0))
    keep = (row.x >= 330.0) & (row.x <= 350.0)
    assert np.array_equal(view.y, row.y[:, keep]) and view.title == row.title
    assert (view.y[1] <= high[1]).all()


def test_a_quiet_sxr_fan_is_drawn_as_it_is(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    tree.no_fetch(monkeypatch)
    fan = _sxr(p, spike=False)
    [row] = saw.sxr_panels(SHOT, paths=p)
    assert np.array_equal(row.y, fan[list(CRASHING)])
    assert row.title == SXR_TITLE


def test_an_ece_burst_wider_than_the_median_is_kept_and_not_called_clipped(
    tmp_path, monkeypatch
):
    p = tree.paths(tmp_path)
    tree.no_fetch(monkeypatch)
    t = tree.times(0.0, 100.0, 500_000)
    y = 1.0 + tree.noise(36, t, 0.01)
    y[21, 20_000:20_100] = 50.0  # at 40 ms: 100 samples, 0.2 % of the record
    tree.write(p.corpus_file(SHOT), {"ece": (t, y)})
    tree.cohort(p, [tree.queue_row(SHOT, 0, window=(20.0, 80.0))])
    rows = saw.ece_panels(SHOT, paths=p)
    first = rows[0]
    assert first.title == (
        "ECE Te, inversion side A, ch 20-23 "
        "(0.05 ms median; q=1 mapping unavailable)"
    )
    assert not any(row.title.endswith(_shared.CLIPPED) for row in rows)
    ceiling = _shared.robust_limits(first.x, first.y, (20.0, 80.0))[1]
    assert ceiling[1] < 2, "a robust clip would have cut the burst to this"
    assert first.y[1].max() == pytest.approx(50.0), "a burst is a crash's extreme"
