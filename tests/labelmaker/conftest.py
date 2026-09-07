"""Opt-in gating for the tests that fetch live data through fdp.

Everything else in this directory is hermetic: it must pass under a plain
`pixi run -e labelmaker python -m pytest tests/labelmaker`, with no server,
no token and no network. A live test is marked `live` and is skipped unless
it is asked for explicitly, either with `--run-live` or with
`LABELMAKER_FDP=1` (the environment switch the plan's command blocks use).

A live run also needs the `fdp run` wrapper, which supplies the server
configuration: without it PTDATA fails with `getservbyname failed for task
'PTSERVER'` and MDSplus with `TREE-E-FOPENR`.

    pixi run -e labelmaker fdp run python -m pytest tests/labelmaker \
        -q -W error --run-live

It is also where the fixtures more than one test module needs live: `wired`,
a synthetic archive and corpus, and `synth_mask`, the drawn coherent map the
event layer is measured against.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

#: Environment switch, equivalent to `--run-live`.
LIVE_ENV = "LABELMAKER_FDP"


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="run the tests that fetch real data through fdp",
    )


def pytest_configure(config) -> None:
    # Registered, not just used: `-W error` promotes pytest's
    # PytestUnknownMarkWarning to an error, so an unregistered marker would
    # fail the hermetic gate rather than merely warn.
    config.addinivalue_line(
        "markers",
        "live: fetches real data through fdp; needs `fdp run` and a SciToken",
    )


def pytest_collection_modifyitems(config, items) -> None:
    if config.getoption("--run-live") or os.environ.get(LIVE_ENV) == "1":
        return
    skip = pytest.mark.skip(
        reason="live fdp fetch is opt-in: --run-live or LABELMAKER_FDP=1"
    )
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """A synthetic archive and corpus, a fake model, and an empty data root.

    Lives here rather than in test_run.py so test_analyze.py can use it
    without re-importing another test module's fixture.
    """
    from labelmaker.models import registry

    from .test_run import _archive, _corpus, _fake_adapter

    monkeypatch.setattr(registry, "load_adapter", lambda slug: _fake_adapter())
    monkeypatch.setattr(registry, "verify_artifacts", lambda slug, d: None)
    monkeypatch.setattr(
        registry, "read_card",
        lambda slug: {"labelmaker": {"upstream": {"sha256": {"fake.h5": "00"}}}},
    )
    return {
        "archive": _archive(tmp_path),
        "corpus": _corpus(tmp_path),
        "root": tmp_path / "out",
    }


# ------------------------------------------------- the synthetic mask fixture

#: The synthetic channel `synth_mask` draws is a ZOOM pass of a 500 kHz
#: record: 0.1220703125 kHz per bin, 1.024 ms per column. Both axes come
#: from `events.masks`, so the fixture is on the pipeline's own grid and a
#: test can assert a frequency in kHz rather than in rows.
SYNTH_FS_HZ = 5.0e5
SYNTH_DECIM = 4
SYNTH_T = 2048
SYNTH_BIN_KHZ = 0.1220703125
SYNTH_HOP_S = 1.024e-3

#: Where each feature is drawn, in kHz and in columns. A band is three rows
#: centred on the row nearest the frequency; the columns are half-open.
SYNTH_PICKUP_KHZ = 1.95
SYNTH_EHO_KHZ = 8.0
SYNTH_H2_KHZ = 16.0
SYNTH_H3_KHZ = 24.0
SYNTH_SPLIT_KHZ = 44.0
SYNTH_EHO_COLS = ((200, 600), (620, 1200))       # a 20-column gap: one track
SYNTH_HARMONIC_COLS = (200, 1200)
SYNTH_SPLIT_COLS = ((1300, 1400), (1460, 1560))  # a 60-column gap: two tracks

#: The fishbone: `SYNTH_CHIRP_N` columns starting at `SYNTH_CHIRP_COL0`,
#: each a bar of `SYNTH_CHIRP_ROWS` rows whose centroid sits on the line
#: `SYNTH_CHIRP_F0_KHZ + SYNTH_CHIRP_KHZ_PER_MS * (t - t0)`. Eleven columns
#: of 1.024 ms is 10.24 ms, over which -0.8 kHz/ms takes 20.0 kHz down to
#: 11.8; 11 x 24 = 264 pixels, comfortably over `tracks.MIN_AREA`.
SYNTH_CHIRP_COL0 = 1700
SYNTH_CHIRP_N = 11
SYNTH_CHIRP_ROWS = 24
SYNTH_CHIRP_F0_KHZ = 20.0
SYNTH_CHIRP_KHZ_PER_MS = -0.8

#: Probabilities, one per feature, so a test can tell tracks apart by
#: `mean_prob` and `conf` as well as by frequency.
SYNTH_PROB = {
    "pickup": 0.95, "eho": 0.90, "h2": 0.70, "h3": 0.60,
    "split": 0.85, "chirp": 0.80, "salt": 0.90,
}

#: Salt: single pixels well above every feature (rows 420-500 is 51-61 kHz),
#: sparse enough that no blob of them reaches `tracks.MIN_AREA`.
SYNTH_SALT_ROWS = (420, 500)
SYNTH_SALT_N = 400
SYNTH_SALT_SEED = 7

#: The log-power on a lit pixel and off it. Flat over any one feature, so
#: that a power-weighted centroid is the geometric one and every expected
#: value in a test is arithmetic rather than a fit of the fixture.
SYNTH_LOGPOW_LIT = 2.5
SYNTH_LOGPOW_FLOOR = 0.4


def _synth_band(freq_khz, khz: float, half: int = 1) -> tuple[int, int]:
    """The half-open rows of a `2*half+1`-row band centred on `khz`."""
    row = int(np.argmin(np.abs(freq_khz - khz)))
    return row - half, row + half + 1


@pytest.fixture
def synth_mask():
    """`synth_mask(T=2048) -> (prob, raw_logpow, freq_khz, t_s)`.

    One synthetic coherent-probability map with five things in it, every one
    of them drawn from the constants above so that a test asserts what was
    drawn rather than what came out:

    * **pickup** - three rows at 1.95 kHz lit in every column of the record,
      which is what a receiver line looks like and what `pickup` is for;
    * **an EHO fundamental** at 8 kHz over columns 200-1200 with a
      20-column hole in it, plus its **2nd and 3rd harmonics** at 16 and 24
      kHz over the same columns - `merge` must bridge the hole (one track,
      two components) and `harmonics` must find the two partners;
    * **a split mode** at 44 kHz in two pieces 60 columns apart, which
      `merge` must NOT bridge - 44 kHz is chosen to be more than 7% from
      every 2-5x multiple of every other feature, so it is nobody's
      harmonic;
    * **a fishbone** chirping 20 -> 12 kHz at -0.8 kHz/ms;
    * **salt** - single pixels at 51-61 kHz, which `MIN_AREA` must erase.

    Returned rather than cached because a test may want a different record
    length; `T` shorter than the last feature is refused.
    """
    from labelmaker.events import masks

    def build(T: int = SYNTH_T):
        T = int(T)
        end = SYNTH_CHIRP_COL0 + SYNTH_CHIRP_N
        if T < end:
            raise ValueError(f"the features run to column {end}; T={T}")
        freq_khz = masks.freq_axis_khz(SYNTH_FS_HZ, SYNTH_DECIM)
        t_s = masks.col_times_s(T, SYNTH_FS_HZ, SYNTH_DECIM, 0.0)
        prob = np.zeros((freq_khz.size, T), dtype=np.float32)

        def band(khz, spans, value):
            r0, r1 = _synth_band(freq_khz, khz)
            for c0, c1 in spans:
                prob[r0:r1, c0:c1] = value

        band(SYNTH_PICKUP_KHZ, ((0, T),), SYNTH_PROB["pickup"])
        band(SYNTH_EHO_KHZ, SYNTH_EHO_COLS, SYNTH_PROB["eho"])
        band(SYNTH_H2_KHZ, (SYNTH_HARMONIC_COLS,), SYNTH_PROB["h2"])
        band(SYNTH_H3_KHZ, (SYNTH_HARMONIC_COLS,), SYNTH_PROB["h3"])
        band(SYNTH_SPLIT_KHZ, SYNTH_SPLIT_COLS, SYNTH_PROB["split"])

        # The bar of column k is centred on the line, so its centroid is
        # `f0 + slope * k * hop_ms` to within half a bin.
        half = (SYNTH_CHIRP_ROWS - 1) / 2.0
        for k in range(SYNTH_CHIRP_N):
            f = SYNTH_CHIRP_F0_KHZ + SYNTH_CHIRP_KHZ_PER_MS * k * SYNTH_HOP_S * 1e3
            r0 = round(f / SYNTH_BIN_KHZ - 1.0 - half)
            prob[r0:r0 + SYNTH_CHIRP_ROWS, SYNTH_CHIRP_COL0 + k] = (
                SYNTH_PROB["chirp"]
            )

        rng = np.random.default_rng(SYNTH_SALT_SEED)
        lo, hi = SYNTH_SALT_ROWS
        rows = rng.integers(lo, hi, SYNTH_SALT_N)
        cols = rng.integers(0, T, SYNTH_SALT_N)
        prob[rows, cols] = SYNTH_PROB["salt"]

        raw_logpow = np.where(
            prob > 0.0, SYNTH_LOGPOW_LIT, SYNTH_LOGPOW_FLOOR
        ).astype(np.float32)
        return prob, raw_logpow, freq_khz, t_s

    return build
