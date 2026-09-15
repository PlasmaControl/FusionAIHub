"""Opt-in gating for the tests that fetch live data through fdp.

Everything else in this directory is hermetic: it must pass under a plain
`pixi run -e labelmaker python -m pytest tests/labeler`, with no server,
no token and no network. A live test is marked `live` and is skipped unless
it is asked for explicitly, either with `--run-live` or with
`LABELER_FDP=1` (the environment switch the plan's command blocks use).

A live run also needs the `fdp run` wrapper, which supplies the server
configuration: without it PTDATA fails with `getservbyname failed for task
'PTSERVER'` and MDSplus with `TREE-E-FOPENR`.

    pixi run -e labelmaker fdp run python -m pytest tests/labeler \
        -q -W error --run-live

It is also where the fixtures more than one test module needs live: `wired`,
a synthetic archive and corpus, and `synth_mask`, the drawn coherent map the
event layer is measured against.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from labeler.env import getenv

#: Environment switch, equivalent to `--run-live`.
LIVE_ENV = "LABELER_FDP"


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
    if config.getoption("--run-live") or getenv(LIVE_ENV) == "1":
        return
    skip = pytest.mark.skip(
        reason="live fdp fetch is opt-in: --run-live or LABELER_FDP=1"
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
    from labeler.models import registry

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
    from labeler.events import masks

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


# ------------------------------------------------- the synthetic shot fixture

#: The ECE array `synth_shot` draws: 48 channels at 50 kHz over 0.8 s, which
#: is 800 bins of the 1 ms envelope every heuristic here runs on.
SYNTH_ECE_FS_HZ = 5.0e4
SYNTH_ECE_N = 40001
SYNTH_ECE_CHANNELS = 48

#: The sawtooth train: ten crashes 76 ms apart - the period measured on shot
#: 198658 - each sitting `SYNTH_CRASH_LEAD_MS` BEFORE a 1 ms envelope edge.
#: That offset is deliberate. A crash exactly on an edge lands in whichever
#: bin the last bit of the time axis rounds it into, and a crash in the
#: middle of a bin splits its drop over two bin differences and so over two
#: candidates; a tenth of a millisecond early puts five of the fifty samples
#: of the last pre-crash bin on the far side of the crash, which moves that
#: bin's mean by 0.5% - far under `DROP_FRAC` - and leaves exactly one
#: candidate per crash whichever way the arithmetic rounds.
SYNTH_SAWTOOTH_PERIOD_MS = 76.0
SYNTH_SAWTOOTH_N = 10
SYNTH_CRASH_LEAD_MS = 0.1

#: Half-open channel ranges: 4-20 lose Te at the crash, 21-40 gain it, and
#: 0-3 and 41-47 see nothing. The inversion is therefore BETWEEN channels 20
#: and 21, and the dropping block is interior to the array - which is what
#: `inversion_block` demands and what an ELM cannot produce.
SYNTH_CORE_CH = (4, 21)
SYNTH_EDGE_CH = (21, 41)
#: The crash drops the core by 5% of its pre-crash value and lifts the edge
#: by 2% of its own. Both baselines are 1.0, so the heat pulse is 40% of the
#: drop in ABSOLUTE terms and clears `REL_POS` (30%) - the displaced heat has
#: to arrive somewhere, and on unequal baselines it would not have to.
SYNTH_CORE_DROP = 0.05
SYNTH_EDGE_RISE = 0.02

#: The impostor: at 114 ms EVERY channel drops 5% together for 10 ms, which
#: is what an ELM or a gas puff looks like on the array. It is a crash
#: candidate and must not be a sawtooth.
SYNTH_ELM_MS = 114.0
SYNTH_ELM_WIDTH_MS = 10.0
SYNTH_ELM_DROP = 0.05

#: D-alpha: filterscopes 0-7 at 10 kHz. The L->H at 0.30 s drops it 45% (so
#: the confidence is 0.45/0.6 = 0.75, a number a test can tell from 1.0) and
#: the H->L at 0.60 s puts it back.
SYNTH_DALPHA_FS_HZ = 1.0e4
SYNTH_DALPHA_CHANNELS = 8
SYNTH_LH_S = 0.30
SYNTH_HL_S = 0.60
SYNTH_DALPHA_L = 1.0
SYNTH_DALPHA_H = 0.55

#: Line-averaged density, 1 kHz: flat, then +20% over the 50 ms after the
#: L->H and back down over the 50 ms after the H->L.
SYNTH_SCALAR_FS_HZ = 1.0e3
SYNTH_NE_L = 2.0
SYNTH_NE_H = 2.4
SYNTH_NE_RAMP_S = 0.05

#: Actuators. The beams are on 0.10-0.70 s at 3 MW; the torque is co-current
#: except over 0.30-0.50 s, where it flips against a positive Ip.
SYNTH_PINJ_KW = 3000.0
SYNTH_PINJ_ON_S = (0.10, 0.70)
SYNTH_TINJ_NM = 5.0
SYNTH_COUNTER_S = (0.30, 0.50)
SYNTH_IP_A = 1.0e6
SYNTH_BETAN_L = 1.0
SYNTH_BETAN_H = 2.0

SYNTH_T_COV = (0.0, 0.8)


def _synth_ece():
    """`(t_s, y, crash_times_s)`: the 48-channel array with ten crashes in it.

    Between crashes the core ramps back up and the edge decays back down by
    exactly the amount the crash moved them, so the train is periodic and
    every crash is the same size - no drift a detector could mistake for a
    trend. The ELM impostor is a multiplicative dip applied to the whole
    array on top of that.
    """
    period_s = SYNTH_SAWTOOTH_PERIOD_MS * 1e-3
    t = np.arange(SYNTH_ECE_N, dtype=np.float64) / SYNTH_ECE_FS_HZ
    crash0 = period_s - SYNTH_CRASH_LEAD_MS * 1e-3
    crashes = crash0 + np.arange(SYNTH_SAWTOOTH_N) * period_s
    phase = np.mod(t - crash0, period_s) / period_s

    # A crash that takes `1 - drop` of the level must be undone by a ramp of
    # `drop / (1 - drop)`, or the train would walk down the record.
    core_ramp = SYNTH_CORE_DROP / (1.0 - SYNTH_CORE_DROP)
    edge_decay = 1.0 - 1.0 / (1.0 + SYNTH_EDGE_RISE)
    y = np.ones((SYNTH_ECE_CHANNELS, t.size), dtype=np.float64)
    y[slice(*SYNTH_CORE_CH)] = 1.0 + core_ramp * phase
    y[slice(*SYNTH_EDGE_CH)] = 1.0 - edge_decay * phase

    elm0 = (SYNTH_ELM_MS - SYNTH_CRASH_LEAD_MS) * 1e-3
    elm1 = elm0 + SYNTH_ELM_WIDTH_MS * 1e-3
    y[:, (t >= elm0) & (t < elm1)] *= 1.0 - SYNTH_ELM_DROP
    return t, y.astype(np.float32), crashes


def _synth_dalpha():
    """`(t_s, y)`: eight filterscope channels, high in L-mode and low in H."""
    n = round(SYNTH_T_COV[1] * SYNTH_DALPHA_FS_HZ) + 1
    t = np.arange(n, dtype=np.float64) / SYNTH_DALPHA_FS_HZ
    level = np.where(
        (t >= SYNTH_LH_S) & (t < SYNTH_HL_S), SYNTH_DALPHA_H, SYNTH_DALPHA_L
    )
    # Per-channel gains, because the detector takes the channel MEDIAN and a
    # median of eight identical traces would not prove it took one.
    gains = 1.0 + 0.05 * np.arange(SYNTH_DALPHA_CHANNELS, dtype=np.float64)
    return t, (gains[:, None] * level[None, :]).astype(np.float32)


def _synth_scalars():
    """The 1 kHz scalars: `ne`, `betan`, `pinj` (kW), `tinj` (N m), `ip` (A)."""
    n = round(SYNTH_T_COV[1] * SYNTH_SCALAR_FS_HZ) + 1
    t = np.arange(n, dtype=np.float64) / SYNTH_SCALAR_FS_HZ
    up = np.clip((t - SYNTH_LH_S) / SYNTH_NE_RAMP_S, 0.0, 1.0)
    down = np.clip((t - SYNTH_HL_S) / SYNTH_NE_RAMP_S, 0.0, 1.0)
    shape = up - down
    ne = SYNTH_NE_L + (SYNTH_NE_H - SYNTH_NE_L) * shape
    betan = SYNTH_BETAN_L + (SYNTH_BETAN_H - SYNTH_BETAN_L) * shape
    on = (t >= SYNTH_PINJ_ON_S[0]) & (t <= SYNTH_PINJ_ON_S[1])
    pinj = np.where(on, SYNTH_PINJ_KW, 0.0)
    counter = (t >= SYNTH_COUNTER_S[0]) & (t <= SYNTH_COUNTER_S[1])
    tinj = np.where(on, np.where(counter, -SYNTH_TINJ_NM, SYNTH_TINJ_NM), 0.0)
    ip = np.full(t.shape, SYNTH_IP_A)
    return t, ne, betan, pinj, tinj, ip


@pytest.fixture
def synth_shot():
    """One synthetic shot's ECE, D-alpha, density, betan and actuators.

    Everything a `events.heuristics` detector reads, drawn from the constants
    above so that a test asserts what was PUT there: ten sawtooth crashes 76
    ms apart with the inversion between channels 20 and 21, one ELM impostor
    that drops the whole array together, an L->H at 0.30 s and an H->L at
    0.60 s with the density following, beams on 0.10-0.70 s and their torque
    counter-current over 0.30-0.50 s.

    A dict rather than a dataclass because a test routinely wants one of
    these arrays altered - a NaN channel, a flat density, no beams - and
    `{**synth_shot, "ne_y": ...}` is how it says so.
    """
    ece_t, ece_y, crashes = _synth_ece()
    dalpha_t, dalpha_y = _synth_dalpha()
    scalar_t, ne, betan, pinj, tinj, ip = _synth_scalars()
    return {
        "ece_t_s": ece_t,
        "ece_y": ece_y,
        "crash_times_s": crashes,
        "elm_impostor_s": (SYNTH_ELM_MS - SYNTH_CRASH_LEAD_MS) * 1e-3,
        "dalpha_t_s": dalpha_t,
        "dalpha_y": dalpha_y,
        "ne_t_s": scalar_t,
        "ne_y": ne,
        "betan_t_s": scalar_t,
        "betan_y": betan,
        "pinj_t_s": scalar_t,
        "pinj_y": pinj,
        "tinj_t_s": scalar_t,
        "tinj_y": tinj,
        "ip_t_s": scalar_t,
        "ip_y": ip,
        "t_cov": SYNTH_T_COV,
    }


@pytest.fixture(autouse=True)
def _isolate_package_environment(monkeypatch):
    """Strip legacy names still exported by the transitional main pixi manifest.

    Preserve new names, including values set by module- or session-scoped fixtures.
    Tests that need a new name absent must delete it explicitly.
    """
    for name in tuple(os.environ):
        if name.startswith(("IDEATE_", "LABELMAKER_")):
            monkeypatch.delenv(name)


@pytest.fixture
def truth_archive(tmp_path):
    """Synthetic continuous/binary truth and shot IDs for validation unit tests.

    These tests mock per-shot data or request an absent shot. Their initial
    archive-shape check must also use synthetic input, never the production store.
    """
    root = tmp_path / "truth-archive"
    root.mkdir()
    n = 2000
    np.save(root / "y.npy", np.column_stack((np.linspace(0.1, 4.0, n), np.arange(n) % 2)))
    np.save(root / "z.npy", np.full(n, 111, dtype=np.int64))
    return root
