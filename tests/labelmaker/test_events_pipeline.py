"""One shot end to end: `events.pipeline.process_shot` and `run events`.

Hermetic, and deliberately so. The corpus file is written into `tmp_path`
from the `synth_shot` fixture's own arrays - the ten sawtooth crashes, the
L->H at 0.30 s, the beams and the counter-current torque - so a test asserts
what was PUT in the file, and the network is a nine-line stand-in that
paints a rectangle and an ELM comb instead of the vendored U-Net.

Why a stand-in rather than the real network with a random state dict: the
mask path is already pinned against the real weights by
`test_events_unet.py`'s golden array and by `test_events_masks.py`, and what
is under test HERE is the orchestration - which channels are planned, which
step's failure is a skip and which is an error, what reaches the masks file,
the events file and the index. A 7.8M-parameter forward pass over seven
channels' tiles costs seconds of CPU per test and would pin none of that; a
model whose output is drawn by hand pins all of it, and lets a test say
"this track is at these columns" rather than "some track appeared".

The stand-in's geometry is chosen so every record here is ONE tile wide
(320 columns on the wide pass, 86 on the zoom, both under `masks.TILE`), so
a painted column index is the column index of the stitched mask.
"""
from __future__ import annotations

import json
import math
from dataclasses import replace

import h5py
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from labelmaker import run
from labelmaker.config import Paths
from labelmaker.events import (
    heuristics,
    masks,
    schema,
    text_weak,
    tracks,
    transients,
    unet,
)
from labelmaker.events import lexicon as lx
from labelmaker.events import pipeline as pl

from .conftest import SYNTH_COUNTER_S

SHOT = 199999
#: A stand-in sha, so nothing here needs the pinned checkpoint on disk.
FAKE_SHA = "0" * 64

#: What the stand-in paints. Rows 100-139 of 512 are 4.93-6.83 kHz on this
#: fixture's 50 kHz record; 200 lit columns of 320 is under
#: `tracks.PICKUP_ROW_FRACTION`, so the band is a `coherent_mode` and not a
#: pickup line, and 40 x 200 pixels is well over `tracks.MIN_AREA`.
TRACK_ROWS = (100, 140)
TRACK_COLS = (40, 240)
#: Five columns where every row is transient: the ELM comb. Interior, so
#: `find_peaks` sees a neighbour on both sides of each.
ELM_COLS = (60, 100, 140, 180, 220)
#: sigmoid(8) = 0.99966 and sigmoid(-8) = 3.4e-4, either side of
#: `ae.labels.PROB_THRESHOLD` by a wide margin.
LOGIT = 8.0


class PaintedNet(torch.nn.Module):
    """A U-Net-shaped stand-in: `forward` -> a 1-tuple of `(B, 2, H, W)` logits.

    Same contract as the vendored `BigTFUNetModel` (`unet.probabilities`
    takes `model(x)[0]` and applies the sigmoid), and its output does not
    depend on the input at all: what a test wants to know is where the
    orchestration put the mask it was given, not what the network thinks.
    """

    def forward(self, x):
        b, _, h, w = x.shape
        logits = torch.full((b, 2, h, w), -LOGIT, dtype=torch.float32)
        logits[:, 0, TRACK_ROWS[0]:TRACK_ROWS[1], TRACK_COLS[0]:TRACK_COLS[1]] = (
            LOGIT
        )
        for col in ELM_COLS:
            if col < w:
                logits[:, 1, :, col] = LOGIT
        return (logits,)


def _noise(shape, seed):
    return np.random.default_rng(seed).normal(0.0, 1e-4, shape)


def _write_corpus(corpus_dir, shot, s, *, groups=("all",)):
    """The synthetic shot as a corpus file: `<shot>_processed.h5`.

    Flat groups of `xdata` (seconds) and `ydata` `(C, T)` float32 and no
    attributes anywhere, which is the corpus' own layout
    (`features/resolve_corpus.py`). `bes` is written as the corpus'
    absent-signal sentinel - a `(64, 1)` placeholder - because that is what
    an absent group looks like on 67% of real shots, and `mirnov` is left
    out entirely because `mhr` is here to stand in for.
    """
    corpus_dir.mkdir(parents=True, exist_ok=True)
    path = corpus_dir / f"{shot}_processed.h5"
    ece_t, ece_y = s["ece_t_s"], np.asarray(s["ece_y"], dtype=np.float64)
    scalar_t = s["pinj_t_s"]
    n_fast, n_slow = ece_t.size, scalar_t.size
    ne_fast = np.interp(ece_t, s["ne_t_s"], s["ne_y"])

    mhr = np.sin(2 * np.pi * 5.0e3 * ece_t)[None, :] + _noise((8, n_fast), 1)
    mhr[:, -1] = np.nan                      # every fast group ends in NaN
    co2 = np.tile(ne_fast, (4, 1)) + _noise((4, n_fast), 2)
    co2[:, -1] = np.nan

    fscope = np.full((104, s["dalpha_t_s"].size), np.nan)
    fscope[:8] = np.asarray(s["dalpha_y"], dtype=np.float64)

    want = set(groups)

    def keep(name):
        return "all" in want or name in want

    with h5py.File(path, "w") as f:
        def put(name, x, y):
            if not keep(name):
                return
            g = f.create_group(name)
            g.create_dataset("xdata", data=np.asarray(x, dtype=np.float32))
            g.create_dataset("ydata", data=np.asarray(y, dtype=np.float32))

        put("mhr", ece_t, mhr)
        put("ece", ece_t, ece_y + _noise((48, n_fast), 3))
        put("co2", ece_t, co2)
        put("bes", [0.0], np.zeros((64, 1)))
        put("filterscopes", s["dalpha_t_s"], fscope)
        # The corpus holds the eight beams in W and the eight torques in
        # N m; `pipeline` sums them into the canonical kW and N m.
        put("pinj", scalar_t, np.tile(s["pinj_y"] * 1e3 / 8.0, (8, 1)))
        put("tinj", scalar_t, np.tile(s["tinj_y"] / 8.0, (8, 1)))
        put("ech_power", scalar_t, np.zeros((12, n_slow)))
        gas = np.zeros((11, n_slow))
        # The puff on channel 0 - namespace's `gas` IS `gas_raw#0` - and a
        # dead valve idling above the threshold on channel 3, which is what
        # 198658's channel 3 does for its whole 105 s record.
        gas[0] = np.where((scalar_t >= 0.1) & (scalar_t <= 0.7), 2.0, 0.0)
        gas[3] = 0.8
        put("gas_raw", scalar_t, gas)
        rmp = np.zeros((12, n_slow))
        rmp[5] = np.where((scalar_t >= 0.2) & (scalar_t <= 0.5), 800.0, 0.0)
        put("rmp", scalar_t, rmp)
    return path


@pytest.fixture
def paths(tmp_path):
    return Paths(
        root=tmp_path / "root",
        corpus=tmp_path / "corpus",
        text_root=tmp_path / "bundles",
        logs_jsonl=tmp_path / "logs.jsonl",
    )


@pytest.fixture
def shot_file(paths, synth_shot):
    return _write_corpus(paths.corpus, SHOT, synth_shot)


@pytest.fixture
def model():
    return PaintedNet().eval()


def test_finish_publishes_dalpha_clock_points_without_any_mask(paths):
    """A missing U-Net block must not hide measured D-alpha peaks."""
    paths.corpus.mkdir(parents=True)
    t = np.arange(10001) / 10000
    peak_times = np.array([0.2, 0.4, 0.6, 0.8])
    signal = sum(np.exp(-0.5 * ((t - p) / 0.001) ** 2) for p in peak_times)
    y = np.full((104, t.size), np.nan)
    y[:8] = signal
    y[:, :100] = np.nan
    y[:, -100:] = np.nan
    with h5py.File(paths.corpus_file(SHOT), "w") as f:
        f["filterscopes/xdata"] = t
        f["filterscopes/ydata"] = y
    result = pl.finish_shot(pl.ShotResult(shot=SHOT), paths,
                            paths.corpus_file(SHOT), [], run_id="dalpha")
    assert not result.error
    events = schema.read_events(paths.events_file(SHOT))
    points = events[events.phenomenon == "elm"]
    assert len(points) == 4
    np.testing.assert_allclose(points.t0_s, peak_times, atol=0.0001)
    np.testing.assert_array_equal(points.t0_s, points.t1_s)
    assert set(points.source) == {"elm_clock"}
    assert set(points.evidence_kind) == {"heuristic"}
    assert set(points.diag) == {"filterscopes"}
    assert set(points.t_cov0_s) == {t[100]}
    assert set(points.t_cov1_s) == {t[-101]}
    for raw in points["attrs"]:
        attrs = json.loads(raw)
        assert set(attrs) == {"prominence", "width_ms", "channel", "rate_hz_local"}
        assert attrs["channel"] == 0
        assert attrs["prominence"] > 0.9
        assert 2 < attrs["width_ms"] < 3
        assert attrs["rate_hz_local"] == 10
    assert result.n_elms == 4
    source = schema.read_sources(paths.sources_file(SHOT))
    clock = source[source.source == "elm_clock"].iloc[0]
    assert clock.status == "ok"
    assert clock.n_events == len(events[events.source == "elm_clock"])
    assert clock.t_cov0_s == t[100] and clock.t_cov1_s == t[-101]
    assert events[events.phenomenon == "elm_free"].shape[0] == 5


def test_outside_track_does_not_prevent_independent_dalpha_events(
    shot_file, paths, model, monkeypatch,
):
    with h5py.File(shot_file, "a") as f:
        t = f["filterscopes/xdata"][:]
        signal = sum(np.exp(-0.5 * ((t - p) / 0.002) ** 2)
                     for p in (0.1, 0.3, 0.5, 0.7))
        f["filterscopes/ydata"][:8, :] = signal
    describe = pl.describe_block

    def outside_track(prepared, *args, **kwargs):
        block = describe(prepared, *args, **kwargs)
        if block.diag == "ece" and block.channel == 8:
            at = block.t_cov[1] + 0.001
            # Its valid track co-occurs with the surviving blocks, but one
            # wholly outside descriptor rejects this entire block.
            block = replace(block, tracks=[
                *block.tracks, replace(block.tracks[0], t0_s=at, t1_s=at),
            ])
        return block

    monkeypatch.setattr(pl, "describe_block", outside_track)
    result = _run(paths, model)
    assert not result.error
    assert result.n_elms == 4
    assert "must not exceed" in result.skipped["track ece:8:wide"]
    events = schema.read_events(paths.events_file(SHOT))
    assert len(events[events.phenomenon == "elm"]) == 4
    track_rows = events[events.source == "tokeye_track"]
    assert result.n_tracks == len(track_rows) == 6
    assert not ((track_rows.diag == "ece") & (track_rows.channel == 8)).any()
    sources = schema.read_sources(paths.sources_file(SHOT))
    failed = sources[(sources.source == "tokeye_track")
                     & (sources.diag == "ece") & (sources.channel == 8)].iloc[0]
    assert failed.status == "skipped" and failed.n_events == 0
    assert np.isnan(failed.t_cov0_s) and np.isnan(failed.t_cov1_s)
    assert sources[sources.source == "elm_clock"].iloc[0].status == "ok"
    published = {
        f"{r.diag}:{r.channel}:{r.pass_name}"
        for r in sources.itertuples()
        if r.source == "tokeye_track" and r.status == "ok"
    }
    partners = [
        target for r in track_rows.itertuples()
        for target in json.loads(r.attrs)["cooccurrent_with"]
    ]
    assert partners  # Successful blocks must still corroborate each other.
    assert all(target.split("#")[0] in published for target in partners)
    assert "mhr:4:wide#0" in partners  # Original block-local track identity.


def test_quiet_dalpha_writes_coverage_and_no_observed_elm(paths):
    paths.corpus.mkdir(parents=True)
    with h5py.File(paths.corpus_file(SHOT), "w") as f:
        f["filterscopes/xdata"] = np.arange(1001) / 10000
        f["filterscopes/ydata"] = np.zeros((104, 1001))
    pl.finish_shot(pl.ShotResult(shot=SHOT), paths, paths.corpus_file(SHOT), [])
    events = schema.read_events(paths.events_file(SHOT))
    assert list(events.phenomenon) == ["elm_free"]
    source = schema.read_sources(paths.sources_file(SHOT))
    clock = source[source.source == "elm_clock"].iloc[0]
    assert clock.status == "ok" and clock.n_events == 1
    assert (clock.t_cov0_s, clock.t_cov1_s) == (0.0, 0.1)


def test_dalpha_falls_back_past_isolated_finite_samples(paths):
    paths.corpus.mkdir(parents=True)
    t = np.arange(1001) / 10000
    y = np.zeros((104, t.size))
    y[0] = np.nan
    y[0, [0, -1]] = 1.0
    with h5py.File(paths.corpus_file(SHOT), "w") as f:
        f["filterscopes/xdata"] = t
        f["filterscopes/ydata"] = y
    result = pl.finish_shot(pl.ShotResult(shot=SHOT), paths,
                            paths.corpus_file(SHOT), [])
    assert "elm_clock" not in result.skipped
    assert result.elm_reference == "filterscopes_01"
    rows = schema.read_events(paths.events_file(SHOT))
    assert list(rows.phenomenon) == ["elm_free"]
    assert list(rows.channel) == [1]


def test_missing_dalpha_is_skipped_without_borrowing_mask_coverage(
    shot_file, paths, model,
):
    with h5py.File(shot_file, "a") as f:
        del f["filterscopes"]
    _run(paths, model)
    events = schema.read_events(paths.events_file(SHOT))
    assert not (events.phenomenon == "elm").any()
    assert not (events.phenomenon == "elm_free").any()
    transient = events[events.source == "tokeye_transient"]
    assert set(transient.phenomenon) == {"transient"}
    assert len(transient) == len(ELM_COLS)
    source = schema.read_sources(paths.sources_file(SHOT))
    clock = source[source.source == "elm_clock"].iloc[0]
    assert clock.status == "skipped" and clock.diag == "filterscopes"
    assert np.isnan(clock.t_cov0_s) and np.isnan(clock.t_cov1_s)


def _run(paths, model, **kw):
    kw.setdefault("passes", ("wide",))
    kw.setdefault("tile_batch", 4)
    kw.setdefault("run_id", "test-run")
    kw.setdefault("unet_sha256", FAKE_SHA)
    return pl.process_shot(SHOT, paths, model=model, device="cpu", **kw)


# --------------------------------------------------------------- the masks


def test_every_runnable_channel_becomes_a_block_and_the_rest_a_reason(
    shot_file, paths, model,
):
    res = _run(paths, model)
    assert res.error == ""
    assert res.n_blocks == 7
    assert masks.list_blocks(paths.masks_file(SHOT)) == [
        "co2_00_wide", "co2_02_wide", "ece_08_wide", "ece_20_wide",
        "ece_40_wide", "mhr_00_wide", "mhr_04_wide",
    ]
    # `bes` is the corpus' (64, 1) placeholder; `mirnov` is `mhr`'s stand-in
    # and `mhr` is here. Neither is an error.
    assert res.skipped["channel bes:26"] == "group absent"
    assert res.skipped["channel mirnov:0"] == "fallback not needed"


def test_both_passes_write_their_own_blocks(shot_file, paths, model):
    res = _run(paths, model, passes=("wide", "zoom"))
    assert res.n_blocks == 14
    blocks = masks.list_blocks(paths.masks_file(SHOT))
    assert "mhr_00_zoom" in blocks and "mhr_00_wide" in blocks
    meta = masks.read_mask(paths.masks_file(SHOT), "mhr_00_zoom_meta")
    assert meta["decim"] == masks.ZOOM_DECIM


# -------------------------------------------------------------- the events


def test_the_track_transient_and_clock_sources_all_reach_the_file(
    shot_file, paths, model,
):
    _run(paths, model)
    df = schema.read_events(paths.events_file(SHOT))
    assert {"tokeye_track", "tokeye_transient", "elm_clock"} <= set(df["source"])
    tracks = df[df["source"] == "tokeye_track"]
    assert len(tracks) == 7                       # one painted band per block
    assert set(tracks["phenomenon"]) == {"coherent_mode"}
    # The band is rows 100-139 of a 0.048828 kHz/bin axis.
    assert tracks["f0_khz"].min() == pytest.approx(101 * 0.048828125, rel=1e-3)


def test_the_tracks_are_measured_on_the_probabilities_not_the_packed_mask(
    shot_file, paths, model,
):
    # Task L4's note: a stored mask is boolean, so `tracks_for_block` reads
    # `conf`/`mean_prob` of 1.0 (or NaN) off it. The pipeline must describe
    # the track while the probabilities are still in memory - here 0.99966,
    # which is neither.
    _run(paths, model)
    df = schema.read_events(paths.events_file(SHOT), source="tokeye_track")
    attrs = json.loads(df.iloc[0]["attrs"])
    assert attrs["mean_prob"] == pytest.approx(0.99966, abs=1e-4)
    assert 0.0 < float(df.iloc[0]["confidence"]) < 1.0


def test_transients_keep_one_mask_reference_and_elms_use_filterscopes(
    shot_file, paths, model,
):
    # Every magnetics block carries the same comb, and writing an ELM per
    # channel puts every crash in the table N times - which is what
    # `windows.EventTable`'s de-duplication exists to survive, not what it
    # should be fed. One reference channel, named in the result.
    res = _run(paths, model)
    df = schema.read_events(paths.events_file(SHOT), source="tokeye_transient")
    assert set(zip(df["diag"], df["channel"], df["pass_name"], strict=True)) == {
        ("mhr", 0, "wide")
    }
    assert set(df.phenomenon) == {"transient"}
    assert len(df) == len(ELM_COLS)
    assert res.elm_reference == "filterscopes_00"
    assert res.n_elms == len(schema.read_events(
        paths.events_file(SHOT), phenomenon="elm",
    ))


def test_rerunning_legacy_transients_removes_obsolete_elm_index_rows(
    shot_file, paths, model,
):
    legacy = schema.Event(shot=SHOT, source="tokeye_transient", phenomenon="elm",
                          t0_s=0.2, t1_s=0.2, t_cov0_s=0, t_cov1_s=1)
    schema.write_events(paths.events_file(SHOT), SHOT, [legacy], run_id="old")
    old = schema.index_rows(paths.events_file(SHOT))
    pl.append_index(paths.events_index, [*old, {**old[0], "shot": SHOT + 1}],
                    keys=["shot", "source", "phenomenon"])
    _run(paths, model)
    index = pd.read_parquet(paths.events_index)
    ours = index[index.shot == SHOT]
    assert not ((ours.source == "tokeye_transient") & (ours.phenomenon == "elm")).any()
    transient = ours[ours.source == "tokeye_transient"]
    assert list(transient.phenomenon) == ["transient"]
    assert list(transient.n_events) == [len(ELM_COLS)]
    assert (index.shot == SHOT + 1).sum() == 1


def test_the_cooccurring_tracks_name_each_other(shot_file, paths, model):
    _run(paths, model)
    df = schema.read_events(paths.events_file(SHOT), source="tokeye_track")
    row = df[df["diag"] == "mhr"].iloc[0]
    assert json.loads(row["attrs"])["cooccurrent_with"] == ["mhr:4:wide#0"]


def test_the_heuristics_add_their_own_sources(shot_file, paths, model):
    res = _run(paths, model)
    df = schema.read_events(paths.events_file(SHOT))
    assert {"ece_sawtooth", "dalpha_lh", "actuator"} <= set(df["source"])
    assert res.n_sawteeth == 10                   # the fixture's ten crashes
    assert set(df[df["source"] == "dalpha_lh"]["phenomenon"]) == {
        "lh_transition", "hl_transition"
    }
    on = set(df[df["source"] == "actuator"]["phenomenon"])
    assert {"nbi_on", "rmp_on", "gas_on"} <= on


def test_the_gas_valve_is_channel_0_and_not_the_group_maximum(shot_file,
                                                              paths, model):
    # `features/namespace.py`'s `gas` IS `gas_raw#0` (PTDATA `gasa`), and
    # taking the loudest of the eleven valves instead is defeated by one
    # idling channel: on shot 198658 channel 3 sits at 0.675-0.876 V for the
    # whole 105 s record, so `gas_on` came back as one interval covering it.
    # The fixture reproduces that - channel 3 flat at 0.8 V - and the puff
    # on channel 0 is what must be reported.
    _run(paths, model)
    df = schema.read_events(paths.events_file(SHOT), source="actuator")
    gas = df[df["phenomenon"] == "gas_on"]
    assert len(gas) == 1
    assert float(gas.iloc[0]["t0_s"]) == pytest.approx(0.1, abs=0.01)
    assert float(gas.iloc[0]["t1_s"]) == pytest.approx(0.7, abs=0.01)


def test_nbi_counter_is_a_recorded_skip_when_no_store_serves_the_current(
    shot_file, paths, model,
):
    """No features file, so no `ip`, so counter-injection is not evaluated.

    `ip` is not a corpus group, so this is what every shot did before the
    features store reached this stage - and what a shot whose features
    file has not been written still does.
    """
    res = _run(paths, model)
    df = schema.read_events(paths.events_file(SHOT), source="actuator")
    assert "nbi_counter" not in set(df["phenomenon"])
    assert "ip" in res.skipped["nbi_counter"]


def test_the_index_gets_one_row_per_source_and_phenomenon(shot_file, paths,
                                                          model):
    _run(paths, model)
    index = pd.read_parquet(paths.events_index)
    df = schema.read_events(paths.events_file(SHOT))
    assert len(index) == df.groupby(["source", "phenomenon"]).ngroups
    assert set(index["shot"]) == {SHOT}
    assert index["n_events"].sum() == len(df)


# ------------------------------------------- the per-source completion record


def test_every_source_that_ran_is_a_row_even_with_no_events(shot_file, paths,
                                                            synth_shot, model):
    # An events file says what was FOUND. `ech_power_total` is zero for
    # this whole shot, and with the beams zeroed below there is no NBI-on
    # interval for the QH proxy to intersect with, so it finds no candidate
    # either: neither writes an event row - and both ran. (The proxy needs
    # the features store's `ip` before it can run at all; without one it is
    # SKIPPED rather than a successful zero, which is its own test. And the
    # fixture's D-alpha carries no ELM burst, so with the beams ON the proxy
    # DOES find the 5 kHz line - that is the intersection test below.)
    _write_features(paths, SHOT, synth_shot)
    with h5py.File(shot_file, "a") as f:
        f["pinj/ydata"][...] = 0.0
    _run(paths, model)
    src = schema.read_sources(paths.sources_file(SHOT))
    ev = schema.read_events(paths.events_file(SHOT))
    assert set(src.columns) == set(schema.SOURCE_COLUMNS)
    ok = src[src["status"] == "ok"]
    quiet = ok[ok["n_events"] == 0]
    assert {("actuator", "ech_power_total"), ("qh_proxy", "")} <= set(
        zip(quiet["source"], quiet["diag"], strict=True)
    )
    assert set(quiet["reason"]) == {""}
    # Every row that ran and DID produce events agrees with the file.
    counted = ev.groupby(
        ["source", "diag", "channel", "pass_name"], dropna=False
    ).size()
    for _, row in ok.iterrows():
        key = (row["source"], row["diag"], row["channel"], row["pass_name"])
        assert int(row["n_events"]) == int(counted.get(key, 0)), key


def test_a_skipped_step_reaches_the_sources_file_with_its_reason(shot_file,
                                                                 paths, model):
    _run(paths, model)
    src = schema.read_sources(paths.sources_file(SHOT))
    rows = {
        (r["source"], r["diag"], r["channel"]): (r["status"], r["reason"])
        for _, r in src.iterrows()
    }
    # A planned channel the corpus does not serve.
    assert rows[("tokeye_track", "bes", 26)] == ("skipped", "group absent")
    # A step that could not be EVALUATED is skipped on its own source's
    # key, not recorded as a successful zero somewhere else: counter
    # -injection on `("actuator", "tinj_total")`, which is where its rows
    # would have been counted, and the proxy on `qh_proxy` itself.
    assert rows[("actuator", "tinj_total", -1)][0] == "skipped"
    assert rows[("qh_proxy", "", -1)][0] == "skipped"
    assert "flat-top" in rows[("qh_proxy", "", -1)][1]
    # And the caveat naming WHICH input was missing keeps its own row.
    assert "not a corpus group" in rows[("qh_flattop", "", -1)][1]
    assert rows[("text", "", -1)] == ("skipped", "no lexicon passed")
    assert set(src["shot"]) == {SHOT}


def test_a_failed_transient_step_does_not_skip_the_dalpha_clock(
    shot_file, paths, model, monkeypatch,
):
    def boom(*a, **kw):
        raise RuntimeError("no transient trace")

    monkeypatch.setattr(transients, "elm_events", boom)
    _run(paths, model)
    src = schema.read_sources(paths.sources_file(SHOT))
    got = src[src["source"].isin(["tokeye_transient", "elm_clock"])]
    assert len(got) == 2
    got = got.set_index("source")
    assert got.loc["tokeye_transient", "status"] == "skipped"
    assert "RuntimeError" in got.loc["tokeye_transient", "reason"]
    assert got.loc["elm_clock", "status"] == "ok"


def test_each_actuator_source_records_its_own_axis(shot_file, paths, model,
                                                   synth_shot):
    # Defect 2a at the pipeline end: one row per canonical feature, each
    # with the coverage of the group it was read off, over finite samples.
    _run(paths, model)
    src = schema.read_sources(paths.sources_file(SHOT))
    act = src[src["source"] == "actuator"].set_index("diag")
    assert set(act.index) == {
        "pinj_total", "tinj_total", "ech_power_total", "rmp", "gas"
    }
    # `tinj_total` is the counter-injection row and there is no `ip` here,
    # so it is the one that is skipped rather than covered; the other four
    # carry the axis they were read off.
    assert act.loc["tinj_total", "status"] == "skipped"
    act = act.drop(index="tinj_total")
    scalar_t = synth_shot["pinj_t_s"]
    for name in act.index:
        assert act.loc[name, "t_cov0_s"] == pytest.approx(float(scalar_t[0]))
        assert act.loc[name, "t_cov1_s"] == pytest.approx(float(scalar_t[-1]))
    # And every actuator EVENT carries the axis it was measured on.
    ev = schema.read_events(paths.events_file(SHOT), source="actuator")
    assert dict(zip(ev["phenomenon"], ev["diag"], strict=True)) == {
        "gas_on": "gas", "nbi_on": "pinj_total", "rmp_on": "rmp",
    }


def test_the_lh_detector_covers_only_where_all_three_inputs_were_measured(
    shot_file, paths, model, synth_shot,
):
    # D-alpha AND the line density AND the injected power. The fixture's
    # `co2` ends in a NaN column, as every real fast group does, so the
    # intersection stops one sample short of the D-alpha's own axis.
    _run(paths, model)
    src = schema.read_sources(paths.sources_file(SHOT))
    lh = src[src["source"] == "dalpha_lh"].iloc[0]
    ece_t = synth_shot["ece_t_s"]
    dalpha_t = synth_shot["dalpha_t_s"]
    assert lh["t_cov1_s"] == pytest.approx(float(ece_t[-2]))
    assert lh["t_cov1_s"] < float(dalpha_t[-1])
    ev = schema.read_events(paths.events_file(SHOT), source="dalpha_lh")
    assert ev["t_cov1_s"].to_numpy() == pytest.approx(float(ece_t[-2]))


def test_a_rerun_of_one_channel_replaces_only_that_channels_rows(shot_file,
                                                                 paths, model):
    _run(paths, model)
    before = schema.read_sources(paths.sources_file(SHOT))
    schema.write_sources(
        paths.sources_file(SHOT), SHOT,
        [{"source": "tokeye_track", "diag": "mhr", "channel": 0,
          "pass_name": "wide", "status": "ok", "reason": "",
          "t_cov0_s": 0.0, "t_cov1_s": 9.0, "n_events": 42}],
        run_id="rerun",
    )
    after = schema.read_sources(paths.sources_file(SHOT))
    assert len(after) == len(before)
    row = after[(after["source"] == "tokeye_track") & (after["diag"] == "mhr")
                & (after["channel"] == 0)].iloc[0]
    assert (row["n_events"], row["run_id"]) == (42, "rerun")
    assert set(after[after["run_id"] == "rerun"]["diag"]) == {"mhr"}


def test_write_false_writes_no_sources_file(shot_file, paths, model):
    _run(paths, model, write=False)
    assert not paths.sources_file(SHOT).exists()
    assert schema.read_sources(paths.sources_file(SHOT)).empty


# ------------------------------------------------------- failure isolation


def test_a_missing_group_is_a_skip_and_not_an_error(paths, synth_shot, model):
    # No `ece` at all: no mask blocks for it, no sawtooth, and everything
    # else still runs.
    _write_corpus(paths.corpus, SHOT, synth_shot,
                  groups=("mhr", "co2", "filterscopes", "pinj", "tinj"))
    res = _run(paths, model)
    assert res.error == ""
    assert res.n_blocks == 4
    assert res.n_sawteeth == 0
    assert "sawtooth" in res.skipped
    assert res.n_tracks == 4


def test_a_failing_step_is_recorded_and_the_shot_carries_on(
    shot_file, paths, model, monkeypatch,
):
    def boom(*a, **kw):
        raise RuntimeError("the envelope exploded")

    monkeypatch.setattr(heuristics, "sawtooth_events", boom)
    res = _run(paths, model)
    assert res.error == ""
    assert "RuntimeError" in res.skipped["sawtooth"]
    assert res.n_blocks == 7
    df = schema.read_events(paths.events_file(SHOT))
    assert "ece_sawtooth" not in set(df["source"])
    assert "tokeye_track" in set(df["source"])


def test_an_unreadable_corpus_file_is_an_error_and_writes_nothing(paths,
                                                                  model):
    paths.corpus.mkdir(parents=True, exist_ok=True)
    paths.corpus_file(SHOT).write_bytes(b"not an hdf5 file at all")
    res = _run(paths, model)
    assert res.error
    assert res.n_blocks == 0
    assert not paths.masks_file(SHOT).exists()
    assert not paths.events_file(SHOT).exists()


def test_write_false_computes_everything_and_stores_nothing(shot_file, paths,
                                                            model):
    res = _run(paths, model, write=False)
    assert res.n_blocks == 7 and res.n_tracks == 7 and res.n_sawteeth == 10
    assert not paths.masks_file(SHOT).exists()
    assert not paths.events_file(SHOT).exists()
    assert not paths.events_index.exists()


# ------------------------------------------------------------- the text end


def _write_text(paths, shot, prose):
    paths.text_cache.mkdir(parents=True, exist_ok=True)
    paths.logs_subset.write_text(
        json.dumps({"shot": shot, "log_text":
                    f"### [PHYSICS_OPERATOR] smithj 2024-05-17 13:12:07\n{prose}\n"})
        + "\n",
        encoding="utf-8",
    )
    paths.text_root.mkdir(parents=True, exist_ok=True)
    paths.text_file(shot).write_text(
        "## Shot-specific context (from summary.html)\n"
        f"SHOT: {shot}\n\nSHOT TABLE ROW (name -> value)\n"
        "- SHOT_TYPE: plasma\n- PULSE-LENGTH: 0.80\n",
        encoding="utf-8",
    )


def test_the_shot_scope_text_becomes_events_when_the_subset_has_a_record(
    shot_file, paths, model,
):
    _write_text(paths, SHOT, "fishbones through the current ramp")
    res = _run(paths, model, lexicon=lx.load_lexicon())
    df = schema.read_events(paths.events_file(SHOT), source="text")
    assert list(df["phenomenon"]) == ["fishbone"]
    assert res.n_text == 1
    assert set(df["evidence_kind"]) == {"text"}


def test_a_shot_with_no_logbook_record_is_not_an_error(shot_file, paths,
                                                       model):
    # And the 616 MB source is never opened for it: the subset says no.
    res = _run(paths, model, lexicon=lx.load_lexicon())
    assert res.error == ""
    assert res.n_text == 0
    assert "no logbook record" in res.skipped["text"]


# ---------------------------------------------------------- the norm switch


def test_plasma_norm_restandardises_inside_the_coverage_intersection(
    shot_file, paths, model,
):
    _run(paths, model, norm="plasma")
    meta = masks.read_mask(paths.masks_file(SHOT), "mhr_00_wide_meta")
    assert meta["norm"] == "plasma"
    # The statistics are the ones a z-score over exactly those columns
    # gives, and the window really is a subset: column 0 sits three hops
    # BEFORE the first sample (`masks.COL_ORIGIN`), outside every group's
    # coverage, so the padded head is not in it.
    y, fs_hz, t0_s, _ = masks.read_waveform(paths.corpus_file(SHOT), "mhr", 0)
    spectrogram, own = masks.prep(y, fs_hz=fs_hz)
    raw = masks.unstandardise(spectrogram, own)
    t_s = masks.col_times_s(own["n_cols"], fs_hz, 1, t0_s)
    inside = (t_s >= meta["norm_t0_s"]) & (t_s <= meta["norm_t1_s"])
    assert 0 < inside.sum() < own["n_cols"]
    assert meta["spec_mean"] == pytest.approx(float(raw[:, inside].mean()),
                                              rel=1e-6)
    assert meta["spec_mean"] != pytest.approx(own["spec_mean"], rel=1e-9)


def test_record_norm_is_the_default_and_is_the_ae_paths_own(shot_file, paths,
                                                            model):
    _run(paths, model)
    meta = masks.read_mask(paths.masks_file(SHOT), "mhr_00_wide_meta")
    assert meta["norm"] == "record"
    y, fs_hz, _, _ = masks.read_waveform(paths.corpus_file(SHOT), "mhr", 0)
    _, own = masks.prep(y, fs_hz=fs_hz)
    assert meta["spec_mean"] == pytest.approx(own["spec_mean"])
    assert meta["spec_std"] == pytest.approx(own["spec_std"])


# ------------------------------------------------------------ the run stage


@pytest.fixture
def staged(paths, monkeypatch, model):
    """`run events` with the network and the logbook stubbed out."""
    monkeypatch.setattr(unet, "load_unet",
                        lambda path=None, device="cpu", **kw: model)
    return paths


def _only_run(paths):
    """The one `runs/events/<run_id>.json` this run wrote."""
    written = list((paths.runs / "events").glob("*.json"))
    assert len(written) == 1, written
    return written[0]


def _argv(paths, *extra):
    return [
        "events", "--shots", str(SHOT), "--root", str(paths.root),
        "--corpus-dir", str(paths.corpus), *extra,
    ]


def test_the_events_stage_writes_the_masks_and_the_events(shot_file, staged,
                                                          paths):
    assert run.main(_argv(paths, "--passes", "wide")) == 0
    assert paths.masks_file(SHOT).exists()
    assert schema.read_events(paths.events_file(SHOT))["source"].nunique() >= 4


def test_the_events_stage_calls_process_shot_per_shot_and_honours_limit(
    staged, paths, monkeypatch,
):
    seen: list[int] = []

    def fake(shot, paths_, **kw):
        seen.append(shot)
        return pl.ShotResult(shot=shot, n_blocks=2, n_tracks=1)

    monkeypatch.setattr(pl, "process_shot", fake)
    assert run.main(_argv(paths, "--shots", "11", "22", "33",
                          "--limit", "2")) == 0
    assert seen == [11, 22]


def test_the_events_stage_writes_a_json_summary(staged, paths, monkeypatch):
    monkeypatch.setattr(pl, "process_shot",
                        lambda shot, p, **kw: pl.ShotResult(shot=shot,
                                                            n_blocks=3))
    assert run.main(_argv(paths)) == 0
    payload = json.loads(_only_run(paths).read_text())
    assert payload["settings"]["device"] == "cpu"
    assert payload["git_sha"]
    assert [r["shot"] for r in payload["shots"]] == [SHOT]
    assert payload["shots"][0]["n_blocks"] == 3


def test_the_log_subset_is_built_once_for_the_whole_run(staged, paths,
                                                        monkeypatch):
    calls: list[tuple] = []

    def fake_build(shots, *, paths=None, refresh_missing=False):
        calls.append((sorted(shots), refresh_missing))
        return 7

    monkeypatch.setattr(text_weak, "build_logs_subset", fake_build)
    monkeypatch.setattr(pl, "process_shot",
                        lambda shot, p, **kw: pl.ShotResult(shot=shot))
    assert run.main(_argv(paths, "--shots", "11", "22", "33")) == 0
    assert calls == [([11, 22, 33], False)]
    payload = json.loads(_only_run(paths).read_text())
    assert payload["n_log_records_added"] == 7
    assert payload["logs_subset"] == str(paths.logs_subset)


def test_refresh_text_is_how_a_recorded_miss_is_re_asked(staged, paths,
                                                         monkeypatch):
    calls: list[bool] = []
    monkeypatch.setattr(
        text_weak, "build_logs_subset",
        lambda shots, *, paths=None, refresh_missing=False: (
            calls.append(refresh_missing) or 0
        ),
    )
    monkeypatch.setattr(pl, "process_shot",
                        lambda shot, p, **kw: pl.ShotResult(shot=shot))
    assert run.main(_argv(paths, "--refresh-text")) == 0
    assert calls == [True]


def test_a_shot_that_runs_over_its_budget_is_recorded_and_the_run_goes_on(
    staged, paths, monkeypatch, capsys,
):
    def slow(shot, p, **kw):
        if shot == 11:
            while True:
                pass
        return pl.ShotResult(shot=shot, n_blocks=1)

    monkeypatch.setattr(pl, "process_shot", slow)
    assert run.main(_argv(paths, "--shots", "11", "22", "--timeout", "1")) == 0
    out = capsys.readouterr().out
    assert "1 error" in out and "1 ok" in out
    payload = json.loads(_only_run(paths).read_text())
    rows = {r["shot"]: r for r in payload["shots"]}
    assert rows[11]["status"] == "error"
    assert rows[22]["status"] == "ok"


# ------------------------------------------------- the curated label tables


def _label_tables(tmp_path, rows, *, stem="rwm_fixture", version=1):
    """A one-table label root: `tables.yaml` and one CSV of `(shot, ms)`."""
    import yaml

    root = tmp_path / "labels"
    (root / "resistive_wall_mode/format").mkdir(parents=True, exist_ok=True)
    (root / "tables.yaml").write_text(yaml.safe_dump({
        "version": version,
        "tables": [{
            "stem": stem,
            "raw_file": f"{stem}.csv", "format_stem": stem, "converter": "csv",
            "made_at": "2026-09-13T00:00:00Z",
            "dir": "resistive_wall_mode",
            "phenomenon": "rwm",
            "kind": "point",
            "shot_col": "SHOT",
            "t_col": "ONSET_TIME",
            "t_units": "ms",
            "attr_cols": ["NTOR"],
            "attr_types": {"NTOR": "int"},
            "provenance": "a fixture",
        }],
    }), encoding="utf-8")
    import csv

    with (root / "resistive_wall_mode/format" / f"{stem}.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["shot", "t0_s", "t1_s", "phenomenon", "evidence_kind",
                         "source", "confidence", "attrs"])
        for shot, ms in rows:
            writer.writerow([shot, ms / 1000, ms / 1000, "rwm", "database",
                             f"database:{stem}", "", json.dumps({"NTOR": 1,
                                                                "table": stem})])
    return root


def test_a_table_that_names_the_shot_adds_its_rows_and_one_source_record(
    shot_file, paths, model, tmp_path,
):
    root = _label_tables(tmp_path, [(SHOT, 300.0), (SHOT, 450.0),
                                    (SHOT + 1, 100.0)])
    res = _run(replace(paths, label_tables=root), model)
    assert res.error == "" and "database" not in res.skipped
    df = schema.read_events(paths.events_file(SHOT),
                            source="database:rwm_fixture")
    assert list(df["t0_s"]) == [0.3, 0.45]
    assert list(df["t1_s"]) == [0.3, 0.45]
    assert set(df["evidence_kind"]) == {"database"}
    assert set(df["phenomenon"]) == {"rwm"}
    # A listing is not a coverage claim, and a curated list has no
    # calibrated probability.
    assert df["t_cov0_s"].isna().all() and df["t_cov1_s"].isna().all()
    assert df["confidence"].isna().all()
    assert res.by_source["database:rwm_fixture"] == 2
    # One record per table that named this shot, on the result...
    (record,) = res.database_sources
    assert record["source"] == "database:rwm_fixture"
    assert record["status"] == "ok" and record["n_events"] == 2
    assert record["reason"] == ""
    assert np.isnan(record["t_cov0_s"]) and np.isnan(record["t_cov1_s"])
    # ...and written to the sources file with every other source's, on the
    # same contract: ok, no reason, NaN coverage. `ok` + NaN coverage is
    # what identifies a curated source, because no detector writes that.
    srcs = schema.read_sources(paths.sources_file(SHOT))
    curated = srcs[srcs["source"] == "database:rwm_fixture"]
    assert len(curated) == 1
    (row,) = curated.to_dict("records")
    assert row["status"] == "ok" and row["reason"] == ""
    assert row["n_events"] == 2
    assert np.isnan(row["t_cov0_s"]) and np.isnan(row["t_cov1_s"])
    assert (row["diag"], row["channel"], row["pass_name"]) == ("", -1, "")
    # And no detector's row looks like that.
    others = srcs[srcs["source"] != "database:rwm_fixture"]
    assert not (
        (others["status"] == "ok") & others["t_cov0_s"].isna()
    ).any()


def test_a_shot_no_table_names_gets_no_database_row_and_no_source_record(
    shot_file, paths, model, tmp_path,
):
    # Nobody looked, so there is nothing to say - not `n_events=0`, which
    # over 16,909 shots would be a coverage claim no author made.
    root = _label_tables(tmp_path, [(SHOT + 1, 100.0)])
    res = _run(replace(paths, label_tables=root), model)
    assert res.database_sources == []
    assert "database" not in res.skipped
    df = schema.read_events(paths.events_file(SHOT))
    assert (df["evidence_kind"] == "database").sum() == 0
    assert not any(s.startswith("database:") for s in df["source"])
    # And NO source row either - the shot was not looked at by any table.
    srcs = schema.read_sources(paths.sources_file(SHOT))
    assert not any(str(s).startswith("database:") for s in srcs["source"])


def test_a_broken_manifest_is_a_skip_and_not_a_lost_shot(shot_file, paths,
                                                         model, tmp_path):
    root = _label_tables(tmp_path, [(SHOT, 300.0)])
    (root / "tables.yaml").write_text("version: 1\ntables: [{stem: x}]\n",
                                      encoding="utf-8")
    res = _run(replace(paths, label_tables=root), model)
    assert res.error == ""
    assert "DatabaseError" in res.skipped["database"]
    assert res.database_sources == []
    df = schema.read_events(paths.events_file(SHOT))
    assert "tokeye_track" in set(df["source"])       # the rest of the shot


def test_the_curated_rows_are_in_the_index_like_any_other_source(
    shot_file, paths, model, tmp_path,
):
    root = _label_tables(tmp_path, [(SHOT, 300.0)])
    _run(replace(paths, label_tables=root), model)
    index = pd.read_parquet(paths.events_index)
    row = index[index["source"] == "database:rwm_fixture"]
    assert len(row) == 1
    assert row["phenomenon"].item() == "rwm"
    assert row["evidence_kind"].item() == "database"
    assert row["n_events"].item() == 1


# ------------------------------------------------ run events --databases-only


@pytest.fixture
def no_network(monkeypatch):
    """Nothing in `--databases-only` may reach the U-Net or the logbook."""
    def refuse(*a, **kw):
        raise AssertionError("--databases-only loaded the network")

    monkeypatch.setattr(unet, "load_unet", refuse)
    return refuse


def test_databases_only_needs_neither_the_corpus_nor_the_network(
    paths, tmp_path, monkeypatch, no_network,
):
    # A curated list is knowledge ABOUT a shot; we may hold no signals for
    # it at all, and for the RWM tables we hold none for any of the 33.
    root = _label_tables(tmp_path, [(SHOT, 300.0), (SHOT, 450.0)])
    monkeypatch.setenv("LABELMAKER_LABEL_TABLES", str(root))
    assert not paths.corpus_file(SHOT).exists()
    assert run.main(_argv(paths, "--databases-only")) == 0
    df = schema.read_events(paths.events_file(SHOT))
    assert list(df["source"]) == ["database:rwm_fixture"] * 2
    assert df["t_cov0_s"].isna().all()
    assert not paths.masks_file(SHOT).exists()


def test_databases_only_says_how_many_shots_any_table_names(
    paths, tmp_path, monkeypatch, capsys, no_network,
):
    root = _label_tables(tmp_path, [(SHOT, 300.0)])
    monkeypatch.setenv("LABELMAKER_LABEL_TABLES", str(root))
    assert run.main(["events", "--databases-only", "--root", str(paths.root),
                     "--shots", str(SHOT), "11", "22"]) == 0
    out = capsys.readouterr().out
    assert "1 of 3 shots are named by any table" in out


def test_databases_only_over_shots_no_table_names_writes_nothing_and_exits_ok(
    paths, tmp_path, monkeypatch, capsys, no_network,
):
    # The `recommender_v1` case, which must READ as an answer and not as a
    # failure: the RWM tables span 156785-176092 and the corpus starts at
    # 185601, so zero is the honest count.
    root = _label_tables(tmp_path, [(156785, 856.0)])
    monkeypatch.setenv("LABELMAKER_LABEL_TABLES", str(root))
    assert run.main(_argv(paths, "--databases-only")) == 0
    assert "0 of 1 shots are named by any table" in capsys.readouterr().out
    assert not paths.events_file(SHOT).exists()
    assert not paths.events_index.exists()


def test_databases_only_writes_the_run_json_with_the_per_table_totals(
    paths, tmp_path, monkeypatch, no_network,
):
    root = _label_tables(tmp_path, [(SHOT, 300.0), (SHOT, 450.0)])
    monkeypatch.setenv("LABELMAKER_LABEL_TABLES", str(root))
    assert run.main(_argv(paths, "--databases-only")) == 0
    payload = json.loads(_only_run(paths).read_text())
    assert payload["settings"]["databases_only"] is True
    assert payload["settings"]["label_tables"] == str(root)
    totals = payload["totals"]
    assert totals["n_shots_named"] == 1 and totals["n_shots"] == 1
    assert totals["n_events"] == 2
    assert totals["n_source_records"] == 1
    assert totals["events_by_source"] == {"database:rwm_fixture": 2}
    assert totals["tables"] == ["database:rwm_fixture"]
    (row,) = payload["shots"]
    assert row["n_events"] == 2 and row["status"] == "ok"
    assert row["sources"] == ["database:rwm_fixture"]
    (record,) = row["source_records"]
    assert record["status"] == "ok" and record["reason"] == ""
    assert record["n_events"] == 2
    # `null`, not a bare `NaN` literal: the run JSON has to be JSON.
    assert record["t_cov0_s"] is None and record["t_cov1_s"] is None
    raw = _only_run(paths).read_text()
    assert "NaN" not in raw
    json.loads(raw, parse_constant=_no_constants)


def _no_constants(name):
    raise AssertionError(f"non-standard JSON literal: {name}")


def test_databases_only_writes_the_sources_file_too(
    paths, tmp_path, monkeypatch, no_network,
):
    """The standalone mode is on the same sources contract as the GPU path.

    Otherwise a table ingested here would reach `events.parquet` with no
    row saying anybody consulted it, and a consumer asking "did any table
    look at this shot" would get the same silence as for a shot nothing
    has run on.
    """
    root = _label_tables(tmp_path, [(SHOT, 300.0), (SHOT, 450.0),
                                    (SHOT + 1, 100.0)])
    monkeypatch.setenv("LABELMAKER_LABEL_TABLES", str(root))
    assert run.main(_argv(paths, "--databases-only")) == 0
    (row,) = schema.read_sources(paths.sources_file(SHOT)).to_dict("records")
    assert row["source"] == "database:rwm_fixture"
    assert row["status"] == "ok" and row["reason"] == ""
    assert row["n_events"] == 2
    assert np.isnan(row["t_cov0_s"]) and np.isnan(row["t_cov1_s"])
    assert (row["diag"], row["channel"], row["pass_name"]) == ("", -1, "")
    # The shot no table names gets no file at all.
    assert not paths.sources_file(SHOT + 2).exists()


def test_databases_only_leaves_another_sources_rows_alone(
    shot_file, staged, paths, tmp_path, monkeypatch,
):
    # The tables are ingested over a shot list in seconds, repeatedly and at
    # any time; a re-ingest must not cost the shot its detector rows.
    assert run.main(_argv(paths, "--passes", "wide")) == 0
    before = schema.read_events(paths.events_file(SHOT))
    assert "tokeye_track" in set(before["source"])
    root = _label_tables(tmp_path, [(SHOT, 300.0)])
    monkeypatch.setenv("LABELMAKER_LABEL_TABLES", str(root))
    assert run.main(_argv(paths, "--databases-only")) == 0
    after = schema.read_events(paths.events_file(SHOT))
    assert len(after) == len(before) + 1
    assert set(before["event_id"]) < set(after["event_id"])


def test_databases_only_belongs_to_the_events_stage_alone(paths, tmp_path):
    with pytest.raises(SystemExit) as exc:
        run.main(["features", "--models", "d3d_ae_activity_seldnet",
                  "--shots", str(SHOT), "--root", str(paths.root),
                  "--databases-only"])
    assert exc.value.code == 2


def test_an_unreadable_manifest_stops_the_run_before_any_shot(
    paths, tmp_path, monkeypatch, capsys, no_network,
):
    root = _label_tables(tmp_path, [(SHOT, 300.0)])
    (root / "tables.yaml").write_text("version: 1\ntables: [{stem: x}]\n",
                                      encoding="utf-8")
    monkeypatch.setenv("LABELMAKER_LABEL_TABLES", str(root))
    assert run.main(_argv(paths, "--databases-only")) == run.EXIT_BAD_LABEL_TABLE
    assert "kind" in capsys.readouterr().err
    assert not paths.events_file(SHOT).exists()


# --------------------------------------------------- the features store steps

#: The q-min band the feature file below paints, and the record it sits in.
#: 0.10-0.70 s of `qmin = 1.2` is 600 ms of hybrid inside the flat-top, and
#: the shot's own record is 0-0.8 s (`SYNTH_T_COV`).
FEATURE_QMIN_BAND = (0.10, 0.70)
FEATURE_QMIN_VALUE = 1.2
FEATURE_QMIN_STEP_S = 0.02


def _write_features(paths, shot, synth, *, qmin=True, ip=True,
                    qmin_value=FEATURE_QMIN_VALUE):
    """A features file for `shot`, written through the store's own writer.

    `ip` is the fixture's own constant current, so the flat-top is the
    whole record; `qmin` is a 20 ms axis - EFIT01's cadence - holding
    `qmin_value` over `FEATURE_QMIN_BAND` and 0.8 (no regime) either side.
    """
    from labelmaker.features import store as fs

    paths.features.mkdir(parents=True, exist_ok=True)
    arrays = {}
    if ip:
        arrays["ip"] = fs.FeatureArray(
            x=np.asarray(synth["ip_t_s"], dtype=np.float64),
            y=np.asarray(synth["ip_y"], dtype=np.float64)[None, :],
            attrs={"resolver": "fdp"},
        )
    if qmin:
        t = (np.arange(round(0.8 / FEATURE_QMIN_STEP_S) + 1)
             * FEATURE_QMIN_STEP_S).round(10)
        y = np.full(t.size, 0.8)
        lo, hi = FEATURE_QMIN_BAND
        y[(t >= lo) & (t <= hi)] = float(qmin_value)
        arrays["qmin"] = fs.FeatureArray(x=t, y=y[None, :],
                                         attrs={"resolver": "fdp"})
    fs.write_features(paths.features_file(shot), shot, arrays, {},
                      merge=False)
    return paths.features_file(shot)


def _sources(paths, shot=SHOT):
    """`{(source, diag): (status, reason, t_cov0, t_cov1, n_events)}`."""
    df = schema.read_sources(paths.sources_file(shot))
    return {
        (r["source"], r["diag"]): (
            r["status"], r["reason"], r["t_cov0_s"], r["t_cov1_s"],
            r["n_events"],
        )
        for _, r in df.iterrows()
    }


def test_the_flattop_comes_off_the_features_store(shot_file, paths, synth_shot,
                                                  model):
    _write_features(paths, SHOT, synth_shot)
    res = _run(paths, model)
    assert "features" not in res.skipped
    assert "qh_flattop" not in res.skipped
    assert "nbi_counter" not in res.skipped


def test_nbi_counter_is_claimed_once_the_store_supplies_the_current(
    shot_file, paths, synth_shot, model,
):
    """The skip that stood on every shot is gone, and the rows are there.

    `ip` is not a corpus group, so before the features store reached this
    stage `nbi_counter` was a recorded skip on 100% of shots and the QH
    proxy intersected with the empty set. This test fails if either skip
    comes back.
    """
    _write_features(paths, SHOT, synth_shot)
    res = _run(paths, model)
    df = schema.read_events(paths.events_file(SHOT))
    counter = df[df["phenomenon"] == "nbi_counter"]
    assert not counter.empty
    # The fixture flips the torque counter-current over 0.30-0.50 s.
    assert counter["t0_s"].min() == pytest.approx(SYNTH_COUNTER_S[0], abs=2e-3)
    assert counter["t1_s"].max() == pytest.approx(SYNTH_COUNTER_S[1], abs=2e-3)
    assert res.by_source.get("actuator", 0) >= len(counter)


class QuietNet(PaintedNet):
    """`PaintedNet` with the ELM comb moved off the track.

    The default fixture paints ELMs straight through the drawn mode, so
    its `elm_free` intervals are exactly the record either side of the
    track and the QH proxy is empty whatever the flat-top says - which
    makes it useless for testing the flat-top. Here the two ELMs are at
    columns 10 and 300, outside `TRACK_COLS`, so the track runs in an
    ELM-free NBI-heated stretch and the ONLY remaining gate is the one
    under test.
    """

    def forward(self, x):
        b, _, h, w = x.shape
        logits = torch.full((b, 2, h, w), -LOGIT, dtype=torch.float32)
        logits[:, 0, TRACK_ROWS[0]:TRACK_ROWS[1], TRACK_COLS[0]:TRACK_COLS[1]] = (
            LOGIT
        )
        for col in (10, 300):
            if col < w:
                logits[:, 1, :, col] = LOGIT
        return (logits,)


def test_the_qh_proxy_intersects_a_real_flattop(shot_file, paths, synth_shot):
    """The proxy claims where all four hold - and nothing outside them.

    Three runs of one shot. Without a features file there is no flat-top,
    the proxy intersects with the empty set and claims nothing - which is
    what EVERY shot did before this task. With the fixture's flat current
    the flat-top is the whole record and the proxy claims. With a current
    that collapses at 0.4 s the flat-top ends there, and so does the
    claim: this test fails if the flat-top stops reaching `qh_candidates`
    in either direction.
    """
    model = QuietNet().eval()
    blind = _run(paths, model)
    assert blind.skipped["qh_flattop"] == pl.NO_FLATTOP
    assert blind.by_source.get("qh_proxy", 0) == 0

    _write_features(paths, SHOT, synth_shot)
    wide = _run(paths, model)
    assert "qh_flattop" not in wide.skipped
    assert wide.by_source.get("qh_proxy", 0) >= 1
    whole = schema.read_events(paths.events_file(SHOT))
    whole = whole[whole["source"] == "qh_proxy"]

    ip = np.asarray(synth_shot["ip_y"], dtype=np.float64).copy()
    ip[np.asarray(synth_shot["ip_t_s"]) > 0.4] = 0.0
    _write_features(paths, SHOT, {**synth_shot, "ip_y": ip})
    _run(paths, model)
    narrow = schema.read_events(paths.events_file(SHOT))
    narrow = narrow[narrow["source"] == "qh_proxy"]
    assert not narrow.empty
    assert whole["t1_s"].max() > 0.4
    assert narrow["t1_s"].max() == pytest.approx(0.4, abs=5e-3)


def test_the_q_min_rule_writes_its_band_and_its_own_source(
    shot_file, paths, synth_shot, model,
):
    _write_features(paths, SHOT, synth_shot)
    _run(paths, model)
    df = schema.read_events(paths.events_file(SHOT))
    band = df[df["source"] == "qmin_rule"]
    assert list(band["phenomenon"]) == ["qmin_hybrid"]
    assert (band["t0_s"].iloc[0], band["t1_s"].iloc[0]) == pytest.approx(
        FEATURE_QMIN_BAND
    )
    assert band["confidence"].isna().all()
    assert json.loads(band["attrs"].iloc[0])["efit"] == "efit01"
    rows = _sources(paths)
    status, reason, cov0, cov1, n = rows[("qmin_rule", "qmin")]
    assert (status, reason, n) == ("ok", "", 1)
    # The flat-top met with the q-min record: the fixture's current is flat
    # over the whole 0-0.8 s record, so the q-min axis is what bounds it.
    assert (cov0, cov1) == pytest.approx((0.0, 0.8))


def test_each_features_quantity_records_its_own_span(shot_file, paths,
                                                     synth_shot, model):
    _write_features(paths, SHOT, synth_shot)
    _run(paths, model)
    rows = _sources(paths)
    ip_status, _, ip0, ip1, _ = rows[("features", "ip")]
    q_status, _, q0, q1, _ = rows[("features", "qmin")]
    assert (ip_status, q_status) == ("ok", "ok")
    assert (ip0, ip1) == pytest.approx(
        (synth_shot["ip_t_s"][0], synth_shot["ip_t_s"][-1])
    )
    assert (q0, q1) == pytest.approx((0.0, 0.8))


def test_a_shot_with_no_features_file_is_a_skip_with_a_reason(
    shot_file, paths, model,
):
    res = _run(paths, model)
    assert "features" in res.skipped
    assert "features" in res.skipped["features"].lower()
    assert "qmin_rule" in res.skipped
    assert pl.NO_IP in res.skipped["nbi_counter"]
    assert res.skipped["qh_flattop"] == pl.NO_FLATTOP
    df = schema.read_events(paths.events_file(SHOT))
    assert df[df["source"] == "qmin_rule"].empty
    rows = _sources(paths)
    assert rows[("features", "")][0] == "skipped"
    assert rows[("qmin_rule", "qmin")][0] == "skipped"


def test_a_file_without_q_min_still_serves_the_current(shot_file, paths,
                                                       synth_shot, model):
    """The two quantities fail independently: `ip` runs, `qmin` says why.

    17 of the 878 shots in the real features store are exactly this shape.
    """
    _write_features(paths, SHOT, synth_shot, qmin=False)
    res = _run(paths, model)
    assert "features" not in res.skipped
    assert "qmin" in res.skipped
    assert res.skipped["qmin_rule"] == pl.NO_QMIN
    assert "nbi_counter" not in res.skipped
    assert "qh_flattop" not in res.skipped
    rows = _sources(paths)
    assert rows[("features", "ip")][0] == "ok"
    assert rows[("features", "qmin")][0] == "skipped"
    assert "qmin" in rows[("features", "qmin")][1]


def test_a_file_without_the_current_leaves_the_rule_ungateable(
    shot_file, paths, synth_shot, model,
):
    _write_features(paths, SHOT, synth_shot, ip=False)
    res = _run(paths, model)
    assert res.skipped["qmin_rule"] == pl.NO_QMIN_FLATTOP
    assert pl.NO_IP in res.skipped["nbi_counter"]
    assert res.skipped["qh_flattop"] == pl.NO_FLATTOP
    df = schema.read_events(paths.events_file(SHOT))
    assert df[df["source"] == "qmin_rule"].empty


def test_a_flattop_with_no_regime_in_it_still_declares_the_rule_ran(
    shot_file, paths, synth_shot, model,
):
    """The answer only the sources file can carry.

    A shot whose q-min never leaves the lowest band writes no event row at
    all, and "it was under 0.95 all flat-top" and "nobody computed it" are
    then the same empty query without this row.
    """
    _write_features(paths, SHOT, synth_shot, qmin_value=0.5)
    _run(paths, model)
    df = schema.read_events(paths.events_file(SHOT))
    assert df[df["source"] == "qmin_rule"].empty
    status, reason, cov0, cov1, n = _sources(paths)[("qmin_rule", "qmin")]
    assert (status, reason, n) == ("ok", "", 0)
    assert (cov0, cov1) == pytest.approx((0.0, 0.8))


def test_the_regime_rows_reach_the_index_like_any_other_source(
    shot_file, paths, synth_shot, model,
):
    _write_features(paths, SHOT, synth_shot)
    _run(paths, model)
    index = pd.read_parquet(paths.events_index)
    got = index[index["source"] == "qmin_rule"]
    assert list(got["phenomenon"]) == ["qmin_hybrid"]
    assert int(got["n_events"].iloc[0]) == 1


def test_a_features_file_that_cannot_be_read_is_a_skip_and_not_an_error(
    shot_file, paths, synth_shot, model,
):
    paths.features.mkdir(parents=True, exist_ok=True)
    paths.features_file(SHOT).write_bytes(b"not an hdf5 file")
    res = _run(paths, model)
    assert res.status == "ok"
    assert "ip" in res.skipped and "qmin" in res.skipped
    assert res.skipped["qmin_rule"] == pl.NO_QMIN


# ------------------- a missing prerequisite is never an evaluated zero

def test_a_missing_flattop_makes_the_qh_source_itself_skipped(shot_file, paths,
                                                              model):
    """The critic's iteration-0 defect 2, pinned at the writer.

    Before this, a shot with no `ip` wrote TWO incompatible rows -
    `qh_flattop skipped NaN..NaN` and `qh_proxy ok -0.097..4.097 0 events`
    - so ideate read `coverage_state: observed` for a phenomenon nobody
    could compute (measured on shot 198658). The proxy's OWN row has to
    carry the skip, because that is the source the registry consults.
    """
    _run(paths, model)
    rows = _sources(paths)
    status, reason, cov0, cov1, n = rows[("qh_proxy", "")]
    assert status == "skipped"
    assert "flat-top" in reason
    assert math.isnan(cov0) and math.isnan(cov1)
    assert n == 0


def test_with_a_flattop_the_qh_source_covers_the_intersection_of_its_inputs(
    shot_file, paths, synth_shot, model,
):
    """And when it CAN be evaluated, the window is every input's, not one.

    The proxy needs an EHO track, an ELM-free stretch, NBI and the
    flat-top at once. Here the stored current holds only over 0.2-0.6 s,
    which is narrower than the magnetics reference span the old code
    declared, so the row has to shrink to it.
    """
    ip = np.asarray(synth_shot["ip_y"], dtype=np.float64).copy()
    t = np.asarray(synth_shot["ip_t_s"], dtype=np.float64)
    ip[(t < 0.2) | (t > 0.6)] = 0.0
    _write_features(paths, SHOT, {**synth_shot, "ip_y": ip})
    _run(paths, model)
    status, reason, cov0, cov1, _ = _sources(paths)[("qh_proxy", "")]
    assert (status, reason) == ("ok", "")
    assert (cov0, cov1) == pytest.approx((0.2, 0.6), abs=2e-3)
    # Narrower than the block the ELM clock ran on, which is what the row
    # used to claim on its own.
    ref = _sources(paths)[("tokeye_transient", "mhr")]
    assert cov0 > ref[2] and cov1 < ref[3]


def test_the_qh_proxy_takes_the_dalpha_clocks_span_not_the_magnetics_reference(
    shot_file, paths, synth_shot, model,
):
    """The ELM-free input is the D-alpha clock's, so its window is the clock's.

    Task L-A moved the ELM clock off the magnetics reference block and onto
    `filterscopes`; task L-D2's proxy coverage was written against the
    reference. Here the D-alpha record is finite only over 0.1-0.5 s while
    every magnetics block spans the whole 0-0.8 s, so a proxy row that read
    the reference's span would over-claim by 0.4 s.
    """
    _write_features(paths, SHOT, synth_shot)
    with h5py.File(shot_file, "a") as f:
        t = f["filterscopes/xdata"][:]
        y = f["filterscopes/ydata"][:]
        y[:, (t < 0.1) | (t > 0.5)] = np.nan
        f["filterscopes/ydata"][...] = y
    _run(paths, model)
    rows = _sources(paths)
    status, reason, cov0, cov1, _ = rows[("qh_proxy", "")]
    assert (status, reason) == ("ok", "")
    assert (cov0, cov1) == pytest.approx((0.1, 0.5), abs=2e-3)
    clock = rows[("elm_clock", "filterscopes")]
    assert (clock[2], clock[3]) == pytest.approx((cov0, cov1), abs=1e-9)
    ref = rows[("tokeye_transient", "mhr")]
    assert ref[2] < 0.05 and ref[3] > 0.75


def test_a_skipped_dalpha_clock_makes_the_qh_proxy_unevaluable(
    shot_file, paths, synth_shot, model,
):
    """No D-alpha, no ELM-free intervals, no proxy - however many magnetics
    blocks ran. Before this the reference block stood in for the clock and
    the proxy declared `ok` over a span nobody measured ELMs on."""
    _write_features(paths, SHOT, synth_shot)
    with h5py.File(shot_file, "a") as f:
        del f["filterscopes"]
    _run(paths, model)
    rows = _sources(paths)
    status, reason, cov0, cov1, n = rows[("qh_proxy", "")]
    assert status == "skipped" and n == 0
    assert pl.QH_NEEDS_CLOCK in reason
    assert np.isnan(cov0) and np.isnan(cov1)
    assert rows[("tokeye_transient", "mhr")][0] == "ok"


def test_rejected_blocks_do_not_donate_tracks_or_coverage_to_the_qh_proxy(
    shot_file, paths, synth_shot, model, monkeypatch,
):
    """A block whose conversion failed is unknown, not quiet.

    Every block's conversion fails here, so nothing was PUBLISHED - and the
    proxy, which is built on published tracks, must say it had none, even
    though the tracker ran on every block.
    """
    def boom(*a, **kw):
        raise ValueError("invalid padded track")

    monkeypatch.setattr(tracks, "tracks_to_events", boom)
    _write_features(paths, SHOT, synth_shot)
    res = _run(paths, model)
    assert any(k.startswith("track mhr:") for k in res.skipped)
    status, reason, _, _, n = _sources(paths)[("qh_proxy", "")]
    assert status == "skipped" and n == 0
    assert pl.QH_NEEDS_TRACKS in reason


def test_counter_injection_is_skipped_or_covers_all_three_of_its_inputs(
    shot_file, paths, synth_shot, model,
):
    """Same rule for `nbi_counter`: the torque AND the power AND the current.

    Without a features file there is no `ip` and the row is skipped; with
    one whose current is measured over a stretch narrower than the torque
    record, the row is the intersection and not the torque's own axis.
    """
    _run(paths, model)
    status, reason, cov0, cov1, _ = _sources(paths)[("actuator", "tinj_total")]
    assert status == "skipped"
    assert "ip" in reason
    assert math.isnan(cov0) and math.isnan(cov1)

    t = np.asarray(synth_shot["ip_t_s"], dtype=np.float64)
    inside = (t >= 0.2) & (t <= 0.6)
    ip = np.where(inside, np.asarray(synth_shot["ip_y"], dtype=np.float64),
                  np.nan)
    _write_features(paths, SHOT, {**synth_shot, "ip_y": ip})
    _run(paths, model)
    status, reason, cov0, cov1, n = _sources(paths)[("actuator", "tinj_total")]
    assert (status, reason) == ("ok", "")
    assert (cov0, cov1) == pytest.approx((0.2, 0.6), abs=2e-3)
    # The row counts the rows it is the coverage OF: `_actuator_event`
    # stamps `diag="tinj_total"` on `nbi_counter` and on nothing else.
    assert n >= 1
    ev = schema.read_events(paths.events_file(SHOT), source="actuator")
    assert set(ev[ev["diag"] == "tinj_total"]["phenomenon"]) == {"nbi_counter"}


def test_inputs_that_never_overlap_are_no_coverage_at_all(shot_file, paths,
                                                          synth_shot, model):
    """Measured everywhere, together nowhere: still not an observation."""
    # The stored current is a perfectly good record on its own axis - it
    # simply runs after the corpus' beam and torque records have stopped,
    # so there is no instant at which all three were known.
    t = np.asarray(synth_shot["ip_t_s"], dtype=np.float64) + 2.0
    _write_features(paths, SHOT, {**synth_shot, "ip_t_s": t})
    res = _run(paths, model)
    assert "no common coverage" in res.skipped["nbi_counter"]
    rows = _sources(paths)
    assert rows[("actuator", "tinj_total")][0] == "skipped"


# ------------------------------------------------------------- --rules-only

def test_rules_only_needs_neither_the_corpus_nor_the_network(paths, synth_shot,
                                                             monkeypatch):
    """The whole point of the mode: no corpus file, no U-Net, no masks."""
    _write_features(paths, SHOT, synth_shot)
    paths.mkdirs()
    monkeypatch.setattr(
        unet, "load_unet",
        lambda *a, **k: pytest.fail("--rules-only loaded the network"),
    )
    code = run.main([
        "events", "--rules-only", "--shots", str(SHOT),
        "--root", str(paths.root), "--corpus-dir", str(paths.corpus),
    ])
    assert code == 0
    assert not list(paths.masks.glob("*.npz"))
    df = schema.read_events(paths.events_file(SHOT))
    assert list(df["phenomenon"]) == ["qmin_hybrid"]
    assert list(df["source"]) == ["qmin_rule"]


def test_rules_only_writes_the_sources_file_and_the_index(paths, synth_shot):
    _write_features(paths, SHOT, synth_shot)
    paths.mkdirs()
    assert run.main([
        "events", "--rules-only", "--shots", str(SHOT),
        "--root", str(paths.root),
    ]) == 0
    rows = _sources(paths)
    assert rows[("qmin_rule", "qmin")][0] == "ok"
    assert rows[("features", "ip")][0] == "ok"
    index = pd.read_parquet(paths.events_index)
    assert set(index["source"]) == {"qmin_rule"}


def test_rules_only_counts_the_bands_it_found_over_the_shot_list(
    paths, synth_shot, capsys,
):
    """The summary the 500-shot run is read off."""
    _write_features(paths, SHOT, synth_shot)
    _write_features(paths, SHOT + 1, synth_shot, qmin_value=3.0)
    _write_features(paths, SHOT + 2, synth_shot, qmin=False)
    paths.mkdirs()
    assert run.main([
        "events", "--rules-only", "--root", str(paths.root),
        "--shots", str(SHOT), str(SHOT + 1), str(SHOT + 2),
    ]) == 0
    out = capsys.readouterr().out
    assert "qmin_hybrid=1" in out
    assert "qmin_high=1" in out
    payload = json.loads(
        max((paths.runs / "events").glob("*.json")).read_text()
    )
    totals = payload["totals"]
    assert totals["shots_with_band"] == {"qmin_hybrid": 1, "qmin_high": 1}
    assert totals["n_shots"] == 3
    assert totals["shots_skipping"]["qmin_rule"] == 1


def test_rules_only_records_a_shot_with_no_features_file_and_carries_on(
    paths, synth_shot,
):
    _write_features(paths, SHOT, synth_shot)
    paths.mkdirs()
    assert run.main([
        "events", "--rules-only", "--root", str(paths.root),
        "--shots", str(SHOT), str(SHOT + 9),
    ]) == 0
    payload = json.loads(
        max((paths.runs / "events").glob("*.json")).read_text()
    )
    rows = {int(r["shot"]): r for r in payload["shots"]}
    assert rows[SHOT + 9]["status"] == "ok"
    assert "features" in rows[SHOT + 9]["skipped"]
    assert rows[SHOT]["n_events"] == 1


def test_rules_only_belongs_to_the_events_stage_alone(paths, tmp_path):
    with pytest.raises(SystemExit):
        run.main(["features", "--rules-only", "--shots", "1",
                  "--root", str(tmp_path)])
    with pytest.raises(SystemExit):
        run.main(["events", "--rules-only", "--databases-only",
                  "--shots", "1", "--root", str(tmp_path)])


@pytest.mark.parametrize("damage", ["phenomenon", "columns", "malformed", "missing"])
def test_an_invalid_format_csv_stops_the_run_before_any_shot(
    paths, tmp_path, monkeypatch, capsys, no_network, damage,
):
    root = _label_tables(tmp_path, [(SHOT, 300.0)])
    table = next(root.glob("*/format/*.csv"))
    if damage == "missing":
        table.unlink()
    elif damage == "malformed":
        table.write_text('shot,t0_s\n"unterminated\n', encoding="utf-8")
    else:
        frame = pd.read_csv(table, keep_default_na=False)
        if damage == "phenomenon":
            frame["phenomenon"] = "not_a_lexicon_id"
        else:
            frame = frame.drop(columns="attrs")
        frame.to_csv(table, index=False)
    monkeypatch.setenv("LABELMAKER_LABEL_TABLES", str(root))
    assert run.main([
        "events", "--databases-only", "--root", str(paths.root),
        "--shots", str(SHOT), str(SHOT + 1),
    ]) == run.EXIT_BAD_LABEL_TABLE
    captured = capsys.readouterr()
    assert "refusing to run" in captured.err and str(table) in captured.err
    assert ": ERROR" not in captured.out
    assert not paths.events_file(SHOT).exists()
    assert not paths.sources_file(SHOT).exists()
    assert not paths.events_index.exists()
