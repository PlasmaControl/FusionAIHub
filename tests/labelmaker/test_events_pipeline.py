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
    transients,
    unet,
)
from labelmaker.events import lexicon as lx
from labelmaker.events import pipeline as pl

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


def test_the_elm_clock_runs_on_one_reference_channel(shot_file, paths, model):
    # Every magnetics block carries the same comb, and writing an ELM per
    # channel puts every crash in the table N times - which is what
    # `windows.EventTable`'s de-duplication exists to survive, not what it
    # should be fed. One reference channel, named in the result.
    res = _run(paths, model)
    df = schema.read_events(paths.events_file(SHOT), source="tokeye_transient")
    assert set(zip(df["diag"], df["channel"], df["pass_name"], strict=True)) == {
        ("mhr", 0, "wide")
    }
    assert res.elm_reference == "mhr_00_wide"
    assert res.n_elms == len(ELM_COLS)


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


def test_nbi_counter_is_a_recorded_skip_because_ip_is_not_in_the_corpus(
    shot_file, paths, model,
):
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
                                                            model):
    # An events file says what was FOUND. `ech_power_total` is zero for
    # this whole shot and `qh_proxy` claims nothing without an Ip
    # flat-top, so neither writes an event row - and both ran.
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
    # And the two caveats that fire while the run succeeds get rows of
    # their own rather than colliding with the step they qualify.
    assert rows[("nbi_counter", "", -1)][0] == "skipped"
    assert "not a corpus group" in rows[("qh_flattop", "", -1)][1]
    assert rows[("text", "", -1)] == ("skipped", "no lexicon passed")
    assert set(src["shot"]) == {SHOT}


def test_a_failed_step_leaves_both_of_its_sources_visible(shot_file, paths,
                                                          model, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("no transient trace")

    monkeypatch.setattr(transients, "elm_events", boom)
    _run(paths, model)
    src = schema.read_sources(paths.sources_file(SHOT))
    got = src[src["source"].isin(["tokeye_transient", "elm_clock"])]
    assert len(got) == 2                       # one step, two sources, two rows
    assert set(got["status"]) == {"skipped"}
    assert all("RuntimeError" in r for r in got["reason"])


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
