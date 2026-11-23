"""Ladder passes, cache-side errors and the choice of K: rules 2 to 5 of the plan, as code."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from tokamak_foundation_model.ignite.mast import tasks, transform
from tokamak_foundation_model.ignite.mast.tokenizers import assign, codecs, ladder

IP = tasks.SignalSpec("summary-ip", 0.00025, (1,), 3.5e5, 3.0e5)
KAPPA = tasks.SignalSpec("equilibrium-elongation", 0.005, (1,), 1.6, 0.3)
LCFS = tasks.SignalSpec("equilibrium-lcfs_r", 0.005, (170,), 0.9, 0.4)
CC, OMV = (tasks.SignalSpec(n, 0.001024, (3, 256), 0.0, 1.0) for n in tasks.SPECTROGRAM_SIGNALS)
KS = [1, 2, 3, 4, 6, 8, 12, 16]
IP_FLOORS = [0.076, 0.038, 0.025, 0.016, 0.012, 0.009, 0.005, 0.002]
SXR_FLOORS = [0.098, 0.081, 0.075, 0.069, 0.067, 0.065, 0.060, 0.055]


def rows(signal, domain, ks, floors, raw=np.nan, passed=None):
    passed = [np.nan] * len(ks) if passed is None else passed
    return pd.DataFrame({"signal": signal, "domain": domain, "k": ks, "floor": floors, "raw": raw, "passed": passed})


def band_rows(signal, low, rest, contrast):
    a = pd.DataFrame({"signal": signal, "step": "a", "k0": [4, 8, 16, 32], "k": 32, "floor": low, "contrast": np.nan})
    b = pd.DataFrame({"signal": signal, "step": "b", "k0": np.nan, "k": [1, 2, 4, 8, 16, 32], "floor": rest,
                      "contrast": contrast})
    return pd.concat([a, b])


def spectro(contrast_cc, contrast_omv):
    return pd.concat([
        band_rows(CC.name, [0.760, 0.700, 0.688, 0.676], [0.6890, 0.6885, 0.6882, 0.6881, 0.6880, 0.6880], contrast_cc),
        band_rows(OMV.name, [0.477, 0.438, 0.424, 0.422], [0.470, 0.440, 0.428, 0.425, 0.424, 0.424], contrast_omv)])


def unit_stats(*shape):
    return transform.ChannelStats(compress="none", scale=np.ones(shape), mean=np.zeros(shape), std=np.ones(shape))


def test_the_passes_of_a_task_cover_every_domain_and_rung(toy_task):
    p = ladder.passes(toy_task, {"summary-ip": "none"})
    assert [(d, i) for d, i, _ in p] == [("linear", i) for i in range(8)]
    assert p[3][2] == {"summary-ip": assign.Setting("frame_dct", (4,))}
    two = dataclasses.replace(toy_task, outputs=(IP, KAPPA))
    p = ladder.passes(two, {"summary-ip": "asinh", "equilibrium-elongation": "none"})
    assert len(p) == 16 and p[8][:2] == ("compressed", 0) and p[8][2]["summary-ip"].domain == "compressed"
    assert all(s["equilibrium-elongation"] == assign.Setting("frame_dct", (1,)) for _, _, s in p)
    assert ladder.values_per_frame(LCFS, assign.Setting("linear", (32, 4), "signal")) == 32
    assert ladder.values_per_frame(CC, assign.Setting("band_dct", (16, 8, 8, 8, 8, 8, 8, 8), "band")) == 3 * 72
    assert ladder.values_per_frame(tasks.SignalSpec("c", 0.00025, (10,), 0.0, 1.0), assign.Setting("frame_dct", (4,))) == 40
    coils = tasks.SignalSpec("c", 0.00025, (10,), 0.0, 1.0)                 # the two codec names of plan 1c
    assert ladder.values_per_frame(coils, assign.Setting("recoded", (4,), source="klt.series")) == 40
    assert ladder.values_per_frame(coils, assign.Setting("recoded_concat", (24,), source="pca.c")) == 24
    assert ladder.values_per_frame(LCFS, assign.Setting("recoded", (32,), "signal", source="net.profile.s0")) == 32
    assert ladder.values_per_frame(CC, assign.Setting("recoded", (16, 8, 8, 8, 8, 8, 8, 8), "band", source="klt.band")) == 3 * 72
    with pytest.raises(ValueError):
        ladder.values_per_frame(coils, assign.Setting("something", (4,)))


def test_run_ladder_scores_every_pass_and_keeps_what_was_not_measured_missing(toy_task):
    seen = []

    def build(settings):
        return {"summary-ip": codecs.FrameDCT("summary-ip", 0.00025, (1,), unit_stats(1), k=settings["summary-ip"].k[0])}

    def score(kind, tokenizers, name):
        seen.append((kind, name))
        return {"summary-ip": 0.08 / tokenizers["summary-ip"].n_val} if kind == "floor" else None

    table = ladder.run_ladder(toy_task, ladder.passes(toy_task, {"summary-ip": "none"}), build, score)
    assert table["k"].tolist() == KS and table["values_per_frame"].tolist() == KS
    assert table["floor"].tolist() == pytest.approx([0.08 / k for k in KS])
    assert table["passed"].isna().all() and table["raw"].isna().all() and (table["domain"] == "linear").all()
    assert [n for kind, n in seen if kind == "floor"] == [f"ladder-linear-{i}" for i in range(8)]
    assert seen.count(("raw", "raw")) == 1


def test_rule_2_the_smallest_rung_that_meets_the_target():
    lin = rows("summary-ip", "linear", KS, IP_FLOORS)
    domain, index, note = ladder.choose(lin, 0.0176, ("linear",))
    assert (domain, index) == ("linear", 3) and note.startswith("meets 0.0176 with 0.0160")
    com = rows("summary-ip", "compressed", KS, [0.070, 0.030, 0.017, 0.015, 0.012, 0.009, 0.005, 0.002])
    assert ladder.choose(pd.concat([lin, com]), 0.0176, ("linear", "compressed"))[:2] == ("compressed", 2)
    tie = pd.concat([lin, rows("summary-ip", "compressed", KS, IP_FLOORS)])
    assert ladder.choose(tie, 0.0176, ("linear", "compressed"))[:2] == ("linear", 3)      # a tie: the earlier domain


def test_rule_3_when_no_rung_meets_the_target_the_pass_through_cost_decides():
    name = "soft_x_rays-horizontal_cam_lower"
    sxr = rows(name, "linear", KS, SXR_FLOORS, raw=0.2000,
               passed=[0.2100, 0.2030, 0.2016, 0.2010, 0.2008, 0.2005, 0.2003, 0.2001])
    domain, index, note = ladder.choose(sxr, 0.0424, ("linear",))
    assert (domain, index) == ("linear", 2)                    # 0.2016 / 0.2000 - 1 = 0.8%, the first at or under 1%
    assert note.startswith("misses 0.0424: floor 0.0750") and "pass-through cost 0.8%" in note
    domain, index, note = ladder.choose(rows(name, "linear", KS, SXR_FLOORS), 0.0424, ("linear",))
    assert (domain, index) == ("linear", 7) and "no pass-through measurement" in note
    costly = rows(name, "linear", KS, SXR_FLOORS, raw=0.2, passed=[0.25] * 8)
    assert ladder.choose(costly, 0.0424, ("linear",))[1] == 7                  # no rung within 1%: the largest


def test_a_ladder_that_was_not_scored_is_not_a_choice():
    with pytest.raises(ValueError, match="no ladder"):
        ladder.choose(rows("summary-ip", "linear", [], []), 0.0176, ("linear",))


def test_rule_5_two_steps_for_the_spectrograms_and_the_contrast_condition():
    k, note = ladder.choose_band(spectro([0.30, 0.55, 0.80, 0.93, 0.99, 1.0], [0.35, 0.60, 0.85, 0.95, 0.99, 1.0]))
    assert k == (16, 8, 8, 8, 8, 8, 8, 8)        # the benchmark alone would take 4; the contrast asks for 8
    assert "0.6760" in note and "0.4220" in note                               # the frame oracle of both signals
    assert ladder.choose_band(spectro([0.95] * 6, [0.95] * 6))[0] == (16, 4, 4, 4, 4, 4, 4, 4)


def test_the_cache_side_error_is_in_tokamark_units_and_skips_what_is_missing():
    spec = tasks.SignalSpec("s", 0.00025, (1,), 0.0, 2.0)
    z = np.tile(np.r_[np.zeros(10), np.ones(10)], 3)[None].astype(np.float32)   # 3 frames, each half 0 and half 1
    c1 = codecs.FrameDCT("s", 0.00025, (1,), unit_stats(1), k=1)
    assert ladder.shot_error(c1, spec, z) == pytest.approx(0.25)               # rms 0.5 about the frame mean, std 2
    assert ladder.shot_error(codecs.FrameDCT("s", 0.00025, (1,), unit_stats(1), k=20), spec, z) == pytest.approx(0.0, abs=1e-9)
    z[0, 20:40] = np.nan
    assert ladder.shot_error(c1, spec, z) == pytest.approx(0.25)               # a frame without data adds no zero
    assert ladder.shot_error(c1, spec, z[:, :50]) == pytest.approx(0.25)       # a last partial frame is left out
    assert ladder.shot_error(c1, spec, np.full((1, 60), np.nan, np.float32)) is None


def test_the_contrast_falls_when_a_spectral_line_is_averaged_away():
    st = transform.ChannelStats(compress="asinh", scale=np.ones((1, 256)), mean=np.zeros((1, 256)), std=np.ones((1, 256)))
    p = np.ones((1, 256, 50))                                                  # linear power, 11 frames
    p[0, 100] = 100.0                                                          # a line at bin 100
    z = transform.to_model(p, st).astype(np.float32)
    whole = ladder.band_contrast(codecs.BandDCT("m", 0.001024, (1, 256), st, [32] * 8), z)
    flat = ladder.band_contrast(codecs.BandDCT("m", 0.001024, (1, 256), st, [1] * 8), z)
    assert whole[0] == pytest.approx(1.0, abs=1e-4) and whole[1] > 0 and flat[0] < 0.5
    assert ladder.band_contrast(codecs.BandDCT("m", 0.001024, (1, 256), st, [1] * 8),
                                np.full((1, 256, 50), np.nan, np.float32)) is None


def test_build_assignment_applies_the_rules_signal_by_signal():
    pca = [0, 8, 16, 24, 32, 48, 64, 96]
    ladders = pd.concat([
        rows("summary-ip", "linear", KS, IP_FLOORS),
        rows("equilibrium-lcfs_r", "signal", pca, [0.20, 0.012, 0.0093, 0.0080, 0.0072, 0.0058, 0.0046, 0.0030]),
        rows("equilibrium-lcfs_r", "linear", pca, [0.20, 0.013, 0.0100, 0.0090, 0.0080, 0.0070, 0.0050, 0.0030]),
        spectro([0.95] * 6, [0.95] * 6)])
    compress = {"summary-ip": "none", "equilibrium-elongation": "none", "equilibrium-lcfs_r": "none",
                CC.name: "asinh", OMV.name: "asinh"}
    targets = {"summary-ip": 0.0176, "equilibrium-elongation": 0.0176, "equilibrium-lcfs_r": 0.0074,
               CC.name: None, OMV.name: None}
    a, notes = ladder.build_assignment([IP, KAPPA, LCFS, CC, OMV], compress, ladders, targets)
    assert a["summary-ip"] == assign.Setting("frame_dct", (4,))
    assert a["equilibrium-elongation"] == assign.Setting("frame_dct", (1,))
    assert a["equilibrium-lcfs_r"] == assign.Setting("linear", (32, 4), "signal")
    assert a[CC.name] == a[OMV.name] == assign.Setting("band_dct", (16, 4, 4, 4, 4, 4, 4, 4), "band")
    assert set(notes) == set(a) and notes["equilibrium-elongation"].startswith("one sample per frame")
    with pytest.raises(ValueError, match="summary-ip"):
        ladder.build_assignment([IP], compress, ladders[ladders["signal"] != "summary-ip"], targets)


def test_channel_errors_gives_per_channel_sums_and_counts_over_the_same_samples_as_shot_error():
    spec = tasks.SignalSpec("s2", 0.00025, (2,), 0.0, 2.0)
    half = np.r_[np.zeros(10), np.ones(10)]
    z = np.stack([np.tile(half, 3), np.tile(half, 3)]).astype(np.float32)       # 2 channels, 3 frames of 20
    z[1, 25] = np.nan                                                           # one sample of channel 1 missing
    codec = codecs.FrameDCT("s2", 0.00025, (2,), unit_stats(2), k=1)
    sq, n = ladder.channel_errors(codec, spec, z)
    assert sq.shape == n.shape == (2,)
    assert n.tolist() == [60, 59]                                               # NaN is not a sample, not a zero
    assert sq[0] == pytest.approx(60 * 0.25 ** 2)                               # error 0.5 / std 2 at every sample
    assert sq[1] == pytest.approx(59 * 0.25 ** 2, rel=0.02)                     # frame 1's mean moved a little
    whole = ladder.channel_errors(codec, spec, z[:, :50])                       # the last 10 samples are not a frame
    assert whole[1].tolist() == [40, 39]
    assert ladder.shot_error(codec, spec, z) == pytest.approx(np.sqrt(sq.sum() / n.sum()))
    nothing = ladder.channel_errors(codec, spec, np.full((2, 40), np.nan, np.float32))
    assert nothing[1].tolist() == [0, 0] and nothing[0].tolist() == [0.0, 0.0]

