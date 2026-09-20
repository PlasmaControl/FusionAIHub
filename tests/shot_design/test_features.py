"""Ported from shot-recommender-system (shotrec) @565d548."""

# tests/test_features.py
import numpy as np
import pytest

from shot_design import config
from shot_design.schema import Segment
from shot_design.shotdb import features, legacy_raw

CFG = config.load_yaml("retrieval.yaml")["segments"]


def _sig(shot, name, group, col, paths, **kw):
    return legacy_raw.read_signal(
        shot, config.SignalSpec(name=name, group=group, col=col, **kw), paths
    )


def test_find_segments_trapezoid(paths, staged_shot_a):
    # scale=1.0e6: ipsip is megaamps on disk (see signals.yaml's ip.scale comment).
    ip = _sig(staged_shot_a, "ip", "ip", "ipsip", paths, abs=True, scale=1.0e6)
    segs = {s.name: s for s in features.find_segments(ip, CFG)}
    assert set(segs) == {"full", "ramp_up", "flat_top", "ramp_down"}
    # 0.85 * 1.2 MA is crossed at t=680 ms on the way up and at t=4905 ms on the way down
    assert abs(segs["flat_top"].t0_ms - 680) <= 15 and abs(segs["flat_top"].t1_ms - 4905) <= 15
    assert abs(segs["ramp_up"].t0_ms - 80) <= 15 and abs(segs["ramp_down"].t1_ms - 5430) <= 15


def test_no_plasma_gives_no_segments(paths, staged_shot_a):
    # scale=1.0e6: ipsip is megaamps on disk (see signals.yaml's ip.scale comment).
    ip = _sig(staged_shot_a, "ip", "ip", "ipsip", paths, abs=True, scale=1.0e6)
    ip.y = ip.y * 0.01  # 12 kA: below min_ip_a
    assert features.find_segments(ip, CFG) == []


def test_proxy_segments_splits_pulse_length_into_the_four_segments():
    segs = {s.name: s for s in features.proxy_segments(5.0, CFG)}
    assert set(segs) == {"full", "ramp_up", "flat_top", "ramp_down"}
    assert segs["full"].t0_ms == pytest.approx(0.0)
    assert segs["full"].t1_ms == pytest.approx(5000.0)
    assert segs["ramp_up"].t0_ms == pytest.approx(0.0)
    assert segs["ramp_up"].t1_ms == pytest.approx(1000.0)
    assert segs["flat_top"].t0_ms == pytest.approx(1000.0)
    assert segs["flat_top"].t1_ms == pytest.approx(4730.0)
    assert segs["ramp_down"].t0_ms == pytest.approx(4730.0)
    assert segs["ramp_down"].t1_ms == pytest.approx(5000.0)


@pytest.mark.parametrize("pulse_length_s", [1.2, None, float("nan")])
def test_proxy_segments_is_empty_when_there_is_no_flat_top(pulse_length_s):
    assert features.proxy_segments(pulse_length_s, CFG) == []


def test_stats_follow_the_peak_convention(paths, staged_shot_a):
    beam = _sig(
        staged_shot_a,
        "pnbi_15L",
        "p_inj",
        "pinjf_15l",
        paths,
        system="nbi",
        stats=["mean", "peak", "on_frac"],
    )
    seg = Segment(name="flat_top", t0_ms=680.0, t1_ms=4905.0)
    raw, derived = features.segment_scalars(
        {"pnbi_15L": beam},
        seg,
        [
            config.SignalSpec(
                name="pnbi_15L",
                group="p_inj",
                col="pinjf_15l",
                system="nbi",
                stats=["mean", "peak", "on_frac"],
            )
        ],
    )
    assert derived == {}
    assert abs(raw["pnbi_15L_peak"] - 2.0e6) < 1.0  # 95th percentile of the nonzero samples
    assert abs(raw["pnbi_15L_mean"] - 2.0e6) < 1.0  # mean of the same nonzero subset
    assert 0.80 < raw["pnbi_15L_on_frac"] < 0.86  # beam on 1.0-4.5 s inside a 0.68-4.9 s window


def test_peak_is_95th_percentile_not_max():
    """test_stats_follow_the_peak_convention's beam is constant (2.0e6) for the whole "on" span,
    so mean/peak/max all coincide there and a `max()` bug would pass it unnoticed. Here 4% of the
    in-window samples are a 100x spike -- below the 5% the 95th percentile discounts -- so peak
    must equal the base level while the true max sits far above it."""
    y = np.concatenate([np.full(96, 10.0), np.full(4, 1000.0)])
    t = np.arange(y.size, dtype=np.float64)
    assert features.STATS["peak"](t, y, False) == 10.0
    assert float(np.max(y)) == 1000.0


def test_efit_needs_three_slices(paths, staged_shot_a):
    betan = _sig(staged_shot_a, "betan", "beta", "betan", paths, tier="derived", stats=["mean"])
    spec = config.SignalSpec(
        name="betan", group="beta", col="betan", tier="derived", stats=["mean"]
    )
    _, ok = features.segment_scalars(
        {"betan": betan}, Segment(name="flat_top", t0_ms=1000, t1_ms=2000), [spec]
    )
    _, short = features.segment_scalars(
        {"betan": betan}, Segment(name="flat_top", t0_ms=1000, t1_ms=1040), [spec]
    )
    assert abs(ok["betan_mean"] - 1.9) < 1e-6 and short["betan_mean"] is None


def test_totals_and_shape(paths, staged_shot_a):
    systems = config.actuator_systems(staged_shot_a)
    signals = {
        f"pnbi_{b}": _sig(
            staged_shot_a, f"pnbi_{b}", "p_inj", f"pinjf_{b.lower()}", paths, system="nbi"
        )
        for b in systems["nbi"].members
    }
    totals = features.system_totals(signals, systems)
    tot = totals["pnbi_total"]
    assert abs(float(tot.y[(tot.t_ms > 1500) & (tot.t_ms < 4000)].mean()) - 4.5e6) < 1.0
    shape = features.waveform_shape(tot, Segment(name="flat_top", t0_ms=680, t1_ms=4905), n=20)
    assert shape.shape == (20,) and shape.max() <= 1.0 + 1e-6 and shape[0] == 0.0


def test_ip_outcome_detects_disruption(paths, staged_shot_a, our_shot_c):
    for shot, expect_quench in ((staged_shot_a, False), (our_shot_c, True)):
        # scale=1.0e6: ipsip is megaamps on disk; iptipp is already amps (see signals.yaml).
        ip = _sig(shot, "ip", "ip", "ipsip", paths, abs=True, scale=1.0e6)
        tgt = _sig(shot, "ip_target", "ip", "iptipp", paths, abs=True)
        segs = features.find_segments(ip, CFG)
        out = features.ip_outcome(ip, tgt, segs, CFG)
        assert out["fast_quench"] is expect_quench and out["ended_early"] is expect_quench
    assert out["end_reason"] == "fast_current_quench"


def test_infer_nbi_target_unit():
    assert features.infer_nbi_target_mw(5.0, 5.0e6) == (5.0, "MW")  # log10 ratio 6 -> MW
    assert features.infer_nbi_target_mw(5000.0, 5.0e6) == (5.0, "kW")  # log10 ratio 3 -> kW
    assert features.infer_nbi_target_mw(50.0, 5.0e6) == (None, None)  # ambiguous band


def test_system_total_survives_unrecorded_members():
    """Real pattern confirmed against /scratch/gpfs/EKOLEMEN/d3d_fusion_data/160904.h5's
    ech/ecr2dfpwrc column (199995 of 201006 samples NaN): a gyrotron can be almost entirely
    unrecorded while a sibling on the same system reports fine. The system total must track the
    healthy member everywhere, and only go unknown (NaN) where literally nobody recorded."""
    t = np.arange(1000.0)
    healthy_y = np.full(1000, 800.0)
    healthy_y[7] = np.nan  # even the "healthy" member has one genuinely unrecorded sample
    healthy = legacy_raw.Signal(t, healthy_y, "W", "staged", "ech", "tower_a")
    flaky_y = np.full(1000, np.nan)
    flaky_recorded = np.arange(1000) % 10 == 0
    flaky_y[flaky_recorded] = 50.0  # recorded (and genuinely reading 50 W) at 10% of samples
    flaky = legacy_raw.Signal(t, flaky_y, "W", "staged", "ech", "tower_b")
    systems = {"ech": config.SystemSpec(name="ech", prefix="pech", members=["a", "b"])}
    total = features.system_totals({"pech_a": healthy, "pech_b": flaky}, systems)["pech_total"]
    # A nonzero recorded value on the flaky member pins the actual sum -- an implementation that
    # dropped a partially-recorded member entirely, instead of summing what it did record, would
    # still pass a same-as-healthy-alone comparison if that member's readings were all zero.
    expect = healthy_y + np.where(flaky_recorded, 50.0, 0.0)
    elsewhere = np.arange(1000) != 7
    assert np.array_equal(total.y[elsewhere], expect[elsewhere])  # flaky's real value is summed in
    assert np.isnan(total.y[7])  # nobody recorded here -> unknown, not a lying zero


def test_system_totals_no_fabricated_zero_outside_member_coverage():
    """A member on a shorter time grid is resampled onto the reference grid; outside its own
    recorded span that resample must not fabricate a 0.0 (the old np.interp(..., left=0.0,
    right=0.0)) -- paired with a sibling that is entirely unrecorded, an invented zero there would
    read as a confident "system off" instead of the honest "nobody recorded this" NaN."""
    ref_t = np.arange(0.0, 100.0, 1.0)
    dead = legacy_raw.Signal(ref_t, np.full(100, np.nan), "W", "staged", "ech", "tower_a")
    short_t = np.arange(20.0, 60.0, 1.0)  # covers only the middle 40% of the reference window
    partial = legacy_raw.Signal(short_t, np.full(40, 5.0), "W", "staged", "ech", "tower_b")
    systems = {"ech": config.SystemSpec(name="ech", prefix="pech", members=["a", "b"])}
    total = features.system_totals({"pech_a": dead, "pech_b": partial}, systems)["pech_total"]
    outside = (total.t_ms < 20.0) | (total.t_ms >= 60.0)
    inside = (total.t_ms >= 20.0) & (total.t_ms < 60.0)
    assert np.all(np.isnan(total.y[outside]))  # nobody recorded here -- not a lying zero
    assert np.allclose(total.y[inside], 5.0)  # the one member that did cover this span is summed


def test_all_stats_return_none_for_all_nan_window():
    """An instrument that recorded nothing at all over the whole window is uncomputable, not
    zero -- a NaN silently written into the database would poison every downstream distance
    computation that touches this column."""
    t = np.arange(10.0)
    y = np.full(10, np.nan)
    for name, fn in features.STATS.items():
        assert fn(t, y, False) is None, name
        assert fn(t, y, True) is None, name


def test_stats_compute_over_finite_samples_only():
    """A partially-recorded window must give exactly the answer a fully-recorded window of just
    its finite samples would -- the unrecorded samples are dropped, never averaged, minimized, or
    fit into as if they were real zeros or any other real value."""
    t_partial = np.arange(10.0)
    y_partial = np.array([5.0, 6.0, np.nan, 7.0, 9.0, np.nan, 4.0, 8.0, np.nan, 3.0])
    finite = np.isfinite(y_partial)
    t_clean, y_clean = t_partial[finite], y_partial[finite]
    for name, fn in features.STATS.items():
        got, want = fn(t_partial, y_partial, False), fn(t_clean, y_clean, False)
        assert got == want, name


def test_on_frac_none_when_peak_uncomputable_not_zero():
    """The old code asked "is this ever above 5% of its peak?" and silently answered "no"
    whenever the peak itself could not be computed -- indistinguishable from a genuinely idle
    actuator. An entirely unrecorded window must say "unknown", not "off"."""
    t = np.arange(5.0)
    y = np.full(5, np.nan)
    assert features.stat_on_frac(t, y, True) is None
    assert features.stat_on_frac(t, y, False) is None


def test_on_frac_zero_for_recorded_idle_actuator():
    """A recorded, idle actuator is the opposite failure from the test above: real pattern
    confirmed against /scratch/gpfs/EKOLEMEN/d3d_fusion_data/160904.h5's ech/ecr2dfpwrc column
    (1,011 samples recorded, all exactly 0.0) -- an installed gyrotron that simply never fired
    during this window. Unlike the all-NaN case, this *is* knowable: it was confidently never on,
    so on_frac must be a real 0.0, not the same "unknown" None as a channel nothing was recorded
    on at all."""
    t = np.arange(20.0)
    y = np.zeros(20)  # recorded throughout, never positive
    assert features.stat_on_frac(t, y, True) == 0.0
    # stat_min proves the window really was recorded (a real number, not None), so this row can't
    # be confused downstream with the unrecorded case either.
    assert features.stat_min(t, y, True) == 0.0


def test_segment_scalars_min_samples_gate_counts_finite_only():
    """Three raw samples but only one finite one must not be enough to report a number -- the
    min_samples floor exists so an entry is never quietly backed by less real data than it claims."""
    sig = legacy_raw.Signal(
        np.array([0.0, 1.0, 2.0]), np.array([5.0, np.nan, np.nan]), "A", "staged", "grp", "col"
    )
    spec = config.SignalSpec(name="x", group="grp", col="col", stats=["mean"])
    raw, _ = features.segment_scalars(
        {"x": sig}, Segment(name="flat_top", t0_ms=0.0, t1_ms=2.0), [spec]
    )
    assert raw["x_mean"] is None


def test_waveform_shape_normalizes_with_partial_nan_gaps():
    """A source with an ecr2dfpwrc-style gap (long NaN runs around a short recorded span, per
    /scratch/gpfs/EKOLEMEN/d3d_fusion_data/160904.h5) must still normalize on what it did record,
    not fall through to raw ~1e6 magnitudes sitting next to every other, normalized, feature."""
    t = np.arange(0.0, 1000.0, 1.0)
    y = np.full_like(t, np.nan)
    y[400:600] = 1.0e6  # only the middle fifth of the window was actually recorded
    sig = legacy_raw.Signal(t, y, "W", "staged", "ech", "tower")
    shape = features.waveform_shape(sig, Segment(name="flat_top", t0_ms=0.0, t1_ms=999.0), n=20)
    recorded = shape[np.isfinite(shape)]
    assert recorded.size > 0
    assert np.all(np.abs(recorded) <= 1.0 + 1e-6)
    assert recorded.max() > 0.5  # actually normalized, not left at ~1e6 physical units


def test_waveform_shape_zero_for_recorded_flat_zero_signal():
    """A fully recorded, identically zero trace (an idle gyrotron -- see
    test_on_frac_zero_for_recorded_idle_actuator for the same real column) has no magnitude to
    normalize by, but flat zero *is* its correct shape. It must come back as zeros, not collapse
    into the same all-NaN return used for a signal nothing was recorded on at all."""
    t = np.arange(0.0, 1000.0, 1.0)
    y = np.zeros_like(t)  # fully recorded, genuinely off throughout
    sig = legacy_raw.Signal(t, y, "W", "staged", "ech", "tower")
    shape = features.waveform_shape(sig, Segment(name="flat_top", t0_ms=0.0, t1_ms=999.0), n=20)
    assert shape.shape == (20,)
    assert np.all(shape == 0.0)


def test_ip_outcome_unusable_signal_not_confused_with_no_plasma():
    """A NaN run at least as wide as the median-filter kernel (11 samples at this 1 kHz cadence,
    per retrieval.yaml's median_ms=10.0) is too wide for _nanmedian_filter to fully heal and
    leaves a residual NaN patch in the smoothed signal; that residual still makes the (unchanged,
    still NaN-propagating) percentile reference level NaN, so every threshold comparison in
    find_segments is False and it returns [] exactly as it would for a shot that never had plasma.
    Measured: this 200-sample gap is far past that threshold -- a gap shorter than the kernel, like
    test_short_ip_gap_keeps_segments's 5 samples, now heals completely instead. ip_outcome must
    still tell the two apart."""
    t = np.arange(-500.0, 6500.0, 1.0)
    y = np.zeros_like(t)
    y[(t >= 800) & (t <= 4800)] = 1.2e6
    ramp_up = (t >= 0) & (t < 800)
    y[ramp_up] = 1.2e6 * t[ramp_up] / 800.0
    ramp_dn = (t > 4800) & (t <= 5500)
    y[ramp_dn] = 1.2e6 * (5500.0 - t[ramp_dn]) / 700.0
    y[(t >= 2000) & (t < 2200)] = np.nan  # a 200 ms DAQ dropout, well inside the ip_ref window
    ip = legacy_raw.Signal(t, y, "A", "staged", "ip", "ipsip")
    segs = features.find_segments(ip, CFG)
    assert segs == []  # confirms this hits the same empty-segments path as genuine "no plasma"
    out = features.ip_outcome(ip, None, segs, CFG)
    assert out["end_reason"] == "ip_signal_unusable"


def test_short_ip_gap_keeps_segments():
    """A measured example: a brief DAQ dropout well after the plasma has already ended must not
    take the whole shot's segmentation down with it. This 5-sample gap is shorter than the
    11-sample kernel (see test_ip_outcome_unusable_signal_not_confused_with_no_plasma), so
    _nanmedian_filter heals it completely and every segment survives -- at essentially the same
    boundaries as the identical, gap-free trapezoid in test_find_segments_trapezoid."""
    t = np.arange(-500.0, 6500.0, 1.0)
    y = np.zeros_like(t)
    y[(t >= 800) & (t <= 4800)] = 1.2e6
    ramp_up = (t >= 0) & (t < 800)
    y[ramp_up] = 1.2e6 * t[ramp_up] / 800.0
    ramp_dn = (t > 4800) & (t <= 5500)
    y[ramp_dn] = 1.2e6 * (5500.0 - t[ramp_dn]) / 700.0
    y[(t >= 5800) & (t < 5805)] = np.nan  # a 5-sample dropout, long after the plasma has ended
    ip = legacy_raw.Signal(t, y, "A", "staged", "ip", "ipsip")
    segs = {s.name: s for s in features.find_segments(ip, CFG)}
    assert set(segs) == {"full", "ramp_up", "flat_top", "ramp_down"}
    assert abs(segs["flat_top"].t0_ms - 680) <= 15 and abs(segs["flat_top"].t1_ms - 4905) <= 15


# --- Fix wave 3: edge-healed gaps, honest target-comparison outcomes, unusable short records --


def test_gap_heals_at_start_middle_and_end_up_to_kernel_width():
    """_nanmedian_filter used to pad a boundary window by repeating the boundary sample itself
    (scipy's mode="nearest"), so a gap touching either end of the array padded itself with more
    NaN and only healed up to half the kernel width there -- confirmed: at k=11, a 10-sample
    interior gap fully healed but a 6-sample gap at the very start or end did not. The fix shifts
    a boundary window inward (never shrinks or pads it) so every k-wide window is built entirely
    from real samples. Every gap length from 1 to k-1=10 must now heal completely regardless of
    where it falls -- this is the direct numerical claim task-7-report.md's Fix wave 3 records."""
    n, k, level = 200, 11, 1.2e6
    base = np.full(n, level)
    for gap_len in range(1, k):  # 1..10, every length shorter than the kernel
        for where, start in (("start", 0), ("middle", 95), ("end", n - gap_len)):
            y = base.copy()
            y[start : start + gap_len] = np.nan
            healed = features._nanmedian_filter(y, k)
            assert np.isfinite(healed).all(), f"gap_len={gap_len} at {where} left a residual NaN"
            np.testing.assert_allclose(healed, level)


def test_too_few_ip_samples_reports_unusable_not_no_plasma():
    """Fewer than ten recorded samples in the analysis window is a data-quality problem, not
    evidence the shot never had a plasma: _smooth_ip's own floor (t.size < 10) used to return a
    reference level of 0.0, which find_segments reads as "no plasma" and ip_outcome then reports
    as end_reason="no_plasma" -- a confident, wrong conclusion from a handful of samples that
    prove nothing either way. The fix returns a non-finite reference level instead, routing this
    into the branch that already exists for a current signal too gappy to trust."""
    t = np.arange(0.0, 9.0, 1.0)  # 9 samples: one short of the 10-sample floor
    y = np.full(9, 1.2e6)
    ip = legacy_raw.Signal(t, y, "A", "staged", "ip", "ipsip")
    segs = features.find_segments(ip, CFG)
    assert segs == []
    out = features.ip_outcome(ip, None, segs, CFG)
    assert out["end_reason"] == "ip_signal_unusable"


def _clean_trapezoid_ip(t):
    y = np.zeros_like(t)
    y[(t >= 800) & (t <= 4800)] = 1.2e6
    ramp_up = (t >= 0) & (t < 800)
    y[ramp_up] = 1.2e6 * t[ramp_up] / 800.0
    ramp_dn = (t > 4800) & (t <= 5500)
    y[ramp_dn] = 1.2e6 * (5500.0 - t[ramp_dn]) / 700.0
    return y


def test_one_unrecorded_target_sample_does_not_move_the_reported_error():
    """ip_outcome used to average the raw windowed target/measurement arrays with plain np.mean,
    so a single unrecorded sample in either one turned the mean -- and then the error -- into
    NaN, and `abs(NaN) <= tolerance` is always False: a confident "missed the target" for a shot
    we actually know nothing about. Computed from recorded samples only, one dropped sample must
    not move the reported error at all versus the same shot with no gap."""
    t = np.arange(-500.0, 6500.0, 1.0)
    ip = legacy_raw.Signal(t, _clean_trapezoid_ip(t), "A", "staged", "ip", "ipsip")
    segs = features.find_segments(ip, CFG)
    target_y = np.full_like(t, 1.25e6)
    clean = features.ip_outcome(
        ip, legacy_raw.Signal(t, target_y.copy(), "A", "staged", "ip", "iptipp"), segs, CFG
    )
    gappy_y = target_y.copy()
    gappy_y[(t >= 2000) & (t < 2001)] = np.nan  # one unrecorded sample, well inside the flat top
    gappy = features.ip_outcome(
        ip, legacy_raw.Signal(t, gappy_y, "A", "staged", "ip", "iptipp"), segs, CFG
    )
    assert clean["ip_target_err"] is not None
    assert abs(gappy["ip_target_err"] - clean["ip_target_err"]) < 1e-9
    assert gappy["ip_target_hit"] == clean["ip_target_hit"]


def test_fully_unrecorded_target_window_yields_unknown_not_a_miss():
    """The opposite extreme from the test above: a flat top with zero recorded target samples
    cannot say whether the target was hit or missed. It must come back None (unknown) for both
    ip_target_err and ip_target_hit -- never NaN, and never a confident-but-baseless False."""
    t = np.arange(-500.0, 6500.0, 1.0)
    ip = legacy_raw.Signal(t, _clean_trapezoid_ip(t), "A", "staged", "ip", "ipsip")
    segs = features.find_segments(ip, CFG)
    tgt = legacy_raw.Signal(t, np.full_like(t, np.nan), "A", "staged", "ip", "iptipp")
    out = features.ip_outcome(ip, tgt, segs, CFG)
    assert out["ip_target_err"] is None
    assert out["ip_target_hit"] is None


def test_ended_early_unknown_when_target_window_unrecorded_and_not_fast():
    """The post-shot early-termination check has the same NaN-vs-unknown flaw as the flat-top
    error above: np.max is also NaN-propagating, so a target signal entirely unrecorded in the
    500 ms after the shot ends used to silently read as "definitely not ended early". Only
    fast_quench (computed from Ip alone, independent of the target signal) can rule early
    termination out on its own; when it says no and the target-based half is unrecorded, the
    combined answer must stay unknown rather than default to a confident False."""
    t = np.arange(-500.0, 6500.0, 1.0)
    y = _clean_trapezoid_ip(t)  # a slow, programmed rampdown -- not a fast quench
    ip = legacy_raw.Signal(t, y, "A", "staged", "ip", "ipsip")
    segs = features.find_segments(ip, CFG)
    tgt = legacy_raw.Signal(t, np.full_like(t, np.nan), "A", "staged", "ip", "iptipp")
    out = features.ip_outcome(ip, tgt, segs, CFG)
    assert out["fast_quench"] is False
    assert out["ended_early"] is None
    assert out["end_reason"] == "target_signal_unusable"


# --- Fix wave 4: honest ended_early/fast_quench when there is nothing to judge them on --------


def test_ended_early_unknown_when_target_signal_absent_not_just_unrecorded():
    """`ended_early` used to default to a confident False and only get overwritten when
    `ip_target is not None` -- so a shot whose target channel was simply never fetched (a normal,
    expected data-coverage gap, not a corrupted read: see legacy_raw.read_signal returning None) fell
    straight through to that default and was reported as a plain "programmed_rampdown", identical
    to a shot we actually know followed its program. No existing test reaches this path: every
    other ip_outcome call in this file either passes a real (if gappy or all-NaN) target Signal,
    or hits the earlier `flat is None` return before `ended_early` is ever touched -- confirmed by
    inspection of every `ip_outcome(` call site and every existing `ended_early`/`end_reason`
    assertion in this file before writing this test. This constructs the one case that reaches it:
    segments found, target signal absent."""
    t = np.arange(-500.0, 6500.0, 1.0)
    ip = legacy_raw.Signal(t, _clean_trapezoid_ip(t), "A", "staged", "ip", "ipsip")
    segs = features.find_segments(ip, CFG)
    assert {s.name for s in segs} >= {"flat_top", "full"}  # confirms this reaches ended_early
    out = features.ip_outcome(ip, None, segs, CFG)
    assert out["ended_early"] is None
    assert out["end_reason"] == "target_signal_unusable"


def test_fast_quench_unknown_when_post_flattop_region_is_unrecorded():
    """fast_quench's own comparisons (`y > 0.5*ip_ref`, `y < 0.1*ip_ref`) are False for every NaN
    sample -- the same failure shape already fixed above for the target comparisons. A residual
    gap (wider than _nanmedian_filter's kernel, left unhealed) spanning the *entire* region after
    the last above-50%-of-ip_ref sample would otherwise read as a confident "never dropped below
    10%" -- "no quench" -- even though that region was never actually observed.

    The gap must fall outside window_ms=[0, 6000] (retrieval.yaml) or it also poisons ip_ref's own
    percentile and takes the flat top down with it too (confirmed while building this test: NaN
    comparisons are False there as well, so `y >= flat_top_frac * NaN` matches nothing and
    find_segments returns [] -- so this situation genuinely is reachable, just not with an
    ordinary post-rampdown gap sitting inside the reference window). Holding the current at its
    plateau level well past 6000 ms, instead of letting it ramp down there, sidesteps that: the
    entire post-plateau gap then falls outside the ip_ref window, so the flat top is still found
    intact, and there is still nothing recorded to judge the quench criterion on.
    """
    t = np.arange(-500.0, 6500.0, 1.0)
    y = np.zeros_like(t)
    up = (t >= 0) & (t < 800)
    y[up] = 1.2e6 * t[up] / 800.0
    y[(t >= 800) & (t < 6050)] = 1.2e6  # held high well past window_ms's 6000 ms upper bound
    y[t >= 6050] = np.nan  # DAQ never recovers again -- far past the 11-sample kernel width
    ip = legacy_raw.Signal(t, y, "A", "staged", "ip", "ipsip")
    segs = features.find_segments(ip, CFG)
    assert {s.name for s in segs} >= {"flat_top", "full"}  # the gap did not also erase these
    out = features.ip_outcome(ip, None, segs, CFG)
    assert out["fast_quench"] is None


def test_fast_quench_still_definite_for_a_normal_shot():
    """Pairs with the test above: a fully-recorded rampdown must still report a real True/False
    verdict, not fall back to the new unknown path just because it now exists."""
    t = np.arange(-500.0, 6500.0, 1.0)
    ip = legacy_raw.Signal(t, _clean_trapezoid_ip(t), "A", "staged", "ip", "ipsip")
    segs = features.find_segments(ip, CFG)
    out = features.ip_outcome(ip, None, segs, CFG)
    assert out["fast_quench"] is False


def test_std_of_a_float32_dalpha_sized_signal_is_finite():
    """D-alpha is ~1e15-1e18 ph/cm^2/sr/s and the raw store is float32: a float32 sum of squares
    overflows to inf, which the first 200-shot build wrote into dalpha_std for 5 segments."""
    t = np.arange(20_000, dtype=np.float64)
    y = (1.0e18 + 1.0e17 * np.sin(t / 50.0)).astype(np.float32)
    s = features.stat_std(t, y, actuator=False)
    assert s is not None and np.isfinite(s)
    assert s == pytest.approx(float(np.std(y.astype(np.float64))), rel=1e-6)
    assert features.stat_mean(t, y, actuator=False) == pytest.approx(1.0e18, rel=1e-3)


def test_a_statistic_is_never_inf():
    t = np.arange(4, dtype=np.float64)
    assert (
        features.stat_min(t, np.array([1.0, np.inf, 2.0, 3.0]), actuator=False) == 1.0
    )  # inf dropped
    # a variance that overflows even float64 comes back as None, not inf
    assert features.stat_std(t[:2], np.array([1e308, -1e308]), actuator=False) is None
