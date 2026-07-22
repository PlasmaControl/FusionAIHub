"""Known-answer validation of the IGNITE acceptance gate.

This is the whole point of an oracle gate: each metric is exercised on synthetic tensors
whose *correct* answer we know analytically, and we assert the metric returns it. The
gate decides whether a codec's codes may train a world model (design §4.4), so a metric
that silently mis-scores is worse than no gate at all.

CPU only, small tensors, numpy + a torch-input smoke check (framework-agnostic contract).
"""
from __future__ import annotations

import numpy as np
import pytest

from tokamak_foundation_model.ignite import gate


rng = np.random.default_rng(0)


# ======================================================================================
# 1. stability  — fraction of code positions unchanged under the nuisance δ-shift
# ======================================================================================
def _rand_codes(B=4, n_tok=24, fsq_dim=5, levels=8):
    return rng.integers(0, levels, size=(B, n_tok, fsq_dim)).astype(np.int64)


def test_stability_identical_is_one():
    """Identical codes => every position matches => stability == 1.0 exactly."""
    codes = _rand_codes()
    assert gate.stability(codes, codes.copy()) == pytest.approx(1.0)


def test_stability_all_different_is_zero():
    """Codes that differ at EVERY position => stability == 0.0 exactly."""
    a = np.zeros((3, 24, 5), dtype=np.int64)
    b = np.ones((3, 24, 5), dtype=np.int64)  # differs everywhere
    assert gate.stability(a, b) == pytest.approx(0.0)


def test_stability_known_fraction_flipped():
    """Flip a KNOWN fraction of positions => stability == 1 - that fraction.

    We build a (10, 10, 10) = 1000-position tensor and flip exactly 250 positions, so the
    unchanged fraction must be exactly 0.75.
    """
    shape = (10, 10, 10)
    a = np.zeros(shape, dtype=np.int64)
    b = a.copy()
    flat = b.reshape(-1)
    n_flip = 250
    idx = rng.choice(flat.size, size=n_flip, replace=False)
    flat[idx] = 7  # any different symbol
    assert gate.stability(a, b) == pytest.approx(1.0 - n_flip / flat.size)


def test_stability_partial_fsq_flip_counts_proportionally():
    """Flipping some FSQ dims at a token counts proportionally (position-level metric).

    One token of one sample has 2 of 5 fsq dims changed; everything else identical.
    Unchanged fraction = (total - 2) / total.
    """
    a = _rand_codes(B=2, n_tok=3, fsq_dim=5)
    b = a.copy()
    b[0, 0, 0] = (b[0, 0, 0] + 1) % 8
    b[0, 0, 1] = (b[0, 0, 1] + 1) % 8
    total = a.size
    assert gate.stability(a, b) == pytest.approx((total - 2) / total)


def test_stability_meets_gate_threshold_on_good_codec():
    """A codec that flips <20% of codes passes the mandate (stability >= 0.80)."""
    a = np.zeros((10, 10, 10), dtype=np.int64)
    b = a.copy()
    b.reshape(-1)[: 100] = 3  # 10% changed
    s = gate.stability(a, b)
    assert s == pytest.approx(0.9)
    assert s >= 0.80  # passes design §4.4 mandate


def test_stability_shape_mismatch_raises():
    with pytest.raises(ValueError):
        gate.stability(_rand_codes(B=2), _rand_codes(B=3))


def test_stability_accepts_torch_input():
    """Framework-agnostic contract: torch tensors are accepted and give the same answer."""
    torch = pytest.importorskip("torch")
    a = _rand_codes()
    at = torch.from_numpy(a)
    assert gate.stability(at, at.clone()) == pytest.approx(1.0)


# ======================================================================================
# 2. persistence — fraction of code positions unchanged frame-to-frame
# ======================================================================================
def test_persistence_repeated_frame_is_one():
    """Identical consecutive frames => codes fully persist => 1.0."""
    c = _rand_codes()
    assert gate.persistence(c, c.copy()) == pytest.approx(1.0)


def test_persistence_all_change_is_zero():
    a = np.zeros((3, 24, 5), dtype=np.int64)
    b = a + 1
    assert gate.persistence(a, b) == pytest.approx(0.0)


def test_persistence_known_fraction():
    """Change a known fraction of positions => persistence == 1 - fraction."""
    shape = (8, 8, 8)  # 512 positions
    a = np.zeros(shape, dtype=np.int64)
    b = a.copy()
    n_change = 128  # quarter
    b.reshape(-1)[:n_change] = 5
    assert gate.persistence(a, b) == pytest.approx(1.0 - n_change / a.size)


def test_persistence_steady_mask_restricts_to_subset():
    """steady_mask restricts the average to the masked (steady) samples only.

    Sample 0 is fully steady (no change); sample 1 changes everywhere. A per-sample mask
    selecting only sample 0 must yield persistence 1.0; selecting only sample 1 => 0.0.
    """
    a = np.zeros((2, 4, 5), dtype=np.int64)
    b = a.copy()
    b[1] = 1  # sample 1 changes everywhere, sample 0 unchanged
    steady_only_0 = np.array([True, False])
    steady_only_1 = np.array([False, True])
    assert gate.persistence(a, b, steady_mask=steady_only_0) == pytest.approx(1.0)
    assert gate.persistence(a, b, steady_mask=steady_only_1) == pytest.approx(0.0)
    # Unmasked average = half the positions changed:
    assert gate.persistence(a, b) == pytest.approx(0.5)


def test_persistence_full_shape_mask():
    """A full (B, n_tok, fsq_dim) boolean mask selects arbitrary positions."""
    a = np.zeros((2, 3, 4), dtype=np.int64)
    b = a.copy()
    b[0, 0, 0] = 9  # one changed position
    mask = np.zeros_like(a, dtype=bool)
    mask[0, 0, 0] = True   # select the changed position
    mask[1, 1, 1] = True   # and one unchanged position
    # 1 of 2 selected positions unchanged => 0.5
    assert gate.persistence(a, b, steady_mask=mask) == pytest.approx(0.5)


def test_persistence_steady_higher_than_transition():
    """Design mandate: persistence measurably higher on steady than transition segments.

    Steady frame pair (mostly repeated codes) must score well above a transition pair
    (many codes moved). This is exactly the stratified comparison §4.4 asks for.
    """
    n_tok, fsq = 24, 5
    base = _rand_codes(B=6, n_tok=n_tok, fsq_dim=fsq)
    # steady: flip only ~5% of codes
    steady_next = base.copy()
    flat = steady_next.reshape(-1)
    flat[rng.choice(flat.size, size=int(0.05 * flat.size), replace=False)] = 1
    # transition: flip ~60% of codes
    trans_next = base.copy()
    flat = trans_next.reshape(-1)
    flat[rng.choice(flat.size, size=int(0.60 * flat.size), replace=False)] = 1
    p_steady = gate.persistence(base, steady_next)
    p_trans = gate.persistence(base, trans_next)
    assert p_steady >= 0.50          # mandate floor
    assert p_steady > p_trans + 0.2  # measurably lower on transition


def test_persistence_empty_mask_raises():
    a = np.zeros((2, 3, 4), dtype=np.int64)
    with pytest.raises(ValueError):
        gate.persistence(a, a.copy(), steady_mask=np.zeros((2,), dtype=bool))


# ======================================================================================
# 3. forecastability — cheap Markov probe vs persistence baseline, per stratum
# ======================================================================================
def _drifting_sequence(B=8, n_frames=12, n_tok=4, fsq_dim=2, levels=8):
    """A perfectly PREDICTABLE non-persistent sequence: each code index increments by 1
    (mod levels) every frame. Persistence (predict next==current) is always WRONG here,
    so a first-order Markov probe (which learns 'i -> i+1') must strictly beat it.
    """
    seq = np.zeros((B, n_frames, n_tok, fsq_dim), dtype=np.int64)
    start = rng.integers(0, levels, size=(B, n_tok, fsq_dim))
    for f in range(n_frames):
        seq[:, f] = (start + f) % levels
    return seq


def _frozen_sequence(B=8, n_frames=12, n_tok=4, fsq_dim=2, levels=8):
    """A perfectly PERSISTENT sequence: codes never change. Persistence is optimal;
    the probe can at best tie (there are no transitions to beat on).
    """
    frame = rng.integers(0, levels, size=(B, n_tok, fsq_dim))
    return np.broadcast_to(frame[:, None], (B, n_frames, n_tok, fsq_dim)).copy()


def test_forecastability_probe_beats_persistence_on_drift():
    """On a drifting (predictable, non-persistent) sequence, the Markov probe must beat
    persistence on the TRANSITION stratum (margin > 0) — the gate's pass condition."""
    seq = _drifting_sequence()
    res = gate.forecastability(seq)
    assert res["n_transition"] > 0
    assert res["n_stable"] == 0            # every step moves
    assert res["margin_transition"] > 0.0  # probe strictly better
    assert res["beats_persistence"] is True


def test_forecastability_ties_persistence_on_frozen():
    """On a frozen sequence, there are no transitions; persistence is optimal and the
    probe ties it (transition margin reported as 0.0, stable margin ~ 0)."""
    seq = _frozen_sequence()
    res = gate.forecastability(seq)
    assert res["n_transition"] == 0
    assert res["n_stable"] > 0
    assert res["margin_transition"] == pytest.approx(0.0)  # no transition stratum
    # On stable steps the probe cannot beat a perfect persistence predictor:
    assert res["margin_stable"] <= 1e-9


def test_forecastability_probe_not_worse_than_persistence_on_drift_overall():
    """Sanity: overall the probe's mean log-lik >= persistence's on the predictable seq."""
    seq = _drifting_sequence()
    res = gate.forecastability(seq)
    assert res["probe_ll_transition"] >= res["persistence_ll_transition"]
    assert res["margin_overall"] > 0.0


def test_forecastability_persistence_wins_on_sticky_transitions():
    """A sequence that mostly repeats but occasionally jumps to a RANDOM new code:
    the transitions are unpredictable, so the probe should NOT be able to beat
    persistence by much on them (margin should be small / non-positive). This guards
    against a probe that spuriously reports skill on noise (cf. the mode-audit finding
    that the OLD codec's codes were realization-driven and unpredictable).
    """
    B, Fr, n_tok, fsq, levels = 12, 16, 4, 2, 8
    seq = np.zeros((B, Fr, n_tok, fsq), dtype=np.int64)
    seq[:, 0] = rng.integers(0, levels, size=(B, n_tok, fsq))
    for f in range(1, Fr):
        seq[:, f] = seq[:, f - 1]
        # ~15% of positions jump to a fresh random symbol (unpredictable)
        jump = rng.random((B, n_tok, fsq)) < 0.15
        seq[:, f][jump] = rng.integers(0, levels, size=int(jump.sum()))
    res = gate.forecastability(seq)
    assert res["n_transition"] > 0
    # Unpredictable jumps: the cheap probe should not manufacture large skill.
    assert res["margin_transition"] < res["margin_overall"] + 0.5
    # And on the predictable-drift case the margin is clearly larger than here:
    drift_margin = gate.forecastability(_drifting_sequence())["margin_transition"]
    assert drift_margin > res["margin_transition"]


def test_forecastability_respects_transition_mask_override():
    """An explicit transition_mask overrides the data-derived split."""
    seq = _drifting_sequence(B=3, n_frames=5)
    B, Fr = seq.shape[0], seq.shape[1]
    # Mark everything as STABLE via the override, even though the data drifts.
    tm = np.zeros((B, Fr - 1), dtype=bool)
    res = gate.forecastability(seq, transition_mask=tm)
    assert res["n_transition"] == 0
    assert res["n_stable"] == B * (Fr - 1)


def test_forecastability_requires_two_frames():
    with pytest.raises(ValueError):
        gate.forecastability(np.zeros((2, 1, 4, 2), dtype=np.int64))


def test_forecastability_bad_ndim_raises():
    with pytest.raises(ValueError):
        gate.forecastability(np.zeros((2, 4, 2), dtype=np.int64))


# ======================================================================================
# 4. decode_fidelity — mode-structure match + sharpness of reconstructed spectrogram
# ======================================================================================
def _mode_spectrogram(B=2, C=1, F=64, T=48, mode_freqs=(10, 30, 50), seed=1):
    """Synthetic log-power spectrogram with sharp narrow-band modes at known freqs.

    Background is low; each mode is a narrow high-power band with slight time texture, so
    there is real high-frequency gradient energy to blur away.
    """
    r = np.random.default_rng(seed)
    spec = 0.1 * r.standard_normal((B, C, F, T)) - 2.0  # low, noisy background
    freqs = np.arange(F)
    for mf in mode_freqs:
        band = np.exp(-0.5 * ((freqs - mf) / 1.5) ** 2)  # narrow gaussian in freq
        # time modulation so the mode has structure along T too
        tmod = 1.0 + 0.3 * np.sin(np.linspace(0, 6 * np.pi, T))
        spec += (3.0 * band[None, None, :, None]) * tmod[None, None, None, :]
    return spec


def test_decode_fidelity_identical_is_perfect():
    """recon == target => envelope corr ~1, peak F1 ~1, sharpness == 1 exactly."""
    tgt = _mode_spectrogram()
    res = gate.decode_fidelity(tgt.copy(), tgt)
    assert res["envelope_corr"] == pytest.approx(1.0, abs=1e-6)
    assert res["peak_f1"] == pytest.approx(1.0)
    assert res["sharpness"] == pytest.approx(1.0, rel=1e-6)


def test_decode_fidelity_blurred_collapses_sharpness():
    """A low-pass (blurred) reconstruction => sharpness << 1 (collapsed HF energy).

    We blur the target with a separable box filter; mode envelope is largely preserved
    (still correlated) but the high-frequency gradient energy drops sharply.
    """
    tgt = _mode_spectrogram()
    recon = _boxblur(tgt, k=5)
    res = gate.decode_fidelity(recon, tgt)
    assert res["sharpness"] < 0.5           # HF energy strongly reduced
    assert res["envelope_corr"] > 0.9       # coarse mode structure still there


def test_decode_fidelity_mean_collapse_is_worst():
    """The pathological mean-collapse (constant output = dataset mean) => envelope corr
    undefined-flat (returned 0), peak F1 low, sharpness ~0. This is exactly the failure
    §4.4 / the mode-collapse memory warns about.
    """
    tgt = _mode_spectrogram()
    recon = np.full_like(tgt, tgt.mean())
    res = gate.decode_fidelity(recon, tgt)
    assert res["sharpness"] == pytest.approx(0.0, abs=1e-9)  # no gradient energy at all
    assert res["envelope_corr"] == pytest.approx(0.0, abs=1e-9)  # flat env => 0
    assert res["peak_f1"] < 0.5             # no modes detected in recon


def test_decode_fidelity_wrong_modes_drop_peak_f1():
    """Modes at the WRONG frequencies => envelope corr drops and peak F1 drops.

    Same sharpness family, but the modes are placed at disjoint frequencies, so the
    peak-overlap detector should find little agreement.
    """
    tgt = _mode_spectrogram(mode_freqs=(10, 30, 50))
    recon = _mode_spectrogram(mode_freqs=(15, 35, 55), seed=2)  # shifted, disjoint peaks
    res = gate.decode_fidelity(recon, tgt)
    good = gate.decode_fidelity(tgt.copy(), tgt)
    assert res["peak_f1"] < good["peak_f1"]
    assert res["peak_f1"] < 0.5             # little mode overlap
    # sharpness ~1 (still sharp) but structure is wrong -> corr well below perfect
    assert res["envelope_corr"] < 0.9


def test_decode_fidelity_noisier_recon_sharpness_above_one():
    """If the recon is sharper/noisier than target, sharpness > 1 (documented behavior)."""
    tgt = _mode_spectrogram()
    recon = tgt + 0.5 * np.random.default_rng(3).standard_normal(tgt.shape)
    res = gate.decode_fidelity(recon, tgt)
    assert res["sharpness"] > 1.0


def test_decode_fidelity_shape_mismatch_raises():
    with pytest.raises(ValueError):
        gate.decode_fidelity(_mode_spectrogram(B=2), _mode_spectrogram(B=3))


def test_decode_fidelity_accepts_torch_input():
    torch = pytest.importorskip("torch")
    tgt = _mode_spectrogram()
    res = gate.decode_fidelity(torch.from_numpy(tgt.copy()), torch.from_numpy(tgt))
    assert res["sharpness"] == pytest.approx(1.0, rel=1e-6)


# --------------------------------------------------------------------------------------
# small test-local helper (not part of the gate API): separable box blur
# --------------------------------------------------------------------------------------
def _boxblur(spec: np.ndarray, k: int = 5) -> np.ndarray:
    """Separable moving-average blur over the (F, T) plane; a simple low-pass filter."""
    ker = np.ones(k) / k
    out = spec.astype(np.float64).copy()
    # blur along freq
    out = np.apply_along_axis(lambda m: np.convolve(m, ker, mode="same"), -2, out)
    # blur along time
    out = np.apply_along_axis(lambda m: np.convolve(m, ker, mode="same"), -1, out)
    return out


# ======================================================================================
# 5. utilization  — codebook-usage / collapse detector (anti-posterior-collapse)
# ======================================================================================
from tokamak_foundation_model.ignite.config import SpectroCodecConfig  # noqa: E402


def _util_cfg() -> SpectroCodecConfig:
    # small codebook (4*4*4 = 64) so a diverse code set can genuinely exceed the
    # utilization threshold; default gate thresholds (0.02 / 0.3).
    return SpectroCodecConfig(fsq_levels=[4, 4, 4])


def _all_codes(cfg: SpectroCodecConfig, B: int = 4) -> np.ndarray:
    """A code set that visits EVERY codebook entry (max diversity)."""
    g = np.stack(
        np.meshgrid(*[np.arange(l) for l in cfg.fsq_levels], indexing="ij"), axis=-1
    ).reshape(-1, cfg.fsq_dim)
    need = B * cfg.n_tok
    reps = int(np.ceil(need / g.shape[0]))
    tiled = np.tile(g, (reps, 1))[:need]
    return tiled.reshape(B, cfg.n_tok, cfg.fsq_dim).astype(np.int64)


def test_utilization_collapsed_all_identical():
    """All codes identical => 1 distinct code, tiny frac, zero entropy => collapsed."""
    cfg = _util_cfg()
    codes = np.zeros((4, cfg.n_tok, cfg.fsq_dim), dtype=np.int64)  # all the same code
    u = gate.utilization(codes, cfg=cfg)
    assert u["n_distinct_codes"] == 1
    assert u["frac_codes_used"] == pytest.approx(1.0 / cfg.codebook_size)
    assert u["frac_codes_used"] < cfg.gate_min_utilization
    assert u["min_dim_entropy"] == pytest.approx(0.0, abs=1e-9)
    assert u["collapsed"] is True


def test_utilization_diverse_not_collapsed():
    """A code set covering the whole codebook => frac 1.0, high entropy, not collapsed."""
    cfg = _util_cfg()
    codes = _all_codes(cfg)
    u = gate.utilization(codes, cfg=cfg)
    assert u["frac_codes_used"] == pytest.approx(1.0)
    # every dim is exercised across many levels -> high (near-max) per-dim entropy.
    assert u["min_dim_entropy"] > 0.9
    assert u["min_dim_entropy"] >= cfg.gate_min_code_entropy
    assert u["collapsed"] is False


def test_utilization_flags_one_collapsed_dim():
    """Even with diverse flat codes, a single frozen FSQ dim => collapsed via entropy."""
    cfg = _util_cfg()
    codes = _all_codes(cfg).copy()
    codes[..., 0] = 0  # freeze dim 0 to a single level -> its entropy is 0
    u = gate.utilization(codes, cfg=cfg)
    assert u["min_dim_entropy"] == pytest.approx(0.0, abs=1e-9)
    assert u["collapsed"] is True


def test_utilization_flattens_frame_axis():
    """4D (B, n_frames, n_tok, fsq_dim) input is accepted (frames flattened)."""
    cfg = _util_cfg()
    codes4d = np.zeros((2, 3, cfg.n_tok, cfg.fsq_dim), dtype=np.int64)
    u = gate.utilization(codes4d, cfg=cfg)
    assert u["n_distinct_codes"] == 1
    assert u["collapsed"] is True


def test_utilization_codebook_size_kwarg_without_cfg():
    """Thresholds + codebook_size can be passed explicitly (no cfg)."""
    cfg = _util_cfg()
    codes = _all_codes(cfg)
    u = gate.utilization(
        codes,
        codebook_size=cfg.codebook_size,
        min_utilization=0.02,
        min_code_entropy=0.3,
    )
    assert u["collapsed"] is False
    assert u["frac_codes_used"] == pytest.approx(1.0)


def test_utilization_accepts_torch_input():
    torch = pytest.importorskip("torch")
    cfg = _util_cfg()
    codes = torch.zeros((4, cfg.n_tok, cfg.fsq_dim), dtype=torch.long)
    u = gate.utilization(codes, cfg=cfg)
    assert u["collapsed"] is True


def test_utilization_large_codebook_small_eval_not_flagged():
    """REGRESSION: a healthy codec on a SMALL eval set against the REAL large codebook
    (32768) must NOT be flagged collapsed. The absolute frac_codes_used is confounded here
    (distinct / 32768 is tiny even if every eval token is distinct), so the collapse
    decision must use frac_of_observable + min_dim_entropy, not an absolute frac threshold."""
    cfg = SpectroCodecConfig()  # real default: fsq_levels [8,8,8,8,8] => codebook 32768
    n_tok = 64
    rng = np.random.default_rng(0)
    codes = np.stack(
        [rng.integers(0, cfg.fsq_levels[i], size=(1, n_tok)) for i in range(cfg.fsq_dim)],
        axis=-1,
    ).astype(np.int64)  # (1, n_tok, fsq_dim), diverse per-dim
    u = gate.utilization(codes, cfg=cfg)
    # The OLD absolute-frac criterion WOULD have flagged this (max achievable
    # frac_codes_used = n_tok / codebook_size = 64/32768 ~ 0.002 < gate_min_utilization).
    assert u["frac_codes_used"] < cfg.gate_min_utilization
    # But it is genuinely healthy: high per-dim entropy + high eval-relative utilization.
    assert u["min_dim_entropy"] > cfg.gate_min_code_entropy
    assert u["frac_of_observable"] > cfg.gate_min_utilization
    assert u["collapsed"] is False
