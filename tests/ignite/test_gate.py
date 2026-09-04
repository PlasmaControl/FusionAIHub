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
    """REGRESSION: a healthy codec on a SMALL eval set against a LARGE codebook (>> n_tok)
    must NOT be flagged collapsed. The absolute frac_codes_used is confounded here
    (distinct / codebook_size is tiny even if every eval token is distinct), so the collapse
    decision must use frac_of_observable + min_dim_entropy, not an absolute frac threshold.

    Uses the historical 32768 (=8^5) codebook explicitly (the pre-right-size default) as the
    large-codebook scenario, so the confound is present regardless of the current default."""
    cfg = SpectroCodecConfig(fsq_levels=[8, 8, 8, 8, 8])  # large codebook = 32768
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


# ======================================================================================
# 4a. patch_lattice_metrics — the checkerboard / patch-seam detector
# ======================================================================================
def _tiled_texture_spectrogram(B=2, C=1, F=64, T=48, pf=16, pt=16, amp=1.0, seed=3):
    """A spectrogram whose decoder painted every (pf x pt) patch from ONE shared texture.

    This is exactly what a per-token linear ``to_pixels`` + tile does: the same high-frequency
    basis pattern is replicated on the patch lattice, on top of smooth per-patch levels.
    """
    r = np.random.default_rng(seed)
    texture = r.standard_normal((C, pf, pt))                       # ONE shared patch texture
    tiled = np.tile(texture, (1, F // pf, T // pt))                # replicated on the lattice
    levels = r.standard_normal((B, C, F // pf, T // pt))           # smooth per-patch content
    smooth = np.repeat(np.repeat(levels, pf, axis=2), pt, axis=3)
    return smooth + amp * tiled[None]


def test_patch_lattice_ratio_is_one_on_lattice_free_data():
    """Ground-truth-like data (no tiled texture) reads ~1: the lattice bins look exactly like
    their neighbours. This is the control that makes an elevated recon number meaningful."""
    gt = _mode_spectrogram(B=4, F=64, T=48)
    m = gate.patch_lattice_metrics(gt, patch_f=16, patch_t=16)
    assert m["patch_lattice_ratio"] == pytest.approx(1.0, abs=0.35)
    assert m["lattice_energy_frac"] < 0.10
    # The seam ratios are DIAGNOSTIC, not primary: they read one position per boundary, so a
    # narrow mode that happens to sit on a boundary moves them (this synthetic has modes at
    # freq 10/30/50 vs boundaries at 15/31/47). On real mhr ground truth they read 1.003 /
    # 0.971. Assert only that they are not wildly elevated.
    assert 0.5 < m["seam_ratio_freq"] < 2.0
    assert 0.5 < m["seam_ratio_time"] < 2.0


def test_patch_lattice_ratio_fires_on_a_tiled_patch_texture():
    """A shared per-patch texture tiled on the 16x16 grid is detected, and MORE of it reads
    higher — the metric is monotone in artifact strength, not just a boolean."""
    weak = gate.patch_lattice_metrics(
        _tiled_texture_spectrogram(amp=0.3), patch_f=16, patch_t=16)
    strong = gate.patch_lattice_metrics(
        _tiled_texture_spectrogram(amp=3.0), patch_f=16, patch_t=16)
    assert strong["patch_lattice_ratio"] > weak["patch_lattice_ratio"] > 2.0
    assert strong["lattice_energy_frac"] > weak["lattice_energy_frac"]


def test_sharpness_cannot_see_the_lattice_but_the_metric_can():
    """The reason this metric had to exist: the lattice IS high-frequency gradient energy, so
    ``sharpness`` reads ~1 (or above) on a badly checkerboarded reconstruction."""
    tgt = _mode_spectrogram(B=4, F=64, T=48)
    recon = tgt + _tiled_texture_spectrogram(B=4, F=64, T=48, amp=0.5)
    res = gate.decode_fidelity(recon, tgt, patch_f=16, patch_t=16)
    assert res["sharpness"] > 1.0                       # blind: looks "sharp", not blurred
    assert res["patch_lattice_ratio"] > 3.0             # the artifact IS detected
    assert res["target_patch_lattice_ratio"] < 2.0      # ...and the GT control is not


def test_decode_fidelity_lattice_keys_absent_without_patch_size():
    """Default call is unchanged for every existing caller (no new keys, no cost)."""
    tgt = _mode_spectrogram()
    res = gate.decode_fidelity(tgt.copy(), tgt)
    assert "patch_lattice_ratio" not in res
    assert set(res) == {"envelope_corr", "peak_f1", "sharpness",
                        "hf_energy_recon", "hf_energy_target"}


def test_patch_lattice_metrics_rejects_indivisible_shape():
    with pytest.raises(ValueError):
        gate.patch_lattice_metrics(_mode_spectrogram(F=60, T=48), patch_f=16, patch_t=16)


def test_remove_patch_lattice_deletes_the_artifact_and_keeps_the_signal():
    """The ideal notch drives the lattice ratio to ~0 while leaving lattice-free content alone.

    This is the diagnostic behind the measured claim that 92% of the mhr baseline's HF gradient
    energy is the artifact: notch the recon, re-measure, compare.
    """
    tgt = _mode_spectrogram(B=4, F=64, T=48)
    recon = tgt + _tiled_texture_spectrogram(B=4, F=64, T=48, amp=1.0)
    before = gate.patch_lattice_metrics(recon, 16, 16)
    notched = gate.remove_patch_lattice(recon, 16, 16)
    after = gate.patch_lattice_metrics(notched, 16, 16)
    assert before["patch_lattice_ratio"] > 3.0
    assert after["lattice_energy_frac"] == pytest.approx(0.0, abs=1e-9)
    # the notch is a projection: applying it twice changes nothing further.
    assert np.allclose(notched, gate.remove_patch_lattice(notched, 16, 16))
    # and it moves the recon TOWARD the (lattice-free) target rather than away from it.
    assert np.abs(notched - tgt).mean() < np.abs(recon - tgt).mean()
    assert notched.shape == recon.shape


# ======================================================================================
# 4c. full_spectro_metrics / trivial_spectro_baselines — the FULL-array recon check
# ======================================================================================
def test_full_spectro_metrics_self_comparison_is_exact():
    """The wiring self-check: target vs itself MUST read nrmse 0.0 and corr2d 1.0."""
    tgt = _mode_spectrogram(B=3, C=2, F=64, T=48)
    m = gate.full_spectro_metrics(tgt.copy(), tgt, band_bins=32)
    assert m["spec_nrmse"] == pytest.approx(0.0, abs=1e-12)
    assert m["spec_corr2d"] == pytest.approx(1.0, abs=1e-12)
    assert m["spec_nrmse_band"] == pytest.approx(0.0, abs=1e-12)
    assert m["spec_corr2d_band"] == pytest.approx(1.0, abs=1e-12)
    assert m["spec_valid_frac"] == pytest.approx(1.0)


def test_full_spectro_nrmse_anchor_is_the_per_window_mean():
    """The stated normalisation: predicting each (window, channel)'s own scalar mean scores
    EXACTLY nrmse 1.0 and corr2d 0.0 — that is what makes ``< 1`` mean something."""
    tgt = _mode_spectrogram(B=3, C=2, F=64, T=48)
    const = np.broadcast_to(tgt.mean(axis=(-2, -1), keepdims=True), tgt.shape)
    m = gate.full_spectro_metrics(const, tgt, band_bins=None)
    assert m["spec_nrmse"] == pytest.approx(1.0, abs=1e-12)
    assert m["spec_corr2d"] == pytest.approx(0.0, abs=1e-12)
    assert "spec_nrmse_band" not in m  # band_bins=None disables the band variant


def _drifting_mode_spectrogram(B=3, C=2, F=64, T=48, seed=7):
    """Spectrogram whose modes DRIFT in frequency across the window — a mode TRACK.

    ``_mode_spectrogram``'s modes sit at a fixed frequency with a mild time modulation, so its
    time-averaged envelope already explains ~97 % of the array and the two metrics barely
    separate. Real mhr/ece windows carry drifting tracks and bursts, which is exactly the
    structure the time-collapsed metrics cannot see.
    """
    r = np.random.default_rng(seed)
    freqs = np.arange(F)[None, :, None].astype(float)
    tt = np.linspace(0.0, 1.0, T)[None, None, :]
    spec = 0.1 * r.standard_normal((B * C, F, T)) - 2.0
    for _ in range(3):
        f0 = r.uniform(0.15 * F, 0.75 * F, size=(B * C, 1, 1))
        drift = r.uniform(-0.25 * F, 0.25 * F, size=(B * C, 1, 1))
        spec += 3.0 * np.exp(-0.5 * ((freqs - (f0 + drift * tt)) / 1.5) ** 2)
    return spec.reshape(B, C, F, T)


def test_full_spectro_metrics_sees_what_envelope_corr_cannot():
    """A recon that reproduces the time-averaged envelope PERFECTLY but has no temporal
    structure scores envelope_corr 1.0 while corr2d stays well below it. This is the whole
    reason the metric exists."""
    tgt = _drifting_mode_spectrogram(B=3, C=2, F=64, T=48)
    flat = np.broadcast_to(tgt.mean(axis=-1, keepdims=True), tgt.shape).copy()
    old = gate.decode_fidelity(flat, tgt)
    new = gate.full_spectro_metrics(flat, tgt, band_bins=None)
    assert old["envelope_corr"] == pytest.approx(1.0, abs=1e-9)  # blind by construction
    assert new["spec_corr2d"] < old["envelope_corr"] - 0.2
    assert 0.0 < new["spec_nrmse"] < 1.0


def test_full_spectro_metrics_excludes_constant_target_channels():
    """Absent/dead diagnostics (constant target) have no structure to reconstruct: they are
    dropped from both means and counted in ``spec_valid_frac`` instead of dividing by zero."""
    tgt = _mode_spectrogram(B=2, C=4, F=64, T=48)
    tgt[:, 0] = 7.0  # one dead channel of four
    m = gate.full_spectro_metrics(tgt.copy(), tgt, band_bins=None)
    assert m["spec_valid_frac"] == pytest.approx(0.75)
    assert m["spec_nrmse"] == pytest.approx(0.0, abs=1e-12)
    assert m["spec_corr2d"] == pytest.approx(1.0, abs=1e-12)


def test_full_spectro_metrics_rejects_bad_shapes():
    with pytest.raises(ValueError):
        gate.full_spectro_metrics(_mode_spectrogram(B=2), _mode_spectrogram(B=3))
    with pytest.raises(ValueError):
        gate.full_spectro_metrics(np.zeros((4, 8, 6)), np.zeros((4, 8, 6)))


def test_trivial_spectro_baselines_are_ordered_and_anchored():
    """self (0.0 / 1.0) is the best possible, wcmean (1.0 / 0.0) the anchor, and the
    time-mean envelope baseline sits strictly between them."""
    tgt = _mode_spectrogram(B=3, C=2, F=64, T=48)
    b = gate.trivial_spectro_baselines(tgt, band_bins=32)
    assert b["base_self_spec_nrmse"] == pytest.approx(0.0, abs=1e-12)
    assert b["base_self_spec_corr2d"] == pytest.approx(1.0, abs=1e-12)
    assert b["base_wcmean_spec_nrmse"] == pytest.approx(1.0, abs=1e-12)
    assert b["base_wcmean_spec_corr2d"] == pytest.approx(0.0, abs=1e-12)
    assert 0.0 < b["base_tmean_spec_nrmse"] < b["base_wcmean_spec_nrmse"]
    assert b["base_wcmean_spec_corr2d"] < b["base_tmean_spec_corr2d"] < 1.0
    assert "base_self_spec_valid_frac" not in b  # identical for every baseline; not repeated


def test_decode_fidelity_full_spec_is_opt_in_and_additive():
    """DEFAULT OFF: the default dict is unchanged; full_spec=True only ADDS keys and leaves
    every pre-existing value bit-identical."""
    tgt = _mode_spectrogram(B=2, C=2, F=64, T=48)
    recon = tgt + 0.3 * np.random.default_rng(4).standard_normal(tgt.shape)
    off = gate.decode_fidelity(recon, tgt, patch_f=16, patch_t=16)
    on = gate.decode_fidelity(recon, tgt, patch_f=16, patch_t=16, full_spec=True)
    assert "spec_corr2d" not in off
    assert set(off).issubset(set(on))
    for k, v in off.items():
        assert on[k] == v  # bit-identical, not approx
    assert on["spec_nrmse"] > 0.0 and on["base_self_spec_nrmse"] == 0.0


# --------------------------------------------------------------------------------------- #
# MODE-TRACK metrics — the replacements for spec_nrmse as the RANKING key
# --------------------------------------------------------------------------------------- #
def _ridge_window(seed: int = 0, F: int = 128, T: int = 64, rise: bool = True):
    """(1, 2, F, T) log-power-like array with a narrow ridge that MOVES in frequency."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (1, 2, F, T))
    f0 = (F // 3) + (np.arange(T) * (F // 6) // T if rise else np.zeros(T, dtype=int))
    for t in range(T):
        x[:, :, int(f0[t]), t] += 12.0
        x[:, :, int(f0[t]) + 1, t] += 6.0
    return x


def test_mode_track_metrics_are_perfect_on_self_and_low_on_the_time_mean():
    """The whole point: a static envelope must NOT score well. ``peak_f1`` gives it 1.0."""
    x = _ridge_window()
    tm = np.broadcast_to(x.mean(axis=-1, keepdims=True), x.shape).copy()

    s = gate.mode_structure_metrics(x, x, band_bins=None)
    assert s["mode_track_f1"] == pytest.approx(1.0)
    assert s["ridge_traj_corr"] == pytest.approx(1.0)
    assert s["spectral_contrast_ratio"] == pytest.approx(1.0)
    assert s["ms_ssim"] == pytest.approx(1.0, abs=1e-6)
    assert s["mode_track_n_gt_peaks"] > 1.0          # the target really does have peaks

    b = gate.mode_structure_metrics(tm, x, band_bins=None)
    assert b["mode_track_f1"] < 0.5, b["mode_track_f1"]
    assert abs(b["ridge_traj_corr"]) < 0.1, b["ridge_traj_corr"]
    assert b["spectral_contrast_ratio"] < 0.4, b["spectral_contrast_ratio"]
    assert b["ms_ssim"] < s["ms_ssim"]

    # ...whereas the OLD time-collapsed metric cannot tell them apart at all — this is the
    # documented defect that motivated the new metrics.
    assert gate.decode_fidelity(tm, x)["envelope_corr"] == pytest.approx(1.0, abs=1e-6)


def test_ridge_traj_corr_follows_a_moving_ridge_but_not_a_static_one():
    """A reconstruction with the ridge at a FIXED frequency must score ~0, not ~1."""
    x = _ridge_window(rise=True)
    static = _ridge_window(rise=False)
    assert gate.ridge_traj_corr(x, x, band_bins=None)["ridge_traj_corr"] == pytest.approx(1.0)
    assert abs(gate.ridge_traj_corr(static, x, band_bins=None)["ridge_traj_corr"]) < 0.2


def test_patch_lattice_inflates_hf_ratio_far_more_than_the_new_metrics():
    """MEASURED artifact sensitivity — the reason ``sharpness`` cannot be a quality signal.

    Adding a synthetic checkerboard to a BLURRED reconstruction inflates ``sharpness`` by
    12-19x (consistent with the measured 92-94 % of the mhr reconstruction's HF energy sitting
    on the patch lattice), while ``spectral_contrast_ratio`` moves 2-3x — about 6x less — and
    is EXACTLY unchanged by a purely TIME-periodic lattice. ``mode_track_f1`` and ``ms_ssim``
    are the artifact-SAFE members: a lattice makes both WORSE, because its spurious peaks are
    false positives. So ``spectral_contrast_ratio`` is ROBUST, not immune: a frequency-periodic
    lattice is a set of narrow frequency peaks, which it rewards by design, and its inflation
    grows as the target's own contrast falls. The invariant asserted here is the RATIO of
    sensitivities, which does not depend on the ridge geometry.
    """
    x = _ridge_window()
    tm = np.broadcast_to(x.mean(axis=-1, keepdims=True), x.shape).copy()
    F, T = x.shape[2], x.shape[3]
    fg, tg = np.meshgrid(np.arange(F), np.arange(T), indexing="ij")
    lat_f = tm + 3.0 * (fg % 16 == 0).astype(float)[None, None]
    lat_t = tm + 3.0 * (tg % 16 == 0).astype(float)[None, None]

    def _m(c):
        d = gate.decode_fidelity(c, x)
        s = gate.mode_structure_metrics(c, x, band_bins=None)
        return d["sharpness"], s["spectral_contrast_ratio"], s["mode_track_f1"], s["ms_ssim"]

    hf0, c0, f0, ss0 = _m(tm)
    hf1, c1, f1, ss1 = _m(lat_f)
    hf2, c2, f2, ss2 = _m(lat_t)

    # sharpness is inflated by an order of magnitude by EITHER lattice orientation
    assert hf1 / max(hf0, 1e-9) > 5.0
    assert hf2 / max(hf0, 1e-9) > 5.0
    # the contrast ratio is at least 3x LESS sensitive to the artifact than sharpness is,
    # and a TIME-periodic lattice does not move it at all
    assert (hf1 / max(hf0, 1e-9)) > 3.0 * (c1 / max(c0, 1e-9))
    assert c2 == pytest.approx(c0, rel=1e-6)
    # and the two artifact-safe metrics get WORSE, not better
    assert f1 < f0 and ss1 < ss0 and ss2 < ss0


def test_detrended_envelope_scores_exactly_zero_not_rounding_noise():
    """REGRESSION: the detrended time-mean envelope must score EXACTLY 0, not float32 noise.

    ``mode_track_f1(detrend=True)`` subtracts each pair's own time-average, so the envelope's
    detrended array is analytically zero and can contain no peaks. But the prominence test is
    RELATIVE to each frame's own std, and after the metrics were moved to float32 the residual
    is ~1.8e-07 rather than ~3.3e-16 — enough for the relative test to peak-pick pure rounding
    noise and report ~0.17. ``gate._PEAK_MIN_SD`` is the absolute floor that prevents it; real
    log-power z-scored frames have sd ~0.5-1.5, so the floor never fires on actual data.
    """
    x = _ridge_window(F=256, T=96)
    tm = np.broadcast_to(x.mean(axis=-1, keepdims=True), x.shape).copy()

    assert gate.mode_track_f1(tm, x, band_bins=None, detrend=True)["mode_track_f1"] == 0.0
    # ...and the floor must NOT suppress peaks on real-scale data
    assert gate.mode_track_f1(x, x, band_bins=None, detrend=True)["mode_track_f1"] == \
        pytest.approx(1.0)
    assert gate.mode_track_f1(x, x, band_bins=None)["mode_track_f1"] == pytest.approx(1.0)
    assert gate.mode_track_f1(x, x, band_bins=None)["mode_track_n_gt_peaks"] > 1.0


# ===================================================================================== #
# VIDEO full-array metrics (2026-09-03) — the tangtv analogue of full_slowts_metrics.
# The reference points below are the SAME "known ordering" discipline that rejected four
# 2-D spectrogram metrics on the same day: a metric that cannot order these is not allowed
# to order codecs.
# ===================================================================================== #
def _vid(seed=0, B=3, C=2, T=5, H=40, W=60):
    import numpy as np
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    base = np.sin(yy / 6.0) * np.cos(xx / 9.0)
    x = (base[None, None, None] * (1.0 + 0.3 * rng.standard_normal((B, C, T, 1, 1)))
         + 0.4 * rng.standard_normal((B, C, T, H, W)))
    return x


def test_full_video_metrics_self_and_constant_anchors():
    import numpy as np
    from tokamak_foundation_model.ignite import gate

    t = _vid()
    b = gate.trivial_video_baselines(t)
    assert abs(b["base_self_video_nrmse"]) < 1e-12
    assert abs(b["base_self_video_corr"] - 1.0) < 1e-12
    assert abs(b["base_self_video_std_ratio"] - 1.0) < 1e-12
    assert abs(b["base_self_video_hf_ratio"] - 1.0) < 1e-12
    # the per-(window, channel) constant mean is the EXACT 1.0 normalisation anchor
    assert abs(b["base_wcmean_video_nrmse"] - 1.0) < 1e-9
    assert abs(b["base_wcmean_video_std_ratio"]) < 1e-12
    # the per-pixel time mean is "perfect still image, ZERO temporal structure"
    assert abs(b["base_tmean_video_std_ratio_t"]) < 1e-12
    assert b["base_tmean_video_nrmse"] < b["base_wcmean_video_nrmse"]


def test_video_std_ratio_reports_amplitude_collapse_that_nrmse_rewards():
    import numpy as np
    from tokamak_foundation_model.ignite import gate

    t = _vid(1)
    mu = t.mean(axis=(2, 3, 4), keepdims=True)
    shrunk = mu + 0.40 * (t - mu)
    m = gate.full_video_metrics(shrunk, t)
    assert abs(m["video_std_ratio"] - 0.40) < 1e-6         # the collapse is a NUMBER
    assert m["video_nrmse"] < 1.0                          # ... which nRMSE alone accepts
    assert abs(m["video_corr"] - 1.0) < 1e-9               # ... and corr cannot see at all


def test_full_video_metrics_masking_ignores_dead_cameras():
    import numpy as np
    from tokamak_foundation_model.ignite import gate

    t = _vid(2)
    # camera 1 stops recording PART-WAY through the window (frames 3-4 zero-filled), which is
    # the case the "constant target" guard cannot catch on its own: the group still has
    # variance, so an unmasked metric happily scores the zero-fill.
    t_dead = t.copy()
    t_dead[:, 1, 3:] = 0.0
    mask = np.ones(t.shape[:3])
    mask[:, 1, 3:] = 0.0
    recon = t.copy()
    recon[:, 1, 3:] = 50.0                                 # garbage where nothing was recorded
    m = gate.full_video_metrics(recon, t_dead, mask=mask)
    assert abs(m["video_nrmse"]) < 1e-9                    # the garbage is excluded
    assert abs(m["video_present_frac"] - 0.8) < 1e-9       # 2 of 10 (channel, frame) slots dead
    # unmasked, the SAME reconstruction is catastrophic
    assert gate.full_video_metrics(recon, t_dead)["video_nrmse"] > 10.0
    # a fully dead camera (constant target) is dropped by the valid-group guard either way
    t2, r2 = t.copy(), t.copy()
    t2[:, 1] = 0.0
    r2[:, 1] = 12345.0
    m2 = np.ones(t.shape[:3]); m2[:, 1] = 0.0
    assert abs(gate.full_video_metrics(r2, t2, mask=m2)["video_nrmse"]) < 1e-9
    assert abs(gate.full_video_metrics(r2, t2)["video_valid_frac"] - 0.5) < 1e-9


def test_video_patch_lattice_detects_an_injected_checkerboard():
    import numpy as np
    from tokamak_foundation_model.ignite import gate

    t = _vid(3, H=120, W=180)                # 6 x 9 patches, like the real 6 x 18 grid
    clean = gate.video_patch_lattice(t, 20, 20)["patch_lattice_ratio"]
    rng = np.random.default_rng(5)
    tile = rng.standard_normal((20, 20))
    grid = np.tile(tile, (t.shape[-2] // 20, t.shape[-1] // 20))
    dirty = gate.video_patch_lattice(t + 1.0 * t.std() * grid, 20, 20)
    assert clean < 2.0                       # smooth frames: ~1 = no lattice
    assert dirty["patch_lattice_ratio"] > 5.0 * clean         # a tiled texture is caught
    assert dirty["lattice_energy_frac"] > 10 * gate.video_patch_lattice(
        t, 20, 20)["lattice_energy_frac"]


def test_code_rate_bits_matches_hand_computable_cases():
    import numpy as np
    from tokamak_foundation_model.ignite import gate

    # every token always the same code -> ZERO delivered bits at every position
    dead = np.zeros((50, 4, 3), dtype=np.int64)
    r = gate.code_rate_bits(dead, 1000, n_tok=4)
    assert abs(r["rate_positional"]) < 1e-12
    assert abs(r["rate_pooled"]) < 1e-12
    assert r["n_distinct_codes"] == 1

    # each POSITION emits its own fixed distinct code: pooled entropy log2(4) but the honest
    # positional rate is still ZERO -- the exact confusion the pooled number would hide.
    fixed = np.zeros((50, 4, 1), dtype=np.int64)
    fixed[:, :, 0] = np.arange(4)[None, :]
    r = gate.code_rate_bits(fixed, 1000, n_tok=4)
    assert abs(r["bits_pooled_per_token"] - 2.0) < 1e-9
    assert abs(r["bits_positional_per_token"]) < 1e-12
    assert r["bits_available_per_frame"] == 4 * np.log2(1000)
