"""Tests for best-by-gate checkpoint selection (+ optional EMA) in the IGNITE spike.

All CPU, all synthetic — no real HDF5, no SLURM, no GPU, no FAITH model code.

Covers (per the task spec):
  * the composite gate-score formula + the HARD collapse gate (-inf);
  * ``run_spike`` tracks the best score, calls ``on_best`` only on strict improvement,
    and does NOT regress to a worse eval;
  * a collapsed eval (min_dim_entropy below the floor / ``collapsed=True``) never becomes
    best;
  * ``--ema`` maintains a shadow that updates toward the live weights and writes
    ``codec_ema.pt``; the CLI writes ``codec_best.pt`` with the expected payload.

Run:
    .pixi/envs/default/bin/python -m pytest tests/ignite/test_spike_best_ema.py -q
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tokamak_foundation_model.ignite import spike
from tokamak_foundation_model.ignite.config import SpectroCodecConfig


# --------------------------------------------------------------------------------------
# gate_score: reconstruction+forecastability+utilization score; recon-floor disqualifier
# --------------------------------------------------------------------------------------
def _score_formula(env_corr, peak_f1, margin_tr, min_dim_entropy, frac_obs):
    """Reference re-implementation of spike.gate_score's finite branch (for exact asserts).

    Mirrors the documented weighted sum: recon(1.0) + f1(1.0) + fcast(1.0) + util(0.5),
    with MARGIN_SCALE=0.1 and util = 0.5*min_dim_entropy + 0.5*frac_of_observable.
    """
    c01 = lambda v: max(0.0, min(1.0, v))
    recon = c01(env_corr)
    f1 = c01(peak_f1)
    fcast = c01(margin_tr / 0.1)
    util = 0.5 * c01(min_dim_entropy) + 0.5 * c01(frac_obs)
    return 1.0 * recon + 1.0 * f1 + 1.0 * fcast + 0.5 * util


def _gate(
    margin_tr,
    min_dim_entropy,
    peak_f1,
    collapsed,
    *,
    envelope_corr=0.84,   # ece/bes-style: reconstructs well by default -> above recon floor
    frac_of_observable=0.5,
):
    """Minimal gate dict carrying what gate_score reads.

    ``envelope_corr`` defaults to a healthy 0.84 (like ece/bes) so a codec is NOT
    disqualified merely by the dim-entropy ``collapsed`` flag; set it to ``nan`` (or below
    ``gate_recon_floor``) to model a genuine reconstruction failure (like co2's dead codec).
    """
    return {
        "forecastability": {"margin_transition": margin_tr},
        "utilization": {
            "min_dim_entropy": min_dim_entropy,
            "collapsed": collapsed,
            "frac_of_observable": frac_of_observable,
        },
        "decode": {"peak_f1": peak_f1, "envelope_corr": envelope_corr},
    }


def test_gate_score_is_documented_weighted_sum():
    """Above the recon floor, the score is the documented weighted sum of clamped metrics."""
    g = _gate(
        margin_tr=0.10, min_dim_entropy=0.50, peak_f1=0.30, collapsed=False,
        envelope_corr=0.84, frac_of_observable=0.5,
    )
    expect = _score_formula(0.84, 0.30, 0.10, 0.50, 0.5)
    assert spike.gate_score(g) == pytest.approx(expect)


def test_gate_score_recon_fail_nan_is_neg_inf():
    """co2-style dead codec: envelope_corr NaN => disqualified (-inf) no matter what else."""
    g = _gate(margin_tr=5.0, min_dim_entropy=1.0, peak_f1=1.0, collapsed=True,
              envelope_corr=float("nan"))
    assert spike.gate_score(g) == float("-inf")


def test_gate_score_recon_below_floor_is_neg_inf():
    """envelope_corr below the recon floor => disqualified (-inf)."""
    g = _gate(margin_tr=0.5, min_dim_entropy=1.0, peak_f1=1.0, collapsed=False,
              envelope_corr=0.05)
    assert spike.gate_score(g, recon_floor=0.2) == float("-inf")


def test_gate_score_dead_dim_but_reconstructs_is_finite_positive():
    """ece/bes-style: reconstructs well (env_corr 0.84) but one FSQ dim is dead
    (min_dim_entropy=0, collapsed=True). Must yield a FINITE POSITIVE score, NOT -inf."""
    g = _gate(margin_tr=0.05, min_dim_entropy=0.0, peak_f1=0.757, collapsed=True,
              envelope_corr=0.84, frac_of_observable=0.4)
    s = spike.gate_score(g)
    assert math.isfinite(s)
    assert s > 0.0


def test_gate_score_prefers_better_utilized_among_reconstructing():
    """Among two codecs that BOTH reconstruct well and forecast identically, the
    better-utilized one (higher min_dim_entropy / frac_of_observable) must score higher."""
    base = dict(margin_tr=0.05, peak_f1=0.5, collapsed=False, envelope_corr=0.84)
    worse_util = _gate(min_dim_entropy=0.0, frac_of_observable=0.1, **base)
    better_util = _gate(min_dim_entropy=0.6, frac_of_observable=0.6, **base)
    assert spike.gate_score(better_util) > spike.gate_score(worse_util)


def test_gate_score_orders_evals():
    lo = _gate(0.01, 0.20, 0.10, collapsed=False)
    hi = _gate(0.02, 0.60, 0.40, collapsed=False)
    assert spike.gate_score(hi) > spike.gate_score(lo)


# --------------------------------------------------------------------------------------
# _EMA: shadow exists, updates toward live weights
# --------------------------------------------------------------------------------------
def test_ema_shadow_tracks_live_weights():
    torch.manual_seed(0)
    m = torch.nn.Linear(4, 3)
    ema = spike._EMA(m, decay=0.9)

    # shadow starts equal to the live params
    for name, p in m.state_dict().items():
        assert torch.allclose(ema.shadow[name], p)

    before = {k: v.clone() for k, v in ema.shadow.items()}

    # move the live weights, then update the EMA once
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    ema.update(m)

    live = m.state_dict()
    for name in before:
        s_new = ema.shadow[name]
        if not live[name].is_floating_point():
            continue
        # shadow moved toward live (which increased) but did NOT jump all the way
        assert torch.all(s_new >= before[name] - 1e-6)
        assert torch.all(s_new < live[name] - 1e-6)
        # exact EMA recurrence: s = d*before + (1-d)*live
        expect = 0.9 * before[name] + 0.1 * live[name]
        assert torch.allclose(s_new, expect, atol=1e-6)


def test_ema_converges_toward_live_over_many_steps():
    torch.manual_seed(1)
    m = torch.nn.Linear(3, 2)
    ema = spike._EMA(m, decay=0.5)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.full_like(p, 10.0))
    for _ in range(30):
        ema.update(m)
    for name, p in m.state_dict().items():
        if p.is_floating_point():
            assert torch.allclose(ema.shadow[name], p, atol=1e-3)


def test_ema_state_dict_is_loadable_shape():
    m = torch.nn.Linear(4, 3)
    ema = spike._EMA(m, decay=0.99)
    sd = ema.state_dict()
    # detached copy, same keys/shapes as the module state
    assert set(sd.keys()) == set(m.state_dict().keys())
    m.load_state_dict(sd)  # must be a drop-in weight set


# --------------------------------------------------------------------------------------
# run_spike best-tracking: monkeypatch compute_gate to script the eval trajectory
# --------------------------------------------------------------------------------------
def _tiny_cfg():
    return SpectroCodecConfig(
        channels=2,
        freq_bins=64,
        time_frames=32,
        patch_f=32,
        patch_t=16,
        d_model=16,
        enc_depth=1,
        dec_depth=1,
        heads=2,
        fsq_levels=[4, 4, 3],
    )


def _tiny_batches(cfg, n=2, bs=2):
    return spike.synthetic_batches(cfg, n_batches=n, batch_size=bs, device="cpu", seed=0)


def _scripted_gate(monkeypatch, scores_and_collapse):
    """Patch spike.compute_gate to emit a scripted sequence of gate dicts.

    ``scores_and_collapse`` is a list of (margin_tr, min_dim_entropy, peak_f1, collapsed)
    OR (margin_tr, min_dim_entropy, peak_f1, collapsed, envelope_corr); each successive eval
    pops the next entry (last is reused if exhausted). ``envelope_corr`` defaults to 0.5 (a
    healthy, reconstructing codec above the recon floor); pass ``float("nan")`` or a value
    below the recon floor to model a genuine reconstruction failure (disqualified, -inf).
    """
    seq = list(scores_and_collapse)
    call = {"i": 0}

    def fake_compute_gate(codec, eval_pairs, frame_seq, cfg):
        idx = min(call["i"], len(seq) - 1)
        call["i"] += 1
        entry = seq[idx]
        m, h, f1, col = entry[:4]
        env_corr = entry[4] if len(entry) > 4 else 0.5
        g = _gate(m, h, f1, col)
        # fill the rest of the keys compute_gate normally returns (log_fn reads them)
        g["stability"] = 0.5
        g["persistence"] = 0.5
        g["forecastability"].update(
            {"margin_stable": 0.0, "margin_overall": 0.0, "beats_persistence": m > 0}
        )
        g["decode"].update({"envelope_corr": env_corr, "sharpness": 1.0})
        g["utilization"].update({"frac_codes_used": 0.5, "frac_of_observable": 0.5})
        g["pass_stability"] = True
        g["pass_persistence"] = True
        g["pass_utilization"] = not col
        return g

    monkeypatch.setattr(spike, "compute_gate", fake_compute_gate)


def test_best_improves_then_holds_against_worse(monkeypatch):
    cfg = _tiny_cfg()
    # eval trajectory (eval_every=1), all reconstructing (env_corr 0.5 via _scripted_gate);
    # only min_dim_entropy + peak_f1 vary, so scores go LOW -> HIGH (best) -> MID (worse).
    # Scripted (margin_tr, min_dim_entropy, peak_f1, collapsed):
    #   (0.0, 0.1, 0.2)  -> peak_f1 0.2, low util   (lowest)
    #   (0.0, 0.4, 0.5)  -> peak_f1 0.5, high util  (best)
    #   (0.0, 0.1, 0.3)  -> peak_f1 0.3, low util   (worse than best)
    _scripted_gate(monkeypatch, [
        (0.0, 0.1, 0.2, False),
        (0.0, 0.4, 0.5, False),
        (0.0, 0.1, 0.3, False),
    ])
    # expected finite scores (env_corr=0.5, frac_of_observable=0.5 from _scripted_gate)
    s0 = _score_formula(0.5, 0.2, 0.0, 0.1, 0.5)
    s1 = _score_formula(0.5, 0.5, 0.0, 0.4, 0.5)
    s2 = _score_formula(0.5, 0.3, 0.0, 0.1, 0.5)
    assert s1 > s0 and s1 > s2  # eval-1 is strictly best
    best_calls = []

    def on_best(step, codec, disc, g):
        best_calls.append((step, g["score"]))

    out = spike.run_spike(
        cfg, _tiny_batches(cfg), steps=3, device="cpu", eval_every=1,
        log_fn=None, on_best=on_best, seed=0,
    )

    # on_best fired for eval-0 (first ever) and eval-1 (improvement), NOT eval-2 (worse)
    assert [c[0] for c in best_calls] == [0, 1]
    assert math.isclose(best_calls[0][1], s0, abs_tol=1e-9)
    assert math.isclose(best_calls[1][1], s1, abs_tol=1e-9)
    assert math.isclose(out["best_score"], s1, abs_tol=1e-9)
    assert out["best_step"] == 1


def test_recon_failed_eval_never_becomes_best(monkeypatch):
    cfg = _tiny_cfg()
    # A recon-FAILED eval (envelope_corr NaN) with sky-high raw terms must never win — only
    # genuine reconstruction failure disqualifies now, NOT the dim-entropy collapse flag.
    _scripted_gate(monkeypatch, [
        (0.0, 0.10, 0.20, False, 0.5),          # reconstructs (0.5) -> the only legit best
        (5.0, 1.00, 1.00, True, float("nan")),  # recon FAILED -> -inf, must NOT overwrite
    ])
    s0 = _score_formula(0.5, 0.20, 0.0, 0.10, 0.5)
    best_calls = []
    spike.run_spike(
        cfg, _tiny_batches(cfg), steps=2, device="cpu", eval_every=1,
        log_fn=None, on_best=lambda s, c, d, g: best_calls.append((s, g["score"])),
        seed=0,
    )
    assert [s for s, _ in best_calls] == [0]  # only the reconstructing eval won
    assert math.isclose(best_calls[0][1], s0, abs_tol=1e-9)


def test_dead_dim_but_reconstructs_still_becomes_best(monkeypatch):
    cfg = _tiny_cfg()
    # ece/bes-style: collapsed=True (one dead FSQ dim) BUT reconstructs well (env_corr 0.84).
    # This MUST be selectable as best (regression against the old hard collapse gate).
    _scripted_gate(monkeypatch, [
        (0.05, 0.0, 0.757, True, 0.84),   # dead dim, but reconstructs -> finite positive
    ])
    best_calls = []
    out = spike.run_spike(
        cfg, _tiny_batches(cfg), steps=1, device="cpu", eval_every=1,
        log_fn=None, on_best=lambda s, c, d, g: best_calls.append((s, g["score"])),
        seed=0,
    )
    assert len(best_calls) == 1                     # it DID become best
    assert math.isfinite(best_calls[0][1])
    assert best_calls[0][1] > 0.0
    assert out["best_step"] == 0


def test_all_recon_failed_yields_no_best(monkeypatch):
    cfg = _tiny_cfg()
    _scripted_gate(monkeypatch, [
        (5.0, 1.0, 1.0, True, float("nan")),
        (5.0, 1.0, 1.0, True, float("nan")),
    ])
    best_calls = []
    out = spike.run_spike(
        cfg, _tiny_batches(cfg), steps=2, device="cpu", eval_every=1,
        log_fn=None, on_best=lambda s, c, d, g: best_calls.append(s), seed=0,
    )
    assert best_calls == []
    assert out["best_step"] is None
    assert out["best_score"] == float("-inf")


def test_run_spike_ema_shadow_updates_and_reaches_on_ema(monkeypatch):
    cfg = _tiny_cfg()
    _scripted_gate(monkeypatch, [(0.0, 0.5, 0.5, False)])
    ema_calls = []

    def on_ema(step, ema_state, g):
        ema_calls.append((step, ema_state))

    spike.run_spike(
        cfg, _tiny_batches(cfg), steps=2, device="cpu", eval_every=1,
        log_fn=None, ema=True, ema_decay=0.5, on_ema=on_ema, seed=0,
    )
    assert ema_calls, "on_ema was never called with --ema"
    # ema_state is a real weight dict (has codec params)
    _, sd = ema_calls[-1]
    assert isinstance(sd, dict) and len(sd) > 0
    assert all(isinstance(v, torch.Tensor) for v in sd.values())


def test_no_ema_by_default(monkeypatch):
    cfg = _tiny_cfg()
    _scripted_gate(monkeypatch, [(0.0, 0.5, 0.5, False)])
    ema_calls = []
    spike.run_spike(
        cfg, _tiny_batches(cfg), steps=1, device="cpu", eval_every=1,
        log_fn=None, ema=False, on_ema=lambda s, e, g: ema_calls.append(s), seed=0,
    )
    assert ema_calls == []  # on_ema not invoked when ema=False


# --------------------------------------------------------------------------------------
# CLI end-to-end: codec_best.pt payload + codec_ema.pt written under --ema
# --------------------------------------------------------------------------------------
def _tiny_cfg_patch(monkeypatch):
    from tokamak_foundation_model.ignite import config as cfg_mod

    orig = cfg_mod.SpectroCodecConfig

    def _small(**kw):
        kw.setdefault("channels", 4)
        kw["freq_bins"] = 64
        kw["time_frames"] = 32
        kw["patch_f"] = 32
        kw["patch_t"] = 16
        kw["d_model"] = 32
        kw["enc_depth"] = 1
        kw["dec_depth"] = 1
        kw["heads"] = 2
        kw["fsq_levels"] = [4, 4, 3]
        return orig(**kw)

    monkeypatch.setattr(spike, "SpectroCodecConfig", _small)


def _cli_args(out_dir: Path, steps: int, extra=None):
    args = [
        "--synthetic",
        "--n_batches", "3",
        "--eval_n_batches", "2",
        "--batch_size", "2",
        "--steps", str(steps),
        "--eval_every", "1",
        "--n_frames", "3",
        "--out_dir", str(out_dir),
        "--device", "cpu",
        "--seed", "0",
    ]
    if extra:
        args += extra
    return args


def test_cli_writes_best_checkpoint_and_summary(tmp_path, monkeypatch):
    _tiny_cfg_patch(monkeypatch)
    # Script a reconstructing improving trajectory (env_corr 0.5 via _scripted_gate) so the
    # best-ckpt PERSISTENCE path is exercised end-to-end through the CLI. Eval-1 has the
    # highest peak_f1 + utilization, so it is the composite best.
    _scripted_gate(monkeypatch, [
        (0.0, 0.10, 0.20, False),   # lowest
        (0.0, 0.40, 0.50, False),   # <- best
        (0.0, 0.10, 0.30, False),   # worse than best
    ])
    best_score = _score_formula(0.5, 0.50, 0.0, 0.40, 0.5)  # eval-1
    out_dir = tmp_path / "run"
    gate = spike.main(_cli_args(out_dir, steps=3))

    best = out_dir / "codec_best.pt"
    assert best.exists(), "codec_best.pt not written"

    ck = torch.load(best, map_location="cpu", weights_only=False)
    for key in ("codec", "disc", "step", "score", "gate"):
        assert key in ck, f"missing '{key}' in codec_best.pt"
    assert isinstance(ck["score"], float)
    # rolling last ckpt still written (unchanged behavior)
    assert (out_dir / "codec_last.pt").exists()

    # score payload matches the scripted best @ eval-1
    assert math.isclose(ck["score"], best_score, abs_tol=1e-9)

    # summary carries best score/step
    s = json.loads((out_dir / "summary.json").read_text())
    assert "best_score" in s and "best_step" in s
    assert math.isclose(gate["best_score"], best_score, abs_tol=1e-9)
    assert gate["best_step"] == 1


def test_cli_ema_writes_ema_checkpoint(tmp_path, monkeypatch):
    _tiny_cfg_patch(monkeypatch)
    out_dir = tmp_path / "run_ema"
    spike.main(_cli_args(out_dir, steps=3, extra=["--ema", "--ema_decay", "0.5"]))

    ema_ckpt = out_dir / "codec_ema.pt"
    assert ema_ckpt.exists(), "codec_ema.pt not written under --ema"
    ck = torch.load(ema_ckpt, map_location="cpu", weights_only=False)
    assert "codec" in ck and isinstance(ck["codec"], dict) and len(ck["codec"]) > 0
    assert "step" in ck

    s = json.loads((out_dir / "summary.json").read_text())
    assert s["config"]["ema"] is True
    assert s["config"]["ema_decay"] == 0.5


def test_cli_no_ema_checkpoint_by_default(tmp_path, monkeypatch):
    _tiny_cfg_patch(monkeypatch)
    out_dir = tmp_path / "run_noema"
    spike.main(_cli_args(out_dir, steps=2))
    assert not (out_dir / "codec_ema.pt").exists()
