"""Phase-B MaskGIT: masked-token training loss, valid iterative decode, commit-code
rollout."""

from pathlib import Path

import pytest
import torch

from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig, ModalitySpec
from tokamak_foundation_model.ignite.frame_layout import code_to_coords, coords_to_code
from tokamak_foundation_model.ignite.maskgit import (
    MaskGITDynamics,
    _cosine_keep_fractions,
    _latent_nll,
    _latent_sample,
)
from tokamak_foundation_model.ignite.train_dynamics import resolve_mod_weights


def _tiny(depth=2, decode=4):
    return DynamicsConfig(
        modalities=(
            ModalitySpec("a", "spectro", 3, 5),
            ModalitySpec("b", "slowts", 2, 4),
        ),
        d_model=16,
        depth=depth,
        n_heads=2,
        ffn_mult=2,
        k0_seed=2,
        n_predict=3,
        maskgit_decode_steps=decode,
        actuator_dim=6,
    )


def _codes(cfg, B, F, gen=None):
    return {
        m.name: torch.randint(0, m.codebook_size, (B, F, m.n_tok), generator=gen)
        for m in cfg.modalities
    }


def test_cosine_schedule_reveals_everything():
    fr = _cosine_keep_fractions(6)
    assert fr[0] < 1.0 and abs(fr[-1]) < 1e-6  # ends fully revealed
    assert all(a >= b - 1e-9 for a, b in zip(fr, fr[1:]))  # monotone non-increasing


def test_random_mask_uses_mask_id_and_at_least_one_per_frame():
    cfg = _tiny()
    mg = MaskGITDynamics(cfg)
    codes = _codes(cfg, B=2, F=4)
    masked, mask = mg._random_mask(codes, gen=None)
    for m in cfg.modalities:
        mid = m.codebook_size
        assert (masked[m.name][mask[m.name]] == mid).all()  # masked -> MASK id
        assert (
            masked[m.name][~mask[m.name]] == codes[m.name][~mask[m.name]]
        ).all()  # rest kept
        assert mask[m.name].any(dim=-1).all()  # >=1 masked per frame


def test_training_loss_finite_and_grads_flow():
    cfg = _tiny()
    mg = MaskGITDynamics(cfg).train()
    codes = _codes(cfg, B=2, F=5)
    act = torch.randn(2, 5, cfg.actuator_dim)
    loss = mg.training_loss(codes, act, generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    g = mg.backbone.blocks[0].spatial.qkv.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_generate_frame_commits_valid_codes():
    cfg = _tiny(decode=4)
    mg = MaskGITDynamics(cfg).eval()
    past = _codes(cfg, B=2, F=cfg.k0_seed)
    act = torch.randn(2, cfg.k0_seed + 1, cfg.actuator_dim)
    gen = torch.Generator().manual_seed(1)
    nxt = mg.generate_frame(past, act, generator=gen)
    for m in cfg.modalities:
        assert nxt[m.name].shape == (2, m.n_tok)
        assert (nxt[m.name] >= 0).all() and (
            nxt[m.name] < m.codebook_size
        ).all()  # NEVER the MASK id


def test_rollout_shapes_seed_preserved_and_valid():
    cfg = _tiny()
    mg = MaskGITDynamics(cfg).eval()
    seed = _codes(cfg, B=2, F=cfg.k0_seed)
    act = torch.randn(2, cfg.k0_seed + cfg.n_predict, cfg.actuator_dim)
    traj = mg.rollout(seed, act, generator=torch.Generator().manual_seed(2))
    for m in cfg.modalities:
        assert traj[m.name].shape == (2, cfg.k0_seed + cfg.n_predict, m.n_tok)
        # seed frames untouched
        assert torch.equal(traj[m.name][:, : cfg.k0_seed], seed[m.name])
        # generated frames are valid codes
        gen_part = traj[m.name][:, cfg.k0_seed :]
        assert (gen_part >= 0).all() and (gen_part < m.codebook_size).all()


def test_rollout_is_deterministic_with_generator():
    cfg = _tiny()
    mg = MaskGITDynamics(cfg).eval()
    seed = _codes(cfg, B=1, F=cfg.k0_seed)
    act = torch.randn(1, cfg.k0_seed + cfg.n_predict, cfg.actuator_dim)
    t1 = mg.rollout(seed, act, generator=torch.Generator().manual_seed(7))
    t2 = mg.rollout(seed, act, generator=torch.Generator().manual_seed(7))
    for m in cfg.modalities:
        assert torch.equal(t1[m.name], t2[m.name])


def test_ss_fraction_ramps_from_zero_to_final():
    mg = MaskGITDynamics(_tiny())
    assert mg.ss_fraction(0) == 0.0
    assert mg.ss_fraction(mg.cfg.ss_ramp_steps) == pytest.approx(
        mg.cfg.ss_ramp_final_frac
    )
    mid = mg.ss_fraction(mg.cfg.ss_ramp_steps // 2)
    assert 0.0 < mid < mg.cfg.ss_ramp_final_frac


def test_training_loss_with_scheduled_sampling_finite_and_trains():
    cfg = _tiny()
    mg = MaskGITDynamics(cfg).train()
    codes = _codes(cfg, B=2, F=5)
    act = torch.randn(2, 5, cfg.actuator_dim)
    loss = mg.training_loss(
        codes, act, generator=torch.Generator().manual_seed(3), ss_frac=0.5
    )
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert mg.backbone.blocks[0].spatial.qkv.weight.grad is not None


def test_rollout_future_actuator_does_not_change_earlier_frames():
    # committing frame t must not depend on actuator_{>t+1} (causal control)
    cfg = _tiny()
    mg = MaskGITDynamics(cfg).eval()
    seed = _codes(cfg, B=1, F=cfg.k0_seed)
    act = torch.randn(1, cfg.k0_seed + cfg.n_predict, cfg.actuator_dim)
    g = 11
    base = mg.rollout(seed, act, generator=torch.Generator().manual_seed(g))
    act2 = act.clone()
    act2[:, -1, :] += 5.0  # perturb the LAST actuator
    pert = mg.rollout(seed, act2, generator=torch.Generator().manual_seed(g))
    # the FIRST generated frame (t=K0) cannot see the last actuator
    for m in cfg.modalities:
        assert torch.equal(base[m.name][:, cfg.k0_seed], pert[m.name][:, cfg.k0_seed])


# ------------------------------------------------------------------ per-modality
# readout --- The equal-weight aggregate CANNOT judge a reweighting arm: reweighting
# moves the aggregate by construction. Before 2026-09-10 `_val_loss`/`_gen_val_loss`
# also passed the TRAINING weights into the metric, so a weighted arm was ranked against
# unweighted arms on a different quantity -- which is why `mod_weight_pow 0.5` read as
# "no gain". These pin both halves of the fix: per-modality terms are reported, and they
# are independent of the weighting.


def _fixed(cfg, B=2, F=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return _codes(cfg, B, F, gen=g), torch.randn(B, F, cfg.actuator_dim, generator=g)


def test_per_modality_is_populated_and_matches_equal_weight_aggregate():
    cfg = _tiny()
    mg = MaskGITDynamics(cfg).eval()
    codes, act = _fixed(cfg)
    pm = {}
    with torch.no_grad():
        agg = mg.training_loss(
            codes, act, generator=torch.Generator().manual_seed(7), per_modality=pm
        )
    assert set(pm) == {m.name for m in cfg.modalities}
    assert all(v > 0 and torch.isfinite(torch.tensor(v)) for v in pm.values())
    # with no weights the aggregate IS the plain mean of the per-modality terms
    assert abs(float(agg) - sum(pm.values()) / len(pm)) < 1e-5


def test_mod_weights_move_aggregate_but_leave_per_modality_terms_untouched():
    cfg = _tiny()
    mg = MaskGITDynamics(cfg).eval()
    codes, act = _fixed(cfg)
    w = {"a": 0.25, "b": 1.75}  # mean 1, as resolve_mod_weights guarantees
    pm_u, pm_w = {}, {}
    with torch.no_grad():
        agg_u = float(
            mg.training_loss(
                codes,
                act,
                generator=torch.Generator().manual_seed(7),
                per_modality=pm_u,
            )
        )
        agg_w = float(
            mg.training_loss(
                codes,
                act,
                generator=torch.Generator().manual_seed(7),
                per_modality=pm_w,
                mod_weights=w,
            )
        )
    # the per-modality CE is a property of the model, NOT of the loss weighting
    for n in pm_u:
        assert abs(pm_u[n] - pm_w[n]) < 1e-6, (
            f"{n} per-modality term moved with the weights"
        )
    # the aggregate is the weighted mean -> so it is NOT comparable across weightings,
    # and the per-modality dict makes the weighted view reconstructible offline
    assert abs(agg_w - sum(w[n] * pm_u[n] for n in pm_u) / sum(w.values())) < 1e-5
    assert abs(agg_u - agg_w) > 1e-4  # they really do differ


# -------------------------------------------------------------- resolve_mod_weights ---


NAMES = ["a", "b", "c", "d"]


def test_resolve_normalises_to_mean_one_and_preserves_ratios():
    w = resolve_mod_weights('{"a": 4, "b": 2, "c": 1, "d": 1}', NAMES)
    assert abs(sum(w.values()) / len(NAMES) - 1.0) < 1e-9  # mean 1 keeps LR comparable
    assert abs(w["a"] / w["c"] - 4.0) < 1e-9  # ratios survive normalisation


def test_resolve_defaults_missing_modalities_to_one():
    w = resolve_mod_weights('{"a": 3}', NAMES)
    assert abs(w["b"] - w["c"]) < 1e-9 and abs(w["c"] - w["d"]) < 1e-9
    assert w["a"] > w["b"]


def test_resolve_ignores_underscore_metadata_keys():
    w = resolve_mod_weights('{"_provenance": {"measured_on": "x"}, "a": 2}', NAMES)
    assert set(w) == set(NAMES) and w["a"] > w["b"]


def test_resolve_rejects_unknown_modality_rather_than_silently_ignoring_it():
    # a typo'd name must NOT weight nothing and pass -- the failure mode that killed 7
    # jobs
    with pytest.raises(ValueError, match="unknown modalities"):
        resolve_mod_weights('{"mirnov_typo": 2}', NAMES)


@pytest.mark.parametrize(
    "spec,match",
    [
        ('{"a": -1}', "non-negative"),
        ('{"a": "0.5"}', "non-negative"),
        ('{"a": 0, "b": 0, "c": 0, "d": 0}', "sum to zero"),
        ('{"_only": 1}', "only metadata"),
        ("{}", "non-empty"),
        ("[1, 2]", "non-empty"),
    ],
)
def test_resolve_rejects_malformed_specs(spec, match):
    with pytest.raises(ValueError, match=match):
        resolve_mod_weights(spec, NAMES)


def test_resolve_missing_file_raises_rather_than_defaulting_to_equal_weights():
    with pytest.raises(FileNotFoundError):
        resolve_mod_weights("/nonexistent/weights.json", NAMES)


def test_shipped_predictability_weight_files_load_against_the_real_modality_set():
    from tokamak_foundation_model.ignite.dynamics_config import FROZEN_MODALITIES

    names = [m.name for m in FROZEN_MODALITIES]
    root = Path(__file__).resolve().parents[2] / "analysis" / "mod_weights"
    files = sorted(root.glob("predictability_a*.json"))
    assert files, f"no shipped weight files under {root}"
    for f in files:
        w = resolve_mod_weights(str(f), names)
        assert set(w) == set(names)
        assert abs(sum(w.values()) / len(names) - 1.0) < 1e-6
        # the measured ordering must survive: mirnov is the least predictable of the
        # fifteen
        if "mirnov" in w and "cer_ti" in w:
            assert w["mirnov"] < w["cer_ti"], f"{f.name} lost the measured ordering"


def test_per_modality_collection_does_not_change_the_loss_value():
    """REGRESSION GUARD. `_val_loss` now always requests a per-modality dict; if that
    request perturbed the returned CE, every historical val/gen number would silently
    shift and the whole record table (rs2 1.4223, nfulllr3 1.4430, ...) would stop being
    comparable."""
    cfg = _tiny()
    mg = MaskGITDynamics(cfg).eval()
    codes, act = _fixed(cfg, seed=3)
    with torch.no_grad():
        a = float(
            mg.training_loss(codes, act, generator=torch.Generator().manual_seed(11))
        )
        b = float(
            mg.training_loss(
                codes, act, generator=torch.Generator().manual_seed(11), per_modality={}
            )
        )
        # and mod_weights=None must equal passing nothing at all
        c = float(
            mg.training_loss(
                codes,
                act,
                generator=torch.Generator().manual_seed(11),
                per_modality={},
                mod_weights=None,
            )
        )
    assert a == b == c, f"loss moved: {a} vs {b} vs {c}"


# ------------------------------------------------------------------- post-shot frame
# masking --- The cache holds a fixed 219 frames/shot from the ACQUISITION grid while
# the median discharge ends at 5.47 s, so a large minority of scored frames are dead
# time. Measured 2026-09-10: dead windows are markedly cheaper to predict and inflate
# the forecast metric by ~0.24 nats. `live` zero-weights them. These pin the three
# properties that make that safe.


def test_live_none_takes_the_unweighted_path_exactly():
    """REGRESSION GUARD. Arms WITHOUT --live_end_json pass live=None, which must hit the
    plain `F.cross_entropy` branch and be EXACTLY identical to the pre-change code. (An
    all-ones mask is mathematically equivalent but routes through the weighted mean, so
    it differs by ~2e-7 of float32 rounding -- see the next test. Only live=None is the
    exactness guarantee.)"""
    cfg = _tiny()
    mg = MaskGITDynamics(cfg).eval()
    codes, act = _fixed(cfg, seed=5)
    with torch.no_grad():
        a = float(
            mg.training_loss(codes, act, generator=torch.Generator().manual_seed(3))
        )
        b = float(
            mg.training_loss(
                codes,
                act,
                generator=torch.Generator().manual_seed(3),
                live=None,
                present=None,
            )
        )
    assert a == b


def test_all_live_mask_matches_the_unweighted_loss():
    cfg = _tiny()
    mg = MaskGITDynamics(cfg).eval()
    codes, act = _fixed(cfg, seed=5)
    ones = torch.ones(codes[cfg.modalities[0].name].shape[:2])  # (B, F)
    with torch.no_grad():
        a = float(
            mg.training_loss(codes, act, generator=torch.Generator().manual_seed(3))
        )
        b = float(
            mg.training_loss(
                codes, act, generator=torch.Generator().manual_seed(3), live=ones
            )
        )
    assert abs(a - b) < 1e-5, f"all-live mask changed the loss: {a} vs {b}"


def test_dead_frames_are_excluded_from_the_loss():
    """Zeroing the LAST frames must make their CONTENT irrelevant: the CE is a weighted
    mean, so excluded positions cannot influence it."""
    cfg = _tiny()
    mg = MaskGITDynamics(cfg).eval()
    B, F = 2, cfg.max_frames
    g = torch.Generator().manual_seed(9)
    codes = _codes(cfg, B, F, gen=g)
    act = torch.randn(B, F, cfg.actuator_dim, generator=g)
    live = torch.ones(B, F)
    live[:, F - 2 :] = 0.0  # last two frames dead

    codes2 = {k: v.clone() for k, v in codes.items()}
    for m in cfg.modalities:  # corrupt ONLY the dead frames
        codes2[m.name][:, F - 2 :] = (codes2[m.name][:, F - 2 :] + 1) % m.codebook_size
    with torch.no_grad():
        l1 = float(
            mg.training_loss(
                codes, act, generator=torch.Generator().manual_seed(4), live=live
            )
        )
        l2 = float(
            mg.training_loss(
                codes2, act, generator=torch.Generator().manual_seed(4), live=live
            )
        )
    assert abs(l1 - l2) < 1e-5, f"dead-frame content leaked into the loss: {l1} vs {l2}"


def test_live_and_present_multiply():
    """A token counts only if its diagnostic was recorded AND its frame is live."""
    cfg = _tiny()
    mg = MaskGITDynamics(cfg).eval()
    B, F = 2, cfg.max_frames
    g = torch.Generator().manual_seed(11)
    codes = _codes(cfg, B, F, gen=g)
    act = torch.randn(B, F, cfg.actuator_dim, generator=g)
    present = torch.ones(B, len(cfg.modalities))
    live0 = torch.ones(B, F)
    live0[:, F - 2 :] = 0.0
    with torch.no_grad():
        only_live = float(
            mg.training_loss(
                codes, act, generator=torch.Generator().manual_seed(6), live=live0
            )
        )
        with_present = float(
            mg.training_loss(
                codes,
                act,
                generator=torch.Generator().manual_seed(6),
                present=present,
                live=live0,
            )
        )
    assert (
        abs(only_live - with_present) < 1e-5
    )  # all-present adds nothing on top of live


# ----------------------------------------------------------------------------- LATENT
# HEAD --- Regress the token's FSQ grid coordinate (Gaussian head) instead of
# classifying the code. Motivation, measured 2026-09-23 on all 15 v4 codecs: encoders
# ADD temporal stability (lag-1 ac 0.486 -> 0.749) and the discrete token discards it
# (flip 0.878 while the quantised latent stays 0.686 correlated). These pin the four
# properties that make the swap safe.


def _tiny_latent(depth=2, decode=4):
    # levels whose product == codebook_size on BOTH modalities (5 = one FSQ dim of 5
    # levels)
    return DynamicsConfig(
        modalities=(
            ModalitySpec("a", "spectro", 3, 5),
            ModalitySpec("b", "slowts", 2, 5),
        ),
        d_model=16,
        depth=depth,
        n_heads=2,
        ffn_mult=2,
        k0_seed=2,
        n_predict=3,
        maskgit_decode_steps=decode,
        actuator_dim=6,
        latent_head=True,
        fsq_levels=(5,),
    )


def test_fsq_coordinate_roundtrip_is_exact_for_all_1000_production_codes():
    L = (8, 5, 5, 5)
    idx = torch.arange(1000)
    assert torch.equal(coords_to_code(code_to_coords(idx, L), L), idx)
    c = code_to_coords(idx, L)
    assert c.shape == (1000, 4) and float(c.min()) == 0.0 and float(c.max()) == 1.0


def test_discrete_path_is_untouched_when_flag_off():
    cfg = _tiny()  # latent_head defaults False
    mg = MaskGITDynamics(cfg)
    for m in cfg.modalities:
        assert mg.backbone.tok.heads[m.name].out_features == m.codebook_size
    assert mg.backbone.tok.latent_head is False


def test_latent_head_width_loss_finite_and_grads_flow():
    cfg = _tiny_latent()
    mg = MaskGITDynamics(cfg).train()
    for m in cfg.modalities:  # 2 * D, D = 1
        assert mg.backbone.tok.heads[m.name].out_features == 2
    codes = _codes(cfg, B=2, F=5)
    act = torch.randn(2, 5, cfg.actuator_dim)
    loss = mg.training_loss(codes, act, generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(loss)
    loss.backward()
    g = mg.backbone.blocks[0].spatial.qkv.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_latent_head_weighting_paths_agree_with_the_discrete_machinery():
    """present/live/per_modality must work identically under the latent head."""
    cfg = _tiny_latent()
    mg = MaskGITDynamics(cfg).eval()
    codes, act = _fixed(cfg, seed=2)
    pm = {}
    with torch.no_grad():
        a = float(
            mg.training_loss(
                codes, act, generator=torch.Generator().manual_seed(1), per_modality=pm
            )
        )
        ones_p = torch.ones(2, len(cfg.modalities))
        ones_l = torch.ones(2, 5)
        b = float(
            mg.training_loss(
                codes,
                act,
                generator=torch.Generator().manual_seed(1),
                present=ones_p,
                live=ones_l,
            )
        )
    assert set(pm) == {m.name for m in cfg.modalities}
    assert abs(a - b) < 1e-4  # all-ones == unweighted


def test_latent_generate_frame_returns_valid_codes_and_is_deterministic():
    cfg = _tiny_latent()
    mg = MaskGITDynamics(cfg).eval()
    codes = _codes(cfg, B=2, F=3, gen=torch.Generator().manual_seed(4))
    act = torch.randn(2, 4, cfg.actuator_dim)
    with torch.no_grad():
        o1 = mg.generate_frame(codes, act, generator=torch.Generator().manual_seed(7))
        o2 = mg.generate_frame(codes, act, generator=torch.Generator().manual_seed(7))
    for m in cfg.modalities:
        assert o1[m.name].shape == (2, m.n_tok)
        assert int(o1[m.name].min()) >= 0 and int(o1[m.name].max()) < m.codebook_size
        assert torch.equal(o1[m.name], o2[m.name])  # same seed -> same codes


def test_latent_nll_makes_a_neighbour_miss_cheap_and_a_far_miss_dear():
    """THE HYPOTHESIS. Cross-entropy charges a neighbour-cell miss the same as a far
    one; the latent NLL must not. Levels (5,): codes 0..4 at coords 0, .25, .5, .75, 1.
    """
    L = (5,)
    tg = torch.tensor([2])  # coord 0.5

    def nll_at(code_pred):
        mu = code_to_coords(torch.tensor([code_pred]), L)  # (1, 1)
        out = torch.cat([mu, torch.zeros_like(mu)], -1)  # log-var 0
        return float(_latent_nll(out, tg, L))

    exact, neighbour, far = nll_at(2), nll_at(3), nll_at(0)
    assert exact < neighbour < far
    assert (far - exact) > 3.5 * (neighbour - exact)  # (0.5^2)/(0.25^2) = 4x


def test_latent_sample_temperature_zero_is_the_mean_code():
    L = (8, 5, 5, 5)
    mu = code_to_coords(torch.tensor([123, 999, 0]), L)
    out = torch.cat([mu, torch.full_like(mu, -2.0)], -1)
    codes, conf, mean_codes = _latent_sample(
        out, L, 0.0, torch.Generator().manual_seed(0)
    )
    assert torch.equal(codes, torch.tensor([123, 999, 0])) and torch.equal(
        mean_codes, codes
    )
    assert conf.shape == (3,) and bool((conf == 8.0).all())  # -sum(logvar) = -4*(-2)


# ---------------------------------------------------------------- block teacher forcing


def _btf_cfg(**kw):
    return DynamicsConfig(
        modalities=(
            ModalitySpec("a", "spectro", 3, 5),
            ModalitySpec("b", "slowts", 2, 4),
        ),
        d_model=16,
        depth=2,
        n_heads=2,
        ffn_mult=2,
        k0_seed=2,
        n_predict=3,
        actuator_dim=6,
        grad_checkpointing=False,
        **kw,
    )


@pytest.mark.parametrize("cross", [False, True])
def test_block_forcing_query_equals_standard_forward_on_the_past(cross):
    """Query frame t must see exactly what inference sees: frames < t, plus itself."""
    torch.manual_seed(0)
    cfg = _btf_cfg(act_cross_attn=cross)
    model = MaskGITDynamics(cfg).eval()
    gen = torch.Generator().manual_seed(1)
    real = _codes(cfg, 2, cfg.max_frames, gen)
    query, _mask = model._random_mask(real, gen)
    act = torch.randn(2, cfg.max_frames, 6, generator=gen)
    with torch.no_grad():
        hidden = model.backbone.encode_block_forcing(real, query, act)
        for t in range(cfg.max_frames):
            sequence = {
                n: torch.cat([real[n][:, :t], query[n][:, t : t + 1]], dim=1)
                for n in real
            }
            reference = model.backbone.encode(sequence, act[:, : t + 1])[:, -1]
            assert torch.allclose(hidden[:, t], reference, atol=1e-5), t


def test_block_forcing_loss_is_finite_and_trains_every_parameter():
    torch.manual_seed(0)
    cfg = _btf_cfg()
    model = MaskGITDynamics(cfg).train()
    gen = torch.Generator().manual_seed(2)
    codes = _codes(cfg, 2, cfg.max_frames, gen)
    act = torch.randn(2, cfg.max_frames, 6, generator=gen)
    loss = model.training_loss(
        codes, act, generator=gen, ss_frac=0.75, block_forcing=True
    )
    assert torch.isfinite(loss)
    loss.backward()
    without_grad = [n for n, p in model.named_parameters() if p.grad is None]
    assert not without_grad, without_grad


def test_block_forcing_gradients_match_with_checkpointing():
    grads = []
    for checkpointing in (False, True):
        torch.manual_seed(0)
        cfg = _btf_cfg()
        cfg.grad_checkpointing = checkpointing
        model = MaskGITDynamics(cfg).train()
        gen = torch.Generator().manual_seed(3)
        codes = _codes(cfg, 2, cfg.max_frames, gen)
        act = torch.randn(2, cfg.max_frames, 6, generator=gen)
        loss = model.training_loss(
            codes, act, generator=gen, ss_frac=0.5, block_forcing=True
        )
        loss.backward()
        grads.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
    assert torch.allclose(grads[0], grads[1], atol=1e-6)


def test_forecast_context_never_sees_the_frame_it_replaces():
    """A replaced history frame must be a forecast from earlier frames only."""
    torch.manual_seed(0)
    cfg = _btf_cfg()
    model = MaskGITDynamics(cfg).eval()
    codes = _codes(cfg, 1, cfg.max_frames, torch.Generator().manual_seed(4))
    act = torch.zeros(1, cfg.max_frames, 6)
    altered = {n: c.clone() for n, c in codes.items()}
    last = cfg.max_frames - 1
    for n in altered:
        altered[n][:, last] = (altered[n][:, last] + 1) % 4
    first = model._forecast_context(codes, act, 1.0, torch.Generator().manual_seed(5))
    second = model._forecast_context(
        altered, act, 1.0, torch.Generator().manual_seed(5)
    )
    for n in codes:
        assert torch.equal(first[n][:, last], second[n][:, last])


def test_explicit_earlier_and_self_attention_matches_masked_sdpa():
    import torch.nn.functional as F

    from tokamak_foundation_model.ignite.dynamics import (
        _attend_earlier_and_self_explicit,
    )

    torch.manual_seed(0)
    q, hk, hv, sk, sv = (torch.randn(3, 2, 7, 4) for _ in range(5))
    pos = torch.arange(7)
    mask = torch.cat(
        [pos.view(1, -1) < pos.view(-1, 1), pos.view(1, -1) == pos.view(-1, 1)], 1
    )
    reference = F.scaled_dot_product_attention(
        q, torch.cat([hk, sk], 2), torch.cat([hv, sv], 2), attn_mask=mask
    )
    assert torch.allclose(
        _attend_earlier_and_self_explicit(q, hk, hv, sk, sv), reference, atol=1e-6
    )


def test_seed_dropout_off_leaves_the_loss_bit_identical():
    torch.manual_seed(0)
    cfg = _btf_cfg()
    model = MaskGITDynamics(cfg).train()
    codes = _codes(cfg, 2, cfg.max_frames, torch.Generator().manual_seed(3))
    act = torch.randn(2, cfg.max_frames, 6)
    losses = []
    for options in ({}, {"seed_drop_p": 0.0, "seed_column_drop_p": 0.0}):
        torch.manual_seed(5)
        gen = torch.Generator().manual_seed(4)
        losses.append(model.training_loss(codes, act, generator=gen, **options))
    assert torch.equal(losses[0], losses[1])


def test_seed_dropout_hides_only_seed_frames_of_varying_modalities():
    cfg = _btf_cfg()
    model = MaskGITDynamics(cfg)
    codes = _codes(cfg, 3, cfg.max_frames, torch.Generator().manual_seed(0))
    codes["b"][1] = 2  # constant over the window: absent or static, never hidden
    gen = torch.Generator().manual_seed(1)
    hidden = model._seed_dropout(codes, 1.0, 0.0, gen)
    k0 = cfg.k0_seed
    assert hidden["a"][:, :k0].all() and not hidden["a"][:, k0:].any()
    assert hidden["b"][[0, 2], :k0].all() and not hidden["b"][1].any()
    assert model._seed_dropout(codes, 0.0, 0.0, gen) is None


def test_column_dropout_hides_the_same_columns_in_every_seed_frame():
    cfg = _btf_cfg()
    model = MaskGITDynamics(cfg)
    codes = _codes(cfg, 64, cfg.max_frames, torch.Generator().manual_seed(0))
    hidden = model._seed_dropout(codes, 0.0, 1.0, torch.Generator().manual_seed(2))
    for frames in hidden.values():
        assert torch.equal(frames[:, 0], frames[:, cfg.k0_seed - 1])
        assert not frames[:, cfg.k0_seed :].any()
    a = hidden["a"][:, 0]
    assert a.any() and not a.all()  # a random share, not all or nothing


@pytest.mark.parametrize("block_forcing", [False, True])
def test_dropped_seed_tokens_are_invisible_and_scored(monkeypatch, block_forcing):
    """The point of the augmentation: a hidden seed modality must be predicted, from
    nothing of itself, even when scheduled sampling refills its frames."""
    torch.manual_seed(0)
    cfg = _btf_cfg()
    model = MaskGITDynamics(cfg).train()
    codes = _codes(cfg, 2, cfg.max_frames, torch.Generator().manual_seed(1))
    act = torch.randn(2, cfg.max_frames, 6)
    seen = {}
    backbone, tok = model.backbone, model.backbone.tok
    encode, encode_bf = backbone.encode, backbone.encode_block_forcing
    masked_logits = tok.masked_logits

    def spy_encode(inputs, actuators, act_null=None):
        if torch.is_grad_enabled():
            seen["inputs"] = [inputs]
        return encode(inputs, actuators, act_null)

    def spy_encode_bf(history, query, actuators, act_null=None):
        if torch.is_grad_enabled():
            seen["inputs"] = [history, query]
        return encode_bf(history, query, actuators, act_null)

    def spy_logits(hidden, mask):
        if torch.is_grad_enabled():
            seen["mask"] = mask
        return masked_logits(hidden, mask)

    monkeypatch.setattr(backbone, "encode", spy_encode)
    monkeypatch.setattr(backbone, "encode_block_forcing", spy_encode_bf)
    monkeypatch.setattr(tok, "masked_logits", spy_logits)
    loss = model.training_loss(
        codes,
        act,
        generator=torch.Generator().manual_seed(2),
        ss_frac=1.0,
        block_forcing=block_forcing,
        seed_drop_p=1.0,
    )
    assert torch.isfinite(loss)
    k0 = cfg.k0_seed
    for m in cfg.modalities:
        for stream in seen["inputs"]:
            assert (stream[m.name][:, :k0] == m.codebook_size).all()
        assert seen["mask"][m.name][:, :k0].all()


def test_fused_earlier_and_self_attention_matches_explicit_with_gradients():
    from tokamak_foundation_model.ignite.dynamics import (
        _attend_earlier_and_self,
        _attend_earlier_and_self_explicit,
    )

    torch.manual_seed(0)
    inputs = [torch.randn(3, 2, 7, 4) for _ in range(5)]
    outputs, grads = [], []
    for attend in (_attend_earlier_and_self, _attend_earlier_and_self_explicit):
        leaves = [x.clone().requires_grad_() for x in inputs]
        out = attend(*leaves)
        (out * torch.linspace(-1, 1, out.numel()).view_as(out)).sum().backward()
        outputs.append(out.detach())
        grads.append([leaf.grad for leaf in leaves])
    assert torch.allclose(outputs[0], outputs[1], atol=1e-6)
    for fused, explicit in zip(*grads):
        assert torch.allclose(fused, explicit, atol=1e-6)


# ------------------------------------------- classifier-free guidance, multi-step BTF


def test_act_null_removes_every_actuator_effect_and_false_changes_nothing():
    torch.manual_seed(0)
    for cross in (False, True):
        cfg = _btf_cfg(act_cross_attn=cross)
        model = MaskGITDynamics(cfg).eval()
        if cross:  # the cross-attention output is zero-initialised: make it bite
            torch.nn.init.normal_(model.backbone.act_cross.o.weight, std=0.1)
        codes = _codes(cfg, 2, cfg.max_frames, torch.Generator().manual_seed(1))
        act = torch.randn(2, cfg.max_frames, 6)
        other = torch.randn(2, cfg.max_frames, 6)
        null = torch.tensor([True, False])
        with torch.no_grad():
            plain = model.backbone.encode(codes, act)
            off = model.backbone.encode(codes, act, torch.tensor([False, False]))
            a = model.backbone.encode(codes, act, null)
            b = model.backbone.encode(codes, other, null)
        assert torch.equal(plain, off)
        assert torch.allclose(a[0], b[0])  # null sample ignores its actuators
        assert torch.allclose(a[1], plain[1])  # the other sample is untouched
        assert not torch.allclose(a[0], plain[0])  # and null really differs


def test_cfg_and_depth_defaults_leave_the_loss_bit_identical():
    torch.manual_seed(0)
    cfg = _btf_cfg()
    model = MaskGITDynamics(cfg).train()
    codes = _codes(cfg, 2, cfg.max_frames, torch.Generator().manual_seed(3))
    act = torch.randn(2, cfg.max_frames, 6)
    losses = []
    for options in ({}, {"act_drop_p": 0.0, "forecast_depth": 1, "hist_gap_p": 0.0}):
        torch.manual_seed(5)
        losses.append(
            model.training_loss(
                codes, act, generator=torch.Generator().manual_seed(4),
                ss_frac=0.5, block_forcing=True, **options,
            )
        )
    assert torch.equal(losses[0], losses[1])


def test_actuator_dropout_and_deep_forecasts_train_every_parameter():
    torch.manual_seed(0)
    cfg = _btf_cfg()
    model = MaskGITDynamics(cfg).train()
    gen = torch.Generator().manual_seed(2)
    codes = _codes(cfg, 4, cfg.max_frames, gen)
    act = torch.randn(4, cfg.max_frames, 6, generator=gen)
    loss = model.training_loss(
        codes, act, generator=gen, ss_frac=0.75, block_forcing=True,
        act_drop_p=0.5, forecast_depth=3,
    )
    assert torch.isfinite(loss)


def test_straight_through_sample_is_one_hot_with_a_softmax_gradient():
    from tokamak_foundation_model.ignite.maskgit import _straight_through_sample

    logits = torch.randn(4, 7, requires_grad=True)
    weights = _straight_through_sample(logits, 1.0, torch.Generator().manual_seed(0))
    assert torch.allclose(weights.sum(-1), torch.ones(4))
    assert torch.equal(weights.detach(), (weights.detach() > 0.5).float())
    (weights * torch.randn(4, 7)).sum().backward()
    assert logits.grad.abs().sum() > 0


def test_embedded_codes_match_the_code_lookup():
    cfg = _btf_cfg()
    tok = MaskGITDynamics(cfg).backbone.tok
    codes = _codes(cfg, 2, cfg.max_frames, torch.Generator().manual_seed(0))
    vectors = {
        m.name: tok.soft_code_vectors(
            m.name, torch.nn.functional.one_hot(codes[m.name], m.codebook_size).float()
        )
        for m in cfg.modalities
    }
    assert torch.allclose(tok.embed(vectors), tok.embed(codes), atol=1e-6)


def _spy_rollout(monkeypatch, model):
    """Record the per-step logits and the sampled codes of a rollout loss."""
    import tokamak_foundation_model.ignite.maskgit as maskgit_module

    tok, seen = model.backbone.tok, {"logits": [], "samples": []}
    logits, sample = tok.logits, maskgit_module._straight_through_sample

    def spy_logits(hidden):
        out = logits(hidden)
        seen["logits"].append(out)
        return out

    def spy_sample(frame_logits, temperature, generator):
        weights = sample(frame_logits, temperature, generator)
        seen["samples"].append(weights.argmax(-1))
        return weights

    monkeypatch.setattr(tok, "logits", spy_logits)
    monkeypatch.setattr(maskgit_module, "_straight_through_sample", spy_sample)
    return seen


def test_rollout_loss_forward_is_an_exact_rollout(monkeypatch):
    """Step j must see the real seed, the hard codes sampled at steps < j and an
    all-[MASK] frame: the condition of generate_frame's first decoding step."""
    torch.manual_seed(0)
    cfg = _btf_cfg()
    model = MaskGITDynamics(cfg).eval()
    codes = _codes(cfg, 2, cfg.max_frames, torch.Generator().manual_seed(1))
    act = torch.randn(2, cfg.max_frames, 6)
    seen = _spy_rollout(monkeypatch, model)
    k0, horizon = cfg.k0_seed, 3
    with torch.no_grad():
        model.rollout_loss(
            codes, act, horizon, generator=torch.Generator().manual_seed(2)
        )
    order = [m.name for m in cfg.modalities]
    sampled = [
        dict(zip(order, seen["samples"][j * len(order) : (j + 1) * len(order)]))
        for j in range(horizon - 1)
    ]
    for j in range(horizon):
        prefix = {
            m.name: torch.cat(
                [codes[m.name][:, :k0]]
                + [sampled[i][m.name] for i in range(j)]
                + [torch.full((2, 1, m.n_tok), m.codebook_size)],
                dim=1,
            )
            for m in cfg.modalities
        }
        with torch.no_grad():
            heads = model.backbone.tok.heads
            hidden = model.backbone.encode(prefix, act[:, : k0 + j + 1])[:, -1:]
            for m, (s, e) in zip(cfg.modalities, cfg.modality_token_slices()):
                expected = heads[m.name](hidden[:, :, s:e])
                assert torch.allclose(
                    seen["logits"][j][m.name], expected, atol=1e-5
                ), (j, m.name)


def test_rollout_loss_trains_earlier_steps_through_the_chain(monkeypatch):
    torch.manual_seed(0)
    cfg = _btf_cfg()
    model = MaskGITDynamics(cfg).train()
    codes = _codes(cfg, 2, cfg.max_frames, torch.Generator().manual_seed(1))
    act = torch.randn(2, cfg.max_frames, 6)
    seen = _spy_rollout(monkeypatch, model)
    loss = model.rollout_loss(codes, act, 2, generator=torch.Generator().manual_seed(2))
    assert torch.isfinite(loss)
    first, second = seen["logits"]
    k0 = cfg.k0_seed
    second_ce = sum(
        torch.nn.functional.cross_entropy(
            second[m.name].reshape(-1, m.codebook_size),
            codes[m.name][:, k0 + 1].reshape(-1),
        )
        for m in cfg.modalities
    )
    grads = torch.autograd.grad(
        second_ce, [first[m.name] for m in cfg.modalities], retain_graph=True
    )
    assert all(g.abs().sum() > 0 for g in grads)  # step 2's error reaches step 1
    loss.backward()
    without_grad = [n for n, p in model.named_parameters() if p.grad is None]
    assert not without_grad, without_grad


@pytest.mark.parametrize("depth", [1, 3])
def test_deep_forecast_context_never_sees_the_frame_it_replaces(depth):
    torch.manual_seed(0)
    cfg = _btf_cfg()
    model = MaskGITDynamics(cfg).eval()
    codes = _codes(cfg, 1, cfg.max_frames, torch.Generator().manual_seed(4))
    act = torch.zeros(1, cfg.max_frames, 6)
    altered = {n: c.clone() for n, c in codes.items()}
    last = cfg.max_frames - 1
    for n in altered:
        altered[n][:, last] = (altered[n][:, last] + 1) % 4
    first = model._forecast_context(
        codes, act, 1.0, torch.Generator().manual_seed(5), depth=depth
    )
    second = model._forecast_context(
        altered, act, 1.0, torch.Generator().manual_seed(5), depth=depth
    )
    for n in codes:
        assert torch.equal(first[n][:, last], second[n][:, last])


def test_deeper_forecasts_differ_from_one_step_forecasts():
    torch.manual_seed(0)
    cfg = _btf_cfg()
    model = MaskGITDynamics(cfg).eval()
    codes = _codes(cfg, 2, cfg.max_frames, torch.Generator().manual_seed(4))
    act = torch.randn(2, cfg.max_frames, 6)
    one = model._forecast_context(codes, act, 1.0, torch.Generator().manual_seed(5))
    two = model._forecast_context(
        codes, act, 1.0, torch.Generator().manual_seed(5), depth=2
    )
    assert any(not torch.equal(one[n], two[n]) for n in codes)


def test_guidance_one_is_plain_conditioning_and_zero_ignores_actuators():
    torch.manual_seed(0)
    cfg = _btf_cfg()
    model = MaskGITDynamics(cfg).eval()
    past = _codes(cfg, 2, cfg.k0_seed, torch.Generator().manual_seed(1))
    act = torch.randn(2, cfg.k0_seed + 1, 6)
    other = torch.randn(2, cfg.k0_seed + 1, 6)
    with torch.no_grad():
        plain = model.generate_frame(past, act, generator=torch.Generator().manual_seed(3))
        one = model.generate_frame(
            past, act, generator=torch.Generator().manual_seed(3), guidance=1.0
        )
        zero_a = model.generate_frame(
            past, act, generator=torch.Generator().manual_seed(3), guidance=0.0
        )
        zero_b = model.generate_frame(
            past, other, generator=torch.Generator().manual_seed(3), guidance=0.0
        )
        guided = model.rollout(
            past, torch.randn(2, cfg.max_frames, 6),
            generator=torch.Generator().manual_seed(3), guidance=3.0,
        )
    for m in cfg.modalities:
        assert torch.equal(plain[m.name], one[m.name])
        assert torch.equal(zero_a[m.name], zero_b[m.name])
        gen_part = guided[m.name][:, cfg.k0_seed :]
        assert (gen_part >= 0).all() and (gen_part < m.codebook_size).all()


# ---------------------------------------------- confidence gate, consensus rollout


def _rollout_inputs(cfg, n_predict=3):
    seed = _codes(cfg, 2, cfg.k0_seed, torch.Generator().manual_seed(1))
    act = torch.randn(2, cfg.k0_seed + n_predict, 6, generator=torch.Generator().manual_seed(2))
    return seed, act


def test_gate_and_consensus_off_are_the_plain_rollout():
    torch.manual_seed(0)
    cfg = _btf_cfg()
    model = MaskGITDynamics(cfg).eval()
    seed, act = _rollout_inputs(cfg)
    with torch.no_grad():
        plain = model.rollout(seed, act, generator=torch.Generator().manual_seed(3))
        off = model.rollout(
            seed, act, generator=torch.Generator().manual_seed(3),
            gate=0.0, consensus_n=1, consensus_q=0.9,
        )
    for m in cfg.modalities:
        assert torch.equal(plain[m.name], off[m.name])


@pytest.mark.parametrize("option", [{"gate": 1.01}, {"consensus_n": 4, "consensus_q": 1.01}])
def test_impossible_threshold_freezes_the_last_seed_frame(option):
    torch.manual_seed(0)
    cfg = _btf_cfg()
    model = MaskGITDynamics(cfg).eval()
    seed, act = _rollout_inputs(cfg)
    with torch.no_grad():
        out = model.rollout(seed, act, generator=torch.Generator().manual_seed(3), **option)
    for m in cfg.modalities:
        last = seed[m.name][:, -1:]
        assert torch.equal(out[m.name][:, cfg.k0_seed :], last.expand_as(out[m.name][:, cfg.k0_seed :]))


def test_consensus_commits_the_near_one_majority():
    cfg = _btf_cfg()
    model = MaskGITDynamics(cfg)
    # One token, FSQ (8,5,5,5): codes 0 and 1 are one cell apart, 999 far from both.
    candidates = torch.tensor([[0], [1], [0], [999]])  # B=1, N=4
    previous = torch.tensor([[500]])
    assert model._consensus(candidates, previous, 4, 0.75).item() in (0, 1)
    assert model._consensus(candidates, previous, 4, 0.76).item() == 500
    lonely = torch.tensor([[0], [999], [500], [250]])
    assert model._consensus(lonely, previous, 4, 0.5).item() == 500


def test_consensus_rollout_gives_valid_codes_and_is_deterministic():
    torch.manual_seed(0)
    cfg = _btf_cfg()
    model = MaskGITDynamics(cfg).eval()
    seed, act = _rollout_inputs(cfg)
    with torch.no_grad():
        a = model.rollout(seed, act, generator=torch.Generator().manual_seed(3), consensus_n=4, gate=0.05)
        b = model.rollout(seed, act, generator=torch.Generator().manual_seed(3), consensus_n=4, gate=0.05)
    for m in cfg.modalities:
        assert torch.equal(a[m.name], b[m.name])
        part = a[m.name][:, cfg.k0_seed :]
        assert a[m.name].shape == (2, cfg.k0_seed + 3, m.n_tok)
        assert (part >= 0).all() and (part < m.codebook_size).all()


def test_mean_consensus_rounds_the_mean_coordinates():
    from tokamak_foundation_model.ignite.frame_layout import code_to_coords, coords_to_code

    model = MaskGITDynamics(_btf_cfg())
    levels = list(model.cfg.fsq_levels)
    same = torch.tensor([[7], [7], [7], [7]])
    assert model._consensus(same, torch.tensor([[0]]), 4, 0.5, "mean").item() == 7
    mixed = torch.tensor([[0], [999], [123], [456]])
    expected = coords_to_code(code_to_coords(mixed.view(1, 4, 1), levels).mean(1), levels)
    assert torch.equal(model._consensus(mixed, torch.tensor([[0]]), 4, 0.5, "mean"), expected)


def test_mean_consensus_rollout_gives_valid_codes():
    torch.manual_seed(0)
    cfg = _btf_cfg()
    model = MaskGITDynamics(cfg).eval()
    seed, act = _rollout_inputs(cfg)
    with torch.no_grad():
        out = model.rollout(
            seed, act, generator=torch.Generator().manual_seed(3),
            consensus_n=4, consensus_mode="mean",
        )
    for m in cfg.modalities:
        part = out[m.name][:, cfg.k0_seed :]
        assert (part >= 0).all() and (part < m.codebook_size).all()


@pytest.mark.parametrize("mode", ["majority", "mean"])
def test_independent_ensemble_is_valid_deterministic_and_keeps_the_seed(mode):
    torch.manual_seed(0)
    cfg = _btf_cfg()
    model = MaskGITDynamics(cfg).eval()
    seed, act = _rollout_inputs(cfg)
    runs = []
    with torch.no_grad():
        for _ in range(2):
            runs.append(model.rollout(
                seed, act, generator=torch.Generator().manual_seed(3),
                consensus_n=4, consensus_mode=mode, consensus_shared=False,
            ))
    for m in cfg.modalities:
        assert torch.equal(runs[0][m.name], runs[1][m.name])
        assert torch.equal(runs[0][m.name][:, : cfg.k0_seed], seed[m.name])
        part = runs[0][m.name][:, cfg.k0_seed :]
        assert part.shape == (2, 3, m.n_tok)
        assert (part >= 0).all() and (part < m.codebook_size).all()


def test_independent_ensemble_of_one_equals_the_plain_rollout():
    torch.manual_seed(0)
    cfg = _btf_cfg()
    model = MaskGITDynamics(cfg).eval()
    seed, act = _rollout_inputs(cfg)
    with torch.no_grad():
        plain = model.rollout(seed, act, generator=torch.Generator().manual_seed(3))
        one = model.rollout(
            seed, act, generator=torch.Generator().manual_seed(3),
            consensus_n=1, consensus_shared=False,
        )
    for m in cfg.modalities:
        assert torch.equal(plain[m.name], one[m.name])


def test_history_gap_hides_a_stretch_and_trains_every_parameter():
    torch.manual_seed(0)
    cfg = _btf_cfg()
    model = MaskGITDynamics(cfg).train()
    gen = torch.Generator().manual_seed(6)
    codes = _codes(cfg, 4, cfg.max_frames, gen)
    gapped = model._history_gap(codes, 1.0, 3, torch.Generator().manual_seed(7))
    name = cfg.modalities[0].name
    hidden = (gapped[name] == model._mask_id(name)).all(-1)
    assert not hidden[:, 0].any()  # the first frame always stays visible
    lengths = hidden.sum(1)
    assert ((lengths >= 1) & (lengths <= 3)).all()
    for n in codes:  # one stretch, the same for every modality
        assert torch.equal((gapped[n] == model._mask_id(n)).all(-1), hidden)
    assert model._history_gap(codes, 0.0, 3, gen) is codes
    act = torch.randn(4, cfg.max_frames, 6, generator=gen)
    loss = model.training_loss(
        codes, act, generator=gen, ss_frac=0.5, block_forcing=True, hist_gap_p=1.0,
        hist_gap_max=3,
    )
    assert torch.isfinite(loss)
    loss.backward()
    without_grad = [n for n, p in model.named_parameters() if p.grad is None]
    assert not without_grad, without_grad


def test_rollout_loss_rejects_a_horizon_past_the_window():
    cfg = _btf_cfg()
    model = MaskGITDynamics(cfg)
    codes = _codes(cfg, 1, cfg.max_frames)
    with pytest.raises(ValueError, match="outside"):
        model.rollout_loss(codes, torch.randn(1, cfg.max_frames, 6), cfg.n_predict + 1)
