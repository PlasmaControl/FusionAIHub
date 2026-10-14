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
