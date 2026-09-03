import torch

from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig, ModalitySpec
from tokamak_foundation_model.ignite.maskgit import MaskGITDynamics
from tokamak_foundation_model.ignite.scoring import best_of_n, masked_pseudo_likelihood


def _tiny():
    return DynamicsConfig(
        modalities=(ModalitySpec("a", "spectro", 4, 5), ModalitySpec("b", "slowts", 3, 4)),
        d_model=16, depth=2, n_heads=2, ffn_mult=2, k0_seed=2, n_predict=3,
        maskgit_decode_steps=3, actuator_dim=6)


def _model(cfg):
    torch.manual_seed(0)
    return MaskGITDynamics(cfg).eval()


def test_pseudo_likelihood_is_finite_and_deterministic_given_a_seed():
    cfg = _tiny()
    mg = _model(cfg)
    codes = {m.name: torch.randint(0, m.codebook_size, (1, 5, m.n_tok)) for m in cfg.modalities}
    act = torch.randn(1, 5, cfg.actuator_dim)
    a = masked_pseudo_likelihood(mg, codes, act, n_draws=2,
                                 generator=torch.Generator().manual_seed(0))
    b = masked_pseudo_likelihood(mg, codes, act, n_draws=2,
                                 generator=torch.Generator().manual_seed(0))
    assert a == b and a > 0 and a == a          # finite, reproducible


def test_pseudo_likelihood_prefers_model_consistent_codes():
    """Codes the model itself generated should score better than uniform-random codes."""
    cfg = _tiny()
    mg = _model(cfg)
    seed = {m.name: torch.randint(0, m.codebook_size, (1, cfg.k0_seed, m.n_tok))
            for m in cfg.modalities}
    act = torch.randn(1, cfg.max_frames, cfg.actuator_dim)
    own = mg.rollout(seed, act, n_predict=3, generator=torch.Generator().manual_seed(1))
    rand = {m.name: torch.randint(0, m.codebook_size, own[m.name].shape) for m in cfg.modalities}
    win = slice(cfg.k0_seed, cfg.k0_seed + 3)
    s_own = masked_pseudo_likelihood(mg, own, act[:, :own["a"].shape[1]], frames=win,
                                     n_draws=4, generator=torch.Generator().manual_seed(2))
    s_rand = masked_pseudo_likelihood(mg, rand, act[:, :own["a"].shape[1]], frames=win,
                                      n_draws=4, generator=torch.Generator().manual_seed(2))
    assert s_own < s_rand


def test_best_of_n_returns_a_valid_trajectory_and_all_scores():
    cfg = _tiny()
    mg = _model(cfg)
    seed = {m.name: torch.randint(0, m.codebook_size, (1, cfg.k0_seed, m.n_tok))
            for m in cfg.modalities}
    act = torch.randn(1, cfg.max_frames, cfg.actuator_dim)
    traj, scores = best_of_n(mg, seed, act, n=3, n_predict=3,
                             generator=torch.Generator().manual_seed(5))
    assert len(scores) == 3
    for m in cfg.modalities:
        assert traj[m.name].shape == (1, cfg.k0_seed + 3, m.n_tok)
        assert (traj[m.name] < m.codebook_size).all()
