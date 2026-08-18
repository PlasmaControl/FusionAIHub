"""Regenerate the Phase-B golden fixture. Run ONLY against known-good code."""
import torch
from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig, ModalitySpec
from tokamak_foundation_model.ignite.maskgit import MaskGITDynamics

CFG_KW = dict(d_model=16, depth=2, n_heads=2, ffn_mult=2,
              k0_seed=2, n_predict=3, maskgit_decode_steps=4, actuator_dim=6)
MODS = (ModalitySpec("a", "spectro", 3, 5), ModalitySpec("b", "slowts", 2, 4))


def build():
    cfg = DynamicsConfig(modalities=MODS, **CFG_KW)
    torch.manual_seed(0)
    model = MaskGITDynamics(cfg).eval()
    return cfg, model


def main():
    cfg, model = build()
    g = torch.Generator().manual_seed(1)
    seed = {m.name: torch.randint(0, m.codebook_size, (1, cfg.k0_seed, m.n_tok), generator=g)
            for m in cfg.modalities}
    act = torch.randn(1, cfg.max_frames, cfg.actuator_dim, generator=g)
    roll = model.rollout(seed, act, n_predict=3, generator=torch.Generator().manual_seed(7))
    codes = {m.name: torch.randint(0, m.codebook_size, (2, 5, m.n_tok),
                                   generator=torch.Generator().manual_seed(3))
             for m in cfg.modalities}
    tact = torch.randn(2, 5, cfg.actuator_dim, generator=torch.Generator().manual_seed(4))
    loss = model.training_loss(codes, tact, generator=torch.Generator().manual_seed(5))
    torch.save({"rollout_codes": roll, "train_loss": float(loss),
                "cfg_kw": CFG_KW}, "tests/ignite/fixtures/phaseb_golden.pt")
    print("wrote fixture; loss =", float(loss))


if __name__ == "__main__":
    main()
