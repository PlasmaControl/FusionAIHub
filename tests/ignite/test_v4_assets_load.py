from pathlib import Path

import pytest

CODECS = Path("/lustre/orion/fus187/proj-shared/models/ignite_codecs_v4")
DYN = Path("/lustre/orion/fus187/proj-shared/models/ignite_prod_v4/runs/mskfull/dynamics_best.pt")
FAMILIES = {
    "ece": "spectro",
    "bes": "spectro",
    "mhr": "spectro",
    "co2": "spectro",
    "mirnov": "spectro",
    "tangtv_lower": "video",
    "tangtv_upper": "video",
    "ts_core_density": "slowts",
    "ts_core_temp": "slowts",
    "ts_tangential_density": "slowts",
    "ts_tangential_temp": "slowts",
    "cer_ti": "slowts",
    "cer_rot": "slowts",
    "mse": "slowts",
    "filterscopes": "fastts",
}

pytestmark = pytest.mark.skipif(not CODECS.exists(), reason="v4 assets are Frontier-only")


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_every_v4_codec_loads_strictly(name):
    from tokamak_foundation_model.ignite import train_dynamics as td

    codec = td._load_codec(FAMILIES[name], CODECS / name / "codec_best.pt")
    assert codec is not None


def test_mskfull_dynamics_checkpoint_loads_with_88_actuators():
    from tokamak_foundation_model.ignite import eval_dynamics as ed

    model, cfg, step = ed.load_model(DYN, "cpu")
    assert step == 3200
    assert len(cfg.modalities) == 15
    assert model.backbone.act_embed.weight.shape[1] == 88
