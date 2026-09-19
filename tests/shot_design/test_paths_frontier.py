"""The Frontier paths file is a complete, loadable stand-in for `paths.yaml`.

`config.Paths` has no optional keys, so a paths file that dropped one loads on
Stellar and raises on Frontier. These two tests are what keeps the OLCF file in
step with the Stellar one it was copied from.
"""

from pathlib import Path

import yaml

from shot_design import config

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "shot_design"


def test_frontier_paths_has_every_key_of_the_default_file():
    default = yaml.safe_load((CONFIG / "paths.yaml").read_text())
    frontier = yaml.safe_load((CONFIG / "paths.frontier.yaml").read_text())
    assert set(frontier) == set(default)


def test_frontier_paths_loads_and_points_at_proj_shared(monkeypatch):
    monkeypatch.delenv("SHOT_DESIGN_DATA_ROOT", raising=False)
    p = config.load_paths(CONFIG / "paths.frontier.yaml")
    assert str(p.data_root) == "/lustre/orion/fus187/proj-shared/nchen/shot_design"
    assert str(p.foundation_model_processed_dir) == (
        "/lustre/orion/fus187/proj-shared/foundation_model"
    )
    assert str(p.models_dir) == (
        "/lustre/orion/fus187/proj-shared/nchen/shot_design/models"
    )
