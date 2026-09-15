"""Environment isolation must preserve roots set by broader-scoped fixtures."""

import pytest

from shot_design import config


@pytest.fixture(scope="module")
def module_root(tmp_path_factory):
    with pytest.MonkeyPatch.context() as monkeypatch:
        root = tmp_path_factory.mktemp("shot-design-module-root")
        monkeypatch.setenv("SHOT_DESIGN_DATA_ROOT", str(root))
        yield root


def test_module_scoped_new_root_survives_environment_isolation(module_root):
    assert config.load_paths().db_dir == module_root / "db"
