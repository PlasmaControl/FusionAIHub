"""Session isolation removes caller exports before module fixtures set roots."""

import os

import pytest

from shot_design import config


@pytest.fixture(scope="module")
def caller_package_variables():
    """Snapshot caller settings before the module fixture adds its own root."""
    return [
        name for name in os.environ
        if name.startswith(("SHOT_DESIGN_", "IDEATE_", "LABELER_", "LABELMAKER_"))
    ]


@pytest.fixture(scope="module")
def module_root(tmp_path_factory, caller_package_variables):
    with pytest.MonkeyPatch.context() as monkeypatch:
        root = tmp_path_factory.mktemp("shot-design-module-root")
        monkeypatch.setenv("SHOT_DESIGN_DATA_ROOT", str(root))
        yield root


def test_caller_package_exports_are_isolated(caller_package_variables):
    assert caller_package_variables == []


def test_module_scoped_new_root_survives_environment_isolation(module_root):
    assert config.load_paths().db_dir == module_root / "db"
