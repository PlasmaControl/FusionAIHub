"""Environment isolation must preserve roots set by broader-scoped fixtures."""

import pytest

from labeler.config import Paths


@pytest.fixture(scope="module")
def module_root(tmp_path_factory):
    with pytest.MonkeyPatch.context() as monkeypatch:
        root = tmp_path_factory.mktemp("labeler-module-root")
        monkeypatch.setenv("LABELER_ROOT", str(root))
        yield root


def test_module_scoped_new_root_survives_environment_isolation(module_root):
    assert Paths.from_env().root == module_root
