import sys
from types import SimpleNamespace

from labeler.config import Paths
from labeler.confinement import __main__ as cli


def test_default_outputs_are_local_without_redirecting_diagnostic_inputs(
    monkeypatch, tmp_path
):
    captured = []
    before = Paths.from_env()
    monkeypatch.setattr(sys, "argv", ["confinement", "data"])
    monkeypatch.setattr(cli, "run_directory", lambda: tmp_path)
    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda name: SimpleNamespace(main=lambda args: captured.extend(args)),
    )
    cli.main()
    assert captured == ["--labels", str(tmp_path), "--out", str(tmp_path)]
    assert Paths.from_env().raw_cache == before.raw_cache
    assert Paths.from_env().features == before.features


def test_explicit_output_override_is_preserved(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(sys, "argv", ["confinement", "labels", "--out=/custom/run"])
    monkeypatch.setattr(cli, "run_directory", lambda: tmp_path)
    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda name: SimpleNamespace(main=lambda args: captured.extend(args)),
    )
    cli.main()
    assert captured == ["--out=/custom/run"]
