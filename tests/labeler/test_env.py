"""Environment settings are read under their LABELER_* names."""

import subprocess
import sys
from pathlib import Path

import pytest

from labeler.config import Paths


@pytest.mark.parametrize("suffix,field", [
    ("ROOT", "root"),
    ("CORPUS", "corpus"),
    ("TEXT_ROOT", "text_root"),
    ("LOGS_JSONL", "logs_jsonl"),
    ("LABEL_TABLES", "label_tables"),
])
@pytest.mark.parametrize("mode", ["set", "unset"])
def test_paths_read_the_labeler_names(suffix, field, mode, monkeypatch, tmp_path):
    name = f"LABELER_{suffix}"
    monkeypatch.delenv(name, raising=False)
    default = getattr(Paths.from_env(), field)
    if mode == "set":
        monkeypatch.setenv(name, str(tmp_path / "value"))
    expected = tmp_path / "value" if mode == "set" else default
    assert getattr(Paths.from_env(), field) == expected


def test_empty_root_is_an_empty_path_not_the_default(monkeypatch):
    monkeypatch.setenv("LABELER_ROOT", "")
    assert Paths.from_env().root == Path("")


@pytest.mark.parametrize("suffix", [
    "ROOT", "CORPUS", "TEXT_ROOT", "LOGS_JSONL", "LABEL_TABLES", "FDP", "GIT_SHA",
    "DEMO_OUT", "DEMO_CONFIG", "DEMO_FORCE",
])
@pytest.mark.parametrize("mode", ["set", "empty", "unset"])
def test_environment_lookup(suffix, mode, monkeypatch):
    from labeler.env import getenv

    name = "LABELER_" + suffix
    monkeypatch.delenv(name, raising=False)
    if mode != "unset":
        monkeypatch.setenv(name, "" if mode == "empty" else "value")
    want = {"set": "value", "empty": "", "unset": "default"}[mode]
    assert getenv(name, "default") == want


def test_shell_entrypoint_resolves_settings_without_importing_the_package(monkeypatch):
    from labeler import env

    monkeypatch.setenv("LABELER_ROOT", "a path with spaces")
    result = subprocess.run(
        [sys.executable, env.__file__, "LABELER_ROOT", "default"],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout == "a path with spaces\n"
    assert result.stderr == ""


# ------------------------------------------------------------------ cluster defaults


def _host(monkeypatch, name, *, cluster=None, slurm=None):
    """Pretend to run on host `name`, with LABELER_CLUSTER and SLURM_CLUSTER_NAME as given."""
    from types import SimpleNamespace

    from labeler import env

    monkeypatch.setattr(env, "socket", SimpleNamespace(gethostname=lambda: name))
    for var, value in (("LABELER_CLUSTER", cluster), ("SLURM_CLUSTER_NAME", slurm)):
        if value is None:
            monkeypatch.delenv(var, raising=False)
        else:
            monkeypatch.setenv(var, value)


@pytest.mark.parametrize("host,expected", [
    ("stellar-vis2.princeton.edu", "stellar"),
    ("stellar-i01n1", "stellar"),
    ("stellar-m03n2", "stellar"),
    ("stellarai-vis2.princeton.edu", "stellarai"),
    ("stellar-ai-m01n3", "stellarai"),
    ("login01.frontier.olcf.ornl.gov", None),
])
def test_cluster_follows_the_hostname_prefix_longest_first(host, expected, monkeypatch):
    from labeler import env

    _host(monkeypatch, host)
    assert env.cluster() == expected


def test_cluster_variable_wins_and_empty_or_unknown_means_none(monkeypatch):
    from labeler import env

    for named, expected in (("stellarai", "stellarai"), ("stellar", "stellar"),
                            ("", None), ("frontier", None)):
        _host(monkeypatch, "stellar-vis2", cluster=named)
        assert env.cluster() == expected
        assert (env.default("LABELER_ROOT") is None) == (expected is None)


def test_an_unknown_cluster_name_is_noted_once_on_stderr_and_a_known_or_empty_one_is_not(
    monkeypatch, capsys
):
    from labeler import env

    monkeypatch.setattr(env, "_NOTED", set())
    for named in ("stellar", "stellarai", ""):
        monkeypatch.setenv("LABELER_CLUSTER", named)
        env.cluster()
    assert capsys.readouterr().err == ""
    monkeypatch.setenv("LABELER_CLUSTER", "Stellar")
    assert env.cluster() is None
    assert env.cluster() is None
    err = capsys.readouterr().err
    assert err.count("LABELER_CLUSTER='Stellar'") == 1
    assert "'stellar', 'stellarai'" in err


def test_shell_entrypoint_notes_a_mistyped_cluster_on_stderr_and_still_answers():
    typo = _run_cli("LABELER_ROOT", "arg", LABELER_CLUSTER="Stellar")
    assert typo.stdout == "arg\n"
    assert "LABELER_CLUSTER='Stellar'" in typo.stderr
    for named in ("stellar", ""):
        assert _run_cli("LABELER_ROOT", "arg", LABELER_CLUSTER=named).stderr == ""


def test_slurm_cluster_name_counts_only_after_the_hostname_and_only_as_a_key(monkeypatch):
    from labeler import env

    _host(monkeypatch, "node042", slurm="stellarai")
    assert env.cluster() == "stellarai"
    _host(monkeypatch, "node042", slurm="elsewhere")
    assert env.cluster() is None
    _host(monkeypatch, "stellar-i01n1", slurm="stellarai")
    assert env.cluster() == "stellar"


def test_the_rows_name_each_clusters_roots_and_tables(monkeypatch):
    from labeler import env

    want = {
        "stellar": ("/scratch/gpfs/EKOLEMEN/nc1514/labeler",
                    "/scratch/gpfs/EKOLEMEN/nc1514/shot_design",
                    "/scratch/gpfs/EKOLEMEN/nc1514/events"),
        "stellarai": ("/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler",
                      "/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/shot_design",
                      "/scratch/gpfs/EKOLEMEN/D3D/events"),
    }
    for name, values in want.items():
        _host(monkeypatch, "anywhere", cluster=name)
        got = tuple(env.default(var) for var in (
            "LABELER_ROOT", "SHOT_DESIGN_DATA_ROOT", "LABELER_LABEL_TABLES"))
        assert got == values
    assert env.default("LABELER_CORPUS") is None


def test_paths_take_the_cluster_row_and_an_export_wins(monkeypatch, tmp_path):
    from labeler import env

    _host(monkeypatch, "anywhere", cluster="stellar")
    for var in ("LABELER_ROOT", "LABELER_LABEL_TABLES"):
        monkeypatch.delenv(var, raising=False)
    paths = Paths.from_env()
    assert paths.root == Path(env.CLUSTERS["stellar"]["env"]["LABELER_ROOT"])
    assert paths.label_tables == Path(env.CLUSTERS["stellar"]["env"]["LABELER_LABEL_TABLES"])
    monkeypatch.setenv("LABELER_ROOT", str(tmp_path / "mine"))
    assert Paths.from_env().root == tmp_path / "mine"
    _host(monkeypatch, "anywhere", cluster="")
    monkeypatch.delenv("LABELER_ROOT")
    assert Paths.from_env().root == Paths().root


def _run_cli(*args, **variables):
    """`python env.py ...` with a clean LABELER_* environment plus `variables`."""
    import os

    from labeler import env

    clean = {k: v for k, v in os.environ.items() if not k.startswith("LABELER_")}
    return subprocess.run(
        [sys.executable, env.__file__, *args], env={**clean, **variables},
        capture_output=True, text=True, check=True,
    )


def _cli(*args, **variables):
    return _run_cli(*args, **variables).stdout


def test_shell_entrypoint_resolves_the_export_then_the_cluster_then_the_argument():
    assert _cli("LABELER_ROOT", "arg", LABELER_CLUSTER="stellar", LABELER_ROOT="/mine") == "/mine\n"
    assert _cli("LABELER_ROOT", "arg", LABELER_CLUSTER="stellar") == (
        "/scratch/gpfs/EKOLEMEN/nc1514/labeler\n")
    assert _cli("LABELER_LABEL_TABLES", "arg", LABELER_CLUSTER="stellarai") == (
        "/scratch/gpfs/EKOLEMEN/D3D/events\n")
    assert _cli("LABELER_ROOT", "arg", LABELER_CLUSTER="") == "arg\n"
    assert _cli("LABELER_DEMO_FORCE", LABELER_CLUSTER="stellar") == "\n"


def test_shell_entrypoint_prints_the_cluster():
    assert _cli("--cluster", LABELER_CLUSTER="stellarai") == "stellarai\n"
    assert _cli("--cluster", LABELER_CLUSTER="") == "\n"


def test_a_launcher_never_exports_a_value_it_just_resolved_in_one_command():
    """`export X="$(env.py ...)"` returns export's status, so a failing env.py is hidden and X is
    empty; assigning first lets `set -e` stop the job."""
    import re

    scripts = Path(__file__).resolve().parents[2] / "scripts"
    pattern = re.compile(r'^\s*export\s.*="\$\(.*env\.py', re.MULTILINE)
    found = [str(p.relative_to(scripts)) for p in scripts.rglob("*")
             if p.suffix in {".sh", ".sbatch"} and pattern.search(p.read_text())]
    assert found == []


def test_a_labeler_launcher_takes_its_pixi_env_from_the_override_variable():
    """A bare `ENV=/scratch/...`, an `ENV="$REPO/.pixi/..."` or an inline quoted
    `"/scratch/.../.pixi/envs/.../bin/python"` pins the launcher to Stellar's main checkout, so
    on StellarAI even `env.py` could not run; `LABELER_PIXI_ENV` is the way out."""
    import re

    scripts = Path(__file__).resolve().parents[2] / "scripts" / "labeler"
    pattern = re.compile(
        r'^ENV=/scratch/|^ENV="\$REPO/\.pixi/|"/scratch/[^"$]*/\.pixi/envs/[^"$]*/bin/python"',
        re.MULTILINE,
    )
    found = [p.name for p in scripts.iterdir()
             if p.suffix in {".sh", ".sbatch"} and pattern.search(p.read_text())]
    assert found == []
