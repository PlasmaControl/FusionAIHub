"""`shot_design model --download` follows `model.repo_id`, not the generation number.

v4 was pinned from local checkpoints on Frontier (`--pin`) and, since 2026-09-21, is also
published as the private Hub repo `nc1/IGNITE-v4`, so a machine without the Frontier
checkpoints (Stellar) installs it with `--download --full`. The gate is therefore "is a
repo_id pinned", never "is this v2".
"""

from __future__ import annotations

import argparse

import pytest

from shot_design import cli
from shot_design.shotdb import ignite

V4 = {
    "generation": "v4",
    "local_name": "IGNITE_v4",
    "repo_id": "nc1/IGNITE-v4",
    "revision": "d2f12b82d2d0efc236293dbf5372f678f82aa309",
    "dynamics_file": "ignite_dynamics_prod_v4_mskfull_step3200.pt",
    "n_tok": {"ece": 192},
}


def _args(**flags):
    base = {"download": False, "full": False, "pin": False, "check": False}
    return argparse.Namespace(**{**base, **flags})


def test_download_uses_the_pinned_repo_for_a_published_v4(paths, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(ignite, "model_cfg", lambda: dict(V4))
    monkeypatch.setattr(cli.config, "load_paths", lambda: paths)
    monkeypatch.setattr(
        ignite, "download_bundle", lambda p, full=False: calls.append(full) or p.models_dir
    )
    rc = cli.cmd_model(_args(download=True, full=True))
    assert calls == [True]
    out = capsys.readouterr().out
    assert "nc1/IGNITE-v4 @ d2f12b82d2d0" in out
    # No manifest was actually downloaded (stubbed), so the status part reports the gap
    # and points at --download, not at --pin.
    assert rc == 1 and "shot_design model --download" in out


def test_download_is_refused_when_no_repo_is_pinned(paths, monkeypatch, capsys):
    cfg = {k: v for k, v in V4.items() if k not in ("repo_id", "revision")}
    monkeypatch.setattr(ignite, "model_cfg", lambda: cfg)
    monkeypatch.setattr(cli.config, "load_paths", lambda: paths)
    monkeypatch.setattr(
        ignite, "download_bundle", lambda *a, **k: pytest.fail("must not download")
    )
    rc = cli.cmd_model(_args(download=True))
    err = capsys.readouterr().err
    assert rc == 1 and "not published" in err and "--pin" in err
