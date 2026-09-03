"""CPU smoke test for the Phase-A gate-spike CLI (`spike.main`).

Exercises the whole CLI end-to-end on the ``--synthetic`` path (NO real HDF5, NO SLURM,
NO GPU): a few steps on tiny tensors, then asserts the out_dir gets a ``gate_<step>.json``,
a ``summary.json`` and a ``codec_last.pt`` checkpoint, and that ``--resume`` warm-starts
from that checkpoint (continuing the global step count).

Run:
    .pixi/envs/default/bin/python -m pytest tests/ignite/test_spike_cli.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from tokamak_foundation_model.ignite import spike


# Tiny cfg so the synthetic path + transformer stay CPU-fast. main() builds a
# channels=40 ece cfg internally; we monkeypatch SpectroCodecConfig's defaults via a
# thin subclass injected at call time so the test is quick without touching production.
def _tiny_cli_args(out_dir: Path, steps: int, extra=None):
    args = [
        "--synthetic",
        "--n_batches", "3",
        "--eval_n_batches", "2",
        "--batch_size", "2",
        "--steps", str(steps),
        "--eval_every", "1",
        "--n_frames", "3",
        "--out_dir", str(out_dir),
        "--device", "cpu",
        "--seed", "0",
    ]
    if extra:
        args += extra
    return args


def _tiny_cfg_patch(monkeypatch):
    """Shrink the codec so the synthetic CPU run is fast (small grid + shallow nets)."""
    from tokamak_foundation_model.ignite import config as cfg_mod

    orig = cfg_mod.SpectroCodecConfig

    def _small(**kw):
        kw.setdefault("channels", 4)
        kw["freq_bins"] = 64
        kw["time_frames"] = 32
        kw["patch_f"] = 32
        kw["patch_t"] = 16
        kw["d_model"] = 32
        kw["enc_depth"] = 1
        kw["dec_depth"] = 1
        kw["heads"] = 2
        kw["fsq_levels"] = [4, 4, 3]
        return orig(**kw)

    # spike.main constructs `SpectroCodecConfig(channels=40)` via the name imported into
    # the spike module namespace; patch that binding.
    monkeypatch.setattr(spike, "SpectroCodecConfig", _small)


def test_cli_smoke_writes_gate_summary_ckpt(tmp_path, monkeypatch):
    _tiny_cfg_patch(monkeypatch)
    out_dir = tmp_path / "run"

    gate = spike.main(_tiny_cli_args(out_dir, steps=3))

    # gate dict sanity
    assert gate["steps"] == 3
    assert gate["global_step"] == 3
    assert 0.0 <= gate["stability"] <= 1.0
    assert 0.0 <= gate["persistence"] <= 1.0

    # artifacts written
    gate_jsons = sorted(out_dir.glob("gate_*.json"))
    assert gate_jsons, "no gate_<step>.json written"
    # eval_every=1 with 3 steps -> gate_0/1/2
    steps_written = {int(p.stem.split("_")[1]) for p in gate_jsons}
    assert steps_written == {0, 1, 2}, steps_written

    ckpt = out_dir / "codec_last.pt"
    summary = out_dir / "summary.json"
    assert ckpt.exists(), "codec_last.pt not written"
    assert summary.exists(), "summary.json not written"

    # gate json is valid + carries pass flags
    g = json.loads(gate_jsons[-1].read_text())
    for key in ("stability", "persistence", "pass_stability", "pass_persistence",
                "forecastability", "decode", "step"):
        assert key in g, f"missing '{key}' in gate json"
    assert isinstance(g["pass_stability"], bool)

    # summary carries the config echo
    s = json.loads(summary.read_text())
    assert s["config"]["synthetic"] is True
    assert s["config"]["steps"] == 3

    # checkpoint is loadable and records the count of completed steps
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    for key in ("codec", "disc", "opt_g", "opt_d", "step"):
        assert key in ck, f"missing '{key}' in checkpoint"
    assert ck["step"] == 3  # 3 completed steps


def test_cli_resume_continues_global_step(tmp_path, monkeypatch):
    _tiny_cfg_patch(monkeypatch)
    out_dir = tmp_path / "run"

    # first run -> checkpoint records 3 completed steps (eval_every=1)
    spike.main(_tiny_cli_args(out_dir, steps=3))
    ckpt = out_dir / "codec_last.pt"
    assert ckpt.exists()
    completed = torch.load(ckpt, map_location="cpu", weights_only=False)["step"]
    assert completed == 3

    # resume for 2 more steps -> global step continues from the checkpoint
    resume_dir = tmp_path / "resume"
    gate = spike.main(
        _tiny_cli_args(resume_dir, steps=2, extra=["--resume", str(ckpt)])
    )
    assert gate["steps"] == 2
    # start_step = completed (3) -> global steps 3,4 -> global_step == 5
    assert gate["global_step"] == completed + 2, gate["global_step"]

    # resume run wrote its own gate jsons at the continued global step numbers (>= 3)
    resumed_gate_jsons = sorted(resume_dir.glob("gate_*.json"))
    assert resumed_gate_jsons
    steps_written = {int(p.stem.split("_")[1]) for p in resumed_gate_jsons}
    assert min(steps_written) >= completed, steps_written
