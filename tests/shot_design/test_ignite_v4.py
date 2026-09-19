"""The IGNITE v4 generation: a locally pinned bundle, verified by its own sha256 manifest.

v2 came from the Hugging Face Hub and was pinned by a git revision. v4 is copied out of a
proj-shared training directory whose files are live symlinks into run directories, so the pin has
to be a real copy plus a digest of what was copied -- otherwise the weights behind every
embedding in the database can change without anything here noticing.
"""

from __future__ import annotations

import hashlib
import json

import pytest
import torch

from shot_design.shotdb import ignite


def _fake_codec(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": {"d_model": 8}, "codec": {}}, path)


def test_pin_bundle_copies_resolves_symlinks_and_records_sha(paths, tmp_path):
    src = tmp_path / "codecs_v4"
    real = tmp_path / "real"
    _fake_codec(real / "ece.pt")
    (src / "ece").mkdir(parents=True)
    (src / "ece" / "codec_best.pt").symlink_to(real / "ece.pt")
    dyn = tmp_path / "dyn.pt"
    torch.save({"step": 3200, "modalities": [("ece", "spectro", 192, 1000)], "model": {}}, dyn)
    out = ignite.pin_bundle(
        paths,
        codec_tmpl=str(src / "{m}" / "codec_best.pt"),
        dynamics_src=dyn,
        names=["ece"],
        t0_start=1.0,
    )
    copied = out / "codecs" / "ece" / "codec_best.pt"
    assert copied.exists() and not copied.is_symlink()
    man = json.loads((out / "codecs" / "MANIFEST.json").read_text())
    assert man["modalities"]["ece"] == {"family": "spectro", "n_tok": 192, "codebook_size": 1000}
    assert man["sha256"]["codecs/ece/codec_best.pt"] == hashlib.sha256(copied.read_bytes()).hexdigest()
    assert man["t0_start_s"] == 1.0 and man["frame_tokens"] == 192
    assert ignite.check_bundle(paths) == []


def test_check_bundle_reports_a_changed_codec(paths, tmp_path):
    test_pin_bundle_copies_resolves_symlinks_and_records_sha(paths, tmp_path)
    p = ignite.bundle_dir(paths) / "codecs" / "ece" / "codec_best.pt"
    p.write_bytes(b"tampered")
    bad = ignite.check_bundle(paths)
    assert bad and "codecs/ece/codec_best.pt" in bad[0]
    with pytest.raises(ignite.CheckpointMissing):
        ignite.load_codecs(ignite.bundle_dir(paths))


def test_model_cfg_declares_fifteen_v4_modalities():
    cfg = ignite.model_cfg()
    assert cfg["generation"] == "v4"
    assert len(cfg["production_vocabs"]) == 15 and set(cfg["production_vocabs"].values()) == {1000}
    assert sum(cfg["n_tok"].values()) == cfg["frame_tokens"] == 1209
    assert cfg["t0_start_s"] == 1.0
