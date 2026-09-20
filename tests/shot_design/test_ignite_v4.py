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

from shot_design.config import load_paths
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


# --- fix round 1: what the digest actually has to protect ----------------------


def _pin_fake(paths, tmp_path, names=("ece",), ck_mods=None):
    """Pin dummy codecs + a dummy dynamics checkpoint; return the bundle dir.

    Same shape as the test above, but parameterised: `ck_mods` is the
    `modalities` tuple the fake dynamics checkpoint declares, which is what the
    manifest is cross-checked against.
    """
    src = tmp_path / "src"
    for name in names:
        _fake_codec(src / name / "codec_best.pt")
    dyn = tmp_path / "dyn.pt"
    cfg = ignite.model_cfg()
    default_mods = [
        (n, cfg["families"][n], cfg["n_tok"][n], cfg["production_vocabs"][n])
        for n in names
    ]
    torch.save(
        {
            "step": 3200,
            "modalities": ck_mods if ck_mods is not None else default_mods,
            "model": {},
        },
        dyn,
    )
    return ignite.pin_bundle(
        paths,
        codec_tmpl=str(src / "{m}" / "codec_best.pt"),
        dynamics_src=dyn,
        names=list(names),
        t0_start=float(cfg["t0_start_s"]),
    )


@pytest.fixture
def stub_codec_loader(monkeypatch):
    """`_load_codec` without the real codec classes -- this is about digests."""
    from types import SimpleNamespace

    td = SimpleNamespace(
        _load_codec=lambda family, path: (SimpleNamespace(), SimpleNamespace(d_model=8))
    )
    monkeypatch.setattr(ignite, "_dynamics", lambda: td)


def test_load_codecs_hashes_the_codecs_not_the_dynamics_checkpoint(
    paths, tmp_path, stub_codec_loader
):
    """`load_codecs` never reads the 3.3 GB dynamics file, and the design path
    calls it per request -- so it must not pay to hash it. `--check` still does."""
    out = _pin_fake(paths, tmp_path)
    (out / ignite.model_cfg()["dynamics_file"]).write_bytes(b"a different checkpoint")
    assert set(ignite.load_codecs(out)) == {"ece"}
    bad = ignite.check_bundle(paths)
    assert any(ignite.model_cfg()["dynamics_file"] in line for line in bad)
    (out / "codecs" / "ece" / "codec_best.pt").write_bytes(b"tampered")
    with pytest.raises(ignite.CheckpointMissing, match="changed on disk"):
        ignite.load_codecs(out)


def test_check_bundle_reports_a_manifest_disagreeing_with_the_checkpoint(
    paths, tmp_path
):
    """The vocab check compares the manifest to the yaml it was generated from,
    so alone it is tautological right after a pin. The checkpoint is the
    independent witness."""
    assert _pin_fake(paths, tmp_path) and ignite.check_bundle(paths) == []
    _pin_fake(paths, tmp_path, ck_mods=[("ece", "spectro", 4, 1000)])
    bad = ignite.check_bundle(paths)
    assert any("checkpoint" in line and "ece" in line for line in bad), bad


def test_bundle_identity_and_incremental_add_refuse_a_rebuilt_bundle(
    paths, tmp_path, monkeypatch, stub_codec_loader
):
    """Re-pinning rewrites codecs, dynamics and manifest together, so a
    re-pinned bundle passes its own sha256 check. What it must not pass is
    `shot_design add` against a database built from the PREVIOUS pin -- that
    would merge two codec generations into one matrix."""
    from types import SimpleNamespace

    from shot_design.shotdb import build as build_mod

    _pin_fake(paths, tmp_path)
    ident = ignite.bundle_identity(paths)
    assert ident["generation"] == "v4" and len(ident["manifest_sha256"]) == 64
    assert ignite.check_same_bundle(ident, paths) is None

    db = SimpleNamespace(
        manifest={"ignite": {"dims": [8], "modalities": ["ece"], "model": dict(ident)}}
    )
    one = {"ece": (None, SimpleNamespace(d_model=8), "spectro")}
    monkeypatch.setattr(ignite, "load_codecs", lambda *a, **k: one)
    _pin_fake(paths, tmp_path / "again", ck_mods=[("ece", "spectro", 192, 1000)])
    assert ignite.bundle_identity(paths)["manifest_sha256"] != ident["manifest_sha256"]
    with pytest.raises(RuntimeError, match="codec manifest digest"):
        build_mod._carry_over(tmp_path, db, [], None, None, paths, 1)


# --- the pinned bundle on this machine -----------------------------------------

_real = ignite.bundle_dir(load_paths())
_real_dyn = _real / ignite.model_cfg()["dynamics_file"]


@pytest.mark.real_data
@pytest.mark.skipif(
    not _real_dyn.exists(), reason=f"no pinned dynamics checkpoint at {_real_dyn}"
)
def test_the_pinned_manifest_agrees_with_the_real_checkpoints_modalities():
    """The manifest's table is written from hand-maintained yaml; the
    checkpoint carries its own. `modalities_from_manifest` calls the manifest
    "what the checkpoint was trained with", so that has to be checked against
    the checkpoint itself, on the real pin, whenever this runs."""
    man = json.loads(ignite.codec_manifest(_real).read_text())["modalities"]
    assert ignite.checkpoint_modalities(_real_dyn) == [
        (name, e["family"], e["n_tok"], e["codebook_size"]) for name, e in man.items()
    ]
    total = sum(e["n_tok"] for e in man.values())
    assert total == ignite.model_cfg()["frame_tokens"] == 1209


# --- mirnov, and production's own frame codes ----------------------------------


@pytest.fixture
def prod_cache(tmp_path, monkeypatch):
    """A stand-in for `model.frame_codes_cache`, the read-only cache production trained on.

    `model_cfg()` is patched rather than the dict it returns: `load_yaml` hands back a DEEP COPY
    on every call, so mutating one caller's dict is invisible to the next. Both modules that read
    the key are patched, because each imported the function by name.
    """
    from shot_design.design import program_reference as pr

    d = tmp_path / "prod_frame_codes"
    d.mkdir()
    cfg = ignite.model_cfg() | {"frame_codes_cache": str(d)}
    monkeypatch.setattr(ignite, "model_cfg", lambda: cfg)
    monkeypatch.setattr(pr, "model_cfg", lambda: cfg)
    return d


def _payload(n_frames: int = 3) -> dict:
    """A frame-code cache in the shipped layout for the pinned generation."""
    cfg = ignite.model_cfg()
    return {
        "codes": {
            n: torch.zeros(n_frames, cfg["n_tok"][n], dtype=torch.int32)
            for n in cfg["production_vocabs"]
        },
        "actuators": torch.zeros(n_frames, 88, dtype=torch.float16),
        "n_frames": int(n_frames),
        "vocabs": dict(cfg["production_vocabs"]),
    }


def test_cache_path_prefers_the_production_v4_cache(paths, prod_cache):
    """Production's copy wins over ours: its codes are the ones the checkpoint was trained on,
    and ours are only bit-identical to them by luck (a BLAS thread count flips spectro tokens)."""
    from shot_design.design import program_reference as pr

    (prod_cache / "190000.pt").write_bytes(b"x")
    ours = paths.data_root / "frame_codes"
    ours.mkdir()
    (ours / "190000.pt").write_bytes(b"y")
    assert pr._cache_path(190000, paths) == prod_cache / "190000.pt"
    assert pr._cache_path(190001, paths) is None


def test_validate_cache_rejects_a_v2_vocabulary(prod_cache):
    """A v2 cache and a v4 one are otherwise indistinguishable, and the generation has to be
    reported BEFORE the structural checks -- 'actuators must be float16' would send the reader
    looking for a bug in a file whose only fault is that it is a generation old."""
    from shot_design.design import program_reference as pr

    cache = _payload(1)
    cache["vocabs"]["ece"] = 32768
    with pytest.raises(ValueError, match="ece"):
        pr.validate_cache(cache)


def test_frame_codes_reads_the_production_cache_instead_of_re_encoding(paths, prod_cache):
    """No corpus file exists for this shot, so an encode could not even start."""
    torch.save(_payload(3), prod_cache / "190000.pt")
    got = ignite.frame_codes(190000, {"ece": (None, None, "spectro")}, paths, max_frames=2)
    assert set(got) == {"ece"} and got["ece"].shape == (2, ignite.model_cfg()["n_tok"]["ece"])


def test_frame_codes_encodes_when_the_cache_is_refused(paths, prod_cache):
    """`use_cache=False` is what a parity gate passes: a gate that compared production's cache
    with a copy of production's cache would pass by construction."""
    torch.save(_payload(3), prod_cache / "190000.pt")
    with pytest.raises(FileNotFoundError):
        ignite.frame_codes(
            190000, {"ece": (None, None, "spectro")}, paths, use_cache=False
        )


def test_frame_codes_refuses_a_cached_shot_from_another_generation(paths, prod_cache):
    payload = _payload(3)
    payload["vocabs"]["ece"] = 32768
    torch.save(payload, prod_cache / "190000.pt")
    with pytest.raises(ValueError, match="ece"):
        ignite.frame_codes(190000, {"ece": (None, None, "spectro")}, paths)
