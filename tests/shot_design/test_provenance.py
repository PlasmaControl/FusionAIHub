"""`design.provenance`: the sidecar that says how a frame-code cache was made.

A `frame_codes/<shot>.pt` has exactly four keys and must keep having them -- the shipped bundle's
own caches have those four and `ignite_infer.validate_shot` compares against them, so a fifth key
is not a place provenance can live. It lives in `frame_codes/<shot>.json` beside the cache
instead, which is why every test here checks the two things together: the sidecar says what it
should, and the payload it describes is untouched.

Nothing here reads the production store: the synthetic caches are `torch.save`d dicts in
`tmp_path` and the run manifests are the JSON `ideate encode` writes, hand-built.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ideate.design import provenance


def _cache(out_dir: Path, shot: int, frames: int = 3) -> Path:
    """A minimal but structurally honest frame-code cache: the four keys, nothing else."""
    import numpy as np
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "codes": {"mse": torch.zeros((frames, 4), dtype=torch.int32)},
        "actuators": torch.from_numpy(np.zeros((frames, 88), dtype=np.float16)),
        "n_frames": frames,
        "vocabs": {"mse": 1000},
    }
    path = out_dir / f"{shot}.pt"
    torch.save(payload, path)
    return path


def _run_manifest(runs_dir: Path, stamp: str, out_dir: Path, device: str, shots) -> Path:
    """What `ideate encode` writes under `runs/encode/` -- `seed.encode_many`'s report."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"encode_{stamp}_0of1.json"
    path.write_text(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "device": device,
                "include_video": True,
                "n_requested": len(shots),
                "n_encoded": len(shots),
                "n_skipped": 0,
                "elapsed_by_shot": {str(s): 1.0 for s in shots},
                "failed": {},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


# ------------------------------------------------------------------------- the sidecar itself


def test_a_sidecar_carries_every_field_the_payload_cannot(tmp_path):
    """The four-key payload records no device, no thread count, no code revision and no input
    identity, and the encode product is device-mixed (10 CUDA pilot caches, 490 CPU). Everything
    a reader needs to know whether two caches are comparable is here or nowhere."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    h5 = corpus / "190090_processed.h5"
    h5.write_bytes(b"not really hdf5, but it has a size and an mtime")

    got = provenance.build_sidecar(
        190090,
        device="cpu",
        input_file=h5,
        bundle=tmp_path / "bundle",
        n_frames=239,
        modalities=("mse",),
        include_video=False,
        run_manifest=tmp_path / "runs" / "encode" / "encode_x_0of1.json",
    )
    assert set(got) == set(provenance.SIDECAR_KEYS)
    assert got["schema"] == provenance.SCHEMA
    assert got["shot"] == 190090
    assert got["device"] == "cpu"
    assert isinstance(got["torch_threads"], int)
    assert got["torch_version"]
    assert got["git_sha"]
    assert got["backfilled"] is False
    assert got["encoded_at"].endswith("+00:00")
    assert got["run_manifest"].endswith("encode_x_0of1.json")
    assert got["n_frames"] == 239
    assert got["modalities"] == ["mse"]


def test_the_input_is_fingerprinted_by_size_and_mtime_and_says_so(tmp_path):
    """A corpus file is 2-5 GB and there are 500 of them: sha256 over the set is ~40 minutes of
    GPFS reads for a field nobody would wait for during an encode. mtime+size is what is
    recorded, and the sidecar has to SAY that -- a reader who assumes a hash gets a weaker
    guarantee than they think (a rewritten file with the same size and a restored mtime is
    indistinguishable)."""
    h5 = tmp_path / "190090_processed.h5"
    h5.write_bytes(b"x" * 1234)

    fp = provenance.fingerprint(h5)
    assert fp["kind"] == "mtime+size"
    assert fp["size_bytes"] == 1234
    assert isinstance(fp["mtime_ns"], int)
    assert fp["sha256"] is None

    hashed = provenance.fingerprint(h5, sha256=True)
    assert hashed["kind"] == "sha256"
    assert len(hashed["sha256"]) == 64


def test_a_missing_input_file_is_recorded_as_missing_not_as_a_zero(tmp_path):
    fp = provenance.fingerprint(tmp_path / "nope.h5")
    assert fp["kind"] == "missing"
    assert fp["size_bytes"] is None and fp["sha256"] is None


def test_write_and_read_round_trip_and_a_missing_sidecar_reads_as_none(tmp_path):
    codes = tmp_path / "frame_codes"
    codes.mkdir()
    body = provenance.build_sidecar(190090, device="cuda", input_file=None, bundle=None)
    path = provenance.write_sidecar(codes, 190090, body)
    assert path == codes / "190090.json"
    assert provenance.read_sidecar(codes, 190090) == body
    assert provenance.read_sidecar(codes, 999999) is None


def test_an_unreadable_sidecar_is_none_rather_than_an_exception(tmp_path):
    """A half-written JSON beside a good cache must not take down `describe_shot`."""
    codes = tmp_path / "frame_codes"
    codes.mkdir()
    (codes / "190090.json").write_text("{not json", encoding="utf-8")
    assert provenance.read_sidecar(codes, 190090) is None


def test_device_of_reads_the_sidecar_and_says_nothing_when_there_is_none(tmp_path):
    codes = tmp_path / "frame_codes"
    codes.mkdir()
    provenance.write_sidecar(
        codes, 190090, provenance.build_sidecar(190090, device="cuda", input_file=None, bundle=None)
    )
    assert provenance.device_of([codes], 190090) == "cuda"
    assert provenance.device_of([codes], 204346) is None


# ---------------------------------------------------------------------------------- backfill


def test_backfill_writes_one_sidecar_per_cache_from_the_run_manifests(tmp_path):
    """The 500 production caches were written before this sidecar existed. What can be recovered
    is recovered from `runs/encode/*.json` -- which shots a task encoded, on what device -- and
    what cannot is null rather than guessed."""
    codes = tmp_path / "frame_codes"
    runs = tmp_path / "runs" / "encode"
    for shot in (185786, 185955, 190090, 204346):
        _cache(codes, shot)
    _run_manifest(runs, "20260907T154930", codes, "cuda", [185786, 185955])
    _run_manifest(runs, "20260907T174227", codes, "cpu", [190090])

    report = provenance.backfill(codes_dir=codes, runs_dir=runs)

    assert report["n_caches"] == 4
    assert report["n_written"] == 4
    assert report["by_device"] == {"cuda": 2, "cpu": 2}
    assert report["n_without_manifest"] == 1

    cuda = provenance.read_sidecar(codes, 185786)
    assert cuda["device"] == "cuda"
    assert cuda["backfilled"] is True
    assert cuda["device_source"].startswith("run manifest")
    assert cuda["run_manifest"].endswith("encode_20260907T154930_0of1.json")
    assert cuda["torch_threads"] == 1  # scripts/ideate/encode.sbatch: OMP_NUM_THREADS=1

    cpu = provenance.read_sidecar(codes, 190090)
    assert cpu["device"] == "cpu"
    assert cpu["torch_threads"] == 4  # scripts/ideate/encode_cpu.sbatch: OMP_NUM_THREADS=4

    orphan = provenance.read_sidecar(codes, 204346)
    assert orphan["run_manifest"] is None
    assert orphan["device"] == "cpu"
    assert "assumed" in orphan["device_source"]
    assert orphan["git_sha"] is None  # the run manifests record none, and it is not inferred


def test_a_backfilled_sidecar_does_not_describe_the_process_doing_the_backfill(tmp_path):
    """The reconstruction runs weeks later in a different environment. Its torch version, its
    clock and its checkout's commit are facts about the backfill, not about the encode, and a
    provenance field holding a plausible wrong answer is worse than one holding null."""
    import os

    codes = tmp_path / "frame_codes"
    runs = tmp_path / "runs" / "encode"
    cache = _cache(codes, 185786)
    os.utime(cache, (1_757_000_000, 1_757_000_000))
    _run_manifest(runs, "20260907T154930", codes, "cuda", [185786])

    provenance.backfill(codes_dir=codes, runs_dir=runs)
    got = provenance.read_sidecar(codes, 185786)
    assert got["torch_version"] is None
    assert got["git_sha"] is None
    assert got["encoded_at"].startswith("2025-")  # the cache's own mtime, not now
    assert "mtime of 185786.pt" in got["encoded_at_source"]

    live = provenance.build_sidecar(1, device="cpu", input_file=None, bundle=None)
    assert live["torch_version"] and live["git_sha"]
    assert live["encoded_at_source"] == "the encoding process's clock"


def test_backfill_ignores_a_manifest_written_for_another_output_directory(tmp_path):
    """`runs/encode/` also holds the worker/thread sweeps, which encoded the same shots into
    scratch directories. Reading their device off would attribute a sweep's run to the
    production cache."""
    codes = tmp_path / "frame_codes"
    sweep = tmp_path / "runs" / "encode" / "sweep" / "w8"
    runs = tmp_path / "runs" / "encode"
    _cache(codes, 185786)
    _run_manifest(runs, "20260907T155251", sweep, "cuda", [185786])

    report = provenance.backfill(codes_dir=codes, runs_dir=runs)
    assert report["n_without_manifest"] == 1
    assert provenance.read_sidecar(codes, 185786)["run_manifest"] is None


def test_backfill_takes_the_latest_manifest_that_encoded_a_shot(tmp_path):
    codes = tmp_path / "frame_codes"
    runs = tmp_path / "runs" / "encode"
    _cache(codes, 185786)
    _run_manifest(runs, "20260907T154930", codes, "cuda", [185786])
    _run_manifest(runs, "20260907T180000", codes, "cpu", [185786])

    provenance.backfill(codes_dir=codes, runs_dir=runs)
    got = provenance.read_sidecar(codes, 185786)
    assert got["device"] == "cpu"
    assert got["run_manifest"].endswith("encode_20260907T180000_0of1.json")


def test_backfill_never_overwrites_a_sidecar_a_real_encode_wrote(tmp_path):
    """A backfilled sidecar is a reconstruction; one written by the encode itself is a
    measurement. The reconstruction must not replace the measurement."""
    codes = tmp_path / "frame_codes"
    runs = tmp_path / "runs" / "encode"
    _cache(codes, 185786)
    _run_manifest(runs, "20260907T154930", codes, "cuda", [185786])
    live = provenance.build_sidecar(185786, device="cpu", input_file=None, bundle=None)
    provenance.write_sidecar(codes, 185786, live)

    report = provenance.backfill(codes_dir=codes, runs_dir=runs)
    assert report["n_written"] == 0
    assert report["n_kept"] == 1
    assert provenance.read_sidecar(codes, 185786)["backfilled"] is False


def test_backfill_dry_run_writes_nothing(tmp_path):
    codes = tmp_path / "frame_codes"
    runs = tmp_path / "runs" / "encode"
    _cache(codes, 185786)
    _run_manifest(runs, "20260907T154930", codes, "cuda", [185786])

    report = provenance.backfill(codes_dir=codes, runs_dir=runs, dry_run=True)
    assert report["n_written"] == 1
    assert provenance.read_sidecar(codes, 185786) is None


def test_backfill_leaves_the_payload_byte_identical(tmp_path):
    """The compatibility claim in one assertion: the shipped bundle's loader reads these files,
    and a sidecar is only safe because it is a SEPARATE file."""
    import torch

    codes = tmp_path / "frame_codes"
    runs = tmp_path / "runs" / "encode"
    path = _cache(codes, 185786)
    before = path.read_bytes()
    _run_manifest(runs, "20260907T154930", codes, "cuda", [185786])

    provenance.backfill(codes_dir=codes, runs_dir=runs)
    assert path.read_bytes() == before
    payload = torch.load(path, weights_only=False, map_location="cpu")
    assert set(payload) == {"codes", "actuators", "n_frames", "vocabs"}


def test_the_audit_is_a_read_only_census_of_what_the_sidecars_actually_say(tmp_path):
    """What the 500 delivered sidecars CONTAIN was asserted in prose and never counted: the
    report said the input fingerprint was size+mtime, which is true of a live encode and false of
    every backfilled file, where it is `unknown` with `torch_version`, `git_sha` and
    `ignite_bundle_sha` all null. `audit` counts it instead of describing it, and writes
    nothing."""
    codes = tmp_path / "frame_codes"
    runs = tmp_path / "runs" / "encode"
    for shot in (185786, 190090, 204346):
        _cache(codes, shot)
    _run_manifest(runs, "20260907T154930", codes, "cuda", [185786])
    provenance.backfill(codes_dir=codes, runs_dir=runs)
    # one sidecar from a real encode, with a real fingerprint and a real torch version
    h5 = tmp_path / "204346.h5"
    h5.write_bytes(b"x" * 11)
    provenance.write_sidecar(
        codes, 204346,
        provenance.build_sidecar(204346, device="cuda", input_file=h5, bundle=None,
                                 torch_version="2.5.1", git_sha="deadbee"),
    )
    before = {p.name: p.stat().st_mtime_ns for p in sorted(codes.iterdir())}

    report = provenance.audit(codes_dir=codes)

    assert report["n_caches"] == 3
    assert report["n_sidecars"] == 3
    assert report["n_missing_sidecars"] == 0
    assert report["by_fingerprint_kind"] == {"unknown": 2, "mtime+size": 1}
    assert report["by_device"] == {"cuda": 2, "cpu": 1}
    assert report["backfilled"] == {"true": 2, "false": 1}
    assert report["n_null"]["torch_version"] == 2
    assert report["n_null"]["git_sha"] == 2
    assert report["n_null"]["ignite_bundle_sha"] == 3
    assert report["n_null"]["run_manifest"] == 2  # 190090 had none; the live sidecar has none
    assert {p.name: p.stat().st_mtime_ns for p in sorted(codes.iterdir())} == before, \
        "an audit reads"


def test_the_audit_says_when_a_cache_has_no_sidecar_at_all(tmp_path):
    codes = tmp_path / "frame_codes"
    for shot in (1, 2):
        _cache(codes, shot)
    provenance.write_sidecar(
        codes, 1, provenance.build_sidecar(1, device="cpu", input_file=None, bundle=None)
    )
    report = provenance.audit(codes_dir=codes)
    assert report["n_caches"] == 2 and report["n_sidecars"] == 1
    assert report["n_missing_sidecars"] == 1
    assert report["missing_sidecars"] == [2]


def test_the_backfill_script_is_a_thin_wrapper_over_the_module():
    """The script exists so the backfill can be run once from a shell; the behaviour under test
    is the module's, and this is what keeps the two from being two implementations."""
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "scripts" / "ideate" / "frame_codes_provenance.py"
    spec = importlib.util.spec_from_file_location("frame_codes_provenance_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.provenance.backfill is provenance.backfill
    assert module.provenance.audit is provenance.audit
    with pytest.raises(SystemExit):
        module.main([])  # one of --backfill/--audit is required: the script does not guess


def test_the_audit_flag_writes_nothing_and_prints_the_census(tmp_path, capsys):
    """`--audit` is the reproducible form of a claim about the delivered sidecars, so it must be
    safe to run against the production store: it opens files and writes none."""
    import importlib.util

    codes = tmp_path / "frame_codes"
    _cache(codes, 190090)
    provenance.write_sidecar(
        codes, 190090, provenance.build_sidecar(190090, device="cpu", input_file=None, bundle=None)
    )
    before = sorted((p.name, p.stat().st_mtime_ns) for p in codes.iterdir())

    path = Path(__file__).resolve().parents[2] / "scripts" / "ideate" / "frame_codes_provenance.py"
    spec = importlib.util.spec_from_file_location("frame_codes_provenance_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--audit", "--codes-dir", str(codes)]) == 0

    got = json.loads(capsys.readouterr().out)
    assert got["n_caches"] == 1 and got["by_fingerprint_kind"] == {"unknown": 1}
    assert sorted((p.name, p.stat().st_mtime_ns) for p in codes.iterdir()) == before
