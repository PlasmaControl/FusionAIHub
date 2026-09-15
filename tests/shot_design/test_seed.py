"""`encode_frame_codes`: one corpus shot -> a frame-code cache in the shipped IGNITE layout.

Both tests need the frozen codecs, so both are `real_data`. The structure test reads the bundle's
own `frame_codes/190090.pt` and asserts our writer produces the same dict -- keys, dtypes and
token widths -- because that file is the contract the dynamics checkpoint validates against
(`ignite_infer.validate_shot`), and a cache that merely "looks right" is accepted silently and
rolls out confident nonsense. The determinism test uses a tiny synthetic corpus file and the
eight cheap (slow-TS / fast-TS) codecs on CPU, which is enough to pin "two runs, same bytes"
without a 3 GB spectro read.
"""

from __future__ import annotations

import importlib.util
from itertools import chain
from pathlib import Path

import h5py
import numpy as np
import pytest

from shot_design.config import load_paths
from shot_design.design import seed
from shot_design.shotdb import ignite
from shot_design.shotdb.corpus import CorpusReader

SHIPPED = 190090
#: The eight modalities whose codecs consume a handful of kilobytes per frame. The four spectro
#: modalities want 500 kHz arrays and the two video ones want 240x720 frames; neither belongs in
#: a unit test, and the G-ENC gate (scripts/shot_design/g_enc.py) covers them on real shots.
CHEAP = (
    "ts_core_density",
    "ts_core_temp",
    "ts_tangential_density",
    "ts_tangential_temp",
    "cer_ti",
    "cer_rot",
    "mse",
    "filterscopes",
)
SLOW_CHANNELS = {
    "ts_core_density": 44,
    "ts_core_temp": 44,
    "ts_tangential_density": 10,
    "ts_tangential_temp": 10,
    "cer_ti": 48,
    "cer_rot": 48,
    "mse": 69,
}
SYNTH_SHOT = 999001


@pytest.fixture(scope="module")
def bundle():
    paths = load_paths()
    if not ignite.codec_manifest(ignite.bundle_dir(paths)).exists():
        pytest.skip(f"no IGNITE bundle at {ignite.bundle_dir(paths)}")
    return paths


@pytest.fixture
def synth_corpus(tmp_path: Path) -> Path:
    """One synthetic shot with the eight cheap modalities, long enough for 8 frames.

    The traces are noisy sinusoids rather than constants on purpose: the spectro/slow-TS datasets
    re-draw a window whose std is below a floor, and a re-draw is exactly the kind of hidden
    randomness a determinism test has to be able to see.
    """
    d = tmp_path / "corpus"
    d.mkdir()
    p = d / f"{SYNTH_SHOT}_processed.h5"
    rng = np.random.default_rng(20260907)
    with h5py.File(p, "w") as f:

        def put(name, x, y):
            g = f.create_group(name)
            g.create_dataset("xdata", data=np.asarray(x, dtype=np.float32))
            g.create_dataset("ydata", data=np.asarray(y, dtype=np.float32))

        x_slow = np.arange(0.0, 2.0, 0.01)  # 100 Hz, the Thomson/CER rate
        for name, n_ch in SLOW_CHANNELS.items():
            phase = (np.arange(n_ch)[:, None] + 1.0) * x_slow[None, :]
            put(name, x_slow, np.sin(2 * np.pi * phase) + 0.1 * rng.standard_normal(phase.shape))
        x_fast = np.arange(0.0, 1.0, 1e-4)  # 10 kHz, the filterscope rate
        fast = np.sin(2 * np.pi * 30 * x_fast)[None, :] * np.ones((104, 1))
        put("filterscopes", x_fast, fast + 0.05 * rng.standard_normal(fast.shape))
    return d


@pytest.mark.real_data
def test_encode_frame_codes_writes_the_shipped_dict_structure(bundle, tmp_path):
    """Keys, dtypes and token widths equal to the bundle's own cache for the same shot."""
    import torch

    ref = torch.load(
        ignite.bundle_dir(bundle) / "frame_codes" / f"{SHIPPED}.pt",
        weights_only=False,
        map_location="cpu",
    )
    corpus = Path(bundle.foundation_model_processed_dir)
    if not (corpus / f"{SHIPPED}_processed.h5").is_file():
        pytest.skip(f"no corpus file for {SHIPPED}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = seed.encode_frame_codes(
        SHIPPED,
        reader=CorpusReader(corpus),
        device=device,
        include_video=True,
        out_dir=tmp_path,
        n_frames=4,
        paths=bundle,
    )
    assert out == tmp_path / f"{SHIPPED}.pt"
    got = torch.load(out, weights_only=False, map_location="cpu")

    assert set(got) == set(ref)
    assert set(got["codes"]) == set(ref["codes"])
    assert got["vocabs"] == ref["vocabs"]
    assert got["n_frames"] == 4
    assert got["actuators"].dtype == ref["actuators"].dtype
    assert got["actuators"].shape == (4, 88)
    for name, codes in ref["codes"].items():
        assert got["codes"][name].dtype == codes.dtype, name
        assert got["codes"][name].shape == (4, codes.shape[1]), name

    # ... and the provenance goes BESIDE it, not into it. The payload above is the compatibility
    # contract; the sidecar is where device/threads/revision/input identity live.
    from shot_design.design import provenance

    side = provenance.read_sidecar(tmp_path, SHIPPED)
    assert set(side) == set(provenance.SIDECAR_KEYS)
    assert side["device"] == device
    assert side["backfilled"] is False
    assert side["shot"] == SHIPPED
    assert side["n_frames"] == 4
    assert side["input_file"].endswith(f"{SHIPPED}_processed.h5")
    assert side["input_fingerprint"]["kind"] == "mtime+size"
    assert side["ignite_bundle"] and side["ignite_revision"]
    assert set(side["modalities"]) == set(ref["codes"])


@pytest.mark.real_data
def test_encode_frame_codes_is_deterministic(bundle, synth_corpus, tmp_path):
    import torch

    kw = {
        "reader": CorpusReader(synth_corpus),
        "device": "cpu",
        "include_video": False,
        "n_frames": 8,
        "modalities": CHEAP,
        "paths": bundle,
        "workers": 0,
    }
    a = seed.encode_frame_codes(SYNTH_SHOT, out_dir=tmp_path / "a", **kw)
    b = seed.encode_frame_codes(SYNTH_SHOT, out_dir=tmp_path / "b", **kw)
    da = torch.load(a, weights_only=False, map_location="cpu")
    db = torch.load(b, weights_only=False, map_location="cpu")
    assert set(da["codes"]) == set(CHEAP)
    assert da["n_frames"] == db["n_frames"] == 8
    for name in CHEAP:
        assert torch.equal(da["codes"][name], db["codes"][name]), name
        assert da["codes"][name].dtype == torch.int32
    assert torch.equal(da["actuators"], db["actuators"])


@pytest.mark.real_data
def test_encode_frame_codes_zero_fills_the_actuators_of_a_shot_with_none(
    bundle, synth_corpus, tmp_path
):
    """The synthetic shot carries no actuator group at all: the cache still has to be 88 wide,
    finite, and all zeros -- a NaN there would poison the dynamics model's actuator embedding."""
    import torch

    out = seed.encode_frame_codes(
        SYNTH_SHOT,
        reader=CorpusReader(synth_corpus),
        device="cpu",
        include_video=False,
        out_dir=tmp_path,
        n_frames=8,
        modalities=CHEAP,
        paths=bundle,
        workers=0,
    )
    act = torch.load(out, weights_only=False, map_location="cpu")["actuators"]
    assert act.shape == (8, 88)
    assert torch.isfinite(act).all()
    assert not act.any()


# ------------------------------------------------------------------- splitting a shot list

def test_chunk_of_is_contiguous_and_loses_nothing():
    shots = list(range(500))
    halves = [seed.chunk_of(shots, i, 2) for i in range(2)]
    assert halves[0] == shots[:250]
    assert halves[1] == shots[250:]
    assert list(chain.from_iterable(seed.chunk_of(shots, i, 7) for i in range(7))) == shots


def test_chunk_of_handles_a_list_shorter_than_the_split():
    assert seed.chunk_of([1, 2], 0, 4) == []
    assert list(chain.from_iterable(seed.chunk_of([1, 2], i, 4) for i in range(4))) == [1, 2]


def test_chunk_of_rejects_an_out_of_range_task():
    with pytest.raises(ValueError, match="not a valid split"):
        seed.chunk_of([1, 2, 3], 2, 2)


def test_wanted_modalities_drops_video_on_request_and_rejects_an_unknown_name():
    assert len(seed.wanted_modalities()) == 14
    without = seed.wanted_modalities(include_video=False)
    assert len(without) == 12
    assert not set(without) & set(seed.VIDEO_MODALITIES)
    assert seed.wanted_modalities(modalities=["mse", "mse"]) == ("mse",)
    with pytest.raises(ValueError, match="not IGNITE modalities"):
        seed.wanted_modalities(modalities=["ip"])


# ------------------------------------------------------------------- re-runnability

def test_encode_many_skips_a_shot_whose_cache_is_already_written(tmp_path, monkeypatch):
    """`--skip-existing` is what makes the encode job re-runnable: a task that hits the wall
    clock is resubmitted rather than restarted, so the flag has to actually skip -- not
    re-encode and overwrite, and not count a skipped shot as encoded."""
    calls: list[int] = []
    monkeypatch.setattr(seed.ignite, "bundle_dir", lambda paths: tmp_path / "bundle")
    monkeypatch.setattr(seed.ignite, "load_codecs", lambda ckpt, names, device: {"mse": None})
    monkeypatch.setattr(
        seed,
        "encode_frame_codes",
        lambda shot, **kw: calls.append(int(shot)) or _touch(kw["out_dir"], shot),
    )
    out = tmp_path / "frame_codes"
    out.mkdir()
    _touch(out, 111)  # already there from an earlier task

    report = seed.encode_many(
        [111, 222], reader=None, out_dir=out, device="cpu", skip_existing=True,
        paths=object(), log=lambda *_: None,
    )
    assert calls == [222]
    assert (report["n_encoded"], report["n_skipped"]) == (1, 1)

    calls.clear()
    report = seed.encode_many(
        [111, 222], reader=None, out_dir=out, device="cpu", skip_existing=False,
        paths=object(), log=lambda *_: None,
    )
    assert calls == [111, 222]
    assert (report["n_encoded"], report["n_skipped"]) == (2, 0)


def _touch(out_dir, shot):
    path = Path(out_dir) / f"{shot}.pt"
    path.write_bytes(b"")
    return path


# ------------------------------------------------------------------- the G-ENC gate's compare

def _g_enc():
    """`scripts/shot_design/g_enc.py`, imported by path -- it is a script, not a package module."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "shot_design" / "g_enc.py"
    spec = importlib.util.spec_from_file_location("g_enc_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: The 202537 reference cache's own structure, read off the shipped file (see
#: `test_the_synthetic_reference_matches_the_shipped_202537_cache`, which fails if it drifts).
#: Every G-ENC test below builds its payloads from this table, so a test that says "one changed
#: token" changes one token of a cache the checkpoint would actually accept.
REF_SHOT = 202537
REF_FRAMES = 239
REF_TOKENS = {
    "ece": 192, "bes": 192, "mhr": 192, "co2": 192,
    "tangtv_lower": 108, "tangtv_upper": 108,
    "ts_core_density": 4, "ts_core_temp": 4, "ts_tangential_density": 4,
    "ts_tangential_temp": 4, "cer_ti": 4, "cer_rot": 4, "mse": 4,
    "filterscopes": 5,
}
REF_VOCABS = {
    "ece": 32768, "bes": 64000, "mhr": 32768, "co2": 32768,
    "tangtv_lower": 64000, "tangtv_upper": 64000,
    "ts_core_density": 1000, "ts_core_temp": 1000, "ts_tangential_density": 1000,
    "ts_tangential_temp": 1000, "cer_ti": 1000, "cer_rot": 1000, "mse": 1000,
    "filterscopes": 1000,
}


def _synthetic_reference(frames: int = REF_FRAMES, seed_value: int = 202537) -> dict:
    """A faithful copy of the shipped 202537 cache: same keys, modalities, widths, vocabs, dtypes.

    Values are random rather than the shipped file's, because none of these tests is about the
    values -- they are about what `compare`/`verdict` do with a cache whose STRUCTURE is the
    shipped one and whose contents have been perturbed in one named way.
    """
    import torch

    rng = np.random.default_rng(seed_value)
    codes = {
        name: torch.from_numpy(
            rng.integers(0, REF_VOCABS[name], size=(frames, width), dtype=np.int64)
        ).to(torch.int32)
        for name, width in REF_TOKENS.items()
    }
    act = rng.standard_normal((frames, 88)).astype(np.float16)
    return {
        "codes": codes,
        "actuators": torch.from_numpy(act),
        "n_frames": int(frames),
        "vocabs": dict(REF_VOCABS),
    }


def _copy(payload: dict) -> dict:
    return {
        "codes": {k: v.clone() for k, v in payload["codes"].items()},
        "actuators": payload["actuators"].clone(),
        "n_frames": int(payload["n_frames"]),
        "vocabs": dict(payload["vocabs"]),
    }


@pytest.mark.real_data
def test_the_synthetic_reference_matches_the_shipped_202537_cache():
    """The tables above are a transcription of a real file; this is what keeps them one.

    Read-only: the shipped cache is opened, its structure compared, and nothing written. If the
    bundle ever ships a different revision, every G-ENC test below is testing the wrong contract
    and this test is the one that says so.
    """
    import torch

    paths = load_paths()
    ref_path = ignite.bundle_dir(paths) / "frame_codes" / f"{REF_SHOT}.pt"
    if not ref_path.is_file():
        pytest.skip(f"no shipped cache at {ref_path}")
    real = torch.load(ref_path, weights_only=False, map_location="cpu")
    synth = _synthetic_reference()
    assert set(real) == set(synth) == {"codes", "actuators", "n_frames", "vocabs"}
    assert int(real["n_frames"]) == REF_FRAMES
    assert real["vocabs"] == REF_VOCABS
    assert real["actuators"].shape == synth["actuators"].shape == (REF_FRAMES, 88)
    assert real["actuators"].dtype == synth["actuators"].dtype == torch.float16
    for name, width in REF_TOKENS.items():
        assert tuple(real["codes"][name].shape) == (REF_FRAMES, width), name
        assert real["codes"][name].dtype == torch.int32, name
        assert tuple(synth["codes"][name].shape) == tuple(real["codes"][name].shape), name


def test_g_enc_compare_refuses_two_different_frame_counts():
    """Truncating both sides to the shorter one lets a SHORT encode compare its prefix and pass:
    a cache with 4 of 239 frames would agree with the shipped file on all four and be declared
    bit-identical. The gate has to fail loudly instead."""
    g_enc = _g_enc()
    ref = _synthetic_reference()
    got = _synthetic_reference(frames=4)
    with pytest.raises(ValueError, match="frame count"):
        g_enc.compare(got, ref)


def test_g_enc_compare_refuses_a_cache_whose_n_frames_lies_about_its_tensors():
    """`n_frames` is a separate int from the tensors' own first dimension, so it can disagree
    with them -- and every downstream slice is taken at `n_frames`."""
    g_enc = _g_enc()
    ref = _synthetic_reference()
    got = _copy(ref)
    got["n_frames"] = 200
    with pytest.raises(ValueError, match="n_frames"):
        g_enc.compare(got, ref)


def test_g_enc_compare_refuses_a_short_shot_against_the_gates_239_frames():
    """A full shot is 239 frames. Two caches that agree with each other at 8 frames agree about
    nothing the gate is asking about."""
    g_enc = _g_enc()
    short = _synthetic_reference(frames=8)
    with pytest.raises(ValueError, match="239"):
        g_enc.compare(_copy(short), short, expect_frames=g_enc.FULL_SHOT_FRAMES)


def test_g_enc_compare_refuses_a_dtype_that_is_not_the_shipped_one():
    """int32 codes and float16 actuators are what `validate_shot` accepts; an int64 code array
    compares equal to its int32 twin and is a cache the checkpoint would refuse."""
    g_enc = _g_enc()
    import torch

    ref = _synthetic_reference()
    got = _copy(ref)
    got["codes"]["mse"] = got["codes"]["mse"].to(torch.int64)
    with pytest.raises(ValueError, match="int32"):
        g_enc.compare(got, ref)

    got = _copy(ref)
    got["actuators"] = got["actuators"].to(torch.float32)
    with pytest.raises(ValueError, match="float16"):
        g_enc.compare(got, ref)


def test_g_enc_compare_refuses_a_token_outside_its_own_vocabulary():
    """`vocabs` is the field that tells a v2 cache from a v3 one. A token at or past the codebook
    size is read by the model as some other codebook's entry, silently."""
    g_enc = _g_enc()
    ref = _synthetic_reference()
    got = _copy(ref)
    got["codes"]["ece"][0, 0] = REF_VOCABS["ece"]
    with pytest.raises(ValueError, match="vocab"):
        g_enc.compare(got, ref)

    got = _copy(ref)
    got["vocabs"]["ece"] = 16384
    with pytest.raises(ValueError, match="vocab"):
        g_enc.compare(got, ref)


def test_g_enc_compare_refuses_a_token_width_that_is_not_the_shipped_one():
    g_enc = _g_enc()
    ref = _synthetic_reference()
    got = _copy(ref)
    got["codes"]["mse"] = got["codes"]["mse"][:, :2].contiguous()
    with pytest.raises(ValueError, match="shape"):
        g_enc.compare(got, ref)


def test_g_enc_fails_when_a_requested_modality_was_never_encoded():
    """THE DEFECT. A modality missing from the encoded side used to get `equal=None`, and
    `verdict` rejected only `equal is False` -- so dropping `ece` from an otherwise identical
    cache PASSED the gate. A requested modality that produced nothing is a failure."""
    g_enc = _g_enc()
    ref = _synthetic_reference()
    got = _copy(ref)
    del got["codes"]["ece"]
    del got["vocabs"]["ece"]

    result = g_enc.compare(got, ref, requested=tuple(ref["codes"]))
    assert result["modalities"]["ece"] == {
        "requested": True, "equal": None, "agreement": None, "note": "not encoded"
    }
    ok, reasons = g_enc.verdict(result, ())
    assert ok is False
    assert any("ece" in r and "not encoded" in r for r in reasons)


def test_g_enc_does_not_fail_for_a_modality_nobody_asked_for():
    """`--no-video` is a legitimate narrower run: the two video modalities are absent because
    they were not requested, and that is recorded as such rather than as a missing encode. The
    header says in as many words that such a run is not the three-shot gate."""
    g_enc = _g_enc()
    ref = _synthetic_reference()
    got = _copy(ref)
    for name in ("tangtv_lower", "tangtv_upper"):
        del got["codes"][name]
        del got["vocabs"][name]

    requested = tuple(n for n in ref["codes"] if n not in ("tangtv_lower", "tangtv_upper"))
    result = g_enc.compare(got, ref, requested=requested)
    assert result["modalities"]["tangtv_lower"]["requested"] is False
    ok, _ = g_enc.verdict(result, ())
    assert ok is True


def test_g_enc_fails_for_a_single_changed_token():
    g_enc = _g_enc()
    ref = _synthetic_reference()
    got = _copy(ref)
    got["codes"]["co2"][17, 3] = (int(got["codes"]["co2"][17, 3]) + 1) % REF_VOCABS["co2"]

    result = g_enc.compare(got, ref, requested=tuple(ref["codes"]))
    assert result["modalities"]["co2"]["equal"] is False
    assert result["modalities"]["co2"]["n_mismatched_tokens"] == 1
    assert result["modalities"]["co2"]["n_frames_affected"] == 1
    ok, reasons = g_enc.verdict(result, ())
    assert ok is False
    assert any(r.startswith("co2 not bit-identical") for r in reasons)


def test_g_enc_fails_at_81_of_88_actuator_channels_and_passes_at_82():
    """The gate's number is 82/88 within 2e-3 z. 81 is a failure and has to read as one."""
    g_enc = _g_enc()
    ref = _synthetic_reference()

    def with_n_failing(n: int) -> dict:
        got = _copy(ref)
        act = got["actuators"].float().numpy().copy()
        act[0, :n] += 1.0  # far outside 2e-3 z
        import torch

        got["actuators"] = torch.from_numpy(act).to(torch.float16)
        return got

    seven = g_enc.compare(with_n_failing(7), ref, requested=tuple(ref["codes"]))
    assert seven["actuators"]["within_tol"] == 81
    ok, reasons = g_enc.verdict(seven, ())
    assert ok is False
    assert any("81/88" in r for r in reasons)

    six = g_enc.compare(with_n_failing(6), ref, requested=tuple(ref["codes"]))
    assert six["actuators"]["within_tol"] == 82
    assert g_enc.verdict(six, ())[0] is True


def test_g_enc_passes_only_when_every_requested_modality_is_bit_identical():
    g_enc = _g_enc()
    ref = _synthetic_reference()
    result = g_enc.compare(_copy(ref), ref, requested=tuple(ref["codes"]))
    assert all(m["equal"] for m in result["modalities"].values())
    assert result["actuators"]["within_tol"] == 88
    assert g_enc.verdict(result, ())[0] is True


def test_g_enc_marks_anything_narrower_than_the_three_shot_gate_as_a_diagnostic():
    """A one-shot CPU smoke run, `--no-video` or `--allow-partial` can PASS while the gate fails,
    and its report is what gets quoted - so the report says which kind of run it was, from ONE
    rule. The rule lives here rather than inline in `main`, where no test reaches it and where
    the critic's "one-shot smoke result does not satisfy the three-shot gate" was a sentence and
    not a field."""
    g = _g_enc()
    gate = list(g.DEFAULT_SHOTS)
    assert g.is_diagnostic(gate, no_video=False, allow_partial=False) is False
    # Order is not a narrowing.
    assert g.is_diagnostic(list(reversed(gate)), no_video=False, allow_partial=False) is False
    # Each of the three ways a run is narrower than the gate is, on its own, a diagnostic.
    assert g.is_diagnostic([202537], no_video=False, allow_partial=False) is True
    assert g.is_diagnostic(gate, no_video=True, allow_partial=False) is True
    assert g.is_diagnostic(gate, no_video=False, allow_partial=True) is True


def test_g_enc_header_does_not_claim_a_demonstrated_input_difference():
    """The 190735/190736 residual demonstrates OUTPUT disagreement. Calling it a demonstrated
    input difference asserts a cause nothing here measured -- no input file was ever hashed."""
    g_enc = _g_enc()
    doc = " ".join(g_enc.__doc__.split())
    assert "output disagreement; input difference not established without matched input hashes" in doc
    for phrase in ("An input difference IS demonstrated", "is actually DEMONSTRATED"):
        assert phrase not in doc
    assert "is actually DEMONSTRATED" not in " ".join(seed.__doc__.split())


def test_encode_frame_codes_refuses_to_silently_drop_a_requested_modality(tmp_path, monkeypatch):
    """THE OTHER HALF OF THE DEFECT. The encoder reduced `names` to whatever codecs the loader
    handed back and then checked completeness against the reduction, so a bundle missing a codec
    produced a cache that was complete by its own definition."""
    monkeypatch.setattr(seed.ignite, "bundle_dir", lambda paths: tmp_path / "bundle")
    with pytest.raises(seed.ignite.CheckpointMissing, match="ece"):
        seed.encode_frame_codes(
            999002,
            reader=None,
            out_dir=tmp_path,
            modalities=["ece", "mse"],
            codecs={"mse": None},
            paths=object(),
        )


def test_encode_frame_codes_allows_a_named_partial_run_for_diagnostics(tmp_path, monkeypatch):
    """`allow_partial=True` is the diagnostic escape hatch: the run proceeds without the codec,
    the cache simply lacks that modality, and G-ENC's `compare`/`verdict` then fail on it
    because it was requested. It is never the three-shot gate."""
    calls: dict = {}

    def fake_frame_codes(shot, codecs, paths, **kw):
        calls["codecs"] = tuple(codecs)
        raise RuntimeError("stop here: the codec set is what this test is about")

    monkeypatch.setattr(seed.ignite, "bundle_dir", lambda paths: tmp_path / "bundle")
    monkeypatch.setattr(seed.ignite, "frame_codes", fake_frame_codes)
    monkeypatch.setattr(seed.ignite, "model_cfg", lambda: {"t0_start_s": 0.0})
    with pytest.raises(RuntimeError, match="stop here"):
        seed.encode_frame_codes(
            999002,
            reader=type("R", (), {"corpus_dir": tmp_path})(),
            out_dir=tmp_path,
            modalities=["ece", "mse"],
            codecs={"mse": None},
            paths=object(),
            allow_partial=True,
        )
    assert calls["codecs"] == ("mse",)
