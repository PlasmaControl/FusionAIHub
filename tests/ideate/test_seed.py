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

from ideate.config import load_paths
from ideate.design import seed
from ideate.shotdb import ignite
from ideate.shotdb.corpus import CorpusReader

SHIPPED = 190090
#: The eight modalities whose codecs consume a handful of kilobytes per frame. The four spectro
#: modalities want 500 kHz arrays and the two video ones want 240x720 frames; neither belongs in
#: a unit test, and the G-ENC gate (scripts/ideate/g_enc.py) covers them on real shots.
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
    """`scripts/ideate/g_enc.py`, imported by path -- it is a script, not a package module."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "ideate" / "g_enc.py"
    spec = importlib.util.spec_from_file_location("g_enc_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_g_enc_compare_refuses_two_different_frame_counts():
    """Truncating both sides to the shorter one lets a SHORT encode compare its prefix and pass:
    a cache with 4 of 239 frames would agree with the shipped file on all four and be declared
    bit-identical. The gate has to fail loudly instead."""
    g_enc = _g_enc()
    with pytest.raises(ValueError, match="frame"):
        g_enc.compare({"n_frames": 4, "codes": {}}, {"n_frames": 239, "codes": {}})
