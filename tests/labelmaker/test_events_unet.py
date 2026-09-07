"""The vendored TokEye U-Net still is the network the checkpoint was trained for.

Three things are pinned here, and each catches a different kind of drift.
The **parameter count** catches an edit to the vendored architecture that
still loads (a changed dropout, a lost block). The **checkpoint sha256**
catches a swapped or truncated weight file - `load_unet` refuses to hand
back a model built from bytes nobody vouched for. The **golden output**
catches everything else: a silently reordered layer, a changed padding, a
torch upgrade that moves an arithmetic result. Together they mean a label
produced by this network next year is the label it would have been today.

The golden was written by `scripts/labelmaker/pin_unet.py`, which also ran
the original `tokeye` implementation on the same input and recorded the
difference; `test_the_golden_records_a_faithful_vendoring` is what makes
that number part of the suite rather than a line in a report.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from labelmaker.events import unet  # noqa: E402

GOLDEN = Path(__file__).parent / "data" / "unet_golden.npz"
CHECKPOINT = unet.default_checkpoint_path()

needs_checkpoint = pytest.mark.skipif(
    not CHECKPOINT.exists(), reason=f"checkpoint not mounted: {CHECKPOINT}"
)
needs_golden = pytest.mark.skipif(
    not GOLDEN.exists(), reason=f"golden file missing: {GOLDEN}"
)


@pytest.fixture(autouse=True)
def _single_threaded():
    """One BLAS thread, as the golden was made: reductions must not reassociate."""
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def _version_note(recorded: str) -> str:
    """Why a golden comparison is likeliest to have moved."""
    return (
        f"golden written under torch {recorded}; running torch "
        f"{torch.__version__}"
    )


@pytest.fixture(scope="module")
def model():
    if not CHECKPOINT.exists():
        pytest.skip(f"checkpoint not mounted: {CHECKPOINT}")
    return unet.load_unet()


def test_the_config_defaults_are_the_ones_the_checkpoint_was_trained_with():
    cfg = unet.BigTFUNetConfig()
    assert (cfg.in_channels, cfg.out_channels) == (1, 2)
    assert (cfg.num_layers, cfg.first_layer_size) == (5, 32)
    assert cfg.dropout_rate == 0.2


def test_the_parameter_count_is_the_pinned_one():
    # No checkpoint needed: this is a property of the vendored code alone,
    # so it fails on an architecture edit even where the weights are absent.
    built = unet.BigTFUNetModel(unet.BigTFUNetConfig())
    assert sum(p.numel() for p in built.parameters()) == unet.N_PARAMS
    assert unet.N_PARAMS == 7_852_002


def test_the_default_path_follows_the_model_artifact_convention(monkeypatch):
    monkeypatch.setenv("LABELMAKER_ROOT", "/tmp/not-a-real-root")
    got = unet.default_checkpoint_path()
    assert got == Path("/tmp/not-a-real-root/models/tokeye/big_tf_unet_251210.pt")
    assert unet.CHECKPOINT_NAME == "big_tf_unet_251210.pt"


@needs_checkpoint
def test_load_unet_returns_an_evaluating_cpu_model(model):
    assert isinstance(model, unet.BigTFUNetModel)
    assert not model.training                    # dropout and batchnorm frozen
    assert all(p.device.type == "cpu" for p in model.parameters())
    assert sum(p.numel() for p in model.parameters()) == unet.N_PARAMS


@needs_checkpoint
def test_the_checkpoint_on_disk_has_the_pinned_hash():
    from labelmaker.config import sha256_of

    assert sha256_of(CHECKPOINT) == unet.CHECKPOINT_SHA256


@needs_checkpoint
def test_a_corrupted_checkpoint_is_refused_and_both_hashes_are_named(tmp_path):
    bad = tmp_path / unet.CHECKPOINT_NAME
    shutil.copyfile(CHECKPOINT, bad)
    data = bytearray(bad.read_bytes())
    data[len(data) // 2] ^= 0x01                 # one bit, deep in the weights
    bad.write_bytes(bytes(data))

    with pytest.raises(ValueError) as excinfo:
        unet.load_unet(bad)
    message = str(excinfo.value)
    assert unet.CHECKPOINT_SHA256 in message     # what was expected
    from labelmaker.config import sha256_of
    assert sha256_of(bad) in message             # and what is actually there


@needs_checkpoint
def test_verify_sha256_false_is_the_only_way_past_the_guard(monkeypatch):
    monkeypatch.setattr(unet, "CHECKPOINT_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        unet.load_unet()
    # The escape hatch exists for a re-pinning run, and it is the *only*
    # thing that skips the check - so it has to be asked for by name.
    assert isinstance(unet.load_unet(verify_sha256=False), unet.BigTFUNetModel)


@needs_checkpoint
def test_a_wrong_parameter_count_is_refused(monkeypatch):
    monkeypatch.setattr(unet, "N_PARAMS", unet.N_PARAMS + 1)
    with pytest.raises(ValueError, match="parameter"):
        unet.load_unet()


@needs_checkpoint
def test_the_forward_pass_gives_two_probability_channels(model):
    x = torch.zeros(1, 1, 128, 128)
    with torch.no_grad():
        logits = model(x)
        probs = unet.probabilities(model, x)
    assert isinstance(logits, tuple) and len(logits) == 1
    assert logits[0].shape == (1, 2, 128, 128)   # ch0 coherent, ch1 transient
    assert probs.shape == (1, 2, 128, 128)
    assert probs.dtype == torch.float32
    assert torch.isfinite(probs).all()
    assert float(probs.min()) >= 0.0 and float(probs.max()) <= 1.0
    torch.testing.assert_close(probs, torch.sigmoid(logits[0]), rtol=0, atol=0)


@needs_checkpoint
def test_a_non_square_tile_keeps_its_shape(model):
    # The decoder pads odd upsample sizes back to the skip connection, so a
    # tile whose axes are multiples of 16 comes out exactly as it went in.
    with torch.no_grad():
        probs = unet.probabilities(model, torch.zeros(2, 1, 64, 160))
    assert probs.shape == (2, 2, 64, 160)


@needs_golden
def test_the_golden_records_a_faithful_vendoring():
    with np.load(GOLDEN, allow_pickle=False) as z:
        assert str(z["sha256"]) == unet.CHECKPOINT_SHA256
        assert int(z["n_params"]) == unet.N_PARAMS
        # The vendored module and `tokeye`'s own agreed exactly on the same
        # input: nothing was lost in the copy.
        assert float(z["max_abs_diff_vs_tokeye"]) == 0.0
        assert str(z["torch_version"])
        assert z["x"].shape == (1, 1, 128, 128) and z["x"].dtype == np.float32
        assert z["y"].shape == (2, 128, 128) and z["y"].dtype == np.float32


@needs_golden
@needs_checkpoint
def test_the_model_reproduces_the_golden_output_exactly(model):
    with np.load(GOLDEN, allow_pickle=False) as z:
        x, want, recorded = z["x"], z["y"], str(z["torch_version"])
    with torch.no_grad():
        got = unet.probabilities(model, torch.from_numpy(x)).numpy()[0]
    assert got.shape == want.shape and got.dtype == want.dtype
    # The message carries both torch versions: the likeliest cause of a
    # non-zero difference is an upgrade, and the failure should say so at
    # the failure site rather than send a reader to the golden's contents.
    assert np.abs(got - want).max() == 0.0, _version_note(recorded)


@needs_golden
def test_a_golden_mismatch_would_name_both_torch_versions():
    with np.load(GOLDEN, allow_pickle=False) as z:
        recorded = str(z["torch_version"])
    note = _version_note(recorded)
    assert recorded in note and torch.__version__ in note


def test_probabilities_never_builds_a_graph():
    # `probabilities` is called once per tile over a whole shot; a caller who
    # forgets `no_grad` must not be able to accumulate a graph over them.
    built = unet.BigTFUNetModel(unet.BigTFUNetConfig()).eval()
    x = torch.zeros(1, 1, 64, 64, requires_grad=True)
    probs = unet.probabilities(built, x)
    assert not probs.requires_grad
    assert probs.grad_fn is None
