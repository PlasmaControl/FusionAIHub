"""d3d_ae_activity_seldnet end to end, on a synthetic record in `tmp_path`.

Hermetic: the checkpoint is a freshly initialised (tiny) `AeSeldNet` saved
here, the waveform is four sine chords, and nothing reads the data root or
the real weights. What is being checked is the adapter's arithmetic - the
window aggregation, the validity rules, the frequency NaN rule, the output
shapes and the attributes a stored label carries - not whether the trained
network is any good; that is the card's `model-index` and the training run's
own evaluation.
"""
import numpy as np
import pytest
import torch

from labeler.ae import transform as tr
from labeler.ae.model import AeSeldNet, AeSeldNetConfig
from labeler.features import namespace as ns
from labeler.features.store import FeatureArray
from labeler.models import registry
from labeler.models.d3d_ae_activity_seldnet import spec as ae
from labeler.models.runners import torch_pt

FS_HZ = 5.0e5
#: Eight 25 ms rows: enough for the aggregation and the record-edge rules,
#: small enough that a forward pass is milliseconds.
GRID = ns.STEP_S * np.arange(8, dtype=np.float64)


def _tiny_config() -> AeSeldNetConfig:
    # n_freq must stay 348: it is the transform's band, not a free parameter.
    return AeSeldNetConfig(
        in_channels=4, n_freq=348, pool_sizes=(6, 2, 29), conv_channels=4,
        rnn_sizes=(8, 8), fnn_size=8, dropout=0.0, n_outputs=2,
    )


def _checkpoint(tmp_path, *, name="ae.pt", bias=None):
    torch.manual_seed(0)
    module = AeSeldNet(_tiny_config())
    if bias is not None:
        # Drive the activity logit to a known constant so the aggregation can
        # be checked against a number rather than against itself.
        with torch.no_grad():
            module.fnn[-1].weight.zero_()
            module.fnn[-1].bias[:] = torch.tensor(bias, dtype=torch.float32)
    path = tmp_path / name
    torch.save(
        {
            "state_dict": module.state_dict(),
            "config": _tiny_config().as_dict(),
            "freq_lo_khz": 80.0,
            "freq_span_khz": 170.0,
        },
        path,
    )
    return path


def _waveform(*, t0=-0.05, t1=0.25, f_khz=120.0) -> FeatureArray:
    n = round((t1 - t0) * FS_HZ) + 1
    x = t0 + np.arange(n, dtype=np.float64) / FS_HZ
    y = np.stack([
        np.sin(2.0 * np.pi * (f_khz + 5.0 * ch) * 1e3 * x) for ch in range(4)
    ]).astype(np.float32)
    return FeatureArray(x=x, y=y, attrs={"resolver": "corpus"})


def _built(features):
    return ae.INPUT_SPEC.build(features, GRID)


def test_the_card_matches_the_spec_and_the_model_is_implemented():
    assert registry.card_discrepancies(ae.SLUG) == []
    assert ae.SLUG in registry.implemented()
    assert ae.ADAPTER.framework == "torch_pt"
    assert [f.name for f in ae.ADAPTER.output_spec.fields] == [
        "ae_active", "ae_frequency"
    ]


def test_the_180_training_shots_are_recorded_and_do_not_overlap_the_pool():
    assert len(ae.TRAINING_SHOTS) == 180
    assert min(ae.TRAINING_SHOTS) == 170659
    assert max(ae.TRAINING_SHOTS) == 178879
    assert ae.ADAPTER.training_shots == ae.TRAINING_SHOTS


def test_the_waveform_reaches_predict_unsampled(tmp_path):
    arr = _waveform()
    built = _built({"co2": arr})
    assert set(built.waveforms) == {"co2"}
    # Untouched: same object, native rate, every sample.
    np.testing.assert_array_equal(built.waveforms["co2"].y, arr.y)
    assert built.scalars.shape == (GRID.size, 0)
    assert built.valid.all()
    assert built.resolvers == {"co2": "corpus"}


def test_an_absent_waveform_invalidates_every_row_and_yields_nan(tmp_path):
    built = _built({})
    assert not built.valid.any()
    assert built.missing == ("co2",)
    assert "co2 absent" in built.invalid_reasons
    predict = ae.make_load("ae.pt")(_checkpoint(tmp_path).parent)
    out = predict(built)
    assert out.shape == (1, GRID.size, 2)
    assert np.isnan(out).all()


def test_predict_produces_probabilities_on_the_grid(tmp_path):
    path = _checkpoint(tmp_path)
    predict = ae.make_load(path.name)(path.parent)
    built = _built({"co2": _waveform()})
    out = predict(built)
    assert out.shape == (1, GRID.size, 2)
    active = out[0, :, 0]
    assert np.isfinite(active).all()
    assert ((active >= 0.0) & (active <= 1.0)).all()
    freq = out[0, :, 1]
    finite = np.isfinite(freq)
    # Whatever the untrained head says, a reported frequency is inside the band.
    assert ((freq[finite] >= 80.0) & (freq[finite] <= 250.0)).all()
    # Row 0's window is (-25 ms, 0]; the record starts at -50 ms, so it is
    # covered here and every row is valid.
    assert built.valid.all()


def test_a_record_that_starts_at_zero_invalidates_row_zero(tmp_path):
    path = _checkpoint(tmp_path)
    predict = ae.make_load(path.name)(path.parent)
    built = _built({"co2": _waveform(t0=0.0)})
    predict(built)
    assert not built.valid[0]
    assert built.valid[1:].all()


def test_a_confident_model_gives_ae_active_one_and_a_frequency(tmp_path):
    # logit +20 -> sigmoid 1.0 on every frame; normalised frequency 0.5 -> 165 kHz.
    path = _checkpoint(tmp_path, bias=[20.0, 0.5])
    predict = ae.make_load(path.name)(path.parent)
    built = _built({"co2": _waveform()})
    out = predict(built)
    np.testing.assert_allclose(out[0, :, 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(out[0, :, 1], 165.0, atol=1e-3)


def test_a_quiet_model_gives_nan_frequency(tmp_path):
    path = _checkpoint(tmp_path, bias=[-20.0, 0.5])
    predict = ae.make_load(path.name)(path.parent)
    built = _built({"co2": _waveform()})
    out = predict(built)
    np.testing.assert_allclose(out[0, :, 0], 0.0, atol=1e-6)
    assert np.isnan(out[0, :, 1]).all()


def test_labels_carry_the_checkpoint_digest_and_the_transform(tmp_path):
    path = _checkpoint(tmp_path)
    predict = ae.make_load(path.name)(path.parent)
    attrs = dict(predict.output_spec.fields[0].attrs)
    assert attrs["checkpoint"] == path.name
    assert len(attrs["checkpoint_sha256"]) == 64
    assert attrs["transform"] == tr.TRANSFORM_NAME
    assert attrs["band_khz"] == "80-250"
    assert "0.8" in attrs["notch_rule"]
    assert attrs["notched_bins_khz"] == "none"
    freq_attrs = dict(predict.output_spec.fields[1].attrs)
    assert "UNVALIDATED" in freq_attrs["caveat"]


def test_aggregate_puts_a_frame_in_the_window_that_ends_at_the_row():
    grid = ns.STEP_S * np.arange(3, dtype=np.float64)
    # One frame per row, at the row's own stamp, plus one just after row 0.
    t = np.array([0.0, 0.001, 0.025, 0.050])
    prob = np.array([1.0, 0.0, 0.5, 0.5])
    freq = np.array([100.0, 200.0, 150.0, 150.0])
    active, frequency, counts = ae.aggregate(t, prob, freq, grid)
    np.testing.assert_array_equal(counts, [1, 2, 1])
    # t=0 falls in row 0's window (-25 ms, 0]; t=0.001 falls in row 1's.
    np.testing.assert_allclose(active, [1.0, 0.25, 0.5])
    assert frequency[0] == pytest.approx(100.0)
    # Row 1 is below the activity floor, so its frequency is withheld.
    assert np.isnan(frequency[1])
    assert frequency[2] == pytest.approx(150.0)


def test_clip_to_grid_keeps_only_what_a_window_can_reach():
    x = np.linspace(-2.0, 9.0, 1_100_001)
    y = np.zeros((4, x.size), dtype=np.float32)
    got = ae.clip_to_grid(x, y, ns.GRID_S)
    assert got is not None
    xs, ys = got
    assert xs[0] >= -ns.STEP_S - 1e-6
    assert xs[-1] <= ns.GRID_S[-1] + ns.STEP_S + 1e-6
    assert ys.shape == (4, xs.size)


def test_clip_to_grid_refuses_a_record_shorter_than_one_fft():
    x = np.linspace(0.0, 0.001, 100)
    assert ae.clip_to_grid(x, np.zeros((4, 100)), ns.GRID_S) is None


def test_moving_average_preserves_a_constant_and_the_record_ends():
    p = np.full(20, 0.75)
    np.testing.assert_allclose(ae.moving_average(p), 0.75)
    # A single spike is spread over the smoothing width, not lost.
    spike = np.zeros(21)
    spike[10] = 1.0
    got = ae.moving_average(spike)
    assert got[10] == pytest.approx(0.2)
    assert got.sum() == pytest.approx(1.0)


def test_run_windows_matches_a_single_pass(tmp_path):
    path = _checkpoint(tmp_path)
    module, blob = torch_pt.load_module(
        path, lambda cfg: AeSeldNet(AeSeldNetConfig.from_dict(cfg))
    )
    assert blob["freq_lo_khz"] == 80.0
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 300, 348)).astype(np.float32)
    with torch.no_grad():
        whole = module(torch.from_numpy(x).unsqueeze(0))[0].numpy()
    windowed = torch_pt.run_windows(module, x, window=64, context=64)
    assert windowed.shape == whole.shape
    # The seams are an approximation with a bounded width, so this is a
    # closeness check, not an equality one - and it is what makes the card's
    # claim about the seam checkable.
    assert np.abs(windowed - whole).max() < 5e-3


def test_a_checkpoint_without_a_config_is_an_error(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"state_dict": {}}, path)
    with pytest.raises(KeyError, match="config"):
        torch_pt.load_module(path, lambda cfg: AeSeldNet())
