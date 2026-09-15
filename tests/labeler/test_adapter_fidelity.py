"""The torch evaluator equals TensorFlow on the upstream reference inputs.

The golden file was produced once by a one-off script under
tensorflow-cpu 2.15.1 (see the plan, Task 14). It pins both the inputs and
Keras' own outputs, so this test needs no TensorFlow.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from labeler import validate
from labeler.config import Paths
from labeler.models.d3d_tearing_onset_cnn1d import spec as tm
from labeler.models.runners.keras_h5 import load_ensemble, predict_members

GOLDEN = Path(__file__).parent / "data" / "tearing_golden.npz"
UPSTREAM = Path(
    "/projects/EKOLEMEN/simple_ae_predictor/models/rt_multi_io/mse_bin_os_w"
)

pytestmark = [
    pytest.mark.skipif(not GOLDEN.exists(), reason=f"golden file missing: {GOLDEN}"),
    pytest.mark.skipif(not UPSTREAM.exists(), reason=f"weights missing: {UPSTREAM}"),
]


def test_golden_file_records_its_provenance():
    with np.load(GOLDEN, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
    assert meta["keras"].startswith("2.")
    assert meta["inputs_0d"][0] == "bt" and len(meta["inputs_0d"]) == 11
    assert len(meta["inputs_1d"]) == 5


def test_torch_evaluator_matches_keras_to_1e4():
    with np.load(GOLDEN, allow_pickle=False) as z:
        x0, x1, want = z["x0"], z["x1"], z["members"]
    graphs = load_ensemble(UPSTREAM / name for name in tm.ARTIFACTS)
    # Positional, not a dict: the ten members carry different layer names
    # (input_1/input_2 through input_19/input_20), so only the order - 0-D
    # block first, profile block second - is common across members. See
    # predict_members' own docstring.
    got = predict_members(graphs, [x0, x1])
    assert got.shape == want.shape
    # Regression floor, not the real gate: measured max_abs_diff is ~5.6e-5.
    # `validate.adapter_fidelity`'s three scale-aware gates (Task 14b) are
    # what actually decides pass/fail; this assertion only catches a real
    # blowup (a genuine implementation error measured ~1e-2 in Task 14),
    # kept two orders of magnitude below that.
    assert np.abs(got - want).max() < 1e-4


def test_adapter_fidelity_report_is_a_pass(tmp_path):
    report = validate.adapter_fidelity("d3d_tearing_onset_cnn1d", golden=GOLDEN)
    assert report["passed"] is True
    # Regression floor on the raw number, AND (since FidelityTolerances grew
    # a fourth gate) one of the things `passed` actually checks now - see
    # the docstring for the hole an error confined to a handful of rows
    # would otherwise open.
    assert report["max_abs_diff"] < 1e-4
    assert 0 < report["max_abs_diff_float32"] < 1e-3
    assert report["n_rows"] > 1000 and report["n_members"] == 10

    # The evidence behind the docstring's "not a directional bias, not a
    # uniform spread" corrections: both are computed and returned, not only
    # asserted in prose.
    assert len(report["max_abs_diff_by_member"]) == 10
    assert max(report["max_abs_diff_by_member"]) == report["max_abs_diff"]
    assert len(report["mean_signed_diff_by_column"]) == 2
    # Measured: the logit column's signed mean is negative and an order of
    # magnitude past what zero-mean rounding on 16,730 samples would give.
    assert report["mean_signed_diff_by_column"][1] < -1e-7

    # The proof behind the docstring's reasoning: our own two dtypes
    # disagree with each other by about the same amount as either dtype
    # disagrees with TensorFlow - measured ~5.48e-5, next to a raw
    # max_abs_diff of ~5.60e-5. Loosely bounded (not pinned to the exact
    # float) because this is measured from live torch/BLAS arithmetic, not
    # read back from a fixture.
    assert 1e-5 < report["self_max_abs_diff_float64_vs_float32"] < 1e-4
    assert 1e-7 < report["self_median_abs_diff_float64_vs_float32"] < 1e-6

    # The four measured gates that actually decide `passed`, each well
    # inside its tolerance - see FidelityTolerances for what each catches.
    tolerances = report["tolerances"]
    assert report["scale_normalized_max_abs_diff"] < tolerances["scale_normalized_max"]
    assert report["scale_normalized_max_abs_diff"] < 5e-6
    assert report["median_abs_diff"] < tolerances["median_abs_diff"]
    assert 1e-7 < report["median_abs_diff"] < 1e-6
    assert report["label_max_abs_diff"] < tolerances["label_max_abs_diff"]
    assert report["label_max_abs_diff"] < 2e-6
    assert "tm_prob" in report["label_max_abs_diff_by_field"]
    assert report["max_abs_diff"] < tolerances["max_abs_diff"]

    out = validate.write_report(
        Paths(root=tmp_path), "d3d_tearing_onset_cnn1d",
        "adapter_fidelity", report,
    )
    assert json.loads(out.read_text())["passed"] is True


def test_adapter_fidelity_gate_is_conjunctive():
    """A tightened gate on any one of the three measurements alone fails."""
    from labeler.validate import FidelityTolerances

    report = validate.adapter_fidelity(
        "d3d_tearing_onset_cnn1d",
        golden=GOLDEN,
        tolerances=FidelityTolerances(median_abs_diff=1e-9),
    )
    assert report["passed"] is False
    assert report["median_abs_diff"] > report["tolerances"]["median_abs_diff"]
