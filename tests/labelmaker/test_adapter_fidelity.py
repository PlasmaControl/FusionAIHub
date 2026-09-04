"""The torch evaluator equals TensorFlow on the upstream reference inputs.

The golden file was produced once by a one-off script under
tensorflow-cpu 2.15.1 (see the plan, Task 14). It pins both the inputs and
Keras' own outputs, so this test needs no TensorFlow.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from labelmaker import validate
from labelmaker.config import Paths
from labelmaker.models.d3d_tearing_onset_cnn1d import spec as tm
from labelmaker.models.runners.keras_h5 import load_ensemble, predict_members

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
    # 1e-4, not 1e-5: measured max_abs_diff is ~5.6e-5 at this evaluator's
    # default float64, and ~8.2e-5 re-measured at float32 (Task 14b) - the
    # same order, not two apart, which rules out "TensorFlow's own float32
    # rounding against our float64 arithmetic" as the explanation (that was
    # Task 14's untested guess). See validate.adapter_fidelity's docstring
    # for the localization and why 1e-4 is kept rather than adjusted to fit
    # either number: both are still two orders of magnitude below the ~1e-2
    # a genuine implementation error produces.
    assert np.abs(got - want).max() < 1e-4


def test_adapter_fidelity_report_is_a_pass(tmp_path):
    report = validate.adapter_fidelity("d3d_tearing_onset_cnn1d", golden=GOLDEN)
    assert report["passed"] is True
    assert report["max_abs_diff"] < 1e-4
    # Both dtype measurements are reported regardless of which one gates
    # `passed` - see validate.adapter_fidelity's docstring for why float64
    # is still the one that gates it.
    assert report["max_abs_diff_float64"] == report["max_abs_diff"]
    assert 0 < report["max_abs_diff_float32"] < 1e-3
    assert report["n_rows"] > 1000 and report["n_members"] == 10
    out = validate.write_report(
        Paths(root=tmp_path), "d3d_tearing_onset_cnn1d",
        "adapter_fidelity", report,
    )
    assert json.loads(out.read_text())["passed"] is True
