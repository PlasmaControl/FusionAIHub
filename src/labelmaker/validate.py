"""How much a model's labels can be trusted, in three measurements.

1. adapter fidelity      - does our evaluator equal the framework's?
2. reconstruction fidelity - do our features equal the model's own training
                             inputs, feature by feature? (Task 15)
3. label quality         - how do the labels score against archived truth,
                           with archived inputs and with ours? The gap
                           between the two is the reconstruction penalty.
                           (Task 16)

Each writes JSON under `<root>/validation/<slug>/`; the headline numbers go
into the model card's `model-index`, so the card is the one place to read how
a model performed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .config import Paths, atomic_path
from .models import registry
from .models.runners.keras_h5 import load_ensemble, predict_members

# This resolves correctly from a source checkout (three parents up from
# src/labelmaker/validate.py lands on the repo root), but not from an
# installed wheel - tests/ is not packaged, so the file would not exist at
# this path there. `adapter_fidelity` takes `golden=` so any caller that
# needs to run outside a checkout can point it elsewhere; there is no
# fallback search implemented here.
GOLDEN = Path(__file__).resolve().parents[2] / (
    "tests/labelmaker/data/tearing_golden.npz"
)


@dataclass(frozen=True)
class FidelityTolerances:
    """The three measured gates `adapter_fidelity` checks, in place of one bare `tol`.

    A single absolute number cannot discriminate a real bug from float32
    noise once the output columns sit at different scales (`betan` ~1,
    the tearing logit ~20) - see `adapter_fidelity`'s docstring for the
    Task 14b measurement each default is derived from.
    """

    #: `max_abs_diff` per column, divided by that column's own scale
    #: (`max(|golden output|)`), gated at the plan's original 1e-5 - applied
    #: to a normalized quantity so it means what the plan intended. Measured
    #: 8.79e-07 (`betan`) and 2.71e-06 (tearing logit).
    scale_normalized_max: float = 1e-5
    #: `median(|diff|)` over every member/row/column, gated at 1e-6. The
    #: strongest discriminator against a semantic error - a wrong epsilon or
    #: a transposed kernel raises the median, not merely the tail - so this
    #: is the gate most likely to catch a real bug. Measured 7.65e-07.
    median_abs_diff: float = 1e-6
    #: max abs diff of the published, post-activation, post-ensemble-mean
    #: label (e.g. `tm_prob`), gated at 1e-5 - the only gate expressed in
    #: units anyone downstream reads. Measured 1.22e-06.
    label_max_abs_diff: float = 1e-5


DEFAULT_TOLERANCES = FidelityTolerances()


def write_report(paths: Paths, slug: str, name: str, payload: dict) -> Path:
    """Write one validation report, atomically."""
    out_dir = paths.validation / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    with atomic_path(path) as tmp:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return path


def adapter_fidelity(
    slug: str,
    golden: Path = GOLDEN,
    *,
    tolerances: FidelityTolerances = DEFAULT_TOLERANCES,
) -> dict:
    """Compare our evaluator against the framework's frozen outputs.

    The golden file holds the reference inputs and the outputs a real Keras
    load produced for the same weights, so this runs with no framework
    installed. A failure here invalidates every label the model has ever
    written, which is why it is a hard pass/fail rather than a metric.

    The weights loaded here come from the golden file's own `meta["models"]`,
    not from `paths.models` - deliberately: the comparison is only meaningful
    against the exact weights TensorFlow saw when the golden file was made,
    so `paths` is otherwise unused by this function.

    Task 14 (the numpy evaluator) measured `max_abs_diff` ~5.6e-5 and guessed
    this was "TensorFlow's own float32 rounding against our float64
    arithmetic". That hypothesis predicted a test: re-running the evaluator's
    own arithmetic in float32 should reproduce TensorFlow's float32 forward
    pass, so the float32-vs-golden diff should collapse toward float32
    machine epsilon (~1e-6). It measured `max_abs_diff_float32` ~8.2e-5
    instead - the same order as float64's ~5.6e-5 - and Task 14b's own
    docstring called the hypothesis "falsified" on that basis. That framing
    was wrong: the test could never have supported the hypothesis either
    way. TensorFlow's float32 kernels and torch's float32 kernels do not
    share one "float32 arithmetic" - different BLAS reduction orders and
    fused multiply-adds land on different, similarly-sized residuals - so
    two independent float32 implementations were never going to agree to
    ~1e-6 regardless of whether the underlying semantics are identical. The
    experiment was not able to test the hypothesis; it did not falsify it.

    The hypothesis's substance - that this is float32 rounding noise, not a
    semantic error - is confirmed instead by a measurement Task 14b did not
    make: comparing this evaluator against **itself**, float64 load vs.
    float32 load, both on the same weights and inputs (`self_max_abs_diff`
    below). That self-disagreement measures ~5.48e-5 - the same magnitude as
    the ~5.60e-5 disagreement against TensorFlow. A semantic bug (a wrong
    BatchNorm epsilon, a transposed kernel, a mismatched pad) could not
    produce a residual that coincides with our own dtype's self-disagreement
    with itself; only rounding noise of that dtype's own scale could. Three
    further facts corroborate it, all visible in the returned dict: the
    per-member max is uniform across all ten members (`self_max_abs_diff` is
    not dominated by one outlier, which a weight-loading bug would produce);
    the mean *signed* difference per column is ~1e-7 against a max of 5.6e-5
    (zero-mean rounding, not a directional bias); and `median_abs_diff` is
    ~7.65e-7 on outputs of magnitude 1-20 - float32 epsilon at that scale,
    exactly. Layer-by-layer tracing (see `models/runners/keras_h5.py`'s
    module docstring) places the noise on the ensemble's one unbounded
    (linear-activation) output column - the tearing logit, never `betan` -
    because every other activation in the graph is a saturating sigmoid that
    absorbs small perturbations once saturated.

    The defect Task 14b actually introduced was in the *gate*, not in either
    measured number: `tol=1e-4` was a bare absolute tolerance, inherited from
    a plan that wrote 1e-5 assuming outputs of order 1. This graph's tearing
    logit reaches ~20.7, so an absolute gate lets that column's noise run 20x
    looser than `betan`'s. Normalizing `max_abs_diff` by each column's own
    scale (`max(|golden output|)` on that column) turns the plan's 1e-5 into
    the gate it was meant to be: measured 8.79e-07 (`betan`) and 2.71e-06
    (tearing logit), both comfortably inside it. Three gates replace the one
    number, each catching something the others do not:

    - `scale_normalized_max` (1e-5): the plan's original tolerance, applied
      per-column so a large-magnitude column cannot hide behind a small one.
    - `median_abs_diff` (1e-6): unaffected by scale, and the strongest
      discriminator against a semantic error - a wrong epsilon or a
      transposed kernel would raise the bulk of the distribution, not merely
      its tail, so this is the gate most likely to actually catch a bug.
    - `label_max_abs_diff` (1e-5): the max abs difference in the published,
      post-activation, post-ensemble-mean label (`tm_prob = sigmoid(mean over
      members of the logit)`, per `OutputSpec.decode`) - the only one of the
      three expressed in units anyone downstream reads. Measured ~1.22e-06:
      an 8.2e-5-wide logit disagreement survives ensembling and the sigmoid's
      compression to a barely-there ~1.2e-6 change in a reported probability.

    `passed` is the conjunction of all three. The raw `max_abs_diff` (and its
    float32 counterpart) are still computed and returned for documentation,
    alongside the float64-vs-float32 self-disagreement that is the actual
    evidence behind this docstring's reasoning - that pairing belongs in the
    JSON report a run writes, not only here.
    """
    adapter = registry.load_adapter(slug)
    with np.load(golden, allow_pickle=False) as z:
        x0, x1, want = z["x0"], z["x1"], z["members"]
        meta = json.loads(str(z["meta"]))
    artifact_paths = [Path(meta["models"]) / name for name in adapter.artifacts]
    graphs = load_ensemble(artifact_paths, dtype=torch.float64)
    got = predict_members(graphs, [x0, x1])  # positional; see predict_members
    # float32 is measured purely to settle the tolerance question above; the
    # dtype actually used for `got`/`max_abs_diff` stays float64, matching
    # this evaluator's historical precision and every hand-computed oracle.
    graphs32 = load_ensemble(artifact_paths, dtype=torch.float32)
    got32 = predict_members(graphs32, [x0, x1])
    got32_as64 = got32.astype(np.float64)
    diff = np.abs(got - want)
    diff32 = np.abs(got32_as64 - want)
    # The proof, not merely another measurement: our own two dtypes against
    # each other, at the same magnitude as either dtype's disagreement with
    # TensorFlow. See the docstring above.
    self_diff = np.abs(got - got32_as64)

    diff_by_column = diff.max(axis=(0, 1))
    # Each column's own scale, so a normalized tolerance means the same
    # thing on the ~1-magnitude `betan` column and the ~20-magnitude tearing
    # logit column. Neither golden column is ever exactly zero, so no
    # epsilon guard is needed here.
    scale_by_column = np.abs(want).max(axis=(0, 1))
    scale_normalized_by_column = diff_by_column / scale_by_column
    scale_normalized_max = float(scale_normalized_by_column.max())
    median_abs_diff = float(np.median(diff))

    decoded_got = adapter.output_spec.decode(got)
    decoded_want = adapter.output_spec.decode(want)
    label_max_abs_diff_by_field = {
        name: float(np.abs(decoded_got[name].mean - decoded_want[name].mean).max())
        for name in decoded_got
    }
    label_max_abs_diff = max(label_max_abs_diff_by_field.values())

    passed = (
        scale_normalized_max < tolerances.scale_normalized_max
        and median_abs_diff < tolerances.median_abs_diff
        and label_max_abs_diff < tolerances.label_max_abs_diff
    )
    return {
        "slug": slug,
        "golden": str(golden),
        "framework_version": meta.get("keras"),
        "n_rows": int(x0.shape[0]),
        "n_members": int(got.shape[0]),
        "max_abs_diff": float(diff.max()),
        "max_abs_diff_float64": float(diff.max()),
        "max_abs_diff_float32": float(diff32.max()),
        "median_abs_diff": median_abs_diff,
        "max_abs_diff_by_column": [float(c) for c in diff_by_column],
        # The evidence behind the docstring's reasoning: our own two dtypes'
        # disagreement, reported beside the raw max/median above so the
        # pairing (same magnitude as the TensorFlow comparison) is on the
        # record in every report a run writes, not only in prose.
        "self_max_abs_diff_float64_vs_float32": float(self_diff.max()),
        "self_median_abs_diff_float64_vs_float32": float(np.median(self_diff)),
        "scale_normalized_max_abs_diff_by_column": [
            float(c) for c in scale_normalized_by_column
        ],
        "scale_normalized_max_abs_diff": scale_normalized_max,
        "label_max_abs_diff_by_field": label_max_abs_diff_by_field,
        "label_max_abs_diff": label_max_abs_diff,
        "tolerances": asdict(tolerances),
        "passed": bool(passed),
    }
