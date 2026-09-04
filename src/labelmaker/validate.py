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
from pathlib import Path

import numpy as np

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


def write_report(paths: Paths, slug: str, name: str, payload: dict) -> Path:
    """Write one validation report, atomically."""
    out_dir = paths.validation / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    with atomic_path(path) as tmp:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return path


def adapter_fidelity(slug: str, golden: Path = GOLDEN, *, tol: float = 1e-4) -> dict:
    """Compare our evaluator against the framework's frozen outputs.

    The golden file holds the reference inputs and the outputs a real Keras
    load produced for the same weights, so this runs with no framework
    installed. A failure here invalidates every label the model has ever
    written, which is why it is a hard pass/fail rather than a metric.

    The weights loaded here come from the golden file's own `meta["models"]`,
    not from `paths.models` - deliberately: the comparison is only meaningful
    against the exact weights TensorFlow saw when the golden file was made,
    so `paths` is otherwise unused by this function.

    `tol` is 1e-4, not 1e-5: on the real tearing ensemble against the
    upstream reference shots, `max_abs_diff` measures ~5.6e-5. This is not
    evaluator imprecision to chase down - `load_graph` reads every weight as
    float64 and `KerasGraph.__call__` upcasts every input to float64 before
    the first layer runs, so this evaluator never computes in float32 at
    all - it is TensorFlow's own float32 forward pass rounding, accumulated
    across the Conv1D/BatchNorm/Dense chain onto outputs of order 10-20.
    That is two orders of magnitude below the ~1e-2 a genuine implementation
    error would produce, so 1e-4 stays a tight, bug-catching gate while
    tolerating the framework's own arithmetic.
    """
    adapter = registry.load_adapter(slug)
    with np.load(golden, allow_pickle=False) as z:
        x0, x1, want = z["x0"], z["x1"], z["members"]
        meta = json.loads(str(z["meta"]))
    graphs = load_ensemble(Path(meta["models"]) / name for name in adapter.artifacts)
    got = predict_members(graphs, [x0, x1])  # positional; see predict_members
    diff = np.abs(got - want)
    return {
        "slug": slug,
        "golden": str(golden),
        "framework_version": meta.get("keras"),
        "n_rows": int(x0.shape[0]),
        "n_members": int(got.shape[0]),
        "max_abs_diff": float(diff.max()),
        "median_abs_diff": float(np.median(diff)),
        "max_abs_diff_by_column": [float(c) for c in diff.max(axis=(0, 1))],
        "tolerance": tol,
        "passed": bool(diff.max() < tol),
    }
