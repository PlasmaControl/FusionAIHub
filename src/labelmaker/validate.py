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
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .catalog import TM_ARCHIVE
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
    """The four measured gates `adapter_fidelity` checks, in place of one bare `tol`.

    A single absolute number cannot discriminate a real bug from float32
    noise once the output columns sit at different scales (`betan` ~3.89,
    the tearing logit ~20.7) - see `adapter_fidelity`'s docstring for the
    Task 14b measurement each default is derived from.
    """

    #: `max_abs_diff` per column, divided by that column's own scale
    #: (`max(|golden output|)`), gated at the plan's original 1e-5 - applied
    #: to a normalized quantity so it means what the plan intended. Measured
    #: 8.79e-07 (`betan`) and 2.71e-06 (tearing logit) - 3.7x headroom on the
    #: binding (larger) column.
    scale_normalized_max: float = 1e-5
    #: `median(|diff|)` over every member/row/column, gated at 1e-6. The
    #: strongest discriminator against a semantic error - a wrong epsilon or
    #: a transposed kernel raises the median, not merely the tail - so this
    #: is the gate most likely to catch a real bug. Measured 7.65e-07, 1.31x
    #: headroom - the tightest margin of the four; a torch or BLAS upgrade
    #: is the thing that would move it.
    median_abs_diff: float = 1e-6
    #: max abs diff of the published, post-activation, post-ensemble-mean
    #: label MEAN (e.g. `tm_prob`), gated at 1e-5. `Decoded.lo`/`.hi` - the
    #: ensemble min/max the label store also publishes as the label's
    #: spread - are not checked by this gate. Measured 1.24e-06, 8.1x
    #: headroom.
    label_max_abs_diff: float = 1e-5
    #: the raw, un-normalized `max_abs_diff` (see `adapter_fidelity`'s
    #: return value) - restored here rather than left to the tests alone,
    #: because the three gates above can all pass while this one does not:
    #: an error confined to a handful of rows moves the max without moving
    #: the median of 16,730 values. Measured 5.60e-05.
    max_abs_diff: float = 1e-4


# median_abs_diff and label_max_abs_diff, unlike scale_normalized_max, are
# absolute numbers in output units, not scale-normalized. Applied here, by
# slug, as one shared default, they would spuriously fail a future model
# whose outputs sit an order of magnitude larger than this one's (~1-20) -
# the same defect this dataclass was written to fix in the old bare `tol`.
# Not building a per-model tolerance mechanism for that now: a future model
# needs its own FidelityTolerances passed explicitly until one exists.
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
    semantic error - is supported instead by a measurement Task 14b did not
    make: comparing this evaluator against **itself**, float64 load vs.
    float32 load, both on the same weights and inputs (`self_max_abs_diff`
    below). That self-disagreement measures ~5.48e-5 - the same magnitude as
    the ~5.60e-5 disagreement against TensorFlow. This does not exclude a
    semantic bug; it bounds one. The self-comparison is a noise floor: this
    graph amplifies float32-level perturbations to ~5.5e-5 on the logit, so
    a float32 reference *cannot resolve* any semantic error smaller than
    that. It still fully licenses the pass verdict below - a bug large
    enough to matter would clear that floor and show up in
    `median_abs_diff`, the gate built to catch exactly that kind of error
    (see below) - but "cannot resolve anything smaller" is a bound, not
    proof that nothing smaller is there.

    Four further facts corroborate that the residual is float32-scale
    rounding, all visible in the returned dict:

    - the per-member max ranges 2.71e-05 to 5.60e-05 across the ten members
      (`max_abs_diff_by_member` below) - a 2.07x spread, not a uniform one,
      but with no single outlier the way a weight-loading bug on one member
      would produce;
    - `median_abs_diff` is ~7.65e-7 on outputs of magnitude 1-20.7 - float32
      epsilon at that scale, exactly;
    - the mean *signed* difference per column (`mean_signed_diff_by_column`
      below) is `[-2.17e-07, -8.96e-07]`. On the logit column this is NOT
      the zero-mean rounding it might look like next to a max of 5.6e-5:
      57.3% of differences are negative, `mean|diff|` is 5.47e-06, and for
      16,730 i.i.d. zero-mean samples of that magnitude the mean would be
      ~4.23e-08 - the measured -8.96e-07 is 21x that. The bias is real, and
      it is **magnitude-proportional relative rounding**: `mean(signed_diff
      / logit)` over rows with `|logit| > 1` is 1.28e-07 - float32-epsilon
      scale - on a logit distribution whose own mean is -6.47. A mostly-
      negative quantity, rounded relatively, yields a mostly-negative signed
      difference; that is a property of representing the quantity in
      float32, not a directional error in the graph;
    - the same bias appears in the evaluator's own float64-vs-float32
      self-comparison, which involves no TensorFlow at all: mean signed diff
      -2.91e-07, 54.7% negative, 8x its own zero-mean expectation. Seeing it
      there too is what shows this is a property of float32 arithmetic on
      this graph, not a framework difference.

    Layer-by-layer tracing (see `models/runners/keras_h5.py`'s module
    docstring) places the noise on the ensemble's one unbounded
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
    (tearing logit), 3.7x headroom on the binding column. Four gates now
    replace the one number, each catching something the others do not:

    - `scale_normalized_max` (1e-5): the plan's original tolerance, applied
      per-column so a large-magnitude column cannot hide behind a small one.
    - `median_abs_diff` (1e-6): the strongest discriminator against a
      semantic error - a wrong epsilon or a transposed kernel would raise
      the bulk of the distribution, not merely its tail, so this is the
      gate most likely to actually catch a bug. Unaffected by tail outliers
      (not by scale - it is an absolute number in output units; see the
      comment on `DEFAULT_TOLERANCES` for the limit that implies). Measured
      7.65e-07, 1.31x headroom - the tightest margin of the four.
    - `label_max_abs_diff` (1e-5): the max abs difference in the published,
      post-activation, post-ensemble-mean label MEAN (`tm_prob = sigmoid(mean
      over members of the logit)`, per `OutputSpec.decode`) - units anyone
      downstream reads. Measured ~1.24e-06, 8.1x headroom: an 8.2e-5-wide
      logit disagreement survives ensembling and the sigmoid's compression
      to a barely-there ~1.2e-6 change in a reported probability. This gate
      checks `Decoded.mean` only; `Decoded.lo`/`.hi` (the ensemble min/max
      the label store also publishes as the label's spread) are not covered.
    - `max_abs_diff` (1e-4): the raw bound the other three replaced,
      restored rather than left to the tests alone. On the logit column
      (scale 20.686) `scale_normalized_max`'s 1e-5 admits an absolute error
      up to 2.07e-04 - twice as loose as this bound. The median gate is the
      binding one for an error affecting the bulk of rows, but an error
      confined to a handful of rows (a `'same'`-padding edge case at an
      unusual length, a saturation path) moves the max without moving the
      median of 16,730 values, so it could pass all three scale-aware gates;
      this one still catches it. Measured 5.60e-05.

    `passed` is the conjunction of all four. The float32 counterpart of
    `max_abs_diff` is still computed and returned for documentation,
    alongside the float64-vs-float32 self-disagreement that is the actual
    evidence behind this docstring's reasoning - that pairing is written
    into whatever report calls `write_report` on this function's return
    value (there is no dedicated `validate` stage in `run.py` yet - a later
    task adds one), not only kept here in prose.
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

    # The evidence behind the docstring's "not a directional bias" fix: the
    # per-member spread (not perfectly uniform) and the signed mean (not
    # zero-mean) that the reasoning above depends on being auditable from
    # this function's return value, not only from prose.
    max_abs_diff_by_member = diff.max(axis=(1, 2))
    mean_signed_diff_by_column = (got - want).mean(axis=(0, 1))

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

    max_abs_diff = float(diff.max())

    passed = (
        scale_normalized_max < tolerances.scale_normalized_max
        and median_abs_diff < tolerances.median_abs_diff
        and label_max_abs_diff < tolerances.label_max_abs_diff
        and max_abs_diff < tolerances.max_abs_diff
    )
    return {
        "slug": slug,
        "golden": str(golden),
        "framework_version": meta.get("keras"),
        "n_rows": int(x0.shape[0]),
        "n_members": int(got.shape[0]),
        "max_abs_diff": max_abs_diff,
        "max_abs_diff_float32": float(diff32.max()),
        "median_abs_diff": median_abs_diff,
        "max_abs_diff_by_column": [float(c) for c in diff_by_column],
        "max_abs_diff_by_member": [float(m) for m in max_abs_diff_by_member],
        "mean_signed_diff_by_column": [float(c) for c in mean_signed_diff_by_column],
        # The evidence behind the docstring's reasoning: our own two dtypes'
        # disagreement, reported beside the raw max/median above so the
        # pairing (same magnitude as the TensorFlow comparison) is on the
        # record in whatever report calls write_report on this dict, not
        # only in prose.
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


#: Columns of the archived x0 that are bit-identical to the archive store,
#: measured on shot 185945: bt, ip, tritop, tribot, gapin. They are what
#: makes the archived rows addressable - the arrays carry no timestamps.
MATCH_COLUMNS = (0, 1, 6, 7, 8)

#: The `spec.scalar_fields[i].model_name` this module expects at each of
#: `MATCH_COLUMNS`, for `d3d_tearing_onset_cnn1d` - the only model with a
#: training archive (I10). `reconstruction_fidelity` asserts this holds
#: before doing any per-shot work: a different slug's scalar order would
#: otherwise compare the wrong columns and report a plausible-looking
#: distance instead of failing loudly.
_EXPECTED_MATCH_COLUMN_NAMES = ("bt", "ip", "tritop_EFIT01", "tribot_EFIT01", "gapin_EFIT01")


@lru_cache(maxsize=1)
def _archive_shot_ids(archive: Path) -> np.ndarray:
    """`z.npy`, loaded and cast to int64 once per process.

    `archive_rows` used to re-open and fully materialize this 639,555-row
    array (a memmap forced into memory by `np.asarray`, then copied again by
    `.astype`) on every call - once per shot in a `reconstruction_fidelity`
    run over as many as a few hundred shots - for what is otherwise a
    constant lookup table.
    """
    z = np.load(archive / "z.npy", mmap_mode="r")
    return np.asarray(z).astype(np.int64)


def archive_rows(shot: int, archive: Path = TM_ARCHIVE) -> dict | None:
    """The archived training rows for one shot, or None if it has none.

    `z.npy` holds the shot of every row, so this is a mask, not a lookup.
    The arrays are memory-mapped: x1 alone is 422 MB.
    """
    rows = np.where(_archive_shot_ids(archive) == int(shot))[0]
    if rows.size == 0:
        return None
    out = {}
    for name in ("x0", "x1", "y"):
        arr = np.load(archive / f"{name}.npy", mmap_mode="r")
        out[name] = np.asarray(arr[rows], dtype=np.float64)
    out["rows"] = rows
    return out


def match_rows(archived_x0: np.ndarray, built, *, tol: float = 1e-3) -> dict:
    """Map each archived row to the timestep of our own inputs.

    Nearest neighbour on the columns that are bit-identical between the two
    sources, each scaled by its own spread so no single column dominates.
    The upstream filter dropped rows, so the mapping is a strictly increasing
    subsequence; `monotonic` is the check that it really is one, and a large
    `median_distance` means the mapping is not to be trusted at all.

    An archived row can be unmatchable: if one of the five match columns is
    missing for this shot (a per-shot gap `resolve_archive` documents), every
    distance from that row to every one of our timesteps is NaN. Bare
    `nanargmin`/`nanmin` raise `ValueError: All-NaN slice encountered` on
    such a row, which would abort the whole shot instead of reporting one
    unmatched row - guarded below by masking row-by-row, and the reductions
    that follow run only over the rows that did match. `passed` therefore
    requires every archived row to have matched: a partial mapping is not a
    mapping to be trusted either.

    The returned dict names each count for what it actually counts (I4):
    `n_archived_rows` is how many rows were presented for matching,
    `n_matched` is how many of those actually found a finite-distance match,
    and `n_unique_matched` is how many *distinct* timesteps those matches
    landed on - so `n_matched > n_unique_matched` means two archived rows
    collided onto the same one. `fail_reason` names which of the three ways
    this can fail actually happened, rather than making a caller infer it
    from a bare median.
    """
    ours = np.asarray(built.scalars, dtype=np.float64)[:, MATCH_COLUMNS]
    theirs = np.asarray(archived_x0, dtype=np.float64)[:, MATCH_COLUMNS]
    scale = np.nanstd(ours, axis=0)
    if not np.all(scale > 0):
        raise ValueError(
            f"match columns are constant in our inputs (std={scale}); "
            "cannot align without variation"
        )
    d = np.linalg.norm(
        (theirs[:, None, :] - ours[None, :, :]) / scale, axis=2
    )
    n = theirs.shape[0]
    index = np.full(n, -1, dtype=np.int64)
    distance = np.full(n, np.nan, dtype=np.float64)
    has_finite = np.isfinite(d).any(axis=1)
    if np.any(has_finite):
        rows = np.flatnonzero(has_finite)
        index[rows] = np.nanargmin(d[rows], axis=1)
        distance[rows] = np.nanmin(d[rows], axis=1)
    # Reductions on the already-filtered, all-finite subset: `np.median` and
    # `np.max` never see a NaN here, so neither can raise numpy's "All-NaN
    # slice encountered" warning - which `-W error` promotes to an exception,
    # exactly like `nanmean`'s "Mean of empty slice" (see
    # `timebase.window_mean`'s docstring for the same trap).
    finite_distance = distance[has_finite]
    median = float(np.median(finite_distance)) if finite_distance.size else float("nan")
    max_distance = float(np.max(finite_distance)) if finite_distance.size else float("nan")
    matched_index = index[has_finite]
    n_matched = int(matched_index.size)
    n_unique_matched = int(np.unique(matched_index).size) if n_matched else 0
    monotonic = bool(np.all(np.diff(matched_index) > 0)) if matched_index.size > 1 else True
    passed = bool(n_matched == n and median < tol and n_unique_matched == n)
    fail_reason = None
    if not passed:
        if n_matched != n:
            fail_reason = (
                f"{n - n_matched} of {n} archived rows unmatched "
                "(no finite distance to any of our timesteps)"
            )
        elif n_unique_matched != n:
            fail_reason = (
                f"{n - n_unique_matched} of {n} matches collided onto a "
                "timestep another archived row also matched"
            )
        else:
            fail_reason = f"median distance {median:.3g} exceeds tol {tol:.3g}"
    return {
        "index": index,
        "distance": distance,
        "median_distance": median,
        "max_distance": max_distance,
        "monotonic": monotonic,
        "n_archived_rows": int(n),
        "n_matched": n_matched,
        "n_unique_matched": n_unique_matched,
        "fail_reason": fail_reason,
        "passed": passed,
    }


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: `max|ECDF_a(x) - ECDF_b(x)|`.

    Replaces `scipy.stats.ks_2samp(a, b).statistic` (I2). scipy is used
    nowhere else in this module, and importing it here was the reason
    `import labelmaker.validate` could crash outside pytest: this module
    loads torch at module scope (needed by `adapter_fidelity`), torch's
    bundled `libstdc++` shadows the newer system one, and scipy's compiled
    `_ckdtree` extension then fails with `ImportError: version
    'GLIBCXX_3.4.29' not found` - reliably, and only when torch imports
    first. No import-order fix in this module is sufficient (a later task's
    `--stage all` loads torch via the `infer` stage before `validate` is
    imported at all), so the fix is to need no scipy import at runtime.

    Both samples are sorted and the right-continuous ECDF of each is
    evaluated at every value in the pooled sample via `searchsorted`; the
    statistic is the largest gap between the two. This is the same
    definition scipy's `statistic` uses (verified against it as an exact
    oracle in `tests/labelmaker/test_ks_statistic.py`, which does import
    scipy - that import succeeds under pytest, since some test-collection
    plugin loads a compatible `libstdc++` before torch does, and never
    happens at runtime here since scipy is no longer imported outside that
    one test file).
    """
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    pooled = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, pooled, side="right") / a.size
    cdf_b = np.searchsorted(b, pooled, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _stats(ours: np.ndarray, theirs: np.ndarray) -> dict:
    """Agreement between two samples of the same quantity.

    Both branches return the same key set - `mean_ours`/`mean_archive`
    included - even when there are too few points to compute them (`None`
    rather than absent). These dicts are serialized to JSON and read by a
    later stage; a shape that varies with `n` would make every consumer
    write a `.get` guard instead of a plain lookup.
    """
    a = np.asarray(ours, dtype=np.float64).ravel()
    b = np.asarray(theirs, dtype=np.float64).ravel()
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 10:
        return {
            "n": int(good.sum()),
            "median_rel": None,
            "corr": None,
            "ks": None,
            "mean_ours": None,
            "mean_archive": None,
        }
    a, b = a[good], b[good]
    rel = np.median(np.abs(a - b) / (np.abs(b) + 1e-12))
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None
    return {
        "n": int(good.size),
        "median_rel": float(rel),
        "corr": corr,
        "ks": _ks_statistic(a, b),
        "mean_ours": float(a.mean()),
        "mean_archive": float(b.mean()),
    }


def reconstruction_fidelity(
    slug: str,
    shots,
    paths: Paths,
    *,
    archive: Path = TM_ARCHIVE,
) -> dict:
    """Price every substitution, per feature, against the training rows.

    C1: every shot is isolated. The body below reads an archive file, an
    HDF5 feature file, calls `InputSpec.build` and `match_rows` - any of
    which can raise on real data (a truncated feature file, a per-shot
    column `resolve_archive` did not carry, `nan_policy="zero"` turning an
    absent feature into an all-zero column that makes `match_rows`' own
    variance guard raise `ValueError`). None of that is allowed to cost the
    rest of the run: one bad shot goes to `skipped` with its cause, exactly
    like the two `continue`s already in this loop for a missing archive row
    or feature file.
    """
    from .features import namespace as ns
    from .features.store import missing_names, present, read_feature
    from .models.base import sample_by_resolver

    # Materialized once, up front: `shots` is frequently a generator
    # (`catalog.overlap_shots`, or a comprehension over it), and the loop
    # below consumes it. Computing `len(shots)` after the loop - as an
    # earlier draft of this function did - reports 0 for any such caller.
    shots = list(shots)
    adapter = registry.load_adapter(slug)
    spec = adapter.input_spec
    names_0d = [f.model_name for f in spec.scalar_fields]
    ech_rho_field = next((f for f in spec.fields if f.canonical == "ech_rho"), None)

    # I10: MATCH_COLUMNS is specific to d3d_tearing_onset_cnn1d's scalar
    # column order. Checked once, before any per-shot work, so a different
    # slug fails loudly here instead of silently comparing the wrong
    # columns and reporting a plausible-looking distance. This is a
    # programming error, not a per-shot failure, so it is not caught by the
    # per-shot guard below.
    match_names = tuple(names_0d[i] for i in MATCH_COLUMNS)
    if match_names != _EXPECTED_MATCH_COLUMN_NAMES:
        raise ValueError(
            f"{slug}: MATCH_COLUMNS {MATCH_COLUMNS} indexes {match_names}, "
            f"not the expected {_EXPECTED_MATCH_COLUMN_NAMES} "
            "(bt, ip, tritop, tribot, gapin); reconstruction_fidelity's row "
            "alignment is specific to d3d_tearing_onset_cnn1d's scalar "
            "column order and would silently compare the wrong columns for "
            "any other slug"
        )

    pooled: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    n_pooled_shots: dict[str, int] = {}
    match_info: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    incomplete: dict[str, dict] = {}
    resolvers: dict[str, set] = {}
    used: list[int] = []
    ech_conflicts = 0
    ech_rows = 0
    n_valid_pooled = 0
    n_invalid_pooled = 0

    for shot in shots:
        try:
            got = archive_rows(shot, archive)
            if got is None:
                skipped[str(shot)] = "no archived rows"
                continue
            fpath = paths.features_file(shot)
            if not fpath.exists():
                skipped[str(shot)] = "no feature file"
                continue
            stored = present(fpath)
            features = {
                name: read_feature(fpath, name)
                for name in spec.canonical_names
                if name in stored
            }
            built = spec.build(features, ns.GRID_S)
            for canonical, source in built.resolvers.items():
                resolvers.setdefault(canonical, set()).add(source)
            info = match_rows(got["x0"], built)
            match_info[str(shot)] = {
                k: v for k, v in info.items() if k not in ("index", "distance")
            }
            if not info["passed"]:
                skipped[str(shot)] = f"match rejected: {info['fail_reason']}"
                continue
            idx = info["index"]
            used.append(int(shot))
            # I3: a canonical in `built.missing` was never resolved for this
            # shot, so `build`'s `nan_policy="zero"` fill is a zero-filled
            # placeholder, not a reading - pooling it against the real
            # archive column would price a substitution that never
            # happened. `pinj_total` (corpus-only) on a shot with no corpus
            # file is the common case, not an edge case.
            for j, f in enumerate(spec.scalar_fields):
                if f.canonical in built.missing:
                    continue
                pooled.setdefault(f.model_name, []).append(
                    (built.scalars[idx, j], got["x0"][:, j])
                )
                n_pooled_shots[f.model_name] = n_pooled_shots.get(f.model_name, 0) + 1
            for j, f in enumerate(spec.profile_fields):
                if f.canonical in built.missing:
                    continue
                pooled.setdefault(f.model_name, []).append(
                    (built.profiles[idx, :, j], got["x1"][:, :, j])
                )
                n_pooled_shots[f.model_name] = n_pooled_shots.get(f.model_name, 0) + 1
            # Part 3: `pooled` prices every matched row regardless of
            # `built.valid`, so a row for which labelmaker would publish no
            # label is priced alongside one it would. Not filtered out here
            # (that would need re-deriving each pair's row count per
            # feature), but counted, so a reader can see how much of the
            # price above belongs to rows that are never actually published.
            valid_at_idx = np.asarray(built.valid)[idx]
            n_valid_pooled += int(valid_at_idx.sum())
            n_invalid_pooled += int((~valid_at_idx).sum())
            # Diagnostic for the zero-filled ECH deposition location: how
            # often is the location unknown while power is actually being
            # injected? Read the field's own declared lag rather than
            # assuming "t+dt": a future model spec could carry `ech_rho` at
            # plain "t", and hardcoding the offset would silently report a
            # conflict count for a time the model never actually sees.
            # Sampled through `sample_by_resolver` (I5) rather than a bare
            # `sample_at`, so this diagnostic and `InputSpec.build` cannot
            # drift on what "the right way to read this resolver" means -
            # today `ech_rho` is archive-only, so this is a no-op change in
            # behaviour, but it stays correct if that ever stops being true.
            if "ech_power_total" in features and "ech_rho" in features and ech_rho_field:
                raw = features["ech_rho"]
                t = ns.GRID_S + spec.dt_s if ech_rho_field.lag == "t+dt" else ns.GRID_S
                ech_resolver = str(raw.attrs.get("resolver", "unknown"))
                rho = np.asarray(
                    sample_by_resolver(raw.x, raw.y, t, ech_resolver, spec.dt_s)
                ).ravel()[idx]
                power = built.scalars[idx, names_0d.index("ech_pwr_total")]
                ech_rows += int(power.size)
                ech_conflicts += int(((power > 0) & ~np.isfinite(rho)).sum())
            incomplete[str(shot)] = missing_names(fpath)
        except Exception as exc:  # noqa: BLE001 - per-shot isolation, see docstring
            skipped[str(shot)] = f"{type(exc).__name__}: {exc}"
            continue

    per_feature = {}
    for name, pairs in pooled.items():
        stats = _stats(
            np.concatenate([np.asarray(a).ravel() for a, _ in pairs]),
            np.concatenate([np.asarray(b).ravel() for _, b in pairs]),
        )
        # I3: how many shots (not rows) actually contributed to this
        # feature's pooled comparison, so a reader can tell a feature priced
        # over the whole `n_shots_used` from one that is corpus-only and
        # only ever resolved on a handful of them.
        stats["n_shots"] = n_pooled_shots[name]
        per_feature[name] = stats
    return {
        "slug": slug,
        "n_shots_requested": len(shots),
        "n_shots_used": len(used),
        "shots_used": used,
        "skipped": skipped,
        "incomplete_features": {k: v for k, v in incomplete.items() if v},
        "match": match_info,
        "per_feature": per_feature,
        "resolvers": {k: sorted(v) for k, v in resolvers.items()},
        # Part 3: rows priced above that `built.valid` says are never
        # actually published, versus rows that are - see the comment where
        # these are accumulated.
        "pooled_row_validity": {
            "valid": n_valid_pooled,
            "invalid": n_invalid_pooled,
        },
        "ech_location_unknown_while_powered": {
            "rows": ech_rows,
            "conflicts": ech_conflicts,
        },
    }
