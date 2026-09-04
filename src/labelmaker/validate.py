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


@dataclass
class _ShotMatch:
    """One shot's archive-vs-reconstruction alignment, or why it has none.

    `built` and `info` are populated as far as the pipeline actually got,
    even when `skip_reason` is set - not only on success. A shot that fails
    `match_rows`' variance/median gate still had `spec.build` run on it, and
    `built.missing`/`built.resolvers` are exactly the diagnosis task-16's
    review (I3) found being computed and thrown away at skip time instead of
    reported. `got` is the archived rows, present unless the shot has none
    in the archive at all.
    """

    got: dict | None = None
    built: object | None = None
    info: dict | None = None
    features: dict | None = None
    skip_reason: str | None = None


def _matched_shot(shot: int, spec, paths: Paths, archive: Path) -> _ShotMatch:
    """Archive rows aligned to our own reconstructed features, for one shot.

    M10 (task-16 review): `reconstruction_fidelity` and `label_quality`
    independently hand-rolled the same four calls - `archive_rows`, the
    feature-file existence check, `spec.build`, `match_rows` - which is how
    the two could in principle drift apart on how a shot gets skipped. This
    returns a `_ShotMatch` rather than raising on the failure paths, because
    a caller needs `built` for I3's per-shot diagnosis even when the shot is
    about to be skipped - a bare exception would throw that information away
    a second time.
    """
    from .features import namespace as ns
    from .features.store import present, read_feature

    got = archive_rows(shot, archive)
    if got is None:
        return _ShotMatch(skip_reason="no archived rows")
    fpath = paths.features_file(shot)
    if not fpath.exists():
        return _ShotMatch(got=got, skip_reason="no feature file")
    stored = present(fpath)
    features = {
        name: read_feature(fpath, name)
        for name in spec.canonical_names
        if name in stored
    }
    built = spec.build(features, ns.GRID_S)
    info = match_rows(got["x0"], built)
    if not info["passed"]:
        return _ShotMatch(
            got=got, built=built, info=info, features=features,
            skip_reason=f"match rejected: {info['fail_reason']}",
        )
    return _ShotMatch(got=got, built=built, info=info, features=features)


def _skip_category(reason: str) -> str:
    """Collapse a skip reason to a bucket a histogram can actually count (I3).

    Skip reasons routinely carry per-shot numeric detail - a numpy array
    repr inside a caught `ValueError`'s message (`"...constant in our inputs
    (std=[0. 0. 0. 0. 0.])"`), or an exact row count in `match_rows`' own
    `fail_reason` - which makes the raw string unique to nearly every shot.
    That is exactly the failure mode a histogram exists to avoid: 69 shots
    sharing one real cause (a dead fdp resolver, C2) would otherwise render
    as up to 69 distinct entries and hide the pattern entirely.
    """
    if reason in ("no archived rows", "no feature file"):
        return reason
    if reason.startswith("match rejected:"):
        fail = reason.split(":", 1)[1]
        if "unmatched" in fail:
            return "match rejected: rows unmatched"
        if "collided" in fail:
            return "match rejected: rows collided onto the same timestep"
        if "median distance" in fail:
            return "match rejected: median distance exceeds tolerance"
        return "match rejected: other"
    # A caught exception's `f"{type(exc).__name__}: {exc}"` - keep the
    # exception type and the message's fixed prefix (up to the first
    # parenthesis, where a numpy repr or a count usually starts), and drop
    # the rest.
    exc_type, _, detail = reason.partition(":")
    prefix = detail.split("(")[0].strip()
    return f"{exc_type}: {prefix}" if prefix else exc_type


def _skip_report(skipped: dict[str, str], resolvers: dict[str, set]) -> dict:
    """I3: a histogram of *why* shots were skipped, plus a warning when the
    run looks broken rather than merely gappy.

    Two triggers, either of which would have surfaced this task's own defect
    (a dead fdp scaling path, C2) directly in this JSON instead of needing a
    separate investigation: one cause accounting for most of the skips, or a
    whole feature source contributing nothing across every shot the run
    touched.
    """
    histogram: dict[str, int] = {}
    for reason in skipped.values():
        cat = _skip_category(reason)
        histogram[cat] = histogram.get(cat, 0) + 1
    warnings: list[str] = []
    if histogram:
        total = sum(histogram.values())
        cause, count = max(histogram.items(), key=lambda kv: kv[1])
        if total >= 5 and count / total >= 0.5:
            warnings.append(
                f"{count} of {total} skips ({count / total:.0%}) share one "
                f"cause: {cause!r} - this looks systemic, not scattered "
                "per-shot data gaps; see task-16 review findings I3/C2"
            )
    from .features import namespace as ns

    served = {s for sources in resolvers.values() for s in sources}
    for source in ns.SOURCES:
        if source not in served:
            warnings.append(
                f"no shot resolved any feature through {source!r} this run - "
                "that source served nothing at all, which is either an empty "
                "request or a systemic failure (this is exactly the shape "
                "C2's dead fdp scaling path took)"
            )
    return {"histogram": dict(sorted(histogram.items())), "warnings": warnings}


def _shot_budget(timeout_s: int | None):
    """M8 (task-16 review): the per-shot SIGALRM guard, opt-in.

    `run.py`'s module docstring promises "one try/except and one SIGALRM
    timeout per shot" for every stage, but `reconstruction_fidelity` and
    `label_quality` used to have only the try/except half - the exact
    hung-read mode the rest of the package guards against (IGNITE measured
    ~56 of 3,000 corpus shots hanging on reads), left uncovered here because
    `validate` runs single-threaded in the parent rather than through
    `run.py`'s worker pool. `run.py`'s validate branch passes `--timeout`
    through; deferred-imports `run.time_limit` rather than importing it at
    module scope, since `run.py` only imports `validate` lazily (inside
    `main`'s stage branch) and this keeps that lazy-import direction the
    only one that exists between the two modules. A bare `contextlib.
    nullcontext()` when `timeout_s` is `None` preserves every existing
    caller (tests, notebooks) that has no `run.py` timeout to pass.
    """
    if timeout_s is None:
        from contextlib import nullcontext

        return nullcontext()
    from .run import time_limit

    return time_limit(timeout_s)


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: `max|ECDF_a(x) - ECDF_b(x)|`.

    Replaces `scipy.stats.ks_2samp(a, b).statistic` (I2). scipy is used
    nowhere else in this module, and importing it here was the reason
    `import labelmaker.validate` could crash outside pytest - NOT because
    scipy is uniquely cursed, but because it was the module that happened to
    surface a loader-ordering problem shared by every compiled extension in
    this environment that needs a newer GLIBCXX than the system provides.
    This module loads torch at module scope (needed by `adapter_fidelity`),
    and `import torch` binds the SYSTEM `/lib64/libstdc++.so.6` - which
    lacks `GLIBCXX_3.4.29` - ahead of the pixi env's own newer copy unless
    something puts the latter first on the loader's path. With the system
    one bound, scipy's compiled `_ckdtree` extension fails with
    `ImportError: version 'GLIBCXX_3.4.29' not found` - reliably, and only
    when torch imports first. No import-order fix in this module is
    sufficient (a later task's `--stage all` loads torch via the `infer`
    stage before `validate` is imported at all), so the fix here is to need
    no scipy import at runtime.

    The SAME loader-ordering problem is what silently disabled labelmaker's
    entire fdp scaling path (Task 16b, see `features/resolve_fdp.py`'s
    module docstring): `toksearch_d3d`, `fdp` and `pyxrootd` all need the
    same symbol and all failed the same way. That task's fix,
    `pyproject.toml`'s `tool.pixi.feature.fdp` activation table, puts the
    pixi env's own `libstdc++` ahead of the system one via `LD_LIBRARY_PATH`
    for the whole `labelmaker` environment - which also fixes scipy's
    import here, in any context, not just under pytest. This replacement is
    kept anyway: needing no scipy import at runtime is strictly fewer
    moving parts, independent of whether the loader is configured correctly.

    Both samples are sorted and the right-continuous ECDF of each is
    evaluated at every value in the pooled sample via `searchsorted`; the
    statistic is the largest gap between the two. This is the same
    definition scipy's `statistic` uses (verified against it as an exact
    oracle in `tests/labelmaker/test_ks_statistic.py`, which does import
    scipy - now unconditionally importable here, per the previous
    paragraph, but still guarded with `importorskip` so this test degrades
    gracefully in an environment where the activation fix does not apply).
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
    timeout_s: int | None = None,
) -> dict:
    """Price every substitution, per feature, against the training rows.

    C1: every shot is isolated. The body below reads an archive file, an
    HDF5 feature file, calls `InputSpec.build` and `match_rows` (via
    `_matched_shot`, M10) - any of which can raise on real data (a
    truncated feature file, a per-shot column `resolve_archive` did not
    carry, `nan_policy="zero"` turning an absent feature into an all-zero
    column that makes `match_rows`' own variance guard raise `ValueError`).
    None of that is allowed to cost the rest of the run: one bad shot goes
    to `skipped` with its cause, exactly like the `_ShotMatch.skip_reason`
    cases below.

    M8 (task-16 review): `timeout_s`, when given, wraps each shot's body in
    the same SIGALRM guard `run.py`'s `_guarded` applies to the `features`
    and `infer` stages - this function otherwise reads GPFS memmaps and
    HDF5 with nothing but a try/except, the exact hung-read mode the rest of
    the package guards against. `run.py`'s validate branch passes its own
    `--timeout`; a caller with no `run.py` context (a notebook, a test) gets
    no timeout by default, matching every other keyword-optional guard in
    this module.
    """
    from .features import namespace as ns
    from .features.store import missing_names
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
    skip_diagnosis: dict[str, dict] = {}
    incomplete: dict[str, dict] = {}
    resolvers: dict[str, set] = {}
    used: list[int] = []
    ech_conflicts = 0
    ech_rows = 0
    n_valid_pooled = 0
    n_invalid_pooled = 0

    for shot in shots:
        try:
            with _shot_budget(timeout_s):
                # I3 (related): recorded for EVERY shot reached, not only
                # the ones that end up `used` - a shot skipped for a match
                # failure or a missing archive row can still have partial
                # feature misses worth showing, and the old placement (after
                # the `passed` check) meant a skipped shot's misses appeared
                # in neither this report nor `skipped`.
                fpath = paths.features_file(shot)
                incomplete[str(shot)] = missing_names(fpath)

                m = _matched_shot(shot, spec, paths, archive)
                if m.built is not None:
                    for canonical, source in m.built.resolvers.items():
                        resolvers.setdefault(canonical, set()).add(source)
                if m.info is not None:
                    match_info[str(shot)] = {
                        k: v for k, v in m.info.items() if k not in ("index", "distance")
                    }
                if m.skip_reason:
                    skipped[str(shot)] = m.skip_reason
                    # I3: the diagnosis that used to be in scope and thrown
                    # away at skip time - `built.missing`/`built.resolvers`
                    # when a build was reached, plus the per-feature miss
                    # causes `store.missing_names` already tracks.
                    skip_diagnosis[str(shot)] = {
                        "missing_features": (
                            sorted(m.built.missing) if m.built is not None else []
                        ),
                        "resolvers": (
                            {k: sorted(v) for k, v in m.built.resolvers.items()}
                            if m.built is not None else {}
                        ),
                        "feature_misses": incomplete[str(shot)],
                    }
                    continue
                got, built, features = m.got, m.built, m.features
                idx = m.info["index"]
                used.append(int(shot))
                # I3: a canonical in `built.missing` was never resolved for
                # this shot, so `build`'s `nan_policy="zero"` fill is a
                # zero-filled placeholder, not a reading - pooling it against
                # the real archive column would price a substitution that
                # never happened. `pinj_total` (corpus-only) on a shot with
                # no corpus file is the common case, not an edge case.
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
                # `built.valid`, so a row for which labelmaker would publish
                # no label is priced alongside one it would. Not filtered out
                # here (that would need re-deriving each pair's row count per
                # feature), but counted, so a reader can see how much of the
                # price above belongs to rows that are never actually
                # published.
                valid_at_idx = np.asarray(built.valid)[idx]
                n_valid_pooled += int(valid_at_idx.sum())
                n_invalid_pooled += int((~valid_at_idx).sum())
                # Diagnostic for the zero-filled ECH deposition location: how
                # often is the location unknown while power is actually being
                # injected? Read the field's own declared lag rather than
                # assuming "t+dt": a future model spec could carry `ech_rho`
                # at plain "t", and hardcoding the offset would silently
                # report a conflict count for a time the model never
                # actually sees. Sampled through `sample_by_resolver` (I5)
                # rather than a bare `sample_at`, so this diagnostic and
                # `InputSpec.build` cannot drift on what "the right way to
                # read this resolver" means - today `ech_rho` is
                # archive-only, so this is a no-op change in behaviour, but
                # it stays correct if that ever stops being true.
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
        except Exception as exc:  # noqa: BLE001 - per-shot isolation, see docstring
            skipped[str(shot)] = f"{type(exc).__name__}: {exc}"
            skip_diagnosis[str(shot)] = {
                "missing_features": [], "resolvers": {},
                "feature_misses": incomplete.get(str(shot), {}),
            }
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
        # I3: the diagnosis that used to be computed and thrown away at skip
        # time, plus the histogram/warning a reader would otherwise have to
        # re-derive by hand from `skipped` (as this task's own review did).
        "skip_diagnosis": skip_diagnosis,
        "skip_reasons": _skip_report(skipped, resolvers),
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


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Mid-rank ranks: average the ordinal ranks within each tie group.

    Replaces `scipy.stats.rankdata(a, method="average")` (task-16 addendum
    item 1), for the same loader-ordering reason `_ks_statistic` above
    replaces `scipy.stats.ks_2samp` - see that docstring for the mechanism
    (`import torch` binding the SYSTEM `libstdc++`, which the pixi env's own
    copy needs to come before) and for why this is not a defect unique to
    scipy: the identical problem silently disabled labelmaker's fdp scaling
    path (Task 16b). `run.py`'s `--stage all` guarantees the failing order
    here, since the `infer` stage loads torch before `validate` is imported
    at all, and no import-order fix inside this module is sufficient, so the
    fix is to need no scipy import at runtime - kept even after Task 16b's
    `pyproject.toml` activation fix, for the independent reduced-dependency
    benefit. Verified against `scipy.stats.rankdata` as an exact oracle in
    `tests/labelmaker/test_label_quality.py`, guarded with `importorskip`
    (now unconditionally importable here after the activation fix, but the
    guard costs nothing and keeps this test honest in an environment where
    it does not apply).

    Mid-ranks - the average of the ordinal ranks tied values would otherwise
    occupy, not the first or last of them - are required, not a stylistic
    choice: `binary_metrics`'s AUROC is the Mann-Whitney rank-sum identity,
    and averaging within a tie group is what makes a model that emits an
    identical probability for two rows of opposite truth score exactly 0.5
    on that pair - see `test_auroc_matches_hand_computed_cases`'s all-tied
    case.
    """
    a = np.asarray(a, dtype=np.float64)
    n = a.size
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    ordinal = np.arange(1, n + 1, dtype=np.float64)
    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    new_group[1:] = sorted_a[1:] != sorted_a[:-1]
    group_id = np.cumsum(new_group) - 1
    group_mean = np.bincount(group_id, weights=ordinal) / np.bincount(group_id)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = group_mean[group_id]
    return ranks


def binary_metrics(prob: np.ndarray, truth: np.ndarray, *, bins: int = 10) -> dict:
    """AUROC, F1 at 0.5, Brier, and a calibration curve.

    AUROC is the Mann-Whitney rank-sum identity over `_rankdata`, not a
    trapezoid over a sampled ROC and not scikit-learn (not in this
    environment, and not worth adding for three metrics) - exact, and needs
    no scipy at runtime either (see `_rankdata`'s docstring). Ties get
    mid-ranks, the correct convention for a model that emits identical
    probabilities for different rows.
    """
    prob = np.asarray(prob, dtype=np.float64).ravel()
    truth = np.asarray(truth, dtype=np.float64).ravel()
    good = np.isfinite(prob) & np.isfinite(truth)
    prob, truth = prob[good], (truth[good] > 0.5)
    n_pos, n_neg = int(truth.sum()), int((~truth).sum())
    out: dict = {
        "n": int(prob.size),
        "n_positive": n_pos,
        "positive_fraction": float(n_pos / prob.size) if prob.size else None,
        "auroc": None,
        "f1_at_0.5": None,
        "precision_at_0.5": None,
        "recall_at_0.5": None,
        "brier": None,
        "ece": None,
        "calibration": [],
    }
    if prob.size == 0 or n_pos == 0 or n_neg == 0:
        return out
    ranks = _rankdata(prob)
    out["auroc"] = float(
        (ranks[truth].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    )
    pred = prob >= 0.5
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    out["f1_at_0.5"] = float(2 * tp / (2 * tp + fp + fn)) if tp or fp or fn else 0.0
    out["precision_at_0.5"] = float(tp / (tp + fp)) if tp + fp else None
    out["recall_at_0.5"] = float(tp / (tp + fn)) if tp + fn else None
    out["brier"] = float(np.mean((prob - truth.astype(np.float64)) ** 2))
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        sel = which == b
        if not sel.any():
            out["calibration"].append(
                {"bin": b, "n": 0, "mean_prob": None, "observed": None}
            )
            continue
        mean_prob = float(prob[sel].mean())
        observed = float(truth[sel].mean())
        out["calibration"].append(
            {"bin": b, "n": int(sel.sum()), "mean_prob": mean_prob, "observed": observed}
        )
        ece += sel.sum() / prob.size * abs(mean_prob - observed)
    out["ece"] = float(ece)
    return out


def regression_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    pred = np.asarray(pred, dtype=np.float64).ravel()
    truth = np.asarray(truth, dtype=np.float64).ravel()
    good = np.isfinite(pred) & np.isfinite(truth)
    pred, truth = pred[good], truth[good]
    if pred.size == 0:
        return {"n": 0, "rmse": None, "bias": None, "corr": None}
    return {
        "n": int(pred.size),
        "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "bias": float(np.mean(pred - truth)),
        "corr": float(np.corrcoef(pred, truth)[0, 1])
        if pred.std() > 0 and truth.std() > 0
        else None,
    }


#: `y.npy`'s own columns (the archived truth): column 1 is `tm_label`,
#: column 0 is `betan`. `OUTPUT_SPEC` independently declares
#: `OutputField("betan", column=0)` / `OutputField("tm_prob", column=1)` for
#: the MODEL's raw output order - a different array that happens to agree
#: only because upstream built both in the same order (task-16 addendum item
#: 5). `label_quality` asserts the two agree once, before any shot is
#: touched, so a future spec or archive change that broke the coincidence
#: fails loudly instead of silently scoring against the wrong truth column.
_TRUTH_COLUMNS = {"tm_prob": 1, "betan": 0}


def _assert_truth_column_shapes(archive: Path) -> None:
    """I6 (task-16 review): close the half of the truth-column coincidence
    the model-side guard above cannot see.

    `label_quality`'s other assertion (`OUTPUT_SPEC.column ==
    _TRUTH_COLUMNS[name]`) catches a change on the MODEL side - a spec whose
    output order no longer matches this module's hardcoded map. It does
    nothing for a change on the ARCHIVE side: if `y.npy` were re-exported
    with its two columns swapped, that assertion still passes (both sides
    still agree the map is `{tm_prob: 1, betan: 0}`) and every metric in
    this module would silently score against the wrong truth column while
    looking entirely plausible - `betan`'s RMSE would be computed against a
    ~8%-positive binary column and `tm_prob`'s AUROC against a continuous
    one, both of which still *run*, just wrongly.

    Closed here with a property of the data itself, not of the mapping:
    measured on 200,000 archive rows, the `tm_prob` truth column is exactly
    binary (`{0.0, 1.0}`, ~8.07% positive) and the `betan` column is
    continuous (198,520 distinct values over 0.002-4.92). A column swap
    flips both properties at once, so this is checked once, before any shot
    is touched, using a small slice of `y.npy` rather than the full
    639,555-row array.
    """
    y = np.load(archive / "y.npy", mmap_mode="r")
    sample = np.asarray(y[: min(200_000, y.shape[0])], dtype=np.float64)
    tm_col = sample[:, _TRUTH_COLUMNS["tm_prob"]]
    betan_col = sample[:, _TRUTH_COLUMNS["betan"]]
    tm_values = set(np.unique(tm_col).tolist())
    if not tm_values <= {0.0, 1.0}:
        raise ValueError(
            f"y.npy column {_TRUTH_COLUMNS['tm_prob']} (expected tm_prob, "
            f"binary) is not binary - found {sorted(tm_values)[:5]}...; this "
            "looks like a column swap in the archive export (task-16 review "
            "finding I6), and every metric in this module would silently "
            "score against the wrong truth column if this were allowed to "
            "proceed"
        )
    if np.unique(betan_col).size < 1000:
        raise ValueError(
            f"y.npy column {_TRUTH_COLUMNS['betan']} (expected betan, "
            f"continuous) has only {np.unique(betan_col).size} distinct "
            "values in a 200,000-row sample - too few to be a continuous "
            "quantity; this looks like a column swap in the archive export "
            "(task-16 review finding I6)"
        )


def _score_field(task: str, pred: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> dict:
    """Score one output field twice: every matched row, and only the rows
    labelmaker's own validity rule would actually publish a label for.

    Task-16 addendum item 6: labelmaker masks invalid rows out of what it
    emits, so a metric computed over every matched row measures something
    the package would never publish - misstating the one number this whole
    task exists to produce. `all_matched` is kept for reference and
    diagnosis only; `published` (the valid-row-only score) is the one
    `label_quality` surfaces for both `archived_inputs_valid` and
    `reconstructed_inputs_valid` (C1, task-16 review), which is what makes
    the two comparable at all: they now share the same row set as well as
    the same truth.
    """
    scorer = binary_metrics if task == "binary" else regression_metrics
    valid = np.asarray(valid, dtype=bool)
    pred = np.asarray(pred)
    truth = np.asarray(truth)
    return {
        "all_matched": scorer(pred, truth),
        "published": scorer(pred[valid], truth[valid]),
        "n_valid": int(valid.sum()),
        "n_invalid": int((~valid).sum()),
    }


def label_quality(
    slug: str,
    shots,
    paths: Paths,
    *,
    archive: Path = TM_ARCHIVE,
    timeout_s: int | None = None,
) -> dict:
    """Score the model twice against the same truth AND the same rows.

    C1 (task-16 review): the two scores this function exists to produce must
    differ in exactly one thing - the input source - not also in which rows
    got counted. So every field is scored four ways, all from one row-
    matched, valid-masked pair underneath:

    - `archived_inputs_all`: the model's ceiling, every matched row.
    - `archived_inputs_valid`: the same ceiling, restricted to the rows
      labelmaker's own validity rule would publish a label for.
    - `reconstructed_inputs_valid`: what labelmaker actually publishes - its
      own reconstructed features, valid rows only.
    - `reconstructed_inputs_all`: reconstructed features over every matched
      row regardless of validity, kept for diagnosis only.

    `reconstruction_penalty` is computed ONLY from the row-matched pair,
    `reconstructed_inputs_valid` minus `archived_inputs_valid` - both sides
    scored over the identical set of rows, so the difference is attributable
    to the input source alone. Scoring the published `reconstructed_inputs`
    against an `archived_inputs` that was never masked (as an earlier
    version of this function did) let the row set change along with the
    input source in one step: measured on the n=31 proof-of-concept run,
    that mistake understated the AUROC penalty by ~34% and inverted the F1
    comparison entirely (a "reconstruction improves F1" reading that was
    purely the row-selection artifact, not a property of the reconstruction
    - see task-16 review finding C1 for the full numbers).

    I4: shots skip exactly when their archive lacks the five `MATCH_COLUMNS`
    features, which correlates with whatever else that shot's archive group
    is missing - so `shots_used` is not a random sample of
    `shots_requested` even at full coverage. `selection_effect_note` states
    this in the report itself, not only here.

    I3: every skip records its diagnosis (`skip_diagnosis`) rather than
    discarding `built.missing`/`built.resolvers`/`store.missing_names` at
    the point they were last in scope, and `skip_reasons` aggregates
    identical causes into a histogram with a warning when one cause (or one
    whole feature source contributing nothing) looks systemic - both would
    have surfaced this task's own C2 defect (a dead fdp scaling path)
    directly in this JSON.

    I6: `y.npy`'s two truth columns are asserted to have the shapes their
    names imply (`_assert_truth_column_shapes`) before any shot is touched,
    closing the half of the truth-column coincidence the model-side
    `_TRUTH_COLUMNS`/`OUTPUT_SPEC.column` assertion below cannot see - a
    swapped archive export, not a swapped spec.

    M8: `timeout_s`, when given, applies the same per-shot SIGALRM guard
    `reconstruction_fidelity` now has - see `_shot_budget`.

    Every shot is isolated (C1 of Task 15's review, matching
    `reconstruction_fidelity`): a missing archive column, a truncated
    feature file, or a `match_rows` variance-guard `ValueError` on a
    per-shot basis costs only that shot, not the run.
    """
    from .features.store import missing_names
    from .models.base import BuiltInputs

    # Materialized up front: `shots` is frequently a generator, and a
    # `len()` taken after the loop below (which consumes it) would report 0.
    shots = list(shots)
    adapter = registry.load_adapter(slug)
    spec = adapter.input_spec

    # I10, shared with `reconstruction_fidelity`: MATCH_COLUMNS indexes
    # d3d_tearing_onset_cnn1d's scalar column order specifically, and this
    # function performs the same row alignment. Checked once, before any
    # per-shot work and before the (potentially slow) weight load below, so
    # a different slug's scalar order fails loudly here instead of silently
    # comparing the wrong columns.
    names_0d = [f.model_name for f in spec.scalar_fields]
    match_names = tuple(names_0d[i] for i in MATCH_COLUMNS)
    if match_names != _EXPECTED_MATCH_COLUMN_NAMES:
        raise ValueError(
            f"{slug}: MATCH_COLUMNS {MATCH_COLUMNS} indexes {match_names}, not "
            f"the expected {_EXPECTED_MATCH_COLUMN_NAMES}; label_quality's row "
            "alignment is specific to d3d_tearing_onset_cnn1d's scalar column "
            "order and would silently compare the wrong columns for any other "
            "slug"
        )

    # Addendum item 5: the coincidence label_quality depends on, asserted
    # once, before any shot is touched. This is the MODEL-side half; I6's
    # `_assert_truth_column_shapes` below is the ARCHIVE-side half neither
    # this nor the addendum's original check can see.
    for f in adapter.output_spec.fields:
        want = _TRUTH_COLUMNS.get(f.name)
        if want is not None and f.column != want:
            raise ValueError(
                f"{slug}: output field {f.name!r} declares column {f.column}, "
                f"but the archived truth column map says its truth column is "
                f"{want}; label_quality scores one array against the other and "
                "cannot proceed if that coincidence breaks - see task-16 "
                "addendum item 5"
            )
    _assert_truth_column_shapes(archive)  # I6

    predict = adapter.load(paths.models / slug)

    arch_pred: dict[str, list[np.ndarray]] = {}
    ours_pred: dict[str, list[np.ndarray]] = {}
    valid_masks: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    used: list[int] = []
    skipped: dict[str, str] = {}
    skip_diagnosis: dict[str, dict] = {}
    resolvers: dict[str, set] = {}

    for shot in shots:
        try:
            with _shot_budget(timeout_s):
                m = _matched_shot(shot, spec, paths, archive)
                if m.built is not None:
                    for canonical, source in m.built.resolvers.items():
                        resolvers.setdefault(canonical, set()).add(source)
                if m.skip_reason:
                    skipped[str(shot)] = m.skip_reason
                    # I3: the diagnosis that used to be in scope and thrown
                    # away at skip time.
                    skip_diagnosis[str(shot)] = {
                        "missing_features": (
                            sorted(m.built.missing) if m.built is not None else []
                        ),
                        "resolvers": (
                            {k: sorted(v) for k, v in m.built.resolvers.items()}
                            if m.built is not None else {}
                        ),
                        "feature_misses": missing_names(paths.features_file(shot)),
                    }
                    continue
                got, built = m.got, m.built
                idx = m.info["index"]
                theirs = BuiltInputs(
                    t=built.t[idx], scalars=got["x0"], profiles=got["x1"],
                    valid=np.ones(idx.size, bool), missing=(), resolvers={},
                )
                ours = BuiltInputs(
                    t=built.t[idx], scalars=built.scalars[idx],
                    profiles=built.profiles[idx], valid=built.valid[idx],
                    missing=built.missing, resolvers=built.resolvers,
                )
                a = adapter.output_spec.decode(predict(theirs))
                b = adapter.output_spec.decode(predict(ours))
                for name in a:
                    arch_pred.setdefault(name, []).append(a[name].mean)
                    ours_pred.setdefault(name, []).append(b[name].mean)
                truths.append(got["y"])
                valid_masks.append(np.asarray(ours.valid, dtype=bool))
                used.append(int(shot))
        except Exception as exc:  # noqa: BLE001 - per-shot isolation, see docstring
            skipped[str(shot)] = f"{type(exc).__name__}: {exc}"
            skip_diagnosis[str(shot)] = {
                "missing_features": [], "resolvers": {}, "feature_misses": {},
            }
            continue

    report: dict = {
        "slug": slug,
        "truth": str(archive / "y.npy"),
        "n_shots_requested": len(shots),
        "n_shots_used": len(used),
        "shots_used": used,
        "skipped": skipped,
        "skip_diagnosis": skip_diagnosis,
        "skip_reasons": _skip_report(skipped, resolvers),
        "resolvers": {k: sorted(v) for k, v in resolvers.items()},
        "selection_effect_note": (
            "I4 (task-16 review): shots skip exactly when their archive "
            "lacks bt/ip/tritop/tribot/gapin (the five MATCH_COLUMNS), which "
            "correlates with whatever else that shot's archive group is "
            "missing - so shots_used is not a random sample of "
            "shots_requested even at full coverage. Treat the metrics below "
            "as measured on the subset of shots whose archive group happens "
            "to be complete enough to align, not on a representative draw."
        ),
        "archived_inputs_all": {},
        "archived_inputs_valid": {},
        "reconstructed_inputs_valid": {},
        "reconstructed_inputs_all": {},
        "row_counts": {},
        "row_counts_total": {},
        "reconstruction_penalty": {},
        "reconstruction_penalty_note": (
            "C1 (task-16 review): computed as reconstructed_inputs_valid "
            "minus archived_inputs_valid - the SAME valid-row mask applied "
            "to both sides, so the two scores differ only in input source "
            "(archived vs. reconstructed), never also in which rows were "
            "counted. archived_inputs_all and reconstructed_inputs_all "
            "additionally score every matched row regardless of "
            "labelmaker's validity flag, kept for reference only; computing "
            "the penalty from *_all instead (or from *_valid on one side "
            "and *_all on the other, as an earlier version of this function "
            "did) mixes a row-selection change into the reconstruction "
            "price and can even invert which direction a metric moved - "
            "measured, on the n=31 proof-of-concept run, to understate the "
            "AUROC penalty by ~34% and invert the F1 comparison outright."
        ),
    }
    if not used:
        return report
    y = np.concatenate(truths)
    valid = np.concatenate(valid_masks)
    report["row_counts_total"] = {
        "valid": int(valid.sum()), "invalid": int((~valid).sum()),
    }
    for name, arch_values in arch_pred.items():
        col = _TRUTH_COLUMNS.get(name)
        if col is None:
            continue
        t = y[:, col]
        pa = np.concatenate(arch_values)
        po = np.concatenate(ours_pred[name])
        field = next(f for f in adapter.output_spec.fields if f.name == name)
        # C1: the ceiling is now scored the SAME two ways the published
        # score already was (`_score_field` does both in one call), so the
        # row-matched pair (`*_valid` on both sides) and the diagnostic pair
        # (`*_all` on both sides) are each internally consistent.
        archived_scored = _score_field(field.task, pa, t, valid)
        reconstructed_scored = _score_field(field.task, po, t, valid)
        report["archived_inputs_all"][name] = archived_scored["all_matched"]
        report["archived_inputs_valid"][name] = archived_scored["published"]
        report["reconstructed_inputs_valid"][name] = reconstructed_scored["published"]
        report["reconstructed_inputs_all"][name] = reconstructed_scored["all_matched"]
        report["row_counts"][name] = {
            "matched": int(pa.size),
            "valid": archived_scored["n_valid"],
            "invalid": archived_scored["n_invalid"],
        }
    for name in report["archived_inputs_valid"]:
        a_v = report["archived_inputs_valid"][name]
        o_v = report["reconstructed_inputs_valid"][name]
        key = "auroc" if "auroc" in a_v else "rmse"
        if a_v.get(key) is not None and o_v.get(key) is not None:
            report["reconstruction_penalty"][name] = {
                key: float(o_v[key] - a_v[key]),
                "n_rows": report["row_counts"][name]["valid"],
                "computed_from": (
                    "reconstructed_inputs_valid minus archived_inputs_valid "
                    "(row-matched - see reconstruction_penalty_note)"
                ),
            }
    return report


def model_index_results(reports: dict) -> list[dict]:
    """The headline numbers, in HuggingFace `model-index` shape.

    Reads `label_quality`'s `archived_inputs_valid` and
    `reconstructed_inputs_valid` - the row-matched pair C1 (task-16 review)
    exists to produce - never `*_all`, which mixes a different row set into
    the comparison (see `label_quality`'s docstring). `dataset.name` states
    both denominators a card-only reader needs to see this is a like-for-
    like comparison: how many of the requested shots were used, and how
    many of the matched rows passed labelmaker's own validity mask (C1/I4)
    - without them, "d3d overlap shots (n=31)" alone hides that the
    headline numbers are a 31-of-100-shot, valid-rows-only measurement.
    """
    quality = reports.get("label_quality") or {}
    results: list[dict] = []
    row_counts = quality.get("row_counts_total") or {}
    n_valid, n_invalid = row_counts.get("valid"), row_counts.get("invalid")
    if n_valid is not None and n_invalid is not None:
        dataset_name = (
            f"d3d overlap shots (n_shots={quality.get('n_shots_used')}"
            f"/{quality.get('n_shots_requested')} requested, "
            f"n_rows={n_valid}/{n_valid + n_invalid} valid after "
            "labelmaker's validity mask)"
        )
    else:
        # No rows scored (e.g. n_shots_used == 0) - fall back to the shot
        # count alone rather than a dataset name with a bare "None" in it.
        dataset_name = f"d3d overlap shots (n={quality.get('n_shots_used')})"
    for source, suffix in (
        ("archived_inputs_valid", "archived inputs"),
        ("reconstructed_inputs_valid", "reconstructed inputs"),
    ):
        for label, metrics in (quality.get(source) or {}).items():
            entries = []
            for key, mtype in (("auroc", "roc_auc"), ("f1_at_0.5", "f1"),
                               ("rmse", "rmse")):
                value = metrics.get(key)
                if value is None:
                    continue
                entries.append(
                    {"name": f"{key} ({suffix})", "type": mtype, "value": float(value)}
                )
            if not entries:
                continue
            kinds = {e["type"] for e in entries}
            task_type = (
                "tabular-regression" if kinds == {"rmse"} else "tabular-classification"
            )
            results.append(
                {
                    "task": {"type": task_type, "name": label},
                    "dataset": {"name": dataset_name, "type": "d3d-faith-corpus"},
                    "metrics": entries,
                }
            )
    return results
