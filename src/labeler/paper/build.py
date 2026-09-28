"""Every paper product into `$LABELER_ROOT/paper/`, and the copy for the manuscript.

    PYTHONPATH=src pixi run -e labelmaker \\
        python scripts/labeler/paper/make_figures.py \\
        [--out DIR] [--shot SHOT] [--examples N] [--version V] [--copy-to DIR]

reads what the round-two runs wrote (`inputs`) and draws what they allow:

- `fig_scores`, `fig_mhd`, `table_ae_scores.tex`: the AE evaluation;
- `fig_segmentation`, `table_seg_scores.tex`: the segmentation's evaluation;
- `table_differences.tex`: the paired differences both evaluations hold;
- `fig_coverage`, `table_datasets.tex`: the owner's live AE review, with the
  chosen model's split (the reviewed shots in no split apart) and the
  extension's summary where they exist. A cross-validated version (its
  `chosen.json` names `folds_sha256`, or its `cv/folds.csv` exists) shows its
  train and test shots alone, with no validation split, whatever its folds.
  The train shots are marked with the number of folds only when `cv/folds.csv`
  is there, has a `fold` column and is the one `chosen.json` names
  (`folds_check`, D18's check applied to the folds); otherwise both products
  are `partial`, saying why;
- `fig_interpreter`, `fig_examples`: the chosen model run over its test shots
  (`split.csv`), scored against its own copy of the labels,
  `<candidate>/review/labels.csv` (D18), with the segmentation's mask, SegNet
  run over the same stores, where its `model.pt` exists. The copy's sha256 must be the one the AE evaluation names
  (`labels_sha256`); if it is not, or none is named, both are skipped.

The owner's live labels are read for the coverage alone; every scored product
uses the labels the model was scored against.

A product whose inputs are missing is listed in `manifest.json` under `skipped`,
with the reason and the missing paths, and one drawn without some of them under
`partial`, each missing part with its reason (the extension's from the AE bar,
D47; a test shot without a saved label by number). Nothing is drawn into `out`
itself: the build draws into a directory beside it and swaps that in whole at
the end (`staging`), the products and the manifest together, so a failure
leaves `out` as it was. The old output is deleted only once the new one is in
place; a swap that can neither finish nor put the old output back (another
build's output landed at `out` meanwhile, say) deletes nothing and raises
`Stranded`, which names where each one is. The build claims only the names it
writes (`OWNED`: the manifest and each product's own `.pdf` and `.png`, or
`.tex`); anything else in `out` stays, moved into the new output (a symlink as
itself), and a failed build deletes none of it. Of the old output only those
names are deleted: an entry written into it late, through a handle held on it,
is moved into the new output too, or, if its name is taken there, kept with the
old directory, which the build names on stderr and in its JSON line
(`old_output_kept`) and still succeeds. `out` is resolved once, first: a
symlinked `--out` keeps its link, the new output swapped in at its target, with
the staging directory and the old output's holder beside that. A broken link is
built into its target, which the build makes, as it makes a new `--out`.

`--version` (default `v1`) names the frame model's version the inputs come
from: `models/ae_xpower/<version>/` (`chosen.json`, `evaluation.json`, the
chosen `<candidate>/{model.pt, split.csv, review/labels.csv}` and, for a
cross-validated version such as v2, `cv/folds.csv`) and the extension's
summary. Until a version's records exist its AE products are `skipped`, with
the paths missing, and the coverage is still drawn from the owner's live
labels, which have no version.

`--seg-version` (default `v1`) names the segmentation's, apart: a new frame
model does not retrain SegNet, so v2's frame figures stand beside SegNet v1.
fig_segmentation, table_seg_scores, the segmentation's rows of
table_differences, the shot figures' mask and the interpreter pick's points of
interest read `models/ae_seg/<seg-version>/` and `poi/.../ae_seg-<seg-version>/`. The
segmentation's own copy of the labels (`<seg-version>/review/labels.csv`) is
pinned and checked against its record's `labels_sha256` (`seg_labels_match`),
not against the frame model's; a copy missing or unlike the record makes its
products `partial`. The manifest names the frame model SegNet was evaluated
beside (`seg_ae_model`, from its record), and so does table_seg_scores' comment.

**The second look.** A version whose test shots include an earlier version's
(`labeler.ae.xpower.evaluate.SUBSET_OF`: v2's include v1's) says so in the AE
tables' `%` comments and the manifest's `second_look`: how many of its test
shots were the earlier version's, and the model and time that version's test
was scored with (`evaluate.second_look`). They are counted from the earlier
version's records, its `evaluation.json` and its model's `split.csv`, pinned
as `ae_earlier_evaluation` and `ae_earlier_split`; without them the tables are
`partial`. Whether the owner accepts the second look is the ledger's record.

The manifest pins every file the build reads (the chosen `model.pt`, its
`split.csv`, its labels and each spectrogram store among them) with its
sha256, and records the full commit and whether the tree was dirty, whether the
frame model's copy of the labels is the one its evaluation names
(`labels_match`; the segmentation's own check is `seg_labels_match`), whether
a cross-validated version's `cv/folds.csv` is the one its `chosen.json` names
(`folds_match`: None for a version that is not cross-validated, as v1, or when
either is missing), each sha256 in `labels_sha256` (the live table's too), the
time, the
interpreter's shot, its pool and the branch of the rule that fired, the example
shots, the rules that picked them, and the drawn shots' F1 over 0-2 s and over
the whole window. The shots are ranked by their F1 over 0-2 s
(`shots.rank_keys`).

**The sha256s are of the bytes drawn** (`snapshot.Snapshot`): each input is
read once, hashed, and parsed from those bytes; a second read of one is refused,
so its pin stays the bytes drawn. At the end every input is hashed again; one
that changed while the build ran (the owner saving, say) is listed under
`changed_during_build`, with both sha256s, and `consistent` is false.

`--copy-to` copies this run's PDFs and `.tex` tables into a directory (the
manuscript's `dev/label_paper/figures/`); it never runs git there, so nothing is
committed or pushed.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import shutil
import sys
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import torch

from ..ae import seg as ae_seg
from ..ae import xpower
from ..ae.xpower import evaluate as ae_evaluate
from ..config import Paths, atomic_path, git_dirty, git_sha
from ..events.review import labels
from . import AE, coverage, paper_dir, scores, shots
from .snapshot import Snapshot
from .staging import Stranded, discard, staging_dir, swap

VERSION = xpower.VERSION  # the frame model's version; `--version` names another
SEG_VERSION = ae_seg.VERSION  # the segmentation's; `--seg-version` names another
LOGIN_THREADS = 2
COPIED = (".pdf", ".tex")
MANIFEST = "manifest.json"
KEPT = "old_output_kept"  # in `build`'s answer and the JSON line
KEPT_SAID = (
    "{out}: the new output is in place, but the old one's directory still holds "
    "entries that could neither be deleted nor moved into it, so it is kept: {kept}"
)
FIGURE = (".pdf", ".png")  # what `paper.save` writes for a figure
TABLE = (".tex",)
PRODUCTS = {
    "fig_scores": FIGURE,
    "fig_mhd": FIGURE,
    "table_ae_scores": TABLE,
    "fig_segmentation": FIGURE,
    "table_seg_scores": TABLE,
    "table_differences": TABLE,
    "fig_coverage": FIGURE,
    "table_datasets": TABLE,
    "fig_interpreter": FIGURE,
    "fig_examples": FIGURE,
}


def files_of(product: str) -> list[str]:
    """The files a product writes into `out`."""
    return [product + suffix for suffix in PRODUCTS[product]]


# Every name the build writes into `out`; the swap keeps anything else there.
OWNED = frozenset({MANIFEST, *(f for p in PRODUCTS for f in files_of(p))})

# Why a product is skipped or drawn in part (`skipped`, `partial`).
MISSING = "missing inputs"
NO_EVALUATION = "extension not run: no AE evaluation to gate it (D47)"
NO_SUMMARY = "the extension passed its gate (D47) but has not written summary.csv"
NO_BAR = "extension not run: the AE evaluation records no bar (D47)"
UNDECIDED = "extension not run: the AE evaluation's bar leaves {bars} undecided (D47)"
EXTENSION_BARS = ("A1", "A2")  # D47: the extension runs only if both pass
NO_CHOSEN = "no model chosen, so no split"
NO_SPLIT = "the chosen model has no split.csv"
NO_FOLDS = (
    "the chosen model is cross-validated but its cv/folds.csv is missing, so its "
    "folds are not counted"
)
FOLDS_UNNAMED = (
    "chosen.json names no folds_sha256, so cv/folds.csv cannot be checked and its "
    "folds are not counted"
)
FOLDS_DIFFER = (
    "cv/folds.csv is not the one chosen.json names (folds_sha256), so its folds "
    "are not counted"
)
FOLDS_NO_COLUMN = "cv/folds.csv has no fold column, so its folds are not counted"
CV_VAL = (
    "the chosen model is cross-validated, which holds no shot out for "
    "validation, but its split.csv calls these reviewed shots val: they are in "
    "no bar or cell"
)
NO_POI = (
    "no points of interest (poi.csv): the interpreter pick cannot prefer shots with one"
)
NO_MASK = "no mask: the segmentation has no model.pt to run over the test shots"
NO_LABEL = "a test shot with no saved AE label"
NO_STORE = "a test shot with no spectrogram store"
NO_SCORED_SHOT = "no test shot has both a saved label and a store"
NO_F1 = (
    "no test shot has an F1 over 0-2 s: neither the owner nor the model calls "
    "a scored frame there present"
)
NO_NAMED = "the named shot has no saved label in the model's copy, or no store"
AE_LABEL_KEYS = ("labels_sha256", "labels_copy_sha256")  # in the AE record's meta
LABELS_UNNAMED = "the AE evaluation names no labels_sha256, so D18 cannot be checked"
LABELS_DIFFER = "the model's review/labels.csv is not what its evaluation scored (D18)"
SEG_PRODUCTS = ("fig_segmentation", "table_seg_scores")
NO_SEG_COPY = "the segmentation has no review/labels.csv to check its record against"
SEG_LABELS_UNNAMED = "the segmentation's evaluation names no labels_sha256"
SEG_LABELS_DIFFER = (
    "the segmentation's review/labels.csv is not what its evaluation scored (D18)"
)
LOOK_PRODUCTS = ("table_ae_scores", "table_differences")  # say the second look
NO_EARLIER = (
    "the earlier version's records are missing, so the test shots it scored "
    "first are not counted"
)
NO_LOOK_SPLIT = "no split.csv, so the earlier version's test shots are not counted"


def inputs(
    paths: Paths, version: str = VERSION, seg_version: str = SEG_VERSION
) -> dict[str, Path]:
    """Where the round-two runs leave what the paper reads, for the frame
    model's `version` and the segmentation's `seg_version`; the build adds the
    chosen model, its split, its copy of the labels (`ae_scored_labels`) and the
    stores it reads."""
    models = xpower.model_dir(paths, version)
    seg = ae_seg.model_dir(paths).parent / seg_version
    poi = ae_seg.poi_dir(paths).parent / f"{ae_seg.METHOD}-{seg_version}"
    return {
        "ae_evaluation": models / "evaluation.json",
        "ae_chosen": models / "chosen.json",
        "ae_folds": models / "cv" / "folds.csv",  # cross-validated versions only
        "ae_labels": labels.labels_path(xpower.event_dir(paths)),
        "seg_evaluation": seg / "evaluation.json",
        "seg_labels": labels.labels_path(seg),
        "seg_model": seg / "model.pt",
        "summary": xpower.suggestions_dir(paths, version) / "summary.csv",
        "poi": poi / "poi.csv",
    }


def _recorded(record: dict | None, *where: str) -> str | None:
    value = record
    for step in where:
        value = value.get(step) if isinstance(value, dict) else None
    return value or None


def labels_check(
    scored: str | None, live: str | None, ae: dict | None
) -> tuple[bool | None, dict[str, str]]:
    """Whether the chosen frame model's copy of the labels (`scored`) is the one
    its evaluation names (None when there is no copy or the record names none),
    and the sha256s: the copy's, the live table's and each recorded."""
    shas = {k: v for k, v in (("scored", scored), ("live", live)) if v}
    for key, where in (
        ("ae_evaluation", ("meta", "labels_sha256")),
        ("ae_evaluation_copy", ("meta", "labels_copy_sha256")),
    ):
        if value := _recorded(ae, *where):
            shas[key] = value
    recorded = [v for k, v in shas.items() if k not in ("scored", "live")]
    match = all(v == scored for v in recorded) if scored and recorded else None
    return match, shas


def seg_check(
    scored: str | None, seg: dict | None
) -> tuple[bool | None, dict[str, str]]:
    """The same for the segmentation, against its own copy (`scored`): None
    when there is no copy or its record names none."""
    named = _recorded(seg, "meta", "inputs", "labels_sha256")
    shas = {k: v for k, v in (("seg_scored", scored), ("seg_evaluation", named)) if v}
    return (named == scored if scored and named else None), shas


class FoldsCheck(NamedTuple):
    """`folds_check`'s answer."""

    cross_validated: bool
    match: bool | None = None
    count: int | None = None
    why: dict | None = None  # the `partial` entry: why the folds are not counted


def _table(data: bytes) -> pd.DataFrame:
    """A CSV's rows; none for bytes that are not a table."""
    try:
        return pd.read_csv(io.BytesIO(data))
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def folds_check(chosen: dict | None, path: Path, snap: Snapshot) -> FoldsCheck:
    """Whether the chosen model is cross-validated, and its folds checked
    against its record as D18 checks the labels. A version is cross-validated
    when its `chosen.json` names `folds_sha256` or its `cv/folds.csv` (`path`)
    exists; without a chosen model there is no split to fold, and nothing to
    check against, so the file is not read. `match` is the file's pinned
    sha256 against the record's, None unless both exist. `count`, the number
    of folds, is given only when the file is there, is the one the record
    names and has a `fold` column; otherwise `why` says why not."""
    if chosen is None or ("folds_sha256" not in chosen and not path.is_file()):
        return FoldsCheck(False)
    if not path.is_file():
        return FoldsCheck(True, why={"reason": NO_FOLDS, "missing": [str(path)]})
    data = snap.read("ae_folds", path)
    sha, named = snap.sha("ae_folds"), _recorded(chosen, "folds_sha256")
    if named is None:
        return FoldsCheck(True, why={"reason": FOLDS_UNNAMED})
    if named != sha:
        why = f"{FOLDS_DIFFER}: the file is {sha[:12]}, chosen.json names "
        return FoldsCheck(True, False, why={"reason": why + str(named)[:12]})
    count = coverage.fold_count(_table(data))
    if count is None:
        return FoldsCheck(True, True, why={"reason": FOLDS_NO_COLUMN})
    return FoldsCheck(True, True, count)


def scored_refusal(scored: str | None, ae: dict | None) -> str | None:
    """Why the shot products cannot be scored on the model's copy of the labels
    (D18): the AE evaluation names no labels, or others; None when it is that copy."""
    named = {k: _recorded(ae, "meta", k) for k in AE_LABEL_KEYS}
    named = {k: v for k, v in named.items() if v}
    if not named:
        return LABELS_UNNAMED
    wrong = ", ".join(f"{k} {v[:12]}" for k, v in named.items() if v != scored)
    if wrong:
        copy = (scored or "")[:12]
        return f"{LABELS_DIFFER}: the copy is {copy}, the record names {wrong}"
    return None


def _number(x: float) -> float | None:
    return None if math.isnan(x) else round(float(x), 4)


def _store(paths: Paths, shot: int) -> Path:
    return paths.spectrogram_file(xpower.EVENT, shot)


def _write(path: Path, text: str) -> None:
    with atomic_path(path) as tmp:
        tmp.write_text(text)


def extension_reason(ae: dict | None) -> str:
    """Why the extension's summary is missing, from the AE evaluation's `bar`:
    D47 runs the extension only if A1 and A2 pass. It passed its gate only when
    both are recorded True; one recorded False is named as failed, and one
    missing or neither True nor False leaves the gate undecided."""
    if ae is None:
        return NO_EVALUATION
    bar = ae.get("bar")
    if not isinstance(bar, dict):
        return NO_BAR
    failed = [k for k in EXTENSION_BARS if bar.get(k) is False]
    if failed:
        return f"extension not run: {' and '.join(failed)} failed (D47)"
    undecided = [k for k in EXTENSION_BARS if bar.get(k) is not True]
    if undecided:
        return UNDECIDED.format(bars=" and ".join(undecided))
    return NO_SUMMARY


def second_look(
    paths: Paths,
    found: dict[str, Path],
    snap: Snapshot,
    version: str,
    chosen: dict | None,
    split: dict[int, str] | None,
) -> tuple[dict | None, dict | None]:
    """The earlier version's look at this version's test shots
    (`evaluate.second_look`), from its `evaluation.json` and its model's
    `split.csv`, each pinned; or, when it cannot be counted, the `partial`
    entry that says why. Both None for a version with no earlier one."""
    earlier = ae_evaluate.SUBSET_OF.get(version)
    if earlier is None:
        return None, None
    models = xpower.model_dir(paths, earlier)
    evaluation = found["ae_earlier_evaluation"] = models / "evaluation.json"
    if not evaluation.is_file():
        return None, {"reason": NO_EARLIER, "missing": [str(evaluation)]}
    record = snap.json("ae_earlier_evaluation", evaluation)
    candidate = str(record.get("meta", {}).get("candidate"))
    split_file = found["ae_earlier_split"] = models / candidate / "split.csv"
    if not split_file.is_file():
        return None, {"reason": NO_EARLIER, "missing": [str(split_file)]}
    theirs = snap.split("ae_earlier_split", split_file)
    if split is None:
        return None, {"reason": NO_LOOK_SPLIT}
    test = {s for s, v in split.items() if v == "test"}
    return ae_evaluate.second_look(version, test, record, theirs, chosen), None


def build(
    paths: Paths,
    out: Path,
    *,
    shot: int | None = None,
    examples: int = 3,
    version: str = VERSION,
    seg_version: str = SEG_VERSION,
) -> dict:
    """Draw every product the inputs allow, and the manifest, into a directory
    beside `out`, then swap it in whole; on a failure `out` is left as it was,
    and if the swap can neither finish nor undo itself, nothing is deleted.
    The manifest; if the swap had to keep the old output's directory, the
    answer also names it (`KEPT`, which the manifest, written before the swap,
    cannot), and so does stderr."""
    out = Path(out).resolve()  # a link's target: the link keeps pointing at it
    staged = staging_dir(out)
    try:
        with tempfile.TemporaryDirectory(prefix="paper-inputs-") as scratch:
            snap = Snapshot(Path(scratch))
            manifest = _draw(
                paths,
                staged,
                snap,
                shot=shot,
                examples=examples,
                version=version,
                seg_version=seg_version,
            )
        kept = swap(staged, out, owned=OWNED)
    except Stranded:
        raise  # `staged` holds the new output, and the message says so
    except BaseException:
        discard(staged, owned=OWNED)  # the build's own files, never the owner's
        raise
    if kept is None:
        return manifest
    print(KEPT_SAID.format(out=out, kept=kept), file=sys.stderr, flush=True)
    return manifest | {KEPT: str(kept)}


def _draw(
    paths: Paths,
    out: Path,
    snap: Snapshot,
    *,
    shot: int | None,
    examples: int,
    version: str,
    seg_version: str,
) -> dict:
    found = inputs(paths, version, seg_version)
    made: dict[str, list[str]] = {}
    skipped: dict[str, dict] = {}
    partial: dict[str, list[dict]] = {}

    def ready(products: tuple[str, ...], *needs: str) -> bool:
        missing = [str(found[k]) for k in needs if not found[k].is_file()]
        for product in products if missing else ():
            skipped[product] = {"reason": MISSING, "missing": missing}
        return not missing

    def lacking(products: tuple[str, ...], reason: str, **what) -> None:
        """Drawn without part of its inputs, for `reason`: `what` names them."""
        entry = {"reason": reason, **what}
        for product in products:
            partial.setdefault(product, []).append(entry)

    def figure(name: str, draw: Callable, *args) -> None:
        draw(*args, out / name)
        made[name] = files_of(name)

    def table(name: str, text: str) -> None:
        [file] = files_of(name)
        _write(out / file, text)
        made[name] = [file]

    def read(key: str, parse: Callable):
        return parse(key, found[key]) if found[key].is_file() else None

    ae = read("ae_evaluation", snap.json)
    seg = read("seg_evaluation", snap.json)
    chosen = read("ae_chosen", snap.json)
    if chosen is not None:
        candidate = found["ae_chosen"].parent / chosen["candidate"]
        found["ae_model"] = candidate / "model.pt"
        found["ae_split"] = candidate / "split.csv"
        found["ae_scored_labels"] = labels.labels_path(candidate)
    split = read("ae_split", snap.split) if "ae_split" in found else None
    look, unlooked = (None, None)
    if ae is not None:
        look, unlooked = second_look(paths, found, snap, version, chosen, split)
    said = look["said"] if look else None
    if ready(("fig_scores", "fig_mhd", "table_ae_scores"), "ae_evaluation"):
        figure("fig_scores", scores.draw_scores, ae)
        figure("fig_mhd", scores.draw_mhd, ae)
        table("table_ae_scores", scores.table_ae(ae, said))
    seg_match, seg_shas = None, {}
    if ready(SEG_PRODUCTS, "seg_evaluation"):
        figure("fig_segmentation", scores.draw_segmentation, seg)
        table("table_seg_scores", scores.table_segmentation(seg))
        copy = found["seg_labels"]
        if copy.is_file():
            snap.read("seg_labels", copy)
        seg_match, seg_shas = seg_check(snap.sha("seg_labels"), seg)
        if not copy.is_file():
            lacking(SEG_PRODUCTS, NO_SEG_COPY, missing=[str(copy)])
        elif "seg_evaluation" not in seg_shas:
            lacking(SEG_PRODUCTS, SEG_LABELS_UNNAMED)
        elif seg_match is False:
            named = seg_shas["seg_evaluation"][:12]
            why = f"{SEG_LABELS_DIFFER}: the copy is {seg_shas['seg_scored'][:12]}"
            lacking(SEG_PRODUCTS, f"{why}, the record names {named}")
    evaluations = ("ae_evaluation", "seg_evaluation")
    if ae is not None or seg is not None:
        table("table_differences", scores.table_differences(ae, seg, said))
        absent = [str(found[k]) for k in evaluations if not found[k].is_file()]
        if absent:
            lacking(("table_differences",), MISSING, missing=absent)
    else:
        ready(("table_differences",), *evaluations)
    if unlooked:
        lacking(LOOK_PRODUCTS, **unlooked)
    folds = folds_check(chosen, found["ae_folds"], snap)
    live = read("ae_labels", snap.labels)
    counted = ("fig_coverage", "table_datasets")
    if ready(counted, "ae_labels"):
        summary = read("summary", snap.csv)
        counts = {
            AE: coverage.ae_counts(
                live,
                split,
                summary,
                folds=folds.count,
                cross_validated=folds.cross_validated,
            )
        }
        figure("fig_coverage", coverage.draw_coverage, counts)
        table("table_datasets", coverage.table_datasets(counts))
        if chosen is None:
            lacking(counted, NO_CHOSEN, missing=[str(found["ae_chosen"])])
        elif split is None:
            lacking(counted, NO_SPLIT, missing=[str(found["ae_split"])])
        elif folds.why:
            lacking(counted, **folds.why)
        held = sorted(s for s, v in (split or {}).items() if v == "val" and s in live)
        if folds.cross_validated and held:
            lacking(counted, CV_VAL, shots=held)
        if summary is None:
            lacking(counted, extension_reason(ae), missing=[str(found["summary"])])
    picked: dict = {}
    figures = ("fig_interpreter", "fig_examples")
    scored = None
    if ready(figures, "ae_evaluation", "ae_chosen") and ready(
        figures, "ae_model", "ae_split", "ae_scored_labels"
    ):
        saved = read("ae_scored_labels", snap.labels)
        scored = snap.sha("ae_scored_labels")
        why = scored_refusal(scored, ae)
        why, picked = (
            (why, {})
            if why
            else _shot_figures(
                paths, found, snap, saved, split, shot, examples, figure, lacking
            )
        )
        for product in figures if why else ():
            made.pop(product, None)
            skipped[product] = {"reason": why, "missing": []}
    elif "ae_scored_labels" in found and found["ae_scored_labels"].is_file():
        read("ae_scored_labels", snap.labels)
        scored = snap.sha("ae_scored_labels")
    match, shas = labels_check(scored, snap.sha("ae_labels"), ae)
    changed = snap.changed()
    manifest = {
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(full=True),
        "git_dirty": git_dirty(),
        "version": version,
        "seg_version": seg_version,
        "labels_match": match,
        "seg_labels_match": seg_match,
        "folds_match": folds.match,
        "labels_sha256": shas | seg_shas,
        "seg_ae_model": _recorded(seg, "meta", "ae_model"),
        "second_look": look,
        "consistent": not changed,
        "changed_during_build": changed,
        "inputs": {
            key: {"path": str(path), "sha256": sha}
            for key, (path, sha) in snap.pinned.items()
        },
        "products": made,
        "skipped": skipped,
        "partial": partial,
        **picked,
    }
    _write(out / MANIFEST, json.dumps(manifest, indent=1) + "\n")
    return manifest


def _shot_figures(
    paths: Paths,
    found: dict[str, Path],
    snap: Snapshot,
    saved: dict,
    split: dict[int, str],
    shot: int | None,
    examples: int,
    figure: Callable,
    lacking: Callable,
) -> tuple[str | None, dict]:
    """Score every test shot against the model's copy of the labels (D18) and
    draw the two shot figures. Why they could not be drawn (None when they
    were), and the picks for the manifest."""
    figures = ("fig_interpreter", "fig_examples")
    model = snap.model("ae_model", found["ae_model"], split)
    segmentation = (
        snap.segmentation("seg_model", found["seg_model"])
        if found["seg_model"].is_file()
        else None
    )
    poi = snap.csv("poi", found["poi"]) if found["poi"].is_file() else None
    tested = shots.test_shots(model.split)
    unlabelled = [s for s in tested if s not in saved]
    unstored = [_store(paths, s) for s in tested if not _store(paths, s).is_file()]
    usable = [s for s in tested if s in saved and _store(paths, s).is_file()]
    if not usable:
        return NO_SCORED_SHOT, {}

    def one(s: int) -> shots.AEShot:
        """The shot's picture, its store read once."""
        data = snap.read(f"store_{s}", _store(paths, s))
        return shots.picture(
            s, label=saved[s], model=model, store=data, segmentation=segmentation
        )

    pictures = {s: one(s) for s in usable}
    ranked = [shots.rank_keys(p) for p in pictures.values()]
    f1 = {r.shot: r.f1 for r in ranked}
    if all(math.isnan(v) for v in f1.values()):
        return NO_F1, {}
    pick = shots.interpreter_pick(f1, poi, {r.shot: r.gap for r in ranked})
    picked = {
        "interpreter_shot": pick["shot"] if shot is None else shot,
        "interpreter_rule": shots.INTERPRETER_RULE
        if shot is None
        else "named by --shot",
        "interpreter_branch": pick["branch"] if shot is None else None,
        "interpreter_pool": pick["pool"],
        "example_shots": shots.pick_examples(f1, examples),
        "example_rule": shots.EXAMPLES_RULE,
    }
    named = picked["interpreter_shot"]
    if named not in pictures and named in saved and _store(paths, named).is_file():
        pictures[named] = one(named)
    drawn = {s: pictures[s] for s in (named, *picked["example_shots"]) if s in pictures}
    if named not in drawn:
        return f"{NO_NAMED}: {named}", picked
    figure("fig_interpreter", shots.draw_interpreter, drawn[named])
    figure(
        "fig_examples",
        shots.draw_examples,
        [drawn[s] for s in picked["example_shots"]],
    )
    if segmentation is None:
        lacking(figures, NO_MASK, missing=[str(found["seg_model"])])
    if poi is None:
        lacking(figures, NO_POI, missing=[str(found["poi"])])
    if unlabelled:
        lacking(figures, NO_LABEL, shots=unlabelled)
    if unstored:
        lacking(figures, NO_STORE, missing=[str(p) for p in unstored])
    picked["shot_f1"] = {
        str(s): {"f1_0_2s": _number(d.f1), "f1_window": _number(d.f1_window)}
        for s, d in sorted(drawn.items())
    }
    return None, picked


def copy_into(out: Path, manifest: dict, dest: Path) -> list[str]:
    """Copy this run's PDFs and tables into `dest`; the names copied."""
    dest.mkdir(parents=True, exist_ok=True)
    names = sorted(
        name
        for files in manifest["products"].values()
        for name in files
        if Path(name).suffix in COPIED
    )
    for name in names:
        shutil.copy2(out / name, dest / name)
    return names


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, help="default: $LABELER_ROOT/paper")
    parser.add_argument(
        "--shot", type=int, help="the interpreter's shot (default: `interpreter_shot`)"
    )
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument(
        "--version",
        default=VERSION,
        help=f"the frame model's version (default {VERSION})",
    )
    parser.add_argument(
        "--seg-version",
        default=SEG_VERSION,
        help=f"the segmentation's version (default {SEG_VERSION})",
    )
    parser.add_argument(
        "--copy-to", type=Path, help="also copy the PDFs and tables here"
    )
    args = parser.parse_args(argv)
    torch.set_num_threads(LOGIN_THREADS)  # one shot at a time, on the login node
    paths = Paths.from_env()
    out = args.out or paper_dir(paths)
    manifest = build(
        paths,
        out,
        shot=args.shot,
        examples=args.examples,
        version=args.version,
        seg_version=args.seg_version,
    )
    copied = [] if args.copy_to is None else copy_into(out, manifest, args.copy_to)
    print(
        json.dumps(
            {
                "out": str(out),
                "products": sorted(manifest["products"]),
                "skipped": manifest["skipped"],
                "copied": copied,
                KEPT: manifest.get(KEPT),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
