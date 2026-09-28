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
  extension's summary where they exist;
- `fig_interpreter`, `fig_examples`: the chosen model run over its test shots
  (`split.csv`), scored against its own copy of the labels,
  `<candidate>/review/labels.csv` (D18), with the points of interest where they
  exist. The copy's sha256 must be the one the AE evaluation names
  (`labels_sha256`); if it is not, or none is named, both are skipped.

The owner's live labels are read for the coverage alone; every scored product
uses the labels the model was scored against.

A product whose inputs are missing is listed in `manifest.json` under `skipped`,
with the reason and the missing paths, and one drawn without some of them under
`partial`, each missing part with its reason (the extension's from the AE bar,
D47; a test shot without a saved label by number). Nothing is drawn into `out`
itself: the build draws into a directory beside it and swaps that in whole at
the end, the products and the manifest together, so a failure leaves `out` as
it was. A file in `out` that is no product's stays.

`--version` (default `v1`) names the models' version the inputs come from. The
manifest pins every file the build reads (the chosen `model.pt`, its
`split.csv`, its labels and each spectrogram store among them) with its
sha256, and records the full commit and whether the tree was dirty, whether the
model's copy of the labels is the one every evaluation names (`labels_match`,
with each sha256 in `labels_sha256`, the live table's too), the time, the
interpreter's shot, its pool and the branch of the rule that fired, the example
shots, the rules that picked them, and the drawn shots' F1 over 0-2 s and over
the whole window. The shots are ranked by their F1 over 0-2 s
(`shots.rank_keys`).

**The sha256s are of the bytes drawn** (`Snapshot`): each input is read once,
hashed, and parsed from those bytes. At the end every input is hashed again; one
that changed while the build ran (the owner saving, say) is listed under
`changed_during_build`, with both sha256s, and `consistent` is false.

`--copy-to` copies this run's PDFs and `.tex` tables into a directory (the
manuscript's `dev/label_paper/figures/`); it never runs git there, so nothing is
committed or pushed.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch

from ..ae import seg as ae_seg
from ..ae import xpower
from ..ae.xpower.train import read_split
from ..config import Paths, atomic_path, git_dirty, git_sha, sha256_of
from ..events.review import labels
from . import AE, coverage, paper_dir, scores, shots

VERSION = xpower.VERSION  # the models' version; `--version` names another
LOGIN_THREADS = 2
COPIED = (".pdf", ".tex")
MANIFEST = "manifest.json"
PRODUCTS = (
    "fig_scores",
    "fig_mhd",
    "table_ae_scores",
    "fig_segmentation",
    "table_seg_scores",
    "table_differences",
    "fig_coverage",
    "table_datasets",
    "fig_interpreter",
    "fig_examples",
)
# Why a product is skipped or drawn in part (`skipped`, `partial`).
MISSING = "missing inputs"
NO_EVALUATION = "extension not run: no AE evaluation to gate it (D47)"
NO_SUMMARY = "the extension passed its gate (D47) but has not written summary.csv"
EXTENSION_BARS = ("A1", "A2")  # D47: the extension runs only if both pass
NO_CHOSEN = "no model chosen, so no split"
NO_SPLIT = "the chosen model has no split.csv"
NO_POI = "no points of interest: the segmentation has not run over the test shots"
NO_LABEL = "a test shot with no saved AE label"
NO_STORE = "a test shot with no spectrogram store"
NO_SCORED_SHOT = "no test shot has both a saved label and a store"
NO_NAMED = "the named shot has no saved label in the model's copy, or no store"
AE_LABEL_KEYS = ("labels_sha256", "labels_copy_sha256")  # in the AE record's meta
LABELS_UNNAMED = "the AE evaluation names no labels_sha256, so D18 cannot be checked"
LABELS_DIFFER = "the model's review/labels.csv is not what its evaluation scored (D18)"


def inputs(paths: Paths, version: str = VERSION) -> dict[str, Path]:
    """Where the round-two runs leave what the paper reads, for the models'
    `version`; the build adds the chosen model, its split, its copy of the labels
    (`ae_scored_labels`) and the stores it reads."""
    models = xpower.model_dir(paths, version)
    return {
        "ae_evaluation": models / "evaluation.json",
        "ae_chosen": models / "chosen.json",
        "ae_labels": labels.labels_path(xpower.event_dir(paths)),
        "seg_evaluation": ae_seg.model_dir(paths).parent / version / "evaluation.json",
        "summary": xpower.suggestions_dir(paths, version) / "summary.csv",
        "poi": ae_seg.poi_dir(paths).parent / f"{ae_seg.METHOD}-{version}" / "poi.csv",
    }


def _recorded(record: dict | None, *where: str) -> str | None:
    value = record
    for step in where:
        value = value.get(step) if isinstance(value, dict) else None
    return value or None


def labels_check(
    scored: str | None, live: str | None, ae: dict | None, seg: dict | None
) -> tuple[bool | None, dict[str, str]]:
    """Whether the chosen model's copy of the labels (`scored`) is the one every
    evaluation record names (None when there is no copy or no record names one),
    and the sha256s: the copy's, the live table's and each recorded."""
    shas = {k: v for k, v in (("scored", scored), ("live", live)) if v}
    for key, record, where in (
        ("ae_evaluation", ae, ("meta", "labels_sha256")),
        ("ae_evaluation_copy", ae, ("meta", "labels_copy_sha256")),
        ("seg_evaluation", seg, ("meta", "inputs", "labels_sha256")),
    ):
        if value := _recorded(record, *where):
            shas[key] = value
    recorded = [v for k, v in shas.items() if k not in ("scored", "live")]
    match = all(v == scored for v in recorded) if scored and recorded else None
    return match, shas


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
    """Why the extension's summary is missing: D47 runs it only if A1 and A2 pass."""
    if ae is None:
        return NO_EVALUATION
    failed = [k for k in EXTENSION_BARS if ae.get("bar", {}).get(k) is False]
    if failed:
        return f"extension not run: {' and '.join(failed)} failed (D47)"
    return NO_SUMMARY


def _staging(out: Path) -> Path:
    """A fresh directory beside `out` to draw into."""
    for k in range(1000):
        staged = out.with_name(f".{out.name}.staging-{os.getpid()}-{k}")
        try:
            staged.mkdir(parents=True)
            return staged
        except FileExistsError:
            continue
    raise FileExistsError(f"{out}: no free staging directory beside it")


def _ours(name: str) -> bool:
    """A file the build owns in `out`: a product's or the manifest."""
    return name == MANIFEST or Path(name).stem in PRODUCTS


def _swap(staged: Path, out: Path) -> None:
    """Put `staged` where `out` is, in two renames. The old products and
    manifest leave together; a file in `out` that is not the build's stays."""
    if not out.exists():
        staged.rename(out)
        return
    for kept in out.iterdir():
        if not _ours(kept.name) and not (staged / kept.name).exists():
            if kept.is_dir():
                shutil.copytree(kept, staged / kept.name, symlinks=True)
            else:
                shutil.copy2(kept, staged / kept.name, follow_symlinks=False)
    holder = Path(tempfile.mkdtemp(prefix=f".{out.name}.old-", dir=out.parent))
    out.rename(holder / out.name)
    try:
        staged.rename(out)
    except BaseException:
        (holder / out.name).rename(out)
        raise
    finally:
        shutil.rmtree(holder, ignore_errors=True)


def build(
    paths: Paths,
    out: Path,
    *,
    shot: int | None = None,
    examples: int = 3,
    version: str = VERSION,
) -> dict:
    """Draw every product the inputs allow, and the manifest, into a directory
    beside `out`, then swap it in whole; on a failure `out` is left as it was."""
    out = Path(out)
    staged = _staging(out)
    try:
        with tempfile.TemporaryDirectory(prefix="paper-inputs-") as scratch:
            snap = Snapshot(Path(scratch))
            manifest = _draw(
                paths, staged, snap, shot=shot, examples=examples, version=version
            )
        _swap(staged, out)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    return manifest


class Snapshot:
    """Each input read once: its bytes hashed, then parsed from those bytes, so
    the manifest pins what was drawn. `changed` hashes every input again."""

    def __init__(self, scratch: Path):
        self.scratch = scratch  # for a parser that needs a file: the bytes' copy
        self.pinned: dict[str, tuple[Path, str]] = {}
        self._kept: dict[str, bytes] = {}

    def read(self, key: str, path: Path, *, keep: bool = True) -> bytes:
        if key in self._kept:
            return self._kept[key]
        data = Path(path).read_bytes()
        self.pinned[key] = (Path(path), hashlib.sha256(data).hexdigest())
        if keep:
            self._kept[key] = data
        return data

    def sha(self, key: str) -> str | None:
        return self.pinned[key][1] if key in self.pinned else None

    def json(self, key: str, path: Path) -> dict:
        return json.loads(self.read(key, path))

    def csv(self, key: str, path: Path) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(self.read(key, path)))

    def split(self, key: str, path: Path) -> dict[int, str]:
        return read_split(path, data=self.read(key, path))

    def labels(self, key: str, path: Path) -> dict:
        """A `review/labels.csv`, parsed as the review parses it."""
        event = self.scratch / key
        copy = labels.labels_path(event)
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(self.read(key, path))
        return labels.read_saved(event)

    def model(self, key: str, path: Path, split) -> shots.Model:
        return shots.Model.load(io.BytesIO(self.read(key, path, keep=False)), split)

    def changed(self) -> dict[str, dict]:
        """The inputs whose bytes are no longer the ones drawn."""
        found = {}
        for key, (path, drawn) in self.pinned.items():
            now = sha256_of(path) if path.is_file() else None
            if now != drawn:
                found[key] = {"path": str(path), "drawn": drawn, "now": now}
        return found


def _draw(
    paths: Paths,
    out: Path,
    snap: Snapshot,
    *,
    shot: int | None,
    examples: int,
    version: str,
) -> dict:
    found = inputs(paths, version)
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
        made[name] = [f"{name}.pdf", f"{name}.png"]

    def table(name: str, text: str) -> None:
        _write(out / f"{name}.tex", text)
        made[name] = [f"{name}.tex"]

    def read(key: str, parse: Callable):
        return parse(key, found[key]) if found[key].is_file() else None

    ae = read("ae_evaluation", snap.json)
    seg = read("seg_evaluation", snap.json)
    if ready(("fig_scores", "fig_mhd", "table_ae_scores"), "ae_evaluation"):
        figure("fig_scores", scores.draw_scores, ae)
        figure("fig_mhd", scores.draw_mhd, ae)
        table("table_ae_scores", scores.table_ae(ae))
    if ready(("fig_segmentation", "table_seg_scores"), "seg_evaluation"):
        figure("fig_segmentation", scores.draw_segmentation, seg)
        table("table_seg_scores", scores.table_segmentation(seg))
    evaluations = ("ae_evaluation", "seg_evaluation")
    if ae is not None or seg is not None:
        table("table_differences", scores.table_differences(ae, seg))
        absent = [str(found[k]) for k in evaluations if not found[k].is_file()]
        if absent:
            lacking(("table_differences",), MISSING, missing=absent)
    else:
        ready(("table_differences",), *evaluations)
    chosen = read("ae_chosen", snap.json)
    if chosen is not None:
        candidate = found["ae_chosen"].parent / chosen["candidate"]
        found["ae_model"] = candidate / "model.pt"
        found["ae_split"] = candidate / "split.csv"
        found["ae_scored_labels"] = labels.labels_path(candidate)
    split = read("ae_split", snap.split) if "ae_split" in found else None
    live = read("ae_labels", snap.labels)
    counted = ("fig_coverage", "table_datasets")
    if ready(counted, "ae_labels"):
        summary = read("summary", snap.csv)
        counts = {AE: coverage.ae_counts(live, split, summary)}
        figure("fig_coverage", coverage.draw_coverage, counts)
        table("table_datasets", coverage.table_datasets(counts))
        if chosen is None:
            lacking(counted, NO_CHOSEN, missing=[str(found["ae_chosen"])])
        elif split is None:
            lacking(counted, NO_SPLIT, missing=[str(found["ae_split"])])
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
    match, shas = labels_check(scored, snap.sha("ae_labels"), ae, seg)
    changed = snap.changed()
    manifest = {
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(full=True),
        "git_dirty": git_dirty(),
        "version": version,
        "labels_match": match,
        "labels_sha256": shas,
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
    poi = snap.csv("poi", found["poi"]) if found["poi"].is_file() else None
    tested = shots.test_shots(model.split)
    unlabelled = [s for s in tested if s not in saved]
    unstored = [_store(paths, s) for s in tested if not _store(paths, s).is_file()]
    usable = [s for s in tested if s in saved and _store(paths, s).is_file()]
    if not usable:
        return NO_SCORED_SHOT, {}

    def one(s: int) -> shots.AEShot:
        """The shot's picture, its store read once."""
        data = snap.read(f"store_{s}", _store(paths, s), keep=False)
        return shots.picture(s, label=saved[s], model=model, store=data)

    pictures = {s: one(s) for s in usable}
    ranked = [shots.rank_keys(p) for p in pictures.values()]
    f1 = {r.shot: r.f1 for r in ranked}
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
    drawn = {
        s: dataclasses.replace(pictures[s], boxes=shots.boxes_of(poi, s))
        for s in (named, *picked["example_shots"])
        if s in pictures
    }
    if named not in drawn:
        return f"{NO_NAMED}: {named}", picked
    figure("fig_interpreter", shots.draw_interpreter, drawn[named])
    figure(
        "fig_examples",
        shots.draw_examples,
        [drawn[s] for s in picked["example_shots"]],
    )
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
        "--version", default=VERSION, help=f"the models' version (default {VERSION})"
    )
    parser.add_argument(
        "--copy-to", type=Path, help="also copy the PDFs and tables here"
    )
    args = parser.parse_args(argv)
    torch.set_num_threads(LOGIN_THREADS)  # one shot at a time, on the login node
    paths = Paths.from_env()
    out = args.out or paper_dir(paths)
    manifest = build(
        paths, out, shot=args.shot, examples=args.examples, version=args.version
    )
    copied = [] if args.copy_to is None else copy_into(out, manifest, args.copy_to)
    print(
        json.dumps(
            {
                "out": str(out),
                "products": sorted(manifest["products"]),
                "skipped": manifest["skipped"],
                "copied": copied,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
