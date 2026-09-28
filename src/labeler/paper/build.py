"""Every paper product into `$LABELER_ROOT/paper/`, and the copy for the manuscript.

    PYTHONPATH=src pixi run -e labelmaker \\
        python scripts/labeler/paper/make_figures.py \\
        [--out DIR] [--shot SHOT] [--examples N] [--copy-to DIR]

reads what the round-two runs wrote (`inputs`) and draws what they allow:

- `fig_scores`, `fig_mhd`, `table_ae_scores.tex`: the AE evaluation;
- `fig_segmentation`, `table_seg_scores.tex`: the segmentation's evaluation;
- `fig_coverage`, `table_datasets.tex`: the owner's AE review, with the chosen
  model's split and the extension's summary where they exist;
- `fig_interpreter`, `fig_examples`: the gallery's index and the chosen model,
  with the points of interest where they exist.

A product whose inputs are missing is listed in `manifest.json` under `skipped`,
with the missing paths. The manifest also records each input's sha256, the
commit, the time, the interpreter's shot and the example shots, the rules that
picked them, and the drawn shots' F1 over 0-2 s and over the whole window. The
shots are ranked by their F1 over 0-2 s, the model run over every reviewed test
shot in the gallery's index (`shots.score_shot`). `--copy-to` copies this run's
PDFs and `.tex` tables into a directory (the manuscript's
`dev/label_paper/figures/`); it never runs git there, so nothing is committed
or pushed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch

from ..ae import seg as ae_seg
from ..ae import xpower
from ..ae.xpower.evaluate import chosen_model
from ..config import Paths, atomic_path, git_sha
from ..events.review import labels
from . import AE, coverage, paper_dir, scores, shots

LOGIN_THREADS = 2
COPIED = (".pdf", ".tex")


def inputs(paths: Paths) -> dict[str, Path]:
    """Where the round-two runs leave what the paper reads."""
    models = xpower.model_dir(paths)
    return {
        "ae_evaluation": models / "evaluation.json",
        "ae_chosen": models / "chosen.json",
        "ae_labels": labels.labels_path(xpower.event_dir(paths)),
        "seg_evaluation": ae_seg.model_dir(paths) / "evaluation.json",
        "summary": xpower.suggestions_dir(paths) / "summary.csv",
        "gallery_index": xpower.gallery_dir(paths) / "index.csv",
        "poi": ae_seg.poi_dir(paths) / "poi.csv",
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(x: float) -> float | None:
    return None if math.isnan(x) else round(float(x), 4)


def _write(path: Path, text: str) -> None:
    with atomic_path(path) as tmp:
        tmp.write_text(text)


def build(
    paths: Paths, out: Path, *, shot: int | None = None, examples: int = 3
) -> dict:
    """Draw every product the inputs allow into `out`; the manifest."""
    found = inputs(paths)
    made: dict[str, list[str]] = {}
    skipped: dict[str, str] = {}
    partial: dict[str, list[str]] = {}

    def ready(products: tuple[str, ...], *needs: str) -> bool:
        missing = [str(found[k]) for k in needs if not found[k].is_file()]
        for product in products if missing else ():
            skipped[product] = "missing " + ", ".join(missing)
        return not missing

    def lacking(products: tuple[str, ...], missing: list[Path]) -> None:
        """Drawn, but without `missing`: those parts say so."""
        for product in products if missing else ():
            partial[product] = [str(p) for p in missing]

    def figure(name: str, draw: Callable, *args) -> None:
        draw(*args, out / name)
        made[name] = [f"{name}.pdf", f"{name}.png"]

    def table(name: str, text: str) -> None:
        _write(out / f"{name}.tex", text)
        made[name] = [f"{name}.tex"]

    if ready(("fig_scores", "fig_mhd", "table_ae_scores"), "ae_evaluation"):
        ae = scores.read(found["ae_evaluation"])
        figure("fig_scores", scores.draw_scores, ae)
        figure("fig_mhd", scores.draw_mhd, ae)
        table("table_ae_scores", scores.table_ae(ae))
    if ready(("fig_segmentation", "table_seg_scores"), "seg_evaluation"):
        seg = scores.read(found["seg_evaluation"])
        figure("fig_segmentation", scores.draw_segmentation, seg)
        table("table_seg_scores", scores.table_segmentation(seg))
    model_file = (
        chosen_model(xpower.model_dir(paths)) if found["ae_chosen"].is_file() else None
    )
    if ready(("fig_coverage", "table_datasets"), "ae_labels"):
        split = None if model_file is None else model_file.parent / "split.csv"
        counts = {
            AE: coverage.ae_counts(
                found["ae_labels"].parent.parent, split, found["summary"]
            )
        }
        figure("fig_coverage", coverage.draw_coverage, counts)
        table("table_datasets", coverage.table_datasets(counts))
        lacking(
            ("fig_coverage", "table_datasets"),
            [
                *([found["ae_chosen"]] if split is None else []),
                *([split] if split is not None and not split.is_file() else []),
                *([] if found["summary"].is_file() else [found["summary"]]),
            ],
        )
    picked: dict = {}
    if ready(("fig_interpreter", "fig_examples"), "gallery_index", "ae_chosen"):
        index = pd.read_csv(found["gallery_index"])
        poi = pd.read_csv(found["poi"]) if found["poi"].is_file() else None
        ranked = [
            shots.score_shot(paths, s, model_file=model_file)
            for s in shots.reviewed_test_shots(index)
        ]
        f1 = {r.shot: r.f1 for r in ranked}
        mixed = {r.shot for r in ranked if r.mixed}
        picked = {
            "interpreter_shot": shot
            if shot is not None
            else shots.interpreter_shot(index, poi, f1=f1, mixed=mixed),
            "interpreter_rule": "named by --shot"
            if shot is not None
            else shots.INTERPRETER_RULE,
            "example_shots": shots.pick_examples(index, examples, f1=f1),
            "example_rule": shots.EXAMPLES_RULE,
        }
        drawn: dict[int, shots.AEShot] = {}

        def one(s: int) -> shots.AEShot:
            if s not in drawn:
                drawn[s] = shots.ae_shot(paths, s, model_file=model_file, poi=poi)
            return drawn[s]

        figure(
            "fig_interpreter", shots.draw_interpreter, one(picked["interpreter_shot"])
        )
        figure(
            "fig_examples",
            shots.draw_examples,
            [one(s) for s in picked["example_shots"]],
        )
        lacking(
            ("fig_interpreter", "fig_examples"),
            [] if poi is not None else [found["poi"]],
        )
        picked["shot_f1"] = {
            str(s): {"f1_0_2s": _number(d.f1), "f1_window": _number(d.f1_window)}
            for s, d in sorted(drawn.items())
        }
    manifest = {
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "inputs": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in found.items()
            if path.is_file()
        },
        "products": made,
        "skipped": skipped,
        "partial": partial,
        **picked,
    }
    _write(out / "manifest.json", json.dumps(manifest, indent=1) + "\n")
    return manifest


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
        "--copy-to", type=Path, help="also copy the PDFs and tables here"
    )
    args = parser.parse_args(argv)
    torch.set_num_threads(LOGIN_THREADS)  # one shot at a time, on the login node
    paths = Paths.from_env()
    out = args.out or paper_dir(paths)
    manifest = build(paths, out, shot=args.shot, examples=args.examples)
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
