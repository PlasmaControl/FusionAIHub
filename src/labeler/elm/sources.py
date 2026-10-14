"""Score the ELM review-source candidates on the reviewed shots, by divertor side.

    python -m labeler.elm.sources [--jump v2] [--run cv2] [--out-dir DIR]

The candidates for the ELM review source are suggestion tables whose present rows
(category 1) are ELMy spans: `elm_jump` (`labeler.events.elm_jump`, side-aware
jump-and-hold onsets), the D-alpha clock `elm_clock` v1 (the source since
2026-09-27) and v2, beside `elm-ours` (its out-of-fold predictions of the `--run`,
as `scripts/labeler/elm_ours_evaluate.py` scores it). They are scored as that
script scores the clock: the benchmark's 50 ms bins and its two sets
(`compare.load_sets`: `all119`, every reviewed shot with fetched inputs, and
`bes73`), a bin called when a present span touches it (`methods.span_part`), 95 %
percentile intervals over 1000 shot resamples shared by every method on a set, and
paired differences from `elm_jump`. A shot a table does not hold calls no bin.

**By side.** The `all119` shots split by the divertor side `elm_jump` read
(`per_shot.side` of its meta: "lower", "upper", or "fallback" where the strike
points could not be read), each part with its own resamples.

**Reference.** The live review labels (`review/labels.csv`, regridded to 10 ms on
2026-10-11); `--review` scores against another copy of them (the committed one
reproduces the published clock numbers). Shots without fetched inputs
(`round4/elm/inputs`) are left out, as the benchmark leaves them.

Writes `<out-dir>/evaluation.json` (default `outputs/labeler/elm/sources`).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ..config import Paths, git_sha, sha256_of
from . import compare, methods, prepare, score, train

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "outputs" / "labeler" / "elm" / "sources"
RUN = "cv2"
JUMP = "elm-jump"
#: The elm_jump version the ELM review source points to (v1 earlier on 2026-10-14).
JUMP_VERSION = "v2"
CLOCK_V2 = "elm-clock-v2"
SIDES = ("lower", "upper", "fallback")


def table(paths: Paths, method: str, version: str) -> Path:
    return (
        paths.root
        / "suggestions"
        / method
        / version
        / f"edge_localized_mode_suggest_{method}_{version}.csv"
    )


def present_spans(path: Path) -> dict[int, pd.DataFrame]:
    """A suggestion table's present rows as `methods.span_frame`s by shot."""
    frame = pd.read_csv(path)
    frame = frame[frame.category == 1].rename(
        columns={"t_start": "t_start_ms", "t_end": "t_end_ms"}
    )
    return {
        int(s): g[["t_start_ms", "t_end_ms"]].reset_index(drop=True)
        for s, g in frame.groupby("shot")
    }


def held(path: Path) -> set[int]:
    return set(pd.read_csv(path, usecols=["shot"]).shot.astype(int))


def parts_for(shots, data, bins_of, cover_of, oof, spans_by_method) -> dict:
    """Each method's `score.ShotScore` per shot, in `shots` order."""
    out: dict[str, list[score.ShotScore]] = {name: [] for name in spans_by_method}
    out[compare.NAME["ours"]] = []
    for shot in shots:
        d, bins, cover = data[shot], bins_of[shot], cover_of[shot]
        for name, found in spans_by_method.items():
            spans = found.get(shot, methods.span_frame([], []))
            out[name].append(methods.span_part(d.spans, shot, bins, cover, spans))
        out[compare.NAME["ours"]].append(
            methods.trace_part(
                d.spans, shot, bins, cover, oof.trace(shot)[0], oof.threshold[shot]
            )
        )
    return out


def summary(parts: dict, shots) -> dict:
    boot = score.draws(len(shots))
    out = {"shots": [int(s) for s in shots], "n_shots": len(shots)}
    out["bins"] = int(sum(len(p.truth) for p in parts[JUMP]))
    out.update(methods.summarise_methods(parts, boot, JUMP))
    out["per_kind"] = {
        name: methods.kind_summary(plist, boot) for name, plist in parts.items()
    }
    return out


def evaluate(paths: Paths, data: dict, oof, tables: dict[str, Path], sides) -> dict:
    """The record's `sets` and `by_side` for the reviewed shots in `data`."""
    sets = compare.load_sets(paths, data)
    found = {name: present_spans(path) for name, path in tables.items()}
    record: dict = {"sets": {}, "by_side": {}}
    for tag in ("all119", "bes73"):
        s = sets[tag]
        parts = parts_for(s.shots, data, s.bins, s.cover, oof, found)
        record["sets"][tag] = summary(parts, s.shots)
    full = sets["all119"]
    for side in SIDES:
        shots = [s for s in full.shots if sides.get(s) == side]
        if not shots:
            continue
        parts = parts_for(shots, data, full.bins, full.cover, oof, found)
        record["by_side"][side] = summary(parts, shots)
    return record


def jump_sides(path: Path) -> dict[int, str]:
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    return {int(s): v.get("side", "") for s, v in meta.get("per_shot", {}).items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jump", default=JUMP_VERSION, help="the elm_jump version")
    ap.add_argument("--run", default=RUN, help="the elm-ours `train` run")
    ap.add_argument("--review", type=Path, help="score against this labels.csv")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args(argv)
    paths = Paths.from_env()
    review = args.review or prepare.review_csv(paths)
    if args.review:
        prepare.review_csv = lambda _paths: args.review  # train.load reads it
    reviewed = set(pd.read_csv(review, usecols=["shot"]).shot.astype(int))
    inputs = {int(p.stem) for p in prepare.inputs_dir(paths).glob("*.npy")}
    data = train.load(paths, shots=sorted(reviewed & inputs))
    oof = methods.Oof(paths.root / "round4" / "elm" / "cv" / args.run)
    tables = {
        JUMP: table(paths, "elm_jump", args.jump),
        compare.NAME["clock"]: table(paths, "elm_clock", "v1"),
        CLOCK_V2: table(paths, "elm_clock", "v2"),
    }
    sides = jump_sides(tables[JUMP])
    record = {
        "git": git_sha(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "reference": str(review),
        "reference_sha256": sha256_of(review),
        "shots_without_inputs": sorted(reviewed - inputs),
        "run": args.run,
        "tables": {
            name: {
                "path": str(path),
                "sha256": sha256_of(path),
                "reviewed_shots_held": len(held(path) & set(data)),
            }
            for name, path in tables.items()
        },
        "sides": {
            side: sum(sides.get(s) == side for s in data) for side in (*SIDES, "")
        },
        **evaluate(paths, data, oof, tables, sides),
    }
    clock = record["sets"]["all119"]["methods"][compare.NAME["clock"]]["point"]["f1"]
    jump = record["sets"]["all119"]["methods"][JUMP]["point"]["f1"]
    record["decision"] = {
        "rule": "move the ELM review pointer to elm_jump if its all119 F1 beats "
        "elm_clock v1's",
        "elm_jump_f1": jump,
        "elm_clock_v1_f1": clock,
        "elm_jump_beats_clock_v1": bool(jump > clock),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "evaluation.json").write_text(json.dumps(record, indent=1) + "\n")
    for scope, block in (
        *((f"set {k}", v) for k, v in record["sets"].items()),
        *((f"side {k}", v) for k, v in record["by_side"].items()),
    ):
        print(scope, block["n_shots"], "shots", block["bins"], "bins")
        for name, res in block["methods"].items():
            ci = res.get("ci95", {}).get("f1")
            print(
                f"  {name:15s} F1 {res['point']['f1']:.3f}",
                f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "",
                f"P {res['point']['precision']:.3f} R {res['point']['recall']:.3f}",
            )
    print(json.dumps(record["decision"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
