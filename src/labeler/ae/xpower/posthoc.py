"""The AE model's post-hoc record, computed after its one test.

    python -m labeler.ae.xpower.posthoc --version v2 [--limit N]

**Post hoc.** Everything here is computed after the version's test, from the
same model, threshold and frames the test scored, and no decision depends on
it: the choice, the bar's verdict and the tier stand as `chosen.json` and
`evaluation.json` record them. It writes `posthoc.json` and `posthoc.md` in
`$LABELER_ROOT/models/ae_xpower/<version>/`, and nothing else; the test record
(`evaluation.*`) is never written, and `evaluate --test` still refuses a second
scoring. Both files open with that statement (`POST_HOC`). A pilot (`--limit N`,
the first N test shots, N <= 20) writes only under
`$LABELER_ROOT/runs/ae_xpower/posthoc/<version>/`. Run again, it rewrites the
two files (after the seed study, say); nothing reads them to decide anything.

**The predictions.** P(AE) is recomputed from `model.pt` for the split's test
shots, through the test's own `evaluate.load_test`, so with every check the
test makes of its inputs. What the test record names must be what is read now:
the model, `chosen.json`, the split, the labels (the directory's copy and the
ones the model learned from) and the source table by sha256, the candidate,
threshold and band (`evaluate.identity`), and a cross-validated version's
checks; otherwise it refuses, naming the key. The record must be the full
test's (`limit` 0), and a full run's frames must count what the test's did.

**(a) MHD false positives by place.** The test's MHD frames (the owner says
absent and TokEye sees a 0-60 kHz line) by where they lie in the shot's 0-2 s
(`PLACES`): the lead-in, before the owner's first present frame; inside AE,
between the first and last present frames; after AE, after the last; and the
shots with no present frame. The owner's states are read over all of 0-2 s;
only scored frames count. For each place and each method the test scores: the
false positives, the frames and the rate with its 95 % shot-bootstrap interval
(`labeler.scoring.stats`, 2000 replicates, seed 20260923, as the test), and the
paired difference of each method's rate from SELDNet's.

**(b) Prevalence.** The share of scored frames the owner calls present, and of
MHD frames the owner calls absent, in the pooled out-of-fold frames (`cv/`, the
chosen candidate's five folds, `cv.oof_frames`) and in the test frames.

**(c) The seed study** (`cv.seed_study`): per training seed, the pooled
out-of-fold scores at the chosen threshold, and their spread beside the A1
margin (0.03) and the test's miss of it; "not run" while no study seed has
all five folds.

**(d) The second look** (`evaluate.second_look`): how many test shots were the
earlier version's (`evaluate.SUBSET_OF`) test shots, with that version's model
and when its test was scored, from its records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from ...config import Paths, atomic_path, git_sha
from ...events.catalog.states import PRESENT
from . import CV_VERSIONS, PILOT_MAX, cv, evaluate, model_dir
from .train import read_split

POST_HOC = (
    "Post hoc: computed after the test, from the same model, threshold and "
    "frames it scored. No decision depends on it."
)
#: Where an MHD frame lies in its shot's 0-2 s, by the owner's present frames.
PLACES = {
    "lead_in": "before the owner's first present frame (lead-in)",
    "inside": "between the owner's first and last present frames (inside AE)",
    "after": "after the owner's last present frame (after AE)",
    "no_present": "in shots with no present frame",
}
METRICS = ("f1", "precision", "recall", "fp_rate_mhd")


def out_dir(paths: Paths, version: str, limit: int) -> Path:
    """The version's models directory; a pilot's own directory under `runs/`."""
    if limit:
        return paths.runs / "ae_xpower" / "posthoc" / version
    return model_dir(paths, version)


def places(frames: evaluate.ShotFrames) -> dict[str, np.ndarray]:
    """Each place's frames of one shot, by the owner's first and last present
    frames over all of its 0-2 s."""
    n = len(frames.owner)
    at = np.arange(n)
    present = np.flatnonzero(np.asarray(frames.owner) == PRESENT)
    none = np.zeros(n, dtype=bool)
    if not len(present):
        return {**dict.fromkeys(PLACES, none), "no_present": np.ones(n, dtype=bool)}
    first, last = present[0], present[-1]
    return {
        "lead_in": at < first,
        "inside": (at > first) & (at < last),
        "after": at > last,
        "no_present": none,
    }


def by_place(shots, methods=evaluate.METHODS) -> dict:
    """(a): per place, the MHD frames and each method's false positives on
    them, its rate with interval, and its paired difference from SELDNet's."""
    out = {}
    for place, text in PLACES.items():

        def where(f, place=place):
            return evaluate.mhd_absent(f) & places(f)[place]

        per_shot = [int((f.scored & where(f)).sum()) for f in shots]
        entry = {
            "where": text,
            "mhd_frames": sum(per_shot),
            "shots": sum(n > 0 for n in per_shot),
            "methods": {},
            "minus_seldnet": {},
        }
        theirs = (
            evaluate.cells(shots, "seldnet", where) if "seldnet" in methods else None
        )
        for m in methods:
            c = evaluate.cells(shots, m, where)
            entry["methods"][m] = {
                "fp": int(c[:, 1].sum()),
                "frames": int((c[:, 1] + c[:, 3]).sum()),
                "rate": evaluate._estimate(c, evaluate.fp_rate),
            }
            if theirs is not None and m != "seldnet":
                entry["minus_seldnet"][m] = evaluate._difference(
                    c, theirs, evaluate.fp_rate
                )
        out[place] = entry
    return out


def prevalence(shots) -> dict:
    """(b): the share of scored frames present, and MHD with the owner absent."""
    n = evaluate.frame_counts(shots)
    scored = n["scored"]
    return {
        **{k: n[k] for k in ("shots", "scored", "present", "mhd_absent")},
        "present_fraction": n["present"] / scored if scored else None,
        "mhd_fraction": n["mhd_absent"] / scored if scored else None,
    }


def checked_inputs(paths: Paths, models: Path, version: str, record: dict):
    """The test's inputs as `evaluate.load_test` reads and checks them, refused
    unless the test record names each of them as it is now."""
    file = models / "evaluation.json"
    meta = record.get("meta", {})
    if meta.get("limit") != 0:
        raise ValueError(f"{file}: not a full test (limit {meta.get('limit')!r})")
    inputs = evaluate.load_test(paths, models, version)
    now = json.loads(json.dumps(evaluate.identity(inputs, version) | inputs.cv_meta))
    for key, value in now.items():
        if meta.get(key) != value:
            raise ValueError(
                f"{file}: {key} {meta.get(key)!r} is not what is read now "
                f"({value!r}); the post-hoc record is of the model the test scored"
            )
    return inputs


def seed_summary(paths: Paths, version: str, record: dict) -> dict:
    """(c): `cv.seed_study`, beside the A1 margin and the test's miss of it."""
    study = cv.seed_study(paths, version)
    bound = evaluate.BAR["f1_vs_seldnet_low"]
    low = record["differences"]["f1_minus_seldnet"]["low"]
    ran = [r for r in study["rows"] if r["seed"] != study["default_seed"]]
    return study | {
        "status": "run" if ran else "not run",
        "a1_margin": -bound,
        "test_f1_minus_seldnet_low": low,
        "test_missed_by": None if low is None else bound - low,
    }


def look(paths: Paths, version: str, test: list[int], chosen: dict) -> dict | None:
    """(d): the second look, from the earlier version's test record and split."""
    earlier = evaluate.SUBSET_OF.get(version)
    if earlier is None:
        return None
    models = model_dir(paths, earlier)
    file = models / "evaluation.json"
    if not file.is_file():
        return {"version": earlier, "said": f"not counted: {file} is missing"}
    record = json.loads(file.read_text())
    split = read_split(models / record["meta"]["candidate"] / "split.csv")
    return evaluate.second_look(version, test, record, split, chosen)


def run(paths: Paths, version: str, limit: int = 0) -> dict:
    """Read, check, recompute, and write `posthoc.json` and `posthoc.md`."""
    if version not in CV_VERSIONS:
        raise ValueError(f"version {version} was not cross-validated; no record")
    if not 0 <= limit <= PILOT_MAX:
        raise ValueError(f"--limit is 0 (all) or a pilot's 1 to {PILOT_MAX}")
    models = model_dir(paths, version)
    file = models / "evaluation.json"
    if not file.is_file():
        raise FileNotFoundError(f"{file}: no test yet; the record follows the test")
    data = file.read_bytes()
    record = json.loads(data)
    inputs = checked_inputs(paths, models, version, record)
    test = sorted(s for s, v in inputs.split.items() if v == "test")
    shots, _ = evaluate.frames_of_test(
        paths, inputs, test[:limit] if limit else test, whole_window=False
    )
    counts = evaluate.frame_counts(shots)
    if not limit and counts != record.get("frames"):
        raise ValueError(f"{file}: the test counted other frames ({counts})")
    out = {
        "post_hoc": POST_HOC,
        "version": version,
        "limit": limit,
        "evaluation": str(file),
        "evaluation_sha256": hashlib.sha256(data).hexdigest(),
        "identity": {k: record["meta"][k] for k in ("candidate", "threshold")},
        "frames": counts,
        "places": by_place(shots),
        "prevalence": {
            "out_of_fold": prevalence(cv.oof_frames(paths, version)),
            "test": prevalence(shots),
        },
        "seed_study": seed_summary(paths, version, record),
        "second_look": look(paths, version, test, json.loads(inputs.chosen_bytes)),
        "git_sha": git_sha(),
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if file.read_bytes() != data:
        raise ValueError(f"{file}: changed while the record was computed")
    where = out_dir(paths, version, limit)
    where.mkdir(parents=True, exist_ok=True)
    with atomic_path(where / "posthoc.json") as tmp:
        tmp.write_text(json.dumps(out, indent=1) + "\n")
    with atomic_path(where / "posthoc.md") as tmp:
        tmp.write_text(posthoc_md(out))
    return out


def _num(x, spec=".3f") -> str:
    return "n/a" if x is None else format(x, spec)


def posthoc_md(record: dict) -> str:
    """`posthoc.md`: the record's four parts, headed as post hoc."""
    n, ident = record["frames"], record["identity"]
    lines = [
        f"# AE {record['version']}: post hoc",
        "",
        f"**{POST_HOC}**",
        "",
        (
            f"The test's model {ident['candidate']} at {ident['threshold']}, "
            f"recomputed on {n['shots']} test shots ({n['scored']} frames of 0-2 s, "
            f"{n['present']} present, {n['mhd_absent']} MHD frames the owner called "
            f"absent)"
            + (f"; a pilot of {record['limit']}" if record["limit"] else "")
            + f". Test record sha256 {record['evaluation_sha256'][:12]}..."
        ),
        "",
        "## (a) MHD false positives by place",
        "",
        (
            "95 % shot-bootstrap intervals; the difference is the method's rate "
            "minus SELDNet's on the same shots."
        ),
        "",
    ]
    for entry in record["places"].values():
        lines += [
            (
                f"**MHD frames {entry['where']}:** {entry['mhd_frames']} on "
                f"{entry['shots']} shots."
            ),
            "",
            "| method | false positives | rate [95 %] | minus SELDNet [95 %] |",
            "|---|---|---|---|",
        ]
        for m, s in entry["methods"].items():
            diff = entry["minus_seldnet"].get(m)
            lines.append(
                f"| {m} | {s['fp']} of {s['frames']} | {evaluate._fmt(s['rate'])} | "
                f"{evaluate._fmt(diff) if diff else '-'} |"
            )
        lines.append("")
    lines += [
        "## (b) Prevalence",
        "",
        "| frames | shots | scored | present | MHD, owner absent |",
        "|---|---|---|---|---|",
    ]
    for name, p in record["prevalence"].items():
        lines.append(
            f"| {name.replace('_', '-')} | {p['shots']} | {p['scored']} | "
            f"{_num(p['present_fraction'])} | {_num(p['mhd_fraction'])} |"
        )
    study = record["seed_study"]
    lines += ["", "## (c) The seed study", ""]
    if study["status"] == "not run":
        lines += [f"Not run: seeds {study['seeds']} have no complete folds.", ""]
    lines += [
        (
            f"{study['candidate']}'s pooled out-of-fold frames at "
            f"{study['threshold']}, per training seed:"
        ),
        "",
        "| seed | F1 | precision | recall | MHD FP | F1 >= 0.90 |",
        "|---|---|---|---|---|---|",
    ]
    for r in study["rows"]:
        lines.append(
            f"| {r['seed']} | {_num(r['f1'])} | {_num(r['precision'])} | "
            f"{_num(r['recall'])} | {_num(r['fp_rate_mhd'])} | "
            f"{'yes' if r['meets_f1_min'] else 'no'} |"
        )
    lines.append("")
    for key, s in study["spread"].items():
        lines.append(f"- {key}: range {s['range']:.4f}, std {s['std']:.4f} (ddof 1)")
    if study["incomplete"]:
        lines.append(f"- incomplete (folds missing): {study['incomplete']}")
    missed = study["test_missed_by"]
    how = (
        "has no lower bound"
        if missed is None
        else f"missed by {missed:.4f}"
        if missed > 0
        else f"met with {-missed:.4f} to spare"
    )
    lines += [
        "",
        (
            f"Beside them: A1's margin, {study['a1_margin']:.2f} (the F1 "
            "difference from SELDNet must have a lower bound of at least "
            f"-{study['a1_margin']:.2f}); the test's lower bound, "
            f"{_num(study['test_f1_minus_seldnet_low'], '.4f')}, {how}."
        ),
        "",
        "## (d) The second look",
        "",
    ]
    second = record["second_look"]
    lines += [second["said"] + "." if second else "No earlier version.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--version", default="v2")
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="a pilot's first N test shots, into runs/ae_xpower/posthoc/<version>",
    )
    args = p.parse_args(argv)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    paths = Paths.from_env()
    try:
        record = run(paths, args.version, args.limit)
    except (OSError, ValueError, KeyError) as error:
        p.error(str(error))
    where = out_dir(paths, args.version, args.limit)
    print(
        f"wrote {where / 'posthoc.json'}; seed study {record['seed_study']['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
