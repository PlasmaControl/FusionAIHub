r"""fig2_coverage: how much labelled data each label set holds, legacy against
Tokamak-SI, for the coverage row of Figure 2 (`fig_benchmarks.py`).

    PYTHONPATH=src pixi run --frozen -e labelmaker \
        python scripts/labeler/paper/fig2_coverage.py [--out PATH]

One definition for every set (`DEFINITION`): a set's labelled time is, shot by
shot, the union of the intervals its table states present or absent (catalog
states 1 and 0; the four regimes of confinement); intervals stated uncertain,
not observable or unassessed are not labelled time, and a shot is labelled when
it has any. Tokamak-SI counts every released tier and every split; the reviewed
subset counts the shots whose labels a person reviewed (the review store's
`history.jsonl` names the reviewer). Every count is computed here from the label
CSV inventory (`data/events/<category>/...`, the committed tables), except where
a stream already wrote a record that carries it (`docs/labeler/figure2_tm.json`,
the Smith evaluation, the detachment record): those are read, never retyped.

A set that cannot meet the definition says so in its `definition` and `notes`
and the figure's CSV carries the same note; nothing is changed silently:

- AE: legacy time is the whole annotated window, so its absent time is
  unannotated time (the Heidbrink annotation under-counts), not reviewed absence;
- ELMs: legacy is D. Smith's windows (1 ms cells selected around known ELMs);
- tearing modes: the legacy label is the Seo growth-phase label's support, and the
  Tokamak-SI count is the stream's record (10 ms bins from the measured plasma
  start), not the union of the interval table;
- RWM: legacy is a list of onset points (shots only, no time); Tokamak-SI absent
  time is assumed before the first precursor;
- sawtooth: no legacy label set exists (the published detectors are from another
  machine and ship no table); Tokamak-SI is the released population labels, whose
  shards are verified against the manifest's digests, and the reviewed subset is
  the three shots a person drew spans on;
- detachment: no legacy set, no human review; the record is the detachment
  stream's committed `docs/labeler/figure2_detach.json`.

The record, `docs/labeler/figure2_coverage.json`, is what the figure draws.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
TABLES = REPO / "data" / "events"
OUT = REPO / "docs" / "labeler" / "figure2_coverage.json"
MAIN = "/scratch/gpfs/nc1514/FusionAIHub"
#: the curated confinement intervals live in the main checkout's run directory
CONFINEMENT_INTERVALS = Path(
    os.environ.get(
        "CONFINEMENT_INTERVALS",
        f"{MAIN}/runs/labeler/confinement/v1/merged_intervals.csv",
    )
)
TM_RECORD = REPO / "docs" / "labeler" / "figure2_tm.json"
SMITH_RECORD = REPO / "outputs" / "labeler" / "elm" / "smith" / "evaluation.json"
#: the detachment stream's record, committed in this repository
DETACH_RECORD = Path(
    os.environ.get(
        "LABELER_DETACH_RECORD", REPO / "docs" / "labeler" / "figure2_detach.json"
    )
)
#: the sawtooth stream's label manifest (the shards' digests) and, in the
#: checkout, the untracked integration copy of the shards
SAW_MANIFEST = (
    REPO / "outputs" / "labeler" / "sawtooth" / "fix5" / "label_manifest.json"
)
SAW_TABLES = TABLES / "sawtooth_oscillation"
#: x order of the figure
SETS = ("ae", "confinement", "elm", "tm", "sawtooth", "rwm", "detachment")
NAMES = {
    "ae": "AE",
    "confinement": "confinement",
    "elm": "ELMs",
    "tm": "tearing modes",
    "sawtooth": "sawtooth",
    "rwm": "RWM",
    "detachment": "detachment",
}
DEFINITION = (
    "Labelled time is, shot by shot, the union of the intervals a label table states "
    "present or absent (the four regimes for confinement); intervals stated "
    "uncertain, not observable or unassessed are not labelled time. Labelled shots "
    "are the shots with any labelled time. Tokamak-SI counts every released tier and "
    "split; the reviewed subset counts the shots whose labels a person reviewed."
)
SHARED = "shared definition"
MS_PER_S = 1000.0
sources: dict[str, str] = {}


def note_source(path: Path | str, digest: str | None = None) -> str:
    """The path as the record names it (repo-relative when inside), with its
    sha256 kept for the record."""
    path = Path(path)
    name = str(path.relative_to(REPO)) if REPO in path.parents else str(path)
    if digest is None and path.is_file():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sources[name] = digest or "not a file"
    return name


def table(rel: str) -> Path:
    """A committed label table: the checkout's own first, then the label tables
    root of the environment."""
    from labeler.config import Paths

    for root in (TABLES, Paths.from_env().label_tables):
        if (root / rel).is_file():
            return root / rel
    raise FileNotFoundError(f"label table {rel} not found")


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def union_seconds(frame: pd.DataFrame) -> pd.Series:
    """Per shot, the length in seconds of the union of its intervals (ms in the
    tables). Overlaps count once; a point has no length."""
    out = {}
    for shot, rows in frame.groupby("shot"):
        starts = rows.t_start.to_numpy(float)
        ends = rows.t_end.to_numpy(float)
        order = np.argsort(starts, kind="stable")
        starts, ends = starts[order], ends[order]
        total, cur_s, cur_e = 0.0, starts[0], ends[0]
        for a, b in zip(starts[1:], ends[1:]):
            if a <= cur_e:
                cur_e = max(cur_e, b)
            else:
                total += cur_e - cur_s
                cur_s, cur_e = a, b
        total += cur_e - cur_s
        out[shot] = total / MS_PER_S
    return pd.Series(out, dtype=float)


def labelled(frame: pd.DataFrame, states, shots=None) -> dict:
    """Shots and seconds the intervals in `states` cover (on `shots` if given)."""
    rows = frame[frame.category.isin(list(states))]
    if shots is not None:
        rows = rows[rows.shot.isin(list(shots))]
    if rows.empty:
        return {"shots": 0, "seconds": 0.0}
    seconds = union_seconds(rows)
    seconds = seconds[seconds > 0]
    return {"shots": len(seconds), "seconds": round(float(seconds.sum()), 3)}


def reviewed_shots(history: Path) -> set[int]:
    """The shots a person saved a review of, from the store's history."""
    note_source(history)
    return {json.loads(line)["shot"] for line in open(history) if line.strip()}


def side(status: str, label: str, definition: str, paths, **counts) -> dict:
    """One setting of one set. `definition` is "shared definition" (optionally
    "... on <what is specific>") when the count follows DEFINITION, else it says how
    the count departs from it."""
    return {
        "status": status,
        "label": label,
        "shots": counts.pop("shots", None),
        "seconds": counts.pop("seconds", None),
        "shared_definition": definition.startswith(SHARED),
        "definition": definition,
        "paths": [note_source(p) for p in paths],
        **counts,
    }


def reviewed_block(status: str, shots=None, seconds=None, note: str = "") -> dict:
    return {"status": status, "shots": shots, "seconds": seconds, "note": note}


def ae() -> dict:
    legacy_path = table("alfven_eigenmode/format/alfven_eigenmode_format_2026_v1.csv")
    review = table("alfven_eigenmode/review/labels.csv")
    history = review.with_name("history.jsonl")
    legacy_rows = pd.read_csv(legacy_path)
    rows = pd.read_csv(review)
    si = labelled(rows, (0, 1))
    seen = reviewed_shots(history)
    rev = labelled(rows, (0, 1), seen)
    present = labelled(legacy_rows, (1,))
    ece_path = table("alfven_eigenmode/format/alfven_eigenmode_ece_format_2026_v1.csv")
    ece = labelled(pd.read_csv(ece_path), (0, 1))
    return {
        "legacy": side(
            "measured",
            "Heidbrink annotation",
            "the annotated shots' whole windows; unannotated time is "
            "listed absent, which the annotation under-counts, so legacy absent "
            "time is not reviewed absence",
            [legacy_path],
            annotated_present_shots=present["shots"],
            annotated_present_seconds=present["seconds"],
            **labelled(legacy_rows, (0, 1)),
        ),
        "tokamak_si": side(
            "measured",
            "expert-reviewed AE labels",
            SHARED,
            [review, history],
            reviewed=reviewed_block(
                "measured",
                rev["shots"],
                rev["seconds"],
                "every shot of the table was saved by a reviewer",
            ),
            **si,
        ),
        "notes": [
            (
                "the ECE hand-drawn boxes (a second legacy annotation, on another "
                "diagnostic) are not counted"
            )
        ],
        "alternatives": [
            {
                "name": "ECE hand-drawn boxes (second legacy annotation)",
                "shots": ece["shots"],
                "seconds": ece["seconds"],
                "paths": [note_source(ece_path)],
            }
        ],
    }


def confinement() -> dict:
    roster = table("confinement/extend_confine_ours/roster.csv")
    meta = roster.with_suffix(".meta.json")
    cohort = table("catalog/cohort.csv")
    blind = set(pd.read_csv(cohort).query("split == 'test'").shot)
    curated = pd.read_csv(CONFINEMENT_INTERVALS)
    curated = curated[
        curated.regimes.isin(["L", "H", "QH", "WP"]) & ~curated.shot.isin(blind)
    ].assign(category=1)
    rows = pd.read_csv(roster)
    return {
        "legacy": side(
            "measured",
            "Gill and Butt tables merged",
            "the curated L, H, QH and WPQH intervals of the merged table, less the "
            "blind split (never read) and the one H/L conflict interval",
            [CONFINEMENT_INTERVALS, cohort],
            **labelled(curated, (1,)),
        ),
        "tokamak_si": side(
            "experimental",
            "confine-ours roster",
            "shared definition on classes 1 to 4 (high, low, qh, wpqh) of the "
            "roster; class 5 (below the confidence floor) is uncertain; beam-on "
            "time only, other time is not assessed",
            [roster, meta],
            reviewed=reviewed_block(
                "none", note="the roster is a network's reading; no person reviewed it"
            ),
            **labelled(rows, (1, 2, 3, 4)),
        ),
        "notes": [
            (
                "Tokamak-SI is the experimental, unreviewed roster: segments are the "
                "network's reading of D-alpha, density, beta_N, W_MHD and beam power, "
                "labels and inputs share the 0D traces"
            ),
        ],
        "alternatives": [],
    }


def elm() -> dict:
    review = table("edge_localized_mode/review/labels.csv")
    history = review.with_name("history.jsonl")
    onset = table("edge_localized_mode/format/edge_localized_mode_format_2026_v1.csv")
    smith = load_json(SMITH_RECORD)
    cell = "1ms cells" in smith["protocol"]["occupancy"]
    if not cell:  # the record's cell width is what turns bins into seconds
        raise ValueError("Smith record no longer says its cells are 1 ms")
    rows = pd.read_csv(review)
    seen = reviewed_shots(history)
    rev = labelled(rows, (0, 1), seen)
    old = labelled(pd.read_csv(onset), (0, 1))
    return {
        "legacy": side(
            "measured",
            "D. Smith's windows",
            "the 1 ms cells wholly inside a Smith window, shots with a hand-labelled "
            "ELM; windows are selected around known ELMs, so absent time is not "
            "continuous-discharge absence",
            [SMITH_RECORD],
            shots=int(smith["shots"]),
            seconds=round(smith["counts"]["bins"] / MS_PER_S, 3),
            smith_hand_onsets=int(smith["counts"]["hand_onsets"]),
        ),
        "tokamak_si": side(
            "measured",
            "reviewed ELM spans",
            SHARED,
            [review, history],
            reviewed=reviewed_block(
                "measured",
                rev["shots"],
                rev["seconds"],
                "every shot of the table was saved by a reviewer",
            ),
            **labelled(rows, (0, 1)),
        ),
        "notes": [
            (
                "legacy is Smith's windows as the benchmark uses them; the older WPQH "
                "onset table is the other legacy source (alternatives)"
            )
        ],
        "alternatives": [
            {
                "name": (
                    "legacy onset table (Hiro's ELM labels, bins over a fixed window)"
                ),
                "shots": old["shots"],
                "seconds": old["seconds"],
                "paths": [note_source(onset)],
            }
        ],
    }


def tm() -> dict:
    record = load_json(TM_RECORD)
    cover = record["coverage"]
    cohort_csv = table("neoclassical_tearing_mode/extend_tm_interval/tm_interval.csv")
    union = labelled(pd.read_csv(cohort_csv), (0, 1))
    tm_path = note_source(TM_RECORD)
    return {
        "legacy": side(
            "measured",
            "Seo growth-phase labels",
            "deviation: " + cover["legacy_seo"]["definition"],
            [TM_RECORD],
            shots=cover["legacy_seo"]["labelled_shots"],
            seconds=round(cover["legacy_seo"]["labelled_seconds"], 3),
            record_key="coverage.legacy_seo",
        ),
        "tokamak_si": side(
            "measured",
            "interval labels, population",
            "deviation: the stream's record, " + cover["ours_population"]["definition"],
            [TM_RECORD],
            shots=cover["ours_population"]["labelled_shots"],
            seconds=round(cover["ours_population"]["labelled_seconds"], 3),
            record_key="coverage.ours_population",
            reviewed=reviewed_block(
                "none", note="a rule's output; no person reviewed the intervals"
            ),
        ),
        "notes": [
            (
                "Tokamak-SI is the interval labeller's output on the population, "
                "scored as development cross-validation; the blind shots carry no "
                "tearing-mode labels"
            ),
            (
                "legacy and Tokamak-SI time are not on one definition: the legacy "
                "figure is the label support on observable bins, the Tokamak-SI figure "
                "the labelled bins from the measured plasma start"
            ),
        ],
        "alternatives": [
            {
                "name": "development cohort (record)",
                "shots": cover["ours"]["labelled_shots"],
                "seconds": round(cover["ours"]["labelled_seconds"], 3),
                "paths": [tm_path],
                "record_key": "coverage.ours",
            },
            {
                "name": "development cohort (union of the interval table)",
                "shots": union["shots"],
                "seconds": union["seconds"],
                "paths": [note_source(cohort_csv)],
            },
            {
                "name": "survival legacy labels",
                "shots": cover["legacy_survival"]["labelled_shots"],
                "seconds": round(cover["legacy_survival"]["labelled_seconds"], 3),
                "paths": [tm_path],
                "record_key": "coverage.legacy_survival",
            },
        ],
    }


def saw_shards() -> tuple[list[Path], dict]:
    """The released population label shards and the manifest that digests them.
    The checkout's integration copy is used when it holds the shards, else the
    label store the manifest names; each shard must match its digest."""
    from labeler.config import Paths

    manifest = load_json(SAW_MANIFEST)
    entries = [f for f in manifest["files"] if f["file"].startswith("population-")]
    folders = (
        SAW_TABLES / "extend_saw_physics",
        Path(manifest["labels_path"]),
        Paths.from_env().root / "round4" / "saw" / "fix5" / "labels",
    )
    folder = next((f for f in folders if (f / entries[0]["file"]).is_file()), None)
    if folder is None:
        raise FileNotFoundError("the sawtooth population label shards are not found")
    shards = [folder / e["file"] for e in entries]
    for shard, entry in zip(shards, entries):
        if hashlib.sha256(shard.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"{shard} does not match the manifest's digest")
    return shards, manifest


def sawtooth() -> dict:
    shards, manifest = saw_shards()
    frame = pd.concat(
        [
            pd.read_csv(f, usecols=["shot", "category", "t_start", "t_end"])
            for f in shards
        ]
    )
    review = table("sawtooth_oscillation/review/labels.csv")
    history = review.with_name("history.jsonl")
    seen = reviewed_shots(history)
    rev = labelled(pd.read_csv(review), (0, 1), seen)
    return {
        "legacy": side(
            "none",
            "no legacy label set",
            "the published sawtooth detectors are from another machine and ship no "
            "label table",
            [],
        ),
        "tokamak_si": side(
            "measured",
            "released population labels",
            "shared definition on the physics-rule labels: present (a periodic "
            "relaxation train) and absent (ECE-tested absence) count; the q-prior "
            "states, other uncertain time and unassessed time do not",
            [SAW_MANIFEST, review, history],
            reviewed=reviewed_block(
                "measured",
                rev["shots"],
                rev["seconds"],
                "the shots a person drew spans on, anchored to earlier rule suggestions",
            ),
            shards=len(shards),
            shards_sha256sums=manifest["manifest_sha256"],
            **labelled(frame, (0, 1)),
        ),
        "notes": [
            (
                "the labels are rule output, unvalidated against an independent "
                "truth; only the reviewed spans are human-drawn"
            )
        ],
        "alternatives": [],
    }


def rwm() -> dict:
    legacy_path = table("resistive_wall_mode/format/rwm_format_2026_v1.csv")
    windows = table("resistive_wall_mode/extend_rwm_onset_window/rwm_windows.csv")
    meta = windows.with_suffix(".meta.json")
    onsets = pd.read_csv(legacy_path)
    return {
        "legacy": side(
            "point_events",
            "Hanson onset list",
            "onset times only: a shot count, no labelled time",
            [legacy_path],
            shots=int(onsets.shot.nunique()),
            seconds=None,
        ),
        "tokamak_si": side(
            "measured",
            "onset windows",
            "shared definition on the Hanson shots: present is one minimal slice "
            "from each listed onset; absent is assumed before the first "
            "precursor; the pre-onset window, the screened comparison shots "
            "and the unassessed time are not labelled",
            [windows, meta],
            reviewed=reviewed_block(
                "none", note="a completeness assumption; no person reviewed the spans"
            ),
            **labelled(pd.read_csv(windows), (0, 1)),
        ),
        "notes": [
            (
                "present time is a convention (one slice per onset), not a measured "
                "duration; absent time rests on a completeness assumption"
            )
        ],
        "alternatives": [],
    }


def detach_record() -> tuple[dict | None, str | None]:
    """The detachment stream's committed record (`DETACH_RECORD`), and the name it
    is cited by; none when the file is absent."""
    if not DETACH_RECORD.is_file():
        return None, None
    return load_json(DETACH_RECORD), note_source(DETACH_RECORD)


def detachment() -> dict:
    record, name = detach_record()
    none = side("none", "no legacy label set", "no earlier label set exists", [])
    if record is None:
        pending = side(
            "pending",
            "detachment labels",
            "the detachment stream's record is not committed",
            [],
            reviewed=reviewed_block("pending"),
        )
        return {
            "legacy": none,
            "tokamak_si": pending,
            "notes": ["pending the detachment record"],
            "alternatives": [],
        }
    cover = record["coverage"]["labelled_certain_or_tangtv_only"]
    return {
        "legacy": none,
        "tokamak_si": {
            **side(
                "measured",
                "detachment labels",
                (
                    "deviation: the bins that carry a state (TangTV, with or without "
                    "an agreeing second indicator); bins with no vote or a conflict "
                    "are not labelled"
                ),
                [],
                shots=cover["shots"],
                seconds=round(cover["seconds"], 3),
                record_key="coverage.labelled_certain_or_tangtv_only",
                reviewed=reviewed_block(
                    "none", note="indicator votes; no person reviewed the bins"
                ),
            ),
            "paths": [name],
        },
        "notes": [
            (
                "the record is the detachment stream's committed file; its tiers "
                "are TangTV with an agreeing Afrac vote and TangTV only"
            )
        ],
        "alternatives": [
            {
                "name": "assessed bins (a state or not)",
                "shots": record["coverage"]["assessed"]["shots"],
                "seconds": round(record["coverage"]["assessed"]["seconds"], 3),
                "paths": [name],
                "record_key": "coverage.assessed",
            }
        ],
    }


BUILDERS = {
    "ae": ae,
    "confinement": confinement,
    "elm": elm,
    "tm": tm,
    "sawtooth": sawtooth,
    "rwm": rwm,
    "detachment": detachment,
}


def head() -> str:
    run = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return run.stdout.strip() or "unknown"


def build() -> dict:
    sources.clear()
    entries = {name: BUILDERS[name]() for name in SETS}
    for name, entry in entries.items():
        entry["name"] = NAMES[name]
    return {
        "schema": "figure2_coverage",
        "made_by": "scripts/labeler/paper/fig2_coverage.py",
        "git": head(),
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "definition": DEFINITION,
        "units": {"shots": "shots", "seconds": "seconds of labelled time"},
        "order": list(SETS),
        "sets": entries,
        "sources": dict(sorted(sources.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    record = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(record, f, indent=1)
        f.write("\n")
    for name, entry in record["sets"].items():
        parts = []
        for side_name in ("legacy", "tokamak_si"):
            s = entry[side_name]
            parts.append(
                f"{side_name} {s['status']} {s['shots']} shots {s['seconds']} s"
            )
        print(f"{name:12s} " + "; ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
