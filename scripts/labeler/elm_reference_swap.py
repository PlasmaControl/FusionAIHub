#!/usr/bin/env python
"""The reference swap for ELMs: score the ELM methods against two references.

    python scripts/labeler/elm_reference_swap.py --run cv2 [--out-dir DIR]

Needs `elm_ours_evaluate.py`'s predictions (`train` run `--run`) and
`elm_dsm_evaluate.py`'s scores (`$LABELER_ROOT/round4/elm/dsm/`).

**1. Overlap, stated first.** Which reviewed shots each legacy ELM reference covers:
Hiro's table of 50 ms onset bins, the shot-level `elm_all_ground_truth`, D. Smith's
labelled windows.

**2. Conversion.** Hiro's table is converted to the scored bins as described in
`labeler.elm.swap`; the bins are those of `elm_dsm_evaluate.py` (50 ms bins wholly
inside one reviewed absent, non-crowd or crowd span, in analysed time, with the DSM's
rows), so every method sits on the same bins under both references.

**3. Finding 1.** What the legacy table misses: `|M|` (review-present bins it marks
absent), `|P|` (legacy-present bins the review marks absent), and its recall and
precision against the review.

**4. Finding 2.** `elm-ours`, `elm-dsm` (DSM refit, limited inputs (60 of the original 124) and as a detector), `elm-elmo`
and `elm-clock` against both references: AUROC and F1 with 95 % shot-bootstrap
intervals, the order each reference gives, and whether the order changes. The legacy
table covers few reviewed shots, so every interval is wide; the JSON says so with the
numbers.

**5. Proxy references (not independent).** The same swap on many shots with a
reference built from a detector's output: the bins the ELM clock's ELMy spans touch (all
119 shots; the reviewers started from the clock, so it is not independent of the review)
and the bins holding an ELM-O onset (73 shots, the legacy table's convention). A
detector is not scored against its own reference. They show how a detector-made
reference reorders methods when the shots are many; they are not legacy annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from labeler.config import Paths, git_sha
from labeler.elm import compare, dsm, labels, methods, score, swap, train

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "labeler" / "elm" / "swap"
GROUND_TRUTH = Path("/scratch/gpfs/nc1514/FusionAIHub/.tmp/elm_all_ground_truth.csv")
NAME = compare.NAME
ALWAYS = "always present"
ORACLE = "expert dense labels"
LEGACY = "legacy table"


def annotate_prefetch_swap():
    """Keep historical scores, adding accurate method names and archive scope."""
    path = OUT / "prefetch_evaluation.json"
    if not path.exists():
        return
    record = json.loads(path.read_text())
    record["method_display_names"] = compare.DISPLAY_NAME
    record["record_note"] = (
        "Archived pre-Ip/Bt-fetch scores and ranking pairs, unchanged. Historical "
        "individual-kind fields mean non-crowd present spans. Historical conversion "
        "and interpretation are superseded by the current evaluation.json."
    )
    for collection in ("swap", "proxy"):
        for result in record[collection].values():
            for reference in ("reviewed", "legacy", "proxy"):
                if reference not in result:
                    continue
                for name, row in result[reference]["methods"].items():
                    row["display_name"] = compare.DISPLAY_NAME.get(name, name)
    path.write_text(json.dumps(record, indent=1))


def interval_coverage_audit(table, data, shots) -> dict:
    """AE-style all-covered-bin audit, explicitly partitioning unknown review time."""
    rows, per_shot, excluded = [], {}, []
    total_bins = total_positive = 0
    for shot in shots:
        spans = data[shot].spans
        bins, legacy, status = swap.coverage_bins(table, shot, spans)
        known = bins.truth >= 0
        c = swap.agreement_counts(bins.truth[known], bins.kind[known], legacy[known])
        rows.append(c)
        unknown = [
            {
                "shot": int(shot),
                "t_start_ms": float(t),
                "t_end_ms": float(t + labels.BIN_MS),
                "review_status": str(st),
                "legacy_present": bool(lp),
            }
            for t, st, lp in zip(bins.t0[~known], status[~known], legacy[~known])
        ]
        excluded.extend(unknown)
        positive = table[(table.shot == shot) & (table.category == 1)]
        per_shot[str(shot)] = {
            "review_window_ms": [float(spans.t_start.min()), float(spans.t_end.max())],
            "legacy_covered_bins": len(bins.t0),
            "legacy_present_bins": int(legacy.sum()),
            "counts": dict(zip(swap.AGREEMENT_NAMES, map(int, c), strict=True)),
            "excluded_review_bins": unknown,
            "review_spans": spans[["t_start", "t_end", "kind", "crowd"]].to_dict(
                "records"
            ),
            "legacy_positive_extent_ms": [
                float(positive.t_start.min()),
                float(positive.t_end.max()),
            ],
        }
        total_bins += len(bins.t0)
        total_positive += int(legacy.sum())
    by_status = {}
    for state in ("uncertain", "not_observable", "mixed_or_unlabelled"):
        selected = [r for r in excluded if r["review_status"] == state]
        by_status[state] = {
            "bins": len(selected),
            "legacy_present_bins": sum(r["legacy_present"] for r in selected),
            "bin_list": selected,
        }
    return {
        "definition": "All legacy-covered 50 ms bins intersecting the review "
        "window; >=25 ms majority rule, no DSM/diagnostic restriction. M/P compare "
        "onset-bin presence with reviewed interval occupancy, not physical ELM omissions.",
        "n_shots": len(shots),
        "shots": shots,
        "legacy_covered_bins": total_bins,
        "legacy_present_bins": total_positive,
        "known_review_majority": swap.agreement_summary(
            np.stack(rows), score.draws(len(shots))
        ),
        "excluded_review_states": by_status,
        "per_shot": per_shot,
    }


def onset_agreement_audit(paths, table, data, shots) -> dict:
    """Original positive onset samples, separately compared to span starts."""
    from labeler.events.source_formatters import read_elm

    raw = paths.label_tables / "edge_localized_mode" / "raw"
    files = [raw / f"elm_labels_dict{s}.pkl" for s in ("", "_wpqh")]
    onsets = {s: t[y == 1] for s, t, y in read_elm(*files) if s in set(shots)}
    per_shot = {}
    for s in shots:
        bins, legacy, _ = swap.coverage_bins(table, s, data[s].spans)
        per_shot[str(s)] = {
            "onset_bins_match_legacy": bool(
                np.array_equal(swap.onset_truth(onsets[s], bins), legacy)
            ),
            "legacy_onsets_in_review_window_ms": onsets[s][
                (onsets[s] >= data[s].spans.t_start.min())
                & (onsets[s] < data[s].spans.t_end.max())
            ].tolist(),
            "by_tolerance": {
                f"tol_{k}ms": swap.start_agreement(data[s].spans, onsets[s], k)
                for k in (5, 10, 50)
            },
        }
    summary = {}
    boot = score.draws(len(shots))
    for k in (5, 10, 50):
        key = f"tol_{k}ms"
        summary[key] = {}
        for kind in ("non_crowd", "crowd"):
            counts = np.array(
                [
                    [
                        per_shot[str(s)]["by_tolerance"][key][kind][col]
                        for col in ("matched", "spans")
                    ]
                    for s in shots
                ]
            )
            hit, n = counts.sum(axis=0)
            drawn = counts[boot].sum(axis=1)
            share = np.divide(
                drawn[:, 0],
                drawn[:, 1],
                out=np.full(len(boot), np.nan),
                where=drawn[:, 1] > 0,
            )
            summary[key][kind] = {
                "matched": int(hit),
                "spans": int(n),
                "share": float(hit / n) if n else float("nan"),
                "ci95": score._ci(share),
            }
    return {
        "definition": "Legacy onset within inclusive +/-k ms of each reviewed "
        "present span start. Proximity per annotation, not one-to-one ELM recall; "
        "crowd starts are not its non-crowd ELMs. Review starts are not "
        "independently verified millisecond onset truth.",
        "source_files": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
        "summary": summary,
        "per_shot": per_shot,
    }


def overlap_report(paths, reviewed, table, ground_truth, smith) -> dict:
    rs = set(reviewed)
    t_shots = {int(s) for s in table.shot.unique()}
    g_shots = {int(s) for s in ground_truth.shot}
    s_shots = {int(s) for s in smith.shot.unique()}
    both = sorted(rs & t_shots)
    return {
        "reviewed_shots": len(rs),
        "legacy_table": {
            "file": str(swap.LEGACY_TABLE),
            "shots": len(t_shots),
            "overlap": len(both),
            "overlap_shots": both,
            "grid": swap.table_alignment(table, both),
        },
        "shot_level_ground_truth": {
            "file": str(GROUND_TRUTH),
            "shots": len(g_shots),
            "overlap": len(rs & g_shots),
            "overlap_shots": sorted(rs & g_shots),
            "values_on_overlap": {
                int(r.shot): bool(r.ground_truth)
                for r in ground_truth[ground_truth.shot.isin(rs)].itertuples()
            },
            "note": "shot-level: it says whether a shot has an ELM, not where, so it "
            "cannot score bins; it is checked against the review's present spans",
        },
        "smith_windows": {
            "shots": len(s_shots),
            "overlap": len(rs & s_shots),
            "note": "the ELM-O benchmark truth; none of its shots is in the review",
        },
    }


def always_part(part: score.ShotScore) -> score.ShotScore:
    return score.ShotScore(
        part.shot, part.truth, part.kind, np.ones(len(part.truth), bool), None, {}
    )


def score_reference(parts, truths, boot, skip=()) -> dict:
    """Every method against `truths` (shot -> truth per bin, -1 dropped)."""
    shots = [p.shot for p in next(iter(parts.values()))]
    out: dict = {"methods": {}, "paired": {}}
    kept: dict[str, list[score.ShotScore]] = {}
    for name, plist in parts.items():
        if name in skip:
            continue
        kept[name] = [
            swap.retruth(p, truths[s]) for p, s in zip(plist, shots, strict=True)
        ]
    kept[ALWAYS] = [always_part(p) for p in next(iter(kept.values()))]
    for name, plist in kept.items():
        res = score.summarise(plist, boot)
        if name == ALWAYS:
            res["point"]["auroc"] = 0.5
            res["ci95"].pop("auroc", None)
        out["methods"][name] = res
        res["display_name"] = compare.DISPLAY_NAME.get(name, name)
        res["degenerate_f1"] = bool(res["point"]["recall"] >= 0.99)
    out["bins"] = int(sum(len(p.truth) for p in kept[ALWAYS]))
    out["prevalence"] = float(np.concatenate([p.truth for p in kept[ALWAYS]]).mean())
    out["ranking"] = swap.rankings(
        {n: r["point"] for n, r in out["methods"].items() if n != ALWAYS}
    )
    ours = kept[NAME["ours"]]
    for name, plist in kept.items():
        if name in (NAME["ours"], ALWAYS):
            continue
        for metric in ("auroc", "f1"):
            if metric == "auroc" and any(p.score is None for p in plist):
                continue
            out["paired"][f"{NAME['ours']} - {name}: {metric}"] = (
                score.paired_difference(ours, plist, boot, metric)
            )
    return out


def flips(a: dict, b: dict) -> dict:
    """Pairs of methods whose order differs between two references' point values."""
    out = {}
    for metric in ("auroc", "f1"):
        pa = {
            n: r["point"][metric]
            for n, r in a["methods"].items()
            if n != ALWAYS and metric in r["point"]
        }
        pb = {
            n: r["point"][metric]
            for n, r in b["methods"].items()
            if n != ALWAYS and metric in r["point"]
        }
        names = sorted(set(pa) & set(pb))
        out[metric] = [
            [x, y]
            for i, x in enumerate(names)
            for y in names[i + 1 :]
            if (pa[x] - pa[y]) * (pb[x] - pb[y]) < 0
        ]
    return out


def compare_references(a: dict, b: dict) -> dict:
    """Whether the order of methods changes between two references, per metric, and how
    `elm-ours`' paired differences from every other method move."""
    changes = {
        m: a["ranking"][m] != b["ranking"][m] for m in a["ranking"] if m in b["ranking"]
    }
    moves = {}
    for key, va in a["paired"].items():
        vb = b["paired"].get(key)
        if vb is None:
            continue
        moves[key] = {
            "first": va["value"],
            "second": vb["value"],
            "same_sign": va["value"] * vb["value"] > 0,
            "first_ci_excludes_zero": va["ci95"][0] > 0 or va["ci95"][1] < 0,
            "second_ci_excludes_zero": vb["ci95"][0] > 0 or vb["ci95"][1] < 0,
        }
    return {
        "order_changes": changes,
        "order_flips": flips(a, b),
        "paired_differences": moves,
        "auroc_changes": {
            name: {
                "review": va["point"]["auroc"],
                "legacy": b["methods"][name]["point"]["auroc"],
                "legacy_minus_review": b["methods"][name]["point"]["auroc"]
                - va["point"]["auroc"],
            }
            for name, va in a["methods"].items()
            if name != ALWAYS and "auroc" in va["point"]
        },
    }


def agreement(parts_ref, legacy, shots, boot) -> dict:
    """Finding 1 from per-shot counts of the legacy marks against the review."""
    rows = [
        swap.agreement_counts(p.truth, p.kind, legacy[s])
        for p, s in zip(parts_ref, shots, strict=True)
    ]
    out = swap.agreement_summary(np.stack(rows), boot)
    out["per_shot"] = {
        int(s): dict(zip(swap.AGREEMENT_NAMES, (int(v) for v in r), strict=True))
        for s, r in zip(shots, rows, strict=True)
    }
    return out


def oracle_rows(parts_ref, legacy, shots, boot) -> dict:
    """The review and the legacy table as methods against the other reference."""
    ref = [
        swap.as_method(
            s,
            p.truth[legacy[s] >= 0].astype(bool),
            p.truth[legacy[s] >= 0],
            p.kind[legacy[s] >= 0],
        )
        for p, s in zip(parts_ref, shots, strict=True)
    ]
    leg = [
        swap.as_method(
            s,
            legacy[s][legacy[s] >= 0] == 1,
            p.truth[legacy[s] >= 0],
            p.kind[legacy[s] >= 0],
        )
        for p, s in zip(parts_ref, shots, strict=True)
    ]
    return {
        "legacy_table_vs_reviewed": score.summarise(leg, boot),
        "expert_vs_reviewed": score.summarise(ref, boot),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args(argv)
    annotate_prefetch_swap()
    paths = Paths.from_env()
    work = paths.root / "round4" / "elm" / "dsm"
    data = train.load(paths)
    oof = methods.Oof(paths.root / "round4" / "elm" / "cv" / args.run)
    norm = dsm.load_norm(paths)
    shots_all = sorted(data)
    rows = {s: dsm.cached_rows(paths, s, norm, work) for s in shots_all}
    dscores = compare.DsmScores.load(work, rows, variants=compare.VARIANTS)
    sets = compare.load_sets(paths, data)
    elmo_spans, clock_spans = compare.load_detected(paths)
    elmo_sweep = compare.load_elmo_sweep(paths)

    table = swap.legacy_table(paths.label_tables / swap.LEGACY_TABLE)
    gt = pd.read_csv(GROUND_TRUTH)
    smith = pd.read_csv(
        paths.root / "benchmarks" / "elm" / "elmo" / "smith_windows.csv"
    )
    over = overlap_report(paths, shots_all, table, gt, smith)
    shots_over = over["legacy_table"]["overlap_shots"]
    print("overlap:", json.dumps({k: v for k, v in over.items()}, default=str)[:600])
    reviewed_present = {
        int(s): int((data[s].spans.kind.isin(["non_crowd", "crowd"])).sum())
        for s in over["shot_level_ground_truth"]["overlap_shots"]
    }
    over["shot_level_ground_truth"]["reviewed_present_spans"] = reviewed_present
    dsm_record = json.loads((OUT.parent / "dsm" / "evaluation.json").read_text())
    dsm_source = dsm_record["own_target"]

    record = {
        "git": git_sha(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "run": args.run,
        "cohort_test_shots_used": 0,
        "cohort_test_shots_used_scope": "reviewed-label detector training/tuning; "
        "the fixed DSM refit and initialized detector inherit prior pretraining "
        "overlap, recorded separately in legacy_trained_method",
        "overlap": over,
        "conversion": swap.__doc__,
        "legacy_trained_method": {
            "id": NAME["dsm"],
            "display_name": compare.DISPLAY_NAME[NAME["dsm"]],
            "training_label_source": "Hiro's onsets via elm_survival_labels.pkl "
            "and the wpqh_elm_hiro survival-row split; same source as legacy onset table",
            "ae_analogue": "legacy-trained RCN/LSTM",
            "reviewed_source_overlap": dsm_source[
                "reviewed_shot_ids_in_published_split"
            ],
            "cohort_source_overlap": dsm_source["cohort_physical_shot_overlap"],
            "own_target_selection_role": dsm_source["selection_role"],
        },
        "interval_audit": interval_coverage_audit(table, data, shots_over),
        "onset_agreement": onset_agreement_audit(paths, table, data, shots_over),
        "swap": {},
        "proxy": {},
    }

    # --- the legacy table: the overlap shots; ELM-O on those with BES
    for tag, base in (("overlap", sets["all119"]), ("overlap_bes", sets["bes73"])):
        sel = [s for s in base.shots if s in set(shots_over)]
        sdef = compare.SetDef(
            tag,
            sel,
            {s: base.bins[s] for s in sel},
            {s: base.cover[s] for s in sel},
            base.has_elmo,
        )
        if not sel:
            record["swap"][tag] = {"shots": [], "note": "no shot"}
            continue
        parts, bins_of = compare.common_parts(
            sdef, data, oof, dscores, elmo_spans, clock_spans, elmo_sweep
        )
        boot = score.draws(len(sel))
        legacy = {s: swap.table_truth(table, s, bins_of[s]) for s in sel}
        review = {s: np.where(legacy[s] >= 0, bins_of[s].truth, -1) for s in sel}
        res = {
            "shots": [int(s) for s in sel],
            "n_shots": len(sel),
            "restriction_deviation": "Ranking uses bins wholly inside one known "
            "review span and analysed time, with DSM forecast/detection rows; "
            "unlike the AE all-frame audit. Full majority counts are interval_audit.",
            "reviewed": score_reference(parts, review, boot),
            "legacy": score_reference(parts, legacy, boot),
        }
        ref_parts = parts[NAME["ours"]]
        res["finding_1"] = agreement(ref_parts, legacy, sel, boot)
        res["finding_1"]["as_methods"] = oracle_rows(ref_parts, legacy, sel, boot)
        res["comparison"] = compare_references(res["reviewed"], res["legacy"])
        record["swap"][tag] = res

    # --- proxy references on all the reviewed shots (not independent)
    for tag, key, base, found, own, how in (
        ("clock_spans", "all119", sets["all119"], clock_spans, NAME["clock"], "spans"),
        ("elmo_onsets", "bes73", sets["bes73"], elmo_spans, NAME["elmo"], "onsets"),
    ):
        parts, bins_of = compare.common_parts(
            base, data, oof, dscores, elmo_spans, clock_spans, elmo_sweep
        )
        boot = score.draws(len(base.shots))
        proxy = {}
        for s in base.shots:
            sp = found.get(s)
            if sp is None:
                sp = methods.span_frame([], [])
            if how == "onsets":
                proxy[s] = swap.onset_truth(sp.t_start_ms.to_numpy(float), bins_of[s])
            else:
                sp = sp.sort_values("t_start_ms")
                proxy[s] = labels.hard_hits(
                    sp.t_start_ms.to_numpy(float),
                    sp.t_end_ms.to_numpy(float),
                    bins_of[s],
                ).astype(np.int8)
        review = {s: bins_of[s].truth.astype(np.int8) for s in base.shots}
        res = {
            "set": key,
            "shots": len(base.shots),
            "reference_producer_excluded": own,
            "label": f"{own} onset bins" if how == "onsets" else f"{own} span bins",
            "conversion": (
                "a bin is present when a detected ELM onset falls in it"
                if how == "onsets"
                else "a bin is present when a detected ELMy span touches it (the "
                "clock's category-1 rows are spans, one per ELMy period, not onsets)"
            ),
            "note": "NOT independent: a reference made from a detector's output; the "
            "producing detector is not scored against it",
            "reviewed": score_reference(parts, review, boot, skip=(own,)),
            "proxy": score_reference(parts, proxy, boot, skip=(own,)),
        }
        res["comparison"] = compare_references(res["reviewed"], res["proxy"])
        record["proxy"][tag] = res

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "evaluation.json").write_text(json.dumps(record, indent=1))
    from labeler.elm import swap_tex

    swap_tex.write(record, args.out_dir)
    print("overlap shots:", shots_over)
    for tag, r in record["swap"].items():
        if "finding_1" not in r:
            continue
        f1 = r["finding_1"]
        print(
            tag, r["n_shots"], "shots", f1["bins"], "bins", "M", f1["M"], "P", f1["P"]
        )
        print("  ranking reviewed", r["reviewed"]["ranking"])
        print(
            "  ranking legacy  ",
            r["legacy"]["ranking"],
            "changes:",
            r["comparison"]["order_changes"],
        )
    for tag, r in record["proxy"].items():
        print(tag, "reviewed", r["reviewed"]["ranking"], "proxy", r["proxy"]["ranking"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
