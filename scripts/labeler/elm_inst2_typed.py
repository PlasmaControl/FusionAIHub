#!/usr/bin/env python
"""Score `elm-inst2` again with a type per detection: its second round on the key.

    python scripts/labeler/elm_inst2_typed.py evaluate   # development record, round 2
    python scripts/labeler/elm_inst2_typed.py key        # the challenge key, once

Round 1 (`elm_inst2.py`) reported every detection as the detection-only type 5 and
chose its thresholds at type-5 prices, under which a found small ELM saves 0.5 and a
false detection costs 9.5 (`spikes.prices`); it scored the key once, and `key.lock`
refuses it again. On 2026-10-24 the owner asked for two changes, scored once more under
a second lock (`key2.lock`):

1. **Type-2 prices.** Every detection reported as a small ELM (type 2): a found large
   ELM saves 7.0, a found small one 4.0, a false detection costs 3.0. The threshold is
   chosen again on round 1's out-of-fold counts (`elm_inst2.tradeoff`), by its two
   rules (`elm_inst2.rule_gains`) at type-2 prices: `elm-inst2-t2` at the
   conservative rule's maximum, `elm-inst2-t2-alt` at the as-trained rule's.
2. **Size typing** (`elm-inst2-sized`, the primary): `elm-inst2-t2`'s detections,
   each reported as a large (1) or a small (2) ELM, whichever has the lower expected
   cost (`spikes.report_type`) under two logistic regressions on its size measures
   (`sizes.features`), fitted on the out-of-fold detections at the served threshold:
   * P(large | ELM): Smith's onsets found (large) against his onsets found inside a
     crowd (small), the two classes weighted equally: the key's mix is not used;
   * P(no event): detections judged false (unmatched in Smith's windows, in guarded
     reviewed absent time, his `fp` and `empty`) against detections that found an
     onset of Smith's or his, at their out-of-fold mix.

`evaluate` fixes every choice in the development record before the key is read. `key`
checks that record against round 1's, rebuilds `elm-inst`'s graded files as round 1
did (`elm_inst2.reproduce`), decodes and types every detection, and only then takes
the lock and grades.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np

from labeler.config import Paths, git_sha, sha256_of
from labeler.elm import challenge, inputs, prepare, sizes, spikes
from labeler.events.source_formatters import read_elm_challenge

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import elm_inst as base  # the sys.path insert above makes this work
import elm_inst2 as r1

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/labeler/elm/inst2/evaluation_typed.json"
KEY_OUT = REPO / "outputs/labeler/elm/inst2/key_typed.json"
LOCK = "key2.lock"  # beside round 1's key.lock, which stays
SCORES = "key2_merit_scores.json"
RECORD = "key2_record.json"
DETECTIONS = "typed_detections.npz"
T2 = "elm-inst2-t2"
T2_ALT = "elm-inst2-t2-alt"
SIZED = "elm-inst2-sized"
#: `elm-inst` reported as type 2, graded by another session on the owner's word
#: (2026-10-24); graded again here beside these for one record.
AS2 = "elm-inst-as2"
REPORTED = 2
COMMENT = (
    "FusionAIHub elm-inst2 (round 2): peaks of the five fold models' mean 0.1 ms onset "
    "heatmap, 3 ms apart, at a threshold chosen on development data at type-2 prices, "
    "each event the training shots' median clock event duration long"
)
LABELS = ("large", "small", "elm", "false")


def run_evaluate(paths: Paths, args) -> None:
    """Round 2's development record: the type-2 thresholds, the two size models and
    their out-of-fold views, and the three decodings the key is scored with."""
    directory = r1.work(paths)
    if (directory / LOCK).exists():
        raise SystemExit(
            f"{directory / LOCK} exists: the key was scored with the development "
            "record as it is"
        )
    round1 = json.loads(r1.OUT.read_text())
    data = r1.load_all(paths)
    records = [
        json.loads((directory / f"fold{k}/fold.json").read_text())
        for k in range(base.FOLDS)
    ]
    for k, r in enumerate(records):
        if sha256_of(directory / f"fold{k}/model.pt") != r["checkpoint_sha256"]:
            raise ValueError(f"fold{k}/model.pt is not the checkpoint fold.json names")
        if r["checkpoint_sha256"] != round1["hashes"]["checkpoints"][f"fold{k}"]:
            raise ValueError(f"fold{k} is not the checkpoint round 1 recorded")
    fold_of = {
        s: k
        for k, r in enumerate(records)
        for v in r["shots"]["test"].values()
        for s in v
    }
    pred = directory / "pred"
    candidates = {
        s: r1.candidates_of(
            np.load(pred / f"{s}.npz")["inst"].astype(np.float32), spikes.MIN_SEP_MS
        )
        for s in data
    }
    duration = float(np.mean([r["event_duration_ms"] for r in records]))
    table = r1.tradeoff(data, candidates, duration)
    count_keys = r1.TRADEOFF_COUNTS
    recorded = round1["tradeoff_pooled_out_of_fold"]["elm-inst2"]
    if [{k: r[k] for k in count_keys} for r in table] != [
        {k: r[k] for k in count_keys} for r in recorded
    ]:
        raise SystemExit("the out-of-fold counts differ from round 1's record")
    for row in table:
        t2 = r1.rule_gains(row, duration, reported=REPORTED)
        row["gain_as_trained_t2"] = t2["gain_as_trained"]
        row["gain_conservative_t2"] = t2["gain_conservative"]
    best = max(table, key=lambda r: (r["gain_conservative_t2"], r["threshold"]))
    best_alt = max(table, key=lambda r: (r["gain_as_trained_t2"], r["threshold"]))
    served = {
        **round1["served"],
        "onset_threshold": best["threshold"],
        "event_type": REPORTED,
        "rule": "round 1's decoding (five-model mean heatmap, 3 ms apart, ungated, "
        "each event the mean fold event duration long) at the grid threshold where "
        "the pooled out-of-fold counts save the most by the conservative rule at "
        "type-2 prices (elm_inst2.rule_gains, reported=2; the higher threshold on a "
        "tie); every detection reported as type 2",
    }
    alt = {
        **served,
        "onset_threshold": best_alt["threshold"],
        "rule": "as elm-inst2-t2, at the as-trained rule's maximum at type-2 prices",
    }

    key_traces = {
        p.stem: sha256_of(p) for p in sorted((directory / "key").glob("*.npz"))
    }
    if key_traces != round1["hashes"]["key_traces"]:
        raise SystemExit("the key traces are not the ones round 1 hashed")
    det = detections(data, candidates, fold_of, served["onset_threshold"])
    np.savez_compressed(directory / DETECTIONS, **det)
    oof = out_of_fold(det)
    models = size_models(det, np.ones(len(det["label"]), bool))
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_sha(full=True),
        "run": str(directory),
        "round1_record_sha256": sha256_of(r1.OUT),
        "frozen": frozen(paths, sorted(int(s) for s in key_traces)),
        "protocol": "Development data only, out of fold, as round 1 (elm_inst2.py): "
        "the same checkpoints, traces and counts. The decodings the key is scored with "
        "are fixed here. The size models are fitted on detections at the served "
        "threshold of single out-of-fold models; the key's detections are the "
        "five-model mean's. Their labels are proxies: Smith's ELMs stand for large "
        "ELMs and his ELMs inside crowds for small ones. The challenge key's events and "
        "its mix of types are not read here; its shots' traces are only hashed.",
        "primary": SIZED,
        "prices": {
            str(t): dict(zip(("large_found", "small_found", "false"), spikes.prices(t)))
            for t in sorted(spikes.PENALTY)
        },
        "served": served,
        "alt": alt,
        "sized": {
            **served,
            "event_type": "per detection, spikes.report_type(p_none, (1 - p_none) "
            "p_large, (1 - p_none) (1 - p_large)) over types 2 and 1, with p_large and "
            "p_none from the size models below on sizes.features at the detection's "
            "onset",
            "rule": "elm-inst2-t2's detections, each reported as type 1 or 2 by "
            "event_type",
        },
        "size_models": {
            "features": list(sizes.FEATURES),
            "large_given_elm": models["large_given_elm"],
            "no_event": models["no_event"],
            "large_given_elm_rule": "logistic, Smith's onsets found (1) against his "
            "onsets found inside a crowd (0), each class half the weight",
            "no_event_rule": "logistic, detections judged false (1: unmatched in "
            "Smith's windows, in guarded reviewed absent time, his fp and empty) "
            "against detections that found an onset (0), unweighted",
        },
        "tradeoff_pooled_out_of_fold": {
            "rules": "elm_inst2.rule_gains: gain_* at type 5 (round 1), gain_*_t2 at "
            "type 2; counts as round 1's",
            "elm-inst2": table,
        },
        "detections_at_served_threshold": {
            label: int((det["label"] == label).sum()) for label in LABELS
        }
        | {"unjudged": int((det["label"] == "").sum())},
        "size_models_out_of_fold": oof,
        "hashes": {
            "checkpoints": round1["hashes"]["checkpoints"],
            "key_traces": key_traces,
            "detections": sha256_of(directory / DETECTIONS),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=1, default=float) + "\n")
    for r in table:
        print(
            json.dumps(
                {
                    k: round(r[k], 2)
                    for k in ("threshold", "gain_conservative_t2", "gain_as_trained_t2")
                }
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                k: record[k]
                for k in (
                    "served",
                    "alt",
                    "detections_at_served_threshold",
                    "size_models",
                    "size_models_out_of_fold",
                )
            },
            indent=1,
            default=float,
        )
    )


def frozen(paths: Paths, key_shots) -> dict:
    """sha256 of what decides the key's submissions besides the record: the code that
    decodes, measures, types, writes and grades them, the key shots' input records,
    and the `elm-inst-as2` files graded beside them. `key` refuses on any change."""
    code = (
        Path(__file__),
        HERE / "elm_inst2.py",
        HERE / "elm_inst.py",
        HERE / "elm_challenge_merit.py",
        *(
            REPO / "src/labeler/elm" / f"{m}.py"
            for m in ("spikes", "sizes", "inst", "inputs", "challenge")
        ),
    )
    signals = prepare.signals_dir(paths)
    as2 = paths.root / r1.SUBMISSIONS / AS2
    return {
        "code": {p.name: sha256_of(p) for p in code},
        "key_signals": {str(s): sha256_of(signals / f"{s}.npz") for s in key_shots},
        AS2: {p.name: sha256_of(p) for p in sorted(as2.glob("*.json"))},
    }


def detections(data, candidates, fold_of, threshold: float) -> dict:
    """Every out-of-fold detection at `threshold` with its shot, fold, source, onset,
    heatmap height, size measures and label: `large` (found one of Smith's onsets),
    `small` (found his onset inside a crowd), `elm` (found his onset outside one),
    `false` (unmatched in Smith's windows, in guarded reviewed absent time, or his
    `fp` or `empty`, `spikes.judge`), or `""` (unjudged)."""
    cols = {k: [] for k in ("shot", "fold", "source", "t", "h", "label", "X")}
    for s, d in data.items():
        t, h = candidates[s]
        keep = h >= threshold
        t, h = t[keep], h[keep]
        label = np.full(len(t), "", dtype=object)
        if d.source == "smith":
            inside = base.in_any(t, list(zip(d.window_t0, d.window_t1, strict=True)))
            matched, _ = spikes.match_pairs(t[inside], d.truth, base.TOL_MS)
            label[np.flatnonzero(inside)] = np.where(matched, "large", "false")
        elif d.source == "review":
            label[base.in_any(t, base.guarded_absent(d.spans))] = "false"
        else:
            order = np.argsort(t, kind="stable")
            t, h = t[order], h[order]
            j = spikes.judge(t, d.onsets, d.spans)
            crowd = spikes.in_crowd(t, d.spans)
            kind = j["kind"]
            label = np.where(
                kind == "tp",
                np.where(crowd, "small", "elm"),
                np.where((kind == "fp") | (kind == "empty"), "false", ""),
            ).astype(object)
        cols["shot"].append(np.full(len(t), s))
        cols["fold"].append(np.full(len(t), fold_of[s]))
        cols["source"].append(np.full(len(t), d.source, dtype=object))
        cols["t"].append(t)
        cols["h"].append(h)
        cols["label"].append(label)
        cols["X"].append(sizes.features(d.x, t))
    out = {k: np.concatenate(v) for k, v in cols.items()}
    for k in ("source", "label"):
        out[k] = out[k].astype(str)
    return out


def size_models(det: dict, rows: np.ndarray) -> dict:
    """The two logistic regressions on the detections `rows` of `det`."""
    label, X = det["label"], det["X"]
    sized = rows & np.isin(label, ("large", "small"))
    judged = rows & np.isin(label, LABELS)
    return {
        "large_given_elm": sizes.fit(
            X[sized],
            label[sized] == "large",
            weight=sizes.balanced(label[sized] == "large"),
        ),
        "no_event": sizes.fit(X[judged], label[judged] == "false"),
    }


def typed(models: dict, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(p_large, p_none, type)` of detections with size measures `X`."""
    q = sizes.prob(models["large_given_elm"], X)
    f = sizes.prob(models["no_event"], X)
    return q, f, spikes.report_type(f, (1 - f) * q, (1 - f) * (1 - q))


def out_of_fold(det: dict) -> dict:
    """Each fold's detections typed by models fitted on the other folds: how well the
    two models separate their classes, and what typing would change on development
    data against every detection reported as type 2."""
    n = len(det["label"])
    q, f, ty = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, int)
    for k in np.unique(det["fold"]):
        test = det["fold"] == k
        q[test], f[test], ty[test] = typed(size_models(det, ~test), det["X"][test])
    label = det["label"]
    sized = np.isin(label, ("large", "small"))
    judged = np.isin(label, LABELS)
    # reported 1 instead of 2: a large ELM saves 5.0, a small one costs 3.0 and a
    # false detection 7.0 more (spikes.PENALTY)
    change = {"large": 5.0, "small": -3.0, "false": -7.0}
    as1 = ty == 1
    bins = np.linspace(0, 1, 6)
    w = sizes.balanced(label[sized] == "large")
    reliability = []
    for lo, hi in pairwise(bins):
        b = (q[sized] >= lo) & (q[sized] < hi if hi < 1 else q[sized] <= hi)
        reliability.append(
            {
                "from": float(lo),
                "to": float(hi),
                "n": int(b.sum()),
                "mean_p_large": float(q[sized][b].mean()) if b.any() else None,
                "large_share_balanced": float(
                    (w[b] * (label[sized][b] == "large")).sum() / w[b].sum()
                )
                if b.any()
                else None,
            }
        )
    return {
        "auroc_large_vs_small": sizes.auroc(q[sized], label[sized] == "large"),
        "auroc_false_vs_found": sizes.auroc(f[judged], label[judged] == "false"),
        "reliability_large_given_elm_balanced": reliability,
        "reported_large": {lab: int((as1 & (label == lab)).sum()) for lab in LABELS}
        | {"unjudged": int((as1 & (label == "")).sum())},
        "reported_small": {lab: int((~as1 & (label == lab)).sum()) for lab in LABELS}
        | {"unjudged": int((~as1 & (label == "")).sum())},
        "change_against_all_type_2": float(
            sum(v * (as1 & (label == lab)).sum() for lab, v in change.items())
        ),
        "change_rule": "per detection reported 1 instead of 2: +5.0 on a large ELM, "
        "-3.0 on a small one, -7.0 on a false detection; his ELMs outside crowds and "
        "unjudged detections uncounted; at the development mix, which is not the key's",
        "median_features": {
            lab: dict(
                zip(
                    sizes.FEATURES,
                    np.nanmedian(det["X"][label == lab], axis=0).tolist(),
                    strict=True,
                )
            )
            for lab in LABELS
            if (label == lab).any()
        },
    }


# ---------------------------------------------------------------------- key


def typed_submission(
    shot: int, spans, types, window, *, method: str, comment: str
) -> dict:
    """`challenge.submission` of `spans` with each event reported as its type."""
    spans = np.asarray(spans, dtype=float).reshape(-1, 2)
    types = np.asarray(types, dtype=int).reshape(-1)
    if len(types) != len(spans):
        raise ValueError(f"{shot}: {len(types)} types for {len(spans)} events")
    order = np.argsort(spans[:, 0], kind="stable")
    spans, types = spans[order], types[order]
    record = challenge.submission(shot, spans, window, method=method, comment=comment)
    if len(challenge.flag_spans(record)) != len(spans):
        raise ValueError(f"{shot}: events would merge; types cannot be assigned")
    record["event_type"] = [0, *(int(t) for t in types for _ in range(4)), 0]
    if read_types(record) != types.tolist():
        raise ValueError(f"{shot}: the merit function would read other types")
    return record


def read_types(record: dict) -> list[int]:
    """Each event's type as the merit function reads it: the type at the sample where
    the flag rises (`count_and_compare_events`)."""
    t, f, ty = (np.asarray(record[k]) for k in ("time", "event_flag", "event_type"))
    if not len(t) == len(f) == len(ty):
        raise ValueError("time, event_flag and event_type differ in length")
    return ty[np.append(False, np.diff(f) == 1)].astype(int).tolist()


def classification_view(merit: dict, scopes: dict) -> dict:
    """Per graded folder and scope, the sums the merit function gives for wrong types."""
    out = {}
    for name, scores in merit["folders"].items():
        per_case = {
            r1.shot_of(v["label"]): v
            for c, v in scores.items()
            if c not in ("total", "total_not_counting_missing") and not v["missing"]
        }
        out[name] = {
            scope: {
                k: float(sum(per_case[s][k] for s in want if s in per_case))
                for k in ("classification_score", "incorrect_classification")
            }
            for scope, want in scopes.items()
        }
    return out


def run_key(paths: Paths, args) -> None:
    """The challenge key, round 2, once: the three decodings the development record
    fixed, graded beside `elm-inst`, `elm-inst-as2` and no detection."""
    directory = r1.work(paths)
    lock, scores_file = directory / LOCK, directory / SCORES
    for path in (KEY_OUT, lock, scores_file):
        if path.exists():
            raise SystemExit(
                f"{path} exists: the key is scored once (delete it by hand)"
            )
    dev = json.loads(OUT.read_text())
    round1 = json.loads(r1.OUT.read_text())
    if sha256_of(r1.OUT) != dev["round1_record_sha256"]:
        raise SystemExit("round 1's development record changed since round 2's")
    if sha256_of(directory / DETECTIONS) != dev["hashes"]["detections"]:
        raise SystemExit(f"{directory / DETECTIONS} is not the one the record hashed")
    submissions = paths.root / r1.SUBMISSIONS
    if not any((submissions / AS2).glob("*.json")):
        raise SystemExit(f"{submissions / AS2} holds no submission")
    cases = {c["shot"]: c for c in read_elm_challenge(paths.label_tables / r1.KEY)}
    shots = sorted(cases)
    now = frozen(paths, shots)
    if now != dev["frozen"]:
        changed = [
            f"{part}/{name}"
            for part, sums in dev["frozen"].items()
            for name in set(sums) | set(now.get(part, {}))
            if sums.get(name) != now.get(part, {}).get(name)
        ]
        raise SystemExit(f"changed since the development record: {sorted(changed)}")
    scopes = r1.scopes_of(paths, cases)
    found_inst1, merit_inst1, check = r1.reproduce(
        paths, round1, cases, directory / "key2_check"
    )
    r1.views(cases, {"elm-inst": found_inst1}, merit_inst1, scopes)
    classification_view(merit_inst1, scopes)
    traces = r1.traces_of(directory / "key", shots, dev["hashes"]["key_traces"])
    found = {
        method: {s: r1.decode(traces[s], served, r1.window_of(cases[s])) for s in shots}
        for method, served in ((T2, dev["served"]), (T2_ALT, dev["alt"]))
    }
    models = dev["size_models"]
    signals = prepare.signals_dir(paths)
    types, p_large, p_none = {}, {}, {}
    for s in shots:
        x = inputs.read_channels(signals / f"{s}.npz")
        q, f, ty = typed(models, sizes.features(x, found[T2][s][:, 0]))
        types[s], p_large[s], p_none[s] = ty, q, f
    # every record built and checked before the lock; written after it
    built = {}
    for method, spans, served in (
        (T2, found[T2], dev["served"]),
        (T2_ALT, found[T2_ALT], dev["alt"]),
        (SIZED, found[T2], dev["served"]),
    ):
        comment = f"{COMMENT}; threshold {served['onset_threshold']:g}"
        if method == SIZED:
            comment += "; each event typed large or small by its size"
        built[method] = {
            s: typed_submission(
                s,
                spans[s],
                types[s] if method == SIZED else np.full(len(spans[s]), REPORTED),
                r1.window_of(cases[s]),
                method=method,
                comment=comment,
            )
            for s in shots
        }
    claim_note = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_sha(full=True),
        "job": os.environ.get("SLURM_JOB_ID"),
        "development_record_sha256": sha256_of(OUT),
    }
    r1.claim(lock, claim_note)
    for method, records in built.items():
        folder = submissions / method
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.json"):
            old.unlink()
        for s, record in records.items():
            (folder / f"DIII-D_{s}.json").write_text(json.dumps(record) + "\n")
    folders = (T2, T2_ALT, SIZED, AS2, "elm-inst", "no-detection")
    merit = r1.grade(paths, {m: submissions / m for m in folders}, scores_file)
    found["elm-inst"] = found_inst1
    out = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_sha(full=True),
        "clone_commit": merit["clone_commit"],
        "key_sha256": merit["key_sha256"],
        "development_record_sha256": sha256_of(OUT),
        "statement": "scored with the challenge merit function at commit b71f991, once "
        "for round 2; not an official leaderboard result. elm-inst2-sized is the "
        "primary; elm-inst2-t2 and elm-inst2-t2-alt are its type-2 parts. All three "
        "were declared in the development record before the key was read. The key had "
        "been graded before (elm-inst2 round 1, elm-inst retyped as 2, the key fed back "
        "as a submission), so this is not a held-out result. Sums are over graded "
        "cases only.",
        "primary": SIZED,
        "decodings": {T2: dev["served"], T2_ALT: dev["alt"], SIZED: dev["sized"]},
        "reproduction_check": check,
        "frozen": now,
        "types_reported": {
            str(s): {
                "large": int((types[s] == 1).sum()),
                "small": int((types[s] == 2).sum()),
            }
            for s in shots
        },
        "typing_per_detection": {
            str(s): {
                "onset_ms": found[T2][s][:, 0].tolist(),
                "p_large": p_large[s].tolist(),
                "p_none": p_none[s].tolist(),
                "type": types[s].astype(int).tolist(),
            }
            for s in shots
        },
        "classification": classification_view(merit, scopes),
        **r1.views(
            cases,
            {T2: found[T2], T2_ALT: found[T2_ALT], "elm-inst": found_inst1},
            merit,
            scopes,
        ),
    }
    text = json.dumps(out, indent=1, default=float) + "\n"
    KEY_OUT.parent.mkdir(parents=True, exist_ok=True)
    KEY_OUT.write_text(text)
    (directory / RECORD).write_text(text)
    print(
        json.dumps(
            {n: {sc: v[sc]["total"] for sc in scopes} for n, v in out["merit"].items()},
            indent=1,
        )
    )
    print(json.dumps(out["classification"], indent=1))
    print(json.dumps(out["types_reported"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("evaluate", "key"))
    args = ap.parse_args(argv)
    paths = Paths.from_env()
    {"evaluate": run_evaluate, "key": run_key}[args.stage](paths, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
