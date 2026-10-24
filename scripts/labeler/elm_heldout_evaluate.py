#!/usr/bin/env python
"""Score the frozen ELM models on Ricardo Shousha's review of 2026-10-23, no refit.

    python scripts/labeler/elm_heldout_evaluate.py [--shots S ...] [--skip-old]

His 81 shots (the last save of each, `review/history.jsonl` of the pin `2026-10-23`)
are in no cross-validation fold, none in Smith's 211, none in the cohort's blind test
split. They split into `B1`, the 58 that are not on a `protect: holdout` list, and
`B2`, the 23 of the ELM challenge key's 24 shots, which are evaluation only. The
reference panel `A` is the old reviewed set (pin `2026-10-17`) scored by the same code
from the saved out-of-fold traces, which reproduces `outputs/labeler/elm/sources`.

Models, all frozen:

* `elm-ours`, the serving ensemble: the mean of the five `cv2` fold models' event
  probability, at the mean of their thresholds (0.4173), run here on every shot (the
  traces `labeler.elm.serve` saved for 46 of the shots are checked against it);
* `elm-inst`, the five `inst1` fold models' mean occupancy at the mean occupancy
  threshold, and its onsets at the mean onset threshold in the decoding its development
  record chose;
* ELM-O (`elmo_benchmark.review_shot`), only on the shots with BES in the corpus and
  a fetched record, in memory (the benchmark's own work files are not touched);
* the ELM clock (v1, v2) and `elm_jump` v2, the suggestion tables.

Two scores. **Occupancy bins**: the benchmark's 50 ms bins from his crowd and absent
rows (`labeler.elm.heldout`). **Onset agreement** with his individual 10 ms rows, which
are the accepted `elm_jump` v2 prefill (17,109 of 17,128): circular for `elm_jump`, and
generous to anything that behaves like it. Each row carries its provenance,
`prefill_accepted` or `edited`.

Inputs are the fetched records `$LABELER_ROOT/benchmarks/elm/elmo/signals/<shot>.npz`,
read into memory (never into `round4/elm/inputs`); nothing is fetched. Writes
`<out-dir>/evaluation.json` (default `outputs/labeler/elm/heldout_2026_10_23`) and, under
`$LABELER_ROOT/round4/elm/heldout/2026-10-23`, each shot's traces and the individual
rows' provenance.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from labeler.config import Paths, git_sha, sha256_of
from labeler.elm import (
    challenge,
    compare,
    heldout,
    inputs,
    inst,
    labels,
    methods,
    onset,
    prepare,
    serve,
    sources,
    train,
)
from labeler.events import shotlists

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/labeler/elm/heldout_2026_10_23"
PIN_OLD, PIN_NEW = "2026-10-17", "2026-10-23"
OURS_RUN = "cv2"
INST_RUN = "round4/elm/inst/inst1"
FOLDS = 5
OURS, INST, ELMO = "elm-ours", "elm-inst", "elm-elmo"
CLOCKS = {"elm-clock": "v1", "elm-clock-v2": "v2"}
CLOCK_PEAKS = "elm-clock-peaks"
JUMP = "elm-jump"
SAVED_SERVE = ("silver_gold", "gold_shots")
PUBLISHED = Path("/scratch/gpfs/nc1514/FusionAIHub/outputs/labeler/elm")


def elmo_module():
    """`scripts/labeler/elmo_benchmark.py` as a module its worker processes can find."""
    spec = importlib.util.spec_from_file_location(
        "elmo_benchmark", REPO / "scripts/labeler/elmo_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["elmo_benchmark"] = module
    spec.loader.exec_module(module)
    return module


def inst_models(directory: Path):
    """The five `inst1` fold models (eval mode) and their fold records."""
    models, records = [], []
    for k in range(FOLDS):
        model = inst.InstNet()
        model.load_state_dict(
            torch.load(directory / f"fold{k}/model.pt", map_location="cpu", weights_only=True)
        )
        models.append(model.eval())
        records.append(json.loads((directory / f"fold{k}/fold.json").read_text()))
    return models, records


def run_elmo(elmo, paths: Paths, table: pd.DataFrame, shots, workers: int) -> dict[int, dict]:
    """ELM-O over each shot's labelled window, for the shots with BES and a fetched
    record: `{shot: review_shot's result}`."""
    signals = prepare.signals_dir(paths)
    windows = table.groupby("shot").agg(lo=("t_start", "min"), hi=("t_end", "max"))
    ready = [
        s for s in shots if (signals / f"{s}.npz").is_file() and elmo.bes_samples(s) > 1
    ]
    jobs = [(s, (float(windows.lo[s]), float(windows.hi[s])), signals) for s in ready]
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        return {job[0]: out for job, out in zip(jobs, pool.map(elmo.review_shot, jobs))}


def elmo_spans(result: dict) -> pd.DataFrame:
    rows = [r for r in result["elms"] if r[1] == "paper"]
    return methods.span_frame([r[2] for r in rows], [r[3] for r in rows]).sort_values(
        "t_start_ms"
    )


def old_panel(paths: Paths, published: Path) -> dict:
    """The old reviewed set scored by the same code from the saved traces."""
    reviewed = set(
        labels.review_table(prepare.review_csv(paths, PIN_OLD)).shot.astype(int)
    )
    have = {int(f.stem) for f in prepare.inputs_dir(paths).glob("*.npy")}
    data = train.load(paths, shots=sorted(reviewed & have))
    sets = compare.load_sets(paths, data)
    elmo_found, _ = compare.load_detected(paths)
    oof = methods.Oof(paths.root / "round4/elm/cv" / OURS_RUN)
    directory = paths.root / INST_RUN
    records = [
        json.loads((directory / f"fold{k}/fold.json").read_text()) for k in range(FOLDS)
    ]
    inst_fold = {
        s: k for k, r in enumerate(records) for s in r["shots"]["test"]["review"]
    }
    tables = {JUMP: sources.table(paths, "elm_jump", "v2")}
    tables.update({n: sources.table(paths, "elm_clock", v) for n, v in CLOCKS.items()})
    spans = {n: sources.present_spans(p) for n, p in tables.items()}
    out = {}
    for tag in ("all119", "bes73"):
        sdef = sets[tag]
        panels = {
            s: heldout.Panel(s, data[s].spans, sdef.cover[s], sdef.bins[s])
            for s in sdef.shots
        }
        called = {
            OURS: {
                s: heldout.Called(oof.trace(s)[0], oof.threshold[s]) for s in sdef.shots
            },
            INST: {
                s: heldout.Called(
                    np.load(directory / "pred" / f"{s}.npz")["occ"],
                    records[inst_fold[s]]["occ_threshold"],
                )
                for s in sdef.shots
                if s in inst_fold and (directory / "pred" / f"{s}.npz").is_file()
            },
        }
        for name, found in spans.items():
            called[name] = {
                s: heldout.Called(spans=found.get(s, methods.span_frame([], [])))
                for s in sdef.shots
            }
        if sdef.has_elmo:
            called[ELMO] = {
                s: heldout.Called(spans=elmo_found.get(s, methods.span_frame([], [])))
                for s in sdef.shots
            }
        out[tag] = heldout.occupancy_set(panels, called)
    record = {
        "pin": PIN_OLD,
        "protocol": "elm-ours and elm-inst are out of fold: each shot's own fold model "
        "and threshold (the 119 are in four of the five folds' training, so the serving "
        "ensemble would be in sample); ELM-O is the published run; the clock and "
        "elm_jump are tables",
        "sets": out,
    }
    reference = published / "sources/evaluation.json"
    if reference.is_file():
        old = json.loads(reference.read_text())
        record["reproduction"] = {
            "published_record": str(reference),
            "published_sha256": sha256_of(reference),
            "f1": {
                tag: {
                    name: {
                        "published": old["sets"][tag]["methods"][name]["point"]["f1"],
                        "here": out[tag]["methods"][name]["point"]["f1"],
                    }
                    for name in (JUMP, "elm-clock", OURS)
                    if name in old["sets"][tag]["methods"]
                    and name in out[tag]["methods"]
                }
                for tag in ("all119", "bes73")
            },
            "bins": {
                tag: {"published": old["sets"][tag]["bins"], "here": out[tag]["bins"]}
                for tag in ("all119", "bes73")
            },
        }
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shots", type=int, nargs="+", help="only these of his shots (a pilot)")
    ap.add_argument("--skip-old", action="store_true", help="leave out the old panel A")
    ap.add_argument("--no-run-days", action="store_true", help="skip the run-day overlap")
    ap.add_argument("--workers", type=int, default=8, help="ELM-O processes")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--work-dir", type=Path, help="traces and provenance rows")
    ap.add_argument("--published", type=Path, default=PUBLISHED, help="the published records")
    args = ap.parse_args(argv)
    paths = Paths.from_env()
    work = args.work_dir or paths.root / "round4/elm/heldout" / PIN_NEW
    (work / "traces").mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- the labels: two pins, never the live table
    pins = {}
    for pin in (PIN_OLD, PIN_NEW):
        directory = prepare.pins_dir(paths) / pin
        pins[pin] = {"dir": str(directory), "sha256": heldout.verify_pin(directory)}
    old = labels.review_table(prepare.review_csv(paths, PIN_OLD))
    new = labels.review_table(prepare.review_csv(paths, PIN_NEW))
    saves = heldout.last_saves(prepare.pins_dir(paths) / PIN_NEW / "history.jsonl")
    mine = sorted(saves)
    if not set(mine) <= set(new.shot):
        raise SystemExit("saves of his without rows in the table")
    if set(mine) & set(old.shot):
        raise SystemExit("his shots overlap the old reviewed set")
    shared = new[new.shot.isin(set(old.shot))].reset_index(drop=True)
    pins_agree = bool(shared.equals(old))
    if args.shots:
        mine = [s for s in mine if s in set(args.shots)]
    holdout = [s for s in mine if shotlists.held_out(paths, s)]
    groups = {"B1": [s for s in mine if s not in holdout], "B2": holdout}
    spans = {s: new[new.shot == s].reset_index(drop=True) for s in mine}

    # ---- ELM-O first: its worker processes fork before torch starts its threads
    signals = prepare.signals_dir(paths)
    elmo = elmo_module()
    elmo_out = run_elmo(
        elmo, paths, new, [s for s in mine if (signals / f"{s}.npz").is_file()], args.workers
    )

    # ---- the inputs, in memory
    valid, no_record = {}, {}
    ours_dir = paths.root / "round4/elm/cv" / OURS_RUN
    ours_models, ours_thresholds = serve.load_models(ours_dir)
    ours_thr = float(np.mean(ours_thresholds))
    ours_onset_thr = float(
        np.mean(
            [
                json.loads((ours_dir / f"fold{k}/fold.json").read_text())["onset_threshold"]
                for k in range(FOLDS)
            ]
        )
    )
    inst_nets, inst_folds = inst_models(paths.root / INST_RUN)
    occ_thr = float(np.mean([f["occ_threshold"] for f in inst_folds]))
    onset_thr = float(np.mean([f["onset_threshold"] for f in inst_folds]))
    record_path = args.published / "inst/evaluation.json"
    primary = json.loads(record_path.read_text())["primary_decoding"]
    gated = primary == "gated"
    ours_trace, ours_onsets, inst_occ, inst_onsets = {}, {}, {}, {}
    for shot in mine:
        record = signals / f"{shot}.npz"
        if not record.is_file():
            marker = signals / f"{shot}.missing.json"
            no_record[shot] = (
                json.loads(marker.read_text())["error"].split(". If this kernel")[0]
                if marker.is_file()
                else "not fetched"
            )
            continue
        x = inputs.read_channels(record)
        valid[shot] = x[inputs.VALID]
        heads = np.mean([train.predict(m, x, "cpu") for m in ours_models], axis=0)
        ours_trace[shot] = heads[0]  # the event head, as `serve.ensemble` returns it
        ours_onsets[shot] = inputs.GRID0_MS + onset.peaks(heads[1], ours_onset_thr) + 0.5
        outs = [inst.predict(m, x, "cpu") for m in inst_nets]
        occ = np.mean([o[0] for o in outs], axis=0)
        heat = np.mean([o[1] for o in outs], axis=0).astype(np.float32)
        inst_occ[shot] = occ
        inst_onsets[shot] = inst.onsets(
            heat, onset_thr, occ if gated else None, occ_thr if gated else None
        )
        np.savez_compressed(
            work / "traces" / f"{shot}.npz",
            ours=heads.astype(np.float32),
            inst_occ=occ.astype(np.float32),
            inst_heat=heat.astype(np.float16),
        )
        print(json.dumps({"shot": shot, "scored": True}), flush=True)
    scored = sorted(valid)

    # the traces `serve` saved before: the same ensemble, so the same numbers
    saved = {}
    for shot in scored:
        for name in SAVED_SERVE:
            path = paths.root / "round4/elm/serve" / name / f"{shot}.npz"
            if path.is_file():
                old_trace = np.load(path)["p"].astype(np.float32)
                n = min(len(old_trace), len(ours_trace[shot]))
                saved[shot] = {
                    "max_abs_diff": float(np.abs(old_trace[:n] - ours_trace[shot][:n]).max()),
                    "length_saved": len(old_trace),
                    "length_here": len(ours_trace[shot]),
                }
                break

    # ---- the tables
    tables = {n: sources.table(paths, "elm_clock", v) for n, v in CLOCKS.items()}
    tables[JUMP] = sources.table(paths, "elm_jump", "v2")
    held = {n: sources.held(p) for n, p in tables.items()}
    found_spans = {n: sources.present_spans(p) for n, p in tables.items()}
    jump_table = tables[JUMP]
    onsets_all = pd.read_csv(jump_table.with_name(jump_table.stem + "_onsets.csv"))
    onsets_of = {
        int(s): g.t_ms.to_numpy(float)
        for s, g in onsets_all[onsets_all.shot.isin(mine)].groupby("shot")
    }

    # the clock's own per-ELM detections (the challenge evaluation's `elm-clock`): the
    # published tables hold none of the 23 key shots
    clock_peaks, clock_info, clock_failed = {}, {}, {}
    for shot in scored:
        try:
            found_ms, clock_info[shot] = challenge.clock_spans(shot, paths)
        except Exception as error:  # noqa: BLE001 - recorded in the evaluation, never hidden
            clock_failed[shot] = f"{type(error).__name__}: {str(error)[:160]}"
            continue
        found_ms = np.asarray(found_ms, dtype=float).reshape(-1, 2)
        clock_peaks[shot] = methods.span_frame(found_ms[:, 0], found_ms[:, 1])

    # ---- panels and what each method called
    panels = {s: heldout.panel_of(s, spans[s], valid[s]) for s in scored}
    elmo_out = {s: out for s, out in elmo_out.items() if out["cover"]}
    bes = {s: heldout.bes_panel(panels[s], elmo_out[s]["cover"]) for s in elmo_out}
    called = {
        OURS: {s: heldout.Called(ours_trace[s], ours_thr) for s in scored},
        INST: {s: heldout.Called(inst_occ[s], occ_thr) for s in scored},
    }
    for name in tables:
        called[name] = {
            s: heldout.Called(spans=found_spans[name].get(s, methods.span_frame([], [])))
            for s in scored
            if s in held[name]
        }
    called[CLOCK_PEAKS] = {s: heldout.Called(spans=f) for s, f in clock_peaks.items()}
    called_bes = {
        **called,
        ELMO: {s: heldout.Called(spans=elmo_spans(elmo_out[s])) for s in elmo_out},
    }

    # ---- the individual rows: provenance and onset truth
    provenance, truth, per_shot = [], {}, {}
    totals = {"prefill_accepted": 0, "edited": 0, "onsets_removed": 0}
    for shot in mine:
        rows = spans[shot][spans[shot].kind == "non_crowd"]
        shown = onsets_of.get(shot, np.zeros(0))
        window = saves[shot]["window"]
        marked, removed = heldout.individual_provenance(rows, shown, window)
        provenance.append(marked)
        accepted = (marked.provenance == "prefill_accepted").to_numpy()
        totals["prefill_accepted"] += int(accepted.sum())
        totals["edited"] += int((~accepted).sum())
        totals["onsets_removed"] += removed
        agreement = heldout.span_agreement(
            found_spans[JUMP].get(shot, methods.span_frame([], [])).rename(
                columns={"t_start_ms": "t_start", "t_end_ms": "t_end"}
            ),
            spans[shot][spans[shot].kind == "crowd"],
        )
        per_shot[str(shot)] = {
            "panel": "B2" if shot in holdout else "B1",
            "saves": saves[shot]["saves"],
            "individual_rows": len(rows),
            "prefill_accepted": int(accepted.sum()),
            "edited": int((~accepted).sum()),
            "onsets_removed": removed,
            "crowd_rows": int((spans[shot].kind == "crowd").sum()),
            "uncertain_ms": float(
                (spans[shot].t_end - spans[shot].t_start)[spans[shot].kind == "uncertain"].sum()
            ),
            "span_agreement_with_elm_jump_draft": agreement,
            "scored": shot in valid,
            "elm_ours_saved_serve_trace": saved.get(shot),
            "has_bes": shot in elmo_out,
            "in_tables": [n for n in tables if shot in held[n]],
            "clock_peaks": clock_info.get(shot) or {"failed": clock_failed.get(shot)},
            "crowd_bins": int((panels[shot].bins.kind == "crowd").sum()) if shot in panels else None,
            "absent_bins": int((panels[shot].bins.kind == "absent").sum()) if shot in panels else None,
        }
        if shot in valid:
            n_ms = len(valid[shot]) // inputs.CELLS_PER_MS
            defined, absent = heldout.label_masks(spans[shot], n_ms)
            truth[shot] = heldout.OnsetTruth(
                ((rows.t_start + rows.t_end) / 2).to_numpy(float), accepted, defined, absent
            )
    provenance = pd.concat(provenance, ignore_index=True)
    provenance.to_csv(work / "individual_provenance.csv", index=False)

    found = {
        OURS: {s: ours_onsets[s] for s in scored},
        INST: {s: inst_onsets[s] for s in scored},
        JUMP: {s: onsets_of.get(s, np.zeros(0)) for s in scored if s in held[JUMP]},
        CLOCK_PEAKS: {s: f.t_start_ms.to_numpy(float) for s, f in clock_peaks.items()},
    }
    found_bes = {
        **found,
        ELMO: {s: elmo_spans(elmo_out[s]).t_start_ms.to_numpy(float) for s in elmo_out},
    }

    def cells(by_method):
        return {
            n: {s: heldout.to_grid(v) for s, v in by_shot.items()}
            for n, by_shot in by_method.items()
        }

    sets = {}
    for tag, shots in groups.items():
        ok = [s for s in shots if s in panels]
        sets[tag] = {
            "shots": shots,
            "shots_without_record": {str(s): no_record[s] for s in shots if s in no_record},
            "all": {
                "occupancy": heldout.occupancy_set({s: panels[s] for s in ok}, called),
                "onset": heldout.onset_set(
                    ok, {s: panels[s].cover for s in ok}, truth, found
                ),
                "onset_cells": heldout.onset_set(
                    ok, {s: panels[s].cover for s in ok}, truth, cells(found)
                ),
            },
        }
        with_bes = [s for s in ok if s in bes]
        sets[tag]["bes"] = {
            "occupancy": heldout.occupancy_set({s: bes[s] for s in with_bes}, called_bes),
            "onset": heldout.onset_set(
                with_bes, {s: bes[s].cover for s in with_bes}, truth, found_bes
            ),
            "onset_cells": heldout.onset_set(
                with_bes, {s: bes[s].cover for s in with_bes}, truth, cells(found_bes)
            ),
        }

    history = prepare.pins_dir(paths) / PIN_NEW / "history.jsonl"
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_sha(full=True),
        "pilot_shots": args.shots,
        "protocol": "frozen models, no refit, no threshold change; B1 and B2 are never "
        "pooled; his individual 10 ms rows are the accepted elm_jump v2 prefill, so "
        "onset agreement is circular for elm_jump and generous to jump-like methods; "
        "uncertain rows are in no score",
        "scripts_sha256": {
            p.name: sha256_of(p)
            for p in (
                Path(__file__),
                REPO / "src/labeler/elm/heldout.py",
                REPO / "scripts/labeler/elmo_benchmark.py",
            )
        },
        "pins": pins,
        "pins_agree_on_the_old_shots": pins_agree,
        "reviewer": heldout.REVIEWER,
        "history_sha256": sha256_of(history),
        "shots": {"B1": groups["B1"], "B2": groups["B2"], "without_record": no_record},
        "models": {
            OURS: {
                "run": OURS_RUN,
                "threshold": ours_thr,
                "fold_thresholds": ours_thresholds,
                "onset_threshold": ours_onset_thr,
                "checkpoints_sha256": [
                    sha256_of(paths.root / f"round4/elm/cv/{OURS_RUN}/fold{k}/model.pt")
                    for k in range(FOLDS)
                ],
                "saved_serve_traces": {
                    "shots": len(saved),
                    "max_abs_diff": max((v["max_abs_diff"] for v in saved.values()), default=None),
                    "length_mismatches": sorted(
                        s for s, v in saved.items() if v["length_saved"] != v["length_here"]
                    ),
                },
            },
            INST: {
                "run": INST_RUN,
                "occupancy_threshold": occ_thr,
                "onset_threshold": onset_thr,
                "primary_decoding": primary,
                "checkpoints_sha256": [f["checkpoint_sha256"] for f in inst_folds],
            },
            ELMO: {"shots_run": sorted(elmo_out), "setting": "published (eta 0.997)"},
        },
        "elm_clock_peaks": {
            "rule": "challenge.clock_spans: the ELM clock's peaks on the first finite "
            "filterscope, each the span of its half-prominence width",
            "shots": len(clock_peaks),
            "failed": {str(s): v for s, v in clock_failed.items()},
        },
        "suggestion_tables": {n: {"path": str(p), "sha256": sha256_of(p)} for n, p in tables.items()},
        "jump_onsets_sha256": sha256_of(jump_table.with_name(jump_table.stem + "_onsets.csv")),
        "provenance": {
            **totals,
            "individual_rows": len(provenance),
            "rows_file": str(work / "individual_provenance.csv"),
            "rows_sha256": sha256_of(work / "individual_provenance.csv"),
            "last_saves": len(saves),
            "last_saves_flagged_prefilled": sum(bool(v.get("prefilled")) for v in saves.values()),
            "last_saves_with_suggestions_shown": sum(
                bool(v.get("suggestions_shown")) for v in saves.values()
            ),
            "note": "the page flags the save that wrote the prefill; a later save of the "
            "same shot is not flagged, so the per-row provenance, not the flag, is the evidence",
        },
        "per_shot": per_shot,
        "panels": sets,
    }
    if not args.skip_old:
        record["panels"]["A"] = old_panel(paths, args.published)
    if not args.no_run_days:
        trained = sorted(set(old.shot))
        days = heldout.run_days(paths, [*mine, *trained])
        trained_days = {days[s] for s in trained if s in days}
        record["run_days"] = {
            "trained_shots": len(trained),
            "shots_missing_a_day": sorted(set(mine) - set(days)),
            "sharing_a_trained_run_day": {
                tag: sorted(s for s in shots if days.get(s) in trained_days)
                for tag, shots in groups.items()
            },
        }
    (args.out_dir / "evaluation.json").write_text(
        json.dumps(heldout.finite(record), indent=1) + "\n"
    )
    for tag, block in sets.items():
        for view in ("all", "bes"):
            occ = block[view]["occupancy"]
            print(tag, view, occ["n_shots"], "shots", occ["bins"], "bins", occ["bins_by_kind"])
            for name, res in occ["methods"].items():
                p = res["point"]
                print(
                    f"  {name:13s} n {len(res['shots_scored']):3d} F1 {p['f1']:.3f}",
                    f"crowd recall {p['crowd_bin_recall']:.3f}",
                    f"false alarm {p['false_alarm_bin_rate']:.3f}",
                )
            for name, res in block[view]["onset_cells"].items():
                p = res["tolerances_ms"]["10.0"]["point"]
                print(
                    f"  onsets {name:15s} n {len(res['shots_scored']):3d} @10 ms cells",
                    f"P {p['precision']:.3f} R {p['recall']:.3f} F1 {p['f1']:.3f}",
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
