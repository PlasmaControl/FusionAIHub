#!/usr/bin/env python
"""Gather a batch's designs and simulations into one table for analysis.

One row per prompt of the batch: the source shot and theme, the design the assistant saved
(reference / comparison shots, which channels it scaled and by how much, its explanation),
and the simulation's state plus its per-modality metrics scraped from ``report.md`` --
``frac_static``, ``token_acc``, ``persistence_acc``, ``skill`` (= token_acc - persistence_acc)
and ``divergence_vs_real`` (fraction of predicted tokens that differ between the real and the
proposed arm). Nothing is computed here that the pipeline did not already write; a design
without a simulation, or a failed one, keeps its row with the state and the error.

    SHOT_DESIGN_DATA_ROOT=<batch root> python scripts/shot_design/batch_collect.py \
        --designs <root>/designs/designs.jsonl --out <root>/summary
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import pandas as pd

METRICS = ("frac_static", "token_acc", "persistence_acc", "skill", "divergence_vs_real")


def _table(report: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if not report.exists():
        return out
    for line in report.read_text().splitlines():
        if (
            not line.startswith("|")
            or line.startswith("| modality")
            or set(line) <= {"|", "-", " "}
        ):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 6:
            continue
        try:
            out[cells[0]] = dict(
                zip(METRICS, (float(c) for c in cells[1:]), strict=True)
            )
        except ValueError:
            continue
    return out


def _scale_summary(edits: dict, h5_meta: dict) -> tuple[dict[str, float], list[str]]:
    scales = {
        k: float(v) for k, v in (h5_meta.get("scales") or {}).items() if float(v) != 1.0
    }
    return scales, sorted(edits)


def _design(ident: str, root: Path) -> dict:
    rec: dict = {"design_id": ident}
    prog = root / "actuations" / "designs" / f"{ident}.json"
    if prog.exists():
        d = json.loads(prog.read_text())
        rec.update(
            reference_shot=d.get("reference_shot"),
            comparison_shots=list(d.get("comparison_shots") or []),
            start_s=d.get("start_s"),
            end_s=d.get("end_s"),
            edited_channels=sorted(d.get("edits") or {}),
            n_edits=len(d.get("edits") or {}),
        )
    h5 = root / "outputs" / f"{ident}.h5"
    meta: dict = {}
    if h5.exists():
        try:
            with h5py.File(h5, "r") as f:
                meta = json.loads(f["metadata"][()])
        except (OSError, KeyError, ValueError):
            meta = {}
    scales = {
        k: float(v) for k, v in (meta.get("scales") or {}).items() if float(v) != 1.0
    }
    rec.update(
        llm_model=meta.get("model"),
        baseline=meta.get("baseline"),
        goal=meta.get("goal"),
        explanation=meta.get("explanation"),
        scales=scales,
        n_scaled=len(scales),
        scale_min=min(scales.values()) if scales else None,
        scale_max=max(scales.values()) if scales else None,
        n_candidates=len(meta.get("retrieved_candidates") or []),
        n_unusable=len(meta.get("unusable_candidates") or []),
        candidate_shots=[
            c.get("shot") for c in (meta.get("retrieved_candidates") or [])
        ],
        n_warnings=len(((meta.get("checks") or {}).get("warnings")) or []),
    )
    sim = root / "outputs" / ident / "simulation"
    st = sim / "status.json"
    status = json.loads(st.read_text()) if st.exists() else {}
    rec.update(
        sim_state=status.get("state", "not_run"),
        sim_error=status.get("error"),
        sim_started=status.get("started"),
        sim_finished=status.get("finished"),
        sim_dir=str(sim) if sim.exists() else None,
    )
    if status.get("started") and status.get("finished"):
        t0, t1 = pd.Timestamp(status["started"]), pd.Timestamp(status["finished"])
        rec["sim_seconds"] = round((t1 - t0).total_seconds(), 1)
    h5s = sim / "simulation.h5"
    if h5s.exists():
        try:
            with h5py.File(h5s, "r") as f:
                rec["codec_generation"] = f.attrs.get("codec_generation")
                rec["frame_origin_s"] = float(f.attrs["frame_origin_s"])
                rec["dynamics_step"] = int(f.attrs.get("dynamics_step", -1))
                rec["window_s"] = [float(x) for x in f.attrs.get("window_s", [])]
        except OSError:
            pass
    for m, vals in _table(sim / "report.md").items():
        for k, v in vals.items():
            rec[f"{m}.{k}"] = v
    panels = sorted((sim / "panels").glob("*.png")) if (sim / "panels").exists() else []
    rec["panels"] = [str(p) for p in panels]
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--designs", required=True, type=Path, help="batch_design.py's output"
    )
    ap.add_argument(
        "--out",
        required=True,
        type=Path,
        help="directory for summary.{parquet,json,md}",
    )
    args = ap.parse_args()
    root = Path(
        os.environ.get("SHOT_DESIGN_DATA_ROOT") or sys.exit("set SHOT_DESIGN_DATA_ROOT")
    )
    rows = []
    for line in args.designs.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        row = {
            "shot": r["shot"],
            "theme": r.get("theme"),
            "year": r.get("year"),
            "run_id": r.get("run_id"),
            "mpid": r.get("mpid"),
            "mp_title": r.get("mp_title"),
            "prompt": r.get("prompt"),
            "design_error": r.get("error"),
            "design_seconds": r.get("elapsed_s"),
        }
        if r.get("design_id"):
            row.update(_design(r["design_id"], root))
        else:
            row["design_id"] = None
            row["sim_state"] = "no_design"
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("shot").reset_index(drop=True)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out / "summary.parquet", index=False)
    df.to_json(args.out / "summary.json", orient="records", indent=1)
    n = len(df)
    designed = int(df["design_id"].notna().sum())
    done = int((df["sim_state"] == "complete").sum())
    failed = int((df["sim_state"] == "failed").sum())
    anchored = (
        int((df.get("reference_shot") == df["shot"]).sum())
        if "reference_shot" in df
        else 0
    )
    lines = [
        "# Batch summary",
        "",
        "| prompts | designs | reference = source shot | simulations done | simulations failed |",
        "|---|---|---|---|---|",
        f"| {n} | {designed} | {anchored} | {done} | {failed} |",
        "",
    ]
    if done:
        mods = sorted({c.split(".")[0] for c in df.columns if c.endswith(".skill")})
        lines += [
            "| modality | median token_acc | median persistence_acc | median skill | median divergence |",
            "|---|---|---|---|---|",
        ]
        ok = df[df["sim_state"] == "complete"]
        for m in mods:
            lines.append(
                f"| {m} | {ok[f'{m}.token_acc'].median():.3f} | {ok[f'{m}.persistence_acc'].median():.3f} | "
                f"{ok[f'{m}.skill'].median():.3f} | {ok[f'{m}.divergence_vs_real'].median():.3f} |"
            )
        lines.append("")
    if "theme" in df:
        lines += ["| theme | prompts | designs | sims done |", "|---|---|---|---|"]
        for theme, g in df.groupby(df["theme"].fillna("none")):
            lines.append(
                f"| {theme} | {len(g)} | {int(g['design_id'].notna().sum())} | {int((g['sim_state'] == 'complete').sum())} |"
            )
        lines.append("")
    (args.out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {args.out}/summary.{{parquet,json,md}} ({n} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
