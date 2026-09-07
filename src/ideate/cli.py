"""The ideate CLI: build, add, show, export, coverage, query, model, llm and actuation.

Build from legacy raw signals and operator text, retrieve similar shots, and inspect
saved actuator sets. Thresholds, node names and units live in configs/ideate/.

Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path
from typing import get_args

import yaml

from . import config
from .retrieval import describe as describe_mod
from .retrieval import rank as rank_mod
from .schema import QueryState, Range, SegName, ShotRecord, to_summary
from .shotdb import build as build_mod
from .shotdb import census, corpus_signals, legacy_raw, store
from .shotdb import logs as logs_mod
from .shotdb import select as select_mod
from .shotdb.reader import ShotFailed

# sentence_transformers reaches into huggingface_hub on every model load, even though the MiniLM
# checkpoint is already in ~/.cache/huggingface/hub. On a compute node with no outbound route that
# call does not fail -- it hangs, for minutes, inside httpx.connect_tcp (measured during Task 12;
# the same load takes ~5 s with HF_HUB_OFFLINE=1). Set here rather than in the Makefile so that
# `uv run ideate build` behaves the same however it is invoked, and at import time so the build's
# worker processes inherit it. Below the imports is early enough -- huggingface_hub reads this
# variable when IT is imported, which is lazily, inside text._load_model. IDEATE_HF_ONLINE=1 opts
# out, which is what a first run on a machine with no cached checkpoint needs.
if os.environ.get("IDEATE_HF_ONLINE") != "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

WIDTH = 98

# Reading order for the non-actuator scalars in `show`. Presentation only -- a quantity that is
# not listed still prints, after these, in alphabetical order.
HEADLINE = (
    "ip", "bt", "q95", "qmin", "q0", "betan", "betap", "li", "wmhd", "ne_line", "ne0", "te0",
    "kappa", "tritop", "tribot", "aminor", "r0", "volume", "drsep", "zxpt1", "gapin", "gapout",
    "vloop", "dalpha", "neutrons", "n1rms", "n2rms",
)  # fmt: skip


# ------------------------------------------------------------------------------- shot selection


def _shots(args, default_list: str | None = None) -> list[int]:
    """--shots, --list and --list-file, unioned.
    `default_list` applies only if none of the three was given.

    Not `elif`: `ideate build --list poc_v1 --list-file extra.yaml` silently built only the
    extra file's shots and dropped all 200 of poc_v1's when these were exclusive branches.
    """
    shots = set(getattr(args, "shots", None) or [])
    list_file, list_name = getattr(args, "list_file", None), getattr(args, "list", None)
    if list_file:
        shots |= set(config.load_shot_list(path=Path(list_file)))
    if list_name:
        shots |= set(config.load_shot_list(list_name))
    if not (shots or list_file or list_name) and default_list:
        shots = set(config.load_shot_list(default_list))
    return sorted(shots)


def _shot_source(args, default_list: str | None = None) -> str:
    """How `_shots` chose this build's shots, as one recordable string.

    `list:<name>`, `list-file:<path>`, `shots:<n>` -- joined with `+` when more than one selector
    was given, because `_shots` unions them. The manifest keeps this instead of `args.list`, which
    was null for two of the three ways to select shots.
    """
    parts = []
    if getattr(args, "list", None):
        parts.append(f"list:{args.list}")
    if getattr(args, "list_file", None):
        parts.append(f"list-file:{args.list_file}")
    if getattr(args, "shots", None):
        parts.append(f"shots:{len(args.shots)}")
    return "+".join(parts) if parts else (f"list:{default_list}" if default_list else "none")


def _ip_spec() -> config.SignalSpec:
    """Ip's registry entry. It is a fixed signal, so it does not depend on the shot."""
    return config.SignalSpec(name="ip", **config.load_yaml("signals.yaml")["signals"]["ip"])


def _with_ip(shots: list[int], paths: config.Paths, reader=None) -> tuple[list[int], list[int]]:
    """Split a shot list into (has an Ip trace on disk, does not).

    `build_record` does not fail on a shot with no raw file: it returns a record with zero
    segments and an `end_reason` of "no_ip_signal", which is the right answer for one shot and the
    wrong thing to put in a database for ninety. Ip is the discriminator because it is what
    `find_segments` needs -- a file that exists but is still being written by a bulk fetch has no
    usable Ip yet and belongs on the skipped side too.

    Asked of the SAME reader the build will use. Asking the d3d_fusion_data layout about a corpus
    shot answers "no Ip" for every one of them -- the two layouts hold disjoint shot ranges -- and
    the build would then skip its entire list.
    """
    spec = _ip_spec()
    reader = reader or legacy_raw.LegacyReader(paths)
    have = []
    for s in shots:
        try:
            if reader.signal_status(s, spec) == "present":
                have.append(s)
        except ShotFailed:  # an unopenable raw file is not a shot with an Ip trace
            continue
    return have, [s for s in shots if s not in set(have)]


def _brief(shots: list[int], n: int = 8) -> str:
    head = ", ".join(str(s) for s in shots[:n])
    return head if len(shots) <= n else f"{head}, ... (+{len(shots) - n})"


# --------------------------------------------------------------------------------- db plumbing


def _open_db(paths: config.Paths) -> store.ShotDB | None:
    if not (paths.db_dir / "manifest.json").exists():
        print(
            f"no database at {paths.db_dir} -- run `ideate build --list poc_v1` first",
            file=sys.stderr,
        )
        return None
    return store.ShotDB.load(paths.db_dir)


def _record(db: store.ShotDB, shot: int) -> ShotRecord | None:
    if shot not in db.shots.index:
        held = sorted(int(s) for s in db.shots.index)
        span = f"{held[0]}-{held[-1]}" if held else "(empty)"
        print(
            f"shot {shot} is not in the database ({len(held)} shots, {span}). "
            f"Add it with `ideate add {shot}`.",
            file=sys.stderr,
        )
        return None
    return db.get(shot)


# ------------------------------------------------------------------------------------ printing


def _fill(text: str, indent: str = "  ", hang: str = "    ") -> str:
    return textwrap.fill(text, width=WIDTH, initial_indent=indent, subsequent_indent=hang)


def _num(v: float) -> str:
    return f"{v:.4g}"


def _system_line(prefix: str, vals: dict[str, float | None], units: dict[str, str]) -> str:
    """One actuator system on one line: the total, the members that ran, the ones that did not,
    and -- kept separate on purpose -- the ones nothing was recorded for.

    "recorded and reading zero" (an idle gyrotron) and "not recorded" (no data for that channel)
    are different facts everywhere else in this codebase; collapsing them here would be the one
    place a reader could not tell them apart.
    """
    stat = "peak" if f"{prefix}_total_peak" in vals else "mean"
    unit = units.get(f"{prefix}_total", "")
    members: dict[str, float | None] = {}
    for col, v in vals.items():
        base, s = rank_mod.split_stat(col)
        if s == stat and base.startswith(f"{prefix}_") and base != f"{prefix}_total":
            members[base[len(prefix) + 1 :]] = v
    on = {m: v for m, v in members.items() if v is not None and v > 0}
    idle = sorted(m for m, v in members.items() if v == 0)
    unknown = sorted(m for m, v in members.items() if v is None)
    total = vals.get(f"{prefix}_total_{stat}")
    parts = [f"total {_num(total) if total is not None else '[not recorded]'} {unit} ({stat})"]
    parts.append(
        "on: " + ", ".join(f"{m} {_num(v)}" for m, v in sorted(on.items())) if on else "on: none"
    )
    if idle:
        parts.append(f"idle ({len(idle)}): {' '.join(idle)}")
    if unknown:
        parts.append(f"not recorded ({len(unknown)}): {' '.join(unknown)}")
    return "; ".join(parts)


def _print_segment(seg, units: dict[str, str], systems: dict[str, str], full: bool) -> None:
    vals = {**seg.raw, **seg.derived}
    recorded = {k: v for k, v in vals.items() if v is not None}
    print(
        f"[{seg.name}] {seg.t0_ms:.0f}-{seg.t1_ms:.0f} ms "
        f"({seg.t1_ms - seg.t0_ms:.0f} ms), {len(recorded)}/{len(vals)} scalars recorded"
    )
    if full:
        print(_fill(", ".join(f"{k}={_num(v)}" for k, v in sorted(recorded.items()))))
        return
    plasma: dict[str, dict[str, float]] = {}
    for col, v in recorded.items():
        base, stat = rank_mod.split_stat(col)
        if any(base.startswith(f"{p}_") for p in systems):
            continue
        plasma.setdefault(base, {})[stat or "value"] = v
    order = {name: i for i, name in enumerate(HEADLINE)}
    items = sorted(plasma.items(), key=lambda kv: (order.get(kv[0], len(order)), kv[0]))
    if items:
        shown = []
        for base, stats in items:
            v = stats.get("mean", stats.get("peak", next(iter(stats.values()))))
            shown.append(f"{base} {_num(v)} {units.get(base, '')}".strip())
        print(_fill(", ".join(shown)))
    for prefix, name in sorted(systems.items(), key=lambda kv: kv[1]):
        if any(c.startswith(f"{prefix}_") for c in vals):
            print(_fill(f"{name:<5} {_system_line(prefix, vals, units)}"))


def _print_record(rec: ShotRecord, segment: str, full: bool) -> None:
    h, units = rec.human, rank_mod.units()
    systems = {s.prefix: n for n, s in config.actuator_systems(rec.shot).items()}
    print(
        f"Shot {rec.shot}   {rec.shot_date or 'date [?]'}   run {h.run_id or '[?]'}   "
        f"campaign {rec.campaign}"
    )
    if h.mpid or h.mp_title:
        print(_fill(f"MP {h.mpid or '[?]'}  {h.mp_title or ''}".strip()))
    # The headline is retrieval.describe's segment line -- the same text a query result prints
    # for this shot -- so `show` and `query` cannot disagree about a number or its unit.
    headline = describe_mod.segment_line(rec, "flat_top" if segment == "all" else segment)
    if headline is None:
        headline = describe_mod.segment_line(rec, "full")
    print(_fill(headline or "no segment scalars recorded"))
    print()

    ops = ", ".join(sorted(rec.labels.operational)) or "none"
    print(f"labels    regime {rec.labels.regime} (source: {rec.labels.regime_source})")
    print(_fill(f"operational: {ops}"))
    print(f"verdict   {h.verdict}")
    if h.chief_operator_status:
        print(_fill(f"chief operator: {h.chief_operator_status}"))
    outcome = {
        k: _num(v) if isinstance(v, float) else v
        for k, v in rec.outcome.model_dump(exclude_none=True).items()
    }
    print(
        _fill(
            ", ".join(f"{k}={v}" for k, v in outcome.items()) or "(nothing derived)",
            indent="outcome   ",
            hang="  ",
        )
    )
    print()

    print(
        "segments  " + " | ".join(f"{s.name} {s.t0_ms:.0f}-{s.t1_ms:.0f} ms" for s in rec.segments)
    )
    wanted = [s for s in rec.segments if segment in ("all", s.name)]
    if not wanted and rec.segments:
        print(f"  (no segment named {segment!r})")
    for seg in wanted:
        _print_segment(seg, units, systems, full)
    print()

    by_status: dict[str, list[str]] = {}
    for name, status in rec.coverage.items():
        by_status.setdefault(status, []).append(name)
    n_present = len(by_status.get("present", []))
    print(f"coverage  {n_present}/{len(rec.coverage)} registry fields present")
    for status in ("pending", "unavailable", "not_installed"):
        names = sorted(by_status.get(status, []))
        if names:
            print(_fill(f"{status} ({len(names)}): {' '.join(names)}"))

    groups: dict[str, list[str]] = {}
    for name, p in rec.derived_provenance.items():
        key = " ".join(
            filter(
                None,
                [
                    f"{p.tool}{p.version or ''}",
                    f"tree={p.tree}" if p.tree else "",
                    "(run id assumed: staged files do not record it)" if p.assumed else "",
                ],
            )
        )
        groups.setdefault(key, []).append(name)
    print("provenance")
    for key, names in sorted(groups.items()):
        print(_fill(f"{key}: {' '.join(sorted(names))}"))
    src = Counter(rec.raw_sources.values())
    print(
        f"raw       {', '.join(f'{n} {s}' for s, n in sorted(src.items())) or 'nothing read'}"
        f"   built {rec.built_at:%Y-%m-%d %H:%M} UTC by {rec.builder_sha}"
    )


# ------------------------------------------------------------------------------------ commands


def _coverage_table(db: store.ShotDB) -> str:
    """Recomputed from the records rather than read from manifest["coverage"], which `add()`
    does not update -- after an incremental add the manifest's copy describes the previous
    build's shot set."""
    records = [db.get(int(s)) for s in db.shots.index]
    return build_mod.format_coverage(build_mod.coverage_report(records))


def cmd_build(args) -> int:
    paths, cfg = config.load_paths(), build_mod.load_build_cfg()
    wanted = _shots(args, "poc_v1")
    source, n_requested = _shot_source(args, "poc_v1"), len(wanted)
    # `--limit` before anything else, and on the sorted list, so a pilot and the full run agree on
    # which shots the pilot measured.
    if args.limit is not None:
        wanted = wanted[: args.limit]
    reader = build_mod.make_reader(args.reader, paths)
    shots, skipped = (wanted, []) if args.all else _with_ip(wanted, paths, reader)
    if skipped:
        print(
            _fill(
                f"{len(skipped)} of {len(wanted)} shots have no Ip signal on disk yet and are "
                f"skipped (fetch them first, or pass --all to build them as empty records): "
                f"{_brief(skipped)}",
                indent="",
                hang="  ",
            )
        )
    if not shots:
        print("nothing to build", file=sys.stderr)
        return 1
    report = build_mod.build(
        shots,
        paths,
        cfg,
        workers=args.workers,
        encode=not args.no_encode,
        reuse=not args.reencode,
        reader_kind=args.reader,
        shot_source=source,
        n_requested=n_requested,
        limit=args.limit,
    )
    print(
        f"built {len(report.shots)} shots / {report.n_segments} segments in "
        f"{report.elapsed_s:.1f} s -> {report.db_dir}"
    )
    for shot, err in sorted(report.failed.items()):
        print(f"  FAILED {shot}: {err}")
    db = _open_db(paths)
    if db is None:
        return 1
    ignite = db.manifest.get("ignite", {})
    # The flag belongs to this layer: `build()` records what was skipped, and the CLI names the
    # flag that skipped it, so a programmatic build's manifest never claims a flag was passed.
    flag = " (--no-encode)" if args.no_encode and ignite.get("status") == "disabled" else ""
    print(f"ignite embeddings: {ignite.get('status')} -- {ignite.get('reason')}{flag}")
    print()
    print(_coverage_table(db))
    return 0 if report.shots else 1


def cmd_model(args) -> int:
    """Where the IGNITE bundle is, what it holds, and -- with --download -- fetch it."""
    from .shotdb import ignite

    paths = config.load_paths()
    mcfg = ignite.model_cfg()
    target = ignite.bundle_dir(paths)
    if args.download:
        print(f"downloading {mcfg['repo_id']} @ {mcfg['revision'][:12]} -> {target}")
        try:
            ignite.download_bundle(paths, full=args.full)
        except Exception as e:  # noqa: BLE001 — optional operation; preserve the fallback contract
            print(f"download failed: {type(e).__name__}: {e}", file=sys.stderr)
            print(
                "the repo is private: `uv run hf auth login` (or HF_TOKEN) with an account that "
                "can read it, then retry",
                file=sys.stderr,
            )
            return 1
    manifest = ignite.codec_manifest(target)
    if not manifest.exists():
        print(f"no bundle at {target} -- run `ideate model --download`")
        return 1
    entries = json.loads(manifest.read_text())["modalities"]
    have = [n for n in entries if (target / "codecs" / n / "codec_best.pt").exists()]
    dyn = target / mcfg["dynamics_file"]
    print(f"bundle: {target}  ({mcfg['repo_id']} @ {mcfg['revision'][:12]})")
    print(f"codecs: {len(have)}/{len(entries)} present -- {', '.join(have)}")
    print(
        f"dynamics model: {'present' if dyn.exists() else 'not downloaded (--download --full)'} "
        f"({mcfg['dynamics_file']})"
    )
    for n, e in entries.items():
        mark = " " if n in have else "!"
        print(
            f"  {mark} {n:24s} {e['family']:8s} {e['channels']:3d} ch  vocab {e['codebook_size']}"
        )
    return 0


def cmd_add(args) -> int:
    paths, cfg = config.load_paths(), build_mod.load_build_cfg()
    if not (paths.db_dir / "manifest.json").exists():
        print(
            f"no database at {paths.db_dir} -- `ideate build` before `ideate add`", file=sys.stderr
        )
        return 1
    report = build_mod.add(args.shots, paths, cfg, workers=args.workers)
    print(
        f"added {len(report.shots)} shot{'s' if len(report.shots) != 1 else ''}: "
        f"{_brief(report.shots)}; database now holds {report.n_segments} segments "
        f"({report.elapsed_s:.1f} s)"
    )
    for shot, err in sorted(report.failed.items()):
        print(f"  FAILED {shot}: {err}")
    return 0 if report.shots else 1


def cmd_show(args) -> int:
    db = _open_db(config.load_paths())
    if db is None:
        return 1
    rec = _record(db, args.shot)
    if rec is None:
        return 1
    if args.json:
        print(rec.model_dump_json(indent=1))
        return 0
    _print_record(rec, args.segment, args.full)
    return 0


def cmd_export(args) -> int:
    db = _open_db(config.load_paths())
    if db is None:
        return 1
    if args.format == "parquet" and not args.out:
        print("--format parquet needs --out PATH", file=sys.stderr)
        return 2
    rows = []
    for shot in args.shots:
        rec = _record(db, shot)
        if rec is None:
            return 1
        rows.append(to_summary(rec, describe_mod.describe(rec)).model_dump(mode="json"))
    if args.format == "json":
        blob = json.dumps(rows, indent=1, default=str)
        if not args.out:
            print(blob)
            return 0
        Path(args.out).write_text(blob)
    else:
        import pandas as pd

        pd.json_normalize(rows).to_parquet(args.out)
    print(f"wrote {len(rows)} ShotSummary rows -> {args.out}", file=sys.stderr)
    return 0


def cmd_encode(args) -> int:
    """IGNITE frame-code caches for a shot list: the seeds a Phase-5 rollout starts from."""
    from .design import seed as seed_mod
    from .shotdb.corpus import CorpusReader

    paths = config.load_paths()
    shots = sorted(_shots(args, "recommender_v1"))
    if args.n_chunks > 1:
        shots = seed_mod.chunk_of(shots, args.chunk, args.n_chunks)
    if args.limit is not None:
        shots = shots[: args.limit]
    if not shots:
        print("nothing to encode", file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else Path(paths.data_root) / "frame_codes"
    report = seed_mod.encode_many(
        shots,
        reader=CorpusReader(paths.foundation_model_processed_dir),
        out_dir=out,
        device=args.device,
        include_video=not args.no_video,
        skip_existing=args.skip_existing,
        workers=args.workers,
        paths=paths,
    )
    run_dir = Path(paths.data_root) / "runs" / "encode"
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    manifest = run_dir / f"encode_{stamp}_{args.chunk}of{args.n_chunks}.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"encoded {report['n_encoded']}/{report['n_requested']} shots "
        f"({report['n_skipped']} already present, {len(report['failed'])} failed) in "
        f"{report['elapsed_s']:.1f} s -> {out}"
    )
    if report["s_per_shot_mean"] is not None:
        print(
            f"  {report['s_per_shot_mean']:.2f} s/shot mean, "
            f"{report['s_per_shot_median']:.2f} s median, {report['s_per_shot_max']:.2f} s max"
        )
    for shot, err in report["failed"].items():
        print(f"  FAILED {shot}: {err}", file=sys.stderr)
    print(f"  run manifest -> {manifest}")
    return 0 if report["n_encoded"] or report["n_skipped"] else 1


def cmd_coverage(args) -> int:
    db = _open_db(config.load_paths())
    if db is None:
        return 1
    print(_coverage_table(db))
    return 0


def cmd_llm(args) -> int:
    from .llm.client import LLMClient

    client = LLMClient()
    ok, hint = client.available()
    if not ok:
        print(hint)
        return 1
    ep = client.endpoint()
    print(
        f"{ep.url}  models {', '.join(ep.models) or '?'}  host {ep.host or '?'}"
        + (f"  job {ep.job_id}" if ep.job_id else "")
        + (f"  since {ep.started}" if ep.started else "")
    )
    return 0


def cmd_blurb(args) -> int:
    from .llm.client import LLMClient
    from .shotdb.build import write_blurbs

    client = LLMClient()
    ok, hint = client.available()
    if not ok:
        print(hint)
        return 1
    n = write_blurbs(config.load_paths(), client, only_missing=not args.all)
    print(f"{n} blurbs written by {client.model(client.cfg['blurb']['model'])}")
    return 0


def cmd_actuation(args) -> int:
    from .retrieval import actuation

    paths = config.load_paths()
    if args.what == "list":
        items = actuation.list_sets(paths)
        if not items:
            print(f"no saved actuation sets in {paths.actuations_dir}")
            return 0
        for it in items:
            print(
                f"{it['id']}  {it['created']}  shot {it['source_shot']}  "
                f"{it['n_waveforms']} waveforms"
            )
        return 0
    try:
        aset = actuation.load(args.id, paths)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    if aset is None:
        print(f"no actuation set {args.id} in {paths.actuations_dir}", file=sys.stderr)
        return 1
    if args.json:
        print(aset.model_dump_json(indent=1))
        return 0
    print(
        f"{aset.id}  created {aset.created}  source shot {aset.source_shot}  segment {aset.segment}"
    )
    for key, w in aset.waveforms.items():
        pts = ", ".join(f"({v.t_s:g} s, {_num(v.y)})" for v in w.vertices)
        print(f"  {key} [{w.units or '?'}] seed={w.seed}: {pts}")
    if aset.gas_species:
        print(
            "  gas species: " + ", ".join(f"{v}={s}" for v, s in sorted(aset.gas_species.items()))
        )
    for f in aset.flags:
        print(f"  {f.severity}: {f.message}")
    if aset.notes:
        print(f"  notes: {aset.notes}")
    return 0


# --------------------------------------------------------------------------- phase 2 (wired)


def _constraint(text: str) -> tuple[str, Range]:
    """COLUMN=LO:HI, or COLUMN:LO:HI -- the same thing with a colon, which is how --where reads."""
    col, sep, span = text.partition("=") if "=" in text else text.partition(":")
    lo, colon, hi = span.partition(":")
    if not col or not sep or not colon:
        raise argparse.ArgumentTypeError(
            f"expected COLUMN=LO:HI (either bound may be empty), got {text!r}"
        )
    try:
        return col, Range(lo=float(lo) if lo else None, hi=float(hi) if hi else None)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"{text!r}: {e}") from e


def _actuator(text: str) -> tuple[str, float]:
    name, sep, value = text.partition("=")
    if not name or not sep:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {text!r}")
    try:
        return name, float(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"{text!r}: {e}") from e


def _query_state(args) -> QueryState:
    return QueryState(
        text=args.text,
        negatives=args.negative,
        ref_shot=args.ref_shot,
        ref_window_ms=tuple(args.ref_window) if args.ref_window else None,
        segment=args.segment,
        constraints=dict(args.constraint),
        actuators=dict(args.actuator),
        require_labels=set(args.require),
        avoid_labels=set(args.avoid),
        exclude_shots=set(args.exclude_shot),
        exclude_runs=set(args.exclude_run),
        prefer_outcome=args.prefer_outcome,
        n=args.n,
    )


def _query_header(state: QueryState, report: dict, fired: dict[str, int]) -> None:
    """What the query asked, what survived the hard filter, and which channels had anything to
    say -- printed before the results because a top-5 out of three candidates is a different
    answer from a top-5 out of three thousand, and the reader cannot tell them apart otherwise."""
    asked = []
    if state.ref_shot:
        asked.append(f"ref {state.ref_shot}")
    if state.text:
        asked.append(f"text {state.text!r}")
    for col, rng in state.constraints.items():
        asked.append(
            f"{col} in [{rng.lo if rng.lo is not None else ''}, {rng.hi if rng.hi is not None else ''}]"
        )
    for name, value in state.actuators.items():
        asked.append(f"{name}={_num(value)}")
    if state.require_labels:
        asked.append("require " + ",".join(sorted(state.require_labels)))
    if state.avoid_labels:
        asked.append("avoid " + ",".join(sorted(state.avoid_labels)))
    print(f"query     {'; '.join(asked) or '(nothing specified)'}   segment {state.segment}")
    print(f"candidates {report['candidates']} of {report['segment_rows']} {state.segment} rows")
    for col, n in report["nan_excluded"].items():
        print(
            f"  note    {n} of them were dropped because {col} was never recorded, not because of its value"
        )
    print("channels  " + ", ".join(f"{k} {v}" for k, v in fired.items()))
    print()


def _print_proposal(state: QueryState, flags) -> None:
    """The user's own `--actuator` settings checked against configs/ideate/flags.yaml, printed
    before the results. "Can DIII-D do this at all" is a separate answer from "what has it done
    like this", and it was not being given: the rules ran on the result rows only, so
    `--actuator nbi.total=5e7` (2.5x the installed beam power) printed no flag."""
    if not state.actuators:
        return
    print("proposal  " + ", ".join(f"{k}={_num(v)}" for k, v in state.actuators.items()))
    loud = [f for f in flags if f.severity != "info"]
    for f in loud:
        print(_fill(f"{f.severity:<9} {f.message}", indent="  ", hang="            "))
    if not loud:
        print("  ok        no configured limit violated")
    # The rules that had nothing to check: with one actuator proposed, most of them. Listed by id
    # so "no limit violated" is never read as "every limit checked".
    skipped = [f.rule_id for f in flags if f.severity == "info"]
    if skipped:
        print(
            _fill(
                f"info      {len(skipped)} rules had no input to check: {', '.join(skipped)}",
                indent="  ",
                hang="            ",
            )
        )
    print()


def _print_results(results, db, state: QueryState) -> None:
    # What the query itself asks for, so each similar/differs line can show the pair it compared
    # rather than a lone number and a z-distance the reader cannot check.
    qvals = rank_mod.query_values(state, db)
    for i, r in enumerate(results, start=1):
        head = f"{i:2d}. {r.shot}  score {r.score:.4f}  run {r.run_id or '[?]'}"
        if r.labels.regime != "unknown":
            head += f"  regime {r.labels.regime}"
        print(head)
        print(_fill(r.description, indent="    ", hang="    "))
        e = r.explanation
        for line in e.matched_constraints:
            print(_fill(f"met       {line}", indent="    ", hang="              "))
        for label, feats in (("similar", e.top_similar), ("differs", e.top_different)):
            if feats:
                shown = ", ".join(_feature(c, d, v, qvals.get(c)) for c, d, v in feats)
                print(_fill(f"{label:<9} {shown}", indent="    ", hang="              "))
        if e.channel_ranks:
            print(
                "    ranked    "
                + ", ".join(f"{k} #{v}" for k, v in sorted(e.channel_ranks.items()))
            )
        if e.text_highlight:
            print(_fill(f'log       "{e.text_highlight}"', indent="    ", hang="              "))
        for f in r.flags:
            if f.severity != "info":
                print(_fill(f"{f.severity:<9} {f.message}", indent="    ", hang="              "))
        # Every info-level flag on this database is one rule reporting that it could not run for
        # want of an input, and there are five of them on most shots. They are the same five on
        # every result, so listing them in full under each one buries the results it is meant to
        # annotate; --json still carries each flag with its message.
        skipped = [f.rule_id for f in r.flags if f.severity == "info"]
        if skipped:
            print(
                _fill(
                    f"info      {len(skipped)} rules did not run: {', '.join(skipped)}",
                    indent="    ",
                    hang="              ",
                )
            )
        print()


def _feature(col: str, delta: float, value: float, want: float | None) -> str:
    """Example: `irmp_IL210_peak 958 vs 12 A (29 sd)` -- this result, then the query, one unit."""
    if want is None:
        return f"{col} {rank_mod.display(col, value)}"
    return f"{col} {rank_mod.display_pair(col, value, want)} ({delta:.2g} sd)"


# ---------------------------------------------------------------------------------- corpus


def _shot_file(path: str) -> list[int]:
    """Shots from a file: a shot-list YAML (the project's own format) or one integer per line,
    `#` starting a comment. Both because the census is run both ways -- against a curated list
    and against whatever a job script scraped together."""
    p = Path(path)
    if p.suffix in (".yaml", ".yml"):
        return config.load_shot_list(path=p)
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        head = line.split("#", 1)[0].strip()
        if head:
            out.append(int(head))
    return out


def cmd_corpus_select(args) -> int:
    """`corpus select`: the shot list of plan §5.7, written as YAML and as a plain shot-per-line
    file for the labelmaker mask job.

    Two passes over rule (d). Pass one estimates the Ip flat-top for the whole 13,313-shot pool;
    `--verify-flattop` adds pass two, which measures it for the 500 SELECTED shots and refills what
    the measurement drops (`select.verify_flattop`). A selected shot with no feature file comes out
    of pass two as `pending` and is written to `--pending-out` for labelmaker's features stage;
    `--allow-pending` is what lets such a list be written at all, and `--finalize` is the second
    invocation that will not.

    `--from-list <yaml>` is how that second invocation must be run, and the reason is that the
    first one changes its own inputs: it asks for features, the features arrive, and the enlarged
    store moves both rule (d)'s first pass and the `preferred` tie-break -- so a plain re-run
    SELECTS A DIFFERENT LIST (measured: 78 of 500 shots changed after 17 new feature files) and
    the features-then-finalize loop never converges. With `--from-list` the committed 500 are the
    eligibility snapshot: exactly those shots are re-measured, the pool is consulted only to
    replace what the measurement rejects, and the seed comes off the document
    (`select.reverify_flattop`).

    Three things a run may not do, each of them a way an unverified list could take the verified
    one's place: `--finalize` without `--from-list` may not overwrite an existing target (it
    would re-select over it); a document may only carry `finalized: true` -- and the `from_list`
    that makes the claim checkable -- when `--from-list` was given; and a `--from-list --finalize`
    run whose replacements could not fill every hole is refused rather than written short.
    """
    import pandas as pd

    if getattr(args, "finalize", False) and args.allow_pending:
        print(
            "--finalize and --allow-pending contradict each other: finalize IS the run that "
            "refuses a list with unmeasured rows",
            file=sys.stderr,
        )
        return 2
    from_list = Path(args.from_list) if getattr(args, "from_list", None) else None
    listed: list[select_mod.Candidate] | None = None
    source_doc: dict = {}
    out = Path(args.out) if args.out else config.CONFIG_DIR / "shot_lists" / f"{args.name}.yaml"
    if getattr(args, "finalize", False) and from_list is None and out.exists():
        # `--finalize` without `--from-list` re-SELECTS: `eligible()` and `diversify()` run over
        # today's store, which is bigger than the one the existing file was drawn against, so it
        # returns a DIFFERENT list -- and it would overwrite the committed one with it. Refused
        # where the target already exists, which is exactly the case where the damage is silent.
        print(
            f"{out} already exists: `--finalize` without `--from-list` would RE-SELECT over it "
            f"and overwrite the committed list with a different one. Pass "
            f"--from-list {out} to re-verify it, or --out somewhere else to select afresh.",
            file=sys.stderr,
        )
        return 2
    if from_list is not None:
        if not from_list.exists():
            print(f"no shot list at {from_list}", file=sys.stderr)
            return 1
        source_doc = yaml.safe_load(from_list.read_text(encoding="utf-8")) or {}
        if not source_doc.get("shots"):
            print(f"{from_list} carries no shots: to select a new list, omit --from-list",
                  file=sys.stderr)
            return 1
    # The seed is the document's when it has one, so the replacements a re-verification draws are
    # reproducible from the list itself and not from whoever remembers the flag.
    seed = args.seed if args.seed is not None else int(source_doc.get("seed", select_mod.DEFAULT_SEED))
    # `--from-list` is a verification by definition: there is nothing else it could do.
    verify = args.verify_flattop or getattr(args, "finalize", False) or from_list is not None
    census_path = Path(args.census) if args.census else None
    paths = None
    if not (census_path and args.text_dir and args.frame_codes is not None):
        paths = config.load_paths()
        census_path = census_path or paths.db_dir / "corpus_coverage.parquet"
    if not census_path.exists():
        print(f"no census at {census_path}; run `ideate corpus scan` first", file=sys.stderr)
        return 1
    text_dir = Path(args.text_dir) if args.text_dir else paths.per_shot_txt_dir
    # Both of these have exactly one definition, and it is not here: the feature store is
    # `corpus_signals.default_features_dir` (the same directory the corpus build reads) and the
    # frame-code locations are `build.frame_codes_dirs` (the same ones `has_frame_codes` counts),
    # so `corpus select` and `build` cannot disagree about the same shot.
    features_dir = Path(args.features) if args.features else corpus_signals.default_features_dir()
    code_dirs = (
        [Path(args.frame_codes)]
        if args.frame_codes
        else (build_mod.frame_codes_dirs(paths) if paths else [])
    )

    group_spans = select_mod.spans(pd.read_parquet(census_path, columns=_SELECT_COLUMNS))
    corpus_shots = set(pd.read_parquet(census_path, columns=["shot"])["shot"].astype(int))
    bundles = select_mod.read_bundles(text_dir, sorted(corpus_shots))

    preferred = select_mod.preferred_shots(features_dir=features_dir, frame_codes_dirs=code_dirs)
    if source_doc:
        listed = select_mod.candidates_from_rows(source_doc["shots"], preferred=preferred)
    themes = select_mod.lexicon_themes()
    reasons: Counter = Counter()
    kept: list[select_mod.ShotFacts] = []
    for shot in sorted(bundles):
        facts = select_mod.parse_facts(shot, bundles[shot])
        # The measured flat-top only for shots that already have a feature file -- rule (d)'s
        # first source. There is no fdp pass: the pool after (a)-(c) is far past the ~900-shot
        # ceiling the brief set for that path, so every other shot answers (d) by the proxy.
        got = select_mod.measured_flattop(shot, features_dir)
        if got is not None:
            facts = select_mod.with_flattop(facts, got, select_mod.FLATTOP_MEASURED)
        ok, why = select_mod.eligible(
            facts, group_spans.get(shot, {}), min_shot_chars=args.min_shot_chars
        )
        reasons.update(why)
        if ok:
            kept.append(facts)

    logs = Path(args.logs) if args.logs else (paths.logs_jsonl if paths else None)
    if kept and not (logs and logs.exists()):
        print(f"no logbook at {logs}; the <= 5 per mpid cap is skipped", file=sys.stderr)
    mpids = select_mod.mpid_index(logs, [f.shot for f in kept])
    candidates = [
        select_mod.Candidate(
            shot=f.shot,
            run_id=f.run_id,
            mpid=mpids.get(f.shot),
            year=f.year,
            theme=select_mod.assign_theme(f.title, themes, subject=f.mp_subject),
            has_co2=group_spans.get(f.shot, {}).get("co2", 0.0) > 0.0,
            has_bes=group_spans.get(f.shot, {}).get("bes", 0.0) > 0.0,
            has_tangtv=group_spans.get(f.shot, {}).get("tangtv", 0.0) > 0.0,
            ip_sign=f.ip_sign,
            flattop_s=f.flattop_s,
            flattop_source=f.flattop_source,
            preferred=f.shot in preferred,
        )
        for f in kept
    ]
    def measure(shot: int) -> float | None:
        return select_mod.measured_flattop(shot, features_dir)

    replacements: list[dict] = []
    pending: list[int] = []
    if listed is not None:
        # The committed list, re-measured. `diversify` is deliberately NOT run: re-selecting on
        # today's store is what makes the loop non-convergent (see the docstring).
        quotas = select_mod.Quotas(n=len(listed))
        print(f"re-verifying the {len(listed)} shot(s) of {from_list}")
        selected, replacements, _ = select_mod.reverify_flattop(
            listed, candidates, quotas, seed=seed, measure=measure
        )
    else:
        quotas = select_mod.Quotas(n=args.n)
        selected = select_mod.diversify(candidates, quotas, seed=seed)
        if verify:
            selected, replacements = select_mod.verify_flattop(
                selected, candidates, quotas, seed=seed, measure=measure
            )
    if verify:
        # Read off the rows rather than off `reverify_flattop`'s own list, so that a REPLACEMENT
        # that is itself unmeasurable is in the work order too.
        pending = [c.shot for c in selected if c.flattop_source == select_mod.FLATTOP_PENDING]
        pending_out = _pending_path(args)
        if pending and pending_out:
            # Written even when the list itself is refused below: it is the work order that
            # clears the refusal, and the run that produced it is the only thing that knows
            # which 500 shots the features stage has to cover.
            pending_out.parent.mkdir(parents=True, exist_ok=True)
            pending_out.write_text("".join(f"{s}\n" for s in pending), encoding="utf-8")
            print(f"wrote {pending_out} ({len(pending)} shot(s) awaiting features)")
        if pending and not args.allow_pending:
            noun = "listed" if listed is not None else "selected"
            print(
                f"{len(pending)} {noun} shot(s) have no measured flat-top; refusing to write "
                f"{args.name}. Run labelmaker's features stage over "
                f"{pending_out or 'those shots (set $LABELMAKER_ROOT or --pending-out)'} and "
                "repeat with --finalize, or pass --allow-pending to write the list as it stands.",
                file=sys.stderr,
            )
            return 1
    # A finalized list shorter than the number it was drawn for is not a finalized list. It
    # happens when the measurement drops a shot and nothing measured is left to replace it
    # (`_refill` leaves the hole rather than admitting an unverified row), and writing it anyway
    # published a 499-row `recommender_v1` stamped `finalized: true`, exit 0 -- after which every
    # downstream count reads 500 because that is what the name says.
    if listed is not None and getattr(args, "finalize", False) and len(selected) < quotas.n:
        holes = [r["dropped"] for r in replacements if r["replacement"] is None]
        print(
            f"refusing to finalize a short list: {len(selected)} of {quotas.n} shot(s). "
            f"{len(holes)} dropped shot(s) had no measured replacement available "
            f"({_brief(holes)}). Fetch features for more of the eligible pool "
            "(scripts/labelmaker/fetch_features.py) and repeat.",
            file=sys.stderr,
        )
        return 1

    summary = select_mod.summarize(
        n_candidates=len(candidates),
        reasons=reasons,
        selected=selected,
        quotas=quotas,
        candidates=candidates,
        replacements=replacements,
        # A written document can only be finalized if this run was a `--finalize --from-list`
        # run: one that still had a pending row, or came back short, returned 1 above without
        # writing anything. `--finalize` alone re-selects (`diversify` over today's store), so
        # whatever it writes is a FIRST invocation however measured its rows are, and stamping
        # it `finalized` would make it indistinguishable from the list that was re-verified.
        finalized=bool(getattr(args, "finalize", False)) and from_list is not None,
        from_list=from_list,
        store=select_mod.store_fingerprint(features_dir, code_dirs),
    )
    doc = select_mod.document(
        selected,
        summary,
        name=args.name,
        seed=seed,
        n=len(selected),
        hand_review=source_doc.get("hand_review"),
    )

    # `out` was resolved at the top of this function, before anything was read: the `--finalize`
    # refusal there has to know which file this run would overwrite.
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100), encoding="utf-8"
    )
    print(f"wrote {out}")
    if args.txt_out:
        txt_out = Path(args.txt_out)
        txt_out.parent.mkdir(parents=True, exist_ok=True)
        txt_out.write_text("".join(f"{c.shot}\n" for c in selected), encoding="utf-8")
        print(f"wrote {txt_out}")
    print()
    print(select_mod.format_summary(summary))
    return 0


# The census columns the selection reads. Named so that a census that stopped writing one of them
# fails here with a KeyError on this list rather than with a silently empty span mapping.
_SELECT_COLUMNS = ["shot", "group", "present", "t0_s", "t1_s"]


def _pending_path(args) -> Path | None:
    """Where the selected shots that still need a labelmaker features run are listed.

    `$LABELMAKER_ROOT/<name>_pending_features.txt` by default: the file is a work order for
    labelmaker's features stage, so it belongs next to the feature store that stage writes into
    and not in this repo. None when neither `--pending-out` nor `$LABELMAKER_ROOT` says where --
    in which case the count is still printed and the list is still refused, because not knowing
    where to file the work order is not a reason to publish an unverified list.
    """
    if args.pending_out:
        return Path(args.pending_out)
    root = os.environ.get("LABELMAKER_ROOT")
    return Path(root) / f"{args.name}_pending_features.txt" if root else None


def cmd_corpus(args) -> int:
    """`corpus scan` counts what every corpus file carries; `corpus summary` reads that count
    back as the availability table; `corpus select` picks the development shot list."""
    if args.what == "select":
        return cmd_corpus_select(args)
    if args.what == "summary":
        import pandas as pd

        df = pd.read_parquet(args.parquet)
        print(census.format_summary(census.summary(df)))
        return 0

    paths = config.load_paths() if not (args.corpus and args.out) else None
    corpus_dir = Path(args.corpus) if args.corpus else Path(paths.foundation_model_processed_dir)
    out = Path(args.out) if args.out else paths.db_dir / "corpus_coverage.parquet"
    if not corpus_dir.is_dir():
        print(f"no corpus directory at {corpus_dir}", file=sys.stderr)
        return 1
    shots = _shot_file(args.shots) if args.shots else None
    started = time.perf_counter()
    df = census.scan(
        corpus_dir,
        workers=args.workers,
        limit=args.limit,
        shots=shots,
        progress=sys.stdout.isatty(),
    )
    elapsed = time.perf_counter() - started
    n_files = int(df["shot"].nunique())
    n_openable = int(df.loc[df["openable"], "shot"].nunique())
    sidecar = census.write(df, out, corpus_dir=corpus_dir, workers=args.workers, elapsed_s=elapsed)
    print(
        f"scanned {n_files:,} files ({n_openable:,} openable, {n_files - n_openable:,} not) "
        f"in {elapsed:.1f} s -> {len(df):,} rows"
    )
    print(f"wrote {out} and {sidecar.name}")
    print()
    print(census.format_summary(census.summary(df)))
    return 0


# ------------------------------------------------------------------------------------ logs


def cmd_logs(args) -> int:
    """`logs missing` says what to run at GA; `logs import` brings a synced runs/ tree back.

    Neither touches the network. The text corpus is d3dlogfetching's output and that tool only
    runs inside the GA fusion network, so the only honest thing this side can do is name the
    commands and be byte-compatible with what they produce (see shotdb/logs.py).
    """
    paths = None
    if args.what == "missing":
        # Every path this branch needs, or none: falling back per path would leave `--index`
        # silently unset when the caller gave the other two, and every missing shot would then be
        # reported as unplaceable rather than as belonging to a run the corpus already knows.
        if not (args.text_dir and args.index and (args.list or args.census)):
            paths = config.load_paths()
        text_dir = Path(args.text_dir) if args.text_dir else paths.per_shot_txt_dir
        index = Path(args.index) if args.index else (paths.shot_index_json if paths else None)
        if args.all_corpus:
            import pandas as pd

            parquet = Path(args.census) if args.census else paths.db_dir / "corpus_coverage.parquet"
            if not parquet.exists():
                print(f"no census at {parquet}; run `ideate corpus scan` first", file=sys.stderr)
                return 1
            shots = sorted(set(pd.read_parquet(parquet, columns=["shot"])["shot"].astype(int)))
        else:
            listed = args.list or "recommender_v1"
            path = Path(listed)
            if not path.exists():
                path = config.CONFIG_DIR / "shot_lists" / f"{listed}.yaml"
            if not path.exists():
                print(f"no shot list at {listed}", file=sys.stderr)
                return 1
            shots = _shot_file(str(path))
        print(
            logs_mod.format_missing(
                logs_mod.missing(shots, text_dir), n_total=len(shots), index_json=index
            )
        )
        return 0

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        print(f"no runs directory at {runs_dir}", file=sys.stderr)
        return 1
    if not (args.raw_dir and args.text_dir):
        paths = config.load_paths()
    raw_dir = Path(args.raw_dir) if args.raw_dir else paths.shotsummary_raw_dir
    text_dir = Path(args.text_dir) if args.text_dir else paths.per_shot_txt_dir
    report = logs_mod.import_runs(
        runs_dir, raw_dir=raw_dir, txt_dir=text_dir, dry_run=args.dry_run, force=args.force
    )
    verb = "would write" if args.dry_run else "wrote"
    print(f"{len(report.runs)} run(s): {', '.join(report.runs) or 'none'}")
    if report.skipped:
        print(f"skipped (no {logs_mod.RUN_MARKER}): {', '.join(report.skipped)}")
    print(f"{'would copy' if args.dry_run else 'copied'} {report.files_copied} run file(s) -> {raw_dir}")
    print(f"{verb} {report.bundles_written} bundle(s) -> {text_dir}")
    if report.bundles_kept:
        print(f"kept {report.bundles_kept} existing bundle(s) (--force to rewrite)")
    for name in report.would_write:
        print(f"  {verb} {name}")
    return 0


# ------------------------------------------------------------------------------------ labels


def cmd_labels(args) -> int:
    """`labels join`: labelmaker's per-time labels and events become the DB's three label tables.

    Succeeds with `n_shots_with_events = 0` when `$LABELMAKER_ROOT/events/` does not exist -- it
    does not until the mask job has run, and the labels are useful before then.
    """
    from labelmaker.config import Paths as LabelmakerPaths

    from .labels import join as join_mod

    paths = config.load_paths()
    shots = _shots(args, "recommender_v1")
    if not shots:
        print("no shots selected", file=sys.stderr)
        return 1
    root = Path(args.labelmaker_root) if args.labelmaker_root else LabelmakerPaths.from_env().root
    if not (root / "labels").is_dir():
        print(f"no labels directory at {root / 'labels'}", file=sys.stderr)
        return 1
    db_dir = Path(args.db) if args.db else paths.db_dir
    text_root = None if args.no_text else Path(args.text_root or paths.text_root)
    if text_root is not None and not text_root.is_dir():
        print(f"no text corpus at {text_root}; joining without claims", file=sys.stderr)
        text_root = None

    started = time.perf_counter()
    result = join_mod.join(
        shots, labelmaker_root=root, text_root=text_root, lexicon_path=args.lexicon
    )
    block = join_mod.write_tables(
        db_dir, result.labels_wide, result.events, result.claims, result.manifest
    )
    m = result.manifest
    print(
        f"joined {m['n_shots']:,} shots in {time.perf_counter() - started:.1f} s "
        f"({m['n_shots_with_labels']:,} labelled, {m['n_shots_with_events']:,} with events)"
    )
    n_models = len(set(result.labels_wide["slug"]))
    print(f"labels_wide {m['n_labels_wide_rows']:,} rows over {n_models} model(s)")
    for slug, n in sorted(Counter(result.labels_wide["slug"]).items()):
        print(f"  {slug:<45} {n:>7,}")
    if not result.labels_wide.empty:
        # Whose threshold each row carries: a card's operating point, ideate's own alarm level, or
        # none at all. An aggregate count cannot say it per row, but it can say how far it reaches.
        sources = Counter(result.labels_wide["thr_source"])
        print("  thr_source " + ", ".join(
            f"{src or 'none'} {n:,}" for src, n in sorted(sources.items())
        ))
    print(f"events      {m['n_events']:,} rows, of which {m['n_forecast_events']:,} forecasts")
    for (phen, horizon), n in sorted(
        Counter(
            zip(
                result.events.loc[result.events["evidence_kind"] == "forecast", "phenomenon"],
                result.events.loc[result.events["evidence_kind"] == "forecast", "horizon_s"],
                strict=True,
            )
        ).items()
    ):
        print(f"  forecast {phen:<12} horizon {horizon:>6.3f} s  {n:>7,}")
    print(f"text_claims {m['n_text_claims']:,} rows from {m['lexicon'] or 'no lexicon'}")
    if not result.claims.empty:
        for col in ("polarity", "temporality"):
            print("  " + ", ".join(f"{k} {v:,}" for k, v in
                                   sorted(Counter(result.claims[col]).items())))
    if m["n_shots_missing_labels"]:
        print(f"no labels file: {_brief(m['labels_missing'])}")
    print(f"wrote {db_dir}/{{labels_wide,events,text_claims}}.parquet and manifest.json")
    print(f"manifest labels block: {json.dumps(block, default=str)}")
    return 0


def cmd_query(args) -> int:
    """Multi-channel retrieval over the built database. See ideate.retrieval.search."""
    paths = config.load_paths()
    db = _open_db(paths)
    if db is None:
        return 1
    state = _query_state(args)
    if state.ref_shot is not None and f"{state.ref_shot}:{state.segment}" not in db.segments.index:
        if _record(db, state.ref_shot) is None:
            return 1
        print(f"shot {state.ref_shot} has no {state.segment} segment", file=sys.stderr)
        return 1
    try:
        report = rank_mod.search_report(state, db)
    except KeyError as e:
        print(
            f"{e.args[0]}. Columns are the ones `ideate show SHOT --full` prints.", file=sys.stderr
        )
        return 2
    # One pass: the per-channel counts on the "channels" line come from the same rankings the
    # results were fused from. Running every channel once more for the counts doubled the cost of
    # a --text query, which embeds its text with MiniLM on each pass (~1.9 s each, measured).
    found = rank_mod.search(state, db)
    fired = {name: len(ranking) for name, ranking in found.rankings.items()}
    if args.json:
        doc = {
            "proposal_flags": [f.model_dump(mode="json") for f in found.proposal_flags],
            "results": [r.model_dump(mode="json") for r in found.items],
        }
        print(json.dumps(doc, indent=1, default=str))
        return 0 if any(fired.values()) else 2
    _query_header(state, report, fired)
    _print_proposal(state, found.proposal_flags)  # answered even when nothing resembles it
    if not any(fired.values()):
        print(
            "no channel had anything to search on -- give --ref SHOT, --text, --where or "
            "--actuator (`ideate query --help`).",
            file=sys.stderr,
        )
        return 2
    results = found.items
    _print_results(results, db, state)
    if not results:
        print("nothing passed the filters.", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------------------- main


def _add_selection(p: argparse.ArgumentParser) -> None:
    p.add_argument("--list", help="configs/ideate/shot_lists/<name>.yaml")
    p.add_argument("--list-file", help="a shot-list YAML at an explicit path")
    p.add_argument("--shots", type=int, nargs="*", help="explicit shot numbers")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ideate",
        description="DIII-D shot database: build from local signals, retrieve and inspect shots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build", help="full rebuild of the database (atomic: writes db.tmp, swaps)")
    _add_selection(p)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--reader",
        choices=("legacy", "corpus"),
        default="legacy",
        help="raw layer to build from: the d3d_fusion_data layout (default) or the FAITH corpus "
        "plus $LABELMAKER_ROOT/features",
    )
    p.add_argument(
        "--no-encode",
        action="store_true",
        help="skip the optional IGNITE waveform channel (the scalar and text embeddings are "
        "part of the database and are built either way)",
    )
    p.add_argument(
        "--reencode",
        action="store_true",
        help="re-encode every shot instead of carrying unchanged shots' IGNITE rows over",
    )
    p.add_argument("--all", action="store_true", help="build shots with no Ip on disk too")
    p.add_argument(
        "--limit", type=int, help="build only the first N shots of the selection (pilot runs)"
    )
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("add", help="incremental upsert of one or more shots")
    p.add_argument("shots", type=int, nargs="+")
    p.add_argument("--workers", type=int, default=1)
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("model", help="IGNITE bundle status; --download copies it from the Hub")
    p.add_argument("--download", action="store_true", help="snapshot the pinned revision once")
    p.add_argument("--full", action="store_true", help="include the 3.5 GB dynamics checkpoint")
    p.set_defaults(func=cmd_model)

    p = sub.add_parser("encode", help="IGNITE frame-code caches for a shot list (needs a GPU)")
    _add_selection(p)
    p.add_argument("--out", help="output directory (default: <data_root>/frame_codes)")
    p.add_argument("--device", help="cuda | cpu (default: cuda when available)")
    p.add_argument("--workers", type=int, help="CPU dataloader workers per codec (default: 8)")
    p.add_argument("--limit", type=int, help="encode only the first N shots (pilot runs)")
    p.add_argument("--skip-existing", action="store_true", help="leave caches already written")
    p.add_argument("--no-video", action="store_true", help="skip the two tangtv modalities")
    p.add_argument("--chunk", type=int, default=0, help="which contiguous slice this task takes")
    p.add_argument("--n-chunks", type=int, default=1, help="how many tasks share the list")
    p.set_defaults(func=cmd_encode)

    p = sub.add_parser("show", help="print one shot's record")
    p.add_argument("shot", type=int)
    p.add_argument("--segment", default="flat_top", help="segment name, or 'all'")
    p.add_argument("--full", action="store_true", help="every scalar, by column name")
    p.add_argument("--json", action="store_true", help="the whole ShotRecord as JSON")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("export", help="ShotSummary rows as JSON or Parquet")
    p.add_argument("shots", type=int, nargs="+")
    p.add_argument("--format", choices=["json", "parquet"], default="json")
    p.add_argument("--out", help="output path (JSON goes to stdout when omitted)")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("coverage", help="present/unavailable/pending per registry field")
    p.set_defaults(func=cmd_coverage)

    p = sub.add_parser("corpus", help="the FAITH corpus: what each shot file actually carries")
    what = p.add_subparsers(dest="what", required=True)
    s = what.add_parser("scan", help="census every corpus file (header-only, parallel)")
    s.add_argument("--corpus", help="corpus directory (default: paths.yaml's)")
    s.add_argument("--out", help="parquet to write (default: <db_dir>/corpus_coverage.parquet)")
    s.add_argument("--workers", type=int, default=os.cpu_count())
    s.add_argument("--limit", type=int, help="only the first N files, in shot order")
    s.add_argument("--shots", metavar="FILE", help="a shot-list YAML, or one shot per line")
    s = what.add_parser("summary", help="the availability table from a written census")
    s.add_argument("parquet")
    s = what.add_parser("select", help="the development shot list (plan §5.7)")
    s.add_argument("--n", type=int, default=500, help="how many shots to select")
    s.add_argument("--name", default="recommender_v1", help="the list's name in the YAML")
    s.add_argument("--census", help="corpus_coverage.parquet (default: <db_dir>/...)")
    s.add_argument("--text-dir", help="per_shot_txt directory (default: paths.yaml's)")
    s.add_argument("--out", help="shot-list YAML (default: configs/ideate/shot_lists/<name>.yaml)")
    s.add_argument("--txt-out", help="also write one shot per line here (the mask job reads it)")
    s.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"the list is a function of it (default {select_mod.DEFAULT_SEED}, or the seed "
        "--from-list's document recorded)",
    )
    s.add_argument("--features", help="labelmaker feature store (default: $LABELMAKER_ROOT/features)")
    s.add_argument("--frame-codes", help="IGNITE frame_codes directory")
    s.add_argument("--logs", help="sql/logs.jsonl, the source of mpid (default: paths.yaml's)")
    s.add_argument(
        "--min-shot-chars",
        type=int,
        default=select_mod.MIN_SHOT_CHARS,
        help=f"rule (b) shot-text floor (default {select_mod.MIN_SHOT_CHARS}; see select.py)",
    )
    s.add_argument(
        "--verify-flattop",
        action="store_true",
        help="pass two of rule (d): measure the Ip flat-top of every SELECTED shot and refill",
    )
    s.add_argument(
        "--allow-pending",
        action="store_true",
        help="write the list even though some rows have no measured flat-top yet",
    )
    s.add_argument(
        "--finalize",
        action="store_true",
        help="verify and accept nothing less: --verify-flattop with --allow-pending refused",
    )
    s.add_argument(
        "--from-list",
        metavar="YAML",
        help="re-verify the shots of an existing list instead of selecting a new one: the "
        "second invocation of the two-pass rule (d). --n and --seed come off that document",
    )
    s.add_argument(
        "--pending-out",
        help="where to list the shots awaiting features (default: "
        "$LABELMAKER_ROOT/<name>_pending_features.txt)",
    )
    p.set_defaults(func=cmd_corpus)

    p = sub.add_parser("logs", help="the d3dlogfetching shot-log contract")
    what = p.add_subparsers(dest="what", required=True)
    s = what.add_parser("missing", help="shots with no shot_<N>.txt, and how to fetch them")
    s.add_argument("--list", help="shot-list name or file (default: recommender_v1)")
    s.add_argument("--all-corpus", action="store_true", help="every shot in the census instead")
    s.add_argument("--census", help="corpus_coverage.parquet (default: <db_dir>/...)")
    s.add_argument("--text-dir", help="per_shot_txt directory (default: paths.yaml's)")
    s.add_argument("--index", help="sql/index.json, shot -> run_id (default: paths.yaml's)")
    s = what.add_parser("import", help="a synced runs/ tree -> raw/<run_id>/ + per-shot bundles")
    s.add_argument("runs_dir")
    s.add_argument("--raw-dir", help="shotsummary/raw (default: paths.yaml's)")
    s.add_argument("--text-dir", help="per_shot_txt directory (default: paths.yaml's)")
    s.add_argument("--dry-run", action="store_true", help="print what would be written")
    s.add_argument("--force", action="store_true", help="rewrite an existing per-shot bundle")
    p.set_defaults(func=cmd_logs)

    p = sub.add_parser("labels", help="labelmaker's labels and events -> the ideate DB")
    what = p.add_subparsers(dest="what", required=True)
    s = what.add_parser("join", help="labels_wide.parquet, events.parquet, text_claims.parquet")
    s.add_argument("--list", help="configs/ideate/shot_lists/<name>.yaml (default: recommender_v1)")
    s.add_argument("--list-file", help="a shot-list YAML at an explicit path")
    s.add_argument("--shots", type=int, nargs="*", help="explicit shot numbers")
    s.add_argument("--labelmaker-root", help="$LABELMAKER_ROOT (default: labelmaker's own)")
    s.add_argument("--db", help="database directory (default: paths.yaml's db_dir)")
    s.add_argument("--text-root", help="the text corpus root (default: paths.yaml's)")
    s.add_argument("--lexicon", help="phenomenon aliases (default: labelmaker's lexicons.yaml)")
    s.add_argument("--no-text", action="store_true", help="skip the text claims entirely")
    p.set_defaults(func=cmd_labels)

    p = sub.add_parser("query", help="find similar shots")
    p.add_argument("--text", help="free text, e.g. 'wide pedestal QH at low torque'")
    p.add_argument("--negative", action="append", default=[], help="text to move away from")
    p.add_argument("--ref-shot", "--ref", type=int, help="use this shot as the reference")
    p.add_argument("--ref-window", type=float, nargs=2, metavar=("T0_MS", "T1_MS"))
    # The schema's own literal, so a typo (`--segment flattop`) is an argparse message here rather
    # than a pydantic traceback after the database has loaded. `show --segment` also takes 'all'.
    p.add_argument("--segment", default="flat_top", choices=get_args(SegName))
    p.add_argument(
        "--constraint",
        "--where",
        action="append",
        default=[],
        type=_constraint,
        metavar="COLUMN=LO:HI",
        help="hard range filter; either bound may be empty; COLUMN:LO:HI also accepted",
    )
    p.add_argument(
        "--actuator",
        action="append",
        default=[],
        type=_actuator,
        metavar="NAME=VALUE",
        help="target actuator setting, e.g. nbi.total=5e6",
    )
    p.add_argument("--require", action="append", default=[], metavar="LABEL")
    p.add_argument("--avoid", action="append", default=[], metavar="LABEL")
    p.add_argument("--exclude-shot", action="append", default=[], type=int)
    p.add_argument("--exclude-run", action="append", default=[])
    p.add_argument("--prefer-outcome", choices=["any", "success"], default="any")
    p.add_argument("-n", "--n", type=int, default=10, help="how many results")
    p.add_argument("--json", action="store_true", help="results as JSON, explanations included")
    p.set_defaults(func=cmd_query)

    p = sub.add_parser(
        "llm", help="is a language model reachable? prints the endpoint or how to start one"
    )
    p.set_defaults(func=cmd_llm)

    p = sub.add_parser(
        "blurb", help="write the model's per-shot blurbs into shots.parquet (needs a running model)"
    )
    p.add_argument(
        "--all", action="store_true", help="rewrite every blurb, not only the template ones"
    )
    p.set_defaults(func=cmd_blurb)

    p = sub.add_parser("actuation", help="inspect saved actuator waveform sets")
    what = p.add_subparsers(dest="what", required=True)
    what.add_parser("list", help="every saved set, newest first")
    s = what.add_parser("show", help="one saved set")
    s.add_argument("id")
    s.add_argument("--json", action="store_true", help="the stored JSON verbatim")
    p.set_defaults(func=cmd_actuation)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError) as e:
        print(f"ideate: {e}", file=sys.stderr)
        return 1
    except SystemExit as e:
        # No `cmd_*` raises SystemExit itself, but a library one calls can, and with either shape:
        # an exit status (`SystemExit(3)`) or a message (`raise SystemExit("refusing to write
        # into ...")`). `int()` on a message raised ValueError out of main and the message was
        # never shown; the interpreter's own convention is: print it, exit 1.
        if e.code is None or isinstance(e.code, int):
            return int(e.code or 0)
        print(e.code, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
