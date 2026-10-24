#!/usr/bin/env python
"""Fetch the raw DIII-D signals `prepare_data.py` reads, through fdp, into `<shot>.h5`.

This is the fdp/toksearch counterpart of `scripts/data_fetching_omega/`, for a cluster
that reaches DIII-D only over GA's Pelican/OSDF route (no MDSplus login on atlas, no
route to chiron). It fetches exactly the nodes named in the data-generation config,
`src/tokamak_foundation_model/data/config/modalities/modalities.yaml`, and writes them in
the mdsh5 layout that `prepare_data.py`'s `SignalLoader` reads:

    <out_dir>/<shot>.h5
      <shot>/<tree>/<node>/data   float32, channels first for 2-D leaves (Thomson)
      <shot>/<tree>/<node>/dim0   time in ms, float64

`<tree>` and `<node>` are spelled exactly as in modalities.yaml, so the file needs no
renaming before processing. A signal that returns no data is simply absent.

TREES. A `\\D3D::TOP.<SUB>.<rest>` node is fetched as `\\<SUB>::TOP.<rest>` from the
subsystem's own tree: the cross-subsystem `d3d` tree resolves the same leaves 10-20x
slower (measured in configs/shot_design/ignite_modalities.yaml, and here: ECE TECEF
36 s per channel through `d3d`). If the subsystem spelling fails for any reason but
"no data", the original node is retried through `d3d`. PTDATA names go through
`PtDataSignal`. A group's `fallback` (CO2's PTDATA point names) is fetched only for a
channel whose primary node came back empty.

WHERE IT RUNS. The login node, under the `fdp run` wrapper (compute nodes have no
route out, and without the wrapper PTDATA fails with `getservbyname ... PTSERVER`):

    pixi run --frozen -e fdp fdp -D d3d run python scripts/data_fetching_fdp/fetch_raw_fdp.py \\
        --shot-file /scratch/gpfs/EKOLEMEN/D3D/shot_lists/recommender_v1.txt \\
        --out-dir /scratch/gpfs/EKOLEMEN/D3D/d3d_time_series_data --workers 24

Each worker handles one shot at a time and writes `<shot>.h5.part`, renamed to
`<shot>.h5` when the shot is done, next to a `<shot>.fetch.json` record of every node's
status, shape and fetch time. An existing `<shot>.h5` is skipped, so rerunning resumes.
The pool is forked before toksearch is imported: its ptserver reader is not fork-safe.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO / "src/tokamak_foundation_model/data/config/modalities/modalities.yaml"
)
D3D_NODE = re.compile(r"^\\D3D::TOP\.([A-Z0-9_]+)\.(.+)$", re.IGNORECASE)
SUBTREE_NODE = re.compile(r"^\\([A-Z0-9_]+)::", re.IGNORECASE)


def read_shots(path: Path) -> list[int]:
    """Shot numbers from a text file, one per line; `#` starts a comment."""
    shots = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            shots.append(int(line))
    return shots


def signal_plan(config: dict, groups: list[str] | None) -> list[dict]:
    """One entry per modality group: its tree, nodes and optional fallback."""
    plan = []
    for name, cfg in config["signals"].items():
        if groups and name not in groups:
            continue
        fb = cfg.get("fallback")
        plan.append({
            "group": name,
            "tree": cfg["tree"],
            "nodes": list(cfg["input_key"]),
            "fallback": (
                {"tree": fb["tree"], "nodes": list(fb["input_key"])} if fb else None
            ),
        })
    return plan


def subsystem_spelling(node: str) -> tuple[str, str]:
    """The (expression, tree) the fast path fetches a modalities.yaml node from."""
    m = D3D_NODE.match(node)
    if m:
        return f"\\{m.group(1)}::TOP.{m.group(2)}", m.group(1).lower()
    m = SUBTREE_NODE.match(node)
    if m:
        return node, m.group(1).lower()
    return node, "d3d"


def is_nodata(exc: Exception) -> bool:
    """True for MDSplus/PTDATA's "this shot has nothing here" answer."""
    text = f"{type(exc).__name__}: {exc}"
    return "NODATA" in text or "No data" in text


def is_permanent(exc: Exception) -> bool:
    """A miss that the same fetch would repeat: no data, no such node, an aborting
    expression, or a PTDATA point the Pelican index does not carry.

    TdiABORT is deterministic (the same SXR leaves abort on 185955 and 198807, through
    both trees) and costs up to 40 s per try. A point missing from the index makes the
    ptdata engine fall back to a ptserver this host has no route to, which surfaces as
    `PtDataError ... getservbyname failed for task 'PTSERVER'`; that is the same miss TDI
    reports as TdiABORT, and a missing point does not affect later fetches.
    """
    text = f"{type(exc).__name__}: {exc}"
    return isinstance(exc, BadRecord) or is_nodata(exc) or any(
        s in text for s in ("TreeNNF", "TdiABORT", "getservbyname failed for task 'PTSERVER'")
    )


class BadRecord(ValueError):
    """A stored signal whose data and time base disagree in length."""


class ShotTrees:
    r"""Per-shot MDSplus trees, opened once, and the PTDATA substitution evaluator.

    Many MDSplus leaves are TDI expressions over PTDATA: bare wrappers
    (`ptdata("sx165r1s9", $SHOT)`), calibrations with an inline call (ECE TECEF:
    `_ece = PTDATA2(\ECE::ECPTNAMF[i], $SHOT, 4), Build_Signal(_ece * (...))`), or
    expressions over other nodes that wrap PTDATA (gas FLOW over `\GASFLOW.x:RAW`, NB
    VOLTAGE_CAL over `\NB15L:PTDATA_RAW`). Letting TDI evaluate the PTDATA call is the
    slow part over Pelican (ECE 17-40 s against 3.9 s per channel through
    `PtDataSignal`; SXR 18 s against 0.6 s), and it does not work at all on staged
    local files (libd3's ptdata_ path aborts; the Python ptdata engine reads them).

    So `evaluate` fetches every PTDATA call in the record, and every referenced node
    whose own record (recursively) leads to PTDATA, through `PtDataSignal`, passes them
    to TDI as `$1..$n`, and lets TDI evaluate the rest of the expression unchanged.
    Checked against the plain TDI value on 185955 (ECE channel 5, SX165R1S09): equal
    data and time base.
    """

    def __init__(self, shot: int):
        self.shot = shot
        self.trees: dict[str, object] = {}
        self.leads: dict[tuple[str, str], bool] = {}

    def tree(self, sub: str):
        if sub not in self.trees:
            import MDSplus

            self.trees[sub] = MDSplus.Tree(sub, self.shot)
        return self.trees[sub]

    def record(self, expr: str, sub: str) -> str | None:
        """The node's stored record as TDI text, or None if it cannot be read."""
        try:
            return str(self.tree(sub).getNode(expr).getRecord())
        except Exception:  # noqa: BLE001 - absent, empty or unreadable: fetch normally
            return None

    def leads_to_ptdata(self, text: str, sub: str, depth: int = 0) -> bool:
        if PTDATA_CALL.search(text) or PTHEAD_CALL.search(text):
            return True
        if depth > 4:
            return False
        for ref in node_refs(text):
            key = (sub, ref)
            if key not in self.leads:
                self.leads[key] = False  # cycle guard
                rec = self.record(ref, sub)
                self.leads[key] = bool(rec) and self.leads_to_ptdata(rec, sub, depth + 1)
            if self.leads[key]:
                return True
        return False

    def evaluate(self, text: str, sub: str, depth: int = 0):
        """TDI-evaluate `text` with its PTDATA fetched through PtDataSignal."""
        import MDSplus

        tree = self.tree(sub)
        args: list = []

        def arg(value) -> str:
            args.append(value)
            return f"${len(args)}"

        def pointname(text: str) -> str:
            if text.startswith('"'):
                return text[1:-1].strip()
            name = tree.tdiExecute(text).data()
            return (name.decode() if isinstance(name, bytes) else str(name)).strip()

        def ptdata_call(m: re.Match) -> str:
            # TDI's ptdata2 returns float32 unless asked for double; match it, so a
            # threshold such as NB BEAMSTAT's `PTDATA_RAW - 2.5 > 0` cuts identically.
            try:
                rec = fetch_ptdata(pointname(m.group(1)), self.shot, int(m.group(2) or 1))
            except Exception as exc:  # noqa: BLE001
                if is_permanent(exc):
                    return "ABORT()"  # what TDI's PTDATA does; IF_ERROR(...) still applies
                raise
            data = np.asarray(rec["data"])
            if np.issubdtype(data.dtype, np.floating):
                data = data.astype(np.float32)
            return arg(MDSplus.Signal(
                MDSplus.makeArray(data), None,
                MDSplus.makeArray(np.asarray(rec["dim0"], dtype=np.float64)),
            ))

        def pthead_call(m: re.Match) -> str:
            # PTHEAD_REAL32 returns real32[2:] (tdi/d3d/ptdata/pthead_real32.fun) and
            # PTHEAD_ASCII the packed header characters; read through the Python engine.
            import warnings

            import ptdata

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    head = ptdata.PtDataHeader(pointname(m.group(2)), self.shot)
            except Exception:  # noqa: BLE001 - no header: TDI aborts too
                return "ABORT()"
            if m.group(1).upper() == "REAL32":
                return arg(MDSplus.makeArray(np.asarray(head.real32[2:], dtype=np.float32)))
            raw = np.asarray(head.ascii[2:], dtype="<i4").tobytes().replace(b"\0", b"")
            text = raw.decode("ascii", "replace")
            return arg(MDSplus.String(text + " " * (-len(text) % 4)))

        def node_ref(m: re.Match) -> str:
            ref = m.group(0)
            rec = self.record(ref, sub)
            if rec and depth < 4 and self.leads_to_ptdata(rec, sub):
                return arg(self.evaluate(rec, sub, depth + 1))
            return ref

        # PTDATA calls first (their pointname is a quoted string), then node references
        # outside string literals.
        text = PTHEAD_CALL.sub(pthead_call, PTDATA_CALL.sub(ptdata_call, text))
        parts = re.split(r'("(?:[^"\\]|\\.)*")', text)
        text = "".join(p if p.startswith('"') else NODE_REF.sub(node_ref, p) for p in parts)
        return tree.tdiExecute(text, *args).evaluate()


PTDATA_CALL = re.compile(
    r'PTDATA2?\s*\(\s*("[^"]*"|\\[A-Za-z0-9_.:$]+(?:\[\d+\])?)\s*,\s*\$?SHOT\s*'
    r'(?:,\s*(\d+)\s*)?\)',
    re.IGNORECASE,
)
PTHEAD_CALL = re.compile(
    r'PTHEAD_(REAL32|ASCII)\s*\(\s*("[^"]*"|\\[A-Za-z0-9_.:$]+(?:\[\d+\])?)\s*,\s*\$?SHOT\s*\)',
    re.IGNORECASE,
)
NODE_REF = re.compile(r"\\[A-Za-z_][A-Za-z0-9_$]*(?:::[A-Za-z0-9_.:$]+|[.:][A-Za-z0-9_.:$]+)")


def node_refs(text: str) -> list[str]:
    """Backslash node references outside string literals, in order, unique."""
    parts = re.split(r'("(?:[^"\\]|\\.)*")', text)
    seen = []
    for p in parts:
        if not p.startswith('"'):
            for m in NODE_REF.finditer(p):
                if m.group(0) not in seen:
                    seen.append(m.group(0))
    return seen


def fetch_ptdata(name: str, shot: int, ical: int = 1) -> dict:
    from toksearch_d3d import PtDataSignal

    rec = PtDataSignal(name, ical=ical).fetch(shot)
    return {"data": rec["data"], "dim0": rec["times"]}


def fetch_one(tree: str, node: str, shot: int, trees: ShotTrees) -> dict:
    """One node, as {"data", "dim0"} arrays; raises on a fetch failure."""
    if tree.upper() == "PTDATA":
        return fetch_ptdata(node, shot)

    expr, sub = subsystem_spelling(node)
    record = trees.record(expr, sub)
    if record is not None and trees.leads_to_ptdata(record, sub):
        value = trees.evaluate(record, sub)
        data = np.asarray(value.data())
        try:
            dim0 = np.asarray(value.dim_of().data())
        except Exception:  # noqa: BLE001 - not a Signal: no time base
            dim0 = np.array([])
        return {"data": data, "dim0": dim0}

    from toksearch import MdsSignal

    try:
        rec = MdsSignal(expr, sub).fetch(shot)
    except Exception as exc:  # noqa: BLE001 - backend errors are arbitrary
        # No data, or an aborting expression, is the same answer through `d3d`;
        # a node-not-found may be a spelling the subsystem tree does not carry.
        if sub == "d3d" or (is_permanent(exc) and "TreeNNF" not in type(exc).__name__):
            raise
        rec = MdsSignal(node, "d3d").fetch(shot)
    return {"data": rec["data"], "dim0": rec.get("times", np.array([]))}


def normalise(rec: dict) -> dict | None:
    """float32 data, float64 ms time, channels first for a 2-D leaf; None if empty."""
    data = np.asarray(rec["data"])
    times = np.asarray(rec["dim0"], dtype=np.float64)
    if data.size == 0:
        return None
    if np.issubdtype(data.dtype, np.floating) and data.dtype != np.float32:
        with np.errstate(invalid="ignore"):
            data = data.astype(np.float32)
    # Thomson leaves come back (channels, time) or (time, channels); settle it by length.
    if data.ndim == 2 and times.size and data.shape[0] == times.size != data.shape[1]:
        data = data.T
    # As stored, not a fetch error: ECH system 3's RAY_AZI/RAY_POLAR on 195625, 195646 and
    # 195648 hold 1011 samples over a 205000-point time base. prepare_data cannot use it.
    if data.ndim == 1 and times.size and data.size != times.size:
        raise BadRecord(f"{data.size} samples but {times.size} times")
    return {"data": data, "dim0": times}


def fetch_with_retry(tree: str, node: str, shot: int, retries: int, trees: ShotTrees) -> tuple:
    """(record or None, status, error text) with `retries` extra tries on failure."""
    for attempt in range(retries + 1):
        try:
            rec = normalise(fetch_one(tree, node, shot, trees))
            return rec, ("ok" if rec is not None else "empty"), None
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {str(exc)[:300]}"
            if is_permanent(exc):
                return None, ("nodata" if is_nodata(exc) else "missing"), err
            if attempt == retries:
                return None, "error", err
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def fetch_shot(job: tuple) -> dict:
    """`_fetch_shot`, with a crash reported as a summary instead of ending the pool."""
    try:
        return _fetch_shot(*job)
    except Exception as exc:  # noqa: BLE001
        return {"shot": job[0], "seconds": 0, "bytes": 0,
                "counts": {"crash": f"{type(exc).__name__}: {exc}"}}


def use_local_stage(shot_dir: Path) -> None:
    """Point MDSplus and the ptdata engine at a staged shot (stage_shots_fdp.py).

    Called in the worker before toksearch or MDSplus is imported: the ptdata reader
    captures its configuration from the environment once per process. Without the
    Pelican index variables the engine reads `<SYS_D3>/<shot>.<EXT>` from disk, and
    MDSplus finds `<tree>_<shot>.*` through `default_tree_path`; nothing needs the
    network, so this runs on a compute node.
    """
    # MDSplus tree paths are lower-case `<tree>_path`; LD_LIBRARY_PATH and MDS_PATH are not.
    for key in [k for k in os.environ if k.endswith("_path")]:
        os.environ.pop(key)
    for key in ("PTDATA_JSON_INDEX_DIR", "PTDATA_JSON_INDEX_PATTERN", "PTDATA_PLUGIN_LIB"):
        os.environ.pop(key, None)
    prefix = Path(sys.prefix)
    os.environ.update({
        "SYS_D3": str(shot_dir / "ptdata"),
        "SYS_D3_DELIM": ";",
        "PTDATA_LOC": "1",
        "PTDATA_LIBRARY": str(prefix / "lib/libd3.so"),
        "MDS_PATH": str(prefix / "tdi"),
        "D3DATA": "yes",
        "default_tree_path": str(shot_dir / "mdsplus/shots"),
    })


def _fetch_shot(
    shot: int, plan: list[dict], out_dir: str, retries: int, stage_dir: str | None = None
) -> dict:
    """Fetch every planned node for one shot and write `<shot>.h5` atomically."""
    if stage_dir:
        use_local_stage(Path(stage_dir) / str(shot))
    out_dir = Path(out_dir)
    final = out_dir / f"{shot}.h5"
    part = out_dir / f"{shot}.h5.part"
    started = time.monotonic()
    records = []
    trees = ShotTrees(shot)
    counts = {"ok": 0, "empty": 0, "nodata": 0, "missing": 0, "error": 0}
    with h5py.File(part, "w") as f:
        root = f.create_group(str(shot))
        for entry in plan:
            fb = entry["fallback"]
            for i, node in enumerate(entry["nodes"]):
                targets = [(entry["tree"], node)]
                if fb:
                    targets.append((fb["tree"], fb["nodes"][i]))
                for tree, nd in targets:
                    if nd in root.get(tree, {}):
                        break  # repeated node in the config (filterscopes lists PMT04 twice)
                    t0 = time.monotonic()
                    rec, status, err = fetch_with_retry(tree, nd, shot, retries, trees)
                    counts[status] += 1
                    records.append({
                        "group": entry["group"], "tree": tree, "node": nd,
                        "status": status, "error": err,
                        "shape": list(rec["data"].shape) if rec else None,
                        "seconds": round(time.monotonic() - t0, 2),
                    })
                    if rec is not None:
                        g = root.require_group(tree).create_group(nd)
                        g.create_dataset("data", data=rec["data"])
                        g.create_dataset("dim0", data=rec["dim0"])
                        break  # primary had data: no fallback fetch
    os.chmod(part, 0o664)
    part.rename(final)
    summary = {
        "shot": shot, "host": socket.gethostname(),
        "seconds": round(time.monotonic() - started, 1),
        "bytes": final.stat().st_size, "counts": counts,
    }
    (out_dir / f"{shot}.fetch.json").write_text(
        json.dumps({**summary, "signals": records}, indent=1)
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--shot-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--groups", nargs="+", help="only these modality groups")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--limit", type=int, help="first N pending shots only")
    parser.add_argument(
        "--stage-dir", type=Path,
        help="read staged files (stage_shots_fdp.py) instead of the network; only "
             "shots with a complete <stage-dir>/<shot>/ are processed",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    plan = signal_plan(config, args.groups)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shots = read_shots(args.shot_file)
    pending = [s for s in shots if not (args.out_dir / f"{s}.h5").exists()]
    if args.limit:
        pending = pending[: args.limit]
    n_nodes = sum(len(e["nodes"]) for e in plan)
    print(
        f"{len(shots)} shots, {len(shots) - len(pending)} already fetched, "
        f"{len(pending)} to fetch; {len(plan)} groups, {n_nodes} nodes; "
        f"{args.workers} workers -> {args.out_dir}",
        flush=True,
    )
    if args.stage_dir:
        pending = [s for s in pending if (args.stage_dir / str(s)).is_dir()]
    stage = str(args.stage_dir) if args.stage_dir else None
    jobs = [(s, plan, str(args.out_dir), args.retries, stage) for s in pending]
    done = 0
    # Forked before any toksearch import: fetch_one imports it inside the worker.
    with Pool(args.workers, maxtasksperchild=1) as pool:
        for summary in pool.imap_unordered(fetch_shot, jobs):
            done += 1
            print(
                f"[{done}/{len(pending)}] {summary['shot']} {summary['seconds']:.0f}s "
                f"{summary['bytes'] / 1e9:.2f} GB {summary['counts']}",
                flush=True,
            )
    sys.exit(0)


if __name__ == "__main__":
    main()
