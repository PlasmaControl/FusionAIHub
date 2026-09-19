"""Read Stellar's `jobstats` and `sacct` reports, and gate a job on utilisation.

WHY THIS EXISTS. Every production job in this project is sized by hand, and a
job that used a fifth of what it reserved is a job that has to be re-sized
before the next one is submitted. Reading the report by eye is how that gets
skipped, so the plan makes it a dependent job: `sbatch --dependency=afterany`
a task that runs this gate and exits 1 when any of CPU, CPU-memory, GPU or
GPU-memory utilisation came in under the threshold (70 % by default).

THE TRAP THIS MODULE IS SHAPED AROUND. `jobstats` does not always know. On
2925412_5 it printed

    CPU memory usage (Value was erroneously found to be >100%)
    ...
        stellar-i01n1: 16.0PB/16GB (4.0PB/4GB per core of 4)
    * The CPU memory usage could not be determined. Try the grafana dashboard.

16 petabytes on a 16 GB allocation is not a measurement. A parser that reads
that line as a number gets 100000000 %; one that shrugs and calls it 0 % fails
a job that in fact used 65.6 % of its memory, which `sacct`'s `MaxRSS` knows.
So an undetermined utilisation is `None` here - never 0.0, never dropped - the
gate first tries `MaxRSS/ReqMem` instead, and only if THAT is missing too does
it fail the job with the reason `undetermined`. Silence is not a pass.

TWO MEMORY NUMBERS, NOT ONE. `jobstats`' CPU-memory figure and `sacct`'s
`MaxRSS/ReqMem` are different measurements of different things: the first is
sampled over the run, the second is the PEAK of one step. Across our own 22
preserved reports the second is systematically higher - 70.0 % vs 99.9 % on
2925387_0 - so a ledger that quoted them interchangeably would not be
comparable to itself. The gate uses one, says in `source` which, and records
BOTH in `checks["cpu_mem"]` as `jobstats_pct` and `sacct_pct`.

AND THE STATE OUTRANKS BOTH. 2925387_1 was killed for exceeding its 10 GB,
and `jobstats` reports its memory utilisation as 74 % - the sampled mean of a
job that hit 100 % and died. Utilisation is a sizing signal only for a job
that ran to completion, so a job or step in any of `FATAL_STATES` fails the
gate outright, with the state as the reason, whatever the percentages say.

WHAT COUNTS AS THE VALUE. `jobstats` prints each number twice: a headline bar
under "Overall Utilization", which is the floor of the real figure, and a
detail line under "Detailed Utilization", which carries a decimal. Both are
kept - `overall_pct` is the bar, and the fields the gate reads are the precise
ones - because the bar is what a human quotes back at you ("23/6/1/97" for the
old AE dataset job) and the decimal is what a threshold at 70 % needs.

USE
    from labeler.jobstats import parse_jobstats, gate
    stats = parse_jobstats(Path("2925387_0.jobstats.txt").read_text())
    verdict = gate(stats, min_cpu=70, min_cpu_mem=70, cpu_only=None)

or, from SLURM, through `scripts/labeler/jobstats_check.py`:

    sbatch --dependency=afterany:$JOBID --wrap "... jobstats_check.py \
        --job-id $JOBID --wait-for-data 300 --preserve-dir $ROOT/runs/slurm"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

#: How long to wait between `jobstats` polls while `--wait-for-data` is set.
#: jobstats is populated from the exporters some time AFTER a job ends - on
#: 2026-09-07 a finished task needed about two minutes before its report had
#: any utilisation in it at all - so a dependent job that reads it the instant
#: the parent exits reads a blank report unless it waits.
POLL_SECONDS = 20.0

#: The default floor for every check, in per cent. The plan's number.
DEFAULT_MIN = 70.0

#: The `sacct` fields the gate needs. `MaxRSS` is per STEP, `ReqMem` is on the
#: job row only, and `AllocCPUS` is needed to expand a per-core `ReqMem`.
SACCT_FORMAT = (
    "JobID,State,Elapsed,ExitCode,MaxRSS,TotalCPU,AllocCPUS,ReqMem,AllocTRES,"
    "NodeList"
)

#: States in which a job's utilisation is not a sizing signal. A job that was
#: killed did not choose how much it used, and the sampled mean of a job that
#: OOMed at 100 % reads like a job with memory to spare.
FATAL_STATES = frozenset({
    "OUT_OF_MEMORY", "TIMEOUT", "NODE_FAIL", "BOOT_FAIL", "DEADLINE",
    "PREEMPTED", "REVOKED", "CANCELLED", "FAILED",
})

#: Which fatal state to name when several are on the record. `OUT_OF_MEMORY`
#: on a step beats the `FAILED` its job row inherits from it: the specific
#: cause is the one that tells you what to re-size.
FATAL_ORDER = ("OUT_OF_MEMORY", "TIMEOUT", "NODE_FAIL", "BOOT_FAIL",
               "DEADLINE", "PREEMPTED", "REVOKED", "CANCELLED", "FAILED")

#: What each memory number actually measures. Carried in the check so a
#: ledger entry says which scale its percentage is on.
MEASUREMENTS = {
    "jobstats": "jobstats: RSS sampled over the run, against the allocation",
    "sacct": "sacct: MaxRSS, the PEAK of one step, over ReqMem",
    "undetermined": "no usable measurement from either source",
}

#: The centred banners `jobstats` divides its report with.
_SECTIONS = {
    "Slurm Job Statistics": "header",
    "Overall Utilization": "overall",
    "Detailed Utilization": "detailed",
    "Notes": "notes",
}

#: Overall-utilisation row label -> the short name used everywhere else.
_METRICS = {
    "CPU utilization": "cpu",
    "CPU memory usage": "cpu_mem",
    "GPU utilization": "gpu",
    "GPU memory usage": "gpu_mem",
}

#: Detailed-utilisation subsection label -> `per_node` key.
_DETAIL_SECTIONS = {
    "CPU utilization per node": "cpu_util",
    "CPU memory usage per node": "cpu_mem",
    "GPU utilization per node": "gpu_util",
    "GPU memory usage per node": "gpu_mem",
}

_BINARY = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3,
           "T": 1024 ** 4, "P": 1024 ** 5, "E": 1024 ** 6}

#: A memory figure as `jobstats` writes it: `7.0GB`, `494.4MB`, `16.0PB`.
_SIZE_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGTPE]?)B\s*$")
#: A memory figure as `sacct` writes it: `10999176K`, `16G`, `2800Mc`, `1500M`.
_SLURM_SIZE_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGTPE]?)([nc]?)\s*$",
                            re.IGNORECASE)
_HEADER_RE = re.compile(r"^\s+([A-Za-z][\w /]*?):\s+(\S.*?)\s*$")
_PCT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%")
#: A ratio above this is not a measurement; see the module docstring.
_ABSURD_PCT = 150.0


# ------------------------------------------------------------------- values

def parse_duration(value: str) -> float | None:
    """`[DD-]HH:MM:SS[.mmm]` or `MM:SS.mmm` -> seconds.

    Both `jobstats` (`Run Time: 00:42:15`) and `sacct` (`TotalCPU: 16:50.339`)
    use this, and `sacct` drops the hours field when there are none, which is
    why the parts are read from the RIGHT.
    """
    value = (value or "").strip()
    if not value:
        return None
    days = 0.0
    if "-" in value:
        head, _, value = value.partition("-")
        try:
            days = float(head)
        except ValueError:
            return None
    parts = value.split(":")
    if len(parts) > 3:
        return None
    try:
        numbers = [float(p) for p in parts]
    except ValueError:
        return None
    seconds = 0.0
    for number in numbers:                       # left to right, base 60
        seconds = seconds * 60.0 + number
    return days * 86400.0 + seconds


def _duration_field(value: str) -> float | None:
    """`Run Time: 00:10:00 (in progress)` -> 600.0.

    A RUNNING report annotates its own run time, and those are exactly the
    reports `--wait-for-data` exists to catch mid-flight, so the trailer is
    dropped rather than allowed to blank the field.
    """
    return parse_duration(re.sub(r"\s*\(.*\)\s*$", "", value or ""))


def parse_size(value: str) -> int | None:
    """A `jobstats` memory figure (`7.0GB`, `16.0PB`) in bytes.

    Binary multipliers: `jobstats` reports a 10 GiB allocation as `10GB`, and
    a percentage taken against the wrong base would be off by 7 %.
    """
    m = _SIZE_RE.match(value or "")
    if not m:
        return None
    return round(float(m.group(1)) * _BINARY[m.group(2).upper()])


def parse_req_mem(req_mem: str, alloc_cpus: int | None = None) -> int | None:
    """`sacct`'s `ReqMem` in bytes: `16G`, `1500M`, `2800Mn`, `4Gc`.

    The trailing letter is Slurm's: `n` is per node, `c` is per CPU-core, and
    a per-core request has to be multiplied by `AllocCPUS` before it means
    anything. Newer Slurm drops the letter and always reports the total.
    """
    m = _SLURM_SIZE_RE.match(req_mem or "")
    if not m:
        return None
    total = float(m.group(1)) * _BINARY[m.group(2).upper()]
    if m.group(3).lower() == "c":
        total *= alloc_cpus or 1
    return round(total)


# --------------------------------------------------------------- jobstats

@dataclass
class JobStats:
    """One parsed `jobstats` report.

    Every utilisation is `float | None`, and `None` means UNDETERMINED - the
    report said so, or the row was absent. It never means zero.
    """

    job_id: str = ""
    state: str = ""
    nodes: int | None = None
    cpu_cores: int | None = None
    cpu_mem_gb: float | None = None
    gpus: int = 0
    run_time_s: float | None = None
    time_limit_s: float | None = None
    #: The report was taken while the job was still running.
    in_progress: bool = False
    cpu_util_pct: float | None = None
    cpu_mem_pct: float | None = None
    gpu_util_pct: float | None = None
    gpu_mem_pct: float | None = None
    #: The headline bars, as printed: `{"cpu": 23.0, "cpu_mem": 6.0, ...}`.
    #: Floors of the precise figures, and the numbers a human quotes.
    overall_pct: dict[str, float | None] = field(default_factory=dict)
    #: The detail lines, per node: `cpu_util`, `cpu_mem`, `gpu_util`,
    #: `gpu_mem`, each `{node: {...}}`. Kept verbatim (16.0PB included) so a
    #: reader can see why a value was refused.
    per_node: dict[str, dict] = field(default_factory=dict)
    header: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def has_utilisation(stats: JobStats) -> bool:
    """Has the report been populated yet?

    The exporters fill `jobstats` some minutes after a job ends, and until
    they do the report has a header and no "Overall Utilization" rows. That
    is what `--wait-for-data` waits for.
    """
    return stats.cpu_util_pct is not None or stats.overall_pct.get("cpu") is not None


def _split_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {name: [] for name in _SECTIONS.values()}
    current = "header"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in _SECTIONS:
            current = _SECTIONS[stripped]
            continue
        if set(stripped) == {"="}:               # the rule under each banner
            continue
        sections[current].append(line)
    return sections


def _parse_header(lines: list[str]) -> dict[str, str]:
    header = {}
    for line in lines:
        m = _HEADER_RE.match(line)
        if m:
            header[m.group(1)] = m.group(2)
    return header


def _parse_overall(lines: list[str]) -> tuple[dict[str, float | None], list[str]]:
    overall: dict[str, float | None] = {}
    notes: list[str] = []
    for line in lines:
        stripped = line.strip()
        for label, name in _METRICS.items():
            if not stripped.startswith(label):
                continue
            rest = stripped[len(label):].strip()
            m = _PCT_RE.search(rest)
            if m and rest.startswith("["):
                overall[name] = float(m.group(1))
            else:
                # `(Value was erroneously found to be >100%)` and anything
                # else parenthetical: the report is telling us it does not
                # know. Record the None and keep the words.
                overall[name] = None
                if rest:
                    notes.append(f"{label} {rest}")
            break
    return overall, notes


def _parse_detailed(lines: list[str]) -> dict[str, dict]:
    per_node: dict[str, dict] = {}
    current: str | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        matched = next(
            (key for label, key in _DETAIL_SECTIONS.items()
             if stripped.startswith(label)), None
        )
        if matched:
            current = matched
            per_node.setdefault(current, {})
            continue
        if current is None or ": " not in stripped:
            continue
        node, _, rest = stripped.partition(": ")
        entry = _parse_detail_entry(current, rest)
        if entry is not None:
            per_node[current][node] = entry
    return per_node


def _parse_detail_entry(kind: str, rest: str) -> dict | None:
    if kind == "cpu_util":
        # `02:39:25/03:31:15 (efficiency=75.5%)`
        head = rest.split(" ", 1)[0]
        used, _, run = head.partition("/")
        used_s, run_s = parse_duration(used), parse_duration(run)
        m = re.search(r"efficiency=([0-9.]+)%", rest)
        return {
            "used_s": used_s,
            "run_s": run_s,
            "efficiency_pct": float(m.group(1)) if m else None,
        }
    if kind == "cpu_mem":
        # `7.0GB/10GB (1.4GB/2GB per core of 5)`
        head = rest.split(" ", 1)[0]
        used, _, alloc = head.partition("/")
        used_b, alloc_b = parse_size(used), parse_size(alloc)
        return {
            "used_bytes": used_b,
            "alloc_bytes": alloc_b,
            "detail": rest,
            "pct": _ratio_pct(used_b, alloc_b),
        }
    if kind == "gpu_util":
        # `0.6%`
        m = _PCT_RE.search(rest)
        return {"pct": float(m.group(1)) if m else None}
    if kind == "gpu_mem":
        # `38.6GB/40GB (96.6%)`
        head = rest.split(" ", 1)[0]
        used, _, total = head.partition("/")
        used_b, total_b = parse_size(used), parse_size(total)
        m = _PCT_RE.search(rest)
        pct = float(m.group(1)) if m else _ratio_pct(used_b, total_b)
        return {"used_bytes": used_b, "total_bytes": total_b, "pct": pct}
    return None


def _ratio_pct(numerator: int | None, denominator: int | None) -> float | None:
    if not numerator or not denominator:
        return None
    return round(100.0 * numerator / denominator, 1)


def _parse_notes(lines: list[str]) -> list[str]:
    """The `* ...` bullets, each rewrapped onto one line."""
    notes: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("* "):
            notes.append(stripped[2:])
        elif notes:
            notes[-1] = f"{notes[-1]} {stripped}"
    return notes


def _int(value: str | None) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_jobstats(text: str) -> JobStats:
    """Parse one `jobstats` report. Never raises on a blank or partial one."""
    sections = _split_sections(text or "")
    header = _parse_header(sections["header"])
    overall, overall_notes = _parse_overall(sections["overall"])
    per_node = _parse_detailed(sections["detailed"])
    notes = _parse_notes(sections["notes"]) + overall_notes

    stats = JobStats(
        job_id=header.get("Job ID", ""),
        state=header.get("State", ""),
        nodes=_int(header.get("Nodes")),
        cpu_cores=_int(header.get("CPU Cores")),
        cpu_mem_gb=_cpu_mem_gb(header.get("CPU Memory", "")),
        gpus=_int(header.get("GPUs")) or 0,
        run_time_s=_duration_field(header.get("Run Time", "")),
        time_limit_s=_duration_field(header.get("Time Limit", "")),
        in_progress="(in progress)" in header.get("Run Time", ""),
        overall_pct=overall,
        per_node=per_node,
        header=header,
        notes=notes,
    )
    stats.cpu_util_pct = _cpu_util(per_node.get("cpu_util", {}), overall)
    stats.cpu_mem_pct = _cpu_mem(per_node.get("cpu_mem", {}), overall)
    stats.gpu_util_pct = _mean_pct(per_node.get("gpu_util", {}), overall, "gpu")
    stats.gpu_mem_pct = _mean_pct(per_node.get("gpu_mem", {}), overall, "gpu_mem")
    return stats


def _cpu_mem_gb(value: str) -> float | None:
    """`10GB (2GB per CPU-core)` -> 10.0."""
    size = parse_size(value.split(" ", 1)[0]) if value else None
    return None if size is None else round(size / 1024 ** 3, 3)


def _cpu_util(nodes: dict, overall: dict) -> float | None:
    """CPU utilisation, preferring the decimal on the detail lines.

    Over several nodes the per-node efficiencies are averaged weighted by run
    time, which is what "overall" means; over one node - every job we run -
    the answer is that node's reported efficiency exactly.
    """
    effs = [(e["efficiency_pct"], e["run_s"] or 1.0) for e in nodes.values()
            if e.get("efficiency_pct") is not None]
    if effs:
        weight = sum(w for _, w in effs)
        return round(sum(v * w for v, w in effs) / weight, 1)
    used = sum(e["used_s"] for e in nodes.values() if e.get("used_s"))
    run = sum(e["run_s"] for e in nodes.values() if e.get("run_s"))
    if used and run:
        return round(100.0 * used / run, 1)
    return overall.get("cpu")


def _cpu_mem(nodes: dict, overall: dict) -> float | None:
    """CPU memory utilisation, or `None` when the report disowned its own.

    `overall` holding the key with a `None` value is the report saying "the
    value was erroneously found to be >100%", and that verdict outranks the
    detail lines - they are where the 16.0PB is.
    """
    if "cpu_mem" in overall and overall["cpu_mem"] is None:
        return None
    used = sum(e["used_bytes"] for e in nodes.values() if e.get("used_bytes"))
    alloc = sum(e["alloc_bytes"] for e in nodes.values() if e.get("alloc_bytes"))
    pct = _ratio_pct(used, alloc)
    if pct is not None:
        return None if pct > _ABSURD_PCT else pct
    return overall.get("cpu_mem")


def _mean_pct(nodes: dict, overall: dict, key: str) -> float | None:
    values = [e["pct"] for e in nodes.values() if e.get("pct") is not None]
    if values:
        return round(sum(values) / len(values), 1)
    return overall.get(key)


# ------------------------------------------------------------------ sacct

@dataclass(frozen=True)
class SacctRow:
    """One line of `sacct -P` output - a job or one of its steps."""

    job_id: str = ""
    state: str = ""
    elapsed_s: float | None = None
    exit_code: str = ""
    max_rss_bytes: int | None = None
    total_cpu_s: float | None = None
    alloc_cpus: int | None = None
    req_mem: str = ""
    #: `billing=256,cpu=4,gres/gpu=1,mem=48G,node=1` - the ALLOCATION record,
    #: and the only independent witness to whether a job had a GPU.
    alloc_tres: str = ""
    node_list: str = ""
    raw: dict[str, str] = field(default_factory=dict)

    @property
    def is_step(self) -> bool:
        return "." in self.job_id


def parse_sacct(text: str) -> list[SacctRow]:
    """Parse pipe-separated `sacct -P` output, header row included.

    Read by column NAME, not position: the captures in the repository were
    taken with two different `--format` lists and both have to work.
    """
    lines = [line for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return []
    names = lines[0].split("|")
    if "JobID" not in names:                     # already `-n`, no header
        return []
    rows = []
    for line in lines[1:]:
        raw = dict(zip(names, line.split("|")))
        rows.append(SacctRow(
            job_id=raw.get("JobID", ""),
            state=raw.get("State", ""),
            elapsed_s=parse_duration(raw.get("Elapsed", "")),
            exit_code=raw.get("ExitCode", ""),
            max_rss_bytes=parse_req_mem(raw.get("MaxRSS", "")),
            total_cpu_s=parse_duration(raw.get("TotalCPU", "")),
            alloc_cpus=_int(raw.get("AllocCPUS")),
            req_mem=raw.get("ReqMem", "").strip(),
            alloc_tres=raw.get("AllocTRES", "").strip(),
            node_list=raw.get("NodeList", ""),
            raw=raw,
        ))
    return rows


def max_rss_bytes(rows: list[SacctRow]) -> int | None:
    """The largest `MaxRSS` over every step, in bytes.

    Per step, because the job row never carries one: the payload's RSS is on
    the `.0` step and the `.batch` step only ever shows the shell's few MB.
    """
    values = [r.max_rss_bytes for r in rows if r.max_rss_bytes]
    return max(values) if values else None


def max_rss_step(rows: list[SacctRow]) -> str | None:
    """Which step supplied `max_rss_bytes`. Needed to know whether to trust it."""
    best = None
    for row in rows:
        if row.max_rss_bytes and (best is None or row.max_rss_bytes > best[0]):
            best = (row.max_rss_bytes, row.job_id)
    return best[1] if best else None


def _is_shell_step(job_id: str) -> bool:
    """`.batch` is the sbatch shell and `.extern` the container - not payloads."""
    return job_id.endswith((".batch", ".extern"))


def _has_payload_rss(rows: list[SacctRow]) -> bool:
    return any(r.max_rss_bytes and r.is_step and not _is_shell_step(r.job_id)
               for r in rows)


def tres_gpus(alloc_tres: str) -> int:
    """GPUs in a `sacct` TRES string: `...,gres/gpu=1,...` -> 1."""
    m = re.search(r"gres/gpu(?::[\w.]+)?=(\d+)", alloc_tres or "")
    return int(m.group(1)) if m else 0


def sacct_memory_pct(rows: list[SacctRow] | None) -> tuple[float | None, str]:
    """`MaxRSS/ReqMem` as a percentage, or `(None, why not)`.

    Three ways this refuses to answer, and each of them was a bug once:

    * an absurd ratio - `sacct` has the same glitch `jobstats` does, and a
      `MaxRSS` of 17179869184K against 16G is 102 400 %, which before this
      guard was compared to the threshold and PASSED;
    * a `MaxRSS` that is only the batch shell's - a few MB of bash, which
      renders as a confident `0 %` and would send someone to shrink a
      reservation that may have been nearly full. Note that in an sbatch with
      no `srun` the `.batch` step IS the payload, so this refuses only when
      the record shows no numbered step at all: undetermined-and-FAIL is the
      safe direction, and `jobstats` usually has the number anyway;
    * no `ReqMem`, or no `MaxRSS` anywhere.
    """
    if not rows:
        return None, "no sacct rows"
    req = req_mem_bytes(rows)
    if not req:
        return None, "sacct recorded no ReqMem"
    rss = max_rss_bytes(rows)
    if not rss:
        return None, "sacct recorded no MaxRSS"
    step = max_rss_step(rows)
    if step and _is_shell_step(step) and not _has_payload_rss(rows):
        return None, (f"the only MaxRSS on the record is {step}'s, which is "
                      "the batch shell and not the payload")
    pct = _ratio_pct(rss, req)
    if pct is None or pct > _ABSURD_PCT:
        return None, (f"MaxRSS/ReqMem is {pct} %, which is not a measurement "
                      "(see the 16.0PB case)")
    return pct, ""


def req_mem_bytes(rows: list[SacctRow]) -> int | None:
    """`ReqMem` from the job row, in bytes, with a per-core suffix expanded."""
    for row in rows:
        if not row.is_step and row.req_mem:
            return parse_req_mem(row.req_mem, row.alloc_cpus)
    for row in rows:
        if row.req_mem:
            return parse_req_mem(row.req_mem, row.alloc_cpus)
    return None


# --------------------------------------------------------------- frontier

#: The header `scripts/slurm_frontier/_gpu_sampler.sh` writes, and the order of
#: its columns: `epoch_s,gpu_pct,vram_used_mb,vram_total_mb`.
GPU_SAMPLE_HEADER = "epoch_s,gpu_pct,vram_used_mb,vram_total_mb"


def frontier_backend() -> bool:
    """Is this a cluster with `sacct` but no `jobstats`? (Frontier is.)

    `jobstats` is a Princeton tool; OLCF does not have it, and on Frontier the
    gate would otherwise read every check as `undetermined` and fail every job
    it was pointed at. Detected rather than configured so the same command line
    works on both clusters.
    """
    return shutil.which("jobstats") is None and shutil.which("sacct") is not None


def parse_gpu_samples(text: str | None) -> tuple[float | None, float | None]:
    """`_gpu_sampler.sh`'s CSV -> `(mean GPU busy %, peak VRAM %)`.

    MEAN for utilisation and PEAK for memory, which is the pairing the rest of
    this module already uses: a job is sized on the memory it needed at its
    worst and on the compute it used on average. Malformed lines - the header,
    a final line truncated by the job's death - are skipped, and a sample whose
    `vram_total_mb` is 0 (the sampler's placeholder when `rocm-smi` failed)
    cannot be a denominator.
    """
    busy: list[float] = []
    used: list[float] = []
    total: list[float] = []
    for line in (text or "").splitlines():
        parts = line.strip().split(",")
        if len(parts) < 4:
            continue
        try:
            sample = [float(p) for p in parts[1:4]]
        except ValueError:
            continue                             # the header, or a torn line
        busy.append(sample[0])
        used.append(sample[1])
        total.append(sample[2])
    mean = round(sum(busy) / len(busy), 1) if busy else None
    cap = round(max(total)) if total else 0
    peak = _ratio_pct(round(max(used)), cap) if used and cap else None
    return mean, peak


def parse_frontier(sacct_text: str, rocm_samples: str | None) -> JobStats:
    """A `JobStats` built from `sacct` and, if the job had one, the GPU samples.

    `TotalCPU / (Elapsed * AllocCPUS)` is the same CPU efficiency `jobstats`
    prints, computed from the accounting record instead of read off a report.
    The peak `MaxRSS` over the steps against `ReqMem` is the memory figure -
    `sacct_memory_pct`, refusals included, so an absurd ratio or a batch-shell
    MaxRSS stays `None` here exactly as it does on Stellar.

    The GPU keys go into `overall_pct` ONLY when there are samples: that is
    what `_has_gpu` reads, so a CPU job is auto-detected as CPU-only and is not
    failed for the GPU it never had.
    """
    rows = parse_sacct(sacct_text)
    job = next((r for r in rows if not r.is_step), rows[0] if rows else SacctRow())
    steps = [r.total_cpu_s for r in rows if r.is_step and r.total_cpu_s]
    total_cpu_s = max(steps) if steps else job.total_cpu_s
    core_seconds = (job.elapsed_s or 0.0) * (job.alloc_cpus or 0)
    cpu_util = (round(100.0 * total_cpu_s / core_seconds, 1)
                if total_cpu_s and core_seconds else None)
    cpu_mem, _ = sacct_memory_pct(rows)
    gpu_util, gpu_mem = parse_gpu_samples(rocm_samples)
    req = req_mem_bytes(rows)

    overall: dict[str, float | None] = {"cpu": cpu_util, "cpu_mem": cpu_mem}
    if rocm_samples is not None:
        overall["gpu"] = gpu_util
        overall["gpu_mem"] = gpu_mem
    return JobStats(
        job_id=job.job_id,
        state=job.state,
        cpu_cores=job.alloc_cpus,
        cpu_mem_gb=None if req is None else round(req / 1024 ** 3, 3),
        run_time_s=job.elapsed_s,
        cpu_util_pct=cpu_util,
        cpu_mem_pct=cpu_mem,
        gpu_util_pct=gpu_util,
        gpu_mem_pct=gpu_mem,
        overall_pct=overall,
        notes=["utilisation computed from sacct"
               + (" and rocm-smi samples" if rocm_samples is not None else "")
               + ": this cluster has no jobstats"],
    )


def gpu_samples_path(job_id: str) -> Path:
    """Where `_gpu_sampler.sh` left this job's samples.

    Under the shot_design data root, beside the job's own `.out`, because that
    is the directory the sbatch `--output` line already points at. The labeler
    root is the fallback for a job submitted without the shot_design env.
    """
    root = os.environ.get("SHOT_DESIGN_DATA_ROOT") or os.environ.get("IDEATE_DATA_ROOT")
    base = Path(root) / "runs" / "slurm" if root else default_out().parent
    return base / f"{job_id}.gpu.csv"


def read_gpu_samples(job_id: str) -> str | None:
    """The job's GPU samples, or `None` - a CPU job, or a sampler that never ran."""
    try:
        return gpu_samples_path(job_id).read_text()
    except OSError:
        return None


# ------------------------------------------------------------------- gate

@dataclass
class Verdict:
    """The gate's answer: what was checked, what failed, and whether it counts."""

    passed: bool = False
    checks: dict[str, dict] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    #: A pilot is measured and reported like any other job but never fails the
    #: exit code - the point of a 20-shot pilot is to LEARN the sizing.
    exempt: bool = False

    @property
    def ok(self) -> bool:
        """Should the process exit 0?"""
        return self.passed or self.exempt

    def to_dict(self) -> dict:
        return asdict(self)


def _has_gpu(stats: JobStats, rows: list[SacctRow] | None = None) -> bool:
    """Did this job have a GPU? `jobstats` first, then the allocation record.

    `jobstats`' GPU rows are not always there; `sacct`'s `AllocTRES` is what
    Slurm actually handed out and it always is. Without the cross-check a GPU
    job whose report arrived with CPU rows only would auto-detect as CPU-only
    and pass on CPU alone - the wrong-PASS this gate exists to prevent.
    """
    if (stats.gpus or stats.per_node.get("gpu_util")
            or stats.per_node.get("gpu_mem") or "gpu" in stats.overall_pct):
        return True
    return any(tres_gpus(row.alloc_tres) for row in rows or [])


def job_states(stats: JobStats, rows: list[SacctRow] | None) -> list[tuple[str, str]]:
    """Every state on the record - jobstats header, sacct job row, every step."""
    states = []
    if stats.state:
        states.append((stats.state, "jobstats"))
    for row in rows or []:
        if row.state:
            states.append((row.state, f"sacct {row.job_id}"))
    return states


def _fatal_state(states: list[tuple[str, str]]) -> tuple[str | None, str | None]:
    """The most specific fatal state on the record, and where it was found."""
    best: tuple[int, str, str] | None = None
    for state, where in states:
        head = state.split()[0].upper() if state.split() else ""
        if head not in FATAL_STATES:
            continue
        rank = FATAL_ORDER.index(head)
        if best is None or rank < best[0]:
            best = (rank, state, where)
    return (best[1], best[2]) if best else (None, None)


def _check(name: str, value: float | None, threshold: float, source: str,
           reasons: list[str], *, detail: str = "",
           extra: dict | None = None) -> dict:
    check = dict(extra or {})
    if value is None:
        reasons.append(
            f"{name} undetermined: {detail or 'no usable value was reported'}"
        )
        check.update(value=None, threshold=threshold, passed=False,
                     source="undetermined")
        return check
    passed = value >= threshold
    if not passed:
        reasons.append(f"{name} {value:g} % < {threshold:g} % ({source})")
    check.update(value=value, threshold=threshold, passed=passed, source=source)
    return check


def gate(
    stats: JobStats,
    *,
    min_cpu: float = DEFAULT_MIN,
    min_cpu_mem: float = DEFAULT_MIN,
    min_gpu: float = DEFAULT_MIN,
    min_gpu_mem: float = DEFAULT_MIN,
    cpu_only: bool | None = None,
    sacct: list[SacctRow] | None = None,
    exempt: bool = False,
    tool_error: str = "",
) -> Verdict:
    """Judge one job. `cpu_only=None` auto-detects it from the absence of GPUs.

    A CPU-only job is judged on CPU and CPU-memory only; asking a job with no
    GPU how well it used its GPU would fail every CPU job on the cluster.
    """
    if cpu_only is None:
        cpu_only = not _has_gpu(stats, sacct)
    reasons: list[str] = []
    checks: dict[str, dict] = {}

    # `jobstats`' stderr distinguishes "no such job" from "not populated
    # yet", and without it a failed check job is undiagnosable. Collapsed to
    # one line: a reason is a line, and jobstats wraps its errors.
    error = " ".join((tool_error or "").split())

    def because(base: str) -> str:
        return f"{base} [{error}]" if error else base

    # State first, and it outranks every percentage below it: see the module
    # docstring. A job that was killed is not evidence about its own sizing.
    states = job_states(stats, sacct)
    if states:
        fatal, where = _fatal_state(states)
        if fatal:
            reasons.append(
                f"state {fatal} ({where}): the job did not run to completion, "
                f"so its utilisation is not a sizing signal"
            )
        checks["state"] = {
            "value": fatal or states[0][0],
            "threshold": None,
            "passed": fatal is None,
            "source": where or states[0][1],
        }

    checks["cpu"] = _check("cpu", stats.cpu_util_pct, min_cpu, "jobstats",
                           reasons,
                           detail=because("jobstats reported no CPU utilisation"))

    # Both memory numbers, always, and a note saying which scale was used.
    jobstats_mem = stats.cpu_mem_pct
    sacct_mem, sacct_note = sacct_memory_pct(sacct)
    if jobstats_mem is not None:
        mem, source = jobstats_mem, "jobstats"
    elif sacct_mem is not None:
        mem, source = sacct_mem, "sacct"
    else:
        mem, source = None, "undetermined"
    checks["cpu_mem"] = _check(
        "cpu_mem", mem, min_cpu_mem, source, reasons,
        detail=because(
            "jobstats reported no usable value and "
            + (sacct_note or "there was no sacct fallback")
        ),
        extra={"jobstats_pct": jobstats_mem, "sacct_pct": sacct_mem,
               "measurement": MEASUREMENTS[source]},
    )

    if not cpu_only:
        gpu_detail = because(
            "the job was allocated a GPU but jobstats reported no GPU rows"
        )
        checks["gpu"] = _check("gpu", stats.gpu_util_pct, min_gpu, "jobstats",
                               reasons, detail=gpu_detail)
        checks["gpu_mem"] = _check("gpu_mem", stats.gpu_mem_pct, min_gpu_mem,
                                   "jobstats", reasons, detail=gpu_detail)
    return Verdict(
        passed=all(c["passed"] for c in checks.values()),
        checks=checks,
        reasons=reasons,
        exempt=exempt,
    )


# -------------------------------------------------------- array expansion

def expand_array(job_id: str, sacct_text: str) -> list[str]:
    """`2925387` -> `["2925387_0", ...]`, from `sacct -j ID -X -n -P`.

    An id that already names a task is returned as itself. A still-pending
    array prints its remaining tasks as a range (`99_[3-5,8%2]`), so the
    ranges are expanded too and the `%` throttle dropped.
    """
    base, _, suffix = job_id.partition("_")
    if suffix:
        return [job_id]
    ids: list[str] = []
    for line in (sacct_text or "").splitlines():
        line = line.strip()
        # Exact match on the id, or `2925387` would claim `29253870_1`.
        if not line or not (line == base or line.startswith(base + "_")):
            continue
        for task in _expand_one(line):
            if task not in ids:
                ids.append(task)
    return ids or [job_id]


def _expand_one(entry: str) -> list[str]:
    entry = re.sub(r"%\d+", "", entry)           # drop the `%N` throttle
    m = re.match(r"^(.+)_\[(.+)\]$", entry)
    if not m:
        return [entry]
    prefix, spec = m.group(1), m.group(2)
    tasks = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        lo, dash, hi = piece.partition("-")
        try:
            span = range(int(lo), int(hi) + 1) if dash else [int(lo)]
        except ValueError:
            continue
        tasks.extend(f"{prefix}_{i}" for i in span)
    return tasks or [entry]


# --------------------------------------------------------- talking to slurm

def _sleep(seconds: float) -> None:
    """Indirection so a test can wait without waiting."""
    time.sleep(seconds)


def _capture(cmd: list[str]) -> tuple[str, str]:
    """Run a read-only Slurm query; return `(stdout, stderr)`.

    stderr is kept because "no such job" and "the report is not populated
    yet" both come back as an empty report, and a check job whose ledger
    entry does not say which of the two happened cannot be diagnosed.
    """
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return "", f"{cmd[0]}: {exc}"
    return done.stdout or "", (done.stderr or "").strip()


def fetch_jobstats(job_id: str,
                   wait_for_data: float = 0.0) -> tuple[str, str, JobStats]:
    """`jobstats ID`, retried until the report has utilisation in it.

    The exporters populate the report a few minutes after the job ends, so a
    dependent check job that reads it immediately reads a header and nothing
    else. Polling is the only way; giving up leaves every value `None`, which
    the gate treats as undetermined and therefore a failure - not a pass.
    """
    text, err = _capture(["jobstats", job_id])
    stats = parse_jobstats(text)
    waited = 0.0
    while not has_utilisation(stats) and waited < wait_for_data:
        _sleep(POLL_SECONDS)
        waited += POLL_SECONDS
        text, err = _capture(["jobstats", job_id])
        stats = parse_jobstats(text)
    return text, err, stats


def fetch_sacct(job_id: str) -> tuple[str, str]:
    return _capture(["sacct", "-j", job_id, "-P", f"--format={SACCT_FORMAT}"])


def fetch_array_listing(job_id: str) -> tuple[str, str]:
    return _capture(["sacct", "-j", job_id, "-X", "-n", "-P", "--format=JobID"])


# -------------------------------------------------------------------- CLI

def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:g} %"


def format_line(job_id: str, stats: JobStats, verdict: Verdict) -> str:
    """`2925387_0  CPU 75.5 %  CPU-mem 70 %  GPU n/a  GPU-mem n/a  PASS`."""
    fallback = {
        "cpu": stats.cpu_util_pct, "cpu_mem": stats.cpu_mem_pct,
        "gpu": stats.gpu_util_pct, "gpu_mem": stats.gpu_mem_pct,
    }
    cells = []
    for key, label in (("cpu", "CPU"), ("cpu_mem", "CPU-mem"),
                       ("gpu", "GPU"), ("gpu_mem", "GPU-mem")):
        check = verdict.checks.get(key)
        value = check["value"] if check else fallback[key]
        cells.append(f"{label} {_fmt(value)}")
    status = "PASS" if verdict.passed else "FAIL"
    if verdict.exempt:
        status += " (exempt)"
    return "  ".join([job_id, *cells, status])


def default_out() -> Path:
    """`$LABELER_ROOT/runs/slurm/jobstats.json`.

    Under the DATA root, not the repository: nothing this project runs writes
    a new file into the source tree.
    """
    from labeler.config import Paths

    return Paths.from_env().runs / "slurm" / "jobstats.json"


def _atomic_write(path: Path, text: str) -> None:
    """Replace `path` in one step, so no reader ever sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_record(out: Path, record: dict) -> None:
    """Append-or-replace one job's record in the JSON ledger, keyed by id."""
    doc = {"jobs": {}}
    if out.is_file():
        try:
            loaded = json.loads(out.read_text())
            if isinstance(loaded.get("jobs"), dict):
                doc = loaded
        except (json.JSONDecodeError, OSError, AttributeError):
            pass                                 # a corrupt ledger is replaced
    doc["jobs"][record["job_id"]] = record
    doc["updated"] = record["timestamp"]
    # Read-modify-write, replaced atomically. Two check jobs finishing at the
    # same instant can still lose a record, but neither can leave a truncated
    # ledger behind; `--out` per array is the way to avoid the race entirely.
    _atomic_write(out, json.dumps(doc, indent=2, sort_keys=True) + "\n")


class _Formatter(argparse.ArgumentDefaultsHelpFormatter):
    """`ArgumentDefaultsHelpFormatter`, minus `(default: )` on the empty ones."""

    def _get_help_string(self, action):
        default = action.default
        if (default is None or default == "" or isinstance(default, bool)
                or default is argparse.SUPPRESS):
            return action.help
        return super()._get_help_string(action)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jobstats_check.py",
        description=(
            "Gate a finished SLURM job on its utilisation. Exits 1 if any of "
            "CPU, CPU-memory, GPU or GPU-memory came in under the threshold, "
            "or if jobstats could not determine one of them."
        ),
        formatter_class=_Formatter,
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--job-id", default="",
        help="run `jobstats` and `sacct` for this id; an array id is expanded "
             "and every task is gated",
    )
    source.add_argument(
        "--jobstats-file", default="",
        help="a saved `jobstats` report to read instead (offline)",
    )
    p.add_argument("--sacct-file", default="",
                   help="a saved `sacct -P` capture to go with --jobstats-file")
    p.add_argument("--min-cpu", type=float, default=DEFAULT_MIN,
                   help="floor for CPU utilisation, per cent")
    p.add_argument("--min-cpu-mem", type=float, default=DEFAULT_MIN,
                   help="floor for CPU memory utilisation, per cent")
    p.add_argument("--min-gpu", type=float, default=DEFAULT_MIN,
                   help="floor for GPU utilisation, per cent")
    p.add_argument("--min-gpu-mem", type=float, default=DEFAULT_MIN,
                   help="floor for GPU memory utilisation, per cent")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--cpu-only", action="store_true",
                      help="judge on CPU and CPU-memory only (default: auto, "
                           "from whether the job had GPUs)")
    mode.add_argument("--gpu", action="store_true",
                      help="judge on the GPU checks too, whatever jobstats says")
    p.add_argument("--pilot", action="store_true",
                   help="exempt: measure and report, but always exit 0")
    p.add_argument("--wait-for-data", type=float, default=0.0, metavar="SECONDS",
                   help=f"poll `jobstats` every {POLL_SECONDS:g}s for this long "
                        "until the report is populated")
    p.add_argument("--out", default="",
                   help="JSON ledger to append to (default: "
                        "$LABELER_ROOT/runs/slurm/jobstats.json)")
    p.add_argument("--preserve-dir", default="",
                   help="also write <id>.jobstats.txt and <id>.sacct.txt here")
    p.add_argument("--quiet", action="store_true", help="print nothing")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cpu_only = True if args.cpu_only else (False if args.gpu else None)
    thresholds = {"cpu": args.min_cpu, "cpu_mem": args.min_cpu_mem,
                  "gpu": args.min_gpu, "gpu_mem": args.min_gpu_mem}
    out = Path(args.out) if args.out else default_out()
    preserve = Path(args.preserve_dir) if args.preserve_dir else None

    # A saved report is replayed as what it is, whatever cluster reads it back.
    frontier = frontier_backend() and not args.jobstats_file
    if args.jobstats_file:
        path = Path(args.jobstats_file)
        sacct_text = (
            Path(args.sacct_file).read_text() if args.sacct_file else ""
        )
        captures = [(path.name.split(".")[0], path.read_text(), "", sacct_text, "")]
    else:
        captures = [
            (task, *_capture_task(task, args.wait_for_data, frontier))
            for task in _tasks(args.job_id)
        ]

    failed = False
    for name, jobstats_text, jobstats_err, sacct_text, sacct_err in captures:
        stats = (parse_frontier(sacct_text, read_gpu_samples(name)) if frontier
                 else parse_jobstats(jobstats_text))
        rows = parse_sacct(sacct_text)
        tool_error = (f"jobstats: {jobstats_err}" if jobstats_err
                      else (f"sacct: {sacct_err}" if sacct_err else ""))
        verdict = gate(
            stats, min_cpu=args.min_cpu, min_cpu_mem=args.min_cpu_mem,
            min_gpu=args.min_gpu, min_gpu_mem=args.min_gpu_mem,
            cpu_only=cpu_only, sacct=rows or None, exempt=args.pilot,
            tool_error=tool_error,
        )
        job_id = stats.job_id or name
        record = {
            "job_id": job_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "thresholds": thresholds,
            "cpu_only": (cpu_only if cpu_only is not None
                         else not _has_gpu(stats, rows)),
            "jobstats_text": jobstats_text,
            "jobstats_stderr": jobstats_err,
            "sacct_text": sacct_text,
            "sacct_stderr": sacct_err,
            "stats": stats.to_dict(),
            "verdict": verdict.to_dict(),
        }
        write_record(out, record)
        if preserve is not None:
            # Only ever WRITE a capture, never truncate one: an expired
            # `--wait-for-data` returns an empty report, and the directory
            # this points at is where the evidence of the run that worked is
            # kept.
            _preserve(preserve, f"{job_id}.jobstats.txt", jobstats_text)
            _preserve(preserve, f"{job_id}.sacct.txt", sacct_text)
        if not args.quiet:
            print(format_line(job_id, stats, verdict), flush=True)
            for reason in verdict.reasons:
                print(f"    {reason}", flush=True)
        failed = failed or not verdict.ok
    return 1 if failed else 0


def _preserve(directory: Path, name: str, text: str) -> None:
    if not text.strip():
        return
    _atomic_write(directory / name, text)


def _tasks(job_id: str) -> list[str]:
    _, _, suffix = job_id.partition("_")
    if suffix:
        return [job_id]
    listing, _ = fetch_array_listing(job_id)
    return expand_array(job_id, listing)


def _capture_task(task: str, wait_for_data: float,
                  frontier: bool = False) -> tuple[str, str, str, str]:
    if frontier:
        # There is no `jobstats` here to run or to wait for, and calling it
        # would fill the record's `jobstats_stderr` with a "no such file"
        # that the gate then reports as the reason a job failed.
        sacct_text, sacct_err = fetch_sacct(task)
        return "", "", sacct_text, sacct_err
    text, err, _ = fetch_jobstats(task, wait_for_data)
    sacct_text, sacct_err = fetch_sacct(task)
    return text, err, sacct_text, sacct_err


if __name__ == "__main__":                       # pragma: no cover
    raise SystemExit(main())
