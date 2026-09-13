"""`frame_codes/<shot>.json`: how a frame-code cache was made, beside the cache it describes.

WHY A SIDECAR AND NOT A FIFTH KEY. A `frame_codes/<shot>.pt` is a `torch.save` dict with exactly
`codes`, `actuators`, `n_frames`, `vocabs` -- see `design.seed`'s docstring. The shipped bundle's
own caches have those four and `ignite_infer.validate_shot` compares against them, so a fifth key
is a compatibility risk for a field no consumer reads. The file stays byte-comparable with the
bundle's and everything the payload cannot say lives in a JSON sibling.

WHAT IT HAS TO SAY, AND WHY. The 500-shot encode product is DEVICE-MIXED: ten CUDA pilot caches
and 490 CPU ones. That would be a footnote if the encoder were device-independent, and it is not
-- measured on the login node, 185955's `bes`/`mhr` codes differ between cuda and cpu, and
between cpu at four threads and cpu at eight, by one to four isolated tokens, on the SAME machine
with only `OMP_NUM_THREADS` changed. So "which device, how many threads, which code revision,
which codec bundle, which input file" is the difference between two caches that can be compared
and two that cannot, and none of it was recorded anywhere a reader of the cache would look.

WHAT THE INPUT FINGERPRINT IS. `mtime+size`, not sha256, and the sidecar says so in
`input_fingerprint.kind`. A corpus file is 2-5 GB and there are 500 of them; hashing the set is
tens of minutes of GPFS reads per encode run, for a field nobody would wait for. mtime+size does
not survive a rewrite that preserves both, and a reader who assumed a hash would be claiming more
than this field supports -- which is exactly why the kind is recorded rather than implied.
`fingerprint(path, sha256=True)` is available for the cases where the cost is worth paying.

THE BACKFILL IS A RECONSTRUCTION. `backfill()` writes sidecars for caches that predate this
module, from the run manifests `ideate encode` left under `runs/encode/`. Those manifests record
the output directory, the device and which shots the task encoded; they record no commit and no
thread count. So `git_sha` is null -- not inferred from a timestamp -- and `torch_threads` is the
number the sbatch that ran the job exports, marked as coming from the script rather than from the
run. Every backfilled sidecar carries `backfilled: true`, and `backfill()` never overwrites one a
real encode wrote: a reconstruction must not replace a measurement.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

_log = logging.getLogger(__name__)

SCHEMA = "ideate-frame-codes-provenance-v1"

#: Every key a sidecar has, in this order. Asserted by `test_provenance`, so a field added here
#: without a reader is visible rather than silently absent on the shots written yesterday.
SIDECAR_KEYS: tuple[str, ...] = (
    "schema",
    "shot",
    "device",
    "device_source",
    "torch_threads",
    "torch_threads_source",
    "torch_version",
    "git_sha",
    "ignite_bundle",
    "ignite_bundle_sha",
    "ignite_revision",
    "input_file",
    "input_fingerprint",
    "n_frames",
    "modalities",
    "include_video",
    "encoded_at",
    "encoded_at_source",
    "run_manifest",
    "backfilled",
)

#: `OMP_NUM_THREADS` as the two encode jobs export it. Neither the run manifest nor the cache
#: records the thread count, so a backfilled sidecar can only quote the script that ran the job,
#: and `torch_threads_source` says that is what it is doing.
SBATCH_THREADS = {"cuda": 1, "cpu": 4}
SBATCH_FOR = {
    "cuda": "scripts/ideate/encode.sbatch",
    "cpu": "scripts/ideate/encode_cpu.sbatch",
}


def sidecar_path(codes_dir: Path | str, shot: int) -> Path:
    return Path(codes_dir) / f"{int(shot)}.json"


def fingerprint(path: Path | str | None, sha256: bool = False) -> dict:
    """`{kind, size_bytes, mtime_ns, sha256}` for one input file.

    `kind` is `mtime+size` (the default), `sha256` when the caller paid for a hash, `missing`
    when the file is not there, and `unknown` when no path was given at all. Naming the kind is
    the point of the field: a reader must not have to guess how strong the identity is.
    """
    if path is None:
        return {"kind": "unknown", "size_bytes": None, "mtime_ns": None, "sha256": None}
    path = Path(path)
    try:
        st = path.stat()
    except OSError:
        return {"kind": "missing", "size_bytes": None, "mtime_ns": None, "sha256": None}
    out = {
        "kind": "mtime+size",
        "size_bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "sha256": None,
    }
    if sha256:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(8 << 20), b""):
                digest.update(block)
        out["kind"] = "sha256"
        out["sha256"] = digest.hexdigest()
    return out


def _git_sha() -> str | None:
    """This checkout's short commit, through `shotdb.build`'s implementation rather than a copy.

    Imported inside the function: `shotdb.build` pulls in pandas and the whole feature stack, and
    `design.seed` is imported by the encode path, which does not otherwise need it.
    """
    try:
        from ..shotdb.build import _git_sha as sha

        return sha() or None
    except Exception:  # noqa: BLE001 - provenance may not fail the thing it describes
        return None


def _bundle_identity(bundle: Path | str | None) -> tuple[str | None, str | None, str | None]:
    """`(path, codec-manifest sha256, HF revision)` for the codec bundle a cache was encoded with.

    The bundle is 3.5 GB, so the digest is of `codecs/MANIFEST.json` -- a few kilobytes that name
    every codec file, its source and its size, and which changes whenever the pinned snapshot
    does. The revision is the Hugging Face one from `configs/ideate/ignite_modalities.yaml`.
    """
    revision = None
    try:
        from ..shotdb import ignite

        revision = str(ignite.model_cfg().get("revision") or "") or None
    except Exception:  # noqa: BLE001
        revision = None
    if bundle is None:
        return None, None, revision
    bundle = Path(bundle)
    manifest = bundle / "codecs" / "MANIFEST.json"
    sha = None
    try:
        sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    except OSError:
        sha = None
    return str(bundle), sha, revision


def build_sidecar(
    shot: int,
    *,
    device: str,
    input_file: Path | str | None,
    bundle: Path | str | None,
    n_frames: int | None = None,
    modalities: Sequence[str] | None = None,
    include_video: bool | None = None,
    run_manifest: Path | str | None = None,
    workers: int | None = None,
    torch_threads: int | None = None,
    device_source: str = "the encode run",
    torch_threads_source: str = "torch.get_num_threads() in the encoding process",
    torch_version: str | None = "",
    git_sha: str | None = "",
    backfilled: bool = False,
    encoded_at: str | None = None,
    encoded_at_source: str = "the encoding process's clock",
    sha256_input: bool = False,
) -> dict:
    """One sidecar body.

    `git_sha` and `torch_version` take `""` for "ask this process" and `None` for "not
    recoverable". The distinction is the whole point of the field on a BACKFILLED sidecar: the
    torch version of the process doing the reconstruction is not the one that did the encode, and
    writing it there would put a wrong answer in a provenance field, which is worse than a null.
    """
    if torch_threads is None:
        try:
            import torch

            torch_threads = int(torch.get_num_threads())
        except Exception:  # noqa: BLE001
            torch_threads = None
    if torch_version == "":
        try:
            import torch

            torch_version = str(torch.__version__)
        except Exception:  # noqa: BLE001
            torch_version = None
    path, sha, revision = _bundle_identity(bundle)
    return {
        "schema": SCHEMA,
        "shot": int(shot),
        "device": str(device) if device else None,
        "device_source": device_source,
        "torch_threads": torch_threads,
        "torch_threads_source": torch_threads_source,
        "torch_version": torch_version,
        "git_sha": _git_sha() if git_sha == "" else git_sha,
        "ignite_bundle": path,
        "ignite_bundle_sha": sha,
        "ignite_revision": revision,
        "input_file": str(input_file) if input_file is not None else None,
        "input_fingerprint": fingerprint(input_file, sha256=sha256_input),
        "n_frames": None if n_frames is None else int(n_frames),
        "modalities": None if modalities is None else list(modalities),
        "include_video": include_video,
        "encoded_at": encoded_at or datetime.now(UTC).isoformat(timespec="seconds"),
        "encoded_at_source": encoded_at_source,
        "run_manifest": str(run_manifest) if run_manifest is not None else None,
        "backfilled": bool(backfilled),
    }


def write_sidecar(codes_dir: Path | str, shot: int, body: dict) -> Path:
    """Write `<codes_dir>/<shot>.json` through a temporary sibling, like the cache itself."""
    codes_dir = Path(codes_dir)
    codes_dir.mkdir(parents=True, exist_ok=True)
    final = sidecar_path(codes_dir, shot)
    tmp = final.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(body, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(final)
    return final


def read_sidecar(codes_dir: Path | str, shot: int) -> dict | None:
    """The sidecar, or None -- for an absent one AND for an unreadable one.

    A half-written JSON beside a good cache is a provenance gap, not a reason for `describe_shot`
    to raise: the caller's question is "what do we know about this cache", and "nothing" is an
    answer it can render.
    """
    try:
        return json.loads(sidecar_path(codes_dir, shot).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def device_of(codes_dirs: Iterable[Path | str], shot: int) -> str | None:
    """Which device encoded this shot's cache, from the first directory that has a sidecar."""
    for d in codes_dirs:
        body = read_sidecar(d, shot)
        if body and body.get("device"):
            return str(body["device"])
    return None


# ---------------------------------------------------------------------------------- backfill


def _encode_runs(runs_dir: Path, codes_dir: Path) -> dict[int, tuple[Path, str]]:
    """`shot -> (manifest, device)` for every shot a run manifest says it encoded INTO `codes_dir`.

    The directory also holds the worker- and thread-sweep runs, which encoded some of the same
    shots into scratch directories; reading a sweep's device off would attribute an experiment's
    settings to the production cache, so `out_dir` is matched and the rest ignored. Manifests are
    walked oldest first, so the last run that actually encoded a shot wins -- a shot a later run
    SKIPPED (`--skip-existing`) is not in that run's `elapsed_by_shot` at all, which is what makes
    "last writer" recoverable from these files.
    """
    out: dict[int, tuple[Path, str]] = {}
    target = codes_dir.resolve()
    for path in sorted(Path(runs_dir).glob("encode_*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            _log.warning("unreadable run manifest %s", path)
            continue
        if Path(str(report.get("out_dir", ""))).resolve() != target:
            continue
        device = str(report.get("device") or "")
        for shot in report.get("elapsed_by_shot") or {}:
            out[int(shot)] = (path, device)
    return out


def backfill(
    *,
    codes_dir: Path | str,
    runs_dir: Path | str,
    default_device: str = "cpu",
    dry_run: bool = False,
    force: bool = False,
    log=lambda *_: None,
) -> dict:
    """Write a sidecar for every `<codes_dir>/<shot>.pt` that has none. Returns a report.

    `default_device` is what a cache with no run manifest is recorded as, marked `assumed` in
    `device_source`: the three OOM-killed array tasks of the CPU encode died before writing their
    manifest, and their caches survive. Every CPU job ran `--device cpu`, so the assumption is
    the right one -- and it is still an assumption, which is why it says so in the file.
    """
    codes_dir, runs_dir = Path(codes_dir), Path(runs_dir)
    runs = _encode_runs(runs_dir, codes_dir) if runs_dir.is_dir() else {}
    caches = sorted(int(p.stem) for p in codes_dir.glob("*.pt"))

    written = kept = 0
    without_manifest = 0
    by_device: dict[str, int] = {}
    for shot in caches:
        existing = read_sidecar(codes_dir, shot)
        if existing is not None and not (force and existing.get("backfilled")):
            kept += 1
            by_device[str(existing.get("device"))] = by_device.get(str(existing.get("device")), 0) + 1
            continue
        manifest, device = runs.get(shot, (None, ""))
        if manifest is None:
            without_manifest += 1
            device = default_device
            device_source = (
                f"assumed {default_device}: no run manifest under {runs_dir} names this shot "
                f"(the OOM-killed array tasks died before writing one); every CPU encode job ran "
                f"--device cpu"
            )
        else:
            device_source = f"run manifest {manifest.name}"
        # The cache file's own mtime, per shot, rather than the run manifest's: it is the moment
        # THIS cache was written, it is unambiguous (an epoch, not a local-time stamp in a
        # filename), and the seventy caches whose task was OOM-killed have no manifest at all.
        try:
            cache_mtime = datetime.fromtimestamp(
                (codes_dir / f"{shot}.pt").stat().st_mtime, UTC
            ).isoformat(timespec="seconds")
        except OSError:
            cache_mtime = None
        body = build_sidecar(
            shot,
            device=device,
            input_file=None,
            bundle=None,
            run_manifest=manifest,
            torch_threads=SBATCH_THREADS.get(device),
            encoded_at=cache_mtime,
            encoded_at_source=f"mtime of {shot}.pt (the run manifests record no per-shot clock)",
            # The reconstruction runs in a different process from the encode; its torch version
            # and this checkout's commit describe the backfill, not the artifact.
            torch_version=None,
            device_source=device_source,
            torch_threads_source=(
                f"{SBATCH_FOR.get(device, 'the encode job')} exports OMP_NUM_THREADS="
                f"{SBATCH_THREADS.get(device)}; the run manifest records no thread count"
            ),
            # The run manifests written before this change carry no commit. Deriving one from the
            # file's timestamp would put a plausible sha in a provenance field, which is worse
            # than an empty one: the next reader could not tell the two apart.
            git_sha=None,
            backfilled=True,
        )
        by_device[device] = by_device.get(device, 0) + 1
        written += 1
        if not dry_run:
            write_sidecar(codes_dir, shot, body)
        log(f"{shot} {device} {'(dry run)' if dry_run else ''}".rstrip())
    return {
        "codes_dir": str(codes_dir),
        "runs_dir": str(runs_dir),
        "n_caches": len(caches),
        "n_written": written,
        "n_kept": kept,
        "n_without_manifest": without_manifest,
        "by_device": by_device,
        "dry_run": bool(dry_run),
    }


__all__ = [
    "SBATCH_THREADS",
    "SCHEMA",
    "SIDECAR_KEYS",
    "backfill",
    "build_sidecar",
    "device_of",
    "fingerprint",
    "read_sidecar",
    "sidecar_path",
    "write_sidecar",
]
