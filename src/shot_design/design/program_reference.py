"""Raw reference previews and explicit preparation of source-bound IGNITE caches."""

from __future__ import annotations

import errno
import hashlib
import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np
import torch

from ..config import Paths, load_yaml
from ..env import getenv
from ..shotdb import build
from ..shotdb.corpus import CorpusReader
from ..shotdb.reader import ShotFailed, Unavailable
from . import actuators as act


@dataclass
class Reference:
    cache: dict | None
    controls: act.Actuators
    available: np.ndarray  # (88, F): a finite measurement in this frame
    digest: str
    source_digest: str = ""
    cache_error: str | None = None
    needs_seed: bool = False


def file_identity(path: Path) -> tuple:
    st = path.stat()
    return str(path.resolve()), st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def _source_digest(identity: tuple) -> str:
    """Pin a raw draft to its corpus even before simulation tokens exist."""
    return hashlib.sha256(repr(identity).encode()).hexdigest()


def _corpus(paths: Paths) -> Path:
    return Path(getenv("SHOT_DESIGN_CORPUS") or paths.foundation_model_processed_dir)


def _cache_path(shot: int, paths: Paths) -> Path | None:
    """This shot's frame-code cache, PRODUCTION's copy first.

    `model.frame_codes_cache` is the corpus the pinned dynamics checkpoint was trained on --
    thousands of shots already encoded, read-only, and never written by anything here. When it
    holds the shot, that file is not merely the cheapest answer (a re-encode is ~40 s of GPU and
    a 2-5 GB corpus read): it is the only one guaranteed to carry the codes the checkpoint was
    fitted to. Our own encode agrees bit for bit only on some shots -- changing a BLAS thread
    count on one machine is enough to move a spectro token (see `scripts/shot_design/g_enc.py`).
    The two writable locations follow it, for the shots production never encoded.

    The search order is `shotdb.build.frame_codes_dirs`, not a second list kept here:
    `build` (the `has_frame_codes` column) and `corpus select` read the same function,
    and a root added to one and not the other is exactly how the two commands used to
    disagree about one shot.
    """
    for root in build.frame_codes_dirs(paths):
        path = root / f"{shot}.pt"
        if path.is_file():
            return path
    source = _corpus(paths) / f"{shot}_processed.h5"
    if source.is_file():
        path = (
            paths.ignite_inputs_dir / "reference_cache"
            / _source_digest(file_identity(source)) / f"{shot}.pt"
        )
        if path.is_file():
            return path
    return None


def validate_cache(cache: dict) -> None:
    """Reject incompatible files before slicing or exposing any cached tensors."""
    if not isinstance(cache, dict) or set(cache) != {
        "codes",
        "actuators",
        "n_frames",
        "vocabs",
    }:
        raise ValueError(
            "Seed cache must contain codes, actuators, n_frames and vocabs"
        )
    contract = load_yaml("ignite_modalities.yaml")
    specs = contract["modalities"]
    expected_vocabs = contract["model"]["production_vocabs"]
    codes, vocabs = cache["codes"], cache["vocabs"]
    # THE GENERATION COMES FIRST. Two caches of different codec generations are otherwise
    # indistinguishable -- same four keys, same dtypes, plausible integers -- and `vocabs` is the
    # only field that tells them apart, which is why it is checked before anything structural.
    # Reported the other way round, a v2 cache meeting a v4 checkpoint complains about its
    # actuator dtype or its token width and sends the reader hunting for a bug in a file whose
    # only fault is its age.
    if not isinstance(vocabs, dict) or set(vocabs) != set(expected_vocabs):
        only_one_side = set(expected_vocabs) ^ set(vocabs if isinstance(vocabs, dict) else {})
        raise ValueError(
            f"Seed cache names {len(vocabs) if isinstance(vocabs, dict) else 0} modalities, "
            f"the pinned production generation has {len(expected_vocabs)}; on one side only: "
            f"{', '.join(sorted(only_one_side)) or '(none)'}"
        )
    for name, vocab in vocabs.items():
        if type(vocab) is not int or vocab < 1:
            raise ValueError(f"Seed cache {name} has invalid vocabulary metadata")
        if vocab != expected_vocabs[name]:
            raise ValueError(
                f"Seed cache {name} vocabulary {vocab} does not match pinned production "
                f"vocabulary {expected_vocabs[name]}; use this checkpoint's codec generation"
            )
    frames = cache["n_frames"]
    if type(frames) is not int or frames < 21:
        raise ValueError(
            "Seed cache needs at least 20 history frames and one prediction"
        )
    values = cache["actuators"]
    if (
        not isinstance(values, torch.Tensor)
        or values.dtype != torch.float16
        or tuple(values.shape) != (frames, 88)
        or not torch.isfinite(values).all()
    ):
        raise ValueError(
            "Seed cache actuators must be finite float16 with shape (F, 88)"
        )
    if not isinstance(codes, dict) or set(codes) != set(specs):
        raise ValueError(
            f"Seed cache must contain all {len(specs)} production modalities"
        )
    for name, tokens in codes.items():
        vocab = vocabs[name]
        if (
            not isinstance(tokens, torch.Tensor)
            or tokens.dtype != torch.int32
            or tuple(tokens.shape) != (frames, specs[name]["n_tok"])
        ):
            raise ValueError(f"Seed cache {name} has invalid token dimensions or dtype")
        if tokens.min() < 0 or tokens.max() >= vocab:
            raise ValueError(f"Seed cache {name} tokens exceed its vocabulary")


class _MemoryReader:
    """Read each actuator group once while deriving statistics and availability."""

    def __init__(self, reader):
        self.reader = reader
        self.groups = {}

    def read(self, shot, group, channels=None):
        if group not in self.groups:
            times, values = self.reader.read(shot, group)
            if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise ValueError(f"Reference {group} has invalid time coordinates")
            self.groups[group] = times, values
        t, y = self.groups[group]
        return t, y if channels is None else y[channels]

    def coverage(self, shot, group):
        lo, hi = self.reader.coverage(shot, group)
        # read() strips all-NaN trailing data, but production frame arithmetic
        # uses the original span, including that pad's timestamp.
        if not np.isfinite([lo, hi, hi - lo]).all() or hi <= lo:
            raise ValueError(f"Reference {group} has invalid time coverage endpoints")
        return lo, hi


@lru_cache(maxsize=16)
def _read_controls(shot: int, corpus_identity: tuple, n_frames: int):
    corpus_path = Path(corpus_identity[0])
    reader = _MemoryReader(CorpusReader(corpus_path.parent))
    controls = act.build_actuators(shot, reader, n_frames)
    available = np.zeros_like(controls.raw, dtype=bool)
    for spec in act.ACT_SPEC:
        try:
            times, values = reader.read(shot, spec.group)
            lo, hi = reader.coverage(shot, spec.group)
        except Unavailable:
            continue
        if len(times) < 2:
            continue
        count = act._record_length(times, hi - lo)
        bounds = act._frame_bounds(times, count, hi - lo, controls.n_frames, 0.05, 0)
        # Per-channel availability must precede production's NaN->zero padding.
        for i in range(min(spec.n_channels, values.shape[0])):
            cumulative = np.r_[0, np.cumsum(np.isfinite(values[i]))]
            clipped = np.clip(bounds, 0, values.shape[1])
            available[spec.offset + i] = (
                cumulative[clipped[:, 1]] > cumulative[clipped[:, 0]]
            )
    if file_identity(corpus_path) != corpus_identity:
        raise ValueError("Reference source changed while reading; retry preview")
    return controls, available


def comparison_reference(shot: int, paths: Paths, n_frames: int) -> Reference:
    """Overlays need raw measurements only, never diagnostic seed tokens."""
    corpus = Path(getenv("SHOT_DESIGN_CORPUS") or paths.foundation_model_processed_dir)
    source = corpus / f"{shot}_processed.h5"
    try:
        controls, available = _read_controls(shot, file_identity(source), n_frames)
    except (OSError, ShotFailed, IndexError, TypeError) as exc:
        raise ValueError(f"Cannot read comparison shot {shot}: {exc}") from exc
    return Reference({}, controls, available, "")


@lru_cache(maxsize=4)
def _read_reference(
    shot: int, corpus_identity: tuple, cache_identity: tuple
) -> Reference:
    corpus_path, cache_path = Path(corpus_identity[0]), Path(cache_identity[0])
    try:
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Cannot safely load seed cache for {shot}: {exc}") from exc
    validate_cache(cache)
    controls, available = _read_controls(shot, corpus_identity, cache["n_frames"])
    if (
        file_identity(corpus_path) != corpus_identity
        or file_identity(cache_path) != cache_identity
    ):
        raise ValueError("Reference source changed while reading; retry preview")
    digest = hashlib.sha256()
    digest.update(repr((shot, corpus_identity, cache_identity)).encode())
    digest.update(controls.raw.tobytes())
    digest.update(cache["actuators"].numpy().tobytes())
    for name in sorted(cache["codes"]):
        digest.update(name.encode())
        digest.update(str(cache["vocabs"][name]).encode())
        digest.update(cache["codes"][name].numpy().tobytes())
    return Reference(
        cache, controls, available, digest.hexdigest(), _source_digest(corpus_identity)
    )


def _raw_reference(shot: int, identity: tuple, error: str, missing: bool) -> Reference:
    reader = _MemoryReader(CorpusReader(Path(identity[0]).parent))
    ends = []
    for spec in act.ACT_SPEC:
        try:
            _, hi = reader.coverage(shot, spec.group)
            ends.append(hi / 1000)
        except Unavailable:
            continue
    frames = int(np.floor(max(ends, default=0) / act.FRAME_S + 1e-7))
    if frames < 21:
        raise ValueError(f"Reference shot {shot} has insufficient actuator history")
    controls, available = _read_controls(shot, identity, frames)
    digest = _source_digest(identity)
    return Reference(None, controls, available, digest, digest, error, missing)


def reference(shot: int, paths: Paths) -> Reference:
    source = _corpus(paths) / f"{shot}_processed.h5"
    if not source.is_file():
        raise ValueError(f"Reference shot {shot} is missing its processed corpus file")
    try:
        identity = file_identity(source)
        cache = _cache_path(shot, paths)
        if cache is not None:
            try:
                return _read_reference(shot, identity, file_identity(cache))
            except ValueError as exc:
                return _raw_reference(shot, identity, str(exc), False)
        return _raw_reference(
            shot, identity,
            f"Shot {shot} has no IGNITE seed cache yet. You can edit and save its "
            "waveforms; choose Prepare IGNITE input to enable simulation export.",
            True,
        )
    except (OSError, ShotFailed, IndexError, TypeError) as exc:
        raise ValueError(f"Cannot read reference shot {shot}: {exc}") from exc


_PREPARE_LOCK = Lock()


def prepare_reference(shot: int, paths: Paths) -> None:
    """Encode with existing production codecs; publish only a complete valid cache."""
    from .seed import encode_frame_codes

    if not _PREPARE_LOCK.acquire(blocking=False):
        raise ValueError("Another IGNITE input is being prepared. Try again shortly.")
    try:
        if _cache_path(shot, paths) is not None:
            return
        corpus = _corpus(paths)
        source = corpus / f"{shot}_processed.h5"
        identity = file_identity(source)
        root = paths.ignite_inputs_dir / "reference_cache"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=root) as temp:
            output = encode_frame_codes(
                shot, reader=CorpusReader(corpus), out_dir=Path(temp), paths=paths,
                device="cuda" if torch.cuda.is_available() else "cpu", workers=2,
            )
            validate_cache(torch.load(output, map_location="cpu", weights_only=True))
            if file_identity(source) != identity:
                raise ValueError("Reference source changed during preparation; retry")
            # Source-specific directory binds tokens to the file they encoded.
            # Publish cache and provenance together, without replacing a winner
            # from another process preparing this same source concurrently.
            try:
                os.rename(temp, root / _source_digest(identity))
            except OSError as exc:
                if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
                if not (root / _source_digest(identity) / f"{shot}.pt").is_file():
                    raise
    except (OSError, RuntimeError, ImportError) as exc:
        raise ValueError(f"Could not prepare IGNITE input for shot {shot}: {exc}") from exc
    finally:
        _PREPARE_LOCK.release()
