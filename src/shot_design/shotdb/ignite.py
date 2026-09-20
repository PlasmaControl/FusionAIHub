"""IGNITE embeddings from existing foundation-model processed HDF5 files.

Frozen Phase-A codecs encode 50 ms frames into pre-FSQ continuous features for retrieval.
Missing modalities stay NaN. frame_codes applies the production quantizer unchanged,
verified against the bundle's shipped frame-code cache. Codecs are imported from the
sibling tokamak_foundation_model.ignite package; weights remain in the configured bundle.

Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Paths, load_yaml
from . import legacy_raw

_log = logging.getLogger(__name__)

FRAME_MS = 50.0  # IGNITE's production frame duration
_open = legacy_raw.open_h5

ENCODABLE_FAMILIES = ("spectro", "slowts", "fastts")


class CheckpointMissing(RuntimeError):
    """Raised by load_codecs when the local checkpoint directory is absent or empty."""


def model_cfg() -> dict:
    """The pinned IGNITE generation: `model:` in configs/shot_design/ignite_modalities.yaml.

    Generation v2 was a Hugging Face snapshot (`repo_id` + `revision`, `download_bundle`);
    generation v4 is a local copy taken with `pin_bundle` from `codec_tmpl`/`dynamics_src` and
    verified by the sha256 table in its own manifest. `generation` says which, and it is the only
    key that decides: nothing here infers the generation from the shape of the table.
    """
    return load_yaml("ignite_modalities.yaml")["model"]


def bundle_dir(paths: Paths) -> Path:
    """Where the pinned bundle lives locally: <models_dir>/<model.local_name>.

    The name carries the generation (IGNITE_v4), so pinning a new generation lands beside the old
    one rather than on top of it and a database built against either can still find its weights.
    """
    return paths.models_dir / model_cfg()["local_name"]


def codec_manifest(ckpt_dir: Path) -> Path:
    return Path(ckpt_dir) / "codecs" / "MANIFEST.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pin_bundle(
    paths: Paths,
    *,
    codec_tmpl: str,
    dynamics_src: Path,
    names: list[str],
    t0_start: float,
) -> Path:
    """Copy one generation's codecs + dynamics checkpoint into <models_dir> and digest them.

    This is generation v4's replacement for `download_bundle`. The sources are training
    directories, not a published snapshot: `codec_tmpl.format(m=name)` is typically a SYMLINK into
    whichever run currently holds the best checkpoint, and those runs keep training. So the copy
    is a real copy of the RESOLVED file (`shutil.copy2` after `Path.resolve()`, never a symlink
    that would follow the run), and `codecs/MANIFEST.json` records a sha256 of every file written.
    `check_bundle` re-hashes them, which makes "the weights behind the database changed" a
    detectable event rather than a silent one.

    The manifest is the same shape the v2 Hub bundle shipped -- `modalities` maps each name to its
    family, n_tok and codebook_size, in canonical token order -- so `load_codecs` and
    `dynamics_config.modalities_from_manifest` read either generation unchanged.
    """
    cfg = model_cfg()
    families, n_tok, vocabs = cfg["families"], cfg["n_tok"], cfg["production_vocabs"]
    unknown = [n for n in names if n not in families or n not in n_tok or n not in vocabs]
    if unknown:
        raise KeyError(
            f"not modalities of generation {cfg.get('generation')}: {', '.join(unknown)}"
        )
    out = bundle_dir(paths)
    out.mkdir(parents=True, exist_ok=True)
    sha: dict[str, str] = {}
    for name in names:
        src = Path(codec_tmpl.format(m=name)).resolve()
        if not src.is_file():
            raise CheckpointMissing(f"no codec for {name} at {codec_tmpl.format(m=name)} -> {src}")
        rel = f"codecs/{name}/codec_best.pt"
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        sha[rel] = _sha256(dst)
    dyn_src = Path(dynamics_src).resolve()
    if not dyn_src.is_file():
        raise CheckpointMissing(f"no dynamics checkpoint at {dynamics_src} -> {dyn_src}")
    dyn_rel = cfg["dynamics_file"]
    shutil.copy2(dyn_src, out / dyn_rel)
    sha[dyn_rel] = _sha256(out / dyn_rel)
    manifest = {
        "_meta": {
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "generation": cfg.get("generation"),
            "codec_tmpl": codec_tmpl,
            "dynamics_src": str(dynamics_src),
            "copy_mode": "shutil.copy2, symlinks resolved",
        },
        "modalities": {
            name: {
                "family": families[name],
                "n_tok": int(n_tok[name]),
                "codebook_size": int(vocabs[name]),
            }
            for name in names
        },
        "t0_start_s": float(t0_start),
        "frame_tokens": sum(int(n_tok[name]) for name in names),
        "sha256": sha,
    }
    codec_manifest(out).write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def checkpoint_modalities(path: Path) -> list[tuple[str, str, int, int]] | None:
    """A dynamics checkpoint's OWN layout: [(name, family, n_tok, codebook_size)].

    Read with `mmap=True`, so the 3.3 GB of weights beside it are never
    materialised. None when the checkpoint carries no `modalities` entry
    (nothing to cross-check against).
    """
    import torch

    ck = torch.load(Path(path), map_location="cpu", weights_only=False, mmap=True)
    mods = ck.get("modalities") if isinstance(ck, dict) else None
    if not mods:
        return None
    out = []
    for m in mods:
        if isinstance(m, list | tuple):
            name, family, n_tok, vocab = m
        else:  # a ModalitySpec, as older checkpoints stored it
            name, family, n_tok, vocab = m.name, m.family, m.n_tok, m.codebook_size
        out.append((str(name), str(family), int(n_tok), int(vocab)))
    return out


def _check_dir(ckpt_dir: Path, *, codecs_only: bool = False) -> list[str]:
    """What no longer matches the bundle's own manifest, one line each ([] = intact).

    `codecs_only` restricts the digests to the `codecs/` entries: that is what
    `load_codecs` verifies, because it is what `load_codecs` reads. Hashing the
    3.3 GB dynamics checkpoint on a per-request path (design/seed.py loads
    codecs per prepare) costs seconds for a file that call never opens.
    `model --check` runs the full check, dynamics and cross-check included.
    """
    manifest = codec_manifest(ckpt_dir)
    if not manifest.exists():
        return [f"{manifest}: missing -- nothing is pinned here"]
    man = json.loads(manifest.read_text())
    bad: list[str] = []
    for rel, want in man.get("sha256", {}).items():
        if codecs_only and not rel.startswith("codecs/"):
            continue
        path = Path(ckpt_dir) / rel
        if not path.is_file():
            bad.append(f"{rel}: missing")
            continue
        got = _sha256(path)
        if got != want:
            bad.append(f"{rel}: expected {want[:12]} got {got[:12]}")
    # The vocabularies are the other thing that silently changes meaning: a codec generation with
    # different codebook sizes produces codes a checkpoint trained on this one cannot read.
    vocabs = model_cfg().get("production_vocabs", {})
    entries = man.get("modalities", {})
    for name, entry in entries.items():
        want_v = vocabs.get(name)
        if want_v is not None and int(entry["codebook_size"]) != int(want_v):
            bad.append(
                f"codecs/{name}: manifest vocab {entry['codebook_size']} != pinned "
                f"production vocab {want_v}"
            )
    # ... but that compares the manifest against the yaml it was WRITTEN from,
    # so on its own it is tautological right after a pin. The dynamics
    # checkpoint is the independent witness: it carries the layout it was
    # actually trained with, and `modalities_from_manifest` claims the manifest
    # IS that layout. Checked here, where the file is in hand, and not on the
    # codecs-only path -- it is the one check that has to open the checkpoint.
    dyn = Path(ckpt_dir) / model_cfg()["dynamics_file"]
    if not codecs_only and entries and dyn.is_file():
        mine = [
            (n, e["family"], int(e["n_tok"]), int(e["codebook_size"]))
            for n, e in entries.items()
        ]
        try:
            theirs = checkpoint_modalities(dyn)
        except Exception as e:  # noqa: BLE001 — an unreadable pin IS a finding
            bad.append(
                f"{dyn.name}: cannot read its modalities "
                f"({type(e).__name__}: {e})"
            )
            theirs = None
        if theirs is not None and theirs != mine:
            differ = [a for a, b in zip(mine, theirs, strict=False) if a != b]
            extra = {n for n, *_ in mine} ^ {n for n, *_ in theirs}
            bad.append(
                f"{dyn.name}: manifest modalities != the checkpoint's own "
                f"({len(mine)} vs {len(theirs)}; differing: "
                f"{', '.join(n for n, *_ in differ) or '-'}; only one side: "
                f"{', '.join(sorted(extra)) or '-'})"
            )
    return bad


def check_bundle(paths: Paths) -> list[str]:
    """`_check_dir` for the configured bundle: [] when every pinned file still hashes the same."""
    return _check_dir(bundle_dir(paths))


def bundle_identity(paths: Paths) -> dict:
    """What identifies the weights a database was built with: the generation
    and the digest of the bundle's codec manifest.

    v2 used the Hub `revision`; a pinned bundle has none, and its own sha256
    table cannot serve either -- re-pinning rewrites the codecs, the dynamics
    file AND the manifest together, so a re-pinned bundle passes its own check.
    What separates one pin from the next is the digest OF the manifest, which
    `design.provenance._bundle_identity` already computes for the encode
    sidecars; it is reused rather than recomputed so both records mean the same
    thing.
    """
    from ..design.provenance import _bundle_identity

    _, manifest_sha, revision = _bundle_identity(bundle_dir(paths))
    return {
        "generation": model_cfg().get("generation", "v2"),
        "revision": revision,
        "manifest_sha256": manifest_sha,
    }


def check_same_bundle(old_model: dict, paths: Paths) -> str | None:
    """Why the current bundle is not the one `old_model` records, or None when it is."""
    now = bundle_identity(paths)
    for key, label in (
        ("generation", "model generation"),
        ("revision", "model revision"),
        ("manifest_sha256", "codec manifest digest"),
    ):
        was = old_model.get(key, "v2" if key == "generation" else None)
        if was != now[key]:
            return f"{label} changed since the database was built ({was} -> {now[key]})"
    return None


def download_bundle(paths: Paths, full: bool = False, revision: str | None = None) -> Path:
    """Copy the model bundle from the Hub into <models_dir>, once.

    The Hub is the source of the copy, never a runtime dependency: everything else in this module
    reads the local directory only. By default the 3.5 GB dynamics checkpoint is left out -- the
    retrieval embedding needs the 14 codecs (453 MB) and `ignite_min`; `full=True` adds the
    dynamics model for the Phase-5 rollout. `snapshot_download` skips files already present, so
    re-running is a no-op check. The revision is pinned in configs/shot_design/ignite_modalities.yaml so
    a re-upload cannot silently change every embedding in the database.
    """
    from huggingface_hub import snapshot_download

    cfg = model_cfg()
    if cfg.get("repo_id") is None:
        raise RuntimeError(
            f"generation {cfg.get('generation')} is not published to the Hub: it is pinned "
            f"locally with `shot_design model --pin` (see pin_bundle)."
        )
    ignore = None if full else [cfg["dynamics_file"], "frame_codes/*"]
    out = snapshot_download(
        cfg["repo_id"],
        revision=revision or cfg["revision"],
        local_dir=bundle_dir(paths),
        ignore_patterns=ignore,
        max_workers=8,
    )
    return Path(out)


def _dynamics():
    """Import the sibling codecs with the Torch 2.14 compatibility warning scoped here."""
    # x_transformers uses @torch.jit.script at import time. Torch 2.14 deprecates
    # it but still executes it correctly; keep production codecs and codes intact.
    # This filter does not hide other warnings or alter the caller's filters.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`torch\.jit\.script` is deprecated\.",
            category=FutureWarning,
            module=r"torch\.jit\._script",
        )
        from tokamak_foundation_model.ignite import train_dynamics
    return train_dynamics


def load_codecs(
    ckpt_dir: Path, names: list[str] | None = None, device: str | None = None
) -> dict[str, tuple[Any, Any, str]]:
    """Load the frozen Phase-A codecs from the LOCAL bundle -> {name: (codec, cfg, family)}.

    `ckpt_dir` is the pinned bundle (`pin_bundle`, or `download_bundle` for v2):
    `codecs/MANIFEST.json` names the generation's modalities (v4: 15) and their family, plus a
    sha256 per copied file, and each codec is `codecs/<modality>/codec_best.pt`, a
    torch.save dict carrying `cfg` (the codec's own config dataclass, with the per-channel
    standardisation statistics the trainer injected) and `codec` (the state dict). The dataclass
    unpickles against the sibling `tokamak_foundation_model.ignite.config` package,
    which is byte-identical to the bundle's `ignite_min/`.

    By default only the encodable families load (spectro, slowts, fastts): the video codecs are in
    the bundle but no store here has camera frames for them to consume.
    """
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        raise CheckpointMissing(
            f"no IGNITE bundle at {ckpt_dir}. Install it once with\n"
            f"    shot_design model --pin        (generation v4: copies the local checkpoints)\n"
            f"    shot_design model --download   (generation v2: Hugging Face snapshot)\n"
            f"whichever `generation` ignite_modalities.yaml pins; the location is "
            f"paths.yaml:models_dir / ignite_modalities.yaml:model.local_name."
        )
    # Look for the weights BEFORE importing FusionAIHub. The common case by far is "the bundle
    # has not arrived yet", and that has to report a missing checkpoint -- not whatever import
    # error an unrelated submodule happens to raise, which `shot_design build` would classify as a
    # failed encode rather than an absent model.
    manifest = codec_manifest(ckpt_dir)
    if not manifest.exists():
        raise CheckpointMissing(
            f"{ckpt_dir} exists but holds no codec checkpoints: expected "
            f"codecs/MANIFEST.json and codecs/<modality>/codec_best.pt. Re-run "
            f"`shot_design model --pin` (or `--download` for generation v2)."
        )
    man = json.loads(manifest.read_text())
    entries = man["modalities"]
    # A pinned bundle carries digests of what was copied. Check them BEFORE loading anything: the
    # sources are live training directories, so "the codec under this path is no longer the one
    # the database was built with" is a real event, and it has to stop the run rather than quietly
    # re-embed a corpus against different weights.
    if man.get("sha256"):
        bad = _check_dir(ckpt_dir, codecs_only=True)
        if bad:
            raise CheckpointMissing(
                "pinned bundle changed on disk: " + "; ".join(bad) + f" (in {ckpt_dir}). "
                "Re-pin with `shot_design model --pin`, and rebuild anything embedded with it."
            )
    td = _dynamics()

    wanted = names or [n for n, e in entries.items() if e["family"] in ENCODABLE_FAMILIES]
    codecs: dict[str, tuple[Any, Any, str]] = {}
    missing: list[str] = []
    for name in wanted:
        entry = entries.get(name)
        path = ckpt_dir / "codecs" / name / "codec_best.pt"
        if entry is None or not path.exists():
            missing.append(name)
            continue
        codec, cfg = td._load_codec(entry["family"], path)
        if device is not None:
            codec = codec.to(device)
        codecs[name] = (codec, cfg, entry["family"])
    if not codecs:
        raise CheckpointMissing(
            f"{ckpt_dir} holds a codec manifest but none of the {len(wanted)} requested "
            f"modalities has a checkpoint; missing: {', '.join(missing) or 'all'}."
        )
    if missing:
        _log.info("codecs not in %s (skipped): %s", ckpt_dir, ", ".join(missing))
    return codecs


@dataclass
class ShotEmbedding:
    """Per-frame continuous features for one shot, plus which modalities were real."""

    shot: int
    modalities: tuple[str, ...]
    dims: tuple[int, ...]  # d_model per modality, in the same order
    features: np.ndarray  # (n_frames, sum(dims)) float32, NaN where the modality was absent
    mask: np.ndarray  # (n_frames, n_modalities) bool -- False where the modality was absent
    t0: float  # seconds; frame f covers [t0 + f*0.05, t0 + (f+1)*0.05)

    @property
    def frame_times(self) -> np.ndarray:
        return self.t0 + np.arange(self.features.shape[0]) * (FRAME_MS / 1000.0)

    @property
    def present(self) -> tuple[str, ...]:
        """The modalities that contributed at least one real frame."""
        return tuple(m for m, ok in zip(self.modalities, self.mask.any(axis=0), strict=True) if ok)

    def segment(self, t_start: float, t_end: float) -> np.ndarray:
        """The segment embedding: the mean over the frames whose start falls in [t_start, t_end).

        Mean over frames of the concatenated per-modality features, which keeps a segment
        comparable to any other regardless of how many frames it happens to contain. Frames whose
        modality was absent contribute NaN there, so the mean is nan-aware per column and a
        modality missing for the whole segment stays NaN rather than silently becoming zero.
        """
        t = self.frame_times
        sel = (t >= t_start) & (t < t_end)
        if not sel.any():
            return np.full(self.features.shape[1], np.nan, dtype=np.float32)
        return _nanmean_rows(self.features[sel])

    def windows(self, window_s: float) -> list[tuple[float, float, np.ndarray]]:
        """Fixed windows of `window_s` from t0: [(t_start, t_end, mean features), ...].

        These are what a fragment query (`--ref-window`) averages over, so they are cut on the
        same grid for every shot; a trailing partial window is kept when it holds at least one
        frame."""
        n_per = max(1, round(window_s / (FRAME_MS / 1000.0)))
        out = []
        for start in range(0, self.features.shape[0], n_per):
            block = self.features[start : start + n_per]
            t_start = self.t0 + start * (FRAME_MS / 1000.0)
            out.append(
                (t_start, t_start + block.shape[0] * (FRAME_MS / 1000.0), _nanmean_rows(block))
            )
        return out


def _nanmean_rows(vals: np.ndarray) -> np.ndarray:
    """Column means over the finite entries only; a column NaN in every row stays NaN.

    np.nanmean would do this, but warns "Mean of empty slice" for a column that is NaN in every
    row -- which is the ordinary case for a modality the shot does not have."""
    finite = np.isfinite(vals)
    n = finite.sum(axis=0)
    total = np.where(finite, vals, 0.0).sum(axis=0)
    return np.where(n > 0, total / np.maximum(n, 1), np.nan).astype(np.float32)


def filled_channels(processed: Path) -> dict[str, int]:
    """Per group of a processed file, how many channels hold any finite sample.

    This is the presence test the encoder trusts. The codec datasets do NOT refuse a placeholder
    group: measured on 185601, whose `bes` is the (64, 1) all-NaN placeholder, CodecPairDataset
    still yields 239 frames of a constant (std 0.0) spectrogram, which the codec would happily
    encode into a meaningless but finite feature. Absence has to be decided from the file.
    """
    with _open(processed) as f:
        return {
            name: int(
                sum(
                    1
                    for i in range(f[name]["ydata"].shape[0])
                    if np.isfinite(f[name]["ydata"][i]).any()
                )
            )
            for name in f
            if f[name]["xdata"].shape[0] > 1
        }


def _quiet(ds):
    # TokamakMultiFileDataset prints a "[w-pid…] prof_worker …" line every 50 __getitem__ calls;
    # there is no switch for it other than this counter.
    ds._prof_log_every = 1 << 62
    return ds


def _frames(
    shot: int,
    codecs: dict[str, tuple[Any, Any, str]],
    data_dir: Path,
    filled: dict[str, int],
    t0_start: float,
    max_frames: int | None,
    device: str,
    workers: int,
    batch_size: int,
    want_codes: bool,
) -> dict[str, np.ndarray]:
    """Run each codec over the shot's 50 ms frames -> {modality: (n_frames, d_model) features}
    (or (n_frames, n_tok) flat FSQ code indices when `want_codes`).

    The windowing is FusionAIHub's own dataset classes, not a reimplementation. That is the whole
    point: a spectro codec does not consume a waveform but a log-power spectrogram, and the STFT
    parameters, standardisation, channel selection and window origin that produced the model's
    training inputs live in those classes. Reproducing them by hand would be an unverifiable
    guess. Verified instead: with t0_start=0.0 this path reproduces the production frame-code
    cache shipped in the bundle BIT FOR BIT on shot 190090 -- all 239 frames of all 12 non-video
    modalities (tests/test_ignite.py keeps a slice of that check).

    Frames go through a DataLoader with CPU workers because the per-frame STFT of 500 kHz data is
    the cost, not the GPU: 239 co2 frames took 38 s single-process and 1.0 s with 8 workers, with
    identical codes.
    """
    import torch
    from torch.utils.data import DataLoader, Subset

    td = _dynamics()

    out: dict[str, np.ndarray] = {}
    for name, (codec, cfg, family) in codecs.items():
        if filled.get(name, 0) == 0:
            continue  # absent in the file: NaN, never the constant frames the dataset would give
        codec = codec.to(device)
        ds = _quiet(
            td._single_shot_dataset(name, family, cfg, str(shot), data_dir, t0_start=t0_start)
        )
        n = len(ds) if max_frames is None else min(len(ds), max_frames)
        if n == 0:
            continue
        loader = DataLoader(
            Subset(ds, range(n)), batch_size=batch_size, shuffle=False, num_workers=workers
        )
        chunks = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0] if isinstance(batch, list | tuple) else batch
                feats = codec.encode(x.to(device))  # (B, n_tok, d_model), pre-FSQ
                if want_codes:
                    chunks.append(codec.quantizer.fsq(feats)[1].long().cpu().numpy())
                else:
                    chunks.append(feats.mean(dim=1).float().cpu().numpy())
        out[name] = np.concatenate(chunks)[:n]
    return out


def _resolve_input(shot: int, paths: Paths, data_dir: Path | None) -> tuple[Path, dict[str, int]]:
    data_dir = Path(data_dir) if data_dir is not None else paths.foundation_model_processed_dir
    processed = data_dir / f"{shot}_processed.h5"
    if not processed.is_file():
        raise FileNotFoundError(f"no processed IGNITE input at {processed}")
    return data_dir, filled_channels(data_dir / f"{shot}_processed.h5")


def _default_workers() -> int:
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def encode_shot(
    shot: int,
    codecs: dict[str, tuple[Any, Any, str]],
    paths: Paths,
    data_dir: Path | None = None,
    t0_start: float | None = None,
    max_frames: int | None = None,
    device: str | None = None,
    workers: int | None = None,
    batch_size: int = 32,
    filled: dict[str, int] | None = None,
) -> ShotEmbedding:
    """Window `shot` into 50 ms frames and keep each codec's PRE-FSQ features per frame.

    Per frame and modality we store `codec.encode(x).mean(dim=tokens)` -- the mean over the
    modality's tokens of the continuous encoder features. `encode` is exactly the pre-FSQ tensor
    (SpectroCodec.encode's docstring calls it that, and forward() returns it as `feats` beside the
    quantised `quant` and the discrete `codes`), so this is the representation before the
    bottleneck throws away magnitude -- which is what a nearest-neighbour search needs.

    A modality the file does not hold is a NaN block with `mask` False, and modalities that stop
    at different times (each group spans its own support) are NaN-padded to the
    longest one rather than truncated to the shortest.
    """
    import torch

    cfg = model_cfg()
    t0_start = float(cfg["t0_start_s"]) if t0_start is None else t0_start
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if filled is None or data_dir is None:
        data_dir, filled = _resolve_input(shot, paths, data_dir)
    per_modality = _frames(
        shot,
        codecs,
        Path(data_dir),
        filled,
        t0_start,
        max_frames,
        device,
        _default_workers() if workers is None else workers,
        batch_size,
        want_codes=False,
    )
    names = tuple(codecs)
    dims = tuple(int(codecs[n][1].d_model) for n in names)
    n_frames = max([v.shape[0] for v in per_modality.values()] + [0])
    blocks, mask = [], []
    for name, d in zip(names, dims, strict=True):
        arr = np.full((n_frames, d), np.nan, dtype=np.float32)
        got = per_modality.get(name)
        if got is not None:
            arr[: got.shape[0]] = got
        blocks.append(arr)
        mask.append(np.isfinite(arr).any(axis=1))
    features = np.concatenate(blocks, axis=1) if blocks else np.zeros((n_frames, 0), np.float32)
    return ShotEmbedding(
        shot=shot,
        modalities=names,
        dims=dims,
        features=features,
        mask=np.stack(mask, axis=1) if mask else np.zeros((n_frames, 0), bool),
        t0=t0_start,
    )


def production_cache_path(shot: int) -> Path | None:
    """`model.frame_codes_cache`/<shot>.pt, when the pinned generation's own corpus holds it.

    Read-only: this directory belongs to the training runs, and nothing here ever writes into it.
    """
    root = model_cfg().get("frame_codes_cache")
    if not root:
        return None
    path = Path(root) / f"{int(shot)}.pt"
    return path if path.is_file() else None


def _cached_frame_codes(
    shot: int, codecs: dict[str, tuple[Any, Any, str]], max_frames: int | None
) -> dict[str, np.ndarray] | None:
    """Production's own codes for `shot`, or None when they cannot serve this request.

    A generation mismatch RAISES rather than falling back to an encode: a cache whose vocabs are
    not this checkpoint's is a pin that disagrees with itself, and re-encoding around it would
    hide exactly the event `validate_cache` and the bundle digest exist to surface. A cache that
    simply lacks a requested modality is not an error -- it is a narrower corpus -- and the
    encoder takes over, for ALL of the modalities: half-cached, half-encoded codes would mix two
    machines' arithmetic inside one frame.
    """
    path = production_cache_path(shot)
    if path is None:
        return None
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = model_cfg()["production_vocabs"]
    for name, vocab in payload.get("vocabs", {}).items():
        want = expected.get(name)
        if want is not None and int(vocab) != int(want):
            raise ValueError(
                f"{path}: {name} vocabulary {int(vocab)} is not the pinned production vocabulary "
                f"{int(want)} -- this cache was written by another codec generation"
            )
    cached = payload.get("codes", {})
    absent = [n for n in codecs if n not in cached]
    if absent:
        _log.info(
            "shot %s: %s is not in %s, encoding the whole frame instead",
            shot,
            ", ".join(absent),
            path.parent,
        )
        return None
    return {n: cached[n][:max_frames].to(torch.int64).numpy() for n in codecs}


def frame_codes(
    shot: int,
    codecs: dict[str, tuple[Any, Any, str]],
    paths: Paths,
    data_dir: Path | None = None,
    t0_start: float | None = None,
    max_frames: int | None = None,
    device: str | None = None,
    workers: int | None = None,
    use_cache: bool = True,
) -> dict[str, np.ndarray]:
    """{modality: (n_frames, n_tok) int64 flat FSQ code index} -- the Phase-B token contract.

    Not used by retrieval (see the module docstring for why the pre-FSQ features are), but it is
    the same loader as `encode_shot` with the quantiser applied, which makes it the check that the
    loader is right: the bundle ships the production codes for ten shots, and this must reproduce
    them exactly. It is also what a Phase-5 rollout seeds from.

    PRODUCTION'S CACHE FIRST. The pinned generation was trained on `model.frame_codes_cache`, and
    for a shot in there its file is returned unchanged: it is both cheaper (no 2-5 GB corpus read,
    no GPU) and more faithful than re-encoding, because our codes agree with production's bit for
    bit only on some shots -- changing a BLAS thread count moves a spectro token. `use_cache=False`
    forces the encode, and is what a parity gate has to pass: a gate comparing production's cache
    against a copy of production's cache would pass by construction.
    """
    import torch

    if use_cache:
        cached = _cached_frame_codes(shot, codecs, max_frames)
        if cached is not None:
            return cached
    t0_start = float(model_cfg()["t0_start_s"]) if t0_start is None else t0_start
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_dir, filled = _resolve_input(shot, paths, data_dir)
    return _frames(
        shot,
        codecs,
        data_dir,
        filled,
        t0_start,
        max_frames,
        device,
        _default_workers() if workers is None else workers,
        32,
        want_codes=True,
    )


def encode_records(
    records: list,
    codecs: dict[str, tuple[Any, Any, str]],
    paths: Paths,
    workers: int = 8,
    device: str | None = None,
    log=None,
) -> tuple[dict[int, ShotEmbedding], dict[int, str]]:
    """Encode every record's shot -> ({shot: ShotEmbedding}, {shot: error})."""
    shots = [r.shot for r in records]
    errors: dict[int, str] = {}
    embeddings: dict[int, ShotEmbedding] = {}
    for i, shot in enumerate(shots):
        try:
            embeddings[shot] = encode_shot(
                shot,
                codecs,
                paths,
                device=device,
                workers=workers,
            )
        except Exception as e:  # noqa: BLE001 — keep going; the manifest lists what failed and why
            errors[shot] = f"{type(e).__name__}: {e}"
            _log.warning("shot %s: encode failed (%s)", shot, errors[shot])
            continue
        if log is not None and (i + 1) % 10 == 0:
            log(f"  encoded {i + 1}/{len(shots)} shots")
    return embeddings, errors


def segment_matrix(
    embeddings: dict[int, ShotEmbedding], records: list, seg_ids: list[str], width: int
) -> np.ndarray:
    """(len(seg_ids), width) float32 in seg_ids order; NaN rows for shots without an embedding."""
    mat = np.full((len(seg_ids), width), np.nan, dtype=np.float32)
    pos = {sid: i for i, sid in enumerate(seg_ids)}
    for rec in records:
        emb = embeddings.get(rec.shot)
        if emb is None:
            continue
        for seg in rec.segments:
            i = pos.get(f"{rec.shot}:{seg.name}")
            if i is not None:
                mat[i] = emb.segment(seg.t0_ms / 1000.0, seg.t1_ms / 1000.0)
    return mat


def window_table(
    embeddings: dict[int, ShotEmbedding], window_s: float, width: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """windows.parquet rows + the matching (n_windows, width) float16 matrix, shots in order."""
    rows, vecs = [], []
    for shot in sorted(embeddings):
        for t_start, t_end, vec in embeddings[shot].windows(window_s):
            rows.append(
                {
                    "id": f"{shot}:{round(t_start * 1000)}",
                    "shot": shot,
                    "t0_ms": t_start * 1000.0,
                    "t1_ms": t_end * 1000.0,
                }
            )
            vecs.append(vec.astype(np.float16))
    df = pd.DataFrame(rows, columns=["id", "shot", "t0_ms", "t1_ms"]).set_index("id")
    mat = np.stack(vecs) if vecs else np.zeros((0, width), np.float16)
    return df, mat


def coverage_from_matrix(
    mat: np.ndarray, seg_shots: list[int], dims: list[int], modalities: list[str]
) -> tuple[dict[str, int], int]:
    """({modality: shots with >= 1 finite block}, shots with any finite cell), read off the final
    segment matrix so a carried-over row counts exactly like a freshly encoded one."""
    shots = np.asarray(seg_shots)
    coverage: dict[str, int] = {}
    off = 0
    any_finite = np.zeros(mat.shape[0], dtype=bool)
    for name, d in zip(modalities, dims, strict=True):
        ok = np.isfinite(mat[:, off : off + d]).any(axis=1)
        off += d
        coverage[name] = len(set(shots[ok].tolist()))
        any_finite |= ok
    return coverage, len(set(shots[any_finite].tolist()))


def manifest_block(
    codecs: dict[str, tuple[Any, Any, str]],
    coverage: dict[str, int],
    n_encoded: int,
    errors: dict[int, str],
    reason: str,
    elapsed_s: float,
    paths: Paths,
) -> dict:
    cfg = model_cfg()
    names = list(codecs)
    return {
        "status": "ok",
        "reason": reason,
        "channels": names,
        "modalities": names,
        "dims": [int(codecs[n][1].d_model) for n in names],
        "model": {
            # v2 identified the weights by a Hub revision; a pinned bundle is
            # identified by the digest of its codec manifest, which is what
            # `check_same_bundle` compares on an incremental add so that two
            # pins cannot end up in one matrix.
            **bundle_identity(paths),
            "repo_id": cfg.get("repo_id"),
            "bundle_dir": str(bundle_dir(paths)),
        },
        "t0_start_s": float(cfg["t0_start_s"]),
        "window_ms": float(cfg["window_ms"]),
        "pooling": "mean over tokens of the pre-FSQ codec features per 50 ms frame; "
        "mean over frames per segment / window; NaN where a modality is absent",
        "n_encoded": int(n_encoded),
        "modality_coverage": coverage,
        "failed": {str(k): v for k, v in sorted(errors.items())},
        "elapsed_s": round(elapsed_s, 1),
    }


def encode_db(
    tmp: Path, records: list, paths: Paths, workers: int = 8, device: str | None = None, log=None
) -> dict:
    """The hook `shotdb.build` calls: write emb_ignite_seg.npy (+ the window table) into `tmp`.

    Returns the manifest's `ignite` block and never raises for a missing model: a scalar-only
    database is fully usable and the block has to say plainly why the waveform channel is absent.
    Row order of `emb_ignite_seg.npy` is segments.parquet's, which is what every other `emb_*`
    matrix and `ShotDB.knn` rely on; a shot that fails to encode is a NaN row and a named entry in
    `failed`, not a shifted matrix.
    """
    t_start = time.perf_counter()
    try:
        codecs = load_codecs(bundle_dir(paths))
    except CheckpointMissing as e:
        return {
            "status": "not_installed",
            "reason": f"shot_design.shotdb.ignite has no checkpoint: {e}",
            "channels": [],
        }
    segments = pd.read_parquet(Path(tmp) / "segments.parquet", columns=["shot"])
    seg_ids = segments.index.tolist()
    dims = [int(codecs[n][1].d_model) for n in codecs]
    embeddings, errors = encode_records(
        records, codecs, paths, workers=workers, device=device, log=log
    )
    mat = segment_matrix(embeddings, records, seg_ids, sum(dims))
    np.save(Path(tmp) / "emb_ignite_seg.npy", mat)
    win_df, win_mat = window_table(embeddings, float(model_cfg()["window_ms"]) / 1000.0, sum(dims))
    win_df.to_parquet(Path(tmp) / "windows.parquet")
    np.save(Path(tmp) / "emb_ignite_win.npy", win_mat)
    coverage, n_encoded = coverage_from_matrix(mat, segments["shot"].tolist(), dims, list(codecs))
    return manifest_block(
        codecs,
        coverage,
        n_encoded,
        errors,
        f"{n_encoded}/{len(records)} shots encoded with {len(codecs)} codecs",
        time.perf_counter() - t_start,
        paths,
    )
