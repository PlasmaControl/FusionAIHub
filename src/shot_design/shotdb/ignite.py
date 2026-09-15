"""IGNITE embeddings from existing foundation-model processed HDF5 files.

Frozen Phase-A codecs encode 50 ms frames into pre-FSQ continuous features for retrieval.
Missing modalities stay NaN. frame_codes applies the production quantizer unchanged,
verified against the bundle's shipped frame-code cache. Codecs are imported from the
sibling tokamak_foundation_model.ignite package; weights remain in the configured bundle.

Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import json
import logging
import os
import time
import warnings
from dataclasses import dataclass
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
    return load_yaml("ignite_modalities.yaml")["model"]


def bundle_dir(paths: Paths) -> Path:
    """Where the Hugging Face bundle lives locally: <models_dir>/<model.local_name>."""
    return paths.models_dir / model_cfg()["local_name"]


def codec_manifest(ckpt_dir: Path) -> Path:
    return Path(ckpt_dir) / "codecs" / "MANIFEST.json"


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

    `ckpt_dir` is the Hugging Face bundle (`download_bundle`): `codecs/MANIFEST.json` names the
    14 modalities and their family, and each codec is `codecs/<modality>/codec_best.pt`, a
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
            f"no IGNITE bundle at {ckpt_dir}. Download it once with\n"
            f"    shot_design model --download\n"
            f"(repo {model_cfg()['repo_id']}, needs a Hugging Face token with access); the "
            f"location is paths.yaml:models_dir / ignite_modalities.yaml:model.local_name."
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
            f"`shot_design model --download`."
        )
    entries = json.loads(manifest.read_text())["modalities"]
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


def frame_codes(
    shot: int,
    codecs: dict[str, tuple[Any, Any, str]],
    paths: Paths,
    data_dir: Path | None = None,
    t0_start: float | None = None,
    max_frames: int | None = None,
    device: str | None = None,
    workers: int | None = None,
) -> dict[str, np.ndarray]:
    """{modality: (n_frames, n_tok) int64 flat FSQ code index} -- the Phase-B token contract.

    Not used by retrieval (see the module docstring for why the pre-FSQ features are), but it is
    the same loader as `encode_shot` with the quantiser applied, which makes it the check that the
    loader is right: the bundle ships the production codes for ten shots, and this must reproduce
    them exactly. It is also what a Phase-5 rollout seeds from.
    """
    import torch

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
            "repo_id": cfg["repo_id"],
            "revision": cfg["revision"],
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
