"""Dataset construction for e2e evaluation — trainer-identical semantics.

The train/val split is reproduced by calling the *trainer's own*
``resolve_shot_files`` (never a re-implementation), and the resulting val set
is asserted against the training-era lengths cache before any analysis runs.
Split leakage would silently invalidate every memorization / probe result, so
mismatches are a hard failure, not a warning.

Window grid (prediction mode): input window ``[t0, t0 + chunk)`` with
``t0 = warmup_s + idx * step_size_s`` seconds from shot start, target window
``[t0 + chunk, t0 + chunk + horizon)``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[3]
for _p in (str(_REPO / "src"), str(_REPO / "scripts" / "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tokamak_foundation_model.data.data_loader import (  # noqa: E402
    TokamakH5Dataset,
    collate_fn_prediction,
)
from tokamak_foundation_model.data.multi_file_dataset import (  # noqa: E402
    TokamakMultiFileDataset,
)

DEFAULT_DATA_DIR = Path("/lustre/orion/fus187/proj-shared/foundation_model")
DEFAULT_META_DIR = Path("/lustre/orion/fus187/proj-shared/foundation_model_meta")
DEFAULT_STATS_PATH = DEFAULT_META_DIR / "preprocessing_stats.pt"
# Written by the stage-1 training run itself — the ground truth for which
# shots were validation. The Jul-29 ``lengths_e2e_stage1_{train,val}.pt``
# caches in the same directory belong to a DIFFERENT split; never use them.
TRAINING_VAL_CACHE = DEFAULT_META_DIR / "lengths_eval_stage1_val.pt"

# Raw tangtv indices a 2-channel checkpoint was trained on, before the
# MovieConfig default grew to 7 channels (multi_file_dataset.py override doc).
_TANGTV_2CH_RAW = [4, 6]


def resolve_split(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    seed: int = 42,
    val_fraction: float = 0.1,
) -> Tuple[List[Path], List[Path]]:
    """Return ``(train_files, val_files)`` exactly as the stage-1 trainer did."""
    # Lazy: pulls in the full trainer module; label-only drivers skip the cost.
    import train_e2e_stage1 as _t1

    return _t1.resolve_shot_files(
        data_dir=Path(data_dir),
        train_shots_yaml=None,
        val_shots_yaml=None,
        max_files=None,
        val_fraction=val_fraction,
        seed=seed,
    )


def assert_val_matches_training_cache(
    val_files: Sequence[Path | str],
    cache_path: Path | str = TRAINING_VAL_CACHE,
) -> None:
    """Hard-fail unless val basenames equal the training-era cache's.

    The cached paths carry the Princeton prefix (``/scratch/gpfs/...``), so
    comparison is by basename only.
    """
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    cached = {Path(p).name for p in cache["paths"]}
    ours = {Path(p).name for p in val_files}
    if cached != ours:
        missing = sorted(cached - ours)
        extra = sorted(ours - cached)
        raise RuntimeError(
            f"val split does not match training-era cache {cache_path}: "
            f"{len(missing)} cached-only (e.g. {missing[:5]}), "
            f"{len(extra)} resolved-only (e.g. {extra[:5]}). "
            "Refusing to run — analyses would mix train shots into val."
        )


def matched_train_subset(train_files: Sequence[Path], n: int = 875) -> List[Path]:
    """Deterministic same-size train sample for train-vs-val comparisons.

    ``train_files`` from :func:`resolve_split` is already seed-42 shuffled, so
    the first *n* entries are an unbiased, reproducible sample.
    """
    return list(train_files[:n])


def load_stats(path: Path | str = DEFAULT_STATS_PATH) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def video_channels_override_for(diagnostics: Sequence[Any]) -> Optional[Dict[str, list]]:
    """Per-camera raw-channel reselection needed to feed this checkpoint.

    Compares each video ``DiagnosticConfig`` in the checkpoint against today's
    ``MOVIE_CONFIGS`` defaults. Channel-count drift with no known raw-index
    mapping is an error — silently feeding the wrong camera channels would
    corrupt every video metric.
    """
    movie_by_name = {mc.name: mc for mc in TokamakH5Dataset.MOVIE_CONFIGS}
    overrides: Dict[str, list] = {}
    for cfg in diagnostics:
        if getattr(cfg, "kind", None) != "video":
            continue
        mc = movie_by_name.get(cfg.name)
        if mc is None:
            raise RuntimeError(f"checkpoint video {cfg.name!r} not in MOVIE_CONFIGS")
        if mc.channels == cfg.n_channels:
            continue
        if cfg.name == "tangtv" and cfg.n_channels == 2:
            overrides[cfg.name] = list(_TANGTV_2CH_RAW)
        else:
            raise RuntimeError(
                f"video {cfg.name!r}: checkpoint has {cfg.n_channels} channels, "
                f"MovieConfig default has {mc.channels}, and no raw-index "
                "mapping is known for this combination."
            )
    return overrides or None


def make_prediction_dataset(
    files: Sequence[Path | str],
    stats: dict,
    ck_args: Dict[str, Any],
    diagnostics: Sequence[Any],
    actuators: Sequence[Any],
    *,
    step_size_s: Optional[float] = None,
    prediction_horizon_s: Optional[float] = None,
    lengths_cache_path: Optional[Path | str] = None,
    max_open_files: int = 256,
) -> TokamakMultiFileDataset:
    """Prediction-mode dataset mirroring the trainer's ``build_datasets``.

    ``step_size_s`` / ``prediction_horizon_s`` override the checkpoint values
    (e.g. 0.25 s stride for non-overlapping latent extraction, ``K * horizon``
    for rollout targets). ``lengths_cache_path`` must live under a writable
    directory (``data/outputs/...``) — the checkpoint dir is read-only.
    """
    input_signals = [c.name for c in diagnostics]
    target_signals = input_signals + [a.name for a in actuators]
    return TokamakMultiFileDataset(
        [Path(f) for f in files],
        chunk_duration_s=ck_args["chunk_duration_s"],
        prediction_mode=True,
        prediction_horizon_s=(
            ck_args["prediction_horizon_s"]
            if prediction_horizon_s is None
            else prediction_horizon_s
        ),
        step_size_s=(
            ck_args["step_size_s"] if step_size_s is None else step_size_s
        ),
        warmup_s=ck_args.get("warmup_s", 0.0),
        preprocessing_stats=stats,
        input_signals=input_signals,
        target_signals=target_signals,
        max_open_files=max_open_files,
        lengths_cache_path=lengths_cache_path,
        video_channels_override=video_channels_override_for(diagnostics),
    )


def make_eval_loader(
    dataset: TokamakMultiFileDataset,
    batch_size: int = 8,
    num_workers: int = 0,
) -> DataLoader:
    """Deterministic sequential loader (file-major order, no shuffling)."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_prediction,
        pin_memory=torch.cuda.is_available(),
    )


def shot_window_ranges(
    dataset: TokamakMultiFileDataset,
) -> List[Tuple[Path, int, int]]:
    """``(shot_path, global_start_idx, n_windows)`` per valid file, in order.

    Lets drivers map a global dataset index back to (shot, local window idx)
    without touching HDF5.
    """
    out: List[Tuple[Path, int, int]] = []
    start = 0
    for vi, n in zip(dataset._valid_indices, dataset._valid_lengths):
        out.append((dataset.hdf5_paths[vi], start, int(n)))
        start += int(n)
    return out


def window_start_s(local_idx: int, warmup_s: float, step_size_s: float) -> float:
    """Start time (s from shot start) of the input window at ``local_idx``."""
    return warmup_s + local_idx * step_size_s
