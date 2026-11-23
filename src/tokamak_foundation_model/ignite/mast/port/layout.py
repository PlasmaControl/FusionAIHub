"""Patch dev-peter's IGNITE trainer and evaluator to use a MAST frame-code cache.

All patches are applied by attribute assignment on the imported module objects
(``train_dynamics`` and ``eval_dynamics``) without modifying any existing file:
  - ``train_dynamics.cache_modality_specs`` -> reads ``_codec_manifest.json``
    in canonical manifest order.
  - ``train_dynamics.resolve_split`` -> reads ``splits.json`` (TokaMark train/val/test;
    test is never trained or validated on) and freezes ``out_dir/split_shots.json``.
  - ``train_dynamics.build_presence`` -> reads the exact ``present`` flags stored in
    each cached ``{shot}.pt`` (falling back to dev-peter's heuristic if absent).
  - ``train_dynamics._ACT_SPEC`` / ``_FRAME_S`` and ``eval_dynamics.FRAME_S`` /
    ``_global_actuator_stats`` -> populated from ``_codec_manifest.json`` so
    actuator widths and counterfactual actuator modes match MAST.
"""

from __future__ import annotations

import contextlib
import json
import os
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# Importing train_dynamics/eval_dynamics pulls in x_transformers (which still uses
# @torch.jit.script, deprecated as FutureWarning on torch 2.15). Scope the filter
# strictly to that third-party import so no warnings in our code are hidden.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r".*torch\.jit\.script.*is deprecated.*",
        category=FutureWarning,
    )
    from ... import eval_dynamics as ev
    from ... import train_dynamics as td
from ...dynamics_config import ModalitySpec

_ORIG = {
    "td_cache_modality_specs": td.cache_modality_specs,
    "td_resolve_split": td.resolve_split,
    "td_build_presence": td.build_presence,
    "td_act_spec": td._ACT_SPEC,
    "td_frame_s": td._FRAME_S,
    "ev_frame_s": ev.FRAME_S,
    "ev_global_actuator_stats": ev._global_actuator_stats,
}


def load_manifest(cache_dir: Path | str) -> dict:
    """Read ``<cache_dir>/_codec_manifest.json``."""
    path = Path(cache_dir) / "_codec_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"MAST codec manifest not found: {path}")
    return json.loads(path.read_text())


def modality_specs_from_manifest(cache_dir: Path | str) -> Tuple[ModalitySpec, ...]:
    """Return ``ModalitySpec`` tuple in canonical manifest order."""
    manifest = load_manifest(cache_dir)
    entries = manifest["modalities"]
    return tuple(
        ModalitySpec(
            name=str(name),
            family=str(entry["family"]),
            n_tok=int(entry["n_tok"]),
            codebook_size=int(entry["codebook_size"]),
        )
        for name, entry in entries.items()
    )


def actuator_spec_from_manifest(cache_dir: Path | str) -> Tuple[Tuple[str, int], ...]:
    """Return ``((actuator_name, n_channels), ...)`` from the MAST manifest."""
    act = load_manifest(cache_dir)["actuators"]
    return tuple((str(n), int(c)) for n, c in zip(act["names"], act["channels"]))


def _actuator_stats_from_manifest(cache_dir: Path | str) -> Tuple[torch.Tensor, torch.Tensor]:
    act = load_manifest(cache_dir)["actuators"]
    return (
        torch.as_tensor(act["mu"], dtype=torch.float32),
        torch.as_tensor(act["sd"], dtype=torch.float32),
    )


def resolve_mast_split(
    cache_dir,
    out_dir,
    val_n: int,
    split_seed: int = 0,
    train_cap: int = 0,
    test_n: int = 0,
    pin_val=(),
    pin_train=(),
    pin_both=(),
    write: bool = True,
    log=print,
) -> Tuple[List[str], List[str], List[str]]:
    """Use TokaMark's frozen ``splits.json`` from ``cache_dir``; test is never touched."""
    cache_path = Path(cache_dir)
    splits_path = cache_path / "splits.json"
    if not splits_path.exists():
        raise FileNotFoundError(f"MAST splits.json not found: {splits_path}")
    splits = json.loads(splits_path.read_text())
    cached = {p.stem for p in cache_path.glob("*.pt") if not p.stem.startswith("_")}

    train = [str(s) for s in splits["train"] if str(s) in cached]
    val = [str(s) for s in splits["val"] if str(s) in cached]
    test = [str(s) for s in splits["test"] if str(s) in cached]

    if set(test) & (set(train) | set(val)):
        raise ValueError("splits.json overlaps test with train or val")

    if write and out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        record = {
            "source": str(splits_path.resolve()),
            "cache_dir": str(cache_path.resolve()),
            "split_seed": split_seed,
            "val_n": len(val),
            "test_n": len(test),
            "pin_val": list(pin_val),
            "pin_train": list(pin_train),
            "pin_both": list(pin_both),
            "train": train,
            "val": val,
            "test": test,
        }
        target = out_path / "split_shots.json"
        tmp = target.with_suffix(f".json.tmp.{os.getpid()}")
        tmp.write_text(json.dumps(record, indent=1))
        os.replace(tmp, target)
        log(
            f"[mast-port] split loaded from {splits_path} -> {target} "
            f"(train {len(train)}/val {len(val)}/test {len(test)} HELD OUT)"
        )
    else:
        log(
            f"[mast-port] split loaded from {splits_path} "
            f"(train {len(train)}/val {len(val)}/test {len(test)} HELD OUT)"
        )
    if train_cap:
        train = train[: int(train_cap)]
    return train, val, test


def build_mast_presence(
    cache_dir, out_path=None, log=print
) -> Dict[str, Dict[str, bool]]:
    """Build exact presence map from ``entry['present']`` stored in MAST cache entries."""
    cache = Path(cache_dir)
    stems = sorted(p.stem for p in cache.glob("*.pt") if not p.stem.startswith("_"))
    presence: Dict[str, Dict[str, bool]] = {}
    fallback_needed = False
    for shot in stems:
        entry = torch.load(cache / f"{shot}.pt", map_location="cpu", weights_only=False)
        if "present" not in entry:
            fallback_needed = True
            break
        presence[shot] = {str(k): bool(v) for k, v in entry["present"].items()}
    if fallback_needed:
        return _ORIG["td_build_presence"](cache_dir, out_path=out_path, log=log)
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
        tmp.write_text(
            json.dumps({"cache_dir": str(cache), "presence": presence}, indent=0)
        )
        os.replace(tmp, out)
        log(f"[mast-port] presence -> {out} ({len(presence)} shots)")
    return presence


def patch_ignite(cache_dir: Path | str | None = None) -> None:
    """Assign MAST layout, split, presence, and actuator hooks onto dev-peter's modules."""
    td.cache_modality_specs = modality_specs_from_manifest
    td.resolve_split = resolve_mast_split
    td.build_presence = build_mast_presence
    if cache_dir is not None:
        manifest_path = Path(cache_dir) / "_codec_manifest.json"
        if manifest_path.exists():
            manifest = load_manifest(cache_dir)
            td._ACT_SPEC = actuator_spec_from_manifest(cache_dir)
            dt_frame = float(manifest.get("dt_frame", 0.005))
            td._FRAME_S = dt_frame
            ev.FRAME_S = dt_frame
            stats = _actuator_stats_from_manifest(cache_dir)
            ev._global_actuator_stats = lambda _s=stats: _s


def unpatch_ignite() -> None:
    """Restore dev-peter's original module attributes."""
    td.cache_modality_specs = _ORIG["td_cache_modality_specs"]
    td.resolve_split = _ORIG["td_resolve_split"]
    td.build_presence = _ORIG["td_build_presence"]
    td._ACT_SPEC = _ORIG["td_act_spec"]
    td._FRAME_S = _ORIG["td_frame_s"]
    ev.FRAME_S = _ORIG["ev_frame_s"]
    ev._global_actuator_stats = _ORIG["ev_global_actuator_stats"]


@contextlib.contextmanager
def patched_ignite(cache_dir: Path | str | None = None) -> Iterator[None]:
    """Context manager applying :func:`patch_ignite` and restoring on exit."""
    patch_ignite(cache_dir)
    try:
        yield
    finally:
        unpatch_ignite()
