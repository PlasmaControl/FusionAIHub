"""One corpus shot -> the `frame_codes/<shot>.pt` the IGNITE dynamics checkpoint seeds from.

THE SHIPPED STRUCTURE. A frame-code cache is a plain `torch.save` dict with exactly four keys,
and `ignite_infer.validate_shot` refuses a rollout unless they line up with the checkpoint:

    codes      {modality: (F, n_tok) int32}  flat FSQ token indices, 14 modalities
                 ece / mhr / co2      spectro  192 tok   vocab 32768
                 bes                  spectro  192 tok   vocab 64000
                 tangtv_lower/_upper  video    108 tok   vocab 64000
                 ts_core_density, ts_core_temp, ts_tangential_density, ts_tangential_temp,
                 cer_ti, cer_rot, mse slowts     4 tok   vocab  1000
                 filterscopes         fastts     5 tok   vocab  1000
    actuators  (F, 88) float16  the control trajectory, ALREADY per-shot z-scored
    n_frames   int              F; 239 for a full shot
    vocabs     {modality: int}  codebook size per modality -- the field that tells a v2 cache
                                from a v3 one, which are otherwise indistinguishable and which
                                the model reinterprets silently

Nothing else. A fifth key would be dropped by nobody and read by nobody, so provenance for a
cache written here lives beside it in the run log and in the G-ENC gate's JSON, not inside the
file, and the file stays byte-comparable with the bundle's own.

WHY IT IS ENCODED THIS WAY. `shotdb.ignite.frame_codes` is the verified loader -- the STFT
parameters, channel selection and window origin that produced the model's training inputs live
in FusionAIHub's own dataset classes, and reproducing them by hand would be an unverifiable
guess. This module calls it and adds the 88-channel actuator block from `design.actuators`.

The one thing it cannot get from `frame_codes` is a modality the corpus file stores as an
all-NaN `(C, 1)` placeholder. `frame_codes` skips those deliberately: for a retrieval embedding,
a codec's output on a placeholder is a finite, meaningless number and NaN is the honest answer.
A frame-code cache is the opposite case -- production encoded the placeholder anyway, and the
checkpoint's modality table requires all 14 slots -- so the placeholders are topped up here
through the same loader with the presence test bypassed, which reproduces the shipped constant
codes exactly (bes = 51210, co2 = 312 on 190090). Dropping them instead would produce a cache
`validate_shot` rejects.

VERIFIED. `scripts/ideate/g_enc.py` compares freshly encoded shots with the caches shipped in
the bundle. Over all ten, on a V100S against production's MI250X: nine of the fourteen
modalities are bit-identical on every shot, `mhr` on 8/10, `co2` on 4/10 and `ece` on 3/10 (the
misses agree on >= 99.2 % of tokens), the two video modalities on 8/10 and 6/10, and the 88
actuator channels are bit-identical in float16 on 8/10. The two exceptions are 190735 and
190736, at 78/88 and 77/88 within 2e-3 z; on those two shots the actuator block is pure NumPy
arithmetic that disagrees by 1.8-2.4 z, which is the one place the disagreement cannot be this
module's codec arithmetic -- output disagreement, not a demonstrated input difference, which
would need matched input hashes nobody has. Everywhere else the residual is a scatter of isolated single tokens at
the quantiser's bin boundaries, consistent with a cross-vendor numerics difference and not
attributed to anything stronger -- no production input was ever compared. The gate's docstring
carries the measurements and the limits of what they support.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .. import config
from ..config import Paths
from ..shotdb import ignite
from . import actuators as act
from . import provenance as prov

_log = logging.getLogger(__name__)

#: The checkpoint's modality table, in the bundle's own order (`frame_codes/*.pt`'s `codes`).
MODALITIES: tuple[str, ...] = (
    "ece",
    "bes",
    "mhr",
    "co2",
    "tangtv_lower",
    "tangtv_upper",
    "ts_core_density",
    "ts_core_temp",
    "ts_tangential_density",
    "ts_tangential_temp",
    "cer_ti",
    "cer_rot",
    "mse",
    "filterscopes",
)
VIDEO_MODALITIES: tuple[str, ...] = ("tangtv_lower", "tangtv_upper")
#: `shotdb.ignite.frame_codes` hard-codes this; the placeholder top-up has to match it exactly.
_BATCH_SIZE = 32

#: The two `shotdb.ignite` symbols this module needs that carry a leading underscore, aliased
#: once here rather than reached for at every call site. `frame_codes` is the sanctioned entry
#: point and is what `encode_frame_codes` uses; the engine underneath it is needed for exactly
#: one thing (`_placeholder_codes`, see below), and naming that dependency in one place means a
#: refactor of `ignite` breaks an import at module load rather than a call deep in a 250-shot
#: SLURM task. `shotdb/ignite.py` is not modified by this module.
encode_frames = ignite._frames
default_workers = ignite._default_workers


def wanted_modalities(
    include_video: bool = True, modalities: Sequence[str] | None = None
) -> tuple[str, ...]:
    if modalities is not None:
        names = tuple(dict.fromkeys(modalities))
        unknown = [n for n in names if n not in MODALITIES]
        if unknown:
            raise ValueError(f"not IGNITE modalities: {', '.join(unknown)}")
    else:
        names = MODALITIES
    if not include_video:
        names = tuple(n for n in names if n not in VIDEO_MODALITIES)
    return names


def encode_frame_codes(
    shot: int,
    *,
    reader,
    device: str | None = None,
    include_video: bool = True,
    out_dir: Path,
    n_frames: int | None = None,
    modalities: Sequence[str] | None = None,
    codecs: dict[str, tuple[Any, Any, str]] | None = None,
    paths: Paths | None = None,
    workers: int | None = None,
    allow_partial: bool = False,
    run_manifest: Path | None = None,
) -> Path:
    """Encode `shot` into `<out_dir>/<shot>.pt` in the shipped layout; return that path.

    `reader` is a `shotdb.corpus.CorpusReader`: its `corpus_dir` is the directory of
    `<shot>_processed.h5` files the codecs stream from, and it is also what the actuator block
    reads its traces through. `n_frames` caps the cache (a full shot is 239 frames); without it
    every modality is encoded to the end of its own record and the cache is trimmed to the
    SHORTEST, which is production's rule and the only one that keeps the frame index meaning the
    same instant in every modality.

    A requested modality the bundle has no codec for RAISES. It used to be dropped silently --
    `codecs` was reduced to whatever the loader returned and the completeness check below then
    ran against that reduction, so the cache came out complete by its own definition, and G-ENC,
    whose only notion of "missing" was the same reduced dictionary, passed it. `allow_partial=True`
    is the diagnostic escape hatch: the run proceeds without the codec and the cache simply lacks
    that modality, which the gate then FAILS on because it was requested. It is never the gate.
    """
    import torch

    paths = paths or config.load_paths()
    names = wanted_modalities(include_video, modalities)
    # The codec set is settled BEFORE anything is read: a run that cannot answer what it was
    # asked for should say so in the first second, not after the first 3 GB HDF5 read.
    if codecs is None:
        codecs = ignite.load_codecs(ignite.bundle_dir(paths), names=list(names), device=device)
    absent = [n for n in names if n not in codecs]
    if absent and not allow_partial:
        raise ignite.CheckpointMissing(
            f"{shot}: no codec for requested modalities {', '.join(absent)} -- the loaded set is "
            f"{', '.join(sorted(codecs)) or '(empty)'}. Pass allow_partial=True for a diagnostic "
            f"run that skips them; such a run fails the G-ENC gate, which is the point."
        )
    if absent:
        _log.warning(
            "shot %s: PARTIAL encode, no codec for %s -- this cache is not a gate artifact",
            shot,
            ", ".join(absent),
        )
    codecs = {n: codecs[n] for n in names if n in codecs}
    if not codecs:
        raise ignite.CheckpointMissing(f"no codec for any of {', '.join(names)}")

    data_dir = Path(reader.corpus_dir)
    t0_start = float(ignite.model_cfg()["t0_start_s"])
    started = time.perf_counter()
    per = ignite.frame_codes(
        shot,
        codecs,
        paths,
        data_dir=data_dir,
        t0_start=t0_start,
        max_frames=n_frames,
        device=device,
        workers=workers,
    )
    per.update(
        _placeholder_codes(shot, codecs, per, data_dir, t0_start, n_frames, device, workers)
    )
    missing = [n for n in codecs if n not in per]
    if missing:
        raise RuntimeError(f"{shot}: no frames encoded for {', '.join(missing)}")

    frames = min(int(v.shape[0]) for v in per.values())
    if n_frames is not None:
        frames = min(frames, int(n_frames))
    if frames < 1:
        raise RuntimeError(f"{shot}: no encodable frames")

    trajectory = act.build_actuators(shot, reader, frames, t0_s=t0_start)
    payload = {
        "codes": {
            n: torch.from_numpy(np.ascontiguousarray(per[n][:frames])).to(torch.int32)
            for n in codecs
        },
        "actuators": torch.from_numpy(np.ascontiguousarray(trajectory.z.T)).to(torch.float16),
        "n_frames": int(frames),
        "vocabs": {n: int(codecs[n][0].quantizer.fsq.codebook_size) for n in codecs},
    }
    out = _save(payload, Path(out_dir), shot)
    # A fifth key in `payload` would break byte-comparability with the bundle's own caches, so
    # the provenance goes in a JSON sibling. Written AFTER the cache and never allowed to fail
    # the encode: a cache with no sidecar is a provenance gap, a lost cache is 40 s of GPU time.
    try:
        prov.write_sidecar(
            Path(out_dir),
            shot,
            prov.build_sidecar(
                shot,
                device=device or _default_device(),
                input_file=getattr(reader, "path", lambda s: None)(shot),
                bundle=ignite.bundle_dir(paths),
                n_frames=frames,
                modalities=list(codecs),
                include_video=include_video,
                run_manifest=run_manifest,
            ),
        )
    except Exception:  # noqa: BLE001 - provenance must not lose the artifact it describes
        _log.exception("shot %s: could not write the provenance sidecar", shot)
    _log.info(
        "shot %s: %d frames, %d modalities, %d absent actuator groups, %.1f s",
        shot,
        frames,
        len(codecs),
        len(trajectory.missing),
        time.perf_counter() - started,
    )
    return out


def _placeholder_codes(
    shot: int,
    codecs: dict[str, tuple[Any, Any, str]],
    have: dict[str, np.ndarray],
    data_dir: Path,
    t0_start: float,
    n_frames: int | None,
    device: str | None,
    workers: int | None,
) -> dict[str, np.ndarray]:
    """The modalities `frame_codes` skipped, encoded anyway -- see the module docstring.

    Same loader, same codecs, same window origin and the same batch size; only the "does this
    shot actually have the diagnostic" test is bypassed, by handing the encoder a presence map
    that says yes. The batch size is `frame_codes`'s own 32 and is deliberately not a parameter:
    it is not a free knob. Measured on 190090, encoding `tangtv_lower` in batches of 1, 2, 4 and
    32 gives four different sets of codes (0.2 % of tokens move), because cuDNN picks a different
    convolution algorithm per batch shape and the video codec's tokens sit on quantiser bin
    boundaries. Two shots encoded at different batch sizes would not be comparable.
    """
    todo = {n: c for n, c in codecs.items() if n not in have}
    if not todo:
        return {}
    _log.debug("shot %s: encoding placeholder modalities %s", shot, ", ".join(todo))
    return encode_frames(
        shot,
        todo,
        data_dir,
        dict.fromkeys(todo, 1),
        t0_start,
        n_frames,
        device or _default_device(),
        default_workers() if workers is None else workers,
        _BATCH_SIZE,
        want_codes=True,
    )


def _default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _save(payload: dict, out_dir: Path, shot: int) -> Path:
    """Write `<out_dir>/<shot>.pt` through a temporary file in the same directory.

    A frame-code cache is read by a different process (and by `--skip-existing` on the next
    encode run), so a half-written file at the final name is indistinguishable from a good one.
    `Path.replace` is atomic within a filesystem, which is why the temp file is a sibling.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"{shot}.pt"
    tmp = out_dir / f"{shot}.pt.tmp"
    import torch

    torch.save(payload, tmp)
    tmp.replace(final)
    return final


# ------------------------------------------------------------------ encoding a whole shot list


def chunk_of(shots: Sequence[int], chunk: int, n_chunks: int) -> list[int]:
    """Task `chunk` of `n_chunks`, cut CONTIGUOUSLY out of the sorted list.

    Contiguous rather than strided so that two array tasks reading the same corpus directory
    walk two different regions of it: the corpus files are 2-5 GB each and a strided split would
    have both tasks competing for the same GPFS blocks at the same time.
    """
    if not 1 <= n_chunks or not 0 <= chunk < n_chunks:
        raise ValueError(f"chunk {chunk} of {n_chunks} is not a valid split")
    shots = list(shots)
    lo = (len(shots) * chunk) // n_chunks
    hi = (len(shots) * (chunk + 1)) // n_chunks
    return shots[lo:hi]


def encode_many(
    shots: Sequence[int],
    *,
    reader,
    out_dir: Path,
    device: str | None = None,
    include_video: bool = True,
    skip_existing: bool = False,
    workers: int | None = None,
    paths: Paths | None = None,
    run_manifest: Path | None = None,
    log=print,
) -> dict:
    """Encode every shot into `out_dir`, one process, codecs loaded once.

    Loading the fourteen codecs costs ~5 s and 0.5 GB of GPU memory, so they are loaded once and
    reused for the whole list -- which is the entire reason this is a batch entry point rather
    than a loop over `encode_frame_codes` in a shell script. A shot that fails is logged and the
    run continues: a corpus file that will not open (2.3 % of them) is a fact about that shot,
    not a reason to lose the other 249.
    """
    paths = paths or config.load_paths()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = wanted_modalities(include_video)
    device = device or _default_device()
    codecs = ignite.load_codecs(ignite.bundle_dir(paths), names=list(names), device=device)

    started = time.perf_counter()
    done: dict[int, float] = {}
    skipped: list[int] = []
    failed: dict[int, str] = {}
    for i, shot in enumerate(shots, 1):
        if skip_existing and (out_dir / f"{shot}.pt").is_file():
            skipped.append(int(shot))
            continue
        t0 = time.perf_counter()
        try:
            encode_frame_codes(
                shot,
                reader=reader,
                device=device,
                include_video=include_video,
                out_dir=out_dir,
                codecs=codecs,
                paths=paths,
                workers=workers,
                run_manifest=run_manifest,
            )
        except Exception as e:  # noqa: BLE001 -- one bad shot must not end a 250-shot task
            failed[int(shot)] = f"{type(e).__name__}: {e}"
            log(f"[{i}/{len(shots)}] {shot} FAILED {failed[int(shot)]}")
            continue
        done[int(shot)] = round(time.perf_counter() - t0, 2)
        log(f"[{i}/{len(shots)}] {shot} {done[int(shot)]:.2f} s")
    elapsed = time.perf_counter() - started
    per_shot = sorted(done.values())
    return {
        "out_dir": str(out_dir),
        "device": device,
        "include_video": include_video,
        "n_requested": len(shots),
        "n_encoded": len(done),
        "n_skipped": len(skipped),
        "elapsed_s": round(elapsed, 1),
        "s_per_shot_mean": round(sum(per_shot) / len(per_shot), 2) if per_shot else None,
        "s_per_shot_median": round(per_shot[len(per_shot) // 2], 2) if per_shot else None,
        "s_per_shot_max": max(per_shot) if per_shot else None,
        "elapsed_by_shot": done,
        "failed": {str(k): v for k, v in sorted(failed.items())},
    }
