"""IGNITE Phase-B trainer + smoke test: stream real shots -> FROZEN-codec-encode each modality
into frame codes + 70-ch actuators -> MaskGIT training loop with the scheduled-sampling ramp.

The frozen Phase-A codecs are read-only here (eval, no_grad): each modality's 50 ms window is
encoded to its FLAT FSQ index (``codec.quantizer.fsq(codec.encode(x))[1]``, in [0, codebook_size)),
which is exactly the token vocab the FrameTokenizer/MaskGIT consume. Frame *t* = chunk *t* of a
shot (all modalities share the 50 ms windowing), so a single-shot per-modality codec dataset
yields consecutive frames by index. Actuators are the 7 raw actuator signals (70 ch) averaged
over each frame window.

`smoke()` proves the whole path end-to-end on a few REAL shots (a handful of MaskGIT steps →
finite, decreasing loss → a short rollout of valid committed codes) BEFORE any real training run.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from . import spike
from . import train_codec as tc
from .dynamics_config import DynamicsConfig
from .maskgit import MaskGITDynamics

_M = Path("/lustre/orion/fus187/proj-shared/models")
# best-ckpt manifest for the FROZEN codec set (2026-07-27). family drives the loader + dataset.
FROZEN_CODEC_CKPTS: Dict[str, Tuple[str, str]] = {
    "ece": ("spectro", "eval_runs/ignite_d5_ece/codec_best.pt"),
    "bes": ("spectro", "eval_runs/ignite_d5_bes/codec_best.pt"),
    "mhr": ("spectro", "eval_runs/ignite_d5_mhr/codec_best.pt"),
    "co2": ("spectro", "eval_runs/ignite_co2_rawstd192/codec_best.pt"),
    "tangtv_lower": ("video", "eval_runs/ignite_d4_tangtv_lower/codec_best.pt"),  # 108 tok, best 2.07
    "tangtv_upper": ("video", "eval_runs/ignite_d4_tangtv_upper/codec_best.pt"),  # 108 tok, best 1.66
    "ts_core_density": ("slowts", "eval_runs/ignite_d5_ts_core_density/codec_best.pt"),
    "ts_core_temp": ("slowts", "eval_runs/ignite_d5_ts_core_temp/codec_best.pt"),
    "ts_tangential_density": ("slowts", "eval_runs/ignite_d5_ts_tangential_density/codec_best.pt"),
    "ts_tangential_temp": ("slowts", "eval_runs/ignite_d5_ts_tangential_temp/codec_best.pt"),
    "cer_ti": ("slowts", "eval_runs/ignite_d5_cer_ti/codec_best.pt"),
    "cer_rot": ("slowts", "eval_runs/ignite_d5_cer_rot/codec_best.pt"),
    "mse": ("slowts", "eval_runs/ignite_mse_ent5/codec_best.pt"),
    # fast-TS (4th family). PENDING: no codec_best.pt yet (collapsed, needs a fix like co2/mse);
    # smoke skips it until then, but it IS part of the frame layout.
    "filterscopes": ("fastts", "eval_runs/ignite_fastts_filterscopes/codec_best.pt"),
}

# RESERVED-SLOT modalities: part of the 1017-token frame layout, but whose codec is NOT yet frozen
# (pending a redesign). Their n_tok slots are filled with a CONSTANT placeholder code during
# precompute so the frame keeps its full width; when the real codec lands, swap placeholder->real
# codes WITHOUT changing frame dimensions and the model finetunes into them (no full retrain).
# filterscopes (fast-TS ELM envelope) collapsed to 1 code on real data — the random-init encoder
# under-spreads the standardized envelope near the FSQ centre (feats std 0.13 vs 0.84 on varied
# input; diagnosed 2026-07-27, NOT a config fix) -> reserved until the codec is redesigned.
PLACEHOLDER_MODALITIES: Tuple[str, ...] = ("filterscopes",)
_PLACEHOLDER_CODE = 0

# (H5 group key, n_channels) for the 7 actuator signals -> 70 ch. Keys are the ACTUAL hdf5_keys
# (SignalConfig name 'pin'->'pinj', 'tin'->'tinj'); the rest match. Some shots miss some -> zeros.
_ACT_SPEC = (("ech_power", 12), ("pinj", 8), ("beam_voltage", 8), ("tinj", 8),
             ("gas_flow", 11), ("gas_raw", 11), ("rmp", 12))


def _load_codec(family: str, path: Path):
    """Rebuild a FROZEN codec from its best-ckpt (eval, no grad). Reuses the codec class per family."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    if family == "spectro":
        from .codec import SpectroCodec
        codec = SpectroCodec(cfg)
    elif family == "video":
        from .video_codec import VideoCodec
        codec = VideoCodec(cfg)
    elif family == "slowts":
        from .slow_ts_codec import SlowTSCodec
        codec = SlowTSCodec(cfg)
    elif family == "fastts":
        from .fastts_codec import FastTSCodec
        codec = FastTSCodec(cfg)
    else:
        raise ValueError(f"unknown codec family {family!r}")
    codec.load_state_dict(ck["codec"])
    codec.eval()
    for p in codec.parameters():
        p.requires_grad_(False)
    return codec, cfg


def resolve_codec_path(name: str, repo: Path, tmpl: str = None):
    """-> repo-relative checkpoint path for ``name``, or None.

    ``tmpl`` (e.g. ``eval_runs/ignite_codec_{m}_v6/codec_best.pt`` or a pinned snapshot
    ``.../codecs/{m}/codec_best.pt``) is tried FIRST; the frozen manifest is the fallback.
    A PLACEHOLDER modality becomes REAL when a template checkpoint exists for it (e.g.
    filterscopes since its v6 gate-passing codec)."""
    if tmpl:
        rel = tmpl.format(m=name)
        if (repo / rel).exists():
            return rel
    if name in PLACEHOLDER_MODALITIES:
        return None
    rel = FROZEN_CODEC_CKPTS[name][1]
    return rel if (repo / rel).exists() else None


def load_frozen_codecs(names: List[str], repo: Path = None, tmpl: str = None):
    """{name: (codec, cfg, family)} for every name whose checkpoint resolves (see
    :func:`resolve_codec_path`); unresolved names are skipped (precompute emits them as
    placeholder constants; eval simply doesn't render them)."""
    repo = repo or Path.cwd()
    out = {}
    for n in names:
        rel = resolve_codec_path(n, repo, tmpl)
        if rel is None:
            continue
        fam = FROZEN_CODEC_CKPTS[n][0]
        out[n] = (*(_load_codec(fam, repo / rel)), fam)
    return out


@torch.no_grad()
def encode_flat(codec, x: torch.Tensor) -> torch.Tensor:
    """(B, ...) modality window -> (B, n_tok) FLAT FSQ code index in [0, codebook_size).

    Device-safe: moves the input to the codec's device (codecs may be on GPU for fast precompute)
    and returns the codes on CPU (the cache is CPU int16).
    """
    dev = next(codec.parameters()).device
    feats = codec.encode(x.to(dev))
    return codec.quantizer.fsq(feats)[1].long().cpu()


def _single_shot_dataset(name: str, fam: str, cfg, shot: str, data_dir, t0_start: float = 1.0):
    """``t0_start`` is the window origin in shot time (default 1.0 = the production warmup
    skip; 0.0 = include the ramp-up second, e.g. for a seed-from-shot-start eval)."""
    if fam == "spectro":
        return tc.CodecPairDataset(name, [shot], cfg, data_dir=data_dir,
                                   lengths_cache_path=None, t0_start=t0_start)
    if fam == "video":
        return tc.VideoCodecPairDataset(name, [shot], cfg, data_dir=data_dir,
                                        lengths_cache_path=None, t0_start=t0_start)
    if fam == "fastts":
        from .fastts_train import FastTSCodecPairDataset  # shots-first signature (only filterscopes)
        return FastTSCodecPairDataset([shot], cfg, data_dir=data_dir,
                                      lengths_cache_path=None, t0_start=t0_start)
    return tc.SlowTSCodecPairDataset(name, [shot], cfg, data_dir=data_dir,
                                     lengths_cache_path=None, t0_start=t0_start)


@torch.no_grad()
def build_frame_codes(shot: str, codecs: Dict, n_frames: int, data_dir) -> Dict[str, torch.Tensor]:
    """One shot -> {modality: (n_frames, n_tok)} flat codes over consecutive frames (chunks)."""
    out = {}
    for name, (codec, cfg, fam) in codecs.items():
        ds = _single_shot_dataset(name, fam, cfg, shot, data_dir)
        if len(ds) < n_frames:
            raise RuntimeError(f"{name}@{shot}: only {len(ds)} frames < {n_frames}")
        frames = []
        for t in range(n_frames):
            item = ds[t]
            x = (item[0] if isinstance(item, tuple) else item).unsqueeze(0)  # (1, ...) window
            frames.append(encode_flat(codec, x)[0])                          # (n_tok,)
        out[name] = torch.stack(frames, 0)                                    # (n_frames, n_tok)
    return out


def actuator_frames(shot: str, n_frames: int, data_dir, t0_start: float = 1.0) -> torch.Tensor:
    """(n_frames, 70): the 7 actuator signals averaged over each frame's 50 ms window.

    Uses the ACTUAL H5 group keys (SignalConfig hdf5_keys) and zero-fills any signal a shot lacks,
    so the actuator vector is always the fixed 70 channels regardless of per-shot availability.
    ``t0_start`` must match the code windows' origin (default 1.0 = production warmup skip).

    TIME BASE (FIXED 2026-08-11 — was a real misalignment, see below). Windows are resolved in
    SHOT time exactly as the diagnostics loader resolves them (``data_loader._load_signal_raw``):
    index ``round((t - xdata[0]) * fs)`` with ``fs = (n - 1) / (xdata[-1] - xdata[0])``. Both
    halves of that convention are load-bearing:
      * The actuator groups do NOT start at t=0, so the old ``int(t * fs)`` (index counted from
        the ARRAY START) read actuators from seconds BEFORE the frame's shot time. Measured over
        40 random shots: gas_flow/gas_raw start at -10.0 s, beam_voltage ~-6.3 s, rmp ~-1.04 s
        and SHOT-DEPENDENT (-0.91..-1.15, so not even a learnable constant lag), ech_power
        -0.1 s; only pinj/tinj start at 0.0. On the 219-frame production cache window of shot
        200729 that left ~83% of the actuator input VARIANCE decorrelated from the true
        trajectory (corr: rmp +0.24, gas_raw -0.11, gas_flow +0.02, beam_voltage -0.28), with
        gas read entirely from pre-shot idle.
      * ``xdata`` is float32, so ``1/median(diff(x))`` loses the sample step to cancellation and
        is off by up to +0.82% (gas) — a GROWING drift, ~1.8 frames by the end of an 11 s window.
        The span-based rate is exact (10000.00 Hz for every actuator group).
    Indices are CLAMPED into the record: a negative index would silently wrap to the array tail.
    """
    import h5py
    frame_s, warmup = 0.05, float(t0_start)
    cols = []
    with h5py.File(Path(data_dir) / f"{shot}_processed.h5", "r") as f:
        for key, nch in _ACT_SPEC:
            if key in f and "ydata" in f[key] and f[key]["ydata"].shape[-1] > 1:
                y = np.asarray(f[key]["ydata"], dtype=np.float64)      # (C, T_raw)
                x = np.asarray(f[key].get("xdata", np.arange(y.shape[-1]) / 1e4), dtype=np.float64)
                span = float(x[-1] - x[0]) if x.size > 1 else 0.0
                fs = (x.size - 1) / span if span > 0 else 10_000.0
                x0 = float(x[0]) if x.size else 0.0                   # record start in SHOT time
                t_raw = y.shape[-1]
                frames = []
                for t in range(n_frames):
                    t0 = warmup + t * frame_s
                    i0 = min(max(int(round((t0 - x0) * fs)), 0), t_raw)
                    i1 = min(max(int(round((t0 + frame_s - x0) * fs)), 0), t_raw)
                    seg = y[:, i0:i1]
                    frames.append(np.nan_to_num(seg).mean(axis=1) if seg.size else np.zeros(y.shape[0]))
                col = np.stack(frames, 0)                              # (n_frames, C_actual)
                if col.shape[1] != nch:                               # normalize to the fixed width
                    fixed = np.zeros((n_frames, nch))
                    fixed[:, : min(nch, col.shape[1])] = col[:, :nch]
                    col = fixed
            else:
                col = np.zeros((n_frames, nch))                       # missing actuator -> zeros
            cols.append(col)
    act = np.concatenate(cols, axis=1)                                # (n_frames, 70)
    act = (act - act.mean(0, keepdims=True)) / (act.std(0, keepdims=True) + 1e-6)
    return torch.tensor(act, dtype=torch.float32)


def build_batch(shots: List[str], codecs: Dict, n_frames: int, data_dir):
    """-> ({modality:(B,F,n_tok)}, actuators (B,F,70)) over B shots."""
    per_shot_codes, acts = [], []
    for s in shots:
        per_shot_codes.append(build_frame_codes(s, codecs, n_frames, data_dir))
        acts.append(actuator_frames(s, n_frames, data_dir))
    codes = {n: torch.stack([c[n] for c in per_shot_codes], 0) for n in per_shot_codes[0]}
    return codes, torch.stack(acts, 0)


class _ShotTimeout(Exception):
    """A single shot exceeded the per-shot encode budget (pathological H5 / redraw loop)."""


@torch.no_grad()
def precompute_frame_codes(shots: List[str], codecs: Dict, out_dir, data_dir,
                           placeholder_specs: Dict[str, int] = None,
                           shot_timeout_s: int = 900, t0_start: float = 1.0,
                           log=print) -> int:
    """Encode every frame of each shot ONCE with the frozen codecs -> per-shot code cache.

    The codecs are frozen, so the codes are fixed; pre-encoding removes the codec forward from the
    training hot loop (Phase-B then trains on cached int codes only). Writes {shot}.pt =
    {'codes': {modality: int16 (F, n_tok)}, 'actuators': float16 (F, 70), 'n_frames': F}. Skips a
    shot if any modality has too few frames. ``placeholder_specs`` (name -> n_tok) are RESERVED-SLOT
    modalities emitted as a constant code (no codec yet) so the frame keeps its full width. Returns
    the number of shots cached.
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    placeholder_specs = placeholder_specs or {}
    # Per-shot wall-time guard: ~56 of the first 3000 shots HANG during encode (measured
    # 2026-08-08/09 — 2 h with no file and no exception). SIGALRM turns a hang into a
    # logged skip so a full-dataset sweep cannot stall on one shot. Main-thread only.
    import signal

    def _on_alarm(_sig, _frm):
        raise _ShotTimeout(f"shot exceeded {shot_timeout_s}s encode budget")
    can_alarm = False
    try:
        signal.signal(signal.SIGALRM, _on_alarm)
        can_alarm = True
    except ValueError:
        log("[precompute] NOTE: not in main thread — per-shot timeout guard disabled")
    n_ok = 0
    for si, shot in enumerate(shots):
        p = out / f"{shot}.pt"
        if p.exists():
            n_ok += 1
            continue
        if can_alarm:
            signal.alarm(shot_timeout_s)
        try:
            # frames available = min over REAL modalities of their single-shot dataset length
            per = {}
            nmin = None
            for name, (codec, cfg, fam) in codecs.items():
                ds = _single_shot_dataset(name, fam, cfg, shot, data_dir, t0_start=t0_start)
                per[name] = (ds, codec)
                nmin = len(ds) if nmin is None else min(nmin, len(ds))
            if not nmin or nmin < 8:
                continue
            codes = {}
            for name, (ds, codec) in per.items():
                frames = []
                for t in range(nmin):
                    item = ds[t]
                    x = (item[0] if isinstance(item, tuple) else item).unsqueeze(0)
                    frames.append(encode_flat(codec, x)[0].to(torch.int32))
                codes[name] = torch.stack(frames, 0)                      # (nmin, n_tok)
            for pname, ptok in placeholder_specs.items():                 # reserved slots -> constant
                codes[pname] = torch.full((nmin, ptok), _PLACEHOLDER_CODE, dtype=torch.int32)
            # per-modality vocab sizes ride along so the trainer derives the layout from the
            # cache (v6+ codecs change n_tok/vocab vs the static FROZEN_MODALITIES table).
            vocabs = {name: int(codec.quantizer.fsq.codebook_size)
                      for name, (ds, codec) in per.items()}
            act = actuator_frames(shot, nmin, data_dir,
                                  t0_start=t0_start).to(torch.float16)  # (nmin, 70)
            tmp = out / f"{shot}.pt.tmp"
            torch.save({"codes": codes, "actuators": act, "n_frames": nmin,
                        "vocabs": vocabs}, tmp)
            tmp.replace(p)
            n_ok += 1
        except Exception as e:
            log(f"[precompute] skip {shot}: {type(e).__name__}: {e}")
        finally:
            if can_alarm:
                signal.alarm(0)
        if (si + 1) % 50 == 0:
            log(f"[precompute] {si + 1}/{len(shots)} shots, {n_ok} cached")
    return n_ok


def split_shots(cache_dir, val_n: int, split_seed: int = 0, train_cap: int = 0,
                test_n: int = 0, pin_val=()):
    """Deterministic (train, val, test) shot split over the cached pool.

    ``split_seed = 0`` keeps the legacy sorted split (val = the sorted tail — the probe-v1
    behavior). ``split_seed > 0`` SHUFFLES the sorted list with that seed before splitting,
    so the partitions are uniform random subsets of the pool (user convention 2026-08-09:
    randomly selected validation/test subsets; also avoids the sorted-first-N campaign bias).
    ``train_cap`` keeps only the first N train shots (the N-shots probe arms).

    ``test_n`` carves a THIRD partition that nothing in training or validation ever touches
    (production fractions 0.90/0.05/0.05, user 2026-08-09). ``pin_val`` forces named shots
    into val whatever the shuffle says — the standing example shots (200729) must never be
    trained on. Pinned shots COUNT toward ``val_n`` so the val size stays as requested.
    """
    shots = sorted(p.stem for p in Path(cache_dir).glob("*.pt"))
    if split_seed:
        import random as _random
        _random.Random(split_seed).shuffle(shots)
    # Pin first so a pinned shot can never be drawn into train or test.
    pinned = [s for s in dict.fromkeys(pin_val) if s in set(shots)]
    rest = [s for s in shots if s not in set(pinned)]
    # NOTE the explicit `if n else` guards throughout: `xs[:-0]` is [], not xs.
    test = rest[-test_n:] if test_n else []
    pool = rest[:-test_n] if test_n else rest
    take = max(0, val_n - len(pinned))
    val = pinned + (pool[-take:] if take else [])
    train = pool[:-take] if take else pool
    if train_cap:
        train = train[:train_cap]
    return train, val, test


def resolve_split(cache_dir, out_dir, val_n: int, split_seed: int = 0, train_cap: int = 0,
                  test_n: int = 0, pin_val=(), write: bool = True, log=print):
    """The split actually used by a run — computed ONCE, then frozen to ``out_dir``.

    A seeded shuffle is only stable for a FIXED pool. The production cache is built by a
    chained precompute, so a split recomputed while it grows would reassign shots between
    partitions between legs — silently leaking test shots into train. Persisting the lists
    on first use (and reloading them afterwards) makes the split immune to cache growth and
    gives the run an auditable record, which is what the 3-way convention requires.

    Returns (train, val, test). Reloaded lists are intersected with what is actually cached,
    so a run started mid-precompute trains on what exists without ever re-drawing.

    DDP: ``write`` must be True on EXACTLY ONE rank (job 5224307 died when all 128 ranks
    raced on a single shared ``.tmp`` — one rank's replace() moved it away and the next
    rank's failed with FileNotFoundError). Non-writing ranks recompute the split in memory;
    it is a pure function of (cache contents, val_n, split_seed, test_n, pin_val), so every
    rank derives byte-identical partitions without touching the filesystem.
    """
    import json
    import os as _os
    p = Path(out_dir) / "split_shots.json"
    if p.exists():
        d = json.loads(p.read_text())
        have = {q.stem for q in Path(cache_dir).glob("*.pt")}
        tr = [s for s in d["train"] if s in have]
        va = [s for s in d["val"] if s in have]
        te = [s for s in d["test"] if s in have]
        miss = len(d["train"]) - len(tr)
        log(f"[dynamics] split RELOADED from {p} (train {len(tr)}/val {len(va)}/test {len(te)}"
            + (f"; {miss} train shots not yet cached" if miss else "") + ")")
        if train_cap:
            tr = tr[:train_cap]
        return tr, va, te
    # Freeze the UNCAPPED partitions: train_cap is a per-run knob (the N-shots probe arms),
    # not a property of the split. Baking it in would leave a resumed run permanently
    # restricted to the capped pool. Apply it after persisting, exactly as the reload path does.
    tr, va, te = split_shots(cache_dir, val_n, split_seed=split_seed, train_cap=0,
                             test_n=test_n, pin_val=pin_val)
    if write:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        # PID-unique tmp so a stray concurrent writer can never clobber ours mid-flight.
        tmp = p.with_suffix(f".json.tmp.{_os.getpid()}")
        tmp.write_text(json.dumps({"split_seed": split_seed, "val_n": val_n, "test_n": test_n,
                                   "pin_val": list(pin_val), "cache_dir": str(cache_dir),
                                   "train": tr, "val": va, "test": te}, indent=1))
        tmp.replace(p)                       # atomic within a rank; only one rank does this
        log(f"[dynamics] split FROZEN -> {p} (train {len(tr)}/val {len(va)}/test {len(te)}, "
            f"seed={split_seed}, pinned={list(pin_val)})")
    else:
        log(f"[dynamics] split computed in memory (train {len(tr)}/val {len(va)}/"
            f"test {len(te)}); rank 0 persists it")
    if train_cap:
        tr = tr[:train_cap]
    return tr, va, te


class _EnvRank:
    """Plain env-var shard identity for the embarrassingly-parallel cache passes.

    NO torch.distributed: these passes write per-shot .pt files (tmp+replace) and ranks finish
    at very different times. An initialized NCCL group kills the run at teardown: with a
    trailing barrier, idle ranks hit the 600 s watchdog (5208340/41); WITHOUT it, ranks that
    exit break the survivors' communicators (DistBackendError, 5218815). Env sharding has
    neither failure mode.
    """

    def __init__(self):
        import os as _os
        self.rank = int(_os.environ.get("RANK", _os.environ.get("SLURM_PROCID", 0)))
        self.world_size = int(_os.environ.get("WORLD_SIZE", "1"))
        local = int(_os.environ.get("LOCAL_RANK", 0))
        idx = local if torch.cuda.device_count() > 1 else 0
        self.device = torch.device(f"cuda:{idx}" if torch.cuda.is_available() else "cpu")
        self.is_main = self.rank == 0
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)


def _legacy_actuator_frames(shot: str, n_frames: int, data_dir, t0_start: float = 1.0):
    """The PRE-2026-08-11 (misaligned) actuator windowing, kept ONLY to verify a cache entry
    before :func:`patch_cache_actuators` overwrites it.

    Reproduces the exact indexing the buggy :func:`actuator_frames` used — ``int(t * fs)`` from
    the array start with ``fs = 1/median(diff(xdata))`` — so the patch can PROVE each cached
    ``actuators`` block was produced by the known code path (and by the manifest's ``t0_start``)
    rather than assuming it. Do not use for anything else; it is wrong on purpose.
    """
    import h5py
    frame_s, warmup = 0.05, float(t0_start)
    cols = []
    with h5py.File(Path(data_dir) / f"{shot}_processed.h5", "r") as f:
        for key, nch in _ACT_SPEC:
            if key in f and "ydata" in f[key] and f[key]["ydata"].shape[-1] > 1:
                y = np.asarray(f[key]["ydata"], dtype=np.float64)
                x = np.asarray(f[key].get("xdata", np.arange(y.shape[-1]) / 1e4), dtype=np.float64)
                fs = 1.0 / np.median(np.diff(x)) if x.size > 1 else 10_000.0
                frames = []
                for t in range(n_frames):
                    t0 = warmup + t * frame_s
                    i0, i1 = int(t0 * fs), int((t0 + frame_s) * fs)
                    seg = y[:, i0:i1]
                    frames.append(np.nan_to_num(seg).mean(axis=1) if seg.size else np.zeros(y.shape[0]))
                col = np.stack(frames, 0)
                if col.shape[1] != nch:
                    fixed = np.zeros((n_frames, nch))
                    fixed[:, : min(nch, col.shape[1])] = col[:, :nch]
                    col = fixed
            else:
                col = np.zeros((n_frames, nch))
            cols.append(col)
    act = np.concatenate(cols, axis=1)
    act = (act - act.mean(0, keepdims=True)) / (act.std(0, keepdims=True) + 1e-6)
    return torch.tensor(act, dtype=torch.float32)


def patch_cache_actuators(shots: List[str], cache_dir, data_dir, t0_start: float,
                          backup_path=None, verify: bool = True, dry_run: bool = False,
                          log=print) -> Dict[str, int]:
    """Rewrite ONLY the ``actuators`` block of existing frame-code cache entries in place.

    The 2026-08-11 time-base fix in :func:`actuator_frames` changes the actuators but NOT the
    codes (diagnostics were always windowed on the correct absolute-time base), so the cache is
    repaired without any codec re-encode — an I/O pass, not a re-precompute.

    Safety contract:
      * ``verify`` recomputes the LEGACY (misaligned) actuators and requires the stored block to
        match them before overwriting. An entry that does not match was not produced by the code
        path this patch assumes, so it is SKIPPED rather than rewritten (counted as ``mismatch``).
        Tolerance is float16 resolution on the stored values.
      * ``backup_path`` receives {shot: old actuators} so the pass is reversible without
        duplicating the (much larger) codes.
      * Writes are tmp+replace, and ``codes`` / ``n_frames`` / ``vocabs`` are carried through
        untouched.
    """
    out = {"patched": 0, "mismatch": 0, "missing": 0, "error": 0}
    backup = {}
    for si, shot in enumerate(shots):
        p = Path(cache_dir) / f"{shot}.pt"
        if not p.exists():
            out["missing"] += 1
            continue
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
            n = int(d["n_frames"])
            old = d["actuators"]
            if verify:
                legacy = _legacy_actuator_frames(shot, n, data_dir, t0_start=t0_start)
                # stored as float16: compare at float16 resolution of the stored magnitudes
                tol = 1e-2 + 1e-3 * float(old.float().abs().max())
                dev = float((legacy - old.float()).abs().max())
                if dev > tol:
                    log(f"[patch_act] MISMATCH {shot}: stored actuators are not the legacy "
                        f"t0={t0_start} output (max dev {dev:.4g} > tol {tol:.4g}) — SKIPPED")
                    out["mismatch"] += 1
                    continue
            new = actuator_frames(shot, n, data_dir, t0_start=t0_start).to(old.dtype)
            if new.shape != old.shape:
                log(f"[patch_act] SHAPE {shot}: {tuple(old.shape)} -> {tuple(new.shape)} — SKIPPED")
                out["mismatch"] += 1
                continue
            backup[shot] = old.clone()
            if not dry_run:
                d["actuators"] = new
                tmp = p.with_suffix(".pt.tmp")
                torch.save(d, tmp)
                tmp.replace(p)
            out["patched"] += 1
        except Exception as e:                                   # noqa: BLE001 (per-shot isolation)
            log(f"[patch_act] skip {shot}: {type(e).__name__}: {e}")
            out["error"] += 1
        if (si + 1) % 200 == 0:
            log(f"[patch_act] {si + 1}/{len(shots)} shots, {out['patched']} patched")
    if backup_path is not None and backup and not dry_run:
        Path(backup_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"actuators": backup, "t0_start": t0_start,
                    "note": "pre-2026-08-11 misaligned actuators; restore to undo the time-base fix"},
                   backup_path)
    return out


def run_patch_actuators(cache_dir, data_dir, dry_run: bool = False, log=print) -> int:
    """Distributed driver for :func:`patch_cache_actuators`: shard the cache's shots across ranks.

    ``t0_start`` is read from the cache's own ``_codec_manifest.json`` (written by
    :func:`run_precompute`) — never guessed, since the wrong origin would silently write
    actuators for the wrong window.
    """
    import json
    ddp = _EnvRank()
    cdir = Path(cache_dir)
    man = cdir / "_codec_manifest.json"
    if not man.exists():
        raise SystemExit(f"--patch_actuators needs {man} to recover the cache's t0_start")
    with open(man) as f:
        t0_start = float(json.load(f)["t0_start"])
    shots = sorted(p.stem for p in cdir.glob("[0-9]*.pt"))
    shard = shots[ddp.rank::ddp.world_size]
    if ddp.is_main:
        log(f"[patch_act] {len(shots)} cached shots / {ddp.world_size} ranks "
            f"(~{len(shard)}/rank); t0_start={t0_start} cache={cdir} dry_run={dry_run}", flush=True)
    bak = cdir / "_actuators_prefix_backup" / f"rank{ddp.rank:03d}.pt"
    res = patch_cache_actuators(shard, cdir, data_dir, t0_start, backup_path=bak,
                                dry_run=dry_run, log=log)
    log(f"[patch_act rank {ddp.rank}] {res} (backup -> {bak})")
    return res["patched"]


def run_precompute(cache_dir, data_dir, max_shots: int = 0, codec_tmpl: str = None,
                   shot_sample: int = 0, shot_seed: int = 0, t0_start: float = 1.0,
                   shot_timeout_s: int = 900, log=print) -> int:
    """Distributed frame-code precompute: shard shots across ranks, each encodes its shard on GPU.

    Embarrassingly parallel (per-shot .pt files, tmp+replace) so no inter-rank comm beyond a final
    barrier. Real codecs are loaded once per rank and moved to the rank's device; modalities whose
    checkpoint does NOT resolve (see :func:`resolve_codec_path`; ``codec_tmpl`` is tried first)
    are emitted as a constant placeholder code so the frame keeps its full width. Rank 0 writes
    ``_codec_manifest.json`` (resolved checkpoint per modality) into the cache for provenance.
    """
    import json
    from .dynamics_config import FROZEN_MODALITIES

    ddp = _EnvRank()
    repo = Path.cwd()
    resolved = {m.name: resolve_codec_path(m.name, repo, codec_tmpl) for m in FROZEN_MODALITIES}
    real = [n for n, rel in resolved.items() if rel is not None]
    placeholder_specs = {m.name: m.n_tok for m in FROZEN_MODALITIES if resolved[m.name] is None}
    codecs = load_frozen_codecs(real, repo=repo, tmpl=codec_tmpl)
    if ddp.is_main:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(cache_dir) / "_codec_manifest.json", "w") as f:
            json.dump({"resolved": resolved, "codec_tmpl": codec_tmpl,
                   "t0_start": t0_start}, f, indent=2)
    for _n, (c, _cfg, _fam) in codecs.items():
        c.to(ddp.device)
    all_shots = sorted(spike.discover_shots(data_dir))
    if shot_sample:
        # seeded uniform random sample over the WHOLE dataset (all campaigns) — the
        # random-selection probe / production convention (user 2026-08-09).
        import random as _random
        all_shots = sorted(_random.Random(shot_seed).sample(
            all_shots, min(shot_sample, len(all_shots))))
    elif max_shots:
        all_shots = all_shots[:max_shots]           # sanity-run cap (0 = full dataset)
    shard = all_shots[ddp.rank::ddp.world_size]
    if ddp.is_main:
        log(f"[precompute] {len(all_shots)} shots / {ddp.world_size} ranks (~{len(shard)}/rank); "
            f"real={len(real)} placeholders={list(placeholder_specs)} cache={cache_dir}", flush=True)
    n = precompute_frame_codes(shard, codecs, cache_dir, data_dir,
                               placeholder_specs=placeholder_specs, t0_start=t0_start,
                               shot_timeout_s=shot_timeout_s,
                               log=log)
    log(f"[precompute rank {ddp.rank}] cached {n}/{len(shard)} shots on this shard")
    # NO trailing dist.barrier(): shards are embarrassingly parallel and finish at very
    # different times (video-heavy shots) — idle ranks waiting in the barrier hit the NCCL
    # watchdog timeout (600 s) and the whole step gets killed mid-encode (measured on legs
    # 5208340/41, 2026-08-08). Ranks simply exit; srun ends when the slowest rank is done.
    return n


def build_presence(cache_dir, out_path=None, log=print) -> Dict[str, Dict[str, bool]]:
    """Per-(shot, modality) PRESENCE mask, derived from the null codeword of each modality.

    A diagnostic that did not record feeds a constant input to its frozen codec, so it
    encodes to a byte-identical frame vector in every shot where it is missing. Measured on
    the production cache, the frequently-absent modalities are the SMALL ones — and because
    ``training_loss`` averages per-modality means (each modality weighted EQUALLY regardless
    of token count), roughly 38% of the loss terms are computed on those constants.

    A shot's modality is ABSENT iff every frame equals that modality's null codeword. This is
    stricter than "the codes never change": a real diagnostic that happens to sit flat over a
    window is NOT null, because its constant value is not the shared null pattern.

    Single pass: each shot is loaded once. Returns {shot: {modality: True if PRESENT}}.
    """
    import json
    import os as _os
    from collections import Counter
    cache = Path(cache_dir)
    stems = sorted(p.stem for p in cache.glob("*.pt") if not p.stem.startswith("_"))
    frozen_hash: Dict[str, Counter] = {}
    per_shot: Dict[str, Dict[str, tuple]] = {}
    for s in stems:
        try:
            codes = torch.load(cache / f"{s}.pt", map_location="cpu")["codes"]
        except Exception as e:
            log(f"[presence] skip {s}: {type(e).__name__}: {e}")
            continue
        rec = {}
        for m, a in codes.items():
            is_frozen = bool((a == a[0:1]).all())
            h = hash(a[0].numpy().tobytes())
            rec[m] = (is_frozen, h)
            if is_frozen:
                frozen_hash.setdefault(m, Counter())[h] += 1
        per_shot[s] = rec
    # The null codeword per modality = the dominant pattern among frozen shots.
    null_of = {m: c.most_common(1)[0][0] for m, c in frozen_hash.items() if c}
    presence = {s: {m: not (fr and null_of.get(m) == h) for m, (fr, h) in rec.items()}
                for s, rec in per_shot.items()}
    if per_shot:
        mods = sorted(next(iter(per_shot.values())))
        log(f"[presence] {len(presence)} shots x {len(mods)} modalities")
        for m in mods:
            absent = sum(1 for s in presence if not presence[s][m])
            share = len(frozen_hash.get(m, {})) or 0
            log(f"[presence]   {m:<22} absent in {absent:5d}/{len(presence)} "
                f"({100*absent/max(1,len(presence)):4.1f}%)  distinct frozen patterns={share}")
        mean_absent = sum(1 for s in presence for m in mods if not presence[s][m]) \
            / max(1, len(presence) * len(mods))
        log(f"[presence] MEAN per-modality absence = {100*mean_absent:.1f}% "
            f"(= the share of loss TERMS that would score constants)")
    if out_path:
        tmp = Path(str(out_path) + f".tmp.{_os.getpid()}")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps({"cache_dir": str(cache), "presence": presence}, indent=0))
        tmp.replace(Path(out_path))
        log(f"[presence] -> {out_path}")
    return presence


class FrameCodeDataset(torch.utils.data.Dataset):
    """Streams (shot, start_frame) windows of length cfg.max_frames from the pre-encoded cache.

    Loads all cached shots into RAM once (int16 codes ~1-2 GB for the full dataset), so __getitem__
    is a cheap slice — no codec forward, no HDF5 read in the training loop.
    """

    def __init__(self, cache_dir, shots, cfg: DynamicsConfig, presence=None):
        self.cfg = cfg
        self.win = cfg.max_frames
        self.shots = []
        self.index = []                                     # (shot_i, start_frame)
        self.presence = []                                  # per shot: (n_modalities,) float mask
        names = {m.name for m in cfg.modalities}
        for s in shots:
            p = Path(cache_dir) / f"{s}.pt"
            if not p.exists():
                continue
            d = torch.load(p, map_location="cpu", weights_only=False)
            if not names.issubset(d["codes"]):              # cache must cover this frame layout
                continue
            self.shots.append(d)
            # 1.0 = this diagnostic recorded in this shot, 0.0 = absent (null codeword).
            # No presence map => everything present, i.e. the pre-masking behaviour.
            pr = (presence or {}).get(str(s), {})
            self.presence.append(torch.tensor(
                [1.0 if pr.get(m.name, True) else 0.0 for m in cfg.modalities]))
            si = len(self.shots) - 1
            for st in range(0, d["n_frames"] - self.win + 1):
                self.index.append((si, st))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        si, st = self.index[i]
        d = self.shots[si]
        codes = {m.name: d["codes"][m.name][st: st + self.win].long() for m in self.cfg.modalities}
        act = d["actuators"][st: st + self.win].float()
        return codes, act, self.presence[si]


def _collate_frames(batch):
    codes = {n: torch.stack([b[0][n] for b in batch], 0) for n in batch[0][0]}
    act = torch.stack([b[1] for b in batch], 0)
    present = torch.stack([b[2] for b in batch], 0)          # (B, n_modalities)
    return codes, act, present


def smoke(n_shots: int = 2, n_steps: int = 30, depth: int = 4, d_model: int = 128, seed: int = 0):
    """End-to-end proof on REAL frames: encode -> MaskGIT train (loss ↓) -> rollout (valid codes).

    Runs on CPU with a SMALL d_model — the goal is to prove the data+model pipeline, not d1024
    capacity (production trains d_model=1024 on GPU). Frozen codecs + frame layout are unchanged.
    """
    torch.manual_seed(seed)
    dd = spike.DEFAULT_DATA_DIR
    repo = Path(__file__).resolve().parents[3]
    from .dynamics_config import FROZEN_MODALITIES
    # use only modalities whose FROZEN ckpt exists (video ckpts pending path/format resolution);
    # the model is modality-agnostic, so this still proves the pipeline end-to-end.
    avail = tuple(m for m in FROZEN_MODALITIES if (repo / FROZEN_CODEC_CKPTS[m.name][1]).exists())
    skipped = [m.name for m in FROZEN_MODALITIES if m not in avail]
    if skipped:
        print(f"[smoke] NOTE skipping (no ckpt yet): {skipped}", flush=True)
    cfg = DynamicsConfig(modalities=avail, d_model=d_model, depth=depth, k0_seed=4, n_predict=3)
    names = [m.name for m in cfg.modalities]
    print(f"[smoke] loading {len(names)} frozen codecs ...", flush=True)
    codecs = load_frozen_codecs(names, repo=repo)
    n_frames = cfg.max_frames                                       # 4 + 3 = 7
    # pick co2-present shots so every modality (incl co2) has real data
    from .train_codec import _shot_paths
    from ..data.multi_file_dataset import filter_signal_present_files
    present = [p.name.split("_")[0] for p in filter_signal_present_files(
        _shot_paths(spike.discover_shots(dd), dd), "co2",
        cache_path=Path("/lustre/orion/fus187/proj-shared/foundation_model_meta/codec_co2_present.pt"))]
    shots = present[:n_shots]
    print(f"[smoke] building real frame codes for shots {shots} ({n_frames} frames) ...", flush=True)
    codes, act = build_batch(shots, codecs, n_frames, dd)
    print(f"[smoke] frame codes ok: co2 {tuple(codes['co2'].shape)} mse {tuple(codes['mse'].shape)} "
          f"actuators {tuple(act.shape)}", flush=True)

    model = MaskGITDynamics(cfg).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    gen = torch.Generator().manual_seed(seed)
    losses = []
    for step in range(n_steps):
        opt.zero_grad()
        loss = model.training_loss(codes, act, generator=gen, ss_frac=model.ss_fraction(step))
        loss.backward(); opt.step()
        losses.append(loss.item())
    print(f"[smoke] MaskGIT loss: start {losses[0]:.3f} -> end {losses[-1]:.3f} "
          f"(min {min(losses):.3f}) over {n_steps} steps", flush=True)

    model.eval()
    seed_codes = {n: codes[n][:, : cfg.k0_seed] for n in codes}
    traj = model.rollout(seed_codes, act, n_predict=cfg.n_predict, generator=gen)
    ok = all(int(v[:, cfg.k0_seed:].min()) >= 0 and
             int(v[:, cfg.k0_seed:].max()) < dict((m.name, m.codebook_size) for m in cfg.modalities)[n]
             for n, v in traj.items())
    print(f"[smoke] rollout -> {tuple(traj['co2'].shape)} per modality; all committed codes valid: {ok}",
          flush=True)
    print(f"[smoke] RESULT: loss decreased={losses[-1] < losses[0]}  rollout_valid={ok}", flush=True)
    return losses, ok


def _render_loss_curve(hist_path, png_path) -> None:
    """Render train/val masked-CE curves from ``loss_history.jsonl`` (overwritten in place).

    The jsonl is the durable loss record (survives 2 h chain legs; resume truncates entries
    past the restart step, so the curve is exact across a whole chain). Train points are the
    per-50-step logged losses (raw + EMA); val points are the fixed-batch masked CE."""
    import json as _json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr_s, tr_v, va_s, va_v = [], [], [], []
    with open(hist_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = _json.loads(line)
            if "val_loss" in r:
                va_s.append(r["step"]); va_v.append(r["val_loss"])
            elif "loss" in r:
                tr_s.append(r["step"]); tr_v.append(r["loss"])
    if not tr_s:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(tr_s, tr_v, color="C0", alpha=0.3, lw=0.8, label="train masked CE (per-log)")
    ema, sm = None, []
    for v in tr_v:
        ema = v if ema is None else 0.98 * ema + 0.02 * v
        sm.append(ema)
    ax.plot(tr_s, sm, color="C0", lw=1.6, label="train (EMA)")
    if va_s:
        ax.plot(va_s, va_v, "o-", color="C1", ms=3, lw=1.2, label="val masked CE")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("masked-token CE")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    plt.close(fig)


def cache_modality_specs(cache_dir):
    """Derive the frame layout from the cache itself: (ModalitySpec tuple, in canonical order).

    n_tok comes from each modality's code shape and vocab from the cache's ``vocabs`` entry
    (written at precompute since the v6 codecs; older caches fall back to the static
    FROZEN_MODALITIES codebook sizes). This keeps the trainer correct when the codec set
    changes the layout (ece 768 tok, fsq6 64k vocabs)."""
    from .dynamics_config import FROZEN_MODALITIES, ModalitySpec
    files = sorted(Path(cache_dir).glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"no cached shots under {cache_dir}")
    d = torch.load(files[0], map_location="cpu", weights_only=False)
    vocabs = d.get("vocabs", {})
    static = {m.name: m for m in FROZEN_MODALITIES}
    specs = []
    for m in FROZEN_MODALITIES:
        if m.name not in d["codes"]:
            continue
        specs.append(ModalitySpec(m.name, m.family, int(d["codes"][m.name].shape[1]),
                                  int(vocabs.get(m.name, static[m.name].codebook_size))))
    return tuple(specs)


def train(cache_dir, out_dir, steps: int = 200_000, batch_size: int = 8, lr: float = 3e-4,
          depth: int = 24, d_model: int = 1024, val_frac: float = 0.05, num_workers: int = 4,
          ckpt_every: int = 1000, ss_final_frac: float = None, n_heads: int = None,
          k0_seed: int = None, n_predict: int = None, train_cap: int = 0, val_n: int = 0,
          split_seed: int = 0, warmup_steps: int = 0, min_lr_ratio: float = 0.01,
          beta2: float = 0.999, weight_decay: float = 0.01, patience: int = 0,
          test_n: int = 0, test_frac: float = 0.0, pin_val=(),
          mask_absent: bool = False, presence_path: str = None,
          accum_steps: int = 1, log=print):
    """Production Phase-B training over the pre-encoded code cache (DDP, streaming, checkpointing).

    Reuses the codec trainer's DDP wrapper; streams FrameCodeDataset windows; MaskGIT loss with the
    scheduled-sampling ramp; cosine LR; resumes from + writes ``dynamics_latest.pt``.
    """
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler
    from .train_codec import _DDPState

    ddp = _DDPState()
    device = ddp.device
    # layout DERIVED from the cache (v6 codecs change n_tok/vocab vs the static table)
    specs = cache_modality_specs(cache_dir)
    cfg_kw = dict(modalities=specs, d_model=d_model, depth=depth)
    if n_heads:
        cfg_kw["n_heads"] = n_heads
    if k0_seed:
        cfg_kw["k0_seed"] = k0_seed
    if n_predict:
        cfg_kw["n_predict"] = n_predict
    cfg = DynamicsConfig(**cfg_kw)
    if ss_final_frac is not None:
        # scheduled sampling's own-code sampling calls backbone.forward = FULL (B,F,1017,vocab)
        # logits (~400 GB at F=100) — infeasible until that path is made memory-efficient. Set 0 to
        # disable for the base run (ss is rollout-drift mitigation; re-enable once the path is fixed).
        cfg.ss_ramp_final_frac = float(ss_final_frac)
    n_all = len(list(Path(cache_dir).glob("*.pt")))
    n_val = val_n if val_n else max(1, int(n_all * val_frac))
    n_test = test_n if test_n else (max(1, int(n_all * test_frac)) if test_frac else 0)
    # split_seed > 0 => seeded-random subsets (identical val across probe arms); 0 => legacy
    # sorted-tail split. Frozen to out_dir on first use so a growing cache cannot reshuffle
    # it mid-chain. See split_shots / resolve_split.
    train_shots, val_shots, test_shots = resolve_split(
        cache_dir, out_dir, n_val, split_seed=split_seed, train_cap=train_cap,
        test_n=n_test, pin_val=pin_val, write=ddp.is_main,
        log=(log if ddp.is_main else (lambda *a, **k: None)))
    # Presence mask (absent diagnostics excluded from the CE; see build_presence /
    # MaskGITDynamics.training_loss). Built once next to the cache and reused; mask_absent=0
    # keeps the historical behaviour so old runs stay reproducible.
    presence, use_presence = None, bool(mask_absent)
    if use_presence:
        import json as _json
        pth = Path(presence_path) if presence_path else Path(cache_dir) / "_presence.json"
        if not pth.exists():
            if ddp.is_main:
                log(f"[dynamics] no presence map at {pth} — building it once (single pass)")
                build_presence(cache_dir, out_path=pth, log=log)
            if ddp.world_size > 1:
                dist.barrier()                 # ranks wait for rank 0 to finish writing
        presence = _json.loads(pth.read_text())["presence"]
        if ddp.is_main:
            log(f"[dynamics] MASK_ABSENT on: presence map {pth} ({len(presence)} shots)")
    ds = FrameCodeDataset(cache_dir, train_shots, cfg, presence=presence)
    if ddp.is_main:
        log(f"[dynamics] cache={cache_dir} shots={n_all} split_seed={split_seed} "
            f"(train {len(train_shots)}/val {len(val_shots)}/test {len(test_shots)} HELD OUT) "
            f"windows={len(ds)} d_model={d_model} depth={depth} heads={cfg.n_heads} "
            f"frame_tokens={cfg.tokens_per_frame} win={cfg.max_frames}")
    if ddp.is_main:
        log(f"[dynamics] batch: micro={batch_size} x ranks={ddp.world_size} "
            f"x accum={accum_steps} -> EFFECTIVE {batch_size * ddp.world_size * accum_steps}"
            + (f"  ({accum_steps} fwd/bwd per optimizer step)" if accum_steps > 1 else ""))
        # Report the allocator config AS SEEN BY THE RANK PROCESS. The launcher's echo runs in
        # the batch step, NOT inside the srun tasks, so it cannot prove the ranks got it —
        # and job 5233441 OOM'd with 19.26 GiB reserved-but-unallocated (fragmentation) while
        # the submitting shell had expandable_segments set.
        import os as _o
        log(f"[dynamics] RANK-SEEN PYTORCH_ALLOC_CONF="
            f"{_o.environ.get('PYTORCH_ALLOC_CONF', '<UNSET>')} "
            f"(HIP={_o.environ.get('PYTORCH_HIP_ALLOC_CONF', '<unset>')})")
    sampler = DistributedSampler(ds, num_replicas=ddp.world_size, rank=ddp.rank, shuffle=True) \
        if ddp.world_size > 1 else None
    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, shuffle=(sampler is None),
                        num_workers=num_workers, collate_fn=_collate_frames, drop_last=True)

    model = MaskGITDynamics(cfg).to(device)
    latest = Path(out_dir) / "dynamics_latest.pt"
    start_step = 0
    ck = None
    if latest.exists():
        ck = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); start_step = int(ck.get("step", 0))
        if ddp.is_main:
            log(f"[dynamics] resumed from {latest} @ step {start_step}")
    # MANUAL gradient sync (no DDP wrapper). torch DDP's autograd hooks are fundamentally
    # incompatible with gradient checkpointing here: non-reentrant corrupts the recompute
    # (t()-on-4D / addmm), reentrant needs static_graph but the graph varies (scheduled sampling)
    # AND drops grads on out-of-checkpoint params (act_embed + all heads, per TORCH_DISTRIBUTED_DEBUG
    # =DETAIL). Checkpointing works perfectly BARE, so run the bare model and all-reduce grads by
    # hand after backward — identical math to DDP (average across ranks), ~100 ms/step for 433M
    # params on Slingshot. Also needs no loss-adapter / static_graph / find_unused gymnastics.
    def _sync_grads():
        if ddp.world_size <= 1:
            return
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= ddp.world_size
    # Optimizer / schedule. Defaults preserve the historical recipe (no warmup, wd 0.01,
    # beta2 0.999, cosine to 1% of peak) so existing runs are unchanged; the Genie-style
    # recipe is opt-in per argument (warmup_steps > 0, beta2 0.9, wd 1e-4, min_lr 10%).
    # The schedule is ONE LambdaLR whose factor is a pure function of the step index —
    # resume-safe by construction (no SequentialLR milestone state to desync across the
    # 2 h chain legs; that class of bug bit the FAITH chain).
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, beta2),
                            weight_decay=weight_decay)

    def _lr_factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / float(warmup_steps)          # linear warmup from ~0
        prog = (step - warmup_steps) / max(1, steps - warmup_steps)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(max(prog, 0.0), 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos     # cosine peak -> min_lr_ratio
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_factor)
    if ck is not None and "opt" in ck:
        # restore optimizer (Adam moments) + scheduler so a 2 h-job chain doesn't reset them each
        # resume; without this Adam m/v reset every ~700 steps -> unstable. See checkpoint-deadlock.
        opt.load_state_dict(ck["opt"])
        # The scheduler state may come from a DIFFERENT scheduler class than the one this
        # run builds (checkpoints written before the warmup+cosine LambdaLR carry
        # CosineAnnealingLR state, whose dict has no 'lr_lambdas' -> KeyError; that broke
        # every ngen2 resume on 2026-08-10). Our LR factor is a PURE FUNCTION of the step
        # index, so replaying start_step steps reproduces the exact schedule either way.
        loaded = False
        if "sched" in ck:
            try:
                sched.load_state_dict(ck["sched"])
                loaded = True
            except Exception as e:                       # incompatible scheduler state
                log(f"[dynamics] scheduler state incompatible ({type(e).__name__}: {e}); "
                    f"replaying {start_step} steps instead")
        if not loaded:
            for _ in range(start_step):
                sched.step()
    else:
        for _ in range(start_step):
            sched.step()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    import os
    if os.environ.get("IGNITE_DETECT_ANOMALY"):        # debug: pinpoint the offending forward op
        torch.autograd.set_detect_anomaly(True)

    import json
    # ---- loss curves: durable jsonl history + fixed-batch validation ------------------------ #
    hist_path = Path(out_dir) / "loss_history.jsonl"
    if ddp.is_main and start_step and hist_path.exists():
        # resume: drop entries past the restart step so chain-leg overlaps don't duplicate
        kept = [l for l in hist_path.read_text().splitlines()
                if l.strip() and json.loads(l).get("step", 0) <= start_step]
        hist_path.write_text("\n".join(kept) + ("\n" if kept else ""))
    # fixed validation batches: identical windows on every rank and every eval, masked with a
    # freshly re-seeded generator each time -> the val series is comparable across the run.
    val_ds = FrameCodeDataset(cache_dir, val_shots, cfg, presence=presence)
    val_batches = []
    if len(val_ds):
        vi = [int(i) for i in torch.linspace(0, len(val_ds) - 1,
                                             steps=min(4 * batch_size, len(val_ds))).tolist()]
        items = [val_ds[i] for i in vi]
        for s in range(0, len(items), batch_size):
            val_batches.append(_collate_frames(items[s:s + batch_size]))
    if ddp.is_main:
        log(f"[dynamics] loss history -> {hist_path} (+ loss_curve.png each ckpt); "
            f"val: {sum(b[1].shape[0] for b in val_batches)} fixed windows")

    @torch.no_grad()
    def _val_loss() -> float:
        if not val_batches:
            return float("nan")
        model.eval()
        g = torch.Generator(device=device)
        g.manual_seed(1234)                     # same masks every eval + every rank (lockstep)
        tot = 0.0
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            for cv, av, pv in val_batches:
                cd = {k: v.to(device) for k, v in cv.items()}
                tot += float(model.training_loss(cd, av.to(device), generator=g, ss_frac=0.0,
                                                 present=pv.to(device) if use_presence else None))
        model.train()
        return tot / len(val_batches)
    best_val, best_step, since_improve, stop_now = float("inf"), -1, 0, False
    _bp = Path(out_dir) / "dynamics_best.pt"
    if _bp.exists():
        try:                                        # chain-safe: keep the global best
            _b = torch.load(_bp, map_location="cpu", weights_only=False)
            best_val, best_step = float(_b.get("val_loss", float("inf"))), int(_b.get("step", -1))
            if ddp.is_main:
                log(f"[dynamics] resumed BEST val masked_ce={best_val:.4f} @ step {best_step}")
        except Exception as e:
            log(f"[dynamics] could not read {_bp} ({type(e).__name__}); best tracking restarts")
    gen = torch.Generator(device="cpu")
    step = start_step
    # Accumulation state. `step` counts OPTIMIZER steps, never micro-batches, so the LR
    # schedule, ckpt_every, val cadence and resume are all unaffected by accum_steps.
    micro, accum_loss = 0, 0.0
    model.train()
    while step < steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for codes, act, present in loader:
            codes = {k: v.to(device) for k, v in codes.items()}
            act = act.to(device)
            present = present.to(device) if use_presence else None
            if micro == 0:
                opt.zero_grad()                         # only at the START of an accumulation group
            ssf = model.ss_fraction(step)               # 0 for the whole run if ss_final_frac=0
            # bf16 autocast: ~2-3x faster + ~2x less activation memory on MI250X; bf16 has fp32
            # dynamic range so no GradScaler needed (unlike fp16). Frozen codes are int (unaffected).
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model.training_loss(codes, act, generator=None, ss_frac=ssf,
                                           present=present)
            # GRADIENT ACCUMULATION: effective batch = batch_size x world_size x accum_steps.
            # Scaling by 1/accum makes the summed grads equal the mean over the whole effective
            # batch, so the update is IDENTICAL to running the large batch in one go. Memory
            # stays at the micro-batch level: measured bs1 30.0 GiB allocated vs bs2 49.0 GiB,
            # and bs2 sat at 62.7/64 GiB reserved — unsurvivable (job 5233441).
            (loss / accum_steps).backward()
            accum_loss += float(loss.item())
            micro += 1
            if micro < accum_steps:
                continue                                # keep accumulating; no sync, no step
            micro = 0
            _sync_grads()                               # manual all-reduce (replaces DDP)
            opt.step(); sched.step()
            step += 1
            # validation runs on ALL ranks (identical fixed batches -> ranks stay in lockstep)
            vl = _val_loss() if step % ckpt_every == 0 else None
            lv = accum_loss / max(1, accum_steps)       # mean over the accumulation group
            accum_loss = 0.0
            if ddp.is_main and step % 50 == 0:
                # Memory telemetry: the RESERVED-minus-ALLOCATED gap is the fragmentation that
                # killed 5233441 (40.85 alloc / 19.26 reserved-unallocated of 64 GiB). With
                # expandable_segments working, reserved should track allocated closely.
                mem = ""
                if device.type == "cuda":
                    ga = torch.cuda.max_memory_allocated() / 2**30
                    gr = torch.cuda.max_memory_reserved() / 2**30
                    mem = f" mem_alloc={ga:.1f}G mem_resv={gr:.1f}G frag={gr - ga:.1f}G"
                log(f"[dynamics] step {step}/{steps} loss={lv:.4f} "
                    f"ss={ssf:.3f} lr={sched.get_last_lr()[0]:.2e}{mem}")
                with open(hist_path, "a") as f:
                    f.write(json.dumps({"step": step, "loss": lv, "ss": ssf,
                                        "lr": sched.get_last_lr()[0]}) + "\n")
            if ddp.is_main and vl is not None:
                log(f"[dynamics] step {step} VAL masked_ce={vl:.4f}")
                with open(hist_path, "a") as f:
                    f.write(json.dumps({"step": step, "val_loss": vl}) + "\n")
                _render_loss_curve(hist_path, Path(out_dir) / "loss_curve.png")
            if ddp.is_main and step % ckpt_every == 0:
                payload = {"model": model.state_dict(), "opt": opt.state_dict(),
                           "sched": sched.state_dict(), "step": step,
                           "cfg_depth": depth, "cfg_d_model": d_model,
                           "cfg_n_heads": cfg.n_heads, "cfg_k0": cfg.k0_seed,
                           "cfg_n_predict": cfg.n_predict,
                           "modalities": [(s.name, s.family, s.n_tok, s.codebook_size)
                                          for s in specs]}
                tmp = Path(out_dir) / "dynamics_latest.pt.tmp"
                torch.save(payload, tmp)
                tmp.replace(latest)
                # BEST-CHECKPOINT RETENTION. dynamics_latest.pt is a ROLLING file, so a run
                # that overfits destroys its own best weights (measured: the ngen2 N=100 arm
                # bottomed at val CE 1.77 @ step 4.6k and was at 2.63 by 8k — those weights
                # were unrecoverable). Keep a separate best-on-validation copy.
                if vl is not None and math.isfinite(vl) and vl < best_val - 1e-6:
                    best_val, best_step, since_improve = vl, step, 0
                    btmp = Path(out_dir) / "dynamics_best.pt.tmp"
                    torch.save({**payload, "val_loss": vl}, btmp)
                    btmp.replace(Path(out_dir) / "dynamics_best.pt")
                    with open(hist_path, "a") as f:
                        f.write(json.dumps({"step": step, "best_val_loss": vl}) + "\n")
                    log(f"[dynamics] new BEST val masked_ce={vl:.4f} @ step {step}")
                elif vl is not None and math.isfinite(vl):
                    since_improve += 1
                    if patience and since_improve >= patience:
                        log(f"[dynamics] EARLY STOP: {since_improve} evals without "
                            f"improvement (best {best_val:.4f} @ step {best_step}); "
                            f"stopping at step {step}")
                        stop_now = True
            if stop_now or step >= steps:
                break
        if stop_now:
            break
    if ddp.is_main:
        log(f"[dynamics] done @ step {step}"
            + (f" | best val masked_ce {best_val:.4f} @ step {best_step} "
               f"-> dynamics_best.pt" if math.isfinite(best_val) else ""))
    return step


def build_arg_parser():
    import argparse
    p = argparse.ArgumentParser("IGNITE Phase-B dynamics trainer")
    p.add_argument("--cache_dir", required=True, help="pre-encoded frame-code cache dir")
    p.add_argument("--out_dir", default=None, help="model output dir (required for training, not --precompute)")
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--depth", type=int, default=24)      # the depth-study knob
    p.add_argument("--d_model", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--ckpt_every", type=int, default=200,
                   help="checkpoint cadence (steps). MUST be < steps-per-job (~700 at 2h/g1) or a "
                        "chained run never checkpoints and every resume restarts from step 0.")
    p.add_argument("--ss_final_frac", type=float, default=None,
                   help="override DynamicsConfig.ss_ramp_final_frac; set 0 to DISABLE scheduled "
                        "sampling (its full-logits sampling OOMs at F=100 until made memory-efficient).")
    p.add_argument("--precompute", action="store_true",
                   help="run the distributed frame-code precompute (build the cache) then exit")
    p.add_argument("--data_dir", default=None,
                   help="H5 dir for --precompute (default: spike.DEFAULT_DATA_DIR)")
    p.add_argument("--max_shots", type=int, default=0,
                   help="cap total shots for a --precompute sanity run (0 = full dataset)")
    p.add_argument("--codec_tmpl", default=None,
                   help="codec ckpt template tried before the frozen manifest during "
                        "--precompute, e.g. 'path/codecs/{m}/codec_best.pt' (pinned snapshot)")
    p.add_argument("--n_heads", type=int, default=None, help="override DynamicsConfig.n_heads")
    p.add_argument("--k0_seed", type=int, default=None, help="override seed frames (window sizing)")
    p.add_argument("--n_predict", type=int, default=None,
                   help="override predicted frames (window = k0_seed + n_predict)")
    p.add_argument("--train_cap", type=int, default=0,
                   help="N-shots generalization probe: train on only the FIRST N train shots")
    p.add_argument("--val_n", type=int, default=0,
                   help="FIXED validation tail size (shots); 0 = val_frac fraction")
    p.add_argument("--split_seed", type=int, default=0,
                   help=">0: seeded-RANDOM train/val split over the cache (0 = sorted tail)")
    p.add_argument("--warmup_steps", type=int, default=0,
                   help="linear LR warmup steps (0 = none, the historical recipe)")
    p.add_argument("--min_lr_ratio", type=float, default=0.01,
                   help="cosine floor as a fraction of peak LR (Genie recipe: 0.1)")
    p.add_argument("--beta2", type=float, default=0.999,
                   help="AdamW beta2 (Genie recipe: 0.9)")
    p.add_argument("--weight_decay", type=float, default=0.01,
                   help="AdamW weight decay (Genie recipe: 1e-4)")
    p.add_argument("--patience", type=int, default=0,
                   help="EARLY STOP after this many validation evals with no improvement "
                        "(0 = never stop; dynamics_best.pt is written either way)")
    p.add_argument("--shot_sample", type=int, default=0,
                   help="--precompute: seeded uniform random sample of this many shots "
                        "from the WHOLE dataset (overrides --max_shots)")
    p.add_argument("--shot_seed", type=int, default=0, help="seed for --shot_sample")
    p.add_argument("--t0_start", type=float, default=1.0,
                   help="window origin in shot time [s] for --precompute. 0.0 = include the "
                        "ramp-up second so a K0=20 seed spans [0,1) s and PREDICTION STARTS "
                        "AT t=1.0 s (the standing convention). NOTE: the frozen codecs never "
                        "trained on ramp-up windows.")
    p.add_argument("--accum_steps", type=int, default=1,
                   help="gradient-accumulation micro-steps per optimizer step. Effective batch "
                        "= batch_size x world_size x accum_steps, at the MEMORY cost of "
                        "batch_size alone (measured: bs1 30.0 GiB vs bs2 49.0 GiB allocated; "
                        "bs2 reserved 62.7/64 GiB, which killed job 5233441). Throughput is "
                        "~unchanged: step time scales linearly with batch at this sequence "
                        "length (200 steps: bs1 13:57 vs bs2 26:42).")
    p.add_argument("--mask_absent", action="store_true",
                   help="exclude ABSENT diagnostics from the masked-CE (they encode to a "
                        "constant null codeword; ~38% of loss TERMS on the production cache, "
                        "since the loss weights every modality equally). Their tokens still "
                        "enter the model as input. Builds/loads <cache>/_presence.json.")
    p.add_argument("--presence_path", default=None,
                   help="override the presence-map location (default <cache_dir>/_presence.json)")
    p.add_argument("--build_presence", action="store_true",
                   help="build the presence map for --cache_dir and exit (no training)")
    p.add_argument("--test_n", type=int, default=0,
                   help="size of the HELD-OUT test partition in shots (overrides "
                        "--test_frac). Nothing in training or validation ever touches it.")
    p.add_argument("--test_frac", type=float, default=0.0,
                   help="test partition as a fraction of the cached pool; production "
                        "convention is 0.05 alongside --val_frac 0.05 (0.90/0.05/0.05)")
    p.add_argument("--pin_val", type=str, default="",
                   help="comma-separated shots FORCED into val whatever the shuffle says "
                        "(standing example shots, e.g. 200729, must never be trained on)")
    p.add_argument("--patch_actuators", action="store_true",
                   help="rewrite ONLY the 'actuators' block of an EXISTING frame-code cache "
                        "with the 2026-08-11 time-base fix (see actuator_frames). Codes are "
                        "untouched, so no codec re-encode. t0_start comes from the cache's own "
                        "_codec_manifest.json; each entry is verified against the legacy output "
                        "before being overwritten, and the old block is backed up under "
                        "<cache>/_actuators_prefix_backup/.")
    p.add_argument("--dry_run", action="store_true",
                   help="--patch_actuators: verify + report, write nothing")
    p.add_argument("--shot_timeout_s", type=int, default=900,
                   help="--precompute: per-shot SIGALRM encode budget [s]. A shot exceeding it "
                        "is logged and skipped so a full sweep cannot stall. Raise it for a "
                        "RECOVERY pass over shots the first sweep timed out on (precompute "
                        "skips shots already in the cache, so a re-run only retries the gaps).")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.build_presence:
        out = args.presence_path or (Path(args.cache_dir) / "_presence.json")
        build_presence(args.cache_dir, out_path=out)
        return 0
    if args.patch_actuators:
        return run_patch_actuators(args.cache_dir, args.data_dir or spike.DEFAULT_DATA_DIR,
                                   dry_run=args.dry_run)
    if args.precompute:
        return run_precompute(args.cache_dir, args.data_dir or spike.DEFAULT_DATA_DIR,
                              max_shots=args.max_shots, codec_tmpl=args.codec_tmpl,
                              shot_sample=args.shot_sample, shot_seed=args.shot_seed,
                              t0_start=args.t0_start, shot_timeout_s=args.shot_timeout_s)
    if not args.out_dir:
        raise SystemExit("--out_dir is required for training (omit only with --precompute)")
    return train(args.cache_dir, args.out_dir, steps=args.steps, batch_size=args.batch_size,
                 lr=args.lr, depth=args.depth, d_model=args.d_model, num_workers=args.num_workers,
                 ckpt_every=args.ckpt_every, ss_final_frac=args.ss_final_frac,
                 n_heads=args.n_heads, k0_seed=args.k0_seed, n_predict=args.n_predict,
                 train_cap=args.train_cap, val_n=args.val_n, split_seed=args.split_seed,
                 warmup_steps=args.warmup_steps, min_lr_ratio=args.min_lr_ratio,
                 beta2=args.beta2, weight_decay=args.weight_decay,
                 patience=args.patience, test_n=args.test_n, test_frac=args.test_frac,
                 pin_val=tuple(s.strip() for s in args.pin_val.split(",") if s.strip()),
                 mask_absent=args.mask_absent, presence_path=args.presence_path,
                 accum_steps=args.accum_steps)


if __name__ == "__main__":
    main()
