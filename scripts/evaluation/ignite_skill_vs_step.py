"""Rollout skill as a function of training step — the exposure-bias regression test.

The documented pathology is that rollout skill INVERTS with more training (band-power
line: TM-band skill +0.153 @ step 11k -> -0.862 @ step 20k) while teacher-forced CE
keeps improving. A fix for the train/test gap must flatten or reverse that curve, so
this harness plots skill against step for one or more checkpoint directories.

TWO metrics, because they do not agree and only one of them sees the pathology:

1. DECODED band-restricted nRMSE skill (PRIMARY) — ``1 - nrmse/nrmse_persistence`` on the
   band-power series obtained by dequantizing the codes and averaging over a frequency
   band, scored over the DATA-VALID predicted region. This is the metric the documented
   inversion lives in. Reported per (modality, band) and NEVER averaged across bands: over
   the same two bp128 steps mhr TM 1-20 kHz went +0.153 -> -0.862 while mhr AE 50-250 kHz
   went +0.175 -> +0.767, so any band-aggregated number cancels the failure out.
2. TOKEN accuracy skill (SECONDARY) — token accuracy over the predicted region minus the
   persistence baseline (fraction of tokens equal to the last seed frame). Persistence is
   mandatory: discrete codes at 50 ms are highly persistent, so raw accuracy is not
   interpretable. The MAJORITY-TOKEN baseline is reported too — a modality that cannot beat
   the frequency of its commonest ground-truth code is degenerate and its skill number
   means nothing.

MEASURED 2026-08-18 (bp128_d512L8, shot 199597, steps 10500 -> 20000): token skill is BLIND
to the inversion — mhr +0.249 -> +0.259, co2 +0.131 -> +0.130, flat or slightly BETTER, and
robust to the RNG seed (seed 0: +0.255 -> +0.280), to trimming the frames past the digitiser
record (+0.254 -> +0.319), and to using dynamics_best.pt @10500 as the stand-in for the
vanished step-11000 checkpoint (+0.249 vs +0.255). The failure is DISTRIBUTIONAL — variance
over-prediction inside one frequency band — so per-token argmax agreement survives it. Gate
on metric 1; read metric 2 only as a degeneracy check.

    python ignite_skill_vs_step.py --run_dir <dir> [--run_dir <dir2>] \
        --cache_dir <frame_codes> --shots 199597,190735 --out_dir <out>

Validate the decoded metric implementation against the archived 2026-08-15 rollouts:

    python ignite_skill_vs_step.py --from_traces data/outputs/ignite_bp_cases/tm_199597 \
        --from_traces data/outputs/ignite_bp_cases/tm_199597_step20000 --out_dir <out>
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402
import torch                                                             # noqa: E402

# Durable copy of the bp128 cache EXTENDED with 199597/199598/200730 (the canonical
# ignite_bp128/frame_codes was built before those were encoded and does NOT contain them).
# 600 of its 603 entries are absolute symlinks into the canonical cache. Rebuild recipe:
# scripts/evaluation/ignite_build_bp_cache.py — see PROVENANCE.txt in that directory.
DEFAULT_CACHE = "/lustre/orion/fus187/proj-shared/nchen/ignite_bp128_frame_codes_ext"
DEFAULT_BP_ROOT = "/lustre/orion/fus187/proj-shared/models/ignite_bp128"
DEFAULT_CODEC_ROOT = "/lustre/orion/fus187/proj-shared/models/ignite_codecs_current"
DEFAULT_DATA_DIR = "/lustre/orion/fus187/proj-shared/additional_data"

# ------------------------------------------------------------------------------------------- #
# band-power inverse + band-restricted skill
#
# LIFTED (deliberately copied, not imported) from scripts/evaluation/ignite_bp_cases.py
# (levels_to_value / band_slice / band_trace / valid_frames, whose own math is replicated from
# ignite_bandpower/bp_eval2.py) and scripts/evaluation/ignite_case_panels.py
# (decoded_space_skill). Those scripts are UNTRACKED user eval scripts; a committed harness
# that imported them would be broken for anyone else, so the minimal functions live here.
# Semantics are identical — verified by --from_traces reproducing +0.153 / -0.862 exactly.
# ------------------------------------------------------------------------------------------- #
FRAME_DT_S = 0.05
SPECTRO_FS_HZ, SPECTRO_N_FFT = 500e3, 1024
DF_HZ = SPECTRO_FS_HZ / SPECTRO_N_FFT      # 488.28 Hz per STFT bin (DC dropped)
N_LEV = 8                                  # band-power tokenizer levels (vocab 8)
BANDS_HZ = [("TM 1-20 kHz", 1e3, 20e3), ("AE 50-250 kHz", 50e3, 250e3)]


def decoded_space_skill(gt, pred, pers, K0) -> tuple:
    """(nrmse, nrmse_persistence, skill) over the PREDICTED region, in DECODED space.

    gt and pred both come from codes pushed through the SAME deterministic dequantization, so
    tokenizer error cancels and what remains is the DYNAMICS model's error. skill > 0 means
    the rollout beats freezing the last real frame.
    """
    g, p, q = gt[K0:], pred[K0:], pers[K0:]
    ok = np.isfinite(g) & np.isfinite(p) & np.isfinite(q)
    if ok.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    g, p, q = g[ok], p[ok], q[ok]
    den = g.std()
    if den < 1e-12:
        return float("nan"), float("nan"), float("nan")
    nr = float(np.sqrt(((p - g) ** 2).mean()) / den)
    npers = float(np.sqrt(((q - g) ** 2).mean()) / den)
    return nr, npers, float(1.0 - nr / npers) if npers > 0 else float("nan")


def load_edges(bp_root):
    """The TRAINED quantile bin edges. Using Peter's own file is mandatory: the levels are
    quantiles, meaningful only relative to the edges the model was trained against."""
    bp_root = Path(bp_root)
    p = next((f for f in (bp_root / "bin_edges_128.npz", bp_root / "bin_edges.npz")
              if f.exists()), None)
    if p is None:
        return None
    print(f"[skill] trained bin edges: {p}")
    return np.load(p)


def levels_to_value(edges, mod: str, tk: np.ndarray) -> np.ndarray:
    """Level index -> representative log-power. Interior bins take the midpoint of their two
    cut points; the two outer (unbounded) bins take the edge -/+ half the neighbouring width."""
    q = edges[mod]                                   # (N_LEV-1, D)
    D = q.shape[1]
    reps = np.zeros((N_LEV, D))
    reps[1:-1] = 0.5 * (q[:-1] + q[1:])
    w = np.maximum(q[1] - q[0], 1e-6)
    reps[0] = q[0] - 0.5 * w
    w = np.maximum(q[-1] - q[-2], 1e-6)
    reps[-1] = q[-1] + 0.5 * w
    return np.take_along_axis(reps, np.clip(tk, 0, N_LEV - 1), axis=0)


def band_slice(n_band: int, n_freq: int, f_lo: float, f_hi: float) -> tuple:
    """Which of the n_band equal-width frequency bands overlap [f_lo, f_hi]."""
    e = np.linspace(0, n_freq, n_band + 1).astype(int)
    lo_bin = max(0, int(round(f_lo / DF_HZ)) - 1)
    hi_bin = min(n_freq - 1, int(round(f_hi / DF_HZ)) - 1)
    idx = [i for i in range(n_band) if e[i + 1] > lo_bin and e[i] <= hi_bin]
    return (idx[0], idx[-1]) if idx else (0, 0)


def band_trace(edges, mod: str, codes: np.ndarray, n_ch: int, n_band: int,
               n_freq: int, f_lo: float, f_hi: float) -> np.ndarray:
    """(n_frames, C*n_band) levels -> per-frame mean log-power over the requested band."""
    vals = levels_to_value(edges, mod, codes)                       # (n_frames, C*n_band)
    v = vals.reshape(vals.shape[0], n_ch, n_band)
    b0, b1 = band_slice(n_band, n_freq, f_lo, f_hi)
    return v[:, :, b0:b1 + 1].mean(axis=(1, 2))


def valid_frames(shot: str, mod: str, data_dir, n_frames: int) -> int:
    """How many frames a modality actually has DATA for.

    Diagnostics are fixed-length digitiser records -- mhr is 2**21+1 samples = 4.194 s -- so a
    5 s rollout window runs off the end of them. The cache still holds codes there because the
    encoder was handed padding, and those codes decode to plausible-looking values. Scoring
    them compares the model against fabricated ground truth, so metrics stop here.
    """
    import h5py
    p = Path(data_dir) / f"{shot}_processed.h5"
    if not p.exists():
        print(f"[skill] WARNING {p} missing — cannot trim {mod}@{shot} to its digitiser "
              f"record; scoring the FULL {n_frames} frames, padding included")
        return n_frames
    try:
        with h5py.File(p, "r") as f:
            if mod not in f:
                return n_frames
            t_end = float(f[mod]["xdata"][-1])
    except Exception as exc:
        print(f"[skill] WARNING {mod}@{shot} record length unreadable "
              f"({type(exc).__name__}) — scoring the full {n_frames} frames")
        return n_frames
    return max(0, min(n_frames, int(np.floor(t_end / FRAME_DT_S))))


def bp_geometry(names, n_tok_by_name, codec_root):
    """{mod: (n_ch, n_band, n_freq)}; None if the geometry cannot be established.

    D = C * n_band, so n_band follows from the codec's channel count, and n_freq is the codec
    input width the equal-width bands were cut from. Both come from the codec cfg rather than
    being assumed, since a codec re-train could change either.
    """
    geom = {}
    for m in names:
        p = Path(codec_root) / m / "codec_best.pt"
        if not p.exists():
            print(f"[skill] no codec cfg at {p} — decoded metrics unavailable for {m}")
            return None
        cfgc = torch.load(p, map_location="cpu", weights_only=False)["cfg"]
        n_ch, n_freq = int(cfgc.channels), int(cfgc.freq_bins)
        n_tok = int(n_tok_by_name[m])
        if n_ch <= 0 or n_tok % n_ch:
            print(f"[skill] {m}: n_tok {n_tok} not divisible by channels {n_ch} — "
                  f"decoded metrics unavailable")
            return None
        geom[m] = (n_ch, n_tok // n_ch, n_freq)
        print(f"[skill] {m}: channels={n_ch} bands={n_tok // n_ch} tokens={n_tok} "
              f"n_freq={n_freq}")
    return geom


def decoded_metrics(edges, geom, mod, gt_codes, pred_codes, K0, F, kvalid, horizons):
    """Per-band decoded skill for one modality/shot: full data-valid window + k prefixes.

    Returns {band_label: {...}}. The k prefix window is [K0, min(K0+k, kvalid)); a k that runs
    past the digitiser record is TRUNCATED and says so, because a silently-shortened horizon
    would otherwise be read as a full-length result.
    """
    n_ch, n_band, n_freq = geom[mod]
    out = {}
    for label, f_lo, f_hi in BANDS_HZ:
        gt = band_trace(edges, mod, gt_codes[:F], n_ch, n_band, n_freq, f_lo, f_hi)
        pred = band_trace(edges, mod, pred_codes[:F], n_ch, n_band, n_freq, f_lo, f_hi)
        # persistence = freeze the last SEED frame for the whole predicted region
        pers = np.concatenate([gt[:K0], np.repeat(gt[K0 - 1], F - K0)])
        nr, npers, sk = decoded_space_skill(gt[:kvalid], pred[:kvalid], pers[:kvalid], K0)
        hz = {}
        for k in horizons:
            end = min(K0 + k, kvalid)
            _n, _p, _s = decoded_space_skill(gt[:end], pred[:end], pers[:end], K0)
            hz[str(k)] = {"skill": _s, "end_frame": int(end),
                          "truncated": bool(K0 + k > kvalid)}
        out[label] = {"skill": sk, "nrmse": nr, "nrmse_persistence": npers,
                      "k_valid": int(kvalid), "horizons": hz}
    return out


def token_metrics(gt, pred, K0, F):
    """Token accuracy skill vs persistence, plus the majority-token degeneracy guard."""
    g, p = gt[K0:F], pred[K0:F]
    acc = float((g == p).float().mean())
    last = gt[K0 - 1: K0]
    pers = float((g == last).float().mean())
    _vals, cnt = torch.unique(g, return_counts=True)
    major = float(cnt.max()) / float(g.numel())
    return {"skill": acc - pers, "acc": acc, "persistence": pers,
            "majority": major, "beats_majority": acc > major}


# ------------------------------------------------------------------------------------------- #
# checkpoint discovery
# ------------------------------------------------------------------------------------------- #
def _payload_step(path: Path):
    """The `step` field of a checkpoint, read WITHOUT materializing the weights.

    mmap=True keeps the tensor storages on disk, so this touches only the pickle header
    and never the GPU (MEASURED 0.8 s on a warm 412 MB bp128 checkpoint; a COLD Lustre
    read of the same file can take tens of seconds, so do not expect it to be free on
    the first pass over a run dir). map_location "meta" also works on these payloads;
    "cpu" is used because it is the same read path load_model takes, and weights_only=False
    for the same reason — these payloads carry non-tensor config (`modalities` is a tuple
    of tuples), and a future entry that the safe unpickler's allowlist rejects would break
    a probe that only wants an int.
    """
    try:
        ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except Exception as exc:                                             # pragma: no cover
        print(f"[skill] WARNING cannot read {path.name}: {type(exc).__name__}: {exc}")
        return None
    step = ck.get("step")
    del ck
    return None if step is None else int(step)


def checkpoints(run_dir: Path):
    """Every distinct-step checkpoint in a run dir, ascending by step.

    Step-tagged files are preferred, but the production runs do NOT keep any: bp128_d512L8
    holds ONLY dynamics_best.pt and dynamics_latest.pt (the trainer overwrites both every
    ckpt_every steps). Ingesting `best` as well as `latest` is what makes the two-point
    regression curve — and the documented 11k -> 20k inversion — measurable at all; with
    `latest` alone the curve is a single point and this harness proves nothing.

    Steps are read from the PAYLOAD for both un-tagged files, never guessed from mtime.
    """
    by_step: dict = {}
    for p in sorted(run_dir.glob("dynamics_step*.pt")):
        m = re.search(r"step(\d+)", p.name)
        if m:
            by_step.setdefault(int(m.group(1)), p)
    # de-dup by STEP, step-tagged wins: `best` and `latest` are aliases of steps that may
    # already be present under an explicit name (and are each other's alias when the last
    # checkpoint was also the best).
    for name in ("dynamics_best.pt", "dynamics_latest.pt"):
        p = run_dir / name
        if not p.exists():
            continue
        step = _payload_step(p)
        if step is None:
            print(f"[skill] WARNING {p} carries no `step` field — skipped")
            continue
        if step in by_step:
            print(f"[skill] {p.name} is step {step}, already covered by "
                  f"{by_step[step].name} — skipped")
            continue
        by_step[step] = p
    return sorted(by_step.items())


# ------------------------------------------------------------------------------------------- #
# reporting
# ------------------------------------------------------------------------------------------- #
def _mean(dicts, key):
    vals = [d[key] for d in dicts if d[key] == d[key]]          # drop NaN
    return sum(vals) / len(vals) if vals else float("nan")


def summarize(per_shot_tok, per_shot_dec, shots):
    """Collapse per-shot metrics to one record, keeping the per-shot values visible."""
    tok = {n: {k: (_mean(v, k) if isinstance(v[0][k], float) else all(d[k] for d in v))
               for k in v[0]}
           for n, v in per_shot_tok.items()}
    dec = {}
    for key, v in per_shot_dec.items():                          # key = "mod|band"
        hs = sorted({h for d in v for h in d["horizons"]}, key=int)
        dec[key] = {
            "skill": _mean(v, "skill"),
            "nrmse": _mean(v, "nrmse"),
            "nrmse_persistence": _mean(v, "nrmse_persistence"),
            "per_shot": {s: d["skill"] for s, d in zip(shots, v)},
            "k_valid": {s: d["k_valid"] for s, d in zip(shots, v)},
            "horizons": {h: {"skill": _mean([d["horizons"][h] for d in v], "skill"),
                             "truncated": any(d["horizons"][h]["truncated"] for d in v)}
                         for h in hs},
        }
    return {"decoded": dec, "token": tok}


def print_record(tag, rec):
    for key, d in sorted(rec["decoded"].items()):
        hz = "  ".join(f"k{h}={v['skill']:+.3f}" + ("*" if v["truncated"] else "")
                       for h, v in d["horizons"].items())
        print(f"[skill] {tag}  DECODED {key:26s} skill={d['skill']:+.3f}   {hz}", flush=True)
    for n, d in sorted(rec["token"].items()):
        print(f"[skill] {tag}  token   {n:26s} skill={d['skill']:+.3f}"
              + ("" if d["beats_majority"] else "  (DEGENERATE: below majority token)"),
              flush=True)


def make_figure(results, out_dir):
    dec_keys = sorted({k for a in results.values() for s in a.values() for k in s["decoded"]})
    tok_keys = sorted({k for a in results.values() for s in a.values() for k in s["token"]})
    rows = [("decoded", k) for k in dec_keys] + [("token", k) for k in tok_keys]
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 1, figsize=(9, 2.6 * len(rows)), sharex=True,
                             squeeze=False)
    for ax, (kind, key) in zip(axes[:, 0], rows):
        for arm, by_step in results.items():
            xs = sorted(by_step)
            ys = [by_step[s][kind].get(key, {}).get("skill", float("nan")) for s in xs]
            ax.plot(xs, ys, marker="o", label=arm)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        lab = "decoded\nband skill" if kind == "decoded" else "token\nskill"
        ax.set_ylabel(f"{key}\n{lab}", fontsize=8)
        ax.legend(fontsize=7)
    axes[-1, 0].set_xlabel("training step")
    fig.suptitle("Rollout skill vs training step (skill must not invert)\n"
                 "DECODED per-band rows are the gate; token rows are the degeneracy check",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "skill_vs_step.png", dpi=110)
    print("wrote", out_dir / "skill_vs_step.png")


# ------------------------------------------------------------------------------------------- #
def run_from_traces(args, out_dir, horizons):
    """VALIDATION GATE for the decoded metric implementation.

    Recomputes decoded band skill from the ARCHIVED 2026-08-15 rollouts through this harness's
    own dequantize -> band-reduce -> skill path (starting from the archived CODES, not the
    archived traces, so the whole chain is exercised), and cross-checks against the traces the
    original script stored. mhr TM 1-20 kHz must come out +0.153 @11k and -0.862 @20k.
    """
    edges = load_edges(args.bp_root)
    if edges is None:
        raise SystemExit(f"no bin_edges*.npz under {args.bp_root}")
    results: dict = defaultdict(dict)
    for d in args.from_traces:
        d = Path(d)
        z = np.load(d / "bp_cases_traces.npz", allow_pickle=True)
        meta = json.loads((d / "bp_cases_meta.json").read_text())
        step, K0, F = int(meta["step"]), int(z["K0"]), int(z["F"])
        mods = [m for m in meta["data_valid_frames"]]
        geom = bp_geometry(mods, {m: z[f"codes_gt__{m}"].shape[1] for m in mods},
                           args.codec_root)
        if geom is None:
            raise SystemExit("cannot establish bp geometry")
        per_dec, per_tok = defaultdict(list), defaultdict(list)
        for m in mods:
            gt_c, pr_c = z[f"codes_gt__{m}"], z[f"codes_pred__real__{m}"]
            kv = int(meta["data_valid_frames"][m])
            for band, rec in decoded_metrics(edges, geom, m, gt_c, pr_c, K0, F, kv,
                                             horizons).items():
                per_dec[f"{m}|{band}"].append(rec)
                # cross-check against the trace the ORIGINAL script archived
                tkey = f"trace_gt__{m}__{band}"
                if tkey in z.files:
                    _n, _p, s_arch = decoded_space_skill(
                        z[tkey][:kv], z[f"trace_pred__real__{m}__{band}"][:kv],
                        z[f"trace_pers__{m}__{band}"][:kv], K0)
                    print(f"[check] step {step} {m}|{band}: from CODES {rec['skill']:+.6f}"
                          f"  vs from ARCHIVED TRACE {s_arch:+.6f}"
                          f"  delta {abs(rec['skill'] - s_arch):.2e}")
            per_tok[m].append(token_metrics(torch.from_numpy(gt_c.astype(np.int64)),
                                            torch.from_numpy(pr_c.astype(np.int64)), K0, F))
        results[meta.get("shot", d.name)][step] = summarize(per_tok, per_dec,
                                                           [meta.get("shot", d.name)])
        print_record(f"{d.name} step {step}", results[meta.get('shot', d.name)][step])
    (out_dir / "skill_vs_step_from_traces.json").write_text(json.dumps(results, indent=2))
    print("wrote", out_dir / "skill_vs_step_from_traces.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", action="append",
                    help="checkpoint directory; repeat for multiple arms")
    ap.add_argument("--cache_dir", default=DEFAULT_CACHE)
    ap.add_argument("--shots", default="199597")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k0", type=int, default=20, help="seed frames (0 => use cfg.k0_seed)")
    ap.add_argument("--seed", type=int, default=1234, help="rollout RNG seed (reproducible)")
    ap.add_argument("--horizons", default="10,40,80",
                    help="decoded skill is also reported over [k0, k0+k) for each k")
    # decoded-metric inputs. --bp_root "" disables the decoded metrics (token skill only),
    # which is what a non-band-power tokenizer must do: the quantile edges would be meaningless.
    ap.add_argument("--bp_root", default=DEFAULT_BP_ROOT,
                    help="supplies the TRAINED bin_edges*.npz; '' = token skill only")
    ap.add_argument("--codec_root", default=DEFAULT_CODEC_ROOT,
                    help="supplies each modality's channels/freq_bins")
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                    help="H5 dir, used ONLY to find where each diagnostic's record ends")
    ap.add_argument("--from_traces", action="append",
                    help="validation mode: recompute decoded skill from an archived "
                         "ignite_bp_cases output dir instead of rolling out")
    # Task-11 sampler flags, constructed exactly as eval_dynamics.main does so an arm
    # measured here is the same decode policy the single-shot eval reports.
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--global_pool", action="store_true")
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--revision_rounds", type=int, default=0)
    ap.add_argument("--cfg_scale", type=float, default=1.0)
    ap.add_argument("--best_of_n", type=int, default=1)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    horizons = [int(k) for k in args.horizons.split(",") if k.strip()]

    if args.from_traces:
        return run_from_traces(args, out_dir, horizons)
    if not args.run_dir:
        raise SystemExit("--run_dir is required (or use --from_traces)")

    from tokamak_foundation_model.ignite.eval_dynamics import (
        load_model, load_shot_cache, rollout_shot)
    from tokamak_foundation_model.ignite.sampling import SamplerConfig

    shots = [s.strip() for s in args.shots.split(",") if s.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampler = SamplerConfig(temperature=args.temperature, top_p=args.top_p,
                            global_pool=args.global_pool,
                            revision_rounds=args.revision_rounds,
                            cfg_scale=args.cfg_scale)
    edges = load_edges(args.bp_root) if args.bp_root else None
    if edges is None:
        print("[skill] decoded band-restricted skill DISABLED (no trained bin edges) — "
              "token skill only, which is NOT sufficient to gate an exposure-bias fix")
    print(f"[skill] device={device} cache={args.cache_dir} shots={shots} k0={args.k0} "
          f"seed={args.seed} horizons={horizons} global_pool={args.global_pool} "
          f"top_p={args.top_p} revision_rounds={args.revision_rounds} "
          f"cfg_scale={args.cfg_scale} best_of_n={args.best_of_n}", flush=True)
    results: dict = defaultdict(dict)

    for run in args.run_dir:
        arm = Path(run).name
        ckpts = checkpoints(Path(run))
        if not ckpts:
            raise SystemExit(f"no checkpoints with a readable step in {run}")
        print(f"[skill] {arm}: {[(s, p.name) for s, p in ckpts]}", flush=True)
        for step, ckpt in ckpts:
            # load_model(ckpt_path, device) -> (model, cfg, step)   [eval_dynamics.py:121]
            model, cfg, _ = load_model(Path(ckpt), device)
            k0 = args.k0 or int(cfg.k0_seed)
            names = [m.name for m in cfg.modalities]
            geom = (bp_geometry(names, {m.name: m.n_tok for m in cfg.modalities},
                                args.codec_root) if edges is not None else None)
            # Guard the dequantization: the trained edges only describe a vocab-8 band-power
            # tokenizer. Applying them to a 64k-vocab FSQ modality would silently produce
            # numbers that look fine and mean nothing.
            if geom is not None:
                bad = [m.name for m in cfg.modalities
                       if int(m.codebook_size) != N_LEV or m.name not in edges.files
                       or edges[m.name].shape[1] != int(m.n_tok)]
                if bad:
                    print(f"[skill] decoded metrics DISABLED: {bad} are not vocab-{N_LEV} "
                          f"band-power modalities matching the trained edges")
                    geom = None
            per_tok, per_dec = defaultdict(list), defaultdict(list)
            for shot in shots:
                cache = load_shot_cache(Path(args.cache_dir), shot)
                # rollout_shot(model, cfg, cache, K0, temperature, generator, device, ...)
                # -> (gt_codes, pred_codes, K0, F); tensors are (F, n_tok) cpu long, NO batch dim.
                # The generator must live on `device` — that is the convention the rest of
                # eval_dynamics uses (eval_dynamics.py:1061); a CPU generator raises inside
                # torch.multinomial on a CUDA distribution.
                gt, pred, K0, F = rollout_shot(
                    model, cfg, cache, k0, args.temperature,
                    torch.Generator(device=device).manual_seed(args.seed), device,
                    sampler=sampler, best_of=args.best_of_n)
                for name in pred:
                    per_tok[name].append(token_metrics(gt[name], pred[name], K0, F))
                    if geom is None:
                        continue
                    kv = valid_frames(shot, name, args.data_dir, F)
                    if kv <= K0 + 2:
                        print(f"[skill] {name}@{shot}: only {kv} data-valid frames "
                              f"(K0={K0}) — decoded metrics skipped")
                        continue
                    for band, rec in decoded_metrics(
                            edges, geom, name, gt[name].numpy(), pred[name].numpy(),
                            K0, F, kv, horizons).items():
                        per_dec[f"{name}|{band}"].append(rec)
            results[arm][step] = summarize(per_tok, per_dec, shots)
            print_record(f"{arm} step {step}", results[arm][step])
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    with open(out_dir / "skill_vs_step.json", "w") as f:
        json.dump(results, f, indent=2)
    make_figure(results, out_dir)


if __name__ == "__main__":
    main()
