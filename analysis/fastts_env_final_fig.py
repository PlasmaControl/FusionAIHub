"""FINAL fast-TS (filterscopes) ENVELOPE codec figure: GT vs reconstruction, full shot.

Companion to the slow-TS deliverable (eval_runs/codec_recon_figs/slowts_FINAL_fullshot.png).
Evaluation + rendering ONLY -- reads the production envelope codec read-only, trains nothing.

Windows that carry no valid data are left as GAPS (never filled, never interpolated); each
panel states what fraction of the time axis actually has data.

Per channel the panel reports, all pooled over HELD-OUT shots:
    nRMSE        RMSE / that channel's own global std  => a CONSTANT predictor scores 1.000
    k=5 floor    PCA at the codec's own token budget (5 tokens), fit on DISJOINT shots
    std_ratio    std(recon)/std(GT); pooled, and therefore LEVEL-DOMINATED — see ac_amp
    ac_amp       std of the MEAN-REMOVED part, recon/GT: the real within-window AMPLITUDE
                 check. Validated on references of known ordering: a 0.4x-shrunk target reads
                 std_ratio 0.9926 (looks fine) but ac_amp 0.4000 (the truth). Never judge
                 burst amplitude on std_ratio.
    level corr   Pearson r of the reconstructed envelope against ground truth
    burst kept   fraction of a real ELM burst's height that survives the codec

THE BAR (2026-09-03). Every table also carries the RATE-MATCHED window-mean encoder: transmit
ONLY each channel's window mean, quantised to the codec's OWN bit budget (5 tokens x
log2(1000) = 49.83 bits => 74 levels/channel). Measured 0.1935 pooled nRMSE against the
shipped codec's 0.5252 — a codec that loses to "transmit 8 numbers" at its own rate is not
doing its job, so this is the number to beat, not the 1.0000 constant anchor.

Modes
-----
    # the original single-checkpoint figure (production checkpoint, unchanged default)
    python analysis/fastts_env_final_fig.py

    # any checkpoint
    python analysis/fastts_env_final_fig.py --ckpt <path> --out <png>

    # ARM TABLE: score many checkpoints on the same held-out pool, no figure
    python analysis/fastts_env_final_fig.py --table --arms <dir_with_arm_subdirs> [...]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

from tokamak_foundation_model.ignite.fastts_codec import FastTSCodec
from tokamak_foundation_model.ignite.fastts_train import FastTSCodecPairDataset, CHUNK_S

PROD_CKPT = "/lustre/orion/fus187/proj-shared/models/ignite_codecs_prod_noinorm/filterscopes/codec_best.pt"
DATA = "/lustre/orion/fus187/proj-shared/foundation_model"
HELD_OUT = [201874, 193367, 193803, 191258]
PCA_FIT = [190000, 190001, 190002, 190003, 190004, 190005, 190006, 190007]


def load_codec(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    codec = FastTSCodec(cfg).to(device)
    codec.load_state_dict(ck["codec"])
    codec.eval()
    return codec, cfg, ck


def walk_shot(shot, cfg, codec, device="cpu"):
    """Full-shot consecutive-window walk. Returns GT/recon (n_win, C, E) with NaN gaps.

    Uses ``_build_pair`` DIRECTLY so the activity-stratified re-draw is bypassed: a window with
    no valid data stays a gap instead of being silently replaced by a different window.
    """
    ds = FastTSCodecPairDataset([shot], cfg, data_dir=DATA)
    _ = ds[0]                                   # binds the per-worker h5 handle for this shot
    n = len(ds)
    C, E = cfg.channels, cfg.env_bins
    gt = np.full((n, C, E), np.nan, dtype=np.float32)
    for i in range(n):
        try:
            pair = ds._build_pair(i)
        except Exception:
            pair = None
        if pair is not None:
            gt[i] = pair[0].numpy()
    live = np.isfinite(gt).all(axis=(1, 2))
    rec = np.full_like(gt, np.nan)
    if live.any():
        with torch.no_grad():
            x = torch.from_numpy(gt[live]).to(device)
            out = []
            for s in range(0, x.shape[0], 128):
                out.append(codec.forward(x[s:s + 128])["recon"].cpu().numpy())
            rec[live] = np.concatenate(out, 0)
    t0 = float(getattr(ds, "warmup_s", 1.0))
    # bin j of window i is centred at t0 + i*CHUNK_S + (j+0.5)*(CHUNK_S/E)
    t = (t0 + np.arange(n)[:, None] * CHUNK_S
         + (np.arange(E)[None, :] + 0.5) * (CHUNK_S / E))
    return gt, rec, live, t


def channel_metrics(G, R):
    """Per-channel pooled metrics from already-walked GT/recon arrays ``(N, C, E)``."""
    C = G.shape[1]
    out = {}
    for c in range(C):
        g, r = G[:, c].ravel(), R[:, c].ravel()
        sd = g.std()
        # AC (mean-removed) amplitude: the pooled std_ratio below is dominated by the
        # between-window level (96.2% of the variance) and reads ~0.99 even on a prediction
        # whose bursts are at 40% height. This is the amplitude number that is not fooled.
        gac = G[:, c] - G[:, c].mean(1, keepdims=True)
        rac = R[:, c] - R[:, c].mean(1, keepdims=True)
        # TWO different correlations, deliberately both reported:
        #   corr      full-envelope Pearson r over every (window, bin) -- what the original
        #             figure printed as "level corr" (0.870 for the shipped codec).
        #   lev_corr  Pearson r of the per-WINDOW level itself. The window-mean baseline scores
        #             EXACTLY 1.000 here (it transmits the level) while its full-envelope corr
        #             is 0.981, which is why the two must not be conflated.
        lt, lr = G[:, c].mean(1), R[:, c].mean(1)
        out[c] = dict(
            nrmse=float(np.sqrt(((r - g) ** 2).mean()) / (sd + 1e-12)),
            std_ratio=float(r.std() / (sd + 1e-12)),
            ac_amp=float(rac.std() / (gac.std() + 1e-12)),
            corr=float(np.corrcoef(g, r)[0, 1]) if r.std() > 1e-12 else 0.0,
            lev_corr=float(np.corrcoef(lt, lr)[0, 1]) if lr.std() > 1e-12 else 0.0,
            bias=float((r.mean() - g.mean()) / (sd + 1e-12)),
            burst=_burst_capture(G[:, c], R[:, c]),
        )
    return out


def walk_pool(shots, cfg, codec, cache=None):
    """Walk a shot list once and return pooled ``(N, C, E)`` GT + recon (NaN gaps dropped).

    ``cache`` (a dict) memoises the GT walk per shot so an 8-arm table reads the HDF5 once
    instead of once per arm.
    """
    G, R = [], []
    # Key the GT cache by the fields that DEFINE the envelope, so an arm with different
    # envelope geometry can never be served another arm's ground truth.
    key = (cfg.channels, cfg.env_bins, cfg.pool, cfg.baseline_win, cfg.env_eps,
           tuple(cfg.channel_mean or ()), tuple(cfg.channel_std or ()))
    for s in shots:
        if cache is not None and (key, s) in cache:
            gt, live = cache[(key, s)]
            rec = _apply_codec(gt, live, cfg, codec)
        else:
            gt, rec, live, _ = walk_shot(s, cfg, codec)
            if cache is not None:
                cache[(key, s)] = (gt, live)
        if live.any():
            G.append(gt[live]); R.append(rec[live])
    return np.concatenate(G, 0), np.concatenate(R, 0)


def _apply_codec(gt, live, cfg, codec):
    rec = np.full_like(gt, np.nan)
    if live.any() and codec is not None:
        with torch.no_grad():
            x = torch.from_numpy(gt[live])
            out = [codec.forward(x[i:i + 128])["recon"].cpu().numpy()
                   for i in range(0, x.shape[0], 128)]
            rec[live] = np.concatenate(out, 0)
    return rec


def pooled_metrics(shots, cfg, codec, cache=None):
    G, R = walk_pool(shots, cfg, codec, cache=cache)
    return channel_metrics(G, R), G, R


def _burst_capture(g2, r2):
    """Fraction of a real burst's height that survives the codec.

    Peak-above-median on the top-1% most active windows: the ELM bursts are the whole point of
    a fast-TS channel, and nRMSE (dominated by the quiet 99%) barely notices if they are flattened.
    """
    g, r = g2.max(1), r2.max(1)
    k = max(1, int(0.01 * len(g)))
    idx = np.argsort(g)[-k:]
    base = np.median(g2)
    return float((r[idx] - base).mean() / ((g[idx] - base).mean() + 1e-12))


def rate_matched_wmean(G, cfg):
    """Trivial baseline at the CODEC'S OWN bit rate: transmit only each channel's window mean.

    5 tokens x log2(1000) = 49.83 bits/window; split 8 ways that is 6.23 bits => 74 levels per
    channel. So this is a like-for-like rate comparison, not a free-information cheat.
    """
    C, E = G.shape[1], G.shape[2]
    bits = cfg.n_env_patch * np.log2(float(np.prod(cfg.fsq_levels)))
    lv = int(2 ** (bits / C))
    out = {}
    for c in range(C):
        g = G[:, c]; wm = g.mean(1, keepdims=True)
        lo, hi = wm.min(), wm.max()
        q = np.round((wm - lo) / (hi - lo + 1e-12) * (lv - 1)) / (lv - 1) * (hi - lo) + lo
        out[c] = float(np.sqrt(((np.repeat(q, E, 1) - g) ** 2).mean()) / (g.std() + 1e-12))
    return out, bits, lv


def pca_floor(fit_shots, G_test, cfg, codec, k):
    """PCA at the codec's OWN token budget, fit on shots DISJOINT from the test pool."""
    F = []
    for s in fit_shots:
        gt, _, live, _ = walk_shot(s, cfg, codec)
        if live.any():
            F.append(gt[live])
    F = np.concatenate(F, 0)
    Xf = F.reshape(len(F), -1).astype(np.float64)
    mu = Xf.mean(0)
    # numpy SVD on the host: torch.linalg.eigh is a known silent-SIGKILL on this machine.
    _, _, Vt = np.linalg.svd(Xf - mu, full_matrices=False)
    P = Vt[:k]
    Xt = G_test.reshape(len(G_test), -1).astype(np.float64)
    Rt = ((Xt - mu) @ P.T) @ P + mu
    C, E = G_test.shape[1], G_test.shape[2]
    Rt = Rt.reshape(-1, C, E)
    return {c: float(np.sqrt(((Rt[:, c] - G_test[:, c]) ** 2).mean())
                     / (G_test[:, c].std() + 1e-12)) for c in range(C)}


def baselines(G):
    """Trivial anchors in the ENVELOPE domain (the domain this codec is scored in)."""
    C = G.shape[1]; b = {}
    for c in range(C):
        g = G[:, c]; sd = g.std()
        f = lambda p: float(np.sqrt(((p - g) ** 2).mean()) / (sd + 1e-12))
        b[c] = dict(
            self=f(g),
            gmean=f(np.full_like(g, g.mean())),                       # global constant
            wmean=f(np.repeat(g.mean(1, keepdims=True), g.shape[1], 1)),  # per-window mean
        )
    return b


# --------------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------------- #
def render(shot, out_png, device="cpu", zoom_s=1.5, ckpt=PROD_CKPT, label=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    codec, cfg, ck = load_codec(ckpt, device)
    met, G, R = pooled_metrics(HELD_OUT, cfg, codec)
    floor = pca_floor(PCA_FIT, G, cfg, codec, k=cfg.n_env_patch)
    base = baselines(G)
    rmw, bits, lv = rate_matched_wmean(G, cfg)
    gt, rec, live, t = walk_shot(shot, cfg, codec, device)

    C, E = cfg.channels, cfg.env_bins
    tf = t.ravel()
    cov = 100.0 * live.mean()

    # pick the zoom window: the most ACTIVE live stretch (largest GT variance across channels)
    wm = np.where(live[:, None, None], gt, np.nan).reshape(len(gt), -1)
    wm = np.nanmean(wm, axis=1)
    d = np.abs(np.diff(wm, prepend=wm[0]))
    d[~live] = -np.inf
    w = max(1, int(round(zoom_s / CHUNK_S)))
    zc = int(np.nanargmax(np.where(np.isfinite(d), d, -np.inf)))
    z0 = int(np.clip(zc - w // 2, 0, max(0, len(gt) - w - 1)))
    zt = (t[z0, 0], t[min(len(t) - 1, z0 + w), -1])

    fig, axes = plt.subplots(C, 2, figsize=(25, 2.75 * C),
                             gridspec_kw={"width_ratios": [3.1, 1.0]})
    mn = float(np.nanmean([met[c]["nrmse"] for c in met]))
    mf = float(np.nanmean(list(floor.values())))
    ms = float(np.nanmean([met[c]["std_ratio"] for c in met]))
    mc = float(np.nanmean([met[c]["corr"] for c in met]))
    bw = float(np.nanmean([base[c]["wmean"] for c in base]))
    mb = float(np.nanmean([abs(met[c]["bias"]) for c in met]))
    mr = float(np.nanmean(list(rmw.values())))
    mu = 100.0 * float(np.nanmean([met[c]["burst"] for c in met]))
    ma = float(np.nanmean([met[c]["ac_amp"] for c in met]))
    gsx = getattr(cfg, "n_gain_tok", 0)
    arch = (f"GAIN-SHAPE {gsx} gain + {cfg.n_shape_tok} shape tokens"
            if gsx else "flat 5-token codec")
    verdict = (f"BEATS the rate-matched bar by {mr / mn:.2f}x" if mn < mr else
               f"LOSES to a trivial rate-matched encoder {mn / mr:.1f}x")
    fig.suptitle(
        f"FAST-TS (filterscopes) ENVELOPE CODEC — {label or ckpt} (step {ck['step']}), "
        f"{arch}, shot {shot} HELD OUT.  GAPS = no data (never filled).\n"
        f"Pooled over {len(HELD_OUT)} held-out shots ({len(G)} windows):   "
        f"mean nRMSE {mn:.4f}   vs   k={cfg.n_env_patch} PCA floor {mf:.4f}   |   "
        f"AC amplitude {ma:.3f} (ideal 1.0; pooled std_ratio {ms:.3f} is level-dominated)"
        f"   |   level corr {mc:.3f}   |   mean |level bias| {mb:.2f} sd   |   "
        f"burst height kept {mu:.0f}%\n"
        f"Anchors:  self 0.0000   |   global constant 1.0000   |   per-window-mean {bw:.4f} "
        f"(the level-only FLOOR)   |   RATE-MATCHED window-mean "
        f"({bits:.1f} bits/window = {lv} levels/ch) {mr:.4f}  <-- THE BAR: this codec "
        f"{verdict}",
        fontsize=12.5, y=0.997)

    for c in range(C):
        g = gt[:, c, :].ravel()
        r = rec[:, c, :].ravel()
        m = met[c]
        for j, ax in enumerate(axes[c]):
            ax.plot(tf, g, color="k", lw=1.1, label="GT envelope", zorder=3)
            ax.plot(tf, r, color="crimson", lw=1.0, ls="--", label="codec recon", zorder=4)
            ax.grid(alpha=0.25)
            if j == 0:
                # two short lines instead of one long one: the single-line form ran off the
                # left panel and collided with the zoom panel's title.
                ax.set_title(
                    f"ch {c} — shot {shot}, {tf[-1]:.1f} s, {cov:.0f}% of the axis has DATA "
                    f"(gaps = no data)   |   pooled nRMSE {m['nrmse']:.4f}   vs   "
                    f"RATE-MATCHED bar {rmw[c]:.4f}, constant 1.000, k={cfg.n_env_patch} "
                    f"floor {floor[c]:.4f}\n"
                    f"AC amp {m['ac_amp']:.2f} (pooled std_ratio {m['std_ratio']:.2f} is "
                    f"level-dominated)   |   level corr {m['lev_corr']:.2f} "f"(envelope corr {m['corr']:.2f})   |   level bias "
                    f"{m['bias']:+.2f} sd   |   BURST HEIGHT KEPT {100*m['burst']:.0f}%",
                    fontsize=9.5, loc="left")
                ax.set_ylabel("log1p RMS envelope")
                ax.set_xlim(tf[0], tf[-1])
                if c == 0:
                    ax.legend(loc="upper right", fontsize=9, ncol=2)
            else:
                ax.set_title(f"zoom {zt[0]:.2f}–{zt[1]:.2f} s (largest transient)",
                             fontsize=10.5, loc="left")
                ax.set_xlim(*zt)
            if c == C - 1:
                ax.set_xlabel("time (s)")
    fig.tight_layout(rect=[0, 0, 1, 0.962])
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    print("wrote", out_png)
    return met, floor, base, rmw


def arm_table(ckpts, device="cpu"):
    """Score many checkpoints on the SAME held-out pool and print one ranked table.

    Every row carries BOTH anchors (the 1.0000 constant and the rate-matched window-mean bar),
    the amplitude reads, and the utilization expressed as a BIT RATE — because a codec that
    reaches a good nRMSE on 3 codes has not earned it.
    """
    cache: dict = {}
    rows = []
    ref_bits = ref_bar = ref_floor = None

    def _baseline_rows(G, cfg):
        """The trivial encoders as FULL rows, scored by the SAME function as every codec.

        Reporting the bar only as an nRMSE is misleading in the other direction: the
        rate-matched window-mean encoder wins on nRMSE while carrying NO within-window
        information at all (AC amplitude 0.000, and it keeps a burst only to the extent the
        burst raises the window mean). A codec has to beat it on nRMSE *and* still carry the
        shape, so both halves belong in the table.
        """
        E = G.shape[2]
        lev = G.mean(-1, keepdims=True)
        bits = cfg.n_env_patch * np.log2(float(np.prod(cfg.fsq_levels)))
        lv = int(2 ** (bits / G.shape[1]))
        q = np.empty_like(lev)
        for c in range(G.shape[1]):
            w = lev[:, c, 0]
            lo, hi = w.min(), w.max()
            q[:, c, 0] = np.round((w - lo) / (hi - lo + 1e-12) * (lv - 1)) / (lv - 1) * (hi - lo) + lo
        preds = [
            ("~BASE rate-matched", np.repeat(q, E, 2)),
            ("~BASE window-mean", np.repeat(lev, E, 2)),
            ("~BASE constant", np.broadcast_to(G.mean((0, 2))[None, :, None], G.shape)),
        ]
        out = []
        for nm, pr in preds:
            m = channel_metrics(G, np.ascontiguousarray(pr))
            agg = lambda k: float(np.nanmean([m[c][k] for c in m]))
            out.append(dict(arm=nm, step=-1, nrmse=agg("nrmse"), ac_amp=agg("ac_amp"),
                            std_ratio=agg("std_ratio"), corr=agg("corr"),
                            burst=agg("burst"), lev_corr=agg("lev_corr"),
                            gain_tok=0, used_bits=(bits if "rate" in nm else 0.0),
                            max_bits=float(bits), n_codes=0, n_win=int(len(G)), path=""))
        return out

    base_rows = None
    for label, path in ckpts:
        try:
            codec, cfg, ck = load_codec(path, device)
        except Exception as e:                       # an arm that never wrote a checkpoint
            print(f"  !! {label}: {type(e).__name__}: {e}")
            continue
        G, R = walk_pool(HELD_OUT, cfg, codec, cache=cache)
        met = channel_metrics(G, R)
        base = baselines(G)
        rmw, bits, lv = rate_matched_wmean(G, cfg)
        # utilization as a BIT RATE, measured on the same held-out pool.
        with torch.no_grad():
            codes = np.concatenate([
                codec.forward(torch.from_numpy(G[i:i + 128]))["codes"].cpu().numpy()
                for i in range(0, len(G), 128)], 0)          # (N, n_tok, fsq_dim)
        lvls = np.array(cfg.fsq_levels)
        flat = (codes * np.concatenate([[1], np.cumprod(lvls)[:-1]])).sum(-1)  # (N, n_tok)
        ent = 0.0
        for t in range(flat.shape[1]):
            _, cnt = np.unique(flat[:, t], return_counts=True)
            pr = cnt / cnt.sum()
            ent += float(-(pr * np.log2(pr)).sum())
        n_dist = int(len(np.unique(flat)))
        agg = lambda k: float(np.nanmean([met[c][k] for c in met]))
        ref_bits, ref_bar = bits, float(np.nanmean(list(rmw.values())))
        ref_floor = float(np.nanmean([base[c]["wmean"] for c in base]))
        rows.append(dict(
            arm=label, step=int(ck.get("step", -1)), nrmse=agg("nrmse"),
            ac_amp=agg("ac_amp"), std_ratio=agg("std_ratio"), corr=agg("corr"),
            burst=agg("burst"), lev_corr=agg("lev_corr"),
            bar=ref_bar, floor=ref_floor,
            used_bits=ent, max_bits=float(bits), n_codes=n_dist,
            gain_tok=int(getattr(cfg, "n_gain_tok", 0)),
            n_win=int(len(G)), path=path))
        if base_rows is None:
            base_rows = _baseline_rows(G, cfg)
    for b in (base_rows or []):
        b["bar"], b["floor"] = ref_bar, ref_floor
    rows += (base_rows or [])
    rows.sort(key=lambda r: r["nrmse"])
    print()
    print(f"HELD-OUT POOL: {HELD_OUT}  ({rows[0]['n_win'] if rows else 0} windows)")
    print(f"ANCHORS: global constant 1.0000 | per-window-mean FLOOR {ref_floor:.4f} | "
          f"RATE-MATCHED window-mean BAR {ref_bar:.4f}  ({ref_bits:.1f} bits/window)")
    print()
    hdr = (f"{'arm':<14}{'step':>7}{'gainTok':>8}{'nRMSE':>9}{'vs BAR':>9}{'burst%':>8}"
           f"{'AC amp':>8}{'stdRat':>8}{'levCorr':>9}{'envCorr':>9}{'bits used':>11}{'codes':>7}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['arm']:<14}{r['step']:>7}{r['gain_tok']:>8}{r['nrmse']:>9.4f}"
              f"{r['bar'] / r['nrmse']:>8.2f}x{100 * r['burst']:>8.0f}{r['ac_amp']:>8.3f}"
              f"{r['std_ratio']:>8.3f}{r['lev_corr']:>9.3f}{r['corr']:>9.3f}"
              f"{r['used_bits']:>7.2f}/{r['max_bits']:<3.0f}{r['n_codes']:>7}")
    print("-" * len(hdr))
    print("vs BAR > 1.00x = beats the trivial rate-matched window-mean encoder.")
    return rows


def discover_arms(root):
    """(label, path) for every arm subdir under ``root``; both best and last checkpoints."""
    out = []
    for d in sorted(Path(root).iterdir()):
        if not d.is_dir():
            continue
        for tag, fn in (("best", "codec_best.pt"), ("last", "codec_last.pt")):
            if (d / fn).exists():
                out.append((f"{d.name}:{tag}", str(d / fn)))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shot", type=int, default=201874)
    p.add_argument("--out", default="eval_runs/codec_recon_figs/fastts_env_FINAL_fullshot.png")
    p.add_argument("--ckpt", default=PROD_CKPT, help="checkpoint to render / score")
    p.add_argument("--label", default=None, help="figure title label for --ckpt")
    p.add_argument("--table", action="store_true", help="print the ARM TABLE, no figure")
    p.add_argument("--arms", default=None, help="dir of arm subdirs to table")
    p.add_argument("--extra_ckpt", action="append", default=[],
                   help="label=path row to add to the table (repeatable)")
    a = p.parse_args()

    if a.table:
        ck = [("PROD(shipped)", PROD_CKPT)]
        ck += [(x.split("=", 1)[0], x.split("=", 1)[1]) for x in a.extra_ckpt]
        if a.arms:
            ck += discover_arms(a.arms)
        rows = arm_table(ck)
        json.dump(rows, open(Path(a.out).with_suffix(".table.json"), "w"), indent=1)
        print("wrote", Path(a.out).with_suffix(".table.json"))
    else:
        met, floor, base, rmw = render(a.shot, a.out, ckpt=a.ckpt, label=a.label)
        json.dump({"metrics": met, "floor": floor, "baselines": base,
                   "rate_matched_wmean": rmw, "held_out": HELD_OUT, "pca_fit": PCA_FIT,
                   "ckpt": a.ckpt},
                  open(Path(a.out).with_suffix(".json"), "w"), indent=1)
