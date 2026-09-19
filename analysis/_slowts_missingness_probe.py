"""Is the slow-TS "oscillation" REAL DATA or a ZERO-FILL that the validity mask calls valid?

THE HYPOTHESIS (2026-09-03). ``TokamakH5Dataset._load_signal_raw`` fills missing samples with
0 in RAW units and returns a ``nan_mask`` that is 1 only where the HDF5 value was literally
NaN. ``SlowTSCodecPairDataset`` then derives validity as

    zero_is_missing (the 4 Thomson signals) : valid = (raw != 0) & (nan_mask == 0)
    otherwise       (cer_ti / cer_rot / mse): valid = (nan_mask == 0)

and only AFTER that standardizes, ``(x - mean) / std``. So for the three NON-Thomson signals a
raw 0 that was never a NaN is declared VALID and is then mapped to ``-mean/std`` — a large
constant, negative when the channel mean is positive. That is exactly a square wave between the
real level and one fixed extreme, which is what the ``cer_rot`` / ``cer_ti`` / ``mse`` panels of
slowts_retrain_fullshot.png show, and it is precisely the failing set. The 4 Thomson signals
carry ``zero_is_missing=True`` and are therefore immune by construction.

If confirmed, ``slowts_nrmse`` is measured against fill values on those three: the codec is
charged for not reproducing "missing", the target std is inflated by the fill excursions, and
the codec's flat trace may be CORRECT.

WHAT THIS PRINTS, per signal, over consecutive windows of one shot (all in the codec's own
standardized space, from the same dataset the audit and the trainer use):
  * present_frac      — mask occupancy (the honest missing fraction).
  * raw-zero fraction — samples whose RAW value is exactly 0.0.
  * zero_but_valid    — samples that are raw-0 AND mask-valid: the contaminated ones.
  * the standardized value those samples take, vs the predicted ``-mean/std`` fill level.
  * variance share    — how much of the target's total (masked) variance the contaminated
    samples contribute. This is the number that says whether the metric is corrupted or the
    effect is cosmetic.
  * a corrected present_frac / std if raw-0 were ALSO treated as missing for cer/mse.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tokamak_foundation_model.ignite import train_codec as tc  # noqa: E402

SIGNALS = ("cer_rot", "cer_ti", "mse", "ts_core_density", "ts_core_temp",
           "ts_tangential_density", "ts_tangential_temp")


def _cfg(signal, standardize=True):
    cfg = tc.slowts_codec_cfg(signal, tc.modality_channels(signal))
    if standardize:
        method, mean, std = tc.load_slowts_channel_stats(signal)
        cfg.preprocess_method, cfg.channel_mean, cfg.channel_std = method, mean, std
    cfg.min_activity, cfg.active_bias = 0.0, 0.0
    return cfg


def _stream(signal, shot, n_windows, cache_dir, standardize):
    cfg = _cfg(signal, standardize)
    ds = tc.SlowTSCodecPairDataset(
        signal, [shot], cfg, data_dir=tc.DEFAULT_DATA_DIR,
        lengths_cache_path=str(Path(cache_dir) / f"probe_{signal}_{shot}_lengths.pt"),
    )
    n = min(len(ds), n_windows)
    X, M = [], []
    for i in range(n):
        x, m = ds[i]
        X.append(x.numpy())
        M.append(m.numpy())
    return np.stack(X), np.stack(M), cfg, len(ds)


def probe(signal, shot, n_windows, cache_dir):
    Xs, Ms, cfg_s, n_avail = _stream(signal, shot, n_windows, cache_dir, True)
    Xr, Mr, _cfg_r, _ = _stream(signal, shot, n_windows, cache_dir, False)
    assert Xs.shape == Xr.shape and np.array_equal(Ms, Mr), "mask must not depend on standardize"

    C = cfg_s.channels
    Xs, Xr, Ms = Xs[:, :C], Xr[:, :C], Ms[:, :C]      # drop zone-padding tail positions
    valid = Ms > 0.5
    raw_zero = Xr == 0.0
    zbv = raw_zero & valid                             # the contaminated samples

    mean = np.asarray(cfg_s.channel_mean, dtype=np.float64)[:C]
    std = np.asarray(cfg_s.channel_std, dtype=np.float64)[:C]
    method = cfg_s.preprocess_method
    if method == "log_standardize":
        fill_level = (np.log10(np.clip(0.0, -0.99, None) + 1.0) - mean) / np.clip(std, 1e-3, None)
    else:
        fill_level = (0.0 - mean) / np.clip(std, 1e-3, None)

    # variance share of the contaminated samples, in the metric's own grouping (per window,
    # deviations about that window's masked mean over the flattened (C, T) plane)
    B = Xs.shape[0]
    f = Xs.reshape(B, -1)
    mv = valid.reshape(B, -1).astype(np.float64)
    n_ok = np.maximum(mv.sum(1), 1.0)
    mu = (f * mv).sum(1) / n_ok
    dev2 = ((f - mu[:, None]) ** 2) * mv
    sst = dev2.sum()
    share = dev2[zbv.reshape(B, -1)].sum() / sst if sst > 0 else float("nan")

    # std of the target with vs without the contaminated samples (pooled over windows)
    def _pooled_std(mask_bool):
        mm = mask_bool.reshape(B, -1).astype(np.float64)
        nn = np.maximum(mm.sum(1), 1.0)
        m2 = (f * mm).sum(1) / nn
        return float(np.sqrt((((f - m2[:, None]) ** 2) * mm).sum() / max(mm.sum(), 1.0)))

    std_as_is = _pooled_std(valid)
    std_corrected = _pooled_std(valid & ~raw_zero)

    zbv_vals = Xs[zbv]
    print(f"\n=== {signal}  (shot {shot}, {B} of {n_avail} windows, C={C}, T={Xs.shape[2]}, "
          f"method={method}) ===")
    print(f"  present_frac (mask)            {valid.mean():.4f}")
    print(f"  raw-zero fraction              {raw_zero.mean():.4f}")
    print(f"  ZERO-BUT-MASK-VALID fraction   {zbv.mean():.4f}   <-- contaminated samples")
    print(f"  of the VALID samples, raw-0    {(zbv.sum() / max(valid.sum(), 1)):.4f}")
    if zbv.any():
        print(f"  their standardized value       min {zbv_vals.min():+.4f}  max "
              f"{zbv_vals.max():+.4f}  mean {zbv_vals.mean():+.4f}")
        ch = np.where(zbv.any(axis=(0, 2)))[0]
        print(f"  predicted fill level -mean/std over those {len(ch)} channels: "
              f"min {fill_level[ch].min():+.4f}  max {fill_level[ch].max():+.4f}")
    print(f"  VARIANCE SHARE of contaminated {share:.4f}   <-- how corrupted slowts_nrmse is")
    print(f"  pooled target std  as-is {std_as_is:.4f}   raw-0 also masked {std_corrected:.4f}"
          f"   (ratio {std_corrected / max(std_as_is, 1e-12):.4f})")
    print(f"  standardized range  p1 {np.percentile(Xs[valid], 1):+.3f}  p99 "
          f"{np.percentile(Xs[valid], 99):+.3f}")
    return {
        "signal": signal, "present_frac": float(valid.mean()),
        "raw_zero_frac": float(raw_zero.mean()), "zero_but_valid_frac": float(zbv.mean()),
        "zbv_share_of_valid": float(zbv.sum() / max(valid.sum(), 1)),
        "variance_share": float(share), "std_as_is": std_as_is,
        "std_raw0_masked": std_corrected,
    }


def trace(signal, shot, position, t0, t1, cache_dir, n_windows=400):
    """Print consecutive samples of ONE profile position with raw value + validity flag."""
    Xs, Ms, cfg, _ = _stream(signal, shot, n_windows, cache_dir, True)
    Xr, _Mr, _c, _ = _stream(signal, shot, n_windows, cache_dir, False)
    C, T = cfg.channels, Xs.shape[2]
    s = Xs[:, :C].transpose(1, 0, 2).reshape(C, -1)
    r = Xr[:, :C].transpose(1, 0, 2).reshape(C, -1)
    m = Ms[:, :C].transpose(1, 0, 2).reshape(C, -1)
    dt = tc.CHUNK_S / T
    i0, i1 = int(t0 / dt), min(int(t1 / dt), s.shape[1])
    if position is None:                  # pick the position the figure would pick
        occ = m.mean(axis=1)
        spread = np.where(m > 0.5, s, np.nan)
        with np.errstate(invalid="ignore"):
            sp = np.nan_to_num(np.nanstd(spread, axis=1))
        position = int(np.argmax(np.where(occ >= 0.4, occ * sp, -1.0)))
    print(f"\n--- {signal} shot {shot} position {position}: t, RAW, STANDARDIZED, valid ---")
    print(f"    (raw==0 rows marked '<<'; {i1 - i0} samples from {t0}-{t1} s)")
    n0 = 0
    for i in range(i0, i1):
        flag = "<<" if r[position, i] == 0.0 else "  "
        n0 += int(r[position, i] == 0.0)
        print(f"    {i * dt:7.3f}  {r[position, i]:+14.6g}  {s[position, i]:+9.4f}  "
              f"valid={int(m[position, i])} {flag}")
    print(f"    raw-zero samples in window: {n0}/{i1 - i0}")


def spectra(signal, shot, n_windows, cache_dir, seg=64):
    """Is the failing modalities' oscillation RESOLVED physics or near-Nyquist / white content?

    Runs only AFTER the zero-fill hypothesis was refuted (the mask is applied correctly and the
    oscillation sits on VALID samples), so the remaining question is whether the content is
    reconstructable at all. Slow-TS loads at target_fs = 100 Hz via
    ``F.interpolate(mode="linear")`` with NO anti-alias filter (data_loader.py step 6), so a
    natively-faster signal decimated to 100 Hz folds its high-frequency content down.

    Everything is computed on the STITCHED consecutive-window trace in the codec's own
    standardized space, per profile position, over VALID samples only:

      * ``ac1`` / ``ac2`` / ``ac5``  — lag-1/2/5 autocorrelation over sample pairs where BOTH
        ends are valid (lag 5 = one 50 ms codec window). ac1 near 0 or negative == white /
        alias-folded content that no 4-token code and no deeper decoder can carry. ac1 near 1
        == a smooth resolved trend.
      * ``nyq``  — the fraction of a Hann-windowed ``seg``-sample periodogram's power in the
        TOP bin (50 Hz, Nyquist). Uniform white noise gives ~1/(seg/2+1) ~ 0.03.
      * ``hf``   — the fraction above 40 Hz (the top 20 % of the band).
      * ``lf``   — the fraction below 10 Hz (the resolved trend).
    """
    Xs, Ms, cfg, _ = _stream(signal, shot, n_windows, cache_dir, True)
    C, T = cfg.channels, Xs.shape[2]
    x = Xs[:, :C].transpose(1, 0, 2).reshape(C, -1)
    m = Ms[:, :C].transpose(1, 0, 2).reshape(C, -1) > 0.5

    acs = {1: [], 2: [], 5: []}
    pw, nseg = np.zeros(seg // 2 + 1), 0
    win = np.hanning(seg)
    for c in range(C):
        v, xc = m[c], x[c]
        if v.sum() < 32:
            continue
        mu = xc[v].mean()
        sd = xc[v].std()
        if sd <= 1e-8:
            continue
        d = (xc - mu) / sd
        for k in acs:
            both = v[:-k] & v[k:]
            if both.sum() >= 16:
                acs[k].append(float((d[:-k][both] * d[k:][both]).mean()))
        # contiguous valid runs -> Hann-windowed periodogram, 50 % overlap
        i = 0
        while i + seg <= len(v):
            if v[i:i + seg].all():
                sgm = d[i:i + seg] - d[i:i + seg].mean()
                P = np.abs(np.fft.rfft(sgm * win)) ** 2
                tot = P.sum()
                if tot > 0:
                    pw += P / tot
                    nseg += 1
                i += seg // 2
            else:
                i += 1
    freqs = np.fft.rfftfreq(seg, d=1.0 / 100.0)
    if nseg:
        P = pw / nseg
        nyq, hf, lf = float(P[-1]), float(P[freqs >= 40].sum()), float(P[freqs <= 10].sum())
    else:
        nyq = hf = lf = float("nan")
    out = {"signal": signal,
           "ac1": float(np.mean(acs[1])) if acs[1] else float("nan"),
           "ac2": float(np.mean(acs[2])) if acs[2] else float("nan"),
           "ac5": float(np.mean(acs[5])) if acs[5] else float("nan"),
           "nyq": nyq, "hf40": hf, "lf10": lf, "n_seg": nseg, "n_chan": len(acs[1])}
    print(f"  {signal:<24}{out['ac1']:>+8.3f}{out['ac2']:>+8.3f}{out['ac5']:>+8.3f}"
          f"{nyq:>9.4f}{hf:>9.4f}{lf:>9.4f}{nseg:>8d}{out['n_chan']:>7d}", flush=True)
    return out


def levels(signal, shot, ckpt, cache_dir, n_windows=400):
    """ACROSS-WINDOW level tracking — the number that matches what the eye judges in the figure.

    ``gate.full_slowts_metrics``' ``slowts_std_ratio_t`` is a WITHIN-window quantity: it is the
    ratio of temporal std over the T=5 samples of ONE 50 ms window, so it mostly measures
    high-frequency wiggle. Measured 2026-09-03 that is NOT what the full-shot figure shows:
    ``ts_tangential_temp`` reads sr_t 0.426 yet its trace overlays the ground truth almost
    exactly, while ``cer_rot`` reads 0.049 and visibly misses a 2.4 s excursion of 2.5 sigma.

    What the eye is judging is the ACROSS-window level: does the reconstruction's per-window
    DC level move with the target's? This computes, per profile position, the std over
    consecutive windows of the per-(window, position) masked mean, for recon and target, and
    reports the variance-weighted ratio (ideal 1.0) plus the correlation of the two level
    series (ideal 1.0). Redrawn windows are EXCLUDED via ``_build_window`` (never ``ds[w]``).
    """
    import importlib.util
    import torch as _t
    _sp = importlib.util.spec_from_file_location(
        "_rcrf", str(Path(__file__).with_name("render_codec_recon_figs.py")))
    _rc = importlib.util.module_from_spec(_sp)
    _sp.loader.exec_module(_rc)
    codec, cfg, _ck = _rc.load_slowts_codec(Path(ckpt))   # reconciles cfg with the WEIGHTS
    ds = tc.SlowTSCodecPairDataset(
        signal, [shot], _cfg(signal, True), data_dir=tc.DEFAULT_DATA_DIR,
        lengths_cache_path=str(Path(cache_dir) / f"probe_{signal}_{shot}_lengths.pt"))
    ds[0]
    C = cfg.channels
    G, R, M = [], [], []
    with _t.no_grad():
        for w in range(min(len(ds), n_windows)):
            it = ds._build_window(w)
            if it is None:
                continue                       # NEVER a redraw
            sw, mw = it
            rec = codec(sw.unsqueeze(0))["recon"][0]
            G.append(sw[:C].numpy()); R.append(rec[:C].numpy()); M.append(mw[:C].numpy())
    G, R, M = np.stack(G), np.stack(R), np.stack(M) > 0.5     # (N, C, T)
    n = M.sum(-1)
    ok = n >= 1
    gl = np.where(ok, (G * M).sum(-1) / np.maximum(n, 1), np.nan)    # (N, C) level series
    rl = np.where(ok, (R * M).sum(-1) / np.maximum(n, 1), np.nan)
    num = den = 0.0
    cors = []
    for c in range(C):
        g, r = gl[:, c], rl[:, c]
        f = np.isfinite(g) & np.isfinite(r)
        if f.sum() < 8:
            continue
        gd, rd = g[f] - g[f].mean(), r[f] - r[f].mean()
        if gd.std() <= 1e-8:
            continue
        num += (rd * rd).sum(); den += (gd * gd).sum()
        if rd.std() > 1e-8:
            cors.append(float((gd * rd).mean() / (gd.std() * rd.std())))
    return {"signal": signal, "level_std_ratio": float(np.sqrt(num / den)) if den > 0 else float("nan"),
            "level_corr": float(np.mean(cors)) if cors else float("nan"),
            "n_windows": int(G.shape[0]), "n_chan": len(cors)}


def code_readout(signal, ckpt, cache_dir, n_fit=1200, n_score=600, alpha=1.0,
                 eval_n_shots=16, n_eval_shots=12, num_workers=4):
    """DECODER-LIMITED vs CODE-LIMITED: an out-of-sample LINEAR read-out of the codec's OWN
    frozen codes, metered with the IDENTICAL function/mask/units as the codec's own row.

    If a ridge regression from the frozen codes beats the codec's trained decoder, the
    information is IN THE CODES and the decoder cannot extract it (decoder-limited). If it does
    not, the codes themselves do not carry it (code-limited) and a deeper decoder cannot help.

    DESIGN MATRIX: the FSQ code is (n_tok, fsq_dim) level indices, so the natural linear basis
    is a one-hot of every (token, dim, level) triple — 4 x (8+5+5+5) = 92 columns for the
    default [8,5,5,5] slow-TS codec, plus an intercept. That is small enough to fit
    out-of-sample on ~1200 windows without regularisation doing the work, and it is a STRICTLY
    LESS expressive map than the 4-layer transformer decoder it is compared against (it cannot
    even represent an interaction between two tokens), so beating the decoder with it is a
    strong statement.

    Fit shots and score shots are DISJOINT and both held out from training. numpy lstsq on the
    host (torch.linalg on this ROCm build SIGKILLs silently).
    """
    import importlib.util
    import torch as _t
    _sp = importlib.util.spec_from_file_location(
        "_rcrf2", str(Path(__file__).with_name("render_codec_recon_figs.py")))
    _rc = importlib.util.module_from_spec(_sp)
    _sp.loader.exec_module(_rc)
    from tokamak_foundation_model.ignite import gate as gate_mod
    from tokamak_foundation_model.ignite import spike as spike_mod

    codec, cfg, _ck = _rc.load_slowts_codec(Path(ckpt))
    pool = spike_mod.discover_shots(tc.DEFAULT_DATA_DIR)[-eval_n_shots:][:n_eval_shots]
    fit_shots, score_shots = pool[4:12], pool[:4]      # same split as the PCA linear floor
    levels = list(cfg.fsq_levels)
    off, tot = [], 0
    for L in levels:
        off.append(tot); tot += L
    F = int(cfg.n_tok) * tot

    def _collect(shots, n, tag):
        dl, _ = _rc._slowts_load_windows(signal, cfg, shots, n, 32, num_workers, tag)
        X, Y, M = [], [], []
        with _t.no_grad():
            for sig, mask in dl:
                _q, codes = codec.quantize(codec.encode(sig))     # (B, n_tok, fsq_dim)
                c = codes.numpy().astype(int)
                B = c.shape[0]
                oh = np.zeros((B, F), dtype=np.float64)
                for tkn in range(c.shape[1]):
                    for dim in range(c.shape[2]):
                        oh[np.arange(B), tkn * tot + off[dim] + c[:, tkn, dim]] = 1.0
                X.append(oh); Y.append(sig.numpy()); M.append(mask.numpy())
        return (np.concatenate(X), np.concatenate(Y).astype(np.float64),
                np.concatenate(M).astype(np.float64))

    Xf, Yf, Mf = _collect(fit_shots, n_fit, "rofit")
    Xs, Ys, Ms = _collect(score_shots, n_score, "roscore")
    C, T = Yf.shape[1], Yf.shape[2]
    Yf2 = Yf.reshape(Yf.shape[0], -1); Mf2 = (Mf.reshape(Mf.shape[0], -1) > 0.5)
    Xf1 = np.hstack([Xf, np.ones((Xf.shape[0], 1))])
    Xs1 = np.hstack([Xs, np.ones((Xs.shape[0], 1))])
    # per-output masked ridge: solve (X'WX + aI) b = X'W y  column by column (W = validity)
    D = Xf1.shape[1]
    Wt = np.zeros((D, Yf2.shape[1]))
    XtX_all = Xf1.T @ Xf1
    for j in range(Yf2.shape[1]):
        w = Mf2[:, j].astype(np.float64)
        if w.sum() < D:                                  # too few valid rows -> predict its mean
            Wt[-1, j] = (Yf2[:, j] * w).sum() / max(w.sum(), 1.0)
            continue
        Xw = Xf1 * w[:, None]
        A = Xw.T @ Xf1 + alpha * np.eye(D)
        Wt[:, j] = np.linalg.lstsq(A, Xw.T @ Yf2[:, j], rcond=None)[0]
    pred = (Xs1 @ Wt).reshape(-1, C, T)
    m = gate_mod.full_slowts_metrics(pred, Ys, mask=Ms)
    with _t.no_grad():
        own_rec = _t.cat([codec(_t.tensor(Ys[i:i + 32], dtype=_t.float32))["recon"]
                          for i in range(0, Ys.shape[0], 32)]).numpy()
    own = gate_mod.full_slowts_metrics(own_rec, Ys, mask=Ms)
    return {"signal": signal, "ckpt": str(ckpt), "n_fit": int(Xf.shape[0]),
            "n_score": int(Xs.shape[0]), "n_feat": D,
            "readout_pooled": m["slowts_nrmse_pooled"], "readout_corr": m["slowts_corr"],
            "readout_sr_t": m["slowts_std_ratio_t"],
            "decoder_pooled": own["slowts_nrmse_pooled"], "decoder_corr": own["slowts_corr"],
            "decoder_sr_t": own["slowts_std_ratio_t"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shot", default="204990")
    ap.add_argument("--n_windows", type=int, default=220)
    ap.add_argument("--signals", nargs="+", default=list(SIGNALS))
    ap.add_argument("--cache_dir", default=str(REPO / "eval_runs" / "codec_recon_figs" / "_cache"))
    ap.add_argument("--trace", default=None, help="signal to dump a sample-by-sample trace for")
    ap.add_argument("--spectra", action="store_true",
                    help="lag-1/2/5 autocorrelation + Nyquist/HF power share per modality "
                         "(the RESOLVED-vs-ALIASED test; run only after the zero-fill "
                         "hypothesis is refuted)")
    ap.add_argument("--seg", type=int, default=64, help="periodogram segment length (--spectra)")
    ap.add_argument("--readout", default=None,
                    help="LINEAR CODE READ-OUT: checkpoint template with {signal}. Fits a ridge "
                         "from the codec's own FROZEN codes to the target out-of-sample and "
                         "prints it beside that same codec's trained-decoder score "
                         "(decoder-limited vs code-limited).")
    ap.add_argument("--alpha", type=float, default=1.0, help="ridge alpha for --readout")
    ap.add_argument("--levels", default=None,
                    help="ACROSS-WINDOW level tracking: a checkpoint path template containing "
                         "{signal} (e.g. eval_runs/ignite_slowts_all7/{signal}/codec_last.pt)")
    ap.add_argument("--position", type=int, default=None)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=1.8)
    a = ap.parse_args()
    if a.trace:
        trace(a.trace, a.shot, a.position, a.t0, a.t1, a.cache_dir, a.n_windows)
        raise SystemExit(0)
    if a.readout:
        print(f"\nOUT-OF-SAMPLE LINEAR READ-OUT of the codec's OWN FROZEN CODES "
              f"(ridge alpha={a.alpha}, one-hot (token,dim,level) basis)")
        print("  readout < decoder => the info IS in the codes and the DECODER is the limit")
        print(f"  {'signal':<24}{'readout':>9}{'decoder':>9}{'delta':>8}"
              f"{'ro_corr':>9}{'dec_corr':>9}{'ro_sr_t':>9}{'dec_sr_t':>9}{'nfeat':>7}")
        print("  " + "-" * 96)
        for sg in a.signals:
            try:
                r = code_readout(sg, a.readout.format(signal=sg), a.cache_dir, alpha=a.alpha)
                d = r["readout_pooled"] - r["decoder_pooled"]
                print(f"  {sg:<24}{r['readout_pooled']:>9.4f}{r['decoder_pooled']:>9.4f}"
                      f"{d:>+8.4f}{r['readout_corr']:>9.3f}{r['decoder_corr']:>9.3f}"
                      f"{r['readout_sr_t']:>9.3f}{r['decoder_sr_t']:>9.3f}{r['n_feat']:>7d}",
                      flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  {sg:<24} FAILED {e!r}")
        raise SystemExit(0)
    if a.levels:
        print(f"\nACROSS-WINDOW LEVEL TRACKING (shot {a.shot}, redrawn windows EXCLUDED) — "
              f"the quantity the full-shot figure actually shows")
        print(f"  {'signal':<24}{'lvl_std_ratio':>15}{'lvl_corr':>10}{'nwin':>7}{'nch':>6}")
        print("  " + "-" * 62)
        for sg in a.signals:
            try:
                r = levels(sg, a.shot, a.levels.format(signal=sg), a.cache_dir, a.n_windows)
                print(f"  {sg:<24}{r['level_std_ratio']:>15.3f}{r['level_corr']:>10.3f}"
                      f"{r['n_windows']:>7d}{r['n_chan']:>6d}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  {sg:<24} FAILED {e!r}")
        raise SystemExit(0)
    if a.spectra:
        print(f"\nAUTOCORRELATION + SPECTRAL SHARE (shot {a.shot}, standardized space, VALID "
              f"samples only, {a.seg}-sample Hann periodogram at 100 Hz)")
        print("  white noise reference: ac1 ~ 0.000, nyq ~ 0.030, hf40 ~ 0.20, lf10 ~ 0.20")
        print(f"  {'signal':<24}{'ac1':>8}{'ac2':>8}{'ac5':>8}{'nyq':>9}{'hf40':>9}"
              f"{'lf10':>9}{'nseg':>8}{'nch':>7}")
        print("  " + "-" * 90)
        for sg in a.signals:
            spectra(sg, a.shot, a.n_windows, a.cache_dir, seg=a.seg)
        raise SystemExit(0)
    rows = [probe(s, a.shot, a.n_windows, a.cache_dir) for s in a.signals]
    print("\n" + "=" * 104)
    print(f"{'signal':<24}{'present':>9}{'raw0':>8}{'zero&valid':>12}{'ofValid':>9}"
          f"{'varShare':>10}{'std':>8}{'std|raw0 masked':>17}")
    print("-" * 104)
    for r in rows:
        print(f"{r['signal']:<24}{r['present_frac']:>9.4f}{r['raw_zero_frac']:>8.4f}"
              f"{r['zero_but_valid_frac']:>12.4f}{r['zbv_share_of_valid']:>9.4f}"
              f"{r['variance_share']:>10.4f}{r['std_as_is']:>8.4f}{r['std_raw0_masked']:>17.4f}")
