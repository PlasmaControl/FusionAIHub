"""Residual-FSQ mode-prediction proof render (spectro-only overfit model).

The production ``--comparison_figure`` renderer needs the full multimodal model
(video panels); the overfit is spectro-only, so this focused proof reuses the
REAL trained model's 1-window-ahead prediction via ``forward_batch`` (residual
space, since the head self-declares bg_subtract) and shows, for the strongest
mode channel of a mode shot:

    row 0  GT residual (next window)      -- the true modes
    row 1  codec recon-ceiling (residual) -- what the frozen codec can represent
    row 2  MODEL prediction (residual)    -- 1-window-ahead world-model output

columns = the top-N real (non-padding) mode windows. Reports the mode-band
(0-60 kHz) correlation model-vs-GT and the codec ceiling, so the figure is not
judged by eye alone. This is the honest test of the week-long problem: does the
world model predict the coherent modes (not just the broadband envelope)?
"""
import os
import sys
from pathlib import Path

FMH = "/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub"
for p in (f"{FMH}/src", f"{FMH}/scripts/training"):
    if p not in sys.path:
        sys.path.insert(0, p)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from eval_e2e_animation_tokamak import load_model
from train_e2e_stage1 import build_datasets, forward_batch, _core
from tokamak_foundation_model.data.data_loader import collate_fn

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = os.environ.get("CKPT", f"/lustre/orion/fus187/proj-shared/models/e2e_resid_overfit/e2e_stage1_best.pt")
MODS = os.environ.get("MODALITIES", "ece,co2").split(",")
SHOTS = os.environ.get("SHOTS", "200729,190996,204811").split(",")
NCOL = int(os.environ.get("NCOL", "5"))
OUT = Path(os.environ.get("OUT_DIR", f"{FMH}/eval_runs/comparison/resid_overfit_proof"))
OUT.mkdir(parents=True, exist_ok=True)
FS, NFFT, HOP = 500_000.0, 1024, 256
fmax = int(60 / (FS / NFFT / 1e3))          # 0-60 kHz band

model, ckpt = load_model(Path(CKPT), dev)
model.eval()
core = _core(model)
# FSQ code-sampling temperature override (eval already SAMPLES codes; higher T
# flattens the peaked code distribution → more variance → sharper modes, at the
# risk of incoherence). Set SAMPLE_TEMP to sweep.
_SAMPLE_TEMP = os.environ.get("SAMPLE_TEMP")
if _SAMPLE_TEMP is not None:
    for _h in core.diag_heads.values():
        if hasattr(_h, "sample_temperature"):
            _h.sample_temperature = float(_SAMPLE_TEMP)
    print(f"[proof] SAMPLE_TEMP override = {_SAMPLE_TEMP}", flush=True)
a = ckpt["args"]
dn = [d["name"] for d in ckpt["diagnostics"]]
an = [c["name"] for c in ckpt["actuators"]]
dd = Path(a["data_dir"])
stats = torch.load(a["stats_path"], weights_only=False)
sfiles = [dd / f"{s}_processed.h5" for s in SHOTS]
sfiles = [f for f in sfiles if f.exists()]
print(f"[proof] ckpt={CKPT}\n[proof] shots={[f.stem for f in sfiles]} mods={MODS}", flush=True)

_, ds = build_datasets(dd, sfiles, sfiles, stats, a["chunk_duration_s"],
                       a.get("prediction_horizon_s", a["chunk_duration_s"]),
                       a["step_size_s"], a["warmup_s"], dn, an,
                       Path(f"{FMH}/eval_runs/modecode_cache"),
                       history_windows=int(a.get("history_windows", 1)))
ld = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate_fn)

# Accumulate GT / model-pred / codec-recon / persistence(input window) per modality.
GT = {m: [] for m in MODS}
PR = {m: [] for m in MODS}
RC = {m: [] for m in MODS}
IN = {m: [] for m in MODS}
with torch.no_grad():
    for batch in ld:
        preds, din, targets, _, _ = forward_batch(model, batch, dev)
        for m in MODS:
            if m not in targets:
                continue
            head = core.diag_heads[m]
            GT[m].append(targets[m].float().cpu())
            PR[m].append(preds[m].float().cpu())
            # codec ceiling only exists for codec-based heads (FSQ/MaskGIT);
            # generative SpectrogramFlowHead has no decode/encode_target → skip.
            if hasattr(head, "decode") and hasattr(head, "encode_target"):
                rec = head.decode(head.encode_target(targets[m]))
                RC[m].append(rec.float().cpu())
            # persistence = the INPUT (current) window, time-cropped to match target
            xin = din[m].float()
            if xin.dim() == targets[m].dim() + 1:   # multi-window: last input window
                xin = xin[:, -1]
            if xin.shape[-1] != targets[m].shape[-1]:
                xin = xin[..., :targets[m].shape[-1]]
            IN[m].append(xin.cpu())

FREQ = np.arange(NFFT // 2 + 1) * FS / NFFT / 1e3


def _bc(a, b, lo, hi):
    aa, bb = a[lo:hi].ravel(), b[lo:hi].ravel()
    if aa.std() < 1e-9 or bb.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


DC_HI = max(1, int(5.0 / (FS / NFFT / 1e3)))                 # <5 kHz = smooth low-freq residual
MODE_LO, MODE_HI = DC_HI, int(40.0 / (FS / NFFT / 1e3))      # 5-40 kHz = the coherent modes

for m in MODS:
    if not GT[m]:
        print(f"[proof] {m}: no windows", flush=True)
        continue
    g = torch.cat(GT[m], 0).numpy()     # (N,C,F,T) GT next window
    p = torch.cat(PR[m], 0).numpy()     # model prediction
    r = torch.cat(RC[m], 0).numpy() if RC[m] else None   # codec ceiling (None for generative)
    q = torch.cat(IN[m], 0).numpy()     # input window = PERSISTENCE baseline
    N, C, F, T = g.shape
    ch = int(np.argmax(np.abs(g[:, :, :fmax]).sum(axis=(0, 2, 3))))
    wstd = g[:, ch, :fmax].reshape(N, -1).std(1)
    real = wstd > np.median(wstd)
    idx = np.where(real)[0]
    # Rank windows by SUSTAINED mode strength: a sharp peak above the smooth
    # baseline (isolates the coherent mode from DC/broadband) that PERSISTS across
    # the time axis (min over time, not max) — this excludes both mode-free/noisy
    # windows AND transient single-frame bursts (ELM onsets are unpredictable
    # 1-step, so they'd unfairly tank every model incl. persistence). We want the
    # steady, physically-forecastable modes the user pointed at.
    from scipy.ndimage import gaussian_filter1d as _gf0
    def _gt_sustained(w):
        s = np.abs(g[w, ch, MODE_LO:MODE_HI])                 # (Fband, T)
        base = _gf0(s, 6.0, axis=0)                           # smooth over freq
        prom = (s - base).clip(min=0).max(0)                  # peak prominence per time-frame
        return float(np.percentile(prom, 25))                # sustained (lower-quartile over time)
    mode_order = sorted(idx.tolist(), key=lambda w: -_gt_sustained(w))
    order = mode_order[:NCOL]                      # windows shown in the figure
    strong = mode_order[:max(NCOL, min(len(mode_order), 12))]  # sustained-mode subset
    # THE HONEST TEST: is the correlation in the MODE band (5-40 kHz) or only near DC?
    def _profc(a, b, w):
        # time-averaged mode-band frequency profile correlation: does the pred put
        # a mode ridge at the SAME frequency as GT? (the achievable target; the
        # exact 2D pattern is unpredictable 1-step, persistence ceiling ~0.4)
        pa = np.abs(a[w, ch, MODE_LO:MODE_HI]).mean(1)
        pb = np.abs(b[w, ch, MODE_LO:MODE_HI]).mean(1)
        if pa.std() < 1e-9 or pb.std() < 1e-9:
            return np.nan
        return float(np.corrcoef(pa, pb)[0, 1])

    # MODE-CAPTURE (matches the eye): subtract the smooth baseline to isolate the
    # peaks, find GT's mode peak freq, measure how much of ITS prominence the
    # prediction has AT THAT FREQ. Immune to the shared low-freq slope that fooled
    # peak-match/profile-corr. Flat/missed prediction -> ~0. This is THE metric.
    from scipy.ndimage import gaussian_filter1d as _gf
    def _capture(gg, pp, w):
        gp = np.abs(gg[w, ch, MODE_LO:MODE_HI]).mean(1)
        pf = np.abs(pp[w, ch, MODE_LO:MODE_HI]).mean(1)
        gd = gp - _gf(gp, 6.0)          # GT prominence above smooth baseline
        pd = pf - _gf(pf, 6.0)          # pred prominence
        f0 = int(np.argmax(gd))         # GT mode peak
        if gd[f0] < 1e-6:
            return np.nan
        return float(pd[f0] / gd[f0])   # fraction of GT mode captured at its freq
    capture = float(np.nanmedian([_capture(g, p, w) for w in idx]))
    capture_pers = float(np.nanmedian([_capture(g, q, w) for w in idx]))
    # On the STRONGEST-mode windows (where a coherent mode actually exists), does
    # the model capture it — and does it BEAT persistence (i.e. the ridge got
    # moved/sharpened to the right place, not just copied)? This is the number
    # that answers the user's "orange must overlay black" on the real modes.
    capture_strong = float(np.nanmedian([_capture(g, p, w) for w in strong]))
    capture_pers_strong = float(np.nanmedian([_capture(g, q, w) for w in strong]))
    beats = capture_strong > capture_pers_strong + 0.05
    # CODEC CEILING capture: the BEST the FSQ pipeline could do — encode the
    # GROUND-TRUTH mode → codes → decode. If this is high, the codec CAN show
    # modes and the world model's low capture is a PREDICTION problem; if this
    # is also ~0, the frozen codec itself cannot represent the mode amplitude.
    capture_codec_strong = (float(np.nanmedian([_capture(g, r, w) for w in strong]))
                            if r is not None else float("nan"))
    # DIAGNOSTIC: is persistence's gap FREQUENCY-drift (warp fixes) or AMPLITUDE
    # (warp does NOT fix — the mode grows over the horizon)? Measure, on the
    # sustained-mode windows, persistence's peak-freq drift (kHz) and its
    # amplitude ratio at the GT peak. Small drift + low amp-ratio => amplitude
    # is the bottleneck, not frequency.
    def _drift_amp(w):
        gp = np.abs(g[w, ch, MODE_LO:MODE_HI]).mean(1); gd = gp - _gf0(gp, 6.0)
        qp = np.abs(q[w, ch, MODE_LO:MODE_HI]).mean(1); qd = qp - _gf0(qp, 6.0)
        f_gt = int(np.argmax(gd)); f_in = int(np.argmax(qd))
        drift = abs(f_gt - f_in) * (FS / NFFT / 1e3)          # kHz
        amp = float(qp[f_gt] / (gp[f_gt] + 1e-9))             # raw amp ratio at GT peak
        return drift, amp
    _da = [_drift_amp(w) for w in strong]
    drift_kHz = float(np.median([d for d, _ in _da]))
    amp_ratio = float(np.median([a for _, a in _da]))
    fprof = float(np.nanmedian([_profc(g, p, w) for w in idx]))       # MODEL: mode-frequency prediction
    fprof_c = (float(np.nanmedian([_profc(g, r, w) for w in idx]))    # CODEC CEILING (max achievable)
               if r is not None else float("nan"))
    pers = float(np.nanmedian([_profc(g, q, w) for w in idx]))        # PERSISTENCE baseline (copy input)
    dc = float(np.nanmedian([_bc(g[w, ch], p[w, ch], 0, DC_HI) for w in idx]))
    mode = float(np.nanmedian([_bc(g[w, ch], p[w, ch], MODE_LO, MODE_HI) for w in idx]))
    tvr = float(p[real][:, ch, :fmax].var(-1).mean() / (g[real][:, ch, :fmax].var(-1).mean() + 1e-9))
    tvr_codec = (float(r[real][:, ch, :fmax].var(-1).mean()
                       / (g[real][:, ch, :fmax].var(-1).mean() + 1e-9))
                 if r is not None else float("nan"))

    # CHECKERBOARD-ROBUST metric: does the model's dominant mode-band peak land on
    # the GT mode's (shot-varying) frequency? A fixed patch-grid checkerboard peak
    # can't track a mode that sits at different freqs on different shots, so it
    # cannot score here — this is immune to the ConvTranspose artifact.
    tol = max(1, int(2.0 / (FS / NFFT / 1e3)))          # ~2 kHz
    def _peakf(a, w):
        return MODE_LO + int(np.argmax(np.abs(a[w, ch, MODE_LO:MODE_HI]).mean(1)))
    pk_model = float(np.mean([abs(_peakf(g, w) - _peakf(p, w)) <= tol for w in idx]))
    pk_pers = float(np.mean([abs(_peakf(g, w) - _peakf(q, w)) <= tol for w in idx]))

    # PASS requires ALL of: (1) checkerboard-proof tracking near persistence,
    # (2) profile-corr at least matching persistence, and CRUCIALLY (3) VISIBLE
    # amplitude — tvr in [0.6, 1.6] (dampened <0.6 = not visible; >1.6 = noise).
    # (3) is the fix for the "PASS but I can't see it" failure.
    # PASS = actually CAPTURES the mode peak (prominence at GT freq >= half) AND
    # it's visible (tvr in range). This is the metric that matches the eye.
    # Headline judgment uses the SUSTAINED-mode-window capture (capture_strong) —
    # the all-real-windows median is inflated by noisy near-mode-free windows
    # (ratio of two small numbers). The real question is: on the windows that
    # actually carry a steady mode, does the model reproduce it (>= half of GT's
    # prominence) AND is it visible (tvr in range)?
    passed = (capture_strong >= 0.5) and (0.6 <= tvr <= 1.6)
    why = []
    if capture_strong < 0.5: why.append(f"MISSES mode peak (sustained capture={capture_strong:.2f})")
    if tvr < 0.6: why.append("DAMPENED(not visible)")
    if tvr > 1.6: why.append("noise")
    verdict = ("PASS: CAPTURES mode peak, visible" if passed else "FAIL: " + ", ".join(why))
    if passed and beats: verdict += " + BEATS persistence"
    print(f"[RANK] {m} ch{ch}: MODE-CAPTURE(model)={capture:.2f} vs persist={capture_pers:.2f} [prominence @ GT peak] "
          f"|| peak-match={pk_model:.2f} profile-corr={fprof:.2f}(pers {pers:.2f}) | tvr={tvr:.3f} (codec-ceiling tvr={tvr_codec:.2f})  ==> {verdict}",
          flush=True)
    print(f"[RANK] {m} ch{ch}: STRONG-MODE windows (n={len(strong)}): "
          f"capture model={capture_strong:.2f} vs persist={capture_pers_strong:.2f} "
          f"==> {'BEATS persistence' if beats else ('ties persistence' if capture_strong>=capture_pers_strong-0.05 else 'BELOW persistence')}",
          flush=True)
    print(f"[RANK] {m} ch{ch}: GAP DIAGNOSIS (persistence): peak-freq drift={drift_kHz:.1f} kHz | "
          f"amp-ratio@GTpeak={amp_ratio:.2f} ==> {'FREQUENCY-drift dominant (warp helps)' if drift_kHz > 1.5 else 'AMPLITUDE-undershoot dominant (warp will NOT help; need amplitude prediction)'}",
          flush=True)
    print(f"[RANK] {m} ch{ch}: CODEC-CEILING capture (strong)={capture_codec_strong:.2f} vs model={capture_strong:.2f} "
          f"==> {'CODEC caps it (fix CODEC)' if (capture_codec_strong==capture_codec_strong and capture_codec_strong < 0.4) else ('codec OK, model under-predicts (fix PREDICTION)' if capture_codec_strong==capture_codec_strong else 'n/a (non-codec head)')}",
          flush=True)

    # Figure: GT (OWN scale) | PRED (OWN scale) | freq-profile overlay — removes the
    # shared-scale washout so dampened-but-present is distinguishable from truly flat.
    ncol = max(1, len(order))
    # Image rows: GT, [CODEC-ceiling if available], MODEL; last row = profile overlay.
    img_rows = [("GT residual", g)]
    if r is not None:
        img_rows.append(("CODEC-ceiling\n(decode(encode(GT)))", r))
    img_rows.append(("MODEL pred", p))
    n_img = len(img_rows)
    prow = n_img
    fig, ax = plt.subplots(n_img + 1, ncol, figsize=(3.2 * ncol, 2.9 * (n_img + 1)), squeeze=False)
    ext = (0, T * HOP / FS * 1e3, 0, FREQ[fmax - 1])
    for j, w in enumerate(order):
        for i, (lab, d) in enumerate(img_rows):
            vmn, vmx = np.percentile(d[w, ch, :fmax], [2, 98])   # OWN per-panel scale
            ax[i, j].imshow(d[w, ch, :fmax], origin="lower", aspect="auto", cmap="magma",
                            vmin=vmn, vmax=vmx, extent=ext)
            if j == 0:
                ax[i, j].set_ylabel(f"{lab}\nFreq (kHz)", fontsize=9)
            if i == 0:
                ax[i, j].set_title(f"win {w}", fontsize=8)
        ax[prow, j].plot(FREQ[:fmax], np.abs(g[w, ch, :fmax]).mean(1), lw=1.4, color="k", label="GT")
        ax[prow, j].plot(FREQ[:fmax], np.abs(q[w, ch, :fmax]).mean(1), lw=1.0, color="tab:green", label="persistence")
        if r is not None:
            ax[prow, j].plot(FREQ[:fmax], np.abs(r[w, ch, :fmax]).mean(1), lw=1.0, color="tab:purple", label="codec-ceiling")
        ax[prow, j].plot(FREQ[:fmax], np.abs(p[w, ch, :fmax]).mean(1), lw=1.2, color="tab:orange", label="MODEL")
        ax[prow, j].axvspan(FREQ[MODE_LO], FREQ[MODE_HI], color="k", alpha=0.07)
        ax[prow, j].set_xlabel("Freq (kHz)")
        if j == 0:
            ax[prow, j].set_ylabel("|residual| time-avg")
            ax[prow, j].legend(fontsize=7)
    tag = "PASS" if passed else "FAIL"
    fig.suptitle(f"[{tag}] {m} ch{ch}  |  MODE-CAPTURE model={capture:.2f} persist={capture_pers:.2f} "
                 f"(prominence @ GT peak)  |  tvr={tvr:.2f} peak-match={pk_model:.2f} profile-corr={fprof:.2f}",
                 fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for e in ("png", "pdf"):
        fig.savefig(OUT / f"resid_proof_{m}.{e}", dpi=130, bbox_inches="tight")
    print(f"[proof] saved {OUT}/resid_proof_{m}.png", flush=True)
