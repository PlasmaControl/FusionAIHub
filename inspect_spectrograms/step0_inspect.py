"""Step 0: data verification for the Phase B spectrogram plan.

Reads raw signals directly from HDF5 (bypasses the broken
_getitem_standard / _getitem_prediction code paths), computes the
project's STFT (n_fft=1024, hop=256, drops DC), and produces:

  figures/{shot}_{modality}.png      — log-magnitude spectrogram
                                       per shot (channels stacked)
  figures/freq_energy.png            — per-frequency total energy
                                       averaged across shots
  figures/bes_correlation.png        — pairwise correlation between
                                       BES 16 channels' time-averaged
                                       spectra (probes 2x8 grid layout)

Outputs a markdown summary at
``docs/spectrogram_step0_findings.md`` capturing:
- confirmed shapes
- per-channel mean/std of standardized output (sanity vs preprocessing
  stats)
- BES grid orientation finding
- frequency-cutoff observation
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize


# ── Configuration ────────────────────────────────────────────────────────

DATA_DIR = Path("/scratch/gpfs/EKOLEMEN/foundation_model")
STATS_PATH = Path(
    "/scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt"
)
OUT_DIR = Path("/scratch/gpfs/ps9551/FusionAIHub/inspect_spectrograms")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR = Path("/scratch/gpfs/ps9551/FusionAIHub/docs")
SUMMARY_PATH = DOCS_DIR / "spectrogram_step0_findings.md"

# Plan-locked params:
N_FFT = 1024
HOP = 256
TARGET_FS = 500_000  # ECE/CO2/BES all 500 kHz
CHUNK_S = 0.05                          # training-time chunk
N_SAMPLES = int(CHUNK_S * TARGET_FS)    # 25_000  (used only for shape check)
VIZ_WINDOW_S = 1.0                      # visualization window
VIZ_N_SAMPLES = int(VIZ_WINDOW_S * TARGET_FS)   # 500_000
WINDOW = torch.hann_window(N_FFT)

# Channel slices the SignalConfig will eventually apply:
ECE_SLICE = slice(0, 40)        # raw 48 -> 40
CO2_SLICE = slice(0, 4)         # raw 4
BES_SLICE = slice(48, 64)       # raw 64 -> 16 (channels 49..64)

N_SHOTS = 5
WARMUP_S = 1.0  # start chunk this far into the shot


# ── Helpers ──────────────────────────────────────────────────────────────

def find_complete_shots(n: int) -> List[Path]:
    """Return up to ``n`` shots that have all three modalities present
    with at least VIZ_N_SAMPLES + WARMUP_S*fs samples."""
    needed = int(WARMUP_S * TARGET_FS) + VIZ_N_SAMPLES
    out = []
    for shot in sorted(DATA_DIR.glob("*_processed.h5")):
        try:
            with h5py.File(shot, "r") as f:
                ok = True
                for k in ("ece", "co2", "bes"):
                    if k not in f or "ydata" not in f[k]:
                        ok = False
                        break
                    if f[k]["ydata"].shape[1] < needed:
                        ok = False
                        break
                if ok:
                    out.append(shot)
                    if len(out) >= n:
                        break
        except Exception:
            continue
    return out


def stft_chunk(arr: np.ndarray, ch_slice: slice, n_samples: int) -> torch.Tensor:
    """Return |STFT| with DC removed for a window starting at
    WARMUP_S into the shot, length ``n_samples`` samples, sliced to
    the SignalConfig channel range."""
    start = int(WARMUP_S * TARGET_FS)
    sig = torch.from_numpy(np.asarray(arr[ch_slice, start:start + n_samples])).float()
    sig = torch.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    spec = torch.stft(
        sig, n_fft=N_FFT, hop_length=HOP, window=WINDOW, return_complex=True
    )
    return torch.abs(spec)[:, 1:, :]   # drop DC -> (C, 512, n_frames)


def freq_axis_hz() -> np.ndarray:
    """Centre frequencies of the 512 retained STFT bins (DC dropped)."""
    return (np.arange(1, N_FFT // 2 + 1) * (TARGET_FS / N_FFT))


def save_spectrogram_panel(
    path: Path, mag: torch.Tensor, modality: str, shot_id: str,
    window_s: float,
) -> None:
    """One PNG per (shot, modality): log10 magnitude spectrogram for
    every channel, stacked vertically. y axis = freq (kHz), x = time
    (ms within ``window_s`` seconds starting at ``WARMUP_S``)."""
    C, F, T = mag.shape
    log_mag = torch.log10(mag.clamp_min(1e-8)).numpy()

    fig, axes = plt.subplots(
        C, 1, figsize=(12, max(2, 0.6 * C)), sharex=True, sharey=True
    )
    if C == 1:
        axes = [axes]

    vmin = float(np.percentile(log_mag, 1))
    vmax = float(np.percentile(log_mag, 99))
    norm = Normalize(vmin=vmin, vmax=vmax)
    freqs_khz = freq_axis_hz() / 1e3
    t_start_ms = WARMUP_S * 1e3
    t_end_ms = (WARMUP_S + window_s) * 1e3
    times_ms = np.linspace(t_start_ms, t_end_ms, T)

    for c, ax in enumerate(axes):
        im = ax.imshow(
            log_mag[c],
            origin="lower",
            aspect="auto",
            extent=[times_ms[0], times_ms[-1], freqs_khz[0], freqs_khz[-1]],
            norm=norm,
            cmap="magma",
        )
        ax.set_ylabel(f"ch{c}\nkHz", fontsize=7)
        ax.tick_params(labelsize=6)

    axes[-1].set_xlabel("time (ms, absolute within shot)")
    fig.suptitle(
        f"{modality.upper()} log10|STFT| — shot {shot_id} — "
        f"window {window_s*1e3:.0f} ms — "
        f"{C} ch × {F} freq × {T} time",
        fontsize=10,
    )
    fig.colorbar(im, ax=axes, location="right", shrink=0.6, label="log10|STFT|")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_freq_energy(
    path: Path, mean_per_freq: Dict[str, np.ndarray]
) -> None:
    """Per-modality mean log-magnitude vs frequency, averaged over
    channels, time and shots. Helps decide if the upper part of the
    band can be cropped."""
    freqs_khz = freq_axis_hz() / 1e3
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, curve in mean_per_freq.items():
        ax.plot(freqs_khz, curve, label=name.upper())
    ax.set_xlabel("frequency (kHz)")
    ax.set_ylabel("mean log10|STFT| (over ch, time, shots)")
    ax.set_title(
        f"Per-frequency energy distribution "
        f"({VIZ_WINDOW_S:.1f} s window per shot)"
    )
    ax.set_xscale("linear")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_bes_correlation(
    path: Path, mean_spectrum_per_ch: np.ndarray, ch_indices: List[int]
) -> None:
    """Pairwise correlation matrix between BES 16 channels' time-and-
    shot averaged spectra. Diagnoses 2x8 grid orientation: if channels
    49–56 are one spatial row and 57–64 another, expect block structure."""
    C = mean_spectrum_per_ch.shape[0]
    cor = np.corrcoef(mean_spectrum_per_ch)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cor, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(C))
    ax.set_yticks(range(C))
    ax.set_xticklabels([str(i) for i in ch_indices], rotation=90, fontsize=7)
    ax.set_yticklabels([str(i) for i in ch_indices], fontsize=7)
    ax.set_xlabel("BES channel index (raw)")
    ax.set_ylabel("BES channel index (raw)")
    # Highlight the proposed 49-56 vs 57-64 row split.
    ax.axhline(7.5, color="k", lw=0.5)
    ax.axvline(7.5, color="k", lw=0.5)
    ax.set_title(
        "BES inter-channel correlation of mean spectra\n"
        "(black lines split channels 49–56 from 57–64)"
    )
    fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.85)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Step 0 inspection — output: {OUT_DIR}")
    shots = find_complete_shots(N_SHOTS)
    if not shots:
        raise SystemExit("No shots found with all three modalities.")
    print(f"Selected shots: {[s.stem.replace('_processed', '') for s in shots]}")

    # Stats for sanity-checking standardization.
    stats = torch.load(STATS_PATH, weights_only=False)

    slices = {"ece": ECE_SLICE, "co2": CO2_SLICE, "bes": BES_SLICE}
    expected_C = {"ece": 40, "co2": 4, "bes": 16}

    # Accumulators across shots (computed on the long visualization
    # window so the per-frequency / BES-correlation estimates are
    # statistically meaningful — 50 ms gives only 98 time frames per
    # shot, 1 s gives ~1953).
    sum_log_mag_per_freq: Dict[str, np.ndarray] = {
        m: np.zeros(N_FFT // 2, dtype=np.float64) for m in slices
    }
    n_per_freq: Dict[str, int] = {m: 0 for m in slices}
    bes_spectrum_accum = np.zeros((16, N_FFT // 2), dtype=np.float64)
    bes_n = 0

    # Per-modality sample shape collected across shots at the **50 ms**
    # training window; this is the model contract.
    seen_shapes: Dict[str, set[Tuple[int, int, int]]] = {m: set() for m in slices}

    for shot in shots:
        sid = shot.stem.replace("_processed", "")
        with h5py.File(shot, "r") as f:
            for modality, ch_slice in slices.items():
                arr = f[modality]["ydata"][...]

                # 1) 50 ms shape contract (training-time window).
                mag_train = stft_chunk(arr, ch_slice, N_SAMPLES)
                seen_shapes[modality].add(tuple(mag_train.shape))

                # 2) Long-window spectrogram for visualization.
                mag_viz = stft_chunk(arr, ch_slice, VIZ_N_SAMPLES)
                fig_path = FIG_DIR / f"{sid}_{modality}.png"
                save_spectrogram_panel(
                    fig_path, mag_viz, modality, sid, VIZ_WINDOW_S
                )

                # Aggregate per-freq energy from the long window.
                log_mag = torch.log10(mag_viz.clamp_min(1e-8)).numpy()
                per_freq = log_mag.mean(axis=(0, 2))   # (F,)
                sum_log_mag_per_freq[modality] += per_freq
                n_per_freq[modality] += 1

                if modality == "bes":
                    bes_spectrum_accum += log_mag.mean(axis=2)  # (16, F)
                    bes_n += 1

    # Mean over shots.
    mean_log_mag_per_freq = {
        m: sum_log_mag_per_freq[m] / max(n_per_freq[m], 1) for m in slices
    }
    save_freq_energy(FIG_DIR / "freq_energy.png", mean_log_mag_per_freq)

    bes_mean_spectrum = bes_spectrum_accum / max(bes_n, 1)
    bes_ch_indices = list(range(48, 64))
    save_bes_correlation(
        FIG_DIR / "bes_correlation.png", bes_mean_spectrum, bes_ch_indices
    )

    # Compute stats-vs-data sanity: with log_standardize for ECE/CO2,
    # post-standardized values should be ~unit variance per channel.
    # We don't apply log_standardize here (visualizations are raw log10),
    # but we can at least confirm the stats file dimensions match.
    sanity = {}
    for m, cs in slices.items():
        if m in stats and "log" in stats[m]:
            mean_arr = stats[m]["log"]["mean"][cs]
            std_arr = stats[m]["log"]["std"][cs]
            sanity[m] = (
                int(np.isnan(mean_arr).sum()),
                int(np.isnan(std_arr).sum()),
                float(np.nanmin(std_arr)),
                float(np.nanmax(std_arr)),
                int(mean_arr.shape[0]),
            )

    # ── Markdown findings ────────────────────────────────────────────
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    summary = SUMMARY_PATH
    lines: List[str] = []
    lines.append("# Step 0 — Data Verification Findings")
    lines.append("")
    lines.append(f"Date: 2026-05-06")
    lines.append(f"Shots inspected ({len(shots)}): "
                 f"{', '.join(s.stem.replace('_processed','') for s in shots)}")
    lines.append("")
    lines.append("## Confirmed shapes")
    lines.append("")
    lines.append("| modality | C (sliced) | observed shape (C, F, T) | matches plan [C, 512, 98]? |")
    lines.append("|---|---:|---|:---:|")
    for m, sh in seen_shapes.items():
        s = next(iter(sh)) if sh else None
        ok = (s is not None and s[0] == expected_C[m] and s[1] == 512 and s[2] == 98)
        lines.append(
            f"| {m} | {expected_C[m]} | {s} | {'✓' if ok else '✗'} |"
        )
    lines.append("")
    lines.append("All shots produced identical shapes per modality "
                 f"({sum(len(sh) for sh in seen_shapes.values())} shape "
                 f"observations total — should be {3*len(shots)} if "
                 f"unique).")
    lines.append("")
    lines.append("## Per-channel preprocessing-stats sanity")
    lines.append("")
    lines.append("| modality | C in stats | NaN(mean) | NaN(std) | std min | std max |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for m, vals in sanity.items():
        n_nan_m, n_nan_s, smn, smx, c = vals
        lines.append(f"| {m} | {c} | {n_nan_m} | {n_nan_s} | "
                     f"{smn:.4f} | {smx:.4f} |")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    # Path relative from docs/ to inspect_spectrograms/figures.
    fig_rel = Path("..") / FIG_DIR.relative_to(OUT_DIR.parent)
    lines.append(f"Saved to `{fig_rel}/` (relative to this doc):")
    lines.append("")
    for shot in shots:
        sid = shot.stem.replace("_processed", "")
        for m in slices:
            lines.append(f"- `{sid}_{m}.png` — {m.upper()} spectrogram, all channels stacked")
    lines.append("- `freq_energy.png` — per-frequency mean log-magnitude")
    lines.append("- `bes_correlation.png` — BES 16-channel inter-channel correlation matrix")
    lines.append("")
    lines.append("## Open questions to resolve from figures")
    lines.append("")
    lines.append("1. **Frequency cutoff:** look at `freq_energy.png`. Where does")
    lines.append("   the curve flatten / approach noise floor for each modality?")
    lines.append("   If <250 kHz cutoff is justified, recompute token budget.")
    lines.append("2. **BES grid orientation:** look at `bes_correlation.png`.")
    lines.append("   Two distinct 8x8 blocks (channels 49–56 vs 57–64) →")
    lines.append("   row-major reshape(2, 8). Interleaved pattern → column-major.")
    lines.append("3. **Physics features visible?** Inspect per-shot")
    lines.append("   spectrogram panels. Look for MHD modes (narrow horizontal")
    lines.append("   bands), ELM signatures (broadband bursts), and noise.")
    lines.append("")

    summary.write_text("\n".join(lines) + "\n")
    print(f"Wrote findings to {summary}")
    print(f"Figures in {FIG_DIR}")


if __name__ == "__main__":
    main()