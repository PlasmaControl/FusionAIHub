import shutil
from pathlib import Path

import torch

from tokamak_foundation_model.data.preprocess_data import compute_preprocessing_stats


def main():
    hdf5_files = sorted(
        Path("/lustre/orion/fus187/proj-shared/foundation_model").glob("*_processed.h5")
    )

    # Per-bin-only run: restrict to STFT signals so we don't redo the
    # ~25 non-STFT signals already covered by the existing
    # preprocessing_stats.pt. We compute raw + log + log_per_bin for
    # just these 6 spec signals, then merge ONLY the new 'log_per_bin'
    # entries into the existing file — all other keys (raw, log of
    # every modality, video stats, etc.) stay intact.
    stft_signals = {"mhr", "ece", "co2", "mirnov", "langmuir", "bes"}
    all_signals = list(stft_signals)

    zero_is_missing_signals = set()  # none of the STFT signals need this
    hdf5_key_map = {}                # none of the STFT signals need remapping

    stats_path = Path(
        "/lustre/orion/fus187/proj-shared/foundation_model_meta/"
        "preprocessing_stats.pt"
    )
    tmp_path = stats_path.with_suffix(".per_bin_tmp.pt")
    backup_path = stats_path.with_suffix(".pt.bak")

    # 1) Compute fresh stats for the 6 STFT signals (saved to tmp_path).
    new_stats = compute_preprocessing_stats(
        hdf5_paths=hdf5_files,
        signal_names=all_signals,
        output_path=tmp_path,
        stft_signals=stft_signals,
        hdf5_key_map=hdf5_key_map,
        zero_is_missing_signals=zero_is_missing_signals,
        num_workers=15,
        compute_per_bin_for_stft=True,
    )

    # 2) Load the existing stats and add ONLY the new 'log_per_bin'
    #    sub-entries. We deliberately do NOT overwrite the existing
    #    raw / log channel-wise stats (those came from a wider pass
    #    over all modalities and stay authoritative).
    print(f"Loading existing stats from {stats_path}")
    existing = torch.load(stats_path, weights_only=False)
    for sig in stft_signals:
        sig_stats = new_stats.get(sig)
        if not sig_stats or "log_per_bin" not in sig_stats:
            print(f"  WARN: no log_per_bin computed for {sig!r}; skipping")
            continue
        if sig not in existing:
            existing[sig] = {}
        existing[sig]["log_per_bin"] = sig_stats["log_per_bin"]
        m = sig_stats["log_per_bin"]["mean"]
        s = sig_stats["log_per_bin"]["std"]
        print(
            f"  {sig}: per-bin mean shape={tuple(m.shape)}  "
            f"mean-range [{m.min():.4g}, {m.max():.4g}]  "
            f"std-range [{s.min():.4g}, {s.max():.4g}]"
        )

    # 3) Atomic-ish save: back up the original, then overwrite.
    print(f"Backing up original to {backup_path}")
    shutil.copy2(stats_path, backup_path)
    print(f"Saving augmented stats back to {stats_path}")
    torch.save(existing, stats_path)

    # Clean up tmp.
    tmp_path.unlink(missing_ok=True)
    print("done")


if __name__ == "__main__":
    main()
