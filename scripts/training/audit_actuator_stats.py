"""Audit actuator preprocessing stats for correctness.

Loads the preprocessing stats file and checks all actuator channels for:
- NaN/Inf values in min/max/mean/std
- Zero-range channels (max - min < 1e-8)
- Shape mismatches with expected n_channels
- Value range sanity
"""
import sys
from pathlib import Path

import torch
import numpy as np

# Actuator configs (must match train_foundation_model.py)
ACTUATOR_CONFIGS = {
    "pin": {"target_fs": 10_000, "n_channels": 8, "patch_len": 200},
    "tin": {"target_fs": 10_000, "n_channels": 8, "patch_len": 200},
    "beam_voltage": {"target_fs": 10_000, "n_channels": 8, "patch_len": 200},
    "ech_power": {"target_fs": 10_000, "n_channels": 12, "patch_len": 200},
    "ech_tor_angle": {"target_fs": 10_000, "n_channels": 12, "patch_len": 200},
    "ech_pol_angle": {"target_fs": 10_000, "n_channels": 12, "patch_len": 200},
    "gas_flow": {"target_fs": 10_000, "n_channels": 11, "patch_len": 200},
    "ich": {"target_fs": 10_000, "n_channels": 1, "patch_len": 200},
    "rmp": {"target_fs": 10_000, "n_channels": 12, "patch_len": 200},
}


def main():
    stats_path = sys.argv[1] if len(sys.argv) > 1 else \
        "/projects/EKOLEMEN/foundation_model/preprocessing_stats.pt"

    print(f"Loading stats from: {stats_path}")
    stats = torch.load(stats_path, weights_only=False)

    print(f"\nTop-level keys in stats: {sorted(stats.keys())}\n")

    total_issues = 0

    for name, cfg in ACTUATOR_CONFIGS.items():
        expected_ch = cfg["n_channels"]
        print(f"\n{'='*70}")
        print(f"Actuator: {name} (expected {expected_ch} channels)")
        print(f"{'='*70}")

        if name not in stats:
            print(f"  *** NOT FOUND in stats! ***")
            total_issues += 1
            continue

        entry = stats[name]
        # Stats may be nested under "raw" key
        s = entry.get("raw", entry) if isinstance(entry, dict) else entry

        for stat_name in ["min_val", "max_val", "mean", "std"]:
            if stat_name not in s:
                print(f"  *** Missing '{stat_name}' ***")
                total_issues += 1
                continue

            val = np.asarray(s[stat_name])
            n_ch = val.shape[0] if val.ndim > 0 else 1

            # Shape check
            if n_ch != expected_ch:
                print(f"  *** {stat_name}: shape={val.shape}, "
                      f"expected {expected_ch} channels ***")
                total_issues += 1

            # NaN/Inf check
            n_nan = np.isnan(val).sum()
            n_inf = np.isinf(val).sum()
            if n_nan > 0 or n_inf > 0:
                print(f"  *** {stat_name}: {n_nan} NaN, {n_inf} Inf ***")
                total_issues += 1

            print(f"  {stat_name:>8s}: shape={str(val.shape):>10s}  "
                  f"range=[{val.min():12.6f}, {val.max():12.6f}]")

        # Check min-max range
        if "min_val" in s and "max_val" in s:
            s_min = np.asarray(s["min_val"])
            s_max = np.asarray(s["max_val"])
            s_range = s_max - s_min
            zero_range = s_range < 1e-8
            n_zero = zero_range.sum()
            if n_zero > 0:
                idxs = np.where(zero_range)[0]
                print(f"  *** {n_zero} channels with zero range: {idxs.tolist()} ***")
                total_issues += 1
            else:
                print(f"  Range: min={s_range.min():.6f}, "
                      f"max={s_range.max():.6f}, "
                      f"mean={s_range.mean():.6f}")

            # Check if min > max (corrupted)
            inverted = s_min > s_max
            n_inv = inverted.sum()
            if n_inv > 0:
                print(f"  *** {n_inv} channels with min > max! ***")
                total_issues += 1

    # Also check for diagnostic signals
    print(f"\n\n{'='*70}")
    print("Diagnostic signal stats (for reference)")
    print(f"{'='*70}")
    for name in ["filterscopes", "ts_core_density", "ts_core_temp",
                  "ts_tangential_density", "ts_tangential_temp",
                  "mse", "cer_ti", "cer_rot"]:
        if name not in stats:
            print(f"  {name}: NOT FOUND")
            continue
        entry = stats[name]
        # Check both raw and log keys
        for subkey in ["raw", "log"]:
            if isinstance(entry, dict) and subkey in entry:
                s = entry[subkey]
                for stat_name in ["min_val", "max_val", "mean", "std"]:
                    if stat_name in s:
                        val = np.asarray(s[stat_name])
                        n_nan = np.isnan(val).sum()
                        n_inf = np.isinf(val).sum()
                        flag = " ***" if (n_nan + n_inf) > 0 else ""
                        print(f"  {name}.{subkey}.{stat_name}: "
                              f"shape={val.shape}, "
                              f"range=[{val.min():.4f}, {val.max():.4f}]"
                              f"{flag}")

    print(f"\n\nTotal issues found: {total_issues}")
    if total_issues == 0:
        print("All actuator stats look clean!")


if __name__ == "__main__":
    main()
