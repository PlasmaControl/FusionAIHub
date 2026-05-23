#!/usr/bin/env python3
"""
Parse rocm-smi + mpstat CSVs from a profile/ directory and print per-job
summaries: peak/median GPU compute %, peak/median GPU memory %, peak power,
peak temp, and a rough CPU utilization summary.

Usage:
    python3 scripts/slurm_frontier/_summarize_profile.py profile/<jobid>_*

If no args are given, scans every subdirectory of profile/.
"""
import csv
import glob
import os
import sys
from pathlib import Path
from statistics import mean, median


def parse_gpu_csv(path: Path):
    """Yield dicts of {card, gpu_pct, vram_pct, power_w, temp_c}."""
    with open(path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                yield {
                    "card": row["card"],
                    "gpu_pct": float(row["gpu_use_pct"]),
                    "vram_pct": float(row["gpu_memory_use_pct"]),
                    "power_w": float(row["power_w"]),
                    "temp_c": float(row["temp_edge_c"]),
                }
            except (KeyError, ValueError):
                continue


def parse_cpu_mpstat(path: Path):
    """Return a list of `all` rows' %idle from mpstat output (one per second)."""
    idle_pcts = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            # mpstat row format: "HH:MM:SS  CPU  %usr ... %idle"
            # The "all" rows we care about have "all" as the 2nd token.
            if len(parts) >= 12 and parts[1] == "all":
                try:
                    idle_pcts.append(float(parts[-1]))
                except ValueError:
                    pass
    return idle_pcts


def stats(values, fmt="{:6.1f}"):
    if not values:
        return "n/a"
    return f"min={fmt.format(min(values))} med={fmt.format(median(values))} mean={fmt.format(mean(values))} max={fmt.format(max(values))} (n={len(values)})"


def summarize_dir(prof_dir: Path):
    """Print a single-job summary block."""
    print(f"\n=== {prof_dir.name} ===")
    gpu_files = sorted(prof_dir.glob("*_gpu.csv"))
    cpu_files = sorted(prof_dir.glob("*_cpu.csv"))

    if not gpu_files:
        print(f"  (no *_gpu.csv files)")
        return

    for gpu_path in gpu_files:
        host = gpu_path.stem.replace("_gpu", "")
        print(f"  host={host}")
        # Bucket by card
        by_card = {}
        for r in parse_gpu_csv(gpu_path):
            by_card.setdefault(r["card"], []).append(r)
        for card, rows in sorted(by_card.items()):
            gpu_pcts = [r["gpu_pct"] for r in rows]
            vram_pcts = [r["vram_pct"] for r in rows]
            powers = [r["power_w"] for r in rows]
            temps = [r["temp_c"] for r in rows]
            print(f"    {card}:")
            print(f"      gpu_pct  {stats(gpu_pcts)}")
            print(f"      vram_pct {stats(vram_pcts)}")
            print(f"      power_w  {stats(powers, '{:6.1f}')}")
            print(f"      temp_c   {stats(temps, '{:6.1f}')}")

    for cpu_path in cpu_files:
        host = cpu_path.stem.replace("_cpu", "")
        idle = parse_cpu_mpstat(cpu_path)
        if idle:
            busy = [100 - x for x in idle]
            print(f"  host={host} cpu_util(%)  {stats(busy)}")


def main(argv):
    targets = []
    if len(argv) > 1:
        for arg in argv[1:]:
            targets.extend(Path(p) for p in sorted(glob.glob(arg)))
    else:
        if Path("profile").is_dir():
            targets = sorted(p for p in Path("profile").iterdir() if p.is_dir())

    if not targets:
        print("No profile directories found. Pass a path or run from project root.")
        sys.exit(1)

    for t in targets:
        if not t.is_dir():
            print(f"skipping {t}: not a dir")
            continue
        summarize_dir(t)


if __name__ == "__main__":
    main(sys.argv)
