"""Build the HL-3 network's full offline input set for the benchmark shots.

Nine channels at 10 kHz on each shot's native analysis grid (the grid of the
benchmark's own `signals/<shot>.npz`), one `inputs/<shot>.npz` per shot under
`$LABELER_ROOT/round4/hl3/work/hl3_inputs/`, and a small record of what each shot
had, `outputs/labeler/sawtooth/fix5/hl3_full_inputs.json`:

    ece_core_te       the benchmark's existing core ECE average (keV, EFIT-axis rule)
    mirnov_pair_mean  mean of Mirnov rows 0 and 1, 10 kHz FIR-decimated
    ip                plasma current (MA), the benchmark's existing input
    line_density      CO2 interferometer chord V2 (corpus or raw cache), 10 kHz, / 1e14
    sxr_core          SX90 core chord, label-free choice (see `labeler.sawtooth.hl3_inputs`)
    sxr_edge          SX90 edge chord, label-free choice
    stored_energy     EFIT01 WMHD (MJ, 20 ms cadence, bridged to 100 ms gaps)
    nbi_power         summed beam power (MW): corpus `pinj`, else PTDATA BMSPINJ
    ech_power         summed gyrotron power (MW): corpus `ech_power`

Every row is divided by `hl3_inputs.UNIT_SCALE` when stacked: the benchmark stores
training windows as float16, and joules or watts overflow it to infinity.

A signal a shot lacks is a missing channel (NaN here, masked in the network);
the shot is still built. No label, crash time or interval is read: the only
inputs besides the signals are the benchmark's own `t` and `baseline`.

    pixi run --frozen -e labelmaker python scripts/labeler/sawtooth_hl3_inputs.py \\
        --workers 6
    ... --aggregate     # write the shot-coverage record from the per-shot files
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
from sawtooth_physics import OUTPUT, WORK, mean_finite, save_json

from labeler.config import Paths
from labeler.events import equilibrium, heuristics, verify
from labeler.sawtooth import hl3_inputs as hl3
from labeler.sawtooth.preprocessing import sample_native

ROOT = Path(os.environ["LABELER_ROOT"]) / "round4/hl3"
HL3_WORK = ROOT / "work"
INPUTS = HL3_WORK / "hl3_inputs"
RECORDS = ROOT / "inputs_records"
BMS = ROOT / "bms"
DETACH_CACHE = Path(os.environ["LABELER_ROOT"]) / "round4/detach/cache"
#: CO2 chord order in the corpus: R0, V1, V2, V3; the repo's density is V2.
CO2_ROW = 2
WMHD_MAX_GAP_S = 0.1
#: A channel counts as present on a shot when it is finite on this share of the
#: shot's analysis window.
PRESENT = 0.5


def groups(shot, name, paths):
    """Open h5 groups named ``name`` for ``shot``: corpus first, then raw cache."""
    for label, root in (("corpus", paths.corpus), ("raw_cache", paths.raw_cache)):
        path = verify.corpus_path(shot, corpus=root)
        if not path.is_file():
            continue
        file = h5py.File(path, "r", locking=False)
        if name in file and file[name]["ydata"].shape[-1] > verify.SENTINEL_WIDTH:
            yield label, file, file[name]
        else:
            file.close()


def read_native(shot, name, paths, *, rows=None):
    """`(source, t_s, y)` of a corpus group at 10 kHz, from the first store holding it.

    High-rate groups go through the repo's FIR decimator on the idealised uniform
    clock (a float32 clock is checked globally, not chunk by chunk).
    """
    errors = []
    for label, file, group in groups(shot, name, paths):
        try:
            clock = hl3.uniform_clock(np.asarray(group["xdata"]))
            t, y = sample_native(
                {"xdata": clock, "ydata": group["ydata"]}, rows=rows, fs=int(hl3.FS)
            )
            return label, t, y
        except ValueError as error:
            errors.append(f"{label}: {error}")
        finally:
            file.close()
    raise ValueError("; ".join(errors) or f"no {name!r} group in corpus or raw cache")


def beam_power(shot, paths):
    """`(source, t_s, watts)`: summed beams from the corpus, else BMSPINJ."""
    try:
        label, t, y = read_native(shot, "pinj", paths)
        power = hl3.sum_sources(y)
        if power is not None:
            return f"{label}:pinj", t, power
    except ValueError:
        pass
    path = DETACH_CACHE / f"{shot}.npz"
    if path.is_file():
        with np.load(path) as data:
            if "pinj_bms__y" in data.files:
                t = np.asarray(data["pinj_bms__t"], dtype=float) / 1000
                return "detach_cache:BMSPINJ", t, np.asarray(data["pinj_bms__y"]) * 1e6
    path = BMS / f"{shot}.npz"
    if path.is_file():
        with np.load(path) as data:
            return "fetched:BMSPINJ", data["t_s"], np.asarray(data["w"], dtype=float)
    raise ValueError("no beam power in the corpus, raw cache or BMSPINJ stores")


def sxr_pair(shot, paths, window):
    """`(pair, fan, t_s, core, edge)` from the shot's first lit SX90 fan."""

    def read(group, rows):
        _, t, y = read_native(shot, group, paths, rows=slice(rows.start, rows.stop))
        return t, y

    fan, t, y, chords = heuristics.sxr_fan(read, shot=shot)
    keep = (t >= window[0]) & (t <= window[1])
    pair = hl3.select_sxr_chords(y[:, keep], chords=chords, fs=hl3.FS)
    row = {int(c): i for i, c in enumerate(chords)}
    core = None if pair.core is None else y[row[pair.core]]
    edge = None if pair.edge is None else y[row[pair.edge]]
    return pair, fan, t, core, edge


def build_shot(shot, *, sxr_scale="robust"):
    """The nine-channel matrix of one shot and its record, never raising."""
    paths = Paths.from_env()
    out = INPUTS / f"{shot}.npz"
    started = time.monotonic()
    with np.load(WORK / "signals" / f"{shot}.npz") as data:
        t = np.asarray(data["t"], dtype=np.float64)
        baseline = np.asarray(data["baseline"], dtype=np.float32)
        observable = np.asarray(data["observable"], dtype=bool)
    window = (float(t[0]), float(t[-1]))
    record = {
        "shot": int(shot),
        "window_s": list(window),
        "observable_samples": int(observable.sum()),
        "source": {},
        "error": {},
    }
    parts = {
        "ece_core_te": baseline[0],
        "ip": baseline[3],
    }
    record["source"]["ece_core_te"] = "benchmark baseline (EFIT-axis core ECE average)"
    record["source"]["ip"] = "benchmark baseline (Ip / 1e6)"

    def attempt(name, work):
        try:
            return work()
        except (ValueError, KeyError, OSError, IndexError, verify.NoDataError) as e:
            record["error"][name] = f"{type(e).__name__}: {e}"[:200]
            return None

    def mirnov():
        label, tm, ym = read_native(shot, "mirnov", paths, rows=slice(0, 2))
        record["source"]["mirnov_pair_mean"] = f"{label}:mirnov rows 0-1 (FIR 10 kHz)"
        return hl3.interpolate_rows(t, tm, mean_finite(ym)[None])[0]

    def density():
        label, td, yd = read_native(
            shot, "co2", paths, rows=slice(CO2_ROW, CO2_ROW + 1)
        )
        record["source"]["line_density"] = f"{label}:co2 chord V2 (FIR 10 kHz)"
        return hl3.interpolate_rows(t, td, yd)[0]

    def stored_energy():
        # The fetch parked WMHD in its own raw-cache folder, never the shared one.
        for where in (paths, replace(paths, raw_cache=ROOT / "raw")):
            try:
                w = equilibrium.signal(shot, "wmhd", where, fetch=False)
                break
            except verify.NoDataError as error:
                missing = error
        else:
            raise missing
        record["source"]["stored_energy"] = "EFIT01 WMHD (J)"
        return hl3.interpolate_gapped(t, w.x, w.y[0], max_gap_s=WMHD_MAX_GAP_S)

    def beams():
        label, tb, pb = beam_power(shot, paths)
        record["source"]["nbi_power"] = label
        return hl3.interpolate_rows(t, tb, pb[None])[0]

    def ech():
        label, te, ye = read_native(shot, "ech_power", paths)
        power = hl3.sum_sources(ye)
        if power is None:
            raise ValueError("ech_power holds no finite source")
        record["source"]["ech_power"] = f"{label}:ech_power (sum of gyrotrons, W)"
        return hl3.interpolate_rows(t, te, power[None])[0]

    parts["mirnov_pair_mean"] = attempt("mirnov_pair_mean", mirnov)
    parts["line_density"] = attempt("line_density", density)
    parts["stored_energy"] = attempt("stored_energy", stored_energy)
    parts["nbi_power"] = attempt("nbi_power", beams)
    parts["ech_power"] = attempt("ech_power", ech)
    found = attempt("sxr", lambda: sxr_pair(shot, paths, window))
    record["sxr"] = {"fan": None, "method": hl3.METHOD, "status": "no_sxr_pair"}
    if found is not None:
        pair, fan, ts, core, edge = found
        record["sxr"] = {"fan": fan, **pair.to_json(), "status": "pair"}
        for name, chord, values in (
            ("sxr_core", pair.core, core),
            ("sxr_edge", pair.edge, edge),
        ):
            if values is None:
                record["sxr"]["status"] = f"no_{name.split('_')[1]}_chord"
                continue
            on_grid = hl3.interpolate_rows(t, ts, values[None])[0]
            parts[name] = (
                hl3.robust_standardise(on_grid) if sxr_scale == "robust" else on_grid
            )
            record["source"][name] = f"{fan} chord {chord} ({sxr_scale} scale)"
    matrix, finite = hl3.stack_inputs(t, parts)
    # Observability masks the same samples the benchmark masks (input_values).
    record["finite_fraction_window"] = finite
    obs = observable
    record["finite_fraction_observable"] = {
        name: float(np.isfinite(matrix[i][obs]).mean()) if obs.any() else 0.0
        for i, name in enumerate(hl3.CHANNELS)
    }
    record["present"] = {
        name: bool(record["finite_fraction_observable"][name] >= PRESENT)
        for name in hl3.CHANNELS
    }
    INPUTS.mkdir(parents=True, exist_ok=True)
    temporary = INPUTS / f".{shot}.tmp.npz"
    np.savez_compressed(temporary, inputs=matrix, t=t, channels=np.array(hl3.CHANNELS))
    temporary.replace(out)
    RECORDS.mkdir(parents=True, exist_ok=True)
    record["elapsed_s"] = round(time.monotonic() - started, 2)
    save_json(RECORDS / f"{shot}.json", record)
    return record


def population():
    split = json.loads((OUTPUT / "split_manifest.json").read_text())
    shots = set(split["training_cohort"]) | set(split["fixed_validation_supported"])
    shots |= set(split["expert_shots"])
    if shots & set(split["blind_test_excluded"]):
        raise ValueError("the blind test split must never be read")
    return sorted(shots)


def work_one(args):
    shot, scale, redo = args
    built = (INPUTS / f"{shot}.npz").is_file() and (RECORDS / f"{shot}.json").is_file()
    if built and not redo:
        return {"shot": shot, "status": "kept"}
    record = build_shot(shot, sxr_scale=scale)
    return {"shot": shot, "status": "built", "present": record["present"]}


def aggregate():
    shots = population()
    every = [json.loads((RECORDS / f"{s}.json").read_text()) for s in shots]
    # A shot with no observable ECE sample contributes no window and no score.
    rows = [r for r in every if r["observable_samples"] > 0]
    names = hl3.CHANNELS
    lacking = {n: sorted(r["shot"] for r in rows if not r["present"][n]) for n in names}
    methods, fans, structure = {}, {}, []
    for r in rows:
        methods[r["sxr"]["status"]] = methods.get(r["sxr"]["status"], 0) + 1
        fans[str(r["sxr"].get("fan"))] = fans.get(str(r["sxr"].get("fan")), 0) + 1
        if r["sxr"]["status"] != "no_sxr_pair":
            structure.append(r["sxr"])
    sources = {}
    for name in names:
        for r in rows:
            key = r["source"].get(name, "missing")
            if name.startswith("sxr") and key != "missing":
                key = key.split(" chord ")[0] + " chord"
            sources.setdefault(name, {})
            sources[name][key] = sources[name].get(key, 0) + 1
    correlation = np.array(
        [
            x["edge_core_correlation"]
            for x in structure
            if x["edge_core_correlation"] is not None
        ]
    )
    summary = {
        "shots": len(shots),
        "shots_without_observable_support": sorted(
            r["shot"] for r in every if r["observable_samples"] == 0
        ),
        "shots_with_observable_support": len(rows),
        "channels": list(names),
        "unit_scale": {n: hl3.UNIT_SCALE[n] for n in names},
        "unit_scale_note": (
            "each row is divided by this when stacked: stored energy MJ, powers MW, "
            "CO2 chord V2 / 1e14; the sources below name the reader's own units"
        ),
        "present_rule": f"finite on at least {PRESENT:.0%} of the shot's observable samples",
        "shots_present": {n: len(rows) - len(lacking[n]) for n in names},
        "shots_lacking": {n: len(lacking[n]) for n in names},
        "lacking_shot_ids": lacking,
        "sxr_method": hl3.METHOD,
        "sxr_geometry": (
            "SX90 chord impact parameters are in neither MDSplus nor imas_composer "
            "(probed 2026-10-05), so every shot uses the label-free pairing"
        ),
        "sxr_status_counts": methods,
        "sxr_core_with_structure": sum(1 for x in structure if x["core_has_structure"]),
        "sxr_edge_with_structure": sum(1 for x in structure if x["edge_has_structure"]),
        "sxr_edge_correlation_negative": int((correlation < 0).sum()),
        "sxr_edge_correlation_below_minus_0p1": int((correlation < -0.1).sum()),
        "sxr_edge_correlation_median": float(np.median(correlation)),
        "sxr_edge_correlation_shots": len(correlation),
        "sxr_structure_ratio": hl3.STRUCTURE_RATIO,
        "sxr_fan_counts": fans,
        "sources": sources,
        "any_channel_lacking_shots": len({s for x in lacking.values() for s in x}),
        "per_shot": [
            {
                "shot": r["shot"],
                "present": [n for n in names if r["present"][n]],
                "sxr": r["sxr"],
                "finite_fraction_observable": r["finite_fraction_observable"],
                "errors": r["error"],
            }
            for r in rows
        ],
    }
    save_json(OUTPUT / "hl3_full_inputs.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_shot"}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shots", type=int, nargs="*")
    parser.add_argument("--sxr-scale", choices=("robust", "raw"), default="robust")
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("at most eight workers")
    if args.aggregate:
        aggregate()
        return 0
    shots = args.shots or population()
    if args.limit:
        shots = shots[: args.limit]
    jobs = [(s, args.sxr_scale, args.redo) for s in shots]
    context = multiprocessing.get_context("fork")
    with context.Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(work_one, jobs), 1):
            if i % 25 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} {row['shot']} {row['status']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
