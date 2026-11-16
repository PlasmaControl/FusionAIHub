"""Build the designer's index of every tokenized shot: machine limits and shot features.

Two products in :data:`machine.INDEX_DIR`:

``machine_limits.json``
    Per era (block of :data:`ERA_SHOTS` shot numbers) and actuator channel: the
    largest power or flow sustained in any shot (99th percentile of the shot's frames,
    then the maximum over shots), the median voltage and torque per MW of each beam
    source with the share of shots in which it injected co-current, and how many shots
    used the channel. Everything is measured; nothing is a hardware specification.
``shots.npz``
    Per shot and 50 ms frame: total beam power, net beam torque, counter-injected beam
    power, ECH power, gas flow, and plasma quantities decoded from the shot's own tokens
    by the frozen codecs (core toroidal rotation, T_i, T_e, n_e; NaN where the
    diagnostic was not recorded). Plus per-shot scalars (plasma end, I_p peak, which
    sources and gyrotrons ran). :mod:`.precedents` queries it.

Run on a CPU node (about 15 min with 32 workers)::

    srun -N1 --account=kolemen --partition=cpu --mem=96G -c 32 -t 2:00:00 \\
        pixi run --frozen -e stellarai python -m tokamak_foundation_model.ignite.designer.index \\
        --workers 32
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import machine as mc

ERA_SHOTS = 2000
CODEC_TMPL = str(mc.MODELS / "ignite_codecs_v4" / "{m}" / "codec_best.pt")
PLASMA = ("cer_rot", "cer_ti", "ts_core_temp", "ts_core_density")
# CER chords T02, T04-T08 (codec channels 1, 3-7): the core tangential chords recorded in
# every paper shot; T01 and T03 have broken normalisation stats and decode to NaN.
CER_CORE = (1, 3, 4, 5, 6, 7)
# Per channel and block the index keeps this many largest per-shot values, so that a
# limit can be "reached in at least DB_MIN_SHOTS shots" (machine.database_limits).
TOP_N = 5
SERIES = ("pnbi_MW", "torque_Nm", "counter_MW", "ech_MW", "gas_TorrL_s",
          "rot_core_kms", "ti_core_keV", "te_core_keV", "ne_core_1e19")

_P, _T, _V = mc.columns("pinj"), mc.columns("tinj"), mc.columns("beam_voltage")
_E, _G = mc.columns("ech_power"), mc.columns("gas_flow")

_CODECS = None


def _codecs():
    global _CODECS
    if _CODECS is None:
        import torch

        from .. import eval_dynamics as ev
        from .. import train_dynamics as td

        torch.set_num_threads(1)
        _CODECS = td.load_frozen_codecs(list(PLASMA), repo=Path.cwd(), tmpl=CODEC_TMPL)
        _CODECS = (_CODECS, ev.profile_converters(_CODECS, PLASMA))
    return _CODECS


def _plasma(codes: dict, n: int) -> dict:
    """Core plasma quantities per frame from the shot's tokens (NaN when not recorded)."""
    import torch

    from .. import eval_dynamics as ev

    codecs, conv = _codecs()
    out = {}
    for name in PLASMA:
        c = codes.get(name)
        if c is None or (c == c[:1]).all():
            out[name] = np.full(n, np.nan, np.float32)
            continue
        x = ev.decode_flat_chunked(codecs[name][0], torch.as_tensor(np.asarray(c[:n])),
                                   torch.device("cpu"))
        with np.errstate(invalid="ignore", all="ignore"):
            prof = np.nanmean(conv[name](x), axis=-1)  # (F, C) physical
            if name.startswith("cer"):
                val = np.nanpercentile(np.abs(prof[:, list(CER_CORE)]), 90, axis=1)
            elif name == "ts_core_temp":
                val = np.nanpercentile(prof, 90, axis=1)
            else:
                val = np.nanpercentile(prof, 75, axis=1)
        out[name] = val.astype(np.float32)
    return {"rot_core_kms": out["cer_rot"], "ti_core_keV": out["cer_ti"] * 1e-3,
            "te_core_keV": out["ts_core_temp"] * 1e-3,
            "ne_core_1e19": out["ts_core_density"] * 1e-19}


def _one(args):
    """Features of one shot; runs in a worker."""
    shot, set_name = args
    import torch

    ts = next(t for t in mc.TOKEN_SETS if t.name == set_name)
    try:
        entry = torch.load(ts.act_global / f"{shot}.pt", map_location="cpu", weights_only=False)
        mu, sd = mc.act_stats()
        act = entry["actuators_global"].float().numpy() * sd + mu
        cache = torch.load(ts.frame_codes / f"{shot}.pt", map_location="cpu",
                           weights_only=False, mmap=True)
        n = min(act.shape[0], int(cache.get("n_frames", act.shape[0])))
        act = act[:n]
        codes = {k: np.asarray(v) for k, v in cache["codes"].items() if k in PLASMA}
        end = ts.plasma_end / f"{shot}.json"
        info = json.loads(end.read_text()) if end.exists() else {}
        t_end = info.get("t_end")
        ip = info.get("ip_peak_a")
        p = act[:, _P] * 1e-6
        tq = act[:, _T]
        series = {
            "pnbi_MW": p.sum(1), "torque_Nm": tq.sum(1),
            "counter_MW": np.where(tq < -0.05, p, 0.0).sum(1),
            "ech_MW": (act[:, _E] * 1e-6).sum(1), "gas_TorrL_s": act[:, _G].sum(1),
            **_plasma(codes, n),
        }
        live = mc.time_of(np.arange(n)) < (t_end if t_end else np.inf)
        for k in series:
            series[k] = np.where(live, series[k], np.nan).astype(np.float32)
        # Per-channel statistics over the live frames, for the machine limits.
        a = act[live] if live.any() else act[:0]
        chan = {}
        for group in mc.GROUPS:
            x = a[:, mc.columns(group)]
            chan[group] = {
                "p99": np.percentile(np.abs(x), 99, axis=0) if len(x) else np.zeros(x.shape[1]),
            }
        on = a[:, _P] > 3e5
        rate = np.full(8, np.nan)
        kv = np.full(8, np.nan)
        for i in range(8):
            if on[:, i].sum() >= 3:
                rate[i] = np.median(a[on[:, i], _T.start + i] / a[on[:, i], _P.start + i] * 1e6)
                kv[i] = np.median(a[on[:, i], _V.start + i]) * 1e-3
        return {"shot": shot, "set": set_name, "series": series,
                "t_end": np.nan if t_end is None else float(t_end),
                "ip_peak_MA": np.nan if ip is None else abs(float(ip)) / 1e6,
                "chan": chan, "rate": rate, "kv": kv}
    except Exception as e:  # noqa: BLE001 - one bad shot must not stop the index
        return {"shot": shot, "set": set_name, "error": f"{type(e).__name__}: {e}"}


def _limits(rows) -> dict:
    eras = {}
    for r in rows:
        e = str(int(r["shot"]) // ERA_SHOTS * ERA_SHOTS)
        eras.setdefault(e, []).append(r)
    out = {"era_shots": ERA_SHOTS, "built": time.strftime("%Y-%m-%d %H:%M"),
           "method": "power/flow/current: max (and 95th percentile) over shots of each "
                     "shot's 99th percentile over its live 50 ms frame means; voltage and "
                     "torque per MW: median over shots of each shot's median over frames "
                     "with the source above 0.3 MW, separately for shots in which the "
                     "source injected co-current (+) and counter-current (-)",
           "eras": {}}
    for e, rs in sorted(eras.items(), key=lambda kv: int(kv[0])):
        era = {"n_shots": len(rs), "shots": [min(r["shot"] for r in rs), max(r["shot"] for r in rs)],
               "beams": {}, "ech_power": {}, "gas_flow": {}, "rmp": {}, "i_coil": {}}
        for i, b in enumerate(mc.BEAMS):
            pmax = np.array([r["chan"]["pinj"]["p99"][i] for r in rs]) * 1e-6
            rate = np.array([r["rate"][i] for r in rs])
            kv = np.array([r["kv"][i] for r in rs])
            used = np.isfinite(rate)
            co, ctr = used & (rate > 0), used & (rate < 0)

            def med(x):
                return round(float(np.median(x)), 3) if len(x) else None

            era["beams"][b] = {
                "n_shots_used": int(used.sum()),
                "power_MW_max": round(float(pmax.max()), 3) if len(pmax) else 0.0,
                "power_MW_max_shot": int(rs[int(pmax.argmax())]["shot"]) if len(pmax) else None,
                "power_MW_top": _top(pmax, rs),
                "power_MW_p95_of_shots": round(float(np.percentile(pmax[used], 95)), 3)
                if used.any() else None,
                "kv_median": med(kv[used]),
                "kv_max": round(float(np.max(kv[used])), 1) if used.any() else None,
                "n_shots_co": int(co.sum()),
                "n_shots_counter": int(ctr.sum()),
                "torque_per_MW_co": med(rate[co]),
                "torque_per_MW_counter": med(rate[ctr]),
            }
        for group, scale in (("ech_power", 1e-6), ("gas_flow", 1.0), ("rmp", 1e-3), ("i_coil", 1e-3)):
            for i, name in enumerate(mc.CHANNEL_NAMES[group]):
                vals = np.array([r["chan"][group]["p99"][i] for r in rs]) * scale
                thr = {"ech_power": 0.02, "gas_flow": 0.5, "rmp": 0.1, "i_coil": 0.1}[group]
                used = vals > thr
                era[group][name] = {
                    "max": round(float(vals.max()), 3),
                    "max_shot": int(rs[int(vals.argmax())]["shot"]),
                    "top": _top(np.abs(vals), rs),
                    "p95_of_shots": round(float(np.percentile(vals[used], 95)), 3)
                    if used.any() else None,
                    "n_shots_used": int(used.sum())}
        out["eras"][e] = era
    return out


def _top(vals, rs, n=TOP_N):
    """The ``n`` largest per-shot values with their shots, largest first."""
    order = np.argsort(-np.nan_to_num(vals, nan=-np.inf))[:n]
    return [[round(float(vals[i]), 3), int(rs[i]["shot"])] for i in order if np.isfinite(vals[i])]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--limit", type=int, default=0, help="only the first N shots (testing)")
    ap.add_argument("--out", default=str(mc.INDEX_DIR))
    a = ap.parse_args(argv)
    jobs = []
    seen = set()
    for ts in mc.TOKEN_SETS:
        for f in sorted(ts.act_global.glob("*.pt")):
            if f.stem.isdigit() and int(f.stem) not in seen and (ts.frame_codes / f.name).exists():
                seen.add(int(f.stem))
                jobs.append((int(f.stem), ts.name))
    if a.limit:
        jobs = jobs[: a.limit]
    print(f"[index] {len(jobs)} shots, {a.workers} workers", flush=True)
    t0 = time.time()
    rows, errors = [], []
    with ProcessPoolExecutor(a.workers) as pool:
        for i, r in enumerate(pool.map(_one, jobs, chunksize=16)):
            (errors if "error" in r else rows).append(r)
            if (i + 1) % 1000 == 0:
                print(f"[index] {i + 1}/{len(jobs)} in {time.time() - t0:.0f} s", flush=True)
    rows.sort(key=lambda r: r["shot"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "machine_limits.json").write_text(json.dumps(_limits(rows), indent=1))
    lengths = np.array([len(r["series"]["pnbi_MW"]) for r in rows])
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    np.savez_compressed(
        out / "shots.npz",
        shot=np.array([r["shot"] for r in rows]),
        token_set=np.array([r["set"] for r in rows]),
        t_end=np.array([r["t_end"] for r in rows], np.float32),
        ip_peak_MA=np.array([r["ip_peak_MA"] for r in rows], np.float32),
        beam_rate=np.stack([r["rate"] for r in rows]).astype(np.float32),
        offsets=offsets,
        **{f"series__{k}": np.concatenate([r["series"][k] for r in rows]).astype(np.float16)
           for k in SERIES},
    )
    (out / "errors.json").write_text(json.dumps(errors, indent=1))
    print(f"[index] wrote {out}: {len(rows)} shots, {len(errors)} errors, "
          f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
