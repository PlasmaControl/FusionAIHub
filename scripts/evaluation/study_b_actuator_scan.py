"""Study B: actuator-conditioning scans against physics expectations.

Sub-studies (``--study``, default runs all):

  b1    9-actuator × 12-diagnostic finite-difference sensitivity matrix.
        Two perturbation types per actuator: additive ±0.5·σ_pool (z-space,
        wakes even off actuators) and multiplicative ±20 % (physical units,
        via the inverse transform). Sensitivity = mean |Δpred| / σ_modality.
        Plus a signed physics matrix on scalar readouts (gas_flow→nₑ↑,
        pin→Tᵢ/rotation↑, ech_power→Tₑ↑, ...) with sign-consistency stats.

  b2    Dose-response: physical multiplicative scale ∈ {0,0.5,0.8,1,1.2,
        1.5,2} on pin/gas_flow/ech_power/beam_voltage/rmp; median ± IQR of
        Δ(scalar readout) vs scale at K=1.

  swap  Actuator-swap test (targets the prior cos-sim≈0.999 weak-
        conditioning finding): swap whole actuator trajectories between
        pool windows, compare per-modality latent + prediction cosine
        similarity against natural cross-window variability.

Reference pool: first ``--n-shots`` val shots providing ``--windows-per-shot``
flat-top windows (t ∈ [2, 8] s) with core Thomson + ECE valid. Deterministic.

All perturbed passes reuse ``forward_with_latents`` on a batch whose
``targets[<actuator>]`` entry is replaced — identical input prep to
training, no bespoke forward path.

Run inside an sbatch (single GCD):
    python scripts/evaluation/study_b_actuator_scan.py --study all
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "src"))
sys.path.insert(0, str(_HERE.parent))

from tokamak_foundation_model.data.data_loader import (  # noqa: E402
    collate_fn_prediction,
)

from tfm_eval.ckpt import DEFAULT_CKPT, build_model_from_ckpt, load_ckpt  # noqa: E402
from tfm_eval.data import (  # noqa: E402
    assert_val_matches_training_cache,
    load_stats,
    make_prediction_dataset,
    resolve_split,
    shot_window_ranges,
)
from tfm_eval.inverse_preprocess import InversePreprocessor  # noqa: E402
from tfm_eval.latents import forward_with_latents  # noqa: E402

# Scalar readouts computed from model PREDICTIONS (z-space → physical).
READOUT_SIGNALS = [
    ("ne_core", "ts_core_density"),
    ("te_core", "ts_core_temp"),
    ("ne_tang", "ts_tangential_density"),
    ("te_tang", "ts_tangential_temp"),
    ("ti", "cer_ti"),
    ("rotation", "cer_rot"),
    ("d_alpha", "filterscopes"),
]

# Physics sign expectations for (actuator, readout): +1 = readout should
# rise when the actuator rises. Curated, conservative — only entries with
# an unambiguous textbook direction are scored.
EXPECTED_SIGNS = {
    ("gas_flow", "ne_core"): +1,   # fueling raises density
    ("gas_flow", "ne_tang"): +1,
    ("pin", "te_core"): +1,        # NBI heats
    ("pin", "ti"): +1,
    ("pin", "rotation"): +1,       # co-injection torque spins the plasma
    ("pin", "d_alpha"): +1,        # beam fueling/recycling raises D-alpha
    ("ech_power", "te_core"): +1,  # ECH heats electrons
    ("beam_voltage", "ti"): +1,
    ("beam_voltage", "rotation"): +1,
    ("rmp", "ne_core"): -1,        # RMP density pump-out
    ("rmp", "ne_tang"): -1,
    ("rmp", "rotation"): -1,       # RMP torque brakes rotation
}

DOSE_ACTUATORS = ["pin", "gas_flow", "ech_power", "beam_voltage", "rmp"]
DOSE_SCALES = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--study", choices=("b1", "b2", "swap", "all"), default="all")
    ap.add_argument("--n-shots", type=int, default=32)
    ap.add_argument("--windows-per-shot", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--out-root",
        default=str(_HERE.parents[2] / "data" / "outputs" / "eval_suite"
                    / "e2e_stage1_best"),
    )
    return ap.parse_args()


def build_pool(ds, ranges, n_shots, windows_per_shot, stride_s, warmup_s, t0):
    """Deterministic flat-top reference pool: list of per-window samples."""
    lo = int(round((2.0 - warmup_s) / stride_s))       # t = 2 s
    hi_t = 8.0
    samples, shot_names = [], []
    for path, start, n in ranges:
        if len(shot_names) >= n_shots:
            break
        cand = []
        for li in range(lo, n):
            if warmup_s + li * stride_s > hi_t:
                break
            cand.append(start + li)
        picked = []
        for g in cand:
            s = ds[g]
            if (float(s["inputs"].get("ts_core_density_valid", 0)) > 0
                    and float(s["inputs"].get("ece_valid", 0)) > 0):
                picked.append(s)
            if len(picked) == windows_per_shot:
                break
        if len(picked) == windows_per_shot:
            samples.extend(picked)
            shot_names.append(path.name.replace("_processed.h5", ""))
        print(f"[{time.time()-t0:6.1f}s]   pool: {len(shot_names)} shots "
              f"({len(samples)} windows)", end="\r", flush=True)
    print()
    return samples, shot_names


def batched_forward(model, batch, device, bsz):
    """forward_with_latents over minibatches of a big collated batch."""
    n = next(iter(batch["inputs"].values())).shape[0]
    preds, pooled = {}, {}
    for i in range(0, n, bsz):
        sl = {
            side: {k: v[i:i + bsz] if torch.is_tensor(v) else v
                   for k, v in batch[side].items()}
            for side in ("inputs", "targets")
        }
        r = forward_with_latents(model, sl, device)
        for k, v in r["predictions"].items():
            preds.setdefault(k, []).append(v.cpu())
        for k, v in r["pooled"].items():
            pooled.setdefault(k, []).append(v.cpu())
    return (
        {k: torch.cat(v) for k, v in preds.items()},
        {k: torch.cat(v) for k, v in pooled.items()},
    )


def readout_scalars(preds, inv):
    """{readout: (N,) tensor} in physical units from z-space predictions."""
    out = {}
    for scalar, signal in READOUT_SIGNALS:
        if signal in preds:
            phys = inv.inverse(signal, preds[signal].float())
            out[scalar] = phys.mean(dim=tuple(range(1, phys.ndim)))
    return out


def perturb_additive(z, delta_c, sign):
    return z + sign * delta_c.view(1, -1, 1)


def perturb_multiplicative(z, name, factor, inv):
    phys = inv.inverse(name, z.float())
    return inv.apply(name, phys * factor).to(z.dtype)


def act_stats(batch, act_names):
    """Per-channel std of each actuator over the pool + per-window valid."""
    stats = {}
    for name in act_names:
        z = batch["targets"][name].float()
        finite = torch.isfinite(z)
        zc = torch.where(finite, z, torch.zeros_like(z))
        m = finite & (zc != 0)
        cnt = m.sum(dim=(0, 2)).clamp_min(1)
        mean = (zc * m).sum(dim=(0, 2)) / cnt
        var = (((zc - mean.view(1, -1, 1)) * m) ** 2).sum(dim=(0, 2)) / cnt
        stats[name] = {
            "std": var.sqrt(),
            "valid": (batch["targets"][f"{name}_valid"] > 0),
        }
    return stats


def with_actuator(batch, name, new_z):
    out = {"inputs": batch["inputs"],
           "targets": dict(batch["targets"])}
    out["targets"][name] = new_z
    return out


def main() -> int:
    args = parse_args()
    t0 = time.time()
    out_dir = Path(args.out_root) / "study_b"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_ckpt(args.checkpoint)
    ck_args = ckpt.get("args", {}) or {}
    model, diagnostics, actuators = build_model_from_ckpt(ckpt, device=args.device)
    diag_names = [c.name for c in diagnostics]
    act_names = [c.name for c in actuators]
    print(f"[{time.time()-t0:6.1f}s] model ready on {args.device}")

    _, val_files = resolve_split(
        ck_args.get("data_dir", "/lustre/orion/fus187/proj-shared/foundation_model"),
        seed=ck_args.get("seed", 42),
        val_fraction=ck_args.get("val_fraction", 0.1),
    )
    assert_val_matches_training_cache(val_files)
    stats = load_stats(ck_args["stats_path"])
    stride_s, warmup_s = 0.25, ck_args.get("warmup_s", 0.0)
    ds = make_prediction_dataset(
        val_files[: args.n_shots * 3], stats, ck_args, diagnostics, actuators,
        step_size_s=stride_s,
        lengths_cache_path=out_dir / "lengths_pool.pt",
        max_open_files=64,
    )
    samples, shot_names = build_pool(
        ds, shot_window_ranges(ds), args.n_shots, args.windows_per_shot,
        stride_s, warmup_s, t0,
    )
    pool = collate_fn_prediction(samples)
    n_pool = len(samples)
    print(f"[{time.time()-t0:6.1f}s] pool: {n_pool} windows from "
          f"{len(shot_names)} shots")

    inv = InversePreprocessor(
        str(ck_args["stats_path"]),
        dataset_kwargs={"channels_to_use": {"tangtv": [4, 6]}},
    )
    astats = act_stats(pool, act_names)

    # Base pass (shared by all studies).
    base_preds, base_pooled = batched_forward(
        model, pool, args.device, args.batch_size
    )
    base_scalars = readout_scalars(base_preds, inv)
    sigma_d = {
        name: float(base_preds[name].float().std()) for name in diag_names
    }
    print(f"[{time.time()-t0:6.1f}s] base pass done")

    run_all = args.study == "all"

    # ── B1: finite-difference sensitivity + sign matrix ──────────────
    if run_all or args.study == "b1":
        ptypes = ["add_0.5sigma", "mult_20pct"]
        S = np.full((len(act_names), len(ptypes), len(diag_names)), np.nan)
        sign_mean = {}
        sign_frac = {}
        n_valid_windows = {}
        for ai, aname in enumerate(act_names):
            zb = pool["targets"][aname].float()
            valid = astats[aname]["valid"].numpy()
            n_valid_windows[aname] = int(valid.sum())
            for pi, ptype in enumerate(ptypes):
                if ptype.startswith("add"):
                    delta = 0.5 * astats[aname]["std"]
                    zp = perturb_additive(zb, delta, +1.0)
                    zm = perturb_additive(zb, delta, -1.0)
                else:
                    zp = perturb_multiplicative(zb, aname, 1.2, inv)
                    zm = perturb_multiplicative(zb, aname, 0.8, inv)
                pp, _ = batched_forward(
                    model, with_actuator(pool, aname, zp),
                    args.device, args.batch_size,
                )
                pm, _ = batched_forward(
                    model, with_actuator(pool, aname, zm),
                    args.device, args.batch_size,
                )
                for di, dname in enumerate(diag_names):
                    d = (pp[dname].float() - pm[dname].float()).abs()
                    per_w = d.flatten(1).mean(1).numpy()
                    use = valid if valid.any() else np.ones_like(valid)
                    S[ai, pi, di] = float(per_w[use].mean()) / max(
                        sigma_d[dname], 1e-8
                    )
                sp = readout_scalars(pp, inv)
                sm = readout_scalars(pm, inv)
                for scalar in sp:
                    dv = (sp[scalar] - sm[scalar]).numpy()
                    use = valid if valid.any() else np.ones_like(valid)
                    dv = dv[use]
                    dv = dv[np.isfinite(dv)]
                    key = (aname, ptype, scalar)
                    sign_mean[key] = float(dv.mean()) if dv.size else float("nan")
                    sign_frac[key] = (
                        float((np.sign(dv) == np.sign(dv.mean())).mean())
                        if dv.size and dv.mean() != 0 else float("nan")
                    )
                print(f"[{time.time()-t0:7.1f}s] b1 {aname:16s} {ptype} done",
                      flush=True)

        np.savez(
            out_dir / "b1_sensitivity.npz",
            S=S, act_names=np.array(act_names), ptypes=np.array(ptypes),
            diag_names=np.array(diag_names),
            sigma_d=np.array([sigma_d[n] for n in diag_names]),
            act_sigma=np.array(
                [astats[n]["std"].mean().item() for n in act_names]
            ),
            n_valid_windows=np.array(
                [n_valid_windows[n] for n in act_names]
            ),
        )
        sign_rows = []
        for (aname, ptype, scalar), mv in sign_mean.items():
            exp = EXPECTED_SIGNS.get((aname, scalar))
            sign_rows.append({
                "actuator": aname, "ptype": ptype, "readout": scalar,
                "mean_delta": mv, "sign_consistency": sign_frac[
                    (aname, ptype, scalar)],
                "expected_sign": exp,
                "agrees": (None if exp is None or not np.isfinite(mv)
                           else bool(np.sign(mv) == exp)),
            })
        (out_dir / "b1_sign_matrix.json").write_text(
            json.dumps(sign_rows, indent=1)
        )
        scored = [r for r in sign_rows if r["agrees"] is not None
                  and r["ptype"] == "mult_20pct"]
        n_ok = sum(r["agrees"] for r in scored)
        print(f"[{time.time()-t0:7.1f}s] B1 done — physics sign agreement "
              f"{n_ok}/{len(scored)} (mult_20pct)")

    # ── B2: dose-response curves (K=1) ────────────────────────────────
    if run_all or args.study == "b2":
        curves = {}
        for aname in DOSE_ACTUATORS:
            if aname not in act_names:
                continue
            zb = pool["targets"][aname].float()
            valid = astats[aname]["valid"].numpy()
            for scale in DOSE_SCALES:
                if scale == 1.0:
                    sc = base_scalars
                else:
                    zs = perturb_multiplicative(zb, aname, scale, inv)
                    ps, _ = batched_forward(
                        model, with_actuator(pool, aname, zs),
                        args.device, args.batch_size,
                    )
                    sc = readout_scalars(ps, inv)
                for scalar, v in sc.items():
                    dv = (v - base_scalars[scalar]).numpy()
                    use = valid if valid.any() else np.ones_like(valid)
                    curves[f"{aname}|{scale}|{scalar}"] = dv[use]
            print(f"[{time.time()-t0:7.1f}s] b2 {aname} done", flush=True)
        np.savez(
            out_dir / "b2_dose_response.npz",
            **{k: v for k, v in curves.items()},
            scales=np.array(DOSE_SCALES),
            readouts=np.array([s for s, _ in READOUT_SIGNALS]),
            dose_actuators=np.array(
                [a for a in DOSE_ACTUATORS if a in act_names]
            ),
        )
        print(f"[{time.time()-t0:7.1f}s] B2 done")

    # ── swap: actuator-trajectory swap vs natural variability ─────────
    if run_all or args.study == "swap":
        perm = torch.arange(n_pool).roll(1)
        swapped = {"inputs": pool["inputs"], "targets": dict(pool["targets"])}
        for aname in act_names:
            swapped["targets"][aname] = pool["targets"][aname][perm]
            vk = f"{aname}_valid"
            if vk in swapped["targets"]:
                swapped["targets"][vk] = pool["targets"][vk][perm]
        sw_preds, sw_pooled = batched_forward(
            model, swapped, args.device, args.batch_size
        )

        def cos(a, b):
            a = a.float().flatten(1)
            b = b.float().flatten(1)
            return torch.nn.functional.cosine_similarity(a, b, dim=1)

        result = {}
        for name in diag_names + ["diag_global", "act_global"]:
            if name in base_pooled:
                lat_swap = cos(base_pooled[name], sw_pooled[name])
                lat_nat = cos(base_pooled[name], base_pooled[name][perm])
                result[f"lat_swap|{name}"] = lat_swap.numpy()
                result[f"lat_natural|{name}"] = lat_nat.numpy()
            if name in base_preds:
                result[f"pred_swap|{name}"] = cos(
                    base_preds[name], sw_preds[name]
                ).numpy()
                result[f"pred_natural|{name}"] = cos(
                    base_preds[name], base_preds[name][perm]
                ).numpy()
        np.savez(out_dir / "b3_actuator_swap.npz", **result)
        med = {
            k: float(np.median(v)) for k, v in result.items()
            if k.startswith(("lat_swap", "pred_swap"))
        }
        print(f"[{time.time()-t0:7.1f}s] swap done; median cos-sim "
              f"(swap vs base): "
              + ", ".join(f"{k.split('|')[1]}={v:.4f}"
                          for k, v in sorted(med.items())
                          if k.startswith("pred_swap")))

    meta = {
        "n_pool": n_pool, "shots": shot_names,
        "windows_per_shot": args.windows_per_shot,
        "t_range_s": [2.0, 8.0], "stride_s": stride_s,
        "dose_scales": DOSE_SCALES,
        "expected_signs": {f"{a}|{r}": s
                           for (a, r), s in EXPECTED_SIGNS.items()},
        "act_valid_windows": {n: int(astats[n]["valid"].sum())
                              for n in act_names},
    }
    (out_dir / "study_b_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[{time.time()-t0:7.1f}s] ALL DONE → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
