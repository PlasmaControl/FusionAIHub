"""Rollout skill as a function of training step — the exposure-bias regression test.

The documented pathology is that rollout skill INVERTS with more training (band-power
line: TM-band skill +0.153 @ step 11k -> -0.862 @ step 20k) while teacher-forced CE
keeps improving. A fix for the train/test gap must flatten or reverse that curve, so
this harness plots skill against step for one or more checkpoint directories.

Skill = token accuracy over the predicted region minus the persistence baseline
(fraction of tokens equal to the last seed frame). Persistence is mandatory: discrete
codes at 50 ms are highly persistent, so raw accuracy is not interpretable. Compare
against the MAJORITY-TOKEN baseline too — a modality that cannot beat the frequency of
its commonest ground-truth code is degenerate and its skill number means nothing.

    python ignite_skill_vs_step.py --run_dir <dir> [--run_dir <dir2>] \
        --cache_dir <frame_codes> --shots 199597,190735 --out_dir <out>

MEASURED CAVEAT (2026-08-18, bp128_d512L8, shot 199597) — READ BEFORE USING THIS AS THE
ACCEPTANCE GATE. Token skill as computed here is BLIND to the documented pathology. On the
band-power line between step 11k and step 20k:

    decoded mhr TM 1-20 kHz nRMSE skill : +0.153 -> -0.862   (the documented inversion)
    TOKEN skill, mhr, this harness      : +0.249 -> +0.259   (flat / slightly BETTER)
    TOKEN skill, co2, this harness      : +0.131 -> +0.130   (flat)

Verified robust to the RNG seed (seed 0: +0.255 -> +0.280), to trimming the frames past the
digitiser record (mhr is data-valid only to frame 75 on this shot: +0.254 -> +0.319), and to
using dynamics_best.pt @10500 as the stand-in for the vanished step-11000 checkpoint
(+0.249 vs +0.255 on the archived step-11000 rollout). The failure is DISTRIBUTIONAL —
variance over-prediction inside one frequency band — and it leaves per-token accuracy intact
while the decoded band power diverges. Note also that the inversion is band-specific: over
the same two steps mhr AE 50-250 kHz goes +0.175 -> +0.767 and co2 TM goes -1.178 -> +0.064.

So a flat curve here does NOT license the claim that an arm stopped inverting. The plan's own
validation protocol (§5) requires "token skill AND decoded band-restricted skill"; only the
first half is implemented here. The decoded half needs the band-power inverse (deterministic
bin edges, no learned codec) — see scripts/evaluation/ignite_bp_cases.py and
ignite_case_panels.decoded_space_skill, which produced the numbers above.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import torch                                                             # noqa: E402


def _payload_step(path: Path):
    """The `step` field of a checkpoint, read WITHOUT materializing the weights.

    mmap=True keeps the tensor storages on disk, so this touches only the pickle header
    and never the GPU (MEASURED 0.8 s on a warm 412 MB bp128 checkpoint; a COLD Lustre
    read of the same file can take tens of seconds, so do not expect it to be free on
    the first pass over a run dir). map_location
    "meta" also works on these payloads; "cpu" is used because it is the same read path
    load_model takes, and weights_only=False for the same reason — these payloads carry
    non-tensor config (`modalities` is a tuple of tuples), and a future entry that the
    safe unpickler's allowlist rejects would break a probe that only wants an int.
    """
    try:
        ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except Exception as exc:                                             # pragma: no cover
        print(f"[skill] WARNING cannot read {path.name}: {type(exc).__name__}: {exc}")
        return None
    step = ck.get("step")
    del ck
    return None if step is None else int(step)


def checkpoints(run_dir: Path):
    """Every distinct-step checkpoint in a run dir, ascending by step.

    Step-tagged files are preferred, but the production runs do NOT keep any: bp128_d512L8
    holds ONLY dynamics_best.pt and dynamics_latest.pt (the trainer overwrites both every
    ckpt_every steps). Ingesting `best` as well as `latest` is what makes the two-point
    regression curve — and the documented 11k -> 20k inversion — measurable at all; with
    `latest` alone the curve is a single point and this harness proves nothing.

    Steps are read from the PAYLOAD for both un-tagged files, never guessed from mtime.
    """
    by_step: dict = {}
    for p in sorted(run_dir.glob("dynamics_step*.pt")):
        m = re.search(r"step(\d+)", p.name)
        if m:
            by_step.setdefault(int(m.group(1)), p)
    # de-dup by STEP, step-tagged wins: `best` and `latest` are aliases of steps that may
    # already be present under an explicit name (and are each other's alias when the last
    # checkpoint was also the best).
    for name in ("dynamics_best.pt", "dynamics_latest.pt"):
        p = run_dir / name
        if not p.exists():
            continue
        step = _payload_step(p)
        if step is None:
            print(f"[skill] WARNING {p} carries no `step` field — skipped")
            continue
        if step in by_step:
            print(f"[skill] {p.name} is step {step}, already covered by "
                  f"{by_step[step].name} — skipped")
            continue
        by_step[step] = p
    return sorted(by_step.items())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", action="append", required=True,
                    help="checkpoint directory; repeat for multiple arms")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--shots", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k0", type=int, default=20, help="seed frames (0 => use cfg.k0_seed)")
    ap.add_argument("--seed", type=int, default=1234, help="rollout RNG seed (reproducible)")
    # Task-11 sampler flags, constructed exactly as eval_dynamics.main does so an arm
    # measured here is the same decode policy the single-shot eval reports.
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--global_pool", action="store_true")
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--revision_rounds", type=int, default=0)
    ap.add_argument("--cfg_scale", type=float, default=1.0)
    ap.add_argument("--best_of_n", type=int, default=1)
    args = ap.parse_args()

    from tokamak_foundation_model.ignite.eval_dynamics import (
        load_model, load_shot_cache, rollout_shot)
    from tokamak_foundation_model.ignite.sampling import SamplerConfig

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shots = [s.strip() for s in args.shots.split(",") if s.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampler = SamplerConfig(temperature=args.temperature, top_p=args.top_p,
                            global_pool=args.global_pool,
                            revision_rounds=args.revision_rounds,
                            cfg_scale=args.cfg_scale)
    print(f"[skill] device={device} shots={shots} k0={args.k0} seed={args.seed} "
          f"global_pool={args.global_pool} top_p={args.top_p} "
          f"revision_rounds={args.revision_rounds} cfg_scale={args.cfg_scale} "
          f"best_of_n={args.best_of_n}", flush=True)
    results: dict = defaultdict(dict)

    for run in args.run_dir:
        arm = Path(run).name
        ckpts = checkpoints(Path(run))
        if not ckpts:
            raise SystemExit(f"no checkpoints with a readable step in {run}")
        print(f"[skill] {arm}: {[(s, p.name) for s, p in ckpts]}", flush=True)
        for step, ckpt in ckpts:
            # load_model(ckpt_path, device) -> (model, cfg, step)   [eval_dynamics.py:121]
            model, cfg, _ = load_model(Path(ckpt), device)
            k0 = args.k0 or int(cfg.k0_seed)
            per_mod = defaultdict(list)
            for shot in shots:
                cache = load_shot_cache(Path(args.cache_dir), shot)
                # rollout_shot(model, cfg, cache, K0, temperature, generator, device, ...)
                # -> (gt_codes, pred_codes, K0, F); tensors are (F, n_tok) cpu long, NO batch dim.
                # The generator must live on `device` — that is the convention the rest of
                # eval_dynamics uses (eval_dynamics.py:1061); a CPU generator raises inside
                # torch.multinomial on a CUDA distribution.
                gt, pred, K0, F = rollout_shot(
                    model, cfg, cache, k0, args.temperature,
                    torch.Generator(device=device).manual_seed(args.seed), device,
                    sampler=sampler, best_of=args.best_of_n)
                for name in pred:
                    g, p = gt[name][K0:F], pred[name][K0:F]
                    acc = float((g == p).float().mean())
                    last = gt[name][K0 - 1: K0]
                    pers = float((g == last).float().mean())
                    # majority-token guard: the frequency of the commonest GT code
                    vals, cnt = torch.unique(g, return_counts=True)
                    major = float(cnt.max()) / float(g.numel())
                    per_mod[name].append({"skill": acc - pers, "acc": acc,
                                          "persistence": pers, "majority": major,
                                          "beats_majority": acc > major})
            results[arm][step] = {
                n: {k: (sum(d[k] for d in v) / len(v) if isinstance(v[0][k], float)
                        else all(d[k] for d in v))
                    for k in v[0]}
                for n, v in per_mod.items()}
            print(f"[skill] {arm} step {step}: "
                  + ", ".join(f"{n}={d['skill']:+.3f}"
                              + ("" if d["beats_majority"] else "(DEGENERATE)")
                              for n, d in results[arm][step].items()), flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    with open(out_dir / "skill_vs_step.json", "w") as f:
        json.dump(results, f, indent=2)

    mods = sorted({n for arm in results.values() for s in arm.values() for n in s})
    fig, axes = plt.subplots(len(mods), 1, figsize=(9, 2.6 * len(mods)), sharex=True,
                             squeeze=False)
    for ax, n in zip(axes[:, 0], mods):
        for arm, by_step in results.items():
            xs = sorted(by_step)
            ys = [by_step[s].get(n, {}).get("skill", float("nan")) for s in xs]
            ax.plot(xs, ys, marker="o", label=arm)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_ylabel(f"{n}\nskill")
        ax.legend(fontsize=7)
    axes[-1, 0].set_xlabel("training step")
    fig.suptitle("Rollout skill vs training step (skill must not invert)")
    fig.tight_layout()
    fig.savefig(out_dir / "skill_vs_step.png", dpi=110)
    print("wrote", out_dir / "skill_vs_step.png")


if __name__ == "__main__":
    main()
