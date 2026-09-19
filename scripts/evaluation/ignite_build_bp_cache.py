"""Encode shots into a BAND-POWER frame-code cache (the `bp*` model family's format).

Why this exists: at step 13500 the 14-modality dynamics model predicts `mhr` at or BELOW a
constant-token baseline, so AE and tearing-mode questions are unanswerable on it. Peter's
band-power line trains on `mhr` + `co2` only -- exactly the MHD channel -- but its caches
were built before 199597 was encoded, and 199597 is the only genuinely held-out shot of the
three case studies. This builds the missing entries.

The band-power tokeniser has NO learned codec (that is the point): it is deterministic, so a
cache entry can be reproduced exactly. Math replicated from
``ignite_bandpower/build_bp128.py``:

  1. take the codec's RAW INPUT spectrogram window (C, n_freq, n_time) at t0_start
  2. split the FREQUENCY axis into N_BAND equal bands; per (channel, band) take the mean
     log-power over that band's bins AND over time -> (C * N_BAND,) per frame
  3. quantile-bin each column with the TRAINED bin edges -> level in [0, N_LEV)

Using Peter's own ``bin_edges_128.npz`` is mandatory: the levels are only meaningful
relative to the quantiles the model was trained against. Regenerating edges from different
shots would silently shift every token.

Actuators: borrowed from the production cache when the shot is there (those are the
time-base-patched ones the models were trained on); otherwise computed from the H5 with
``actuator_frames``. Which path was used is recorded per shot and the two are cross-checked
against each other whenever both are available.

Usage::

    python scripts/evaluation/ignite_build_bp_cache.py \
        --shots 199597,199598 \
        --data-dir /lustre/orion/fus187/proj-shared/additional_data \
        --bp-root /lustre/orion/fus187/proj-shared/models/ignite_bp128 \
        --out data/outputs/ignite_bp_cache_ext
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "src"))

CODEC_SNAP = Path("/lustre/orion/fus187/proj-shared/models/ignite_codecs_current")
PROD_CACHE = Path("/lustre/orion/fus187/proj-shared/models/ignite_production/frame_codes")
MODS = ["co2", "mhr"]
N_LEV = 8
T0_DEFAULT = 0.0          # must match the cache manifest's window origin


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", required=True, help="comma-separated shot numbers")
    ap.add_argument("--data-dir", required=True, help="dir holding <shot>_processed.h5")
    ap.add_argument("--bp-root", default="/lustre/orion/fus187/proj-shared/models/ignite_bp128",
                   help="bp family root: supplies bin_edges*.npz and the reference frame_codes")
    ap.add_argument("--out", required=True, help="output cache dir (new entries land here)")
    ap.add_argument("--t0-start", type=float, default=T0_DEFAULT)
    ap.add_argument("--force", action="store_true", help="rebuild entries that already exist")
    return ap.parse_args()


def band_stats(mod: str, shot: str, data_dir, n_band: int, t0: float) -> np.ndarray:
    """(n_frames, C*n_band) mean log-power per (channel, band). Peter's build_bp128 math."""
    from tokamak_foundation_model.ignite.train_dynamics import _single_shot_dataset
    cfg = torch.load(CODEC_SNAP / mod / "codec_best.pt", map_location="cpu",
                     weights_only=False)["cfg"]
    ds = _single_shot_dataset(mod, "spectro", cfg, shot, data_dir, t0_start=t0)
    rows = []
    for t in range(len(ds)):
        it = ds[t]
        x = np.asarray(it[0] if isinstance(it, tuple) else it, dtype=np.float64)
        n_freq = x.shape[1]
        e = np.linspace(0, n_freq, n_band + 1).astype(int)
        rows.append(np.stack([x[:, e[i]:e[i + 1], :].mean(axis=(1, 2))
                              for i in range(n_band)], -1).reshape(-1))
    return np.asarray(rows)


def quantize(A: np.ndarray, q: np.ndarray) -> np.ndarray:
    """(n_frames, D) band powers + (N_LEV-1, D) trained quantiles -> int32 levels."""
    tk = np.stack([np.searchsorted(q[:, j], A[:, j]) for j in range(A.shape[1])], -1)
    return np.clip(tk, 0, N_LEV - 1).astype(np.int32)


def main() -> int:
    from tokamak_foundation_model.ignite.train_dynamics import actuator_frames

    args = parse_args()
    bp_root = Path(args.bp_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shots = [s.strip() for s in args.shots.split(",") if s.strip()]

    ed_path = next((f for f in (bp_root / "bin_edges_128.npz", bp_root / "bin_edges.npz")
                    if f.exists()), None)
    if ed_path is None:
        raise SystemExit(f"no bin_edges*.npz under {bp_root}")
    edges = np.load(ed_path)
    print(f"[bp] trained bin edges: {ed_path}")

    # Reference entry fixes the layout the model expects; also infers n_band.
    ref_files = sorted((bp_root / "frame_codes").glob("*.pt"))
    if not ref_files:
        raise SystemExit(f"no reference entries in {bp_root/'frame_codes'}")
    ref = torch.load(ref_files[0], map_location="cpu", weights_only=False)
    ref_layout = {m: (int(v.shape[1]), int(ref["vocabs"][m])) for m, v in ref["codes"].items()}
    print(f"[bp] reference layout from {ref_files[0].name}: {ref_layout}")

    # n_band is not a free parameter: D = C * n_band is fixed by the trained edges, and D must
    # equal the reference token count or the edges and frame_codes came from different builds.
    for m in MODS:
        if edges[m].shape[1] != ref_layout[m][0]:
            raise SystemExit(f"{m}: edges D={edges[m].shape[1]} != reference tokens "
                             f"{ref_layout[m][0]} -- bin edges and frame_codes disagree")
        if edges[m].shape[0] != N_LEV - 1:
            raise SystemExit(f"{m}: edges have {edges[m].shape[0]} cut points, expected "
                             f"{N_LEV - 1} for {N_LEV} levels")

    written, failed = [], []
    for shot in shots:
        dest = out / f"{shot}.pt"
        if dest.exists() and not args.force:
            print(f"[bp] {shot}: already present, skipping")
            continue
        h5 = Path(args.data_dir) / f"{shot}_processed.h5"
        if not h5.exists():
            print(f"[FAIL] {shot}: {h5} missing")
            failed.append(shot)
            continue
        try:
            codes, vocabs, nmin = {}, {}, None
            for m in MODS:
                D = edges[m].shape[1]
                # C is whatever the codec input has; n_band = D / C
                cfg = torch.load(CODEC_SNAP / m / "codec_best.pt", map_location="cpu",
                                 weights_only=False)["cfg"]
                from tokamak_foundation_model.ignite.train_dynamics import _single_shot_dataset
                ds0 = _single_shot_dataset(m, "spectro", cfg, shot, args.data_dir,
                                           t0_start=args.t0_start)
                it0 = ds0[0]
                C = np.asarray(it0[0] if isinstance(it0, tuple) else it0).shape[0]
                if D % C:
                    raise RuntimeError(f"{m}: edges D={D} not divisible by C={C}")
                nb = D // C
                A = band_stats(m, shot, args.data_dir, nb, args.t0_start)
                codes[m] = quantize(A, edges[m])
                vocabs[m] = N_LEV
                nmin = codes[m].shape[0] if nmin is None else min(nmin, codes[m].shape[0])
                print(f"[bp] {shot}/{m}: C={C} n_band={nb} D={D} frames={codes[m].shape[0]}")

            # Actuators: prefer the production cache's patched copy; else derive from the H5.
            prod = PROD_CACHE / f"{shot}.pt"
            if prod.exists():
                p = torch.load(prod, map_location="cpu", weights_only=False)
                act = p["actuators"]
                src = "production cache (patched)"
                nmin = min(nmin, int(p["n_frames"]))
                # cross-check the H5 path against it so a silent time-base drift shows up
                try:
                    a2 = actuator_frames(shot, nmin, args.data_dir,
                                         t0_start=args.t0_start).to(torch.float16)
                    d = (a2[:nmin].float() - act[:nmin].float())
                    rms = float(d.pow(2).mean().sqrt())
                    ref_rms = float(act[:nmin].float().pow(2).mean().sqrt())
                    print(f"[bp] {shot}: actuator cross-check rms(H5 - cache)/rms = "
                          f"{rms / max(ref_rms, 1e-9):.4f}")
                except Exception as e:
                    print(f"[bp] {shot}: actuator cross-check skipped ({type(e).__name__})")
            else:
                act = actuator_frames(shot, nmin, args.data_dir,
                                      t0_start=args.t0_start).to(torch.float16)
                src = "derived from H5"

            payload = {"codes": {m: torch.tensor(codes[m][:nmin]) for m in MODS},
                       "actuators": act[:nmin].clone(),
                       "n_frames": int(nmin), "vocabs": vocabs}
            lay = {m: (int(v.shape[1]), int(payload["vocabs"][m]))
                   for m, v in payload["codes"].items()}
            if lay != ref_layout:
                raise RuntimeError(f"layout {lay} != reference {ref_layout}")
            tmp = out / f".{shot}.tmp"
            torch.save(payload, tmp)
            tmp.replace(dest)
            dead = [m for m, v in payload["codes"].items() if torch.unique(v).numel() <= 1]
            print(f"[ok]   {shot}: n_frames={nmin} actuators {tuple(act[:nmin].shape)} "
                  f"({src}) live={len(MODS) - len(dead)}/{len(MODS)}"
                  + (f" STATIC={dead}" if dead else ""))
            written.append(shot)
        except Exception as e:
            print(f"[FAIL] {shot}: {type(e).__name__}: {e}")
            failed.append(shot)

    print(f"\n[bp] wrote {len(written)}: {written}" + (f"  FAILED {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
