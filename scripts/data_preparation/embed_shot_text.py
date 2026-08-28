"""Embed per-shot DIII-D text bundles with Qwen3-Embedding-8B into a consolidated H5.

Reads the per-shot JSONL text bundles (one `{"doc_id","run_id","shot","text"}` record per
line, one file per run), splits each bundle into the strictly pre-experiment "input" slice
and the whole "total" bundle (see `tokamak_foundation_model.ignite.text_embed.split_bundle`),
embeds both with Qwen3-Embedding-8B, and writes them to a resume-friendly per-shard H5 cache
that `--merge` consolidates into one file for the dynamics-model conditioning task to load.

`--merge` and `--verify` never import transformers/torch-GPU -- only the default (embedding)
mode does, and it does so lazily, so this script stays import-clean without a GPU or
transformers installed.

Usage::

    # pilot on a handful of shots (single process, GPU)
    python scripts/data_preparation/embed_shot_text.py --limit 20 --device cuda

    # sharded run under SLURM (SLURM_PROCID / SLURM_NTASKS pick the shard automatically)
    srun python scripts/data_preparation/embed_shot_text.py

    # consolidate shard files into the final H5 (single process, no GPU)
    python scripts/data_preparation/embed_shot_text.py --merge

    # sanity-check the consolidated H5
    python scripts/data_preparation/embed_shot_text.py --verify
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "src"))

from tokamak_foundation_model.ignite.text_embed import split_bundle  # noqa: E402

EMBED_DIM = 4096


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--jsonl_dir",
        default="/lustre/orion/fus187/proj-shared/foundation_model_text/shotsummary/processed/per_shot_jsonl",
    )
    p.add_argument("--out", default="/lustre/orion/fus187/proj-shared/foundation_model/text_embeddings.h5")
    p.add_argument("--shard_dir", default=None, help="default: <out's parent>/text_embeddings_shards")
    p.add_argument("--model_id", default="Qwen/Qwen3-Embedding-8B")
    p.add_argument("--max_length", type=int, default=32768)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_PROCID", 0)))
    p.add_argument("--n_shards", type=int, default=int(os.environ.get("SLURM_NTASKS", 1)))
    p.add_argument("--limit", type=int, default=0, help="0 = all; pilot uses a small N")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--cache_dir",
        default="/lustre/orion/fus187/proj-shared/models/ignite_production/frame_codes",
        help="verify-mode coverage reference",
    )
    p.add_argument("--merge", action="store_true", help="merge shard files into consolidated H5, then exit")
    p.add_argument("--verify", action="store_true", help="readback checks on --out, then exit")
    args = p.parse_args()
    if args.shard_dir is None:
        args.shard_dir = str(Path(args.out).parent / "text_embeddings_shards")
    return args


def _parse_run_id(raw) -> int:
    """Most run_ids are plain digit strings (e.g. "20240401"); a handful carry a trailing
    letter suffix for same-day re-runs (e.g. "20260303A"). Strip any non-digit suffix so the
    date-based id still round-trips to an int; fall back to 0 if nothing numeric remains."""
    s = str(raw)
    digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else 0


def _load_shot_texts(jsonl_dir):
    """Return {int(shot): (int(run_id), text)} deterministically, skipping shot 0 and
    duplicate shots (first occurrence wins). Also returns the duplicate count."""
    shots: dict[int, tuple[int, str]] = {}
    n_dupes = 0
    for path in sorted(glob.glob(os.path.join(jsonl_dir, "*.jsonl"))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                shot = int(rec["shot"])
                if shot == 0:
                    continue
                if shot in shots:
                    n_dupes += 1
                    continue
                shots[shot] = (_parse_run_id(rec["run_id"]), rec["text"])
    return shots, n_dupes


def _git_sha():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_HERE.parents[2]),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _shard_path(shard_dir, shard, n_shards):
    return Path(shard_dir) / f"shard_{shard:03d}of{n_shards:03d}.h5"


def _embed_texts(tok, mdl, device, texts, max_length):
    """Embed a batch of texts with Qwen3-Embedding-8B's last-token pooling.

    Returns (emb_fp16 (N, EMBED_DIM) ndarray, n_tok (N,) list of pre-truncation token
    counts, truncated (N,) list of bools).
    """
    import torch

    n_tok = [len(tok(t).input_ids) for t in texts]
    truncated = [n > max_length for n in n_tok]

    # documents get NO instruction prefix (instructions are for queries only); append EOS
    # so the pooled (last, with left padding) position is always the true EOS token.
    texts_with_eos = [t + tok.eos_token for t in texts]
    batch = tok(texts_with_eos, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    with torch.no_grad():
        out = mdl(**{k: v.to(device) for k, v in batch.items()})
    emb = out.last_hidden_state[:, -1, :]
    emb = torch.nn.functional.normalize(emb.float(), dim=-1)
    return emb.half().cpu().numpy(), n_tok, truncated


def run_embed(args):
    import torch
    from transformers import AutoModel, AutoTokenizer

    shots, n_dupes = _load_shot_texts(args.jsonl_dir)
    shots_sorted = sorted(shots.keys())
    if args.limit > 0:
        shots_sorted = shots_sorted[: args.limit]
    my_shots = shots_sorted[args.shard :: args.n_shards]
    print(f"[shard {args.shard}/{args.n_shards}] {len(my_shots)} shots to process "
          f"(total {len(shots_sorted)}, {n_dupes} duplicate shots skipped)")

    os.makedirs(args.shard_dir, exist_ok=True)
    shard_path = _shard_path(args.shard_dir, args.shard, args.n_shards)

    tok = AutoTokenizer.from_pretrained(args.model_id, padding_side="left")
    mdl = AutoModel.from_pretrained(args.model_id, dtype=torch.float16, attn_implementation="sdpa")
    mdl = mdl.to(args.device).eval()

    n_skipped_empty = 0
    n_errors = 0
    n_done = 0
    with h5py.File(shard_path, "a") as f:
        for i, shot in enumerate(my_shots):
            key = str(shot)
            if key in f and f[key].attrs.get("complete"):
                continue
            if key in f:
                del f[key]
            run_id, text = shots[shot]
            try:
                input_text, total_text = split_bundle(text)
                if not input_text.strip() or not total_text.strip():
                    n_skipped_empty += 1
                    continue

                emb, n_tok, truncated = _embed_texts(
                    tok, mdl, args.device, [input_text, total_text], args.max_length
                )

                grp = f.create_group(key)
                grp.create_dataset("input", data=emb[0].astype(np.float16))
                grp.create_dataset("total", data=emb[1].astype(np.float16))
                grp.attrs["run_id"] = run_id
                grp.attrs["n_tok_input"] = n_tok[0]
                grp.attrs["n_tok_total"] = n_tok[1]
                grp.attrs["truncated_input"] = truncated[0]
                grp.attrs["truncated_total"] = truncated[1]
                grp.attrs["complete"] = True
                n_done += 1
            except Exception as e:
                n_errors += 1
                print(f"[shard {args.shard}] ERROR shot {shot}: {e}")
                continue

            if (i + 1) % 25 == 0:
                print(f"[shard {args.shard}] {i + 1}/{len(my_shots)} "
                      f"(done={n_done} skipped_empty={n_skipped_empty} errors={n_errors})")

    print(f"[shard {args.shard}] finished: done={n_done} skipped_empty={n_skipped_empty} errors={n_errors}")


def run_merge(args):
    shard_paths = sorted(glob.glob(os.path.join(args.shard_dir, "shard_*.h5")))
    if not shard_paths:
        raise SystemExit(f"no shard files found in {args.shard_dir}")

    rows = {}
    for sp in shard_paths:
        with h5py.File(sp, "r") as f:
            for key in f.keys():
                grp = f[key]
                if not grp.attrs.get("complete"):
                    continue
                shot = int(key)
                if shot in rows:
                    raise AssertionError(f"duplicate shot {shot} found across shard files (last seen in {sp})")
                rows[shot] = {
                    "run_id": int(grp.attrs["run_id"]),
                    "input": grp["input"][:],
                    "total": grp["total"][:],
                    "n_tok_input": int(grp.attrs["n_tok_input"]),
                    "n_tok_total": int(grp.attrs["n_tok_total"]),
                    "truncated_input": bool(grp.attrs["truncated_input"]),
                    "truncated_total": bool(grp.attrs["truncated_total"]),
                }

    shots_sorted = sorted(rows.keys())
    n = len(shots_sorted)
    print(f"merging {n} complete shots from {len(shard_paths)} shard files")

    all_shots, _ = _load_shot_texts(args.jsonl_dir)
    n_skipped = len(all_shots) - n

    out_path = Path(args.out)
    tmp_path = out_path.with_suffix(f".h5.tmp.{os.getpid()}")
    os.makedirs(out_path.parent, exist_ok=True)
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("shots", data=np.array(shots_sorted, dtype=np.int64))
        f.create_dataset("run_id", data=np.array([rows[s]["run_id"] for s in shots_sorted], dtype=np.int64))
        f.create_dataset("input", data=np.stack([rows[s]["input"] for s in shots_sorted]).astype(np.float16))
        f.create_dataset("total", data=np.stack([rows[s]["total"] for s in shots_sorted]).astype(np.float16))
        f.create_dataset(
            "n_tok_input", data=np.array([rows[s]["n_tok_input"] for s in shots_sorted], dtype=np.int32)
        )
        f.create_dataset(
            "n_tok_total", data=np.array([rows[s]["n_tok_total"] for s in shots_sorted], dtype=np.int32)
        )
        f.create_dataset(
            "truncated_input", data=np.array([rows[s]["truncated_input"] for s in shots_sorted], dtype=bool)
        )
        f.create_dataset(
            "truncated_total", data=np.array([rows[s]["truncated_total"] for s in shots_sorted], dtype=bool)
        )
        f.attrs["model_id"] = args.model_id
        f.attrs["hf_revision"] = ""
        f.attrs["max_length"] = args.max_length
        f.attrs["pooling"] = "last_token"
        f.attrs["instruction"] = ""
        f.attrs["normalized"] = True
        f.attrs["embed_dim"] = EMBED_DIM
        f.attrs["input_rule"] = (
            "header + general-session-info/metadata + planned(mini-proposal); "
            "excludes session summaries and shot-specific results"
        )
        f.attrs["source_dir"] = args.jsonl_dir
        f.attrs["git_sha"] = _git_sha()
        f.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        f.attrs["n_shots"] = n
        f.attrs["n_skipped"] = n_skipped
        f.attrs["complete"] = True

    os.replace(tmp_path, out_path)
    print(f"wrote {out_path} ({n} shots)")


def run_verify(args):
    ok = True
    with h5py.File(args.out, "r") as f:
        expected_datasets = {
            "shots": np.int64,
            "run_id": np.int64,
            "input": np.float16,
            "total": np.float16,
            "n_tok_input": np.int32,
            "n_tok_total": np.int32,
            "truncated_input": bool,
            "truncated_total": bool,
        }
        n = f["shots"].shape[0] if "shots" in f else 0
        for name, dtype in expected_datasets.items():
            if name not in f:
                print(f"FAIL: missing dataset /{name}")
                ok = False
                continue
            ds = f[name]
            if ds.shape[0] != n:
                print(f"FAIL: /{name} has {ds.shape[0]} rows, expected {n}")
                ok = False
            if not np.issubdtype(ds.dtype, dtype) and ds.dtype != np.dtype(dtype):
                print(f"FAIL: /{name} dtype {ds.dtype}, expected {dtype}")
                ok = False
        for name in ("input", "total"):
            if f[name].shape != (n, EMBED_DIM):
                print(f"FAIL: /{name} shape {f[name].shape}, expected ({n}, {EMBED_DIM})")
                ok = False

        required_attrs = [
            "model_id", "hf_revision", "max_length", "pooling", "instruction", "normalized",
            "embed_dim", "input_rule", "source_dir", "git_sha", "created_utc", "n_shots",
            "n_skipped", "complete",
        ]
        for attr in required_attrs:
            if attr not in f.attrs:
                print(f"FAIL: missing attr {attr}")
                ok = False
        if not f.attrs.get("complete"):
            print("FAIL: attrs['complete'] is not True")
            ok = False

        input_data = f["input"][:].astype(np.float32)
        total_data = f["total"][:].astype(np.float32)
        for name, data in (("input", input_data), ("total", total_data)):
            if not np.isfinite(data).all():
                print(f"FAIL: NaN/Inf found in /{name}")
                ok = False
            norms = np.linalg.norm(data, axis=1)
            bad = np.sum((norms < 0.98) | (norms > 1.02))
            if bad > 0:
                print(f"FAIL: {bad} rows in /{name} have L2 norm outside [0.98, 1.02]")
                ok = False

        shots = f["shots"][:]
        run_ids = f["run_id"][:]

        cache_stems = set()
        for p in Path(args.cache_dir).glob("*.pt"):
            try:
                cache_stems.add(int(p.stem))
            except ValueError:
                continue
        if cache_stems:
            coverage = len(set(shots.tolist()) & cache_stems) / len(cache_stems)
        else:
            coverage = float("nan")
        print(f"coverage: |shots ∩ cache_stems| / |cache_stems| = {coverage:.3f} "
              f"({len(cache_stems)} cache stems)")

        rng = np.random.default_rng(0)
        by_run: dict[int, list[int]] = {}
        for idx, rid in enumerate(run_ids.tolist()):
            by_run.setdefault(rid, []).append(idx)
        multi_shot_runs = [idxs for idxs in by_run.values() if len(idxs) >= 2]
        rng.shuffle(multi_shot_runs)
        multi_shot_runs = multi_shot_runs[:30]

        def _cos(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

        within_pairs = []
        for idxs in multi_shot_runs:
            idxs = list(idxs)
            i, j = rng.choice(len(idxs), size=2, replace=False)
            within_pairs.append((idxs[i], idxs[j]))

        n_pairs = len(within_pairs)
        cross_pairs = []
        all_idx = np.arange(n)
        while len(cross_pairs) < n_pairs and n >= 2:
            i, j = rng.choice(all_idx, size=2, replace=False)
            if run_ids[i] != run_ids[j]:
                cross_pairs.append((i, j))

        if within_pairs and cross_pairs:
            within_mean = float(np.mean([_cos(input_data[i], input_data[j]) for i, j in within_pairs]))
            cross_mean = float(np.mean([_cos(input_data[i], input_data[j]) for i, j in cross_pairs]))
            cosine_pass = within_mean > cross_mean
            print(f"cosine sanity: within-run mean={within_mean:.4f} cross-run mean={cross_mean:.4f} "
                  f"({'PASS' if cosine_pass else 'FAIL'})")
            ok = ok and cosine_pass
        else:
            print("cosine sanity: SKIPPED (not enough multi-shot runs)")

    print("VERIFY " + ("PASSED" if ok else "FAILED"))
    return ok


def main():
    args = _parse_args()
    if args.merge:
        run_merge(args)
        return
    if args.verify:
        ok = run_verify(args)
        sys.exit(0 if ok else 1)
    run_embed(args)


if __name__ == "__main__":
    main()
