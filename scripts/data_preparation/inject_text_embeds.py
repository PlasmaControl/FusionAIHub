"""Inject per-shot text embeddings from the consolidated H5 into per-shot `{shot}_processed.h5`
files, so each production shot file carries everything about that shot.

The consolidated `text_embeddings.h5` (built by `embed_shot_text.py --merge`) remains the
canonical source that IGNITE training/eval reads directly. This script is pure co-location:
it copies the `input`/`total` embedding vectors and their provenance for each shot into a new
`text_embed` group inside that shot's `{shot}_processed.h5` file, additively, and never touches
any other group or attribute in the shot files.

Single-process, CPU-only, h5py + numpy only (no torch import needed). Runs on a login node.

Usage::

    # pilot dry run, first 5 shots
    python scripts/data_preparation/inject_text_embeds.py --dry_run --limit 5

    # real run
    python scripts/data_preparation/inject_text_embeds.py

    # re-run, replacing existing text_embed groups
    python scripts/data_preparation/inject_text_embeds.py --overwrite

    # sanity-check a sample of already-injected shots
    python scripts/data_preparation/inject_text_embeds.py --verify --limit 25
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

# Attrs copied verbatim from the consolidated H5's root attrs into each shot's text_embed group.
_ROOT_ATTRS_TO_COPY = ["model_id", "embed_dim", "pooling", "normalized", "input_rule", "max_length"]
# Per-shot row attrs (one value per shot, read from the corresponding row).
_ROW_ATTRS = ["n_tok_input", "n_tok_total", "truncated_input", "truncated_total"]


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeds", default="/lustre/orion/fus187/proj-shared/foundation_model/text_embeddings.h5")
    p.add_argument("--shot_dir", default="/lustre/orion/fus187/proj-shared/foundation_model")
    p.add_argument("--group", default="text_embed")
    p.add_argument("--overwrite", action="store_true", help="delete+recreate an existing text_embed group")
    p.add_argument("--dry_run", action="store_true", help="print planned writes, write nothing")
    p.add_argument("--limit", type=int, default=0, help="0 = all shots; N caps to first N shots (pilot/verify sample)")
    p.add_argument("--verify", action="store_true", help="re-open injected shots and assert bit-equality; no writes")
    args = p.parse_args(argv)
    return args


def _shot_file_path(shot_dir, shot) -> Path:
    return Path(shot_dir) / f"{shot}_processed.h5"


def _row_value(ds, i):
    """Read one row from a per-shot dataset, converting numpy scalar types to native python."""
    v = ds[i]
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def run_inject(args):
    embeds_path = Path(args.embeds).resolve()
    with h5py.File(embeds_path, "r") as ef:
        if not ef.attrs.get("complete"):
            raise SystemExit(
                f"{embeds_path} does not have complete=True; run "
                "`embed_shot_text.py --merge` (and --verify) first."
            )

        shots = ef["shots"][:]
        n_total = shots.shape[0]
        n = n_total if args.limit <= 0 else min(args.limit, n_total)

        embed_dim = ef["input"].shape[1]
        assert ef["total"].shape[1] == embed_dim

        root_attrs = {k: ef.attrs[k] for k in _ROOT_ATTRS_TO_COPY if k in ef.attrs}
        source_created_utc = ef.attrs.get("created_utc", "")
        injected_utc = datetime.now(timezone.utc).isoformat()

        counts = {"injected": 0, "skipped_existing": 0, "no_shot_file": 0, "error": 0}
        planned_samples = []

        for i in range(n):
            shot = int(shots[i])
            shot_path = _shot_file_path(args.shot_dir, shot)

            if not shot_path.exists():
                counts["no_shot_file"] += 1
                continue

            try:
                if args.dry_run:
                    with h5py.File(shot_path, "r") as sf:
                        exists = args.group in sf
                    if exists and not args.overwrite:
                        counts["skipped_existing"] += 1
                    else:
                        counts["injected"] += 1
                        if len(planned_samples) < 10:
                            planned_samples.append(shot)
                    continue

                if not args.overwrite:
                    with h5py.File(shot_path, "r") as sf:
                        exists = args.group in sf
                    if exists:
                        counts["skipped_existing"] += 1
                        continue

                with h5py.File(shot_path, "r+") as sf:
                    if args.group in sf:
                        del sf[args.group]

                    grp = sf.create_group(args.group)
                    grp.create_dataset("input", data=ef["input"][i].astype(np.float16))
                    grp.create_dataset("total", data=ef["total"][i].astype(np.float16))
                    for k, v in root_attrs.items():
                        grp.attrs[k] = v
                    for attr in _ROW_ATTRS:
                        grp.attrs[attr] = _row_value(ef[attr], i)
                    grp.attrs["source"] = str(embeds_path)
                    grp.attrs["source_created_utc"] = source_created_utc
                    grp.attrs["injected_utc"] = injected_utc

                counts["injected"] += 1
            except Exception as e:  # noqa: BLE001 - one bad file must not abort the run
                counts["error"] += 1
                print(f"{shot}: {e!r}", file=sys.stderr)
                continue

    if args.dry_run:
        print(f"[dry_run] category counts: {counts} of {n} shots")
        if planned_samples:
            print(f"[dry_run] sample of shots that would be written: {planned_samples}")
    else:
        print(
            f"injected {counts['injected']}, skipped_existing {counts['skipped_existing']}, "
            f"no_shot_file {counts['no_shot_file']}, errors {counts['error']} of {n} shots"
        )

    return counts


def run_verify(args):
    embeds_path = Path(args.embeds).resolve()
    ok = True
    n_checked = 0
    with h5py.File(embeds_path, "r") as ef:
        shots = ef["shots"][:]
        n_total = shots.shape[0]
        n = n_total if args.limit <= 0 else min(args.limit, n_total)

        root_attrs = {k: ef.attrs[k] for k in _ROOT_ATTRS_TO_COPY if k in ef.attrs}

        for i in range(n):
            shot = int(shots[i])
            shot_path = _shot_file_path(args.shot_dir, shot)
            if not shot_path.exists():
                continue
            try:
                with h5py.File(shot_path, "r") as sf:
                    if args.group not in sf:
                        continue
                    grp = sf[args.group]
                    n_checked += 1

                    if not np.array_equal(grp["input"][:], ef["input"][i]):
                        print(f"FAIL: shot {shot} input mismatch")
                        ok = False
                    if not np.array_equal(grp["total"][:], ef["total"][i]):
                        print(f"FAIL: shot {shot} total mismatch")
                        ok = False

                    for k, expected in root_attrs.items():
                        if k not in grp.attrs:
                            print(f"FAIL: shot {shot} missing attr {k}")
                            ok = False
                        elif grp.attrs[k] != expected:
                            print(f"FAIL: shot {shot} attr {k} mismatch: {grp.attrs[k]!r} != {expected!r}")
                            ok = False
                    for attr in _ROW_ATTRS:
                        if attr not in grp.attrs:
                            print(f"FAIL: shot {shot} missing attr {attr}")
                            ok = False
                        elif _row_value(grp.attrs, attr) != _row_value(ef[attr], i):
                            print(f"FAIL: shot {shot} attr {attr} mismatch")
                            ok = False
                    for attr in ("source", "source_created_utc", "injected_utc"):
                        if attr not in grp.attrs:
                            print(f"FAIL: shot {shot} missing attr {attr}")
                            ok = False
            except Exception as e:  # noqa: BLE001
                print(f"FAIL: shot {shot}: {e!r}")
                ok = False

    print(f"VERIFY {'PASSED' if ok else 'FAILED'} ({n_checked} shots checked)")
    return ok


def main(argv=None):
    args = _parse_args(argv)
    if args.verify:
        ok = run_verify(args)
        sys.exit(0 if ok else 1)

    counts = run_inject(args)
    sys.exit(0 if counts["error"] == 0 else 1)


if __name__ == "__main__":
    main()
