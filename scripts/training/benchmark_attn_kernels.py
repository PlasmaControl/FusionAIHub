"""Kernel-level benchmark: flash-attn vs standard attention on MI250X.

Compares four self-attention implementations on synthetic (q, k, v) of
realistic transformer shapes, on one MI250X GCD:

    flash_ext   : flash_attn.flash_attn_func (external pkg, Triton-AMD/aiter)
    sdpa_math   : torch.nn.functional.scaled_dot_product_attention, math
                  backend forced (the "standard" path — what we use today)
    sdpa_flash  : F.scaled_dot_product_attention, flash backend forced
                  (PyTorch native, uses AOTriton on ROCm 7.x — completely
                  different code path from flash_ext)
    sdpa_auto   : F.scaled_dot_product_attention with defaults (PyTorch
                  picks; useful as a "what does torch want" reference)

Measures forward time, backward time, peak alloc. Reports a markdown
table to stdout and a JSON dump.

Why: the e2e profile measured flash_ext as 19% slower / 3.78× memory
than nn.MultiheadAttention at the e2e Stage 1 shape (head_dim=32,
seq_len≈26). Before concluding flash-attn is bad on Frontier, we need
a sanity check at shapes where flash should obviously win.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:
    SDPBackend = None
    sdpa_kernel = None

try:
    from flash_attn import flash_attn_func as _flash_attn_func
except ImportError:
    _flash_attn_func = None


def make_qkv(
    batch: int, seq_len: int, n_heads: int, head_dim: int,
    layout: str, dtype: torch.dtype, device: torch.device,
    requires_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate (q, k, v) in the layout the impl expects.

    layout='bhsd' for SDPA (batch, heads, seq, dim);
    layout='bshd' for flash_attn_func (batch, seq, heads, dim).
    """
    if layout == "bhsd":
        shape = (batch, n_heads, seq_len, head_dim)
    elif layout == "bshd":
        shape = (batch, seq_len, n_heads, head_dim)
    else:
        raise ValueError(layout)
    q = torch.randn(shape, dtype=dtype, device=device, requires_grad=requires_grad)
    k = torch.randn(shape, dtype=dtype, device=device, requires_grad=requires_grad)
    v = torch.randn(shape, dtype=dtype, device=device, requires_grad=requires_grad)
    return q, k, v


def run_flash_ext(q, k, v):
    # flash_attn_func expects (B, S, H, D)
    return _flash_attn_func(q, k, v, causal=False)


def _sdpa_with_backend(backend):
    def _call(q, k, v):
        # SDPA expects (B, H, S, D)
        ctx = sdpa_kernel(backend) if (sdpa_kernel and backend is not None) else nullcontext()
        with ctx:
            return F.scaled_dot_product_attention(q, k, v, is_causal=False)
    return _call


_MHA_CACHE: dict = {}


def _get_nn_mha(d_model: int, n_heads: int, dtype, device) -> torch.nn.MultiheadAttention:
    """Cache an nn.MultiheadAttention so we don't re-init every call.

    Constructed in fp32 then cast — matches typical autocast-style usage.
    """
    key = (d_model, n_heads, dtype)
    mha = _MHA_CACHE.get(key)
    if mha is None:
        mha = torch.nn.MultiheadAttention(
            d_model, n_heads, dropout=0.0, batch_first=True, bias=True,
        ).to(device=device, dtype=dtype)
        _MHA_CACHE[key] = mha
    return mha


def run_nn_mha(q, k, v):
    """Match stage1/2's current backbone: nn.MultiheadAttention(h, h, h).

    Input layout is (B, S, H, D); we collapse heads*dim → embed for MHA, then
    re-split on output. need_weights=False is the path that *could* dispatch
    to SDPA internally — this measurement tells us whether it actually does.
    """
    B, S, H, D = q.shape
    embed = H * D
    qh = q.reshape(B, S, embed)
    # MHA does its own Q/K/V projection; matching the pattern in the backbone
    # which calls self.attn(h, h, h, need_weights=False).
    mha = _get_nn_mha(embed, H, q.dtype, q.device)
    out, _ = mha(qh, qh, qh, need_weights=False)
    return out.reshape(B, S, H, D)


def time_fn_fwd_bwd(
    fn: Callable, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    n_warmup: int, n_iters: int, do_bwd: bool,
) -> dict:
    """Time fn(q, k, v) forward (and optionally backward).

    Returns dict with fwd_ms, bwd_ms (or None), peak_alloc_GB.
    """
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Warmup
    for _ in range(n_warmup):
        out = fn(q, k, v)
        if do_bwd:
            out.sum().backward()
            q.grad = k.grad = v.grad = None
    torch.cuda.synchronize()

    # Forward timing
    fwd_start = torch.cuda.Event(enable_timing=True)
    fwd_end = torch.cuda.Event(enable_timing=True)
    fwd_start.record()
    outs = []
    for _ in range(n_iters):
        out = fn(q, k, v)
        outs.append(out)
    fwd_end.record()
    torch.cuda.synchronize()
    fwd_ms = fwd_start.elapsed_time(fwd_end) / n_iters

    bwd_ms = None
    if do_bwd:
        bwd_start = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)
        bwd_start.record()
        for out in outs:
            out.sum().backward(retain_graph=False)
            q.grad = k.grad = v.grad = None
        bwd_end.record()
        torch.cuda.synchronize()
        bwd_ms = bwd_start.elapsed_time(bwd_end) / n_iters

    peak_alloc_gb = torch.cuda.max_memory_allocated() / 1e9
    return {"fwd_ms": fwd_ms, "bwd_ms": bwd_ms, "peak_alloc_GB": peak_alloc_gb}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=16)
    p.add_argument("--head_dims", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--seq_lens", type=int, nargs="+",
                   default=[32, 128, 512, 2048, 4096])
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--n_warmup", type=int, default=3)
    p.add_argument("--n_iters", type=int, default=10)
    p.add_argument("--no_bwd", action="store_true")
    args = p.parse_args()

    assert torch.cuda.is_available(), "no CUDA/HIP device visible"
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"dtype : {dtype}")
    print(f"shapes: batch={args.batch} n_heads={args.n_heads} "
          f"head_dims={args.head_dims} seq_lens={args.seq_lens}")
    print(f"flash_attn package: {'installed' if _flash_attn_func else 'MISSING'}")
    print(f"sdpa_kernel ctx   : {'available' if sdpa_kernel else 'MISSING (old torch)'}")
    print()

    # Compose impl list. Skip flash_ext if package missing; skip sdpa_flash if
    # the ctx manager is missing (very old torch).
    impls: list[tuple[str, str, Callable]] = []  # (name, layout, fn)
    if _flash_attn_func is not None:
        impls.append(("flash_ext", "bshd", run_flash_ext))
    if sdpa_kernel is not None:
        impls.append(("sdpa_math",  "bhsd", _sdpa_with_backend(SDPBackend.MATH)))
        impls.append(("sdpa_flash", "bhsd", _sdpa_with_backend(SDPBackend.FLASH_ATTENTION)))
    impls.append(("sdpa_auto", "bhsd", _sdpa_with_backend(None)))
    # The one we actually use in production today: nn.MultiheadAttention via
    # backbone.py. Tells us whether it dispatches to SDPA internally on this
    # PyTorch+ROCm build.
    impls.append(("nn_mha", "bshd", run_nn_mha))

    rows: list[dict] = []
    for head_dim in args.head_dims:
        for seq_len in args.seq_lens:
            print(f"-- head_dim={head_dim}  seq_len={seq_len} --")
            for name, layout, fn in impls:
                try:
                    q, k, v = make_qkv(
                        args.batch, seq_len, args.n_heads, head_dim,
                        layout, dtype, device,
                        requires_grad=not args.no_bwd,
                    )
                    res = time_fn_fwd_bwd(
                        fn, q, k, v,
                        n_warmup=args.n_warmup, n_iters=args.n_iters,
                        do_bwd=not args.no_bwd,
                    )
                    rows.append({
                        "impl": name, "head_dim": head_dim, "seq_len": seq_len,
                        "batch": args.batch, "n_heads": args.n_heads,
                        "dtype": args.dtype, **res,
                    })
                    bwd_str = f" bwd={res['bwd_ms']:7.2f}ms" if res["bwd_ms"] else ""
                    print(
                        f"  {name:<10}  fwd={res['fwd_ms']:7.2f}ms"
                        f"{bwd_str}  peak={res['peak_alloc_GB']:5.2f}GB"
                    )
                except Exception as e:
                    print(f"  {name:<10}  FAILED: {type(e).__name__}: {e}")
                    rows.append({
                        "impl": name, "head_dim": head_dim, "seq_len": seq_len,
                        "batch": args.batch, "n_heads": args.n_heads,
                        "dtype": args.dtype, "error": f"{type(e).__name__}: {e}",
                    })
                finally:
                    del q, k, v
                    torch.cuda.empty_cache()
            print()

    # Markdown summary
    md_path = args.out_dir / "summary.md"
    json_path = args.out_dir / "results.json"
    with json_path.open("w") as f:
        json.dump({"args": vars(args) | {"out_dir": str(args.out_dir)}, "rows": rows}, f,
                  indent=2, default=str)

    # Table: for each (head_dim, seq_len), show ratio of each impl vs sdpa_math
    lines: list[str] = []
    lines.append(
        f"# Attention kernel benchmark  ({torch.cuda.get_device_name(0)}, "
        f"{args.dtype}, batch={args.batch}, n_heads={args.n_heads})"
    )
    lines.append("")
    lines.append("Forward + backward time in ms (lower is better). "
                 "Peak alloc in GB. `× math` = ratio of total time to sdpa_math.")
    lines.append("")
    grouped: dict[tuple[int, int], dict[str, dict]] = {}
    for r in rows:
        if "error" in r:
            continue
        key = (r["head_dim"], r["seq_len"])
        grouped.setdefault(key, {})[r["impl"]] = r
    for (head_dim, seq_len), impl_map in sorted(grouped.items()):
        lines.append(f"## head_dim={head_dim}, seq_len={seq_len}")
        lines.append("")
        lines.append("| impl | fwd (ms) | bwd (ms) | total (ms) | × math | peak (GB) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        base = impl_map.get("sdpa_math")
        base_total = (base["fwd_ms"] + (base["bwd_ms"] or 0)) if base else None
        for impl_name in ("sdpa_math", "sdpa_flash", "sdpa_auto", "flash_ext", "nn_mha"):
            if impl_name not in impl_map:
                continue
            r = impl_map[impl_name]
            total = r["fwd_ms"] + (r["bwd_ms"] or 0)
            ratio = f"{total / base_total:5.2f}" if base_total else "  n/a"
            bwd_str = f"{r['bwd_ms']:.2f}" if r["bwd_ms"] else "—"
            lines.append(
                f"| {impl_name} | {r['fwd_ms']:.2f} | {bwd_str} | "
                f"{total:.2f} | {ratio} | {r['peak_alloc_GB']:.2f} |"
            )
        lines.append("")
    md = "\n".join(lines)
    with md_path.open("w") as f:
        f.write(md)
    print()
    print("=" * 60)
    print(md)
    print("=" * 60)
    print(f"\nJSON: {json_path}")
    print(f"MD  : {md_path}")


if __name__ == "__main__":
    main()
