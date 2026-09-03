"""NAN-LOCALIZE — discriminator: is the rollout-step-0 ece NaN ROLLOUT-SPECIFIC
numerics, or MODEL-LATENT (the backbone can't tolerate the hot ece tokens)?

READ-ONLY probe. Does NOT modify rollout.py / the model / any fix; does NOT touch
the production chain. Loads the single-step g3fix model the same way
gate4_kprobe / eval_e2e_animation_tokamak.load_model does.

Measurements (all bf16, matching the trainer's autocast dtype):
  1. RAW-INPUT single-step tolerance vs token-absmax: tokenize raw ece input
     windows spanning absmax ~1500..2410, run FULL single-step forward
     (tokenizer -> backbone -> heads), NO rollout, NO grad-ckpt. finiteness curve.
  2. CODEC-DECODE single-step tolerance: encode_target(window) -> decode(codes)
     -> tokenize -> same single-step forward. finiteness + absmax vs raw.
  3. NaN localization: on a non-finite window, hook every backbone sub-op
     (tokenizer proj, per-block QK^T pre-softmax logits, softmax out, LN outs,
     FFN outs) and report the FIRST non-finite op + the magnitude of its INPUT.
  4. grad-ckpt / backward interaction (only if single-step is finite).

Env: CKPT(argv1 or g3fix beta6 step3000), SHOT(200729), EXTRA_DATA_DIR,
     N_HOT(number of hottest windows to test), OUT_DIR, CACHE_DIR.
"""
import os, sys, json, math
from pathlib import Path
FMH = "/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub"
for p in (f"{FMH}/src", f"{FMH}/scripts/training", f"{FMH}/analysis/mode_audit"):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np
import torch
from torch.utils.data import DataLoader

from eval_e2e_animation_tokamak import load_model
from train_e2e_stage1 import build_datasets, _core
from tokamak_foundation_model.data.data_loader import collate_fn

CKPT = Path(sys.argv[1] if len(sys.argv) > 1
            else "/lustre/orion/fus187/proj-shared/models/e2e_g3fix_anneal/e2e_stage1_beta6.0_step3000.pt")
SHOT = os.environ.get("SHOT", "200729")
EXTRA = os.environ.get("EXTRA_DATA_DIR", "/lustre/orion/fus187/proj-shared/additional_data")
N_HOT = int(os.environ.get("N_HOT", "12"))
BATCH = int(os.environ.get("BATCH", "8"))
MAX_BATCHES = int(os.environ.get("MAX_BATCHES", "40"))    # across all shots
N_EXTRA_SHOTS = int(os.environ.get("N_EXTRA_SHOTS", "6"))
OUT = Path(os.environ.get("OUT_DIR", f"{FMH}/eval_runs/nan_localize")); OUT.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[nl] ckpt={CKPT.name} SHOT={SHOT} device={device}", flush=True)
model, ckpt = load_model(CKPT, device); model.eval()
for p in model.parameters():
    p.requires_grad_(False)
a = ckpt["args"]; core = _core(model)
diag_names = [d["name"] for d in ckpt["diagnostics"]]; act_names = [c["name"] for c in ckpt["actuators"]]
print(f"[nl] d_model={a['d_model']} n_layers={a['n_layers']} n_heads={a['n_heads']} "
      f"diags={diag_names} acts={act_names} spec_freq_stem={a.get('spec_freq_stem', False)} "
      f"backbone_input_skip={a.get('backbone_input_skip', False)} history_windows={a.get('history_windows',1)}",
      flush=True)
assert "ece" in core.diag_tokenizers, "ece tokenizer missing"
ece_tok = core.diag_tokenizers["ece"]
ece_head = core.diag_heads["ece"]
print(f"[nl] ece head type={type(ece_head).__name__} has_codec={hasattr(ece_head,'encode_target')} "
      f"freq_stem_enabled={getattr(ece_tok,'enable_freq_stem',False)}", flush=True)

data_dir = Path(a["data_dir"])
stats = torch.load(a["stats_path"], weights_only=False)
chunk = a["chunk_duration_s"]; horizon = chunk       # single-step: one chunk lookahead
cache = Path(os.environ.get("CACHE_DIR", f"{FMH}/eval_runs/nan_localize_cache")); cache.mkdir(parents=True, exist_ok=True)

def resolve(sh):
    f = data_dir / f"{sh}_processed.h5"
    if f.exists(): return f
    if EXTRA and (Path(EXTRA) / f"{sh}_processed.h5").exists(): return Path(EXTRA) / f"{sh}_processed.h5"
    return None

files = []
f0 = resolve(SHOT)
assert f0 is not None, f"{SHOT} not found"
files.append(f0)
# add extra shots from additional_data SPREAD ACROSS the directory to span the
# corpus absmax range (the question cites a corpus max ~2410; sequential shots
# from one campaign under-sample it). Evenly sample across the sorted list.
if EXTRA and Path(EXTRA).exists():
    allp = [p for p in sorted(Path(EXTRA).glob("*_processed.h5")) if p != f0]
    if allp:
        idxs = np.linspace(0, len(allp) - 1, min(N_EXTRA_SHOTS, len(allp))).astype(int)
        for j in sorted(set(idxs.tolist())):
            files.append(allp[j])
print(f"[nl] files={[f.name for f in files]}", flush=True)

# ── autocast dtype matches the trainer (bf16). ──
AMP = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

def bg_split(x):
    """Mirror eval spectro bg split (residual codec self-declares)."""
    if not getattr(ece_head, "bg_subtract", False):
        return x
    from spectro_bg import baseline_residual_torch
    _, R = baseline_residual_torch(x, float(getattr(ece_head, "bg_sigma", 8.0)))
    return R

# ── Collect windows across shots, compute token-absmax per window (fp32 tokenize,
#    matching how the trainer's step-0 tokenize would produce the tokens). ──
windows = []   # list of (raw_ece_window (1,C,F,T) fp32 on cpu, act_dict placeholder)
absmax_raw = []
for f in files:
    _, va = build_datasets(data_dir, [f], [f], stats, chunk, horizon, a["step_size_s"], a["warmup_s"],
                           diag_names, act_names, cache)
    loader = DataLoader(va, batch_size=BATCH, shuffle=False, num_workers=2, collate_fn=collate_fn, drop_last=False)
    nb = 0
    for batch in loader:
        if nb >= MAX_BATCHES: break
        nb += 1
        raw = batch["inputs"]["ece"].to(device).float()          # (B,C,F,T)
        trunc = ece_tok.trunc_t
        raw = raw[..., :trunc]
        raw = bg_split(raw)
        # per-window token absmax (fp32 tokenizer path)
        with torch.no_grad():
            tok = ece_tok(raw)                                   # (B,n_tok,d)
        am = tok.abs().amax(dim=(1, 2)).detach().cpu().numpy()   # (B,)
        for b in range(raw.shape[0]):
            windows.append(raw[b:b+1].detach().cpu())
            absmax_raw.append(float(am[b]))
print(f"[nl] collected {len(windows)} ece windows; token-absmax(raw) "
      f"min={min(absmax_raw):.1f} max={max(absmax_raw):.1f} "
      f"p50={np.percentile(absmax_raw,50):.1f} p90={np.percentile(absmax_raw,90):.1f}", flush=True)

# sort by absmax, keep the hottest N_HOT plus a spread down to ~1500
order = np.argsort(absmax_raw)[::-1]
hot_idx = list(order[:N_HOT])
# also add a few mid/low windows to draw the finiteness-vs-absmax curve
spread = [int(order[int(x)]) for x in np.linspace(0, len(order) - 1, 8)]
test_idx = sorted(set(hot_idx + spread), key=lambda i: -absmax_raw[i])
print(f"[nl] testing {len(test_idx)} windows; absmax range "
      f"{absmax_raw[test_idx[-1]]:.1f}..{absmax_raw[test_idx[0]]:.1f}", flush=True)

# ── build a dummy actuator input (zeros) matching act tokenizer geometry —
#    single-step forward needs act tokens. We reuse a real batch's act to be safe. ──
# Grab one actuator dict from the first file's loader.
_, va0 = build_datasets(data_dir, [files[0]], [files[0]], stats, chunk, horizon,
                        a["step_size_s"], a["warmup_s"], diag_names, act_names, cache)
_l0 = DataLoader(va0, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
_b0 = next(iter(_l0))
from eval_e2e import split_target_by_step, _clean_and_mask as _cm
act_template = {}
for name in act_names:
    raw = _b0["targets"][name].to(device).float()
    # single-step: take the first chunk-window slice via the split helper
    slc = split_target_by_step(raw, name, 1, chunk)[0]
    cleaned, _ = _cm(slc, None)
    act_template[name] = cleaned    # (1, C, ...) shape for batch=1

# Full diagnostic template: the model has 9 diagnostics; core.tokenize iterates
# ALL of them, so a single-step forward needs a real input for each. We hold the
# non-ece diagnostics FIXED (a real batch's input window) and swap in only the
# ece window under test — isolating the ece pathway's numerics.
diag_template = {}
for cfg in core.diagnostics:
    name = cfg.name
    raw = _b0["inputs"][name].to(device).float()
    cleaned, _ = _cm(raw, None)
    if cfg.kind == "spectrogram":
        cleaned = cleaned[..., :core.diag_tokenizers[name].trunc_t]
        cleaned = bg_split(cleaned) if name == "ece" else cleaned
    diag_template[name] = cleaned
    vk = f"{name}_valid"
    if vk in _b0["inputs"]:
        diag_template[vk] = _b0["inputs"][vk].to(device)

# Per-modality tokenizer-output absmax (identify what dominates the global token
# scale — the 3.45M value seen in v1 was NOT ece). Diagnostic only.
with torch.no_grad(), AMP:
    for cfg in core.diagnostics:
        _t = core.diag_tokenizers[cfg.name](diag_template[cfg.name])
        print(f"[nl] template-tok-absmax diag {cfg.name:22s} = {float(_t.float().abs().max()):12.1f}", flush=True)
    for name in act_names:
        _t = core.act_tokenizers[name](act_template[name])
        print(f"[nl] template-tok-absmax act  {name:22s} = {float(_t.float().abs().max()):12.1f}", flush=True)


def _ece_layout():
    for layout in core.token_layout:
        if getattr(layout, "name", None) == "ece":
            return layout
    return None

_ECE_LAYOUT = _ece_layout()

def single_step_forward(raw_ece_1, use_grad=False, isolate_ece=False):
    """FULL single-step forward on ONE ece window (1,C,F,T). Reports the
    ECE-SLICE input-token absmax (the quantity the question means by "ece token
    absmax ~2216"), the finiteness of the backbone OUTPUT (global + ece slice),
    and the ece head output. Mirrors model.forward's single-window path.

    ``isolate_ece``: zero the NON-ece diagnostic inputs so their tokenizers emit
    only the small learned bias/PE and ece DRIVES the backbone token scale — the
    stress test that removes the confound of a foreign large token forcing the
    first LayerNorm to rescale everything. (Actuators kept from template.)"""
    diag_inputs = {k: v for k, v in diag_template.items()}
    if isolate_ece:
        for cfg in core.diagnostics:
            if cfg.name != "ece" and cfg.name in diag_inputs:
                diag_inputs[cfg.name] = torch.zeros_like(diag_inputs[cfg.name])
    diag_inputs["ece"] = raw_ece_1.to(device)
    diag_inputs["ece_valid"] = torch.ones(1, dtype=torch.long, device=device)
    act_inputs = {k: v for k, v in act_template.items()}
    step_idx = torch.zeros(1, dtype=torch.long, device=device)
    time_off = torch.zeros(1, device=device)
    ctx = torch.enable_grad() if use_grad else torch.no_grad()
    with ctx, AMP:
        tokens = core.tokenize(diag_inputs, act_inputs,
                               actuators_as_film=getattr(core, "use_actuator_film", False))
        tok_out_absmax = float(tokens.detach().float().abs().max())    # global (all modalities)
        ece_tok_absmax = (float(tokens[:, _ECE_LAYOUT.slice_].detach().float().abs().max())
                          if _ECE_LAYOUT is not None else float("nan"))
        tok_finite = bool(torch.isfinite(tokens).all())
        _film = (core._actuator_film_params(act_inputs)
                 if getattr(core, "use_actuator_film", False) else None)
        out_tokens = core.backbone(tokens, step_idx, time_off, film_params=_film)
        if getattr(core, "backbone_input_skip", False):
            out_tokens = tokens + core.backbone_skip_gate * out_tokens
        bb_finite = bool(torch.isfinite(out_tokens).all())
        bb_absmax = float(out_tokens.detach().float().abs().max())
        # ece backbone-output slice + head
        ece_slice = out_tokens[:, _ECE_LAYOUT.slice_] if _ECE_LAYOUT is not None else None
        ece_bb_finite = bool(torch.isfinite(ece_slice).all()) if ece_slice is not None else None
        ece_out = ece_head(ece_slice) if ece_slice is not None else None
        ece_finite = bool(torch.isfinite(ece_out).all()) if ece_out is not None else None
    return dict(tok_out_absmax=tok_out_absmax, ece_tok_absmax=ece_tok_absmax,
                tok_finite=tok_finite, bb_finite=bb_finite, bb_absmax=bb_absmax,
                ece_bb_finite=ece_bb_finite, ece_finite=ece_finite,
                out_tokens=out_tokens if use_grad else None,
                tokens=tokens if use_grad else None)


def codec_decode_window(raw_ece_1):
    """encode_target -> decode: the on-manifold feedback content."""
    with torch.no_grad(), AMP:
        # encode_target / decode are the frozen codec (its own precision handling).
        codes = ece_head.encode_target(raw_ece_1.to(device).float())
        decoded = ece_head.decode(codes)
    return decoded.detach()


# ─────────────────────────────── MEASUREMENT 1 + 2 ───────────────────────────────
# ISOLATE_ECE (default 1): zero non-ece diagnostics so the ece token drives the
# backbone scale (removes the confound that a foreign large token forces the first
# LayerNorm to rescale everything, trivially finitizing the output). We report BOTH
# the FULL forward (all modalities present, matching the real rollout) and the
# ece-isolated forward.
ISO = os.environ.get("ISOLATE_ECE", "1") == "1"
print(f"\n[nl] === MEASUREMENT 1+2: single-step tolerance vs ECE-slice token-absmax "
      f"(isolate_ece={ISO}) ===", flush=True)
rows = []
first_nonfinite_raw = None
first_nonfinite_codec = None
for i in test_idx:
    w = windows[i]
    r_raw = single_step_forward(w)                          # full forward, raw input
    r_raw_iso = single_step_forward(w, isolate_ece=True) if ISO else r_raw
    dec = codec_decode_window(w)
    r_cod = single_step_forward(dec)                        # full forward, codec-decode
    r_cod_iso = single_step_forward(dec, isolate_ece=True) if ISO else r_cod
    rows.append(dict(
        absmax_raw_precomputed=absmax_raw[i],
        # ece-slice input-token absmax (the real discriminating magnitude)
        raw_ece_tok_absmax=r_raw["ece_tok_absmax"],
        codec_ece_tok_absmax=r_cod["ece_tok_absmax"],
        raw_global_tok_absmax=r_raw["tok_out_absmax"],
        codec_global_tok_absmax=r_cod["tok_out_absmax"],
        # FULL forward finiteness (all modalities present, = real rollout)
        raw_bb_finite=r_raw["bb_finite"], raw_ece_bb_finite=r_raw["ece_bb_finite"],
        raw_ece_finite=r_raw["ece_finite"], raw_bb_absmax=r_raw["bb_absmax"],
        codec_bb_finite=r_cod["bb_finite"], codec_ece_bb_finite=r_cod["ece_bb_finite"],
        codec_ece_finite=r_cod["ece_finite"], codec_bb_absmax=r_cod["bb_absmax"],
        # ISOLATED forward finiteness (ece drives the scale)
        raw_iso_bb_finite=r_raw_iso["bb_finite"], raw_iso_ece_bb_finite=r_raw_iso["ece_bb_finite"],
        raw_iso_bb_absmax=r_raw_iso["bb_absmax"], raw_iso_ece_tok_absmax=r_raw_iso["ece_tok_absmax"],
        codec_iso_bb_finite=r_cod_iso["bb_finite"], codec_iso_ece_bb_finite=r_cod_iso["ece_bb_finite"],
        codec_iso_bb_absmax=r_cod_iso["bb_absmax"], codec_iso_ece_tok_absmax=r_cod_iso["ece_tok_absmax"],
    ))
    print(f"  ece_tok_absmax RAW={r_raw['ece_tok_absmax']:8.1f} CODEC={r_cod['ece_tok_absmax']:8.1f} | "
          f"FULL bb_finite raw={r_raw['bb_finite']}/cod={r_cod['bb_finite']} "
          f"ece_bb raw={r_raw['ece_bb_finite']}/cod={r_cod['ece_bb_finite']} | "
          f"ISO bb_finite raw={r_raw_iso['bb_finite']}/cod={r_cod_iso['bb_finite']} "
          f"(iso ece_tok raw={r_raw_iso['ece_tok_absmax']:.1f} cod={r_cod_iso['ece_tok_absmax']:.1f} "
          f"iso bb_absmax raw={r_raw_iso['bb_absmax']:.1f} cod={r_cod_iso['bb_absmax']:.1f})", flush=True)
    # A NaN counts if EITHER the full or isolated forward goes non-finite.
    raw_nan = (not r_raw["bb_finite"]) or (not r_raw_iso["bb_finite"])
    cod_nan = (not r_cod["bb_finite"]) or (not r_cod_iso["bb_finite"])
    if first_nonfinite_raw is None and raw_nan:
        first_nonfinite_raw = (i, r_raw["bb_finite"])          # (idx, full_finite?) → localize picks iso if full ok
    if first_nonfinite_codec is None and cod_nan:
        first_nonfinite_codec = (i, r_cod["bb_finite"])

raw_max_ece_tok = max(r["raw_ece_tok_absmax"] for r in rows)
codec_max_ece_tok = max(r["codec_ece_tok_absmax"] for r in rows)
raw_all_finite = all(r["raw_bb_finite"] and r["raw_iso_bb_finite"] for r in rows)
codec_all_finite = all(r["codec_bb_finite"] and r["codec_iso_bb_finite"] for r in rows)
print(f"\n[nl] RAW single-step: all-finite(full&iso)={raw_all_finite} "
      f"(max ece-tok-absmax tested={raw_max_ece_tok:.1f})", flush=True)
print(f"[nl] CODEC single-step: all-finite(full&iso)={codec_all_finite} "
      f"(max codec ece-tok-absmax={codec_max_ece_tok:.1f})", flush=True)


# ─────────────────────────────── MEASUREMENT 3: NaN localization ───────────────────────────────
def localize(raw_ece_1, label, isolate=False):
    """Instrument the backbone forward to find the FIRST non-finite op + its input
    magnitude. Manually reimplements the SharedBackbone block math so we can inspect
    QK^T logits pre-softmax, softmax out, LN outs, FFN outs. Uses the model's own
    weights. All in bf16 autocast to match training numerics."""
    diag_inputs = {k: v for k, v in diag_template.items()}
    if isolate:
        for cfg in core.diagnostics:
            if cfg.name != "ece" and cfg.name in diag_inputs:
                diag_inputs[cfg.name] = torch.zeros_like(diag_inputs[cfg.name])
    diag_inputs["ece"] = raw_ece_1.to(device)
    diag_inputs["ece_valid"] = torch.ones(1, dtype=torch.long, device=device)
    act_inputs = {k: v for k, v in act_template.items()}
    step_idx = torch.zeros(1, dtype=torch.long, device=device); time_off = torch.zeros(1, device=device)
    bb = core.backbone
    events = []
    def chk(name, t, inp_absmax):
        fin = bool(torch.isfinite(t).all())
        am = float(t.detach().float().abs().max()) if fin else float("inf")
        if not fin:
            events.append((name, inp_absmax, am))
        return fin
    with torch.no_grad(), AMP:
        tokens = core.tokenize(diag_inputs, act_inputs,
                               actuators_as_film=getattr(core, "use_actuator_film", False))
        # tokenizer proj output specifically (ece) — reproduce _encode up to proj
        xin = raw_ece_1.to(device)[..., :ece_tok.trunc_t]
        if getattr(ece_tok, "enable_freq_stem", False):
            import torch.nn.functional as F
            h = xin.transpose(2, 3); h = ece_tok.fs_lin2(F.gelu(ece_tok.fs_lin1(h))); xin = xin + h.transpose(2, 3)
        proj = ece_tok.proj(xin)
        proj_finite = chk("ece_tokenizer.proj", proj, float(xin.float().abs().max()))
        tok_finite = chk("tokenize_full", tokens, float(xin.float().abs().max()))
        # backbone manual forward
        step_embed = bb.step_cond(step_idx, time_off).unsqueeze(1)
        x = tokens + step_embed
        chk("post_step_embed", x, float(tokens.float().abs().max()))
        n_heads = a["n_heads"]; d = a["d_model"]; hd = d // n_heads
        for li, block in enumerate(bb.blocks):
            xin_am = float(x.detach().float().abs().max())
            h = block.norm1(x)
            if not chk(f"block{li}.norm1", h, xin_am):
                break
            # replicate MultiheadAttention QK^T logits pre-softmax
            mha = block.attn
            # in_proj: [q;k;v]
            w = mha.in_proj_weight; b = mha.in_proj_bias
            qkv = torch.nn.functional.linear(h, w, b)             # (1,N,3d)
            q, k, v = qkv.chunk(3, dim=-1)
            B_, N_, _ = q.shape
            qh = q.reshape(B_, N_, n_heads, hd).transpose(1, 2)   # (1,H,N,hd)
            kh = k.reshape(B_, N_, n_heads, hd).transpose(1, 2)
            logits = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(hd)
            if not chk(f"block{li}.attn.QK_logits(pre-softmax)", logits, float(h.float().abs().max())):
                break
            attn = torch.softmax(logits, dim=-1)
            if not chk(f"block{li}.attn.softmax", attn, float(logits.float().abs().max())):
                break
            # use real MHA for the residual add (matches model exactly)
            attn_out, _ = mha(h, h, h, need_weights=False)
            if not chk(f"block{li}.attn.out", attn_out, float(h.float().abs().max())):
                break
            x = x + attn_out
            if not chk(f"block{li}.attn.residual", x, xin_am):
                break
            h2 = block.norm2(x)
            if not chk(f"block{li}.norm2", h2, float(x.float().abs().max())):
                break
            ffn = block.mlp(h2)
            if not chk(f"block{li}.ffn", ffn, float(h2.float().abs().max())):
                break
            x = x + ffn
            if not chk(f"block{li}.ffn.residual", x, xin_am):
                break
        else:
            fn = bb.final_norm(x)
            chk("final_norm", fn, float(x.float().abs().max()))
    print(f"\n[nl] LOCALIZE ({label}): proj_finite={proj_finite} tok_finite={tok_finite}", flush=True)
    if events:
        name0, inp_am0, out_am0 = events[0]
        print(f"[nl] LOCALIZE ({label}): FIRST non-finite op = {name0} "
              f"| triggering INPUT absmax = {inp_am0:.1f} | this op's out = {out_am0}", flush=True)
        for e in events[:6]:
            print(f"       nonfinite: {e[0]:40s} input_absmax={e[1]:.1f}", flush=True)
    else:
        print(f"[nl] LOCALIZE ({label}): NO non-finite op found (backbone forward is finite).", flush=True)
    return events

loc_raw = None; loc_codec = None
if first_nonfinite_raw is not None:
    idx, full_finite = first_nonfinite_raw
    # if the FULL forward was finite, the NaN is in the isolated forward → localize isolated
    loc_raw = localize(windows[idx], "RAW-hot-window", isolate=bool(full_finite))
if first_nonfinite_codec is not None:
    idx, full_finite = first_nonfinite_codec
    dec = codec_decode_window(windows[idx])
    loc_codec = localize(dec, "CODEC-DECODE-hot-window", isolate=bool(full_finite))
if first_nonfinite_raw is None and first_nonfinite_codec is None:
    print("\n[nl] No single-step NaN on any tested window (raw OR codec, full OR isolated).", flush=True)


# ─────────────────────────────── MEASUREMENT 4: grad-ckpt / backward ───────────────────────────────
gc_result = None
if raw_all_finite and codec_all_finite:
    print("\n[nl] === MEASUREMENT 4: grad-ckpt + backward on hottest CODEC-DECODE window ===", flush=True)
    import torch.utils.checkpoint as torch_ckpt
    # hottest by codec ece-slice tok-absmax
    hottest = max(range(len(test_idx)), key=lambda j: rows[j]["codec_ece_tok_absmax"])
    dec = codec_decode_window(windows[test_idx[hottest]]).float()
    diag_inputs = {k: v for k, v in diag_template.items()}
    diag_inputs["ece"] = dec.to(device).requires_grad_(True)
    diag_inputs["ece_valid"] = torch.ones(1, dtype=torch.long, device=device)
    act_inputs = {k: v for k, v in act_template.items()}
    step_idx = torch.zeros(1, dtype=torch.long, device=device); time_off = torch.zeros(1, device=device)
    bb = core.backbone
    # temporarily require grad on backbone params so a real backward path exists
    saved = [(p, p.requires_grad) for p in bb.parameters()]
    for p in bb.parameters():
        p.requires_grad_(True)
    gc_every = int(os.environ.get("GC_EVERY", "10"))
    try:
        with AMP:
            tokens = core.tokenize(diag_inputs, act_inputs,
                                   actuators_as_film=getattr(core, "use_actuator_film", False))
            step_embed = bb.step_cond(step_idx, time_off).unsqueeze(1)
            x = tokens + step_embed
            # grad-ckpt groups mirroring rollout_grad_checkpoint_every semantics:
            # checkpoint each block (recompute in backward).
            for li, block in enumerate(bb.blocks):
                x = torch_ckpt.checkpoint(block, x, None, None, use_reentrant=False)
            out = bb.final_norm(x)
        fwd_finite = bool(torch.isfinite(out).all())
        loss = out.float().pow(2).mean()
        loss.backward()
        # check grads finite
        gfin = all(bool(torch.isfinite(p.grad).all()) for p in bb.parameters() if p.grad is not None)
        gmax = max((float(p.grad.float().abs().max()) for p in bb.parameters() if p.grad is not None), default=0.0)
        gc_result = dict(fwd_finite=fwd_finite, grads_finite=gfin, grad_absmax=gmax,
                         codec_ece_tok_absmax=rows[hottest]["codec_ece_tok_absmax"])
        print(f"[nl] grad-ckpt: fwd_finite={fwd_finite} grads_finite={gfin} grad_absmax={gmax:.3g} "
              f"(codec ece-tok-absmax={rows[hottest]['codec_ece_tok_absmax']:.1f})", flush=True)
    finally:
        for p in bb.parameters():
            p.grad = None
        for p, rg in saved:
            p.requires_grad_(rg)


# ─────────────────────────────── VERDICT ───────────────────────────────
single_step_nan = (not raw_all_finite) or (not codec_all_finite)
verdict = {}
if not raw_all_finite:
    verdict["class"] = "MODEL-LATENT (raw-input single-step NaN)"
elif not codec_all_finite:
    verdict["class"] = "MODEL-LATENT-ish (codec-decode single-step NaN; raw-input finite)"
else:
    verdict["class"] = "ROLLOUT-SPECIFIC (single-step finite on all hot windows)"

summary = dict(
    ckpt=str(CKPT), shot=SHOT, n_windows_collected=len(windows),
    absmax_raw_min=float(min(absmax_raw)), absmax_raw_max=float(max(absmax_raw)),
    raw_single_step_all_finite=raw_all_finite,
    codec_single_step_all_finite=codec_all_finite,
    max_raw_ece_tok_absmax_tested=raw_max_ece_tok,
    max_codec_ece_tok_absmax=codec_max_ece_tok,
    first_nonfinite_op_raw=(loc_raw[0][0] if loc_raw else None),
    first_nonfinite_input_absmax_raw=(loc_raw[0][1] if loc_raw else None),
    first_nonfinite_op_codec=(loc_codec[0][0] if loc_codec else None),
    first_nonfinite_input_absmax_codec=(loc_codec[0][1] if loc_codec else None),
    grad_ckpt=gc_result,
    verdict=verdict["class"],
    rows=rows,
)
json.dump(summary, open(OUT / "nan_localize.json", "w"), indent=2)
print(f"\n[nl] ================= VERDICT: {verdict['class']} =================", flush=True)
print(f"[nl] wrote {OUT}/nan_localize.json", flush=True)
