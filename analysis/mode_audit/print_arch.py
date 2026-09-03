"""INITIALIZE the next-production-run model and PRINT its architecture (from the object,
not from memory). d_model=1024, n_layers=48, FINER 8x16 FSQ spectro codec + video/fast-TS/
slow-TS FSQ codecs. Built via the trainer's own build_configs + E2EFoundationModel ctor.

Env overrides: D_MODEL(1024) N_LAYERS(48) N_HEADS(8) SPEC_CODEC(finer dir) PATCH_F(8) PATCH_T(16).
CPU init (~1.2B params fp32 ~5GB RAM). Prints config, token layout, param table, module types.
"""
import json
import os
import sys

FMH = "/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub"
for p in (f"{FMH}/src", f"{FMH}/scripts/training"):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch
from collections import defaultdict
from train_e2e_stage1 import build_configs
from tokamak_foundation_model.e2e.model import E2EFoundationModel

M = "/lustre/orion/fus187/proj-shared/models"
D_MODEL = int(os.environ.get("D_MODEL", "1024"))
N_LAYERS = int(os.environ.get("N_LAYERS", "48"))
N_HEADS = int(os.environ.get("N_HEADS", "8"))
PATCH_F = int(os.environ.get("PATCH_F", "8")); PATCH_T = int(os.environ.get("PATCH_T", "16"))
SPEC_CODEC = os.environ.get("SPEC_CODEC", f"{M}/fsq_resid_p8_all")

diagnostics, actuators = build_configs(
    0.05, use_video=["tangtv_lower", "tangtv_upper"],
    use_spectro=["ece", "co2", "bes", "mhr"],
    spectro_patch_f=PATCH_F, spectro_patch_t=PATCH_T)

model = E2EFoundationModel(
    diagnostics=diagnostics, actuators=actuators,
    d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, dropout=0.1,
    spectro_fsq=True, spectro_fsq_codec_dir=SPEC_CODEC,
    video_fsq=True, video_fsq_codec_dir=f"{M}/fsq_video_codecs_2ch",
    fastts_fsq=True, fastts_fsq_codec_dir=f"{M}/fsq_fastts_codec_tok80",
    slow_ts_fsq=True, slow_ts_fsq_codec_dir=f"{M}/fsq_slowts_codecs",
)
model.eval()

print("=" * 78)
print("NEXT-PRODUCTION MODEL — architecture (INITIALIZED, not from memory)")
print("=" * 78)
print(f"backbone: d_model={D_MODEL} n_layers={N_LAYERS} n_heads={N_HEADS} mlp_ratio=4.0 "
      f"(class={type(model.backbone).__name__})")
print(f"spectro codec: {SPEC_CODEC}  patch=({PATCH_F},{PATCH_T})  spectro_fsq=True")

# --- token layout (authoritative per-modality token counts) ---
print("\n--- token layout (backbone sequence) ---")
seq = 0
for L in model.token_layout:
    n = L.slice_.stop - L.slice_.start
    seq += n
    kind = "diag" if getattr(L, "is_diagnostic", True) else "act"
    print(f"    {L.name:20s} {n:5d} tokens  [{kind}]")
print(f"    {'TOTAL sequence':20s} {seq:5d} tokens")

# --- params ---
tot = sum(p.numel() for p in model.parameters())
top = defaultdict(int); tmod = defaultdict(int); hmod = defaultdict(int)
for k, v in model.state_dict().items():
    n = v.numel(); parts = k.split(".")
    top[parts[0]] += n
    if parts[0] == "diag_tokenizers" and len(parts) > 1:
        tmod[parts[1]] += n
    if parts[0] == "diag_heads" and len(parts) > 1:
        hmod[parts[1]] += n
print(f"\n--- parameters: TOTAL {tot/1e6:.1f} M ---")
for k, v in sorted(top.items(), key=lambda x: -x[1]):
    print(f"    {k:22s} {v/1e6:9.2f} M")
print("  diag_tokenizers by modality (M): " +
      ", ".join(f"{k} {v/1e6:.1f}" for k, v in sorted(tmod.items(), key=lambda x: -x[1])))
print("  diag_heads by modality (M):      " +
      ", ".join(f"{k} {v/1e6:.1f}" for k, v in sorted(hmod.items(), key=lambda x: -x[1])))

# --- module types per modality ---
print("\n--- module types ---")
for name in [c.name for c in diagnostics]:
    tk = type(model.diag_tokenizers[name]).__name__
    hd = type(model.diag_heads[name]).__name__
    print(f"    {name:20s} tok={tk:28s} head={hd}")

# --- one backbone block (the repeated unit) ---
print("\n--- one backbone block (repeated x%d) ---" % N_LAYERS)
try:
    blk = model.backbone.blocks[0] if hasattr(model.backbone, "blocks") else list(model.backbone.children())[0]
    print(blk)
except Exception as e:
    print(f"(could not introspect block: {e})")
print("=" * 78)

# --- JSON artifact ---
arch = {
    "note": "INITIALIZED model architecture (built via build_configs + E2EFoundationModel), not from memory",
    "backbone": {"class": type(model.backbone).__name__, "d_model": D_MODEL,
                 "n_layers": N_LAYERS, "n_heads": N_HEADS, "mlp_ratio": 4.0, "dropout": 0.1,
                 "params_M": round(top["backbone"] / 1e6, 2)},
    "spectro_codec_dir": SPEC_CODEC, "spectro_patch": [PATCH_F, PATCH_T], "spectro_fsq": True,
    "total_params_M": round(tot / 1e6, 2),
    "seq_len_tokens": seq,
    "params_by_component_M": {k: round(v / 1e6, 2) for k, v in sorted(top.items(), key=lambda x: -x[1])},
    "params_diag_tokenizers_M": {k: round(v / 1e6, 2) for k, v in sorted(tmod.items(), key=lambda x: -x[1])},
    "params_diag_heads_M": {k: round(v / 1e6, 2) for k, v in sorted(hmod.items(), key=lambda x: -x[1])},
    "token_layout": [{"name": L.name, "tokens": L.slice_.stop - L.slice_.start,
                      "is_diagnostic": bool(getattr(L, "is_diagnostic", True))}
                     for L in model.token_layout],
    "module_types": {c.name: {"tokenizer": type(model.diag_tokenizers[c.name]).__name__,
                              "head": type(model.diag_heads[c.name]).__name__}
                     for c in diagnostics},
}
out_json = os.environ.get("OUT_JSON", f"{FMH}/analysis/mode_audit/next_production_arch.json")
json.dump(arch, open(out_json, "w"), indent=2)
print(f"[print_arch] initialized OK — total {tot/1e6:.1f} M params, seq {seq} tokens", flush=True)
print(f"[print_arch] wrote {out_json}", flush=True)
