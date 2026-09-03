"""GROUND TRUTH from the checkpoint — run this FIRST in any audit/debug session.

Every downstream number gets interpreted against these facts, so they must come from the
ARTIFACT, not from memory. Prints (and JSON-dumps) a report header:
  - d_model / n_layers / n_heads (+ chunk/step/horizon/history_windows if present)
  - parameter counts per top-level component AND per-modality tokenizer/head
  - token count per modality (from tokenizer positional-embedding shapes in the state dict)
  - loss weights / config knobs (any arg matching weight|lambda|class_weight|gamma|lr|freeze)
  - per-modality codec cfg (patch, fsq_dim/L, bg_subtract, smooth_frames) from the codec .pt

Env/arg: CKPT (path). Optional OUT_JSON. Usage: CKPT=... python checkpoint_facts.py
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

FMH = "/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub"
for p in (f"{FMH}/src", f"{FMH}/scripts/training"):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch

CKPT = sys.argv[1] if len(sys.argv) > 1 else os.environ["CKPT"]
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
a = ck.get("args", {})
sd = ck.get("model_state_dict", ck.get("model", ck))

facts = {"checkpoint": CKPT, "step": ck.get("step"), "val_loss": ck.get("val_loss"),
         "best_val_loss": ck.get("best_val_loss")}

# --- architecture ---
arch_keys = ["d_model", "n_layers", "n_heads", "dropout", "chunk_duration_s",
             "step_size_s", "prediction_horizon_s", "warmup_s", "history_windows",
             "use_spectro", "use_video", "batch_size", "lr"]
facts["arch"] = {k: a.get(k) for k in arch_keys if k in a}

# --- loss weights / config knobs (artifact, not memory) ---
knob_re = ("weight", "lambda", "class_weight", "gamma", "freeze", "anchor",
           "generative", "codec", "smooth", "bg_", "focal", "band", "resize", "warp")
facts["config_knobs"] = {k: v for k, v in sorted(a.items())
                         if any(t in k.lower() for t in knob_re) and not isinstance(v, (dict, list))}

# --- parameter counts ---
tot = 0
top = defaultdict(int)
mod_tok = defaultdict(int)
mod_head = defaultdict(int)
for k, v in sd.items():
    n = v.numel(); tot += n
    parts = k.split(".")
    top[parts[0]] += n
    if parts[0] == "diag_tokenizers" and len(parts) > 1:
        mod_tok[parts[1]] += n
    if parts[0] == "diag_heads" and len(parts) > 1:
        mod_head[parts[1]] += n
facts["params_total_M"] = round(tot / 1e6, 2)
facts["params_by_component_M"] = {k: round(v / 1e6, 2) for k, v in sorted(top.items(), key=lambda x: -x[1])}
facts["params_diag_tokenizers_M"] = {k: round(v / 1e6, 2) for k, v in sorted(mod_tok.items(), key=lambda x: -x[1])}
facts["params_diag_heads_M"] = {k: round(v / 1e6, 2) for k, v in sorted(mod_head.items(), key=lambda x: -x[1])}

# --- token count per modality (from tokenizer positional-embedding shapes) ---
tok = {}
for k, v in sd.items():
    if k.startswith("diag_tokenizers.") and (k.endswith(".spatial_pe") or k.endswith(".pos_embed") or k.endswith(".temporal_pe")):
        mod = k.split(".")[1]
        tok[mod] = tok.get(mod, 0) + v.shape[0]
facts["tokens_per_modality"] = tok
# actuators token count (context)
facts["actuators"] = [c.get("name") for c in ck.get("actuators", []) if isinstance(c, dict)]

# --- per-modality codec cfg (from the codec .pt referenced by args) ---
codec_dir = a.get("spec_fsq_codec_dir")
facts["spec_fsq_codec_dir"] = codec_dir
facts["codec_cfg"] = {}
if codec_dir and Path(codec_dir).exists():
    for f in sorted(Path(codec_dir).glob("spectro_codec_*.pt")):
        mod = f.stem.replace("spectro_codec_", "")
        try:
            c = torch.load(f, map_location="cpu", weights_only=False)["cfg"]
            facts["codec_cfg"][mod] = {kk: c.get(kk) for kk in
                                       ("patch_f", "patch_t", "fsq_dim", "fsq_L", "C", "Fq", "Tq",
                                        "bg_subtract", "smooth_frames", "d_model")}
        except Exception as e:
            facts["codec_cfg"][mod] = f"load-failed: {e}"

# --- print header ---
print("=" * 72)
print("GROUND TRUTH (from checkpoint — NOT memory)")
print("=" * 72)
print(f"ckpt: {CKPT}")
print(f"step: {facts['step']}  val_loss: {facts['val_loss']}")
print(f"arch: {facts['arch']}")
print(f"TOTAL params: {facts['params_total_M']} M")
print("params by component (M):")
for k, v in facts["params_by_component_M"].items():
    print(f"    {k:22s} {v:9.2f}")
print(f"diag_tokenizers by modality (M): {facts['params_diag_tokenizers_M']}")
print(f"diag_heads by modality (M):      {facts['params_diag_heads_M']}")
print(f"tokens/modality: {facts['tokens_per_modality']}")
print(f"actuators: {facts['actuators']}")
print(f"loss/config knobs: {facts['config_knobs']}")
print(f"codec_dir: {codec_dir}")
for m, c in facts["codec_cfg"].items():
    print(f"    codec[{m}]: {c}")
print("=" * 72)

out_json = os.environ.get("OUT_JSON", f"{FMH}/analysis/mode_audit/ground_truth.json")
json.dump(facts, open(out_json, "w"), indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
print(f"[facts] wrote {out_json}", flush=True)
