"""Production (d1024/48L, FULL-modality) parameter counter.

Extends eval_runs/paper_facts/build_and_count.py to the PAPER PRODUCTION config:
  - backbone d_model=1024, n_layers=48, n_heads=16 (head_dim 64)
  - 7 slow-TS continuous + 1 fast-TS continuous
  - 4 spectrograms FSQ-coded (ece, co2, bes, mhr; frozen codecs)
  - 2 video FSQ-coded (tangtv_lower + tangtv_upper; two separate frozen codecs)
  - 7 actuators (unchanged)

FSQ scope = spectrograms + video (frozen codecs, predicted via code heads);
slow-TS + fast-TS are continuous regression heads (no codec).

Read-only. Builds on CPU. No training, no checkpoint writes, no cache touch.
All spectro + video codecs EXIST on disk -> counted EXACTLY (no projection needed).
"""
import sys, os, torch, importlib.util
from collections import defaultdict

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/training")
from tokamak_foundation_model.e2e.model import E2EFoundationModel

spec = importlib.util.spec_from_file_location("trn", "scripts/training/train_e2e_stage1.py")
trn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trn)

# --- Codec dirs (all EXIST; verified on disk) -----------------------------
# Spectro: the residual patch(8,16) family used by the live d512 chain
# (fsq_resid_p8_all). All 4 modalities present, self-consistent n_tok=384.
SPEC_CODEC_DIR = "/lustre/orion/fus187/proj-shared/models/fsq_resid_p8_all"
# Video: the split upper/lower-divertor codecs (2ch each, n_tok=300).
VIDEO_CODEC_DIR = "/lustre/orion/fus187/proj-shared/models/fsq_video_codecs_2ch"

# The residual spectro codecs were built at patch (8,16). The backbone spectro
# tokenizer patch MUST match the codec's (constructor asserts codec.n_tok ==
# (freq_bins//F_p)*(trunc_t//T_p)), so we build the FSQ config at (8,16).
SPEC_PATCH_F, SPEC_PATCH_T = 8, 16


def make_production(d_model=1024, n_layers=48, n_heads=16):
    diagnostics, actuators = trn.build_configs(
        chunk_duration_s=0.05,
        use_video=["tangtv_lower", "tangtv_upper"],
        use_spectro=["ece", "co2", "bes", "mhr"],
        spectro_patch_f=SPEC_PATCH_F, spectro_patch_t=SPEC_PATCH_T,
        prediction_horizon_s=0.2,
    )
    m = E2EFoundationModel(
        diagnostics=diagnostics, actuators=actuators,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=0.1,
        # --- spectro FSQ (all 4) ---
        spectro_fsq=True, spectro_fsq_codec_dir=SPEC_CODEC_DIR,
        spectro_code_pred_hidden=512, spectro_code_pred_layers=2,
        spectro_code_temperature=1.0,
        spec_descriptor=True, spec_descriptor_tcol=6, spec_descriptor_hidden=512,
        spec_descriptor_horizons=(2, 4),
        # --- video FSQ (split lower/upper) ---
        video_fsq=True, video_fsq_codec_dir=VIDEO_CODEC_DIR,
        video_code_pred_hidden=512, video_code_pred_layers=2,
        video_code_temperature=1.0,
        # --- slow-TS + fast-TS continuous (NO codec) ---
        fastts_fsq=False, slow_ts_fsq=False,
        # --- misc (match live g3fix) ---
        history_windows=1, use_actuator_film=False,
        spectro_seam_refine=False, seam_refine_hidden_ch=16, spectro_refine_kernel=3,
        spectro_inv_stem=False, spectro_inv_stem_ch=64,
        spectro_freq_stem=False, spectro_freq_stem_hidden=128,
    )
    return m, diagnostics, actuators


def breakdown(m):
    total = trainable = frozen = 0
    groups = defaultdict(lambda: [0, 0])  # name -> [trainable, frozen]
    for name, p in m.named_parameters():
        n = p.numel(); total += n
        tr = p.requires_grad
        if tr:
            trainable += n
        else:
            frozen += n
        parts = name.split('.')
        top = parts[0]
        if top == 'diag_tokenizers':
            key = f"diag_tokenizers.{parts[1]}"
        elif top == 'diag_heads':
            is_codec = (len(parts) > 2 and parts[2] == 'codec')
            key = f"diag_heads.{parts[1]}." + ("codec[FROZEN]" if is_codec else "pred")
        elif top == 'act_tokenizers':
            key = f"act_tokenizers.{parts[1]}"
        elif top == 'spec_descriptor_heads':
            key = f"spec_descriptor_heads.{parts[1]}"
        elif top == 'backbone':
            key = "backbone"
        else:
            key = top
        groups[key][0 if tr else 1] += n
    return total, trainable, frozen, groups


m, diags, acts = make_production()
total, trainable, frozen, groups = breakdown(m)

print("############### d1024 / 48L  PRODUCTION (full-modality, FSQ spectro+video) ###############")
print(f"n_total_tokens={m.n_total_tokens}  n_diag_tokens={m.n_diag_tokens}")
print(f"TOTAL={total:,}  TRAINABLE={trainable:,}  FROZEN={frozen:,}")
print("\n--- per-key breakdown (params : trainable / frozen) ---")
for k in sorted(groups, key=lambda x: -(groups[x][0] + groups[x][1])):
    tr, fr = groups[k]
    print(f"  {tr+fr:>13,}  (train {tr:>13,} | froz {fr:>13,})  {k}")

print("\n--- token layout ---")
for ts in m.token_layout:
    print(f"  {ts.name:<24} tokens={ts.slice_.stop-ts.slice_.start:<5} diag={ts.is_diagnostic}")
