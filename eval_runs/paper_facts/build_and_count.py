import sys, torch
from collections import defaultdict
sys.path.insert(0, "src")
sys.path.insert(0, "scripts/training")
# import build_configs from the trainer module without running main
import importlib.util
spec = importlib.util.spec_from_file_location("trn", "scripts/training/train_e2e_stage1.py")
# Avoid executing argparse: import module attributes we need directly.
from tokamak_foundation_model.e2e.model import E2EFoundationModel

# Reconstruct build_configs by importing it from the module. The module top-level
# defines build_configs and the registries with no side effects on import.
trn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trn)  # executes top-level defs; main() is guarded by __main__

CODEC_DIR = "/lustre/orion/fus187/proj-shared/models/fsq_resid_p8_all"

def make_model(d_model, n_layers, n_heads):
    diagnostics, actuators = trn.build_configs(
        0.05, use_video=[], use_spectro=['ece'],
        spectro_patch_f=8, spectro_patch_t=16, prediction_horizon_s=0.2)
    m = E2EFoundationModel(
        diagnostics=diagnostics, actuators=actuators,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=0.1,
        spectro_fsq=True, spectro_fsq_codec_dir=CODEC_DIR,
        spectro_code_pred_hidden=512, spectro_code_pred_layers=2, spectro_code_temperature=1.0,
        spec_descriptor=True, spec_descriptor_tcol=6, spec_descriptor_hidden=512,
        spec_descriptor_horizons=(2,4),
        history_windows=1, use_actuator_film=False,
        spectro_seam_refine=False, seam_refine_hidden_ch=16, spectro_refine_kernel=3,
        spectro_inv_stem=False, spectro_inv_stem_ch=64,
        spectro_freq_stem=False, spectro_freq_stem_hidden=128,
    )
    return m, diagnostics, actuators

def breakdown(m):
    total=trainable=frozen=0
    groups=defaultdict(lambda:[0,0])  # name -> [trainable, frozen]
    for name,p in m.named_parameters():
        n=p.numel(); total+=n
        tr = p.requires_grad
        if tr: trainable+=n
        else: frozen+=n
        parts=name.split('.')
        top=parts[0]
        if top=='diag_tokenizers': key=f"diag_tokenizers.{parts[1]}"
        elif top=='diag_heads':
            key=f"diag_heads.{parts[1]}." + ("codec[FROZEN]" if (len(parts)>2 and parts[2]=='codec') else "pred")
        elif top=='act_tokenizers': key=f"act_tokenizers.{parts[1]}"
        elif top=='spec_descriptor_heads': key=f"spec_descriptor_heads.{parts[1]}"
        elif top=='backbone': key="backbone"
        else: key=top
        groups[key][0 if tr else 1]+=n
    return total,trainable,frozen,groups

def coarse(groups):
    c=defaultdict(lambda:[0,0])
    for k,(tr,fr) in groups.items():
        if k=='backbone': ck='backbone'
        elif k.startswith('diag_tokenizers'): ck='diag_tokenizers (all)'
        elif k.endswith('codec[FROZEN]'): ck='diag_heads codecs [FROZEN]'
        elif k.startswith('diag_heads'): ck='diag_heads pred (trainable)'
        elif k.startswith('act_tokenizers'): ck='act_tokenizers (all)'
        elif k.startswith('spec_descriptor_heads'): ck='spec_descriptor_heads'
        else: ck=k
        c[ck][0]+=tr; c[ck][1]+=fr
    return c

for (dm,nl,nh,label) in [(512,12,8,"d512 (pilot g3fix)"), (1024,48,16,"d1024/48L (production)")]:
    m,diags,acts = make_model(dm,nl,nh)
    total,trainable,frozen,groups=breakdown(m)
    print(f"\n########## {label}  d_model={dm} n_layers={nl} n_heads={nh} ##########")
    print(f"n_total_tokens={m.n_total_tokens}  n_diag_tokens={m.n_diag_tokens}")
    print(f"TOTAL={total:,}  TRAINABLE={trainable:,}  FROZEN={frozen:,}")
    c=coarse(groups)
    print("  --- coarse (trainable / frozen) ---")
    for k in sorted(c, key=lambda x:-(c[x][0]+c[x][1])):
        tr,fr=c[k]; print(f"    {tr+fr:>13,}  (train {tr:>12,} | froz {fr:>11,})  {k}")
    # token layout
    print("  --- token layout ---")
    for ts in m.token_layout:
        print(f"    {ts.name:<24} tokens={ts.slice_.stop-ts.slice_.start:<5} diag={ts.is_diagnostic}")
