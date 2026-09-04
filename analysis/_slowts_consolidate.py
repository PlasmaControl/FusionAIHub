import json, os, math
OUT='eval_runs/codec_recon_figs'
S7=('cer_rot','cer_ti','mse','ts_core_density','ts_core_temp','ts_tangential_density','ts_tangential_temp')
# level tracking (shot 204990, redraws excluded): (prod),(retrained),(best cermse arm)
LVL={'cer_rot':((0.846,0.534),(0.721,0.637),(0.830,0.792)),
     'cer_ti':((0.622,0.245),(0.870,0.809),(0.989,0.833)),
     'mse':((0.687,0.063),(0.362,0.051),(0.214,0.136)),
     'ts_core_density':((0.816,0.724),(0.896,0.821),None),
     'ts_core_temp':((1.058,0.812),(0.943,0.847),None),
     'ts_tangential_density':((0.990,0.988),(0.942,0.988),None),
     'ts_tangential_temp':((0.929,0.985),(0.971,0.996),None)}
BEST={'cer_rot':'cerrot_je05_best','cer_ti':'certi_je05_last','mse':'mse_je05_ew2_best'}
RO={'cer_rot':0.5788,'cer_ti':0.5874,'mse':0.6536}
def load(p):
    return json.load(open(p)) if os.path.exists(p) else []
lines=[]
hdr=(f"{'modality':<22}{'arm':<18}{'pooled':>8}{'k4flr':>7}{'/flr':>6}{'readout':>8}{'corr':>6}"
     f"{'sr_p':>6}{'sr_t':>6}{'lvlR':>6}{'lvlC':>7}{'codes':>6}{'bits':>6}{'%cap':>6}")
lines+=["SLOW-TS CODECS - CONSOLIDATED (12 held-out shots, 2600 windows, 10400 tokens)",
        "headline = slowts_nrmse_pooled (1.0 = that window's own constant mean; wcmean anchors EXACTLY 1.0,",
        "self EXACTLY 0.0). k4flr = out-of-sample PCA rank-4 MATCHED-RATE floor. readout = out-of-sample",
        "93-param linear read-out of the codec's OWN frozen codes. sr_* = std(recon)/std(gt) within-window;",
        "lvlR/lvlC = ACROSS-window level amplitude + correlation (the quantity the full-shot figure shows).",
        "bits = n_tok*H_joint/ln2 out of a 39.9-bit cap.", "", hdr, "-"*len(hdr)]
for s in S7:
    base=load(f'{OUT}/slowts_audit_{s}.json')
    fl=base[0].get('floor_pooled',{}).get('4',float('nan')) if base else float('nan')
    rows=[(r['arm'],r) for r in base]
    for r in load(f'{OUT}/slowts_audit_cermse_{s}.json'):
        if r['arm']==BEST.get(s): rows.append(('cermse:'+r['arm'],r))
    for i,(nm,r) in enumerate(rows):
        lv=LVL[s][i] if i<len(LVL[s]) and LVL[s][i] else (float('nan'),float('nan'))
        ro=RO.get(s,float('nan')) if i==0 else float('nan')
        lines.append(f"{s if i==0 else '':<22}{nm:<18}{r['slowts_nrmse_pooled']:>8.4f}{fl:>7.4f}"
              f"{r['slowts_nrmse_pooled']/fl:>6.2f}{ro:>8.4f}{r['slowts_corr']:>6.3f}"
              f"{r['slowts_std_ratio_pooled']:>6.3f}{r['slowts_std_ratio_t']:>6.3f}"
              f"{lv[0]:>6.3f}{lv[1]:>7.3f}{r['n_distinct_codes']:>6d}{r['bits_realised']:>6.1f}"
              f"{100*r['bits_realised']/r['bits_cap']:>5.0f}%")
    lines.append("")
open(f'{OUT}/SLOWTS_CONSOLIDATED.txt','w').write("\n".join(lines))
print("\n".join(lines))
