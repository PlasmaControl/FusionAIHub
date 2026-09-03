"""Model-free preliminary predictability proxy: does η-removal make the mode more
forecastable t->t+1 (50 ms)? Compares RAW vs DENOISED 2D mode-band pattern correlation
and freq-profile correlation between consecutive windows. No codec, no world model."""
import sys; sys.path.insert(0, "src"); sys.path.insert(0, "scripts/training")
import numpy as np, torch, h5py
from spectro_bg import channel_coherent_denoise, raw_stft_complex
FS,NFFT,HOP=500_000.,1024,256; DF=FS/NFFT/1e3
LO,HI=int(round(5/DF)),int(round(40/DF)); WFR=int(round(0.05*FS/HOP))
shot="200729"; a,b=0,40   # ece channels_to_use
with h5py.File(f"/lustre/orion/fus187/proj-shared/foundation_model/{shot}_processed.h5","r") as f:
    x=f["ece"]["xdata"][:]; i0=int(np.searchsorted(x,1.0)); i1=i0+int(2.0*FS)
    sig=torch.tensor(np.nan_to_num(f["ece"]["ydata"][a:b, i0:i1]),dtype=torch.float32)
S=raw_stft_complex(sig,NFFT,HOP)
raw=S.abs().numpy(); den,_=channel_coherent_denoise(S,2,1,1); den=den.numpy()
C,F,T=raw.shape; nwin=T//WFR
# strongest-mode channel by raw band prominence
from scipy.ndimage import gaussian_filter1d as gf
prom=lambda M,c,w: (lambda p:(p-gf(p,6)))(np.abs(M[c,LO:HI,w*WFR:(w+1)*WFR]).mean(1))
z=[max(float((prom(raw,c,w)).max()) for c in range(C)) for w in range(nwin)]
order=np.argsort(-np.array(z)); act=order[:max(4,nwin//3)]   # active windows
def corr2d(M,c,w1,w2):
    A=np.abs(M[c,LO:HI,w1*WFR:(w1+1)*WFR]).ravel(); B=np.abs(M[c,LO:HI,w2*WFR:(w2+1)*WFR]).ravel()
    return float(np.corrcoef(A,B)[0,1]) if A.std()>0 and B.std()>0 else np.nan
def fprofcorr(M,c,w1,w2):
    A=np.abs(M[c,LO:HI,w1*WFR:(w1+1)*WFR]).mean(1); B=np.abs(M[c,LO:HI,w2*WFR:(w2+1)*WFR]).mean(1)
    return float(np.corrcoef(A,B)[0,1]) if A.std()>0 and B.std()>0 else np.nan
r2r,r2d,fpr,fpd=[],[],[],[]
for w in act:
    if w+1>=nwin: continue
    c=int(np.argmax([float(prom(raw,cc,w).max()) for cc in range(C)]))
    r2r.append(corr2d(raw,c,w,w+1)); r2d.append(corr2d(den,c,w,w+1))
    fpr.append(fprofcorr(raw,c,w,w+1)); fpd.append(fprofcorr(den,c,w,w+1))
print(f"ECE {shot}, n_active_pairs={len(r2r)}  (mode-band 5-40kHz, 50ms stride)")
print(f"  2D-pattern  t->t+1 corr:  RAW={np.nanmedian(r2r):.3f}   DENOISED={np.nanmedian(r2d):.3f}   delta={np.nanmedian(r2d)-np.nanmedian(r2r):+.3f}")
print(f"  freq-profile t->t+1 corr: RAW={np.nanmedian(fpr):.3f}   DENOISED={np.nanmedian(fpd):.3f}   delta={np.nanmedian(fpd)-np.nanmedian(fpr):+.3f}")
print("VERDICT:", "DENOISE IMPROVES pattern predictability" if np.nanmedian(r2d)>np.nanmedian(r2r)+0.03 else "no clear pattern-predictability gain")
