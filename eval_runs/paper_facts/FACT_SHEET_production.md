# PRODUCTION model fact sheet — d1024 / 48L, FULL-modality (paper correction)

**Purpose.** Correct a paper inaccuracy: the PRODUCTION model reported in the paper is the
**full-modality** d1024/48L world model — NOT the ece-only d512 pilot in `FACT_SHEET.md §5`.
This sheet rebuilds the production parameter count by **BUILDING the model on CPU and counting**
(`eval_runs/paper_facts/build_and_count_production.py`), not by estimating.

**Headline result — every component EXACT, ZERO projections required.** All 4 spectro FSQ
codecs AND both split-video FSQ codecs already exist on disk, so the model constructs
fully (all codecs load, no shape errors) and every count below is `[exact]`.

- **TOTAL = 1,203,520,250** (~1.20 B) params
- **TRAINABLE = 1,145,387,460** (~1.15 B)
- **FROZEN = 58,132,790** (~58.1 M — the 4 spectro + 2 video FSQ codecs)
- **Sequence length = 2,524 tokens** (2,489 diagnostic + 35 actuator)

Reproduce: `python eval_runs/paper_facts/build_and_count_production.py`
(sources env via `scripts/slurm_frontier/_frontier_common.sh`, CPU-only, read-only,
no training, no checkpoint writes, running chain + shared cache untouched).

---

## 0. Production config (confirmed against `train_e2e_stage1_d1024_48L.sh` + codecs on disk)

| Field | Value | Source |
|---|---|---|
| Backbone | `d_model=1024, n_layers=48, n_heads=8` (head_dim 128) | launcher `train_e2e_stage1_d1024_48L.sh --n_heads 8`, CONFIRMED live (job 5029250: `n_heads=8 tokens=2524 params=1203.52M`). Head count does NOT affect the param count (8 vs 16 identical); the earlier "16" was an assumption, corrected 2026-07-18. |
| Diagnostics | 14 (7 slow-TS + 1 fast-TS continuous; 4 spectro FSQ; 2 video FSQ) | `build_configs` registries |
| Actuators | 7 | `ACTUATOR_MODALITIES` |
| FSQ scope | spectrograms + video FSQ-coded (frozen codecs); slow-TS + fast-TS continuous | user spec; matches constructor branches |
| `use_spectro` | `ece co2 bes mhr` | launcher L258 |
| `use_video` | `tangtv_lower tangtv_upper` (SPLIT divertor — two separate enc/dec codecs) | launcher L257 |
| spectro patch (F_p, T_p) | **(8, 16)** — matches the residual codec family | see §2 (patch↔codec constraint) |
| spectro codec dir | `/lustre/orion/fus187/proj-shared/models/fsq_resid_p8_all/` | live d512 chain's dir; all 4 present |
| video codec dir | `/lustre/orion/fus187/proj-shared/models/fsq_video_codecs_2ch/` | split lower/upper present |

> **The split-video codecs are NO LONGER "planned-not-trained".** `project-next-run-split-video-codec`
> memory anticipated two separate upper/lower codecs; they now EXIST
> (`video_codec_tangtv_lower.pt` + `video_codec_tangtv_upper.pt`, 2ch each), and
> `VIDEO_MODALITIES` already registers `tangtv_lower`/`tangtv_upper`. No projection needed.

---

## 1. Per-modality shape + token table

STFT geometry (all spectrograms): `n_fft=1024, hop=256, fs=500 kHz` → `freq_bins=512`,
`time_frames = round(0.05·500000/256) = 98`, codec truncates time to `Tq=96`.
Spectro tokens `= (512/F_p)·(96/T_p)`. With patch (8,16): `(512/8)·(96/16)=64·6=384`.

| Modality | Physical name | Kind | Tokenizer | Input shape (per window) | Tokens | Codec? |
|---|---|---|---|---|---:|---|
| ts_core_density | Thomson core density | slow_ts continuous | `SlowTimeSeriesTokenizer` (Linear, 1 tok/ch) | (44, 5) | 44 | none |
| ts_core_temp | Thomson core temperature | slow_ts | same | (44, 5) | 44 | none |
| ts_tangential_density | Thomson tangential density | slow_ts | same | (10, 5) | 10 | none |
| ts_tangential_temp | Thomson tangential temperature | slow_ts | same | (10, 5) | 10 | none |
| cer_ti | CER ion temperature | slow_ts | same | (48, 5) | 48 | none |
| cer_rot | CER rotation | slow_ts | same | (48, 5) | 48 | none |
| mse | Motional Stark Effect | slow_ts | same | (69, 5) | 69 | none |
| filterscopes | fast time-series (fast-TS) | fast_ts continuous | `FastTimeSeriesTokenizer` (Conv1d stem+patch, stride 50) | (8, 500) | 80 | none |
| **ece** | Electron Cyclotron Emission | spectrogram | `SpectrogramTokenizer` Conv2d (8×16) | (40, 512, 96) | **384** | **FSQ (frozen)** |
| **co2** | CO2 interferometer | spectrogram | Conv2d (8×16) | (4, 512, 96) | **384** | **FSQ (frozen)** |
| **bes** | Beam Emission Spectroscopy | spectrogram | Conv2d (8×16) | (16, 512, 96) | **384** | **FSQ (frozen)** |
| **mhr** | Mirnov / magnetics (high-freq) | spectrogram | Conv2d (8×16) | (6, 512, 96) | **384** | **FSQ (frozen)** |
| **tangtv_lower** | tangential-TV, lower divertor (raw cams ch0,ch2) | video | `VideoTokenizer` tube-patch (3,12,12) | (2, 3, 120, 360) | **300** | **FSQ (frozen)** |
| **tangtv_upper** | tangential-TV, upper divertor (raw cams ch4,ch6) | video | tube-patch (3,12,12) | (2, 3, 120, 360) | **300** | **FSQ (frozen)** |
| pin | Neutral-beam injected power | actuator | `ActuatorTokenizer` Conv1d | (8, 2000) | 5 | none |
| beam_voltage | Neutral-beam voltage | actuator | same | (8, 2000) | 5 | none |
| tin | Neutral-beam ion torque | actuator | same | (8, 2000) | 5 | none |
| ech_power | ECH power | actuator | same | (12, 2000) | 5 | none |
| gas_flow | Gas-injection flow | actuator | same | (11, 2000) | 5 | none |
| gas_raw | Gas-injection raw | actuator | same | (11, 2000) | 5 | none |
| rmp | RMP coil current | actuator | same | (12, 2000) | 5 | none |

Video tube-patch geometry: `(120/12)·(360/12)·(3/3) = 10·30·1 = 300` tokens (matches both codecs).
Actuator window = `prediction_horizon_s(0.2)·10 kHz = 2000` samples (spans the horizon, not the input chunk).

**Token-sequence layout** (flat backbone sequence, order `[slow_ts | fast_ts | spectro | video | actuators]`):

| Block | tokens |
|---|---:|
| slow-TS (44+44+10+10+48+48+69) | 273 |
| fast-TS (filterscopes) | 80 |
| spectro (ece+co2+bes+mhr = 4×384) | 1,536 |
| video (tangtv_lower+upper = 2×300) | 600 |
| **n_diag_tokens** | **2,489** |
| actuators (7×5) | 35 |
| **n_total_tokens** | **2,524** |

> vs d512 pilot's 772 tokens (ece-only, no video). Production is **3.3× longer sequence**.

---

## 2. Codec inventory + the patch↔codec constraint

**All required codecs EXIST (loaded + counted; none projected):**

| Modality | Codec `.pt` | patch | n_tok | C | fsq_dim/L | params | status |
|---|---|---|---:|---:|---|---:|---|
| ece spectro | `fsq_resid_p8_all/spectro_codec_ece.pt` | (8,16) | 384 | 40 | 48/16 | 15,601,112 | EXISTS [exact] |
| co2 spectro | `fsq_resid_p8_all/spectro_codec_co2.pt` | (8,16) | 384 | 4 | 48/16 | 13,241,780 | EXISTS [exact] |
| bes spectro | `fsq_resid_p8_all/spectro_codec_bes.pt` | (8,16) | 384 | 16 | 48/16 | 14,028,224 | EXISTS [exact] |
| mhr spectro | `fsq_resid_p8_all/spectro_codec_mhr.pt` | (8,16) | 384 | 6 | 48/16 | 13,372,854 | EXISTS [exact] |
| tangtv_lower video | `fsq_video_codecs_2ch/video_codec_tangtv_lower.pt` | (3,12,12) | 300 | 2 | 24/8 | 944,410 | EXISTS [exact] |
| tangtv_upper video | `fsq_video_codecs_2ch/video_codec_tangtv_upper.pt` | (3,12,12) | 300 | 2 | 24/8 | 944,410 | EXISTS [exact] |
| **frozen total** | | | | | | **58,132,790** | |

Embedded codec counts are byte-identical to the standalone `.pt` files (verified).
All spectro codecs use `bg_subtract=True` (residual/R-space); internal `d_model=256`
(independent of backbone d_model → identical at d512 and d1024).

**Patch↔codec constraint (why patch = (8,16), not the launcher default (512,4)).**
The constructor asserts `codec.n_tok == (freq_bins/F_p)·(trunc_t/T_p)`. Available spectro
codec families and their token budgets:
- `fsq_resid_p8_all` → patch (8,16) → **384 tok** (all 4 modalities; the live d512 chain's dir) ← USED
- `fsq_spectro_residual_codecs` / `fsq_resid_ece_sharpdec` / `fsq_spectro_codecs_tok96` → patch (32,16) → 96 tok (all 4)
- **No spectro codec exists at patch (64,32) → 24 tok** (the memory-preferred p64pe patch).

> **Important design note (paper honesty).** The most-recent full-modality *trained*
> d1024/48L checkpoint (`e2e_stage1_d1024_p64pe`) used patch **(64,32)** but with
> **GENERATIVE spectro heads and resize-conv video — NOT FSQ** (verified from its args:
> `spec_generative=True`, `spec_fsq=None`, `video_fsq=None`, zero `codec` keys in its
> state_dict). So the "FSQ-coded spectro+video production" the user specifies is a
> DISTINCT design point that pairs with the (8,16)/384-tok (or 96-tok) codec families —
> NOT with the p64pe checkpoint's geometry. This build uses the (8,16) residual codecs,
> the canonical FSQ family the live chain relies on. Choosing the 96-tok family instead
> would shrink the spectro tokenizers/heads/codecs and the sequence length (see §4).

---

## 3. PRODUCTION parameter table (d1024 / 48L, full-modality) — all [exact]

| Component | Params | Trainable | Frozen | Tag |
|---|---:|---:|---:|---|
| backbone (48 BackboneBlocks + step_cond MLP + final_norm) | 609,082,368 | 609,082,368 | 0 (+32 buf) | [exact] |
| spectro tokenizers ×4 (ece/co2/bes/mhr) | 414,801,920 | 414,801,920 | 0 | [exact] |
| FROZEN spectro codecs ×4 | 56,243,970 | 0 | 56,243,970 | [exact] |
| fast-TS tokenizer (filterscopes) | 36,892,992 | 36,892,992 | 0 | [exact] |
| fast-TS continuous head | 36,872,513 | 36,872,513 | 0 | [exact] |
| actuator tokenizers ×7 | 28,722,176 | 28,722,176 | 0 | [exact] |
| spectro descriptor heads ×4 | 9,149,856 | 9,149,856 | 0 | [exact] |
| spectro FSQ code heads ×4 | 4,725,760 | 4,725,760 | 0 | [exact] |
| video tokenizers ×2 (tangtv lower/upper) | 3,002,368 | 3,002,368 | 0 | [exact] |
| FROZEN video codecs ×2 | 1,888,820 | 0 | 1,888,820 | [exact] |
| video FSQ code heads ×2 | 1,771,904 | 1,771,904 | 0 | [exact] |
| slow-TS tokenizers ×7 | 329,728 | 329,728 | 0 | [exact] |
| slow-TS continuous heads ×7 | 35,875 | 35,875 | 0 | [exact] |
| **TOTAL** | **1,203,520,250** | **1,145,387,460** | **58,132,790** | [exact] |

Per-item detail (where multiple in a group differ):
- Each spectro tokenizer: ece 106,780,672 · bes 103,634,944 · mhr 102,324,224 · co2 102,062,080
  (the SpectrogramTokenizer dominates the whole model's tokenizer bank; scale with n_channels).
- Each spectro FSQ code head (`.pred`): 1,181,440 (ece=co2=bes=mhr; head is n_channel-independent).
- Each spectro descriptor head: 2,287,464 (×4).
- Each frozen spectro codec: ece 15,601,112 · bes 14,028,224 · mhr 13,372,854 · co2 13,241,780.
- Each video tokenizer: 1,501,184 (lower=upper). Each video FSQ head: 885,952. Each video codec: 944,410.
- Each actuator tokenizer: ech_power=rmp 4,922,368 · gas_flow=gas_raw 4,512,768 · pin=beam_voltage=tin 3,283,968.
- Each slow-TS continuous head: 5,125 (single Linear); slow-TS tokenizers 17,408–77,824 (scale w/ n_channels).

**Frozen mechanics.** The FSQ codecs (spectro `SpectrogramCodeHead.codec`, video
`VideoCodeHead.codec`) are submodules loaded with `requires_grad_(False)` + `.eval()`, so
their params ARE inside `model_state_dict` (excluded from the DDP reducer; AdamW never
updates them). Slow-TS + fast-TS have NO codec (continuous heads, fully trainable).

---

## 4. d512 pilot (trained, NOT production) — kept as-is for comparison

The live/trained chain (`e2e_g3fix_kanneal_v2`) is **ece-only, no video, d512** — verified
from its `latest.pt` args (`d_model=512, n_layers=12, use_spectro=['ece'], use_video=[]`).
This is the method-development pilot, NOT the paper's production model. Numbers from
`FACT_SHEET.md §1/§5` (unchanged here):

| | d512 pilot (ece-only, trained) | d1024/48L PRODUCTION (full, built) |
|---|---:|---:|
| **TOTAL** | **120,702,180** | **1,203,520,250** |
| **TRAINABLE** | **105,101,068** | **1,145,387,460** |
| **FROZEN (FSQ codecs)** | **15,601,112** (ece codec only) | **58,132,790** (4 spectro + 2 video) |
| backbone | 39,011,840 | 609,082,368 |
| spectro tokenizers | 28,224,512 (ece only) | 414,801,920 (×4) |
| video tokenizers | 0 (none) | 3,002,368 (×2) |
| fast-TS tok + head | 20,118,145 | 73,765,505 |
| actuator tokenizers | 14,361,088 | 28,722,176 |
| spectro descriptor | 2,283,368 (×1) | 9,149,856 (×4) |
| sequence length | 772 tokens | 2,524 tokens |

---

## 5. Assumptions + projection status (explicit)

1. **ZERO components projected.** Every number is `[exact]` — the model built cleanly at
   d1024/48L with all 6 codecs loaded; token layout verified; embedded codec counts
   byte-match the standalone `.pt` files.
2. **`n_heads=8`** — the launcher `train_e2e_stage1_d1024_48L.sh` hard-codes `--n_heads 8`
   (head_dim 128), CONFIRMED on the live model. (An earlier draft assumed 16; corrected.)
   This does NOT change the count: `nn.MultiheadAttention` param count is `n_heads`-independent
   at fixed d_model. So the 1.2035 B total holds for either n_heads — 8 is the real config.
3. **Patch = (8,16) → 384 spectro tokens** (the `fsq_resid_p8_all` residual family). This is
   the FSQ family the live chain uses and the only 4-modality family besides the 96-tok
   `_residual_codecs`/`_tok96` families. If the production run instead adopts the **96-tok**
   spectro codecs (patch 32,16), the spectro tokenizers/heads/codecs and sequence length
   shrink accordingly (spectro tokens 1,536→384; re-run the counter with
   `SPEC_CODEC_DIR=…/fsq_spectro_residual_codecs SPEC_PATCH_F=32 SPEC_PATCH_T=16`).
   The **p64pe (64,32) geometry is incompatible with FSQ** (no 24-tok codec exists) — it was
   a generative-head run, so it is NOT the FSQ-production geometry.
4. **All other knobs held at the live-g3fix design** (residual bg_subtract codecs;
   descriptor horizons 2,4; hidden 512; code_pred hidden 512 / layers 2; no freq_stem;
   no seam-refine/inv-stem; actuators-as-tokens, no FiLM; history_windows=1). Any run that
   flips a design flag (freq_stem on, FiLM, MaskGIT head, etc.) will differ.

Build/count script (kept, NOT committed): `eval_runs/paper_facts/build_and_count_production.py`.
No repo `src/` model code modified; no checkpoints copied; running chain + shared cache untouched.
