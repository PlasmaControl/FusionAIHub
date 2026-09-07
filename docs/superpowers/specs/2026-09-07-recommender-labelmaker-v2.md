# Appendix A — labelmaker v2 detailed plan (Opus Plan agent, condensed)

## A0. Corrections to the draft (measured)
A0.1 trailing NaN sample (V1). A0.2 record lengths: mhr (8, 2097153) 4.194 s sliding; ece (48, 3096576) 6.19 s; co2 (4, 4500001) 9.0 s; bes (64, 3145729) 6.29 s; mirnov (29, 8191977) 16.4 s; STFT cols (hop 128): 16,391 / 24,199 / 35,164 / 24,576 / 64,000; wide tiles 37/54/79/55/143; zoom tiles 10/14/20/14/36. A0.3 availability (V2). A0.4 filterscopes 0–7 real (V4). A0.5 QH_Database zero overlap (V6). A0.6 sawtooth O(n²) (V7). A0.7 masks storage, no co2 crosspower (U-Net trained on auto-spectra), SLURM 8 CPUs insufficient.

## A1. Measured constants
U-Net 7,852,002 params; V100S fp16 197 tiles/s flat batch 8–32; peak alloc bs8 2.61 GiB / bs16 5.17 / bs32 10.30; CPU prep 1 core: mhr 0.97 s, ece 1.41 s, co2 2.13 s (+0.19/0.29/0.50 zoom); `ndimage.label` 0.04–0.09 s (560–3437 comps; 7–254 ≥200 px); packbits 1.05–2.25 MB/channel; corpus datasets contiguous uncompressed (`ece[[8,20,40],:]` 0.02 s; full ece 594 MB 0.31 s); file 3.85 GB/shot.
Grids: wide bin k = 0.48828·k kHz (0.49–250), col 0.256 ms, tile 131 ms, stride 114.7 ms; zoom (`scipy.signal.decimate(x,4,ftype="iir",zero_phase=True)`) 0.12207 kHz/bin (0.12–62.5), col 1.024 ms, tile 524 ms, stride 459 ms. Zoom measured on 198658 co2: lit 0.0220→0.0554, centroid 37.3→22.1 kHz; exposes pickup rows (co2 1.7–2.3 kHz, ece 8.4–9.0 kHz).

## A2. Files
```
src/labelmaker/events/{__init__,schema,unet,channels,masks,tracks,transients,heuristics,text_weak,windows}.py, lexicons.yaml
src/labelmaker/annotate/{__init__,priors,candidates,render,sheet,gbdt}.py
src/labelmaker/features/resolve_events.py
src/labelmaker/models/runners/gbdt_npz.py
src/labelmaker/models/d3d_{eho_activity,qcm_activity,fishbone_activity,pickup_noise,qh_regime}_gbdt/{__init__,README.md,spec.py}
src/labelmaker/models/d3d_{sawtooth_activity,lh_transition}_rule/{__init__,README.md,spec.py}
scripts/labelmaker/{tokeye_masks.py,tokeye_masks.sbatch,jobstats_check.py,annotate_candidates.py,annotate_render.py,annotate_train.py,pin_unet.py}
tests/labelmaker/test_{events_schema,events_unet,events_masks,events_channels,events_tracks,events_transients,events_heuristics,events_text_weak,events_windows,annotate_priors,annotate_candidates,annotate_render,annotate_sheet,annotate_gbdt,gbdt_npz_runner,resolve_events,phenomenon_adapter}.py; data/{unet_golden,gbdt_golden}.npz
```
Modified: `features/namespace.py` (`SOURCES += ("events",)`, `SAMPLING_BY_SOURCE["events"]="nearest"`, `FeatureSpec("phenomenon_window_features", kind="waveform", sources=("events",))`), `run.py` (`_resolve_one_source` events branch; `STAGES += ("events","candidates","render")`), `config.py` (`Paths.events/events_file/masks/masks_file/annotate/events_index`, `mkdirs`), `labels/store.append_index(keys=...)`, `docs/LABELMAKER.md`.

## A3. Key signatures
```python
# events/schema.py
@dataclass(frozen=True) class Event: shot, source, phenomenon, t0_s, t1_s, f0_khz=nan, f1_khz=nan, confidence=nan, diag="", channel=-1, pass_name="", attrs={}, t_cov0_s=nan, t_cov1_s=nan, evidence_kind="detector", horizon_s=nan  # __post_init__ validates
def write_events(path, shot, events, *, run_id, merge=True, sources=None)  # atomic; carries forward other sources
def read_events(path, *, source=None, phenomenon=None) -> DataFrame; def index_rows(path); def intervals(events, phenomenon) -> (n,2)
# events/unet.py  (vendored BigTFUNetConfig/Model, 190 lines; CHECKPOINT_SHA256; N_PARAMS=7_852_002)
def load_unet(path=DEFAULT_CHECKPOINT, device="cpu", *, verify_sha256=True) -> nn.Module   # bare state_dict; not torch_pt.load_module
# events/channels.py
ROUND1_PLAN = (mhr 0, mhr 4, ece 8, ece 20, ece 40, co2 0, co2 2, bes 26, bes 28, mirnov 0/8 fallback_for="mhr")
def plan_for(corpus_file, plan=ROUND1_PLAN) -> (specs, reasons)   # header-only reads; shape[-1]<2 absent
def coverage(corpus_file, diag) -> (t_cov0_s, t_cov1_s)
# events/masks.py  TILE=512 OVERLAP=64 PROB_THRESHOLD=0.2 (== ae.labels) ZOOM_DECIM=4 N_BANDS=16
def read_waveform(corpus_file, group, channel) -> (y, fs_hz, t0_s, t1_s)  # strips trailing non-finite; fs from span
def prep(y, *, decim=1) -> ((512,T) float32, meta)   # ae.transform.compute_stft + standardise; NOT model_input
def tile(spec) -> (tiles, meta); def stitch(pred, meta) -> (2,512,T); def infer(model, spec, device, *, batch=96, amp=True)  # OOM → halve batch
def band_logpow(raw_spec, n_bands=16) -> (16,T) f16; def pack/unpack; def write_masks(path, shot, blocks, *, merge=True); def read_mask(path, key)
# events/tracks.py  MIN_AREA=200 MERGE_GAP_COLS=40 FREQ_OVERLAP=0.5 CONF_PCT=95
def components(coh, *, min_area) ; def merge(comps, *, max_gap, freq_overlap)  # raddet port; min_area first, t0-sorted early break; test pins fast path == literal port
def descriptors(...) -> Track(t0_s,t1_s,f0_khz,f1_khz,f_centroid_khz,chirp_khz_per_ms,chirp_r2,bandwidth_khz,duration_ms,duty,mean_prob,conf,n_pix,n_components,row_lit_fraction,pickup)
def harmonics(tracks, *, tol=0.06, min_overlap_s=0.02) ; def cooccurrence(by_channel, *, iou_min=0.2); def tracks_to_events(...)
# events/transients.py (tokeye.elmspec port) column_activity, extract_bursts, elm_events(find_peaks; smooth 0.64 ms, prominence 0.03, min_distance 3 ms), elm_clock, elm_free_intervals(max_rate_hz=5, min_duration_s=0.05)
# events/heuristics.py
def envelope(y, t_s, env_ms=1.0)  # vectorised bincount; ==omnimode to 1e-12, ≥10× faster (timing test)
def sawtooth_events(ece_y, ece_t_s, *, shot, drop_frac=0.02, min_channels=2, min_interval_ms=10, gap_ms=2, span_ms=8)  # envelope once; inversion_block/has_inversion verbatim
def lh_transitions(dalpha_t, dalpha_y, *, ne_t, ne_y, betan_t, betan_y, pinj_t, pinj_y, shot, drop_frac=0.30, drop_window_ms=5, ne_rise_frac=0.05, min_pinj_kw=500)
def actuator_intervals(features, *, shot)  # canonical pinj_total>500 kW, ech_power_total>1e5 W, |rmp|>0.5 kA, gas>0.5 V; hysteresis 2:1; min 20 ms; nbi_counter when sign(tinj)≠sign(ip)
def qh_candidates(eho_tracks, elm_free, nbi, ip_flattop, *, shot)  # proxy; confidence=min(track.conf, elm_free_frac)
# events/text_weak.py  shot_block(shot) after marker; run_context(shot); hits(text, lexicon) space-glued word match; weak_labels(shots) -> DataFrame
# events/windows.py  FEATURE_NAMES (46, frozen, asserted literally); window_features(...) -> (46,); window_grid(width_s=0.34, stride_s=0.17); shot_window_features(shot, paths) -> (T,), (46,T)
# annotate/priors.py  PhenomenonPrior(name, f_khz, duration_ms, chirp_sign, chirp_khz_per_ms, min_harmonics, require_elm_free, require_nbi, require_counter_injection, passes, roles, render_diags, lexicon_key, window_s, negatives_per_positive).score(feats, text_hits) -> (0..1, why); TEXT_ONLY_CEILING=0.25
# annotate/candidates.py  rank(shots, phenomenon, paths, *, top_k=200, per_shot_max=2, seed=20260907) -> [CandidateWindow]; negatives(..., n, seed, positives) thirds
# annotate/render.py  render_window(cw, paths, out_png, *, dpi=130) (Figure+FigureCanvasAgg, no pyplot); render_set(cws, phenomenon, out_dir, paths, *, seed) -> summary
# annotate/sheet.py  read_sheet(path, manifest_path) (validates keys; labels y/n/?/""); sheet_stats(df)
# annotate/gbdt.py   predict_proba(bundle, x) numpy tree traversal + isotonic np.interp; load_bundle(path)
# models/runners/gbdt_npz.py  make_predictor(model_dir, *, feature_names, threshold, hysteresis) -> predict(built) -> (1,T,1); feature-name mismatch raises; narrows built.valid in place where no coverage
```
Normalisation: z-score over the whole record (as `ae_dataset.py` and `run_tokeye.py`); `--norm {record,plasma}` switch; pilot decides.

## A4. Round-1 priors table
| phen | band | duration | chirp | harmonics | ELM-free | pass | mask ch | render panels |
|---|---|---|---|---|---|---|---|---|
| eho | 2–20 kHz | ≥100 ms | flat \|ċ\|<0.02 | ≥2 (res<0.06) | yes (<5 Hz) | zoom | mhr 0,4; co2 0,2 | mhr/co2 zoom spec+mask, Dα, n̄ₑ, pinj/tinj sign, betan, rmp |
| qcm | 40–120 kHz | ≥50 ms | flat | 0–1 | yes | wide | mhr 0,4; bes 26,28; co2 | mhr/bes wide, Dα, n̄ₑ, pinj, ech |
| sawtooth | precursor 2–20 kHz optional | crash train 20–200 ms | — | — | no | zoom | ece 48 heuristic + mhr 0 | ECE raster (inversion), 3 ECE traces, mhr zoom, Dα, ip, ech |
| fishbone | 5–30 kHz | bursts 2–15 ms, repeat 5–60 ms | down −10…−0.3, r²>0.5 | 0 | no | zoom | mhr 0,4; ece 8 | mhr zoom, ece 8 zoom, neutron_rate, pinj, ip |
| pickup | any | ≥80% record | 0 | — | no | both | all | offending channel full-record mask vs clean channel |
| lh | point | — | — | — | — | — | filterscopes 0–7 | Dα ±200 ms, n̄ₑ, betan, pinj, ech, gas, ip, mhr wide |
| qh | eho∧elm_free∧nbi | ≥100 ms | — | — | yes | zoom | mhr 0,4 | mhr zoom+mask, Dα flat, n̄ₑ, tinj vs ip sign, betan, rmp, snippet |

## A5. SLURM (masks)
```bash
#SBATCH --job-name=tokeye-masks --partition=gpu --qos=gpu-stellar --nodes=1 --ntasks=2 --gpus-per-task=1
#SBATCH --cpus-per-task=20 --mem-per-cpu=1G --time=02:00:00 --array=0-7%2
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%A_%a.out
export PYTHONPATH=$REPO/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
srun --cpu-bind=cores $ROOT/envs/phase3/bin/python -u $REPO/scripts/labelmaker/tokeye_masks.py \
  --shot-file $ROOT/recommender_v1.txt --chunk $SLURM_ARRAY_TASK_ID --n-chunks 8 --rank $SLURM_PROCID --world 2 \
  --root $ROOT --corpus /scratch/gpfs/EKOLEMEN/foundation_model --plan round1 --tile-batch 96 --prep-workers 18 --prefetch 4 --amp --timeout 240
# then: sbatch --dependency=afterany:$JOBID --partition=serial --time=00:10:00 --wrap "... jobstats_check.py --job-id $JOBID --min-cpu 70 --min-gpu 70 --min-cpu-mem 70 --min-gpu-mem 70 --out outputs/labelmaker/events/jobstats.json"
```
Cost model: ≈15.2 core-s and ≈379 tiles per shot → 0.84 s A100 GPU → balance 18 cores/GPU. 2,000 shots ≈ 25–40 min on 4 A100s.

## A6. Tasks (dependencies 1→{2,3,7}→4→{5,6}→8→9→{10,11}→12→13→14→15→16(user)→17→18)
1 schema+config+append_index(keys) [451+12 green] · 2 unet vendored + pin_unet golden (max_abs_diff 0.0) · 3 channels+masks [smoke on 198658 reproduces A1 table] · 4 tracks [synth exact; fast merge == literal] · 5 transients+ELM · 6 heuristics [198658: 45±3 sawteeth, 76±5 ms, <1 s; L→H on ≥3/10 H-mode shots] · 7 lexicons+text_weak [≥1 sawtooth hit on ≥5% of 200 shots] · 8 windows+resolve_events+namespace/run edits [full suite green] · 9 tokeye_masks.py [`--limit 5 --device cpu` end-to-end] · 10 jobstats_check.py [parses old dataset.json → 23/6/1/97, exits 1] · 11 sbatch + 20-shot pilot [measure tiles/s, RSS, util; size knobs] · 12 pilot ablation norm/zoom vs aemodes annotations [one-page report] · 13 production 2,000 [jobstats gate; ≥95% events files] · 14 priors+candidates [≥50 candidates/phenomenon; text ceiling] · 15 render+sheet [7 folders × 120 chunks; 10 PNGs each hand-inspected] · 16 user annotates [≥60 labelled, ≥15 positives per trained phenomenon] · 17 annotate_train + gbdt runner + model folders [OOF AUROC vs permutation & text-only baselines; golden vs sklearn 1e-12] · 18 infer end-to-end + docs [existing groups untouched; `source=model` events].

## A7. Tests (≈180) — highlights
Trailing-NaN strip and interior-NaN raise; fs from span; prep == `ae.transform.compute_stft`; zoom axis 0.12207·k; tile/stitch identity & overlap averaging; pack/unpack at every T mod 8; synth_mask analytic descriptors (EHO 3 tracks n_harmonics=2; fishbone chirp −0.78±0.01 r²>0.99; pickup row; ELM period 15.36 ms); merge fast-path == literal port on 200 random sets; envelope == omnimode 1e-12 and ≥10× faster; sawtooth rejects same-sign ELM impostor; lh returns [] without NBI; NaN channels ≥8 don't poison mean; actuator hysteresis + counter-injection sign; text word-boundary (" nt " ≠ "want"); FEATURE_NAMES frozen; TEXT_ONLY_CEILING for every prior; render uses Figure/Agg (pyplot monkeypatched to raise); sheet mismatch raises; gbdt hand arithmetic + golden; runner feature-name mismatch raises; adapters `card_discrepancies == []`.

