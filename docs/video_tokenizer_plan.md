# Video Tokenizer — Implementation Plan (Revised)

**Prerequisites:**
- Phase A Extended Stage 2 running stably
- Spectrogram tokenizers (Phase B) complete
- All decided items from `video_tokenizer_design.md` locked

**Camera order:** tangtv first → irtv second

**Amendment 2026-05-06 — tangtv reduced to 2 channels.** Per the
c_stage1_best eval (job 2735419) only filters 4 and 6 carry plasma
data; channels 0–3 and 5 are background / calibration / dim. The
tangtv MovieConfig now uses `channels=2, channels_to_use=[4, 6]`. All
tokenizer / head / trainer / test references switched from 7 to 2
channels. Token count (300 per camera per 50 ms window) is unchanged
because it is set by the spatial-temporal patch grid, not the input
channel count. What shrank: tokenizer + head params 1.55 M → 0.44 M
(−71%); per-token receptive field 7×3×12×12 = 3024 px → 2×3×12×12 =
864 px (compression 11.8× → 3.4×). The previous c_stage1 run dir was
deleted; Phase C will retrain from scratch on the 2-channel config.

**Amendment 2026-05-06 (later) — freeze API generalised.** As part of
Phase B Step 5 (spectrogram integration), the Phase-C-only
`--freeze_backbone_steps` flag was replaced with four independent
warm-start flags in `train_e2e_stage1.py`:
`--freeze_ts_steps / --freeze_video_steps / --freeze_spectro_steps /
--freeze_backbone_steps`. The flags compose freely; the previous
"freeze everything except video for N steps" behaviour now needs
three flags (`--freeze_ts_steps N --freeze_spectro_steps N
--freeze_backbone_steps N`). Actuator tokenizers, which the previous
monolithic freeze also held fixed, are now always trainable.
**Implication for `scripts/slurm/train_c_stage1.sh`:** **deleted
2026-05-06.** Replaced by the combined Stage 1 launcher
`scripts/slurm/train_bc_stage1.sh`, which adds `--use_video tangtv
--use_spectro ece co2 bes` to a single warm-started run from Phase A
best (output dir `runs/bc_stage1/`). The combined launcher uses
`--freeze_ts_steps 5000 --freeze_backbone_steps 5000` so video and
spectrogram modules can settle without perturbing the Phase A-trained
backbone. **`train_c_stage2.sh` deleted 2026-05-06**, replaced by the
combined Stage 2 launcher `scripts/slurm/train_bc_stage2.sh` (uses
`--use_video tangtv --use_spectro ece co2 bes`, inits from BC-Stage 1
best with Phase A fallback, output dir `runs/bc_stage2_delta/`).
`train_e2e_stage2_delta.py` was extended in the same pass with the
`SPECTROGRAM_MODALITIES` registry, `--use_spectro` flag,
`_spectro_loss_gate` / `split_spectro_target_by_step` helpers, and
the MAE-only spectrogram path through `rollout_forward_loss_delta` /
`validate`. `head_weight_l2` was generalised to cover spectrogram
heads via `head.patch_unembed`.

**O10 decided:** Plain MAE for video. cos_sim in ~900k dimensions (120×360×7×3) is meaningless — dominated by bulk brightness, not spatial structure. Revisit only if MAE produces visibly blurry reconstructions with no plasma structure.

**Note on frame count:** Earlier design session (at 50 fps) locked 2 input + 2 target frames. After confirming cameras run at 100 fps (5 native frames per 50ms window, no alignment issues), frame count was upgraded to 3 input + 3 target (t=0, 20, 40ms → t=50, 70, 90ms) for richer temporal signal. This is an intentional change, not drift.

---

## Critical Pre-Implementation Checks

Before any coding, verify these against live code:

- [x] **Verify token budget** against live DiagnosticConfig/ActuatorConfig in `train_e2e_stage1.py`. Confirm ~398 total before quoting.
- [x] **Check existing `_load_movie_raw`** (`data_loader.py:1227-1379`). It already does trilinear resampling from raw to target resolution.
- [x] ~~**MOVIE_CONFIGS override** (per-instance, not class-level)~~ **Superseded 2026-05-06.** All e2e training is being retrained from scratch on the 2-channel tangtv config, so the channel-selection change was committed at the class level (`MOVIE_CONFIGS["tangtv"]` directly). Per-instance override mechanism is no longer needed for this purpose.
- [x] **Frame subsample location:** Implemented via `MovieConfig.n_output_frames` (data_loader applies `torch.linspace(0, n - 1, n_output_frames)` in `__getitem__` after movie processing). `MOVIE_CONFIGS["tangtv"]` sets `n_output_frames=3`.
- [x] **Check `collate_fn`** in the training scripts — handles `[C, T, H, W]` movie tensors via the existing collation path.
- [x] **Check pixel value range** in raw data. Per-batch (B, C) z-score standardisation applied at the trainer level (`standardize_per_bc` in `train_video_ae.py` and `train_e2e_stage2_delta.py`); preprocessing-stats regen for video deferred and not needed.
- [x] **Checkpoint loading:** explicit `load_state_dict_explicit` (`src/.../e2e/checkpoint.py`) is used in the e2e trainers; raises on unexpected keys, allows declared-missing prefixes.

---

## Step 0: Data Inspection (~2 hours)

Before any code. Can do during Phase A/B training downtime.

**Tasks:**
- [ ] Load 5–10 representative shots with tangtv data from HDF5
- [ ] Visualize raw frames at full resolution (240×720) and after 2× downsample (120×360)
- [ ] Measure spatial scale of physics features (ELM filaments, detachment fronts, MHD activity) in pixel units at 120×360
- [ ] Confirm 2× downsample preserves the relevant structure
- [ ] Check frame availability: what fraction of shots have tangtv? How many dropped frames?
- [ ] Verify native frame times — confirm spacing and alignment with TS windows
- [ ] Check raw pixel value range and distribution — informs preprocessing and stem initialization
- [ ] Repeat for irtv (513×640 → 256×320) — informational only, implementation comes later

**Output:** Brief notes confirming 2× downsample is sufficient, frame availability statistics, pixel value ranges, example frames saved as reference images for test validation.

---

## Step 1: Data Pipeline (~1 day) — COMPLETE

Built and verified during Phase C and the 2026-05-06 channel reduction.

**Tasks:**
- [x] ~~Override MOVIE_CONFIGS per-instance~~ Superseded — class-level edit in `data_loader.py` (`MOVIE_CONFIGS["tangtv"]` set to `channels=2, channels_to_use=[4, 6], n_output_frames=3, height=120, width=360`). All e2e training is being retrained from scratch on the new 2-channel config so the no-class-level guard is no longer needed.
- [x] ~~Add `PreprocessConfig(method='standardize')` for tangtv~~ Superseded — per-batch standardisation at the trainer level (`standardize_per_bc` in `train_video_ae.py` and `train_e2e_stage2_delta.py`). No video stats regen.
- [x] Frame subsample via `MovieConfig.n_output_frames=3`; `__getitem__` picks 3 evenly spaced frames per half-window. Returns input/target tensors `[2, 3, 120, 360]` plus `tangtv_channel_mask` and `tangtv_valid` indicator.
- [x] `collate_fn` handles video tensor shape (verified by `tests/data/test_video_loading.py::test_collation_video_keys`).
- [x] Video behind `--use_video` opt-in flag in `train_e2e_stage1.py` and `train_e2e_stage2_delta.py`. With empty `--use_video`, Stage 1/2 paths are byte-identical to TS-only (G2/G3 guard tests).
- [x] Checkpoint loading via `load_state_dict_explicit` (`src/.../e2e/checkpoint.py`); raises on unexpected keys, allows declared-missing prefixes. No `strict=False`.
- [x] Unit test: `test_n_output_frames_picks_endpoints_and_centre` — frame indices [0, 2, 4] of 5 native frames.
- [x] Unit test: output shape `[2, 3, 120, 360]` (`test_sample_present_shapes_and_keys`, post-2026-05-06).
- [x] Unit test: validity mask False for shots without tangtv (`test_sample_empty_shapes_and_keys`).
- [ ] Benchmark: measure read throughput at batch 128 with 16 workers — not formalised as a benchmark step; observed in production training runs (Phase C Stage 2 with video) without GPU starvation.

**Note:** Every TS window has native video frames available (native frame spacing matches TS stride). No even/odd window distinction. No zero-tensor fallback for stride mismatch.

---

## Step 2: §5.4 Tests (~1 day) — COMPLETE (tests adapted to tube-patch)

Tests live in `tests/e2e/test_video_tokenizer.py` and pass for the
2-channel tube-patch tokenizer (8 tests; the GPU memory-gate is
skipped without CUDA). The contract is `(B, 2, 3, 120, 360) → (B, 300,
256)` — 300 spatiotemporal tube-patches, **not** 16 Perceiver-pool
queries (the Perceiver-pool design described below in Step 3 was
abandoned per `project_phase_c_video_design.md` after three
plateaued iterations).

**File:** `tests/e2e/test_video_tokenizer.py`

**Test 1 — Shape contract:**
```python
def test_tokenizer_output_shape():
    # tangtv: [B, 2, 3, 120, 360] → [B, 16, 256]
    # Verify output is exactly (batch, n_queries, d_model)
```

**Test 2 — Spatial selectivity (stem test):**
```python
def test_spatial_selectivity():
    # Bright square in one corner vs black frame
    # cos_sim(bright_corner, black) < 0.9
    # Tests that the stem extracts spatially distinct features
```

**Test 3 — Motion detection (Perceiver test):**
```python
def test_motion_detection():
    # Static: same frame repeated three times
    # Moving: object shifted across frame 0, 1, 2
    # cos_sim(static_tokens, moving_tokens) < 0.95
    # Tests that joint space×time Perceiver preserves temporal info
```

**Test 4 — Reconstruction fidelity (output head test):**
```python
def test_reconstruction_fidelity():
    # Forward pass through tokenizer + output head
    # Reconstruction MAE < threshold on synthetic patterns
    # Tests the full encode-decode pipeline at 120×360
```

**Test 5 — Memory (OOM gate):**
```python
def test_full_size_forward_no_oom():
    # batch=128, tangtv [B, 2, 3, 120, 360]
    # Full forward + backward pass
    # Must complete without OOM on A100 40GB
```

**Test 6 — Missing camera token:**
```python
def test_missing_camera_produces_learned_token():
    # Input with mask=False
    # Output should be the learned missing-camera token, NOT zeros
    # Distinct from all-black-frame tokens
```

**Test 7 — Modality embedding distinctness (self-contained, no irtv needed):**
```python
def test_modality_embeddings_distinct():
    # Two tangtv tokenizer instances with independently-initialized modality_emb
    # Same input through both → tokens should differ
    # cos_sim < 0.99
    # Tests that modality embedding actually affects output
    # (Full tangtv vs irtv distinctness tested in Step 7)
```

---

## Step 3: Video Tokenizer Module (~2 days) — SUPERSEDED by tube-patch

> **Status (2026-05-06):** the Perceiver-pool design described below was
> abandoned during Phase C. The tube-patch tokenizer that actually
> shipped lives at `src/tokamak_foundation_model/e2e/tokenizers/video.py`
> (`VideoTokenizer`) and the inverse `VideoOutputHead` lives at
> `src/.../e2e/output_heads.py`. Both are implemented, tested
> (`tests/e2e/test_video_tokenizer.py`), and integrated with the e2e
> trainers. The original Perceiver-pool implementation plan in this
> section is kept here only as a reference to the design history; do
> not re-implement from it. See `project_phase_c_video_design.md` for
> the rationale (bounded global tokens cannot encode unbounded local
> structure → switched to local 3D conv patches).

**File:** `src/tokamak_foundation_model/e2e/video_tokenizer.py`

**Architecture (pre-norm, matching backbone convention):**

```python
class VideoTokenizer(nn.Module):
    def __init__(self, n_channels=7, n_frames=3, n_queries=16,
                 d_stem=128, d_model=256,
                 spatial_size=(120, 360)):  # post-downsample
        # Stem: 2-layer stride-2 cascade
        # Conv → Norm → GELU (matching backbone pre-norm convention)
        self.stem = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, d_stem, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, d_stem),
            nn.GELU(),
        )

        # Feature map sizes after stem
        h_out = spatial_size[0] // 4  # e.g. 120 → 30
        w_out = spatial_size[1] // 4  # e.g. 360 → 90
        n_patches = h_out * w_out     # e.g. 2700 per frame

        # Perceiver cross-attention (pre-norm to match backbone)
        self.queries = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.1)
        #                                                              ^^^
        # std=0.1, NOT 0.02 — at 0.02 dot products → ~0 → uniform softmax
        # → all queries collapse to same output → fails §5.4 Test 3 at init
        self.kv_proj = nn.Linear(d_stem, d_model)
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model)

        # Positional encodings — explicit shapes
        self.spatial_pe = nn.Parameter(
            torch.randn(1, n_patches, d_model) * 0.02)        # [1, H'*W', d_model]
        self.temporal_pe = nn.Parameter(
            torch.randn(1, n_frames, 1, d_model) * 0.002)     # [1, 3, 1, d_model]
            # 10× smaller init than spatial PE

        # Modality embedding
        self.modality_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Learned missing-camera token (NOT zero — distinguishable from black frame)
        self.missing_token = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)

    def forward(self, x, mask=None):
        # x: [B, n_channels, n_frames, H, W]
        B = x.shape[0]

        if mask is not None and not mask.all():
            out = self.missing_token.expand(B, -1, -1).clone()
            if mask.any():
                out[mask] = self._encode(x[mask])
            return out

        return self._encode(x)

    def _encode(self, x):
        B = x.shape[0]

        frame_features = []
        for t in range(self.n_frames):
            feat = self.stem(x[:, :, t])             # [B, d_stem, H', W']
            feat = feat.flatten(2).transpose(1, 2)   # [B, H'*W', d_stem]
            feat = self.kv_proj(feat)                 # [B, H'*W', d_model]
            feat = feat + self.spatial_pe             # [1, H'*W', d_model] broadcast
            feat = feat + self.temporal_pe[:, t]      # [1, 1, d_model] broadcast
            frame_features.append(feat)

        kv = torch.cat(frame_features, dim=1)        # [B, 3*H'*W', d_model]

        # Pre-norm cross-attention
        queries = self.queries.expand(B, -1, -1)
        q = self.q_norm(queries)
        k = v = self.kv_norm(kv)
        attn_out, _ = self.cross_attn(q, k, v)
        tokens = queries + attn_out

        # Pre-norm FFN
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        tokens = tokens + self.modality_emb

        return tokens  # [B, n_queries, d_model]
```

---

## Step 4: Video Output Head (~1 day) — SUPERSEDED by per-patch ConvTranspose3d

> **Status (2026-05-06):** the 16-query reshape + ConvTranspose cascade
> below was abandoned together with the Perceiver-pool tokenizer. The
> shipped head is a single `ConvTranspose3d` whose kernel and stride
> equal the patch size, exactly inverting the tube-patch tokenizer.
> Lives in `src/.../e2e/output_heads.py::VideoOutputHead`. With the
> 2-channel tangtv config, ~221 k params (vs the abandoned ~5 M).

**File:** `src/tokamak_foundation_model/e2e/video_output_head.py`

**Concrete architecture for tangtv (120×360):**

**CRITICAL: No MLP blow-up.** Linear(4096, 24576) = 100M params — 2× the backbone.
Instead: reshape 16 tokens into 4×4 grid, 1×1 conv to reduce channels, ConvTranspose cascade to 32×32, bilinear resize to target aspect ratio. ~5M params.

```python
class VideoOutputHead(nn.Module):
    def __init__(self, n_queries=16, d_model=256, n_channels=7,
                 n_frames=3, output_size=(120, 360)):
        self.output_size = output_size
        self.n_channels = n_channels
        self.n_frames = n_frames

        # Reshape 16 tokens into 4×4 spatial grid
        # Each token → d_model channels at one grid position
        self.grid_h, self.grid_w = 4, 4
        assert self.grid_h * self.grid_w == n_queries, \
            f"grid {self.grid_h}×{self.grid_w} must equal n_queries={n_queries}"
        # If n_queries bumped to 32: use 4×8 grid

        # 1×1 conv to reduce channels: 256 → 128
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(d_model, 128, kernel_size=1),
            nn.GroupNorm(16, 128),
            nn.GELU(),
        )

        # ConvTranspose2d cascade: 4×4 → 8×8 → 16×16 → 32×32
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(16, 128), nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, 32), nn.GELU(),
        )
        # 32×32 → bilinear resize to output_size → final 1×1 conv
        self.final = nn.Conv2d(32, n_channels * n_frames, kernel_size=3, padding=1)
        # Total params: ~5M (vs 100M with MLP)

    def forward(self, tokens):
        B = tokens.shape[0]
        # tokens: [B, 16, 256] → reshape to [B, 256, 4, 4]
        x = tokens.transpose(1, 2).view(B, -1, self.grid_h, self.grid_w)
        x = self.channel_reduce(x)                     # [B, 128, 4, 4]
        x = self.decoder(x)                             # [B, 32, 32, 32]
        x = F.interpolate(x, size=self.output_size,
                          mode='bilinear', align_corners=False)  # [B, 32, 120, 360]
        x = self.final(x)                               # [B, 21, 120, 360]
        return x.view(B, self.n_frames, self.n_channels, *self.output_size)
```

**For irtv (256×320):** Same architecture, different `output_size`. The 4×4 grid + bilinear resize handles any aspect ratio.

**Loss:** Plain MAE at full preprocessed resolution. Per-pixel, per-channel, per-frame. Masked for missing cameras.

---

## Step 5: Wire into E2EFoundationModel (~1–2 days) — COMPLETE

All checkboxes below ticked; Stage 1 + Stage 2 trainers integrate the
video kind cleanly. Spectrogram integration (Phase B) shipped in the
same code path on 2026-05-06; see `docs/spectrogram_tokenizer_plan.md`
§"Step 5" / §"Stage 2 trainer integration" for parallel coverage.

**Approach:** Extend DiagnosticConfig with video fields, add `kind="video"` branch.

```python
@dataclass
class DiagnosticConfig:
    name: str
    kind: str = "slow_ts"  # "slow_ts", "fast_ts", "video"
    # video-specific:
    n_frames: int = 3
    height: int = 0
    width: int = 0
    n_queries: int = 16
```

**Token ordering (load-bearing for rollout):**
Video tokens MUST sit in the diagnostic prefix (`out_tokens[:, :self.n_diag_tokens]`) because `rollout.py:149` slices this contiguous prefix for propagation.

```
[slow_ts_tokens | fast_ts_tokens | video_tokens | actuator_tokens]
 ←──────── n_diag_tokens ────────→
```

**Tasks:**
- [x] Extend DiagnosticConfig, add `kind="video"` dispatch in `__init__` and `n_tokens()`
- [x] Video tokenizer/head in `diag_tokenizers` / `diag_heads` ModuleDicts
- [x] Update `token_layout` / `TokenSlice` — video in diagnostic prefix, before actuators (verified by `test_video_tokens_in_diagnostic_prefix`)
- [x] Update `n_diag_tokens` to include video
- [x] `--use_video` flag — disabled by default, Stage 1 resumes unaffected (verified by `test_no_video_state_dict_keys_identical` and `test_no_video_forward_bitwise_identical` G2/G3 guards)
- [x] Checkpoint loading: `load_state_dict_explicit` (allows declared-missing prefixes, raises on unexpected keys) — verified by `test_load_old_checkpoint_into_video_model_succeeds` and `test_load_with_unexpected_key_raises`
- [x] Delete `lengths_*.pt` when window params change — handled by per-run-dir cache files; documented in `project_chunk_cache_bug.md` memory and the spectrogram plan's prerequisites
- [x] Video loss = plain MAE, excluded when `tangtv_valid=0` (Phase C lock per `project_phase_c_video_design.md`)

**Tests:**
- [x] `tests/e2e/test_video_integration.py`: 5 integration tests (G1–G5) all pass
- [x] `tests/e2e/test_rollout.py` covers token-prefix propagation; `test_video_tokens_in_diagnostic_prefix` covers the video specific case
- [x] All existing TS-only tests pass — guarded by G2 (state_dict identity) and G3 (forward bitwise identity) tests
- [x] TS-only checkpoint loads into TS+video model — verified by G4 test

---

## Step 6: Train tangtv — RESET 2026-05-06, NOW JOINT WITH PHASE B

> **Status (2026-05-06):**
> - The prior C-Stage 1 run (`runs/c_stage1`) was deleted in preparation
>   for a clean retrain on the 2-channel (ch4 + ch6) tangtv config.
> - Phase C is no longer trained as a standalone stage — the previous
>   `train_c_stage1.sh` / `train_c_stage2.sh` launchers were replaced
>   with combined Phase B + Phase C launchers
>   (`train_bc_stage1.sh` / `train_bc_stage2.sh`) that train video
>   alongside ECE / CO2 / BES spectrograms in one run.
> - All freeze references below should be read through the new
>   four-flag API: `--freeze_ts_steps`, `--freeze_video_steps`,
>   `--freeze_spectro_steps`, `--freeze_backbone_steps`. Each is
>   independent; the pre-refactor "freeze everything except video"
>   behaviour now requires three flags simultaneously.

**Combined BC training sequence (replaces standalone Phase C):**

**BC-Stage 1** (`scripts/slurm/train_bc_stage1.sh`): single-step
training of TS + tangtv + ECE/CO2/BES spectrograms.
- Init from Phase A best (`runs/e2e_stage1/e2e_stage1_best.pt`),
  snapshotted at job start. Video and spectrogram tokenizer + head
  keys are declared in `allowed_missing_prefixes`.
- Warm-start freeze: `--freeze_ts_steps 5000 --freeze_backbone_steps 5000`.
  Video and spectrogram modules train freely; TS modules and the
  backbone are held fixed for the first 5 k steps so the new
  modalities can settle without perturbing the Phase A-trained TS
  backbone. Actuator tokenizers are always trainable in this API
  (tiny modules, no observed regressions).
- Output dir: `runs/bc_stage1/`.
- Monitor: tangtv + spectrogram MAE decreasing per modality, TS
  metrics within 5% of pre-spectro baseline.

**BC-Stage 2b** (`scripts/slurm/train_bc_stage2.sh`): displacement
loss curriculum (K=1 → 10), full-backprop.
- Init from BC-Stage 1 best (`runs/bc_stage1/e2e_stage1_best.pt`),
  fallback to Phase A best.
- TS uses standard `α·MAE + β·(1−cos) + γ·|log mag|` (1.0 / 0.3 / 0.1).
- Video and spectrogram loss = plain MAE (cosine + magnitude
  meaningless in pixel space; deferred for spectrograms per Open
  Decision #3 in the spectrogram plan).
- Output dir: `runs/bc_stage2_delta/`.
- Monitor: TS direction_cos stable, video / spectrogram MAE
  decreasing.

**BC-Extended Stage 2:** K=10 → 80 curriculum (not yet wired with
spectrograms — `train_e2e_stage2_extended.py` still needs the same
`--use_spectro` extension that Stage 2b got on 2026-05-06).

**Gates (joint):**
- tangtv passes all §5.4 tests (already green for the 2-channel config).
- TS metrics do not degrade > 5%.
- BC-Stage 2 (delta): visual correlation between tangtv and filterscope
  edge-instability signals.

---

## Step 7: Add irtv (~2 days, after tangtv validated)

- [ ] Second VideoTokenizer with `spatial_size=(256, 320)`, init grid 8×10
- [ ] Second VideoOutputHead with `output_size=(256, 320)`
- [ ] Separate modality embedding and missing-camera token
- [ ] Token count: ~398 + 16 + 16 = ~430 (verify against live code)
- [ ] §5.4 tests for irtv shapes
- [ ] OOM test at batch 128 with both cameras — drop to 64 if needed
- [ ] Repeat BC-Stage 1 / BC-Stage 2 training with both cameras (no
      separate Phase C path post 2026-05-06; irtv joins the joint
      TS+video+spectrogram run)

---

## Timeline

```
Pre-checks:    Verify fps, tokens, collate, pixels    ~2 hours
Step 0:        Data inspection                         ~2 hours  (Phase A/B downtime)
Step 1:        Data pipeline                           ~1 day
Step 2:        Tests                                   ~1 day
Step 3:        Tokenizer module                        ~2 days
Step 4:        Output head                             ~1 day
Step 5:        Model integration                       ~1-2 days
Step 6:        Training (tangtv)                       ~ongoing
Step 7:        Add irtv                                ~2 days
                                                       ─────────
Total:                                                 ~9 days coding + training
```

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| 16 queries insufficient | Video adds no information | Config param, bump to 32 |
| Token→grid can't reconstruct 120×360 | Weak gradients | Different init grid; skip connections from stem |
| W-axis blur from asymmetric resize | Spatial detail lost along width | Swap 4×4 init grid for 2×8 to better match 1:3 aspect |
| Video degrades TS metrics | Phase A regressed | Freeze backbone first 5K steps; freeze TS components if needed |
| OOM with both cameras | Batch reduction | Drop to batch 64; measure before adding irtv |
| tangtv mostly missing | Too few samples | Check availability in Step 0 |
| Double resampling | Blurry inputs | Per-instance MOVIE_CONFIGS override (not class-level) |
| Checkpoint break | Training interrupted | `--use_video` opt-in, explicit key check on load |
| Raw pixel range instability | NaN at init | Standardize preprocessing |
| collate_fn incompatible | Dataloader crash | Verify in pre-checks |
| Query init too small | All queries collapse at init | std=0.1 for queries (not 0.02) |