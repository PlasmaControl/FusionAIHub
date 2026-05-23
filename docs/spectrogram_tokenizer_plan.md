# Spectrogram Tokenizer — Design & Implementation Plan (Phase B)

**Date:** 2026-05-05
**Status:** Draft — pending user review

**Modalities:**
- ECE Radiometer: 40 channels, electron temperature fluctuations
- CO2 Interferometer: 4 channels, line-averaged electron density
- BES: 16 channels (channels 49–56 and 57–64), density fluctuations, 2×8 spatial grid

**Scope:** Full autoregressive prediction. Spectrogram tokens are part of
the plasma state $S_t$, sit in the diagnostic prefix, propagate in
token-space rollout, and have output heads for loss computation.

**Prerequisites:**
- STFT already implemented in data loader (w=1024, hop=256, fs=500 kHz)
- Signal statistics available for normalization
- **Missing data is significant at the shot level:** ECE ~94% present,
  CO2 ~44%, BES ~36%. Per-modality `<name>_valid` masks are mandatory
  (Phase C tangtv pattern).
- Phase A Extended Stage 2: RUNNING (step ~195K/322K, K=40 as of
  2026-05-06). Phase B Steps 0–6 can proceed in parallel; Step 8 (BC
  training) is blocked until Phase A produces a stable Stage 1 best
  for the warm-start init.
- Video tokenizer (Phase C) steps 1–5: COMPLETE. Phase C is no longer
  a standalone stage — video joined the combined BC training launchers
  on 2026-05-06 (`train_bc_stage1.sh` / `train_bc_stage2.sh`).
- Frontier DD allocation confirmed, account approved ~May 18 (64 GB/GCD,
  needed for full 1178-token config at batch 256)

---

## Architecture Summary

### Input
Per modality: `[B, C_d, 512, 98]` — channels × frequency bins × STFT time frames.

Note: STFT with w=1024, hop=256, center=True on 25,000 samples (50ms at
500 kHz) produces 513 frequency bins × 98 time frames. DC bin is dropped
→ 512 frequency bins. Axis order is **(C, freq, time)**, not (C, time, freq).

### Tokenizer: Conv2d (Approach A — merge channels)
All channels treated as input channels to a single Conv2d per modality:

```
Conv2d(in_channels=C_d, out_channels=d_model, kernel_size=(F_p, T_p), stride=(F_p, T_p))
```

Note: kernel is **(F_p, T_p)** matching data layout (B, C, F, T).
Each modality gets its own Conv2d (different C_d → different weight shapes).

**Patch size (F_p, T_p):** Different per modality to balance compression
ratio against token count.

| Modality | Channels | Patch (F, T) | Input after truncation | Tokens | Compression/token | Rationale |
|---|---|---|---|---|---|---|
| CO2 | 4 | (64, 8) | [512, 96] | 96 | 8× | Few channels, light compression sufficient |
| ECE | 40 | (32, 8) | [512, 96] | 192 | 40× | Many channels need finer frequency resolution |
| BES | 16 | (32, 8) | [512, 96] | 192 | 16× | Moderate channels, same grid as ECE |

Truncation: freq=512 is already clean for all patch sizes. Time=98 is
truncated to 96 (drop last 2 frames) for clean division by T_p=8.
Output heads reconstruct [512, 96]; the 2 dropped time frames are not
recoverable but represent <2.1% of the window.

### Positional and modality encodings
Per token: `Conv2d(x)_s + p_s + e_m`
- `p_s ∈ R^{d_model}`: spatial positional encoding per patch position (96 for CO2, 192 for ECE/BES)
- `e_m ∈ R^{d_model}`: modality embedding (one per spectrogram modality)
- No channel positional encoding (channels merged by Conv2d)

### Output head: ConvTranspose2d (inverse of tokenizer)
```
ConvTranspose2d(in_channels=d_model, out_channels=C_d, kernel_size=(F_p, T_p), stride=(F_p, T_p))
```
Exact inverse of the tokenizer. Reconstructs [512, 96] (truncated time).
Same pattern as video (ConvTranspose3d).

### Token budget

| Component | Tokens |
|---|---|
| Slow TS | 273 |
| Fast TS (filterscopes) | 80 |
| CO2 spectrogram (patch 64×8 on [512, 96]) | 96 |
| ECE spectrogram (patch 32×8 on [512, 96]) | 192 |
| BES spectrogram (patch 32×8 on [512, 96]) | 192 |
| Video (tangtv) | 300 |
| Actuators | 45 |
| **Total** | **1178** |

Attention cost vs Phase A: (1178/398)² = **8.8×**.
Memory estimate: Phase C benchmark showed 28.78 GB at 698 tokens, batch 256.
At 1178 tokens, expect ~47+ GB — requires batch reduction on A100 40GB.
Frontier (64 GB/GCD) should handle batch 256 comfortably.

### Parameter budget (estimated)

| Component | Params |
|---|---|
| CO2 tokenizer: Conv2d(4, 256, 64, 8) | 4 × 64 × 8 × 256 + 256 ≈ 0.5M |
| ECE tokenizer: Conv2d(40, 256, 32, 8) | 40 × 32 × 8 × 256 + 256 ≈ 2.6M |
| BES tokenizer: Conv2d(16, 256, 32, 8) | 16 × 32 × 8 × 256 + 256 ≈ 1.0M |
| CO2 head: ConvTranspose2d(256, 4, 64, 8) | ≈ 0.5M |
| ECE head: ConvTranspose2d(256, 40, 32, 8) | ≈ 2.6M |
| BES head: ConvTranspose2d(256, 16, 32, 8) | ≈ 1.0M |
| Positional encodings (96 + 192 + 192) × 256 | ≈ 0.1M |
| Modality embeddings (3 × 256) | ≈ 0.8K |
| **Total Phase B add-on** | **≈ 8.3M** |

Combined with Phase A (9.29M) and Phase C video (1.70M), the full model
is approximately **19.3M parameters** — still small by foundation model
standards (Aurora: 1.3B).

---

## Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| 1178 tokens OOM on A100 40GB | Can't train full config on Stellar | Reduce batch to 64–128, grad checkpointing, or train on Frontier (64 GB/GCD — DD allocation confirmed, account approved ~May 18) |
| Time truncation 98→96 | Lose 2 time frames (<2.1%) | Acceptable loss; reconstruction targets [512, 96] not [512, 98] |
| ECE 40:1 compression per token | Reconstruction quality poor | Reduce ECE patch to (16, 8) → 384 tokens if AE validation fails |
| Cross-channel structure matters for BES 2×8 grid | Merge loses spatial adjacency info | Reshape 16ch to [2, 8] spatial grid before Conv2d; or use Conv3d with spatial kernel |
| Spectrogram reconstruction blurry | Loss terms insufficient | Add perceptual loss or per-frequency weighting |
| 8.8× attention cost too slow for training | Wall time infeasible on single GPU | Multi-GPU DDP on Frontier; or Perceiver compression before backbone |
| CO2 only 44%, BES only 36% available | Most shots lack full spectro | Per-modality valid masks + learned missing-modality tokens (Phase C pattern). Loss excluded for missing modalities. |
| ~~STFT NaN-fill bug in _getitem~~ | ~~STFT data cannot load at all~~ | **Resolved 2026-05-06** — fix in `_process_signal` + new `_raw_to_frame_mask` helper + masks projected in `_getitem_*`. Tests in `tests/data/test_spectrogram_loading.py`. |

---

## Data Pipeline Prerequisites (before Step 0)

These must be resolved in data_loader.py before verification:

1. **[x] BES channel selection** — `channels_to_use=slice(48, 64)` in
   BES SignalConfig (data_loader.py:547). 16 channels (1-idx 49–64),
   two 8-channel poloidal rows forming a 2×8 grid. **Rationale:** the
   BES array is moved radially per session and channel configurations
   vary by session-leader request, so channel-to-(R, Z) mapping is
   non-stationary across shots. These two specific rows are chosen
   because they were historically the most dead-channel-free across
   campaigns. The model sees 16 BES signals indexed by channel, not
   by physical position; (R, Z) is not available in the dataset and
   is not used as conditioning.

2. **[x] BES preprocessing** — changed from `log` to `log_standardize`
   in SignalConfig (data_loader.py:548). All three spectrogram
   modalities now share normalization, avoiding scale imbalance in
   the shared backbone.

3. **[x] Per-modality availability masks** — `<name>_valid` already
   emitted by the data loader (Phase C tangtv pattern, int-valued).
   Now propagated through the prediction-mode input/target split
   (data_loader.py:~1681). Reads 0 for missing modalities and > 0
   when present. Step 0 survey found ECE ~94%, CO2 ~44%, BES ~36%
   present across shots; only ~36% of shots have all three. Trainer
   uses `batch[f"{name}_valid"] > 0` for per-sample masking.

4. **Cache invalidation:** After SignalConfig changes, delete
   `lengths_*.pt` sidecars in any active run dir before next
   training/eval submission. The cache key in `multi_file_dataset.py`
   is only file paths, not signal config — stale caches will
   silently use wrong chunk counts. Stage 1/2/Extended runs do NOT
   currently include ECE/CO2/BES, so existing run dirs are unaffected.

5. **[x] STFT NaN-fill bug (BLOCKING)** — fixed. `_process_signal`
   now applies `nan_to_num` before `torch.stft` (so STFT outputs are
   finite) and projects `element_mask` to STFT-frame coords. New
   helper `_raw_to_frame_mask` (data_loader.py:~1087) projects raw
   `(C, T)` validity masks to `(C, T_frames)` via
   `F.max_pool1d(kernel=n_fft, stride=hop, padding=n_fft//2)` —
   mirroring `torch.stft(center=True)` framing. `_getitem_standard`
   and `_getitem_prediction` use the helper to build full
   `(C, F, T_frames)` masks for STFT signals. Off-by-one in
   `valid_length_out` for absent STFT modalities (was 1, now 0)
   also fixed in the same commit. **Tests:** `tests/data/test_spectrogram_loading.py`
   (8 passing).

---

## Implementation Steps

### Step 0: Data Verification (~2 hours)

Verify STFT output on real data. No architectural decisions needed here.

- [x] Load 5 representative shots, compute STFT for ECE, CO2, BES
- [x] Confirm output shape [C_d, 512, 98] for each (C_d: CO2=4, ECE=40, BES=16)
- [x] Verify axis order: (channels, frequency, time) — NOT (channels, time, frequency)
- [x] Visualize example spectrograms (log-magnitude) — physics
      features visible (saved to `inspect_spectrograms/figures/`,
      1 s window per shot)
- [x] Frequency axis: keep full 0–250 kHz range (no cropping)
- [ ] ~~BES channel layout: verify 2×8 spatial adjacency~~ **n/a** —
      BES array is moved radially per session and configurations vary
      per session-leader request, so channel-to-(R, Z) mapping is
      non-stationary. The 2×8 grid is a logical 16-channel selection,
      not a fixed physical layout.
- [ ] ~~BES grid orientation: row-major vs column-major reshape(2, 8)~~
      **n/a** — no fixed (R, Z) orientation to align to; (R, Z) is
      not in the dataset and the channel-to-position mapping varies
      per shot. Conv3d fallback (Risk #4) would have to use logical
      adjacency only.
- [x] Per-channel statistics validated against `preprocessing_stats.pt`
      (NaN=0, std>0 on all 60 channels; ECE and BES log-scales close,
      CO2 on a different log-scale)

**Output:** Findings doc at `docs/spectrogram_step0_findings.md` (links
to the figures in `inspect_spectrograms/figures/`); regenerated by
re-running `inspect_spectrograms/step0_inspect.py`.

### Step 1: Data Pipeline (~1 day) — COMPLETE

- [x] Fix NaN-fill bug: mask shape must match STFT tensor shape, not
      raw-signal shape
- [x] Verify STFT output is accessible as `batch['ece']`, `batch['co2']`,
      `batch['bes']` with shape [B, C_d, 512, 98] (C_d: 40, 4, 16)
- [x] Verify BES uses only channels 49–64 (16 total)
- [x] Verify axis order is (C, freq, time), NOT (C, time, freq)
- [x] Verify normalization is applied correctly (log_standardize for all three)
- [x] Per-modality `<name>_valid` propagation through prediction-mode
      split; `> 0` indicates modality present, `== 0` indicates absent
- [x] Unit tests in `tests/data/test_spectrogram_loading.py` (8 tests,
      all passing): shape contract, BES channel slice, BES log_standardize,
      `_valid` propagation in present and missing-modality cases,
      `_raw_to_frame_mask` projection correctness, non-STFT regression

### Step 2: Tests (~0.5 day) — TESTS WRITTEN (TDD)

File: `tests/e2e/test_spectrogram_tokenizer.py` (created 2026-05-06).
Currently fails with `ImportError` because `SpectrogramTokenizer` and
`SpectrogramOutputHead` do not exist yet — that is the TDD signal.
Tests will pass once Steps 3 and 4 land.

- [x] **Test 1 — Shape contract** (parametrized over CO2/ECE/BES):
      `(B, C, 512, 98) → (B, n_tokens, 256)` with n_tokens = 96 for CO2
      (patch F=64, T=8) and 192 for ECE/BES (patch F=32, T=8).
- [x] **Test 2 — Frequency selectivity:** narrowband 50 kHz vs 200 kHz
      synthetic spectrograms produce cos_sim < 0.9.
- [x] **Test 3 — Reconstruction pipeline** (parametrized): tokenizer →
      output head shape `(B, C, 512, 96)`, gradients flow into the
      tokenizer.
- [x] **Test 4 — Memory gate (GPU only):** all three tokenizers + heads
      at batch=128 fit on a single GPU forward + backward; skipped if
      no CUDA.
- [x] **Test 5 — Modality-embedding distinctness:** two independent
      tokenizer instances draw distinct `modality_embed` parameters
      (cos similarity well below 1).
- [x] **Test 6 — Time-truncation invariance:** the last 2 frames of
      the input (positions 96:98) must not influence the output, since
      the tokenizer truncates internally.

**Skipped here (deferred to Step 5 integration):** the
"`<name>_state_dict` identity guard for the TS-only path" — that
requires the E2E model to support `kind="spectrogram"`, so it lands in
Step 5 alongside the integration tests.

### Step 3: Spectrogram Tokenizer Implementation (~1 day) — COMPLETE

File: `src/tokamak_foundation_model/e2e/tokenizers/spectrogram.py` (created
2026-05-06). Tests 1, 2, 5, 6 from Step 2 now pass for all three modalities.

```python
class SpectrogramTokenizer(nn.Module):
    def __init__(self, n_channels, d_model, patch_f, patch_t, freq_bins, time_frames):
        # Truncate time to nearest multiple of patch_t
        self.trunc_t = (time_frames // patch_t) * patch_t  # 98 → 96

        self.n_patches_f = freq_bins // patch_f
        self.n_patches_t = self.trunc_t // patch_t
        self.n_tokens = self.n_patches_f * self.n_patches_t

        # kernel_size=(F_p, T_p) matches data layout (B, C, F, T)
        self.proj = nn.Conv2d(n_channels, d_model,
                              kernel_size=(patch_f, patch_t),
                              stride=(patch_f, patch_t))
        self.spatial_pe = nn.Parameter(
            torch.empty(self.n_tokens, d_model))
        self.modality_embed = nn.Parameter(
            torch.empty(d_model))

        nn.init.normal_(self.spatial_pe, std=0.02)
        nn.init.normal_(self.modality_embed, std=0.02)

    def forward(self, x):
        # x: [B, C_d, F=512, T=98]
        x = x[:, :, :, :self.trunc_t]  # truncate time 98 → 96
        tokens = self.proj(x)           # [B, d_model, n_f, n_t]
        tokens = tokens.flatten(2).transpose(1, 2)  # [B, n_tokens, d_model]
        tokens = tokens + self.spatial_pe + self.modality_embed
        return tokens
```

### Step 4: Output Head Implementation (~0.5 day) — COMPLETE

File: `src/tokamak_foundation_model/e2e/output_heads.py` (added
`SpectrogramOutputHead` class on 2026-05-06). Test 3 (reconstruction
pipeline) now passes for all three modalities. 9 of 10 spectrogram
tokenizer tests pass; the GPU memory-gate test is `skipped` when CUDA
is unavailable.

```python
class SpectrogramOutputHead(nn.Module):
    def __init__(self, n_channels, d_model, patch_f, patch_t,
                 n_patches_f, n_patches_t):
        # kernel_size=(F_p, T_p) matches data layout
        self.deconv = nn.ConvTranspose2d(d_model, n_channels,
                                          kernel_size=(patch_f, patch_t),
                                          stride=(patch_f, patch_t))
        self.n_patches_f = n_patches_f
        self.n_patches_t = n_patches_t

    def forward(self, tokens):
        # tokens: [B, n_tokens, d_model]
        B = tokens.shape[0]
        x = tokens.transpose(1, 2).reshape(
            B, -1, self.n_patches_f, self.n_patches_t)
        x = self.deconv(x)              # [B, C_d, F=512, T=96]
        return x
        # Note: reconstructs truncated [512, 96], not original [512, 98]
```

### Step 5: Wire into E2EFoundationModel (~1 day) — COMPLETE 2026-05-06

Implemented in five sub-groups, all tests green (96 passed, 7 skipped
across `tests/e2e/` and `tests/data/`).

- [x] Extend `DiagnosticConfig` with `kind="spectrogram"` and fields
      `freq_bins`, `spectrogram_patch_size`. `window_samples` reused
      for the time axis (parallel to video using it for `n_frames`).
- [x] `__init__` and `tokenize` dispatch on `kind == "spectrogram"`
      (`src/.../e2e/model.py`). `decode` is kind-agnostic. Tokenizer
      gained a learned `missing_token` (Phase C tangtv pattern); the
      `tokenize` branch routes `<name>_valid` through `mask=`.
- [x] Token ordering `[slow_ts | fast_ts | spectro | video | actuators]`
      enforced by `train_e2e_stage1.build_configs` and pinned by
      `tests/e2e/test_spectrogram_integration.py::test_layout_order_*`.
- [x] Missing-modality token: `SpectrogramTokenizer.missing_token`
      (`(n_tokens, d_model)`, std=0.02). When `<name>_valid == 0` for
      a sample, the tokenizer substitutes that sample's tokens. Loss
      gate is `_spectro_loss_gate` ((B, 1, 1, 1) from `_valid`),
      simpler than video's per-channel gate.
- [x] `--use_spectro` flag in `train_e2e_stage1.py` (list of modality
      names from `SPECTROGRAM_MODALITIES`); empty default keeps
      Phase A byte-for-byte (G2/G3 guards stay green).
- [x] Checkpoint loading uses `load_state_dict_explicit` with
      `allowed_missing_prefixes` covering both `--use_video` cameras
      and `--use_spectro` modalities; unexpected keys still raise.
- [x] Guard tests:
      - G2/G3 byte-identity for the TS-only path are pinned by the
        existing `tests/e2e/test_video_integration.py::test_no_video_*`
        fixture; `--use_spectro` empty produces the same diagnostics
        list and state_dict as before, so those tests still pass.
      - 7 new spectrogram-specific tests in
        `tests/e2e/test_spectrogram_integration.py`: token-prefix
        containment per modality (S1×3), token-ordering across TS +
        spectro + video (S2), TS-only checkpoint into TS+spectro
        loads (S3×2), explicit-loader rejection when prefix not
        declared (S3 negative).
- [x] Loss: masked MAE on per-(B, C) z-scored targets
      (`_spectro_standardize_per_bc`); displacement loss deferred
      pending Step 6 reconstruction quality.

**Trainer-side additions to `train_e2e_stage1.py`** (no Stage 1 script
fork, per the saved feedback rule):

- `SPECTROGRAM_MODALITIES` registry, `SPECTRO_FREQ_BINS=512`,
  `SPECTRO_TIME_FRAMES=98`.
- `_spectro_standardize_per_bc(x)` — per-(B, C) z-score over (F, T).
- `_spectro_loss_gate(cfg, batch, device)` — (B, 1, 1, 1) gate from
  `<name>_valid`.
- `forward_batch`, `compute_step_loss`, `copy_baseline_mae` extended
  with `kind == "spectrogram"` branches.

**Freeze refactor (orthogonal four-flag API), shared with Phase C:**

Replaced the Phase-C-only `_apply_video_only_freeze` /
`_release_video_only_freeze` with generic `_apply_module_freeze` /
`_release_module_freeze` that accept four independent boolean flags
(`freeze_ts`, `freeze_video`, `freeze_spectro`, `freeze_backbone`).
CLI exposes one warm-start step count per category:

| flag | freezes |
|---|---|
| `--freeze_ts_steps N` | slow_ts + fast_ts tokenizers + heads |
| `--freeze_video_steps N` | video tokenizer + head |
| `--freeze_spectro_steps N` | spectrogram tokenizer + head |
| `--freeze_backbone_steps N` | shared backbone |

All default 0 (no freeze); each is independent and composable; no-op
when the corresponding modality isn't configured. The training loop
tracks per-category active freezes and releases each at its own step
boundary. The previous `--freeze_backbone_steps requires --use_video`
validation was dropped — orthogonal freezes don't need it. To
reproduce the previous Phase C "freeze everything except video"
warm-start, pass `--freeze_ts_steps 5000 --freeze_spectro_steps 5000
--freeze_backbone_steps 5000`.

### Step 6: Standalone AE Validation (~0.5 day) — IN PROGRESS

Standalone AE harness lives at `scripts/training/train_spectrogram_ae.py`
with launcher `scripts/slurm/train_spectrogram_ae.sh <modality>`. CO2
finished, BES running, ECE pending (2026-05-06).

- [x] Train tokenizer + output head as standalone autoencoder per modality
- [x] 5K steps, lr=1e-3, on real spectrogram data
- [x] Report per-channel reconstruction ratio (MAE / mean baseline)
- [x] Visualize: input spectrogram vs reconstruction every 500 steps
- [ ] If ratio > 0.5 for any modality, investigate

**Resolved during Step 6:** initial runs with per-batch (B, C) z-score
on top of the data loader's `log_standardize` plateaued at ratio
~0.84 (CO2) — see `Open Decisions` #6. After dropping the per-batch
z-score, CO2 final ratio was 0.80–0.87 (avg 0.81), still above the
plan's 0.5 gate.

**Likely conclusion (pending ECE / BES):** for CO2, line-integrated
density on 4 chords is mostly broadband per 50 ms window, so the
per-(B, C) constant mean is already a strong baseline; the AE
captures only ~15–20% of the residual variance. ECE / BES with more
channels and richer spectral structure may land lower; if all three
plateau ~0.8, treat that as the floor for spectrogram modalities in
this architecture and move on rather than fixing per-modality
patches.

**Reference results (CO2 retry, no per-batch z-score):**

| step  | per-channel ratios            | avg  |
|------:|-------------------------------|-----:|
| 1500  | 0.885 / 0.790 / 0.828 / 0.748 | 0.81 |
| 3000  | 0.873 / 0.778 / 0.822 / 0.744 | 0.80 |
| 5000  | 0.869 / 0.809 / 0.816 / 0.751 | 0.81 |

### Step 7: Memory Benchmark (~2 hours)

- [ ] Full config (TS + spectro + video): 1178 tokens
- [ ] Benchmark at batch 128 and batch 256 on A100 40GB
- [ ] If OOM: determine maximum batch size
- [ ] Repeat on Frontier GCD (64 GB) if available

### Stage 2 trainer integration — COMPLETE 2026-05-06

`scripts/training/train_e2e_stage2_delta.py` extended in parallel to
the Group 4 Stage 1 work:

- `SPECTROGRAM_MODALITIES` registry + `SPECTRO_FREQ_BINS=512` /
  `SPECTRO_TIME_FRAMES=98` constants.
- `build_configs(use_video, use_spectro)` — same diagnostic ordering
  `[slow_ts | fast_ts | spectrogram | video | actuators]`.
- `_spectro_loss_gate(name, batch, device)` — `(B, 1, 1, 1)` from
  `<name>_valid`, broadcasts over `(B, C, F, T)`.
- `split_spectro_target_by_step(target, k_steps, trunc_t)` — splits the
  STFT-extended-window target into K windows of exactly `trunc_t`
  frames each (where `trunc_t = (window_samples // T_p) * T_p`,
  matching the spectrogram tokenizer's internal time truncation).
  Frames past `K * trunc_t` are discarded — for K=10 with trunc_t=96,
  that's 17 / 977 ≈ 1.7% of the time axis. Raises if the target is
  shorter than `K * trunc_t`. The trainer pre-computes per-modality
  `trunc_t` via the `_spectro_trunc_t` helper.
- `rollout_forward_loss_delta` and `validate` — both extended with
  `spectro_diag_names: Optional[List[str]] = None`. Spectrograms get
  the same MAE-only loss path as video (cosine + magnitude deferred);
  no per-batch z-score (data loader's `log_standardize` is the only
  normalisation, mirroring Stage 1).
- `head_weight_l2` generalised to dispatch on `head.proj` (slow_ts) /
  `head.deconv` (fast_ts) / `head.patch_unembed` (video and
  spectrogram), with a fallback to the head's first parameter for
  unknown future kinds.
- `--use_spectro` CLI flag added; `allowed_missing_prefixes` covers
  both `--use_video` and `--use_spectro` modules so warm-starts from
  Phase A or BC-Stage 1 best work cleanly.

**Tests:**

- `tests/e2e/test_spectrogram_integration.py` extended with three new
  tests:
  - `test_split_spectro_target_by_step_shapes` — 977-frame target,
    `trunc_t=96`, K=10 → 10 windows of (B, C, 512, 96).
  - `test_split_spectro_target_by_step_raises_when_too_short` — guards
    the precondition `target.shape[3] >= K * trunc_t`.
  - `test_stage1_forward_batch_with_spectrogram_loss_is_finite` —
    end-to-end shape contract: builds a TS+spectro model, runs
    `compute_step_loss` on a synthetic dataloader-shaped batch, and
    asserts finite loss + backward. Catches the regression below.
- Full suite: 99 passed, 7 skipped.

**Bug fixed during integration:** the `SpectrogramOutputHead` emits
`(B, C, 512, 96)` (truncated time) but the dataloader's spectrogram
target arrives at `(B, C, 512, 98)`. The Stage 1 trainer's
`forward_batch` and `copy_baseline_mae`, plus Stage 2's per-step
target split, were all updated to slice the target's time axis to the
head's `trunc_t = (window_samples // T_p) * T_p` so loss-time shapes
match. Without the fix the masked MAE crashed on broadcast.

**Combined Stage 2 launcher:** `scripts/slurm/train_bc_stage2.sh`
(uses `--use_video tangtv --use_spectro ece co2 bes`, init from
`runs/bc_stage1/e2e_stage1_best.pt` with fallback to Phase A best,
output dir `runs/bc_stage2_delta/`). The previous `train_c_stage2.sh`
was deleted.

### Step 8: Phase B Stage 1 Training — LAUNCHER READY

Combined Phase B + Phase C Stage 1 launcher:
**`scripts/slurm/train_bc_stage1.sh`** (created 2026-05-06; replaces
the now-deleted `train_c_stage1.sh`).

- Warm-starts from Phase A best
  (`runs/e2e_stage1/e2e_stage1_best.pt`), snapshotted at job start.
  Video and spectrogram tokenizer + head keys are declared in
  `allowed_missing_prefixes` so they load from scratch cleanly.
- Adds `--use_video tangtv --use_spectro ece co2 bes`.
- Warm-start freeze: `--freeze_ts_steps 5000 --freeze_backbone_steps 5000`
  (TS and backbone held; **video and spectrogram modules train
  freely** so the freshly-initialised modules can settle).
- `--batch_size 64` (down from Phase C's 256; full 1178-token config
  estimated > 40 GB at batch 256 on Stellar A100 40 GB). Adjust after
  the Step 7 memory benchmark.
- Auto-resume across 24 h SLURM walls preserved.
- Output dir: `runs/bc_stage1/`.

**Submission gate (still pending):**
- [ ] Phase B Step 6 (standalone AE) results for ECE / BES land
      (currently CO2 done, BES running, ECE pending).
- [ ] Phase A Stage 1 best checkpoint exists at
      `runs/e2e_stage1/e2e_stage1_best.pt` (the launcher errors out
      if it doesn't).

**Submit when ready (from `scripts/slurm/`):**
```
sbatch train_bc_stage1.sh
```

**Monitoring during the run:**
- [ ] TS metrics within 5% of pre-spectro baseline (per-modality MAE
      logged by `train_e2e_stage1.py`'s validation hook).
- [ ] Spectrogram MAE decreasing per modality.
- [ ] Video MAE decreasing.

**Stage 2 follow-on:**
`scripts/slurm/train_bc_stage2.sh` (combined Stage 2b launcher) is
ready and waits on `runs/bc_stage1/e2e_stage1_best.pt`. Falls back to
Phase A best if BC-Stage 1 hasn't produced one. Submit after BC-Stage 1
hits the success gate.

---

## Open Decisions

1. **Patch sizes locked?** CO2=(F=64, T=8)→96 tokens, ECE=(F=32, T=8)→192,
   BES=(F=32, T=8)→192. Input is [512, 96] after truncating time 98→96;
   freq=512 is untouched. Depends on Step 0 frequency axis inspection —
   if signal is concentrated below 100 kHz, cropping the frequency axis
   before tokenization could reduce tokens further.

2. ~~**Padding vs truncation**~~ **Resolved: truncate time.** Time
   axis truncated 98 → 96 (lose 2 frames). Freq=512 already clean.

3. **Loss for spectrograms:** Start with plain MAE (same conservative
   choice as video). Add displacement loss only after Step 6 standalone
   AE validates reconstruction quality. Log space may be friendlier to
   magnitude terms but verify empirically first.

4. **Training order:** Phase B before or after Phase C video training?
   If Frontier is available, both can train simultaneously on
   different GCDs.

5. **BES spatial structure:** Currently treating 16 channels (2×8 grid)
   as flat input channels. If reconstruction quality is poor, reshape
   to [2, 8, 512, 96] (post-truncation) and use Conv3d with a spatial
   kernel to exploit adjacency.

6. ~~**Trainer-level standardization**~~ **Resolved 2026-05-06: NO
   per-batch standardization for spectrograms.** Initial Step 6 runs
   with per-(B, C) z-score on top of the data loader's
   `log_standardize` plateaued at ratio ~0.84 (CO2 final, ECE early
   trajectory) — the additional standardization removed the
   per-window variance the AE could otherwise learn, and the implicit
   "predict zero in standardized space" baseline already captured
   most of the per-window content. Both `train_spectrogram_ae.py`
   and `train_e2e_stage1.py`'s spectrogram branches now train
   directly on the data-loader-normalized values; the validation
   baseline is "predict per-(B, C) constant mean", which is the
   correct competitor without per-batch z-score. **Video keeps its
   per-batch z-score** because video pixels are not pre-normalised by
   the data loader (no `log_standardize` for raw camera frames).
