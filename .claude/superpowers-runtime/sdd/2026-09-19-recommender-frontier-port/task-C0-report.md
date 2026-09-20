# Task C0 — merge Peter's v4 training code into `nathan_dev`

Repo: `/lustre/orion/fus187/scratch/nchen/FusionAIHub`, branch `nathan_dev`.
Merged: `peter/dev-peter` @ `35cfed1` into `61fc1dc`. Merge base: `f71acd4`.

## Commits

| sha | subject |
|---|---|
| `bd0ea9e` | merge peter/dev-peter 35cfed1: v4 codec stack, live-frame weighting, actuator_dim inference (parents `61fc1dc` `35cfed1`) |
| `b6b86be` | ignite: test that the pinned v4 codecs and mskfull dynamics checkpoint load |

Nothing pushed. `nathan_fm`, `dev-nathan`, `main` untouched. Peter's clone under
`proj-shared/ps9551/` untouched; his read-only worktree was read, never written.
The uncommitted `.claude/superpowers/plans/…` edit was left unstaged.

## Step 1 — baseline failure list (before, on `61fc1dc`)

```
FAILED tests/shot_design/test_ignite.py::test_codecs_expose_the_encode_then_quantize_contract
FAILED tests/shot_design/test_mcp.py::test_the_project_mcp_config_points_at_this_server
2 failed, 2147 passed, 25 skipped, 4 warnings in 370.86s (0:06:10)
```

## Step 2 — the failing test

`tests/ignite/test_v4_assets_load.py`, written verbatim from the brief. On `61fc1dc`:
**7 failed / 9 passed** (6 codec families + the mskfull dynamics case).

## Step 3 — conflicts and how each was resolved

Exactly the seven files the brief predicted conflicted.

### `src/tokamak_foundation_model/ignite/dynamics_config.py` — 1 conflict
- **Ours wins**: C1's `modalities_from_manifest` kept verbatim (rule 3).
- **Peter's wins, auto-merged**: `ModalitySpec("mirnov", "spectro", 192, 1000)` in
  `FROZEN_MODALITIES` → 15 modalities, 1209 tokens/frame.
- **Both kept**: the new dataclass fields — ours (`ctf_*`, `modality_loss_weight`,
  `actuator_dropout_p`, `sf_*`, `text_embed_dim`, `text_dropout_p`) followed by Peter's
  (`gen_mask_p`, `gen_horizon_*`, `act_cross_attn`, `lag_embed_k`, `grad_ckpt`, …).
  Both sides' defaults are off, so the constructed config is unchanged for either parent's
  invocation. One blank line added between the two groups for readability.

### `src/tokamak_foundation_model/ignite/dynamics.py` — 2 conflicts
- `__init__`: **both** submodules constructed — Peter's `act_cross` then our `text_embed`.
  Both are `None` unless their flag is set, so the parameter set is bit-identical to each
  parent under that parent's config.
- `encode()`: the one genuine judgment call. Peter's `act_cross` attention and our
  `actuator_dropout_p` / `drop_actuators` CFG path both touch the actuator pathway. Resolved
  so the cross-attention pathway obeys the **same** conditioning drop as the additive one:

  ```python
  a = self.act_embed(actuators)
  keep = None
  if drop_actuators:
      a = torch.zeros_like(a)
  else:
      p = getattr(self.cfg, "actuator_dropout_p", 0.0)
      if self.training and p > 0.0:
          keep = (torch.rand((B, 1, 1), device=a.device) >= p).to(a.dtype)
          a = a * keep
  x = x + a.unsqueeze(2)
  if self.act_cross is not None and not drop_actuators:
      c = self.act_cross(x, actuators)
      if keep is not None:
          c = c * keep.unsqueeze(-1)
      x = x + c
  ```

  Rationale: if `act_cross` ignored the drop, the "unconditional" CFG branch would still see
  the full control program and CFG would be silently broken. With our flag at its default 0.0
  and `drop_actuators=False` this is **bit-identical to Peter's code**; with his
  `act_cross_attn=0` it is bit-identical to ours.
  The text path's inner mask was renamed `keep` → `keep_t` so it cannot shadow the actuator one.

### `src/tokamak_foundation_model/ignite/maskgit.py` — 5 conflicts
- `_scheduled_sample_context`: **Peter's** memory-efficient implementation wins wholesale
  (rule 1); our `text=text` was threaded into its `encode()` call.
- `training_loss` signature: **both** kwarg sets kept.
- Masking branch: **ours** kept (sf/ctf gating), with Peter's kwargs carried into the `else`:
  ```python
  masked, mask = self._random_mask(context, generator, ratio_override=mask_ratio,
                                   history_frames=history_frames, gen_mask_p=gen_mask_p)
  ```
- Per-modality loss weighting: the two schemes combine **multiplicatively** —
  `wt = w_m * (1.0 if mod_weights is None else float(mod_weights.get(m.name, 1.0)))`.
  Ours (`modality_loss_weight="uniform"`) gives `w_m == 1.0` by default and Peter's
  `mod_weights` is `None` by default, so the product is 1.0 → bit-identical to both parents.

### `src/tokamak_foundation_model/ignite/eval_dynamics.py` — 5 conflicts
- `load_model`: merged clean. **Peter's** actuator-width inference wins (rule 1) — it reads
  `backbone.act_embed.weight` from the checkpoint, which is what makes the 88-channel mskfull
  checkpoint load — and restores `cfg_act_cross_attn` / `cfg_dropout`. **Ours** kept:
  `cfg_text_embed_dim` / `cfg_text_dropout_p` restoration.
- The other four conflicts are all `render_figure` cosmetics; **both sides carried**:
  Peter's env-gated `_GT_ONLY` / `_SHOW_SKILL` / `_SHOW_R` / `_SHOW_CH` / `_SHOW_TAG`, his
  `_disp()` helper and Okabe-Ito line constants; our `_SLOWTS_SYMBOL` axis labels,
  `fontsize=_p - 4`, `interpolation="antialiased"`, the z-scored-video guard and the honest
  `"not scored (static GT)"` annotation. One over-indented `ax_p.text(...)` in Peter's hunk
  was corrected.

### `src/tokamak_foundation_model/ignite/train_dynamics.py` — 12 conflicts
- `FrameCodeDataset.__init__`: **Peter's** signature and body win (`_narrow`, `stride`,
  `windows_per_shot`, `window_sample`, `_ACT_GLOBAL_DIR`); our `text_embeds=None` appended
  last and our per-shot text vector / coverage print threaded in, placed before his
  `if self.windows_per_shot: self.set_epoch(0)`.
- `__getitem__` / `_collate_frames`: 3-tuple → 4-tuple `(codes, act, present, text)`.
- `train()` signature: **both** kwarg sets kept (ours: `snapshot_every`, `ctf_*`,
  `modality_loss_weight`, `actuator_dropout_p`, `sf_*`, `text_*`; Peter's: `ss_ramp_steps`,
  `gen_mask_p`, `gen_horizon_*`, `act_cross_attn`, `grad_ckpt`, `best_metric`, `lag_embed_k`,
  `balance_presence`, `window_*`, `mod_weight_pow`, `lr_decay_from`, `wd_exclude_norms`,
  `label_smoothing`, `val_windows_per_shot`, `grad_clip`).
- `_val_loss` / `_gen_val_loss`: **Peter's** `no_grad` + `cache_enabled=False` bodies win (the
  bf16 weight-cache fix is load-bearing); our text threaded through as
  `for cv, av, pv, tv in val_batches` + `**({"text": tv.to(device)} if use_text else {})`.
- `_sync_grads`: **ours** wins — the 128 MiB flat-bucket all-reduce (the RCCL-efficiency fix);
  Peter's side was still the per-tensor loop.
- Training loop: **Peter's** `ds.set_epoch(step)` and `_prof` timing kept, with our 4-tuple
  unpack, `text = txt.to(device) if use_text else None`, and a `training_loss` call that
  carries `label_smoothing`, `mod_weights` and the optional `text`.
- Checkpoint payload: **both** `cfg_*` stamp sets.
- argparse: both flag sets.

### `scripts/slurm_frontier/_frontier_common.sh` — 1 conflict
- **Ours** wins (rule 4): the `rccl-net-plugin/1.0` module block with the `RCCL_PLUGIN` /
  `RCCL_ALT_RDZV` escape hatches. Peter's side still had the old commented-out hand-built
  plugin line.
- **Peter's new env var brought over**: the `OFI_PREFIX` guarded opt-in, as a new block placed
  **after** the `RCCL_PLUGIN` `if/else … fi` (an earlier attempt had it inside the `else`
  branch, where it would only have applied with `RCCL_PLUGIN=0` — caught and fixed).
- `git diff 61fc1dc HEAD` on this file removes **zero** lines: our version is a strict subset.
- `bash -n` clean.

### `scripts/slurm_frontier/train_dynamics.sh` — 1 conflict
- Only the header's env-override comment list actually conflicted; merged both lists.
- **Ours** kept: `source scripts/slurm_frontier/_frontier_settings.sh` (our branch's
  self-contained env file — it does module loads, direct `.pixi/envs/frontier` PATH activation,
  `FLASH_ATTENTION_TRITON_AMD_ENABLE`, and its own copy of the RCCL plugin block) rather than
  Peter's `_frontier_common.sh`, per rule 4. Also kept: the `TEXT_EMBED_*` together-or-neither
  gate and the `EXTRA+=(--text_*)` block.
- **Peter's** LOCKED PRODUCTION CONFIG block and all his new flags merged in
  (`--grad_ckpt`, `--best_metric`, `--gen_horizon_*`, `--act_cross_attn`, `--lag_embed_k`,
  `--window_*`, `--mod_weight_pow`, `--lr_decay_from`, `--wd_exclude_norms`,
  `--label_smoothing`, `--val_windows_per_shot`, `--grad_clip`), including his production
  defaults (`d1024xL16`, `frame_codes_noinorm`, `LR=1e-3`, `MIN_LR_RATIO=1.0`, …), which
  supersede our older `d512xL8` defaults — rule 1, he owns the production config.
- `bash -n` clean.

## Step 4 — verification

`tests/ignite/test_v4_assets_load.py` after the merge:

```
16 passed, 1 warning in 8.04s
```

Full suite after (`pytest tests/ignite tests/shot_design -q -p no:cacheprovider
--ignore=tests/ignite/test_train_codec.py -rf`):

```
FAILED tests/shot_design/test_ignite.py::test_codecs_expose_the_encode_then_quantize_contract
FAILED tests/shot_design/test_mcp.py::test_the_project_mcp_config_points_at_this_server
2 failed, 2249 passed, 25 skipped, 4 warnings in 377.30s (0:06:17)
```

**Before/after diff: identical failure set. No regressions. +102 passing tests.**
Both remaining failures are the two the brief predicted:
`test_codecs_expose_the_encode_then_quantize_contract` still asserts v2's 12 encodable names
and needs `mirnov` (C2's job — the test was **not** edited), and the MCP-config test was
already failing at baseline.

`tests/labeler/test_labels_layout_integration.py` was never run. `tests/ignite/test_train_codec.py`
was ignored per the brief.

### One extra fix folded into the merge commit

`tests/ignite/test_phaseb_frame_layout.py` failed after the merge
(`test_frame_layout_totals_1017`, `test_all_four_codec_families_present`). Investigated by
running the same test file against **Peter's own read-only worktree** with `PYTHONPATH=$WT/src`:
his tree also yields 15 modalities / 1209 tokens / 5 spectro families with that identical test
file, i.e. these are **stale v2 assertions he did not update**, not a regression from my
resolution. Updated to the v4 numbers (1017→1209, 14→15, 4→5 spectro), with a comment recording
that the v2 numbers describe the superseded table. File now passes 10/10. This is the only
non-conflict edit in the merge commit, and it is what keeps `bisect` green across it.

## Deliberately dropped from Peter's side

- Peter's `_frontier_common.sh` as the env file for `train_dynamics.sh` — our
  `_frontier_settings.sh` is kept instead (rule 4). Functionally equivalent plus flash-attn env.
- Peter's `d512L8` → his newer `d1024L16` defaults were *taken*; our older `d512L8` defaults
  were dropped (rule 1). Anyone re-running our old arm must now pass `DEPTH`/`D_MODEL`/`LR`
  explicitly.
- Peter's per-tensor `_sync_grads` loop (superseded by our bucketing fix).
- Nothing else. No hunk was judged undecidable.

## Concerns

1. **`OFI_PREFIX` reaches only `_frontier_common.sh`.** Our production wrappers source
   `_frontier_settings.sh`, which was not a conflicting file and so does not carry Peter's
   new escape hatch. Flagging rather than silently widening the merge; it is a two-line,
   no-op-when-unset block if someone wants parity.
2. **Peter's LOCKED PRODUCTION CONFIG comment in `train_dynamics.sh` is self-inconsistent with
   his own v4 change**: it says "14 modalities, mirnov EXCLUDED" and "frame goes 1593 → 1017
   tokens", while the `FROZEN_MODALITIES` table he shipped in the same commit is 15 modalities
   / 1209 tokens. Comment text only — left as-is, but it will mislead the next reader, and the
   `STEPS=55399` budget derived from the old window count is now approximate.
3. **Pre-existing latent `NameError` in `eval_dynamics.render_figure`**: the `"psrow"` branch
   references an undefined `_spec_scale` (line 973). Present identically on **both** parents
   (ours line 817, Peter's line 928), so not a merge artifact — but it is live code, reachable
   whenever a `psrow` row is laid out. Out of scope for C0; worth a ticket.
4. **`ruff` is not installed in the `shot-design-frontier` env**, so no lint gate was run.
   Substituted: `ast.parse`, a module-wide undefined-global scan (clean apart from item 3),
   `bash -n` on both shell scripts, imports of all five ignite modules, and the full suite.
5. The merge commit was **amended** once (to fold in the frame-layout fix), so its sha is
   `bd0ea9e`, not the `7e2ade0` produced by the first attempt.

## Fix round 1 (controller-applied, commit aa61f24)

**Finding addressed (Important, review-C0.md):** `maskgit.py` CTF branch ignored the caller's
pinned `mask_ratio` / `history_frames` / `gen_mask_p`, so `_gen_val_loss` under `--ctf_frac>0`
or `--sf_frames>0` scored a random CTF/SF window instead of the cold-start conditional.

**Change:** `training_loss` computes `pinned = mask_ratio is not None`; when pinned, the
Self-Forcing rollout arm is skipped (`n_sf = 0`) and the CTF gate is skipped
(`if not pinned and self.cfg.ctf_frac > 0.0`). Neither arm touches the RNG on the pinned path,
so it is a bare `_random_mask(ratio_override, history_frames, gen_mask_p)` stream.

**TDD:** RED — new `tests/ignite/test_pinned_mask_bypasses_curricula.py`, 2 of 3 failed
(`curricula changed a caller-pinned mask layout`; `pinned path consumed extra RNG`); the
regression guard (`ctf_frac=1.0` unpinned still takes CTF) passed before and after.
GREEN — command:
`pixi run --frozen -e shot-design-frontier pytest tests/ignite/test_pinned_mask_bypasses_curricula.py tests/ignite/test_ctf.py tests/ignite/test_selfforce.py tests/ignite/test_phaseb_maskgit.py tests/ignite/test_phaseb_compat.py -q`
→ `31 passed, 1 warning`. The one warning is pre-existing in Peter's
`tests/ignite/test_ctf.py:96` (float() on a requires_grad tensor), untouched by this fix.
New lines ≤ 88 cols (checked with awk; ruff is not installed in the frozen env).
