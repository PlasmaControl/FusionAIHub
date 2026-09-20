# Review — Task C0: merge `peter/dev-peter` 35cfed1 into `nathan_dev`

Scope: `61fc1dc..b6b86be` (merge `bd0ea9e` + test commit `b6b86be`). Task-scoped gate only.
Verified against the supplied diff plus read-only `git show/diff` scoped to the seven conflict
files (`git diff -U0 35cfed1 bd0ea9e -- <files>`, `git show --cc --name-only bd0ea9e`), an AST
duplicate-definition scan, an argparse parse of the merged trainer, `bash -n`, and
`uvx ruff check --select E501`. The test suite was **not** re-run.

---

### Spec Compliance

**Rule 1 — model construction / checkpoint loading follows Peter**

| File | Verdict | Evidence |
|---|---|---|
| `maskgit.py` | ✅ | `git diff -U0 35cfed1 bd0ea9e` shows **no** hunk inside `_random_mask` — Peter's generation-mode masking, horizon sampling, `protect_gen`, `ratio_override`, `history_frames` are verbatim. `_scheduled_sample_context` is Peter's memory-efficient `masked_logits` + chunked-multinomial + `cache_enabled=False` body wholesale, with only `text=text` threaded into its `encode()` call. `training_loss` carries **both** kwarg sets (`text` ∪ `per_modality/mask_ratio/history_frames/mod_weights/gen_mask_p/label_smoothing`); no side dropped. |
| `dynamics.py` | ✅ | `act_cross` construction, `preserve_rng_state` tied to `cfg.dropout`, `grad_checkpointing` are Peter's. `forward` keeps Peter's shape + our optional `text`. |
| `dynamics_config.py` fields | ✅ | Peter's `gen_mask_p`, `lag_embed_k`, `gen_horizon_alpha/max`, `act_cross_attn` all present alongside our `ctf_*`, `sf_*`, `modality_loss_weight`, `actuator_dropout_p`, `text_*`. `FROZEN_MODALITIES` took Peter's `mirnov` entry (verified at runtime: 15 modalities / 1209 tokens / 5 spectro). |
| `eval_dynamics.load_model` | ✅ | The only diff vs Peter's blob in that function is the two additive `cfg_text_*` restores; Peter's `backbone.act_embed.weight` width inference and the `cfg_act_cross_attn`/`cfg_dropout` restore are untouched. |
| codec loading in `train_dynamics.py` | ✅ | First hunk of `git diff 35cfed1 bd0ea9e -- train_dynamics.py` is at line 30 (the `text_embed` import), next at 767 (`FrameCodeDataset`). Everything between — including `_load_codec` / `load_frozen_codecs` / `resolve_codec_path` — is Peter's byte-for-byte. |

**Rule 2 — our flag-gated features survive and stay off by default** ✅

Runtime-verified in the merged env (`pixi run --frozen -e shot-design-frontier`):

- All five ignite modules import cleanly (`train_dynamics`, `eval_dynamics`, `maskgit`, `dynamics`, `dynamics_config`) — no stale-symbol import.
- `build_arg_parser().parse_args(['--cache_dir','/tmp','--out_dir','/tmp'])` yields
  `snapshot_every=0, ctf_frac=0.0, modality_loss_weight='uniform', actuator_dropout_p=0.0,
  sf_frames=0, text_embed_path=None, text_embed_dim=0, text_dropout_p=0.0, text_key='input'`
  and Peter's `gen_mask_p=None, act_cross_attn=None, grad_ckpt=None, lag_embed_k=None,
  best_metric='masked', pin_train='', label_smoothing=0.0, grad_clip=0.0`. Every flag is off /
  inert at its default.
- **No duplicate `add_argument`** in either `train_dynamics.py` or `eval_dynamics.py`
  (`grep -oP 'add_argument\("--[a-z_0-9]+' | sort | uniq -d` → empty), so argparse cannot raise
  at startup.
- **No duplicate function/method definition** in any of the five modules (AST scan per class
  scope → clean). No "one side of a shared function silently dropped" was found.
- `_GUMBEL_SCALE = float(os.environ.get("IGNITE_MASKGIT_GUMBEL","0"))` present, default 0 → the
  historical greedy reveal is bit-identical (no `rand` drawn).
- `CMAP_DIFF` is **gone** (`dir(eval_dynamics)` → only `CMAP_SPECTRO`, `CMAP_VIDEO`); the
  GT|PRED-only video layout and the `"not scored (static GT)"` honest annotation survive
  (`eval_dynamics.py:1055`), as do `apply_actuator_mode` / `divergence_vs_real` / the
  `--actuator_mode` counterfactual panel and our z-score scale guard.
- Both sides' `render_figure` cosmetics coexist: Peter's `_GT_ONLY`/`_SHOW_SKILL`/`_SHOW_R`/
  `_SHOW_CH`/`_SHOW_TAG`/`_disp()`/Okabe-Ito, and our `_SLOWTS_SYMBOL`, `fontsize=_p-4`,
  `interpolation="antialiased"`, PNG-DPI knob. The over-indented `ax_p.text(...)` from Peter's
  hunk is correctly re-indented under `if not _GT_ONLY:`.
- `_sync_grads` is **ours** (128 MiB flat-bucket all-reduce), Peter's per-tensor loop is gone —
  correct, since it is the RCCL-efficiency fix and not a checkpoint-format decision.
- `frame_layout.py` auto-merged: both our `logits_last` and Peter's `lag_embed` are present
  (`git show --cc` for that path is empty → no manual resolution hidden there).
- `selfforce.rollout_context` already takes `text=`, so the SF path threads correctly.

**Rule 3 — `dynamics_config.py` keeps C1's `modalities_from_manifest` verbatim + Peter's fields** ✅
The function is byte-identical to `61fc1dc`, with the `json` / `Path` imports intact; Peter's new
dataclass fields sit in their own block.

**Rule 4 — Slurm wrappers keep our structure with Peter's new vars added** ✅
- `_frontier_common.sh`: our `rccl-net-plugin/1.0` block with the `RCCL_PLUGIN` / `RCCL_ALT_RDZV`
  escape hatches wins; Peter's `OFI_PREFIX` opt-in is appended **after** the `if/else … fi`, so it
  applies on both branches (the report's own "earlier attempt put it inside the `else`" is fixed
  in the shipped file). `git diff 61fc1dc bd0ea9e` on this file removes zero lines.
- `train_dynamics.sh`: keeps `source scripts/slurm_frontier/_frontier_settings.sh` (verified that
  file carries the same RCCL plugin block plus `FLASH_ATTENTION_TRITON_AMD_ENABLE`), keeps the
  `TEXT_EMBED_*` together-or-neither gate and the `EXTRA+=(--text_*)` block, and takes Peter's
  LOCKED PRODUCTION CONFIG and all his new flags/defaults.
- `bash -n` re-run here on both scripts: **clean**.

**Folded fix — `tests/ignite/test_phaseb_frame_layout.py`** ✅ legitimate, not weakened.
`1017→1209`, `14→15`, `spectro 4→5`; the arithmetic comment `192*5 + 108*2 + 4*7 + 5*1 = 1209`
checks out and `5+2+7+1 = 15` checks out. Independently confirmed the assertions were stale on
Peter's own tree: `35cfed1` ships `ModalitySpec("mirnov","spectro",192,1000)` in
`FROZEN_MODALITIES` while its copy of this test still asserts `1017` / `14` / `4` — so this is
Peter's un-updated test, not a regression from the resolution. Every other assertion in the file
(including our `test_logits_last_matches_full_logits_last_frame`) is unchanged.

**`tests/ignite/test_v4_assets_load.py`** ✅ matches the brief: same `CODECS`/`DYN` paths, same
15-entry `FAMILIES` map, same `skipif`, same parametrized strict-load test, same three assertions
(`step == 3200`, `len(cfg.modalities) == 15`, `act_embed.weight.shape[1] == 88`). Only cosmetic
reformatting of the dict and a blank line after the local imports.

**Commit hygiene** ✅ `merge peter/dev-peter 35cfed1: v4 codec stack, live-frame weighting,
actuator_dim inference` and `ignite: test that the pinned v4 codecs and mskfull dynamics
checkpoint load` — both single-line, no body, no attribution block. `b6b86be` touches exactly one
file. The merge commit's combined diff touches only the seven conflict files + the folded
frame-layout test. Branch is `nathan_dev`; nothing pushed.

---

### Strengths

- The one genuinely hard hunk (`DynamicsBackbone.encode`) is resolved with the right invariant:
  `act_cross` now obeys the same conditioning drop as the additive actuator path, so the CFG
  "unconditional" branch really is unconditional. Under Peter's config (`actuator_dropout_p=0`,
  `drop_actuators` never set) it is bit-identical to his code; under ours (`act_cross_attn=0`) it
  is bit-identical to ours. Parameter set unchanged either way — the mskfull checkpoint still
  loads.
- The `keep` → `keep_t` rename in the text path is a real shadowing bug avoided, not cosmetics.
- The multiplicative `wt = w_m * mod_weights.get(...)` composition of the two per-modality
  weighting schemes is the only resolution that reduces to each parent at its own defaults.
- `_val_loss` / `_gen_val_loss` correctly kept Peter's `no_grad` + `cache_enabled=False` bodies
  (the load-bearing bf16 weight-cache fix) rather than our older autocast-only versions.
- The report's own "deliberately dropped" and "concerns" sections are accurate; every claim I
  spot-checked held, including the `_spec_scale` pre-existence and the `OFI_PREFIX` placement.

---

### Issues

#### Critical
None.

#### Important

**1. `maskgit.py:352` — the CTF branch swallows Peter's `mask_ratio` / `history_frames` /
`gen_mask_p`, which silently invalidates `gen_ce` (and `--best_metric gen`) for any CTF or
self-forcing arm.**

```python
if use_ctf:
    masked, mask = (self._boundary_mask(context, generator, min_boundary=sf_at)
                    if sf_at is not None else self._boundary_mask(context, generator))
else:
    masked, mask = self._random_mask(context, generator, ratio_override=mask_ratio,
                                     history_frames=history_frames, gen_mask_p=gen_mask_p)
```

`train_dynamics._gen_val_loss` (`train_dynamics.py:1499`, call at `:1528`) measures the rollout
condition by passing `mask_ratio=1.0, history_frames=cfg.k0_seed, gen_mask_p=0.0`. With
`--ctf_frac > 0` a fraction `ctf_frac` of validation windows takes the `_boundary_mask` branch
instead and those three arguments are dropped on the floor — the number is still logged as
`gen_ce`, still printed in the `gap` line, and is still the selection criterion under
`--best_metric gen`. `--sf_frames > 0` compounds it: `training_loss` will run an actual
`rollout_context` inside `_gen_val_loss` (and `_val_loss`), making the "fixed validation
protocol" both non-fixed and much more expensive.

Neither flag is on by default, so nothing shipped today is wrong — but this is a new cross-side
interaction created by the merge, and it fails silently. Fix: force the diagnostic path when the
caller pins the mask, e.g. `if use_ctf and mask_ratio is None:` (and gate the SF rollout on the
same condition), or have `_val_loss`/`_gen_val_loss` temporarily zero `cfg.ctf_frac` /
`cfg.sf_frames`. Cheap, and it restores "val is a fixed protocol across arms", which is the stated
reason `gen_mask_p=0.0` is pinned three lines away.

#### Minor

**2. `scripts/slurm_frontier/train_dynamics.sh:57,69` — Peter's LOCKED PRODUCTION CONFIG comment
now contradicts the table the same merge ships.** It states "14 modalities, mirnov EXCLUDED" and
"the frame goes 1593 -> 1017 tokens (4x192 spectro …)", while `DynamicsConfig()` on this commit
returns 15 modalities / 1209 tokens / 5 spectro (verified at runtime). The `STEPS=55399` budget
derived from the old window count is likewise stale. Comment-only, correctly left as Peter's
text, but it is the production launcher and it will mislead. One-line correction is worth doing
in a follow-up (out of scope for a merge-fidelity gate).

**3. `scripts/slurm_frontier/train_dynamics.sh:124` — `PIN_TRAIN`'s inline comment is `PIN_VAL`'s,
copied.** `PIN_TRAIN="${PIN_TRAIN:-}"      # standing example shot: pinned to val, never trained`
— the correct semantics (neighbour stratification, comma-separated, never via `--export`) are in
the block comment two lines above. Peter's defect, carried verbatim.

**4. `eval_dynamics.py:973` — `_spec_scale` is undefined in the `"psrow"` branch (`NameError` when
a power-spectrum row is laid out).** Confirmed identical on **both** parents (`61fc1dc:817`,
`35cfed1:928`), so not a merge artifact and correctly out of scope — but it is live, reachable
code and deserves a ticket, as the report says.

**5. `train_dynamics.py:1722 — `"cfg_dropout": float(cfg.dropout),`` is mis-indented** inside the
checkpoint-payload dict (12 spaces vs the surrounding 27). Syntactically fine. Verified present
identically at `35cfed1:1623` — carried verbatim from Peter, not introduced.

**6. Ruff/88 on new lines is technically violated, but only in line with the file's own style.**
Every merge-introduced (`++`) line is ≤ 99 columns; the longest are
`maskgit.py` `wt = w_m * (1.0 if mod_weights is None else float(mod_weights.get(m.name, 1.0)))`
(96) and the two `_frontier_common.sh` comment lines (94). The whole `ignite` package is written
at ~100 columns and already carries thousands of pre-existing `E501`s against the `line-length =
88` in `pyproject.toml` (the module docstring of `train_dynamics.py` is 94 chars at line 1), so
enforcing 88 on these lines alone would make them inconsistent with their neighbours. No action;
noted so the constraint is not silently assumed satisfied. `uvx ruff check --select E501` was run
here to confirm this is pre-existing, not merge-introduced.

**7. `OFI_PREFIX` does not reach the production dynamics job.** Our `train_dynamics.sh` sources
`_frontier_settings.sh`, which was not a conflict file and therefore did not receive Peter's new
escape hatch (it does carry the RCCL plugin block and `FLASH_ATTENTION_TRITON_AMD_ENABLE` —
verified). This is the correct reading of rule 4 ("keep our structure"; `_frontier_settings.sh`
was out of scope), and the implementer flagged it rather than silently widening the merge. Parity
is a two-line, no-op-when-unset addition if wanted.

---

### Assessment

**Task quality: Approved**

Every conflict rule is satisfied and verified independently of the report: Peter's semantics own
model construction and checkpoint loading (his `train_dynamics.py` lines 30–766 are byte-for-byte,
his `_random_mask` is untouched, his `act_embed` width inference is intact), all of our
flag-gated features survive with defaults that parse to off, and no function, argparse flag, or
import was lost or duplicated. The only substantive finding is a silent flag-interaction
(`--ctf_frac` / `--sf_frames` overriding Peter's pinned-mask diagnostic in `_gen_val_loss`) that
is inert at today's defaults and can be fixed with a one-line guard.
