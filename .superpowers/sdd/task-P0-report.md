# Part 0 — IGNITE v4 only

Worktree: `/scratch/gpfs/nc1514/FusionAIHub-v4only`; branch: `v4-only`; base: `7fcd175`.

## Changes and commits

- P0.1: `model:` is the only model block. Deleted generation switching and origin/generation defaults, required recorded frame origin, simplified model status/manifest handling, and removed the historical G-ENC gate.
- P0.2: Editor window defaults to 2–6 s. Omitted `n_predict` fits both the seed and original checkpoint horizon; explicit oversize requests fail before model loading where possible. The resolved value reaches rollout and HDF5 metadata.
- P0.3: Submission command lives in cluster Paths. Added the supplied single-A100 Stellar script and retained Frontier's wrapper. Removed the generation export; polling remains 15 s.
- P0.4: Spectrogram reduction preserves signed mean z. Reports expose proposed `frac_static` and `frac_static_real`; the collector reads named columns in both older and new tables.
- P0.5: Removed generation override tests and the zero-origin pin. Migrated corpus times, windows, vertices, cache structure, browser fixtures and parity checks to v4. Fixed the two baseline failures; Python design defaults now match the editor. Removed the obsolete bundle frame-code lookup.
- P0.6: Updated available cluster/model/corpus/simulation docs and the obsolete checkpoint example in `programs.md`.

```text
e48424a shot_design: use only the IGNITE v4 model block (P0.1)
b6aca59 shot_design: derive v4 rollout defaults from available frames (P0.2)
7277526 shot_design: submit Simulate through cluster paths (P0.3)
5c0082f shot_design: preserve signed decode means and report both arms (P0.4)
c8ca655 shot_design: migrate caches and test fixtures to v4 frames (P0.5)
1b54163 shot_design: document v4 on Stellar and Frontier (P0.6)
```

Every commit ends with the requested Claude co-author trailer.

## Inventory

```bash
git grep -nE 'model_generations|IGNITE_GENERATION|"v2"' -- src/shot_design scripts/shot_design configs/shot_design tests/shot_design docs dev/paper
```

Output: **empty** (exit 1: no matches).

The wider sweep includes `scripts/slurm_frontier` and finds only the plan's allowed chord names, MiniLM name, prompt/evalset versions, codec proposal and measured-vocabulary note:

```text
configs/shot_design/evalsets/README.md:61:sentences in a future v2.
configs/shot_design/evalsets/README.md:220:recorded here and its own test constant, and report both. A number from v1 and a number from v2
configs/shot_design/flags.yaml:86:  # ne_line (the BCI V2 chord), whose unit configs/signals.yaml records as
configs/shot_design/flags.yaml:93:  # Measured on the 201-shot database once the V2 chord unit was confirmed (2026-09-05): the
configs/shot_design/ignite_modalities.yaml:125:    # Channel order is FusionAIHub's: R0 first, then V1, V2, V3. The UF ("unfiltered") pointnames
configs/shot_design/ignite_modalities.yaml:126:    # are the fast chords that feed co2_density; the SLOW version of the V2 chord is a different
configs/shot_design/ignite_modalities.yaml:128:    staged: {group: co2_density, cols: [r0, v1, v2, v3]}
configs/shot_design/llm.yaml:40:                                  # v2: states the word cap explicitly and targets 40-60 words,
configs/shot_design/paths.frontier.yaml:25:sentence_transformers_model: sentence-transformers/all-MiniLM-L6-v2
configs/shot_design/paths.yaml:22:sentence_transformers_model: sentence-transformers/all-MiniLM-L6-v2
configs/shot_design/signals.yaml:41:  # The corpus address is the V2 chord (channel 2 of `co2`, whose four channels are the DPD
configs/shot_design/signals.yaml:42:  # R0/V1/V2/V3 :DENUF nodes in that order per modalities.yaml), NOT channel 0. It has to be:
configs/shot_design/signals.yaml:46:  ne_line:   {group: co2_density_slow, col: v2, units: m/cm3, tier: raw, stats: [mean, peak], fetch: {kind: mds, tree: bci, node: "\\BCI::DENV2"}, corpus: {group: co2, channels: [2], reduce: first}}
docs/clusters/frontier.md:63:Hugging Face cache (`sentence-transformers/all-MiniLM-L6-v2`) is
docs/models/ignite-rollout-quality-plan.md:393:3. **Tokenizer input (statistics-first codec v2 / bp-line upgrade).** The bp
docs/reference/environment-variables.md:54:| `SHOT_DESIGN_HF_ONLINE` | when unset (the default), Hugging Face calls (the `sentence-transformers/all-MiniLM-L6-v2` embedder) run offline from a pre-populated cache |
scripts/shot_design/census.sbatch:31:# first 300: 37.2 s, 298 openable, 9,514 rows. Its per-group fractions are the ones the plan's V2
scripts/slurm_frontier/_submit_spectro_pair_arms.sh:877:# 8753/8753 shots. 1017 matches v2, which is the only reason the MEASURED 2.9055 no-skill CE
src/shot_design/config.py:75:    sentence_transformers_model: str = "sentence-transformers/all-MiniLM-L6-v2"
src/shot_design/flags/rules.py:88:# The BCI V2 chord is vertical at this major radius (DIII-D CO2 interferometer geometry; V1 is
src/shot_design/flags/rules.py:89:# at 1.48 m, V3 at 2.10 m, R0 is the radial chord). Only V2 is a registry signal today.
src/shot_design/flags/rules.py:94:    """n_e / n_G from the V2 chord.
src/shot_design/shotdb/text.py:1250:EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # beat 6 larger encoders on this operator text (shotsearch)
src/shot_design/shotdb/text.py:1260:EMBED_DIM = 384  # all-MiniLM-L6-v2's width; only used when there is nothing to embed, see below
tests/shot_design/conftest.py:612:    * `co2` 4 chords, V2 (index 2) at 5e13 and the others at 1.0
tests/shot_design/test_build_store.py:515:    """A corpus file whose `co2` holds two chords cannot serve `ne_line` (the V2 chord, channel
tests/shot_design/test_config.py:120:    assert paths.sentence_transformers_model == "sentence-transformers/all-MiniLM-L6-v2"
tests/shot_design/test_corpus_signals.py:92:        ("ne_line", 5.0e13),  # co2's V2 chord, not its first channel
tests/shot_design/test_integration_real.py:169:    if not list(cache.glob("models--sentence-transformers--all-MiniLM-L6-v2")):
```

## Validation

Run from the worktree root, with no dependency installation or lockfile operation. The final labeler run uses `export OMP_NUM_THREADS=1` in its shell before the command below (reason and earlier results recorded below); shot_design uses the inherited environment:

```bash
PYTHONPATH=$PWD/src pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e shot-design-cpu python -m pytest tests/shot_design -q -W error
PYTHONPATH=$PWD/src pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python -m pytest tests/labeler -q -W error
```

```text
shot_design: 1853 passed, 4 skipped in 251.10s (0:04:11)
labeler (OMP_NUM_THREADS=1): 2029 passed, 3 skipped in 510.47s (0:08:30)
```

Baseline reproduced: `2 failed, 1834 passed, 3 skipped`. Required behavior regressions were observed failing before their fixes. The additional skip is the shipped-cache structure test correctly checking the configured Frontier training cache, which is not available here; the v4 bundle contains no cache samples. Real-data parity remains enabled when that configured cache is available.

Ruff: `/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/ruff check <all 31 added/modified Python files>` — **All checks passed!**

`bash -n` passed for both changed Stellar scripts. Documentation tests: `4 passed`. Fresh read-only whole-branch review found no Critical or Important regressions.

The first two labeler runs used the inherited environment (PyTorch default: 40 intra-op threads). The first had one timing failure: `test_l14perf_identity.py::test_pooled_timeout_is_an_error_and_next_shot_runs` (`2028 passed, 1 failed, 3 skipped`). The recovery shot took 2.23 s against a 2 s timeout while suites were concurrent; isolated rerun passed (`1 passed in 6.48s`). The full command was rerun after shot_design finished and hit the same test failure (`1 failed, 2028 passed, 3 skipped in 702.36s (0:11:42)`; recovery shot: 3 blocks in 2.57 s); another labeler suite was still running elsewhere on the host. A focused order reproduction (U-Net tests followed by the timeout test) also failed: the recovery shot completed only 6 blocks in 2.49 s. With `export OMP_NUM_THREADS=1` before the same Pixi/pytest wrapper, that sequence passed (`15 passed in 7.24s`), with the recovery shot completing all 14 blocks in 0.72 s. The final full labeler rerun passed with that shell setting; the required suite command, timeout and tests are unchanged. No labeler code or tests were changed. All three full labeler runs emitted an XRootD FutureWarning at interpreter shutdown, outside pytest reporting; the final run exited 0 with no pytest failures.

## Deviations, remaining work and decisions

- **Labeler execution environment:** the inherited 40-thread environment fails the existing two-second timeout/recovery test. The final verification limits OpenMP to one thread; this is an explicit execution-environment difference, not a test or production-code change.
- **Unavailable paper files:** `dev/paper/` is absent from this worktree and its tracked files. Its README, `make_figures.py` and `decode_all_v4.py` were not fabricated or copied from elsewhere. Their requested changes remain undone.
- **Staticness filtering remains unresolved:** this checkout's `report.py` has no filter or threshold; it only computed proposed-arm staticness. Clarification was requested but no answer arrived. Both arms are now measured and collected independently, with all rows retained. Applying the literal filter requires its missing source/threshold. The reviewer agreed this is a reasonable fallback, not fulfillment of the literal filtering requirement.
- **Report metadata:** `report.py` is a writer, not an HDF5 reader. It now requires frame origin when writing; `batch_collect.py` requires the recorded origin when reading and still records generation when present.
- **Additional defaults/cache cleanup:** Python `DesignProgram` and `Intent` defaults also needed 2–6 s after the origin pin was removed. The bundle frame-code lookup was obsolete because v4 ships none. These were discovered and corrected during P0.5 with regression coverage; leaving them would preserve invalid defaults and a retired cache source.
- **Additional docs:** `docs/shot-design/programs.md` also named the obsolete checkpoint, so that example now points to v4.
- **Execution bookkeeping:** a Part 0 ledger was used instead of whole-plan helpers to avoid other agents' Parts A/E; intermediate fixture failures were resolved in P0.5 as ordered.
- **Production operations:** no production writes, submissions, fdp fetches, cache refreshes or retirement operations were performed. The Stellar UI pilot and its memory/time sizing are the post-merge session's work, per Part 0 acceptance.
- **Unchanged scope:** Parts A/E, `pyproject.toml`, `pixi.lock` and `data/events/` have no changes. Existing nondefault `--k0` seed-alignment behavior is unchanged; Part 0's seed contract is 20 frames.

No deferred code-review minor findings.
