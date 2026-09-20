# Review — Task C1: Manifest-driven modality table and `shot_design model --pin`

Base `dc662c2` → head `51035b6`, one commit. Read-only review; nothing in the tree was mutated.
Verified against the diff, the pinned bundle's real `MANIFEST.json` on disk, and four focused
checks outside the diff (`model_cfg()["revision"]` call sites, `MANIFEST.json` readers,
`load_codecs` callers, `frozen_modalities` readers).

### Spec Compliance

| Brief item | |
|---|---|
| `dynamics_config.modalities_from_manifest(path) -> tuple[ModalitySpec, ...]`, file order | ✅ `dynamics_config.py:69-78` — `json.loads(...)["modalities"]` then a tuple comprehension over `entries.items()`; `family`/`n_tok`/`codebook_size` map positionally onto `ModalitySpec(name, family, n_tok, codebook_size)`, ints coerced. Order is dict-insertion order, i.e. JSON key order. |
| `pin_bundle` signature and semantics | ✅ `ignite.py:69-142`. Keyword-only `codec_tmpl / dynamics_src / names / t0_start`, returns `bundle_dir(paths)`. `Path(codec_tmpl.format(m=name)).resolve()` then `shutil.copy2` (real bytes, symlink resolved) into `codecs/<m>/codec_best.pt`; dynamics resolved and copied to `<local_name>/<dynamics_file>`. |
| sha256 of **every** file incl. dynamics | ✅ `ignite.py:109,115` — `sha[rel]` per codec and `sha[dyn_rel]` for the dynamics file. Confirmed on the real pin: 16 entries for 15 codecs + `ignite_dynamics_prod_v4_mskfull_step3200.pt`. |
| Manifest layout (`_meta{created,generation,codec_tmpl,dynamics_src,copy_mode}`, `modalities`, `t0_start_s`, `frame_tokens`, `sha256`) | ✅ `ignite.py:116-136`, all five keys and all five `_meta` sub-keys present. Real file: `t0_start_s 1.0`, `frame_tokens 1209`, order `ece, bes, mhr, co2, mirnov, tangtv_lower, tangtv_upper, ts_core_*(2), ts_tangential_*(2), cer_ti, cer_rot, mse, filterscopes`, every `codebook_size` 1000, `n_tok` summing to 1209. |
| `check_bundle(paths) -> list[str]`, brief's message format | ✅ `ignite.py:145-170` — `f"{rel}: expected {want[:12]} got {got[:12]}"`, plus `": missing"` and a manifest-vs-yaml vocab line (the brief's "vocab check"). |
| `load_codecs` raises `CheckpointMissing` on mismatch, only when `sha256` present | ✅ `ignite.py:262-270`, before `_dynamics()` and before any weight load, so a v2 manifest (no `sha256`) is unaffected. |
| yaml `model:` block verbatim | ✅ `ignite_modalities.yaml:62-138` — `generation: v4`, `local_name: IGNITE_v4`, `codec_tmpl`, `dynamics_src`, `dynamics_file`, `frame_codes_cache`, `frame_tokens: 1209`, `t0_start_s: 1.0`, `window_ms: 250`, flow-style `production_vocabs` / `families` / `n_tok` with 15 names each, all vocabs 1000, n_tok summing to 1209, v2 rollback comment kept. SCOPE paragraph rewritten for 15/1209. `frozen_modalities:` dropped — grep confirms no reader anywhere. |
| New `mirnov` entry, mhr-shaped, 29 channels | ✅ `ignite_modalities.yaml:160-180`. `family/tier: spectro`, `n_tok: 192`, `n_channels: 29`, `target_fs: 500000`. The 29 `staged.cols` / `fetch.nodes` match `data/config/modalities/modalities.yaml:1150-1187` name for name and index for index, including the `MPI1A274D` repeat at indices 24 and 27, which the comment calls out rather than hides. |
| `--download` refuses when `generation != "v2"`; `--check` exit code | ✅ `cli.py:417-425` (stderr + `return 1`), `cli.py:445-450` (`return 1 if bad else 0`; an unpinned bundle yields the manifest-missing line, so exit 1). |
| `build.py` / `g_enc.py` changes minimal | ✅ minimal in size (`g_enc.py:430-432` records `generation` + `.get("revision")`; `build.py:1301-1310` compares `(generation, revision)` tuples) and necessary — grep confirms no `model_cfg()["revision"]` remains and `design/provenance.py:154` already used `.get`. But `build.py`'s substitute guard is weaker than the report admits — Important #1. |
| `test_program.py` still tests what its name says | ✅ `PRODUCTION_VOCABS` is now v4's 15×1000; `test_wrong_codec_generation_refuses_export_even_with_in_range_tokens` keeps `("ece", 16)` and swaps `("ece", 1000)`/`("bes", 32768)` — which under v4 are *correct* vocabs and would no longer be a wrong generation — for v2's `32768`/`64000`. The cases remain in-range-token cases caught only by metadata, which is the behaviour the name claims. |
| Tests never touch the Frontier data root | ✅ `tests/shot_design/conftest.py:36-61` points `models_dir` at `tmp_path/root/models`; `pin_bundle` writes only there. |
| Parked codec gap not papered over | ✅ `tests/shot_design/test_ignite.py:185-210` is untouched: it still asserts v2's 12-name encodable set and now runs (rather than skips) because a bundle exists, failing loudly. The two other `needs_weights` tests stay skipped on the Stellar `FM_DIR` guard. |
| TDD, one-line imperative commit, nothing from Stellar | ✅ RED/GREEN evidence in the report matches the new test files; commit message `ignite: v4 generation pinned by sha256 manifest; 15-modality table from manifest`. |

⚠️ **Not showable from the diff** (report claims, no executable trace):
- That the manifest's key order equals the dynamics checkpoint's own `modalities` order. The values (families, n_tok, vocabs, 1209) were re-derived here from the yaml and the pinned manifest and are self-consistent; the *checkpoint* side is the report's manual `torch.load`. See Important #3.
- `t0_start_s: 1.0` — the yaml comment states plainly that the bit-for-bit frame-code parity check against `frame_codes_cache` that pinned v2's `0.0` has **not** been run for v4. Honest, but it means the one number that silently shifts every frame index is currently taken on trust.
- The `mirnov` `staged:`/`fetch:` block is unverified against a staged file (no staged store reachable from OLCF) and unread by any code — documented as such in the yaml.

### Strengths

- The pin actually does the hard part right: `resolve()` before `copy2` means the copies are real files, not symlinks that would follow a live training run — and that is exactly the failure this task exists to prevent. Verified on the real bundle (16 sha entries, dynamics byte size preserved).
- The digest check is placed *before* `_dynamics()` and any weight load (`ignite.py:262`), so a tampered bundle fails as a checkpoint problem rather than as an unrelated import or a silent re-embed.
- Generation branching is driven only by the explicit `model.generation` key (`ignite.py:44-50` docstring says so and the code honours it) — nothing infers a generation from path shape or table width, which is what makes v2 rollback still coherent.
- `_check_dir(ckpt_dir)` / `check_bundle(paths)` split is the right factoring: `load_codecs` only ever has a directory, and it reuses the identical code path instead of a second implementation.
- Error messages were updated in step with the code: the "no bundle" text now names both installers, still contains `--download` and `models_dir` (so the pre-existing `test_ignite.py` assertion holds), and no longer interpolates the now-absent `repo_id`.
- The `mirnov` yaml entry is derived from the config that actually produced the training corpus, and reproduces the duplicated pointname instead of "fixing" it — with a comment explaining why. That is the difference between a transcription and an engineering decision.
- The report's self-review is unusually candid and every claim I spot-checked held up; the parked codec failure was left visible rather than skipped or deleted.

### Issues

#### Critical

None.

#### Important

**1. `build.py:1301-1310` — the v4 staleness guard does not survive a re-pin, and the comment claims it does.**
Under v4 both sides of the comparison reduce to `("v4", None)`, so the check only catches a *generation* switch. The in-code comment argues this is safe because "`load_codecs` above has already checked" the sha256 — but that check proves the bundle matches **its own manifest**, not that it matches the bundle the database was built with. Re-running `shot_design model --pin` rewrites the codecs, the dynamics file and the manifest together, so a re-pinned bundle passes `load_codecs` and passes this guard, and `shot_design add` then merges embeddings from two different codec generations into one database — silently. This is not hypothetical: the yaml comment at `ignite_modalities.yaml:113-116` and Concern 3 of the report both state that `dynamics_src`/`codec_best.pt` are live symlinks that will move, so re-pinning is the expected workflow.
*Fix:* record the bundle identity in `manifest_block()` (`ignite.py:655-666`) — `design/provenance.py:143-167` already computes exactly the right thing, `sha256` of `codecs/MANIFEST.json` — and compare that in `_carry_over` instead of, or alongside, `revision`. Two lines in each place, and it restores the v2-era property.

**2. `ignite.py:262-270` — every `load_codecs` re-hashes the whole 4.4 GB bundle, including the 3.3 GB dynamics file it never loads.**
The report measures 17.6 s CPU / 4.8 s wall warm-cache and treats it as a fan-out cost. It is worse than that: `design/program_reference.py:272` → `design/seed.py:153` calls `load_codecs` **per request** (no `codecs` passed, under `_PREPARE_LOCK`), so this lands on an interactive "prepare IGNITE input" path, and on a cold Lustre read of a 3.3 GB file it will be much larger than 5 s. `build.py:1297`, `ignite.py:695` and `g_enc.py:422` each pay it per process as well.
*Fix:* in `load_codecs`, digest only the entries under `codecs/` (or only the codecs in `wanted`) and leave the whole-bundle verification to `check_bundle` / `model --check`. The brief's intent — "the weights behind this encode changed" — is fully served by the codec digests; the dynamics file is not a weight this call uses.

**3. The manifest's modality table — the part that changed in this task — is copied out of hand-maintained yaml and never cross-checked against the checkpoint, while the sha256 machinery makes the bundle look self-certifying.**
`pin_bundle` (`ignite.py:124-134`) takes `family`/`n_tok`/`codebook_size` from `model_cfg()`, and `modalities_from_manifest` then presents that as "the canonical token order — it is what the checkpoint was trained with" (`dynamics_config.py:74-77`). Nothing verifies that. The vocab check in `_check_dir` (`ignite.py:157-165`) compares the manifest to the same yaml it was generated from, so immediately after a pin it is tautological; it only fires on a later yaml edit. The report's verification against `dynamics_best.pt`'s own `modalities` tuple was a one-off manual `torch.load` that leaves no trace and will not be repeated on the next re-pin — of a *live* training run whose layout could legitimately change.
*Fix:* a `needs_weights`-style test (the pattern already exists at `tests/shot_design/test_ignite.py:178`) asserting the pinned manifest's `(name, family, n_tok, codebook_size)` tuple equals the dynamics checkpoint's `modalities`, read with `torch.load(..., mmap=True, weights_only=False)` so no storage is materialised. Cheap, and it converts the report's prose into a standing guarantee.

#### Minor

- `ignite.py:96-142` — `pin_bundle` is not atomic and does not prune. A failure partway through a **re**-pin leaves new codec bytes next to the old manifest (detectable — `check_bundle` reports mismatches — but the directory still looks installed), and re-pinning a shorter `names` list leaves orphan modality directories that nothing reads. Write the manifest to a temp file and `os.rename` it, and either prune or list the strays.
- `ignite.py:133` — a partial pin writes `frame_tokens` = sum of the pinned names only, with no cross-check against the yaml's `frame_tokens: 1209` or its 15-name set. Since `modalities_from_manifest` is now the canonical layout for a dynamics model, a partially pinned bundle would hand out a silently short layout. `_check_dir` deliberately tolerates partial pins for the vocab check; add a name-set / `frame_tokens` line there so a partial pin is at least *reported*.
- `cli.py:437-444` — `--pin` reads `codec_tmpl`, `dynamics_src`, `n_tok` unguarded and would `KeyError` on a v2 `model:` block, while `--download` has the symmetric guard at `cli.py:418`. The report flags this as deliberate; a two-line `if generation == "v2": ... return 1` would make the pair symmetric.
- `ignite.py:40-41` — `CheckpointMissing`'s docstring still says "the local checkpoint directory is absent or empty"; it is now also the tamper/digest-mismatch error. One line.
- `ignite.py:105,113` — `sha256` keys are relative to the *bundle root* (`codecs/ece/codec_best.pt`, `ignite_dynamics_...pt`) while the manifest itself lives in `codecs/`. Correct and consistently consumed by `_check_dir(ckpt_dir)`, but nothing says so; a sentence in `pin_bundle`'s docstring would prevent the obvious future mistake of resolving them against the manifest's own directory.
- `t0_start_s` now exists in two places — the yaml (read by `seed.py:172`, `ignite.py:544`) and the pinned manifest (written, never read). They can drift after a yaml edit with nothing noticing. Either read it from the manifest where a bundle is in hand, or note in the yaml that the manifest copy is provenance only.
- `tests/shot_design/test_ignite_v4.py:52` — `test_check_bundle_reports_a_changed_codec` calls the previous test function as a setup helper. It is the brief's own shape and it works, but a fixture would decouple the two, and a failure in the first now fails both with a confusing traceback.
- Ruff: ~65 added Python lines exceed the configured `line-length = 88` (longest 103, `test_ignite_v4.py:47`). `ruff check`'s default rule set does not include E501 so the repo is green, and the report's point that the tree already exceeds 88 is true — but the task constraint said 88 for new lines, and `ruff format` would reflow them.
- `cli.py:1690` — `--full`'s help still says "3.5 GB dynamics checkpoint" (v2's size); the v4 one is 3.3 GB and `--full` does not apply to it at all.

### Assessment

**Task quality: Needs fixes**

Every brief interface was delivered exactly as specified, the real pin verifies clean on disk, and the out-of-brief `build.py`/`g_enc.py`/`test_program.py` edits are genuinely the minimum the missing `revision` key forced — but `build.py`'s replacement guard is weaker than its own comment claims (a re-pin, the expected workflow, silently passes), and `load_codecs` now re-hashes 3.3 GB of dynamics weights it never loads on an interactive per-request path. Both are small, localised fixes; nothing here needs redesign.
