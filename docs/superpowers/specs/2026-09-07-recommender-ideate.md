# Appendix B — ideate detailed plan (Opus Plan agent, condensed)

## B0. Facts (F1–F14) → see V9–V14 above plus: F6 filterscopes rollout skill −0.22/−0.24 vs persistence; F12 IGNITE embedding width 2176; `emb_ignite_win` 7.1 GB at full corpus with 250 ms windows → 1 s windows + float16 + memmap (0.9 GB); F13 encode ~15 s/shot V100S; F14 rollout 143 s/1 s, 1076 s/4 s, 5.39 GiB.

## B1. Layout
```
configs/ideate/{paths,signals,actuators,retrieval,flags,phenomena,themes,ignite_modalities,ui,llm}.yaml, shot_lists/recommender_v1.yaml, evalsets/reference_shot_prompts.csv
src/ideate/{__init__,__main__,schema,config,cli}.py
src/ideate/shotdb/{reader,corpus,legacy_raw,features,text,ignite,build,store}.py
src/ideate/retrieval/{channels,rank,describe,suggest,scenarios,blurb,actuation,phenomena,circumstances}.py
src/ideate/design/{actuators,seed,rollout,decode,spectra,predict,report}.py
src/ideate/labels/{join,events}.py   src/ideate/flags/rules.py   src/ideate/llm/client.py
src/ideate/mcp/{__init__,__main__,server,tools}.py   src/ideate/ui/{app,chat,waveforms,jobs,serve}.py + static/
src/ideate/eval/{prompts,phenomenon_recall,latency}.py
tests/ideate/ (conftest builds corpus-layout fixtures; -W error)   scripts/ideate/{encode,rollout,build}.sbatch
```
Port map: schema (+`Interval, PhenomenonHit, CircumstanceReport, DesignRequest/Report, EventRef`), config (`IDEATE_*`, `_find_configs()`), rawfile→legacy_raw (optional), drop fetch/curate/`to_processed_h5`/pcslayout, features/text unchanged, ignite (delete conversion half; sibling import), build (Reader; labels/events join), store (+events, labels_pivot, `_label_tokens`), retrieval (+phenomena, circumstances), ui in iteration 2, cli (+`corpus select|scan`, `labels join`, `phenomenon`, `events`, `circumstances`, `encode`, `design`, `mcp`, `eval`), tests ~560.

## B2. pixi
```toml
[tool.hatch.build.targets.wheel] packages = ["src/faith","src/tokamak_foundation_model","src/labelmaker","src/ideate"]
[tool.pixi.feature.ideate] platforms=["linux-64"]
[tool.pixi.feature.ideate.dependencies] pyarrow=">=17,<22" pyyaml=">=6,<7" pydantic=">=2.12,<3" scipy="*" scikit-learn=">=1.5,<2" h5py=">=3.15,<4" tqdm=">=4.66,<5" pypdf=">=4,<7" httpx=">=0.27,<1" fastapi=">=0.115,<1" uvicorn=">=0.30,<1" sentence-transformers=">=6,<7" mcp=">=2.1,<3" pytest=">=9,<10" protobuf="<7" duckdb=">=1.1,<2" (optional)
[tool.pixi.feature.ideate.target.unix.activation.env] IDEATE_DATA_ROOT="/scratch/gpfs/EKOLEMEN/nc1514/ideate" LABELMAKER_ROOT="/scratch/gpfs/EKOLEMEN/nc1514/labelmaker" IDEATE_CORPUS="/scratch/gpfs/EKOLEMEN/foundation_model" HF_HUB_OFFLINE="1" TOKENIZERS_PARALLELISM="false"
[tool.pixi.feature.ideate-cpu.pypi-dependencies] torch cpu index; torchvision cpu index
[tool.pixi.feature.ideate.tasks] ideate="python -m ideate" ideate-mcp="python -m ideate.mcp" ideate-test="pytest tests/ideate -q"
[tool.pixi.environments] ideate=["ideate","cuda"]  ideate-cpu=["ideate","ideate-cpu"]
```
Fallback if conda `mcp` blocks the solve: pypi-dependencies `mcp=">=2.1,<3"`. Add pinned-vector test for MiniLM across the sentence-transformers major bump.

## B3. Build
`CorpusReader` (xdata×1000 → ms; `shape[-1]<2` → unavailable; `OSError` → shot failed; `locking=False`). Scalars: corpus actuator totals via `actuators.yaml` `corpus: {group, channels}` (nbi←pinj, nbi_voltage←beam_voltage, nbi_torque←tinj, ech←ech_power, gas←gas_flow, gas_raw, rmp, icoil←i_coil — the 8 groups of ACT_SPEC); EFIT/plasma scalars from `$LABELMAKER_ROOT/features/<shot>_features.h5` via `labelmaker.features.store.read_feature` (ip, bt, betan, kappa, tritop, tribot, li, qmin, gapin, aminor, volume, r0, pcbcoil; profiles reduced to core/edge/peak) with per-shot resolver provenance; corpus diagnostics-as-scalars (ne_line from co2, dalpha from filterscopes[0:8], neutrons). Missing → NaN + `coverage="pending"`, never 0. Segments: `find_segments` on ip (25 ms), `t_max_ms` 10000. IGNITE embeddings via `encode_db` directly on corpus files (width 2176).

## B4. Retrieval additions
`phenomena.yaml` (see §5.6). `retrieval/phenomena.py`: `registry()`, `resolve(text) -> [(id, weight)]` (longest alias, word-boundary, `exclude` vetoes), `evidence(shot, ph, db, segment)`, `locate(ph, db, n, *, segment, constraints, min_confidence) -> [PhenomenonHit]`. `PhenomenonHit(shot, phenomenon, score, intervals[Interval], total_duration_s, quote, quote_role, text_snippets, actuators_at_onset, label_evidence, forecasts, caveats, run_id, mp_title)`. `ShotDB._label_tokens` per segment row → `mask()` unchanged algebra. `CHANNELS["phenomenon"]` weight 1.2. `describe._phenomenon_line`. Quote always via `describe.best_quote` (one LogEntry, never spliced).

## B5. Circumstances — see §7. Output `CircumstanceReport(phenomenon, arm, pre_s, lead_s, segment, n_events, n_event_shots, n_controls, n_unmatched, features[FeatureEffect(name, units, n_*, means, hedges_g, g_ci, auroc, auroc_ci, p_perm, q_bh, direction, correlated_with)], underpowered, timelines[EventTimeline(event_id, shot, t0_s, t1_s, changes[Change], quote)], caveats, method)`.

## B6. Design — `ACT_SPEC`, `frame_means(shot, paths, n_frames=239)`, `z_score`, `build_actuators`, `channel_index`, `apply(aset, raw, stats, policy)`; `encode_frame_codes(shot, paths, device, include_video=True)`; `RolloutResult(shot, seed, k0, n_frames, horizon_s, arm, codes_path, accuracy, live_modalities, elapsed_s, device, dtype)` cached under `runs/<shot>/<hash>`; `spectra.{spectro_axes, unstandardize, to_ae_input}`; `PredictedPhenomenon(phenomenon, intervals, model, confidence ∈ {low, medium}, basis)`; `DesignReport(reference_shot, actuation_id, edited_channels, z_shift, extrapolation_warnings, baseline, counterfactual, token_divergence, decoded_change, predicted, caveats, nearest_real_shots)`. fp16 only if per-modality tok_acc shift ≤0.01.

## B7. MCP (`mcp>=2.1`, stdio)
```python
from mcp.server import MCPServer; mcp = MCPServer("ideate")
@mcp.tool() search_shots(text, ref_shot, segment, constraints, actuators, require_labels, avoid_labels, n)
@mcp.tool() phenomenon_locate(phenomenon, n, segment, constraints) -> list[PhenomenonHit]
@mcp.tool() describe_shot(shot, segment); get_events(shot, phenomenon, t0_s, t1_s); circumstances(phenomenon, pre_s, arm, n_features)
@mcp.tool() suggest_configuration(result_ids, successful_only); check_proposal(actuators); design_rollout(reference_shot, actuation_id, edits, horizon_s, arm, wait=False) -> DesignReport|JobHandle; get_job(job_id)
# resource ideate://manifest ; every tool returns caveats: list[str]; ui/chat.py TOOLS generated from this registry (single-source test)
```

## B8. Eval & latency
200-prompt evalset copied to `configs/ideate/evalsets/`; metrics: coverage ≥95%, hard-filter survival, channel participation, category→phenomenon resolution ≥80% (qh_mode→qh/eho, elm_rmp→elm, fast_ions→ae), run diversity, duplicate rate, 20 hand-graded prompts. `phenomenon_recall.py` vs annotation sheets (`split=test` rows only; refuse n<20). Latency budgets (warm): load <3 s (2k), search text <400 ms, search no-text <200 ms, locate <300 ms, circumstances <20 s, encode ~15 s, rollout 1 s horizon 150–300 s.

## B9. Draft errors fixed by this appendix
88-channel freeze → resolved ACT_SPEC; "encoding impossible" → G-ENC; AE on decoded waveform → spectral bridge + G-DEC; level edits no-op → reference_stats; ELM from rollout → low confidence; pcslayout vendor → delete line; 612 tests → ~560; FastMCP → MCPServer; subset choice → 2,654 pool; iteration 1 gated on human annotation → decoupled (ideate consumes whatever `events/` exists).

