# Labelmaker Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the tearing survival label reliable the way its authors measure it (per-shot alarms), publish the spread the survival model already computes, recalibrate and finish training it, draw a TabPFN regressor beside it, decide with data whether the Alfvén-eigenmode label can be built from a coherent-mode mask, and if so build it; then train labelmaker's own ELM survival model without BES.

**Architecture:** Every change is either a pure-numpy scoring/calibration module wired into `validate`, a new output column set on an existing adapter, or a self-contained training script with its own SLURM launcher whose artifact lands in `<root>/models/<slug>/` with `PROVENANCE.json` and then gets a model folder like every other model. Nothing here touches how labels are stored; new series are new groups in the same per-shot HDF5 files.

**Tech Stack:** Python 3.11 in the `labelmaker` pixi env for everything under `src/labelmaker` and `tests/labelmaker` (numpy, scipy, h5py, torch-cpu, matplotlib; **no scikit-learn there**). Training and GPU work run in a uv venv on group storage (`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/envs/phase3`: torch cu126, scikit-learn, scikit-survival, tqdm, pandas, tabpfn 8.5.0, tifffile) or in the tokeye repo's own `.venv` (torch 2.9.1+cu128, read-only use). SLURM on stellar: `gpu` partition (A100) for GPU jobs, `all` for CPU jobs.

**Design spec:** `docs/superpowers/specs/2026-09-05-labelmaker-phase3-design.md`. Sections 3 to 6 are what this plan builds; section 2 is the upstream evidence every task leans on.

## Global Constraints

- **Repository:** `/scratch/gpfs/nc1514/FusionAIHub`, branch `labelmaker`. Every shell command starts with `cd /scratch/gpfs/nc1514/FusionAIHub &&` or uses absolute paths: the session's working directory resets between calls to `/scratch/gpfs/nc1514/shot-recommender-system`, which is **another session's checkout and must never be read, written, or entered**.
- **Read-only trees:** everything under `/projects/EKOLEMEN/`, `/scratch/gpfs/nc1514/fdp/`, `/scratch/gpfs/nc1514/aemodes/`, `/scratch/gpfs/nc1514/tokeye/`, `/scratch/gpfs/EKOLEMEN/d3d_fusion_data/`, `/scratch/gpfs/EKOLEMEN/hackathon/`, `/scratch/gpfs/EKOLEMEN/foundation_model/`. Reference and read; never write. Running `/scratch/gpfs/nc1514/tokeye/.venv/bin/python` is reading.
- **Where artifacts go:** `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/` (`Paths.from_env()`, override `LABELMAKER_ROOT`). Venvs go under `.../labelmaker/envs/`. Never write data, venvs or caches under `/scratch/gpfs/nc1514/` (near quota) beyond the repo's own tracked files. `figures/` at the repo root is Nathan's and is not touched.
- **Commits:** one per task, subject `labelmaker: <imperative summary>`, body states the evidence (numbers, commands). Made with `git commit -F -` and a quoted heredoc (`<<'MSG'`), never `git commit -m`. **No `Co-Authored-By` trailer of any kind, no "Generated with" footer.** Commit only the paths the task names. **Never push.** Never `git checkout --`, `git restore`, `git reset`, `git stash`.
- **Tests:** `pixi run -e labelmaker python -m pytest tests/labelmaker -q -W error -p no:cacheprovider` must pass (301 passed, 2 skipped at the start of this plan) and stay hermetic: no network, no `/projects`, no data root. Real-data checks live in scripts under `outputs/labelmaker/` or `scripts/labelmaker/`, not in tests. Test code writes only to `tmp_path`.
- **Lint:** `pixi run -e labelmaker ruff check src/labelmaker tests/labelmaker` clean over the whole package after every task. `except Exception` handlers that isolate a shot carry `# noqa: BLE001` and a reason.
- **Conventions:** `@dataclass(frozen=True)` and plain functions; no Pydantic; no imports from `tokamak_foundation_model`; no TensorFlow; new dependencies only in the phase3 venv, never in the pixi env, unless a task says otherwise. Labels stay probabilities/quantities, never thresholded. HDF5 layout: `xdata` float64 seconds, `ydata` float32 `(C, T)`. Every file written atomically via `config.atomic_path`.
- **Cards:** every model folder's `README.md` front matter must satisfy `registry.card_discrepancies(slug) == []` after the task. A number in a card names the row set it was measured on (all aligned pre-onset rows vs onset-only shots are different populations; see spec section 2.1).
- **SLURM:** submit with `sbatch`, then poll `squeue -u nc1514` and `jobstats <jobid>` while the job runs and record CPU/GPU utilisation and memory high-water marks in the report and in the artifact's `PROVENANCE.json`. Say plainly when utilisation is low and what would fix it. Request modest resources first and size the next request from what the first used.
- **Secrets:** never print or copy `~/.fdp/token`, the Hugging Face token, or `TABPFN_TOKEN`.
- **Pool:** the 500-shot pool is `$LABELMAKER_ROOT/shots_500.txt`; 486 of them align to archived rows and 86 of those have an archived onset. Labels for both tearing models already exist for it. Re-running `infer` for one slug rewrites only that slug's group (`write_labels(merge=True)`).

## File structure

| path | responsibility |
|---|---|
| `src/labelmaker/alarm.py` (new) | pure functions: per-shot alarm verdict, warning time, jumps, any-row rule, IPCW time-dependent AUC. No I/O. |
| `src/labelmaker/calibrate.py` (new) | pure functions: prior shift on log-odds, isotonic map (PAVA) fit/apply/serialise. No I/O. |
| `src/labelmaker/validate.py` | gains `_pooled_onset_rows`, `alarm_quality`, `calibration_study`, a `time_to_onset` truth kind; `score_against_truth` gains alarm fields. |
| `src/labelmaker/models/runners/dsm_pickle.py` | gains `mixture`, `quantiles`, `gate_entropy`; `survival` reuses `mixture`. |
| `src/labelmaker/models/base.py` | `OutputField.attrs` (per-label extra HDF5 attributes) flowing through `labels/schema.py` and `labels/store.py`. |
| `src/labelmaker/models/d3d_tearing_time_to_event_dsm/spec.py` + card | 17 -> 20 output columns (risks, quantiles, mixture, entropy, isotonic risks). |
| `src/labelmaker/models/d3d_tearing_time_to_event_dsm_continued/` (new) | variant folder for the retrained artifact; reuses the base spec's code. |
| `src/labelmaker/analyze.py` + `analyze_default.yaml` | band panel for `*_p50` labels with `_p10`/`_p90` siblings. |
| `src/labelmaker/run.py` | validate stage also writes `alarm_quality` and `calibration_study` for slugs with archive truth. |
| `scripts/labelmaker/` (new) | training scripts and sbatch launchers: `phase3-requirements.txt`, `make_phase3_env.sh`, `retrain_tearing_dsm.{py,sbatch}`, `elm_dsm_train.{py,sbatch}`, `ae_*`. |
| `outputs/labelmaker/tabpfn/scripts/time_to_onset_regressor.py` (+sbatch) | TabPFN regressor comparison and figure. |
| `outputs/labelmaker/ae/` (new) | AE measurement outputs: `mask_vs_annotation.json`, figures, README with the go/no-go. |
| `outputs/labelmaker/presentation_continued/` (new) | Phase 2 figures re-rendered for the retrained model. |
| `tests/labelmaker/test_alarm.py`, `test_calibrate.py` (new); `test_dsm_pickle.py`, `test_tearing_dsm_adapter.py`, `test_archived_truth.py`, `test_analyze.py`, `test_label_store.py`, `test_run.py` extended | |

## Dispatch (controller notes)

| task | implementer | reviewer |
|---|---|---|
| 1, 3, 7a, 8b | Codex `gpt-6-astra`, `--effort medium`, `--write`, `--cwd /scratch/gpfs/nc1514/FusionAIHub` | Claude opus |
| 2, 4, 5, 6, 7b, 7c, 8a | Claude opus | Claude opus |

Long SLURM jobs (Tasks 4, 5, 6, 7b, 8a) may run while the next code task is implemented, because they touch disjoint paths; the controller polls `squeue`/`jobstats` and records utilisation in the ledger.

---

## Task 1: per-shot alarm scoring, ported from upstream (`alarm.py`, `validate`)

**Why:** upstream evaluates the survival model per shot (`/projects/EKOLEMEN/survival_tm/metrics_helpers.py`), not per row. Until labelmaker reports the same quantities, none of its numbers are comparable to the model's own paper.

**Read first:** `metrics_helpers.py` (read-only, 110 lines), `src/labelmaker/validate.py` lines 927-1030 (`binary_metrics`), 1400-1527 (`ARCHIVE_TRUTH`, `archived_truth`, `score_against_truth`), `src/labelmaker/run.py` lines 759-830 (validate stage), `tests/labelmaker/test_archived_truth.py` (fixture pattern), `outputs/labelmaker/dsm_onset_quality.py` (how published labels were pooled against the archive in Phase 2).

**Definitions (port these exactly, and fix two upstream quirks, recording both in the docstring):**

1. `label = risk >= threshold` over the shot's valid, finite rows in time order.
2. **Final-label verdict** (`get_classification`): from the LAST row: `TP` if label on and the shot has an onset, `FN` if off and onset, `FP` if on and no onset, `TN` if off and no onset. `alarm = label[-1]`.
3. **Warning time**: for a `TP`, `onset_s - t[k]` where `k` is the first row of the final run of ones (if every row is on, `k = 0`). Upstream reads the dataset's own time-to-event at that row, which is the same number. `None` otherwise.
4. **Jumps** (`get_classification`'s loop): the number of 0 -> 1 transitions that later revert to 0 AND whose on-run lasted at least `JUMP_MIN_DURATION_S = 0.4`. Upstream indexes `t[next_zero_index[0]]` with a window-relative index (a bug: it should be `t[i + next_zero_index[0]]`); implement the intended absolute index and say so. Also report `n_excursions` (every reverted 0 -> 1, regardless of duration). Correct the spec's section 3.1 sentence "a crossing that reverts within 400 ms is a jump" to this definition in the same commit.
5. **Any-row rule** (`fnr_fpr_calculator`): the shot is called at horizon `h` if the risk at `h` is `>= threshold` on ANY valid row. Upstream thresholds survival at 0.7, i.e. risk 0.3; we parametrise on risk.
6. **Pool rates:** FPR = FP / (FP + TN) over quiet shots, FNR = FN / (FN + TP) over tearing shots, both for the final-label rule and the any-row rule, per threshold and per horizon. Horizon-integrated FPR/FNR: trapezoid over the horizons in seconds (upstream `np.trapz(fnrs, prediction_times)`), over the three published horizons (0.25, 0.5, 1.0 s).
7. **IPCW time-dependent AUC** (Uno et al., what `sksurv.metrics.cumulative_dynamic_auc` computes) at horizon `h`: per-row time `T` and event `e` where for a shot with onset `T = onset_s - t`, `e = 1` (rows before onset only) and for a shot without `T = t_end - t`, `e = 0` with `t_end` the last archived row's time; `G` the Kaplan-Meier estimate of the censoring survival function from `(T, 1 - e)`; cases `T <= h, e = 1` weighted `1 / G(T_i)`, controls `T > h` weighted `1 / G(h)`; AUC = weighted fraction of (case, control) pairs with `score_case > score_control` (ties half). Pure numpy, O(n log n) via sorting or O(n_cases * n_controls) if n < 20,000 rows; state which.

**Steps:**

- [ ] Write `tests/labelmaker/test_alarm.py` (hermetic, synthetic arrays) with, at least:
  - final-label verdict for the four cases; `warning_time_s` for an always-on trace equals `onset_s - t[0]`; for a trace that turns on at 1.0 s with onset 1.5 s equals 0.5.
  - jumps: a 0.3 s excursion counts as an excursion but not a jump; a 0.5 s excursion that reverts counts as both; the final run (never reverts) counts as neither.
  - any-row rule differs from final-label rule on a trace that spikes then falls.
  - IPCW AUC equals `binary_metrics`' AUROC of `T <= h` when nothing is censored; a hand-computed 6-row case with censoring; returns `None` when there are no cases or no controls.
  - horizon integral of constant rates equals rate x (1.0 - 0.25).
- [ ] Run them, confirm they fail (module missing).
- [ ] Write `src/labelmaker/alarm.py`: `ShotAlarm` frozen dataclass (`verdict`, `alarm`, `warning_time_s`, `jumps`, `n_excursions`, `n_rows`), `shot_alarm(t, risk, valid, *, threshold, onset_s) -> ShotAlarm`, `any_row_call(risk, valid, *, threshold) -> bool`, `pool_rates(verdicts) -> dict` (fpr, fnr, counts), `horizon_integral(horizons_s, values) -> float`, `km_censoring(T, e) -> callable`, `ipcw_auc(T, e, score, horizon_s) -> float | None`. Module docstring names `metrics_helpers.py` and the two deviations.
- [ ] Run the tests until green.
- [ ] Extend `validate.score_against_truth`: for every binary or `onset_within` label with a `threshold`, add `alarm` (bool), `verdict`, `warning_time_s`, `jumps`, `n_excursions`, `any_row_call`. Onset for the shot is `truth["onset_s"]`. Add tests in `test_archived_truth.py` (use the existing `_fake_match` fixture): a trace on at the end of a tearing shot is `TP` with a positive warning time; a quiet shot with a spike is `TN` under final-label and `True` under `any_row_call`.
- [ ] Add `validate._pooled_onset_rows(slug, shots, paths, *, archive, timeout_s) -> dict`: for each shot in `shots` with a labels file, align via `_matched_shot` with the CNN spec (as `archived_truth` does), read every published label of `slug` that has an `ARCHIVE_TRUTH` entry, and return per-label arrays over pre-onset valid rows: `shot`, `t`, `y`, `onset_s` (nan when none), `t_end`, plus per-shot lists (`shots_used`, `shots_with_onset`, `skipped` with reasons). This is the one place rows are pooled; Task 3 reuses it. Test it on the `wired` fixture in `test_run.py` style (monkeypatch `_matched_shot` and write a small labels file with `write_labels`).
- [ ] Add `validate.alarm_quality(slug, shots, paths, *, archive=TM_ARCHIVE, thresholds=(0.05, 0.1, 0.2, 0.3, 0.5, 0.7), timeout_s=None) -> dict`: per label x threshold: `n_quiet`, `n_tearing`, final-label `fpr`/`fnr`, any-row `fpr`/`fnr`, `warning_time_s` median and quartiles over TPs, `jumps` histogram, `n_excursions` median; per label: plain AUROC (existing `binary_metrics`) and `ipcw_auc` on the same rows; `horizon_integrated` any-row `fpr`/`fnr` per threshold over the three horizons (only for the survival slug, whose three labels share one row set); `row_set` string naming the population ("pre-onset valid rows of every aligned shot"). Report written with `write_report(paths, slug, "alarm_quality", ...)`.
- [ ] Wire it into `run.py`'s validate stage beside `label_quality`, guarded the same way (`except Exception` -> `{"error": ...}`), only for slugs that have at least one `ARCHIVE_TRUTH` key. Extend `test_run.py`'s validate test to assert the report file appears (monkeypatch `alarm_quality` to a stub if the fixture cannot align rows).
- [ ] Full suite + ruff clean.
- [ ] Measure on the pool: `pixi run -e labelmaker python -m labelmaker.run validate --models d3d_tearing_time_to_event_dsm d3d_tearing_onset_cnn1d --shot-file /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/shots_500.txt --workers 8`. If `label_quality`/`reconstruction_fidelity` are too slow to re-run, call `alarm_quality` directly from a one-off script and write the report with `write_report`. Record in the report file: FPR/FNR by rule at thresholds 0.1, 0.2, 0.3, 0.7 for `tm_risk_1s`, median warning time, IPCW AUC vs plain AUROC per horizon.
- [ ] Card: `d3d_tearing_time_to_event_dsm/README.md` Evaluation gains a "Per shot, upstream's way" table with those numbers, the row set named, and one sentence comparing final-label FPR/FNR to the any-row rule. `docs/LABELMAKER.md` gets one line under validation outputs naming `alarm_quality.json`.
- [ ] Commit: `labelmaker: score the survival label per shot the way upstream does` with the measured numbers in the body. Paths: `src/labelmaker/alarm.py`, `src/labelmaker/validate.py`, `src/labelmaker/run.py`, the card, `docs/LABELMAKER.md`, the spec (section 3.1 correction), tests.

---

## Task 2: publish the survival mixture's spread (`dsm_pickle`, tearing DSM spec, `analyze` band panel)

**Why:** `dsm_pickle.survival()` computes the full k=3 log-normal mixture and returns three scalars. The quantiles and the mixture are the model's actual uncertainty statement; spec section 4.1.

**Read first:** `src/labelmaker/models/runners/dsm_pickle.py`, `src/labelmaker/models/d3d_tearing_time_to_event_dsm/spec.py`, `src/labelmaker/models/base.py` lines 517-580 (`OutputField`, `OutputSpec.decode`), `src/labelmaker/analyze.py`, `analyze_default.yaml`, `tests/labelmaker/test_dsm_pickle.py`, `test_tearing_dsm_adapter.py`, `test_analyze.py`, `src/labelmaker/models/registry.py::card_discrepancies`.

**Output columns after this task** (column index -> name, task, units):

```
0-2   tm_risk_250ms, tm_risk_500ms, tm_risk_1s      binary       ''        (unchanged)
3-5   tm_time_p10, tm_time_p50, tm_time_p90         regression   'ms'      quantiles of time to onset
6     tm_time_iqr_log                                regression   ''        ln(p90) - ln(p10)
7-9   tm_mix_w0, tm_mix_w1, tm_mix_w2                regression   ''        softmax gate weights
10-12 tm_mix_mu0..2                                  regression   'ln ms'   component location (ln t)
13-15 tm_mix_sigma0..2                               regression   ''        component log-scale: exp(sigma) is the std of ln t
16    tm_gate_entropy                                regression   'nat'     -sum w ln w, in [0, ln 3]
```

(Task 3 appends columns 17-19.) Component order is the checkpoint's own; the card says the components are not identifiable across retrainings.

**Steps:**

- [ ] Tests first, in `test_dsm_pickle.py`: `mixture(graph, x)` returns `(log_w, mu, sigma)` each `(n, k)` with `exp(log_w)` summing to 1 (atol 1e-12); `survival` is unchanged (the golden test keeps passing); `quantiles(graph, x, (0.1, 0.5, 0.9))` on a k=1 graph equals the analytic log-normal quantile `exp(mu + exp(sigma) * sqrt(2) * erfinv(2q - 1))` to rtol 1e-6 (`scipy.special.erfinv`); on the k=2 fixture `p10 < p50 < p90` and `S(p50) == 0.5` to 1e-6; `gate_entropy` of a uniform gate is `ln k`, of a one-hot gate is 0.
- [ ] Implement in `dsm_pickle.py`: `mixture`, `quantiles` (vectorised bisection in `ln t` between `ln 1e-3` and `ln 1e7` ms, 80 iterations; the mixture survival is monotone so no bracket check is needed beyond the endpoints, but assert `S(lo) >= 1-q >= S(hi)` once and raise otherwise), `gate_entropy`; refactor `survival` to call `mixture`.
- [ ] `spec.py`: `predict` stacks the 17 columns in the order above into `(1, T, 17)`; `OUTPUT_SPEC` declares them (`task="regression"`, `activation="none"`, units as listed). Card `outputs:` updated to match; `card_discrepancies` returns `[]` (add that assertion to `test_tearing_dsm_adapter.py` if not already there). Extend `test_predict_reproduces_the_upstream_preprocessing_step_by_step` or add a sibling asserting the new columns against direct `dsm_pickle` calls on the same `x`.
- [ ] `ARCHIVE_TRUTH` gains `"d3d_tearing_time_to_event_dsm/tm_time_p50": {"kind": "time_to_onset"}`; `score_against_truth` handles that kind: rows before onset of shots WITH an onset, target `(onset_s - t) * 1000` ms, metrics `n`, `median_abs_log_ratio`, `bias_log` (mean of `ln(pred/target)`), `rmse_log`, `kind = "time to archived onset (ms), tearing shots only"`; `scored: False` with a reason on a shot without onset. Test in `test_archived_truth.py`.
- [ ] `analyze`: a listed label whose name ends in `_p50` and whose `_p10` and `_p90` siblings exist in the label file becomes a `band` panel: median line, `p10..p90` shaded band, log y-axis in ms, the archived time-to-onset `(onset_s - t) * 1000` drawn in the truth colour on pre-onset rows when the truth is available, onset line as on other panels, invalid rows shaded as on other panels. `summarize_label` for such a label adds `p50_at_onset_minus_1s` (the median prediction one second before onset, `None` without onset). `analyze_default.yaml` lists `d3d_tearing_time_to_event_dsm/tm_time_p50` after `tm_risk_1s`. Tests in `test_analyze.py` using the `wired` fixture: write a labels file with `_p10/_p50/_p90` and assert the panel kind and that the PNG is produced.
- [ ] Full suite + ruff.
- [ ] Re-publish for the pool: `pixi run -e labelmaker python -m labelmaker.run infer --models d3d_tearing_time_to_event_dsm --shot-file /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/shots_500.txt --workers 8 --force`. Record wall time and the pool medians of `tm_time_p50` and `tm_time_iqr_log` over valid rows, and the fraction of rows where `tm_gate_entropy > 0.9 ln 3` (an undecided gate). Then `analyze` on shots 187199 and 186545 with the default config into `outputs/labelmaker/analysis/` and look at the band panel.
- [ ] Card: Model details describes the 14 new series and states in one paragraph that the spread is the predictive distribution of the event time (aleatoric plus what the network learned), not epistemic uncertainty over weights, and that the upstream `rt_models*.pkl` ensembles would give the latter. Evaluation adds the pool medians and the `tm_time_p50` per-shot score summary from `analyze`.
- [ ] Commit `labelmaker: publish the survival mixture's quantiles, components and gate entropy`.

---

## Task 3: recalibrate to the reported row set (`calibrate.py`, `validate.calibration_study`, published isotonic risks)

**Why:** ECE 0.022 on all aligned pre-onset rows and 0.448 on onset-only shots are both correct; spec section 3.2 asks for two post-hoc maps with their fitting set recorded.

**Read first:** `validate._pooled_onset_rows` (Task 1), `binary_metrics`, `labels/schema.py`, `labels/store.py::write_labels`, `models/base.py::OutputField`, the tearing DSM `spec.py` after Task 2, `/projects/EKOLEMEN/survival_tm_2/data/rt_filtered_{t,e}_bms_pcb_rot.pkl` (numpy arrays pickled; readable with plain `pickle` in the pixi env; `t` in ms, `e` in {0,1}; 914,898 rows).

**Steps:**

- [ ] Tests first, `tests/labelmaker/test_calibrate.py`: `prior_shift(p, from_prevalence=q1, to_prevalence=p1)` is identity when `q1 == p1`, moves 0.5 to `sigmoid(log[(p1/(1-p1)) * ((1-q1)/q1)])`, clips inputs to `[1e-6, 1-1e-6]`; `IsotonicMap.fit(scores, truth)` reproduces the pool-adjacent-violators result on a hand-worked 8-point example (write the expected block means in the test), is non-decreasing, `apply` interpolates linearly between knots and clamps outside; `to_dict`/`from_dict` round-trip; fitting on a synthetic base-rate-shifted sample brings ECE below 0.02 on held-out draws from the same distribution.
- [ ] Implement `src/labelmaker/calibrate.py` (numpy only): `prior_shift`, `IsotonicMap` frozen dataclass (`x`, `y` knot arrays, `n_fit`, `prevalence_fit`), `fit_isotonic(scores, truth) -> IsotonicMap` (PAVA, ties in `scores` pooled first), `apply`, serialisation. Green.
- [ ] `OutputField` gains `attrs: tuple[tuple[str, str], ...] = ()` -> `LabelSpec.attrs` -> written as HDF5 group attributes in `write_labels` (string values). Test in `test_label_store.py` that an attr round-trips through `read_label`.
- [ ] `validate.calibration_study(slug, shots, paths, *, archive=TM_ARCHIVE, seed=0, timeout_s=None) -> dict`: pooled rows from `_pooled_onset_rows`; shots split 50/50 by `np.random.default_rng(seed).permutation` of the sorted aligned shot list into `fit` and `report`; per horizon label: training prevalence `q1(h)` = fraction of the 914,898 upstream training rows with `e == 1 and t <= h` (read once from the two pickles; recorded in the report with the file paths and sha256s); target prevalence `p1(h)` computed on the `fit` half for each of two row sets: `all_pre_onset` and `onset_shots_only`; isotonic fitted on the `fit` half per label per row set. On the `report` half, for each row set and each of raw / prior_shift / isotonic: `n`, `n_positive`, `auroc`, `brier`, `ece`, and the 10-bin calibration curve. Writes `write_report(paths, slug, "calibration_study", ...)` and the maps to `paths.models / slug / "calibration.json"` (atomic) with `fit_on: {shots: [...], n_rows, prevalence, row_set: "all_pre_onset", date, git_sha}` per label. Test with the `wired` fixture and a monkeypatched `_pooled_onset_rows` returning synthetic arrays (and a monkeypatched training-prevalence reader).
- [ ] Tearing DSM `spec.py`: `load` reads `model_dir / "calibration.json"` if present. Output columns 17-19 `tm_risk_250ms_isotonic`, `tm_risk_500ms_isotonic`, `tm_risk_1s_isotonic` (binary, always declared; **NaN when the file is absent**), each with `attrs=(("calibration", "isotonic, fit on all_pre_onset rows"), ("calibration_fit_on", "<json of fit_on>"))` filled at load time - since `OUTPUT_SPEC` is a module constant, build the adapter's `output_spec` in `load` via a small `with_calibration_attrs(fit_on)` helper, or put the fit-on JSON in the attrs at `write_labels` time through `Decoded` - choose the smaller change and say which in the report. Card `outputs:` gains the three; `card_discrepancies == []`. Tests: without the file the three columns are NaN; with a synthetic `calibration.json` they equal `IsotonicMap.apply` of the raw risks and the HDF5 attrs carry the fit-on record.
- [ ] `run.py` validate stage: `calibration_study` written beside `alarm_quality` for the survival slug (guarded). Test as for `alarm_quality`.
- [ ] Full suite + ruff.
- [ ] Run on the pool: `calibration_study` for `d3d_tearing_time_to_event_dsm`, then `infer --force` for the slug to publish the isotonic series, then a quick `analyze` on 187199. Record before/after ECE and Brier on both row sets for all three horizons, and the prior-shift numbers with `q1(h)` and `p1(h)`.
- [ ] Card: Bias/risks paragraph on calibration gains the measured before/after table and names the fitting set; Model details lists the three isotonic series and that the raw series remain the model's output.
- [ ] Commit `labelmaker: prior-shift and isotonic recalibration of the survival risk, fitting set recorded`.

---

## Task 4: continue training the survival model on SLURM (`d3d_tearing_time_to_event_dsm_continued`)

**Why:** the shipped pickle stopped at 20 epochs at lr 1e-5 with validation NLL still falling (0.534 -> 0.471); spec section 3.3.

**Read first:** `/projects/EKOLEMEN/survival_tm/train_tm_model.py` (split at lines 55-85: `unique_shots`, `default_rng(seed)`, `rng.choice` 80% then 10% of the remainder), the `.cfg` in `/projects/EKOLEMEN/survival_tm/` whose `output_filename_base` is `rt_fixed_rot` (find it with grep; it names `database_x_name`, expected `rt_x_pca_bms_pcb_rot`, plus the `_e`, `_t` and `_shots` names), `/projects/EKOLEMEN/survival_tm_2/train_models/auton-survival/auton_survival/models/dsm/utilities.py::train_dsm` (the loop to continue: **skip `pretrain_dsm` and the shape/scale fill**, keep everything from `model.double()` on), `.../dsm/__init__.py::fit` (elbo=True default), `tests/labelmaker/data/make_tearing_dsm_golden.py` (how the fork was imported), `registry.verify_artifacts`, the card, `outputs/labelmaker/presentation/scripts/{pool_rows.py,make_presentation.py}` and its README.

**Steps:**

- [ ] Environment: `scripts/labelmaker/phase3-requirements.txt` (torch==2.14.0 from `https://download.pytorch.org/whl/cu126`, scikit-learn, scikit-survival, tqdm, pandas, h5py, matplotlib, tabpfn==8.5.0, tifffile, pyarrow) and `scripts/labelmaker/make_phase3_env.sh` that runs `uv venv --python 3.11 /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/envs/phase3 && uv pip install --python <venv> -r phase3-requirements.txt`. Run it. Verify `python -c "import torch, sklearn, sksurv, tabpfn"` in the venv.
- [ ] `scripts/labelmaker/retrain_tearing_dsm.py`: arguments `--out-dir`, `--lr 1e-4`, `--batch-size 1000`, `--max-epochs 300`, `--patience 5`, `--seed 0`, `--fork /projects/EKOLEMEN/survival_tm_2/train_models/auton-survival`. Steps inside: (1) load x/e/t/shots from the names in the cfg; (2) rebuild the split exactly as `train_tm_model.py` does; (3) unpickle the shipped `rt_fixed_rot.pkl` with the fork on `sys.path` (full classes); (4) **gate**: `model.compute_nll(x_valid, t_valid, e_valid)` must equal the pickle's stored final validation loss (0.471) within 2e-3; print both and exit non-zero otherwise (a mismatch means the split or the x file is wrong, and nothing after it would be comparable); (5) continue the `train_dsm` loop from the existing `torch_model` with no pretraining, Adam at `--lr`, elbo=True, shuffling with `random_state=epoch` as upstream, validation NLL each epoch, keep every epoch's `state_dict` cost and reload the argmin at the end, stop when the validation cost has not improved for `--patience` epochs; (6) save `[model, train_losses, val_losses, params]` as `rt_fixed_rot_continued.pkl` (same shape as upstream's pickle so `dsm_pickle.load_dsm` reads it), plus `training.json` (epochs run, best epoch, val NLL before/after, lr, batch, seed, split counts, wall time, hostname, job id) and copy `rt_normalizations_dict.pkl` beside it. Also write a `loss_curve.png`.
- [ ] `scripts/labelmaker/retrain_tearing_dsm.sbatch`: `--partition all`, `--cpus-per-task 8`, `--mem 32G`, `--time 08:00:00`, output to `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out`, `OMP_NUM_THREADS=8`, runs the script with the phase3 venv's python, `--out-dir /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/d3d_tearing_time_to_event_dsm_continued`.
- [ ] Before submitting, run the script with `--max-epochs 1` on the login node with a 10-minute timeout to prove the gate passes (record the reproduced 0.471). Then `sbatch`. Poll `squeue -u nc1514` every few minutes and `jobstats <jobid>` at least twice; record CPU utilisation and memory high-water mark. If CPU utilisation is under 50% of 8 cores, note that 4 would do.
- [ ] While it runs: model folder `src/labelmaker/models/d3d_tearing_time_to_event_dsm_continued/` with `__init__.py`, `spec.py` that imports `INPUT_SPEC`, `OUTPUT_SPEC`, `preprocess`, `HORIZONS_MS` from the base module and defines `SLUG`, `CARD_ID = "plasmacontrol/d3d-tearing-time-to-event-dsm-continued"`, `ARTIFACTS = ("rt_fixed_rot_continued.pkl", "rt_normalizations_dict.pkl")`, `UPSTREAM` pointing at the artifact directory, and `ADAPTER` with `load` identical in behaviour (share the base `load` by parametrising it on the artifact names - a `make_load(artifacts)` in the base spec - rather than copying it). Card `README.md`: `status: implemented`, `upstream.path` = the artifact dir, `upstream.trained` = date, `training_code: scripts/labelmaker/retrain_tearing_dsm.py`, sha256 of both artifacts (fill in after the job), inputs/outputs identical to the base card, Training details = the continuation recipe and the measured curve, Evaluation = to be filled below. `PROVENANCE.json` in the artifact dir: upstream data paths with sha256, script git sha, job id, jobstats summary, val NLL before/after.
- [ ] Roster `models/README.md` gains the row (task "survival (risk at 3 horizons), retrained", status implemented). `test_registry.py` slug list updated if it enumerates slugs.
- [ ] When the job finishes: fill sha256s, `registry.verify_artifacts(slug, model_dir)` passes, `card_discrepancies == []`. Full suite + ruff.
- [ ] Publish and compare on the pool: `infer --models d3d_tearing_time_to_event_dsm_continued --shot-file shots_500.txt --workers 8`; then `alarm_quality` and `calibration_study` for the new slug; then copy `outputs/labelmaker/presentation/scripts/pool_rows.py` and `make_presentation.py` into `outputs/labelmaker/presentation_continued/scripts/` parametrised by DSM slug (a `--dsm-slug` argument; do not edit the originals) and render the seven figures into `outputs/labelmaker/presentation_continued/`, with a README whose table has three columns: shipped, continued, difference, for AUROC per horizon (all pre-onset rows), ECE on both row sets, median lead time, final-label FPR/FNR at 0.2.
- [ ] Card Evaluation for the new slug: that table and the validation NLL (0.471 -> measured). Base card Bias/risks: one sentence pointing to the continued model and whether it measured better. Spec section 7 question 5 stays open: the default is unchanged.
- [ ] Commit `labelmaker: continue training the survival model to convergence; publish as a variant` with val NLL before/after, epochs, and job utilisation in the body. Paths: `scripts/labelmaker/*`, the new model folder, roster, base card, `outputs/labelmaker/presentation_continued/`, tests.

---

## Task 5: TabPFN regressor beside the survival band

**Why:** a second Bayesian model's predictive distribution drawn next to the DSM's on the same shot (spec section 4.2). Not a like-for-like comparison and the caption says so.

**Read first:** spec section 4.2, `outputs/labelmaker/tabpfn/README.md` and its existing scripts (Phase 2 Study A/B: how archived inputs `x0/x1` were flattened into TabPFN features), `outputs/labelmaker/presentation/scripts/pool_rows.py` (how per-row arrays for the pool are built: archived inputs, truth, onset, DSM labels).

**Steps:**

- [ ] `outputs/labelmaker/tabpfn/scripts/time_to_onset_regressor.py` (runs in the phase3 venv): from the pool's aligned shots WITH an onset (86), take pre-onset valid rows; features = the CNN's archived inputs flattened as Study A did; target = `(onset_s - t) * 1000` ms. Split the 86 shots 50/50 by `default_rng(0)`. Fit `TabPFNRegressor` (`create_default_for_version("v2", ...)` as in Phase 2; no token) on the fit half (subsample to at most 10,000 rows, seeded), predict quantiles `[0.1, 0.5, 0.9]` on the report half. Read the DSM's published `tm_time_p10/p50/p90` for the same rows from the label files (Task 2). Report, on the report half: `n`, median absolute log ratio of p50 to truth for both models, 10-90 interval coverage for both, median interval width in log units for both. Save `time_to_onset_regressor.json` and `time_to_onset_regressor_rows.npz`.
- [ ] Figure `08_time_to_onset_bands_<shot>.png` for two report-half shots (the one with the longest pre-onset record and 187199 if it is in the half; otherwise the two longest): x = time, y = ms on a log axis; truth line, DSM band and median, TabPFN band and median, onset line; caption text inside the figure: "DSM: hazard model over all rows, censoring included. TabPFN: regression on pre-onset rows of tearing shots only - conditional on tearing. Drawn together to be looked at, not to be ranked." Palette and marks per the repo's `analyze.py` colours.
- [ ] `outputs/labelmaker/tabpfn/scripts/time_to_onset_regressor.sbatch`: `--partition gpu --gres gpu:1 --cpus-per-task 4 --mem 32G --time 02:00:00`. Submit, poll, record GPU utilisation and memory (TabPFN on 10k rows should take minutes; note if the GPU sat idle).
- [ ] `outputs/labelmaker/tabpfn/README.md` gains a section with the table and the figure list. Tearing DSM card Evaluation gets two sentences and a pointer.
- [ ] Commit `labelmaker: TabPFN time-to-onset regressor drawn beside the survival band`.

---

## Task 6: measure the coherent-mode mask against the 180 annotated AE shots (go/no-go)

**Why:** the whole AE label rests on `big_tf_unet`'s coherent channel agreeing with the hand annotations where they overlap. Spec section 5.8 item 1. This task decides, with numbers, and also answers spec section 7 question 3 (which spectrogram transform).

**Read first:** spec section 5 (all), `/scratch/gpfs/nc1514/aemodes/src/aemodes/pipeline/step_0a_make_spectrogram.py`, `step_1_make_semantic.py`, `src/aemodes/utils/dataset.py`, `/scratch/gpfs/nc1514/tokeye/src/tokeye/{transforms.py,hub.py,inference.py}`, `tokeye/README.md` lines 190-220, `tokeye/model/` listing. Data: `aemodes/data/.cache/step_0a/spectrograms/<shot>_{train,valid}.tif` `(4, 512, 7820)` float32 and `step_0a/stats.json`; `aemodes/data/.cache/ae_timeseries/<shot>_{train,valid}.arrow` (r0,v1,v2,v3 at 500 kHz over 0-2000 ms and `label_0..label_4` per sample); `aemodes/data/co2_250_detector.pkl` (2.9 GB; only needed if the arrow labels are missing).

**Steps:**

- [ ] Environment: use `/scratch/gpfs/nc1514/tokeye/.venv/bin/python` (torch 2.9.1+cu128, tokeye importable) read-only. If a needed package (pyarrow, tifffile) is missing there, install into the phase3 venv instead and `pip install --no-deps /scratch/gpfs/nc1514/tokeye` there (non-editable; nothing is written into the tokeye tree). State which env was used.
- [ ] `outputs/labelmaker/ae/scripts/mask_vs_annotation.py`: for each of the 180 shots, produce the coherent mask two ways: (a) **aemodes transform** - the stored tif, standardised with `stats.json` mean/std, one channel at a time, fed to `big_tf_unet` (loaded via `tokeye.hub` or `torch.load` of `model/big_tf_unet_251210.pt`) in column windows of the width tokeye's segmentation inference uses (read `tokeye/inference.py`; if it takes the full width, feed the full width), `sigmoid(out[:, 0])` = coherent probability; (b) **tokeye transform** - `tokeye.transforms.compute_stft` on the arrow time series per channel (n_fft 1024, hop 128, DC clip, percentile clip 1/99, log1p), then the same model. For each: per-frame band occupancy `occ[f] = mean over bins 164-511 of (prob > 0.2)` averaged over the 4 channels, after a first-pass notch that zeroes any bin whose `(prob > 0.2)` fraction over frames exceeds 0.8 (record how many bins per shot). Annotation per frame: `label_1|label_2|label_3|label_4` from the arrow file resampled to the 7820-frame grid by nearest sample (**label_0 excluded**, and also report `label_0` separately).
- [ ] Metrics, pooled over the 180 shots and per shot, for each transform: AUROC of `occ` against the annotation; recall of annotated-active frames at `occ >= {0.005, 0.01, 0.02, 0.05}`; precision at the same thresholds; fraction of `occ >= 0.01` frames that are unannotated, split into "inside an LFM window" and "outside any annotation"; per-shot AUROC distribution (median, 10th percentile, count below 0.6). Also the intensity-weighted centroid frequency of masked pixels per annotated-active frame, its median per shot, to see it lands in 80-250 kHz.
- [ ] Figures: `outputs/labelmaker/ae/figures/<shot>_mask.png` for 6 shots (3 best, 3 worst per-shot AUROC under the better transform): spectrogram channel r0 in the band, mask contour, annotation bar per class along the time axis, notched bins marked. Plus `occupancy_vs_annotation.png`: pooled ROC for both transforms and the per-shot AUROC histogram.
- [ ] `sbatch` file `outputs/labelmaker/ae/scripts/mask_vs_annotation.sbatch` (`gpu`, 1 GPU, 4 CPUs, 32 GB, 3 h). Submit, poll, record utilisation.
- [ ] `outputs/labelmaker/ae/README.md`: the numbers, both transforms side by side, the six figures, and a **verdict** line: GO if, under the better transform, pooled AUROC >= 0.80 and recall >= 0.70 at a threshold whose unannotated fraction outside LFM is <= 0.30; NO-GO otherwise with the failing number named. Do not soften a NO-GO. Record the chosen transform as the answer to spec section 7 question 3 and append the decision to the spec's section 5.8/7 in the same commit.
- [ ] Commit `labelmaker: measure big_tf_unet's coherent mask against the 180 AE annotations`.

**Stop point:** the controller reads the verdict before dispatching Task 7. On NO-GO, Task 7 is replaced by a report to Nathan and the roster row stays `planned`.

---

## Task 7a: AE dataset, notch rule and labels (`scripts/labelmaker/ae_dataset.py`)

**Preconditions:** Task 6 GO, its transform choice, and its occupancy threshold.

**Assumptions recorded as defaults (Nathan's section 7 questions 1 and 4):** `ae_frequency` = intensity-weighted centroid over all masked pixels in the band; the notch rule is an occupancy rule (a bin whose coherent-probability-above-0.2 fraction over the record exceeds a threshold fixed here) with a per-shot review artifact.

- [ ] `scripts/labelmaker/ae_dataset.py` (phase3 venv): for the 180 shots, using Task 6's transform and mask: (1) notch: sweep thresholds {0.5, 0.6, 0.7, 0.8, 0.9} and record per shot which bins each removes; choose the smallest threshold at which no bin inside an annotated AE window's centroid +-3 bins is removed on more than 2 shots; write `notch_review.json` (per shot: bins removed, their kHz, occupancy) and `notch_review.png` (a heat map shot x bin of removed bins); (2) frame labels: `active[f] = (mask occupancy in band >= threshold_from_task6) & annotation_ae[f]` (LFM excluded), `freq_khz[f]` = intensity-weighted centroid over masked band pixels (NaN where inactive); (3) write per-shot `.npz` under `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/ae/dataset/<shot>.npz` with `spec` (4, 348, 7820) float16 band-restricted standardised spectrogram, `active` (7820,) uint8, `freq_khz` (7820,) float32, `notched_bins`, `split` ('train'/'valid' from the pickle's own lists), and a `dataset.json` manifest (counts, class balance, the thresholds, the transform, git sha).
- [ ] A hermetic unit test for the two pure functions (`notch_bins(occupancy, threshold)`, `centroid_khz(spec, mask, bin_khz)`) in `tests/labelmaker/test_ae_labels.py`; put the pure functions in `src/labelmaker/ae/labels.py` so the test needs no torch and the training script imports them.
- [ ] Commit `labelmaker: AE frame labels from the coherent mask and annotations, notch rule fixed at <value>`.

## Task 7b: AE model and training (`scripts/labelmaker/ae_train.{py,sbatch}`, `src/labelmaker/ae/model.py`)

- [ ] `src/labelmaker/ae/model.py`: `AeSeldNet` - SELDNet body as in `aemodes/src/aemodes/models/detection/seldnet.py` (3 conv blocks with pool sizes chosen for 348 bins, e.g. `[6, 6, 2]` -> 348/72 = 4.8: pick pools that divide 348, say `[4, 3, 29]`? No - use `[6, 2, 29]` = 348 exactly, or read the frequency size and choose `pool_sizes` that divide it; state the choice), 2 bidirectional GRUs, FNN head to `(T, 2)`: activity logit and frequency in normalised band coordinate `(f_khz - 80) / 170`. Loss = `BinarySCELoss(alpha=1.0, beta=0.5)` on activity + `lambda_f * Huber(freq, target)` masked to frames the label calls active, `lambda_f = 1.0` recorded. Plain BCE variant selectable by flag. Window length read from the data (`--window-frames 710` default, the model is fully convolutional/recurrent along time so inference runs on any length). Unit test: forward on a random `(2, 4, 710, 348)` gives `(2, 710, 2)`; loss is finite with an all-inactive label.
- [ ] `scripts/labelmaker/ae_train.py`: Lightning-free plain torch loop (the phase3 venv has no Lightning; do not add it): AdamW lr 1e-4 wd 1e-4, cosine to 1e-6 over 30 epochs, bf16 autocast, batch 16 windows, early stop on validation loss patience 5, checkpoint the best; split by shot from the manifest; validation metrics each epoch: frame AUROC of activity, F1 at 0.5, and frequency MAE in kHz on active frames; two runs, `--loss sce` and `--loss bce`; both save `ae_seldnet_<loss>.pt` (state_dict + config) under `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/d3d_ae_activity_seldnet/` with `training.json`.
- [ ] `ae_train.sbatch`: `gpu`, 1 GPU, 8 CPUs, 48 GB, 6 h, two array tasks (sce, bce). Submit, poll, record utilisation; raise `--cpus-per-task` if the dataloader is the bottleneck.
- [ ] Evaluate on the 60 validation shots: against the Task 7a labels AND against the raw pickle annotation (LFM excluded), frame AUROC / F1 / frequency MAE, per loss. Write `evaluation.json` and a 6-shot figure set. The SCE-vs-BCE gap is reported as evidence about label noise (spec section 5.6).
- [ ] Commit `labelmaker: train the AE activity+frequency SELDNet, SCE and BCE variants`.

## Task 7c: labelmaker integration for the AE label

- [ ] `features/namespace.py`: `co2` feature, `kind="waveform"`, 4 channels, source `corpus` group `co2`, native rate kept (`step=0.0`); the feature store must accept a `(4, N)` float32 array of several million samples (check `store.write_features` handles the size; chunked HDF5 dataset). `resolve_corpus` returns it or the standard "absent" outcome. Tests with a synthetic corpus file.
- [ ] `models/runners/torch_pt.py`: load a `state_dict` checkpoint into `AeSeldNet` from `config` in the file; CPU inference; sha256 verified by the card as for every artifact.
- [ ] `models/d3d_ae_activity_seldnet/spec.py`: `InputSpec` with the one waveform input; `predict`: STFT with the Task 6 transform, standardise, band-restrict, notch with the fixed rule (bins recorded), run the model on the full record in windows, sigmoid -> 5-frame moving average, then aggregate to the 25 ms grid: `ae_active` = mean of frame probabilities in the window, `ae_frequency` = probability-weighted mean of the frame frequency over frames with `p > 0.5` (NaN when none). Outputs `ae_active` (binary), `ae_frequency` (regression, kHz). `DomainRule`-style validity: rows invalid when CO2 is absent, when more than 8 bins were notched in the shot, or when the window has fewer than 50 frames. Attrs: `notched_bins_khz`, `transform`, `band_khz = "80-250"`.
- [ ] Card: full model card (`status: implemented`), including the 250 kHz ceiling sentence, the LFM exclusion, the label construction, the SCE/BCE numbers, the notch rule and its threshold, and that the 180 training shots do not intersect the corpus. Roster row `implemented`. `analyze_default.yaml` gains `d3d_ae_activity_seldnet/ae_active` and `ae_frequency`.
- [ ] Run `features` + `infer` on 24 corpus shots (the sampled ones from the spec: 12 have CO2), then `analyze` on two with activity. Record coverage (how many produced labels) and the per-shot notched bins.
- [ ] Commit `labelmaker: d3d_ae_activity_seldnet - AE activity and frequency from CO2 interferometry`.

---

## Task 8a: labelmaker's own ELM survival model, with the BES ablation

**Read first:** spec sections 2.2 and 6; `/projects/EKOLEMEN/wpqh_elm_hiro/hiro_scripts/{new_train_elm_model.py,model.cfg,data_processing.ipynb}`; `/projects/EKOLEMEN/wpqh_elm_hiro/data/train_test_split_model10.pkl` (1.57 GB; keys `train_final_x_normalized`, `train_final_e`, `train_final_t`, `test_*`; columns in `current_diagnostic_order`, 124 wide); the ELM card in this repo.

- [ ] `scripts/labelmaker/elm_dsm_train.py` (phase3 venv, fork on `sys.path`): load the split pickle; **do not re-split** (its by-shot split is the one upstream validated on; the swapped-split bug was in the other trainer); two column sets: `all124` and `no_bes` (slots 0-59); `t` used as-is plus an explicit `+1` ms offset recorded in `training.json` (spec section 6: document it rather than hide it); DSM `k=3, layers=[128], LogNormal, lr=1e-3, batch=1024, dropout=0.2`, `iters=1000` with the fork's patience deciding; seed 0. Save each as `elm_dsm_<set>.pkl` (upstream pickle shape) with `training.json` under `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/d3d_elm_time_to_event_dsm/`. Evaluate both on the test split: validation NLL, and per horizon `h in {5, 10, 20, 50} ms` the AUROC and IPCW AUC of `1 - S(h + 1)` against `t <= h`, using `labelmaker.alarm.ipcw_auc` (import the repo package from the venv via `PYTHONPATH=src`). Write `ablation.json`.
- [ ] `elm_dsm_train.sbatch`: `all`, 8 CPUs, 32 GB, 6 h, two array tasks. Submit, poll, record utilisation.
- [ ] Decision rule written before the run: `no_bes` is adopted if its test NLL is within 0.02 of `all124` and its 20 ms AUROC within 0.02; otherwise report and stop (BES fetch becomes a separate decision for Nathan).
- [ ] Commit `labelmaker: train ELM survival DSMs with and without BES; ablation`.

## Task 8b: ELM adapter and card

**Precondition:** Task 8a adopted `no_bes`.

- [ ] Features: `ece` (48 channels, corpus group `ece`), `co2_slow` (the 4 CO2 chords down-sampled to 1 ms - or reuse Task 7c's `co2` waveform feature with a 1 ms mean; say which), `pcphd02`, `pcphd03` (find their corpus group; if absent in the corpus, record it as `blocked_on` and stop), `gas`, `ip`, `bt`, `pinj_total`, `tinj_total`, `ech_power_total` mapped to the model's 60 columns in `current_diagnostic_order` with the baked-in normalisation replaced by the split pickle's own mean/std (recorded in the artifact dir as `normalization.json`).
- [ ] `spec.py` for `d3d_elm_time_to_event_dsm`: `status: implemented`, `time_step_ms: 1.0`? No - labelmaker's grid is 25 ms; state in the card that the model was trained on 1 ms rows and is evaluated on labelmaker's 25 ms samples (a sampling change, not a model change), outputs `elm_risk_5ms`, `elm_risk_10ms`, `elm_risk_20ms`, `elm_risk_50ms` via `dsm_pickle.survival` at `h + 1`. Card rewritten from scaffold to implemented with the ablation table, the WPQH-phase restriction as a validity caveat (the model has only seen wide-pedestal QH rows), and `blocked_on` cleared or updated.
- [ ] `infer` on 24 corpus shots; record coverage. Commit `labelmaker: d3d_elm_time_to_event_dsm - labelmaker's own 60-column model`.

---

## Order and stop points

1 -> 2 -> 3 -> 4 (SLURM; Task 5 may start once Task 2's labels exist) -> 5 -> 6 (**stop: read the verdict**) -> 7a -> 7b (SLURM) -> 7c -> 8a (SLURM; **stop: read the ablation**) -> 8b. Final whole-branch review after the last task that ran.
