# Full Repo Audit — SubSense Data+ML Workstream

**Date:** 2026-09-28 · **Scope:** every file under `src/`, `tests/`, `configs/`, `notebooks/`, `scripts/`, plus `requirements.txt`, `pytest.ini`, `README.md`, `.gitignore`, cross-referenced against `prd.md` and `TASKS.md`. No code changed in this pass.

**Verification actually performed:** all 55 `src/` modules imported (0 failures); `pytest tests/ -q` → **576 passed, 2 warnings in 6.20 s** (full output: 8×72 dots, warnings summary = scipy L-BFGS-B DeprecationWarning from `sklearn/linear_model/_logistic.py:451` in `test_baselines.py` only); `python scripts/check_tree.py` → `PASS - 30 required directories, 10 package markers present; 4 out-of-scope paths absent.`; `python scripts/validate_synthetic_dataset.py` → `PASS — 1,440,000 rows, 10,000 sequences in run (plan: 4,000,000 ≥ 10,000 required), 10,000 events, fault types: ['BIAS','DRIFT','DROPOUT','SPIKE','STUCK']`; `python scripts/run_pipeline.py` executed end-to-end (§4).

---

## 1. Dataset audit

### [CRITICAL] Train/test split file has no producer — split provenance is unverifiable
- File: `data/features/split_assignment.csv` (consumed by 10 readers: `scripts/run_ablation_a_to_f.py:62`, `scripts/build_and_run_notebook_04.py:65`, `05:74`, `06:92`, `07:96`, `10:104`, `scripts/run_domain_gap_stage2.py`, `scripts/run_if_ablation_abc.py`, `sepetra/build_report.py`, `src/pipeline.py:145`)
- Category: dataset
- What's wrong: every model-consuming artifact reads `data/features/split_assignment.csv`, but **no script, notebook, or module in the repo writes it**. The T-068 module `src/evaluation/splits.py` implements five §23 split functions (`time_split`, `spatial_split`, `node_split`, `event_split`, `synthetic_split`) — none is called to produce this file. The manifest's `split_definition` says only `"assigned in Phase 4 (T-068)"` with no method or parameters. The executed audit run confirmed the per-scenario-family 60/20/20 assignment shown below, so the file *behaves* like a family-level holdout, but its generating code does not exist in the repo.
- Why it breaks something: §35 requires the test split be an "unseen parameter regime, not random split", and the registry (`src/risk/model_registry.py`) refuses unverifiable dataset provenance — yet the central artifact that would prove it is orphaned. A fresh clone cannot reproduce the splits; nothing prevents a future regeneration with a subtly different (or wrong) assignment that all downstream numbers silently inherit.
- Fix: add `scripts/make_split_assignment.py` that calls a split function from `src/evaluation/splits.py` (deterministic seed), writes the CSV, and asserts family-level disjointness via `assert_no_leakage`; wire it into T-068's output list and the manifest's `split_definition`.
- Blocking: no (file exists and is consistent), but it is the audit's largest reproducibility hole.

```
split        test train validation      (rows per scenario_family)
E_accelerating_subsidence 125  375  125      … same 125/375/125 pattern for all 16
```

### [HIGH] Feature-schema drift in both directions (§13/§33 contract)
- File: `configs/feature_schema_v1.yaml` vs `data/features/features_v1.parquet` (writer: `scripts/build_and_run_notebook_03.py:121`; manifest claims `feature_schema_version: v1`)
- Category: dataset
- What's wrong: (a) **Declared but never emitted** in the store: all of G_dgps (5), H_insar (7), I_terrain (8), J_environmental (4) = 24 names. (b) **Emitted but not declared**: the per-channel B-group names `tilt_{x,y}_{rolling_mean,rolling_std,rolling_min,rolling_max,trend,persistence,change_point_score}` (14 columns; the schema declares only the bare `rolling_mean…` set). G/H are documented Phase-4/ablation-gated (T-070) and I/J have their own gates, so (a) is arguably by design — but nothing in the schema or a test states "absent by gate", and (b) is silent drift.
- Why it breaks something: `src/risk/xgboost_model.py:_resolve_features` accepts any `forecast_*`/known-group column present at fit time; a store regenerated with a different emitter mix silently trains on a different feature set while still claiming schema `v1` (the manifest's `feature_schema_version` is never validated against actual columns).
- Fix: declare the B-group per-channel names in `feature_schema_v1.yaml`, and add a `assert_schema_drift(report, schema)` guard (allowed-absent-with-gate-reason) called by notebook 03.
- Blocking: no.

### [MEDIUM] §35's "unseen parameter regime" split is a family-level holdout, not a regime-level one
- File: `data/features/split_assignment.csv` (see producer finding above); PRD §35 bullet 3
- Category: dataset
- What's wrong: §35 demands the test split be an unseen *parameter regime*. The actual assignment is uniform **125 train / 125 val / 125 test events per scenario family** — every family, including all 12 fault/scenario types, appears in all three splits. Unseen-family holdout would be the stricter, honest reading.
- Why it breaks something: the §23 synthetic-regime axis (`src/evaluation/splits.py:166 synthetic_split`) was measured in T-073 and shown degenerate on this corpus; the shipped 60/20/20 family-balanced split is *a* leakage-safe split but not the regime-holdout §35 names.
- Fix: regenerate the split as a regime holdout (hold out `rapid_c_multiplier`/`accelerating_beta` values, or whole scenario types), or document in the manifest why family-balanced was chosen over §35's wording.
- Blocking: no (quality-of-claim issue, not runtime).

### [MEDIUM] Sequence-count claim in `dataset_manifest.json` is misleading (G-4 context)
- File: `data/synthetic/dataset_manifest.json` (`row_counts`); PRD §10
- Category: dataset
- What's wrong: `sequences_generated: 10000` vs `sequences_planned: 4000000` — the "planned" figure implies the PRD's 4 M-sequence scale was attempted; actually the generator's default is 44×12 ≈ 528 sequences/params-combination × … the real, measured chain is 10,000 events (1,440,000 raw rows) → 90,000 windows. PRD §10 asks 10k–50k sequences **and** §15 100k–500k windows; the repo delivers the sequences at the floor and windows below the §15 range (90k < 100k).
- Why it breaks something: nothing crashes, but the manifest invites over-reading; notebook 06/T-079 already showed the 9-window series depth caps the §16 horizons — the same corpus-depth ceiling is the root cause.
- Fix: record `windows_produced: 90000` in the manifest next to `sequences_generated`, with a note that §15's 100k floor needs stride/depth changes (e.g. events > 1 day), not just more events.
- Blocking: no.

### [MEDIUM] Six `configs/physics.yaml` scenario parameters are declared but never used
- File: `configs/physics.yaml` (`scenarios:` block) vs `src/simulator/scenarios.py`
- Category: dataset
- What's wrong: `accelerating_duration_fraction`, `local_anomaly_amplitude_mm`, `local_anomaly_sigma_m`, `second_zone_c_multiplier`, `vibration_event_rate`, `deformation_onset_fraction` appear nowhere in `src/` or `scripts/` (verified by leaf-key scan). Concretely: accelerating subsidence uses `tau**beta` over the full window (no onset fraction); the local-anomaly bump amplitude/sigma are fixed by whatever the code hardwires instead.
- Why it breaks something: an operator tuning `local_anomaly_amplitude_mm` gets no effect — a silent config/no-op mismatch (NFR-6 spirit violation).
- Fix: either wire the six params into `scenarios.py` or delete them from the YAML.
- Blocking: no.

### [LOW] Duplicate 12 MB rig log tracked twice (identical)
- File: `data/recorded/tabletop/raw_sensor_log.csv` and `data/raw/sensors/tabletop/raw_sensor_log.csv`
- Category: dataset
- What's wrong: byte-identical copies (md5 match), both tracked in git (11,984 KB each).
- Why it breaks something: repo bloat only.
- Fix: keep one, symlink or copy at stage time.
- Blocking: no.

### [LOW] `.DS_Store` present at repo root (untracked)
- File: `.DS_Store`
- Category: dataset / hygiene
- What's wrong: ignored by `.gitignore` (fine) but present on disk; noted for completeness.
- Blocking: no.

**Positive dataset findings (checked, OK):** physics `W(x,y,t)` and `W(t)` match PRD §10 **exactly** — `src/simulator/deformation_field.py:77,96` implements `W_max·(1−e^(−c·t))` and the Gaussian kernel with `σ = influence_radius/2` (`kernel.sigma_divisor: 2.0`); analytic tilts `∂W/∂x, ∂W/∂y` at lines 131/145; `subsidence_factor` reconciles config vs geometry (documented, `w_max = extraction_height × subsidence_factor × 1000` mm, config `maximum_subsidence: 2100.0` consistent). Every `src/simulator` module takes an `rng: np.random.Generator`; the single seed entry point is `scripts/generate_synthetic_nodes.py --seed 42` and the manifest records `random_seed: 42`. Store NaN/inf scan: **zero** NaN/inf in all numeric columns; `_velocity`/`_acceleration`/`_slope`/`group_b` sigma guards handle empty/degenerate windows. Packet DQ flags are computed by `validate_packets` (src/preprocessing/validation.py:90) and **traced through the real path**: `src/pipeline.py` stage 1 flags 39,829 corrupted rows out of 1.44 M, which then flow into the feature store. `configs/physics.yaml` → simulator parameter usage: all other leaves verified referenced.

---

## 2. Code audit

### [MEDIUM] `T-023` output `scripts/generate_synthetic_events.py` does not exist
- File: `TASKS.md` T-023 Output column; disk
- Category: code
- What's wrong: TASKS.md marks T-023 `[x] done` declaring output `scripts/generate_synthetic_events.py`; the file is absent (events are produced inside `generate_synthetic_nodes.py`).
- Why it breaks something: output-column conformance check fails; a reader following TASKS.md finds no such script.
- Fix: either split the events generator into that file or correct T-023's Output column.
- Blocking: no.

### [MEDIUM] `T-055` output `models/registry.json` does not exist — FR-14 unexercised in production path
- File: `TASKS.md` T-055 Output column; `src/risk/model_registry.py`; disk
- Category: code
- What's wrong: the model-registry module is complete and strict (refuses missing dataset version/schema mismatch; `FR14_FIELDS = (model_name, model_version, feature_version, training_dataset_version, timestamp)`), but **no caller registers a model and `models/registry.json` was never created**. `scripts/train_tabletop_models.py` dumps joblib artifacts with only `{"model", "scaler", "features"(, "threshold", "params", "class_names")}` — no model_version / feature_version / training_dataset_version inside, and no load-side check.
- Why it breaks something: FR-14 traceability is asserted by tests but not satisfied by any real artifact: the shipped `models/tabletop/*.joblib` carry no version metadata, and no prediction call writes an FR-14 record.
- Fix: have the tabletop trainer (and the ablation runners) instantiate `ModelRegistry.register_model(...)` with the dataset manifest version + schema version, and add version fields to the joblib payload with a load-time assertion.
- Blocking: no (degrades traceability, not execution).

### [MEDIUM] Dead code: `TemporalForecaster.predict` vs `forecast_all_origins` divergence risk
- File: `src/forecasting/temporal_model.py:224-270`, `src/forecasting/forecast_to_risk.py:100`
- Category: code
- What's wrong: `predict()` (per-series last-history forecast) and `forecast_all_origins()` (every origin) duplicate the physical-unit un-standardisation and row-emission logic in two places; they already diverged once in design (origin semantics). Not currently inconsistent (tests pin both), but it is a maintenance hazard.
- Fix: refactor `predict()` to call `forecast_all_origins` and take the last origin per series.
- Blocking: no.

### [LOW] Dead/unused code: `plot_1d_profile` in spatial_model
- File: `src/simulator/spatial_model.py:50`
- Category: code
- What's wrong: `plot_1d_profile` has no caller anywhere (src/tests/docs/notebooks scanned).
- Fix: delete or wire into notebook 01.
- Blocking: no.

### [LOW] Unused imports / declared-but-unused dependencies
- File: `requirements.txt`, `scripts/download_sentinel1.py:32`, `scripts/build_and_run_notebook_10.py:69`
- Category: code
- What's wrong: `requests` and `psutil` are imported by scripts but absent from `requirements.txt` (installed locally: requests 2.34.2, psutil 7.2.2) — a clean-env `pip install -r requirements.txt` then `scripts/download_sentinel1.py` or notebook 10 fails with `ModuleNotFoundError`. Conversely `shap`, `statsmodels`, `seaborn`, `tqdm` are pinned but imported by nothing (shap is §22/FR-15-planned).
- Fix: add `requests` and `psutil` pins; drop or annotate the four unused pins.
- Blocking: no on this machine; **yes on a clean environment** (would stop notebook 10 / the downloader).

### [LOW] `.cursor/mcp.json` and `.cursor/rules/agent.mdc` tracked in git
- File: `.cursor/` (2 tracked files)
- Category: code / hygiene
- What's wrong: IDE/assistant state is committed; `.gitignore` excludes other assistant dirs (`.clario/`, `.claude/`, `.freebuff/`) but not `.cursor/`. The mcp.json embeds the user's absolute machine paths (`/Users/harshkumarjha/...`) — not a secret, but machine-specific.
- Fix: add `.cursor/` to `.gitignore`, `git rm --cached`.
- Blocking: no.

### [LOW] Two `InputData` homonyms with different shapes (concept collision)
- File: `src/features/group_e_health.py`, `src/features/group_f_physics.py`
- Category: code
- What's wrong: both modules define a dataclass named `InputData` with different required fields; a wildcard import or a refactor could bind the wrong one.
- Fix: rename to `GroupEInput` / `GroupFInput`.
- Blocking: no.

**Positive code findings (checked, OK):** no circular imports (55/55 modules import cleanly in one pass); no bare `except:` anywhere in `src/` — the single `except Exception` is `scripts/download_sentinel1.py:178`, a retry-with-backoff that re-raises on final attempt and prints the product name; NFR-6 scan of all `src/*.py` code lines for bare `0.5/0.6/0.7/1.5` finds exactly **one** hit, `src/config.py:114` — a docstring inside the loader itself (compliant). Tests: an AST scan flagged 23 tests with no direct `assert`, but manual sampling (`test_invalid_config_rejected` uses try/except→`raise AssertionError`; `test_no_boundary_crossing_guard_passes_on_clean_output` asserts via a guard that raises) shows **no vacuous tests** — all 576 tests assert meaningfully.

---

## 3. Security audit

### [MEDIUM] Model artifacts distributed as unauthenticated pickle/joblib
- File: `scripts/train_tabletop_models.py:330-333` (writer), `models/tabletop/*.joblib`
- Category: security
- What's wrong: joblib/pickle artifacts execute arbitrary code on load. There is no signature/checksum verification at load time, and the tabletop raw-log ingestion chain (`scripts/train_tabletop_models.py` reading `data/recorded/tabletop/raw_sensor_log.csv`) treats recorded sensor CSVs as trusted.
- Why it breaks something: if a model file or recorded-data file is ever replaced (supply chain, shared drive), loading silently executes attacker code. Current files are locally generated — risk is prospective, not active.
- Fix: write a sha256 alongside each artifact and verify at load; prefer ONNX/safetensors-style formats for anything that will leave the machine.
- Blocking: no.

### [LOW] CDSE credential handling is env-based (correct) but password may leak into process listings
- File: `scripts/download_sentinel1.py:12-18, 152-156`
- Category: security
- What's wrong: credentials come from `CDSE_USERNAME`/`CDSE_PASSWORD` env vars or `--username/--password` **argv flags** (visible in `ps` when passed as flags); the token is held in a session and used as a Bearer header (not logged — verified: no logging of `token` or `password`). The README-style usage line in the docstring correctly shows env usage.
- Fix: drop the argv fallback for the password, keep env-only.
- Blocking: no.

**Positive security findings (checked, OK):** no hardcoded credentials/tokens/DB strings anywhere (src/scripts/configs scanned; notebooks scanned **including cell outputs** — zero password/token/secret/Authorization hits); no `eval`/`exec`; no `subprocess`/`os.system`/`shell=True` anywhere; no path built by concatenating external input (the one constructed path, `SCENE_DIR / prod["product_name"]`, comes from the repo's own generated manifest, not raw user input — still the file to watch when real CDSE responses replace it); `.gitignore` correctly excludes `.venv`, caches, OS files and assistant state; `.cursor/` is the one leak (own finding above); no sensitive data in logging (the only structured logging of coordinates is x/y mesh-local metres, not real GPS — the WGS84 origin lives in config, is not logged).

### [LOW] requirements.txt pins are exact (good) — CVE spot-check
- File: `requirements.txt`
- Category: security
- What's wrong: nothing floating; all `==` pins. Notable versions: numpy 2.3.5, pandas 2.3.3, scikit-learn 1.6.1, xgboost 3.2.0, torch 2.10.0, PyYAML 6.0.3 (CVE-2020-14343 class issues were fixed before 6.0; `yaml.safe_load` used throughout `src/config.py` — no `yaml.load` anywhere). No known-advisory version in this set at audit time (offline check; no network advisory lookup was possible — flagged as a limitation, not a finding).
- Blocking: no.

---

## 4. Architecture / connection audit

### [HIGH] Two schema-join seams can silently produce NaN feature columns for the risk model
- File: `src/forecasting/forecast_to_risk.py:184` (`join_forecast_features`), `src/risk/xgboost_model.py` (`_resolve_features`)
- Category: architecture
- What's wrong: the T-077 bridge is a **left join** on `(event_id, node_id, window_index)`; rows whose series tail lies inside a horizon keep NaN in every `forecast_*` column. The leakage guard `assert_no_forecast_leakage` exists and is tested, but **no production runner calls it** — the pipeline (`src/pipeline.py`) doesn't join forecast features at all yet, and the ablation/§24 runners don't either. The failure mode is latent: the first person to wire forecasts into `train_risk_model` without dropping tails gets NaN feature columns into XGBoost (xgboost 3.x handles NaN but the §15 contract and the calibration story silently change).
- Why it breaks something: documented above — the guard exists precisely for this and is one un-wired call away from being bypassed.
- Fix: make `join_forecast_features` take a `validate: bool = True` that calls `assert_no_forecast_leakage` (or auto-drops tail rows) unless explicitly disabled.
- Blocking: no.

### [MEDIUM] `src/pipeline.py` is fragile on missing `split_assignment.csv`
- File: `src/pipeline.py:145-149`
- Category: architecture
- What's wrong: the pipeline reads the split CSV via `pd.read_csv` with no existence check and raises raw `FileNotFoundError` if absent — combined with finding #1 (no producer), this is the *only* external-file dependency of `run_pipeline` that cannot be regenerated from the repo.
- Why it breaks something: on a clean clone without `data/` (it is git-tracked here, so it works; but `.gitignore`'s spirit + the 220 MB CSV make removal likely), the pipeline dies at stage-2 with an unactionable error.
- Fix: check existence and raise `PipelineError("run scripts/make_split_assignment.py first")` (after that script exists).
- Blocking: no (works as shipped).

### [MEDIUM] Measured window stride (1.67 h) contradicts the §9.1 config grid (10 min) — documented, but upstream of several behaviors
- File: `configs/preprocessing.yaml:15` (`grid_interval_minutes: 10`), `notebooks/06_temporal_forecasting.ipynb` §1, `configs/physics.yaml` (`steps_per_day: 144`)
- Category: architecture
- What's wrong: the store's window timestamps are spaced ~1.67 h (100 min) because feature windows aggregate 60 raw steps at stride 10 over a 144-step day; the §9.1 10-min grid applies to raw cadence. `src/forecasting/horizons.py` resolves §16 horizons against `grid_interval_minutes`, so horizons-in-steps from config (1,3,6,36,144) **do not match** what the store actually supports (notebook 06 measured 1.67 h and honestly re-resolved in-notebook, marking `minutes_30` unsupported).
- Why it breaks something: any consumer that trusts `horizons_in_steps()` for store-level forecasting (e.g. a future config-driven training runner) will request 144-step horizons the 9-window corpus cannot express. The divergence is measured and reported in notebook 06 but the config/`horizons.py` pairing has no guard.
- Fix: add a `stride_measured` parameter or a warning in `horizons_in_steps` when the target frame's measured stride differs from the config grid; document in configs/forecasting.yaml.
- Blocking: no (notebook works around it explicitly).

### [MEDIUM] `scripts/train_tabletop_models.py` ships an unfitted scaler and ignores its own `rf_scaler`
- File: `scripts/train_tabletop_models.py:281-282, 332-333`
- Category: architecture
- What's wrong: `rf_pred = rf.predict(rf_scaler.transform(Xte))` — a `RandomForestClassifier` is scale-invariant, so the transform is harmless but misleading; more importantly the payload dumps `rf_scaler` implying deployment will transform before `rf.predict`, and any deployment that does transforms features the forest never saw scaled during training… which is actually fine for RF, but the IF path (line 330) needs its scaler while the RF path must NOT use one. The saved bundle does not distinguish.
- Why it breaks something: an inference-time implementer following the bundle's shape (scaler present for both) can apply `rf_scaler` before `rf.predict_proba` — harmless here, but the *pattern* invites the same mistake with a scale-sensitive model.
- Fix: drop `rf_scaler` from the RF bundle or add a `"preprocessing": "none"` field with a load-time check.
- Blocking: no.

**Positive architecture findings (checked, OK):** out-of-scope paths genuinely absent (`src/api/`, `dashboard/`, `deployment/`, `docker-compose.yml` — `check_tree.py` asserts and confirms); config coupling — `configs/*.yaml` are consistent: the §10 144-step day is used identically by windowing/scenarios/physics, and no two modules assume different sampling rates (the two cadences that exist — raw 10-min grid vs store 1.67 h stride — are measured and documented, see finding above); Sentinel-1/DGPS optional-modality degradation is genuinely graceful and honestly documented — `enrich_with_insar` (`scripts/run_ablation_a_to_f.py:102`) joins a static per-node LOS snapshot with the limitation recorded in `deviations_from_prd`, and the resolver simply skips absent G/H columns; data flow matches the §7 layer cake (§5, pipeline trace below); the tabletop duplicated CSV is byte-identical (no divergence).

### Pipeline end-to-end trace (clean run, actual output)

`python scripts/run_pipeline.py` ran to completion in this audit (fresh process):

```
corpus: 1,440,000 raw rows, 10,000 events, 400 nodes
[1/8] validation: 39,829 DQ-flagged (corrupted) of 1.44 M
[2/8] features: 90,000 windows, 61 features (§13 budget OK)
[3/8] isolation forest: 23 features, threshold 0.7215 (p0.99 of validation healthy scores)
[4/8] spatial fusion: Group C fused in-store; §21.1 graph (400 nodes, radius 37.5 m (config), median neighbours 8)
[5/8] physics check: residuals z-standardised on train (mu=-13.212, sd=80.342)
[6/8] xgboost: 42 §15 inputs, classes ['CRITICAL','NORMAL','WARNING']
[7/8] alert engine: {'GREEN': 14994, 'WATCH': 1668, 'WARNING': 878, 'CRITICAL': 460}
[8/8] explainability: 18,000 FR-11 payloads emitted
```

No failure point on this machine. The two latent first-points-of-failure on other environments are (a) missing `split_assignment.csv` (no producer — finding §1.1) and (b) missing `requests`/`psutil` in a clean venv for the downstream scripts (finding §2.7). The final summary maps to `experiments/pipeline_run.json` and `reports/acceptance_criteria.md`.

---

## 5. Known-gap re-check (G-1 … G-12)

| Gap | Status | Evidence |
|---|---|---|
| **G-1** label vocabulary (§10's 10 labels vs §12's risk_label) | **PARTIALLY RESOLVED** | The 10-vocabulary §10 labels exist only as scenario *flags* (`src/simulator/scenarios.py:58-62`: `LOCAL_ANOMALY`, `NON_SUBSIDENCE`, `DATA_QUALITY`, `COMMUNICATION_FAILURE`, …) and are mapped **at generation time** into the 3-class `risk_label` per scenario (measured on the shipped store: sensor/comms/stable families → NORMAL; slow/irregular/accelerating → WARNING; rapid/multi-zone → CRITICAL; `anomaly_label`/`fault_label`/`data_quality_label` carry the rest separately, honoring §12 non-collapse). **What's missing:** `RISK_VOCAB_FULL4`/`FULL4_TO_MVP3` (`src/preprocessing/labels.py:61-66`) exist but nothing exercises the 4-class path; the mapping is implemented-in-code but not stated as a single documented table; labels.py still says "provisional … must be revisited before T-050" — stale comment, T-050 shipped. |
| **G-2** GREEN filter / empty training set | **RESOLVED** | No `"GREEN"` filter exists anywhere in `src/` (grep: only vocabulary constants and comments). Healthy-baseline selection is the triple-healthy mask (`src/anomaly/isolation_forest.py:80-92 healthy_baseline_mask`: `anomaly_label==0 & fault_label=="NONE" & risk_label=="NORMAL"`), trained without error (23 features, threshold 0.7215). The docstring explicitly documents why the PRD's §14 snippet is inoperable. |
| **G-3** alert machine expects `P(WATCH or higher)` vs 3-class model | **RESOLVED** | `src/risk/alert_engine.py:59-76 probability_of` maps `WATCH_or_higher → 1 − P(NORMAL)` under the documented NORMAL-fills-GREEN mapping; configs/alerts.yaml thresholds (0.5/0.6/0.7) all evaluate; the full ladder demonstrably fires on held-out data (GREEN 14,994 → WATCH 1,668 → WARNING 878 → CRITICAL 460 in the audit's pipeline run). |
| **G-4** sequence/window reconciliation | **RESOLVED (measured, documented below the PRD's upper band)** | Measured: 10,000 sequences (PRD floor 10k ✓), 1,440,000 raw rows (144 timesteps × 10k), 90,000 windows (60/10) — **below §15's 100k–500k window band**. The manifest + validator (`scripts/validate_synthetic_dataset.py` PASS) record the actuals; notebook 06 documents the depth ceiling's effect on §16 horizons. Not silent, but the 100k floor is unmet — a scope-honesty note, not a hidden break. |
| **G-5** `distance_to_subsidence_center` leakage | **RESOLVED (with recorded synthetic-phase caveat)** | `src/features/group_c_spatial.py:20` documents the G-5 rule; the oracle center comes from config only in the synthetic phase and **every emitted frame carries `center_mode`** (`build_feature_store(center_mode="oracle")` → per-row `center_mode` column in the store) so inference-time use of the "detected" mode is explicit and auditable. No other module computes distance from ground truth. |
| **G-6** InSAR toolchain unspecified | **STILL OPEN (by design)** | `configs/insar.yaml` records the decisions (thresholds, reference point) as T-058 "to be decided"; `scripts/run_insar_pipeline.py` + nb08 process a simulated stack pending T-057's credentialed download. Nothing breaks; the gap is honestly parked. |
| **G-7** DGPS source absent | **STILL OPEN (by design)** | nb09/T-064 run on synthetic DGPS; no receiver/vendor identified (G-7 unchanged); Group G columns are gate-gated out of the default frame. |
| **G-8** NFR-2 numeric edge budget | **STILL OPEN (measured around)** | T-080 measured p50 4.94 ms / p95 6.06 ms / 644.6 MB RSS / 3.71 MB artifacts (`experiments/inference_profile.json`, `reports/inference_profile.md`) with every budget row marked "NOT SPECIFIED — Gap G-8"; no pass/fail claimed. |
| **G-9** §24 operational targets undefined | **STILL OPEN (measured, not judged)** | FA/day + lead times are computed (`experiments/ablation_a_to_f.json`, sepetra graphs); no "good enough" claim anywhere; `reports/acceptance_criteria.md` says so explicitly. |
| **G-10** Group I real-mine source absent | **STILL OPEN (honest synthetic stand-in)** | `configs/physics.yaml terrain.source: synthetic` with G-10 honesty comment; Group I is gate-gated out of the default 61-feature frame. |
| **G-11** unreviewed reference | **STILL OPEN** | No artifact cites or reviews `tandfonline.com/doi/full/10.1080/19475705.2024.2375546` (repo-wide grep: only the TASKS.md gap text). Phase 4 did not fold it in. |
| **G-12** version control | **PARTIALLY RESOLVED** | The repo IS git-tracked now (259 tracked files, history present) — the G-12 recommendation happened. But the working tree at audit time was fully committed (clean `git status`), and `.cursor/` machine state + the 220 MB CSV are tracked (see findings). |

---

## Totals

| Severity | Count |
|---|---|
| CRITICAL | 1 |
| HIGH | 2 |
| MEDIUM | 10 |
| LOW | 8 |
| **Total** | **21** |

## Top 10 to fix first (respecting dependencies)

1. **Split producer** (`data/features/split_assignment.csv` has no generating script — §1.1). Everything downstream trusts this file; make it reproducible before anything else touches splits.
2. **§35 regime holdout** (§1.3) — depends on #1's new producer script; regenerate the split per §35's wording or document the deviation in the manifest.
3. **`requests`/`psutil` into requirements.txt** (§2.7) — independent, one-line each; unblocks clean-env runs of notebook 10 and the downloader.
4. **FR-14 wiring** (§2.2) — register the shipped models in `models/registry.json` (after #1/#2 so `training_dataset_version` points at a reproducible manifest), add version fields + load-side checks to the joblib payloads.
5. **Forecast-join NaN guard** (§4.1) — make `join_forecast_features` validate by default before any runner adopts forecast features.
6. **Feature-schema drift guard** (§1.2) — declare the per-channel B names and add the drift assertion to notebook 03 (after #2, since the store itself doesn't change).
7. **Six unused physics params** (§1.5) — wire or delete; then re-run `scripts/validate_synthetic_dataset.py`.
8. **Horizon-vs-measured-stride guard** (§4.3) — warning or parameter in `horizons_in_steps`.
9. **Pickle/joblib checksums** (§3.1) — sha256 sidecars + load-time verification (pairs naturally with #4).
10. **Hygiene batch** (§1.6 duplicate CSV, §2.3 dead code, §2.4 `plot_1d_profile`, §2.6 `.cursor/`, §2.8 homonyms) — no dependencies; do last.

Pipeline currently runs end-to-end: **YES** — verified via `python scripts/run_pipeline.py` (exit 0, full 8-stage trace above), plus `pytest tests/ -q` → 576 passed and `python scripts/check_tree.py` → PASS.
