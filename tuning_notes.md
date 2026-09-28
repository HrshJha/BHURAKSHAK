# Tuning notes

## Leakage found and fixed

- Group C read `anomaly_label` in both `neighbor_anomaly_fraction` and `hotspot_density`; the old `hotspot_density` had Pearson r **0.9176** and ROC-AUC **0.9663** against `anomaly_label`. Group C now uses only an observable robust-z deformation-rate proxy when a same-time neighbor graph exists. The current corpus has one node per event, so all eight Group C features are NaN and provenance-gated out of XGBoost and Isolation Forest. See `reports/leakage_audit.md`.
- `center_mode="detected"` also used `anomaly_label`, while the feature store and pipeline explicitly requested `oracle`. Center estimation now uses current-snapshot observable displacement only; model builds reject oracle mode. The stored artifact records detected/gated mode.
- Group B persistence and change-point values were computed once from the entire event and copied to earlier windows. They now use only the prefix through each current window; future truncation tests verify the invariant.
- Group F's expected deformation comes from fixed `configs/physics.yaml` parameters, not per-event sampled simulator values. It is marked `config_derived`; deployment must provide the same static site parameters.
- DGPS, InSAR, and synthetic terrain features without a deployable source remain gated. DGPS residuals are evaluation-only.

## Tests fixed

- Pinned environment and 17 baseline failures: see `fix_leak_log.md` rows P0-ENV/P1-*.
- Label-blind builds, label permutation, causal window truncation, Group C gating, oracle refusal, provenance coverage, schema drift, and model-input gates are covered by tests.
- Two Group C expectations were updated because they encoded label-derived behavior; the changes are justified against PRD §13's observable feature inputs and the explicit corpus gate. The IF ablation test no longer expects a Group C input because this corpus has no co-temporal neighbors; the §14 spatial arm remains configured but is inactive until a valid multi-node corpus exists.

## Model comparisons

No post-leakfix baselines or tuning studies have run yet. Prior XGBoost, Isolation Forest, forecaster, ablation, domain-gap, and test numbers are superseded and stored under `reports/superseded_leaky/`. The next valid comparison must use development groups only and grouped CV.

## Test-set status

The old held-out groups are burned. The new locked corpus has not been evaluated. `reports/test_lock.json` and `reports/final_eval.md` will record the new corpus hash and its single final evaluation.

## Open items

- Rebuild the regime split with event and generating-parameter tuple exclusivity.
- Build and hash a new unseen-regime corpus under `data/heldout_locked/`; enforce that only `scripts/final_eval.py` can read it.
- Run development-only baselines, XGBoost/Isolation Forest/forecaster Optuna studies, calibration, alert-engine metrics, and the required robustness suite.
- Regenerate downstream reports, notebook outputs, and the model registry only after those checks.
