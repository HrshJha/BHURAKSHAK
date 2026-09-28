# Inference Profiling — IF + XGBoost on the Edge Path (T-080)

**PRD refs:** NFR-2, §33 (notebook 10) · **Artifacts:** `notebooks/10_inference_profiling.ipynb`, `experiments/inference_profile.json`, `reports/nb10_latency_panels.png`

## The budget question first: G-8

NFR-2 requires edge inference to run "within its compute/power budget" — but **the PRD never states a number**. That gap is recorded in the gap register (**G-8, missing numeric target**), so this report can measure and record, but cannot pass or fail anything against a budget that does not exist. Per the T-080 acceptance note, executing on actual Raspberry Pi 5 hardware is out of workstream scope; everything below is a **workstation measurement** (Apple silicon, macOS 26.6.2, arm64, CPython 3.12.13, scikit-learn 1.6.1, XGBoost 3.2.0) — an indicative order of magnitude, not an on-target result.

## What was profiled

The two §15 artifacts, trained §23-safe exactly as everywhere else in the workstream (T-068 event splits; IF on healthy-baseline train windows, threshold `p0.99` of validation healthy scores; XGBoost on groups A–F + `anomaly_score` + `physics_residual`):

- **Isolation Forest** (T-043): 23 features, pickled artifact **2.33 MB**
- **XGBoost risk model** (T-046): 42 features, 3 classes, pickled artifact **1.38 MB**

Single-window inference = one §15 feature row scored through `anomaly_score` → `predict_proba`. 2,000 individually-timed calls per stage after 50 warm-ups.

## Latency (per single window, workstation)

| stage | p50 | p95 |
|---|---|---|
| IF `anomaly_score` | 4.54 ms | 5.82 ms |
| XGBoost `predict_proba` | 0.23 ms | 0.28 ms |
| **full IF → XGBoost chain** | **4.94 ms** | **6.06 ms** |

Context, not a budget: the §10 window cadence on this corpus (measured stride 1.67 h) means one chain call consumes ≈ **0.0001%** of the time between inference requests. The IF score dominates (~90% of chain latency); XGBoost inference is negligible. If a future budget ever binds, the lever is the IF's 300-tree depth, not the risk model.

## Memory

| metric | value |
|---|---|
| peak RSS after imports (runtime floor) | 227.0 MB |
| **peak RSS with trained artifacts (deployment footprint)** | **644.6 MB** |
| inference delta after 6,000 scored windows | **+0.0 MB** (stable) |
| shipped artifact size (pickled, both models) | 3.71 MB |

Inference is allocation-stable: 6,000 scored windows moved peak RSS by exactly 0.0 MB, so the 644.6 MB figure is both the training and the serving footprint. Note the honest caveat: this workstation number includes the full Python/pandas/sklearn runtime; a constrained edge deployment would ship the 3.71 MB of artifacts into a leaner runtime, and that is precisely the on-target measurement G-8 blocks.

## Against the edge budget

| measurement | value | edge budget (NFR-2) |
|---|---|---|
| single-window chain p50 | 4.94 ms | **NOT SPECIFIED — Gap G-8** |
| single-window chain p95 | 6.06 ms | **NOT SPECIFIED — Gap G-8** |
| peak RSS (trained artifacts in memory) | 644.6 MB | **NOT SPECIFIED — Gap G-8** |
| shipped artifact size | 3.71 MB | **NOT SPECIFIED — Gap G-8** |

**Standing recommendation (also in `experiments/inference_profile.json`):** the moment the PRD's owner supplies one number for NFR-2 — a p95 latency, an RSS ceiling, or a power draw — re-run `notebooks/10_inference_profiling.ipynb` (idempotent) and this table gains its pass/fail column. Until then, no verdict is claimed here.
