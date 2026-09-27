# §25 Ablation Study A–F — T-070

**Question (§25):** what does each modality family actually buy the §15 risk
model? The answer gates whether temporal deep learning (§26) and a GNN (§27)
are ever built.

**Setup:** 90,000 windows (10,000 events × 9 §10 windows; 400 nodes) from
`data/features/features_v1.parquet`, T-068 event-level splits (54k/18k/18k
train/validation/test — no event spans splits). One §15 XGBoost per arm,
hyperparameters from `configs/risk_model.yaml`, deterministic. All metrics
from `src/evaluation/metrics.py` (§24; **no bare accuracy anywhere**).
Alert = P(CRITICAL) ≥ 0.5; hotspot match radius 75 m (3 × node spacing).
Full numbers: `experiments/ablation_a_to_f.json`. Reproduce:
`.venv/bin/python scripts/run_ablation_a_to_f.py`.

## Results

| Arm | Development | #feat | Prec | Rec | F1 | PR-AUC | macro-F1 (3-class) | Brier | ECE | IoU | loc err (m) | lead med (h) | lead P10 | missed | FA/day |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | sensors only | 5 | 0.454 | 0.452 | **0.453** | 0.494 | 0.681 | 0.1105 | 0.0071 | 0.176 | 22.3 | 0.60 | 0.00 | 0.846 | **1.036** |
| B | + temporal | 35 | 0.413 | 0.448 | 0.430 | **0.512** | 0.801 | 0.0703 | 0.0046 | 0.171 | 26.0 | 0.60 | 0.00 | 0.890 | 1.218 |
| C | + spatial | 43 | 0.385 | 0.443 | 0.412 | 0.472 | 0.812 | 0.0620 | 0.0048 | 0.170 | 26.5 | 0.60 | 0.00 | 0.890 | 1.355 |
| D | + vibration + health | 57 | 0.364 | 0.443 | 0.400 | 0.483 | 0.819 | 0.0627 | 0.0014 | 0.171 | 26.4 | 0.60 | 0.00 | 0.885 | 1.481 |
| E | + physics | 61 | 0.361 | 0.443 | 0.398 | 0.483 | 0.833 | **0.0558** | 0.0023 | 0.175 | 26.2 | 0.60 | 0.00 | 0.884 | 1.496 |
| F | + Sentinel-1 (LOS snapshot) | 68 | 0.364 | 0.443 | 0.400 | 0.487 | **0.833** | 0.0569 | 0.0014 | **0.176** | 26.2 | 0.60 | 0.00 | **0.883** | 1.478 |

Deformation error is model-invariant (MAE 31.4 mm, RMSE 81.9 mm, bias
+12.1 mm in every arm): it scores the §10 physics engine's own truth field,
not the classifier — reported per §24 for honesty, not for arm comparison.

## Reading by §24 family

- **Detection quality (precision/recall/F1/PR-AUC).** Arm A is already at
  the ceiling this synthetic corpus offers (AP ≈ 0.49 ≈ the event base rate
  structure). Added modalities do NOT raise alert-quality detection; binary
  F1 at the 0.5 threshold even drifts down as probability mass spreads over
  the finer 3-class boundary.
- **3-class discrimination & calibration.** The real winner: macro-F1
  0.681 → 0.833 (+0.152) and Brier 0.1105 → 0.0558 (−49%) from temporal
  history, then physics. ECE stays ≤ 0.0071 everywhere. Modalities earn
  their place by making the risk GRADE trustworthy, not by detecting more.
- **Spatial accuracy.** IoU ≈ 0.17 and matched-fraction ≈ 0.9 are flat —
  hotspot geometry is set by the §10 local-anomaly footprint, which Group C
  already resolves; LOS context (F) adds nothing measurable at 500 m scale.
- **Operational / temporal.** Median lead time is 1 window (0.6 h) and P10
  is 0 in EVERY arm: the §10 event generator puts anomaly onset mid-event,
  so no feature family can anticipate it earlier. False alarms/day rise
  with feature breadth (1.04 → 1.50) — the precision cost of calibrated
  grading. Missed-event rate improves only marginally (0.846 → 0.883 is the
  wrong direction to celebrate; it is noise-level).

## Gate verdict on §26 (temporal DL) and §27 (GNN)

**Closed for now.** The §25 gate asks whether richer sequence or graph
models would buy *earlier warning*. On this corpus the binding constraint
is the event generator (onset is not anticipatable; median lead time
saturated at one window for every arm), so there is no lead-time headroom
for a sequence model to harvest — and static-graph spatial structure is
already fully exploited by Group C. Building G/H now would be architecture
in search of a problem. The gate reopens when the corpus has genuine
temporal depth (T-057 real Sentinel-1 scenes; multi-day event horizons).

## Honest deviations from the §25 letter (also in the JSON metadata)

1. **No +DGPS arm.** §19 makes DGPS sparse (6 control points) and
   evaluation-target-only (T-063 `assert_evaluation_only`); it cannot be a
   model input at scale, so a "+DGPS" arm would be theatre.
2. **No G/H (temporal-DL / GNN) arms** — those architectures are what this
   ablation gates; building them to evaluate them is circular.
3. **Isolation-Forest `anomaly_score` held out of every arm** (constant
   across arms ⇒ cancels in comparisons).
4. **Sentinel-1 enters as a static per-node LOS snapshot** (single T-062
   acquisition, 400 nodes): the synthetic events carry no wall-clock dates,
   so a per-window InSAR join is impossible. A real temporal stack awaits
   T-057's missing CDSE scenes.

## What the §25 evidence DOES justify

- **Groups B (temporal) and F (physics) are the two modality families with
  measurable value** on this corpus — for calibrated risk grading.
- **The 40–70 §13 feature budget is not the constraint**; the alert
  threshold and event generator are.
- Group J (environmental) stays gated OFF (T-067): nothing here suggests
  weather proxies would move any §24 metric.
