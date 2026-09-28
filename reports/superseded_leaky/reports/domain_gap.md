# Domain-Gap Validation Report — PRD §23.2 (T-075)

**The headline, stated plainly:** the synthetic-trained risk model **degrades
severely** when it leaves the simulator. On the tabletop domain it detects
almost nothing at the operating point it inherited (CRITICAL recall 0.000,
WARNING recall 0.029, F1 0.147), even though its underlying ranking survives
(PR-AUC 0.903). Stage 1 predicted this: ranking transfers across parameter
regimes, calibrated alerting does not.

| stage | question | result |
|---|---|---|
| 1 | internal validity inside the simulator | ranking transfers out-of-regime (AP 0.864); alerting does not (Brier 0.51 vs 0.11 in-regime) |
| 2 | transfer to the physical tabletop domain | **severe degradation, explicitly confirmed** (recall 0.079 binary; F1 0.147 vs 0.398 in-regime) |

Both stages are reproducible:
`scripts/run_domain_gap_stage1.py`, `scripts/run_domain_gap_stage2.py`;
full numbers in `experiments/domain_gap_stage1.json` and
`experiments/domain_gap_stage2.json`.

---

## 1. Stage 1 — synthetic → synthetic (§23.2 stage 1)

**Setup.** The §15 XGBoost (A–F groups + signals, T-068 event-family split
for training) is re-partitioned on the §23 *parameter-regime* axis: the
event `max_deformation` range is banded so the top 20% is held out entirely
from training (configs/validation.yaml `synthetic_split`).

**Two axes were run, because the literal one taught us something:**

### 1a. The literal §23 axis is degenerate on this corpus — reported, not hidden

~75% of generated events (stable-ground, vibration, sensor-fault and
communication families) carry ≈0 mm deformation, so the corpus-wide
top-20% deformation band — [0.68, 2010.8] mm — contains **essentially all
deforming events**. A model trained on the remainder has literally never
seen subsidence. Its test-band scores (Brier 0.677, macro-F1 0.000) are
the arithmetic consequence, and they are a real property of the
generator's parameter space: **the §23 stage-1 question is only meaningful
per scenario family on this corpus.** This finding is an input to any
future regeneration of the dataset (the generator needs deformation
variance inside every family for regime splits to bite).

### 1b. Family-conditional regime split — the answerable form

Within each scenario family: top 20% of `max_deformation` → test, next 20%
→ validation, rest → train (zero-variance families stay in train; there is
no regime to hold out). Held-out test band: [21.4, 2010.8] mm, median
279.6 mm — magnitudes the model never saw for families it knows.

| band | range (mm) | AP | Brier | macro-F1 | CRITICAL prevalence |
|---|---|---|---|---|---|
| train | [0, 19.6] | 0.116* | 0.110 | 0.293 | 0.000 |
| validation | [3.8, 210.5] | 0.645 | 0.423 | 0.000 | 0.032 |
| **test (held-out)** | **[21.4, 2010.8]** | **0.864** | **0.511** | 0.000 | 0.180 |

\* the train band contains zero anomalous windows — its AP is a
near-chance artifact of scoring a band with no positives; recorded as such
in the JSON.

**Stage-1 verdict:** *ranking transfers* (AP 0.864 on unseen magnitudes —
the model orders deformation severity correctly), *calibrated alerting does
not* (P(CRITICAL) never crosses the 0.5 alert threshold out-of-regime;
Brier 0.511 vs 0.110 in-regime). The cause is visible in the numbers: the
train band's deformation caps at 19.6 mm, below the §10 CRITICAL threshold,
so the class the test band asks for never appeared in training. Internal
validity holds for ordering, fails for thresholded decisions.

## 2. Stage 2 — synthetic → tabletop (§23.2 stage 2, §35 bullet 7)

**Setup.** The synthetic-trained model runs **UNMODIFIED** on tabletop
sensor data. The full raw→risk chain is exercised for real:
gravity-derived roll/pitch, ultrasonic-minus-baseline displacement, ToF
crack-opening (as the strain analog), vibration magnitude deviation →
`build_windows` (§10 config, unchanged) → Group A/B emitters (unchanged) →
`predict_proba` (unchanged). 14,552 windows from 26 trials × 4 nodes,
100% matched to the §23.1 independent reference (`known_displacement_mm`).

**Data provenance — stated plainly:** the physical §23.1 rig campaign is
out of workstream scope. The tabletop data is the **recorded stand-in**
(`data/recorded/tabletop/`, seeded rig simulator with the campaign's
schema; folder README documents provenance). The harness is the
deliverable; a real campaign's CSVs drop in unchanged. Every number below
therefore measures the *synthetic→rig-simulator* gap — which is still the
gap §23.2 stage 2 is designed to expose (different sensing geometry, scale
and signal statistics), just not the literal hardware gap.

**Headline model choice.** The A/B feature set (tilt, displacement,
strain, vibration + temporal history) is the only §13 groups present on
BOTH domains; the headline model is trained on exactly those features.
A full-contract diagnostic with NaN-filled C/D/E/F modalities is reported
separately: it collapsed to NORMAL on every window — modality scarcity
alone is fatal, which is why it is not the headline.

### Stage-2 results (degradation stated, as §23.2 requires)

| metric | in-regime (T-070 arm E, test) | tabletop (stage 2) | degradation |
|---|---|---|---|
| binary alert F1 | 0.398 | **0.147** | −0.251 |
| binary recall | 0.443 | **0.079** | −0.364 |
| binary precision | 0.361 | 0.960 | +0.599 (rare-fire effect, see below) |
| PR-AUC | 0.483* | **0.903** | *improves* — see reading |
| Brier (P-CRITICAL) | 0.056 | n/a (3-state refs not comparable) | — |

\* the in-regime PR-AUC sits near the anomaly base rate; the tabletop AP is
computed against the rig's 28.7% disturbed-window rate. Both are reported
with their prevalence context — the AP comparison is about ranking quality,
not absolute performance.

Risk-state recall per class: **NORMAL 0.999, WARNING 0.029,
CRITICAL 0.000.** The model essentially always says NORMAL on the rig.

### Why it fails — and why that is the useful finding

1. **Scale mismatch (the dominant cause).** The rig's full displacement
   range (0.27–54.25 mm) sits inside the synthetic range's bottom sliver
   (0–2010 mm). Stage 1 already showed P(CRITICAL) collapses on
   out-of-regime magnitudes; stage 2 is that finding with a different
   sensor attached.
2. **The operating point does not transfer; the ranking does.** The model
   CAN fire on the rig (max P(CRITICAL) 0.868; 24 windows above 0.5) and
   when it does it is right (precision 0.96) — but it almost never does.
   This is a threshold/calibration failure, which is *fixable* (rig-domain
   threshold calibration, a §15/V2 work item), unlike a representational
   failure.
3. **The sensor→feature bridge itself is sound.** The §23.1 acceptance
   quantity — mesh-estimated vs reference displacement error — is MAE
   0.541 mm / RMSE 0.844 mm / bias −0.40 mm over 7,276 active-node windows.
   The pipeline's plumbing transfers; the statistical domain gap is the
   problem.

## 3. What this means (§35 bullet 7, Honesty Statement)

- **The prototype's risk grades are NOT yet valid outside the simulator.**
  Any deployment claim must wait for stage-2 success against a real
  campaign, with domain-calibrated thresholds at minimum.
- **The §35 acceptance bullet is satisfied the honest way:** the report
  states degradation explicitly rather than omitting it. Passing here
  means *reporting the failure*, because that is what the PRD asks the
  stage to produce.
- **Concrete next actions, in order of leverage:**
  1. calibrate the alert threshold on a small rig-domain labeled set
     (the ranking is already there);
  2. regenerate the synthetic corpus with in-family deformation variance
     spanning the rig's range (makes stage-1 axis non-degenerate and
     stage-2 scales overlapping);
  3. bridge the remaining modalities or retrain an A/B-only reference
     model as the fairness baseline for all future stage-2 runs;
  4. repeat against real rig hardware when the campaign runs (the harness
     is ready; see `docs/tabletop_ground_truth_protocol.md`).

## 4. Reproduce

```bash
.venv/bin/python scripts/run_domain_gap_stage1.py   # → experiments/domain_gap_stage1.json
.venv/bin/python scripts/run_domain_gap_stage2.py   # → experiments/domain_gap_stage2.json
```

Deterministic: config-driven hyperparameters, T-068 splits, fixed seeds,
no sampling. Stage-2 provenance and every substitution are recorded inside
`experiments/domain_gap_stage2.json` (`metadata.data_provenance`,
`metadata.bridge`).
