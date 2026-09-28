# Label vocabulary mapping (G-1 closure)

**Status: RESOLVED in code.** This document is the authoritative description of how the PRD's three label vocabularies relate. The mapping is implemented, tested, and exercised by the shipped dataset — not planned.

## The three vocabularies

1. **§10 scenario taxonomy (10 labels)** — emitted by the scenario engine as *flags/fields*, not as one column: `NORMAL, SENSOR_FAULT, DATA_QUALITY, LOCAL_ANOMALY, NON_SUBSIDENCE, SUBSIDENCE, HIGH_RISK, CRITICAL, MIXED, COMMUNICATION_FAILURE` (`src/simulator/scenarios.py:58-62` and the scenario table in its docstring).
2. **§12 4-class risk vocabulary** — `GREEN, WATCH, WARNING, CRITICAL` (`RISK_VOCAB_FULL4`).
3. **§12 3-class MVP vocabulary (the trained model's classes)** — `NORMAL, WARNING, CRITICAL` (`RISK_VOCAB_MVP3`, the MVP per §12; the model is `src/risk/xgboost_model.py`, exactly 3 classes enforced).

## The two mappings

### §10 taxonomy → separate label fields (never collapsed — §12)

| §10 label | where it lives downstream |
|---|---|
| `NORMAL` | `risk_label = "NORMAL"` |
| `SUBSIDENCE` / `HIGH_RISK` | `risk_label = "WARNING"` (growth scenarios) |
| `CRITICAL` | `risk_label = "CRITICAL"` (rapid / multi-zone) |
| `SENSOR_FAULT` | `fault_label ∈ {BIAS, STUCK, DROPOUT, SPIKE, DRIFT}` (the five §10 fault modes) |
| `DATA_QUALITY` | `data_quality_label = "DATA_QUALITY"` (packet loss) |
| `COMMUNICATION_FAILURE` | `data_quality_label = "COMMUNICATION_FAILURE"` |
| `LOCAL_ANOMALY` | `anomaly_label = 1` + flag `LOCAL_ANOMALY` (single-node disturbance; ground itself stable) |
| `NON_SUBSIDENCE` | `anomaly_label = 1` + flag `NON_SUBSIDENCE` (vibration-only) |
| `MIXED` | composition of the above fields on one event |

Implementation: `src/simulator/scenarios.py` (`_assign_labels`, scenario map at lines ~13–27), vocabulary authority `src/preprocessing/labels.py`.

### 4-class → 3-class (a rename within one column, not a collapse)

`FULL4_TO_MVP3` (`src/preprocessing/labels.py:64`):

| §12 4-class | MVP 3-class |
|---|---|
| GREEN | NORMAL |
| WATCH | WARNING |
| WARNING | WARNING |
| CRITICAL | CRITICAL |

Consumers of the rename: `src/risk/alert_engine.py:probability_of` evaluates `P(WATCH or higher) = 1 − P(NORMAL)` under this mapping (G-3 resolution); the alert engine's levels stay the §21.1 four (`configs/alerts.yaml risk_levels`).

## Where it is pinned

- `tests/test_labels.py::test_full4_vocabulary_accepted_when_selected` — the 4-class vocabulary validates.
- `tests/test_scenarios.py` — every §10 scenario maps to exactly the PRD's label table.
- The shipped store's per-family `risk_label` distribution (all 16 families → the three MVP classes; measured in the fixall audit).

This closes **G-1** as RESOLVED: the mapping exists in code (two explicit tables), not just in prose.
