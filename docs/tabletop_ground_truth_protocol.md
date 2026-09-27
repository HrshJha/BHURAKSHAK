# §23.1 Tabletop Ground-Truth Protocol (T-072)

**Purpose:** a physical-ground-truth validation layer for the SubSense
pipeline. The tabletop rig creates controlled, measurable subsidence; every
trial must link **rig actuator setting → independent reference measurement →
raw sensor data → derived risk state** under one `run_id`. This document is
the protocol; `src/evaluation/tabletop_protocol.py` is its software harness.

> **Scope honesty:** the Data+ML workstream delivers the harness, not the
> rig campaign. Operating hardware is out of scope (TASKS.md T-072). The
> recorded fixture data in `data/recorded/tabletop/` is a seeded stand-in
> with the real campaign's schema, so the harness is executable end to end;
> a real campaign's CSVs drop in unchanged.

## 1. The rig

A planar board carries two actuated trapdoors (TD1, TD2) driven by M8 knob
threads (pitch 1.25 mm/turn). Four nodes observe the board:

| node | role | observes |
|---|---|---|
| N1 | reference | undisturbed corner (baseline) |
| N2 | active | TD1 descent |
| N3 | active | TD2 descent |
| N4 | reference | undisturbed corner (baseline) |

Each node logs at 10 Hz: 3-axis accelerometer + gyroscope, ToF crack-opening
distance, and ultrasonic distance to the trapdoor surface.

## 2. Trial conditions

26 trials across four conditions — `TD1_only`, `TD2_only`,
`both_together` (both doors share one schedule), `both_sequential` (TD2
starts late) — at two speeds and two step patterns (`stepped` produces
discrete onsets with vibration bursts; `continuous` ramps smoothly).
Targets span 0.27–54.25 mm, crossing all severity boundaries.

## 3. The linkage chain (per `run_id`)

1. **Actuator setting** — the knob schedule in turns × 1.25 mm/turn;
   metadata records `max_displacement_mm_target` and the achieved mm per
   trapdoor. The knob is MONOTONIC: it cannot un-turn.
2. **Independent reference** — `known_displacement_mm` per window: the
   mechanism truth at window end, derived from the actuator schedule, NOT
   from any sensor. This is what scores the pipeline.
3. **Raw sensor data** — the 10 Hz log (accel/gyro/ToF/ultrasonic).
4. **Derived risk state** — the pipeline's window risk, computed from
   sensor features ONLY.

The harness (`link_trials`) asserts the chain and fails loudly when broken:
metadata without windows, windows without metadata, raw samples with no
trial, duplicates, or a reference displacement exceeding what the actuator
physically produced.

## 4. Derived risk state

Window risk uses the plan §3.2 severity boundaries applied to the
**sensor-derived** displacement (`displacement_mm`, ultrasonic minus the
window-0 baseline):

| state | sensor displacement |
|---|---|
| NORMAL | ≤ 15 mm |
| WARNING | 15–35 mm |
| CRITICAL | > 35 mm |

The reference is used only to SCORE this state (never to set it) — the
same discipline as the deployment rule "alert when the pipeline says so".

## 5. Acceptance quantities (§23.1)

- **Mesh-estimated vs reference displacement error** — window-level error
  statistics (MAE / RMSE / max-abs / bias) of sensor displacement vs
  `known_displacement_mm` on the active nodes N2/N3. Reference nodes N1/N4
  carry zero truth and are excluded.
- **Risk-state confusion** — derived 3-state risk vs the reference severity
  buckets (collapsed to the same 3-state vocabulary).
- **Per-trial linkage records** — the full chain for every `run_id`, ready
  for the Stage-2 domain-gap scoring (T-074).

## 6. Reproduce

```bash
# regenerate the recorded fixture (or produce real-campaign-schema data)
.venv/bin/python scripts/generate_tabletop_dataset.py data/recorded/tabletop
```

```python
from src.evaluation.tabletop_protocol import evaluate_tabletop
result = evaluate_tabletop(metadata, windowed, raw=None)
```
