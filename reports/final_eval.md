# One-time locked evaluation

All results are from the synthetic held-out corpus.

Lock sha256: `e49000ce6142a6c5e59a9743d8551526da62997b03edbb48173b7d4d03377cbb`; seed `20260929`; evaluations recorded: `1`.

## Risk models

| Model | Macro PR-AUC | Macro F1 | CRITICAL recall | NORMAL false-alarm rate |
|---|---:|---:|---:|---:|
| tuned_xgboost | 0.5029 | 0.4458 | 0.0933 | 0.1508 |
| default_xgboost | 0.5351 | 0.4450 | 0.0457 | 0.0372 |
| logistic_regression | 0.4687 | 0.4297 | 0.7650 | 0.6263 |
| threshold_rule | 0.4146 | 0.3897 | 0.0416 | 0.0174 |

## Anomaly models

| Model | PR-AUC | Recall at development threshold | False-positive rate |
|---|---:|---:|---:|
| default | 0.9717 | 0.5885 | 0.0059 |
| tuned | 0.9740 | 0.8959 | 0.0217 |

## Alert engine

{
  "false_alarms_per_normal_event_day": 0.046079769601152,
  "normal_event_days": 347.22395833333326,
  "false_alert_episodes_on_normal_events": 16,
  "median_lead_time_hours": -3.333400000000001,
  "p10_lead_time_hours": -11.6667,
  "lead_time_events": 887,
  "time_in_state_fraction": {
    "GREEN": 0.6862814814814815,
    "WATCH": 0.3137185185185185
  },
  "neighbour_confirmations_available": false,
  "de_escalation": "not implemented in current AlertEngine; cannot be evaluated",
  "watch_probability_mapping": "P(WATCH or higher) = 1 - P(NORMAL)"
}

The corpus has one node per event, so neighbour confirmation is unavailable and CRITICAL escalation remains gated.
The present alert engine has no de-escalation implementation; that behavior is not reported as measured.
