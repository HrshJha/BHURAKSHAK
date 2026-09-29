# Frozen-model workstation inference profile

Platform: `darwin`. Each number is a single-window CPU measurement over 1,000 warm iterations. RSS is sampled process memory, not device peak memory;   defines no numeric edge budget.

| Model | p50 latency (ms) | p95 latency (ms) | Peak sampled RSS (MiB) |
|---|---:|---:|---:|
| risk_xgboost_tuned | 0.3170 | 0.6131 | 550.6 |
| isolation_forest_tuned | 2.9885 | 3.8288 | 501.4 |
| forecaster_tuned | 0.0417 | 0.0448 | 627.2 |
