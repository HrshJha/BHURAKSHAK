# §14 Isolation Forest ablation — A/B/C (T-045)

Protocol: event-level splits (60/20/20 within each scenario family, seed 42); healthy baseline = triple-healthy mask (G-2); threshold = p0.99 of validation healthy scores; all numbers on held-out test events.

| Set | Features | FAR (healthy test) | Detection (anomalous test) | AUC |
|---|---|---|---|---|
| A_physical_only | 5 | 0.0033 | 0.3023 | 0.9710 |
| B_add_temporal | 15 | 0.0070 | 0.5512 | 0.9907 |
| C_add_spatial | 23 | 0.0069 | 0.6805 | 0.9948 |

**Verdict — does adding spatial coherence reduce false alarms?**

- A (physical only) FAR: 0.0033
- B (+temporal) FAR: 0.0070 (did not reduce vs A)
- C (+spatial) FAR: 0.0069 (reduced vs B)

_Caveat (recorded honestly): on the synthetic gate dataset every event carries a single node, so Group C neighbourhood features degenerate to self-only values; the spatial-coherence effect measured here is bounded by that synthetic limitation and must be re-measured when multi-node events exist._
