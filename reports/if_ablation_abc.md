# §14 Isolation Forest ablation — A/B/C (T-045)

Protocol: event-level splits (60/20/20 within each scenario family, seed 42); healthy baseline = triple-healthy mask (G-2); threshold = p0.99 of validation healthy scores; all numbers on held-out test events.

| Set | Features | FAR (healthy test) | Detection (anomalous test) | AUC |
|---|---|---|---|---|
| A_physical_only | 5 | 0.0104 | 0.2307 | 0.7917 |
| B_add_temporal | 15 | 0.0131 | 0.3672 | 0.8050 |
| C_add_spatial | 23 | 0.0116 | 0.4222 | 0.9413 |

**Verdict — does adding spatial coherence reduce false alarms?**

- A (physical only) FAR: 0.0104
- B (+temporal) FAR: 0.0131 (did not reduce vs A)
- C (+spatial) FAR: 0.0116 (reduced vs B)

_Caveat (recorded honestly): on the synthetic gate dataset every event carries a single node, so Group C neighbourhood features degenerate to self-only values; the spatial-coherence effect measured here is bounded by that synthetic limitation and must be re-measured when multi-node events exist._
