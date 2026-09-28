# Fix-leak log

| ID | Root cause | Files changed | Test / evidence | Before → after |
|---|---|---|---|---|
| P0-ENV | Test run used Python 3.14.3 and packages that drifted from exact requirements pins. This exposed unsupported assumptions about writable NumPy views and datetime tick resolution; no source commit between `pre-fixall` and `pre-leakfix` changed the affected files. | `.python-version`, `reports/env_before.txt`, `reports/env_pinned.txt`, `reports/env_drift.md` | Clean Python 3.12.13 venv installed from `requirements.txt`; full baseline suite | Python 3.14 baseline: 17 failures → pinned baseline: 585 passed |
| P1-IF | `healthy_baseline_mask` performed in-place `&=` on read-only Series views. | `src/anomaly/isolation_forest.py`, `tests/test_isolation_forest.py` | Read-only `Series.to_numpy` monkeypatch and Arrow-backed CoW regression; full suite | 6 failing IF tests → full suite 590 passed |
| P1-DQ | `validate_packets` performed in-place `|=` on a read-only result from pandas/NumPy. | `src/preprocessing/validation.py`, `tests/test_validation.py` | Read-only `Series.to_numpy` monkeypatch and Arrow-backed CoW regression; full suite | 10 failing packet-validation tests → full suite 590 passed |
| P1-UNIT | Group G converted raw datetime ticks as nanoseconds although newer pandas may use microseconds. The feature contract is linear trend in mm/day converted to mm/year; acceleration is mm/year² (PRD §13 Group G and module contract). | `src/features/group_g_dgps.py`, `tests/test_group_g.py` | Explicit datetime64[us] public-path regression; expected CP0 trend −136.97 mm/year | Python 3.14 produced −136,968.75 mm/year → normalized timestamp unit gives −136.97 mm/year |

## Not fixed and why

- Ground-truth-derived Group C features and oracle center mode remain to be removed/gated in Phase 2; no classifier or IF search has run since the leakage finding.
- The old held-out test is burned. A new locked test corpus is required before any final evaluation.

## New problems found

- The checkout contained an untracked `src/risk/artifacts.py` and subsequent registry/tabletop artifact changes appeared during this work. They are preserved and not yet audited or included in the Phase 0/1 commits.
