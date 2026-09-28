#!/usr/bin/env python3
"""Write a provenance-aware univariate scan for the active feature store."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.build_feature_store import feature_names  # noqa: E402
from src.features.provenance import assert_registered, registry  # noqa: E402
from src.evaluation.splits import assert_no_leakage  # noqa: E402

CURRENT = ROOT / "data/features/features_v2.parquet"
OLD = ROOT / "reports/superseded_leaky/data/features/features_v1.parquet"
OUT = ROOT / "reports/leakage_audit.md"


def _measure(frame: pd.DataFrame, feature: str) -> tuple[float | None, float | None, float | None, float | None]:
    if feature not in frame:
        return None, None, None, None
    x = pd.to_numeric(frame[feature], errors="coerce")
    anomaly = pd.to_numeric(frame["anomaly_label"], errors="coerce")
    risk = frame["risk_label"].map({"NORMAL": 0, "WARNING": 1, "CRITICAL": 2})
    valid = x.notna() & anomaly.notna() & risk.notna()
    if valid.sum() < 2 or x[valid].nunique() < 2:
        return None, None, None, None
    corr_anomaly = float(np.corrcoef(x[valid], anomaly[valid])[0, 1])
    corr_risk = float(np.corrcoef(x[valid], risk[valid])[0, 1])
    anomaly_auc = float(roc_auc_score(anomaly[valid], x[valid])) if anomaly[valid].nunique() == 2 else None
    risk_ovr = []
    for label in (0, 1, 2):
        target = (risk[valid] == label).astype(int)
        if target.nunique() == 2:
            risk_ovr.append(max(float(roc_auc_score(target, x[valid])), float(roc_auc_score(target, -x[valid]))))
    risk_auc = max(risk_ovr) if risk_ovr else None
    return corr_anomaly, corr_risk, anomaly_auc, risk_auc


def _fmt(value: float | None) -> str:
    return "n/a (gated/constant)" if value is None or not np.isfinite(value) else f"{value:.4f}"


def main() -> None:
    current = pd.read_parquet(CURRENT)
    previous = pd.read_parquet(OLD)
    splits = pd.read_csv(ROOT / "data/features/split_assignment.csv")
    assert_no_leakage(splits, splits["split"], "event_id")
    assert_no_leakage(splits, splits["split"], "generation_parameter_id")
    names = feature_names()
    assert_registered(names)
    entries = registry()
    lines = [
        "# Feature leakage audit",
        "",
        "The current scan covers the v2 synthetic feature store and compares it with the archived v1 store only to quantify the removed leak. The v1 metrics are invalid and are not model results.",
        "",
        f"- Active artifact: `{CURRENT.relative_to(ROOT)}` ({len(current):,} windows; schema v2).",
        f"- Archived comparison: `{OLD.relative_to(ROOT)}` ({len(previous):,} windows; `superseded_leaky`).",
        "- Labels are present only as targets. `build_feature_store` passes a copy containing structural keys and sensor channels only to feature emitters.",
        "- Group C has no finite values in v2: all 90,000 production snapshots have one node and no co-temporal neighbors. The schema gate excludes C from XGBoost and Isolation Forest. §21.1 neighbor confirmation is not validated by this corpus.",
        "- Group B persistence and change-point values are causal prefixes; truncation tests prove future windows do not alter earlier outputs.",
        f"- Split exclusivity: {len(splits):,} unique events and `{splits['generation_parameter_id'].nunique():,}` generating-parameter groups; zero group overlap across train/validation/test.",
        "- Neighbor confirmations use only nodes co-temporal within the same event/window; alert persistence is isolated by event.",
        "",
        "## Flagged features and provenance review",
        "",
        "AUC is anomaly ROC-AUC and the maximum one-vs-rest risk ROC-AUC over the three classes (allowing either score direction). Correlations use numeric anomaly labels and ordinal risk labels. Values are univariate screening signals, not proof of leakage.",
        "",
        "| Feature | Provenance class | Sources | v1 |r| anomaly | v2 |r| anomaly | v1 |r| risk | v2 |r| risk | v1 anomaly AUC | v2 anomaly AUC | v1 max risk AUC | v2 max risk AUC | Review |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for feature in names:
        before = _measure(previous, feature)
        after = _measure(current, feature)
        entry = entries[feature]
        flagged = any(v is not None and np.isfinite(v) and v >= 0.95 for v in (before[2], before[3], after[2], after[3]))
        flagged |= any(v is not None and np.isfinite(v) and abs(v) >= 0.8 for v in (before[0], before[1], after[0], after[1]))
        review = "manual provenance review" if flagged else "no threshold flag"
        if entry.get("gated"):
            review = "gated; no model input"
        lines.append(
            f"| `{feature}` | `{entry['class']}` | `{', '.join(entry['sources'])}` | "
            f"{_fmt(None if before[0] is None else abs(before[0]))} | {_fmt(None if after[0] is None else abs(after[0]))} | "
            f"{_fmt(None if before[1] is None else abs(before[1]))} | {_fmt(None if after[1] is None else abs(after[1]))} | "
            f"{_fmt(before[2])} | {_fmt(after[2])} | {_fmt(before[3])} | {_fmt(after[3])} | {review} |"
        )
    lines.extend([
        "",
        "## Source audit findings",
        "",
        "| Finding | Before | After | Evidence / disposition |",
        "|---|---|---|---|",
        "| `hotspot_density` read `anomaly_label` | v1 Pearson r = 0.9176 vs `anomaly_label`; exact same-window label-derived source | all v2 values NaN under the single-node gate | Removed label reads; uses a same-snapshot robust-z sensor-rate proxy only when co-temporal neighbors exist. Registry blocks it from current model inputs because the corpus has no neighbor graph. |",
        "| `neighbor_anomaly_fraction` read `anomaly_label` | v1 was constant on one-node events, so correlation is undefined | all v2 values NaN under the gate | Same observable-only proxy when a real same-time graph exists; no cross-event neighbors. |",
        "| Detected center read `anomaly_label`; default store requested oracle center | detected mode was label-dependent; oracle mode used simulator panel geometry | detected mode is recorded; center is based only on current observable displacement; oracle mode raises | `build_feature_store` rejects oracle mode and the pipeline requests detected mode. |",
        "| Group B persistence/change-point used a whole-event scalar | same event-level value was copied to every window | recalculated from prefix through each window | Future-truncation tests compare earlier values byte-for-byte. |",
        "| Pipeline confirmations grouped reused node IDs across events | unrelated events could confirm one another | counts restricted to same-event, same-window nodes; this corpus yields zero confirmations | `tests/test_pipeline_spatial.py` covers separate events with nearby coordinates and valid same-event neighbors. Alert state is keyed by event and node. |",
        "| Group F physics expectation/residuals | configuration prior from `configs/physics.yaml`, not per-event sampled truth | same configured prior in v2 | Classified `config_derived`; deployment requires the same site configuration to be available. No per-event truth is read. |",
        "| Group G DGPS residual and Groups G/H/I absent deployment feeds | not in the current first-iteration feature store | provenance gated from model inputs | DGPS agreement is evaluation-only; synthetic terrain and unavailable modalities stay gated. |",
        "",
        "The script reads only the development feature store and the split metadata. It does not read `data/heldout_locked/` or any test predictions.",
        "",
    ])
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
