#!/usr/bin/env python3
"""T-044 — build and execute notebooks/04_isolation_forest.ipynb.

Acceptance (TASKS.md T-044): the notebook executes end-to-end and renders an
anomaly-score distribution plot plus a numeric separation statistic showing
injected anomalies are demonstrably separated from normal behaviour.

The notebook trains the §14 Isolation Forest on the FULL model input
(A+B+C+D+E feature groups, per §14 "Input: physical + temporal + spatial +
vibration + sensor-health feature groups"), using the persisted event-level
splits from the T-045 ablation run.

Idempotent: rebuilds and re-executes the notebook in place.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "04_isolation_forest.ipynb"

CELLS = [
    new_markdown_cell(
        """# 04 — Isolation Forest: anomaly-score separation (PRD §14, FR-5)

**Claim under test:** an Isolation Forest trained **only on healthy-baseline
windows** of training events separates injected anomalies from normal
behaviour on held-out events — the unsupervised "does this look abnormal?"
trip-wire that feeds XGBoost as one input feature.

**§14 discipline honoured here:**
- hyperparameters exactly `IsolationForest(n_estimators=300, contamination="auto", random_state=42)`;
- healthy baseline = the G-2 triple-healthy mask
  (`anomaly_label==0 AND fault_label=="NONE" AND risk_label=="NORMAL"`) —
  §14's literal `risk_label=="GREEN"` filter is inoperable under the 3-class MVP;
- input = physical + temporal + spatial + vibration + sensor-health feature groups;
- event-level splits (60/20/20 within scenario family, persisted by the T-045 run);
- threshold = p99 of VALIDATION healthy scores; all evaluation on TEST events."""
    ),
    new_code_cell(
        """import sys
from pathlib import Path

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path.cwd().parent))

from src.anomaly.isolation_forest import train_isolation_forest
from src.config import anomaly_config

FEATURES_PATH = Path.cwd().parent / "data" / "features" / "features_v1.parquet"
SPLIT_PATH = Path.cwd().parent / "data" / "features" / "split_assignment.csv"

df = pd.read_parquet(FEATURES_PATH)
splits = pd.read_csv(SPLIT_PATH)
df = df.merge(splits, on="event_id", how="left")
print(f"windows: {len(df):,} | events: {df.event_id.nunique():,}")
print(df.groupby('split').agg(events=('event_id','nunique'), windows=('event_id','size')).to_string())"""
    ),
    new_code_cell(
        """full_groups = ["A_physical", "B_add_temporal", "C_add_spatial", "D_add_vibration", "E_add_sensor_health"]
fitted = train_isolation_forest(df, feature_groups=full_groups)
print(f"features used ({len(fitted.features)}): {fitted.features}")
print(f"trained on {fitted.n_training_windows:,} healthy windows of the TRAIN split")
print(f"threshold: {fitted.threshold:.4f}  ({fitted.threshold_rule})")"""
    ),
    new_markdown_cell(
        "## Score distributions on held-out TEST events (healthy vs injected anomalies)"
    ),
    new_code_cell(
        """test = df[df["split"] == "test"].copy()
test["anomaly_score"] = fitted.anomaly_score(test)
healthy = test["anomaly_label"].to_numpy() == 0
scores_h = test.loc[healthy, "anomaly_score"].to_numpy()
scores_a = test.loc[~healthy, "anomaly_score"].to_numpy()

fig, ax = plt.subplots(figsize=(10, 4.6))
ax.hist(scores_h, bins=60, alpha=0.6, density=True, label=f"healthy test windows (n={len(scores_h):,})")
ax.hist(scores_a, bins=60, alpha=0.6, density=True, label=f"anomalous test windows (n={len(scores_a):,})")
ax.axvline(fitted.threshold, color="r", ls="--", lw=1.6, label=f"threshold = {fitted.threshold:.3f} (p99 of val healthy)")
ax.set_xlabel("anomaly score = -score_samples(X)   (higher = more anomalous)")
ax.set_ylabel("density")
ax.set_title("Isolation Forest anomaly-score separation — held-out test events (§14)")
ax.legend()
fig.tight_layout()
fig.savefig(Path.cwd().parent / "reports" / "nb04_if_separation.png", dpi=110)
plt.show()"""
    ),
    new_markdown_cell(
        """## Numeric separation statistics

Separation is reported three ways so the claim is not one lucky number:
threshold hit rates at the configured ~1% false-alarm operating point, the
healthy-vs-anomalous score gap in units of the healthy spread, and ROC AUC."""
    ),
    new_code_cell(
        """y = test["anomaly_label"].to_numpy()
s = test["anomaly_score"].to_numpy()
flags = s > fitted.threshold

hit_rate = float(flags[~healthy].mean())
false_alarm = float(flags[healthy].mean())
auc = float(roc_auc_score((y > 0).astype(int), s))
gap_sigmas = float((scores_a.mean() - scores_h.mean()) / scores_h.std())

print(f"threshold-hit rate on anomalous test windows : {hit_rate:.4f}")
print(f"false-alarm rate on healthy test windows    : {false_alarm:.4f}  (designed ≈ {1.0 - float(anomaly_config()['far_alpha']):.2f})")
print(f"ROC AUC (healthy vs anomalous, test)        : {auc:.4f}")
print(f"score gap (mean_a - mean_h) / std_h         : {gap_sigmas:.2f} sigma")

separated = (auc > 0.9) and (hit_rate > 2 * max(false_alarm, 1e-9))
print(f"injected anomalies demonstrably separated from normal behaviour: {separated}")
assert separated, "§14 separation gate failed" """
    ),
    new_markdown_cell(
        """## Verdict"""
    ),
    new_code_cell(
        """print("T-044 ISOLATION FOREST SEPARATION")
print(f"  trained on {fitted.n_training_windows:,} healthy windows (train events only)")
print(f"  test AUC {auc:.4f} | hit rate {hit_rate:.3f} at FAR {false_alarm:.3f}")
print(f"  separation gap {gap_sigmas:.2f} sigma of the healthy spread")
print(f"  verdict: {separated}")
assert separated"""
    ),
]


def main() -> int:
    notebook = new_notebook(
        cells=CELLS,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
    )
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, NOTEBOOK_PATH)

    client = NotebookClient(
        notebook,
        timeout=1200,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT / "notebooks")}},
    )
    client.execute()
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"executed OK → {NOTEBOOK_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
