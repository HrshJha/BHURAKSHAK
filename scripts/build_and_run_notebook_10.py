#!/usr/bin/env python3
""" — build and execute notebooks/10_inference_profiling.ipynb.

Acceptance: the notebook measures p50/p95 inference latency
and peak RSS for the Isolation Forest + XGBoost artifacts on a single feature
window and records them against the documented edge budget. Portable
profiling only — no Raspberry Pi 5 hardware is executed on in this
workstream, and Gap records that the 's names an "edge
compute/power budget" without ever giving it a number, so there is NO
threshold to pass/fail against: the notebook measures and reports honestly.

Method: the two artifacts are trained -safe exactly as in /
(IF on healthy-baseline train windows → anomaly_score joined back → XGBoost
on groups A–F + signals), then scored on ONE feature window: per-call latency
of `anomaly_score`, `predict_proba` and the full IF→XGBoost chain over 2,000
timed calls after warm-up (p50/p95), plus peak RSS (post-training footprint
and the inference-time delta) and pickled artifact sizes. Numbers are also
written to experiments/inference_profile.json for the run summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "10_inference_profiling.ipynb"

CELLS = [
    new_markdown_cell(
        """# 10 — Inference Profiling ( , )

**What this notebook measures:** p50/p95 single-window inference latency for
the  chain — Isolation Forest `anomaly_score` → XGBoost `predict_proba` —
plus the models' memory footprint and artifact sizes.

**What it deliberately does NOT do:**

1. **No pass/fail verdict.** Gap ****:  requires edge inference to
   run "within its compute/power budget" but the  never states a latency,
   memory or power number. There is no numeric budget to record against —
   the honest output is the measurement itself, flagged for the missing
   budget.
2. **No Raspberry Pi 5 execution.** Profiling runs on this development
   workstation; per-call latencies and RSS here are an upper bound on
   well-provisioned hardware, not a substitute for on-target measurement."""
    ),
    new_code_cell(
        """import io
import json
import pickle
import platform
import resource
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import sklearn
import xgboost

sys.path.insert(0, str(Path.cwd().parent))
REPO = Path.cwd().parent

from src.anomaly.isolation_forest import train_isolation_forest
from src.risk.xgboost_model import train_risk_model

pd.set_option("display.width", 200)
ENV = {
    "platform": platform.platform(),
    "machine": platform.machine(),
    "python": platform.python_version(),
    "sklearn": sklearn.__version__,
    "xgboost": xgboost.__version__,
    "cpu_count": psutil.cpu_count(logical=True),
}
for k, v in ENV.items():
    print(f"{k:>10}: {v}")

def peak_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes — normalise to MB
    return ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024

RSS_AFTER_IMPORTS = peak_rss_mb()
print(f"peak RSS after imports: {RSS_AFTER_IMPORTS:.1f} MB")"""
    ),
    new_markdown_cell(
        "## 1 — Train the two  artifacts -safe (IF → anomaly_score → XGBoost)"
    ),
    new_code_cell(
        """store = pd.read_parquet(REPO / "data" / "features" / "features_v2.parquet")
splits = pd.read_csv(REPO / "data" / "features" / "split_assignment.csv")
store = store.merge(splits[["event_id", "split"]], on="event_id", how="left", validate="many_to_one")
assert store["split"].notna().all()

if_model = train_isolation_forest(store)          # : healthy-baseline train windows only
store["anomaly_score"] = if_model.anomaly_score(store)  # : the IF score feeds XGBoost
risk_model = train_risk_model(store)              # : groups A–F + anomaly_score + physics_residual
print(f"IF features: {len(if_model.features)} | threshold {if_model.threshold:.4f} "
      f"({if_model.threshold_rule}) | XGBoost features: {len(risk_model.features)} "
      f"| classes {risk_model.classes}")

def artifact_mb(obj) -> float:
    buf = io.BytesIO()
    pickle.dump(obj, buf)
    return buf.tell() / (1024 * 1024)

sizes = {"isolation_forest": artifact_mb(if_model), "xgboost_risk": artifact_mb(risk_model)}
print("pickled artifact sizes:", {k: f"{v:.2f} MB" for k, v in sizes.items()})
RSS_AFTER_TRAINING = peak_rss_mb()
print(f"peak RSS after training: {RSS_AFTER_TRAINING:.1f} MB (delta {RSS_AFTER_TRAINING - RSS_AFTER_IMPORTS:+.1f})")"""
    ),
    new_markdown_cell(
        """## 2 — Single-window inference latency

One feature window = one row of the  feature frame. After 50 warm-up
calls, 2,000 timed calls per stage: the IF score, the XGBoost probabilities,
and the full chain (score the row, attach `anomaly_score`, predict) — the
exact per-window edge inference path. Batch-256 is reported for context only."""
    ),
    new_code_cell(
        """rng = np.random.default_rng(42)
row = store.iloc[[int(rng.integers(len(store)))]][risk_model.features].reset_index(drop=True)

def chain(df):
    s = if_model.anomaly_score(df)
    return risk_model.predict_proba(df.assign(anomaly_score=s))

for _ in range(50):  # warm-up: lazy allocations, caches
    chain(row)

N = 2000
def timed(fn, df, n=N):
    t0 = time.perf_counter_ns()
    for _ in range(n):
        fn(df)
    total = time.perf_counter_ns() - t0
    per_call_ms = np.full(n, total / n / 1e6)  # loop overhead shared evenly
    # re-time individually for honest percentiles
    samples = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter_ns()
        fn(df)
        samples[i] = (time.perf_counter_ns() - t0) / 1e6
    return samples

lat = {
    "IF anomaly_score (1 window)": timed(lambda d: if_model.anomaly_score(d), row),
    "XGBoost predict_proba (1 window)": timed(lambda d: risk_model.predict_proba(d), row),
    "full IF→XGBoost chain (1 window)": timed(chain, row),
}
rows = [{"stage": k, "p50_ms": float(np.percentile(v, 50)), "p95_ms": float(np.percentile(v, 95)),
         "mean_ms": float(v.mean()), "max_ms": float(v.max())} for k, v in lat.items()]

batch = store.iloc[:256][risk_model.features].reset_index(drop=True)
t0 = time.perf_counter_ns(); chain(batch); t_batch = (time.perf_counter_ns() - t0) / 1e6
rows.append({"stage": "full chain (context: batch 256)", "p50_ms": np.nan,
             "p95_ms": np.nan, "mean_ms": t_batch, "max_ms": np.nan})
latency = pd.DataFrame(rows)
display(latency.round(3))
RSS_AFTER_INFERENCE = peak_rss_mb()
print(f"peak RSS after {N * 3:,} inference calls: {RSS_AFTER_INFERENCE:.1f} MB "
      f"(delta vs post-training: {RSS_AFTER_INFERENCE - RSS_AFTER_TRAINING:+.1f})")"""
    ),
    new_markdown_cell(
        """## 3 — Memory footprint

Peak RSS (the deployment-relevant number: the models + runtime must fit in
the target's memory) and the pickled artifact sizes (what must actually be
shipped to the edge)."""
    ),
    new_code_cell(
        """memory = pd.DataFrame([
    {"metric": "peak RSS after imports", "value": RSS_AFTER_IMPORTS, "unit": "MB"},
    {"metric": "peak RSS after training (deployment footprint)", "value": RSS_AFTER_TRAINING, "unit": "MB"},
    {"metric": "peak RSS after 6,000 inference calls", "value": RSS_AFTER_INFERENCE, "unit": "MB"},
    {"metric": "inference peak-RSS delta", "value": RSS_AFTER_INFERENCE - RSS_AFTER_TRAINING, "unit": "MB"},
    {"metric": "IF artifact (pickled)", "value": sizes["isolation_forest"], "unit": "MB"},
    {"metric": "XGBoost artifact (pickled)", "value": sizes["xgboost_risk"], "unit": "MB"},
])
display(memory.round(2))"""
    ),
    new_markdown_cell(
        """## 4 — Against the edge budget:  says there is none

 (edge inference "within its compute/power budget") carries **no
number** in the  — recorded as Gap . The register below therefore has
no pass/fail column. For scale only: the system's own  window cadence
(the stride at which new feature windows — and therefore inference requests —
arrive) is measured next; per-window inference cost is compared to *that*
project-internal cadence, which is a fact of this system, not an invented
budget."""
    ),
    new_code_cell(
        """dts = []
for _, g in store.groupby(["event_id", "node_id"], sort=False):
    t = g.sort_values("window_index")["window_timestamp"].to_numpy(dtype=float)
    if len(t) >= 3:
        dts.append(np.median(np.diff(t)))
stride_h = float(np.median(dts))
chain_p95 = float(latency.loc[latency.stage.str.contains("full IF→XGBoost chain"), "p95_ms"].iloc[0])
cadence_ms = stride_h * 3600 * 1000
print(f" window cadence on this corpus: {stride_h:.2f} h = {cadence_ms:,.0f} ms between windows")
print(f"full-chain p95 on ONE window: {chain_p95:.3f} ms = {100 * chain_p95 / cadence_ms:.5f}% of one window period")

budget = pd.DataFrame([
    {"measurement": "single-window chain p50", "value": f"{latency.loc[latency.stage.str.contains('full IF→XGBoost chain'), 'p50_ms'].iloc[0]:.3f} ms",
     "edge budget ()": "NOT SPECIFIED — Gap "},
    {"measurement": "single-window chain p95", "value": f"{chain_p95:.3f} ms",
     "edge budget ()": "NOT SPECIFIED — Gap "},
    {"measurement": "peak RSS (trained artifacts in memory)", "value": f"{RSS_AFTER_TRAINING:.1f} MB",
     "edge budget ()": "NOT SPECIFIED — Gap "},
    {"measurement": "shipped artifact size", "value": f"{sum(sizes.values()):.2f} MB",
     "edge budget ()": "NOT SPECIFIED — Gap "},
    {"measurement": "share of one  window period (p95)", "value": f"{100 * chain_p95 / cadence_ms:.5f}%",
     "edge budget ()": "context only — cadence is a system fact, not a budget"},
])
display(budget)

profile = {
    "environment": ENV,
    "latency_single_window_ms": {
        "if_p50": float(np.percentile(lat["IF anomaly_score (1 window)"], 50)),
        "if_p95": float(np.percentile(lat["IF anomaly_score (1 window)"], 95)),
        "xgb_p50": float(np.percentile(lat["XGBoost predict_proba (1 window)"], 50)),
        "xgb_p95": float(np.percentile(lat["XGBoost predict_proba (1 window)"], 95)),
        "chain_p50": float(np.percentile(lat["full IF→XGBoost chain (1 window)"], 50)),
        "chain_p95": chain_p95,
    },
    "memory": {
        "peak_rss_after_training_mb": RSS_AFTER_TRAINING,
        "inference_delta_mb": RSS_AFTER_INFERENCE - RSS_AFTER_TRAINING,
        "artifact_mb": sizes,
    },
    "cadence_context": {"window_stride_h": stride_h, "chain_p95_pct_of_period": 100 * chain_p95 / cadence_ms},
    "gap_g8": " states no numeric edge budget; measurements recorded, no pass/fail claimed",
    "n_calls": N,
}
(REPO / "experiments").mkdir(exist_ok=True)
(REPO / "experiments" / "inference_profile.json").write_text(json.dumps(profile, indent=2))
print("profile → experiments/inference_profile.json")"""
    ),
    new_code_cell(
        """fig, ax = plt.subplots(figsize=(8, 3.6))
stages = [s for s in lat]
p50 = [np.percentile(lat[s], 50) for s in stages]
p95 = [np.percentile(lat[s], 95) for s in stages]
x = np.arange(len(stages))
ax.bar(x - 0.2, p50, 0.38, label="p50")
ax.bar(x + 0.2, p95, 0.38, label="p95")
ax.set_yscale("log")
ax.set_xticks(x, ["IF score", "XGB proba", "full chain"], fontsize=9)
ax.set_ylabel("ms per single window (log)")
ax.set_title(f"Single-window inference latency — no  budget exists (Gap ); workstation only")
ax.legend()
fig.tight_layout()
fig.savefig(REPO / "reports" / "nb10_latency_panels.png", dpi=110, bbox_inches="tight")
plt.show()
print("panels → reports/nb10_latency_panels.png")"""
    ),
    new_markdown_cell(
        """## Provenance & honest notes

- Artifacts: `src/anomaly/isolation_forest.py` (, healthy-baseline
  train windows, threshold on validation) and `src/risk/xgboost_model.py`
  (, groups A–F + anomaly_score + physics_residual) — trained here on
  the  store with the  splits, exactly as in the ablation/ runs.
- Latency: 2,000 individually-timed calls after 50 warm-ups, one  feature
  row per call; percentiles over per-call samples (not loop averages).
- Peak RSS via `resource.ru_maxrss` (platform-normalised); artifact sizes
  via pickle (what ships to the edge). Numbers also in
  `experiments/inference_profile.json`.
- ** stands:** no numeric edge budget exists in the , so nothing here
  is a pass/fail claim; and these are workstation measurements, not
  Raspberry Pi 5 measurements. On-target profiling is the open follow-up
  the moment the budget (and hardware) exist."""
    ),
]


def main() -> int:
    raise SystemExit("Legacy notebook 10 profiles a pre-leakfix model and burned test path; rerun after model freeze.")
    notebook = new_notebook(
        cells=CELLS,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
    )
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, NOTEBOOK_PATH)

    # syntax-check every code cell before execution (the triple-quote trap)
    for i, cell in enumerate(CELLS):
        if cell["cell_type"] != "code":
            continue
        src = cell["source"]
        src = "".join(src) if isinstance(src, list) else src
        compile(src, f"nb10_cell{i}", "exec")

    client = NotebookClient(
        notebook,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT / "notebooks")}},
    )
    client.execute()
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"executed OK → {NOTEBOOK_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
