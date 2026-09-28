#!/usr/bin/env python3
"""T-079 — build and execute notebooks/06_temporal_forecasting.ipynb.

Acceptance (TASKS.md T-079): the notebook executes end-to-end and reports
MAE, RMSE, max absolute error and bias per horizon against a persistence
baseline, stating whether the neural forecaster justifies its complexity.

Design: the §16 horizons come from configs/forecasting.yaml (T-078) and are
resolved against the MEASURED window stride of the §23 feature store (the
synthetic corpus is generated at 0.6 h/window — the §9.1 10-minute config
grid is a raw-data cadence, not the store's). Horizons that the corpus'
9-window series depth cannot support (6 h / 24 h here) are stated as
unsupported, never silently dropped. Sequences are built with the T-076
trainer-side helper (`_build_multi_horizon`), split §23-safe at event level
via the T-068 split assignment, and the persistence baseline needs no model:
the last history step of each sequence IS the persistence forecast for every
horizon. §24 error family only (MAE/RMSE/max-abs/bias) — accuracy plays no
role. Every number is computed in-notebook from the store; idempotent.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "06_temporal_forecasting.ipynb"

CELLS = [
    new_markdown_cell(
        """# 06 — Temporal Forecasting (PRD §16, §24, §33)

**Claims under test:**

1. **§16** — the temporal forecaster predicts **physical quantities**
   (displacement / tilt) over the §16 horizons, never "future danger" (T-076
   contract); the horizons are config-driven (T-078) and resolved here
   against the corpus' *measured* window stride.
2. **§24** — the deformation-error family (MAE, RMSE, max absolute error,
   bias) per horizon, for the neural forecaster AND the persistence
   baseline (last observed value carried forward).
3. **Verdict** — whether the neural forecaster justifies its complexity:
   it must beat persistence's MAE by more than 5% on the test split to do so.

**Honest scope:** the §23 synthetic corpus stores 9 windows per event series.
A horizon of *h* steps needs series of at least `history + h` windows, so only
horizons that fit that depth are evaluated; the §16 horizons that do not fit
— or that resolve to less than one full window step on the corpus' *measured*
window stride (computed in §1, not assumed) — are reported as unsupported by
the data, never silently skipped or rounded to a fake "0-step" forecast.
Forecasts feed the XGBoost risk layer as features (T-077); this notebook
scores the forecaster itself."""
    ),
    new_code_cell(
        """import sys
from pathlib import Path

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display

sys.path.insert(0, str(Path.cwd().parent))
REPO = Path.cwd().parent

from src.config import forecasting_config
from src.forecasting.horizons import HORIZON_ORDER, configured_horizon_minutes
from src.forecasting.temporal_model import (
    FORECASTABLE_CHANNELS,
    _build_multi_horizon,
    train_temporal_forecaster,
)

pd.set_option("display.width", 200)
torch.set_num_threads(1)
print("torch", torch.__version__, "| forecastable channels:", FORECASTABLE_CHANNELS)"""
    ),
    new_markdown_cell(
        "## 1 — Load the §23 store, measure the stride, resolve the §16 horizons"
    ),
    new_code_cell(
        """store = pd.read_parquet(REPO / "data" / "features" / "features_v1.parquet")
splits = pd.read_csv(REPO / "data" / "features" / "split_assignment.csv")
store = store.merge(splits[["event_id", "split"]], on="event_id", how="left", validate="many_to_one")
assert store["split"].notna().all(), "every event must carry a §23 split"

# MEASURED corpus stride: median spacing of consecutive window timestamps per series
dts = []
for _, g in store.groupby(["event_id", "node_id"], sort=False):
    t = g.sort_values("window_index")["window_timestamp"].to_numpy(dtype=float)
    if len(t) >= 3:
        dts.append(np.median(np.diff(t)))
STRIDE_H = float(np.median(dts))
print(f"store: {len(store):,} windows | {store.event_id.nunique():,} events | "
      f"measured window stride {STRIDE_H:.3g} h ({STRIDE_H * 60:.0f} min)")

# §16 horizons from config (minutes) → window steps on the MEASURED stride
minutes = configured_horizon_minutes()
depth = store.groupby(["event_id", "node_id"]).size()
MAX_DEPTH = int(depth.max())
HISTORY = 3  # the corpus' 9-window depth leaves (9 - 3 - h + 1) samples per series

resolution, supported = [], []
for key in HORIZON_ORDER:
    m = minutes[key]
    if key == "next_window":
        steps = 1  # next-window is by definition one window ahead
    else:
        steps = int(round(m / (STRIDE_H * 60)))
    # a horizon must be at least one full window step ahead on the measured
    # stride AND fit inside the corpus' series depth — anything else is
    # unsupported by the data (a "0-step forecast" would be the present, not
    # a forecast)
    ok = steps >= 1 and HISTORY + steps <= MAX_DEPTH
    reason = "" if ok else (
        f"resolves to {steps} window steps at the measured {STRIDE_H * 60:.0f}-min stride "
        "(< 1 step — the store cannot represent it)" if steps < 1 else
        f"needs series of {HISTORY + steps}+ windows, corpus max is {MAX_DEPTH}"
    )
    supported.append((key, m, steps, ok, reason))
    resolution.append({
        "horizon": key, "minutes": m, "steps": steps,
        "needed_depth": HISTORY + steps, "corpus_depth": MAX_DEPTH, "supported": ok,
        "note": reason,
    })
resolution_df = pd.DataFrame(resolution)
display(resolution_df)
# Dedup by resolved steps (§16 order first): on this corpus' stride two §16
# horizons can collapse onto the same window step — scored once, stated here.
seen_steps: set[int] = set()
SUPPORTED = []
for key, m, steps, ok, reason in supported:
    if not ok:
        print(f"UNSUPPORTED: {key} ({m} min = {steps} steps) — {reason}")
    elif steps in seen_steps:
        print(f"{key} ({m} min) resolves to the same {steps}-step horizon as an earlier "
              f"§16 horizon on the measured stride — scored once")
    else:
        seen_steps.add(steps)
        SUPPORTED.append((key, m, steps))
UNSUPPORTED = [(k, m, s) for k, m, s, ok, _ in supported if not ok]
HORIZON_STEPS = tuple(s for _, _, s in SUPPORTED)
assert HORIZON_STEPS and HORIZON_STEPS[0] == 1
print("supported horizons (steps):", HORIZON_STEPS)"""
    ),
    new_markdown_cell(
        """## 2 — Build §16 sequences (§23-safe)

`_build_multi_horizon` (the T-076 trainer-side helper) yields, per sample:
history `X (n, 3, C)` and aligned targets `Y (n, H, C)` where `Y[s, i, c]`
is channel *c* exactly `HORIZON_STEPS[i]` windows after the history ends.
Sequences whose targets run past a series are dropped — none invented.
**Persistence baseline for free:** `X[s, -1, c]` (the last observed value)
is the persistence forecast for every horizon."""
    ),
    new_code_cell(
        """CHANNELS = FORECASTABLE_CHANNELS
# the forecaster consumes the build_windows convention ({ch}_mean); the store
# carries the bare §13 A-group names — rename, never recompute
fc_store = store.rename(columns={ch: f"{ch}_mean" for ch in CHANNELS})
X, Y, keys = _build_multi_horizon(fc_store, CHANNELS, history_steps=HISTORY, horizons=HORIZON_STEPS)
key_df = pd.DataFrame(keys, columns=["event_id", "node_id"])
key_df = key_df.merge(splits, on="event_id", how="left", validate="many_to_one")
split_arr = key_df["split"].to_numpy()
n_train = int((split_arr == "train").sum()); n_val = int((split_arr == "validation").sum())
n_test = int((split_arr == "test").sum())
print(f"sequences: {len(keys):,} (train {n_train:,} / validation {n_val:,} / test {n_test:,})")
assert n_train and n_val and n_test
# event-level split ⇒ zero leakage: every sequence belongs wholly to one event
assert key_df["split"].notna().all()"""
    ),
    new_markdown_cell(
        "## 3 — Train the §16 forecaster (LSTM benchmark, config hyper-parameters)"
    ),
    new_code_cell(
        """cfg = forecasting_config()
split_series = fc_store["event_id"].map(splits.set_index("event_id")["split"])
fc = train_temporal_forecaster(
    fc_store,
    split=split_series,
    channels=CHANNELS,
    horizons=HORIZON_STEPS,
    architecture=cfg["architecture"],
    history_steps=HISTORY,
    width=int(cfg["width"]),
    depth=int(cfg["depth"]),
    epochs=int(cfg["epochs"]),
    batch_size=int(cfg["batch_size"]),
    learning_rate=float(cfg["learning_rate"]),
    seed=int(cfg["seed"]),
)
print(f"trained {fc.architecture}: train loss {fc.train_loss:.5f}, val loss {fc.val_loss:.5f}")"""
    ),
    new_markdown_cell(
        """## 4 — §24 error family per horizon: neural vs persistence (test split)

The test split is touched exactly once, for scoring."""
    ),
    new_code_cell(
        """test_idx = np.flatnonzero(split_arr == "test")
Xt = torch.tensor((X[test_idx] - fc.mu) / fc.sd, dtype=torch.float32)
with torch.no_grad():
    pred = fc.torch_module(Xt).numpy() * fc.sd[None, None, :] + fc.mu[None, None, :]
Yt = Y[test_idx]                      # (n, H, C) physical-unit targets
persist = X[test_idx][:, -1, :][:, None, :]  # last observed value, broadcast to all horizons
persist = np.repeat(persist, len(HORIZON_STEPS), axis=1)

def err_family(pred_v, true_v):
    e = pred_v - true_v
    return dict(MAE=float(np.mean(np.abs(e))), RMSE=float(np.sqrt(np.mean(e ** 2))),
                max_abs=float(np.max(np.abs(e))), bias=float(np.mean(e)))

rows = []
for h_i, (key, m, steps) in enumerate(SUPPORTED):
    for c_i, ch in enumerate(CHANNELS):
        neu = err_family(pred[:, h_i, c_i], Yt[:, h_i, c_i])
        per = err_family(persist[:, h_i, c_i], Yt[:, h_i, c_i])
        rows.append({
            "horizon": key, "channel": ch, "steps": steps,
            "neural_MAE": neu["MAE"], "persist_MAE": per["MAE"],
            "neural_RMSE": neu["RMSE"], "persist_RMSE": per["RMSE"],
            "neural_max_abs": neu["max_abs"], "persist_max_abs": per["max_abs"],
            "neural_bias": neu["bias"], "persist_bias": per["bias"],
            "MAE_skill_pct": 100.0 * (per["MAE"] - neu["MAE"]) / per["MAE"] if per["MAE"] else np.nan,
        })
table = pd.DataFrame(rows)
display(table.round(4))"""
    ),
    new_markdown_cell(
        """## 5 — Does the neural forecaster justify its complexity?

**Decision rule (stated before looking):** per horizon, on the displacement
channel, the neural MAE must beat persistence by more than 5% on the test
split. Training cost (epochs × corpus size) is part of the complexity being
justified, so the verdict names it too."""
    ),
    new_code_cell(
        """JUSTIFY_PCT = 5.0
disp = table[table.channel == "displacement"]
verdicts = []
for _, r in disp.iterrows():
    skill = r["MAE_skill_pct"]
    verdicts.append((r["horizon"], skill, skill > JUSTIFY_PCT))
    print(f"{r['horizon']:>12} ({int(r['steps'])} steps): neural MAE {r['neural_MAE']:.4f} vs "
          f"persistence {r['persist_MAE']:.4f} → skill {skill:+.1f}% "
          f"{'NEURAL JUSTIFIED' if skill > JUSTIFY_PCT else 'persistence suffices'}")
any_justified = any(v for _, _, v in verdicts)
n_param = sum(p.numel() for p in fc.torch_module.parameters())
print(f"\\nmodel: {fc.architecture}, {n_param:,} parameters, {cfg['epochs']} epochs "
      f"over {n_train:,} train sequences")
if any_justified:
    print("VERDICT: the neural forecaster justifies its complexity on the horizons above.")
else:
    print("VERDICT: on this corpus the neural forecaster does NOT beat persistence beyond "
          "5% on any supported horizon — persistence remains the §16 baseline until a "
          "corpus with deeper series (and the 6 h/24 h horizons) exists.")"""
    ),
    new_code_cell(
        """fig, axes = plt.subplots(1, 2, figsize=(11, 4))
x = np.arange(len(disp)); w = 0.38
axes[0].bar(x - w / 2, disp["neural_MAE"], w, label="neural (LSTM)")
axes[0].bar(x + w / 2, disp["persist_MAE"], w, label="persistence")
axes[0].set_xticks(x, [f"{h}\\n({int(s)} st)" for h, s in zip(disp["horizon"], disp["steps"])])
axes[0].set_ylabel("MAE (mm)"); axes[0].set_title("Displacement forecast error per horizon (test)")
axes[0].legend()
axes[1].bar(x, disp["MAE_skill_pct"], 0.55, color=["#2a9d8f" if v > JUSTIFY_PCT else "#e76f51" for v in disp["MAE_skill_pct"]])
axes[1].axhline(JUSTIFY_PCT, ls="--", c="k", lw=1)
axes[1].set_xticks(x, disp["horizon"])
axes[1].set_ylabel("MAE skill vs persistence (%)")
axes[1].set_title(f"Skill (justified > {JUSTIFY_PCT:.0f}%)")
fig.tight_layout()
fig.savefig(REPO / "reports" / "nb06_forecast_panels.png", dpi=110, bbox_inches="tight")
plt.show()
print("panels → reports/nb06_forecast_panels.png")"""
    ),
    new_markdown_cell(
        """## Provenance & honest notes

- Store: `data/features/features_v1.parquet` (§10 windows over the §12
  synthetic corpus); splits: `data/features/split_assignment.csv` (T-068,
  event-level). Horizons: `configs/forecasting.yaml` resolved on the
  **measured** corpus stride (the §9.1 10-minute grid is a raw cadence; the
  store's stride is what forecasting actually operates on).
- Sequences: T-076 `_build_multi_horizon` — aligned per-horizon targets, no
  truncation; gappy/short series yield fewer sequences, none invented.
- Unsupported horizons are stated in §1 with the reason (series depth, or
  shorter than one window step on the measured stride); they are **not**
  interpolated, rounded to a 0-step forecast, or dropped silently.
- The forecaster never emits risk (T-076); risk integration is T-077's
  bridge and is not re-claimed here."""
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

    # syntax-check every code cell before execution (the triple-quote trap)
    for i, cell in enumerate(CELLS):
        if cell["cell_type"] != "code":
            continue
        src = cell["source"]
        src = "".join(src) if isinstance(src, list) else src
        compile(src, f"nb06_cell{i}", "exec")

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
