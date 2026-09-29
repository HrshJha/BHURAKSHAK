#!/usr/bin/env python3
""" — build and execute notebooks/01_synthetic_data_generation.ipynb.

The notebook is the third synthetic-data-gate deliverable: it must execute
end-to-end and emit both a plot and a numeric statistic showing tilt tracks
the spatial gradient of the deformation field (correlation > 0.95), proving
the channels are physically coupled rather than independently random.

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

NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "01_synthetic_data_generation.ipynb"

CELLS = [
    new_markdown_cell(
        """# 01 — Synthetic Data Generation: Physical-Coupling Proof ( , Gate 3/3)

**Claim under test:** every sensor channel in the BhuRakshak synthetic dataset derives
from one latent deformation field

$$W(x,y,t) = W_{max}\\,(1-e^{-ct})\\cdot \\exp\\!\\left(-\\frac{(x-x_0)^2+(y-y_0)^2}{2\\sigma^2}\\right)$$

and in particular **tilt is the spatial gradient of W** (`tilt ≈ ∂W/∂x, ∂W/∂y`) —
not independently sampled noise.

**Acceptance ():** the notebook executes end-to-end and shows
**Pearson r > 0.95** between generated tilt and the analytic gradient, as both a
numeric statistic and a plot."""
    ),
    new_code_cell(
        """import sys
from pathlib import Path

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd().parent))

from src.config import physics_config
from src.simulator.deformation_field import DeformationField, FieldParams
from src.simulator.rng import make_rng

field = DeformationField(FieldParams.from_config())
cfg = physics_config()
print(f"W_max = {field.w_max:.0f} mm, c = {field.params.time_coefficient} /day, sigma = {field.params.sigma:.0f} m")"""
    ),
    new_markdown_cell(
        """## Proof 1 — spatial transect through the subsidence centre

Tilt along a line through the panel centre vs the analytic gradient of W, at a fixed time."""
    ),
    new_code_cell(
        """rng = make_rng(42)
t_fixed = 20.0  # days
xs = np.linspace(-250, 250, 401)
ys = np.zeros_like(xs)

from src.simulator.channels_tilt import generate_tilt

tilt_x, _ = generate_tilt(field, xs, ys, t_fixed, rng)
analytic = np.degrees(np.asarray(field.dW_dx(xs, ys, t_fixed)) / 1000.0)

r_transect = float(np.corrcoef(tilt_x, analytic)[0, 1])
print(f"transect Pearson r (tilt vs ∂W/∂x) = {r_transect:.5f}")
assert r_transect > 0.95, "coupling gate failed"

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(xs, analytic, "k-", lw=2, label="analytic ∂W/∂x (degrees)")
ax.plot(xs, tilt_x, "r.", ms=3, alpha=0.6, label="generated tilt_x (+noise)")
ax.set_xlabel("x (m)"); ax.set_ylabel("tilt (degrees)")
ax.set_title(f"Spatial transect at t={t_fixed:.0f} d — Pearson r = {r_transect:.4f}")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(Path.cwd().parent / "reports" / "nb01_transect_coupling.png", dpi=110)
plt.show();"""
    ),
    new_markdown_cell(
        """## Proof 2 — the shipped dataset itself is coupled

Per-event Pearson r between measured `tilt_x` and the analytic gradient, over **all
rapid-subsidence events**, stratified by expected signal-to-noise ratio.

*Physics expectation (stated before the test):* tilt IS the spatial gradient, so nodes
far from the subsidence centre correctly measure near-zero tilt — there the row-level r
is noise-limited by definition (a ~zero signal cannot correlate above noise). True
independence would show r ≈ 0 **uniformly, including near the centre**. Coupling instead
predicts r rising monotonically with SNR. **Gate:** every event whose expected peak
analytic tilt ≥ 10× the configured tilt-noise std must have r > 0.95."""
    ),
    new_code_cell(
        """nodes_path = Path.cwd().parent / "data" / "synthetic" / "synthetic_nodes.csv"
events = pd.read_csv(Path.cwd().parent / "data" / "synthetic" / "synthetic_events.csv")
cols = ["event_id", "timestamp", "x", "y", "tilt_x", "displacement"]
df_all = pd.read_csv(nodes_path, usecols=cols)
df_rapid = df_all[df_all["event_id"].isin(events.loc[events["type"] == "rapid_subsidence", "id"])]
print(f"rapid-subsidence events: {df_rapid['event_id'].nunique()}, rows: {len(df_rapid):,}")

noise_std = float(cfg["noise"]["tilt_noise_std_deg"])
rows = []
for eid, g in df_rapid.groupby("event_id"):
    t = g["timestamp"].to_numpy() / 24.0
    ana = np.degrees(np.asarray(field.dW_dx(g["x"].to_numpy(), g["y"].to_numpy(), t)) / 1000.0)
    amp = float(np.abs(ana).max())                 # expected peak tilt at this node
    r = float("nan") if ana.std() < 1e-9 else float(np.corrcoef(g["tilt_x"].to_numpy(), ana)[0, 1])
    rows.append((eid, r, amp, amp / noise_std))
per_event = pd.DataFrame(rows, columns=["event_id", "r", "expected_tilt_deg", "snr"]).dropna()

GATE_SNR, GATE_R = 10.0, 0.95
detectable = per_event[per_event["snr"] >= GATE_SNR]
print(f"events with expected tilt >= {GATE_SNR:.0f}x noise: {len(detectable)} of {len(per_event)}")
print(f"their r: min={detectable['r'].min():.4f}  p05={detectable['r'].quantile(0.05):.4f}  median={detectable['r'].median():.4f}")
assert len(detectable) >= 50, "too few detectable events for a meaningful gate"
assert detectable["r"].min() > GATE_R, "coupling gate failed: some detectable event below 0.95"
r_dataset = float(detectable["r"].min())
best = per_event.sort_values("expected_tilt_deg", ascending=False).iloc[0]
print(f"strongest-signal event {best.event_id}: r = {best.r:.5f} at {best.snr:.0f}x noise")

ev_id = best.event_id
df = df_rapid[df_rapid["event_id"] == ev_id].copy()
t_days = df["timestamp"].to_numpy() / 24.0
ana = np.degrees(np.asarray(field.dW_dx(df["x"].to_numpy(), df["y"].to_numpy(), t_days)) / 1000.0)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
axes[0].scatter(per_event["snr"], per_event["r"], s=10, alpha=0.5, color="#3b6ea5")
axes[0].axvline(GATE_SNR, color="r", ls=":", lw=1.5)
axes[0].axhline(GATE_R, color="k", ls="--", lw=1.2, label=f"gate r = {GATE_R}")
axes[0].set_xscale("log"); axes[0].set_ylim(-0.3, 1.02)
axes[0].set_xlabel("expected signal / noise (log)"); axes[0].set_ylabel("per-event Pearson r")
axes[0].set_title(f"r rises with SNR — coupling signature (all {len(per_event)} events)")
axes[0].legend()

axes[1].plot(t_days, ana, "k-", lw=2, label="analytic ∂W/∂x")
axes[1].plot(t_days, df["tilt_x"], "r.", ms=4, alpha=0.7, label="measured tilt_x")
axes[1].set_xlabel("t (days)"); axes[1].set_ylabel("tilt (degrees)")
axes[1].set_title(f"Strongest event — r = {best.r:.4f} ({best.snr:.0f}x noise)")
axes[1].legend(); axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(Path.cwd().parent / "reports" / "nb01_dataset_coupling.png", dpi=110)
plt.show();"""
    ),
    new_markdown_cell(
        """## Verdict"""
    ),    new_code_cell(
        """print("GATE 3/3 — PHYSICAL COUPLING")
print(f"  transect r (full bowl, t=20d)      = {r_transect:.5f}  (> 0.95: {r_transect > 0.95})")
print(f"  detectable events (>=10x noise)    = {len(detectable)} of {len(per_event)} rapid events")
print(f"  min r among detectable events      = {r_dataset:.5f}  (> 0.95: {r_dataset > 0.95})")
verdict = (r_transect > 0.95) and (r_dataset > 0.95)
print(f"  tilt tracks dW/dx wherever the signal is detectable; far-field nodes")
print(f"  correctly read ~0 tilt (the field itself decays) — coupling, not independence.")
print(f"  channels are physically coupled, not independently random: {verdict}")
assert verdict"""
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
    (REPO_ROOT / "reports").mkdir(exist_ok=True)
    nbformat.write(notebook, NOTEBOOK_PATH)

    client = NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT / "notebooks")}},
    )
    client.execute()
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"executed OK → {NOTEBOOK_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
