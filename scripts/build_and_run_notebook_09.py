#!/usr/bin/env python3
"""T-065 — build and execute notebooks/09_dgps_validation.ipynb.

Acceptance (TASKS.md T-065): the notebook executes end-to-end and reports
mesh-vs-DGPS displacement agreement and any detected systematic sensor bias.

§19 study design (G-7 honesty): no real DGPS campaign exists, so the survey
is the T-063 synthetic fixture — GNSS-level noise on the §10 physics field at
six sparse control locations. The MESH side (what is audited) is the truth
plus an injected +2 mm systematic bias plus mesh-level noise; the residual
pipeline must recover that bias from the residuals alone. DGPS is used ONLY
as an evaluation target (T-063 ``usage_class`` discipline) — never as a
training feature. Idempotent: rebuilds and re-executes the notebook in place.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "09_dgps_validation.ipynb"

CELLS = [
    new_markdown_cell(
        """# 09 — DGPS Validation: mesh-vs-DGPS agreement & systematic bias (PRD §19)

**Claim under test:** the sensor-mesh displacement estimate agrees with an
INDEPENDENT high-accuracy reference at sparse control locations, and a
systematic mesh-side bias — the classic calibration error §19 sends DGPS to
find — is detectable from the residuals alone.

**Roles (§19, enforced):** the mesh provides coverage; DGPS provides sparse
**evaluation targets**. Every frame here carries
`usage_class = "evaluation_target_only"`; DGPS never enters the model feature
matrix (T-063 guard asserted below).

**G-7 honesty:** §19 names no receiver, vendor or survey partner. The survey
is the T-063 synthetic fixture — GNSS-level noise (σ = 1 mm) on the §10
physics field at 6 control points, three survey passes. The mesh side is the
truth + an injected **+2 mm systematic bias** + mesh noise (σ = 0.8 mm); the
study must recover that bias from the residuals."""
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

from src.features.group_g_dgps import GROUP_G_FEATURES, emit_group_g
from src.geospatial.crs import crs_from_config, local_to_wgs84
from src.geospatial.dgps import (
    assert_evaluation_only,
    ingest_dgps,
    mesh_vs_dgps_residual,
    synthesize_dgps_survey,
)
from src.physics.consistency import physics_engine
from src.simulator.grid import build_grid

REPO = Path.cwd().parent
EPOCH = pd.Timestamp("2026-03-01")
MESH_BIAS_MM = 2.0
SEED = 42
rng = np.random.default_rng(SEED)
grid = build_grid()
engine = physics_engine()
print(f"mesh: {grid.n_nodes} nodes, spacing 25 m (§10)")"""
    ),
    new_markdown_cell("## 1 — Six sparse control locations on the mesh (§19: sparse, not at-scale)"),
    new_code_cell(
        """# control points: mesh corners, centre and edge midpoints — the classic
# geotechnical survey pattern around a working panel
targets = [(-225.0, -225.0), (225.0, 225.0), (-225.0, 225.0), (225.0, -225.0), (0.0, 0.0), (0.0, -225.0)]
idx = [int(np.hypot(grid.x - tx, grid.y - ty).argmin()) for tx, ty in targets]
ctrl = pd.DataFrame({"node_id": [grid.node_ids[i] for i in idx], "x": grid.x[idx], "y": grid.y[idx]})
assert ctrl.node_id.is_unique
crs = crs_from_config()
ctrl_lat, ctrl_lon = local_to_wgs84(crs, ctrl.x.to_numpy(), ctrl.y.to_numpy())
ctrl["lat"], ctrl["lon"] = ctrl_lat, ctrl_lon
print(ctrl.to_string(index=False))"""
    ),
    new_markdown_cell(
        """## 2 — Three survey passes: GNSS-noisy observations of the §10 truth"""
    ),
    new_code_cell(
        """# the event window axis: nine §10-stride windows across one simulated day
window_hours = np.arange(0, 9) * (10 * 6) / 60.0  # window_timestamp hours (60-min cadence)
pass_windows = [0, 4, 8]  # survey passes at t = 0, 4 h, 8 h
pass_times = [(EPOCH + pd.Timedelta(hours=float(window_hours[w]))).isoformat() for w in pass_windows]

truth = np.zeros((len(ctrl), len(pass_windows)))
for k, w in enumerate(pass_windows):
    # vectorised per pass — the engine returns 0-d for scalar inputs
    truth[:, k] = engine.expected_displacement_mm(
        ctrl.x.to_numpy(dtype=float), ctrl.y.to_numpy(dtype=float), window_hours[w]
    )
print("true vertical displacement (mm) at the control points:")
print(pd.DataFrame(truth, index=ctrl.node_id, columns=pass_times).to_string())

survey = synthesize_dgps_survey(
    ctrl.reset_index(drop=True),
    vertical_mm_at=truth,
    timestamps=pass_times,
    noise_std_mm=1.0,
    seed=SEED,
)
print(f"\\nsurvey: {len(survey)} observations, GNSS σ = 1 mm (source: {survey.source.iloc[0]})")"""
    ),
    new_markdown_cell("## 3 — The mesh estimate under audit: truth + injected +2 mm systematic bias"),
    new_code_cell(
        """mesh_rows = []
for i, row in ctrl.iterrows():
    for k, w in enumerate(pass_windows):
        mesh_rows.append({
            "node_id": row.node_id,
            "window_timestamp": float(window_hours[w]),
            "displacement_mean": truth[i, k] + MESH_BIAS_MM + rng.normal(0.0, 0.8),
            "horizontal_displacement_mm": rng.normal(0.0, 0.8),
        })
mesh_est = pd.DataFrame(mesh_rows)
print(f"mesh estimate at control nodes: truth + {MESH_BIAS_MM} mm bias + N(0, 0.8^2) mesh noise")"""
    ),
    new_markdown_cell("## 4 — Residuals: mesh − DGPS at matched windows (±1 h §9.1 tolerance)"),
    new_code_cell(
        """dgps = ingest_dgps(survey)
res = mesh_vs_dgps_residual(mesh_est, dgps, pd.DataFrame({"node_id": grid.node_ids, "x": grid.x, "y": grid.y}), epoch=EPOCH)
assert_evaluation_only(res)
summary = res.attrs["join_summary"]
print("join:", summary)
assert summary["n_matched"] == len(dgps) == 18 and summary["n_out_of_tolerance"] == 0
print(res[["point_id", "node_id", "observation_timestamp", "dgps_vertical_mm",
          "mesh_displacement_mm", "residual_vertical_mm"]].head(8).to_string(index=False))"""
    ),
    new_markdown_cell("## 5 — Agreement metrics per control point (§24 displacement-error family)"),
    new_code_cell(
        """def rmse(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x**2)))

rows = []
for pid, g in res.groupby("point_id"):
    r = g["residual_vertical_mm"].to_numpy(dtype=float)
    rows.append({
        "point_id": pid,
        "node_id": g.node_id.iloc[0],
        "n_matches": len(g),
        "bias_mm": float(r.mean()),
        "MAE_mm": float(np.abs(r).mean()),
        "RMSE_mm": rmse(r),
        "max_abs_mm": float(np.abs(r).max()),
    })
agree = pd.DataFrame(rows)
print(agree.round(3).to_string(index=False))
overall = {"bias_mm": res.residual_vertical_mm.mean(), "MAE_mm": res.residual_vertical_mm.abs().mean(),
           "RMSE_mm": rmse(res.residual_vertical_mm)}
print("\\noverall:", {k: round(v, 3) for k, v in overall.items()})
assert agree["RMSE_mm"].max() < 4.0, 'agreement must stay at the few-mm level (mm-scale truth)'
"""
    ),
    new_code_cell(
        """fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
for pid, g in res.groupby("point_id"):
    g = g.sort_values("observation_timestamp")
    axes[0].plot(g.observation_timestamp, g.mesh_displacement_mm, "o--", alpha=0.75)
    axes[0].plot(g.observation_timestamp, g.dgps_vertical_mm, "s", alpha=0.9)
axes[0].set_ylabel("vertical displacement (mm)")
axes[0].set_title("mesh (dashed) vs DGPS (squares) — control points")
axes[0].tick_params(axis="x", rotation=30)
axes[1].bar(agree.point_id, agree.bias_mm, color="#8b1a1a", alpha=0.85)
axes[1].axhline(MESH_BIAS_MM, color="k", ls="--", lw=1, label=f"injected bias = +{MESH_BIAS_MM} mm")
axes[1].axhline(0, color="k", lw=0.6)
axes[1].set_ylabel("mean residual (mm)"); axes[1].legend()
axes[1].set_title("per-point systematic offset")
fig.tight_layout()
fig.savefig(REPO / "reports" / "nb09_dgps_agreement.png", dpi=110)
plt.show()"""
    ),
    new_markdown_cell(
        """## 6 — Systematic bias detection: is the mesh's offset distinguishable from noise?"""
    ),
    new_code_cell(
        """r = res["residual_vertical_mm"].to_numpy(dtype=float)
se = float(r.std(ddof=1) / np.sqrt(len(r)))
bias_hat = float(r.mean())
t_stat = bias_hat / se
detected = abs(t_stat) > 3.0
print(f"estimated bias  : {bias_hat:+.3f} mm   (injected: {MESH_BIAS_MM:+.1f})")
print(f"standard error  : {se:.3f} mm over {len(r)} matched observations")
print(f"t-like statistic: {t_stat:+.2f}  →  bias detected (|t| > 3): {detected}")
assert detected, "the injected systematic bias must be detectable from residuals alone"
assert abs(bias_hat - MESH_BIAS_MM) < 1.0, "recovered bias within 1 mm of the injected value"
print(f"\\nverdict: the mesh reads {bias_hat:+.1f} mm relative to DGPS; calibration correction: {-bias_hat:+.1f} mm")"""
    ),
    new_markdown_cell("## 7 — Group G on the control nodes (§13 Group G) + evaluation-target discipline"),
    new_code_cell(
        """# one row per (node, pass) for the Group G emitter
frames = []
for k, w in enumerate(pass_windows):
    sel = mesh_est[mesh_est.window_timestamp == float(window_hours[w])].copy()
    sel["event_id"] = "DGPS_STUDY"
    sel["window_index"] = k
    frames.append(sel)
windowed = pd.concat(frames, ignore_index=True)

g = emit_group_g(dgps, windowed, pd.DataFrame({"node_id": grid.node_ids, "x": grid.x, "y": grid.y}), epoch=EPOCH)
assert set(GROUP_G_FEATURES) <= set(g.columns)
print(f"Group G rows: {len(g)} ({g.node_id.nunique()} control nodes × {len(pass_windows)} passes)")
print(g[g.node_id == ctrl.node_id.iloc[4]][["window_index", "vertical_displacement", "velocity",
                                            "mesh_vs_dgps_residual"]].to_string(index=False))
assert g["mesh_vs_dgps_residual"].notna().sum() == 18

# §19 discipline: DGPS frames are evaluation targets, never training labels
assert_evaluation_only(dgps)
assert_evaluation_only(res)
print("\\nDGPS role discipline: evaluation_target_only (asserted) — never a model input at scale")"""
    ),
    new_code_cell(
        """agree.to_csv(REPO / "experiments" / "nb09_dgps_agreement.csv", index=False)
print(f"agreement table → experiments/nb09_dgps_agreement.csv")
print("\\nT-065 VERDICT")
print(f"  mesh-vs-DGPS agreement (6 control points, 3 passes): RMSE {overall['RMSE_mm']:.2f} mm, "
      f"MAE {overall['MAE_mm']:.2f} mm")
print(f"  systematic bias: {bias_hat:+.2f} mm (injected {MESH_BIAS_MM:+.1f}) — detected: {detected}")
print(f"  G-7 honesty: synthetic survey (no real campaign exists in MVP scope)")"""
    ),
    new_markdown_cell(
        """## Verdict

- **Agreement:** mesh-vs-DGPS residuals stay at the few-mm level across all
  six control points — consistent with the mesh-noise + GNSS-noise budget.
- **Systematic bias found:** the residual mean recovers the injected +2 mm
  mesh bias with |t| > 3, i.e. the §19 calibration role works: DGPS finds a
  mesh-wide offset the mesh cannot see about itself.
- **Discipline:** DGPS stayed an evaluation target throughout; Group G's
  residual channel is the only way the audit result reaches the feature
  layer, and only at control nodes (NaN elsewhere — sparse by design)."""
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
