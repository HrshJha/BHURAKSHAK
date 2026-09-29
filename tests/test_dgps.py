""" acceptance tests — DGPS ingestion and mesh-vs-DGPS residuals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.geospatial.dgps import (
    DGPSError,
    EVALUATION_TARGET_ONLY,
    assert_evaluation_only,
    associate_to_nodes,
    ingest_dgps,
    mesh_vs_dgps_residual,
    synthesize_dgps_survey,
)


def _nodes() -> pd.DataFrame:
    return pd.DataFrame(
        {"node_id": ["V_A", "V_B", "V_C"], "x": [0.0, 100.0, 400.0], "y": [0.0, 0.0, 0.0]}
    )


def _survey() -> pd.DataFrame:
    """Two control points: CP0 sits on V_A, CP1 8 m from V_B."""
    rows = []
    for pid, x, y, vals in (
        ("CP0", 5.0, 0.0, [0.0, -5.0, -9.0]),
        ("CP1", 108.0, 0.0, [0.0, 1.0, 2.0]),
    ):
        for d, v in zip(["2026-03-09", "2026-03-21", "2026-04-02"], vals, strict=True):
            rows.append(
                {
                    "point_id": pid,
                    "x": x,
                    "y": y,
                    "observation_timestamp": pd.Timestamp(d),
                    "vertical_displacement_mm": v,
                    "horizontal_displacement_mm": 0.3,
                }
            )
    return pd.DataFrame(rows)


def _mesh(hours=(168.0, 456.0, 744.0)) -> pd.DataFrame:
    """Mesh vertical displacement at V_A/V_B at three windows: 168/456/744 h
 from the 2026-03-02 epoch == the survey dates Mar 9/21, Apr 2."""
    epoch = pd.Timestamp("2026-03-02")
    rows = []
    for node, vals in (("V_A", [0.0, -4.6, -8.7]), ("V_B", [0.0, 1.4, 2.5])):
        for h, v in zip(hours, vals, strict=True):
            rows.append(
                {
                    "node_id": node,
                    "window_timestamp": h,
                    "displacement_mean": v,
                    "horizontal_displacement_mm": 0.1,
                }
            )
    return pd.DataFrame(rows), epoch


def test_ingest_fills_missing_wgs84_from_crs() -> None:
    raw = _survey().drop(columns=["lat", "lon"], errors="ignore")
    out = ingest_dgps(raw)
    assert out["lat"].notna().all() and out["lon"].notna().all()
    assert out.attrs["usage_class"] == EVALUATION_TARGET_ONLY


def test_ingest_drops_qa_failures_and_counts_them() -> None:
    raw = _survey()
    raw.loc[raw.index[-1], "vertical_displacement_mm"] = np.nan
    out = ingest_dgps(raw)
    assert len(out) == len(raw) - 1
    assert out.attrs["n_rows_dropped_qa"] == 1


def test_ingest_rejects_missing_required_columns() -> None:
    with pytest.raises(DGPSError, match="vertical_displacement_mm"):
        ingest_dgps(_survey().drop(columns=["vertical_displacement_mm"]))


def test_ingest_rejects_duplicate_point_timestamp() -> None:
    raw = pd.concat([_survey(), _survey().iloc[[0]]], ignore_index=True)
    with pytest.raises(DGPSError, match="duplicate"):
        ingest_dgps(raw)


def test_association_picks_nearest_node_once_per_point() -> None:
    dgps = ingest_dgps(_survey())
    assoc = associate_to_nodes(dgps, _nodes())
    by_point = assoc.set_index("point_id")
    assert by_point.loc["CP0", "node_id"] == "V_A"
    assert by_point.loc["CP1", "node_id"] == "V_B"
    assert by_point.loc["CP1", "point_distance_m"] == pytest.approx(8.0)
    assert assoc.point_id.is_unique


def test_residuals_match_mesh_minus_dgps_at_nearest_window() -> None:
    dgps = ingest_dgps(_survey())
    mesh, epoch = _mesh()
    res = mesh_vs_dgps_residual(mesh, dgps, _nodes(), epoch=epoch)
    cp0 = res[(res.point_id == "CP0") & (res.observation_timestamp == pd.Timestamp("2026-03-21"))].iloc[0]
    # mesh −4.6 vs DGPS −5.0 → residual +0.4 mm
    assert cp0["mesh_displacement_mm"] == pytest.approx(-4.6)
    assert cp0["residual_vertical_mm"] == pytest.approx(0.4, abs=1e-9)
    assert cp0["residual_horizontal_mm"] == pytest.approx(-0.2, abs=1e-9)
    assert cp0["node_id"] == "V_A"


def test_residuals_carry_staleness_within_one_hour() -> None:
    dgps = ingest_dgps(_survey())
    mesh, epoch = _mesh()
    res = mesh_vs_dgps_residual(mesh, dgps, _nodes(), epoch=epoch)
    assert (res["staleness_hours"].abs() <= 1.0).all(), ": DGPS aligns within ±1 h"
    assert res.attrs["join_summary"]["n_out_of_tolerance"] == 0


def test_out_of_tolerance_observations_are_reported_not_force_joined() -> None:
    dgps = ingest_dgps(_survey())
    dgps.loc[dgps.index[0], "observation_timestamp"] = pd.Timestamp("2026-03-15")  # 6 d from any window
    mesh, epoch = _mesh()
    res = mesh_vs_dgps_residual(mesh, dgps, _nodes(), epoch=epoch)
    assert len(res) == len(dgps) - 1
    assert res.attrs["join_summary"]["n_out_of_tolerance"] == 1


def test_evaluation_only_guard() -> None:
    dgps = ingest_dgps(_survey())
    assert_evaluation_only(dgps)  # passes
    stripped = dgps.copy()
    stripped.attrs.pop("usage_class")
    with pytest.raises(DGPSError, match="evaluation target"):
        assert_evaluation_only(stripped)


def test_residual_frame_is_marked_evaluation_only() -> None:
    dgps = ingest_dgps(_survey())
    mesh, epoch = _mesh()
    res = mesh_vs_dgps_residual(mesh, dgps, _nodes(), epoch=epoch)
    assert res.attrs["usage_class"] == EVALUATION_TARGET_ONLY
    assert_evaluation_only(res)


def test_synthetic_survey_exposes_a_known_bias() -> None:
    """Notebook 09's bias-detection path: with mesh_bias_mm = 2 the mean
 vertical residual must recover ≈ +2 mm (mesh reads high). Signal-free
 truth (all-zero displacement) so the residual mean IS the bias estimate;
 GNSS noise averages out over 6 matched observations (seeded)."""
    nodes = _nodes()
    truth = np.zeros((2, 3))
    survey = synthesize_dgps_survey(
        nodes,
        vertical_mm_at=truth,
        timestamps=["2026-03-09", "2026-03-21", "2026-04-02"],
        mesh_bias_mm=2.0,
        noise_std_mm=0.2,
        seed=7,
    )
    dgps = ingest_dgps(survey)
    mesh, epoch = _mesh()
    # mesh reads truth + bias → residual_vertical = mesh − dgps ≈ +2 everywhere
    mesh["displacement_mean"] = 2.0
    res = mesh_vs_dgps_residual(mesh, dgps, nodes, epoch=epoch)
    assert res.residual_vertical_mm.mean() == pytest.approx(2.0, abs=0.2)
    assert (dgps["source"] == "synthetic ( stand-in)").all()


def test_synthetic_survey_rejects_wrong_shapes() -> None:
    with pytest.raises(DGPSError, match="timestamps for the survey"):
        synthesize_dgps_survey(_nodes(), vertical_mm_at=np.zeros((2, 3)), timestamps=["2026-03-09"])
    with pytest.raises(DGPSError, match="vertical_mm_at"):
        synthesize_dgps_survey(_nodes(), timestamps=["2026-03-09"])
