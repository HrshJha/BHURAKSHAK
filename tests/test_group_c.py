"""T-037 acceptance tests — Feature Group C (spatial)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.group_c_spatial import (
    GROUP_C_FEATURES,
    SpatialFeatureError,
    emit_group_c,
    neighbour_radius_m,
)
from src.features.windowing import build_windows


def _small_mesh(anomaly_nodes: set[str], n_side: int = 5, spacing: float = 25.0):
    """5×5 mesh, 25 m spacing. Anomalous nodes ramp to 12 mm; others are CONSTANT
    (zero slope) so sign-of-slope coherence is unambiguous."""
    coords = (np.arange(n_side) - (n_side - 1) / 2.0) * spacing
    xx, yy = np.meshgrid(coords, coords)
    x, y = xx.ravel(), yy.ravel()  # node i: (x=coords[i%5], y=coords[i//5]); V0012 = (0,0)
    ids = [f"V{i:04d}" for i in range(x.size)]
    coords_df = pd.DataFrame({"node_id": ids, "x": x, "y": y})

    n = 144
    ts = (10.0 / 60.0) * np.arange(n)
    frames = []
    for node in ids:
        anomalous = node in anomaly_nodes
        disp = np.linspace(0.0, 12.0, n) if anomalous else np.full(n, 0.05)
        frames.append(
            pd.DataFrame(
                {
                    "event_id": "E1",
                    "node_id": node,
                    "timestamp": ts,
                    "displacement": disp,
                    "tilt_x": 0.01 * np.ones(n),
                    "tilt_y": -0.01 * np.ones(n),
                    "anomaly_label": int(anomalous),
                }
            )
        )
    windowed = build_windows(pd.concat(frames, ignore_index=True)).df
    return windowed, coords_df


def test_feature_names_match_section_13() -> None:
    assert GROUP_C_FEATURES == (
        "neighbor_mean",
        "neighbor_std",
        "neighbor_anomaly_fraction",
        "spatial_coherence",
        "local_gradient",
        "local_strain",
        "hotspot_density",
        "distance_to_subsidence_center",
    )


def test_default_radius_is_one_and_a_half_grid_spacings() -> None:
    assert neighbour_radius_m() == pytest.approx(1.5 * 25.0)


def test_spatial_coherence_separates_single_node_from_coherent_movement() -> None:
    """§13 acceptance: single-node disturbance ≠ multi-node coherent movement."""
    # (a) ONE node moving alone: its flat neighbours have zero-slope ⇒ incoherent
    alone_w, coords = _small_mesh({"V0012"})
    alone = emit_group_c(alone_w, coords)
    lone = alone[(alone["node_id"] == "V0012") & (alone["window_index"] == 8)].iloc[0]
    assert lone["spatial_coherence"] == pytest.approx(0.0)

    # (b) the whole mesh moving together: every neighbour slopes the same way
    all_nodes = {f"V{i:04d}" for i in range(25)}
    coherent_w, coords = _small_mesh(all_nodes)
    coherent = emit_group_c(coherent_w, coords)
    mid = coherent[(coherent["node_id"] == "V0012") & (coherent["window_index"] == 8)].iloc[0]
    assert mid["spatial_coherence"] == pytest.approx(1.0)


def test_neighbour_statistics_exclude_self() -> None:
    """V0012 ramps to 12 mm; its 8 neighbours are constant 0.05 — if self were
    included, neighbor_mean would be pulled towards ~1.5, not ~0.05."""
    windowed, coords = _small_mesh({"V0012"})
    out = emit_group_c(windowed, coords)
    row = out[(out["node_id"] == "V0012") & (out["window_index"] == 8)].iloc[0]
    assert row["neighbor_mean"] == pytest.approx(0.05, abs=0.01)
    assert row["neighbor_std"] == pytest.approx(0.0, abs=1e-6)


def test_anomaly_fraction_and_hotspot_density() -> None:
    """V0007 sits at (0, 25): 8 neighbours within 37.5 m (one anomalous: V0012);
    hotspot radius 75 m (incl. self) covers 21 nodes, still one anomalous."""
    windowed, coords = _small_mesh({"V0012"})
    out = emit_group_c(windowed, coords)
    row = out[(out["node_id"] == "V0007") & (out["window_index"] == 8)].iloc[0]
    assert row["neighbor_anomaly_fraction"] == pytest.approx(1 / 8)
    assert row["hotspot_density"] == pytest.approx(1 / 21)


def test_local_gradient_and_strain_use_distance_scaling() -> None:
    """V0012 (≈9.2 mm at window 8) vs its neighbours (0.05) 25 m away:
    gradient/strain ≈ (own − 0.05)/25 per metre."""
    windowed, coords = _small_mesh({"V0012"})
    own_disp = windowed.loc[
        (windowed["node_id"] == "V0012") & (windowed["window_index"] == 8), "displacement_mean"
    ].iloc[0]
    out = emit_group_c(windowed, coords)
    row = out[(out["node_id"] == "V0012") & (out["window_index"] == 8)].iloc[0]
    expected = own_disp - 0.05
    assert row["local_strain"] == pytest.approx(expected / 25.0, rel=0.05)
    assert abs(row["local_gradient"]) == pytest.approx(abs(expected) / 25.0, rel=0.05)


def test_distance_to_center_detected_mode_is_inference_safe() -> None:
    """G-5 rule: 'detected' uses only pipeline-visible anomalies, not the oracle."""
    windowed, coords = _small_mesh({"V0012"})
    out = emit_group_c(windowed, coords)
    assert (out["center_mode"] == "detected").all()
    last = out[out["window_index"] == 8]
    centre_row = last[last["node_id"] == "V0012"].iloc[0]
    assert centre_row["distance_to_subsidence_center"] == pytest.approx(0.0), (
        "the only flagged node IS the detected centre"
    )


def test_oracle_mode_is_refused_for_model_features() -> None:
    windowed, coords = _small_mesh({"V0012"})
    with pytest.raises(SpatialFeatureError, match="oracle"):
        emit_group_c(windowed, coords, center_mode="oracle")


def test_no_anomaly_falls_back_to_max_displacement_node() -> None:
    windowed, coords = _small_mesh(set())  # nothing flagged
    out = emit_group_c(windowed, coords)
    last = out[out["window_index"] == 8]
    assert last["distance_to_subsidence_center"].notna().all()
    assert (out["center_mode"] == "detected").all()


def test_single_node_snapshot_gates_all_spatial_context() -> None:
    windowed, coords = _small_mesh({"V0012"})
    one = windowed[windowed["node_id"] == "V0012"]
    one_coord = coords[coords["node_id"] == "V0012"]
    out = emit_group_c(one, one_coord)
    assert out[list(GROUP_C_FEATURES)].isna().all().all()
    assert out["center_mode"].eq("gated_no_cotemporal_neighbors").all()
    assert out["spatial_gate_reason"].eq("single-node events, no co-temporal neighbours").all()


def test_group_c_ignores_permuted_anomaly_labels() -> None:
    windowed, coords = _small_mesh({"V0012"})
    first = emit_group_c(windowed, coords)
    changed = windowed.copy()
    changed["anomaly_label"] = np.random.default_rng(81).permutation(changed["anomaly_label"].to_numpy())
    second = emit_group_c(changed, coords)
    pd.testing.assert_frame_equal(first, second)


def test_invalid_center_mode_raises() -> None:
    windowed, coords = _small_mesh({"V0012"})
    with pytest.raises(SpatialFeatureError, match="center_mode"):
        emit_group_c(windowed, coords, center_mode="psychic")


def test_missing_channel_raises() -> None:
    windowed, coords = _small_mesh({"V0012"})
    with pytest.raises(SpatialFeatureError, match="displacement_mean"):
        emit_group_c(windowed.drop(columns=["displacement_mean"]), coords)


def test_windows_are_processed_per_event_independently() -> None:
    w1, coords = _small_mesh({"V0012"})
    frames = []
    for event, scale in (("E1", 1.0), ("E2", 2.0)):
        w = w1.copy()
        w["event_id"] = event
        w["displacement_mean"] = w["displacement_mean"] * scale
        frames.append(w)
    out = emit_group_c(pd.concat(frames, ignore_index=True), coords)
    counts = out.groupby(["event_id", "window_index"]).size()
    assert counts.nunique() == 1, "every (event, window) snapshot must have the same row count"
