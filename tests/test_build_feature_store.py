""" acceptance tests — feature store assembly and budget."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_feature_store import (
    FeatureBudgetError,
    build_feature_store,
    check_feature_budget,
    feature_names,
)
from src.features.group_c_spatial import GROUP_C_FEATURES
from src.features.provenance import assert_group_registered, assert_registered, model_input_allowlist
from src.features.windowing import build_windows

INTERVAL = 10.0 / 60.0


def _raw(n_nodes: int = 3, n: int = 144) -> tuple[pd.DataFrame, pd.DataFrame]:
    coords_list, frames = [], []
    for i in range(n_nodes):
        node = f"V{i:04d}"
        coords_list.append({"node_id": node, "x": 25.0 * i, "y": 0.0})
        ts = INTERVAL * np.arange(n)
        subs = np.linspace(0.0, 300.0 * (i + 1) / n_nodes, n)
        rng = np.random.default_rng(i)
        frames.append(
            pd.DataFrame(
                {
                    "event_id": "E1",
                    "node_id": node,
                    "timestamp": ts,
                    "tilt_x": 0.01 * rng.standard_normal(n),
                    "tilt_y": -0.01 * rng.standard_normal(n),
                    "tilt_magnitude": np.hypot(0.01, 0.01) * np.ones(n),
                    "displacement": subs + 0.1 * rng.standard_normal(n),
                    "strain": 1e-5 * rng.standard_normal(n),
                    "vibration_rms": 0.02 + 0.05 * (rng.random(n) > 0.95),
                    "vibration_peak": 0.05 + 0.1 * (rng.random(n) > 0.95),
                    "battery": np.linspace(4.1, 3.6, n),
                    "RSSI": -95.0 + 3.0 * rng.standard_normal(n),
                    "SNR": 5.0 * rng.standard_normal(n),
                    "packet_loss": (rng.random(n) > 0.98).astype(float),
                    "anomaly_label": (subs > 150).astype(int),
                    "risk_label": np.where(subs > 150, "WARNING", "NORMAL"),
                    "progression_label": np.where(subs > 150, "SLOW", "STABLE"),
                    "fault_label": "NONE",
                }
            )
        )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(coords_list)


def test_feature_names_within_section_13_budget() -> None:
    names = feature_names()
    assert 40 <= len(names) <= 70
    assert len(names) == len(set(names)), "no duplicate feature names"


def test_budget_guard_raises_outside_bounds() -> None:
    check_feature_budget(list(range(45)))  # within bounds: no raise
    with pytest.raises(FeatureBudgetError):
        check_feature_budget(list(range(39)))
    with pytest.raises(FeatureBudgetError):
        check_feature_budget(list(range(71)))


def test_assembly_produces_all_features_and_labels() -> None:
    raw, coords = _raw()
    model, report = build_feature_store(raw, coords)
    for f in feature_names():
        assert f in model.columns
    for lab in ("anomaly_label", "risk_label", "progression_label"):
        assert lab in model.columns
    assert report.budget_ok is True
    assert report.n_windows == len(model) == 9 * 3


def test_model_input_is_window_level_not_raw() -> None:
    raw, coords = _raw()
    model, _ = build_feature_store(raw, coords)
    assert "window_index" in model.columns
    assert "timestamp" not in model.columns


def test_labels_are_kept_separate_in_the_store() -> None:
    raw, coords = _raw()
    model, _ = build_feature_store(raw, coords)
    #: three independent label columns, not a collapsed binary flag
    assert model["risk_label"].isin(["NORMAL", "WARNING"]).all()
    assert set(model["anomaly_label"].unique()) <= {0, 1}
    assert model["progression_label"].isin(["STABLE", "SLOW"]).all()


def test_group_f_residual_consistent_with_inputs() -> None:
    raw, coords = _raw()
    model, _ = build_feature_store(raw, coords)
    assert model["physics_residual"].notna().all()
    # residual velocity of the first window of each series is zero
    first = model[model["window_index"] == 0]
    assert (first["physics_residual_velocity"] == 0.0).all()


def test_group_c_mode_is_recorded_in_artifact(tmp_path) -> None:
    raw, coords = _raw()
    model, report = build_feature_store(raw, coords)
    artifact = tmp_path / "features.parquet"
    model.to_parquet(artifact, index=False)
    restored = pd.read_parquet(artifact)
    assert report.center_mode == "detected"
    assert restored["center_mode"].isin(["detected", "gated_no_cotemporal_neighbors"]).all()
    assert set(report.gated_features) == set(GROUP_C_FEATURES)


def test_model_feature_build_refuses_oracle_mode() -> None:
    raw, coords = _raw()
    with pytest.raises(ValueError, match="oracle"):
        build_feature_store(raw, coords, center_mode="oracle")


def test_label_blind_and_permuted_labels_leave_features_identical() -> None:
    raw, coords = _raw()
    raw["scenario_family"] = "train_only_metadata"
    raw["true_panel_amplitude"] = 999.0
    baseline, _ = build_feature_store(raw, coords)
    feature_cols = feature_names()

    blind_raw = raw.drop(columns=[c for c in raw if c.endswith("_label") or c in {"scenario_family", "true_panel_amplitude"}])
    blind, _ = build_feature_store(blind_raw, coords)
    pd.testing.assert_frame_equal(
        baseline[feature_cols].reset_index(drop=True), blind[feature_cols].reset_index(drop=True)
    )

    shuffled = raw.copy()
    rng = np.random.default_rng(20260929)
    for label in ("anomaly_label", "risk_label", "progression_label", "fault_label"):
        shuffled[label] = rng.permutation(shuffled[label].to_numpy())
    changed, _ = build_feature_store(shuffled, coords)
    pd.testing.assert_frame_equal(
        baseline[feature_cols].reset_index(drop=True), changed[feature_cols].reset_index(drop=True)
    )


def test_all_feature_store_columns_have_provenance_and_gated_c_is_not_allowed() -> None:
    names = feature_names()
    assert_registered(names)
    allowlist = model_input_allowlist(names)
    assert not (set(names) - allowlist - {name for name in names if name.startswith(("neighbor_", "spatial_", "local_", "hotspot_", "distance_to_subsidence_center"))})


def test_every_group_emitter_feature_has_group_provenance() -> None:
    from src.features.group_a_physical import GROUP_A_FEATURES
    from src.features.group_b_temporal import GROUP_B_FEATURES
    from src.features.group_d_vibration import GROUP_D_FEATURES
    from src.features.group_e_health import GROUP_E_FEATURES
    from src.features.group_f_physics import GROUP_F_FEATURES
    from src.features.group_g_dgps import GROUP_G_FEATURES
    from src.features.group_h_insar import GROUP_H_FEATURES
    from src.features.group_i_terrain import GROUP_I_FEATURES
    from src.features.group_j_environmental import GROUP_J_FEATURES

    for group, names in (
        ("A_physical", GROUP_A_FEATURES), ("B_temporal", GROUP_B_FEATURES),
        ("C_spatial", GROUP_C_FEATURES), ("D_vibration", GROUP_D_FEATURES),
        ("E_sensor_health", GROUP_E_FEATURES), ("F_physics", GROUP_F_FEATURES),
        ("G_dgps", GROUP_G_FEATURES), ("H_insar", GROUP_H_FEATURES),
        ("I_terrain", GROUP_I_FEATURES), ("J_environmental", GROUP_J_FEATURES),
    ):
        assert_group_registered(group, names)


def test_feature_store_is_causal_when_future_rows_are_truncated() -> None:
    raw, coords = _raw()
    full, _ = build_feature_store(raw, coords)
    prefix_raw = raw.groupby(["event_id", "node_id"], sort=False, group_keys=False).head(80)
    prefix, _ = build_feature_store(prefix_raw, coords)
    cols = feature_names()
    prefix_end = int(prefix["window_index"].max())
    full_prefix = full[full["window_index"] <= prefix_end].sort_values(["event_id", "node_id", "window_index"])
    prefix = prefix.sort_values(["event_id", "node_id", "window_index"])
    pd.testing.assert_frame_equal(
        full_prefix[cols].reset_index(drop=True), prefix[cols].reset_index(drop=True)
    )


def test_missing_feature_raises_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.features.build_feature_store as bfs

    real = bfs.emit_group_d

    def broken(windowed):
        out = real(windowed)
        return out.drop(columns=["crest_factor"])

    monkeypatch.setattr(bfs, "emit_group_d", broken)
    raw, coords = _raw()
    with pytest.raises(FeatureBudgetError, match="crest_factor"):
        build_feature_store(raw, coords)


def test_per_series_windows_are_9_for_144_steps() -> None:
    raw, coords = _raw(n_nodes=1)
    model, report = build_feature_store(raw, coords)
    assert report.n_windows == 9
