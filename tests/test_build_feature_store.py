"""T-042 acceptance tests — feature store assembly and §13 budget."""

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
    # §12: three independent label columns, not a collapsed binary flag
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


def test_group_c_distance_uses_oracle_on_synthetic() -> None:
    raw, coords = _raw()
    model, _ = build_feature_store(raw, coords)
    assert (model["center_mode"] == "oracle").all()


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
