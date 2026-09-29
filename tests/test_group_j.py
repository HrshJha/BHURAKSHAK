""" — Feature Group J (environmental) gate & synthetic emission tests."""

from __future__ import annotations

import pandas as pd
import pytest

import src.features.group_j_environmental as gj
from src.config import environmental_config
from src.features.group_j_environmental import (
    GROUP_J_FEATURES,
    GroupJError,
    assert_gate_compliant,
    emit_group_j,
    gate_status,
    model_feature_names,
)


@pytest.fixture()
def windowed() -> pd.DataFrame:
    # 2 events × 2 nodes × 3 windows, timestamps on a 0.6 h stride
    rows = []
    for event in ("EV_A", "EV_B"):
        for node in ("V0000", "V0009"):
            for w in range(3):
                rows.append({"event_id": event, "node_id": node, "window_index": w, "window_timestamp": 0.6 * w})
    return pd.DataFrame(rows)


def _gate(**overrides):
    gate = {
        "status": "not_run",
        "ablation_id": None,
        "metric": None,
        "delta": None,
        "decided_by": None,
        "decided_on": None,
        "min_pr_auc_delta": 0.01,
    }
    gate.update(overrides)
    return {"enabled": False, "gate": gate, "sources": environmental_config()["sources"]}


# --- schema -----------------------------------------------------------------

def test_group_j_feature_names_exact() -> None:
    schema = environmental_config()  # smoke: config loads
    assert schema["sources"]["rainfall"].startswith("synthetic")
    assert GROUP_J_FEATURES == (
        "rainfall",
        "temperature",
        "land_surface_temperature",
        "land_cover",
    )


def test_emit_schema_and_rows_align(windowed: pd.DataFrame) -> None:
    out = emit_group_j(windowed)
    assert len(out) == len(windowed)
    assert list(out["event_id"]) == list(windowed["event_id"])
    assert set(GROUP_J_FEATURES) <= set(out.columns)
    assert out.attrs["gate_status"] == "not_in_model"
    assert out["source"].str.startswith("synthetic").all()


def test_gate_status_default_excluded() -> None:
    assert gate_status() == "excluded"
    assert model_feature_names() == ()


def test_frame_without_gate_status_rejected() -> None:
    with pytest.raises(GroupJError, match="no gate_status"):
        assert_gate_compliant(pd.DataFrame({"rainfall": [0.0]}))


def test_not_in_model_frame_rejected_as_model_input(windowed: pd.DataFrame) -> None:
    out = emit_group_j(windowed)
    with pytest.raises(GroupJError, match="not_in_model"):
        assert_gate_compliant(out)


# --- gate mechanics (monkeypatched config) ----------------------------------

def test_gate_included_but_master_flag_off_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _gate(status="included", ablation_id="ablation_J", metric="pr_auc",
                delta=0.02, decided_by="", decided_on="2026-09-27")
    cfg["enabled"] = False
    monkeypatch.setattr(gj, "environmental_config", lambda: cfg)
    with pytest.raises(GroupJError, match="master flag"):
        model_feature_names()


def test_gate_incomplete_evidence_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _gate(status="included", ablation_id=None, metric="pr_auc",
                delta=0.02, decided_by="", decided_on="2026-09-27")
    cfg["enabled"] = True
    monkeypatch.setattr(gj, "environmental_config", lambda: cfg)
    with pytest.raises(GroupJError, match="ablation_id"):
        model_feature_names()


def test_gate_delta_below_threshold_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _gate(status="included", ablation_id="ablation_J", metric="pr_auc",
                delta=0.005, decided_by="", decided_on="2026-09-27", min_pr_auc_delta=0.01)
    cfg["enabled"] = True
    monkeypatch.setattr(gj, "environmental_config", lambda: cfg)
    with pytest.raises(GroupJError, match="below the inclusion threshold"):
        model_feature_names()


def test_gate_pass_yields_feature_names(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _gate(status="included", ablation_id="ablation_J", metric="pr_auc",
                delta=0.03, decided_by="", decided_on="2026-09-27")
    cfg["enabled"] = True
    monkeypatch.setattr(gj, "environmental_config", lambda: cfg)
    names = model_feature_names()
    assert names == GROUP_J_FEATURES
    assert gate_status() == "included"


def test_excluded_status_never_leaks_names(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _gate(status="pending", ablation_id="ablation_J", metric="pr_auc",
                delta=0.05, decided_by="x", decided_on="2026-09-27")
    cfg["enabled"] = True  # even with the flag on, non-included status ⇒ empty
    monkeypatch.setattr(gj, "environmental_config", lambda: cfg)
    assert model_feature_names() == ()


# --- emission semantics ------------------------------------------------------

def test_temperature_diurnal_shape(windowed: pd.DataFrame) -> None:
    import numpy as np

    out = emit_group_j(windowed)
    temp = out["temperature"].to_numpy(dtype=float)
    assert temp.min() >= 28.0 - 6.0 - 1e-9 and temp.max() <= 28.0 + 6.0 + 1e-9
    # peak at 15 h: t=0 sits 15 h before the peak → cos(−225°) = −√2/2
    expected0 = 28.0 + 6.0 * np.cos(2.0 * np.pi * (0.0 - 15.0) / 24.0)
    assert abs(temp[0] - expected0) < 1e-9


def test_lst_hotter_than_air(windowed: pd.DataFrame) -> None:
    out = emit_group_j(windowed)
    assert (out["land_surface_temperature"] > out["temperature"]).all()


def test_rainfall_shared_per_event_and_deterministic(windowed: pd.DataFrame) -> None:
    a = emit_group_j(windowed)
    b = emit_group_j(windowed)
    pd.testing.assert_series_equal(a["rainfall"], b["rainfall"])
    # regional: all rows of one event see the same rain at the same window
    for _, g in a.groupby("event_id", sort=False):
        by_window = g.groupby("window_index")["rainfall"].nunique()
        assert (by_window <= 1).all()


def test_rainfall_nonnegative(windowed: pd.DataFrame) -> None:
    assert (emit_group_j(windowed)["rainfall"] >= 0.0).all()


def test_land_cover_static_per_node(windowed: pd.DataFrame) -> None:
    out = emit_group_j(windowed)
    assert out.groupby("node_id")["land_cover"].nunique().eq(1).all()
    assert set(out["land_cover"].unique()) <= {0, 1, 2}


def test_missing_window_keys_rejected() -> None:
    with pytest.raises(GroupJError, match="missing"):
        emit_group_j(pd.DataFrame({"event_id": ["EV_A"]}))


def test_timestamps_hours_length_mismatch_rejected(windowed: pd.DataFrame) -> None:
    with pytest.raises(GroupJError, match="align"):
        emit_group_j(windowed, timestamps_hours=[0.0, 0.6])
