"""T-072 — §23.1 tabletop ground-truth protocol harness tests.

The fixture data lives in ``data/recorded/tabletop/`` (recorded campaign
schema; see its README). Tests assert the full §23.1 linkage chain, the
derived-risk-state discipline, and the §23.1 acceptance quantity
(mesh-estimated vs reference displacement error).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.tabletop_protocol import (
    TabletopProtocolError,
    confusion_counts,
    derived_risk_state,
    displacement_error_vs_reference,
    evaluate_tabletop,
    link_trials,
)

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "recorded" / "tabletop"


@pytest.fixture(scope="module")
def metadata() -> pd.DataFrame:
    return pd.read_csv(DATA / "trial_metadata.csv")


@pytest.fixture(scope="module")
def windowed() -> pd.DataFrame:
    return pd.read_csv(DATA / "processed_windowed_dataset.csv")


# --- the recorded fixture itself must satisfy the schema ---------------------

def test_recorded_fixture_present() -> None:
    assert (DATA / "trial_metadata.csv").is_file()
    assert (DATA / "processed_windowed_dataset.csv").is_file()


def test_fixture_links_cleanly(metadata: pd.DataFrame, windowed: pd.DataFrame) -> None:
    links = link_trials(metadata, windowed, raw=None)
    assert len(links) == 26
    assert {l.run_id for l in links} == set(metadata["trial_id"])
    assert all(l.n_windows > 0 for l in links)


def test_fixture_reference_never_exceeds_actuator(metadata: pd.DataFrame, windowed: pd.DataFrame) -> None:
    """§23.1 coherence on the recorded data: mechanism truth ≤ actuator."""
    links = link_trials(metadata, windowed, raw=None)
    for link in links:
        assert link.reference_mm_max <= link.actuator_achieved_mm + 1e-6


# --- linkage assertions (broken chains must raise) ---------------------------

def test_window_trial_missing_from_metadata_raises(metadata: pd.DataFrame, windowed: pd.DataFrame) -> None:
    orphan = windowed.copy()
    orphan.loc[orphan.index[:5], "trial_id"] = "T999"
    with pytest.raises(TabletopProtocolError, match="missing from metadata"):
        link_trials(metadata, orphan, raw=None)


def test_metadata_trial_without_windows_raises(metadata: pd.DataFrame, windowed: pd.DataFrame) -> pd.DataFrame:  # noqa: ARG001
    trimmed = windowed[windowed["trial_id"] != "T013"]
    with pytest.raises(TabletopProtocolError, match="no sensor windows"):
        link_trials(metadata, trimmed, raw=None)


def test_duplicate_window_keys_raise(metadata: pd.DataFrame, windowed: pd.DataFrame) -> None:
    dup = pd.concat([windowed, windowed.iloc[:3]], ignore_index=True)
    with pytest.raises(TabletopProtocolError, match="duplicate"):
        link_trials(metadata, dup, raw=None)


def test_reference_exceeding_actuator_raises(metadata: pd.DataFrame, windowed: pd.DataFrame) -> None:
    bad_meta = metadata.copy()
    bad_meta.loc[bad_meta.index[0], "achieved_displacement_mm_TD1"] = 0.5
    with pytest.raises(TabletopProtocolError, match="linkage broken"):
        link_trials(bad_meta, windowed, raw=None)


def test_missing_required_column_raises(metadata: pd.DataFrame) -> None:
    with pytest.raises(TabletopProtocolError, match="missing"):
        link_trials(metadata.drop(columns=["condition"]), pd.DataFrame({"trial_id": ["T001"]}))


# --- derived risk state -------------------------------------------------------

def test_derived_risk_state_uses_sensor_features_not_reference() -> None:
    # sensor says 20 mm, reference says 0 → the STATE must follow the sensor
    w = pd.DataFrame({
        "displacement_mm": [5.0, 20.0, 40.0],
        "known_displacement_mm": [0.0, 0.0, 0.0],
        "severity_class": [0, 0, 0],
    })
    out = derived_risk_state(w)
    assert out["derived_risk_state"].tolist() == ["NORMAL", "WARNING", "CRITICAL"]


def test_derived_risk_state_boundaries_exact() -> None:
    # plan §3.2: 15 mm belongs to the 0–15 band, 35 mm to the 15–35 band;
    # states flip strictly ABOVE the boundary
    w = pd.DataFrame({
        "displacement_mm": [15.0, 15.5, 35.0, 35.5],
        "known_displacement_mm": [0.0] * 4,
        "severity_class": [0] * 4,
    })
    out = derived_risk_state(w)
    assert out["derived_risk_state"].tolist() == ["NORMAL", "WARNING", "WARNING", "CRITICAL"]


# --- §23.1 acceptance quantity: displacement error ---------------------------

def test_displacement_error_matches_manual_computation(windowed: pd.DataFrame) -> None:
    out = displacement_error_vs_reference(windowed)
    sel = windowed[windowed["node_id"].isin(["N2", "N3"])]
    err = sel["displacement_mm"].to_numpy(dtype=float) - sel["known_displacement_mm"].to_numpy(dtype=float)
    assert out["n_windows"] == len(sel)
    assert out["mae_mm"] == pytest.approx(np.abs(err).mean())
    assert out["rmse_mm"] == pytest.approx(np.sqrt(np.mean(err**2)))
    assert out["bias_mm"] == pytest.approx(err.mean())


def test_displacement_error_excludes_reference_nodes(windowed: pd.DataFrame) -> None:
    out = displacement_error_vs_reference(windowed, nodes=("N1", "N4"))
    assert out["mae_mm"] >= 0.0  # references carry ~0 truth; error is sensor noise only
    sel = windowed[windowed["node_id"].isin(["N1", "N4"])]
    assert out["n_windows"] == len(sel)


def test_displacement_error_unknown_nodes_raise(windowed: pd.DataFrame) -> None:
    with pytest.raises(TabletopProtocolError, match="no windows"):
        displacement_error_vs_reference(windowed, nodes=("N9",))


# --- full protocol run --------------------------------------------------------

def test_evaluate_tabletop_full_run(metadata: pd.DataFrame, windowed: pd.DataFrame) -> None:
    result = evaluate_tabletop(metadata, windowed)
    assert result["n_trials"] == 26
    assert len(result["trial_linkage"]) == 26
    first = result["trial_linkage"][0]
    assert first["run_id"] == "T001"
    assert {"actuator_target_mm", "actuator_achieved_mm", "reference_mm_max",
            "n_raw_samples", "n_raw_nodes"} <= set(first)
    # every active-node window scored exactly once in the confusion
    n_active = windowed[windowed["node_id"].isin(["N2", "N3"])]
    assert sum(sum(v.values()) for v in result["risk_state_confusion"].values()) == len(n_active)


def test_evaluate_tabletop_displacement_error_is_honest(metadata: pd.DataFrame, windowed: pd.DataFrame) -> None:
    """The §23.1 number: sensor-derived vs mechanism-truth displacement."""
    result = evaluate_tabletop(metadata, windowed)
    err = result["displacement_error"]
    # the ultrasonic channel tracks the mechanism to ~mm level (drift + noise)
    assert 0.0 < err["mae_mm"] < 5.0
    assert err["n_windows"] > 0


def test_confusion_counts_helper() -> None:
    cm = confusion_counts(np.array([0, 1, 2, 1]), np.array([0, 1, 1, 1]), ("A", "B", "C"))
    assert cm["A"]["A"] == 1 and cm["B"]["B"] == 2 and cm["C"]["B"] == 1 and cm["C"]["C"] == 0
    assert sum(sum(v.values()) for v in cm.values()) == 4
