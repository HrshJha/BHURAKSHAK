"""Pipeline spatial confirmation must be co-temporal and event-local."""

from __future__ import annotations

import pandas as pd

from src.pipeline import _co_temporal_neighbor_confirmations


def test_unrelated_events_cannot_confirm_reused_nearby_nodes() -> None:
    frame = pd.DataFrame(
        [
            {"event_id": "E1", "node_id": "N1", "window_index": 0, "if_flag": True},
            {"event_id": "E2", "node_id": "N2", "window_index": 0, "if_flag": True},
        ],
        index=[18, 5],
    )
    coords = pd.DataFrame({"node_id": ["N1", "N2"], "x": [0.0, 1.0], "y": [0.0, 0.0]})
    counts = _co_temporal_neighbor_confirmations(frame, coords, radius_m=2.0)
    assert counts.tolist() == [0, 0]


def test_same_event_same_window_neighbors_can_confirm() -> None:
    frame = pd.DataFrame(
        [
            {"event_id": "E1", "node_id": "N1", "window_index": 0, "if_flag": True},
            {"event_id": "E1", "node_id": "N2", "window_index": 0, "if_flag": True},
            {"event_id": "E1", "node_id": "N3", "window_index": 1, "if_flag": True},
        ],
        index=[18, 5, 21],
    )
    coords = pd.DataFrame({"node_id": ["N1", "N2", "N3"], "x": [0.0, 1.0, 1.0], "y": [0.0, 0.0, 0.0]})
    counts = _co_temporal_neighbor_confirmations(frame, coords, radius_m=2.0)
    assert counts.tolist() == [1, 1, 0]
