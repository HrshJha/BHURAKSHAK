""" acceptance tests — leakage-safe validation splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.splits import (
    SplitsError,
    assert_no_leakage,
    event_split,
    node_split,
    random_row_split,
    spatial_split,
    synthetic_split,
    time_split,
)


def _frame(n_events: int = 20, n_nodes: int = 20, steps: int = 144) -> pd.DataFrame:
    """Events named <family>_<idx>_<inst>, one node per event, 144 steps."""
    families = ["stable_ground", "slow_subsidence", "accelerating_subsidence", "rapid_subsidence"]
    rows = []
    rng = np.random.default_rng(0)
    for e in range(n_events):
        eid = f"{families[e % 4]}_{e:04d}_00{e % 10}"
        node = f"V{e % n_nodes:04d}"
        x, y = rng.uniform(-250, 250), rng.uniform(-250, 250)
        ts = (10.0 / 60.0) * np.arange(steps)
        rows.append(
            pd.DataFrame(
                {
                    "event_id": eid,
                    "node_id": node,
                    "x": x,
                    "y": y,
                    "timestamp": ts,
                    # distinct amplitudes so the synthetic-split regime bands are well-defined
                    "displacement": np.linspace(0.0, 6.0 * (e % 5 + 1), steps),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _events_meta(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("event_id").agg(max_deformation=("displacement", "max")).reset_index()
    return g.rename(columns={"event_id": "id"})


def test_time_split_is_early_middle_late_per_series() -> None:
    df = _frame(n_events=4)
    split = time_split(df)
    assert set(split.unique()) == {"train", "validation", "test"}
    # within one series, labels must appear in temporal order
    g = df.assign(_s=split).sort_values("timestamp")
    order = g.groupby(["event_id", "node_id"])["_s"].agg(lambda s: list(dict.fromkeys(s)))
    assert all(o == ["train", "validation", "test"] for o in order)


def test_time_split_fractions_come_from_config() -> None:
    from src.config import validation_config

    cfg = validation_config()["time_split"]
    assert 0 < cfg["train_upper_fraction"] < cfg["val_upper_fraction"] < 1


def test_spatial_split_assigns_whole_nodes() -> None:
    df = _frame(n_events=20)
    split = spatial_split(df)
    assert set(split.unique()) == {"train", "validation", "test"}
    assert_no_leakage(df, split, "node_id")
    # every node keeps one label across all its windows
    work = df.assign(_s=split)
    assert (work.groupby("node_id")["_s"].nunique() == 1).all()


def test_node_split_expected_shape() -> None:
    df = _frame(n_events=20, n_nodes=20)
    split = node_split(df)
    per_node = df.assign(_s=split).groupby("node_id")["_s"].first().value_counts()
    assert per_node["train"] == 15, " example: nodes 1-15 train (of 20)"
    assert per_node["test"] >= 1
    assert_no_leakage(df, split, "node_id")


def test_event_split_slow_train_accelerating_test() -> None:
    df = _frame(n_events=20)
    split = event_split(df)
    work = df.assign(_s=split)
    slow_events = work[work.event_id.str.contains("slow|stable")].event_id.unique()
    fast_events = work[work.event_id.str.contains("accelerating|rapid")].event_id.unique()
    assert (work[work.event_id.isin(slow_events)]["_s"] != "test").all()
    assert (work[work.event_id.isin(fast_events)]["_s"] == "test").all()
    assert_no_leakage(df, split, "event_id")


def test_event_split_covers_all_families_or_raises() -> None:
    df = _frame(n_events=4)
    patched = df.assign(event_id=df["event_id"].str.replace("slow_subsidence", "mystery_family"))
    with pytest.raises(SplitsError, match="not covered"):
        event_split(patched)


def test_synthetic_split_holds_out_top_regime() -> None:
    df = _frame(n_events=20)
    meta = _events_meta(df)
    split = synthetic_split(df, meta)
    work = df.assign(_s=split)
    # the highest-deformation events must be test
    ev_max = work.groupby("event_id").agg(disp=("displacement", "max"), s=("_s", "first"))
    top_events = ev_max.nlargest(4, "disp")
    assert (top_events["s"] == "test").all(), "top parameter regime held out for test"
    # no event spans two splits
    assert_no_leakage(df, split, "event_id")


def test_all_splits_refuse_row_level_leakage() -> None:
    """Each split is exclusive on ITS OWN unit ( table): spatial/node splits
 on node_id, event/synthetic splits on event_id; the time split guarantees
 per-series temporal ordering instead (nodes may recur across events)."""
    df = _frame(n_events=8)
    assert_no_leakage(df, spatial_split(df), "node_id")
    assert_no_leakage(df, node_split(df), "node_id")
    assert_no_leakage(df, event_split(df), "event_id")
    # time split: labels appear in temporal order within every series
    work = df.assign(_s=time_split(df)).sort_values("timestamp")
    order = work.groupby(["event_id", "node_id"])["_s"].agg(lambda s: list(dict.fromkeys(s)))
    assert all(o == ["train", "validation", "test"] for o in order)


def test_random_row_split_is_refused() -> None:
    """: random row-shuffling across time is explicitly disallowed."""
    df = _frame(n_events=4)
    with pytest.raises(SplitsError, match="disallowed"):
        random_row_split(df)


def test_assert_no_leakage_detects_a_real_violation() -> None:
    df = _frame(n_events=4)
    split = pd.Series(["train", "test"] * (len(df) // 2), index=df.index)
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_leakage(df, split, "node_id")


def test_invalid_fractions_raise() -> None:
    import src.evaluation.splits as sp

    real = sp.validation_config

    def fake():
        data = real()
        data["time_split"]["train_upper_fraction"] = 0.9
        data["time_split"]["val_upper_fraction"] = 0.8
        return data

    sp.validation_config = fake
    try:
        with pytest.raises(SplitsError):
            sp.time_split(_frame(n_events=2))
    finally:
        sp.validation_config = real
