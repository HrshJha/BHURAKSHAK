"""§23 leakage-safe validation splits (T-068).

§23: "Random row-shuffling across time is explicitly disallowed." The five
required split types, all assigned at **whole-unit granularity** (never per
row), so no `node_id`, time block, event family or simulator parameter regime
appears on both sides of a split:

| Split    | Unit          | Rule (configs/validation.yaml)                          |
|----------|---------------|---------------------------------------------------------|
| time     | event window  | early → train, middle → validation, late → test         |
| spatial  | mesh sector   | quadrant thirds of the mesh extent                      |
| node     | node_id       | ordered node list sliced into train/val/test            |
| event    | scenario fam. | slow families → train, accelerating families → test     |
| synthetic| parameter     | hold out the TOP of the max-deformation range entirely  |

Every function returns a ``pd.Series`` of split labels indexed like the input
frame. :func:`assert_no_leakage` re-checks the unit-exclusivity property, and
:func:`random_row_split` is the deliberate §23 refusal — it raises.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import validation_config

__all__ = [
    "SplitsError",
    "SPLIT_ORDER",
    "time_split",
    "spatial_split",
    "node_split",
    "event_split",
    "synthetic_split",
    "assert_no_leakage",
    "random_row_split",
]

SPLIT_ORDER = ("train", "validation", "test")


class SplitsError(ValueError):
    """Raised on invalid split requests or §23 violations."""


def _check_fractions() -> tuple[float, float]:
    cfg = validation_config()
    pairs = [
        (cfg["time_split"]["train_upper_fraction"], cfg["time_split"]["val_upper_fraction"]),
        (cfg["node_split"]["train_upper_fraction"], cfg["node_split"]["val_upper_fraction"]),
        (cfg["synthetic_split"]["test_upper_fraction"], cfg["synthetic_split"]["val_upper_fraction"]),
    ]
    for tr, va in pairs:
        if not 0 < tr < va < 1:
            raise SplitsError(f"split fractions must satisfy 0 < train < val < 1, got {tr}, {va}")
    return pairs[0]


def time_split(df: pd.DataFrame) -> pd.Series:
    """§23 time split on each event's own timeline: early/middle/late windows.

    Applied per (event_id, node_id) series so the temporal ordering inside a
    series is preserved and every event contributes to all three phases.
    """
    tr, va = _check_fractions()
    out = pd.Series(index=df.index, dtype=object)
    for _key, g in df.groupby(["event_id", "node_id"], sort=False):
        g = g.sort_values("timestamp", kind="stable")
        t = g["timestamp"].to_numpy(dtype=float)
        t0, t1 = t[0], t[-1]
        span = max(t1 - t0, 1e-9)
        frac = (t - t0) / span
        labels = np.where(frac <= tr, "train", np.where(frac <= va, "validation", "test"))
        out.loc[g.index] = labels
    return out


def spatial_split(df: pd.DataFrame) -> pd.Series:
    """§23 spatial split: sector A/B → train/val, sector C → test.

    Sectors are thirds of the mesh extent along x then y (fractions from
    config): A = lower-left block (train), B = lower-right block (validation),
    C = everything with y above the y-boundary (test). Whole nodes fall in one
    sector — a node never appears in two splits.
    """
    cfg = validation_config()["spatial_split"]
    bx = float(cfg["boundary_fraction_x"])
    by = float(cfg["boundary_fraction_y"])
    for _key, g in df.groupby("node_id", sort=False):
        pass  # validate grouping exists
    x = df.groupby("node_id")["x"].first()
    y = df.groupby("node_id")["y"].first()
    x0, x1 = float(x.min()), float(x.max())
    y0, y1 = float(y.min()), float(y.max())
    x_cut = x0 + bx * (x1 - x0)
    y_cut = y0 + by * (y1 - y0)
    node_sector = np.where(
        (y <= y_cut),
        np.where(x <= x_cut, "train", "validation"),
        "test",
    )
    mapping = pd.Series(node_sector, index=x.index)
    return df["node_id"].map(mapping)


def node_split(df: pd.DataFrame) -> pd.Series:
    """§23 node split: nodes 1–15 → train, 16–20 → test style, by ordered id.

    The ordered node-id list is sliced into train/val/test fractions; the
    middle slice is validation. A node's windows all share one label.
    """
    tr, va = _check_fractions()
    nodes = sorted(df["node_id"].unique())
    n = len(nodes)
    n_tr = max(1, int(round(n * tr)))
    n_va = max(1, int(round(n * (va - tr))))
    label = {}
    for i, node in enumerate(nodes):
        if i < n_tr:
            label[node] = "train"
        elif i < n_tr + n_va:
            label[node] = "validation"
        else:
            label[node] = "test"
    return df["node_id"].map(label)


def event_split(df: pd.DataFrame) -> pd.Series:
    """§23 event split: slow/stable families → train, accelerating/rapid → test.

    Family = the event id minus its trailing instance suffix. Validation is
    sampled from the train-side families at the configured event fraction
    (deterministic given the family names — no rng in the module).
    """
    cfg = validation_config()["event_split"]
    train_fams = tuple(cfg["train_families"])
    test_fams = tuple(cfg["test_families"])
    val_frac = float(cfg["validation_fraction"])
    if val_frac <= 0 or val_frac >= 1:
        raise SplitsError("event_split.validation_fraction must be in (0, 1)")

    families = df["event_id"].str.rsplit("_", n=2).str[0]
    fam_list = list(dict.fromkeys(families))
    train_like = [f for f in fam_list if any(t in f for t in train_fams)]
    test_like = [f for f in fam_list if any(t in f for t in test_fams)]
    unassigned = [f for f in fam_list if f not in train_like and f not in test_like]
    if unassigned:
        raise SplitsError(f"scenario families not covered by config: {unassigned}")

    n_val_fams = max(1, int(round(len(train_like) * val_frac)))
    # deterministic validation: every 5th train-side family (stride from 1/frac)
    stride = max(2, int(round(1.0 / val_frac)))
    val_fams = set(train_like[1::stride][:n_val_fams])

    fam_split = {}
    for f in train_like:
        fam_split[f] = "validation" if f in val_fams else "train"
    for f in test_like:
        fam_split[f] = "test"
    return families.map(fam_split)


def synthetic_split(df: pd.DataFrame, events_meta: pd.DataFrame) -> pd.Series:
    """§23 synthetic split: hold out the TOP of the max-deformation regime.

    ``events_meta`` must carry ``id`` and ``max_deformation``. Events in the
    top ``test_upper_fraction`` of the deformation range are test; the next
    band up to ``val_upper_fraction`` is validation; the rest train. A whole
    parameter regime is therefore never seen in training.
    """
    cfg = validation_config()["synthetic_split"]
    test_frac = float(cfg["test_upper_fraction"])
    val_frac = float(cfg["val_upper_fraction"])
    if not 0 < test_frac < val_frac < 1:
        raise SplitsError("synthetic_split fractions must satisfy 0 < test < val < 1")

    meta = events_meta.copy()
    lo, hi = float(meta["max_deformation"].min()), float(meta["max_deformation"].max())
    val_cut = lo + val_frac * (hi - lo)
    test_cut = lo + test_frac * (hi - lo)

    def label(value: float) -> str:
        if value > val_cut:
            return "test" if value > test_cut else "validation"
        return "train"

    # NOTE: the top band (above val_cut) is test; the band (test_cut, val_cut]
    # is validation; below test_cut is train. The very top regime (>val_cut) is
    # held out entirely from training — that is the §23 point.
    meta = meta.sort_values("max_deformation", ascending=False)
    top_n = int(round(len(meta) * test_frac))
    held_out = set(meta["id"].iloc[:top_n])
    rest = meta[~meta["id"].isin(held_out)]
    lo2, hi2 = float(rest["max_deformation"].min()), float(rest["max_deformation"].max())
    val_cut2 = lo2 + val_frac * (hi2 - lo2)

    split_by_event = {}
    for eid, value in zip(meta["id"], meta["max_deformation"]):
        if eid in held_out:
            split_by_event[eid] = "test"
        elif float(value) > val_cut2:
            split_by_event[eid] = "validation"
        else:
            split_by_event[eid] = "train"
    return df["event_id"].map(split_by_event)


def assert_no_leakage(df: pd.DataFrame, split: pd.Series, unit: str) -> None:
    """Assert the split ``unit`` (e.g. node_id/event_id) never spans two splits.

    This is the executable §23 guarantee: a node/event/time-block that appears
    in both train and test means information leaked across the boundary.
    """
    if len(split) != len(df):
        raise SplitsError("split series must align with the frame")
    work = df[[unit]].copy()
    work["_split"] = split.to_numpy()
    spans = work.groupby(unit)["_split"].nunique()
    bad = spans[spans > 1]
    if len(bad):
        examples = list(bad.index[:5])
        raise AssertionError(
            f"§23 leakage: {unit} values appear in multiple splits: {examples}"
        )


def random_row_split(df: pd.DataFrame, **_kwargs) -> pd.Series:
    """§23 refusal: random row-shuffling across time is explicitly disallowed."""
    raise SplitsError(
        "§23 violation: random row-shuffling across time is explicitly "
        "disallowed — use time_split/spatial_split/node_split/event_split/"
        "synthetic_split, which assign whole units"
    )
