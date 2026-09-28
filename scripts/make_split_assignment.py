#!/usr/bin/env python3
"""Produce data/features/split_assignment.csv — reproducibly (fixall Phase 1.1).

Closes the audit CRITICAL: the split file had ten consumers and no producer.
This script is the single producer. It writes BOTH splits:

- ``split``             — the §35 unseen-parameter-regime holdout
                          (src/evaluation/splits.regime_split; test = whole
                          scenario regimes, default for all §35 claims);
- ``split_family_balanced`` — the legacy 60/20/20 family-balanced assignment
                          (seeded permutation), kept so pre-fixall results
                          stay comparable (the shipped file's assignment).

Both columns are present in one file; readers keyed on ``split`` get the
regime holdout, the comparability column keeps old/new numbers linkable.
Seed and parameters come from configs/validation.yaml (NFR-6); the output is
byte-reproducible from a fresh clone (verified by tests/test_split_assignment.py
and by the Phase-0 sha256 discipline).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import validation_config  # noqa: E402
from src.evaluation.splits import (  # noqa: E402
    SPLIT_ORDER,
    assert_no_leakage,
    regime_split,
)

OUT_PATH = REPO_ROOT / "data" / "features" / "split_assignment.csv"
LEGACY_COLUMN = "split_family_balanced"


def family_balanced_split(events: pd.Series, fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)) -> pd.Series:
    """The legacy family-balanced assignment, made deterministic.

    Within each scenario family, events are sorted by id and sliced 60/20/20.
    This reproduces the original shipped file's *distribution* (375/125/125
    per 625-event family) but not its exact per-event assignment — that came
    from an unknown process and is preserved verbatim in git history at tag
    ``pre-fixall`` (data/features/split_assignment.csv) for old-vs-new metric
    comparisons in reports/split_change.md. No rng: the sorted-order rule is
    byte-reproducible from any clone.
    """
    train_f, val_f, _ = fractions
    out = pd.Series("train", index=events.index, dtype=object)
    families = events.str.rsplit("_", n=2).str[0]
    for fam, idx in events.groupby(families).groups.items():
        order = sorted(idx)
        n = len(order)
        n_train = int(round(train_f * n))
        n_val = int(round(val_f * n))
        out.loc[order[:n_train]] = "train"
        out.loc[order[n_train : n_train + n_val]] = "validation"
        out.loc[order[n_train + n_val :]] = "test"
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", default=str(REPO_ROOT / "data" / "synthetic" / "synthetic_nodes.csv"))
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    raw = pd.read_csv(args.events, usecols=["event_id"])
    events = pd.Series(sorted(raw["event_id"].unique()), name="event_id")
    n = len(events)
    if n == 0:
        raise SystemExit("no events found in the generator output")

    families = events.str.rsplit("_", n=2).str[0]
    meta_path = Path(args.events).with_name("synthetic_events.csv")
    event_meta = pd.read_csv(meta_path).rename(columns={"id": "event_id"})
    required = {"event_id", "type", "center", "max_deformation", "rate"}
    if not required <= set(event_meta.columns):
        raise SystemExit(f"event metadata missing generating parameters: {sorted(required - set(event_meta.columns))}")
    event_meta = event_meta.set_index("event_id").loc[events.to_list()].reset_index()
    event_meta["scenario_family"] = families.to_numpy()
    event_meta["generation_parameter_id"] = [
        hashlib.sha256(json.dumps(
            [fam, typ, center, float(max_d), float(rate)],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        for fam, typ, center, max_d, rate in zip(
            event_meta["scenario_family"], event_meta["type"], event_meta["center"],
            event_meta["max_deformation"], event_meta["rate"],
        )
    ]
    cfg = validation_config()["regime_split"]
    test_families = set(cfg["test_families"])
    val_frac = float(cfg["validation_fraction"])
    stride = max(2, int(round(1.0 / val_frac)))
    regimes = pd.Series("train", index=event_meta.index, dtype=object)
    regimes[event_meta["scenario_family"].isin(test_families)] = "test"
    for family, family_rows in event_meta[~event_meta["scenario_family"].isin(test_families)].groupby("scenario_family", sort=True):
        unique_params = sorted(family_rows["generation_parameter_id"].unique())
        validation_params = set(unique_params[::stride])
        val_mask = family_rows["generation_parameter_id"].isin(validation_params)
        regimes.loc[family_rows.index[val_mask]] = "validation"

    legacy = family_balanced_split(events)

    out = pd.DataFrame(
        {
            "event_id": events,
            "scenario_family": event_meta["scenario_family"],
            "generation_parameter_id": event_meta["generation_parameter_id"],
            "split": regimes,
            LEGACY_COLUMN: legacy,
        }
    )

    # §23 unit-exclusivity on the primary (regime) split
    assert_no_leakage(out, out["split"], "event_id")
    assert_no_leakage(out, out["split"], "generation_parameter_id")
    counts = out["split"].value_counts().to_dict()
    missing = [s for s in SPLIT_ORDER if counts.get(s, 0) == 0]
    if missing:
        raise SystemExit(f"regime holdout produced empty splits: {missing}")

    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {n} events")
    print("regime split counts:", counts)
    print("legacy family-balanced counts:", out[LEGACY_COLUMN].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
