#!/usr/bin/env python3
"""Generate deterministic event-level train, validation, and test splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

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
    """Split each scenario family deterministically using event identifiers."""
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

    # unit-exclusivity on the primary (regime) split
    assert_no_leakage(out, out["split"], "event_id")
    assert_no_leakage(out, out["split"], "generation_parameter_id")
    counts = out["split"].value_counts().to_dict()
    missing = [s for s in SPLIT_ORDER if counts.get(s, 0) == 0]
    if missing:
        raise SystemExit(f"regime holdout produced empty splits: {missing}")

    out.to_csv(args.out, index=False)
    # Keep the corpus manifest useful as the single provenance record even
    # when the generator is rerun (the generator itself knows nothing about
    # downstream split assignments).
    manifest_path = Path(args.events).with_name("dataset_manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["split_definition"] = {
            "type": "synthetic_parameter_holdout",
            "producer": "scripts/make_split_assignment.py",
            "default_split": "regime_holdout",
            "method": "scenario_family_regime_split",
            "seed": int(cfg.get("seed", validation_config().get("seed", 42))),
            "split_column": "split",
            "event_counts": {str(k): int(v) for k, v in counts.items()},
            "parameter_counts": {
                str(k): int(out.loc[out["split"] == k, "generation_parameter_id"].nunique())
                for k in SPLIT_ORDER
            },
            "test_range": {
                "scenario_types": sorted(
                    event_meta.loc[event_meta["scenario_family"].isin(test_families), "type"].astype(str).unique()
                )
            },
            "train_range": {"note": "assigned by event split"},
        }
        manifest["split_assignment_sha256"] = hashlib.sha256(Path(args.out).read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}: {n} events")
    print("regime split counts:", counts)
    print("legacy family-balanced counts:", out[LEGACY_COLUMN].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
