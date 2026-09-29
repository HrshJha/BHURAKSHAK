#!/usr/bin/env python3
""" — run the Isolation Forest ablation (A/B/C) on the feature store.

 ablation plan: (A) physical-only features, (B) + temporal, (C) + spatial —
"measured against whether adding spatial coherence reduces false alarms".

Protocol (-safe):
- split by EVENT (never by row): within each scenario family, events are
 assigned 60% train / 20% validation / 20% test with a seeded rng; the
 assignment is persisted to data/features/split_assignment.csv and reused by
 later model tasks (// build the full machinery).
- train: healthy-baseline windows of train events only ( module);
- validation: healthy windows set the FAR threshold (p99, ≈1% FAR by design);
- test: false-alarm rate on healthy test windows + detection rate on
 anomalous test windows, per feature set.

Outputs: experiments/if_ablation_abc.json, reports/if_ablation_abc.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.anomaly.isolation_forest import ABLATION_STEPS, train_isolation_forest

FEATURES_PATH = REPO_ROOT / "data" / "features" / "features_v2.parquet"
SPLIT_PATH = REPO_ROOT / "data" / "features" / "split_assignment.csv"
OUT_JSON = REPO_ROOT / "experiments" / "if_ablation_abc.json"
OUT_MD = REPO_ROOT / "reports" / "if_ablation_abc.md"

SEED = 42
FAMILY_SPLIT = {"train": 0.60, "validation": 0.20, "test": 0.20}


def build_event_splits(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Assign whole events to train/validation/test within each scenario family.

 The family is the event's scenario base (the label before the trailing
 instance suffix), so every split contains every scenario type. The mapping
 is deterministic given the seed and persisted for reuse.
 """
    rng = np.random.default_rng(seed)
    families = df["event_id"].str.rsplit("_", n=2).str[0]
    rows = []
    for family, events in df.groupby(families)["event_id"].agg(lambda s: sorted(set(s))).items():
        events = list(events)
        rng.shuffle(events)
        n = len(events)
        n_tr = int(round(n * FAMILY_SPLIT["train"]))
        n_va = int(round(n * FAMILY_SPLIT["validation"]))
        for i, e in enumerate(events):
            split = "train" if i < n_tr else ("validation" if i < n_tr + n_va else "test")
            rows.append({"event_id": e, "scenario_family": family, "split": split})
    return pd.DataFrame(rows)


def evaluate(df: pd.DataFrame, groups: list[str], far_alpha: float) -> dict:
    """Train on healthy train windows; report FAR (test healthy) + detection (test anomalous)."""
    fitted = train_isolation_forest(df, feature_groups=groups)
    test = df[df["split"] == "test"]
    scores = fitted.anomaly_score(test)
    labels = test["anomaly_label"].to_numpy(dtype=int)
    healthy = labels == 0
    far = float((scores[healthy] > fitted.threshold).mean())
    det = float((scores[~healthy] > fitted.threshold).mean())
    # separation between test healthy and test anomalous
    y = (labels > 0).astype(int)
    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(y, scores))
    return {
        "feature_groups": list(groups),
        "n_features": len(fitted.features),
        "features": fitted.features,
        "n_training_windows": fitted.n_training_windows,
        "threshold": fitted.threshold,
        "threshold_rule": fitted.threshold_rule,
        "test_far_healthy": far,
        "test_detection_anomalous": det,
        "test_auc": auc,
    }


def main() -> int:
    raise SystemExit("This legacy ablation script reads a previously used test split and cannot produce valid final results.")
    df = pd.read_parquet(FEATURES_PATH)
    print(f"feature store: {len(df):,} windows, {df.event_id.nunique():,} events")

    if SPLIT_PATH.exists():
        splits = pd.read_csv(SPLIT_PATH)
        print("reusing persisted split assignment")
    else:
        splits = build_event_splits(df)
        SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        splits.to_csv(SPLIT_PATH, index=False)
        print("split assignment created and persisted")

    df = df.merge(splits, on="event_id", how="left")
    assert df["split"].notna().all(), "every event must have a split"
    counts = df.groupby(["scenario_family", "split"])["event_id"].nunique()
    print("\nevents per split:")
    print(df.groupby("split")["event_id"].nunique().to_string())
    for split in ("train", "validation", "test"):
        fams = counts.xs(split, level="split")
        assert len(fams) > 0, f"{split} empty"
    print("split mixes every scenario family:", all(
        set(counts.xs(s, level="split").index) == set(counts.index.get_level_values(0).unique())
        for s in ("train", "validation", "test")
    ))

    from src.config import anomaly_config

    far_alpha = float(anomaly_config()["far_alpha"])
    steps = {
        "A_physical_only": ["A_physical"],
        "B_add_temporal": ["A_physical", "B_add_temporal"],
        "C_add_spatial": ["A_physical", "B_add_temporal", "C_add_spatial"],
    }
    results = {}
    for name, groups in steps.items():
        res = evaluate(df, groups, far_alpha)
        results[name] = res
        print(f"\n{name}: features={res['n_features']} "
              f"FAR={res['test_far_healthy']:.4f} detection={res['test_detection_anomalous']:.4f} "
              f"AUC={res['test_auc']:.4f}")

    a, b, c = (results[k] for k in steps)
    verdict = {
        "question": "does adding spatial coherence reduce false alarms?",
        "far_A_physical": a["test_far_healthy"],
        "far_B_add_temporal": b["test_far_healthy"],
        "far_C_add_spatial": c["test_far_healthy"],
        "far_reduced_B_over_A": b["test_far_healthy"] < a["test_far_healthy"],
        "far_reduced_C_over_B": c["test_far_healthy"] < b["test_far_healthy"],
        "auc_A": a["test_auc"], "auc_B": b["test_auc"], "auc_C": c["test_auc"],
    }

    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(
        {"seed": SEED, "far_alpha": far_alpha, "split_rule": "event-level within scenario family, 60/20/20",
         "ablation": results, "verdict": verdict}, indent=2))

    lines = [
        "#  Isolation Forest ablation — A/B/C ()",
        "",
        f"Protocol: event-level splits (60/20/20 within each scenario family, seed {SEED}); "
        f"healthy baseline = triple-healthy mask (); threshold = p{far_alpha:.2f} of validation "
        "healthy scores; all numbers on held-out test events.",
        "",
        "| Set | Features | FAR (healthy test) | Detection (anomalous test) | AUC |",
        "|---|---|---|---|---|",
    ]
    for name in steps:
        r = results[name]
        lines.append(
            f"| {name} | {r['n_features']} | {r['test_far_healthy']:.4f} | "
            f"{r['test_detection_anomalous']:.4f} | {r['test_auc']:.4f} |"
        )
    lines += [
        "",
        f"**Verdict — {verdict['question']}**",
        "",
        f"- A (physical only) FAR: {verdict['far_A_physical']:.4f}",
        f"- B (+temporal) FAR: {verdict['far_B_add_temporal']:.4f} "
        f"({'reduced' if verdict['far_reduced_B_over_A'] else 'did not reduce'} vs A)",
        f"- C (+spatial) FAR: {verdict['far_C_add_spatial']:.4f} "
        f"({'reduced' if verdict['far_reduced_C_over_B'] else 'did not reduce'} vs B)",
        "",
        "_Caveat (recorded honestly): on the synthetic gate dataset every event "
        "carries a single node, so Group C neighbourhood features degenerate to "
        "self-only values; the spatial-coherence effect measured here is bounded "
        "by that synthetic limitation and must be re-measured when multi-node "
        "events exist._",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT_JSON} and {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
