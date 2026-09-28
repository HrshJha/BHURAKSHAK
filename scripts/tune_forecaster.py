#!/usr/bin/env python3
"""Grouped, development-only hyperparameter search for the physical forecaster.

Risk and anomaly models are intentionally excluded: their current spatial
feature inputs contain target-derived anomaly labels (see tuning_notes.md).
The held-out regime test groups are never loaded into the training/evaluation
frames here.
"""
from __future__ import annotations

import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import model_params_config
from src.forecasting.temporal_model import _build_multi_horizon, train_temporal_forecaster

STORE = ROOT / "data/features/features_v2.parquet"
SPLITS = ROOT / "data/features/split_assignment.csv"
OUT = ROOT / "reports/tuning"
PARAMS_PATH = ROOT / "configs/model_params.yaml"
CHANNELS = ("displacement", "tilt_x", "tilt_y")
HORIZONS = (1,)  # only next-window is representable by the 9-window sequences
SEEDS = (42, 43, 44)
FOLDS = 3
EPOCHS = 30
BATCH_SIZE = 2048


def prepared_data() -> pd.DataFrame:
    frame = pd.read_parquet(STORE)
    split_map = pd.read_csv(SPLITS)[["event_id", "split"]]
    frame = frame.merge(split_map, on="event_id", validate="many_to_one")
    # Keep held-out test events physically out of the tuning dataframe.
    frame = frame[frame["split"].isin(("train", "validation"))].copy()
    frame = frame.rename(columns={c: f"{c}_mean" for c in CHANNELS})
    return frame.sort_values(["event_id", "node_id", "window_index"], kind="stable").reset_index(drop=True)


def evaluate(frame: pd.DataFrame, train_events: set[str], val_events: set[str], params: dict, seed: int) -> dict:
    mask_train = frame.event_id.isin(train_events).to_numpy()
    mask_val = frame.event_id.isin(val_events).to_numpy()
    fold_frame = frame[mask_train | mask_val].reset_index(drop=True)
    split = pd.Series(np.where(fold_frame.event_id.isin(train_events), "train", "validation"))
    model = train_temporal_forecaster(
        fold_frame,
        split=split,
        channels=CHANNELS,
        horizons=HORIZONS,
        architecture=params["architecture"],
        history_steps=params["history_steps"],
        width=params["width"],
        depth=params["depth"],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=params["learning_rate"],
        seed=seed,
    )
    x_val, y_val, _ = _build_multi_horizon(
        fold_frame[split.to_numpy() == "validation"], CHANNELS,
        history_steps=params["history_steps"], horizons=HORIZONS,
    )
    x_norm = (x_val - model.mu) / model.sd
    y_norm = (y_val - model.mu) / model.sd
    import torch
    model.torch_module.eval()
    with torch.no_grad():
        pred_norm = model.torch_module(torch.tensor(x_norm, dtype=torch.float32)).numpy()
    pred = pred_norm * model.sd[None, None, :] + model.mu[None, None, :]
    truth = y_val
    persistence = x_val[:, -1:, :]
    err = pred - truth
    p_err = persistence - truth
    return {
        "seed": seed,
        "n_train_rows": int(mask_train.sum()),
        "n_validation_sequences": int(len(x_val)),
        "normalized_mse": float(np.mean(((pred - truth) / model.sd[None, None, :]) ** 2)),
        "persistence_normalized_mse": float(np.mean(((persistence - truth) / model.sd[None, None, :]) ** 2)),
        "mae_by_channel": dict(zip(CHANNELS, np.mean(np.abs(err), axis=(0, 1)).astype(float), strict=True)),
        "rmse_by_channel": dict(zip(CHANNELS, np.sqrt(np.mean(err**2, axis=(0, 1))).astype(float), strict=True)),
        "persistence_mae_by_channel": dict(zip(CHANNELS, np.mean(np.abs(p_err), axis=(0, 1)).astype(float), strict=True)),
        "model": model,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = prepared_data()
    events = np.array(sorted(frame.event_id.unique()))
    groups = frame.groupby("event_id").size()
    if groups.nunique() != 1:
        raise RuntimeError("inconsistent event sequence lengths; grouped CV is unsafe")
    print(f"Development events={len(events)}; held-out test events excluded; windows/event={int(groups.iloc[0])}")

    space = list(itertools.product(("tcn", "gru", "lstm"), (8, 16), (3, 5), (0.001, 0.003)))
    rng = random.Random(20260928)
    chosen = sorted(rng.sample(space, 8))
    candidates = [dict(architecture=a, width=w, history_steps=h, depth=1, learning_rate=lr)
                  for a, w, h, lr in chosen]
    splitter = GroupKFold(n_splits=FOLDS)
    results = []
    for ci, params in enumerate(candidates):
        fold_records = []
        for fi, (tr_idx, va_idx) in enumerate(splitter.split(events, groups=events)):
            tr_events, va_events = set(events[tr_idx]), set(events[va_idx])
            record = evaluate(frame, tr_events, va_events, params, seed=42 + fi)
            fold_records.append({k: v for k, v in record.items() if k != "model"})
        result = {"params": params, "folds": fold_records,
                  "mean_normalized_mse": float(np.mean([r["normalized_mse"] for r in fold_records])),
                  "mean_persistence_normalized_mse": float(np.mean([r["persistence_normalized_mse"] for r in fold_records]))}
        results.append(result)
        print(f"candidate {ci+1}/{len(candidates)} {params} CV={result['mean_normalized_mse']:.5f} persistence={result['mean_persistence_normalized_mse']:.5f}", flush=True)

    results.sort(key=lambda x: x["mean_normalized_mse"])
    top = results[:3]
    repeat_scores = []
    for item in top:
        for seed in SEEDS:
            per_fold = []
            for tr_idx, va_idx in splitter.split(events, groups=events):
                rec = evaluate(frame, set(events[tr_idx]), set(events[va_idx]), item["params"], seed=seed)
                per_fold.append({k: v for k, v in rec.items() if k != "model"})
            repeat_scores.append({"params": item["params"], "seed": seed,
                                  "mean_normalized_mse": float(np.mean([r["normalized_mse"] for r in per_fold])),
                                  "mean_persistence_normalized_mse": float(np.mean([r["persistence_normalized_mse"] for r in per_fold])),
                                  "folds": per_fold})

    best = min(top, key=lambda x: x["mean_normalized_mse"])
    beats = best["mean_normalized_mse"] < best["mean_persistence_normalized_mse"]
    payload = {
        "study": {"seed": 20260928, "sampler": "seeded random sample without replacement",
                  "search_space": {"architecture": ["tcn", "gru", "lstm"], "width": [8, 16],
                                   "history_steps": [3, 5], "depth": [1],
                                   "learning_rate": [0.001, 0.003]},
                  "candidates": len(candidates), "grouping": "event_id", "folds": FOLDS,
                  "repeat_seeds": list(SEEDS), "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                  "horizons": list(HORIZONS), "test_touched": False},
        "candidate_results": results,
        "top3_repeated_cv": repeat_scores,
        "best": best,
        "beats_persistence": bool(beats),
    }
    (OUT / "forecaster_study.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    (OUT / "forecaster_trials.csv").write_text(pd.DataFrame([
        {**r["params"], "mean_normalized_mse": r["mean_normalized_mse"],
         "mean_persistence_normalized_mse": r["mean_persistence_normalized_mse"]} for r in results
    ]).to_csv(index=False))

    config = {
        "forecaster": {
            "status": "tuned_on_development_groups" if beats else "search_complete_did_not_beat_persistence",
            **best["params"], "channels": list(CHANNELS), "horizons": list(HORIZONS),
            "seed": 42, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "cv_mean_normalized_mse": best["mean_normalized_mse"],
            "cv_mean_persistence_normalized_mse": best["mean_persistence_normalized_mse"],
            "search_artifact": "reports/tuning/forecaster_study.json",
        },
        "risk_classifier": {"status": "blocked_pending_target_leakage_fix"},
        "isolation_forest": {"status": "blocked_pending_target_leakage_fix"},
    }
    PARAMS_PATH.write_text(yaml.safe_dump(config, sort_keys=False))
    # Reload via the public config path and use the frozen values for a
    # development-only final fit. The test regime is still excluded.
    final = model_params_config()["forecaster"]
    if beats:
        fitted = train_temporal_forecaster(
            frame,
            channels=tuple(final["channels"]),
            horizons=tuple(final["horizons"]),
            architecture=final["architecture"],
            history_steps=int(final["history_steps"]),
            width=int(final["width"]),
            depth=int(final["depth"]),
            epochs=int(final["epochs"]),
            batch_size=int(final["batch_size"]),
            learning_rate=float(final["learning_rate"]),
            seed=int(final["seed"]),
        )
        manifest = json.loads((ROOT / "data/synthetic/dataset_manifest.json").read_text())
        import torch
        artifact = ROOT / "models/temporal_model/forecaster_tuned.pt"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "metadata": {
                "model_name": "SubSense temporal forecaster",
                "model_version": "1.0.0-tuned-dev-cv",
                "feature_schema_version": manifest["feature_schema_version"],
                "training_dataset_version": manifest["dataset_version"],
                "training_groups": int(frame.event_id.nunique()),
                "held_out_test_used": False,
            },
            "params": final,
            "channels": fitted.channels,
            "horizons": fitted.horizons,
            "mu": fitted.mu,
            "sd": fitted.sd,
            "state_dict": fitted.torch_module.state_dict(),
        }, artifact)
        payload["artifact"] = str(artifact.relative_to(ROOT))
        (OUT / "forecaster_study.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"Best CV MSE={best['mean_normalized_mse']:.5f}; persistence={best['mean_persistence_normalized_mse']:.5f}; beats={beats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
