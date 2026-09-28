#!/usr/bin/env python3
"""Finalize the best development-only forecaster from its completed Optuna study."""
from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tune_forecaster import CHANNELS, EPOCHS, HORIZONS, prepared_data
from src.forecasting.temporal_model import train_temporal_forecaster


def main() -> int:
    study_path = ROOT / "reports/tuning/forecaster_study.json"
    study = json.loads(study_path.read_text(encoding="utf-8"))
    best = study["best"]
    params = dict(best["params"])
    beats = float(best["mean_normalized_mse"]) < float(best["mean_persistence_normalized_mse"])
    cfg_path = ROOT / "configs/model_params.yaml"
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    config["forecaster"] = {
        "status": "tuned_on_development_groups" if beats else "search_complete_did_not_beat_persistence",
        **params, "channels": list(CHANNELS), "horizons": list(HORIZONS),
        "seed": 42, "epochs": EPOCHS,
        "cv_mean_normalized_mse": best["mean_normalized_mse"],
        "cv_mean_persistence_normalized_mse": best["mean_persistence_normalized_mse"],
        "search_artifact": "reports/tuning/forecaster_study.json",
        "requested_trials": study["study"]["requested_trials"],
        "actual_trials": study["study"]["actual_trials"],
    }
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    if beats:
        frame = prepared_data()
        model = train_temporal_forecaster(
            frame, channels=CHANNELS, horizons=HORIZONS,
            architecture=params["architecture"], history_steps=int(params["history_steps"]),
            width=int(params["width"]), depth=int(params["depth"]), epochs=EPOCHS,
            batch_size=int(params["batch_size"]), learning_rate=float(params["learning_rate"]),
            dropout=float(params["dropout"]), weight_decay=float(params["weight_decay"]), seed=42,
        )
        manifest = json.loads((ROOT / "data/synthetic/dataset_manifest.json").read_text(encoding="utf-8"))
        artifact = ROOT / "models/temporal_model/forecaster_tuned.pt"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"metadata": {"model_name": "SubSense temporal forecaster", "model_version": "v2.0.0-tuned-dev",
                                 "feature_schema_version": manifest["feature_schema_version"],
                                 "training_dataset_version": manifest["dataset_version"],
                                 "training_groups": int(frame.event_id.nunique()), "held_out_test_used": False},
                    "params": params, "channels": model.channels, "horizons": model.horizons,
                    "mu": model.mu, "sd": model.sd, "state_dict": model.torch_module.state_dict()}, artifact)
        artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
        artifact.with_suffix(artifact.suffix + ".sha256").write_text(artifact_hash + "\n", encoding="utf-8")
        provenance_hash = hashlib.sha256((ROOT / "configs/feature_provenance.yaml").read_bytes()).hexdigest()
        from src.risk.model_registry import ModelRegistry
        ModelRegistry().register_model(
            model_name="subsense_temporal_forecaster", model_version="v2.0.0-tuned-dev",
            feature_version=str(manifest["feature_schema_version"]),
            training_dataset_version=str(manifest["dataset_version"]), artifact_path=str(artifact),
            split_name="train+validation development groups", seed=42, provenance_hash=provenance_hash,
        )
        study["artifact"] = str(artifact.relative_to(ROOT))
        study["post_study_final_fit"] = {"training_rows": int(len(frame)), "test_touched": False, "seed": 42}
        study_path.write_text(json.dumps(study, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"beats_persistence": beats, "best_mse": best["mean_normalized_mse"],
                      "persistence_mse": best["mean_persistence_normalized_mse"],
                      "config": str(cfg_path.relative_to(ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
