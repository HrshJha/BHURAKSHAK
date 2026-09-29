#!/usr/bin/env python3
"""Profile frozen development models on one window on the current workstation."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
import time
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tune_models import development_frame
from scripts.run_robustness import _latency_profile
from src.risk.artifacts import load_model_artifact


def main() -> int:
    out_path = ROOT / "reports/tuning/robustness.json"
    result = json.loads(out_path.read_text(encoding="utf-8"))
    frame, _ = development_frame()
    config = yaml.safe_load((ROOT / "configs/model_params.yaml").read_text(encoding="utf-8"))
    risk = load_model_artifact(ROOT / "models/tuned/risk_xgboost_tuned.joblib", expected_schema_version="v2")
    probe_frame = frame[list(risk["features"])].iloc[[0]]
    probe = risk["transform"](probe_frame).to_numpy(dtype=float)
    profiles = {"risk_xgboost_tuned": _latency_profile(lambda: risk["model"].predict_proba(probe))}
    forest = load_model_artifact(ROOT / "models/tuned/iforest_tuned.joblib", expected_schema_version="v2")
    forest_probe = forest["transform"](frame).iloc[[0]].to_numpy(dtype=float)
    profiles["isolation_forest_tuned"] = _latency_profile(lambda: forest["model"].score_samples(forest_probe))

    forecast_path = ROOT / "models/temporal_model/forecaster_tuned.pt"
    if forecast_path.exists():
        import torch
        from torch import nn
        from src.forecasting.temporal_model import _build_torch
        frozen = torch.load(forecast_path, map_location="cpu", weights_only=False)
        params = frozen["params"]
        channels = tuple(frozen["channels"])
        horizons = tuple(frozen["horizons"])
        width, depth, architecture = int(params["width"]), int(params["depth"]), str(params["architecture"])
        class FrozenForecaster(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = _build_torch(architecture, len(channels), width, depth)
                self.head = nn.Linear(width, len(horizons) * len(channels))
                self.dropout = nn.Dropout(float(params.get("dropout", 0.0)))
                if architecture == "tcn": self.tcn_proj = nn.Linear(width, width)
            def forward(self, value):
                if architecture == "tcn": last = self.backbone(value.transpose(1, 2))[:, :, -1]
                else:
                    seq, _ = self.backbone(value); last = seq[:, -1, :]
                return self.head(self.dropout(last)).reshape(-1, len(horizons), len(channels))
        model = FrozenForecaster(); model.load_state_dict(frozen["state_dict"]); model.eval()
        features = pd.read_parquet(ROOT / "data/features/features_v2.parquet")
        event = features.event_id.iloc[0]
        series = features[features.event_id == event].sort_values("window_index")
        values = series[list(channels)].tail(int(params["history_steps"])).to_numpy(dtype=np.float32)
        x = torch.tensor(((values - frozen["mu"]) / frozen["sd"])[None, ...], dtype=torch.float32)
        with torch.no_grad(): profiles["forecaster_tuned"] = _latency_profile(lambda: model(x))
    else:
        profiles["forecaster_tuned"] = {"status": "no_frozen_forecaster_artifact"}
    result["workstation_edge_profile"] = {
        "platform": sys.platform, "models": profiles,
        "note": "workstation CPU/process RSS; sampled RSS is not device peak; no numeric edge budget is defined in  ",
    }
    out_path.write_text(json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps(result["workstation_edge_profile"], indent=2))
    return 0

if __name__ == "__main__": raise SystemExit(main())
