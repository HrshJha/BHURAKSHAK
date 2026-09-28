#!/usr/bin/env python3
"""Select a light logistic-regression C on the fixed development event folds."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tune_models import CLASSES, SEED, _fold_frames, _metrics, development_frame, grouped_folds


def main() -> int:
    frame, features = development_frame()
    candidates: dict[str, list[dict]] = {str(c): [] for c in (0.1, 1.0, 10.0)}
    for fold, (train_ids, val_ids) in enumerate(grouped_folds(frame), 1):
        train, val = _fold_frames(frame, train_ids, val_ids)
        y_train = train.risk_label.astype(str).to_numpy()
        y_val = val.risk_label.astype(str).to_numpy()
        scaler = StandardScaler().fit(train[features].to_numpy(dtype=float))
        x_train = scaler.transform(train[features].to_numpy(dtype=float))
        x_val = scaler.transform(val[features].to_numpy(dtype=float))
        for c in (0.1, 1.0, 10.0):
            model = LogisticRegression(C=c, max_iter=2000, class_weight="balanced", random_state=SEED)
            model.fit(x_train, y_train)
            raw = model.predict_proba(x_val)
            prob = np.zeros((len(val), len(CLASSES)))
            for j, label in enumerate(model.classes_):
                prob[:, CLASSES.index(label)] = raw[:, j]
            pred = np.asarray(CLASSES)[prob.argmax(axis=1)]
            candidates[str(c)].append({"fold": fold, **_metrics(y_val, pred, prob, list(CLASSES))})

    result: dict[str, dict] = {}
    for c, records in candidates.items():
        keys = records[0].keys() - {"fold"}
        mean = {key: float(np.mean([r[key] for r in records])) for key in keys}
        std = {key: float(np.std([r[key] for r in records], ddof=1)) for key in keys}
        score = 0.5 * mean["pr_auc_macro"] + 0.3 * mean["recall_critical"] + 0.2 * mean["f1_macro"]
        result[c] = {"params": {"C": float(c)}, "folds": records, "mean": mean, "std": std,
                     "objective": score}
    selected = max(result, key=lambda c: result[c]["objective"])

    report_path = ROOT / "reports/tuning/baselines.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    chosen = result[selected]
    report["logistic_c_search"] = {
        "candidates": result,
        "selected_C": float(selected),
        "selection_objective": "0.5*macro_PR_AUC + 0.3*recall_CRITICAL + 0.2*macro_F1",
        "protocol": "three fixed StratifiedGroupKFold folds grouped by event_id; scaler fit on each training fold",
    }
    report["models"]["logistic"] = {
        "folds": chosen["folds"],
        "mean": {key: {"mean": chosen["mean"][key], "std": chosen["std"][key]} for key in chosen["mean"]},
        "params": chosen["params"],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    config_path = ROOT / "configs/risk_model.yaml"
    lines = config_path.read_text(encoding="utf-8").splitlines()
    in_logistic = False
    found_c = False
    for i, line in enumerate(lines):
        if line == "  logistic:":
            in_logistic = True
            continue
        if in_logistic and line.startswith("  ") and not line.startswith("    "):
            in_logistic = False
        if in_logistic and line.startswith("    C:"):
            lines[i] = f"    C: {float(selected):g}"
            found_c = True
            break
    if not found_c:
        raise RuntimeError("could not find baselines.logistic.C in configs/risk_model.yaml")
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"selected_C": float(selected), "objective": chosen["objective"],
                      "mean": chosen["mean"], "std": chosen["std"], "development_only": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
