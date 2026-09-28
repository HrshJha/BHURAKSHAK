#!/usr/bin/env python3
"""T-081 — run the full §7 chain end-to-end on held-out (test-split) data.

Acceptance (TASKS.md T-081): `python scripts/run_pipeline.py` runs the full
chain (validation → features → Isolation Forest → spatial fusion → physics
check → XGBoost → alert engine → explainability) on held-out data. Every
stage is the module that owns it (provenance recorded in the result); the
TEST split is scored exactly once, after all fitting is done.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import load_synthetic_corpus, run_pipeline  # noqa: E402

OUT_JSON = REPO_ROOT / "experiments" / "pipeline_run.json"


def main() -> int:
    raw, coords = load_synthetic_corpus()
    print(f"corpus: {len(raw):,} raw rows, {raw.event_id.nunique():,} events, "
          f"{raw.node_id.nunique()} nodes", flush=True)

    result = run_pipeline(raw, coords)

    test = result.scored_test
    summary = {
        "validation": result.validation,
        "n_windows": int(result.feature_report.n_windows),
        "n_features": len(result.feature_report.features),
        "if_features": len(result.if_model.features),
        "xgb_features": len(result.risk_model.features),
        "test_windows_scored": int(len(test)),
        "predicted_level_counts": test["predicted_level"].value_counts().to_dict(),
        "alert_level_counts": test["alert_level"].value_counts().to_dict(),
        "explanations_emitted": len(result.explanations),
        "provenance": result.provenance,
    }

    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str))
    print("\n=== held-out (test) summary ===")
    for k, v in summary.items():
        if k != "provenance":
            print(f"{k}: {v}")
    print(f"\nprovenance per stage: {json.dumps(result.provenance, indent=2)}")
    print(f"summary → {OUT_JSON.relative_to(REPO_ROOT)}")

    # the §35 acceptance map is maintained by hand in
    # reports/acceptance_criteria.md — print the pointer, don't duplicate it
    print("\n§35 acceptance map → reports/acceptance_criteria.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
