"""Check required project paths and confirm internal planning files are absent."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED_PATHS = (
    "README.md",
    "configs",
    "data/synthetic/dataset_manifest.json",
    "data/synthetic/synthetic_events.csv",
    "data/synthetic/synthetic_nodes.csv",
    "experiments",
    "models/registry.json",
    "notebooks",
    "reports/final_eval.md",
    "reports/test_lock.json",
    "scripts/run_pipeline.py",
    "src/pipeline.py",
    "tests",
)
ARCHIVED_PATHS = (
    "prd.md",
    "TASKS.md",
    "PROGRESS_LOG.md",
    "reports/superseded_leaky",
)


def main() -> int:
    failures = [f"missing: {path}" for path in REQUIRED_PATHS if not (ROOT / path).exists()]
    failures.extend(f"still present: {path}" for path in ARCHIVED_PATHS if (ROOT / path).exists())
    if failures:
        print("FAIL - repository tree check")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"PASS - {len(REQUIRED_PATHS)} required paths present; {len(ARCHIVED_PATHS)} archived paths absent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
