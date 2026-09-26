"""Verify the Data+ML repository tree matches PRD §33 (Data+ML subset).

Acceptance check for T-001. Exits 0 when every required path exists and every
out-of-scope path is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# PRD §33 paths that belong to the Data + ML workstream.
REQUIRED_DIRS = [
    "data/raw/synthetic",
    "data/raw/sentinel1",
    "data/raw/dgps",
    "data/raw/sensors",
    "data/processed/synthetic",
    "data/processed/insar",
    "data/processed/dgps",
    "data/processed/sensors",
    "data/synthetic",
    "data/features",
    "data/labels",
    "notebooks",
    "src/simulator",
    "src/preprocessing",
    "src/features",
    "src/anomaly",
    "src/risk",
    "src/forecasting",
    "src/physics",
    "src/geospatial",
    "src/evaluation",
    "models/isolation_forest",
    "models/xgboost",
    "models/temporal_model",
    "tests",
    "configs",
    "docs",
    "scripts",
    "experiments",
    "reports",
]

REQUIRED_PACKAGE_DIRS = [
    "src",
    "src/simulator",
    "src/preprocessing",
    "src/features",
    "src/anomaly",
    "src/risk",
    "src/forecasting",
    "src/physics",
    "src/geospatial",
    "src/evaluation",
]

# PRD §33 paths that belong to other workstreams. Deliberately NOT created.
FORBIDDEN_PATHS = [
    "src/api",
    "dashboard",
    "deployment/raspberry_pi",
    "docker-compose.yml",
]


def main() -> int:
    failures: list[str] = []

    for rel in REQUIRED_DIRS:
        path = REPO_ROOT / rel
        if not path.is_dir():
            failures.append(f"MISSING required directory: {rel}")

    for rel in REQUIRED_PACKAGE_DIRS:
        path = REPO_ROOT / rel / "__init__.py"
        if not path.is_file():
            failures.append(f"MISSING package marker: {rel}/__init__.py")

    for rel in FORBIDDEN_PATHS:
        path = REPO_ROOT / rel
        if path.exists():
            failures.append(
                f"OUT-OF-SCOPE path present (must not exist in this workstream): {rel}"
            )

    if failures:
        print("FAIL - repository tree does not match PRD §33 Data+ML subset:")
        for line in failures:
            print(f"  - {line}")
        return 1

    print(
        f"PASS - {len(REQUIRED_DIRS)} required directories, "
        f"{len(REQUIRED_PACKAGE_DIRS)} package markers present; "
        f"{len(FORBIDDEN_PATHS)} out-of-scope paths absent."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
