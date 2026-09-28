#!/usr/bin/env python3
"""Create the fresh, ignored final-evaluation corpus and its one-use lock."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from src.simulator.dataset_builder import build_dataset


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs/heldout.yaml").read_text(encoding="utf-8"))
    out = ROOT.joinpath("data", "heldout_" + "locked")
    lock_path = ROOT / "reports" / "test_lock.json"
    if lock_path.exists():
        raise SystemExit("test lock already exists; refusing to replace a locked test corpus")
    result = build_dataset(
        output_dir=out,
        sequences_per_scenario=int(cfg["sequences_per_scenario"]),
        seed=int(cfg["seed"]),
        dataset_version=str(cfg["dataset_version"]),
        scenario_names=tuple(cfg["scenarios"]),
    )
    hashes = result["artifact_sha256"]
    combined = hashlib.sha256(
        "\n".join(f"{name}:{hashes[name]}" for name in sorted(hashes)).encode("utf-8")
    ).hexdigest()
    lock = {
        "sha256": combined,
        "files": hashes,
        "seed": int(cfg["seed"]),
        "dataset_version": str(cfg["dataset_version"]),
        "feature_schema_version": "v2",
        "scenario_types": list(cfg["scenarios"]),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evals_run": 0,
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"created locked synthetic corpus: {result['events']:,} events, {result['rows']:,} rows")
    print(f"lock: {lock_path.relative_to(ROOT)} sha256={combined}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
