#!/usr/bin/env python3
""" — generate synthetic_nodes.csv (+ dataset_manifest.json).

Usage:
 python scripts/generate_synthetic_nodes.py [--out data/synthetic]
 [--sequences-per-scenario 44] [--seed 42] [--nodes-limit N]

 scale: default run = 16 scenarios × 44 sequences/scenario × 400 nodes
= 281,600 sequences ≥ the 10,000 required. The manifest records both the
generated and planned sequence counts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.simulator.dataset_builder import DEFAULT_SEQUENCES_PER_SCENARIO, build_dataset
from src.simulator.scenarios import Scenario


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--sequences-per-scenario", type=int, default=DEFAULT_SEQUENCES_PER_SCENARIO)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-version", default="v0.1.0")
    parser.add_argument("--nodes-limit", type=int, default=None, help="smoke-test only")
    parser.add_argument("--scenarios", nargs="+", choices=[s.value for s in Scenario], default=None)
    args = parser.parse_args()

    result = build_dataset(
        output_dir=REPO_ROOT / args.out,
        sequences_per_scenario=args.sequences_per_scenario,
        seed=args.seed,
        dataset_version=args.dataset_version,
        nodes_limit=args.nodes_limit,
        scenario_names=tuple(args.scenarios) if args.scenarios else None,
    )
    print(f"rows    : {result['rows']:,}")
    print(f"events  : {result['events']:,}")
    print(f"nodes   : {result['nodes_csv']}")
    print(f"events  : {result['events_csv']}")
    print(f"manifest: {result['manifest']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
