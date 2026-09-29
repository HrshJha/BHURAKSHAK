# Tabletop records and generated stand-in

This directory contains a **seeded generated stand-in**, not measurements from a completed physical-rig campaign. The physical tabletop campaign has not been run. Do not cite these files as field observations, instrument calibration data, or independent validation.

## Files

| File | Contents | Use and limits |
|---|---|---|
| `trial_metadata.csv` | Generated trial identifiers, configured conditions and target displacement | Describes simulator inputs; targets are not independent measured ground truth |
| `processed_windowed_dataset.csv` | Generated sensor features and reference displacement by window | Useful for pipeline/interface exercises; windows derive from shared generated trials and are not independent samples |
| `raw_sensor_log.csv` | Generated time-stamped sensor observations | Can be regenerated from the command below |

The generator is `scripts/generate_tabletop_dataset.py`. Its default deterministic seed is 42. Run from the repository root:

```bash
.venv/bin/python scripts/generate_tabletop_dataset.py data/recorded/tabletop
```

This command rewrites the generated stand-in files in this directory. Preserve any separately collected physical measurements elsewhere with their acquisition protocol, calibration records, units, timestamps, and provenance. They must be audited and versioned before use in model training or evaluation.

## Relationship to v3

The current [`generalization_v3` report](../../../reports/generalization_v3/final_report.md) uses a separate, physically coupled synthetic generator and its own frozen grouped splits and one-use independent synthetic test. These tabletop stand-in CSVs are not the v3 independent test and do not support its reported metrics. The v3 severity boundaries are experimental prototype labels, not safety limits. Neither this data nor v3 constitutes mine validation.
