#!/usr/bin/env python3
""" — domain-gap Stage 1 (synthetic → synthetic) → experiments/domain_gap_stage1.json.

 Stage 1 establishes INTERNAL VALIDITY: can the model handle a
simulator parameter regime it was never trained on? Two partition axes are
reported:

1. **literal synthetic_split** — hold out the TOP ``test_upper_fraction``
 of the corpus-wide ``max_deformation`` range. On THIS corpus the axis is
 degenerate: ~75% of events (stable/vibration/fault families) carry
 ≈0 mm deformation, so the held-out band contains essentially all
 deforming events and the regime-split model trains on no subsidence at
 all. The failure is REPORTED, not hidden — it is a real property of the
 generator's parameter space (and an input to the report).

2. **family-conditional regime split** — the axis 's example
 ("train σ=5–20, test σ=22–30") actually intends: within each scenario
 family, hold out the top 20% of deformation (next 20% = validation,
 zero-variance families stay in train). This answers the stage-1
 question: does a model trained on small/medium subsidence grade large
 subsidence of the SAME families it has seen?

All metrics come from src/evaluation/metrics.py. Deterministic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score as sk_f1

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import validation_config  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    brier_score,
    classification_metrics,
    expected_calibration_error,
    false_alarms_per_day,
    lead_time_stats,
)
from src.evaluation.splits import assert_no_leakage, synthetic_split  # noqa: E402
from src.risk.xgboost_model import train_risk_model  # noqa: E402

OUT_JSON = REPO_ROOT / "experiments" / "domain_gap_stage1.json"
STORE = REPO_ROOT / "data" / "features" / "features_v2.parquet"
EVENTS_META = REPO_ROOT / "data" / "synthetic" / "synthetic_events.csv"
STRIDE_HOURS = 0.6
EVENT_DAYS = 1.0
ALERT_THRESHOLD = 0.5


def family_conditional_split(events: pd.DataFrame) -> pd.Series:
    """Within each event family, hold out the top deformation band.

 Returns a Series indexed by event ``id`` with train/validation/test.
 Test = top ``test_upper_fraction`` of ``max_deformation`` within the
 family; validation = the next band of the same width; train = the rest.
 Families with zero deformation variance cannot yield a regime band and
 stay entirely in train (recorded in the payload metadata).
 """
    test_frac = float(validation_config()["synthetic_split"]["test_upper_fraction"])
    labels = {}
    zero_variance_families: list[str] = []
    for family, g in events.groupby("family", sort=False):
        vals = g["max_deformation"].to_numpy(dtype=float)
        if float(vals.max()) <= 0.0:
            zero_variance_families.append(str(family))
            for eid in g["id"]:
                labels[eid] = "train"
            continue
        q_test = float(np.quantile(vals, 1.0 - test_frac))
        q_val = float(np.quantile(vals, 1.0 - 2.0 * test_frac))
        for eid, v in zip(g["id"], vals, strict=True):
            if v > q_test and q_test > 0.0:
                labels[eid] = "test"
            elif v > q_val and q_val > 0.0:
                labels[eid] = "validation"
            else:
                labels[eid] = "train"
    return pd.Series(labels), zero_variance_families


def score_band(model, classes: list[str], band: pd.DataFrame) -> dict:
    """ metrics on one parameter-regime band."""
    proba = model.predict_proba(band)
    pred_labels = np.asarray(classes)[proba.argmax(axis=1)]
    i_crit = classes.index("CRITICAL")

    y_bin = (band["anomaly_label"] > 0).astype(int).to_numpy()
    pred_bin = (pred_labels != "NORMAL").astype(int)
    if y_bin.sum() == 0:
        # honest: AP is UNDEFINED on a band with no positives — recorded as None
        det = classification_metrics(y_bin, pred_bin)
        det["pr_auc"] = None
        det["pr_auc_note"] = "undefined: band carries no anomalous windows"
    else:
        det = classification_metrics(y_bin, pred_bin, y_score=1.0 - proba[:, classes.index("NORMAL")])
    det["macro_f1_risk3"] = float(
        sk_f1(band["risk_label"], pred_labels, average="macro",
              labels=sorted(set(band["risk_label"]) | set(pred_labels)))
    )
    det["critical_prevalence"] = float((band["risk_label"] == "CRITICAL").mean())

    p_crit = proba[:, i_crit]
    two_col = np.stack([p_crit, 1.0 - p_crit], axis=1)
    y_idx = (band["risk_label"] != "CRITICAL").astype(int).to_numpy()
    calib = {"brier": brier_score(two_col, y_idx), "ece": expected_calibration_error(two_col, y_idx)}

    af = band[["event_id", "window_index", "anomaly_label"]].copy()
    af["p_critical"] = p_crit
    alerts, onsets = {}, {}
    for ev, g in af.groupby("event_id", sort=False):
        g = g.sort_values("window_index")
        fired = g.loc[g["p_critical"] >= ALERT_THRESHOLD, "window_index"]
        if len(fired):
            alerts[str(ev)] = float(fired.iloc[0])
        onset = g.loc[g["anomaly_label"] > 0, "window_index"]
        if len(onset):
            onsets[str(ev)] = float(onset.iloc[0])
    if onsets:
        temporal = lead_time_stats(alerts, onsets, stride_hours=STRIDE_HOURS)
    else:
        temporal = {"median_lead_time_hours": None, "p10_lead_time_hours": None,
                    "n_events_with_lead": 0, "n_missed_events": 0, "missed_event_rate": None,
                    "note": "band carries no anomalous events (deformation below the  threshold)"}
    temporal["false_alarms_per_day"] = false_alarms_per_day(
        y_bin, pred_bin, n_days=float(band["event_id"].nunique()) * EVENT_DAYS
    )
    return {
        "n_windows": int(len(band)),
        "n_events": int(band["event_id"].nunique()),
        "detection": det,
        "calibration": calib,
        "temporal": temporal,
    }


def band_ranges(store: pd.DataFrame, defa: pd.Series) -> dict:
    out = {}
    for band in ("train", "validation", "test"):
        evs = store.loc[store["_split"] == band, "event_id"].unique()
        vals = defa.reindex(evs)
        out[band] = {"min": float(vals.min()), "max": float(vals.max()),
                     "median": float(vals.median()), "n_events": int(len(evs))}
    return out


def run_axis(name: str, store: pd.DataFrame, defa: pd.Series) -> dict:
    print(f"\n=== axis: {name} ===", flush=True)
    ranges = band_ranges(store, defa)
    for band, r in ranges.items():
        print(f"  {band:10s} [{r['min']:8.2f}, {r['max']:8.2f}]  median {r['median']:7.2f}  "
              f"{r['n_events']} events", flush=True)

    model = train_risk_model(store.rename(columns={"_split": "split"}))
    classes = list(model.classes)

    results = {}
    for band in ("train", "validation", "test"):
        part = store[store["_split"] == band].reset_index(drop=True)
        results[band] = score_band(model, classes, part)
        d = results[band]["detection"]
        ap = f"{d['pr_auc']:.3f}" if d["pr_auc"] is not None else "n/a"
        print(f"  {band:10s} P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f} AP={ap} "
              f"macroF1={d['macro_f1_risk3']:.3f} brier={results[band]['calibration']['brier']:.4f}", flush=True)
    return {"band_ranges": ranges, "results": results}


def main() -> int:
    raise SystemExit("Legacy domain-gap evaluation reads the burned test split; rerun only through the Phase 9 workflow.")
    store = pd.read_parquet(STORE)
    events = pd.read_csv(EVENTS_META)
    if not events["id"].is_unique:
        raise SystemExit("events metadata carries duplicate ids")
    events = events.assign(family=events["id"].str.rsplit("_", n=2).str[0])
    defa = events.set_index("id")["max_deformation"]

    literal = synthetic_split(store, events)
    if literal.isna().any():
        missing = sorted(store.loc[literal.isna(), "event_id"].unique())[:5]
        raise SystemExit(f"synthetic_split left events unassigned: {missing}")
    assert_no_leakage(store, literal, "event_id")
    literal_store = store.assign(_split=literal)
    literal_res = run_axis("literal  synthetic_split (corpus-wide deformation quantile)",
                           literal_store, defa)

    # axis 2: family-conditional regime split
    fam_split, zero_var_fams = family_conditional_split(events)
    fam_series = store["event_id"].map(fam_split)
    if fam_series.isna().any():
        missing = sorted(store.loc[fam_series.isna(), "event_id"].unique())[:5]
        raise SystemExit(f"family-conditional split left events unassigned: {missing}")
    assert_no_leakage(store, fam_series, "event_id")
    fam_store = store.assign(_split=fam_series)
    fam_res = run_axis("family-conditional regime split (top 20% of deformation per family)",
                       fam_store, defa)

    payload = {
        "metadata": {
            "task": " ( domain-gap Stage 1: synthetic → synthetic)",
            "store": str(STORE.relative_to(REPO_ROOT)),
            "alert_threshold_p_critical": ALERT_THRESHOLD,
            "axes": {
                "literal_synthetic_split": {
                    "rule": "corpus-wide: top test_upper_fraction of event max_deformation held out",
                    "config": validation_config()["synthetic_split"],
                    "degeneracy_note": (
                        "~75% of events carry ≈0 mm deformation (stable/vibration/fault "
                        "families), so the held-out top band contains essentially all "
                        "deforming events; the literal-axis model trains on no subsidence "
                        "and cannot grade it. Reported as a real property of the generator, "
                        "not hidden."
                    ),
                },
                "family_conditional": {
                    "rule": ("within each scenario family: top 20% of max_deformation → test, "
                             "next 20% → validation, rest → train; zero-variance families stay "
                             "in train (no regime to hold out)"),
                    "zero_variance_families": zero_var_fams,
                },
            },
            "notes": [
                "Leakage assert on both axes: no event_id spans two regime bands.",
                "Test-band prevalence shifts with the axis — prevalence reported alongside "
                "PR-AUC so numbers cannot be read in isolation.",
            ],
        },
        "literal_synthetic_split": literal_res,
        "family_conditional": fam_res,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote {OUT_JSON.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
