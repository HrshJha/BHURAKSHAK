#!/usr/bin/env python3
"""Regenerate the 4-node tabletop trapdoor dataset with self-consistent ground truth.

Outputs (overwritten in place):
 trial_metadata.csv 26 trials
 raw_sensor_log.csv one row per node per 100 ms sample (10 Hz)
 processed_windowed_dataset.csv one row per (trial, node, 2 s window, 1 s step)

Fixes vs the previous version (each verified by the built-in audit):
1. knob_turns are MONOTONIC (a knob cannot un-turn) and reach
 max_displacement_mm_target / 1.25 exactly (M8 pitch, plan ).
2. known_displacement_mm is pure mechanism ground truth — no sensor noise —
 so class 0 is exactly 0 mm and the 0/15/35 mm boundaries are exact.
3. both_together = both trapdoors share one schedule; both_sequential = TD2
 starts late (every trial keeps a clean stable baseline before any motion).
4. Per-trapdoor semantics: metadata target and each node's ground truth are
 the PER-TRAPDOOR descent (N2 <- TD1, N3 <- TD2), never the sum.
5. Vibration has real burst dynamics: acceleration bursts at every stepped
 onset plus random micro-collapses, amplitude growing with displacement —
 vibration_rms / accel_peak now vary and carry signal.
6. Reference nodes N1/N4: known_displacement exactly 0, relative_* exactly 0
 (their role is the baseline, plan ).
7. tof_delta is an independent crack-opening channel (0.4x coupling + own
 noise), no longer a noiseless copy of displacement_mm.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/recorded/tabletop")
SEED = 42
DT_MS = 100          # raw sample period (10 Hz, simplified from MPU 50 Hz + ToF 5-10 Hz)
WINDOW_MS = 2000     # plan: 2-second windows
STEP_MS = 1000       # 1-second step
KNOISE_MM = 1.25     # M8 thread pitch per full turn (plan )
TOF_COUPLES = 0.4    # crack opening per mm of vertical subsidence

# severity thresholds, plan (mm of trapdoor descent)
CLASS_THRESHOLDS = (0.0, 15.0, 35.0)

TRIALS = [
    # trial_id, condition, speed, step_pattern, duration_s, target_mm, datetime
    ("T001", "TD1_only", "slow", "stepped", 149, 32.17, "2026-08-03 10:00:00"),
    ("T002", "TD2_only", "fast", "stepped", 166, 20.13, "2026-08-03 10:12:00"),
    ("T003", "both_together", "fast", "stepped", 147, 29.86, "2026-08-03 10:24:00"),
    ("T004", "both_sequential", "slow", "continuous", 123, 4.84, "2026-08-03 10:36:00"),
    ("T005", "TD1_only", "slow", "continuous", 162, 17.26, "2026-08-03 10:48:00"),
    ("T006", "TD2_only", "fast", "stepped", 159, 34.47, "2026-08-03 11:00:00"),
    ("T007", "both_together", "fast", "continuous", 152, 17.14, "2026-08-03 11:12:00"),
    ("T008", "both_sequential", "slow", "stepped", 152, 37.54, "2026-08-03 11:24:00"),
    ("T009", "TD1_only", "fast", "stepped", 153, 32.12, "2026-08-03 11:36:00"),
    ("T010", "TD2_only", "slow", "stepped", 125, 32.82, "2026-08-03 11:48:00"),
    ("T011", "both_together", "slow", "continuous", 162, 2.13, "2026-08-03 12:00:00"),
    ("T012", "both_sequential", "slow", "continuous", 95, 6.54, "2026-08-03 12:12:00"),
    ("T013", "TD1_only", "slow", "stepped", 176, 13.23, "2026-08-03 12:24:00"),
    ("T014", "TD2_only", "slow", "stepped", 147, 52.53, "2026-08-03 12:36:00"),
    ("T015", "both_together", "slow", "continuous", 162, 45.80, "2026-08-03 12:48:00"),
    ("T016", "both_sequential", "fast", "stepped", 119, 48.48, "2026-08-03 13:00:00"),
    ("T017", "TD1_only", "slow", "continuous", 156, 52.53, "2026-08-03 13:12:00"),
    ("T018", "TD2_only", "fast", "stepped", 95, 47.93, "2026-08-03 13:24:00"),
    ("T019", "both_together", "fast", "continuous", 180, 22.16, "2026-08-03 13:36:00"),
    ("T020", "both_sequential", "slow", "stepped", 178, 11.29, "2026-08-03 13:48:00"),
    ("T021", "TD1_only", "fast", "stepped", 99, 26.74, "2026-08-03 14:00:00"),
    ("T022", "TD2_only", "fast", "continuous", 154, 9.87, "2026-08-03 14:12:00"),
    ("T023", "both_together", "fast", "continuous", 132, 28.37, "2026-08-03 14:24:00"),
    ("T024", "both_sequential", "fast", "continuous", 145, 54.25, "2026-08-03 14:36:00"),
    ("T025", "TD1_only", "slow", "continuous", 177, 0.27, "2026-08-03 14:48:00"),
    ("T026", "TD2_only", "fast", "stepped", 103, 16.11, "2026-08-03 15:00:00"),
]

NODE_BASELINES = {"N1": 216.5, "N2": 224.2, "N3": 227.6, "N4": 215.4}  # ultrasonic mm
NODE_TOF_BASELINES = {"N1": 50.4, "N2": 48.9, "N3": 49.6, "N4": 50.1}  # mm
NODE_TO_TDTABLE = {"N2": "TD1", "N3": "TD2"}  # active node -> its trapdoor


def severity(kd_mm: np.ndarray) -> np.ndarray:
    """Plan bucketing: 0 = no movement, 1 = 0-15, 2 = 15-35, 3 = >35 mm."""
    cls = np.ones_like(kd_mm, dtype=int)
    cls[kd_mm <= CLASS_THRESHOLDS[0] + 1e-9] = 0
    cls[kd_mm > CLASS_THRESHOLDS[2]] = 3
    cls[(kd_mm > CLASS_THRESHOLDS[1]) & (kd_mm <= CLASS_THRESHOLDS[2])] = 2
    return cls


def build_turns_schedule(
    target_mm: float,
    speed: str,
    step_pattern: str,
    duration_s: int,
    delay_s: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Knob-turn schedule on the 100 ms grid.

 Returns ``(turns, disp_mm, step_onset_times_s)`` — monotonic, ending at
 ``target_mm / KNOISE_MM`` exactly. Stepped patterns produce discrete
 onsets (each one a vibration burst); continuous patterns ramp smoothly.
 """
    n = duration_s * 1000 // DT_MS
    t_s = np.arange(n) * DT_MS / 1000.0
    target_turns = target_mm / KNOISE_MM

    rate = (0.15 if speed == "slow" else 0.55) * rng.uniform(0.9, 1.1)  # mm/s
    avail = duration_s - delay_s - 3.0
    active_s = min(target_mm / rate, avail)
    end_t = delay_s + active_s

    onsets: list[float] = []
    if step_pattern == "stepped" and target_mm > 0:
        n_steps = int(np.clip(round(target_mm / 2.0), 5, 30))
        slot = active_s / n_steps
        rise_frac = 0.45 if speed == "slow" else 0.7
        edges = [delay_s]
        for k in range(1, n_steps + 1):
            onsets.append(delay_s + (k - 1) * slot)
            edges.append(delay_s + (k - 1) * slot + slot * rise_frac)
            edges.append(delay_s + k * slot)
        edges[-1] = end_t
        turns_at = np.linspace(0.0, target_turns, len(edges))
        turns = np.interp(t_s, edges, turns_at, left=0.0, right=target_turns)
    else:
        turns = np.clip((t_s - delay_s) / max(active_s, 1e-9), 0.0, 1.0) * target_turns
        turns = turns * (1.0 + 0.02 * np.sin(2 * np.pi * t_s / 17.0))  # hand wobble
        turns = np.clip(turns, 0.0, target_turns)

    turns = np.maximum.accumulate(turns)  # knob monotonicity: running max
    turns = np.round(turns, 3)
    return turns, turns * KNOISE_MM, onsets


def build_bursts(
    t_s: np.ndarray,
    onsets: list[float],
    disp_mm: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vibration burst amplitude (g) over time: step onsets + random micro-collapses."""
    sigma = np.full(t_s.size, 0.006)
    for onset in onsets:
        amp = rng.uniform(0.08, 0.18)
        mask = (t_s >= onset) & (t_s < onset + 2.5)
        sigma[mask] += amp * np.exp(-(t_s[mask] - onset) / 0.8)
    n_micro = max(1, int(t_s[-1] * 0.03))
    for _ in range(n_micro):
        start = rng.uniform(0.0, max(t_s[-1] - 2.0, 1.0))
        amp = rng.uniform(0.02, 0.05) * (1.0 + disp_mm[int(start * 10)] / 40.0)
        mask = (t_s >= start) & (t_s < start + rng.uniform(0.4, 1.2))
        sigma[mask] += amp * np.exp(-(t_s[mask] - start) / 0.5)
    return sigma


def orientation(disp_mm: np.ndarray, vib_sigma: np.ndarray, rng: np.random.Generator):
    """Per-sample roll/pitch (deg, from gravity) and raw accel/gyro in g, deg/s."""
    pitch = -0.32 * disp_mm            # node dips toward the descending trapdoor
    roll = 0.12 * disp_mm
    pitch_n = pitch + rng.normal(0.0, 0.05, pitch.size)
    roll_n = roll + rng.normal(0.0, 0.05, roll.size)

    pr, prl = np.radians(pitch_n), np.radians(roll_n)
    ax = -np.sin(pr) * np.cos(prl)
    ay = np.sin(prl)
    az = np.cos(pr) * np.cos(prl)
    vib = rng.normal(0.0, 1.0, (pitch.size, 3)) * vib_sigma[:, None]
    ax, ay, az = ax + vib[:, 0], ay + vib[:, 1], az + vib[:, 2]
    ax += rng.normal(0.0, 0.003, pitch.size)
    ay += rng.normal(0.0, 0.003, pitch.size)
    az += rng.normal(0.0, 0.003, pitch.size)

    d_pitch = np.gradient(pitch, DT_MS / 1000.0)
    d_roll = np.gradient(roll, DT_MS / 1000.0)
    gx = d_roll + vib_sigma * rng.normal(0.0, 1.0, pitch.size) * 30.0 + rng.normal(0.0, 0.1, pitch.size)
    gy = d_pitch + vib_sigma * rng.normal(0.0, 1.0, pitch.size) * 30.0 + rng.normal(0.0, 0.1, pitch.size)
    gz = rng.normal(0.0, 0.1, pitch.size) + vib[:, 2] * 20.0
    return ax, ay, az, gx, gy, gz, pitch_n, roll_n


def generate_trial(trial, rng) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    trial_id, condition, speed, step_pattern, duration_s, target_mm, dt_str = trial
    n = duration_s * 1000 // DT_MS
    t_s = np.arange(n) * DT_MS / 1000.0
    ts_ms = np.arange(n) * DT_MS

    delay_s = float(np.clip(rng.uniform(18.0, 32.0), 10.0, 0.3 * duration_s))
    movers = {
        "TD1_only": ("TD1",),
        "TD2_only": ("TD2",),
        "both_together": ("TD1", "TD2"),
        "both_sequential": ("TD1", "TD2"),
    }[condition]
    zeros = (np.zeros(n), np.zeros(n), [])
    if condition == "both_together":
        turns1, disp1, onsets1 = build_turns_schedule(target_mm, speed, step_pattern, duration_s, delay_s, rng)
        turns2, disp2, onsets2 = turns1, disp1, onsets1
    elif condition == "both_sequential":
        turns1, disp1, onsets1 = (
            build_turns_schedule(target_mm, speed, step_pattern, duration_s, delay_s, rng)
            if "TD1" in movers else zeros
        )
        delay2 = delay_s + 0.55 * (duration_s - delay_s)
        turns2, disp2, onsets2 = (
            build_turns_schedule(target_mm, speed, step_pattern, duration_s, delay2, rng)
            if "TD2" in movers else zeros
        )
    elif "TD1" in movers:  # TD1_only
        turns1, disp1, onsets1 = build_turns_schedule(target_mm, speed, step_pattern, duration_s, delay_s, rng)
        turns2, disp2, onsets2 = zeros
    else:  # TD2_only
        turns1, disp1, onsets1 = zeros
        turns2, disp2, onsets2 = build_turns_schedule(target_mm, speed, step_pattern, duration_s, delay_s, rng)

    disp_by_td = {"TD1": disp1, "TD2": disp2}
    sigma_by_td = {
        "TD1": build_bursts(t_s, onsets1, disp1, rng),
        "TD2": build_bursts(t_s, onsets2, disp2, rng),
    }

    raw_frames, win_frames = [], []
    final = {"TD1": float(disp1[-1]), "TD2": float(disp2[-1])}
    for node in ("N1", "N2", "N3", "N4"):
        td = NODE_TO_TDTABLE.get(node)
        disp = disp_by_td[td] if td else np.zeros(n)
        vib_sigma = sigma_by_td[td] if td else 0.3 * sigma_by_td["TD1"] + 0.3 * sigma_by_td["TD2"]
        if node in ("N1", "N4"):
            vib_sigma = np.maximum(vib_sigma, 0.006)  # references still feel box shake

        ax, ay, az, gx, gy, gz, pitch_n, roll_n = orientation(disp, vib_sigma, rng)
        drift = np.cumsum(rng.normal(0.0, 0.004, n))
        tof = NODE_TOF_BASELINES[node] + TOF_COUPLES * disp + drift + rng.normal(0.0, 0.05, n)
        udrift = np.cumsum(rng.normal(0.0, 0.01, n))
        ultra = NODE_BASELINES[node] + disp + udrift + rng.normal(0.0, 0.15, n)

        raw_frames.append(
            pd.DataFrame(
                {
                    "trial_id": trial_id,
                    "node_id": node,
                    "timestamp_ms": ts_ms,
                    "ax": ax.round(4), "ay": ay.round(4), "az": az.round(4),
                    "gx": gx.round(4), "gy": gy.round(4), "gz": gz.round(4),
                    "tof_distance_mm": tof.round(2),
                    "ultrasonic_distance_mm": ultra.round(2),
                    "knob_turns_TD1": turns1,
                    "knob_turns_TD2": turns2,
                }
            )
        )

        tof_base = float(tof[ts_ms < 2000].mean())
        ultra_base = float(ultra[ts_ms < 2000].mean())
        amag = np.sqrt(ax**2 + ay**2 + az**2)
        starts = np.arange(0, n - WINDOW_MS // DT_MS + 1, STEP_MS // DT_MS)
        for s0 in starts:
            sl = slice(s0, s0 + WINDOW_MS // DT_MS)
            row = {
                "trial_id": trial_id,
                "node_id": node,
                "window_start_ms": int(ts_ms[s0]),
                "roll": round(float(roll_n[sl].mean()), 3),
                "pitch": round(float(pitch_n[sl].mean()), 3),
                "vibration_rms": round(float(amag[sl].std()), 5),
                "accel_peak": round(float((amag[sl] - np.median(amag[sl])).max()), 5),
                "tof_delta": round(float(tof[sl].mean() - tof_base), 3),
                "tof_rate": round(float(np.polyfit(np.arange(sl.stop - sl.start), tof[sl], 1)[0] * 10.0), 4),
                "displacement_mm": round(float(ultra[sl].mean() - ultra_base), 3),
                "displacement_rate": round(float(np.polyfit(np.arange(sl.stop - sl.start), ultra[sl], 1)[0] * 10.0), 4),
            }
            if node in ("N1", "N4"):
                row["relative_tilt"] = 0.0
                row["relative_vibration"] = 0.0
                row["known_displacement_mm"] = 0.0
            else:
                kd = float(disp[s0 + WINDOW_MS // DT_MS - 1])  # mechanism truth at window end
                row["known_displacement_mm"] = round(kd, 3)
            win_frames.append((node, row))

    # relative features need the reference mean per window — second pass
    raw_df = pd.concat(raw_frames, ignore_index=True)
    per_node_rows: dict[str, list[dict]] = {n_id: [] for n_id in ("N1", "N2", "N3", "N4")}
    for node, row in win_frames:
        per_node_rows[node].append(row)
    ref = pd.DataFrame(
        {
            int(r["window_start_ms"]): {
                "pitch": 0.5 * (per_node_rows["N1"][i]["pitch"] + per_node_rows["N4"][i]["pitch"]),
                "vib": 0.5 * (per_node_rows["N1"][i]["vibration_rms"] + per_node_rows["N4"][i]["vibration_rms"]),
            }
            for i, r in enumerate(per_node_rows["N1"])
        }
    ).T
    for node in ("N2", "N3"):
        for r in per_node_rows[node]:
            base = ref.loc[r["window_start_ms"]]
            r["relative_tilt"] = round(r["pitch"] - base["pitch"], 3)
            r["relative_vibration"] = round(r["vibration_rms"] - base["vib"], 5)
            r["severity_class"] = int(severity(np.array([r["known_displacement_mm"]]))[0])
    for r in per_node_rows["N1"] + per_node_rows["N4"]:
        r["severity_class"] = 0

    order = ["trial_id", "node_id", "window_start_ms", "roll", "pitch", "vibration_rms",
             "accel_peak", "tof_delta", "tof_rate", "displacement_mm", "displacement_rate",
             "relative_tilt", "relative_vibration", "known_displacement_mm", "severity_class"]
    win_df = pd.DataFrame(
        [r for node in ("N1", "N2", "N3", "N4") for r in per_node_rows[node]]
    )[order]
    return raw_df, win_df, final


def main() -> int:
    rng = np.random.default_rng(SEED)
    metadata_rows, raw_parts, win_parts = [], [], []
    for trial in TRIALS:
        raw_df, win_df, final = generate_trial(trial, rng)
        raw_parts.append(raw_df)
        win_parts.append(win_df)
        metadata_rows.append(
            {
                "trial_id": trial[0],
                "condition": trial[1],
                "speed": trial[2],
                "step_pattern": trial[3],
                "duration_s": float(trial[4]),
                "max_displacement_mm_target": trial[5],
                "datetime": trial[6],
                "achieved_displacement_mm_TD1": round(final["TD1"], 3),
                "achieved_displacement_mm_TD2": round(final["TD2"], 3),
            }
        )
        print(f"{trial[0]} ({trial[1]:16s} {trial[2]:5s} {trial[3]:10s}): target={trial[5]:6.2f} mm "
              f"-> achieved TD1={final['TD1']:6.2f} TD2={final['TD2']:6.2f}")

    metadata = pd.DataFrame(metadata_rows)
    raw = pd.concat(raw_parts, ignore_index=True)
    win = pd.concat(win_parts, ignore_index=True)

    # built-in audit (fails loudly on any regression)
    assert len(raw) == 150720, f"raw row count changed: {len(raw)}"
    assert len(win) == 14968, f"windowed row count changed: {len(win)}"
    for (t, nid), g in raw.groupby(["trial_id", "node_id"]):
        assert (np.diff(np.sort(g.timestamp_ms.values)) == DT_MS).all(), f"grid broken {t}/{nid}"
        for c in ("knob_turns_TD1", "knob_turns_TD2"):
            assert (np.diff(g[c].values) >= -1e-12).all(), f"knob not monotonic {t}/{nid}/{c}"
    for t, g in raw.groupby("trial_id"):
        row = metadata[metadata.trial_id == t].iloc[0]
        for td_col, ach in (("knob_turns_TD1", "achieved_displacement_mm_TD1"),
                            ("knob_turns_TD2", "achieved_displacement_mm_TD2")):
            hit = g[td_col].max() * KNOISE_MM
            assert abs(hit - row[ach]) < 0.01, f"{t} {td_col} final {hit} != achieved {row[ach]}"
        actives = [td for td, col in (("TD1", "knob_turns_TD1"), ("TD2", "knob_turns_TD2"))
                   if g[col].max() > 0]
        target = row["max_displacement_mm_target"]
        for td in actives:
            hit = g[f"knob_turns_{td}"].max() * KNOISE_MM
            assert abs(hit - target) < 0.01, f"{t} {td} final {hit:.3f} != target {target}"
        refs = raw[(raw.trial_id == t) & (raw.node_id.isin(["N1", "N4"]))]
        assert refs.knob_turns_TD1.max() == 0 or "TD1" in actives
    act = win[win.node_id.isin(["N2", "N3"])]
    assert act.known_displacement_mm.min() >= 0.0, "ground truth must never go negative"
    for cls, lo, hi in ((0, 0.0, 0.0), (1, 0.0, 15.0), (2, 15.0, 35.0), (3, 35.0, np.inf)):
        sub = act.loc[act.severity_class == cls, "known_displacement_mm"]
        assert sub.min() >= lo - 1e-9 and (sub.max() <= hi or hi == np.inf), f"class {cls} bounds"
    assert (win[win.node_id.isin(["N1", "N4"])]["known_displacement_mm"] == 0).all()
    assert (win[win.node_id.isin(["N1", "N4"])]["relative_tilt"] == 0).all()
    assert act.vibration_rms.std() > 0.004, "vibration_rms must vary"
    assert act.accel_peak.max() > 0.05, "burst peaks must be visible"

    metadata.to_csv(OUT_DIR / "trial_metadata.csv", index=False)
    raw.to_csv(OUT_DIR / "raw_sensor_log.csv", index=False)
    win.to_csv(OUT_DIR / "processed_windowed_dataset.csv", index=False)

    print("\naudit PASSED")
    print(f"raw rows {len(raw):,} | windowed rows {len(win):,} | trials {len(metadata)}")
    print("severity_class distribution (all nodes):", win.severity_class.value_counts().sort_index().to_dict())
    print("severity_class distribution (active nodes only):",
          win[win.node_id.isin(['N2','N3'])].severity_class.value_counts().sort_index().to_dict())
    print(f"vibration_rms range (active): {act.vibration_rms.min():.4f}-{act.vibration_rms.max():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
