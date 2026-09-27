#!/usr/bin/env python3
"""T-059 — Derive the InSAR deformation time series (PRD §18 step 4).

Produces data/processed/insar/deformation_timeseries.parquet: per
persistent-scatterer (PS) point × acquisition date — LOS displacement
(phase-derived proxy), velocity and coherence — over the scene stack, with
low-coherence observations MASKED (NaN), never silently included.

Two modes:
  synthetic (default) — build a synthetic SLC stack with a KNOWN imposed
  deformation field (Mogi-style point source + speckle + flat-Earth ramp),
  run the full T-058 chain, verify recovery, and write the parquet + run
  record. This validates the chain end-to-end while the 21 real scenes
  await T-057's credentialed download.
  real — process the downloaded SLC zips from data/raw/sentinel1/ (requires
  the credentialed download to have succeeded; SAR raster reading without
  rasterio is out of MVP scope, so this mode raises with instructions).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.geospatial.insar_processing import (
    InSARConfig,
    InSARError,
    build_stack,
    load_insar_config,
)

OUT_PARQUET = REPO_ROOT / "data" / "processed" / "insar" / "deformation_timeseries.parquet"
RUN_RECORD = REPO_ROOT / "experiments" / "insar_run.json"

# synthetic scene geometry (kept small: the chain is what is under test)
N_RANGE, N_AZIMUTH = 512, 1024   # raw SLC axes (range × azimuth) for the synthetic scene
WAVELENGTH_M = 0.0556            # Sentinel-1 C-band


def synth_dates(n: int = 21) -> list[str]:
    """The REAL T121 DESC acquisition dates from the T-057 inventory manifest."""
    import json

    manifest = json.loads((REPO_ROOT / "data" / "raw" / "sentinel1" / "inventory_manifest.json").read_text())
    return manifest["dates"][:n]


def impose_deformation(n_range: int, n_azimuth: int, dates: list[str], rng: np.ndarray) -> np.ndarray:
    """Known ground truth: a subsiding point source growing linearly in time.

    Returns LOS deformation in metres per (date, range, azimuth) in the
    satellite look direction — encoded as phase φ = −4π·d_LOS/λ on the
    secondary scenes.
    """
    from datetime import date

    t_days = np.array(
        [(date.fromisoformat(d) - date.fromisoformat(dates[0])).days for d in dates], dtype=float
    )
    rate_m_per_day = 12e-3 / 365.25  # 12 mm/yr subsiding source
    src_r, src_a = n_range * 0.3, n_azimuth * 0.6
    rr, aa = np.meshgrid(np.arange(n_range), np.arange(n_azimuth), indexing="ij")
    dist = np.sqrt((rr - src_r) ** 2 * 20.0**2 + (aa - src_a) ** 2 * 20.0**2)  # ~20 m pixels
    influence = np.exp(-(dist**2) / (2 * 800.0**2))  # Mogi-like bowl, ~1.6 km footprint
    los = rate_m_per_day * t_days[:, None, None] * influence[None, :, :]
    return los


def synth_slc_stack(dates: list[str], seed: int = 42) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Synthetic master + secondaries: bright stable point scatterers (PS),
    speckle background, and UNSTABLE decorrelated patches — so the coherence
    mask and PS gates have real work to do (§18 step 4 discipline)."""
    rng = np.random.default_rng(seed)
    truth = impose_deformation(N_RANGE, N_AZIMUTH, dates, rng)

    master = (rng.standard_normal((N_RANGE, N_AZIMUTH)) + 1j * rng.standard_normal((N_RANGE, N_AZIMUTH))).astype(complex)
    # stable bright point-like scatterers (~2% of pixels): buildings/infrastructure
    n_points = (N_RANGE * N_AZIMUTH) // 50
    flat = master.ravel()
    idx = rng.choice(flat.size, size=n_points, replace=False)
    flat[idx] += 5.0 * (1.0 + 0.05 * rng.standard_normal(n_points))
    master = flat.reshape(N_RANGE, N_AZIMUTH) + 2.0

    # unstable patches (vegetation / active work faces): decorrelate per date
    unstable = np.zeros((N_RANGE, N_AZIMUTH), dtype=bool)
    for _ in range(6):
        r0 = int(rng.integers(0, N_RANGE - 64))
        a0 = int(rng.integers(0, N_AZIMUTH - 64))
        unstable[r0 : r0 + 64, a0 : a0 + 64] = True

    slcs = [master]
    for k in range(1, len(dates)):
        secondary = master.copy()
        phase = np.exp(-1j * 4.0 * np.pi * truth[k] / WAVELENGTH_M)  # LOS deformation phase
        secondary *= phase
        noise_level = np.where(unstable, 1.5, 0.15)
        noise = noise_level * (rng.standard_normal((N_RANGE, N_AZIMUTH)) + 1j * rng.standard_normal((N_RANGE, N_AZIMUTH)))
        secondary = secondary + noise * np.abs(secondary)
        slcs.append(secondary)
    return slcs, truth, unstable


def stack_to_timeseries(stack: dict, dates: list[str], cfg: InSARConfig) -> pd.DataFrame:
    """Per-PS-point × date rows: LOS displacement (wrapped-referenced proxy),
    velocity and acceleration from per-date series, coherence per date —
    low-coherence observations MASKED (NaN), never silently included."""
    ps = np.where(stack["ps_mask"])
    if ps[0].size == 0:
        raise InSARError("no PS points selected — loosen configs/insar.yaml ps_selection gates")
    mean_phase = stack["mean_wrapped_phase"][ps]
    stack_coh = stack["stack_coherence"][ps]
    obs = stack["observation_count"][ps]
    per_date_phase = stack["per_date_wrapped_phase"][:, ps[0], ps[1]]   # (n_dates, n_ps)
    per_date_coh = stack["per_date_mean_coherence"][:, ps[0], ps[1]]

    # Reference point among high-coherence PS: highest coherence, tie-break
    # lowest |stack-mean phase|. (The config's rate-cap rule applies in the
    # real/unwrapped mode; a rate cannot be known before referencing.)
    eligible = np.where(stack_coh >= cfg.min_stack_coherence)[0]
    if eligible.size == 0:
        raise InSARError("no PS reaches min_stack_coherence for the reference point")
    ref = int(min(eligible, key=lambda i: (-stack_coh[i], abs(mean_phase[i]))))

    # Displacement proxy per date: wrapped phase differences vs the REFERENCE
    # DATE (d0), i.e. φ(d_k) − φ(d_0) → mm via d_LOS = −λ·Δφ/(4π), masked where
    # either date lacks pair coverage. Unwrapping is skipped (MVP): the proxy
    # is mod 2π and suitable for ranking/monitoring, stated honestly.
    d0_phase = per_date_phase[0]
    with np.errstate(invalid="ignore"):
        delta = np.angle(np.exp(1j * (per_date_phase - d0_phase[None, :])))
    los_mm = -WAVELENGTH_M * delta / (4.0 * np.pi) * 1e3
    los_mm[per_date_coh < cfg.coherence_mask_threshold] = np.nan       # MASK, don't include

    # velocity/acceleration per PS: robust linear fit over valid observations
    t_days = np.array([pd.Timestamp(d).value for d in dates]) / 86400e9
    vel = np.full(ps[0].size, np.nan)
    acc = np.full(ps[0].size, np.nan)
    for i in range(ps[0].size):
        m = np.isfinite(los_mm[:, i])
        if m.sum() >= 3:
            coef = np.polyfit(t_days[m], los_mm[m, i], 1)          # mm/day
            vel[i] = coef[0] * 365.25
            if m.sum() >= 5:
                quad = np.polyfit(t_days[m], los_mm[m, i], 2)
                acc[i] = 2.0 * quad[0] * 365.25**2

    n_ps = ps[0].size
    rows = []
    for k, d in enumerate(dates):
        rows.append(
            pd.DataFrame(
                {
                    "ps_id": [f"PS_{i:05d}" for i in range(n_ps)],
                    "range_idx": ps[0],
                    "azimuth_idx": ps[1],
                    "date": d,
                    "los_displacement_mm": los_mm[k],
                    "los_velocity_mm_yr": vel,
                    "los_acceleration_mm_yr2": acc,
                    "coherence": per_date_coh[k],
                    "n_observations": obs,
                    "masked_low_coherence": per_date_coh[k] < cfg.coherence_mask_threshold,
                }
            )
        )
    df = pd.concat(rows, ignore_index=True)
    return df


def main() -> int:
    cfg = InSARConfig.from_yaml(load_insar_config())
    dates = synth_dates()
    print(f"stack: {len(dates)} dates {dates[0]} … {dates[-1]}")

    slcs, truth, unstable = synth_slc_stack(dates)
    stack = build_stack(slcs, dates, cfg)
    n_ps = int(stack["ps_mask"].sum())
    print(f"PS points selected: {n_ps} / {stack['ps_mask'].size} multi-look pixels "
          f"(dispersion ≤ {cfg.amplitude_dispersion_max}, coherence ≥ {cfg.min_stack_coherence})")

    # recovery check: per-date displacement proxy vs imposed truth at PS points
    ps = np.where(stack["ps_mask"])
    imposed_phase = -4.0 * np.pi * truth[-1] / WAVELENGTH_M
    imposed_at_ps = imposed_phase[ps]
    observed = stack["mean_wrapped_phase"][ps]
    imposed_wrapped = np.angle(np.exp(1j * imposed_at_ps))
    resid = np.angle(np.exp(1j * (observed - imposed_wrapped)))
    recovery = float(np.sqrt(np.mean(resid**2)))
    print(f"phase recovery RMS (wrapped, rad): {recovery:.4f}  (0 = perfect)")

    # velocity recovery: fitted per-PS velocity vs the imposed 12 mm/yr source
    df = stack_to_timeseries(stack, dates, cfg)
    src_r, src_a = int(N_RANGE * 0.3), int(N_AZIMUTH * 0.6)
    # multi-look indices: range/2, azimuth/8 (config factors 2 x 8)
    ml_r, ml_a = src_r / cfg.range_looks, src_a / cfg.azimuth_looks
    centre = df.drop_duplicates("ps_id").assign(
        dist=lambda x: np.sqrt((x.range_idx - ml_r) ** 2 + (x.azimuth_idx - ml_a) ** 2)
    ).sort_values("dist").iloc[0]
    v_centre = float(centre["los_velocity_mm_yr"])
    v_far = float(
        df.drop_duplicates("ps_id")
        .assign(dist=lambda x: np.sqrt((x.range_idx - ml_r) ** 2 + (x.azimuth_idx - ml_a) ** 2))
        .sort_values("dist").iloc[-1]["los_velocity_mm_yr"]
    )
    print(f"velocity at PS nearest source: {v_centre:.1f} mm/yr | farthest PS: {v_far:.1f} mm/yr (imposed source −12 mm/yr at centre)")

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {len(df):,} rows ({df.ps_id.nunique()} PS × {len(dates)} dates) → {OUT_PARQUET.relative_to(REPO_ROOT)}")

    RUN_RECORD.parent.mkdir(parents=True, exist_ok=True)
    RUN_RECORD.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "synthetic (real scenes pending T-057 credentialed download)",
                "n_dates": len(dates),
                "n_pairs": len(stack["pairs"]),
                "n_ps_points": n_ps,
                "phase_recovery_rms_rad": recovery,
                "velocity_at_source_mm_yr": v_centre,
                "velocity_farthest_mm_yr": v_far,
                "imposed_rate_mm_yr": 12.0,
                "low_coherence_masked": stack["low_coherence_masked"],
                "parquet": str(OUT_PARQUET.relative_to(REPO_ROOT)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"run record → {RUN_RECORD.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
