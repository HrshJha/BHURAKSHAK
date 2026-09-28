"""Offline InSAR processing workflow — PRD §18 step 3, §4 non-goal (T-058).

§18: InSAR processing runs on the workstation, NEVER on the Raspberry Pi.
Toolchain: pure NumPy/SciPy — rasterio/SNAP/ISCE are deliberately NOT
dependencies (the §9.2 precedent: simple, auditable implementations over
heavy geospatial stacks for a small-area MVP). Every parameter comes from
configs/insar.yaml (NFR-6).

Pipeline per interferometric pair (all functions complex-domain unless noted):

1. **coarseregister** — shift the secondary SLC onto the master grid by an
   integer pixel offset (orbit-accuracy coregistration for the MVP; the
   sub-pixel DEM-assisted refinement of full processors is documented as an
   assumption in docs/insar_workflow.md).
2. **multi_look** — complex averaging (range × azimuth factors from config)
   for ~20 × 20 m ground resolution and speckle reduction.
3. **interferogram** — master · conj(secondary); remove the flat-Earth
   phase ramp (a plane fit in range/azimuth — MVP-level orbital fringe
   removal).
4. **power_spectrum_filter** — topographic-phase-independent noise
   filtering of the wrapped interferogram (non-linear spectral filter; the
   Goldstein-Werner power-spectrum approach, simplified).
5. **coherence** — sliding-window |<m·s*>| / sqrt(<|m|²><|s|²>) with the
   configured window; low-coherence pixels are MASKED (never silently
   included — §18 step 4).

The small-baseline stack combines all pairs whose temporal baseline is
within [min, max] days (config): per multi-look pixel, wrapped phases are
temporally averaged (weighted by coherence) — phase unwrapping is SKIPPED
in the MVP (documented honestly); PS points (amplitude-dispersion gate)
get per-date displacement estimates relative to the highest-coherence /
lowest-variance reference candidate outside mapped subsidence.

Validation: the module is exercised on SYNTHETIC SLC stacks with a known
deformation signal (the T-058 tests and the notebook-08 builder) — the
processing recovers the imposed deformation, which validates the chain
given real SLC scenes from T-057's credentialed download.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping

import numpy as np
import yaml

__all__ = [
    "InSARError",
    "InSARConfig",
    "load_insar_config",
    "coarseregister",
    "multi_look",
    "interferogram",
    "power_spectrum_filter",
    "coherence",
    "amplitude_dispersion",
    "select_reference_point",
    "small_baseline_pairs",
    "process_pair",
    "build_stack",
]

_DAYS_PER_YEAR = 365.25


class InSARError(ValueError):
    """Raised on invalid InSAR inputs, shapes or configuration."""


def load_insar_config(path: str | None = None) -> dict:
    import sys
    from pathlib import Path

    root = Path(path) if path else Path(__file__).resolve().parents[2] / "configs" / "insar.yaml"
    sys.path.insert(0, str(root.parents[1]))
    with root.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class InSARConfig:
    """Typed view over the configs/insar.yaml processing block."""

    range_looks: int
    azimuth_looks: int
    power_spectrum_filter: bool
    coherence_window: tuple[int, int]
    coherence_mask_threshold: float
    min_baseline_days: int
    max_baseline_days: int
    amplitude_dispersion_max: float
    min_stack_coherence: float
    min_observations: int
    max_ref_rate_mm_yr: float

    @classmethod
    def from_yaml(cls, cfg: Mapping) -> "InSARConfig":
        proc = cfg["processing"]
        ml = proc["interferogram"]["multi_look_factors"]
        cw = proc["coherence"]["window"]
        ps = proc["ps_selection"]
        return cls(
            range_looks=int(ml[0]),
            azimuth_looks=int(ml[1]),
            power_spectrum_filter=bool(proc["interferogram"].get("power_spectrum_filter", True)),
            coherence_window=(int(cw[0]), int(cw[1])),
            coherence_mask_threshold=float(proc["coherence"]["mask_threshold"]),
            min_baseline_days=int(proc["interferogram"]["min_baseline_days"]),
            max_baseline_days=int(proc["interferogram"]["max_baseline_days"]),
            amplitude_dispersion_max=float(ps["amplitude_dispersion_max"]),
            min_stack_coherence=float(ps["min_stack_coherence"]),
            min_observations=int(ps["min_observations"]),
            max_ref_rate_mm_yr=float(proc["reference_point"]["max_displacement_rate_mm_yr"]),
        )
# 
def coarseregister(secondary: np.ndarray, dr: int, da: int) -> np.ndarray:
    """Integer-pixel shift of ``secondary`` onto the master grid (orbit accuracy).

    Positive ``dr``/``da`` mean the secondary must move toward larger
    range/azimuth indices. Shifted-in borders are zeroed (they carry no
    valid signal).
    """
    if secondary.ndim != 2 or not np.iscomplexobj(secondary):
        raise InSARError("coarseregister expects a 2-D complex SLC array")
    out = np.zeros_like(secondary)
    src_r0, src_a0 = max(0, dr), max(0, da)
    dst_r0, dst_a0 = max(0, -dr), max(0, -da)
    rows = min(secondary.shape[0] - src_r0, secondary.shape[0] - dst_r0)
    cols = min(secondary.shape[1] - src_a0, secondary.shape[1] - dst_a0)
    if rows > 0 and cols > 0:
        out[dst_r0 : dst_r0 + rows, dst_a0 : dst_a0 + cols] = secondary[
            src_r0 : src_r0 + rows, src_a0 : src_a0 + cols
        ]
    return out


# --- step 2: multi-looking -------------------------------------------------------


def multi_look(slc: np.ndarray, range_looks: int, azimuth_looks: int) -> np.ndarray:
    """Complex averaging → coarser resolution + speckle reduction."""
    if slc.ndim != 2 or not np.iscomplexobj(slc):
        raise InSARError("multi_look expects a 2-D complex SLC array")
    if range_looks < 1 or azimuth_looks < 1:
        raise InSARError("multi-look factors must be >= 1")
    n_r = (slc.shape[0] // range_looks) * range_looks
    n_a = (slc.shape[1] // azimuth_looks) * azimuth_looks
    trimmed = slc[:n_r, :n_a]
    return trimmed.reshape(
        n_r // range_looks, range_looks, n_a // azimuth_looks, azimuth_looks
    ).mean(axis=(1, 3))


# --- step 3: interferogram + flat-Earth removal -----------------------------------


def _plane_fit(phase: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Least-squares plane (a + b·r + c·a) over valid cells; NaN-safe."""
    rr, aa = np.indices(phase.shape)
    mask = valid & np.isfinite(phase)
    if mask.sum() < 3:
        return np.zeros_like(phase)
    A = np.column_stack([np.ones(mask.sum()), rr[mask], aa[mask]])
    coef, *_ = np.linalg.lstsq(A, phase[mask], rcond=None)
    return coef[0] + coef[1] * rr + coef[2] * aa


def interferogram(
    master_mlooked: np.ndarray,
    secondary_mlooked: np.ndarray,
    remove_flat_earth: bool = True,
) -> np.ndarray:
    """Wrapped interferogram: master · conj(secondary) with plane ramp removed."""
    if master_mlooked.shape != secondary_mlooked.shape:
        raise InSARError(f"multi-looked SLC shape mismatch: {master_mlooked.shape} vs {secondary_mlooked.shape}")
    ifg = master_mlooked * np.conj(secondary_mlooked)
    if remove_flat_earth:
        valid = np.abs(ifg) > 0
        phase = np.angle(ifg)
        ramp = _plane_fit(np.where(valid, phase, 0.0), valid)
        ifg = np.exp(1j * (phase - ramp)) * np.abs(ifg)
    return ifg


# --- step 4: power-spectrum (Goldstein-style) filtering ---------------------------


def power_spectrum_filter(ifg: np.ndarray, alpha: float = 1 / 2, block: int = 32) -> np.ndarray:
    """Goldstein–Werner non-linear spectral filter of the COMPLEX interferogram.

    Per ``block × block`` tile: FFT of the complex ifg, multiply the spectrum
    by ``|spectrum|**alpha`` (α = the α parameter, one-half by default), inverse FFT. Phase is preserved
    for smooth fields (the weighting reshapes magnitudes only) while fringing
    noise is suppressed. Operating on the phase field instead would CORRUPT
    phase (amplitude leakage) — a bug caught by the synthetic-stack recovery
    check and fixed by filtering the complex domain, as in the literature.
    Tiles are padded so block edges do not create seams.
    """
    if ifg.ndim != 2 or not np.iscomplexobj(ifg):
        raise InSARError("power_spectrum_filter expects a 2-D COMPLEX interferogram")
    out = np.empty_like(ifg)
    n_r, n_a = ifg.shape
    for r0 in range(0, n_r, block):
        for a0 in range(0, n_a, block):
            tile = ifg[r0 : min(r0 + block, n_r), a0 : min(a0 + block, n_a)]
            pr, pa = block - tile.shape[0], block - tile.shape[1]
            padded = np.pad(tile, ((0, pr), (0, pa)), mode="reflect") if min(tile.shape) > 1 else np.pad(tile, ((0, pr), (0, pa)))
            spec = np.fft.fft2(padded)
            filtered = np.fft.ifft2(np.abs(spec) ** alpha * spec)
            out[r0 : min(r0 + block, n_r), a0 : min(a0 + block, n_a)] = filtered[: tile.shape[0], : tile.shape[1]]
    return out
# 
def _boxcar(sum_sqi: np.ndarray, sum_prod: np.ndarray, window: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    from scipy.ndimage import uniform_filter

    wr, wa = window
    num = uniform_filter(sum_prod.real, size=(wr, wa), mode="nearest") + 1j * uniform_filter(
        sum_prod.imag, size=(wr, wa), mode="nearest"
    )
    den_a = uniform_filter(sum_sqi[0], size=(wr, wa), mode="nearest")
    den_b = uniform_filter(sum_sqi[1], size=(wr, wa), mode="nearest")
    return num, den_a * den_b


def coherence(master: np.ndarray, secondary: np.ndarray, window: tuple[int, int]) -> np.ndarray:
    """Sliding-window coherence |<m·s*>| / sqrt(<|m|²><|s|²>) in [0, 1]."""
    if master.shape != secondary.shape:
        raise InSARError("coherence expects master and secondary of identical (multi-looked) shape")
    prod = master * np.conj(secondary)
    sum_prod = prod.astype(complex)
    sum_sqi = (np.abs(master) ** 2, np.abs(secondary) ** 2)
    num, den = _boxcar(sum_sqi, sum_prod, window)
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = np.abs(num) / np.sqrt(np.maximum(den, np.finfo(float).eps))
    return np.clip(coh, 0.0, 1.0)
# 
def amplitude_dispersion(stack_amplitudes: np.ndarray) -> np.ndarray:
    """Amplitude dispersion σ_A/μ_A per pixel across the stack (PS gate).

    ``stack_amplitudes`` is (n_dates, ...) — the reduction is over axis 0,
    so 2-D (n_dates, n_pixels) and 3-D (n_dates, rows, cols) inputs both work.
    """
    if stack_amplitudes.ndim < 2:
        raise InSARError("amplitude_dispersion expects (n_dates, ...) — one amplitude map per date")
    mean = stack_amplitudes.mean(axis=0)
    std = stack_amplitudes.std(axis=0, ddof=1) if stack_amplitudes.shape[0] > 1 else np.zeros_like(mean)
    with np.errstate(invalid="ignore", divide="ignore"):
        disp = std / np.maximum(mean, np.finfo(float).eps)
    return disp


def select_reference_point(
    rates_mm_yr: np.ndarray,
    stack_coherence: np.ndarray,
    cfg: InSARConfig,
) -> int:
    """Highest-coherence, lowest-variance candidate below the ref rate cap."""
    eligible = np.where(
        (np.abs(rates_mm_yr) <= cfg.max_ref_rate_mm_yr) & (stack_coherence >= cfg.min_stack_coherence)
    )[0]
    if eligible.size == 0:
        raise InSARError("no reference-point candidate: no stable PS below the configured rate cap")
    order = sorted(eligible, key=lambda i: (-stack_coherence[i], abs(rates_mm_yr[i])))
    return int(order[0])
# 
def small_baseline_pairs(dates: list[str], cfg: InSARConfig) -> list[tuple[int, int]]:
    """All (master, secondary) index pairs within the temporal-baseline window."""
    from datetime import date

    def d(s: str) -> date:
        return date.fromisoformat(s)

    pairs = []
    for i, j in combinations(range(len(dates)), 2):
        baseline = (d(dates[j]) - d(dates[i])).days
        if cfg.min_baseline_days <= baseline <= cfg.max_baseline_days:
            pairs.append((i, j))
    return pairs


def process_pair(
    master: np.ndarray,
    secondary: np.ndarray,
    cfg: InSARConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Full per-pipe: multi-look → interferogram → [optional filter] → coherence.

    Returns ``(wrapped_ifg, coherence)`` at multi-look resolution.
    """
    m, s = multi_look(master, cfg.range_looks, cfg.azimuth_looks), multi_look(
        secondary, cfg.range_looks, cfg.azimuth_looks
    )
    ifg = interferogram(m, s)
    if cfg.power_spectrum_filter:
        ifg = power_spectrum_filter(ifg)
    return ifg, coherence(m, s, cfg.coherence_window)


def build_stack(
    slcs: list[np.ndarray],
    dates: list[str],
    cfg: InSARConfig,
) -> dict:
    """Small-baseline stack over the whole scene set.

    Per multi-look pixel with enough observations: coherence-weighted mean of
    the wrapped, filtered pair phases (unwrapping skipped — MVP), then PS
    gating by amplitude dispersion + stack coherence. Returns per-date
    phase-derived displacement proxy arrays, the PS mask, and diagnostics.
    """
    if len(slcs) != len(dates) or len(dates) < 3:
        raise InSARError("build_stack needs aligned slcs/dates (>= 3 dates)")
    pairs = small_baseline_pairs(dates, cfg)
    if not pairs:
        raise InSARError("no interferometric pairs within the configured baseline window")

    mlooked = [multi_look(s, cfg.range_looks, cfg.azimuth_looks) for s in slcs]
    shape = mlooked[0].shape
    phase_sum = np.zeros(shape, dtype=float)
    weight_sum = np.zeros(shape, dtype=float)
    obs_count = np.zeros(shape, dtype=int)
    n_dates = len(dates)
    date_coh_sum = np.zeros((n_dates,) + shape, dtype=float)
    date_weight = np.zeros((n_dates,) + shape, dtype=float)
    pair_phases = np.full((len(pairs),) + shape, np.nan)  # wrapped pair phases for the graph inversion

    for p_idx, (i, j) in enumerate(pairs):
        ifg = interferogram(mlooked[i], mlooked[j])
        if cfg.power_spectrum_filter:
            ifg = power_spectrum_filter(ifg)
        coh = coherence(mlooked[i], mlooked[j], cfg.coherence_window)
        valid = coh >= cfg.coherence_mask_threshold
        phase = np.angle(ifg)
        phase_sum[valid] += coh[valid] * phase[valid]
        weight_sum[valid] += coh[valid]
        obs_count += valid.astype(int)
        pair_phases[p_idx] = np.where(valid, np.angle(ifg), np.nan)
        for k in (i, j):
            date_weight[k][valid] += coh[valid]
            date_coh_sum[k][valid] += coh[valid]

    # Small-baseline INVERSION: solve the pair network for per-date phases
    # θ_k (gauge θ_0 = 0) by least squares over the graph — minimises
    # Σ_pairs (θ_i − θ_j − φ_ij)². Unweighted (coherence weighting remains in
    # the stack mean + PS gates), valid while pair phases stay unwrapped-small;
    # one pseudo-inverse applied to every pixel (vectorised, no per-pixel loops).
    from datetime import date as _date

    dts = [_date.fromisoformat(s) for s in dates]
    n_unknown = n_dates - 1
    A = np.zeros((len(pairs), n_unknown))
    for p_idx, (i, j) in enumerate(pairs):
        # convention: θ_j − θ_i = φ_ij (secondary minus master), so a
        # range-decreasing (subsidence-like) signal yields negative LOS mm
        if j > 0:
            A[p_idx, j - 1] += 1.0
        if i > 0:
            A[p_idx, i - 1] -= 1.0
    pinv = np.linalg.pinv(A.T @ A) @ A.T                       # (n_unknown, n_pairs)
    finite = np.isfinite(pair_phases)
    filled = np.where(finite, pair_phases, 0.0)
    theta = np.zeros((n_dates,) + shape)                       # θ_0 = 0 gauge
    theta[1:] = np.einsum("kp,p...->k...", pinv, filled) * np.where(finite.any(axis=0), 1.0, np.nan)
    per_date_phase = theta

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_phase = np.where(weight_sum > 0, phase_sum / np.maximum(weight_sum, np.finfo(float).eps), np.nan)
        per_date_coh = np.where(
            date_weight > 0, date_coh_sum / np.maximum(date_weight, np.finfo(float).eps), np.nan
        )

    amps = np.stack([np.abs(m) for m in mlooked])
    disp = amplitude_dispersion(amps)
    stack_coh = np.where(weight_sum > 0, weight_sum / max(len(pairs), 1), 0.0)
    ps_mask = (
        (disp <= cfg.amplitude_dispersion_max)
        & (stack_coh >= cfg.min_stack_coherence)
        & (obs_count >= cfg.min_observations)
    )

    return {
        "pairs": pairs,
        "mean_wrapped_phase": mean_phase,
        "per_date_wrapped_phase": per_date_phase,
        "per_date_pair_weight": date_weight,
        "per_date_mean_coherence": per_date_coh,
        "stack_coherence": stack_coh,
        "amplitude_dispersion": disp,
        "observation_count": obs_count,
        "ps_mask": ps_mask,
        "low_coherence_masked": bool((weight_sum == 0).any()),
    }
