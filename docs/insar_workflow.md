# T-058 — Offline InSAR Processing Workflow (PRD §18 step 3)

**Runs on the workstation only — never on the Raspberry Pi (§18, §4 non-goal).**

## Toolchain (and why)

Pure **NumPy/SciPy** implementation in `src/geospatial/insar_processing.py`.
rasterio/GAMMA/SNAP/ISCE are deliberately NOT dependencies: the MVP processes
one small area (a ~14 × 22 km bbox on a single track slice), and a compact,
auditable implementation keeps every processing choice visible and
config-driven (NFR-6). Every parameter below lives in `configs/insar.yaml`.

## Pipeline (per interferometric pair)

| Step | Function | Config | Notes / assumptions |
|---|---|---|---|
| 1. Coarseregistration | `coarseregister` | — | Integer-pixel orbit-accuracy shift; shifted-in borders zeroed. Sub-pixel DEM-assisted refinement (SNAP/GAMMA standard) is **assumed unnecessary** for MVP jitters < 0.1 px — recorded as an explicit assumption. |
| 2. Multi-looking | `multi_look` | `multi_look_factors: [2, 8]` | Complex averaging, range × azimuth → ~20 × 20 m ground resolution, speckle reduction. |
| 3. Interferogram | `interferogram` | — | master · conj(secondary); flat-Earth phase removed by a least-squares **plane fit** in range/azimuth (MVP-level orbital fringe removal — real orbital ramps are near-planar over one small slice). |
| 4. Phase filtering | `power_spectrum_filter` | `power_spectrum_filter: false` (MVP: OFF) | Goldstein–Werner-style filter implemented on the COMPLEX interferogram (per 32×32 block, α = 0.5). **Empirically disabled**: on this small scene the block-wise implementation attenuated deformation phase (pair phase +1.04 rad raw → +0.65 rad filtered) — caught by the synthetic-stack recovery check and recorded in configs/insar.yaml. Multi-look averaging + the least-squares stack inversion provide the noise suppression instead; re-enable only after a spatially-adaptive rewrite. |
| 5. Coherence | `coherence` | `window: [5, 5]`, `mask_threshold: 0.45` | Sliding-window estimator; **pixels below the threshold are MASKED, never silently included** (§18 step 4). |

## Stack combination (T-059 feeds from this)

- **Pair selection:** `small_baseline_pairs` — every (master, secondary) combination whose
  temporal baseline is within `[min_baseline_days, max_baseline_days] = [6, 36]` days.
  Longer pairs decorrelate over mining surfaces; shorter pairs dominate the stack.
- **Per-date inversion:** the pair network is solved for per-date phases θ_k by
  least squares over the small-baseline graph (gauge θ_0 = 0, convention
  θ_j − θ_i = φ_ij) — NOT a per-date average of pair phases, which cancels
  signals with partners on both sides. Vectorised via one pseudo-inverse
  (`build_stack`).
- **Stack summary:** coherence-weighted mean wrapped phase per multi-look pixel,
  requiring ≥ `coherence_mask_threshold` per pair.
- **Synthetic-stack validation:** recovery RMS 0.017 rad; per-PS velocity at the
  imposed −12 mm/yr point source recovered at −11.6 mm/yr (far field −0.3); run
  record in `experiments/insar_run.json`.
- **Unwrapping: SKIPPED (MVP, documented honestly).** The stack mean is of *wrapped*
  phase; absolute-displacement anchoring is per-PS relative to the reference point.
  Consequence: per-date displacement is a *phase-derived proxy* (mod 2π), suitable for
  ranking/monitoring; full unwrapping (SNAP snaphu or equivalent) is the first
  post-MVP upgrade.
- **Atmosphere: NONE (MVP).** Atmospheric delay is the dominant unmodelled error;
  stacking suppresses it only partially. Seasonal/regional trends are assessed by
  inspecting the stack means, not removed.
- **PS selection:** amplitude dispersion σ_A/μ_A ≤ 0.25 AND stack coherence ≥ 0.6 AND
  ≥ 15 valid observations of 21 dates.
- **Reference point:** highest-coherence, lowest-variance PS with |rate| ≤ 3 mm/yr —
  i.e. ground assumed outside mapped subsidence zones.

## Validation of the chain (real scenes pending)

The complete chain is validated on **synthetic SLC stacks with an imposed, known
deformation field** (`scripts/run_insar_pipeline.py`, T-058/T-059): a moving point
source (Mogi-style) plus speckle and a flat-Earth ramp is injected; the pipeline
recovers the deformation at the PS points (recovery correlation reported in the run
record). This validates the *processing chain*; the geophysical validity on real
ground awaits T-057's credentialed download of the 21 real SLC scenes and is stated
honestly wherever results are shown.

## Reproducibility

- All geometry/processing parameters: `configs/insar.yaml` (single source of truth).
- Run record with per-stage diagnostics: `experiments/insar_run.json` (T-059).
- Inputs/outputs: `data/raw/sentinel1/` (SLC zips + `inventory_manifest.json` with
  SHA-256 checksums), `data/processed/insar/deformation_timeseries.parquet` (T-059).
