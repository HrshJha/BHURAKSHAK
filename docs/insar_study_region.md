# T-056 — Sentinel-1 Study Region & Orbit Track (PRD §18 step 1)

**Decision: ONE region, ONE geometry, fixed for the entire MVP.**

| Item | Value |
|---|---|
| Study region | Jharia coalfield (East Jharia), Dhanbad district, Jharkhand, India |
| Centre / mesh origin | **23.75° N, 86.45° E** (identical to the `crs` block of `configs/physics.yaml`) |
| Bounding box | lon 86.38–86.52° E, lat 23.65–23.85° N (~14 × 22 km) |
| Track (relative orbit) | **121 — DESCENDING** |
| Product | Sentinel-1 IW SLC (`IW_SLC__1S`), VV+VH |
| Repeat cycle | 12 days per orbit (6-day interleaved pairs at this latitude) |
| Acquisition window | 2026-01-01 → 2026-10-01 → **21 acquisition dates** (≥ 20-scene §18 floor) |
| Catalogue verification | CDSE OData query, 2026-09-27: T121 DESC SLC slices cover the mesh origin on 19 dates Jan–Aug 2026 (5 on S1A through the June fleet handover, 12+ on S1D after) |

## Why this region

- The **mesh deployment origin is already Jharia** (23.75° N, 86.45° E, `configs/physics.yaml`) — satellite, DGPS and mesh features must describe the same ground (§9.2 dual-CRS alignment, FR-13).
- Jharia is one of the most InSAR-studied mining areas in the world: published Sentinel-1 PS/SBAS analyses report mining- and coal-fire-induced subsidence at roughly **cm/yr rates** with persistent scatterers surviving on built mining infrastructure — i.e. the technique demonstrably works on this exact terrain (Riyas et al. 2021, *Remote Sensing* 13:1521; Thakur et al. 2025; Karanam et al. 2021).

## Why track 121 descending

- **Catalogue-verified coverage**: the CDSE OData catalogue (queried 2026-09-27) confirms relative orbit 121 (DESCENDING) SLC slices intersect the mesh origin on every expected 12-day date; the complementary ascending track 85 also covers the region, but §18 mandates ONE consistent geometry for the MVP stack — mixing passes into one stack would break common-master interferometry.
- **Descending geometry** observes predominantly the east–west + vertical component of movement with a favourable incidence over the East Jharia basin; it is the pass used by the published Jharia time-series studies above, keeping our geometry comparable with reported results.
- The 2026 **S1A→S1D handover** (late June) is transparent to interferometry at MVP precision: same orbit tube, same 12-day repeat; the handover date is recorded in `configs/insar.yaml` so cross-handover pairs are flagged in the T-058 workflow.

## Configuration authority

All geometry lives in `configs/insar.yaml` (`study_region`, `acquisition`, `processing`, `outputs` blocks). Downstream tasks consume it: T-057 (search/download filters), T-058 (processing parameters), T-059/T-060/T-061 (outputs and thresholds). No code may hard-code the bounding box or track number.

## Honest scope notes (§35 / Honesty Statement)

1. **21 scenes is the minimum viable stack** — published Jharia PS analyses use longer stacks; with 21 dates the velocity estimates carry wider confidence intervals. The window was chosen to end at the present date.
2. **Atmospheric delay is not corrected** in the MVP workflow (`processing.atmosphere: none`) — it is the dominant unmodelled error and is assessed by stacking, not removed.
3. **Phase unwrapping is skipped** in the MVP (small-baseline stacking of filtered wrapped phase instead); absolute displacement is anchored per PS by the reference point.
4. If scene download (T-057) cannot proceed for credential/scale reasons, the honest fallback is documented there — the workflow code (T-058) is validated on a synthetic SLC pair with the same geometry and clearly labelled as such.
