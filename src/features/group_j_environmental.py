"""Add gated environmental features from available inputs."""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from src.config import environmental_config

__all__ = [
    "GROUP_J_FEATURES",
    "GroupJError",
    "emit_group_j",
    "gate_status",
    "model_feature_names",
    "assert_gate_compliant",
]

GROUP_J_FEATURES = (
    "rainfall",
    "temperature",
    "land_surface_temperature",
    "land_cover",
)

_NOT_IN_MODEL = "not_in_model"
_GATE_KEY = "gate_status"


class GroupJError(ValueError):
    """Raised on invalid Group J inputs or gate violations."""


def _synthetic_params() -> dict:
    cfg = environmental_config()
    synthetic = cfg.get("synthetic")
    if not isinstance(synthetic, dict):
        raise GroupJError("configs/environmental.yaml missing the `synthetic` block")
    return synthetic


def emit_group_j(
    windowed: pd.DataFrame,
    *,
    timestamps_hours: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Emit the four Group J channels, 1:1 with ``windowed`` rows.

 Synthetic stand-ins seeded from config: per-event Poisson rainfall
 (shared across an event's nodes), diurnal air temperature, LST offset
 from air, static per-node land-cover index (distance bands — an index,
 never a classification claim). Output carries
 ``gate_status = "not_in_model"``.
 """
    for col in ("event_id", "node_id", "window_index"):
        if col not in windowed.columns:
            raise GroupJError(f"windowed frame missing {col!r}")

    sp = _synthetic_params()
    seed = int(sp["seed"])
    if timestamps_hours is not None:
        hours = np.asarray(timestamps_hours, dtype=float)
        if len(hours) != len(windowed):
            raise GroupJError("timestamps_hours must align with windowed rows")
    elif "window_timestamp" in windowed.columns:
        hours = pd.to_numeric(windowed["window_timestamp"], errors="coerce").to_numpy(dtype=float)
    else:
        hours = np.zeros(len(windowed), dtype=float)

    cfg = environmental_config()
    nodes = windowed["node_id"].astype(str)

    # --- land cover: STATIC per node, from mesh geometry (deterministic bands)
    node_ids = nodes.drop_duplicates().to_list()
    cover = {}
    for nid in node_ids:
        # Jharia-context banding by node-id parity blocks is arbitrary; use a
        # deterministic hash-free rule instead: node index mod 3 (built-up,
        # vegetation, water) — synthetic provenance declared in config anyway.
        digits = "".join(ch for ch in nid if ch.isdigit())
        idx = int(digits) % 3 if digits else 0
        cover[nid] = idx

    # --- rainfall: one regional storm process per event (shared by nodes)
    rain = np.zeros(len(windowed), dtype=float)
    for event, positions in windowed.groupby("event_id", sort=False).indices.items():
        # CRC32 (NOT hash): string hash is salted per process — the storm
        # seed must be reproducible across runs.
        ev_rng = np.random.default_rng(seed + (zlib.crc32(str(event).encode("utf-8")) % 10000))
        ev_hours = hours[positions]
        span = float(max(np.nanmax(ev_hours) - np.nanmin(ev_hours), 1e-9)) if len(ev_hours) else 1.0
        n_storms = ev_rng.poisson(float(sp["storm_rate_per_day"]) * max(span / 24.0, 1e-9))
        starts = ev_rng.uniform(np.nanmin(ev_hours), np.nanmax(ev_hours) + 1e-9, size=n_storms)
        durations = ev_rng.uniform(1.0, 3.0, size=n_storms)  # hours
        intensities = ev_rng.exponential(float(sp["storm_mean_intensity_mm"]), size=n_storms)
        for s, d, inten in zip(starts, durations, intensities, strict=True):
            wet = (ev_hours >= s) & (ev_hours <= s + d)
            rain[positions[wet]] += inten

    # --- temperature / LST: diurnal sinusoid on the hours axis
    mean_t = float(sp["temp_mean_c"])
    amp_t = float(sp["temp_amplitude_c"])
    peak = float(sp["temp_peak_hour"])
    temp = mean_t + amp_t * np.cos(2.0 * np.pi * (hours - peak) / 24.0)
    lst = temp + float(sp["lst_offset_c"]) + float(sp["lst_damping"]) * amp_t * np.cos(
        2.0 * np.pi * (hours - peak - 1.0) / 24.0
    )

    out = pd.DataFrame(
        {
            "event_id": windowed["event_id"].to_numpy(),
            "node_id": windowed["node_id"].to_numpy(),
            "window_index": windowed["window_index"].to_numpy(),
            "rainfall": rain,
            "temperature": temp,
            "land_surface_temperature": lst,
            "land_cover": nodes.map(cover).to_numpy(),
            "source": [str(cfg["sources"]["rainfall"])] * len(windowed),
        }
    )
    out.attrs[_GATE_KEY] = _NOT_IN_MODEL
    out.attrs["provenance"] = dict(cfg["sources"])
    return out


def gate_status() -> str:
    """The configured gate state: ``'included'`` or ``'excluded'`` (verbatim)."""
    cfg = environmental_config()
    gate = cfg.get("gate", {})
    status = str(gate.get("status", "not_run"))
    if status == "included":
        return "included"
    return "excluded"


def model_feature_names(features=GROUP_J_FEATURES) -> tuple[str, ...]:
    """Group J names IF AND ONLY IF the gate allows them into the model.

 Returns the (deduplicated) Group J feature names when the gate passes,
 an EMPTY tuple when it does not — callers can splice the result into a
 model feature list without branching. Raises only on an incoherent
 configuration (enabled=true but gate missing its evidence fields, or a
 delta below the configured inclusion threshold while claiming inclusion).
 """
    cfg = environmental_config()
    gate = cfg.get("gate", {})
    status = str(gate.get("status", "not_run"))

    if status != "included":
        return ()
    if not bool(cfg.get("enabled", False)):
        raise GroupJError(
            "environmental gate says 'included' but `enabled` is false — flip both, "
            "the master flag alone must never add Group J to the model"
        )
    for field in ("ablation_id", "metric", "delta", "decided_by", "decided_on"):
        if gate.get(field) is None:
            raise GroupJError(
                f"environmental gate incomplete: '{field}' must record the ablation "
                "evidence justifying Group J inclusion (/)"
            )
    delta = float(gate["delta"])
    if delta < float(gate["min_pr_auc_delta"]):
        raise GroupJError(
            f"environmental gate recorded delta {delta} below the inclusion "
            f"threshold {gate['min_pr_auc_delta']} — inclusion is not justified by its own record"
        )
    return tuple(dict.fromkeys(str(f) for f in features))


def assert_gate_compliant(df: pd.DataFrame) -> None:
    """Guard: a frame stamped ``not_in_model`` must never reach a model."""
    status = df.attrs.get(_GATE_KEY)
    if status == _NOT_IN_MODEL:
        raise GroupJError(
            "frame is stamped gate_status='not_in_model' ( gate) — it cannot "
            "be used as a model feature frame; pass the gate first"
        )
    if status is None:
        raise GroupJError("frame carries no gate_status marker — Group J frames must state their gate status")
