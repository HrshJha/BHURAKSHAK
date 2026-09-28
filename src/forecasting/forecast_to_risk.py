"""Forecast → risk-layer bridge — PRD §16 (T-077).

§16's layered, explainable design: the temporal forecaster (T-076) predicts
PHYSICAL quantities — future displacement / tilt — and those forecasts enter
the XGBoost risk layer (§15, T-046) as FEATURES. The forecaster never emits a
risk state, label or probability (asserted in tests/test_temporal_model_fc.py);
the ONLY path from a forecast to "danger" runs through the risk model, whose
splits §22 can interrogate like any other feature.

Join mechanics (leak-free by construction):

- :func:`forecast_all_origins` runs the fitted forecaster at EVERY history
  position of every series — not just the last one like
  :meth:`TemporalForecaster.predict` — so each origin window carries the
  forecast a real deployment would have had at that instant. The forecast
  input is strictly windows ≤ origin: no future information can enter the
  feature, which is what makes it a legal causal feature for the risk model.
- :func:`forecasts_to_feature_frame` pivots forecast rows into one column per
  (channel, horizon): ``forecast_{channel}_h{steps}`` (§13-style names).
- :func:`join_forecast_features` left-joins that frame onto a window-level
  model frame on (event_id, node_id, window_index).
- :func:`assert_no_forecast_leakage` refuses frames whose forecast columns
  contain NaN: for risk-model TRAINING the feature must be fully observed;
  series tails (whose forecast target lies beyond the data) are dropped by
  the caller, never imputed.

Alignment: a forecast from origin window ``o`` (the last window of its
history) with horizon ``h`` targets window ``o + h`` — the same convention
``build_sequences`` uses (target = ``series[origin + h]``), so training-time
features and §16 evaluation (T-079) share identical semantics.

The resolver in ``src/risk/xgboost_model.py`` picks these columns up as the
``I_forecast`` input group whenever they are present, so joining the bridge
output onto a model frame is sufficient for the risk layer to use them.

Pipeline discipline: when forecast features feed the risk model's held-out
evaluation, the forecaster itself must be trained §23-safe (train split only).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.forecasting.temporal_model import ForecastError, TemporalForecaster

__all__ = [
    "LeakageError",
    "FORECAST_FEATURE_PREFIX",
    "is_forecast_feature",
    "forecast_feature_names",
    "forecast_all_origins",
    "forecasts_to_feature_frame",
    "join_forecast_features",
    "assert_no_forecast_leakage",
]

#: §13-style name prefix shared by every forecast feature column.
FORECAST_FEATURE_PREFIX = "forecast_"

#: ``forecast_{channel}_h{steps}`` — channel names contain no underscores-only
#: tokens that could collide, and the ``_h{n}`` suffix is fixed.
_FORECAST_COLUMN_RE = re.compile(r"^forecast_.+_h\d+$")


class LeakageError(RuntimeError):
    """Raised when a forecast feature frame would leak (or impute) at train time."""


def is_forecast_feature(name: str) -> bool:
    """True if ``name`` is a ``forecast_{channel}_h{steps}`` feature column."""
    return bool(_FORECAST_COLUMN_RE.fullmatch(str(name)))


def forecast_feature_names(channels: tuple[str, ...], horizons: tuple[int, ...]) -> list[str]:
    """The deterministic forecast feature column names for a config (T-078)."""
    return [f"{FORECAST_FEATURE_PREFIX}{ch}_h{int(h)}" for ch in channels for h in horizons]


def forecast_all_origins(fc: TemporalForecaster, windowed: pd.DataFrame) -> pd.DataFrame:
    """Forecast every horizon from EVERY valid history position of every series.

    Origin ``o`` (a ``window_index``) is valid for the series when it has a
    full ``fc.history_steps`` window history behind it; each horizon ``h`` is
    emitted only when its target window ``o + h`` exists. Rows carry
    ``window_index`` (the origin), ``target_window_index`` (= origin + horizon)
    and the physical-unit ``predicted_value`` — the same columns as
    :meth:`TemporalForecaster.predict` plus the origin/target keys, and never
    any risk-like column (the
    :class:`~src.forecasting.temporal_model.Forecast` guard still applies
    upstream; this frame is a pure physical-quantity table).

    Series shorter than the history, or origins with non-finite history
    values, yield no rows — never an invented forecast.
    """
    import torch  # lazy (T-076 discipline); threading caps already applied

    fc.torch_module.eval()
    rows: list[dict] = []
    with torch.no_grad():
        for (event, node), g in windowed.groupby(["event_id", "node_id"], sort=False):
            g = g.sort_values("window_index", kind="stable")
            series = np.column_stack(
                [g[f"{ch}_mean"].to_numpy(dtype=float) for ch in fc.channels]
            )
            ts = g["window_timestamp"].to_numpy()
            n = len(series)
            if n < fc.history_steps:
                continue
            origins = list(range(fc.history_steps - 1, n))
            if not origins:
                continue
            batch = np.stack(
                [series[o - fc.history_steps + 1 : o + 1] for o in origins]
            )
            finite = np.isfinite(batch).all(axis=(1, 2))
            if not finite.any():
                continue
            batch, origins = batch[finite], [o for o, ok in zip(origins, finite) if ok]
            xt = (batch - fc.mu) / fc.sd
            out = fc.torch_module(torch.tensor(xt, dtype=torch.float32)).numpy()  # (n_orig, H, C)
            out = out * fc.sd[None, None, :] + fc.mu[None, None, :]  # physical units
            for i, o in enumerate(origins):
                for h_i, h in enumerate(fc.horizons):
                    target = o + h
                    if target > n - 1:
                        continue  # horizon beyond the series — no row, no imputation
                    for c_i, ch in enumerate(fc.channels):
                        rows.append(
                            {
                                "event_id": event,
                                "node_id": node,
                                "window_index": int(o),
                                "window_timestamp": ts[o],
                                "target_window_index": int(target),
                                "channel": ch,
                                "horizon_steps": int(h),
                                "predicted_value": float(out[i, h_i, c_i]),
                            }
                        )
    return pd.DataFrame(
        rows,
        columns=[
            "event_id",
            "node_id",
            "window_index",
            "window_timestamp",
            "target_window_index",
            "channel",
            "horizon_steps",
            "predicted_value",
        ],
    )


def forecasts_to_feature_frame(forecast_rows: pd.DataFrame) -> pd.DataFrame:
    """Pivot forecast rows into one feature column per (channel, horizon).

    Output: one row per (event_id, node_id, window_index) origin with columns
    ``forecast_{channel}_h{steps}`` holding the physical-unit prediction.
    """
    keys = ["event_id", "node_id", "window_index"]
    if forecast_rows.empty:
        return pd.DataFrame(columns=keys)
    missing = {"event_id", "node_id", "window_index", "channel", "horizon_steps", "predicted_value"} - set(
        forecast_rows.columns
    )
    if missing:
        raise ForecastError(f"forecast rows missing columns: {sorted(missing)}")
    pivoted = forecast_rows.pivot_table(
        index=keys,
        columns=["channel", "horizon_steps"],
        values="predicted_value",
        aggfunc="first",
    ).reset_index()
    pivoted.columns = keys + [
        f"{FORECAST_FEATURE_PREFIX}{ch}_h{int(h)}" for ch, h in pivoted.columns[len(keys) :]
    ]
    return pivoted


def join_forecast_features(target: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    """Left-join the pivoted forecast features onto a window-level model frame.

    Rows whose series tail lies inside a horizon keep NaN in the forecast
    columns — real deployments cannot know those values either. Call
    :func:`assert_no_forecast_leakage` before training on the result.
    """
    features = forecasts_to_feature_frame(forecasts)
    if features.empty:
        return target
    merged = target.merge(features, on=["event_id", "node_id", "window_index"], how="left")
    return merged


def forecast_columns_of(df: pd.DataFrame) -> list[str]:
    """The forecast feature columns present in ``df``, deterministic order."""
    return [c for c in df.columns if is_forecast_feature(c)]


def assert_no_forecast_leakage(df: pd.DataFrame) -> None:
    """Refuse a training frame whose forecast features are not fully observed.

    NaN in a forecast column means "the series ended inside that horizon" —
    a value no deployment could have had. Training on imputed versions would
    fabricate foreknowledge; the caller must drop those rows instead.
    """
    cols = forecast_columns_of(df)
    if not cols:
        raise LeakageError("no forecast feature columns present — nothing to assert")
    n_nan = int(df[cols].isna().to_numpy().sum())
    if n_nan:
        raise LeakageError(
            f"forecast features contain {n_nan} NaN values across {len(cols)} columns — "
            "drop series-tail rows (never impute) before training the risk layer"
        )
