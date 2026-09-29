"""Train and apply a temporal model for physical measurements, including subsidence-related displacement."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Threading discipline (root-caused 2026-09-28): when sklearn/scipy are already
# loaded in the same process (full test suite), the OpenMP runtime is
# initialised before torch imports its own, and torch's intra-op parallelism
# oversubscribes the cores inside Adam's per-parameter update loop — training
# appears to hang. Capping the thread pools before the first torch import makes
# torch single-threaded, which costs nothing at these model sizes and removes
# the interaction entirely.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

__all__ = [
    "ForecastError",
    "FORECASTABLE_CHANNELS",
    "Forecast",
    "build_sequences",
    "TemporalForecaster",
    "train_temporal_forecaster",
]

#: The physical channels forecasting may predict (subset of the /
#: movement channels; a forecast is a physical quantity, never a label).
FORECASTABLE_CHANNELS = ("displacement", "tilt_x", "tilt_y")

_ALLOWED_ARCHITECTURES = ("tcn", "gru", "lstm")


class ForecastError(ValueError):
    """Raised on invalid forecasting inputs, config or architecture requests."""


@dataclass
class Forecast:
    """A physical-quantity forecast — explicitly NOT a risk state."""

    rows: pd.DataFrame  # columns: channel, horizon_steps, predicted_value
    channel_units: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        bad = {"risk", "probability", "label", "alert"} & {
            c.lower() for c in self.rows.columns
        }
        if bad:
            raise ForecastError(
                f"Forecast carries risk-like columns {sorted(bad)} —  forbids "
                "the forecaster from emitting anything but physical quantities"
            )

    def values(self, channel: str, horizon_steps: int) -> np.ndarray:
        sel = self.rows[(self.rows["channel"] == channel) & (self.rows["horizon_steps"] == horizon_steps)]
        if sel.empty:
            raise ForecastError(f"no forecast for channel {channel!r} at horizon {horizon_steps}")
        return sel["predicted_value"].to_numpy(dtype=float)


def build_sequences(
    windowed: pd.DataFrame,
    channels: tuple[str, ...],
    *,
    history_steps: int,
    horizon_steps: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    """Sliding (history → target) sequences per (event, node) series.

 Inputs are the windowed frame's ``{ch}_mean`` columns. X: (n, history,
 n_channels); Y: (n, n_channels) — the channel value ``horizon_steps``
 windows ahead. Series shorter than history + horizon produce nothing.
 Returns also the (event_id, node_id) key of every sample so predictions
 can be attributed.
 """
    for ch in channels:
        if f"{ch}_mean" not in windowed.columns:
            raise ForecastError(f"windowed frame missing {ch}_mean for forecasting")
    if history_steps < 2:
        raise ForecastError("history_steps must be >= 2")
    if horizon_steps < 1:
        raise ForecastError("horizon_steps must be >= 1")

    xs, ys, keys = [], [], []
    for (event, node), g in windowed.groupby(["event_id", "node_id"], sort=False):
        g = g.sort_values("window_index", kind="stable")
        series = np.column_stack(
            [g[f"{ch}_mean"].to_numpy(dtype=float) for ch in channels]
        )
        n = len(series)
        for start in range(0, n - history_steps - horizon_steps + 1):
            x = series[start : start + history_steps]
            y = series[start + history_steps - 1 + horizon_steps]
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                continue  # a series with gaps just yields fewer sequences
            xs.append(x)
            ys.append(y)
            keys.append((str(event), str(node)))
    if not xs:
        return (
            np.zeros((0, history_steps, len(channels))),
            np.zeros((0, len(channels))),
            [],
        )
    return np.stack(xs), np.stack(ys), keys


def _build_multi_horizon(
    windowed: pd.DataFrame,
    channels: tuple[str, ...],
    *,
    history_steps: int,
    horizons: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    """Sliding sequences with ALL horizons aligned by index.

 X: (n, history, C); Y: (n, H, C) where Y[s, h_i] is the channel value
 ``horizons[h_i]`` windows after the history window ends. Sequences whose
 targets run past a series' end (or hit a NaN) are dropped — per-horizon
 counts therefore stay aligned by construction (no truncation hacks).
 """
    for ch in channels:
        if f"{ch}_mean" not in windowed.columns:
            raise ForecastError(f"windowed frame missing {ch}_mean for forecasting")
    xs, ys, keys = [], [], []
    max_h = max(horizons)
    for (event, node), g in windowed.groupby(["event_id", "node_id"], sort=False):
        g = g.sort_values("window_index", kind="stable")
        series = np.column_stack([g[f"{ch}_mean"].to_numpy(dtype=float) for ch in channels])
        n = len(series)
        for start in range(0, n - history_steps - max_h + 1):
            x = series[start : start + history_steps]
            try:
                y = np.stack([series[start + history_steps - 1 + h] for h in horizons], axis=0)
            except IndexError:  # pragma: no cover — range already bounds this
                continue
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                continue
            xs.append(x)
            ys.append(y)
            keys.append((str(event), str(node)))
    if not xs:
        return (
            np.zeros((0, history_steps, len(channels))),
            np.zeros((0, len(horizons), len(channels))),
            [],
        )
    return np.stack(xs), np.stack(ys), keys


def _scale_fit(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel standardisation from TRAIN data only."""
    flat = train.reshape(-1, train.shape[-1])
    mu = np.nanmean(flat, axis=0)
    sd = np.nanstd(flat, axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return mu, sd


def _build_torch(arch: str, n_channels: int, width: int, depth: int):
    import torch.nn as nn

    if arch == "lstm":
        return nn.LSTM(input_size=n_channels, hidden_size=width, num_layers=depth, batch_first=True)
    if arch == "gru":
        return nn.GRU(input_size=n_channels, hidden_size=width, num_layers=depth, batch_first=True)
    # TCN: causal-style dilated conv stack (dilation doubles per layer,
    # kernel 3, padding = dilation keeps the length; last step sees the past)
    layers: list[nn.Module] = []
    in_ch = n_channels
    for d in range(depth):
        dilation = 2**d
        layers += [
            nn.Conv1d(in_ch, width, kernel_size=3, padding=dilation, dilation=dilation),
            nn.ReLU(),
        ]
        in_ch = width
    return nn.Sequential(*layers)


@dataclass
class TemporalForecaster:
    """Fitted physical-quantity forecaster (torch backend, any of TCN/GRU/LSTM)."""

    architecture: str
    channels: tuple[str, ...]
    history_steps: int
    horizons: tuple[int, ...]
    mu: np.ndarray
    sd: np.ndarray
    torch_module: object = None
    train_loss: float = float("nan")
    val_loss: float = float("nan")

    def predict(self, windowed: pd.DataFrame) -> Forecast:
        """Forecast every horizon for every series in ``windowed``.

 Each series' LAST complete history window is the input; outputs are
 in the channel's physical unit (unscaled). Series shorter than the
 history produce no rows — never an invented forecast.

 Single source of truth: delegates to:func:`src.forecasting.forecast_to_risk.forecast_all_origins` and keeps
 each series' LAST origin (imported lazily to avoid a module-level
 cycle; forecast_to_risk imports this module's types).
 """
        if self.torch_module is None:
            raise ForecastError("forecaster is not fitted")
        from src.forecasting.forecast_to_risk import forecast_all_origins

        all_rows = forecast_all_origins(self, windowed)
        if all_rows.empty:
            return Forecast(
                rows=pd.DataFrame(columns=["event_id", "node_id", "channel", "horizon_steps", "predicted_value"]),
                channel_units={ch: "mm" if ch == "displacement" else "deg" for ch in self.channels},
            )
        # the last origin per (event, node): max window_index that emitted rows
        last = all_rows.sort_values("window_index", kind="stable").groupby(
            ["event_id", "node_id"], sort=False
        ).tail(len(self.horizons) * len(self.channels))
        last = last[["event_id", "node_id", "channel", "horizon_steps", "predicted_value"]]
        return Forecast(
            rows=last.reset_index(drop=True),
            channel_units={ch: "mm" if ch == "displacement" else "deg" for ch in self.channels},
        )


def train_temporal_forecaster(
    windowed: pd.DataFrame,
    *,
    split: pd.Series | None = None,
    channels: tuple[str, ...] = FORECASTABLE_CHANNELS,
    horizons: tuple[int, ...] = (1,),
    architecture: str = "lstm",
    history_steps: int = 8,
    width: int = 32,
    depth: int = 1,
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    dropout: float = 0.0,
    weight_decay: float = 0.0,
    seed: int = 42,
) -> TemporalForecaster:
    """Fit a TCN/GRU/LSTM forecaster for PHYSICAL channel values.

 ``split``: optional split labels aligned with ``windowed`` (train /
 validation). Without it, all sequences are train (pure-fitting use).
 Early stopping keeps the epoch with the best validation loss; without a
 validation split the final epoch is kept. The LSTM default is the
 literature benchmark.
 """
    import torch
    import torch.nn as nn

    # Belt-and-braces if another module imported torch before this one (the
    # env caps above only apply before the OpenMP runtime is first loaded).
    torch.set_num_threads(1)

    if architecture not in _ALLOWED_ARCHITECTURES:
        raise ForecastError(
            f"architecture must be one of {_ALLOWED_ARCHITECTURES}, got {architecture!r}"
        )
    if split is not None and len(split) != len(windowed):
        raise ForecastError("split must align with windowed")

    if not set(channels) <= set(FORECASTABLE_CHANNELS):
        banned = set(channels) - set(FORECASTABLE_CHANNELS)
        raise ForecastError(
            f"channels {sorted(banned)} are not forecastable physical channels — "
            ": the forecaster predicts displacement/tilt only, never a risk quantity"
        )

    torch.manual_seed(seed)
    windowed = windowed.reset_index(drop=True)
    split_vals = split.to_numpy() if split is not None else None

    # sequences are built per band so train statistics never see val rows
    mask_train = np.ones(len(windowed), dtype=bool) if split_vals is None else (split_vals == "train")
    mask_val = np.zeros(len(windowed), dtype=bool) if split_vals is None else (split_vals == "validation")

    x_tr, y_tr, _ = _build_multi_horizon(
        windowed[mask_train], channels, history_steps=history_steps, horizons=horizons
    )
    x_va, y_va, _ = _build_multi_horizon(
        windowed[mask_val], channels, history_steps=history_steps, horizons=horizons
    )
    if len(x_tr) == 0:
        raise ForecastError("no trainable sequences in the train split")
    mu, sd = _scale_fit(x_tr)

    x_t = torch.tensor((x_tr - mu) / sd, dtype=torch.float32)
    y_t = torch.tensor((y_tr - mu) / sd, dtype=torch.float32)  # (n, H, C)
    n_ch = len(channels)

    class ForecasterModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.architecture = architecture
            self.backbone = _build_torch(architecture, n_ch, width, depth)
            self.head = nn.Linear(width if architecture != "tcn" else width, len(horizons) * n_ch)
            self.dropout = nn.Dropout(float(dropout))
            if architecture == "tcn":
                self.tcn_proj = nn.Linear(width, width)  # (N, width, T) → last-step head

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # (N, T, C)
            if architecture == "tcn":
                z = self.backbone(x.transpose(1, 2))  # (N, width, T)
                last = z[:, :, -1]  # causal: last step sees only the past
            else:
                out, _ = self.backbone(x)
                last = out[:, -1, :]
            return self.head(self.dropout(last)).reshape(-1, len(horizons), n_ch)  # (N, H, C)

    module = ForecasterModule()
    opt = torch.optim.Adam(module.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    x_va_t = torch.tensor((x_va - mu) / sd, dtype=torch.float32) if len(x_va) else None
    y_va_t = torch.tensor((y_va - mu) / sd, dtype=torch.float32) if len(x_va) else None

    best_val, best_state, final_train = float("inf"), None, float("nan")
    generator = torch.Generator().manual_seed(seed)
    for _epoch in range(epochs):
        module.train()
        perm = torch.randperm(len(x_t), generator=generator)
        losses = []
        for i in range(0, len(x_t), batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            pred = module(x_t[idx])
            loss = loss_fn(pred, y_t[idx])
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        final_train = float(np.mean(losses)) if losses else float("nan")
        if x_va_t is not None and len(x_va_t):
            module.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(module(x_va_t), y_va_t))
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in module.state_dict().items()}

    if best_state is not None:
        module.load_state_dict(best_state)

    return TemporalForecaster(
        architecture=architecture,
        channels=tuple(channels),
        history_steps=history_steps,
        horizons=tuple(int(h) for h in horizons),
        mu=mu,
        sd=sd,
        torch_module=module,
        train_loss=final_train,
        val_loss=best_val if best_val != float("inf") else float("nan"),
    )
