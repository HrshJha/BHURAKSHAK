""" — forecast-horizon configuration tests.

Acceptance spine: all five horizons (next-window, 30 min, 1 h, 6 h, 24 h)
are selectable from config and none is hard-coded under ``src/``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from src.config import REPO_ROOT, forecasting_config
from src.forecasting.horizons import (
    HORIZON_ORDER,
    HorizonError,
    configured_horizon_minutes,
    grid_interval_minutes,
    horizons_in_steps,
)

SRC_DIR = REPO_ROOT / "src"
FORECASTING_YAML = REPO_ROOT / "configs" / "forecasting.yaml"


# config file + loader

def test_forecasting_yaml_exists_and_carries_all_five_horizons() -> None:
    assert FORECASTING_YAML.is_file()
    raw = yaml.safe_load(FORECASTING_YAML.read_text(encoding="utf-8"))
    horizons = raw["horizons"]
    assert set(HORIZON_ORDER) <= set(horizons)
    assert set(horizons) == set(HORIZON_ORDER), "no extra horizons beyond 's five"
    assert horizons["next_window"] == 0
    assert horizons["minutes_30"] == 30
    assert horizons["hours_1"] == 60
    assert horizons["hours_6"] == 360
    assert horizons["hours_24"] == 1440


def test_config_reachable_through_loader() -> None:
    cfg = forecasting_config()
    assert cfg["architecture"] == "lstm"  # benchmark stays the default
    assert isinstance(cfg["horizons"], dict)


# resolution to window-steps

def test_horizons_resolve_to_window_steps_on_default_grid() -> None:
    assert grid_interval_minutes() == 10.0  # default mesh cadence
    assert horizons_in_steps() == (1, 3, 6, 36, 144)


def test_horizons_resolve_on_a_different_grid() -> None:
    # a 30-minute grid: 30 min → 1 step, 60 → 2, 360 → 12, 1440 → 48
    assert horizons_in_steps(grid_minutes=30.0) == (1, 2, 12, 48)


def test_horizons_are_strictly_increasing() -> None:
    steps = horizons_in_steps()
    assert steps == tuple(sorted(set(steps)))
    assert len(steps) == 5
    assert all(s >= 1 for s in steps)


# misconfiguration fails loudly

def test_off_grid_horizon_raises() -> None:
    with pytest.raises(HorizonError, match="not a multiple"):
        horizons_in_steps(grid_minutes=45.0)  # 30 min does not land on 45-min grid


def test_missing_horizon_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.config as cfg_mod

    original_loader = cfg_mod.load_config
    broken = dict(forecasting_config()["horizons"])
    del broken["hours_6"]
    monkeypatch.setattr(
        cfg_mod,
        "load_config",
        lambda name: {"horizons": broken} if name == "forecasting" else original_loader(name),
    )
    with pytest.raises(HorizonError, match="hours_6"):
        configured_horizon_minutes()


def test_negative_horizon_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.config as cfg_mod

    original_loader = cfg_mod.load_config
    broken = dict(forecasting_config()["horizons"])
    broken["minutes_30"] = -5
    monkeypatch.setattr(
        cfg_mod,
        "load_config",
        lambda name: {"horizons": broken} if name == "forecasting" else original_loader(name),
    )
    with pytest.raises(HorizonError, match=">= 0"):
        configured_horizon_minutes()


#: horizons live in config, not in code

def test_no_hardcoded_horizon_literals_outside_config() -> None:
    """The horizons must be configuration-driven.

 The distinctive minute values (6 h = 360, 24 h = 1440) appear nowhere as
 bare literals under ``src/``. The colliding values (30/60 also name
 epochs and the window size) are guarded structurally instead: the
 trainer's horizon default stays the neutral next-window ``[1]`` and only
 the horizon resolver may read the forecasting config.
 """
    for value in (360, 1440):
        pattern = re.compile(rf"(?<![\w.]){value}(?![\w.])")
        for path in sorted(SRC_DIR.rglob("*.py")):
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                assert not pattern.search(line), (
                    f" horizon literal {value} (minutes) hard-coded at "
                    f"{path.relative_to(REPO_ROOT)}:{i}: {line.strip()} — "
                    "horizons must come from configs/forecasting.yaml ()"
                )


def test_trainer_horizon_default_is_neutral_and_config_only_in_resolver() -> None:
    import inspect

    from src.forecasting.temporal_model import train_temporal_forecaster

    default = inspect.signature(train_temporal_forecaster).parameters["horizons"].default
    assert default == (1,), "the trainer must not bake a  horizon table into its signature"

    # only the resolver reads the forecasting config
    users = [
        p
        for p in sorted((SRC_DIR / "forecasting").glob("*.py"))
        if "forecasting_config" in p.read_text(encoding="utf-8")
    ]
    assert users == [REPO_ROOT / "src" / "forecasting" / "horizons.py"]


def test_forecaster_defaults_follow_the_config() -> None:
    """The training entry point's model parameters mirror the YAML defaults."""
    import inspect

    from src.forecasting.temporal_model import train_temporal_forecaster

    cfg = forecasting_config()
    sig = inspect.signature(train_temporal_forecaster)
    assert sig.parameters["architecture"].default == cfg["architecture"]
    assert sig.parameters["history_steps"].default == int(cfg["history_steps"])
    assert sig.parameters["width"].default == int(cfg["width"])
    assert sig.parameters["depth"].default == int(cfg["depth"])
    assert sig.parameters["epochs"].default == int(cfg["epochs"])
    assert sig.parameters["batch_size"].default == int(cfg["batch_size"])
    assert float(sig.parameters["learning_rate"].default) == float(cfg["learning_rate"])
    assert sig.parameters["seed"].default == int(cfg["seed"])
