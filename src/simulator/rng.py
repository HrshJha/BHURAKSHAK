"""Create seeded random number generators for simulation."""

from __future__ import annotations

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """The one way any simulator code obtains randomness (PCG64 via default_rng)."""
    return np.random.default_rng(seed)
