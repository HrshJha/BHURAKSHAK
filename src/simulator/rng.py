"""Seeded reproducibility across the generator —.

Single point of RNG construction: every simulator component takes a
``numpy.random.Generator`` created here from the dataset's fixed random seed.
Two runs with the same seed must produce byte-identical output; different
seeds must not (asserted by tests/test_reproducibility.py on real generator
output, not just RNG equality).
"""

from __future__ import annotations

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """The one way any simulator code obtains randomness (PCG64 via default_rng)."""
    return np.random.default_rng(seed)
