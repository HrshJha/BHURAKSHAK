"""Packet loss & communication-failure modes —.

Two distinct comms-degradation modes, distinct from ``SENSOR_FAULT``:

 random packet loss → ``DATA_QUALITY`` (individual packets missing)
 total link outage → ``COMMUNICATION_FAILURE`` (a node/link goes silent)

Both operate as keep-masks over (node, timestep) readings; the label engine
 tags affected rows, and the generator drops lost packets so the
missing-data handling in preprocessing sees realistic gaps.
"""

from __future__ import annotations

import numpy as np

DATA_QUALITY_LABEL = "DATA_QUALITY"
COMMUNICATION_FAILURE_LABEL = "COMMUNICATION_FAILURE"


def random_packet_loss(
    n: int,
    loss_rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent packet loss.

 Returns ``(keep_mask, lost_mask)`` over ``n`` packets; lost packets are
 labelled ``DATA_QUALITY`` by the caller before being dropped.
 """
    if not 0.0 <= loss_rate < 1.0:
        raise ValueError("loss_rate must be in [0, 1)")
    lost = rng.random(n) < loss_rate
    return ~lost, lost


def link_outage(
    n: int,
    outage_start: int,
    outage_duration: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Total link loss for one node: everything in the window is lost.

 Returns ``(keep_mask, lost_mask)``; the outage window is labelled
 ``COMMUNICATION_FAILURE`` — distinct from both DATA_QUALITY and SENSOR_FAULT.
 """
    if outage_duration <= 0:
        raise ValueError("outage_duration must be > 0")
    if not 0 <= outage_start < n:
        raise ValueError("outage_start out of range")
    lost = np.zeros(n, dtype=bool)
    end = min(outage_start + outage_duration, n)
    lost[outage_start:end] = True
    return ~lost, lost


def labels_for_lost_packets(lost_mask: np.ndarray, mode: str) -> np.ndarray:
    """Label array over packets: lost packets carry the mode's label."""
    if mode == "packet_loss":
        label = DATA_QUALITY_LABEL
    elif mode == "outage":
        label = COMMUNICATION_FAILURE_LABEL
    else:
        raise ValueError("mode must be 'packet_loss' or 'outage'")
    labels = np.empty(lost_mask.size, dtype=object)
    labels[:] = ""
    labels[lost_mask] = label
    return labels
