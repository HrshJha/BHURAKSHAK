"""T-017 acceptance tests — packet loss vs link outage, distinct from SENSOR_FAULT."""

from __future__ import annotations

import numpy as np

from src.simulator.comms_degradation import (
    COMMUNICATION_FAILURE_LABEL,
    DATA_QUALITY_LABEL,
    labels_for_lost_packets,
    link_outage,
    random_packet_loss,
)


def test_packet_loss_keeps_majority_at_low_rate() -> None:
    rng = np.random.default_rng(0)
    keep, lost = random_packet_loss(1000, 0.05, rng)
    assert keep.sum() + lost.sum() == 1000
    assert 0 < lost.sum() < 100, "5% loss on 1000 packets should lose a plausible number"
    assert (keep & lost).sum() == 0, "keep and lost masks must be disjoint"


def test_packet_loss_zero_rate_loses_nothing() -> None:
    rng = np.random.default_rng(0)
    keep, lost = random_packet_loss(100, 0.0, rng)
    assert lost.sum() == 0 and keep.all()


def test_packet_loss_invalid_rate_raises() -> None:
    rng = np.random.default_rng(0)
    try:
        random_packet_loss(10, 1.0, rng)
    except ValueError:
        return
    raise AssertionError("loss_rate=1.0 must raise")


def test_outage_loses_exactly_the_window() -> None:
    keep, lost = link_outage(100, outage_start=40, outage_duration=20)
    assert lost.sum() == 20
    assert not lost[:40].any() and not lost[60:].any()
    assert keep.sum() == 80


def test_outage_clamps_at_end() -> None:
    _keep, lost = link_outage(100, outage_start=90, outage_duration=50)
    assert lost.sum() == 10, "outage window must clamp at series end"


def test_outage_invalid_start_raises() -> None:
    try:
        link_outage(10, outage_start=10, outage_duration=2)
    except ValueError:
        return
    raise AssertionError("outage_start == n must raise")


def test_labels_distinguish_data_quality_from_comm_failure() -> None:
    lost_pl = np.array([False, True, False])
    lost_of = np.array([True, False, False])
    pl = labels_for_lost_packets(lost_pl, "packet_loss")
    of = labels_for_lost_packets(lost_of, "outage")
    assert pl[1] == DATA_QUALITY_LABEL and pl[0] == "" and pl[2] == ""
    assert of[0] == COMMUNICATION_FAILURE_LABEL and of[1] == "" and of[2] == ""
    assert DATA_QUALITY_LABEL != COMMUNICATION_FAILURE_LABEL
    # both are distinct from SENSOR_FAULT (T-016)
    from src.simulator.faults import FAULT_LABEL

    assert DATA_QUALITY_LABEL != FAULT_LABEL
    assert COMMUNICATION_FAILURE_LABEL != FAULT_LABEL


def test_labels_invalid_mode_raises() -> None:
    try:
        labels_for_lost_packets(np.array([True]), "bogus")
    except ValueError:
        return
    raise AssertionError("unknown mode must raise")
