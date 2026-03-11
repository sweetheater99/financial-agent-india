"""Tests for margin estimation."""
import pytest
from v7.margin import MarginTracker


@pytest.fixture
def tracker():
    return MarginTracker(capital=300_000)


def test_initial_state(tracker):
    assert tracker.available_margin() == 300_000
    assert tracker.utilization_pct() == 0.0


def test_add_bought_option(tracker):
    tracker.add_position("NIFTY CE 24400", margin=7500)
    assert tracker.used_margin() == 7500
    assert tracker.available_margin() == 292_500


def test_can_add_within_limit(tracker):
    # 70% limit = 210_000
    assert tracker.can_add(200_000) is True
    assert tracker.can_add(220_000) is False


def test_remove_position(tracker):
    tracker.add_position("NIFTY CE", margin=7500)
    tracker.remove_position("NIFTY CE")
    assert tracker.used_margin() == 0


def test_theta_budget(tracker):
    # 40% of 300K = 120K for theta
    assert tracker.theta_budget() == 120_000


def test_directional_budget(tracker):
    # 60% of 300K = 180K for directional (minus buffer)
    # But actual available = capital - buffer(30%) = 210K, minus theta reservation
    assert tracker.directional_budget() > 0


def test_utilization_with_positions(tracker):
    tracker.add_position("pos1", margin=100_000)
    tracker.add_position("pos2", margin=50_000)
    assert tracker.utilization_pct() == pytest.approx(50.0, rel=0.01)


def test_estimate_option_buy_margin():
    tracker = MarginTracker(capital=300_000)
    m = tracker.estimate_option_buy_margin(premium=100, lot_size=75)
    assert m == 7500  # premium * lot_size
