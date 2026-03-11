"""Tests for V7 Theta Engine — weekly Nifty iron condor strategy."""
import pytest
from unittest.mock import MagicMock, patch
from datetime import date, datetime, timedelta
from v7.theta_engine import ThetaEngine, CondorPosition, CondorLeg


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_deps():
    data_feed = MagicMock()
    data_feed.get_vix.return_value = 16.0
    data_feed.get_ltp_batch.return_value = {"NSE:NIFTY 50": 24350.0}
    data_feed.get_option_chain.return_value = _mock_option_chain()

    order_mgr = MagicMock()
    from v7.order_manager import OrderResult
    order_mgr.place_entry_order.return_value = OrderResult(
        order_id="theta_order_1", filled=True, fill_price=25.0, fill_quantity=75,
    )
    order_mgr.place_exit_order.return_value = OrderResult(
        order_id="theta_exit_1", filled=True, fill_price=12.0, fill_quantity=75,
    )

    state = MagicMock()
    state.load_theta_state.return_value = None
    state.save_theta_state = MagicMock()

    strike_selector = MagicMock()
    margin = MagicMock()
    margin.current_utilization_pct.return_value = 25.0
    margin.estimate_condor_margin.return_value = 70000.0

    return {
        "data_feed": data_feed,
        "order_mgr": order_mgr,
        "state": state,
        "strike_selector": strike_selector,
        "margin": margin,
    }


@pytest.fixture
def theta(mock_deps):
    return ThetaEngine(
        data_feed=mock_deps["data_feed"],
        order_mgr=mock_deps["order_mgr"],
        state_mgr=mock_deps["state"],
        strike_selector=mock_deps["strike_selector"],
        margin_tracker=mock_deps["margin"],
        capital=300_000,
    )


def _mock_option_chain():
    """Create a mock option chain with strikes around 24350."""
    chain = []
    for strike in range(23800, 24900, 50):
        entry = {"strikePrice": strike}
        # Delta decreases as we go further from ATM
        dist = abs(strike - 24350)
        delta = max(0.05, 0.50 - dist / 1000)
        if strike < 24350:
            entry["PE"] = {
                "lastTradedPrice": max(5, 200 - dist * 0.3),
                "delta": -delta,
                "openInterest": 50000,
                "volume": 10000,
                "bidPrice": max(4, 199 - dist * 0.3),
                "askPrice": max(6, 201 - dist * 0.3),
                "lotSize": 75,
            }
        if strike > 24350:
            entry["CE"] = {
                "lastTradedPrice": max(5, 200 - dist * 0.3),
                "delta": delta,
                "openInterest": 50000,
                "volume": 10000,
                "bidPrice": max(4, 199 - dist * 0.3),
                "askPrice": max(6, 201 - dist * 0.3),
                "lotSize": 75,
            }
        chain.append(entry)
    return chain


# ── Entry Conditions ──────────────────────────────────────────────────


def test_should_enter_vix_in_range(theta, mock_deps):
    """Entry allowed when VIX is 14-20."""
    mock_deps["data_feed"].get_vix.return_value = 16.0
    assert theta._should_enter_today(date(2026, 3, 9)) is True  # Monday


def test_should_not_enter_vix_too_low(theta, mock_deps):
    """Skip if VIX < 14 (premiums too low)."""
    mock_deps["data_feed"].get_vix.return_value = 12.0
    assert theta._should_enter_today(date(2026, 3, 9)) is False


def test_should_not_enter_vix_too_high(theta, mock_deps):
    """Skip if VIX > 20 (too volatile)."""
    mock_deps["data_feed"].get_vix.return_value = 22.0
    assert theta._should_enter_today(date(2026, 3, 9)) is False


def test_should_enter_only_mon_tue(theta, mock_deps):
    """Only enter on Monday or Tuesday."""
    mock_deps["data_feed"].get_vix.return_value = 16.0
    assert theta._should_enter_today(date(2026, 3, 9)) is True   # Monday
    assert theta._should_enter_today(date(2026, 3, 10)) is True  # Tuesday
    assert theta._should_enter_today(date(2026, 3, 11)) is False # Wednesday
    assert theta._should_enter_today(date(2026, 3, 12)) is False # Thursday


def test_should_not_enter_if_already_have_condor(theta, mock_deps):
    """Skip entry if a condor is already open."""
    theta._condor = CondorPosition(
        entry_date=date(2026, 3, 9),
        expiry_date=date(2026, 3, 12),
        short_ce=CondorLeg("NIFTY2531224700CE", 24700, 30.0, 75, "CE"),
        long_ce=CondorLeg("NIFTY2531224900CE", 24900, 10.0, 75, "CE"),
        short_pe=CondorLeg("NIFTY2531224000PE", 24000, 30.0, 75, "PE"),
        long_pe=CondorLeg("NIFTY2531223800PE", 23800, 10.0, 75, "PE"),
        net_credit=40.0,
    )
    assert theta._should_enter_today(date(2026, 3, 10)) is False


def test_should_not_enter_margin_exceeded(theta, mock_deps):
    """Skip if condor would exceed 40% margin budget."""
    mock_deps["margin"].current_utilization_pct.return_value = 55.0
    assert theta._should_enter_today(date(2026, 3, 9)) is False


# ── Risk Budget ───────────────────────────────────────────────────────


def test_condor_max_risk_within_budget(theta):
    """Max risk = (wing_width - credit) × lot_size ≤ 3% of capital."""
    # Wing width 200pts, credit 40pts → max loss = 160 × 75 = 12000
    # 3% of 300000 = 9000 → this would FAIL
    assert theta._is_within_risk_budget(
        wing_width=200, net_credit=40.0, lot_size=75
    ) is False

    # Wing width 200pts, credit 40pts, lot 50 → max loss = 160 × 50 = 8000 < 9000
    assert theta._is_within_risk_budget(
        wing_width=200, net_credit=40.0, lot_size=50
    ) is True


# ── Daily Management ─────────────────────────────────────────────────


def test_profit_target_50pct_close(theta, mock_deps):
    """Close condor when 50% of credit captured."""
    theta._condor = CondorPosition(
        entry_date=date(2026, 3, 9),
        expiry_date=date(2026, 3, 12),
        short_ce=CondorLeg("NIFTY2531224700CE", 24700, 30.0, 75, "CE"),
        long_ce=CondorLeg("NIFTY2531224900CE", 24900, 10.0, 75, "CE"),
        short_pe=CondorLeg("NIFTY2531224000PE", 24000, 30.0, 75, "PE"),
        long_pe=CondorLeg("NIFTY2531223800PE", 23800, 10.0, 75, "PE"),
        net_credit=40.0,
    )
    # Current value of condor is 18 (less than 50% of 40 = 20 → profit > 50%)
    mock_deps["data_feed"].get_ltp_batch.return_value = {
        "NFO:NIFTY2531224700CE": 12.0,
        "NFO:NIFTY2531224900CE": 3.0,
        "NFO:NIFTY2531224000PE": 5.0,
        "NFO:NIFTY2531223800PE": 2.0,
    }
    action = theta._evaluate_condor_management()
    assert action == "close_profit"


def test_delta_035_tighten_hedge(theta, mock_deps):
    """When short strike delta > 0.35, tighten hedge."""
    theta._condor = CondorPosition(
        entry_date=date(2026, 3, 9),
        expiry_date=date(2026, 3, 12),
        short_ce=CondorLeg("NIFTY2531224700CE", 24700, 30.0, 75, "CE"),
        long_ce=CondorLeg("NIFTY2531224900CE", 24900, 10.0, 75, "CE"),
        short_pe=CondorLeg("NIFTY2531224000PE", 24000, 30.0, 75, "PE"),
        long_pe=CondorLeg("NIFTY2531223800PE", 23800, 10.0, 75, "PE"),
        net_credit=40.0,
    )
    theta._condor.short_ce_delta = 0.38  # above 0.35
    theta._condor.short_pe_delta = -0.15  # fine
    action = theta._evaluate_delta_risk()
    assert action == "tighten_ce"


def test_delta_045_close_side(theta, mock_deps):
    """When short strike delta > 0.45, close the threatened side."""
    theta._condor = CondorPosition(
        entry_date=date(2026, 3, 9),
        expiry_date=date(2026, 3, 12),
        short_ce=CondorLeg("NIFTY2531224700CE", 24700, 30.0, 75, "CE"),
        long_ce=CondorLeg("NIFTY2531224900CE", 24900, 10.0, 75, "CE"),
        short_pe=CondorLeg("NIFTY2531224000PE", 24000, 30.0, 75, "PE"),
        long_pe=CondorLeg("NIFTY2531223800PE", 23800, 10.0, 75, "PE"),
        net_credit=40.0,
    )
    theta._condor.short_pe_delta = -0.47
    theta._condor.short_ce_delta = 0.12
    action = theta._evaluate_delta_risk()
    assert action == "close_pe_side"


def test_delta_050_close_all(theta, mock_deps):
    """When short strike delta > 0.50, close entire condor."""
    theta._condor = CondorPosition(
        entry_date=date(2026, 3, 9),
        expiry_date=date(2026, 3, 12),
        short_ce=CondorLeg("NIFTY2531224700CE", 24700, 30.0, 75, "CE"),
        long_ce=CondorLeg("NIFTY2531224900CE", 24900, 10.0, 75, "CE"),
        short_pe=CondorLeg("NIFTY2531224000PE", 24000, 30.0, 75, "PE"),
        long_pe=CondorLeg("NIFTY2531223800PE", 23800, 10.0, 75, "PE"),
        net_credit=40.0,
    )
    theta._condor.short_ce_delta = 0.52
    action = theta._evaluate_delta_risk()
    assert action == "close_all"


# ── Time Management ──────────────────────────────────────────────────


def test_close_by_wednesday_eod(theta, mock_deps):
    """Close by Wednesday EOD if profit < 50%."""
    theta._condor = CondorPosition(
        entry_date=date(2026, 3, 9),
        expiry_date=date(2026, 3, 12),
        short_ce=CondorLeg("NIFTY2531224700CE", 24700, 30.0, 75, "CE"),
        long_ce=CondorLeg("NIFTY2531224900CE", 24900, 10.0, 75, "CE"),
        short_pe=CondorLeg("NIFTY2531224000PE", 24000, 30.0, 75, "PE"),
        long_pe=CondorLeg("NIFTY2531223800PE", 23800, 10.0, 75, "PE"),
        net_credit=40.0,
    )
    # Wednesday, profit only 30%
    assert theta._should_close_for_time(date(2026, 3, 11)) is True  # Wednesday


def test_never_hold_to_thursday(theta, mock_deps):
    """Never hold condor to Thursday expiry."""
    theta._condor = CondorPosition(
        entry_date=date(2026, 3, 9),
        expiry_date=date(2026, 3, 12),
        short_ce=CondorLeg("NIFTY2531224700CE", 24700, 30.0, 75, "CE"),
        long_ce=CondorLeg("NIFTY2531224900CE", 24900, 10.0, 75, "CE"),
        short_pe=CondorLeg("NIFTY2531224000PE", 24000, 30.0, 75, "PE"),
        long_pe=CondorLeg("NIFTY2531223800PE", 23800, 10.0, 75, "PE"),
        net_credit=40.0,
    )
    assert theta._should_close_for_time(date(2026, 3, 12)) is True  # Thursday


# ── Survival Mode ────────────────────────────────────────────────────


def test_survival_mode_wider_wings(theta):
    """In survival mode, use delta 0.15 instead of 0.20."""
    theta._survival_mode = True
    assert theta._entry_delta() == 0.15


def test_survival_mode_smaller_size(theta):
    """In survival mode, use 50% of normal lot count."""
    theta._survival_mode = True
    assert theta._lot_multiplier() == 0.5


def test_survival_mode_lower_profit_target(theta):
    """In survival mode, profit target is 40% (not 50%)."""
    theta._survival_mode = True
    assert theta._profit_target_pct() == 0.40
