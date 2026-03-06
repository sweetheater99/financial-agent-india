"""Tests for weekly theta harvesting strategy."""
from datetime import date


class TestWeeklyThetaEligibility:
    def test_friday_eligible(self):
        from paper_trade import _is_weekly_theta_day
        assert _is_weekly_theta_day(date(2026, 3, 6)) is True  # Friday

    def test_monday_eligible(self):
        from paper_trade import _is_weekly_theta_day
        assert _is_weekly_theta_day(date(2026, 3, 9)) is True  # Monday

    def test_tuesday_not_eligible(self):
        from paper_trade import _is_weekly_theta_day
        assert _is_weekly_theta_day(date(2026, 3, 10)) is False

    def test_wednesday_not_eligible(self):
        from paper_trade import _is_weekly_theta_day
        assert _is_weekly_theta_day(date(2026, 3, 11)) is False


class TestWeeklyThetaExit:
    def test_target_50pct_decay(self):
        from paper_trade import check_weekly_theta_exit
        pos = {"total_credit": 80, "expiry": "2026-03-10", "entry_date": "2026-03-06"}
        assert check_weekly_theta_exit(pos, current_call_prem=20, current_put_prem=15) == "target"

    def test_stoploss_premium_doubles(self):
        from paper_trade import check_weekly_theta_exit
        pos = {"total_credit": 80, "expiry": "2026-03-10", "entry_date": "2026-03-06"}
        assert check_weekly_theta_exit(pos, current_call_prem=90, current_put_prem=80) == "stoploss"

    def test_hold_normal(self):
        from paper_trade import check_weekly_theta_exit
        pos = {"total_credit": 80, "expiry": "2026-03-10", "entry_date": "2026-03-06"}
        assert check_weekly_theta_exit(pos, current_call_prem=35, current_put_prem=25) is None


class TestStrangleStrikes:
    def test_select_strangle(self):
        from agent_with_options import select_strangle_strikes
        chain = [
            {"strikePrice": 24000, "CE": {"lastPrice": 250, "openInterest": 5000}, "PE": {"lastPrice": 10, "openInterest": 500}},
            {"strikePrice": 24200, "CE": {"lastPrice": 150, "openInterest": 8000}, "PE": {"lastPrice": 20, "openInterest": 2000}},
            {"strikePrice": 24500, "CE": {"lastPrice": 50, "openInterest": 10000}, "PE": {"lastPrice": 45, "openInterest": 10000}},
            {"strikePrice": 24800, "CE": {"lastPrice": 20, "openInterest": 8000}, "PE": {"lastPrice": 150, "openInterest": 8000}},
            {"strikePrice": 25000, "CE": {"lastPrice": 10, "openInterest": 5000}, "PE": {"lastPrice": 250, "openInterest": 5000}},
        ]
        result = select_strangle_strikes(chain, spot=24500, otm_points=250, lot_size=75)
        assert result is not None
        assert result["call_strike"] > 24500
        assert result["put_strike"] < 24500
        assert result["total_credit"] > 0
