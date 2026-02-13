"""
Tests for the paper trading engine.

Pure function tests — no SmartAPI calls. Tests position sizing, exit logic,
P&L calculations, portfolio stats, and persistence.
"""

import json
import math
from datetime import datetime

import pytest

from paper_trade import (
    STOPLOSS_PCT,
    TARGET_PCT,
    PUT_TARGET_PCT,
    PUT_STOPLOSS_PCT,
    MIN_DTE_TO_HOLD,
    TOTAL_CAPITAL,
    _add_trading_days,
    _empty_portfolio,
    _trading_days_between,
    calc_pnl_pct,
    close_position,
    compute_allocations,
    days_to_expiry,
    load_portfolio,
    save_portfolio,
    select_put_strike,
    PORTFOLIO_FILE,
)


# ---------------------------------------------------------------------------
# Position Sizing
# ---------------------------------------------------------------------------

class TestComputeAllocations:
    def test_score_weighted_allocation(self):
        candidates = [
            {"symbol": "A", "score": 6.0, "direction": "bullish"},
            {"symbol": "B", "score": 4.0, "direction": "bearish"},
        ]
        result = compute_allocations(candidates, 100_000)

        assert len(result) == 2
        assert result[0]["weight"] == 0.6
        assert result[1]["weight"] == 0.4
        assert result[0]["allocation"] == 60_000.0
        assert result[1]["allocation"] == 40_000.0

    def test_equal_scores(self):
        candidates = [
            {"symbol": "A", "score": 3.0, "direction": "bullish"},
            {"symbol": "B", "score": 3.0, "direction": "bearish"},
            {"symbol": "C", "score": 3.0, "direction": "bullish"},
        ]
        result = compute_allocations(candidates, 90_000)

        for r in result:
            assert r["weight"] == pytest.approx(1 / 3, abs=0.001)
            assert r["allocation"] == pytest.approx(30_000, abs=1.0)

    def test_single_candidate(self):
        candidates = [{"symbol": "A", "score": 5.0, "direction": "bullish"}]
        result = compute_allocations(candidates, 50_000)

        assert len(result) == 1
        assert result[0]["weight"] == 1.0
        assert result[0]["allocation"] == 50_000.0

    def test_zero_score_returns_empty(self):
        candidates = [{"symbol": "A", "score": 0.0, "direction": "bullish"}]
        assert compute_allocations(candidates, 100_000) == []

    def test_empty_candidates(self):
        assert compute_allocations([], 100_000) == []

    def test_weights_sum_to_one(self):
        candidates = [
            {"symbol": "A", "score": 6.75, "direction": "bullish"},
            {"symbol": "B", "score": 4.5, "direction": "bearish"},
            {"symbol": "C", "score": 3.0, "direction": "bullish"},
            {"symbol": "D", "score": 2.25, "direction": "bearish"},
            {"symbol": "E", "score": 1.5, "direction": "bullish"},
        ]
        result = compute_allocations(candidates, 100_000)
        total_weight = sum(r["weight"] for r in result)
        assert total_weight == pytest.approx(1.0, abs=0.01)

    def test_allocations_sum_to_available(self):
        candidates = [
            {"symbol": "A", "score": 5.0, "direction": "bullish"},
            {"symbol": "B", "score": 3.0, "direction": "bearish"},
        ]
        result = compute_allocations(candidates, 75_000)
        total_alloc = sum(r["allocation"] for r in result)
        assert total_alloc == pytest.approx(75_000, abs=1.0)

    def test_quantity_rounding(self):
        """Verify floor rounding: allocation / LTP -> integer shares."""
        candidates = [{"symbol": "A", "score": 5.0, "direction": "bullish"}]
        result = compute_allocations(candidates, 25_000)
        # Simulate: LTP = 1250.50, qty = floor(25000 / 1250.50) = 19
        ltp = 1250.50
        qty = math.floor(result[0]["allocation"] / ltp)
        assert qty == 19
        assert qty * ltp < result[0]["allocation"]


# ---------------------------------------------------------------------------
# P&L Calculation
# ---------------------------------------------------------------------------

class TestCalcPnlPct:
    def test_bullish_profit(self):
        # Buy at 100, price goes to 105 = +5%
        assert calc_pnl_pct(100, 105, "bullish") == pytest.approx(5.0)

    def test_bullish_loss(self):
        # Buy at 100, price drops to 97 = -3%
        assert calc_pnl_pct(100, 97, "bullish") == pytest.approx(-3.0)

    def test_bearish_profit(self):
        # Short at 100, price drops to 95 = +5%
        assert calc_pnl_pct(100, 95, "bearish") == pytest.approx(5.0)

    def test_bearish_loss(self):
        # Short at 100, price goes to 103 = -3%
        assert calc_pnl_pct(100, 103, "bearish") == pytest.approx(-3.0)

    def test_no_change(self):
        assert calc_pnl_pct(100, 100, "bullish") == 0.0
        assert calc_pnl_pct(100, 100, "bearish") == 0.0

    def test_real_price_bullish(self):
        # Entry 1250.50, LTP 1295 → (1295 - 1250.50) / 1250.50 * 100
        pnl = calc_pnl_pct(1250.50, 1295, "bullish")
        assert pnl == pytest.approx(3.558, abs=0.01)

    def test_real_price_bearish(self):
        # Short 3800, LTP 3720 → (3800 - 3720) / 3800 * 100
        pnl = calc_pnl_pct(3800, 3720, "bearish")
        assert pnl == pytest.approx(2.105, abs=0.01)


# ---------------------------------------------------------------------------
# Exit Logic
# ---------------------------------------------------------------------------

class TestExitLogic:
    def _make_position(self, direction="bullish", entry_price=1000, quantity=10):
        qty = quantity if direction == "bullish" else -quantity
        return {
            "symbol": "TEST",
            "token": "99999",
            "direction": direction,
            "entry_price": entry_price,
            "quantity": qty,
            "allocated": abs(qty) * entry_price,
            "score": 5.0,
            "categories": ["LongBuildUp"],
            "entry_date": "2026-02-10",
            "target_price": entry_price * 1.05 if direction == "bullish" else entry_price * 0.95,
            "stoploss_price": entry_price * 0.97 if direction == "bullish" else entry_price * 1.03,
            "max_hold_date": "2026-02-14",
            "status": "open",
        }

    def _make_portfolio(self, positions=None):
        p = _empty_portfolio()
        if positions:
            p["positions"] = positions
            p["available_capital"] = TOTAL_CAPITAL - sum(pos["allocated"] for pos in positions)
        return p

    def test_target_hit_bullish(self):
        pos = self._make_position("bullish", 1000, 10)
        portfolio = self._make_portfolio([pos])

        # Exit at +5% = 1050
        closed = close_position(portfolio, pos, 1050.0, "target")

        assert closed["pnl_pct"] == pytest.approx(5.0)
        assert closed["pnl"] == pytest.approx(500.0)  # (1050-1000)*10
        assert closed["exit_reason"] == "target"
        assert portfolio["stats"]["winning_trades"] == 1
        assert portfolio["stats"]["total_pnl"] == 500.0

    def test_stoploss_hit_bullish(self):
        pos = self._make_position("bullish", 1000, 10)
        portfolio = self._make_portfolio([pos])

        closed = close_position(portfolio, pos, 970.0, "stoploss")

        assert closed["pnl_pct"] == pytest.approx(-3.0)
        assert closed["pnl"] == pytest.approx(-300.0)
        assert closed["exit_reason"] == "stoploss"
        assert portfolio["stats"]["losing_trades"] == 1

    def test_target_hit_bearish(self):
        pos = self._make_position("bearish", 1000, 10)
        portfolio = self._make_portfolio([pos])

        # Short at 1000, exit at 950 = +5%
        closed = close_position(portfolio, pos, 950.0, "target")

        assert closed["pnl_pct"] == pytest.approx(5.0)
        assert closed["pnl"] == pytest.approx(500.0)  # (1000-950)*10
        assert closed["exit_reason"] == "target"

    def test_stoploss_hit_bearish(self):
        pos = self._make_position("bearish", 1000, 10)
        portfolio = self._make_portfolio([pos])

        # Short at 1000, price goes to 1030 = -3%
        closed = close_position(portfolio, pos, 1030.0, "stoploss")

        assert closed["pnl_pct"] == pytest.approx(-3.0)
        assert closed["pnl"] == pytest.approx(-300.0)

    def test_no_exit_within_bounds(self):
        """Position within bounds should not trigger exit."""
        pos = self._make_position("bullish", 1000, 10)
        # LTP at 1020 → +2%, well within target (5%) and stoploss (-3%)
        pnl = calc_pnl_pct(1000, 1020, "bullish")
        assert pnl < TARGET_PCT
        assert pnl > STOPLOSS_PCT

    def test_max_hold_expiry(self):
        pos = self._make_position("bullish", 1000, 10)
        portfolio = self._make_portfolio([pos])

        # Exit at a small loss due to expiry
        closed = close_position(portfolio, pos, 1010.0, "expiry")

        assert closed["pnl_pct"] == pytest.approx(1.0)
        assert closed["exit_reason"] == "expiry"

    def test_capital_released_on_close(self):
        pos = self._make_position("bullish", 1000, 10)
        portfolio = self._make_portfolio([pos])
        original_available = portfolio["available_capital"]

        close_position(portfolio, pos, 1050.0, "target")

        # Capital should be released (allocated amount returned)
        assert portfolio["available_capital"] == original_available + pos["allocated"]


# ---------------------------------------------------------------------------
# Portfolio Stats
# ---------------------------------------------------------------------------

class TestPortfolioStats:
    def test_running_win_rate(self):
        portfolio = _empty_portfolio()

        # 3 wins, 2 losses
        trades = [
            ("A", "bullish", 100, 105, 10, "target"),   # +5%
            ("B", "bearish", 200, 190, -5, "target"),    # +5%
            ("C", "bullish", 150, 146, 8, "stoploss"),   # -2.67%
            ("D", "bullish", 300, 315, 3, "target"),     # +5%
            ("E", "bearish", 500, 520, -2, "stoploss"),  # -4%
        ]

        for sym, direction, entry, exit_p, qty, reason in trades:
            pos = {
                "symbol": sym, "direction": direction, "entry_price": entry,
                "quantity": qty, "allocated": abs(qty) * entry,
                "entry_date": "2026-02-10", "status": "open",
            }
            portfolio["positions"].append(pos)
            close_position(portfolio, pos, exit_p, reason)

        stats = portfolio["stats"]
        assert stats["total_trades"] == 5
        assert stats["winning_trades"] == 3
        assert stats["losing_trades"] == 2
        assert stats["winning_trades"] / stats["total_trades"] == 0.6

    def test_cumulative_pnl(self):
        portfolio = _empty_portfolio()

        # Trade 1: bullish +5%
        pos1 = {
            "symbol": "A", "direction": "bullish", "entry_price": 100,
            "quantity": 10, "allocated": 1000,
            "entry_date": "2026-02-10", "status": "open",
        }
        portfolio["positions"].append(pos1)
        close_position(portfolio, pos1, 105, "target")
        assert portfolio["stats"]["total_pnl"] == 50.0  # (105-100)*10

        # Trade 2: bearish +5%
        pos2 = {
            "symbol": "B", "direction": "bearish", "entry_price": 200,
            "quantity": -5, "allocated": 1000,
            "entry_date": "2026-02-11", "status": "open",
        }
        portfolio["positions"].append(pos2)
        close_position(portfolio, pos2, 190, "target")
        assert portfolio["stats"]["total_pnl"] == 100.0  # 50 + 50

        # Trade 3: bullish -3%
        pos3 = {
            "symbol": "C", "direction": "bullish", "entry_price": 100,
            "quantity": 10, "allocated": 1000,
            "entry_date": "2026-02-12", "status": "open",
        }
        portfolio["positions"].append(pos3)
        close_position(portfolio, pos3, 97, "stoploss")
        assert portfolio["stats"]["total_pnl"] == 70.0  # 100 - 30

    def test_best_worst_trade(self):
        portfolio = _empty_portfolio()

        trades = [
            ("A", "bullish", 100, 105, 10),   # +5%
            ("B", "bullish", 100, 97, 10),     # -3%
            ("C", "bullish", 100, 102, 10),    # +2%
        ]

        for sym, direction, entry, exit_p, qty in trades:
            pos = {
                "symbol": sym, "direction": direction, "entry_price": entry,
                "quantity": qty, "allocated": qty * entry,
                "entry_date": "2026-02-10", "status": "open",
            }
            portfolio["positions"].append(pos)
            close_position(portfolio, pos, exit_p, "target")

        assert portfolio["stats"]["best_trade"]["symbol"] == "A"
        assert portfolio["stats"]["best_trade"]["pnl_pct"] == 5.0
        assert portfolio["stats"]["worst_trade"]["symbol"] == "B"
        assert portfolio["stats"]["worst_trade"]["pnl_pct"] == -3.0

    def test_pnl_pct_on_capital(self):
        portfolio = _empty_portfolio()

        pos = {
            "symbol": "A", "direction": "bullish", "entry_price": 100,
            "quantity": 10, "allocated": 1000,
            "entry_date": "2026-02-10", "status": "open",
        }
        portfolio["positions"].append(pos)
        close_position(portfolio, pos, 110, "target")

        # PnL = 100, on capital of 100,000 = 0.1%
        assert portfolio["stats"]["total_pnl"] == 100.0
        assert portfolio["stats"]["total_pnl_pct"] == 0.1


# ---------------------------------------------------------------------------
# Trading Day Helpers
# ---------------------------------------------------------------------------

class TestTradingDays:
    def test_add_trading_days_simple(self):
        # Monday 2026-02-09 + 5 trading days = Monday 2026-02-16
        result = _add_trading_days("2026-02-09", 5)
        assert result == "2026-02-16"

    def test_add_trading_days_across_weekend(self):
        # Thursday 2026-02-12 + 3 trading days = Tue 2026-02-17
        result = _add_trading_days("2026-02-12", 3)
        assert result == "2026-02-17"

    def test_add_trading_days_from_friday(self):
        # Friday 2026-02-13 + 1 = Monday 2026-02-16
        result = _add_trading_days("2026-02-13", 1)
        assert result == "2026-02-16"

    def test_trading_days_between(self):
        # Mon 2/9 to Fri 2/13 = 5 weekdays
        assert _trading_days_between("2026-02-09", "2026-02-13") == 5

    def test_trading_days_between_same_day(self):
        assert _trading_days_between("2026-02-10", "2026-02-10") == 1

    def test_trading_days_between_over_weekend(self):
        # Fri 2/13 to Mon 2/16 = 2 weekdays (Fri + Mon)
        assert _trading_days_between("2026-02-13", "2026-02-16") == 2

    def test_trading_days_between_reversed(self):
        # end before start = 0
        assert _trading_days_between("2026-02-13", "2026-02-10") == 0


# ---------------------------------------------------------------------------
# Persistence (load / save)
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load(self, tmp_path, monkeypatch):
        test_file = tmp_path / "portfolio.json"
        monkeypatch.setattr("paper_trade.PORTFOLIO_FILE", test_file)
        monkeypatch.setattr("paper_trade.PORTFOLIO_DIR", tmp_path)

        portfolio = _empty_portfolio()
        portfolio["available_capital"] = 75_000
        portfolio["positions"].append({
            "symbol": "TEST",
            "token": "123",
            "direction": "bullish",
            "entry_price": 100,
            "quantity": 10,
            "allocated": 1000,
            "score": 5.0,
            "categories": ["LongBuildUp"],
            "entry_date": "2026-02-10",
            "target_price": 105,
            "stoploss_price": 97,
            "max_hold_date": "2026-02-14",
            "status": "open",
        })

        save_portfolio(portfolio)
        assert test_file.exists()

        loaded = load_portfolio()
        assert loaded["available_capital"] == 75_000
        assert len(loaded["positions"]) == 1
        assert loaded["positions"][0]["symbol"] == "TEST"

    def test_load_nonexistent_creates_fresh(self, tmp_path, monkeypatch):
        test_file = tmp_path / "does_not_exist.json"
        monkeypatch.setattr("paper_trade.PORTFOLIO_FILE", test_file)

        portfolio = load_portfolio()
        assert portfolio["capital"] == TOTAL_CAPITAL
        assert portfolio["available_capital"] == TOTAL_CAPITAL
        assert portfolio["positions"] == []
        assert portfolio["closed_trades"] == []

    def test_round_trip_preserves_data(self, tmp_path, monkeypatch):
        test_file = tmp_path / "portfolio.json"
        monkeypatch.setattr("paper_trade.PORTFOLIO_FILE", test_file)
        monkeypatch.setattr("paper_trade.PORTFOLIO_DIR", tmp_path)

        portfolio = _empty_portfolio()

        # Add a closed trade
        pos = {
            "symbol": "A", "direction": "bullish", "entry_price": 100,
            "quantity": 10, "allocated": 1000,
            "entry_date": "2026-02-10", "status": "open",
        }
        portfolio["positions"].append(pos)
        close_position(portfolio, pos, 105, "target")
        portfolio["positions"] = [p for p in portfolio["positions"] if p["status"] == "open"]

        save_portfolio(portfolio)
        loaded = load_portfolio()

        assert loaded["stats"]["total_trades"] == 1
        assert loaded["stats"]["winning_trades"] == 1
        assert loaded["stats"]["total_pnl"] == 50.0
        assert len(loaded["closed_trades"]) == 1
        assert loaded["closed_trades"][0]["pnl_pct"] == 5.0


# ---------------------------------------------------------------------------
# Put Option Logic
# ---------------------------------------------------------------------------

class TestPutOptionLogic:
    """Tests for put option support in paper trading."""

    # --- Strike Selection ---

    def test_select_put_strike_atm(self):
        """ATM strike (closest ≤ spot) is picked."""
        chain = [
            {"strikePrice": 5300, "PE": {"lastTradedPrice": 120, "delta": -0.35, "theta": -7, "impliedVolatility": 30, "openInterest": 5000, "gamma": 0.002, "vega": 10}},
            {"strikePrice": 5400, "PE": {"lastTradedPrice": 75, "delta": -0.42, "theta": -8, "impliedVolatility": 28, "openInterest": 8000, "gamma": 0.003, "vega": 12}},
            {"strikePrice": 5500, "PE": {"lastTradedPrice": 40, "delta": -0.25, "theta": -5, "impliedVolatility": 32, "openInterest": 3000, "gamma": 0.001, "vega": 8}},
        ]
        result = select_put_strike(chain, spot_price=5420)
        assert result is not None
        assert result["strike"] == 5400  # closest ≤ 5420

    def test_select_put_strike_between_strikes(self):
        """When spot is between strikes, picks the OTM one (below spot)."""
        chain = [
            {"strikePrice": 5300, "PE": {"lastTradedPrice": 120, "delta": -0.35, "theta": -7, "impliedVolatility": 30, "openInterest": 5000, "gamma": 0.002, "vega": 10}},
            {"strikePrice": 5400, "PE": {"lastTradedPrice": 75, "delta": -0.42, "theta": -8, "impliedVolatility": 28, "openInterest": 8000, "gamma": 0.003, "vega": 12}},
            {"strikePrice": 5500, "PE": {"lastTradedPrice": 40, "delta": -0.25, "theta": -5, "impliedVolatility": 32, "openInterest": 3000, "gamma": 0.001, "vega": 8}},
        ]
        result = select_put_strike(chain, spot_price=5450)
        assert result is not None
        assert result["strike"] == 5400  # closest ≤ 5450

    def test_select_put_strike_exact_match(self):
        """Spot price exactly at a strike."""
        chain = [
            {"strikePrice": 5400, "PE": {"lastTradedPrice": 75, "delta": -0.42, "theta": -8, "impliedVolatility": 28, "openInterest": 8000, "gamma": 0.003, "vega": 12}},
        ]
        result = select_put_strike(chain, spot_price=5400)
        assert result is not None
        assert result["strike"] == 5400

    def test_select_put_strike_empty_chain(self):
        """Empty option chain returns None."""
        assert select_put_strike([], spot_price=5400) is None

    def test_select_put_strike_zero_ltp_filtered(self):
        """Strikes with zero PE LTP are filtered out."""
        chain = [
            {"strikePrice": 5400, "PE": {"lastTradedPrice": 0, "delta": -0.42, "theta": -8, "impliedVolatility": 28, "openInterest": 8000, "gamma": 0.003, "vega": 12}},
            {"strikePrice": 5300, "PE": {"lastTradedPrice": 120, "delta": -0.35, "theta": -7, "impliedVolatility": 30, "openInterest": 5000, "gamma": 0.002, "vega": 10}},
        ]
        result = select_put_strike(chain, spot_price=5450)
        assert result is not None
        assert result["strike"] == 5300  # 5400 filtered out (zero LTP)

    def test_select_put_strike_all_above_spot(self):
        """All strikes above spot returns None (ITM puts excluded)."""
        chain = [
            {"strikePrice": 5500, "PE": {"lastTradedPrice": 40, "delta": -0.25, "theta": -5, "impliedVolatility": 32, "openInterest": 3000, "gamma": 0.001, "vega": 8}},
            {"strikePrice": 5600, "PE": {"lastTradedPrice": 20, "delta": -0.15, "theta": -3, "impliedVolatility": 35, "openInterest": 1000, "gamma": 0.001, "vega": 5}},
        ]
        result = select_put_strike(chain, spot_price=5400)
        assert result is None

    # --- DTE Calculation ---

    def test_days_to_expiry(self, monkeypatch):
        """Known date pair gives expected DTE."""
        from datetime import date
        fake_today = date(2026, 2, 13)
        monkeypatch.setattr("paper_trade.datetime", type("FakeDatetime", (), {
            "now": staticmethod(lambda tz=None: type("D", (), {"date": lambda self: fake_today})()),
            "strptime": datetime.strptime,
        })())
        # Actually, let's just monkeypatch more precisely
        monkeypatch.undo()

        import paper_trade
        original_now = datetime.now

        class FakeDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime(2026, 2, 13, tzinfo=tz)

        monkeypatch.setattr(paper_trade, "datetime", FakeDatetime)
        result = days_to_expiry("27FEB2026")
        assert result == 14  # Feb 13 to Feb 27

    # --- Option P&L ---

    def test_put_pnl_premium_increase(self):
        """Bought put at 75, premium rises to 112.5 → +50% (target)."""
        pnl = calc_pnl_pct(75, 112.5, "bullish")
        assert pnl == pytest.approx(50.0)

    def test_put_pnl_premium_decrease(self):
        """Bought put at 75, premium drops to 45 → -40% (stoploss)."""
        pnl = calc_pnl_pct(75, 45, "bullish")
        assert pnl == pytest.approx(-40.0)

    def test_put_target_threshold(self):
        """Premium increase just above target threshold."""
        pnl = calc_pnl_pct(100, 151, "bullish")
        assert pnl >= PUT_TARGET_PCT

    def test_put_stoploss_threshold(self):
        """Premium decrease just below stoploss threshold."""
        pnl = calc_pnl_pct(100, 59, "bullish")
        assert pnl <= PUT_STOPLOSS_PCT

    # --- close_position for options ---

    def _make_option_position(self, entry_premium=75.0, lot_size=500, num_lots=1):
        qty = lot_size * num_lots
        return {
            "symbol": "PERSISTENT",
            "token": "77777",
            "direction": "bearish",
            "instrument": "OPT",
            "option_type": "PE",
            "strike": 5400.0,
            "expiry": "27FEB2026",
            "contract_symbol": "PERSISTENT27FEB2026PE5400",
            "lot_size": lot_size,
            "num_lots": num_lots,
            "entry_price": entry_premium,
            "quantity": qty,
            "allocated": entry_premium * qty,
            "underlying_price_at_entry": 5452.0,
            "greeks_at_entry": {"delta": -0.42, "theta": -8.5, "iv": 32.5, "gamma": 0.003, "vega": 12.1},
            "score": 6.0,
            "categories": ["ShortBuildUp"],
            "entry_date": "2026-02-13",
            "max_hold_date": "2026-02-20",
            "status": "open",
        }

    def test_close_option_target(self):
        """Option position closed at +50% premium gain."""
        pos = self._make_option_position(75.0, 500, 1)
        portfolio = _empty_portfolio()
        portfolio["positions"].append(pos)
        portfolio["available_capital"] = TOTAL_CAPITAL - pos["allocated"]

        closed = close_position(portfolio, pos, 112.5, "target")

        # P&L uses "bullish" direction: (112.5 - 75) * 500 = 18750
        assert closed["pnl_pct"] == pytest.approx(50.0)
        assert closed["pnl"] == pytest.approx(18750.0)
        assert portfolio["stats"]["winning_trades"] == 1

    def test_close_option_stoploss(self):
        """Option position closed at -40% premium loss."""
        pos = self._make_option_position(75.0, 500, 1)
        portfolio = _empty_portfolio()
        portfolio["positions"].append(pos)
        portfolio["available_capital"] = TOTAL_CAPITAL - pos["allocated"]

        closed = close_position(portfolio, pos, 45.0, "stoploss")

        # P&L: (45 - 75) * 500 = -15000
        assert closed["pnl_pct"] == pytest.approx(-40.0)
        assert closed["pnl"] == pytest.approx(-15000.0)
        assert portfolio["stats"]["losing_trades"] == 1

    def test_close_option_capital_released(self):
        """Allocated capital is freed when option position is closed."""
        pos = self._make_option_position(75.0, 500, 1)
        portfolio = _empty_portfolio()
        portfolio["positions"].append(pos)
        portfolio["available_capital"] = TOTAL_CAPITAL - pos["allocated"]
        avail_before = portfolio["available_capital"]

        close_position(portfolio, pos, 90.0, "target")
        assert portfolio["available_capital"] == avail_before + pos["allocated"]

    # --- Backward Compatibility ---

    def test_legacy_position_no_instrument_field(self):
        """Position without 'instrument' field defaults to equity behavior."""
        pos = {
            "symbol": "TEST",
            "token": "99999",
            "direction": "bearish",
            "entry_price": 1000,
            "quantity": -10,
            "allocated": 10000,
            "entry_date": "2026-02-10",
            "status": "open",
        }
        portfolio = _empty_portfolio()
        portfolio["positions"].append(pos)
        portfolio["available_capital"] = TOTAL_CAPITAL - pos["allocated"]

        # Legacy bearish: (entry - exit) / entry
        closed = close_position(portfolio, pos, 950, "target")
        assert closed["pnl_pct"] == pytest.approx(5.0)
        assert closed["pnl"] == pytest.approx(500.0)

    # --- Persistence with Option Positions ---

    def test_save_load_option_position(self, tmp_path, monkeypatch):
        """Option position round-trips through save/load."""
        test_file = tmp_path / "portfolio.json"
        monkeypatch.setattr("paper_trade.PORTFOLIO_FILE", test_file)
        monkeypatch.setattr("paper_trade.PORTFOLIO_DIR", tmp_path)

        portfolio = _empty_portfolio()
        pos = self._make_option_position(75.0, 500, 1)
        portfolio["positions"].append(pos)
        portfolio["available_capital"] = TOTAL_CAPITAL - pos["allocated"]

        save_portfolio(portfolio)
        loaded = load_portfolio()

        opt_pos = loaded["positions"][0]
        assert opt_pos["instrument"] == "OPT"
        assert opt_pos["option_type"] == "PE"
        assert opt_pos["strike"] == 5400.0
        assert opt_pos["expiry"] == "27FEB2026"
        assert opt_pos["contract_symbol"] == "PERSISTENT27FEB2026PE5400"
        assert opt_pos["lot_size"] == 500
        assert opt_pos["num_lots"] == 1
        assert opt_pos["greeks_at_entry"]["delta"] == -0.42
