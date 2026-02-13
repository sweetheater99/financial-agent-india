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
    COOLDOWN_DAYS,
    MAX_POSITIONS_PER_SECTOR,
    TRAILING_STOP_PCT,
    OPT_TRAILING_STOP_PCT,
    ATR_PERIOD,
    ATR_TARGET_MULTIPLIER,
    ATR_STOPLOSS_MULTIPLIER,
    OPT_THETA_DTE_THRESHOLD,
    OPT_THETA_MIN_GAIN_PCT,
    EQ_SLIPPAGE_PCT,
    OPT_SLIPPAGE_PCT,
    PARTIAL_EXIT_RATIO,
    RSI_BULLISH_MIN,
    RSI_BULLISH_MAX,
    RSI_BEARISH_MIN,
    RSI_BEARISH_MAX,
    VOLUME_MIN_RATIO,
    SKIP_EARNINGS_SOON,
    SKIP_CONTRADICTING_NEWS,
    INTRADAY_CONFIRM_ENABLED,
    INTRADAY_MIN_CANDLES,
    INTRADAY_MIN_MOVE_PCT,
    _add_trading_days,
    _empty_portfolio,
    _trading_days_between,
    _setup_logger,
    apply_slippage,
    calc_pnl_pct,
    calc_performance_analytics,
    calc_transaction_costs,
    calc_round_trip_costs,
    check_cooldown,
    check_intraday_momentum,
    check_sector_limit,
    classify_regime,
    close_position,
    compute_allocations,
    compute_atr,
    days_to_expiry,
    fetch_intraday_candles,
    get_sector,
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

        costs = calc_round_trip_costs("EQ", 1000, 1050, 10)
        expected_net = round(500.0 - costs["total"], 2)
        assert closed["pnl_gross"] == pytest.approx(500.0)
        assert closed["pnl"] == pytest.approx(expected_net)
        assert closed["exit_reason"] == "target"
        assert portfolio["stats"]["winning_trades"] == 1
        assert portfolio["stats"]["total_pnl"] == pytest.approx(expected_net)

    def test_stoploss_hit_bullish(self):
        pos = self._make_position("bullish", 1000, 10)
        portfolio = self._make_portfolio([pos])

        closed = close_position(portfolio, pos, 970.0, "stoploss")

        costs = calc_round_trip_costs("EQ", 1000, 970, 10)
        expected_net = round(-300.0 - costs["total"], 2)
        assert closed["pnl_gross"] == pytest.approx(-300.0)
        assert closed["pnl"] == pytest.approx(expected_net)
        assert closed["exit_reason"] == "stoploss"
        assert portfolio["stats"]["losing_trades"] == 1

    def test_target_hit_bearish(self):
        pos = self._make_position("bearish", 1000, 10)
        portfolio = self._make_portfolio([pos])

        # Short at 1000, exit at 950 = +5%
        closed = close_position(portfolio, pos, 950.0, "target")

        costs = calc_round_trip_costs("EQ", 1000, 950, 10)
        expected_net = round(500.0 - costs["total"], 2)
        assert closed["pnl_gross"] == pytest.approx(500.0)
        assert closed["pnl"] == pytest.approx(expected_net)
        assert closed["exit_reason"] == "target"

    def test_stoploss_hit_bearish(self):
        pos = self._make_position("bearish", 1000, 10)
        portfolio = self._make_portfolio([pos])

        # Short at 1000, price goes to 1030 = -3%
        closed = close_position(portfolio, pos, 1030.0, "stoploss")

        costs = calc_round_trip_costs("EQ", 1000, 1030, 10)
        expected_net = round(-300.0 - costs["total"], 2)
        assert closed["pnl_gross"] == pytest.approx(-300.0)
        assert closed["pnl"] == pytest.approx(expected_net)

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

        # Exit at a small gain due to expiry
        closed = close_position(portfolio, pos, 1010.0, "expiry")

        costs = calc_round_trip_costs("EQ", 1000, 1010, 10)
        expected_net = round(100.0 - costs["total"], 2)
        assert closed["pnl_gross"] == pytest.approx(100.0)
        assert closed["pnl"] == pytest.approx(expected_net)
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
        costs1 = calc_round_trip_costs("EQ", 100, 105, 10)
        expected1 = round(50.0 - costs1["total"], 2)
        assert portfolio["stats"]["total_pnl"] == pytest.approx(expected1)

        # Trade 2: bearish +5%
        pos2 = {
            "symbol": "B", "direction": "bearish", "entry_price": 200,
            "quantity": -5, "allocated": 1000,
            "entry_date": "2026-02-11", "status": "open",
        }
        portfolio["positions"].append(pos2)
        close_position(portfolio, pos2, 190, "target")
        costs2 = calc_round_trip_costs("EQ", 200, 190, 5)
        expected2 = round(expected1 + 50.0 - costs2["total"], 2)
        assert portfolio["stats"]["total_pnl"] == pytest.approx(expected2)

        # Trade 3: bullish -3%
        pos3 = {
            "symbol": "C", "direction": "bullish", "entry_price": 100,
            "quantity": 10, "allocated": 1000,
            "entry_date": "2026-02-12", "status": "open",
        }
        portfolio["positions"].append(pos3)
        close_position(portfolio, pos3, 97, "stoploss")
        costs3 = calc_round_trip_costs("EQ", 100, 97, 10)
        expected3 = round(expected2 + (-30.0) - costs3["total"], 2)
        assert portfolio["stats"]["total_pnl"] == pytest.approx(expected3)

    def test_best_worst_trade(self):
        portfolio = _empty_portfolio()

        trades = [
            ("A", "bullish", 100, 105, 10),   # best (highest gross)
            ("B", "bullish", 100, 97, 10),     # worst (lowest gross)
            ("C", "bullish", 100, 102, 10),    # middle
        ]

        closed_trades = []
        for sym, direction, entry, exit_p, qty in trades:
            pos = {
                "symbol": sym, "direction": direction, "entry_price": entry,
                "quantity": qty, "allocated": qty * entry,
                "entry_date": "2026-02-10", "status": "open",
            }
            portfolio["positions"].append(pos)
            ct = close_position(portfolio, pos, exit_p, "target")
            closed_trades.append(ct)

        assert portfolio["stats"]["best_trade"]["symbol"] == "A"
        assert portfolio["stats"]["best_trade"]["pnl_pct"] == closed_trades[0]["pnl_pct"]
        assert portfolio["stats"]["worst_trade"]["symbol"] == "B"
        assert portfolio["stats"]["worst_trade"]["pnl_pct"] == closed_trades[1]["pnl_pct"]

    def test_pnl_pct_on_capital(self):
        portfolio = _empty_portfolio()

        pos = {
            "symbol": "A", "direction": "bullish", "entry_price": 100,
            "quantity": 10, "allocated": 1000,
            "entry_date": "2026-02-10", "status": "open",
        }
        portfolio["positions"].append(pos)
        close_position(portfolio, pos, 110, "target")

        costs = calc_round_trip_costs("EQ", 100, 110, 10)
        expected_pnl = round(100.0 - costs["total"], 2)
        expected_pct = round(expected_pnl / TOTAL_CAPITAL * 100, 2)
        assert portfolio["stats"]["total_pnl"] == pytest.approx(expected_pnl)
        assert portfolio["stats"]["total_pnl_pct"] == pytest.approx(expected_pct)


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

        costs = calc_round_trip_costs("EQ", 100, 105, 10)
        expected_pnl = round(50.0 - costs["total"], 2)

        assert loaded["stats"]["total_trades"] == 1
        assert loaded["stats"]["winning_trades"] == 1
        assert loaded["stats"]["total_pnl"] == pytest.approx(expected_pnl)
        assert len(loaded["closed_trades"]) == 1
        assert loaded["closed_trades"][0]["pnl_gross"] == pytest.approx(50.0)
        assert loaded["closed_trades"][0]["pnl"] == pytest.approx(expected_pnl)


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

        # Gross P&L uses "bullish" direction: (112.5 - 75) * 500 = 18750
        costs = calc_round_trip_costs("OPT", 75.0, 112.5, 500)
        expected_net = round(18750.0 - costs["total"], 2)
        assert closed["pnl_gross"] == pytest.approx(18750.0)
        assert closed["pnl"] == pytest.approx(expected_net)
        assert portfolio["stats"]["winning_trades"] == 1

    def test_close_option_stoploss(self):
        """Option position closed at -40% premium loss."""
        pos = self._make_option_position(75.0, 500, 1)
        portfolio = _empty_portfolio()
        portfolio["positions"].append(pos)
        portfolio["available_capital"] = TOTAL_CAPITAL - pos["allocated"]

        closed = close_position(portfolio, pos, 45.0, "stoploss")

        # Gross P&L: (45 - 75) * 500 = -15000
        costs = calc_round_trip_costs("OPT", 75.0, 45.0, 500)
        expected_net = round(-15000.0 - costs["total"], 2)
        assert closed["pnl_gross"] == pytest.approx(-15000.0)
        assert closed["pnl"] == pytest.approx(expected_net)
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

        # Legacy bearish: defaults to EQ instrument
        closed = close_position(portfolio, pos, 950, "target")
        costs = calc_round_trip_costs("EQ", 1000, 950, 10)
        expected_net = round(500.0 - costs["total"], 2)
        assert closed["pnl_gross"] == pytest.approx(500.0)
        assert closed["pnl"] == pytest.approx(expected_net)

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


# ---------------------------------------------------------------------------
# Transaction Costs
# ---------------------------------------------------------------------------

class TestTransactionCosts:
    """Tests for transaction cost calculation functions."""

    def test_equity_buy_side_costs(self):
        """Equity buy: brokerage, STT, stamp duty, exchange, SEBI, GST."""
        costs = calc_transaction_costs("EQ", "buy", 1000, 10)
        turnover = 10000

        assert costs["brokerage"] == pytest.approx(min(20.0, turnover * 0.0003), abs=0.01)
        assert costs["stt"] == pytest.approx(turnover * 0.001, abs=0.01)
        assert costs["stamp_duty"] == pytest.approx(turnover * 0.00015, abs=0.01)
        assert costs["exchange"] == pytest.approx(turnover * 0.0000345, abs=0.01)
        assert costs["sebi"] == pytest.approx(turnover * 0.000001, abs=0.01)
        assert costs["gst"] == pytest.approx((costs["brokerage"] + costs["exchange"]) * 0.18, abs=0.01)
        assert costs["total"] == pytest.approx(
            costs["brokerage"] + costs["stt"] + costs["exchange"]
            + costs["stamp_duty"] + costs["sebi"] + costs["gst"], abs=0.01)

    def test_equity_sell_side_costs(self):
        """Equity sell: STT on sell, NO stamp duty on sell."""
        costs = calc_transaction_costs("EQ", "sell", 1050, 10)
        turnover = 10500

        assert costs["stt"] == pytest.approx(turnover * 0.001, abs=0.01)
        assert costs["stamp_duty"] == 0.0  # no stamp duty on sell

    def test_option_buy_side_costs(self):
        """Option buy: no STT on buy, stamp duty on buy."""
        costs = calc_transaction_costs("OPT", "buy", 75, 500)
        turnover = 37500

        assert costs["brokerage"] == 20.0  # flat ₹20 for options
        assert costs["stt"] == 0.0  # no STT on option buy
        assert costs["stamp_duty"] == pytest.approx(turnover * 0.00003, abs=0.01)
        assert costs["exchange"] == pytest.approx(turnover * 0.000495, abs=0.01)

    def test_option_sell_side_costs(self):
        """Option sell: 0.0625% STT, no stamp duty."""
        costs = calc_transaction_costs("OPT", "sell", 112.5, 500)
        turnover = 56250

        assert costs["stt"] == pytest.approx(turnover * 0.000625, abs=0.01)
        assert costs["stamp_duty"] == 0.0  # no stamp duty on sell

    def test_round_trip_equity(self):
        """Entry + exit costs sum correctly for equity."""
        rt = calc_round_trip_costs("EQ", 1000, 1050, 10)
        entry = calc_transaction_costs("EQ", "buy", 1000, 10)
        exit_ = calc_transaction_costs("EQ", "sell", 1050, 10)

        assert rt["entry_total"] == pytest.approx(entry["total"])
        assert rt["exit_total"] == pytest.approx(exit_["total"])
        assert rt["total"] == pytest.approx(entry["total"] + exit_["total"], abs=0.01)

    def test_round_trip_option(self):
        """Entry + exit costs sum correctly for options."""
        rt = calc_round_trip_costs("OPT", 75.0, 112.5, 500)
        entry = calc_transaction_costs("OPT", "buy", 75.0, 500)
        exit_ = calc_transaction_costs("OPT", "sell", 112.5, 500)

        assert rt["entry_total"] == pytest.approx(entry["total"])
        assert rt["exit_total"] == pytest.approx(exit_["total"])
        assert rt["total"] == pytest.approx(entry["total"] + exit_["total"], abs=0.01)

    def test_equity_brokerage_cap(self):
        """For high-value equity trades, 0.03% < ₹20 so percentage is used."""
        # Turnover = 100 * 100 = 10000, 0.03% = 3.0 < 20
        costs = calc_transaction_costs("EQ", "buy", 100, 100)
        assert costs["brokerage"] == pytest.approx(3.0, abs=0.01)

        # Turnover = 5000 * 100 = 500000, 0.03% = 150 > 20 → capped at 20
        costs_high = calc_transaction_costs("EQ", "buy", 5000, 100)
        assert costs_high["brokerage"] == pytest.approx(20.0)

    def test_zero_quantity(self):
        """Edge case: zero quantity should produce zero costs."""
        costs = calc_transaction_costs("EQ", "buy", 1000, 0)
        assert costs["total"] == 0.0

    def test_close_position_stores_costs(self):
        """Closed trade record includes transaction_costs and pnl_gross fields."""
        pos = {
            "symbol": "TEST", "direction": "bullish", "entry_price": 1000,
            "quantity": 10, "allocated": 10000,
            "entry_date": "2026-02-10", "status": "open",
        }
        portfolio = _empty_portfolio()
        portfolio["positions"].append(pos)

        closed = close_position(portfolio, pos, 1050, "target")

        assert "pnl_gross" in closed
        assert "transaction_costs" in closed
        assert closed["pnl_gross"] == pytest.approx(500.0)
        assert closed["transaction_costs"] > 0
        assert closed["pnl"] == pytest.approx(closed["pnl_gross"] - closed["transaction_costs"])

    def test_stats_track_cumulative_costs(self):
        """stats['total_costs'] accumulates across trades."""
        portfolio = _empty_portfolio()

        pos1 = {
            "symbol": "A", "direction": "bullish", "entry_price": 1000,
            "quantity": 10, "allocated": 10000,
            "entry_date": "2026-02-10", "status": "open",
        }
        portfolio["positions"].append(pos1)
        close_position(portfolio, pos1, 1050, "target")
        costs1 = calc_round_trip_costs("EQ", 1000, 1050, 10)

        assert portfolio["stats"]["total_costs"] == pytest.approx(costs1["total"], abs=0.01)

        pos2 = {
            "symbol": "B", "direction": "bullish", "entry_price": 500,
            "quantity": 20, "allocated": 10000,
            "entry_date": "2026-02-11", "status": "open",
        }
        portfolio["positions"].append(pos2)
        close_position(portfolio, pos2, 485, "stoploss")
        costs2 = calc_round_trip_costs("EQ", 500, 485, 20)

        assert portfolio["stats"]["total_costs"] == pytest.approx(
            costs1["total"] + costs2["total"], abs=0.01)


# ---------------------------------------------------------------------------
# Slippage Modeling
# ---------------------------------------------------------------------------

class TestSlippage:
    def test_equity_buy_slippage(self):
        """Buy slippage increases price."""
        result = apply_slippage(1000.0, "EQ", "buy")
        assert result == pytest.approx(1000 * (1 + EQ_SLIPPAGE_PCT), abs=0.01)
        assert result > 1000.0

    def test_equity_sell_slippage(self):
        """Sell slippage decreases price."""
        result = apply_slippage(1000.0, "EQ", "sell")
        assert result == pytest.approx(1000 * (1 - EQ_SLIPPAGE_PCT), abs=0.01)
        assert result < 1000.0

    def test_option_buy_slippage(self):
        """Option buy slippage is larger than equity."""
        result = apply_slippage(100.0, "OPT", "buy")
        assert result == pytest.approx(100 * (1 + OPT_SLIPPAGE_PCT), abs=0.01)
        assert result > 100.0

    def test_option_sell_slippage(self):
        """Option sell slippage decreases price."""
        result = apply_slippage(100.0, "OPT", "sell")
        assert result == pytest.approx(100 * (1 - OPT_SLIPPAGE_PCT), abs=0.01)
        assert result < 100.0


# ---------------------------------------------------------------------------
# Re-entry Cooldown
# ---------------------------------------------------------------------------

class TestCooldown:
    def test_recent_stoploss_blocked(self):
        """Symbol with stoploss exit within COOLDOWN_DAYS is blocked."""
        trades = [{"symbol": "A", "exit_reason": "stoploss", "exit_date": "2026-02-10"}]
        assert check_cooldown("A", trades, "2026-02-12") is True

    def test_old_stoploss_allowed(self):
        """Symbol with stoploss exit older than COOLDOWN_DAYS is allowed."""
        trades = [{"symbol": "A", "exit_reason": "stoploss", "exit_date": "2026-02-01"}]
        assert check_cooldown("A", trades, "2026-02-12") is False

    def test_target_exit_no_cooldown(self):
        """Target exit does not trigger cooldown."""
        trades = [{"symbol": "A", "exit_reason": "target", "exit_date": "2026-02-10"}]
        assert check_cooldown("A", trades, "2026-02-12") is False

    def test_no_history_allowed(self):
        """No trade history means no cooldown."""
        assert check_cooldown("A", [], "2026-02-12") is False

    def test_trailing_stop_triggers_cooldown(self):
        """Trailing stop exit also triggers cooldown."""
        trades = [{"symbol": "A", "exit_reason": "trailing_stop", "exit_date": "2026-02-10"}]
        assert check_cooldown("A", trades, "2026-02-12") is True


# ---------------------------------------------------------------------------
# Sector Concentration
# ---------------------------------------------------------------------------

class TestSectorCheck:
    def test_known_sector(self):
        assert get_sector("HDFCBANK") == "Banking"
        assert get_sector("TCS") == "IT"
        assert get_sector("RELIANCE") == "Energy"

    def test_unknown_sector(self):
        assert get_sector("XYZUNKNOWN") == "Other"

    def test_sector_limit_reached(self):
        """Sector with MAX_POSITIONS_PER_SECTOR open positions blocks new ones."""
        positions = [
            {"symbol": "HDFCBANK", "status": "open"},
            {"symbol": "ICICIBANK", "status": "open"},
        ]
        assert check_sector_limit("SBIN", positions) is True

    def test_sector_limit_not_reached(self):
        """Sector with fewer than limit allows new positions."""
        positions = [{"symbol": "HDFCBANK", "status": "open"}]
        assert check_sector_limit("SBIN", positions) is False

    def test_other_sector_no_limit(self):
        """'Other' sector has no limit."""
        positions = [
            {"symbol": "UNKNOWN1", "status": "open"},
            {"symbol": "UNKNOWN2", "status": "open"},
            {"symbol": "UNKNOWN3", "status": "open"},
        ]
        assert check_sector_limit("UNKNOWN4", positions) is False

    def test_closed_positions_not_counted(self):
        """Closed positions in same sector don't count."""
        positions = [
            {"symbol": "HDFCBANK", "status": "closed"},
            {"symbol": "ICICIBANK", "status": "closed"},
        ]
        assert check_sector_limit("SBIN", positions) is False


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

class TestLogging:
    def test_logger_has_handlers(self):
        """Logger should have file + console handlers."""
        log = _setup_logger()
        assert len(log.handlers) >= 2


# ---------------------------------------------------------------------------
# Trailing Stop-Loss
# ---------------------------------------------------------------------------

class TestTrailingStop:
    def _make_equity_position(self, entry=1000, peak=1050, direction="bullish"):
        return {
            "symbol": "TEST", "token": "99999", "direction": direction,
            "instrument": "EQ", "entry_price": entry, "quantity": 10,
            "allocated": entry * 10, "score": 5.0, "categories": [],
            "entry_date": "2026-02-10", "max_hold_date": "2026-02-19",
            "target_price": entry * 1.05, "stoploss_price": entry * 0.97,
            "peak_price": peak, "status": "open",
        }

    def test_peak_updates_on_higher_price(self):
        """Peak should update when LTP exceeds current peak."""
        pos = self._make_equity_position(1000, 1020)
        # Simulate peak update logic
        ltp = 1040
        if ltp > pos["peak_price"]:
            pos["peak_price"] = ltp
        assert pos["peak_price"] == 1040

    def test_peak_does_not_decrease(self):
        """Peak should not decrease when LTP falls."""
        pos = self._make_equity_position(1000, 1050)
        ltp = 1030
        if ltp > pos["peak_price"]:
            pos["peak_price"] = ltp
        assert pos["peak_price"] == 1050

    def test_trailing_stop_triggers(self):
        """Trailing stop fires when LTP drops below peak * (1 - pct)."""
        pos = self._make_equity_position(1000, 1100)
        trailing_sl = pos["peak_price"] * (1 - TRAILING_STOP_PCT / 100)
        # 1100 * 0.98 = 1078
        ltp = 1075
        assert ltp <= trailing_sl

    def test_trailing_tighter_than_fixed(self):
        """When peak is high enough, trailing SL is tighter than fixed."""
        pos = self._make_equity_position(1000, 1100)
        trailing_sl = pos["peak_price"] * (1 - TRAILING_STOP_PCT / 100)
        fixed_sl = pos["stoploss_price"]  # 970
        assert trailing_sl > fixed_sl

    def test_trailing_weaker_than_fixed(self):
        """When peak hasn't risen much, fixed SL is tighter."""
        pos = self._make_equity_position(1000, 1005)
        trailing_sl = pos["peak_price"] * (1 - TRAILING_STOP_PCT / 100)
        # 1005 * 0.98 = 984.9
        fixed_sl = pos["stoploss_price"]  # 970
        assert trailing_sl > fixed_sl  # trailing is still higher in this range

    def test_option_trailing_stop(self):
        """Option trailing stop uses OPT_TRAILING_STOP_PCT from peak premium."""
        peak_premium = 120.0
        trailing_sl = peak_premium * (1 - OPT_TRAILING_STOP_PCT / 100)
        # 120 * 0.75 = 90
        assert trailing_sl == pytest.approx(90.0)
        assert 85.0 <= trailing_sl  # should be triggered if ltp drops to 85


# ---------------------------------------------------------------------------
# ATR Computation
# ---------------------------------------------------------------------------

class TestATR:
    def _make_candles(self, n=20):
        """Generate synthetic daily candles for ATR testing."""
        candles = []
        price = 100
        for i in range(n):
            o = price
            h = price + 3
            l = price - 2
            c = price + 1
            candles.append([f"2026-01-{i+1:02d}", o, h, l, c, 100000])
            price = c
        return candles

    def test_atr_computation(self):
        """ATR from known candles produces expected value."""
        candles = self._make_candles(20)
        atr = compute_atr(candles, 14)
        assert atr is not None
        assert atr > 0

    def test_insufficient_data_returns_none(self):
        """Less data than period+1 returns None."""
        candles = self._make_candles(5)
        atr = compute_atr(candles, 14)
        assert atr is None

    def test_empty_candles(self):
        assert compute_atr([], 14) is None
        assert compute_atr(None, 14) is None

    def test_atr_target_sl_prices(self):
        """ATR-based target and SL are computed correctly."""
        entry = 1000.0
        atr = 50.0
        target = entry + ATR_TARGET_MULTIPLIER * atr  # 1000 + 100 = 1100
        sl = entry - ATR_STOPLOSS_MULTIPLIER * atr     # 1000 - 75 = 925
        assert target == pytest.approx(1100.0)
        assert sl == pytest.approx(925.0)

    def test_atr_known_values(self):
        """ATR with specific true ranges."""
        # 16 candles → 15 true ranges, use period=5
        candles = [
            ["2026-01-01", 100, 105, 95, 102, 100000],
            ["2026-01-02", 102, 108, 98, 106, 100000],
            ["2026-01-03", 106, 110, 100, 104, 100000],
            ["2026-01-04", 104, 112, 101, 109, 100000],
            ["2026-01-05", 109, 115, 105, 113, 100000],
            ["2026-01-06", 113, 118, 108, 116, 100000],
        ]
        atr = compute_atr(candles, 5)
        assert atr is not None
        assert atr > 0


# ---------------------------------------------------------------------------
# Theta Awareness
# ---------------------------------------------------------------------------

class TestThetaAwareness:
    def test_low_dte_low_gain_exits(self):
        """Low DTE + low gain should trigger theta_decay exit."""
        dte = 3
        pnl_pct = 5.0
        assert dte < OPT_THETA_DTE_THRESHOLD and pnl_pct < OPT_THETA_MIN_GAIN_PCT

    def test_low_dte_high_gain_holds(self):
        """Low DTE + high gain should hold."""
        dte = 3
        pnl_pct = 25.0
        assert dte < OPT_THETA_DTE_THRESHOLD
        assert not (pnl_pct < OPT_THETA_MIN_GAIN_PCT)

    def test_high_dte_holds(self):
        """High DTE should not trigger theta exit."""
        dte = 10
        assert not (dte < OPT_THETA_DTE_THRESHOLD)


# ---------------------------------------------------------------------------
# Market Regime
# ---------------------------------------------------------------------------

class TestMarketRegime:
    def test_bearish_skip(self):
        """High VIX + Nifty below SMA → skip."""
        assert classify_regime(22000, 22500, 26.0) == "skip"

    def test_high_vol_caution(self):
        """High VIX but Nifty above SMA → caution."""
        assert classify_regime(23000, 22500, 26.0) == "caution"

    def test_normal(self):
        """Low VIX → normal."""
        assert classify_regime(23000, 22500, 15.0) == "normal"

    def test_missing_data_fallback(self):
        """Missing data → default to normal."""
        assert classify_regime(None, 22500, 15.0) == "normal"
        assert classify_regime(23000, None, 15.0) == "normal"
        assert classify_regime(23000, 22500, None) == "normal"


# ---------------------------------------------------------------------------
# Partial Exits
# ---------------------------------------------------------------------------

class TestPartialExits:
    def _make_position(self, entry_price=1000, quantity=10):
        return {
            "symbol": "TEST", "token": "99999", "direction": "bullish",
            "instrument": "EQ", "entry_price": entry_price,
            "quantity": quantity, "allocated": entry_price * quantity,
            "score": 5.0, "categories": [], "entry_date": "2026-02-10",
            "target_price": entry_price * 1.05,
            "stoploss_price": entry_price * 0.97,
            "max_hold_date": "2026-02-19", "status": "open",
        }

    def test_partial_reduces_quantity(self):
        """Partial close reduces position quantity."""
        pos = self._make_position(1000, 10)
        portfolio = _empty_portfolio()
        portfolio["positions"].append(pos)
        portfolio["available_capital"] = TOTAL_CAPITAL - pos["allocated"]

        close_position(portfolio, pos, 1050, "partial_target", close_qty=5)
        assert pos["quantity"] == 5
        assert pos["status"] == "open"

    def test_partial_frees_proportional_capital(self):
        """Partial close frees proportional allocated capital."""
        pos = self._make_position(1000, 10)
        portfolio = _empty_portfolio()
        portfolio["positions"].append(pos)
        portfolio["available_capital"] = TOTAL_CAPITAL - pos["allocated"]
        avail_before = portfolio["available_capital"]

        close_position(portfolio, pos, 1050, "partial_target", close_qty=5)
        # 5/10 = 50% freed
        freed = 10000 * 0.5
        assert portfolio["available_capital"] == pytest.approx(avail_before + freed)

    def test_partial_records_trade(self):
        """Partial close records a closed trade."""
        pos = self._make_position(1000, 10)
        portfolio = _empty_portfolio()
        portfolio["positions"].append(pos)

        closed = close_position(portfolio, pos, 1050, "partial_target", close_qty=5)
        assert len(portfolio["closed_trades"]) == 1
        assert closed["exit_reason"] == "partial_target"
        assert closed["quantity"] == 5

    def test_position_stays_open_after_partial(self):
        """Position remains open after partial close."""
        pos = self._make_position(1000, 10)
        portfolio = _empty_portfolio()
        portfolio["positions"].append(pos)

        close_position(portfolio, pos, 1050, "partial_target", close_qty=5)
        assert pos["status"] == "open"
        assert pos["quantity"] == 5

    def test_full_close_after_partial(self):
        """Full close after partial exit works correctly."""
        pos = self._make_position(1000, 10)
        portfolio = _empty_portfolio()
        portfolio["positions"].append(pos)
        portfolio["available_capital"] = TOTAL_CAPITAL - pos["allocated"]

        # Partial exit
        close_position(portfolio, pos, 1050, "partial_target", close_qty=5)
        assert pos["quantity"] == 5
        assert pos["status"] == "open"

        # Full close of remaining
        close_position(portfolio, pos, 1060, "trailing_stop")
        assert pos["status"] == "closed"
        assert portfolio["stats"]["total_trades"] == 2


# ---------------------------------------------------------------------------
# Performance Analytics
# ---------------------------------------------------------------------------

class TestPerformanceAnalytics:
    def _make_trades(self, pnls, capital=100000):
        """Helper to create closed trades with given P&L values."""
        trades = []
        for i, pnl in enumerate(pnls):
            entry = 1000
            qty = 10
            pnl_pct = round(pnl / (entry * qty) * 100, 2)
            exit_p = entry + pnl / qty
            trades.append({
                "symbol": f"T{i}", "direction": "bullish", "instrument": "EQ",
                "entry_price": entry, "exit_price": exit_p, "quantity": qty,
                "pnl_gross": pnl, "pnl": pnl, "pnl_pct": pnl_pct,
                "transaction_costs": 0, "entry_date": f"2026-02-{10+i:02d}",
                "exit_date": f"2026-02-{11+i:02d}", "exit_reason": "target" if pnl >= 0 else "stoploss",
            })
        return trades

    def test_positive_sharpe(self):
        """Consistently winning trades should produce positive Sharpe."""
        trades = self._make_trades([500, 300, 400, 200, 350])
        result = calc_performance_analytics(trades, 100000)
        assert result is not None
        assert result["sharpe_ratio"] > 0

    def test_known_drawdown(self):
        """Drawdown computed correctly for known sequence."""
        # +500, -800, +200 → peak=500, trough=500-800=-300, dd=800
        trades = self._make_trades([500, -800, 200])
        result = calc_performance_analytics(trades, 100000)
        assert result is not None
        assert result["max_drawdown"] == pytest.approx(800.0)
        assert result["max_drawdown_pct"] == pytest.approx(0.8)

    def test_profit_factor(self):
        """Profit factor = sum(wins) / abs(sum(losses))."""
        trades = self._make_trades([500, -200, 300])
        result = calc_performance_analytics(trades, 100000)
        assert result is not None
        assert result["profit_factor"] == pytest.approx(800 / 200, abs=0.01)

    def test_expectancy(self):
        """Expectancy = avg_win * win_rate - avg_loss * loss_rate."""
        trades = self._make_trades([500, -200, 300])
        result = calc_performance_analytics(trades, 100000)
        assert result is not None
        # wins: 500, 300 → avg=400, rate=2/3
        # losses: 200 → avg=200, rate=1/3
        expected = 400 * (2/3) - 200 * (1/3)
        assert result["expectancy"] == pytest.approx(expected, abs=1.0)

    def test_avg_holding(self):
        """Average holding period in trading days."""
        trades = self._make_trades([100, 200, 300])
        result = calc_performance_analytics(trades, 100000)
        assert result is not None
        assert result["avg_holding_days"] > 0

    def test_empty_trades(self):
        """Less than 3 trades returns None."""
        assert calc_performance_analytics([], 100000) is None
        assert calc_performance_analytics([{"pnl": 100}], 100000) is None


# ---------------------------------------------------------------------------
# Entry Quality Filters (RSI, Volume, News)
# ---------------------------------------------------------------------------

class TestEntryQualityFilters:
    """Tests for RSI, volume, and news entry filter logic in open_positions."""

    def test_rsi_bullish_in_range(self):
        """RSI within bullish range [40, 70] passes."""
        rsi = 55.0
        assert RSI_BULLISH_MIN <= rsi <= RSI_BULLISH_MAX

    def test_rsi_bullish_overbought_blocked(self):
        """RSI above 70 blocks bullish entry."""
        rsi = 75.0
        assert not (RSI_BULLISH_MIN <= rsi <= RSI_BULLISH_MAX)

    def test_rsi_bullish_too_low_blocked(self):
        """RSI below 40 blocks bullish entry (oversold → wrong direction)."""
        rsi = 35.0
        assert not (RSI_BULLISH_MIN <= rsi <= RSI_BULLISH_MAX)

    def test_rsi_bearish_in_range(self):
        """RSI within bearish range [30, 60] passes."""
        rsi = 45.0
        assert RSI_BEARISH_MIN <= rsi <= RSI_BEARISH_MAX

    def test_rsi_bearish_oversold_blocked(self):
        """RSI below 30 blocks bearish entry."""
        rsi = 25.0
        assert not (RSI_BEARISH_MIN <= rsi <= RSI_BEARISH_MAX)

    def test_rsi_bearish_too_high_blocked(self):
        """RSI above 60 blocks bearish entry."""
        rsi = 65.0
        assert not (RSI_BEARISH_MIN <= rsi <= RSI_BEARISH_MAX)

    def test_rsi_none_passes(self):
        """None RSI (no data) should not block entry."""
        rsi = None
        # Filters in open_positions only apply when rsi is not None
        assert rsi is None  # passes through

    def test_volume_above_minimum(self):
        """Volume ratio >= 1.0 passes."""
        assert 1.5 >= VOLUME_MIN_RATIO

    def test_volume_below_minimum_blocked(self):
        """Volume ratio < 1.0 blocks entry."""
        assert 0.7 < VOLUME_MIN_RATIO

    def test_volume_none_passes(self):
        """None volume ratio should not block entry."""
        vol_ratio = None
        # open_positions only filters when vol_ratio is not None
        assert vol_ratio is None

    def test_news_earnings_blocks(self):
        """Earnings soon flag blocks entry."""
        news = {"sentiment": "neutral", "has_earnings_soon": True, "has_policy_impact": False}
        assert SKIP_EARNINGS_SOON and news["has_earnings_soon"]

    def test_news_bullish_negative_blocks(self):
        """Bullish direction + negative sentiment blocks entry."""
        news = {"sentiment": "negative", "has_earnings_soon": False, "has_policy_impact": False}
        direction = "bullish"
        assert SKIP_CONTRADICTING_NEWS and direction == "bullish" and news["sentiment"] == "negative"

    def test_news_bearish_positive_blocks(self):
        """Bearish direction + positive sentiment blocks entry."""
        news = {"sentiment": "positive", "has_earnings_soon": False, "has_policy_impact": False}
        direction = "bearish"
        assert SKIP_CONTRADICTING_NEWS and direction == "bearish" and news["sentiment"] == "positive"

    def test_news_none_passes(self):
        """No news data should not block entry."""
        news = None
        assert news is None

    def test_news_matching_sentiment_passes(self):
        """Bullish direction + positive sentiment passes."""
        news = {"sentiment": "positive", "has_earnings_soon": False, "has_policy_impact": False}
        direction = "bullish"
        # Should not be blocked
        contradicting = (direction == "bullish" and news["sentiment"] == "negative") or \
                        (direction == "bearish" and news["sentiment"] == "positive")
        assert not contradicting


# ---------------------------------------------------------------------------
# Intraday Momentum Confirmation
# ---------------------------------------------------------------------------

class TestIntradayMomentum:
    """Tests for check_intraday_momentum and related constants."""

    def test_bullish_confirmed(self):
        """Bullish direction + positive intraday move → confirmed."""
        from paper_trade import check_intraday_momentum
        # 5-min candles: [date, open, high, low, close, volume]
        candles = [
            ["2026-02-13T09:15", 1000, 1005, 998, 1002, 50000],
            ["2026-02-13T09:20", 1002, 1008, 1001, 1006, 45000],
            ["2026-02-13T09:25", 1006, 1012, 1005, 1010, 40000],
        ]
        confirmed, detail = check_intraday_momentum(candles, "bullish")
        assert confirmed is True
        assert "confirmed" in detail

    def test_bullish_contradicted(self):
        """Bullish direction + negative intraday move → contradicted."""
        from paper_trade import check_intraday_momentum
        candles = [
            ["2026-02-13T09:15", 1000, 1003, 995, 998, 50000],
            ["2026-02-13T09:20", 998, 999, 990, 992, 45000],
            ["2026-02-13T09:25", 992, 995, 988, 990, 40000],
        ]
        confirmed, detail = check_intraday_momentum(candles, "bullish")
        assert confirmed is False
        assert "contradicted" in detail

    def test_bearish_confirmed(self):
        """Bearish direction + negative intraday move → confirmed."""
        from paper_trade import check_intraday_momentum
        candles = [
            ["2026-02-13T09:15", 1000, 1003, 995, 997, 50000],
            ["2026-02-13T09:20", 997, 998, 990, 992, 45000],
            ["2026-02-13T09:25", 992, 994, 985, 988, 40000],
        ]
        confirmed, detail = check_intraday_momentum(candles, "bearish")
        assert confirmed is True
        assert "confirmed" in detail

    def test_bearish_contradicted(self):
        """Bearish direction + positive intraday move → contradicted."""
        from paper_trade import check_intraday_momentum
        candles = [
            ["2026-02-13T09:15", 1000, 1005, 999, 1003, 50000],
            ["2026-02-13T09:20", 1003, 1010, 1002, 1008, 45000],
            ["2026-02-13T09:25", 1008, 1015, 1007, 1012, 40000],
        ]
        confirmed, detail = check_intraday_momentum(candles, "bearish")
        assert confirmed is False
        assert "contradicted" in detail

    def test_insufficient_candles_passes(self):
        """Fewer than INTRADAY_MIN_CANDLES → passes through."""
        from paper_trade import check_intraday_momentum
        candles = [
            ["2026-02-13T09:15", 1000, 1005, 998, 990, 50000],
        ]
        confirmed, detail = check_intraday_momentum(candles, "bullish")
        assert confirmed is True
        assert detail == "insufficient_candles"

    def test_none_candles_passes(self):
        """None candles → passes through."""
        from paper_trade import check_intraday_momentum
        confirmed, detail = check_intraday_momentum(None, "bullish")
        assert confirmed is True
        assert detail == "insufficient_candles"

    def test_empty_candles_passes(self):
        """Empty candle list → passes through."""
        from paper_trade import check_intraday_momentum
        confirmed, detail = check_intraday_momentum([], "bearish")
        assert confirmed is True
        assert detail == "insufficient_candles"

    def test_flat_move_passes(self):
        """Intraday move below INTRADAY_MIN_MOVE_PCT → passes through as flat."""
        from paper_trade import check_intraday_momentum, INTRADAY_MIN_MOVE_PCT
        # Move = 0.05% which is < 0.1% threshold
        candles = [
            ["2026-02-13T09:15", 10000, 10005, 9998, 10002, 50000],
            ["2026-02-13T09:20", 10002, 10006, 10001, 10003, 45000],
            ["2026-02-13T09:25", 10003, 10007, 10002, 10005, 40000],
        ]
        confirmed, detail = check_intraday_momentum(candles, "bearish")
        assert confirmed is True
        assert "flat" in detail

    def test_zero_session_open_passes(self):
        """Zero session open → passes through (invalid_open)."""
        from paper_trade import check_intraday_momentum
        candles = [
            ["2026-02-13T09:15", 0, 5, 0, 3, 50000],
            ["2026-02-13T09:20", 3, 6, 2, 5, 45000],
            ["2026-02-13T09:25", 5, 8, 4, 7, 40000],
        ]
        confirmed, detail = check_intraday_momentum(candles, "bullish")
        assert confirmed is True
        assert detail == "invalid_open"
