import datetime
import pytest
import pandas as pd
import numpy as np


def _make_chain(spot=23000.0, vix=14.0, dte=30):
    from fo_data import generate_synthetic_chain
    return generate_synthetic_chain(spot=spot, vix=vix, dte=dte, symbol="NIFTY")


class TestFuturesStrategy:
    def test_should_enter_bullish_high_score(self):
        from fo_strategies import FuturesStrategy
        strat = FuturesStrategy()
        result = strat.should_enter(
            date=datetime.date(2025, 6, 15), spot=23000.0, vix=14.0, atr=200.0,
            score=5.0, direction="bullish", available_capital=1000000.0, current_positions=[],
        )
        assert result is not None
        assert result["direction"] == "bullish"
        assert result["instrument"] == "FUT"
        assert result["entry_price"] > 0
        assert result["quantity"] > 0
        assert result["target_price"] > result["entry_price"]
        assert result["stoploss_price"] < result["entry_price"]

    def test_should_not_enter_low_score(self):
        from fo_strategies import FuturesStrategy
        strat = FuturesStrategy()
        result = strat.should_enter(
            date=datetime.date(2025, 6, 15), spot=23000.0, vix=14.0, atr=200.0,
            score=2.0, direction="bullish", available_capital=1000000.0, current_positions=[],
        )
        assert result is None

    def test_should_enter_bearish(self):
        from fo_strategies import FuturesStrategy
        strat = FuturesStrategy()
        result = strat.should_enter(
            date=datetime.date(2025, 6, 15), spot=23000.0, vix=14.0, atr=200.0,
            score=5.0, direction="bearish", available_capital=1000000.0, current_positions=[],
        )
        assert result is not None
        assert result["direction"] == "bearish"
        assert result["target_price"] < result["entry_price"]
        assert result["stoploss_price"] > result["entry_price"]

    def test_should_exit_target_hit(self):
        from fo_strategies import FuturesStrategy
        strat = FuturesStrategy()
        position = {
            "instrument": "FUT", "direction": "bullish", "entry_price": 23000.0,
            "target_price": 23300.0, "stoploss_price": 22300.0, "peak_price": 23200.0,
            "entry_date": datetime.date(2025, 6, 10), "max_hold_date": datetime.date(2025, 6, 25),
        }
        should_exit, reason = strat.should_exit(
            position=position, date=datetime.date(2025, 6, 12), spot=23350.0, high=23400.0, low=23200.0,
        )
        assert should_exit is True
        assert reason == "target"

    def test_should_exit_stoploss_hit(self):
        from fo_strategies import FuturesStrategy
        strat = FuturesStrategy()
        position = {
            "instrument": "FUT", "direction": "bullish", "entry_price": 23000.0,
            "target_price": 23300.0, "stoploss_price": 22300.0, "peak_price": 23000.0,
            "entry_date": datetime.date(2025, 6, 10), "max_hold_date": datetime.date(2025, 6, 25),
        }
        should_exit, reason = strat.should_exit(
            position=position, date=datetime.date(2025, 6, 12), spot=22200.0, high=22500.0, low=22100.0,
        )
        assert should_exit is True
        assert reason == "stoploss"

    def test_risk_based_sizing(self):
        from fo_strategies import FuturesStrategy
        strat = FuturesStrategy()
        result = strat.should_enter(
            date=datetime.date(2025, 6, 15), spot=23000.0, vix=14.0, atr=50.0,
            score=5.0, direction="bullish", available_capital=1000000.0, current_positions=[],
        )
        max_loss = abs(result["entry_price"] - result["stoploss_price"]) * result["quantity"]
        assert max_loss <= 20000 * 1.1


class TestSpreadStrategy:
    def test_bull_call_spread_entry(self):
        from fo_strategies import SpreadStrategy
        chain = _make_chain(spot=23000.0, vix=14.0, dte=35)
        strat = SpreadStrategy()
        result = strat.should_enter(
            date=datetime.date(2025, 6, 15), spot=23000.0, vix=14.0, atr=200.0,
            score=5.0, direction="bullish", available_capital=1000000.0,
            current_positions=[], chain=chain, dte=35,
        )
        assert result is not None
        assert result["instrument"] == "SPREAD"
        assert result["direction"] == "bullish"
        assert result["long_strike"] < result["short_strike"]
        assert result["net_debit"] > 0
        assert result["max_profit"] > 0

    def test_bear_put_spread_entry(self):
        from fo_strategies import SpreadStrategy
        chain = _make_chain(spot=23000.0, vix=14.0, dte=35)
        strat = SpreadStrategy()
        result = strat.should_enter(
            date=datetime.date(2025, 6, 15), spot=23000.0, vix=14.0, atr=200.0,
            score=5.0, direction="bearish", available_capital=1000000.0,
            current_positions=[], chain=chain, dte=35,
        )
        assert result is not None
        assert result["long_strike"] > result["short_strike"]

    def test_spread_not_entered_low_dte(self):
        from fo_strategies import SpreadStrategy
        chain = _make_chain(spot=23000.0, vix=14.0, dte=10)
        strat = SpreadStrategy()
        result = strat.should_enter(
            date=datetime.date(2025, 6, 15), spot=23000.0, vix=14.0, atr=200.0,
            score=5.0, direction="bullish", available_capital=1000000.0,
            current_positions=[], chain=chain, dte=10,
        )
        assert result is None

    def test_spread_exit_profit_cap(self):
        from fo_strategies import SpreadStrategy
        strat = SpreadStrategy()
        position = {
            "instrument": "SPREAD", "direction": "bullish", "net_debit": 50.0,
            "max_profit": 150.0, "entry_date": datetime.date(2025, 6, 10),
            "expiry": datetime.date(2025, 7, 24),
        }
        should_exit, reason = strat.should_exit(
            position=position, date=datetime.date(2025, 6, 20), current_spread_value=170.0,
        )
        assert should_exit is True
        assert reason == "profit_cap"

    def test_spread_max_risk_sizing(self):
        from fo_strategies import SpreadStrategy
        chain = _make_chain(spot=23000.0, vix=14.0, dte=35)
        strat = SpreadStrategy()
        result = strat.should_enter(
            date=datetime.date(2025, 6, 15), spot=23000.0, vix=14.0, atr=200.0,
            score=5.0, direction="bullish", available_capital=1000000.0,
            current_positions=[], chain=chain, dte=35,
        )
        max_risk = result["net_debit"] * result["quantity"]
        assert max_risk <= 1000000.0 * 0.02 * 1.5  # some tolerance for lot rounding


class TestCondorStrategy:
    def test_condor_entry_normal_vix(self):
        from fo_strategies import CondorStrategy
        chain = _make_chain(spot=23000.0, vix=15.0, dte=30)
        strat = CondorStrategy()
        result = strat.should_enter(
            date=datetime.date(2025, 6, 15), spot=23000.0, vix=15.0,
            available_capital=1000000.0, current_positions=[], chain=chain, dte=30,
        )
        assert result is not None
        assert result["instrument"] == "CONDOR"
        assert result["net_credit"] > 0
        assert result["put_short"] < result["call_short"]

    def test_condor_rejected_high_vix(self):
        from fo_strategies import CondorStrategy
        chain = _make_chain(spot=23000.0, vix=25.0, dte=30)
        strat = CondorStrategy()
        result = strat.should_enter(
            date=datetime.date(2025, 6, 15), spot=23000.0, vix=25.0,
            available_capital=1000000.0, current_positions=[], chain=chain, dte=30,
        )
        assert result is None

    def test_condor_rejected_low_vix(self):
        from fo_strategies import CondorStrategy
        chain = _make_chain(spot=23000.0, vix=10.0, dte=30)
        strat = CondorStrategy()
        result = strat.should_enter(
            date=datetime.date(2025, 6, 15), spot=23000.0, vix=10.0,
            available_capital=1000000.0, current_positions=[], chain=chain, dte=30,
        )
        assert result is None

    def test_condor_exit_profit_target(self):
        from fo_strategies import CondorStrategy
        strat = CondorStrategy()
        position = {
            "instrument": "CONDOR", "net_credit": 100.0,
            "entry_date": datetime.date(2025, 6, 10), "expiry": datetime.date(2025, 7, 24),
        }
        should_exit, reason = strat.should_exit(
            position=position, date=datetime.date(2025, 6, 20), current_condor_value=40.0,
        )
        assert should_exit is True
        assert reason == "profit_target"


class TestMomentumStrategy:
    def test_momentum_entry_high_conviction(self):
        from fo_strategies import MomentumStrategy
        chain = _make_chain(spot=23000.0, vix=14.0, dte=10)
        strat = MomentumStrategy()
        result = strat.should_enter(
            date=datetime.date(2025, 6, 15), spot=23000.0, vix=14.0,
            score=6.0, direction="bullish", available_capital=1000000.0,
            current_positions=[], chain=chain, dte=10,
        )
        assert result is not None
        assert result["instrument"] == "MOMENTUM"
        assert result["option_type"] == "CE"

    def test_momentum_rejected_low_score(self):
        from fo_strategies import MomentumStrategy
        chain = _make_chain(spot=23000.0, vix=14.0, dte=10)
        strat = MomentumStrategy()
        result = strat.should_enter(
            date=datetime.date(2025, 6, 15), spot=23000.0, vix=14.0,
            score=4.0, direction="bullish", available_capital=1000000.0,
            current_positions=[], chain=chain, dte=10,
        )
        assert result is None

    def test_momentum_exit_target(self):
        from fo_strategies import MomentumStrategy
        strat = MomentumStrategy()
        position = {"instrument": "MOMENTUM", "entry_premium": 100.0, "entry_date": datetime.date(2025, 6, 15)}
        should_exit, reason = strat.should_exit(position=position, date=datetime.date(2025, 6, 16), current_premium=195.0)
        assert should_exit is True
        assert reason == "target"

    def test_momentum_exit_stoploss(self):
        from fo_strategies import MomentumStrategy
        strat = MomentumStrategy()
        position = {"instrument": "MOMENTUM", "entry_premium": 100.0, "entry_date": datetime.date(2025, 6, 15)}
        should_exit, reason = strat.should_exit(position=position, date=datetime.date(2025, 6, 16), current_premium=60.0)
        assert should_exit is True
        assert reason == "stoploss"

    def test_momentum_exit_time(self):
        from fo_strategies import MomentumStrategy
        strat = MomentumStrategy()
        position = {"instrument": "MOMENTUM", "entry_premium": 100.0, "entry_date": datetime.date(2025, 6, 15)}
        should_exit, reason = strat.should_exit(position=position, date=datetime.date(2025, 6, 19), current_premium=110.0)
        assert should_exit is True
        assert reason == "time_exit"
