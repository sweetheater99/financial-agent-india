"""V7 Backtest Adapter — runs V7 strategies through FOBacktestEngine.

Maps V7 SetupTypes to existing FO strategies with V7-specific parameters:
- Conviction-based position sizing (HIGH=2%, MEDIUM=1.5%, LOW=0.75%)
- V7 risk limits (4% max daily risk, 4 trades/day, survival mode)
- V7 strike selection parameters (delta 0.40-0.50, VIX filters)
- Breakout detection from support/resistance levels

Usage:
    python -m v7.backtest_adapter --symbol NIFTY --period 2y
    python -m v7.backtest_adapter --symbol NIFTY --period 1y --capital 300000
    python -m v7.backtest_adapter --symbol NIFTY --period 2y --output results.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import compute_atr
from backtest_signals import compute_signals
from fo_backtest import FOBacktestEngine, MAX_CONCURRENT_POSITIONS, CASH_RESERVE_PCT
from fo_data import (
    fetch_real_chain,
    generate_synthetic_chain,
    get_futures_price,
    get_lot_size,
    get_nearest_expiry,
    calc_futures_round_trip,
    calc_options_round_trip,
)
from fo_strategies import (
    BaseStrategy,
    FuturesStrategy,
    SpreadStrategy,
    CondorStrategy,
    bid_ask_spread_pct,
)
from greeks import black_scholes_greeks
from v7.config_v7 import (
    CAPITAL,
    RISK_LIMITS,
    THETA_LIMITS,
    STRIKE_FILTERS,
    TRAILING,
)

logger = logging.getLogger("v7_backtest")

# ---------------------------------------------------------------------------
# V7-tuned strategy overrides
# ---------------------------------------------------------------------------

# V7 uses tighter parameters than the generic FO backtest
V7_STRATEGY_CAPS = {
    "breakout": 0.30,       # breakout_long + breakout_short
    "credit_spread": 0.25,  # credit_spread_bull + credit_spread_bear
    "iron_condor": 0.20,    # theta engine condors
}

# Conviction → risk % per trade
CONVICTION_RISK = {"high": 2.0, "medium": 1.5, "low": 0.75}


class V7BreakoutStrategy(BaseStrategy):
    """Breakout strategy tuned to V7 parameters.

    Detects breakouts from recent highs/lows and enters futures positions.
    Maps to V7 SetupTypes: BREAKOUT_LONG, BREAKOUT_SHORT.
    """
    name = "breakout"

    def __init__(
        self,
        lookback: int = 20,
        score_threshold: float = 3.0,
        target_atr_mult: float = float(TRAILING["atr_multiplier"]),
        sl_atr_mult: float = 2.5,
        max_risk_pct: float = RISK_LIMITS["max_per_trade_risk_pct"] / 100,
        max_hold_days: int = 10,
        max_daily_trades: int = RISK_LIMITS["max_trades_per_day"],
    ):
        self.lookback = lookback
        self.score_threshold = score_threshold
        self.target_atr_mult = target_atr_mult
        self.sl_atr_mult = sl_atr_mult
        self.max_risk_pct = max_risk_pct
        self.max_hold_days = max_hold_days
        self.max_daily_trades = max_daily_trades
        self._trades_today: dict[str, int] = {}  # date_str → count
        self._daily_pnl: dict[str, float] = {}

    def _get_trades_today(self, date: datetime.date) -> int:
        return self._trades_today.get(str(date), 0)

    def _record_trade(self, date: datetime.date):
        key = str(date)
        self._trades_today[key] = self._trades_today.get(key, 0) + 1

    def _record_exit_pnl(self, date: datetime.date, pnl: float):
        key = str(date)
        self._daily_pnl[key] = self._daily_pnl.get(key, 0) + pnl

    def _daily_loss_exceeded(self, date: datetime.date, capital: float) -> bool:
        key = str(date)
        daily = self._daily_pnl.get(key, 0)
        return daily < -(capital * 0.02)

    def should_enter(
        self,
        date: datetime.date,
        spot: float,
        vix: float,
        atr: float,
        score: float,
        direction: str | None,
        available_capital: float,
        current_positions: list[dict],
        symbol: str = "NIFTY",
        high_20: float = 0.0,
        low_20: float = 0.0,
        **kwargs,
    ) -> dict | None:
        # V7 risk gates
        if self._get_trades_today(date) >= self.max_daily_trades:
            return None
        if self._daily_loss_exceeded(date, available_capital / (1 - CASH_RESERVE_PCT)):
            return None

        if score < self.score_threshold or direction is None:
            return None
        if atr <= 0:
            return None
        if vix > 28:
            return None

        # Breakout detection: price near recent high/low
        if direction == "bullish" and high_20 > 0:
            if spot < high_20 * 0.995:  # must be within 0.5% of 20-day high
                return None
        elif direction == "bearish" and low_20 > 0:
            if spot > low_20 * 1.005:
                return None

        lot_size = get_lot_size(symbol, date)
        expiry = get_nearest_expiry(date, min_dte=5)
        dte = (expiry - date).days

        fut_price = get_futures_price(spot, dte)
        slippage = 0.0005
        entry_price = fut_price * (1 + slippage) if direction == "bullish" else fut_price * (1 - slippage)

        if direction == "bullish":
            target = entry_price + self.target_atr_mult * atr
            stoploss = entry_price - self.sl_atr_mult * atr
        else:
            target = entry_price - self.target_atr_mult * atr
            stoploss = entry_price + self.sl_atr_mult * atr

        # V7 conviction-based sizing (use medium by default, high if score > 5)
        conviction = "high" if score >= 5.0 else "medium" if score >= 3.5 else "low"
        risk_pct = CONVICTION_RISK[conviction] / 100
        risk_per_lot = abs(entry_price - stoploss) * lot_size
        max_loss = available_capital * risk_pct
        num_lots = max(1, int(max_loss / risk_per_lot))

        margin_needed = num_lots * lot_size * fut_price * 0.15
        while margin_needed > available_capital * 0.5 and num_lots > 1:
            num_lots -= 1
            margin_needed = num_lots * lot_size * fut_price * 0.15

        max_hold_date = date + datetime.timedelta(days=self.max_hold_days)
        if max_hold_date > expiry:
            max_hold_date = expiry - datetime.timedelta(days=1)

        self._record_trade(date)

        return {
            "instrument": "FUT",
            "strategy": "breakout",
            "v7_setup_type": f"breakout_{'long' if direction == 'bullish' else 'short'}",
            "v7_conviction": conviction,
            "symbol": symbol,
            "direction": direction,
            "entry_price": round(entry_price, 2),
            "quantity": num_lots * lot_size,
            "num_lots": num_lots,
            "lot_size": lot_size,
            "target_price": round(target, 2),
            "stoploss_price": round(stoploss, 2),
            "peak_price": entry_price,
            "entry_date": date,
            "max_hold_date": max_hold_date,
            "expiry": expiry,
            "margin_used": round(margin_needed, 2),
            "score": score,
        }

    def should_exit(
        self,
        position: dict,
        date: datetime.date,
        spot: float,
        high: float,
        low: float,
        **kwargs,
    ) -> tuple[bool, str]:
        direction = position["direction"]
        target = position["target_price"]
        stoploss = position["stoploss_price"]

        if direction == "bullish":
            if high >= target:
                return True, "target"
            if low <= stoploss:
                return True, "stoploss"
        else:
            if low <= target:
                return True, "target"
            if high >= stoploss:
                return True, "stoploss"

        if date >= position["max_hold_date"]:
            return True, "max_hold"
        if date >= position.get("expiry", date + datetime.timedelta(days=30)):
            return True, "expiry"

        return False, ""


class V7CreditSpreadStrategy(BaseStrategy):
    """Credit spread strategy tuned to V7 parameters.

    Maps to V7 SetupTypes: CREDIT_SPREAD_BULL, CREDIT_SPREAD_BEAR.
    Uses V7's spread_sell_delta (0.25) instead of generic scoring.
    """
    name = "credit_spread"

    def __init__(
        self,
        score_threshold: float = 3.0,
        min_dte: int = 15,
        max_dte: int = 35,
        max_risk_pct: float = RISK_LIMITS["max_per_trade_risk_pct"] / 100,
        profit_cap_pct: float = 0.70,
        sl_multiplier: float = 2.0,
        time_exit_days: int = 3,
    ):
        self.score_threshold = score_threshold
        self.min_dte = min_dte
        self.max_dte = max_dte
        self.max_risk_pct = max_risk_pct
        self.profit_cap_pct = profit_cap_pct
        self.sl_multiplier = sl_multiplier
        self.time_exit_days = time_exit_days

    def should_enter(
        self,
        date: datetime.date,
        spot: float,
        vix: float,
        atr: float,
        score: float,
        direction: str | None,
        available_capital: float,
        current_positions: list[dict],
        chain: pd.DataFrame = None,
        dte: int = 0,
        symbol: str = "NIFTY",
        **kwargs,
    ) -> dict | None:
        if score < self.score_threshold or direction is None:
            return None
        if dte < self.min_dte or dte > self.max_dte:
            return None
        if chain is None or chain.empty:
            return None

        lot_size = get_lot_size(symbol, date)
        interval = 50 if symbol == "NIFTY" else 100
        atm = round(spot / interval) * interval

        # V7 uses 2-strike-wide spreads
        spread_width = interval * 2

        if direction == "bullish":
            # Bull put spread: sell higher put, buy lower put
            short_strike = atm - round(atr * 0.5 / interval) * interval
            long_strike = short_strike - spread_width
            opt_type = "PE"
        else:
            # Bear call spread: sell lower call, buy higher call
            short_strike = atm + round(atr * 0.5 / interval) * interval
            long_strike = short_strike + spread_width
            opt_type = "CE"

        short_row = chain[(chain["strike"] == short_strike) & (chain["option_type"] == opt_type)]
        long_row = chain[(chain["strike"] == long_strike) & (chain["option_type"] == opt_type)]
        if short_row.empty or long_row.empty:
            return None

        short_prem = float(short_row.iloc[0]["premium"]) * (1 - bid_ask_spread_pct(short_strike, spot))
        long_prem = float(long_row.iloc[0]["premium"]) * (1 + bid_ask_spread_pct(long_strike, spot))

        net_credit = short_prem - long_prem
        if net_credit <= 5:
            return None

        max_risk_per_lot = spread_width - net_credit
        if max_risk_per_lot <= 0:
            return None

        conviction = "high" if score >= 5.0 else "medium" if score >= 3.5 else "low"
        risk_pct = CONVICTION_RISK[conviction] / 100
        max_risk = available_capital * risk_pct
        num_lots = max(1, int(max_risk / (max_risk_per_lot * lot_size)))

        expiry = get_nearest_expiry(date, min_dte=self.min_dte)

        return {
            "instrument": "SPREAD",
            "strategy": "credit_spread",
            "v7_setup_type": f"credit_spread_{'bull' if direction == 'bullish' else 'bear'}",
            "v7_conviction": conviction,
            "symbol": symbol,
            "direction": direction,
            "option_type": opt_type,
            "long_strike": long_strike,
            "short_strike": short_strike,
            "long_premium": round(long_prem, 2),
            "short_premium": round(short_prem, 2),
            "net_credit": round(net_credit, 2),
            "net_debit": round(-net_credit, 2),  # negative for credit spread
            "max_profit": round(net_credit, 2),
            "max_risk": round(max_risk_per_lot, 2),
            "quantity": num_lots * lot_size,
            "num_lots": num_lots,
            "lot_size": lot_size,
            "entry_date": date,
            "expiry": expiry,
            "score": score,
        }

    def should_exit(
        self,
        position: dict,
        date: datetime.date,
        current_spread_value: float = 0.0,
        **kwargs,
    ) -> tuple[bool, str]:
        net_credit = position["net_credit"]
        expiry = position["expiry"]

        # For credit spreads: profit = credit received - current cost to close
        profit = net_credit - current_spread_value

        if profit >= self.profit_cap_pct * net_credit:
            return True, "profit_target"

        if profit <= -self.sl_multiplier * net_credit:
            return True, "stoploss"

        days_to_expiry = (expiry - date).days
        if days_to_expiry <= self.time_exit_days:
            return True, "time_exit"

        return False, ""


class V7IronCondorStrategy(BaseStrategy):
    """Iron condor strategy tuned to V7 theta engine parameters.

    Maps to V7 SetupType: IRON_CONDOR.
    Uses V7's VIX range (14-20), delta (0.20), wing gap (200pts).
    """
    name = "iron_condor"

    def __init__(
        self,
        min_vix: float = THETA_LIMITS["min_vix"],
        max_vix: float = THETA_LIMITS["max_vix"],
        max_risk_pct: float = THETA_LIMITS["max_risk_pct"] / 100,
        target_pct: float = THETA_LIMITS["profit_target_pct"],
        sl_multiplier: float = 2.0,
        time_exit_days: int = 2,
        wing_gap: int = THETA_LIMITS["wing_gap_nifty"],
    ):
        self.min_vix = min_vix
        self.max_vix = max_vix
        self.max_risk_pct = max_risk_pct
        self.target_pct = target_pct
        self.sl_multiplier = sl_multiplier
        self.time_exit_days = time_exit_days
        self.wing_gap = wing_gap

    def should_enter(
        self,
        date: datetime.date,
        spot: float,
        vix: float,
        available_capital: float,
        current_positions: list[dict],
        chain: pd.DataFrame = None,
        dte: int = 0,
        symbol: str = "NIFTY",
        **kwargs,
    ) -> dict | None:
        if vix < self.min_vix or vix > self.max_vix:
            return None
        if chain is None or chain.empty:
            return None
        if dte < 7:
            return None
        # V7 theta engine prefers weekly condors — limit DTE
        if dte > 14:
            return None

        lot_size = get_lot_size(symbol, date)
        interval = 50 if symbol == "NIFTY" else 100
        atm = round(spot / interval) * interval

        # V7 uses wider OTM distance for safety
        otm_pts = round(spot * 0.025 / interval) * interval  # ~2.5% OTM
        otm_pts = max(otm_pts, 400)

        call_short = atm + otm_pts
        put_short = atm - otm_pts
        call_long = call_short + self.wing_gap
        put_long = put_short - self.wing_gap

        def _get_premium(strike, opt_type):
            row = chain[(chain["strike"] == strike) & (chain["option_type"] == opt_type)]
            return float(row.iloc[0]["premium"]) if not row.empty else 0.0

        call_short_prem = _get_premium(call_short, "CE") * (1 - bid_ask_spread_pct(call_short, spot))
        call_long_prem = _get_premium(call_long, "CE") * (1 + bid_ask_spread_pct(call_long, spot))
        put_short_prem = _get_premium(put_short, "PE") * (1 - bid_ask_spread_pct(put_short, spot))
        put_long_prem = _get_premium(put_long, "PE") * (1 + bid_ask_spread_pct(put_long, spot))

        if call_short_prem <= 0 or put_short_prem <= 0:
            return None

        net_credit = (call_short_prem + put_short_prem) - (call_long_prem + put_long_prem)
        if net_credit <= 0:
            return None

        max_risk_per_lot = self.wing_gap - net_credit
        if max_risk_per_lot <= 0:
            return None

        max_risk = available_capital * self.max_risk_pct
        num_lots = max(1, int(max_risk / (max_risk_per_lot * lot_size)))

        expiry = get_nearest_expiry(date, min_dte=7)

        return {
            "instrument": "CONDOR",
            "strategy": "iron_condor",
            "v7_setup_type": "iron_condor",
            "v7_conviction": "medium",
            "symbol": symbol,
            "call_short": call_short,
            "call_long": call_long,
            "put_short": put_short,
            "put_long": put_long,
            "call_short_premium": round(call_short_prem, 2),
            "call_long_premium": round(call_long_prem, 2),
            "put_short_premium": round(put_short_prem, 2),
            "put_long_premium": round(put_long_prem, 2),
            "net_credit": round(net_credit, 2),
            "max_risk": round(max_risk_per_lot, 2),
            "quantity": num_lots * lot_size,
            "num_lots": num_lots,
            "lot_size": lot_size,
            "entry_date": date,
            "expiry": expiry,
        }

    def should_exit(
        self,
        position: dict,
        date: datetime.date,
        current_condor_value: float = 0.0,
        **kwargs,
    ) -> tuple[bool, str]:
        net_credit = position["net_credit"]
        expiry = position["expiry"]

        profit = net_credit - current_condor_value
        if profit >= self.target_pct * net_credit:
            return True, "profit_target"
        if profit <= -self.sl_multiplier * net_credit:
            return True, "stoploss"

        days_to_expiry = (expiry - date).days
        if days_to_expiry <= self.time_exit_days:
            return True, "time_exit"

        return False, ""


# ---------------------------------------------------------------------------
# V7 Backtest Engine (extends FOBacktestEngine with V7 strategies)
# ---------------------------------------------------------------------------

class V7BacktestEngine(FOBacktestEngine):
    """FOBacktestEngine with V7-specific strategies and risk management."""

    def __init__(self, capital: float = CAPITAL["initial"]):
        super().__init__(capital=capital)
        # Replace generic strategies with V7-tuned ones
        self.strategies = {
            "breakout": V7BreakoutStrategy(),
            "credit_spread": V7CreditSpreadStrategy(),
            "iron_condor": V7IronCondorStrategy(),
        }
        self.strategy_caps = V7_STRATEGY_CAPS

    def _strategy_within_cap(self, strategy_name: str) -> bool:
        cap = self.strategy_caps.get(strategy_name, 0.15)
        return self._strategy_deployed(strategy_name) < self.capital * cap

    def _strategy_deployed(self, strategy_name: str) -> float:
        inst_map = {
            "breakout": "FUT",
            "credit_spread": "SPREAD",
            "iron_condor": "CONDOR",
        }
        target_inst = inst_map.get(strategy_name, "")
        return sum(
            self._deployed_capital(p)
            for p in self.positions
            if p.get("instrument") == target_inst
        )

    def _check_entries(
        self,
        date: datetime.date,
        spot: float,
        vix: float,
        atr: float,
        score: float,
        direction: str | None,
        chain: pd.DataFrame,
        dte: int,
        symbol: str,
    ):
        """Try V7 strategies with V7 risk gates."""
        if len(self.positions) >= RISK_LIMITS["max_concurrent_positions"]:
            return

        available = self._available_capital()
        if available <= 0:
            return

        # Compute 20-day high/low for breakout detection
        high_20 = getattr(self, '_current_high_20', 0.0)
        low_20 = getattr(self, '_current_low_20', 0.0)

        # Try breakout
        if self._strategy_within_cap("breakout"):
            entry = self.strategies["breakout"].should_enter(
                date=date, spot=spot, vix=vix, atr=atr,
                score=score, direction=direction,
                available_capital=available,
                current_positions=self.positions,
                symbol=symbol, high_20=high_20, low_20=low_20,
            )
            if entry and self._deployed_capital(entry) <= available:
                self.positions.append(entry)
                available = self._available_capital()

        # Try credit spread
        if self._strategy_within_cap("credit_spread") and len(self.positions) < RISK_LIMITS["max_concurrent_positions"]:
            entry = self.strategies["credit_spread"].should_enter(
                date=date, spot=spot, vix=vix, atr=atr,
                score=score, direction=direction,
                available_capital=available,
                current_positions=self.positions,
                chain=chain, dte=dte, symbol=symbol,
            )
            if entry and self._deployed_capital(entry) <= available:
                self.positions.append(entry)
                available = self._available_capital()

        # Try iron condor (no score/direction needed — theta play)
        if self._strategy_within_cap("iron_condor") and len(self.positions) < RISK_LIMITS["max_concurrent_positions"]:
            entry = self.strategies["iron_condor"].should_enter(
                date=date, spot=spot, vix=vix,
                available_capital=available,
                current_positions=self.positions,
                chain=chain, dte=dte, symbol=symbol,
            )
            if entry and self._deployed_capital(entry) <= available:
                self.positions.append(entry)

    def _check_exits(
        self,
        date: datetime.date,
        spot: float,
        high: float,
        low: float,
        vix: float,
        dte: int,
        chain: pd.DataFrame,
        symbol: str,
    ):
        """Check exits with V7 logic — credit spreads use different pricing."""
        to_close = []
        iv = vix / 100.0

        for idx, pos in enumerate(self.positions):
            inst = pos.get("instrument", "")
            should_exit = False
            reason = ""
            exit_price = spot

            if inst == "FUT":
                should_exit, reason = self.strategies["breakout"].should_exit(
                    position=pos, date=date, spot=spot, high=high, low=low,
                )
                if should_exit:
                    if reason == "target":
                        exit_price = pos["target_price"]
                    elif reason == "stoploss":
                        exit_price = pos["stoploss_price"]
                    else:
                        exit_price = spot

            elif inst == "SPREAD":
                pos_expiry = pos.get("expiry", date + datetime.timedelta(days=30))
                spread_dte = max(1, (pos_expiry - date).days)
                long_greeks = black_scholes_greeks(
                    spot, pos["long_strike"], spread_dte, iv=iv,
                    option_type=pos["option_type"],
                )
                short_greeks = black_scholes_greeks(
                    spot, pos["short_strike"], spread_dte, iv=iv,
                    option_type=pos["option_type"],
                )
                # Current cost to close: buy back short, sell long
                current_value = abs(short_greeks["theoretical_price"] - long_greeks["theoretical_price"])

                should_exit, reason = self.strategies["credit_spread"].should_exit(
                    position=pos, date=date, current_spread_value=current_value,
                )
                if should_exit:
                    exit_price = current_value

            elif inst == "CONDOR":
                pos_expiry = pos.get("expiry", date + datetime.timedelta(days=30))
                condor_dte = max(1, (pos_expiry - date).days)
                cs = black_scholes_greeks(spot, pos["call_short"], condor_dte, iv=iv, option_type="CE")
                cl = black_scholes_greeks(spot, pos["call_long"], condor_dte, iv=iv, option_type="CE")
                ps = black_scholes_greeks(spot, pos["put_short"], condor_dte, iv=iv, option_type="PE")
                pl = black_scholes_greeks(spot, pos["put_long"], condor_dte, iv=iv, option_type="PE")
                current_condor_value = (cs["theoretical_price"] + ps["theoretical_price"]) - (cl["theoretical_price"] + pl["theoretical_price"])

                should_exit, reason = self.strategies["iron_condor"].should_exit(
                    position=pos, date=date, current_condor_value=current_condor_value,
                )
                if should_exit:
                    exit_price = current_condor_value

            if should_exit:
                to_close.append((idx, reason, exit_price))

        for idx, reason, exit_price in sorted(to_close, reverse=True):
            self._close_position(idx, date, exit_price, reason, symbol)

    def run(self, data: pd.DataFrame, symbol: str = "NIFTY") -> dict:
        """Run V7 backtest with 20-day high/low tracking."""
        # Ensure columns
        col_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        for lc, uc in col_map.items():
            if uc not in data.columns and lc in data.columns:
                data[uc] = data[lc]
        for uc, lc in {v: k for k, v in col_map.items()}.items():
            if lc not in data.columns and uc in data.columns:
                data[lc] = data[uc]

        if "vix" not in data.columns:
            data["vix"] = 15.0

        try:
            signals = compute_signals(data)
        except Exception:
            signals = pd.DataFrame({"score": 0.0, "direction": None}, index=data.index)

        if "score" not in signals.columns:
            signals["score"] = 0.0
        if "direction" not in signals.columns:
            signals["direction"] = None

        atr_series = compute_atr(data)

        # Compute rolling 20-day high/low for breakout detection
        high_20 = data["High"].rolling(20, min_periods=1).max()
        low_20 = data["Low"].rolling(20, min_periods=1).min()

        # Reset state
        self.capital = self.initial_capital
        self.positions = []
        self.closed_trades = []
        self.daily_nav = {}
        self.exit_reasons = defaultdict(int)

        # Reset breakout strategy daily counters
        self.strategies["breakout"]._trades_today = {}
        self.strategies["breakout"]._daily_pnl = {}

        start_idx = min(30, len(data) - 1)
        for i in range(start_idx, len(data)):
            row = data.iloc[i]
            date = data.index[i]
            if isinstance(date, pd.Timestamp):
                date_d = date.date()
            else:
                date_d = date

            spot = float(row["Close"])
            high = float(row["High"])
            low = float(row["Low"])
            vix = float(row.get("vix", 15.0))
            atr_val = float(atr_series.iloc[i]) if not pd.isna(atr_series.iloc[i]) else 0.0
            score = float(signals.iloc[i].get("score", 0)) if i < len(signals) else 0.0
            direction = signals.iloc[i].get("direction", None) if i < len(signals) else None

            # Store 20-day high/low for breakout strategy
            self._current_high_20 = float(high_20.iloc[i]) if not pd.isna(high_20.iloc[i]) else 0.0
            self._current_low_20 = float(low_20.iloc[i]) if not pd.isna(low_20.iloc[i]) else 0.0

            expiry = get_nearest_expiry(date_d, min_dte=7)
            dte = max(1, (expiry - date_d).days)

            try:
                chain = fetch_real_chain(symbol, date_d)
                if chain is None:
                    chain = generate_synthetic_chain(spot, vix, dte, symbol=symbol)
            except Exception:
                chain = pd.DataFrame()

            self._check_exits(date_d, spot, high, low, vix, dte, chain, symbol)
            self._check_entries(date_d, spot, vix, atr_val, score, direction, chain, dte, symbol)

            nav = self.capital + self._total_deployed()
            self.daily_nav[str(date_d)] = round(nav, 2)

        # Close remaining
        if len(data) > 0:
            last_row = data.iloc[-1]
            last_date = data.index[-1]
            if isinstance(last_date, pd.Timestamp):
                last_date = last_date.date()
            self._close_all_remaining(last_date, float(last_row["Close"]), symbol)

        return {
            "trades": self.closed_trades,
            "stats": self._compute_stats(),
            "daily_nav": self.daily_nav,
            "v7_breakdown": self._v7_breakdown(),
        }

    def _v7_breakdown(self) -> dict:
        """V7-specific stats: by setup type, conviction, and strategy."""
        trades = self.closed_trades
        if not trades:
            return {}

        by_setup = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
        by_conviction = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})

        for t in trades:
            setup = t.get("v7_setup_type", t.get("strategy", "unknown"))
            conv = t.get("v7_conviction", "medium")

            by_setup[setup]["trades"] += 1
            by_setup[setup]["pnl"] += t["net_pnl"]
            if t["net_pnl"] > 0:
                by_setup[setup]["wins"] += 1

            by_conviction[conv]["trades"] += 1
            by_conviction[conv]["pnl"] += t["net_pnl"]
            if t["net_pnl"] > 0:
                by_conviction[conv]["wins"] += 1

        result = {"by_setup_type": {}, "by_conviction": {}}
        for key, data in by_setup.items():
            result["by_setup_type"][key] = {
                "trades": data["trades"],
                "win_rate": round(data["wins"] / data["trades"] * 100, 1) if data["trades"] else 0,
                "pnl": round(data["pnl"], 2),
            }
        for key, data in by_conviction.items():
            result["by_conviction"][key] = {
                "trades": data["trades"],
                "win_rate": round(data["wins"] / data["trades"] * 100, 1) if data["trades"] else 0,
                "pnl": round(data["pnl"], 2),
            }
        return result


# ---------------------------------------------------------------------------
# Report & CLI
# ---------------------------------------------------------------------------

def print_v7_report(results: dict):
    """Print V7-specific backtest report."""
    stats = results["stats"]
    v7 = results.get("v7_breakdown", {})

    print("\n" + "=" * 65)
    print("V7 PROFESSIONAL TRADER — BACKTEST REPORT")
    print("=" * 65)
    print(f"  Capital:        Rs.{CAPITAL['initial']:,}")
    print(f"  Total Trades:   {stats['total_trades']}")
    print(f"  Win Rate:       {stats['win_rate']:.1f}%")
    print(f"  Total P&L:      Rs.{stats['total_pnl']:,.2f}")
    print(f"  Return:         {stats['total_pnl'] / CAPITAL['initial'] * 100:.1f}%")
    print(f"  Sharpe Ratio:   {stats['sharpe']:.2f}")
    print(f"  Max Drawdown:   {stats['max_drawdown']:.2f}%")
    print(f"  Profit Factor:  {stats['profit_factor']:.2f}")

    if v7.get("by_setup_type"):
        print("\n  V7 Setup Type Breakdown:")
        for name, data in sorted(v7["by_setup_type"].items(), key=lambda x: -x[1]["pnl"]):
            print(f"    {name:25s}  trades={data['trades']:3d}  "
                  f"win={data['win_rate']:5.1f}%  "
                  f"pnl=Rs.{data['pnl']:,.2f}")

    if v7.get("by_conviction"):
        print("\n  V7 Conviction Breakdown:")
        for name, data in sorted(v7["by_conviction"].items()):
            print(f"    {name:10s}  trades={data['trades']:3d}  "
                  f"win={data['win_rate']:5.1f}%  "
                  f"pnl=Rs.{data['pnl']:,.2f}")

    if stats.get("exit_reasons"):
        print("\n  Exit Reasons:")
        for reason, count in sorted(stats["exit_reasons"].items(), key=lambda x: -x[1]):
            print(f"    {reason:15s}  {count}")

    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="V7 Strategy Backtester")
    parser.add_argument("--symbol", default="NIFTY", help="Symbol to backtest")
    parser.add_argument("--period", default="2y", help="Data period (e.g., 1y, 2y)")
    parser.add_argument("--capital", type=float, default=CAPITAL["initial"], help="Starting capital")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()

    from fo_data import fetch_spot_vix_history

    print(f"Fetching {args.symbol} data for {args.period}...")
    data = fetch_spot_vix_history(symbol=args.symbol, period=args.period)
    if data.empty:
        print("ERROR: No data fetched.")
        sys.exit(1)

    print(f"Running V7 backtest on {len(data)} bars (capital: Rs.{args.capital:,.0f})...")
    engine = V7BacktestEngine(capital=args.capital)
    results = engine.run(data, symbol=args.symbol)

    print_v7_report(results)

    if args.output:
        output = {
            "stats": results["stats"],
            "v7_breakdown": results.get("v7_breakdown", {}),
            "daily_nav": results["daily_nav"],
            "trades": [
                {k: str(v) if isinstance(v, (datetime.date, datetime.datetime)) else v
                 for k, v in t.items()}
                for t in results["trades"]
            ],
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
