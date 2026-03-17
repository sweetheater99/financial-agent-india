"""F&O backtest strategies.

Four strategy classes with common interface:
- FuturesStrategy: directional futures trades
- SpreadStrategy: vertical spreads (bull call / bear put)
- CondorStrategy: iron condors on index
- MomentumStrategy: ATM options for quick directional bets
"""

import datetime
import math

import pandas as pd

from fo_data import get_lot_size, get_futures_price, get_nearest_expiry


def bid_ask_spread_pct(strike: float, spot: float) -> float:
    """Bid-ask spread % that widens with OTM distance. ~2% ATM, ~8% at 500pt OTM."""
    otm_dist = abs(strike - spot) / spot
    return 0.02 + 3.0 * otm_dist


FUT_TARGET_ATR_MULT = 1.5
FUT_SL_ATR_MULT = 3.5
FUT_MARGIN_PCT = 0.15
FUT_MAX_RISK_PCT = 0.02
FUT_MAX_HOLD_DAYS = 15
FUT_SLIPPAGE_PCT = 0.0005

ENTRY_SCORE_THRESHOLD = 2.5
HIGH_CONVICTION_THRESHOLD = 3.5


class BaseStrategy:
    """Common interface for all F&O strategies."""
    name: str = "base"

    def should_enter(self, **kwargs) -> dict | None:
        raise NotImplementedError

    def should_exit(self, **kwargs) -> tuple[bool, str]:
        raise NotImplementedError


class FuturesStrategy(BaseStrategy):
    name = "futures"

    def __init__(
        self,
        score_threshold: float = ENTRY_SCORE_THRESHOLD,
        target_atr_mult: float = FUT_TARGET_ATR_MULT,
        sl_atr_mult: float = FUT_SL_ATR_MULT,
        max_risk_pct: float = FUT_MAX_RISK_PCT,
        max_hold_days: int = FUT_MAX_HOLD_DAYS,
    ):
        self.score_threshold = score_threshold
        self.target_atr_mult = target_atr_mult
        self.sl_atr_mult = sl_atr_mult
        self.max_risk_pct = max_risk_pct
        self.max_hold_days = max_hold_days

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
    ) -> dict | None:
        if score < self.score_threshold or direction is None:
            return None
        if atr <= 0:
            return None
        if vix > 28:
            return None

        lot_size = get_lot_size(symbol, date)
        expiry = get_nearest_expiry(date, min_dte=5)
        dte = (expiry - date).days

        fut_price = get_futures_price(spot, dte)
        if direction == "bullish":
            entry_price = fut_price * (1 + FUT_SLIPPAGE_PCT)
        else:
            entry_price = fut_price * (1 - FUT_SLIPPAGE_PCT)

        if direction == "bullish":
            target = entry_price + self.target_atr_mult * atr
            stoploss = entry_price - self.sl_atr_mult * atr
        else:
            target = entry_price - self.target_atr_mult * atr
            stoploss = entry_price + self.sl_atr_mult * atr

        risk_per_lot = abs(entry_price - stoploss) * lot_size
        max_loss = available_capital * self.max_risk_pct
        num_lots = max(1, int(max_loss / risk_per_lot))

        margin_needed = num_lots * lot_size * fut_price * FUT_MARGIN_PCT
        while margin_needed > available_capital * 0.5 and num_lots > 1:
            num_lots -= 1
            margin_needed = num_lots * lot_size * fut_price * FUT_MARGIN_PCT

        max_hold_date = date + datetime.timedelta(days=self.max_hold_days)
        if max_hold_date > expiry:
            max_hold_date = expiry - datetime.timedelta(days=1)

        return {
            "instrument": "FUT",
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


# ---------------------------------------------------------------------------
# Spread Strategy
# ---------------------------------------------------------------------------

SPREAD_MIN_DTE = 30
SPREAD_MAX_DTE = 45
SPREAD_MAX_RISK_PCT = 0.02
SPREAD_PROFIT_CAP_PCT = 0.80
SPREAD_TIME_EXIT_DAYS = 5
SPREAD_SL_MULTIPLIER = 2.0


class SpreadStrategy(BaseStrategy):
    name = "spread"

    def __init__(
        self,
        score_threshold: float = ENTRY_SCORE_THRESHOLD,
        min_dte: int = SPREAD_MIN_DTE,
        max_dte: int = SPREAD_MAX_DTE,
        max_risk_pct: float = SPREAD_MAX_RISK_PCT,
        profit_cap_pct: float = SPREAD_PROFIT_CAP_PCT,
        sl_multiplier: float = SPREAD_SL_MULTIPLIER,
        time_exit_days: int = SPREAD_TIME_EXIT_DAYS,
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
    ) -> dict | None:
        if score < self.score_threshold or direction is None:
            return None
        if dte < self.min_dte or dte > self.max_dte:
            return None
        if chain is None or chain.empty:
            return None

        lot_size = get_lot_size(symbol, date)
        interval = 50 if symbol == "NIFTY" else 100
        atm_strike = round(spot / interval) * interval
        spread_width = max(interval * 2, round(atr / interval) * interval)

        if direction == "bullish":
            long_strike = atm_strike
            short_strike = atm_strike + spread_width
            opt_type = "CE"
        else:
            long_strike = atm_strike
            short_strike = atm_strike - spread_width
            opt_type = "PE"

        long_row = chain[(chain["strike"] == long_strike) & (chain["option_type"] == opt_type)]
        short_row = chain[(chain["strike"] == short_strike) & (chain["option_type"] == opt_type)]

        if long_row.empty or short_row.empty:
            return None

        # Bid-ask: buy long leg at ask (higher), sell short leg at bid (lower)
        long_premium = float(long_row.iloc[0]["premium"]) * (1 + bid_ask_spread_pct(long_strike, spot))
        short_premium = float(short_row.iloc[0]["premium"]) * (1 - bid_ask_spread_pct(short_strike, spot))
        net_debit = long_premium - short_premium

        if net_debit <= 0:
            return None

        max_profit = abs(long_strike - short_strike) - net_debit
        if max_profit <= 0:
            return None

        max_risk = available_capital * self.max_risk_pct
        num_lots = max(1, int(max_risk / (net_debit * lot_size)))

        expiry = get_nearest_expiry(date, min_dte=self.min_dte)

        return {
            "instrument": "SPREAD",
            "symbol": symbol,
            "direction": direction,
            "option_type": opt_type,
            "long_strike": long_strike,
            "short_strike": short_strike,
            "long_premium": round(long_premium, 2),
            "short_premium": round(short_premium, 2),
            "net_debit": round(net_debit, 2),
            "max_profit": round(max_profit, 2),
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
    ) -> tuple[bool, str]:
        net_debit = position["net_debit"]
        max_profit = position["max_profit"]
        expiry = position["expiry"]

        current_pnl = current_spread_value - net_debit

        if current_pnl >= self.profit_cap_pct * max_profit:
            return True, "profit_cap"

        if current_pnl <= -self.sl_multiplier * net_debit:
            return True, "stoploss"

        days_to_expiry = (expiry - date).days
        if days_to_expiry <= self.time_exit_days:
            return True, "time_exit"

        return False, ""


# ---------------------------------------------------------------------------
# Condor Strategy (Iron Condor)
# ---------------------------------------------------------------------------

CONDOR_MIN_VIX = 18
CONDOR_MAX_VIX = 25
CONDOR_MAX_RISK_PCT = 0.02
CONDOR_TARGET_PCT = 0.50
CONDOR_SL_MULTIPLIER = 2.0
CONDOR_TIME_EXIT_DAYS = 5
CONDOR_OTM_POINTS_NIFTY = 500
CONDOR_WING_WIDTH_NIFTY = 300


class CondorStrategy(BaseStrategy):
    name = "condor"

    def __init__(
        self,
        min_vix: float = CONDOR_MIN_VIX,
        max_vix: float = CONDOR_MAX_VIX,
        max_risk_pct: float = CONDOR_MAX_RISK_PCT,
        target_pct: float = CONDOR_TARGET_PCT,
        sl_multiplier: float = CONDOR_SL_MULTIPLIER,
        time_exit_days: int = CONDOR_TIME_EXIT_DAYS,
    ):
        self.min_vix = min_vix
        self.max_vix = max_vix
        self.max_risk_pct = max_risk_pct
        self.target_pct = target_pct
        self.sl_multiplier = sl_multiplier
        self.time_exit_days = time_exit_days

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
    ) -> dict | None:
        if vix < self.min_vix or vix > self.max_vix:
            return None
        if chain is None or chain.empty:
            return None
        if dte < 15:
            return None

        lot_size = get_lot_size(symbol, date)
        interval = 50 if symbol == "NIFTY" else 100
        atm = round(spot / interval) * interval
        otm_pts = CONDOR_OTM_POINTS_NIFTY
        wing_width = CONDOR_WING_WIDTH_NIFTY

        call_short = atm + otm_pts
        put_short = atm - otm_pts
        call_long = call_short + wing_width
        put_long = put_short - wing_width

        def _get_premium(strike, opt_type):
            row = chain[(chain["strike"] == strike) & (chain["option_type"] == opt_type)]
            return float(row.iloc[0]["premium"]) if not row.empty else 0.0

        # Short legs: sell at bid (lower). Long legs: buy at ask (higher).
        call_short_prem = _get_premium(call_short, "CE") * (1 - bid_ask_spread_pct(call_short, spot))
        call_long_prem = _get_premium(call_long, "CE") * (1 + bid_ask_spread_pct(call_long, spot))
        put_short_prem = _get_premium(put_short, "PE") * (1 - bid_ask_spread_pct(put_short, spot))
        put_long_prem = _get_premium(put_long, "PE") * (1 + bid_ask_spread_pct(put_long, spot))

        if call_short_prem <= 0 or put_short_prem <= 0:
            return None

        net_credit = (call_short_prem + put_short_prem) - (call_long_prem + put_long_prem)
        if net_credit <= 0:
            return None

        max_risk_per_lot = wing_width - net_credit
        # Guard: credit > wing_width is impossible in real markets (synthetic pricing artifact)
        if max_risk_per_lot <= 0:
            return None

        max_risk = available_capital * self.max_risk_pct
        num_lots = max(1, int(max_risk / (max_risk_per_lot * lot_size)))

        expiry = get_nearest_expiry(date, min_dte=15)

        return {
            "instrument": "CONDOR",
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
# Momentum Strategy (ATM options, quick directional)
# ---------------------------------------------------------------------------

MOMENTUM_MAX_RISK_PCT = 0.01
MOMENTUM_SL_PCT = 0.35
MOMENTUM_TARGET_PCT = 0.90
MOMENTUM_TIME_EXIT_DAYS = 3


class MomentumStrategy(BaseStrategy):
    name = "momentum"

    def __init__(
        self,
        score_threshold: float = HIGH_CONVICTION_THRESHOLD,
        max_risk_pct: float = MOMENTUM_MAX_RISK_PCT,
        sl_pct: float = MOMENTUM_SL_PCT,
        target_pct: float = MOMENTUM_TARGET_PCT,
        time_exit_days: int = MOMENTUM_TIME_EXIT_DAYS,
    ):
        self.score_threshold = score_threshold
        self.max_risk_pct = max_risk_pct
        self.sl_pct = sl_pct
        self.target_pct = target_pct
        self.time_exit_days = time_exit_days

    def should_enter(
        self,
        date: datetime.date,
        spot: float,
        vix: float,
        score: float,
        direction: str | None,
        available_capital: float,
        current_positions: list[dict],
        chain: pd.DataFrame = None,
        dte: int = 0,
        symbol: str = "NIFTY",
    ) -> dict | None:
        if score < self.score_threshold or direction is None:
            return None
        if chain is None or chain.empty:
            return None
        if dte < 7 or dte > 14:
            return None

        lot_size = get_lot_size(symbol, date)
        interval = 50 if symbol == "NIFTY" else 100
        atm = round(spot / interval) * interval
        opt_type = "CE" if direction == "bullish" else "PE"

        atm_row = chain[(chain["strike"] == atm) & (chain["option_type"] == opt_type)]
        if atm_row.empty:
            return None

        premium = float(atm_row.iloc[0]["premium"])

        max_risk = available_capital * self.max_risk_pct
        num_lots = max(1, int(max_risk / (premium * lot_size)))

        return {
            "instrument": "MOMENTUM",
            "symbol": symbol,
            "direction": direction,
            "option_type": opt_type,
            "strike": atm,
            "entry_premium": round(premium, 2),
            "quantity": num_lots * lot_size,
            "num_lots": num_lots,
            "lot_size": lot_size,
            "entry_date": date,
            "score": score,
        }

    def should_exit(
        self,
        position: dict,
        date: datetime.date,
        current_premium: float = 0.0,
    ) -> tuple[bool, str]:
        entry_premium = position["entry_premium"]
        entry_date = position["entry_date"]

        pnl_pct = (current_premium - entry_premium) / entry_premium

        if pnl_pct >= self.target_pct:
            return True, "target"

        if pnl_pct <= -self.sl_pct:
            return True, "stoploss"

        days_held = (date - entry_date).days
        if days_held >= self.time_exit_days:
            return True, "time_exit"

        return False, ""
