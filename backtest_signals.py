"""Signal-filtered backtester — replays technical-indicator proxies for screener
confluence on historical data, entering only when multiple signals align.

Unlike backtest.py (which enters at every bar), this module computes RSI,
volume spikes, EMA momentum, ATR breakouts, and trend confirmation as proxies
for the real F&O screener signals, and only enters when the combined score
exceeds a configurable threshold.

Exit logic, transaction costs, and report format mirror backtest.py.

Usage:
    python backtest_signals.py --symbols RELIANCE,TCS --period 1y
    python backtest_signals.py --nifty50 --period 1y
    python backtest_signals.py --nifty50 --period 2y --min-score 4.0 --atr-target 2.0
    python backtest_signals.py --nifty50 --period 1y --output results.json
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from backtest import (
    NIFTY50_SYMBOLS,
    DEFAULT_PARAMS,
    EQ_SLIPPAGE_PCT,
    FUT_SLIPPAGE_PCT,
    FUT_MARGIN_PCT,
    calc_round_trip_costs,
    calc_fut_round_trip_costs,
    compute_atr,
    fetch_data,
)
from indicators_v4 import compute_mfi, compute_obv_divergence, compute_vwap_deviation, compute_adx

# ---------------------------------------------------------------------------
# Nifty 50 sector mapping (for sector momentum filter)
# ---------------------------------------------------------------------------
NIFTY50_SECTORS = {
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking",
    "KOTAKBANK": "Banking", "AXISBANK": "Banking", "INDUSINDBK": "Banking",
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT", "TECHM": "IT",
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma", "APOLLOHOSP": "Pharma",
    "MARUTI": "Auto", "M&M": "Auto", "BAJAJ-AUTO": "Auto",
    "HEROMOTOCO": "Auto", "EICHERMOT": "Auto", "TRENT": "Auto",
    "TATASTEEL": "Metals", "HINDALCO": "Metals", "JSWSTEEL": "Metals",
    "RELIANCE": "Energy", "ONGC": "Energy", "NTPC": "Energy",
    "POWERGRID": "Energy", "BPCL": "Energy", "COALINDIA": "Energy",
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "TITAN": "FMCG", "ASIANPAINT": "FMCG",
    "BAJFINANCE": "FinServices", "BAJAJFINSV": "FinServices",
    "HDFCLIFE": "FinServices", "SBILIFE": "FinServices", "SHRIRAMFIN": "FinServices",
    "LT": "Infra", "ULTRACEMCO": "Infra", "GRASIM": "Infra",
    "ADANIENT": "Infra", "ADANIPORTS": "Infra",
    "BHARTIARTL": "Telecom",
}

# VIX-based position size multipliers (Indian F&O trader standard)
VIX_SIZE_MAP = [
    (12, 1.0),   # Low VIX: full size
    (18, 1.0),   # Normal: full size
    (22, 0.5),   # Elevated: half size
    (28, 0.25),  # High: quarter size
    (99, 0.0),   # Extreme: no trading
]

# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

# Signal weights (proxy for screener weights)
SIGNAL_WEIGHTS = {
    "momentum": 1.0,
    "volume": 1.0,
    "rsi": 1.0,
    "atr_breakout": 1.5,
    "trend_confirm": 2.0,
    "mfi": 1.0,
    "obv_divergence": 1.0,
    "vwap_deviation": 0.5,
    "adx_trend": 1.5,
}

DEFAULT_MIN_SCORE = 3.5


def compute_signals(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """Compute all technical-indicator signals for every bar.

    Returns a DataFrame aligned to *df* with columns:
        direction    : 'bullish' | 'bearish' | None
        score        : float (sum of weighted signals that fired)
        signals_fired: list[str]
    plus individual boolean columns for each signal.
    """
    # Flatten MultiIndex columns from yfinance (e.g. ("Close", "RELIANCE.NS") -> scalar)
    def _col(name):
        col = df[name]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        return col

    close = _col("Close")
    high = _col("High")
    low = _col("Low")
    volume = _col("Volume")

    # --- Indicators ---
    ema20 = close.ewm(span=20, adjust=False).mean()
    atr = compute_atr(df, atr_period)
    day_range = high - low

    # RSI (14-period Wilder)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    # Handle zero/negative-zero avg_loss: when no losses, RSI = 100
    avg_loss_safe = avg_loss.where(avg_loss.abs() > 1e-10, float("nan"))
    rs = avg_gain / avg_loss_safe
    rsi = 100 - (100 / (1 + rs))
    # Where avg_loss is ~0 (all gains), RSI should be 100
    rsi = rsi.fillna(100.0)

    # Volume ratio (current / 20-day SMA)
    vol_sma20 = volume.rolling(20, min_periods=20).mean()
    vol_ratio = volume / vol_sma20.replace(0, float("nan"))

    # 3-day price trend + volume increasing (OI proxy)
    price_up_3d = close > close.shift(3)
    price_down_3d = close < close.shift(3)
    vol_increasing = (volume > volume.shift(1)) & (volume.shift(1) > volume.shift(2))

    # --- V4 indicators (need lowercase columns) ---
    df_lc = pd.DataFrame({
        "open": _col("Open"), "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=df.index)
    mfi = compute_mfi(df_lc, period=14)
    obv_div = compute_obv_divergence(df_lc, lookback=10)
    vwap_dev = compute_vwap_deviation(df_lc)
    adx_result = compute_adx(df_lc, period=14)
    adx_val = adx_result["adx"]
    di_plus = adx_result["di_plus"]
    di_minus = adx_result["di_minus"]

    # --- Per-bar signal evaluation ---
    n = len(df)
    directions = [None] * n
    scores = [0.0] * n
    signals_lists = [[] for _ in range(n)]

    sig_momentum = [False] * n
    sig_volume = [False] * n
    sig_rsi = [False] * n
    sig_atr_breakout = [False] * n
    sig_trend_confirm = [False] * n
    sig_mfi = [False] * n
    sig_obv_divergence = [False] * n
    sig_vwap_deviation = [False] * n
    sig_adx_trend = [False] * n

    for i in range(n):
        # Need at least 20 bars of history for all indicators
        if pd.isna(ema20.iloc[i]) or pd.isna(atr.iloc[i]) or pd.isna(rsi.iloc[i]) or pd.isna(vol_sma20.iloc[i]):
            continue

        c = float(close.iloc[i])
        e = float(ema20.iloc[i])
        r = float(rsi.iloc[i])
        vr = float(vol_ratio.iloc[i]) if not pd.isna(vol_ratio.iloc[i]) else 0
        dr = float(day_range.iloc[i])
        a = float(atr.iloc[i])

        if a <= 0 or c <= 0:
            continue

        # Determine raw direction from price vs EMA + volume
        above_ema = c > e
        below_ema = c < e
        vol_spike = vr > 1.3

        # Direction: bullish if close > EMA20 + volume spike; bearish if close < EMA20 + volume spike
        # If no volume spike, we still allow direction but volume signal won't fire
        if above_ema:
            direction = "bullish"
        elif below_ema:
            direction = "bearish"
        else:
            continue

        fired = []
        score = 0.0

        # 1. Price momentum: close vs 20-day EMA
        if (direction == "bullish" and above_ema) or (direction == "bearish" and below_ema):
            fired.append("momentum")
            score += SIGNAL_WEIGHTS["momentum"]
            sig_momentum[i] = True

        # 2. Volume spike: volume > 1.3x 20-day average
        if vol_spike:
            fired.append("volume")
            score += SIGNAL_WEIGHTS["volume"]
            sig_volume[i] = True

        # 3. RSI filter
        #    Bullish: RSI 40-65 (oversold recovery + momentum, not overbought)
        #    Bearish: RSI 35-60 (not yet oversold bounce territory)
        if direction == "bullish" and 40 <= r <= 65:
            fired.append("rsi")
            score += SIGNAL_WEIGHTS["rsi"]
            sig_rsi[i] = True
        elif direction == "bearish" and 35 <= r <= 60:
            fired.append("rsi")
            score += SIGNAL_WEIGHTS["rsi"]
            sig_rsi[i] = True

        # 4. ATR breakout: today's range > 1.2x ATR
        if dr > 1.2 * a:
            fired.append("atr_breakout")
            score += SIGNAL_WEIGHTS["atr_breakout"]
            sig_atr_breakout[i] = True

        # 5. Trend confirmation (OI proxy): 3-day price trend + volume increasing
        if direction == "bullish" and not pd.isna(price_up_3d.iloc[i]) and price_up_3d.iloc[i]:
            if not pd.isna(vol_increasing.iloc[i]) and vol_increasing.iloc[i]:
                fired.append("trend_confirm")
                score += SIGNAL_WEIGHTS["trend_confirm"]
                sig_trend_confirm[i] = True
        elif direction == "bearish" and not pd.isna(price_down_3d.iloc[i]) and price_down_3d.iloc[i]:
            if not pd.isna(vol_increasing.iloc[i]) and vol_increasing.iloc[i]:
                fired.append("trend_confirm")
                score += SIGNAL_WEIGHTS["trend_confirm"]
                sig_trend_confirm[i] = True

        # 6. MFI: oversold buying (bullish) or overbought selling (bearish)
        if not pd.isna(mfi.iloc[i]):
            m = float(mfi.iloc[i])
            if direction == "bullish" and m < 30:
                fired.append("mfi")
                score += SIGNAL_WEIGHTS["mfi"]
                sig_mfi[i] = True
            elif direction == "bearish" and m > 70:
                fired.append("mfi")
                score += SIGNAL_WEIGHTS["mfi"]
                sig_mfi[i] = True

        # 7. OBV divergence: confirms direction
        if i < len(obv_div) and obv_div.iloc[i] != "none":
            od = obv_div.iloc[i]
            if direction == "bullish" and od == "bullish_div":
                fired.append("obv_divergence")
                score += SIGNAL_WEIGHTS["obv_divergence"]
                sig_obv_divergence[i] = True
            elif direction == "bearish" and od == "bearish_div":
                fired.append("obv_divergence")
                score += SIGNAL_WEIGHTS["obv_divergence"]
                sig_obv_divergence[i] = True

        # 8. VWAP deviation: oversold bounce / extended warning
        if not pd.isna(vwap_dev.iloc[i]):
            vd = float(vwap_dev.iloc[i])
            if direction == "bullish" and vd < -1.0:
                fired.append("vwap_deviation")
                score += SIGNAL_WEIGHTS["vwap_deviation"]
                sig_vwap_deviation[i] = True
            elif direction == "bearish" and vd > 1.0:
                fired.append("vwap_deviation")
                score += SIGNAL_WEIGHTS["vwap_deviation"]
                sig_vwap_deviation[i] = True

        # 9. ADX trend strength: strong trend confirms direction
        if not pd.isna(adx_val.iloc[i]):
            ax = float(adx_val.iloc[i])
            dp = float(di_plus.iloc[i]) if not pd.isna(di_plus.iloc[i]) else 0
            dm = float(di_minus.iloc[i]) if not pd.isna(di_minus.iloc[i]) else 0
            if ax > 25:
                if (direction == "bullish" and dp > dm) or (direction == "bearish" and dm > dp):
                    fired.append("adx_trend")
                    score += SIGNAL_WEIGHTS["adx_trend"]
                    sig_adx_trend[i] = True

        # ADX dampening: choppy market reduces all scores
        if not pd.isna(adx_val.iloc[i]) and float(adx_val.iloc[i]) < 20:
            score *= 0.7

        directions[i] = direction
        scores[i] = score
        signals_lists[i] = fired

    result = pd.DataFrame({
        "direction": directions,
        "score": scores,
        "sig_momentum": sig_momentum,
        "sig_volume": sig_volume,
        "sig_rsi": sig_rsi,
        "sig_atr_breakout": sig_atr_breakout,
        "sig_trend_confirm": sig_trend_confirm,
        "sig_mfi": sig_mfi,
        "sig_obv_divergence": sig_obv_divergence,
        "sig_vwap_deviation": sig_vwap_deviation,
        "sig_adx_trend": sig_adx_trend,
    }, index=df.index)
    result["signals_fired"] = signals_lists
    return result


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

class SignalBacktestEngine:
    """Backtests equity strategy with signal-filtered entries."""

    def __init__(self, params: dict | None = None, min_score: float = DEFAULT_MIN_SCORE):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.min_score = min_score
        self.trades: list[dict] = []
        self.total_bars = 0
        self.entry_bars = 0
        self.signal_counts: dict[str, int] = {
            "momentum": 0,
            "volume": 0,
            "rsi": 0,
            "atr_breakout": 0,
            "trend_confirm": 0,
            "mfi": 0,
            "obv_divergence": 0,
            "vwap_deviation": 0,
            "adx_trend": 0,
        }

    def run(self, symbol: str, df: pd.DataFrame) -> list[dict]:
        """Run signal-filtered backtest on a single symbol."""
        # Flatten MultiIndex columns from yfinance if present
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = [c[0] for c in df.columns]

        p = self.params
        atr_series = compute_atr(df, p["atr_period"])
        signals_df = compute_signals(df, p["atr_period"])

        trades = []
        self.total_bars += len(df)

        i = max(20, p["atr_period"])  # need enough history for all indicators

        while i < len(df) - 1:
            # Check signal score at bar i
            score = signals_df["score"].iloc[i]
            direction = signals_df["direction"].iloc[i]

            if score < self.min_score or direction is None:
                i += 1
                continue

            self.entry_bars += 1
            fired = signals_df["signals_fired"].iloc[i]
            for sig in fired:
                self.signal_counts[sig] = self.signal_counts.get(sig, 0) + 1

            entry_idx = i
            entry_row = df.iloc[entry_idx]
            is_futures = p.get("futures_mode", False)
            slippage = FUT_SLIPPAGE_PCT if is_futures else EQ_SLIPPAGE_PCT

            # Gap filter: skip if today's open gapped >2% from yesterday's close
            if p.get("gap_filter", False) and entry_idx > 0:
                prev_close = float(df.iloc[entry_idx - 1]["Close"])
                today_open = float(entry_row["Open"])
                if prev_close > 0:
                    gap_pct = abs(today_open - prev_close) / prev_close * 100
                    if gap_pct > 2.0:
                        i += 1
                        continue

            entry_price = float(entry_row["Close"]) * (1 + slippage)
            atr = float(atr_series.iloc[entry_idx])

            if atr <= 0 or entry_price <= 0:
                i += 1
                continue

            # Futures can short; equity is long-only
            if direction == "bearish" and not is_futures:
                i += 1
                continue

            is_short = direction == "bearish" and is_futures

            if is_short:
                entry_price = float(entry_row["Close"]) * (1 - slippage)
                target = entry_price - p["atr_target_mult"] * atr
                stoploss = entry_price + p["atr_sl_mult"] * atr
            else:
                target = entry_price + p["atr_target_mult"] * atr
                stoploss = entry_price - p["atr_sl_mult"] * atr

            if is_futures:
                # Margin-based sizing: capital / margin_required_per_share
                margin_per_share = entry_price * FUT_MARGIN_PCT
                quantity = max(1, int(p["capital_per_trade"] / margin_per_share))
            else:
                quantity = max(1, int(p["capital_per_trade"] / entry_price))

            peak = entry_price
            exit_price = None
            exit_reason = None
            exit_idx = None

            for j in range(1, p["max_hold_days"] + 2):
                day_idx = entry_idx + j
                if day_idx >= len(df):
                    exit_price = float(df.iloc[-1]["Close"]) * (1 - slippage)
                    exit_reason = "data_end"
                    exit_idx = len(df) - 1
                    break

                day = df.iloc[day_idx]
                high = float(day["High"])
                low = float(day["Low"])
                close = float(day["Close"])

                if is_short:
                    if low < peak:
                        peak = low  # track lowest for shorts

                    # Target hit (price drops to target)
                    if low <= target:
                        exit_price = target * (1 + slippage)
                        exit_reason = "target"
                        exit_idx = day_idx
                        break

                    # Stoploss hit (price rises to SL)
                    if high >= stoploss:
                        exit_price = stoploss * (1 + slippage)
                        exit_reason = "stoploss"
                        exit_idx = day_idx
                        break

                    # Trailing stop for shorts
                    progress = j / p["max_hold_days"]
                    unrealized_pct = (entry_price - peak) / entry_price * 100

                    if unrealized_pct >= p["trailing_activation_pct"]:
                        trail_sl = peak + p["trailing_tight_mult"] * atr
                        trail_sl = min(trail_sl, entry_price)
                        effective_sl = min(trail_sl, stoploss)
                    else:
                        effective_sl = stoploss

                    if p["time_sl_enabled"]:
                        if progress >= 0.75:
                            time_sl = min(entry_price, peak + p["time_sl_three_quarter_mult"] * atr)
                            effective_sl = min(effective_sl, time_sl)
                        elif progress >= 0.5:
                            time_sl = peak + p["time_sl_half_mult"] * atr
                            effective_sl = min(effective_sl, time_sl)

                    if high >= effective_sl:
                        exit_price = effective_sl * (1 + slippage)
                        is_trailing = unrealized_pct >= p["trailing_activation_pct"]
                        exit_reason = "trailing_stop" if is_trailing else "stoploss"
                        exit_idx = day_idx
                        break
                else:
                    if high > peak:
                        peak = high

                    # Target hit
                    if high >= target:
                        exit_price = target * (1 - slippage)
                        exit_reason = "target"
                        exit_idx = day_idx
                        break

                    # Stoploss hit
                    if low <= stoploss:
                        exit_price = stoploss * (1 - slippage)
                        exit_reason = "stoploss"
                        exit_idx = day_idx
                        break

                    # Trailing stop — only activates after threshold
                    progress = j / p["max_hold_days"]
                    unrealized_pct = (peak - entry_price) / entry_price * 100

                    if unrealized_pct >= p["trailing_activation_pct"]:
                        trail_sl = peak - p["trailing_tight_mult"] * atr
                        trail_sl = max(trail_sl, entry_price)
                        effective_sl = max(trail_sl, stoploss)
                    else:
                        effective_sl = stoploss

                    if p["time_sl_enabled"]:
                        if progress >= 0.75:
                            time_sl = max(entry_price, peak - p["time_sl_three_quarter_mult"] * atr)
                            effective_sl = max(effective_sl, time_sl)
                        elif progress >= 0.5:
                            time_sl = peak - p["time_sl_half_mult"] * atr
                            effective_sl = max(effective_sl, time_sl)

                    if low <= effective_sl:
                        exit_price = effective_sl * (1 - slippage)
                        is_trailing = unrealized_pct >= p["trailing_activation_pct"]
                        exit_reason = "trailing_stop" if is_trailing else "stoploss"
                        exit_idx = day_idx
                        break

                # Max hold expiry (shared)
                if j >= p["max_hold_days"]:
                    exit_price = close * (1 - slippage) if not is_short else close * (1 + slippage)
                    exit_reason = "expiry"
                    exit_idx = day_idx
                    break

            if exit_price is None:
                i += 1
                continue

            if is_short:
                pnl_gross = (entry_price - exit_price) * quantity
            else:
                pnl_gross = (exit_price - entry_price) * quantity

            if is_futures:
                costs = calc_fut_round_trip_costs(entry_price, exit_price, quantity)
            else:
                costs = calc_round_trip_costs(entry_price, exit_price, quantity)
            pnl_net = pnl_gross - costs
            if is_futures:
                margin_used = entry_price * FUT_MARGIN_PCT * quantity
                pnl_pct = (pnl_net / margin_used) * 100 if margin_used > 0 else 0
            else:
                pnl_pct = (pnl_net / (entry_price * quantity)) * 100

            trade = {
                "symbol": symbol,
                "entry_date": str(df.index[entry_idx].date()),
                "exit_date": str(df.index[exit_idx].date()),
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "quantity": quantity,
                "atr": round(atr, 2),
                "target": round(target, 2),
                "stoploss": round(stoploss, 2),
                "pnl_gross": round(pnl_gross, 2),
                "costs": round(costs, 2),
                "pnl_net": round(pnl_net, 2),
                "pnl_pct": round(pnl_pct, 2),
                "exit_reason": exit_reason,
                "holding_days": exit_idx - entry_idx,
                "signal_score": round(score, 1),
                "signals_fired": fired,
                "direction": direction,
            }
            trades.append(trade)

            # Skip to after this trade exits
            i = exit_idx + 1

        return trades

    def run_multi(self, data: dict[str, pd.DataFrame], nifty_df: pd.DataFrame | None = None,
                  vix_df: pd.DataFrame | None = None) -> list[dict]:
        """Run backtest across multiple symbols with full F&O trader filters.

        Filters applied (in order):
        1. Nifty regime (EMA50) — don't go long in bear market
        2. Market breadth — skip longs if <40% of stocks above 20-EMA
        3. Sector momentum — skip stocks in weakening sectors
        4. RS rank — only trade top N by 20-day return
        5. Stock win-rate — skip stocks with poor recent track record
        6. Portfolio risk controls (in _simulate_portfolio)
        """
        # Build Nifty trend filter: date -> bool (is_bullish)
        nifty_bullish = {}
        if nifty_df is not None:
            nifty_close = nifty_df["Close"]
            if isinstance(nifty_close, pd.DataFrame):
                nifty_close = nifty_close.iloc[:, 0]
            nifty_ema50 = nifty_close.ewm(span=50, adjust=False).mean()
            for idx in nifty_df.index:
                d = str(idx.date()) if hasattr(idx, 'date') else str(idx)
                c = float(nifty_close.loc[idx])
                e = float(nifty_ema50.loc[idx])
                nifty_bullish[d] = c > e

        # Build VIX map: date -> vix_value
        vix_map: dict[str, float] = {}
        if vix_df is not None and not vix_df.empty:
            vix_close = vix_df["Close"]
            if isinstance(vix_close, pd.DataFrame):
                vix_close = vix_close.iloc[:, 0]
            for idx in vix_df.index:
                d = str(idx.date()) if hasattr(idx, 'date') else str(idx)
                v = vix_close.loc[idx]
                if not pd.isna(v):
                    vix_map[d] = float(v)

        # Compute market breadth: date -> % of stocks above their 20-EMA
        breadth_map: dict[str, float] = {}
        if self.params.get("breadth_filter", False):
            # For each date, count how many stocks have close > 20-EMA
            stock_above_ema: dict[str, dict[str, bool]] = {}  # symbol -> {date: above_ema}
            for symbol, df in data.items():
                close = df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                ema20 = close.ewm(span=20, adjust=False).mean()
                stock_above_ema[symbol] = {}
                for idx in df.index:
                    d = str(idx.date()) if hasattr(idx, 'date') else str(idx)
                    if not pd.isna(ema20.loc[idx]):
                        stock_above_ema[symbol][d] = float(close.loc[idx]) > float(ema20.loc[idx])

            # Aggregate per date
            all_dates = set()
            for sym_data in stock_above_ema.values():
                all_dates.update(sym_data.keys())
            for d in all_dates:
                above = sum(1 for s in stock_above_ema.values() if s.get(d, False))
                total = sum(1 for s in stock_above_ema.values() if d in s)
                breadth_map[d] = (above / total * 100) if total > 0 else 50

        # Compute sector momentum: date -> {sector: momentum_score}
        # Uses rolling 5-day vs 20-day return differential per sector
        sector_momentum: dict[str, dict[str, float]] = {}  # date -> {sector: score}
        if self.params.get("sector_filter", False):
            # Group stocks by sector, compute sector average returns
            sector_stocks: dict[str, list[str]] = {}
            for symbol in data:
                sym_clean = symbol.replace(".NS", "")
                sector = NIFTY50_SECTORS.get(sym_clean)
                if sector:
                    sector_stocks.setdefault(sector, []).append(symbol)

            # Compute per-sector rolling returns
            for sector, symbols in sector_stocks.items():
                sector_ret5: dict[str, list[float]] = {}  # date -> list of 5d returns
                sector_ret20: dict[str, list[float]] = {}
                for sym in symbols:
                    if sym not in data:
                        continue
                    close = data[sym]["Close"]
                    if isinstance(close, pd.DataFrame):
                        close = close.iloc[:, 0]
                    ret5 = close.pct_change(5)
                    ret20 = close.pct_change(20)
                    for idx in data[sym].index:
                        d = str(idx.date()) if hasattr(idx, 'date') else str(idx)
                        r5 = ret5.loc[idx]
                        r20 = ret20.loc[idx]
                        if not pd.isna(r5):
                            sector_ret5.setdefault(d, []).append(float(r5))
                        if not pd.isna(r20):
                            sector_ret20.setdefault(d, []).append(float(r20))

                for d in sector_ret5:
                    avg5 = sum(sector_ret5[d]) / len(sector_ret5[d]) * 100
                    avg20 = sum(sector_ret20.get(d, [0])) / max(len(sector_ret20.get(d, [1])), 1) * 100
                    score = avg5 - avg20 / 4  # momentum acceleration
                    sector_momentum.setdefault(d, {})[sector] = score

        # Pre-compute per-stock 20-day returns for RS ranking
        rs_returns: dict[str, dict[str, float]] = {}  # symbol -> {date_str: return_20d}
        if self.params.get("rs_rank_filter", False):
            for symbol, df in data.items():
                close = df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                ret20 = close.pct_change(20)
                rs_returns[symbol] = {}
                for idx in df.index:
                    d = str(idx.date()) if hasattr(idx, "date") else str(idx)
                    v = ret20.loc[idx]
                    if not pd.isna(v):
                        rs_returns[symbol][d] = float(v)

        all_trades = []
        for symbol, df in data.items():
            trades = self.run(symbol, df)

            # 1. Regime filter: only longs when Nifty is bullish
            if nifty_bullish:
                trades = [t for t in trades
                          if t["direction"] == "bearish" or nifty_bullish.get(t["entry_date"], True)]

            # 2. VIX filter: tag trades with VIX-based size multiplier
            if vix_map:
                for t in trades:
                    vix_val = vix_map.get(t["entry_date"], 15.0)  # default normal
                    size_mult = 1.0
                    for threshold, mult in VIX_SIZE_MAP:
                        if vix_val <= threshold:
                            size_mult = mult
                            break
                    t["_vix_mult"] = size_mult
                # Drop trades where VIX says no trading
                trades = [t for t in trades if t.get("_vix_mult", 1.0) > 0]

            # 3. Breadth filter: skip longs if market breadth < 40%
            if breadth_map and self.params.get("breadth_filter", False):
                filtered = []
                for t in trades:
                    breadth = breadth_map.get(t["entry_date"], 50)
                    if t["direction"] == "bullish" and breadth < 40:
                        continue  # weak breadth — skip long
                    filtered.append(t)
                trades = filtered

            # 4. Sector momentum filter: skip stocks in weakening sectors
            if sector_momentum and self.params.get("sector_filter", False):
                filtered = []
                for t in trades:
                    sym_clean = t["symbol"].replace(".NS", "")
                    sector = NIFTY50_SECTORS.get(sym_clean)
                    if sector:
                        day_sectors = sector_momentum.get(t["entry_date"], {})
                        sector_score = day_sectors.get(sector, 0)
                        # Skip longs in weakening sectors, shorts in strengthening
                        if t["direction"] == "bullish" and sector_score < -0.5:
                            continue
                        if t["direction"] == "bearish" and sector_score > 0.5:
                            continue
                    filtered.append(t)
                trades = filtered

            all_trades.extend(trades)

        # Rolling stock quality filter: after a symbol accumulates enough trades,
        # skip further entries if its win rate is below threshold.
        # This is forward-safe — each trade only sees past trades for that symbol.
        if self.params.get("stock_wr_filter", False):
            min_lookback = self.params.get("stock_wr_lookback", 5)
            min_wr = self.params.get("stock_wr_min", 0.35)
            all_trades.sort(key=lambda t: t["entry_date"])
            filtered = []
            sym_history: dict[str, list[bool]] = {}
            for t in all_trades:
                sym = t["symbol"]
                hist = sym_history.setdefault(sym, [])
                if len(hist) >= min_lookback:
                    wr = sum(hist[-min_lookback:]) / min_lookback
                    if wr < min_wr:
                        continue  # skip — this stock's signals aren't working
                filtered.append(t)
                hist.append(t["pnl_net"] >= 0)
            all_trades = filtered

        # Relative strength rank filter: on each entry date, rank all symbols by
        # 20-day return. Only keep trades in the top N strongest stocks that day.
        if self.params.get("rs_rank_filter", False) and rs_returns:
            top_n = self.params.get("rs_top_n", 20)
            all_trades.sort(key=lambda t: t["entry_date"])
            filtered = []
            for t in all_trades:
                d = t["entry_date"]
                sym = t["symbol"]
                # Get all symbols' RS on this date
                day_rs = {s: rs_returns.get(s, {}).get(d, float("-inf")) for s in data}
                ranked = sorted(day_rs, key=day_rs.get, reverse=True)[:top_n]
                if sym in ranked:
                    filtered.append(t)
            all_trades = filtered

        all_trades.sort(key=lambda t: t["entry_date"])

        # Portfolio-level risk controls
        max_positions = self.params.get("max_positions", 0)
        risk_per_trade = self.params.get("risk_per_trade_pct", 0)  # max loss per trade as % of capital
        dd_breaker_pct = self.params.get("dd_circuit_breaker_pct", 0)  # pause after X% drawdown
        total_capital = self.params.get("total_capital", 0)

        if total_capital > 0 and (max_positions > 0 or risk_per_trade > 0 or dd_breaker_pct > 0):
            all_trades = self._simulate_portfolio(
                all_trades, total_capital, max_positions, risk_per_trade, dd_breaker_pct
            )

        self.trades = all_trades
        return all_trades

    def _simulate_portfolio(
        self, trades: list[dict], total_capital: float,
        max_positions: int, risk_per_trade_pct: float, dd_breaker_pct: float,
    ) -> list[dict]:
        """Replay trades with portfolio constraints: position limits, risk sizing, circuit breaker."""
        is_futures = self.params.get("futures_mode", False)

        # Build timeline of events (entries and exits)
        events = []
        for idx, t in enumerate(trades):
            events.append(("entry", t["entry_date"], idx, t))
            events.append(("exit", t["exit_date"], idx, t))
        events.sort(key=lambda e: (e[1], 0 if e[0] == "exit" else 1))  # exits before entries on same day

        open_positions = set()  # set of trade indices currently open
        equity = total_capital
        peak_equity = total_capital
        accepted = set()  # indices of accepted trades
        paused_until_date = None  # circuit breaker cooldown date
        dd_cooldown_days = 15  # pause for 15 trading days after breaker trips
        consecutive_losses = 0  # anti-tilt tracker

        for event_type, date, idx, trade in events:
            if event_type == "exit" and idx in accepted:
                open_positions.discard(idx)
                equity += trade["pnl_net"]
                if equity > peak_equity:
                    peak_equity = equity
                # Track consecutive losses
                if trade["pnl_net"] < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0

            elif event_type == "entry":
                # Circuit breaker cooldown check
                if dd_breaker_pct > 0 and paused_until_date is not None:
                    if date < paused_until_date:
                        continue
                    else:
                        paused_until_date = None
                        # Reset peak to current equity after cooldown
                        peak_equity = equity

                # Check drawdown circuit breaker trigger
                if dd_breaker_pct > 0 and peak_equity > 0:
                    dd_pct = (peak_equity - equity) / peak_equity * 100
                    if dd_pct >= dd_breaker_pct:
                        # Pause for cooldown_days trading days
                        from datetime import datetime as dt, timedelta
                        try:
                            d = dt.strptime(date, "%Y-%m-%d")
                            paused_until_date = (d + timedelta(days=dd_cooldown_days * 1.5)).strftime("%Y-%m-%d")
                        except ValueError:
                            paused_until_date = None
                        continue

                # Max positions check
                if max_positions > 0 and len(open_positions) >= max_positions:
                    continue

                # Compute combined size multiplier
                size_mult = 1.0

                # VIX-based sizing
                vix_mult = trade.get("_vix_mult", 1.0)
                size_mult *= vix_mult

                # Consecutive loss scaling (anti-tilt)
                if consecutive_losses >= 5:
                    size_mult *= 0.25
                elif consecutive_losses >= 3:
                    size_mult *= 0.5

                # Risk-per-trade position sizing: resize quantity so max loss = risk_pct% of equity
                if risk_per_trade_pct > 0:
                    max_loss_amount = equity * (risk_per_trade_pct / 100) * size_mult
                    sl_distance = abs(trade["entry_price"] - trade["stoploss"])
                    if sl_distance > 0:
                        risk_qty = max(1, int(max_loss_amount / sl_distance))
                        old_qty = trade["quantity"]
                        if risk_qty < old_qty:
                            ratio = risk_qty / old_qty
                            trade["quantity"] = risk_qty
                            trade["pnl_gross"] = round(trade["pnl_gross"] * ratio, 2)
                            trade["costs"] = round(trade["costs"] * ratio, 2)
                            trade["pnl_net"] = round(trade["pnl_net"] * ratio, 2)
                elif size_mult < 1.0:
                    # Apply size multiplier even without risk-per-trade
                    old_qty = trade["quantity"]
                    new_qty = max(1, int(old_qty * size_mult))
                    if new_qty < old_qty:
                        ratio = new_qty / old_qty
                        trade["quantity"] = new_qty
                        trade["pnl_gross"] = round(trade["pnl_gross"] * ratio, 2)
                        trade["costs"] = round(trade["costs"] * ratio, 2)
                        trade["pnl_net"] = round(trade["pnl_net"] * ratio, 2)

                accepted.add(idx)
                open_positions.add(idx)

        return [t for i, t in enumerate(trades) if i in accepted]

    def compute_stats(self) -> dict:
        """Compute performance statistics from completed trades."""
        trades = self.trades
        if not trades:
            return {"total_trades": 0}

        pnls = [t["pnl_net"] for t in trades]
        pnl_pcts = [t["pnl_pct"] for t in trades]
        wins = [t for t in trades if t["pnl_net"] >= 0]
        losses = [t for t in trades if t["pnl_net"] < 0]

        total_pnl = sum(pnls)
        total_costs = sum(t["costs"] for t in trades)
        win_rate = len(wins) / len(trades) * 100

        avg_win_pct = sum(t["pnl_pct"] for t in wins) / len(wins) if wins else 0
        avg_loss_pct = sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0
        avg_hold = sum(t["holding_days"] for t in trades) / len(trades)

        # Sharpe
        mean_ret = sum(pnl_pcts) / len(pnl_pcts)
        var = sum((r - mean_ret) ** 2 for r in pnl_pcts) / len(pnl_pcts)
        std = math.sqrt(var) if var > 0 else 0
        sharpe = round((mean_ret / std) * math.sqrt(252), 2) if std > 0 else 0

        # Max drawdown
        cumulative = 0
        peak_cum = 0
        max_dd = 0
        for p in pnls:
            cumulative += p
            if cumulative > peak_cum:
                peak_cum = cumulative
            dd = peak_cum - cumulative
            if dd > max_dd:
                max_dd = dd

        # Profit factor
        win_sum = sum(t["pnl_net"] for t in wins)
        loss_sum = abs(sum(t["pnl_net"] for t in losses))
        pf = round(win_sum / loss_sum, 2) if loss_sum > 0 else float("inf")

        # Exit reason breakdown
        reasons = {}
        for t in trades:
            r = t["exit_reason"]
            reasons[r] = reasons.get(r, 0) + 1

        # Per-symbol breakdown
        symbols = {}
        for t in trades:
            s = t["symbol"]
            if s not in symbols:
                symbols[s] = {"trades": 0, "wins": 0, "pnl": 0}
            symbols[s]["trades"] += 1
            symbols[s]["pnl"] += t["pnl_net"]
            if t["pnl_net"] >= 0:
                symbols[s]["wins"] += 1

        # Tax estimates
        stcg_equity = round(max(0, total_pnl) * 0.15, 2)

        # Entry rate
        entry_rate = (self.entry_bars / self.total_bars * 100) if self.total_bars > 0 else 0

        return {
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "total_costs": round(total_costs, 2),
            "avg_win_pct": round(avg_win_pct, 2),
            "avg_loss_pct": round(avg_loss_pct, 2),
            "avg_holding_days": round(avg_hold, 1),
            "sharpe_ratio": sharpe,
            "max_drawdown": round(max_dd, 2),
            "profit_factor": pf,
            "best_trade": max(trades, key=lambda t: t["pnl_pct"]),
            "worst_trade": min(trades, key=lambda t: t["pnl_pct"]),
            "exit_reasons": reasons,
            "per_symbol": symbols,
            "estimated_stcg_tax": stcg_equity,
            "signal_distribution": dict(self.signal_counts),
            "entry_rate_pct": round(entry_rate, 2),
            "total_bars_scanned": self.total_bars,
            "entry_bars": self.entry_bars,
        }

    def print_report(self, stats: dict | None = None):
        """Print formatted backtest report."""
        if stats is None:
            stats = self.compute_stats()

        if stats["total_trades"] == 0:
            print("No trades generated.")
            return

        border = "=" * 60
        sep = "-" * 58

        print(f"\n{border}")
        print("  SIGNAL-FILTERED BACKTEST RESULTS")
        print(f"  Params: ATR target={self.params['atr_target_mult']}x, "
              f"SL={self.params['atr_sl_mult']}x, "
              f"hold={self.params['max_hold_days']}d, "
              f"capital/trade=\u20b9{self.params['capital_per_trade']:,}")
        mode = "FUTURES (margin-based)" if self.params.get("futures_mode") else "EQUITY"
        print(f"  Mode: {mode}  |  Min signal score: {self.min_score}")
        print(border)

        print(f"\n  Total Trades: {stats['total_trades']}  |  "
              f"Win Rate: {stats['win_rate']:.1f}%  "
              f"({stats['winning_trades']}W / {stats['losing_trades']}L)")
        print(f"  Total P&L: \u20b9{stats['total_pnl']:+,.0f}  |  "
              f"Costs: \u20b9{stats['total_costs']:,.0f}")
        print(f"  Avg Win: {stats['avg_win_pct']:+.2f}%  |  "
              f"Avg Loss: {stats['avg_loss_pct']:+.2f}%")
        print(f"  Avg Holding: {stats['avg_holding_days']:.1f} days")

        print(f"\n  {sep}")
        print(f"  RISK METRICS")
        print(f"  {sep}")
        pf = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] != float('inf') else "\u221e"
        print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}  |  "
              f"Profit Factor: {pf}")
        print(f"  Max Drawdown: \u20b9{stats['max_drawdown']:,.0f}")

        print(f"\n  Best:  {stats['best_trade']['symbol']} "
              f"({stats['best_trade']['pnl_pct']:+.1f}%, "
              f"\u20b9{stats['best_trade']['pnl_net']:+,.0f})")
        print(f"  Worst: {stats['worst_trade']['symbol']} "
              f"({stats['worst_trade']['pnl_pct']:+.1f}%, "
              f"\u20b9{stats['worst_trade']['pnl_net']:+,.0f})")

        print(f"\n  {sep}")
        print(f"  SIGNAL DISTRIBUTION")
        print(f"  {sep}")
        print(f"  Entry rate: {stats['entry_rate_pct']:.2f}% "
              f"({stats['entry_bars']} entries / {stats['total_bars_scanned']} bars)")
        for sig, count in sorted(stats["signal_distribution"].items(), key=lambda x: -x[1]):
            if count > 0:
                pct = count / stats["entry_bars"] * 100 if stats["entry_bars"] > 0 else 0
                print(f"  {sig:20s} {count:4d}  ({pct:.0f}% of entries)")

        print(f"\n  {sep}")
        print(f"  EXIT REASONS")
        print(f"  {sep}")
        for reason, count in sorted(stats["exit_reasons"].items(), key=lambda x: -x[1]):
            pct = count / stats["total_trades"] * 100
            print(f"  {reason:20s} {count:4d}  ({pct:.0f}%)")

        print(f"\n  {sep}")
        print(f"  TAX ESTIMATE")
        print(f"  {sep}")
        print(f"  STCG (15% on equity gains): \u20b9{stats['estimated_stcg_tax']:,.0f}")
        print(f"  Net after tax: \u20b9{stats['total_pnl'] - stats['estimated_stcg_tax']:+,.0f}")

        # Top/worst symbols
        if stats["per_symbol"]:
            sorted_syms = sorted(stats["per_symbol"].items(), key=lambda x: -x[1]["pnl"])
            print(f"\n  {sep}")
            print(f"  TOP SYMBOLS")
            print(f"  {sep}")
            for sym, data in sorted_syms[:5]:
                wr = data["wins"] / data["trades"] * 100 if data["trades"] > 0 else 0
                print(f"  {sym:15s} {data['trades']:3d} trades  "
                      f"{wr:.0f}% WR  \u20b9{data['pnl']:+,.0f}")
            if len(sorted_syms) > 5:
                print(f"\n  WORST SYMBOLS")
                for sym, data in sorted_syms[-3:]:
                    wr = data["wins"] / data["trades"] * 100 if data["trades"] > 0 else 0
                    print(f"  {sym:15s} {data['trades']:3d} trades  "
                          f"{wr:.0f}% WR  \u20b9{data['pnl']:+,.0f}")

        print(f"\n{border}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Signal-filtered backtest — enters only on multi-signal confluence"
    )
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols (e.g. RELIANCE,TCS)")
    parser.add_argument("--nifty50", action="store_true", help="Run on all Nifty 50 stocks")
    parser.add_argument("--period", type=str, default="1y", help="yfinance period (1mo, 3mo, 6mo, 1y, 2y)")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
                        help=f"Minimum signal score to enter (default: {DEFAULT_MIN_SCORE})")
    parser.add_argument("--atr-target", type=float, default=2.0, help="ATR target multiplier")
    parser.add_argument("--atr-sl", type=float, default=2.5, help="ATR stoploss multiplier")
    parser.add_argument("--max-hold", type=int, default=15, help="Max holding days")
    parser.add_argument("--trail", type=float, default=2.5, help="Trailing stop ATR multiplier")
    parser.add_argument("--trail-act", type=float, default=3.0, help="Trailing activation %% profit")
    parser.add_argument("--no-regime-filter", action="store_true", help="Disable Nifty EMA50 regime filter")
    parser.add_argument("--stock-filter", action="store_true", help="Enable rolling per-stock win rate filter")
    parser.add_argument("--stock-wr-min", type=float, default=0.35, help="Min win rate to keep trading a stock")
    parser.add_argument("--stock-lookback", type=int, default=5, help="Lookback trades for stock filter")
    parser.add_argument("--rs-filter", action="store_true", help="Enable relative strength rank filter")
    parser.add_argument("--rs-top-n", type=int, default=20, help="Only trade top N stocks by 20d return")
    parser.add_argument("--capital", type=float, default=10000, help="Capital per trade")
    parser.add_argument("--futures", action="store_true", help="Use futures mode (margin-based sizing, lower costs, shorts enabled)")
    parser.add_argument("--vix-filter", action="store_true", help="Enable India VIX-based position sizing")
    parser.add_argument("--breadth-filter", action="store_true", help="Enable market breadth filter (<40%% skips longs)")
    parser.add_argument("--sector-filter", action="store_true", help="Enable sector momentum filter")
    parser.add_argument("--gap-filter", action="store_true", help="Skip entries on >2%% gap days")
    parser.add_argument("--total-capital", type=float, default=0, help="Total portfolio capital (enables portfolio risk controls)")
    parser.add_argument("--max-positions", type=int, default=0, help="Max concurrent open positions (0=unlimited)")
    parser.add_argument("--risk-per-trade", type=float, default=0, help="Max loss per trade as %% of equity (0=disabled)")
    parser.add_argument("--dd-breaker", type=float, default=0, help="Pause trading after X%% drawdown from peak (0=disabled)")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    if not args.symbols and not args.nifty50:
        print("Error: specify --symbols or --nifty50")
        sys.exit(1)

    symbols = NIFTY50_SYMBOLS if args.nifty50 else [s.strip() for s in args.symbols.split(",")]

    params = {
        **DEFAULT_PARAMS,
        "atr_target_mult": args.atr_target,
        "atr_sl_mult": args.atr_sl,
        "max_hold_days": args.max_hold,
        "trailing_mult": args.trail,
        "trailing_activation_pct": args.trail_act,
        "capital_per_trade": args.capital,
        "stock_wr_filter": args.stock_filter,
        "stock_wr_min": args.stock_wr_min,
        "stock_wr_lookback": args.stock_lookback,
        "rs_rank_filter": args.rs_filter,
        "rs_top_n": args.rs_top_n,
        "futures_mode": args.futures,
        "gap_filter": args.gap_filter,
        "breadth_filter": args.breadth_filter,
        "sector_filter": args.sector_filter,
        "total_capital": args.total_capital,
        "max_positions": args.max_positions,
        "risk_per_trade_pct": args.risk_per_trade,
        "dd_circuit_breaker_pct": args.dd_breaker,
    }

    print(f"Fetching data for {len(symbols)} symbol(s), period={args.period}...")
    data = fetch_data(symbols, args.period)
    print(f"Got data for {len(data)} symbols")

    if not data:
        print("No data available")
        sys.exit(1)

    # Fetch Nifty data for regime filter
    nifty_df = None
    if not args.no_regime_filter:
        import yfinance as yf
        print("Fetching Nifty 50 data for regime filter...")
        try:
            nifty_df = yf.download("^NSEI", period=args.period, progress=False)
            if nifty_df.empty:
                print("Warning: Could not fetch Nifty data, running without regime filter")
                nifty_df = None
            else:
                print(f"Nifty data: {len(nifty_df)} rows")
        except Exception as e:
            print(f"Warning: Nifty fetch failed ({e}), running without regime filter")

    # Fetch India VIX for position sizing
    vix_df = None
    if args.vix_filter:
        import yfinance as yf
        print("Fetching India VIX for position sizing...")
        try:
            vix_df = yf.download("^INDIAVIX", period=args.period, progress=False)
            if vix_df.empty:
                print("Warning: Could not fetch VIX data")
                vix_df = None
            else:
                print(f"VIX data: {len(vix_df)} rows")
        except Exception as e:
            print(f"Warning: VIX fetch failed ({e})")

    engine = SignalBacktestEngine(params, min_score=args.min_score)
    engine.run_multi(data, nifty_df=nifty_df, vix_df=vix_df)
    stats = engine.compute_stats()
    engine.print_report(stats)

    if args.output:
        output = {
            "params": {**params, "min_score": args.min_score},
            "stats": stats,
            "trades": engine.trades,
        }
        Path(args.output).write_text(json.dumps(output, indent=2, default=str))
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
