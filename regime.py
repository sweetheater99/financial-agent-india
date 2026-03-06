"""Market regime detection for Trading System V2.

Classifies the current market into one of:
CASH, TRENDING_UP, TRENDING_DOWN, SIDEWAYS, VOLATILE, UNCERTAIN

Uses: ADX(10), 20 EMA, Bollinger Bands, VIX, Nifty daily range.
"""

import numpy as np
import pandas as pd
from ta.trend import ADXIndicator, EMAIndicator
from ta.volatility import BollingerBands

from config import (
    CASH_TRIGGER_MONTHLY_DRAWDOWN_PCT,
    CASH_TRIGGER_CONSECUTIVE_LOSSES,
    CASH_TRIGGER_VIX,
)

# Minimum rows needed for indicators (ADX 10 + EMA 20 + some lookback)
MIN_ROWS = 30


def vix_to_iv_percentile(vix: float) -> int:
    """Map VIX to approximate IV percentile.

    VIX < 12  -> 10th percentile
    12-15     -> 25th percentile
    15-18     -> 50th percentile
    18-22     -> 70th percentile
    22+       -> 90th percentile
    """
    if vix < 12:
        return 10
    elif vix < 15:
        return 25
    elif vix < 18:
        return 50
    elif vix < 22:
        return 70
    else:
        return 90


def _check_cash_regime(vix: float, risk_state: dict | None) -> dict | None:
    """Return regime dict if CASH triggered, else None.

    CASH triggers (any one):
    - Monthly drawdown > CASH_TRIGGER_MONTHLY_DRAWDOWN_PCT (8%)
    - Consecutive losses >= CASH_TRIGGER_CONSECUTIVE_LOSSES (5)
    - VIX > CASH_TRIGGER_VIX (25)
    """
    if risk_state:
        dd = risk_state.get("monthly_drawdown_pct", 0)
        # drawdown is negative, threshold is positive — compare abs
        if abs(dd) >= CASH_TRIGGER_MONTHLY_DRAWDOWN_PCT:
            return {
                "regime": "CASH",
                "confidence": 1.0,
                "reason": f"Monthly drawdown {dd:.1f}% exceeds -{CASH_TRIGGER_MONTHLY_DRAWDOWN_PCT}% threshold",
            }

        losses = risk_state.get("consecutive_losses", 0)
        if losses >= CASH_TRIGGER_CONSECUTIVE_LOSSES:
            return {
                "regime": "CASH",
                "confidence": 1.0,
                "reason": f"{losses} consecutive losses (threshold: {CASH_TRIGGER_CONSECUTIVE_LOSSES})",
            }

    if vix >= CASH_TRIGGER_VIX:
        return {
            "regime": "CASH",
            "confidence": 1.0,
            "reason": f"VIX {vix:.1f} >= {CASH_TRIGGER_VIX} crisis threshold",
        }

    return None


def _check_volatile(nifty_df: pd.DataFrame, vix: float) -> dict | None:
    """Return VOLATILE regime if VIX > 20 OR today's range > 2x 20-day average range."""
    if vix > 20:
        return {
            "regime": "VOLATILE",
            "confidence": 0.85,
            "reason": f"VIX {vix:.1f} > 20 indicates elevated volatility",
        }

    daily_range = nifty_df["high"] - nifty_df["low"]
    avg_range_20 = daily_range.rolling(20).mean()

    if len(daily_range) >= 21:
        today_range = daily_range.iloc[-1]
        avg_20 = avg_range_20.iloc[-1]
        if not np.isnan(avg_20) and avg_20 > 0 and today_range > 2 * avg_20:
            return {
                "regime": "VOLATILE",
                "confidence": 0.80,
                "reason": f"Daily range {today_range:.0f} > 2x 20-day avg range {avg_20:.0f}",
            }

    return None


def _check_trending(nifty_df: pd.DataFrame) -> dict | None:
    """Detect TRENDING_UP or TRENDING_DOWN.

    Conditions:
    - ADX(10) > 20 for 3+ bars
    - Price above/below 20 EMA
    - Price rising/falling for 5+ days
    """
    high = nifty_df["high"]
    low = nifty_df["low"]
    close = nifty_df["close"]

    adx_indicator = ADXIndicator(high, low, close, window=10)
    adx = adx_indicator.adx()

    ema_indicator = EMAIndicator(close, window=20)
    ema20 = ema_indicator.ema_indicator()

    # Check ADX > 20 for last 3 bars
    adx_tail = adx.iloc[-3:]
    if adx_tail.isna().any() or not (adx_tail > 20).all():
        return None

    # Current price vs EMA
    current_close = close.iloc[-1]
    current_ema = ema20.iloc[-1]
    if np.isnan(current_ema):
        return None

    above_ema = current_close > current_ema
    below_ema = current_close < current_ema

    # Check 5-day price direction
    last_5 = close.iloc[-5:]
    rising_days = sum(1 for i in range(1, len(last_5)) if last_5.iloc[i] > last_5.iloc[i - 1])
    falling_days = sum(1 for i in range(1, len(last_5)) if last_5.iloc[i] < last_5.iloc[i - 1])

    # Calculate confidence from ADX strength
    adx_val = adx.iloc[-1]
    confidence = min(0.95, 0.70 + (adx_val - 20) * 0.01)

    if above_ema and rising_days >= 3:
        return {
            "regime": "TRENDING_UP",
            "confidence": round(confidence, 2),
            "reason": f"ADX {adx_val:.1f}>20 for 3+ bars, price above EMA20, {rising_days}/4 days rising",
        }

    if below_ema and falling_days >= 3:
        return {
            "regime": "TRENDING_DOWN",
            "confidence": round(confidence, 2),
            "reason": f"ADX {adx_val:.1f}>20 for 3+ bars, price below EMA20, {falling_days}/4 days falling",
        }

    return None


def _check_sideways(nifty_df: pd.DataFrame) -> dict | None:
    """Detect SIDEWAYS regime.

    Conditions:
    - ADX(10) < 20 for 3+ bars
    - Nifty range < 400 pts for 7+ days
    - BB width at 20-day low
    """
    high = nifty_df["high"]
    low = nifty_df["low"]
    close = nifty_df["close"]

    adx_indicator = ADXIndicator(high, low, close, window=10)
    adx = adx_indicator.adx()

    # Check ADX < 20 for last 3 bars
    adx_tail = adx.iloc[-3:]
    if adx_tail.isna().any() or not (adx_tail < 20).all():
        return None

    # Check Nifty range < 400 pts for last 7 days
    last_7_high = high.iloc[-7:].max()
    last_7_low = low.iloc[-7:].min()
    range_7d = last_7_high - last_7_low
    if range_7d >= 400:
        return None

    # Check BB width at 20-day low
    bb = BollingerBands(close, window=20, window_dev=2)
    bb_width = bb.bollinger_wband()

    if bb_width.isna().all():
        return None

    bb_width_last_20 = bb_width.iloc[-20:]
    bb_width_valid = bb_width_last_20.dropna()
    if len(bb_width_valid) < 2:
        return None

    current_bb_width = bb_width.iloc[-1]
    if np.isnan(current_bb_width):
        return None

    # BB width at or near 20-day low (within 10% of min)
    bb_min = bb_width_valid.min()
    at_low = current_bb_width <= bb_min * 1.10

    if at_low:
        confidence = 0.80
    else:
        # Still sideways but lower confidence without BB confirmation
        confidence = 0.65

    adx_val = adx.iloc[-1]
    return {
        "regime": "SIDEWAYS",
        "confidence": round(confidence, 2),
        "reason": f"ADX {adx_val:.1f}<20 for 3+ bars, 7-day range {range_7d:.0f}<400pts, BB width {'at' if at_low else 'not at'} 20-day low",
    }


def classify_regime(
    nifty_df: pd.DataFrame,
    vix: float,
    risk_state: dict | None = None,
) -> dict:
    """Classify current market regime.

    Priority order:
    1. CASH (overrides everything)
    2. VOLATILE (checked before trend/sideways)
    3. TRENDING_UP / TRENDING_DOWN
    4. SIDEWAYS
    5. UNCERTAIN (default fallback)

    Args:
        nifty_df: DataFrame with columns ['open', 'high', 'low', 'close'], 50+ rows of daily candles
        vix: current India VIX level
        risk_state: optional dict with monthly_drawdown_pct and consecutive_losses

    Returns:
        dict with keys: regime, confidence, reason, iv_percentile
    """
    iv_pct = vix_to_iv_percentile(vix)

    def _result(regime_dict: dict) -> dict:
        regime_dict["iv_percentile"] = iv_pct
        return regime_dict

    # 1. CASH — always check first
    cash = _check_cash_regime(vix, risk_state)
    if cash:
        return _result(cash)

    # Insufficient data guard
    if len(nifty_df) < MIN_ROWS:
        return _result({
            "regime": "UNCERTAIN",
            "confidence": 0.0,
            "reason": f"Insufficient data: {len(nifty_df)} rows (need {MIN_ROWS}+)",
        })

    # 2. VOLATILE
    volatile = _check_volatile(nifty_df, vix)
    if volatile:
        return _result(volatile)

    # 3. TRENDING
    trending = _check_trending(nifty_df)
    if trending:
        return _result(trending)

    # 4. SIDEWAYS
    sideways = _check_sideways(nifty_df)
    if sideways:
        return _result(sideways)

    # 5. UNCERTAIN fallback
    return _result({
        "regime": "UNCERTAIN",
        "confidence": 0.5,
        "reason": "No regime matched with sufficient confidence",
    })
