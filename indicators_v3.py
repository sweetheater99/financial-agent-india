"""V3 indicators: Supertrend and CPR.

Supertrend: ATR-based trailing stop that flips on price crossing.
CPR: Central Pivot Range from previous day's high/low/close.
"""

import numpy as np
import pandas as pd

import config


def compute_supertrend(
    df: pd.DataFrame,
    period: int = None,
    multiplier: float = None,
) -> str:
    """Compute Supertrend signal from OHLC DataFrame.

    Returns "buy", "sell", or "unknown" (if insufficient data).
    """
    period = period or config.SUPERTREND_PERIOD
    multiplier = multiplier or config.SUPERTREND_MULTIPLIER

    if len(df) < period + 1:
        return "unknown"

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    # ATR computation
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            abs(high[1:] - close[:-1]),
            abs(low[1:] - close[:-1]),
        ),
    )
    # Prepend 0 for alignment
    tr = np.insert(tr, 0, high[0] - low[0])

    atr = np.zeros(len(tr))
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    # Supertrend bands
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = np.zeros(len(close))
    direction = np.ones(len(close))  # 1 = up (buy), -1 = down (sell)

    for i in range(1, len(close)):
        if atr[i] == 0:
            supertrend[i] = supertrend[i - 1]
            direction[i] = direction[i - 1]
            continue

        # Adjust bands based on previous values
        if lower_band[i] > supertrend[i - 1] or close[i - 1] > supertrend[i - 1]:
            lower_band[i] = max(lower_band[i],
                                supertrend[i - 1] if direction[i - 1] == 1 else lower_band[i])

        if upper_band[i] < supertrend[i - 1] or close[i - 1] < supertrend[i - 1]:
            upper_band[i] = min(upper_band[i],
                                supertrend[i - 1] if direction[i - 1] == -1 else upper_band[i])

        if direction[i - 1] == 1:  # previous was uptrend
            if close[i] < supertrend[i - 1]:
                direction[i] = -1
                supertrend[i] = upper_band[i]
            else:
                direction[i] = 1
                supertrend[i] = lower_band[i]
        else:  # previous was downtrend
            if close[i] > supertrend[i - 1]:
                direction[i] = 1
                supertrend[i] = lower_band[i]
            else:
                direction[i] = -1
                supertrend[i] = upper_band[i]

    last_dir = direction[-1]
    if last_dir == 1:
        return "buy"
    elif last_dir == -1:
        return "sell"
    return "unknown"


def compute_cpr(prev_high: float, prev_low: float, prev_close: float) -> dict:
    """Compute Central Pivot Range from previous day's candle.

    Returns dict with pivot, tc, bc, cpr_width, cpr_width_pct, day_type.
    """
    pivot = (prev_high + prev_low + prev_close) / 3
    bc = (prev_high + prev_low) / 2
    tc = (pivot - bc) + pivot
    cpr_width = abs(tc - bc)
    cpr_width_pct = (cpr_width / prev_close) * 100 if prev_close > 0 else 0

    if cpr_width_pct < config.CPR_NARROW_PCT:
        day_type = "trending"
    elif cpr_width_pct > config.CPR_WIDE_PCT:
        day_type = "sideways"
    else:
        day_type = "normal"

    return {
        "pivot": round(pivot, 2),
        "tc": round(max(tc, bc), 2),
        "bc": round(min(tc, bc), 2),
        "cpr_width": round(cpr_width, 2),
        "cpr_width_pct": round(cpr_width_pct, 3),
        "day_type": day_type,
    }
