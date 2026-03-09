"""V4 indicators: MFI, OBV divergence, VWAP deviation, ADX, PCR signal, max pain, FII momentum.

Pure functions. OHLCV inputs as pandas DataFrames with columns:
open, high, low, close, volume. No side effects.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Money Flow Index
# ---------------------------------------------------------------------------

def compute_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index (volume-weighted RSI). Returns 0-100 Series, NaN for warmup."""
    n = len(df)
    result = pd.Series(np.nan, index=df.index)

    if n < period + 1:
        return result

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    raw_money_flow = typical_price * df["volume"]

    tp_diff = typical_price.diff()

    pos_flow = pd.Series(0.0, index=df.index)
    neg_flow = pd.Series(0.0, index=df.index)

    pos_flow[tp_diff > 0] = raw_money_flow[tp_diff > 0]
    neg_flow[tp_diff < 0] = raw_money_flow[tp_diff < 0]

    pos_sum = pos_flow.rolling(window=period).sum()
    neg_sum = neg_flow.rolling(window=period).sum()

    # Avoid division by zero
    mfi = 100.0 - 100.0 / (1.0 + pos_sum / neg_sum.replace(0, np.nan))

    # Where neg_sum is 0, MFI = 100 (all positive flow)
    mfi[neg_sum == 0] = 100.0
    # Where pos_sum is 0, MFI = 0
    mfi[(pos_sum == 0) & (neg_sum > 0)] = 0.0

    # NaN out first period bars (need period bars of flow data + 1 for diff)
    mfi.iloc[:period] = np.nan
    result = mfi

    return result


# ---------------------------------------------------------------------------
# 2. OBV Divergence
# ---------------------------------------------------------------------------

def compute_obv_divergence(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """OBV divergence detection. Returns Series of 'bullish_div', 'bearish_div', 'none'."""
    n = len(df)
    close = df["close"].values
    volume = df["volume"].values

    # Compute OBV
    obv = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]

    result = pd.Series("none", index=df.index)

    for i in range(lookback, n):
        window_start = i - lookback
        price_window = close[window_start : i + 1]
        obv_window = obv[window_start : i + 1]

        price_high = np.max(price_window)
        price_low = np.min(price_window)
        obv_high = np.max(obv_window)
        obv_low = np.min(obv_window)

        # Bearish: price at new high in window but OBV below its window high
        if close[i] >= price_high and obv[i] < obv_high:
            result.iloc[i] = "bearish_div"
        # Bullish: price at new low in window but OBV above its window low
        elif close[i] <= price_low and obv[i] > obv_low:
            result.iloc[i] = "bullish_div"

    return result


# ---------------------------------------------------------------------------
# 3. VWAP Deviation
# ---------------------------------------------------------------------------

def compute_vwap_deviation(df: pd.DataFrame, atr_period: int = 14) -> pd.Series:
    """Price deviation from cumulative VWAP, normalized by ATR."""
    n = len(df)
    result = pd.Series(np.nan, index=df.index)

    if n < atr_period + 1:
        return result

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical_price * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    vwap = cum_tp_vol / cum_vol

    # ATR via Wilder's smoothing
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    atr = np.full(n, np.nan)
    atr[atr_period - 1] = np.mean(tr[:atr_period])
    for i in range(atr_period, n):
        atr[i] = (atr[i - 1] * (atr_period - 1) + tr[i]) / atr_period

    atr_series = pd.Series(atr, index=df.index)

    deviation = (df["close"] - vwap) / atr_series
    deviation.iloc[:atr_period] = np.nan

    return deviation


# ---------------------------------------------------------------------------
# 4. ADX + DI+/DI-
# ---------------------------------------------------------------------------

def compute_adx(df: pd.DataFrame, period: int = 14) -> dict[str, pd.Series]:
    """ADX with DI+/DI-. Returns dict with 'adx', 'di_plus', 'di_minus'."""
    n = len(df)
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    # True Range, +DM, -DM
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0

    # Wilder's smoothing: prev * (period-1)/period + new_value
    # NOT the same as EMA — new value is added directly, not multiplied by alpha
    def wilder_smooth(arr: np.ndarray) -> np.ndarray:
        smoothed = np.full(n, np.nan)
        smoothed[period - 1] = np.sum(arr[:period])
        for i in range(period, n):
            smoothed[i] = smoothed[i - 1] - smoothed[i - 1] / period + arr[i]
        return smoothed

    smooth_tr = wilder_smooth(tr)
    smooth_plus_dm = wilder_smooth(plus_dm)
    smooth_minus_dm = wilder_smooth(minus_dm)

    # DI+ and DI-
    di_plus = np.full(n, np.nan)
    di_minus = np.full(n, np.nan)

    for i in range(period, n):
        if smooth_tr[i] > 0:
            di_plus[i] = 100.0 * smooth_plus_dm[i] / smooth_tr[i]
            di_minus[i] = 100.0 * smooth_minus_dm[i] / smooth_tr[i]

    # NaN out first period bars for DI
    di_plus[:period] = np.nan
    di_minus[:period] = np.nan

    # DX
    dx = np.full(n, np.nan)
    for i in range(period, n):
        if not (np.isnan(di_plus[i]) or np.isnan(di_minus[i])):
            denom = di_plus[i] + di_minus[i]
            if denom > 0:
                dx[i] = 100.0 * abs(di_plus[i] - di_minus[i]) / denom

    # ADX = Wilder's smooth of DX, starting from 2*period
    adx = np.full(n, np.nan)
    # First valid DX is at index `period`, need `period` DX values for first ADX
    first_adx_idx = 2 * period
    if first_adx_idx < n:
        # Average of DX from period to 2*period-1
        dx_slice = dx[period:2 * period]
        valid_dx = dx_slice[~np.isnan(dx_slice)]
        if len(valid_dx) > 0:
            adx[first_adx_idx] = np.mean(valid_dx)
            for i in range(first_adx_idx + 1, n):
                if not np.isnan(dx[i]):
                    adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return {
        "adx": pd.Series(adx, index=df.index),
        "di_plus": pd.Series(di_plus, index=df.index),
        "di_minus": pd.Series(di_minus, index=df.index),
    }


# ---------------------------------------------------------------------------
# 5. PCR Signal
# ---------------------------------------------------------------------------

def compute_pcr_signal(pcr: float | None) -> float:
    """Convert Put-Call Ratio to -1..+1 signal.

    PCR > 1.3 = bullish (positive), PCR < 0.7 = bearish (negative).
    Linear interpolation between thresholds.
    """
    if pcr is None:
        return 0.0

    # Neutral zone center = 1.0, bullish threshold = 1.3, bearish threshold = 0.7
    if pcr >= 1.3:
        # At 1.3 = +1.0, beyond stays clamped at 1.0
        return 1.0
    elif pcr <= 0.7:
        # At 0.7 = -1.0, below stays clamped at -1.0
        return -1.0
    else:
        # Between 0.7 and 1.3: interpolate linearly from -1 to +1
        signal = (pcr - 1.0) / 0.3
        return max(-1.0, min(1.0, signal))


# ---------------------------------------------------------------------------
# 6. Max Pain Signal
# ---------------------------------------------------------------------------

def compute_maxpain_signal(price: float, max_pain: float | None) -> float:
    """Max pain proximity to -1..+1 signal.

    Positive if price below max pain (bullish pull up), negative if above (bearish pull down).
    """
    if max_pain is None or max_pain == 0:
        return 0.0

    pct_diff = abs(price - max_pain) / max_pain * 100  # percentage distance
    sign = 1.0 if price < max_pain else -1.0  # below = bullish, above = bearish

    if pct_diff <= 1.0:
        strength = 0.8
    elif pct_diff <= 3.0:
        # Decay from 0.8 to 0.1 over 1% to 3%
        strength = 0.8 - (pct_diff - 1.0) * (0.7 / 2.0)
    else:
        strength = 0.1

    return max(-1.0, min(1.0, sign * strength))


# ---------------------------------------------------------------------------
# 7. FII Momentum
# ---------------------------------------------------------------------------

def compute_fii_momentum(flows_3d: list[float] | None) -> float:
    """Convert 3-day FII flow to -1..+1 signal. Normalize by +/-3000 Cr."""
    if not flows_3d:
        return 0.0

    avg_flow = sum(flows_3d) / len(flows_3d)
    normalized = avg_flow / 3000.0
    return max(-1.0, min(1.0, normalized))
