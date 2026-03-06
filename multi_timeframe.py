"""Multi-timeframe trend confirmation.

Checks daily + weekly trend alignment before allowing entries.
Uses yfinance for weekly candles (free, no API cost).
"""

import logging
import time
from functools import lru_cache

logger = logging.getLogger(__name__)

# Cache TTL: 1 day (86400 seconds)
_CACHE_TTL = 86400
_cache: dict[str, tuple[float, dict]] = {}


def _get_weekly_data(symbol: str, periods: int = 52):
    """Fetch weekly OHLCV data from yfinance.

    Args:
        symbol: NSE stock symbol (e.g. "RELIANCE")
        periods: number of weeks to fetch

    Returns:
        pandas DataFrame with weekly candles, or None on failure.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(f"{symbol}.NS")
        df = ticker.history(period=f"{periods}wk", interval="1wk")
        if df is None or df.empty:
            return None
        return df
    except Exception as e:
        logger.warning("yfinance weekly fetch failed for %s: %s", symbol, e)
        return None


def _compute_ema(series, period: int):
    """Compute EMA on a pandas Series."""
    return series.ewm(span=period, adjust=False).mean()


def _compute_rsi(series, period: int = 14):
    """Compute RSI on a pandas Series."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    # After initial SMA, use EMA for remaining
    for i in range(period, len(avg_gain)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _compute_macd(series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Compute MACD line and signal line."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def check_weekly_trend(symbol: str, direction: str) -> dict:
    """Check if weekly trend agrees with daily entry direction.

    Uses:
    - Weekly close vs 10-week EMA (bullish if above, bearish if below)
    - Weekly RSI (bullish if 40-70, bearish if 30-60)
    - Weekly MACD signal (bullish if MACD > signal, bearish if below)

    Returns: {
        aligned: bool,
        weekly_trend: "bullish" | "bearish" | "neutral",
        weekly_rsi: float,
        weekly_ema_above: bool,
        confidence: float (0-1),
    }
    """
    # Check cache
    cache_key = symbol
    now = time.time()
    if cache_key in _cache:
        cached_time, cached_result = _cache[cache_key]
        if now - cached_time < _CACHE_TTL:
            # Re-evaluate alignment for the requested direction
            result = dict(cached_result)
            trend = result["weekly_trend"]
            if trend == "neutral":
                result["aligned"] = True
            elif direction == "bullish":
                result["aligned"] = trend == "bullish"
            else:
                result["aligned"] = trend == "bearish"
            return result

    df = _get_weekly_data(symbol)
    if df is None or len(df) < 12:
        result = {
            "aligned": True,  # no data → don't block
            "weekly_trend": "neutral",
            "weekly_rsi": 50.0,
            "weekly_ema_above": True,
            "confidence": 0.0,
        }
        _cache[cache_key] = (now, result)
        return result

    close = df["Close"]

    # 10-week EMA
    ema10 = _compute_ema(close, 10)
    latest_close = close.iloc[-1]
    latest_ema = ema10.iloc[-1]
    ema_above = latest_close > latest_ema

    # Weekly RSI (14-period)
    rsi_series = _compute_rsi(close, 14)
    weekly_rsi = rsi_series.iloc[-1] if len(rsi_series) > 0 else 50.0
    # Handle NaN
    if weekly_rsi != weekly_rsi:  # NaN check
        weekly_rsi = 50.0

    # Weekly MACD
    macd_line, signal_line = _compute_macd(close)
    macd_bullish = macd_line.iloc[-1] > signal_line.iloc[-1]

    # Determine weekly trend
    bullish_signals = 0
    bearish_signals = 0

    if ema_above:
        bullish_signals += 1
    else:
        bearish_signals += 1

    if weekly_rsi >= 50:
        bullish_signals += 1
    elif weekly_rsi < 45:
        bearish_signals += 1

    if macd_bullish:
        bullish_signals += 1
    else:
        bearish_signals += 1

    if bullish_signals >= 2:
        weekly_trend = "bullish"
        confidence = bullish_signals / 3.0
    elif bearish_signals >= 2:
        weekly_trend = "bearish"
        confidence = bearish_signals / 3.0
    else:
        weekly_trend = "neutral"
        confidence = 0.5

    # Determine alignment
    if weekly_trend == "neutral":
        aligned = True  # benefit of doubt
    elif direction == "bullish":
        aligned = weekly_trend == "bullish"
    else:
        aligned = weekly_trend == "bearish"

    result = {
        "aligned": aligned,
        "weekly_trend": weekly_trend,
        "weekly_rsi": round(weekly_rsi, 1),
        "weekly_ema_above": ema_above,
        "confidence": round(confidence, 2),
    }

    _cache[cache_key] = (now, result)
    return result


def check_multi_timeframe(symbol: str, direction: str) -> tuple[bool, str, float]:
    """Master function: check if trade direction aligns across timeframes.

    Returns: (aligned: bool, reason: str, confidence: float)
    """
    result = check_weekly_trend(symbol, direction)

    if result["aligned"]:
        reason = (
            f"Weekly trend {result['weekly_trend']} "
            f"(RSI={result['weekly_rsi']:.0f}, EMA={'above' if result['weekly_ema_above'] else 'below'})"
        )
        return True, reason, result["confidence"]
    else:
        reason = (
            f"Weekly trend {result['weekly_trend']} contradicts {direction} "
            f"(RSI={result['weekly_rsi']:.0f}, EMA={'above' if result['weekly_ema_above'] else 'below'})"
        )
        return False, reason, result["confidence"]
