"""
Technical indicators for screener and paper trading.

Shared module: RSI, volume ratio, and ATR computations used by both
screener.py (scoring adjustments) and paper_trade.py (ATR-based exits).

Candle format: [date, open, high, low, close, volume] (SmartAPI getCandleData).
"""


def compute_rsi(candles: list, period: int = 14) -> float | None:
    """
    Compute RSI using Wilder's smoothing method.

    Needs at least period+1 candles. Returns RSI in [0, 100] or None.
    """
    if not candles or len(candles) < period + 1:
        return None

    # Close-over-close changes
    changes = []
    for i in range(1, len(candles)):
        changes.append(float(candles[i][4]) - float(candles[i - 1][4]))

    if len(changes) < period:
        return None

    # Initial averages (simple mean of first `period` changes)
    gains = [max(c, 0) for c in changes[:period]]
    losses = [abs(min(c, 0)) for c in changes[:period]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    # Wilder's smoothing for remaining changes
    for c in changes[period:]:
        gain = max(c, 0)
        loss = abs(min(c, 0))
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return round(rsi, 2)


def compute_volume_ratio(candles: list, lookback: int = 20) -> float | None:
    """
    Compute today's volume relative to the average of the last `lookback` days.

    Needs at least lookback+1 candles. Returns ratio (e.g. 1.5 = 50% above avg).
    """
    if not candles or len(candles) < lookback + 1:
        return None

    today_volume = float(candles[-1][5])
    past_volumes = [float(c[5]) for c in candles[-(lookback + 1):-1]]

    avg_volume = sum(past_volumes) / len(past_volumes)
    if avg_volume <= 0:
        return None

    return round(today_volume / avg_volume, 2)


def compute_atr(candles: list, period: int = 14) -> float | None:
    """
    Compute Average True Range from candle data.

    Returns ATR float or None if insufficient data.
    """
    if not candles or len(candles) < period + 1:
        return None

    true_ranges = []
    for i in range(1, len(candles)):
        high = float(candles[i][2])
        low = float(candles[i][3])
        prev_close = float(candles[i - 1][4])
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)

    if len(true_ranges) < period:
        return None

    # Simple moving average of last `period` true ranges
    return round(sum(true_ranges[-period:]) / period, 2)
