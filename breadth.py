"""Market breadth analysis.

Computes advance/decline metrics for Indian markets.
Uses yfinance for Nifty 50 constituent data.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
import json

logger = logging.getLogger("paper_trade")

CACHE_DIR = Path("data/breadth_cache")

# Nifty 50 top constituents (by weight) — sufficient for breadth signal
# Using .NS suffix for yfinance
NIFTY50_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "HCLTECH.NS", "SUNPHARMA.NS", "TITAN.NS", "WIPRO.NS", "ULTRACEMCO.NS",
    "NTPC.NS", "POWERGRID.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "ONGC.NS",
    "NESTLEIND.NS", "JSWSTEEL.NS", "M&M.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "BAJAJFINSV.NS", "TECHM.NS", "INDUSINDBK.NS", "HINDALCO.NS", "COALINDIA.NS",
    "DRREDDY.NS", "DIVISLAB.NS", "CIPLA.NS", "EICHERMOT.NS", "HEROMOTOCO.NS",
    "BPCL.NS", "GRASIM.NS", "APOLLOHOSP.NS", "TATACONSUM.NS", "BRITANNIA.NS",
    "SBILIFE.NS", "HDFCLIFE.NS", "BAJAJ-AUTO.NS", "BEL.NS", "SHRIRAMFIN.NS",
]


def _get_cache(cache_key: str, max_age_hours: float = 6) -> dict | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            cached_at = datetime.fromisoformat(data.get("_cached_at", "2000-01-01"))
            if datetime.now() - cached_at < timedelta(hours=max_age_hours):
                return data.get("result")
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _set_cache(cache_key: str, result: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    cache_file.write_text(json.dumps({
        "_cached_at": datetime.now().isoformat(),
        "result": result,
    }, indent=2))


def compute_breadth() -> dict | None:
    """Compute Nifty 50 market breadth.

    Returns dict with:
        - advancing: count of stocks above 20-day EMA
        - declining: count below 20-day EMA
        - breadth_pct: advancing / total * 100
        - signal: "healthy" (>60%), "weak" (40-60%), "divergent" (<40%)
        - total: number of stocks checked

    Returns None on failure.
    """
    cache_key = f"breadth_{datetime.now().strftime('%Y-%m-%d')}"
    cached = _get_cache(cache_key, max_age_hours=6)
    if cached:
        return cached

    try:
        import yfinance as yf

        advancing = 0
        declining = 0
        total = 0

        # Batch download is faster than individual
        tickers = " ".join(NIFTY50_SYMBOLS)
        data = yf.download(tickers, period="30d", progress=False, threads=True)

        if data.empty:
            return None

        close = data["Close"]

        for sym in NIFTY50_SYMBOLS:
            try:
                prices = close[sym].dropna()
                if len(prices) < 20:
                    continue
                ema20 = prices.ewm(span=20).mean()
                total += 1
                if prices.iloc[-1] > ema20.iloc[-1]:
                    advancing += 1
                else:
                    declining += 1
            except Exception:
                continue

        if total == 0:
            return None

        breadth_pct = round(advancing / total * 100, 1)

        if breadth_pct >= 60:
            signal = "healthy"
        elif breadth_pct >= 40:
            signal = "weak"
        else:
            signal = "divergent"

        result = {
            "advancing": advancing,
            "declining": declining,
            "total": total,
            "breadth_pct": breadth_pct,
            "signal": signal,
        }

        _set_cache(cache_key, result)
        return result

    except Exception as e:
        logger.debug("Breadth computation failed: %s", e)
        return None
