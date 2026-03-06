"""Sector rotation signal — tracks relative sector momentum week-over-week.

Uses yfinance to fetch sector-representative ETFs/indices and computes
1-week vs 4-week relative strength to detect rotation.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

CACHE_DIR = Path("data/sector_rotation_cache")

# Representative stocks per sector (most liquid names)
SECTOR_REPRESENTATIVES = {
    "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"],
    "IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS"],
    "Pharma": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS"],
    "Auto": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS"],
    "Metals": ["TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS"],
    "Energy": ["RELIANCE.NS", "ONGC.NS", "NTPC.NS"],
    "FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS"],
    "FinServices": ["BAJFINANCE.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS"],
    "Infra": ["LT.NS", "ULTRACEMCO.NS", "ADANIENT.NS"],
}

def _fetch_sector_returns():
    """Fetch 1-week and 4-week returns for each sector.

    Returns dict: {sector: {"1w_return": float, "4w_return": float, "momentum_score": float}}
    momentum_score = 1w_return - 4w_return (positive = accelerating, negative = decelerating)
    """
    import yfinance as yf

    results = {}
    all_tickers = []
    ticker_to_sector = {}

    for sector, tickers in SECTOR_REPRESENTATIVES.items():
        for t in tickers:
            all_tickers.append(t)
            ticker_to_sector[t] = sector

    # Batch download (efficient)
    try:
        data = yf.download(all_tickers, period="2mo", interval="1d", progress=False, threads=True)
        if data.empty:
            return {}
    except Exception:
        return {}

    close = data["Close"] if "Close" in data.columns.get_level_values(0) else data

    for sector, tickers in SECTOR_REPRESENTATIVES.items():
        valid_1w = []
        valid_4w = []
        for t in tickers:
            if t not in close.columns:
                continue
            prices = close[t].dropna()
            if len(prices) < 5:
                continue
            # 1-week return (5 trading days)
            ret_1w = (prices.iloc[-1] / prices.iloc[-5] - 1) * 100 if len(prices) >= 5 else 0
            valid_1w.append(ret_1w)
            # 4-week return (20 trading days)
            if len(prices) >= 20:
                ret_4w = (prices.iloc[-1] / prices.iloc[-20] - 1) * 100
                valid_4w.append(ret_4w)

        if not valid_1w:
            continue

        avg_1w = sum(valid_1w) / len(valid_1w)
        avg_4w = sum(valid_4w) / len(valid_4w) if valid_4w else avg_1w

        results[sector] = {
            "1w_return": round(avg_1w, 2),
            "4w_return": round(avg_4w, 2),
            "momentum_score": round(avg_1w - avg_4w / 4, 2),  # normalized
        }

    return results


def classify_sector_momentum(sector_returns: dict) -> dict:
    """Classify sectors into strengthening, weakening, and neutral.

    Returns:
        {
            "strengthening": [sector_names...],  # momentum_score > 0.5
            "weakening": [sector_names...],       # momentum_score < -0.5
            "neutral": [sector_names...],
            "top_sector": sector_name,
            "worst_sector": sector_name,
            "rotation_detected": bool,  # True if top/worst are very different
            "details": {sector: {1w_return, 4w_return, momentum_score}...},
        }
    """
    if not sector_returns:
        return {
            "strengthening": [], "weakening": [], "neutral": [],
            "top_sector": None, "worst_sector": None,
            "rotation_detected": False, "details": {},
        }

    strengthening = []
    weakening = []
    neutral = []

    for sector, data in sector_returns.items():
        score = data["momentum_score"]
        if score > 0.5:
            strengthening.append(sector)
        elif score < -0.5:
            weakening.append(sector)
        else:
            neutral.append(sector)

    sorted_sectors = sorted(sector_returns.items(), key=lambda x: x[1]["momentum_score"], reverse=True)
    top = sorted_sectors[0][0] if sorted_sectors else None
    worst = sorted_sectors[-1][0] if sorted_sectors else None

    # Rotation detected if spread between top and worst momentum > 2
    spread = 0
    if top and worst:
        spread = sector_returns[top]["momentum_score"] - sector_returns[worst]["momentum_score"]

    return {
        "strengthening": strengthening,
        "weakening": weakening,
        "neutral": neutral,
        "top_sector": top,
        "worst_sector": worst,
        "rotation_detected": spread > 2.0,
        "details": sector_returns,
    }


def _get_cache(max_age_hours=6):
    """Read from file cache if fresh enough."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"rotation_{datetime.now().strftime('%Y-%m-%d')}.json"
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            cached_at = datetime.fromisoformat(data.get("_cached_at", "2000-01-01"))
            if datetime.now() - cached_at < timedelta(hours=max_age_hours):
                return data.get("result")
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _set_cache(result):
    """Write result to file cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"rotation_{datetime.now().strftime('%Y-%m-%d')}.json"
    cache_file.write_text(json.dumps({
        "_cached_at": datetime.now().isoformat(),
        "result": result,
    }, indent=2))


def get_sector_rotation() -> dict:
    """Public API: fetch sector rotation signal with caching.

    Returns classification dict with strengthening/weakening sectors.
    """
    cached = _get_cache(max_age_hours=6)
    if cached is not None:
        return cached

    sector_returns = _fetch_sector_returns()
    result = classify_sector_momentum(sector_returns)
    if result["details"]:
        _set_cache(result)
    return result
