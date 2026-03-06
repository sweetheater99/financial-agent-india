"""Global macro intelligence for Trading System V3.

Fetches US markets (S&P 500, Nasdaq), GIFT Nifty gap, Asia markets,
FII/DII flows, and computes hard gate decisions.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger("paper_trade")

CACHE_DIR = Path("data/global_intel_cache")


def _get_cache(cache_key: str, max_age_hours: float) -> dict | None:
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


def _pct_change_from_history(hist: pd.DataFrame) -> float:
    if hist is None or hist.empty or len(hist) < 2:
        return 0.0
    closes = hist["Close"].values
    prev, current = float(closes[-2]), float(closes[-1])
    if prev == 0:
        return 0.0
    return round(((current - prev) / prev) * 100, 2)


def _compute_us_market_data(sp500_hist: pd.DataFrame, nasdaq_hist: pd.DataFrame) -> dict:
    return {
        "sp500_pct_change": _pct_change_from_history(sp500_hist),
        "nasdaq_pct_change": _pct_change_from_history(nasdaq_hist),
    }


def fetch_us_markets() -> dict:
    cache_key = f"us_markets_{datetime.now().strftime('%Y-%m-%d')}"
    cached = _get_cache(cache_key, config.GLOBAL_INTEL_CACHE_HOURS)
    if cached:
        return cached
    try:
        import yfinance as yf
        sp500 = yf.Ticker("^GSPC").history(period="2d")
        nasdaq = yf.Ticker("^IXIC").history(period="2d")
        result = _compute_us_market_data(sp500, nasdaq)
    except Exception as e:
        logger.debug("US market fetch failed: %s", e)
        result = {"sp500_pct_change": 0.0, "nasdaq_pct_change": 0.0}
    _set_cache(cache_key, result)
    return result


def _compute_gift_nifty_gap(gift_ltp: float | None, prev_nifty_close: float | None) -> dict:
    if gift_ltp is None or prev_nifty_close is None or prev_nifty_close == 0:
        return {"gift_nifty_gap_pct": 0.0, "gift_nifty_ltp": 0.0}
    gap_pct = round(((gift_ltp - prev_nifty_close) / prev_nifty_close) * 100, 2)
    return {"gift_nifty_gap_pct": gap_pct, "gift_nifty_ltp": gift_ltp}


def fetch_gift_nifty_gap(prev_nifty_close: float) -> dict:
    cache_key = f"gift_nifty_{datetime.now().strftime('%Y-%m-%d')}"
    cached = _get_cache(cache_key, config.GLOBAL_INTEL_CACHE_HOURS)
    if cached:
        return cached
    try:
        import yfinance as yf
        nifty_fut = yf.Ticker("^NSEI").history(period="1d")
        ltp = float(nifty_fut["Close"].iloc[-1]) if not nifty_fut.empty else None
        result = _compute_gift_nifty_gap(ltp, prev_nifty_close)
    except Exception as e:
        logger.debug("GIFT Nifty fetch failed: %s", e)
        result = {"gift_nifty_gap_pct": 0.0, "gift_nifty_ltp": 0.0}
    _set_cache(cache_key, result)
    return result


def _compute_asia_data(hang_seng_hist: pd.DataFrame, nikkei_hist: pd.DataFrame) -> dict:
    return {
        "hang_seng_pct": _pct_change_from_history(hang_seng_hist),
        "nikkei_pct": _pct_change_from_history(nikkei_hist),
    }


def fetch_asia_markets() -> dict:
    cache_key = f"asia_markets_{datetime.now().strftime('%Y-%m-%d')}"
    cached = _get_cache(cache_key, config.GLOBAL_INTEL_CACHE_HOURS)
    if cached:
        return cached
    try:
        import yfinance as yf
        hang_seng = yf.Ticker("^HSI").history(period="2d")
        nikkei = yf.Ticker("^N225").history(period="2d")
        result = _compute_asia_data(hang_seng, nikkei)
    except Exception as e:
        logger.debug("Asia market fetch failed: %s", e)
        result = {"hang_seng_pct": 0.0, "nikkei_pct": 0.0}
    _set_cache(cache_key, result)
    return result


def fetch_fii_dii() -> dict:
    cache_key = f"fii_dii_{datetime.now().strftime('%Y-%m-%d')}"
    cached = _get_cache(cache_key, config.FII_DII_CACHE_HOURS)
    if cached:
        return cached
    result = {"fii_net_crores": 0, "dii_net_crores": 0, "date": "", "source": "unavailable"}
    try:
        import requests
        from bs4 import BeautifulSoup
        url = "https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/index.php"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all("td")
                if len(cells) >= 4:
                    label = cells[0].get_text(strip=True).upper()
                    if "FII" in label or "FPI" in label:
                        try:
                            buy = float(cells[1].get_text(strip=True).replace(",", ""))
                            sell = float(cells[2].get_text(strip=True).replace(",", ""))
                            result["fii_net_crores"] = round(buy - sell, 2)
                            result["source"] = "moneycontrol"
                        except ValueError:
                            pass
                    elif "DII" in label:
                        try:
                            buy = float(cells[1].get_text(strip=True).replace(",", ""))
                            sell = float(cells[2].get_text(strip=True).replace(",", ""))
                            result["dii_net_crores"] = round(buy - sell, 2)
                        except ValueError:
                            pass
        result["date"] = datetime.now().strftime("%Y-%m-%d")
    except Exception as e:
        logger.debug("FII/DII scrape failed: %s", e)
    _set_cache(cache_key, result)
    return result


def compute_hard_gate(
    sp500_pct: float = 0,
    nasdaq_pct: float = 0,
    gift_gap_pct: float = 0,
    fii_net: float = 0,
    dii_net: float = 0,
    pcr: float = 1.0,
) -> dict:
    """Compute the strictest hard gate from all macro signals.

    Actions (strictest to mildest): BLOCK_ALL, BLOCK_BULLISH, BLOCK_IT_BULLISH, REDUCE_50, REDUCE_25, NONE
    """
    if sp500_pct <= -config.US_SEVERE_CRASH_PCT:
        return {"action": "BLOCK_ALL", "reason": f"S&P 500 {sp500_pct:+.1f}% severe crash"}

    reasons = []

    if sp500_pct <= -config.US_CRASH_BLOCK_PCT:
        reasons.append(("BLOCK_BULLISH", f"S&P 500 {sp500_pct:+.1f}% crash"))

    if gift_gap_pct <= -config.GIFT_GAP_BLOCK_PCT:
        reasons.append(("BLOCK_BULLISH", f"GIFT Nifty gap {gift_gap_pct:+.1f}%"))

    fii_threshold_heavy = config.FII_HEAVY_SELL_CRORES
    fii_threshold_extreme = config.FII_EXTREME_SELL_CRORES
    if dii_net > 0 and fii_net < 0:
        fii_threshold_heavy *= (1 + config.DII_SUPPORT_MODERATE_PCT)
        fii_threshold_extreme *= (1 + config.DII_SUPPORT_MODERATE_PCT)

    if fii_net <= -fii_threshold_extreme:
        reasons.append(("BLOCK_BULLISH", f"FII net sell {abs(fii_net):.0f}cr (extreme)"))

    if pcr < config.PCR_EUPHORIA:
        reasons.append(("BLOCK_BULLISH", f"PCR {pcr:.2f} euphoria"))

    block_bullish = [r for r in reasons if r[0] == "BLOCK_BULLISH"]
    if block_bullish:
        return {"action": "BLOCK_BULLISH", "reason": block_bullish[0][1]}

    if nasdaq_pct <= -config.NASDAQ_IT_CRASH_PCT:
        return {"action": "BLOCK_IT_BULLISH", "reason": f"Nasdaq {nasdaq_pct:+.1f}% crash"}

    reduce_50_reasons = []
    if -config.US_CRASH_BLOCK_PCT < sp500_pct <= -config.US_MILD_RED_PCT:
        reduce_50_reasons.append(f"S&P 500 {sp500_pct:+.1f}%")
    if -config.GIFT_GAP_BLOCK_PCT < gift_gap_pct <= -config.GIFT_GAP_REDUCE_PCT:
        reduce_50_reasons.append(f"GIFT gap {gift_gap_pct:+.1f}%")
    if -fii_threshold_extreme < fii_net <= -fii_threshold_heavy:
        reduce_50_reasons.append(f"FII net sell {abs(fii_net):.0f}cr")

    if reduce_50_reasons:
        return {"action": "REDUCE_50", "reason": "; ".join(reduce_50_reasons)}

    if pcr < config.PCR_EXTREME_CALL:
        return {"action": "REDUCE_25", "reason": f"PCR {pcr:.2f} extreme call buying"}

    return {"action": "NONE", "reason": "Global cues neutral"}


def fetch_macro_context(prev_nifty_close: float = 0, pcr: float = 1.0) -> dict:
    us = fetch_us_markets()
    gift = fetch_gift_nifty_gap(prev_nifty_close) if prev_nifty_close > 0 else {"gift_nifty_gap_pct": 0.0, "gift_nifty_ltp": 0.0}
    asia = fetch_asia_markets()
    fii_dii = fetch_fii_dii()

    gate = compute_hard_gate(
        sp500_pct=us["sp500_pct_change"],
        nasdaq_pct=us["nasdaq_pct_change"],
        gift_gap_pct=gift["gift_nifty_gap_pct"],
        fii_net=fii_dii["fii_net_crores"],
        dii_net=fii_dii["dii_net_crores"],
        pcr=pcr,
    )

    return {
        **us, **gift, **asia,
        "fii_net_crores": fii_dii["fii_net_crores"],
        "dii_net_crores": fii_dii["dii_net_crores"],
        "hard_gate": gate["action"],
        "hard_gate_reason": gate["reason"],
        "fetched_at": datetime.now().isoformat(),
    }
