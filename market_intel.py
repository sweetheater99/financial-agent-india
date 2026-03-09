"""
Market Intelligence — VWAP computation, OI delta classification, OI support/resistance.

Self-contained module. Only imports config for API_DELAY.
"""

import logging
import time
from datetime import datetime, timedelta, timezone

import config

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))


# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------

def compute_vwap_from_candles(candles: list[dict]) -> float | None:
    """Compute VWAP from intraday candle data.

    Each candle: dict with keys "high", "low", "close", "volume".
    VWAP = sum(typical_price * volume) / sum(volume)
    where typical_price = (high + low + close) / 3.
    Returns float or None if empty / zero volume.
    """
    if not candles:
        return None

    cum_tp_vol = 0.0
    cum_vol = 0.0
    for c in candles:
        tp = (c["high"] + c["low"] + c["close"]) / 3
        cum_tp_vol += tp * c["volume"]
        cum_vol += c["volume"]

    if cum_vol == 0:
        return None

    return cum_tp_vol / cum_vol


def fetch_vwap(smart_api, token: str) -> float | None:
    """Fetch 5-min candles from AngelOne for today and compute VWAP.

    Uses smart_api.getCandleData() with FIVE_MINUTE interval from 9:15 AM
    IST today to now. Respects API_DELAY between calls.
    Returns float or None on failure.
    """
    try:
        now = datetime.now(IST)
        today = now.strftime("%Y-%m-%d")
        params = {
            "exchange": "NSE",
            "symboltoken": token,
            "interval": "FIVE_MINUTE",
            "fromdate": f"{today} 09:15",
            "todate": now.strftime("%Y-%m-%d %H:%M"),
        }
        time.sleep(config.API_DELAY)
        resp = smart_api.getCandleData(params)
        if not resp or not resp.get("data"):
            logger.warning("No candle data for token %s", token)
            return None

        # AngelOne candle format: [timestamp, open, high, low, close, volume]
        candles = []
        for raw in resp["data"]:
            candles.append({
                "high": raw[2],
                "low": raw[3],
                "close": raw[4],
                "volume": raw[5],
            })
        return compute_vwap_from_candles(candles)

    except Exception as e:
        logger.error("fetch_vwap failed for token %s: %s", token, e)
        return None


# ---------------------------------------------------------------------------
# OI Delta Classification
# ---------------------------------------------------------------------------

def classify_oi_change(oi_change_pct: float, price_change_pct: float) -> str:
    """Classify OI pattern based on OI and price change direction.

    Returns one of: long_buildup, short_buildup, short_covering, long_unwinding.
    """
    if oi_change_pct >= 0 and price_change_pct >= 0:
        return "long_buildup"
    elif oi_change_pct >= 0 and price_change_pct < 0:
        return "short_buildup"
    elif oi_change_pct < 0 and price_change_pct >= 0:
        return "short_covering"
    else:
        return "long_unwinding"


# ---------------------------------------------------------------------------
# OI Support / Resistance
# ---------------------------------------------------------------------------

def get_oi_support_resistance(option_chain: dict) -> dict:
    """Find strikes with maximum OI for puts (support) and calls (resistance).

    Input: dict with "CE" and "PE" keys, each a list of {"strike": int, "oi": int}.
    Returns dict with support_strike, resistance_strike, max_put_oi, max_call_oi.
    """
    ce_list = option_chain.get("CE", [])
    pe_list = option_chain.get("PE", [])

    max_call = max(ce_list, key=lambda x: x["oi"]) if ce_list else {"strike": 0, "oi": 0}
    max_put = max(pe_list, key=lambda x: x["oi"]) if pe_list else {"strike": 0, "oi": 0}

    return {
        "support_strike": max_put["strike"],
        "resistance_strike": max_call["strike"],
        "max_put_oi": max_put["oi"],
        "max_call_oi": max_call["oi"],
    }


# ---------------------------------------------------------------------------
# Formatting for Claude prompt
# ---------------------------------------------------------------------------

def format_market_intel_for_claude(
    vwap: float | None,
    ltp: float,
    oi_change: dict | None,
    oi_sr: dict | None,
) -> str:
    """Format market intelligence signals into a multiline string for Claude.

    Args:
        vwap: VWAP value (or None).
        ltp: Last traded price.
        oi_change: dict with "oi_change_pct" and "price_change_pct" keys, or None.
        oi_sr: dict from get_oi_support_resistance, or None.

    Returns:
        Multiline string with VWAP, OI delta, and OI S/R sections.
    """
    lines = ["## Market Intelligence"]

    # VWAP section
    if vwap is not None:
        diff = ltp - vwap
        pct = (diff / vwap) * 100 if vwap else 0
        bias = "BULLISH" if ltp > vwap else "BEARISH"
        lines.append(f"VWAP: {vwap:.2f} | LTP: {ltp:.2f} | Diff: {diff:+.2f} ({pct:+.2f}%)")
        lines.append(f"VWAP Confirmation: {bias} (price {'above' if ltp > vwap else 'below'} VWAP)")
    else:
        lines.append("VWAP: unavailable")

    # OI delta section
    if oi_change is not None:
        oi_pct = oi_change["oi_change_pct"]
        price_pct = oi_change["price_change_pct"]
        pattern = classify_oi_change(oi_pct, price_pct)
        lines.append(f"OI Delta: OI {oi_pct:+.2f}% | Price {price_pct:+.2f}% → {pattern}")
    else:
        lines.append("OI Delta: unavailable")

    # OI Support/Resistance section
    if oi_sr is not None:
        lines.append(
            f"OI S/R: Support {oi_sr['support_strike']} "
            f"(Put OI: {oi_sr['max_put_oi']:,}) | "
            f"Resistance {oi_sr['resistance_strike']} "
            f"(Call OI: {oi_sr['max_call_oi']:,})"
        )
    else:
        lines.append("OI S/R: unavailable")

    return "\n".join(lines)
