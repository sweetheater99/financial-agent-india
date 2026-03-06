"""Unusual options activity detection.

Analyzes option chain data to detect unusual OI buildup that may signal
institutional positioning. Operates on existing option chain data — zero
additional API calls.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

CACHE_DIR = Path("data/oi_unusual_cache")

# Thresholds
OI_SPIKE_MULTIPLIER = 2.0    # OI must be 2x average to flag
MIN_OI_ABSOLUTE = 1000       # Minimum OI to consider (filter noise)
TOP_N_STRIKES = 5            # Number of top strikes to report


def detect_unusual_oi(option_chain: list, symbol: str = "") -> dict:
    """Detect unusual OI buildup in an option chain.

    Args:
        option_chain: list of option chain rows (from fetch_option_chain)
        symbol: stock symbol for logging

    Returns:
        {
            "has_unusual_activity": bool,
            "unusual_calls": [{strike, oi, signal}...],
            "unusual_puts": [{strike, oi, signal}...],
            "call_oi_total": int,
            "put_oi_total": int,
            "max_call_oi_strike": float,
            "max_put_oi_strike": float,
            "oi_skew": "call_heavy" | "put_heavy" | "balanced",
        }
    """
    if not option_chain:
        return {
            "has_unusual_activity": False,
            "unusual_calls": [], "unusual_puts": [],
            "call_oi_total": 0, "put_oi_total": 0,
            "max_call_oi_strike": 0, "max_put_oi_strike": 0,
            "oi_skew": "balanced",
        }

    call_strikes = []
    put_strikes = []

    for row in option_chain:
        strike = row.get("strikePrice", 0)

        # Call side
        ce = row.get("CE") or {}
        ce_oi = int(ce.get("openInterest", 0) or 0)
        ce_vol = int(ce.get("totalTradedVolume", 0) or 0)
        if ce_oi >= MIN_OI_ABSOLUTE:
            call_strikes.append({"strike": strike, "oi": ce_oi, "volume": ce_vol})

        # Put side
        pe = row.get("PE") or {}
        pe_oi = int(pe.get("openInterest", 0) or 0)
        pe_vol = int(pe.get("totalTradedVolume", 0) or 0)
        if pe_oi >= MIN_OI_ABSOLUTE:
            put_strikes.append({"strike": strike, "oi": pe_oi, "volume": pe_vol})

    # Compute averages
    call_ois = [s["oi"] for s in call_strikes]
    put_ois = [s["oi"] for s in put_strikes]

    avg_call_oi = sum(call_ois) / len(call_ois) if call_ois else 0
    avg_put_oi = sum(put_ois) / len(put_ois) if put_ois else 0

    # Detect spikes (OI > 2x average across all strikes)
    unusual_calls = []
    for s in call_strikes:
        if avg_call_oi > 0 and s["oi"] >= avg_call_oi * OI_SPIKE_MULTIPLIER:
            unusual_calls.append({
                "strike": s["strike"],
                "oi": s["oi"],
                "vs_avg": round(s["oi"] / avg_call_oi, 1),
                "signal": "resistance" if s["oi"] > avg_call_oi * 3 else "notable",
            })

    unusual_puts = []
    for s in put_strikes:
        if avg_put_oi > 0 and s["oi"] >= avg_put_oi * OI_SPIKE_MULTIPLIER:
            unusual_puts.append({
                "strike": s["strike"],
                "oi": s["oi"],
                "vs_avg": round(s["oi"] / avg_put_oi, 1),
                "signal": "support" if s["oi"] > avg_put_oi * 3 else "notable",
            })

    # Sort by OI descending, take top N
    unusual_calls.sort(key=lambda x: x["oi"], reverse=True)
    unusual_puts.sort(key=lambda x: x["oi"], reverse=True)
    unusual_calls = unusual_calls[:TOP_N_STRIKES]
    unusual_puts = unusual_puts[:TOP_N_STRIKES]

    # Totals
    call_oi_total = sum(call_ois)
    put_oi_total = sum(put_ois)

    max_call_strike = max(call_strikes, key=lambda x: x["oi"])["strike"] if call_strikes else 0
    max_put_strike = max(put_strikes, key=lambda x: x["oi"])["strike"] if put_strikes else 0

    # Skew
    total_oi = call_oi_total + put_oi_total
    if total_oi > 0:
        call_ratio = call_oi_total / total_oi
        if call_ratio > 0.6:
            oi_skew = "call_heavy"
        elif call_ratio < 0.4:
            oi_skew = "put_heavy"
        else:
            oi_skew = "balanced"
    else:
        oi_skew = "balanced"

    return {
        "has_unusual_activity": len(unusual_calls) > 0 or len(unusual_puts) > 0,
        "unusual_calls": unusual_calls,
        "unusual_puts": unusual_puts,
        "call_oi_total": call_oi_total,
        "put_oi_total": put_oi_total,
        "max_call_oi_strike": max_call_strike,
        "max_put_oi_strike": max_put_strike,
        "oi_skew": oi_skew,
    }


def get_oi_signal_for_trade(unusual_oi: dict, direction: str, spot_price: float) -> dict:
    """Interpret unusual OI for a trade direction.

    Returns:
        {
            "signal": "confirming" | "cautionary" | "neutral",
            "reason": str,
            "key_level": float or None,  # nearest support/resistance from OI
        }
    """
    if not unusual_oi.get("has_unusual_activity"):
        return {"signal": "neutral", "reason": "no unusual OI activity", "key_level": None}

    if direction == "bullish":
        # For bullish: heavy put OI below spot = support = confirming
        # Heavy call OI above spot = resistance = cautionary
        put_supports = [p for p in unusual_oi["unusual_puts"]
                       if p["strike"] < spot_price and p["signal"] == "support"]
        call_resistances = [c for c in unusual_oi["unusual_calls"]
                          if c["strike"] > spot_price and c["signal"] == "resistance"]

        if put_supports and not call_resistances:
            nearest = max(put_supports, key=lambda x: x["strike"])
            return {
                "signal": "confirming",
                "reason": f"strong put OI support at {nearest['strike']} ({nearest['vs_avg']}x avg)",
                "key_level": nearest["strike"],
            }
        elif call_resistances and not put_supports:
            nearest = min(call_resistances, key=lambda x: x["strike"])
            return {
                "signal": "cautionary",
                "reason": f"heavy call OI resistance at {nearest['strike']} ({nearest['vs_avg']}x avg)",
                "key_level": nearest["strike"],
            }

    elif direction == "bearish":
        # For bearish: heavy call OI above spot = ceiling = confirming
        # Heavy put OI below = floor = cautionary
        call_ceilings = [c for c in unusual_oi["unusual_calls"]
                        if c["strike"] > spot_price and c["signal"] == "resistance"]
        put_floors = [p for p in unusual_oi["unusual_puts"]
                     if p["strike"] < spot_price and p["signal"] == "support"]

        if call_ceilings and not put_floors:
            nearest = min(call_ceilings, key=lambda x: x["strike"])
            return {
                "signal": "confirming",
                "reason": f"heavy call OI ceiling at {nearest['strike']} ({nearest['vs_avg']}x avg)",
                "key_level": nearest["strike"],
            }
        elif put_floors and not call_ceilings:
            nearest = max(put_floors, key=lambda x: x["strike"])
            return {
                "signal": "cautionary",
                "reason": f"strong put OI floor at {nearest['strike']} ({nearest['vs_avg']}x avg)",
                "key_level": nearest["strike"],
            }

    return {"signal": "neutral", "reason": "mixed OI signals", "key_level": None}
