# v7/strike_selector.py
"""Mechanical strike selection for V7.

No Claude calls. Pure rules based on delta, liquidity, and budget.
"""
from __future__ import annotations

from v7.config_v7 import STRIKE_FILTERS


def passes_liquidity_filter(oi: int, volume: int, bid_ask_spread: float,
                            symbol: str) -> bool:
    """Check if a strike passes minimum liquidity requirements."""
    if oi < STRIKE_FILTERS["min_oi"]:
        return False
    if volume < STRIKE_FILTERS["min_volume"]:
        return False

    max_spread = STRIKE_FILTERS["max_bid_ask_stock"]
    if symbol == "NIFTY":
        max_spread = STRIKE_FILTERS["max_bid_ask_nifty"]
    elif symbol == "BANKNIFTY":
        max_spread = STRIKE_FILTERS["max_bid_ask_banknifty"]

    if bid_ask_spread > max_spread:
        return False
    return True


def _get_bid_ask_spread(option_data: dict) -> float:
    bid = option_data.get("bidPrice", 0)
    ask = option_data.get("askPrice", 0)
    if bid and ask:
        return ask - bid
    return 999.0


def _estimate_delta(strike: float, spot: float, option_type: str) -> float:
    """Estimate delta from moneyness when Greeks aren't available.

    ATM ~ 0.50, each 1% OTM reduces delta by ~0.04 (weekly options
    with typical IV of 20-40% decay slower than the naive 0.08).
    """
    moneyness = (spot - strike) / spot  # positive = ITM for CE
    if option_type == "PE":
        moneyness = -moneyness  # positive = ITM for PE
    # Moderate linear approximation — 0.04 per 1% OTM
    delta = 0.50 + moneyness * 4.0
    return max(0.05, min(0.95, delta))


def select_directional_strike(
    chain: list[dict], direction: str, spot: float,
    risk_budget: float, lot_size: int, symbol: str,
    target_delta_override: float | None = None,
) -> dict | None:
    """Select the best strike for a directional option buy."""
    option_type = "CE" if direction == "bullish" else "PE"
    delta_min, delta_max = STRIKE_FILTERS["directional_delta_range"]
    min_premium = STRIKE_FILTERS["min_premium"]
    # risk_budget is total capital at risk; max_premium = per-unit cap
    # Floor at 500 Rs/unit to ensure ATM options pass (paper trading viability)
    max_premium = max(risk_budget / lot_size, 500.0)

    candidates = []
    for entry in chain:
        opt = entry.get(option_type)
        if not opt or not opt.get("ltp"):
            continue

        delta = abs(opt.get("delta", 0))
        # Kite doesn't provide Greeks — estimate from moneyness
        if delta == 0:
            delta = _estimate_delta(entry["strikePrice"], spot, option_type)

        premium = opt["ltp"]
        oi = opt.get("oi", 0)
        volume = opt.get("volume", 0)
        spread = _get_bid_ask_spread(opt)

        if delta < delta_min or delta > delta_max:
            continue
        if premium < min_premium or premium > max_premium:
            continue
        if not passes_liquidity_filter(oi, volume, spread, symbol):
            continue

        candidates.append({
            "strike": entry["strikePrice"],
            "option_type": option_type,
            "premium": premium,
            "delta": delta,
            "oi": oi,
            "volume": volume,
            "bid_ask_spread": spread,
            "tradingsymbol": opt.get("tradingsymbol", ""),
            "lot_size": opt.get("lotSize", lot_size),
        })

    # Relaxed fallback: widen delta range, keep budget constraint
    if not candidates:
        for entry in chain:
            opt = entry.get(option_type)
            if not opt or not opt.get("ltp"):
                continue
            delta = abs(opt.get("delta", 0))
            if delta == 0:
                delta = _estimate_delta(entry["strikePrice"], spot, option_type)
            premium = opt["ltp"]
            if 0.15 <= delta <= 0.55 and min_premium <= premium <= max_premium:
                oi = opt.get("oi", 0)
                volume = opt.get("volume", 0)
                spread = _get_bid_ask_spread(opt)
                if passes_liquidity_filter(oi, volume, spread, symbol):
                    candidates.append({
                        "strike": entry["strikePrice"],
                        "option_type": option_type,
                        "premium": premium,
                        "delta": delta,
                        "oi": oi,
                        "volume": volume,
                        "bid_ask_spread": spread,
                        "tradingsymbol": opt.get("tradingsymbol", ""),
                        "lot_size": opt.get("lotSize", lot_size),
                    })

    # Last resort: widest delta, drop liquidity filter (illiquid stock options)
    if not candidates:
        for entry in chain:
            opt = entry.get(option_type)
            if not opt or not opt.get("ltp"):
                continue
            delta = abs(opt.get("delta", 0))
            if delta == 0:
                delta = _estimate_delta(entry["strikePrice"], spot, option_type)
            premium = opt["ltp"]
            oi = opt.get("oi", 0)
            if 0.15 <= delta <= 0.60 and min_premium <= premium <= max_premium and oi >= 100:
                candidates.append({
                    "strike": entry["strikePrice"],
                    "option_type": option_type,
                    "premium": premium,
                    "delta": delta,
                    "oi": oi,
                    "volume": opt.get("volume", 0),
                    "bid_ask_spread": _get_bid_ask_spread(opt),
                    "tradingsymbol": opt.get("tradingsymbol", ""),
                    "lot_size": opt.get("lotSize", lot_size),
                })

    if not candidates:
        return None

    target_delta = target_delta_override if target_delta_override is not None else 0.45
    candidates.sort(key=lambda c: (abs(c["delta"] - target_delta), -c["oi"]))
    return candidates[0]


def select_spread_strikes(
    chain: list[dict], direction: str, spot: float,
    risk_budget: float, lot_size: int, symbol: str,
) -> dict | None:
    """Select strikes for a credit spread."""
    sell_delta = STRIKE_FILTERS["spread_sell_delta"]

    # Bear credit spread = sell CE (call spread), Bull credit spread = sell PE (put spread)
    if direction == "bearish":
        option_type = "CE"
    else:
        option_type = "PE"

    # Pick sell strike at ~0.5% OTM from spot (delta estimator is unreliable for weeklies).
    # For bear call: sell strike ABOVE spot. For bull put: sell strike BELOW spot.
    if direction == "bearish":
        target_strike = spot * 1.005
    else:
        target_strike = spot * 0.995

    sell_candidates = []
    for entry in chain:
        opt = entry.get(option_type)
        if not opt or not opt.get("ltp"):
            continue
        # Must be on the correct side of spot for the spread direction
        if direction == "bearish" and entry["strikePrice"] <= spot:
            continue
        if direction == "bullish" and entry["strikePrice"] >= spot:
            continue
        # Premium must be reasonable (avoid deep OTM lottery strikes)
        if opt["ltp"] < 5:
            continue
        oi = opt.get("oi", 0)
        volume = opt.get("volume", 0)
        spread = _get_bid_ask_spread(opt)
        if not passes_liquidity_filter(oi, volume, spread, symbol):
            continue
        sell_candidates.append({
            "strike": entry["strikePrice"],
            "premium": opt["ltp"],
            "delta": abs(opt.get("delta", 0)) or _estimate_delta(entry["strikePrice"], spot, option_type),
        })

    if not sell_candidates:
        return None

    # Pick the strike closest to target_strike
    sell_candidates.sort(key=lambda c: abs(c["strike"] - target_strike))
    sell = sell_candidates[0]

    strikes = sorted(set(e["strikePrice"] for e in chain))
    sell_idx = strikes.index(sell["strike"]) if sell["strike"] in strikes else -1
    if sell_idx < 0:
        return None

    # Bear call spread: buy HIGHER strike CE (further OTM protection)
    # Bull put spread: buy LOWER strike PE (further OTM protection)
    # 1 strike width (50pt for NIFTY) — keeps max_loss small enough to fit risk budget
    if direction == "bearish":
        buy_idx = sell_idx + 1 if sell_idx + 1 < len(strikes) else len(strikes) - 1
    else:
        buy_idx = sell_idx - 1 if sell_idx >= 1 else 0

    buy_strike = strikes[buy_idx]
    buy_entry = next((e for e in chain if e["strikePrice"] == buy_strike), None)
    if not buy_entry:
        return None

    buy_opt = buy_entry.get(option_type)
    if not buy_opt or not buy_opt.get("ltp"):
        return None

    net_credit = sell["premium"] - buy_opt["ltp"]
    if net_credit < 5:
        return None

    strike_width = abs(sell["strike"] - buy_strike)
    max_loss = (strike_width - net_credit) * lot_size

    # Allow up to 1.25x risk_budget — daily gate will reject if it overflows total daily risk.
    # NIFTY 50pt-wide spread can hit ₹3750 max loss; with low credit it overshoots a tight ₹3000 cap.
    if max_loss > risk_budget * 1.25:
        return None

    # Get tradingsymbols for both legs
    sell_entry = next((e for e in chain if e["strikePrice"] == sell["strike"]), None)
    sell_ts = sell_entry.get(option_type, {}).get("tradingsymbol", "") if sell_entry else ""
    buy_ts = buy_opt.get("tradingsymbol", "")

    return {
        "sell_strike": sell["strike"],
        "buy_strike": buy_strike,
        "sell_premium": sell["premium"],
        "buy_premium": buy_opt["ltp"],
        "sell_tradingsymbol": sell_ts,
        "buy_tradingsymbol": buy_ts,
        "net_credit": net_credit,
        "max_loss": max_loss,
        "option_type": option_type,
        "lot_size": sell_entry.get(option_type, {}).get("lotSize", lot_size) if sell_entry else lot_size,
    }


def select_hedge_strike(
    chain: list[dict], direction: str, spot: float,
    max_cost: float, lot_size: int,
) -> dict | None:
    """Select a protective hedge for overnight carry."""
    option_type = "PE" if direction == "bullish" else "CE"
    max_premium = max_cost / lot_size

    candidates = []
    for entry in chain:
        opt = entry.get(option_type)
        if not opt or not opt.get("ltp"):
            continue
        premium = opt["ltp"]
        if premium <= 0 or premium > max_premium:
            continue
        distance = abs(entry["strikePrice"] - spot)
        candidates.append({
            "strike": entry["strikePrice"],
            "option_type": option_type,
            "premium": premium,
            "delta": abs(opt.get("delta", 0)),
            "distance": distance,
        })

    if not candidates:
        return None

    target_distance = spot * 0.03
    candidates.sort(key=lambda c: abs(c["distance"] - target_distance))
    return candidates[0]
