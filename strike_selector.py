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

    ATM ~ 0.50, each 1% OTM reduces delta by ~0.08.
    """
    moneyness = (spot - strike) / spot  # positive = ITM for CE
    if option_type == "PE":
        moneyness = -moneyness  # positive = ITM for PE
    # Rough linear approximation around ATM
    delta = 0.50 + moneyness * 8.0
    return max(0.05, min(0.95, delta))


def select_directional_strike(
    chain: list[dict], direction: str, spot: float,
    risk_budget: float, lot_size: int, symbol: str,
) -> dict | None:
    """Select the best strike for a directional option buy."""
    option_type = "CE" if direction == "bullish" else "PE"
    delta_min, delta_max = STRIKE_FILTERS["directional_delta_range"]
    min_premium = STRIKE_FILTERS["min_premium"]
    max_premium = risk_budget / lot_size

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

    # Relaxed fallback if strict filters found nothing
    if not candidates:
        for entry in chain:
            opt = entry.get(option_type)
            if not opt or not opt.get("ltp"):
                continue
            delta = abs(opt.get("delta", 0))
            if delta == 0:
                delta = _estimate_delta(entry["strikePrice"], spot, option_type)
            premium = opt["ltp"]
            if 0.35 <= delta <= 0.55 and min_premium <= premium <= max_premium:
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

    if not candidates:
        return None

    target_delta = 0.45
    candidates.sort(key=lambda c: (abs(c["delta"] - target_delta), -c["oi"]))
    return candidates[0]


def select_spread_strikes(
    chain: list[dict], direction: str, spot: float,
    risk_budget: float, lot_size: int, symbol: str,
) -> dict | None:
    """Select strikes for a credit spread."""
    sell_delta = STRIKE_FILTERS["spread_sell_delta"]

    if direction == "bearish":
        option_type = "PE"
    else:
        option_type = "CE"

    sell_candidates = []
    for entry in chain:
        opt = entry.get(option_type)
        if not opt or not opt.get("ltp"):
            continue
        delta = abs(opt.get("delta", 0))
        if abs(delta - sell_delta) < 0.10:
            oi = opt.get("oi", 0)
            volume = opt.get("volume", 0)
            spread = _get_bid_ask_spread(opt)
            if passes_liquidity_filter(oi, volume, spread, symbol):
                sell_candidates.append({
                    "strike": entry["strikePrice"],
                    "premium": opt["ltp"],
                    "delta": delta,
                })

    if not sell_candidates:
        return None

    sell_candidates.sort(key=lambda c: abs(c["delta"] - sell_delta))
    sell = sell_candidates[0]

    strikes = sorted(set(e["strikePrice"] for e in chain))
    sell_idx = strikes.index(sell["strike"]) if sell["strike"] in strikes else -1
    if sell_idx < 0:
        return None

    if direction == "bearish":
        buy_idx = sell_idx - 2 if sell_idx >= 2 else 0
    else:
        buy_idx = sell_idx + 2 if sell_idx + 2 < len(strikes) else len(strikes) - 1

    buy_strike = strikes[buy_idx]
    buy_entry = next((e for e in chain if e["strikePrice"] == buy_strike), None)
    if not buy_entry:
        return None

    buy_opt = buy_entry.get(option_type)
    if not buy_opt or not buy_opt.get("ltp"):
        return None

    net_credit = sell["premium"] - buy_opt["ltp"]
    if net_credit < 15:
        return None

    strike_width = abs(sell["strike"] - buy_strike)
    max_loss = (strike_width - net_credit) * lot_size

    if max_loss > risk_budget:
        return None

    return {
        "sell_strike": sell["strike"],
        "buy_strike": buy_strike,
        "sell_premium": sell["premium"],
        "buy_premium": buy_opt["ltp"],
        "net_credit": net_credit,
        "max_loss": max_loss,
        "option_type": option_type,
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
