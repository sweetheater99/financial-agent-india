"""OI analysis for Trading System V3.

PCR computation, max pain, and OI buildup from option chain data.
Zero additional API calls — operates on chain data already fetched.
"""


def compute_pcr(chain: list[dict]) -> float:
    if not chain:
        return 1.0
    put_oi = sum(s.get("PE", {}).get("openInterest", 0) for s in chain)
    call_oi = sum(s.get("CE", {}).get("openInterest", 0) for s in chain)
    if call_oi == 0:
        return 1.0
    return round(put_oi / call_oi, 2)


def compute_max_pain(chain: list[dict], lot_size: int = 75) -> float:
    if not chain:
        return 0
    strikes = sorted(set(s["strikePrice"] for s in chain))
    min_pain = float("inf")
    max_pain_strike = strikes[len(strikes) // 2]
    for test_strike in strikes:
        total_pain = 0
        for s in chain:
            sp = s["strikePrice"]
            call_oi = s.get("CE", {}).get("openInterest", 0)
            put_oi = s.get("PE", {}).get("openInterest", 0)
            if test_strike > sp:
                total_pain += (test_strike - sp) * call_oi * lot_size
            if test_strike < sp:
                total_pain += (sp - test_strike) * put_oi * lot_size
        if total_pain < min_pain:
            min_pain = total_pain
            max_pain_strike = test_strike
    return max_pain_strike


def get_top_oi_strikes(chain: list[dict], top_n: int = 3) -> dict:
    if not chain:
        return {"call_resistance": [], "put_support": []}
    call_oi_list = [
        {"strike": s["strikePrice"], "oi": s.get("CE", {}).get("openInterest", 0)}
        for s in chain
    ]
    put_oi_list = [
        {"strike": s["strikePrice"], "oi": s.get("PE", {}).get("openInterest", 0)}
        for s in chain
    ]
    call_oi_list.sort(key=lambda x: x["oi"], reverse=True)
    put_oi_list.sort(key=lambda x: x["oi"], reverse=True)
    return {
        "call_resistance": call_oi_list[:top_n],
        "put_support": put_oi_list[:top_n],
    }
