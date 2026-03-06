"""Option Greeks computation and monitoring.

Computes Delta, Theta, Vega, Gamma for option positions using
Black-Scholes model. No external dependencies -- pure Python + math.
"""

import math
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))

# ---------------------------------------------------------------------------
# Normal distribution helpers (no scipy needed)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function (Abramowitz & Stegun)."""
    if x > 10:
        return 1.0
    if x < -10:
        return 0.0
    # Rational approximation (Hart, 1968)
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x_abs = abs(x)
    t = 1.0 / (1.0 + p * x_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x_abs * x_abs / 2.0)
    return 0.5 * (1.0 + sign * y)


def _norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Black-Scholes core
# ---------------------------------------------------------------------------

def _d1_d2(spot: float, strike: float, t: float, r: float, sigma: float) -> tuple[float, float]:
    """Compute d1 and d2 for Black-Scholes formula.

    Args:
        t: time to expiry in years
    """
    if t <= 0 or sigma <= 0:
        return (0.0, 0.0)
    sqrt_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return (d1, d2)


def black_scholes_greeks(
    spot: float,
    strike: float,
    dte: float,
    risk_free: float = 0.065,
    iv: float = 0.20,
    option_type: str = "CE",
) -> dict:
    """Compute all Greeks using Black-Scholes.

    Args:
        spot: underlying price
        strike: option strike price
        dte: days to expiry (calendar days)
        risk_free: annual risk-free rate (e.g., 0.065 for 6.5%)
        iv: implied volatility (e.g., 0.20 for 20%)
        option_type: "CE" for call, "PE" for put

    Returns:
        dict with keys: delta, gamma, theta, vega, iv, theoretical_price
    """
    is_call = option_type.upper() == "CE"

    # Edge case: expired or zero DTE
    if dte <= 0:
        if is_call:
            intrinsic = max(0.0, spot - strike)
            delta = 1.0 if spot > strike else 0.0
        else:
            intrinsic = max(0.0, strike - spot)
            delta = -1.0 if strike > spot else 0.0
        return {
            "delta": delta,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "iv": iv,
            "theoretical_price": intrinsic,
        }

    t = dte / 365.0
    d1, d2 = _d1_d2(spot, strike, t, risk_free, iv)
    sqrt_t = math.sqrt(t)
    pdf_d1 = _norm_pdf(d1)
    discount = math.exp(-risk_free * t)

    # Delta
    if is_call:
        delta = _norm_cdf(d1)
    else:
        delta = _norm_cdf(d1) - 1.0

    # Gamma (same for calls and puts)
    gamma = pdf_d1 / (spot * iv * sqrt_t) if (spot * iv * sqrt_t) > 0 else 0.0

    # Theta (per calendar day)
    theta_part1 = -(spot * pdf_d1 * iv) / (2.0 * sqrt_t)
    if is_call:
        theta_annual = theta_part1 - risk_free * strike * discount * _norm_cdf(d2)
    else:
        theta_annual = theta_part1 + risk_free * strike * discount * _norm_cdf(-d2)
    theta_daily = theta_annual / 365.0

    # Vega (per 1% move in IV, i.e., per 0.01 absolute IV change)
    vega = spot * sqrt_t * pdf_d1 * 0.01

    # Theoretical price
    if is_call:
        price = spot * _norm_cdf(d1) - strike * discount * _norm_cdf(d2)
    else:
        price = strike * discount * _norm_cdf(-d2) - spot * _norm_cdf(-d1)

    return {
        "delta": round(delta, 6),
        "gamma": round(gamma, 6),
        "theta": round(theta_daily, 4),
        "vega": round(vega, 4),
        "iv": iv,
        "theoretical_price": round(max(0.0, price), 4),
    }


# ---------------------------------------------------------------------------
# Implied Volatility (Newton-Raphson)
# ---------------------------------------------------------------------------

def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    dte: float,
    risk_free: float = 0.065,
    option_type: str = "CE",
    max_iter: int = 100,
    tol: float = 1e-6,
) -> float:
    """Compute IV from market price using Newton-Raphson iteration.

    Returns IV as a decimal (e.g., 0.20 for 20%).
    Returns float('nan') if convergence fails or price is impossible.
    """
    if dte <= 0 or market_price <= 0:
        return float("nan")

    is_call = option_type.upper() == "CE"
    t = dte / 365.0
    discount = math.exp(-risk_free * t)

    # Check intrinsic bounds
    if is_call:
        intrinsic = max(0.0, spot - strike * discount)
        upper_bound = spot  # call can never exceed spot
    else:
        intrinsic = max(0.0, strike * discount - spot)
        upper_bound = strike * discount  # put can never exceed PV(strike)

    if market_price < intrinsic - tol or market_price > upper_bound + tol:
        return float("nan")

    # Initial guess
    sigma = 0.30  # start at 30% vol

    for _ in range(max_iter):
        d1, d2 = _d1_d2(spot, strike, t, risk_free, sigma)
        sqrt_t = math.sqrt(t)

        if is_call:
            price = spot * _norm_cdf(d1) - strike * discount * _norm_cdf(d2)
        else:
            price = strike * discount * _norm_cdf(-d2) - spot * _norm_cdf(-d1)

        diff = price - market_price
        if abs(diff) < tol:
            return sigma

        # Vega for Newton step (un-scaled, per unit sigma)
        vega = spot * sqrt_t * _norm_pdf(d1)
        if vega < 1e-12:
            break

        sigma -= diff / vega

        # Keep sigma positive and bounded
        if sigma <= 0.001:
            sigma = 0.001
        if sigma > 10.0:
            break

    return float("nan")


# ---------------------------------------------------------------------------
# Position-level Greeks
# ---------------------------------------------------------------------------

def compute_position_greeks(
    positions: list[dict],
    spot_prices: dict[str, float],
    risk_free: float = 0.065,
) -> list[dict]:
    """Compute Greeks for all option positions.

    Each position needs: symbol, strike, option_type (CE/PE),
    entry_price (premium), quantity, expiry (DDMONYYYY or YYYY-MM-DD)

    spot_prices: {symbol: current_spot_price}

    Returns list of position dicts enriched with computed greeks.
    """
    results = []
    today = datetime.now(IST).date()

    for pos in positions:
        symbol = pos.get("symbol", "")
        instrument = pos.get("instrument", "EQ")

        # Skip non-option positions
        if instrument not in ("OPT", "MOMENTUM"):
            continue

        spot = spot_prices.get(symbol)
        if spot is None:
            continue

        strike = pos.get("strike", 0)
        option_type = pos.get("option_type", "CE")
        entry_price = pos.get("entry_price", 0)
        quantity = abs(pos.get("quantity", 0))
        expiry_str = pos.get("expiry", "")

        # Parse expiry
        expiry_date = _parse_expiry(expiry_str)
        if expiry_date is None:
            continue

        dte = (expiry_date - today).days
        if dte < 0:
            dte = 0

        # Use entry greeks IV if available, otherwise estimate from market
        iv_at_entry = pos.get("greeks_at_entry", {}).get("iv", 0)
        if iv_at_entry and iv_at_entry > 0:
            # IV from API is in percentage terms (e.g., 20.5 for 20.5%)
            iv = iv_at_entry / 100.0 if iv_at_entry > 1.0 else iv_at_entry
        else:
            # Fallback: try to compute IV from entry price
            iv = implied_volatility(entry_price, spot, strike, max(dte, 1), risk_free, option_type)
            if math.isnan(iv):
                iv = 0.25  # default 25% if all else fails

        greeks = black_scholes_greeks(spot, strike, dte, risk_free, iv, option_type)

        result = {
            **pos,
            "spot": spot,
            "dte": dte,
            "computed_greeks": greeks,
            "position_delta": greeks["delta"] * quantity,
            "position_gamma": greeks["gamma"] * quantity,
            "position_theta": greeks["theta"] * quantity,
            "position_vega": greeks["vega"] * quantity,
            "premium_at_risk": entry_price * quantity,
            "theoretical_value": greeks["theoretical_price"] * quantity,
        }
        results.append(result)

    return results


def _parse_expiry(expiry_str: str):
    """Parse expiry string in DDMONYYYY or YYYY-MM-DD format. Returns date or None."""
    if not expiry_str:
        return None
    try:
        return datetime.strptime(expiry_str, "%d%b%Y").date()
    except ValueError:
        pass
    try:
        return datetime.strptime(expiry_str, "%Y-%m-%d").date()
    except ValueError:
        pass
    return None


# ---------------------------------------------------------------------------
# Portfolio-level aggregation
# ---------------------------------------------------------------------------

def portfolio_greeks_summary(position_greeks: list[dict]) -> dict:
    """Aggregate Greeks across all option positions.

    Returns:
        net_delta: sum of position deltas (exposure to underlying movement)
        net_gamma: sum of position gammas
        net_theta_daily: sum of position thetas (daily rupee decay)
        net_vega: sum of position vegas
        total_premium_at_risk: sum of capital deployed in options
        theta_decay_daily_rupees: net theta in rupee terms
        positions_by_delta_exposure: sorted list (highest |delta| first)
    """
    net_delta = 0.0
    net_gamma = 0.0
    net_theta = 0.0
    net_vega = 0.0
    total_premium = 0.0

    for pg in position_greeks:
        net_delta += pg["position_delta"]
        net_gamma += pg["position_gamma"]
        net_theta += pg["position_theta"]
        net_vega += pg["position_vega"]
        total_premium += pg["premium_at_risk"]

    # Sort by absolute delta exposure (largest first)
    sorted_positions = sorted(
        position_greeks,
        key=lambda p: abs(p["position_delta"]),
        reverse=True,
    )

    return {
        "net_delta": round(net_delta, 4),
        "net_gamma": round(net_gamma, 6),
        "net_theta_daily": round(net_theta, 4),
        "net_vega": round(net_vega, 4),
        "total_premium_at_risk": round(total_premium, 2),
        "theta_decay_daily_rupees": round(net_theta, 2),
        "positions_by_delta_exposure": sorted_positions,
    }


# ---------------------------------------------------------------------------
# Telegram-formatted report
# ---------------------------------------------------------------------------

def generate_greeks_report(position_greeks: list[dict], summary: dict) -> str:
    """Telegram-formatted Greeks report (HTML).

    Shows per-position Greeks and portfolio-level exposure.
    """
    lines = []
    lines.append("<b>Greeks Dashboard</b>")
    lines.append("")

    # Portfolio summary
    lines.append("<b>Portfolio Greeks</b>")
    lines.append(f"  Net Delta:  {summary['net_delta']:+.2f}")
    lines.append(f"  Net Gamma:  {summary['net_gamma']:+.6f}")
    lines.append(f"  Net Theta:  {summary['net_theta_daily']:+.2f}/day")
    lines.append(f"  Net Vega:   {summary['net_vega']:+.2f}")
    lines.append(f"  Premium at Risk: Rs.{summary['total_premium_at_risk']:,.0f}")
    lines.append(f"  Theta Decay: Rs.{summary['theta_decay_daily_rupees']:+,.0f}/day")
    lines.append("")

    # Theta warning
    if summary["total_premium_at_risk"] > 0:
        theta_pct = abs(summary["theta_decay_daily_rupees"]) / summary["total_premium_at_risk"] * 100
        if theta_pct >= 2.0:
            lines.append(f"  <b>WARN: Daily theta = {theta_pct:.1f}% of premium</b>")
            lines.append("")

    # Per-position breakdown
    if position_greeks:
        lines.append("<b>Per-Position Greeks</b>")
        for pg in summary.get("positions_by_delta_exposure", position_greeks):
            symbol = pg.get("symbol", "?")
            strike = pg.get("strike", 0)
            opt_type = pg.get("option_type", "?")
            dte = pg.get("dte", 0)
            g = pg.get("computed_greeks", {})
            qty = abs(pg.get("quantity", 0))

            tag = f"{symbol} {opt_type}{int(strike)}"
            lines.append(f"  <b>{tag}</b> (DTE:{dte}, qty:{qty})")
            lines.append(f"    D:{g.get('delta', 0):+.4f}  G:{g.get('gamma', 0):.6f}  T:{g.get('theta', 0):+.4f}  V:{g.get('vega', 0):.4f}")
            lines.append(f"    IV:{g.get('iv', 0)*100:.1f}%  Theo:{g.get('theoretical_price', 0):.2f}")
            lines.append(f"    Pos Delta:{pg['position_delta']:+.2f}  Pos Theta:{pg['position_theta']:+.2f}/day")
            lines.append("")

    if not position_greeks:
        lines.append("  No option positions to analyze.")

    return "\n".join(lines)
