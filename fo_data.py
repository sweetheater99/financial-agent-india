"""F&O backtest data layer.

Provides synthetic option chains, futures pricing, lot sizes, and expiry calendar.
Uses yfinance for historical spot/VIX data + greeks.py for Black-Scholes pricing.
"""

import calendar
import datetime
import math
from pathlib import Path

import numpy as np
import pandas as pd

from greeks import black_scholes_greeks


# ---------------------------------------------------------------------------
# Lot sizes (date-aware)
# ---------------------------------------------------------------------------

_LOT_SIZES = {
    "NIFTY": [
        (datetime.date(2024, 12, 1), 50),
        (datetime.date(2025, 12, 1), 75),
        (datetime.date(2099, 1, 1), 65),
    ],
    "BANKNIFTY": [
        (datetime.date(2024, 12, 1), 25),
        (datetime.date(2025, 12, 1), 30),
        (datetime.date(2099, 1, 1), 28),
    ],
}


def get_lot_size(symbol: str, date: datetime.date) -> int:
    """Get lot size for symbol on given date."""
    tiers = _LOT_SIZES.get(symbol, _LOT_SIZES.get("NIFTY"))
    for cutoff, size in tiers:
        if date < cutoff:
            return size
    return tiers[-1][1]


# ---------------------------------------------------------------------------
# Expiry calendar
# ---------------------------------------------------------------------------

_EXPIRY_SWITCH_DATE = datetime.date(2025, 9, 1)


def _last_weekday_of_month(year: int, month: int, weekday: int) -> datetime.date:
    """Find last occurrence of weekday (0=Mon..6=Sun) in given month."""
    last_day = calendar.monthrange(year, month)[1]
    d = datetime.date(year, month, last_day)
    while d.weekday() != weekday:
        d -= datetime.timedelta(days=1)
    return d


def get_monthly_expiry(year: int, month: int) -> datetime.date:
    """Get monthly expiry date for given year/month."""
    ref = datetime.date(year, month, 1)
    if ref >= _EXPIRY_SWITCH_DATE:
        return _last_weekday_of_month(year, month, 1)  # Tuesday
    else:
        return _last_weekday_of_month(year, month, 3)  # Thursday


def get_weekly_expiries(year: int, month: int) -> list[datetime.date]:
    """Get all weekly expiry dates in a given month."""
    ref = datetime.date(year, month, 1)
    if ref >= _EXPIRY_SWITCH_DATE:
        target_weekday = 1  # Tuesday
    else:
        target_weekday = 3  # Thursday

    expiries = []
    d = datetime.date(year, month, 1)
    while d.weekday() != target_weekday:
        d += datetime.timedelta(days=1)
    while d.month == month:
        expiries.append(d)
        d += datetime.timedelta(days=7)
    return expiries


def get_nearest_expiry(
    from_date: datetime.date, min_dte: int = 0, weekly: bool = False
) -> datetime.date:
    """Find nearest expiry on or after from_date + min_dte."""
    target_date = from_date + datetime.timedelta(days=min_dte)

    for month_offset in range(4):
        year = from_date.year + (from_date.month + month_offset - 1) // 12
        month = (from_date.month + month_offset - 1) % 12 + 1
        if weekly:
            candidates = get_weekly_expiries(year, month)
        else:
            candidates = [get_monthly_expiry(year, month)]
        for exp in candidates:
            if exp >= target_date:
                return exp

    return from_date + datetime.timedelta(days=30)


# ---------------------------------------------------------------------------
# Spot + VIX history
# ---------------------------------------------------------------------------

_YFINANCE_TICKERS = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
}
_VIX_TICKER = "^INDIAVIX"

_STRIKE_INTERVAL = {"NIFTY": 50, "BANKNIFTY": 100}
_STRIKE_RANGE = {"NIFTY": 1500, "BANKNIFTY": 3000}

CACHE_DIR = Path("data/fo_backtest")


def fetch_spot_vix_history(
    symbol: str = "NIFTY", period: str = "2y"
) -> pd.DataFrame:
    """Fetch daily spot OHLCV + India VIX from yfinance.
    Returns DataFrame with columns: open, high, low, close, volume, vix
    """
    import yfinance as yf

    spot_ticker = _YFINANCE_TICKERS.get(symbol, "^NSEI")
    tickers = [spot_ticker, _VIX_TICKER]
    raw = yf.download(tickers, period=period, interval="1d", progress=False, threads=True)

    if raw.empty:
        return pd.DataFrame()

    spot_df = pd.DataFrame({
        "open": raw[("Open", spot_ticker)],
        "high": raw[("High", spot_ticker)],
        "low": raw[("Low", spot_ticker)],
        "close": raw[("Close", spot_ticker)],
        "volume": raw[("Volume", spot_ticker)],
    })

    vix_close = raw[("Close", _VIX_TICKER)]
    spot_df["vix"] = vix_close
    spot_df["vix"] = spot_df["vix"].ffill()
    spot_df = spot_df.dropna(subset=["close"])

    return spot_df


def generate_synthetic_chain(
    spot: float,
    vix: float,
    dte: int,
    symbol: str = "NIFTY",
    risk_free: float = 0.065,
    min_premium: float = 5.0,
    skew_factor: float = 8.0,
) -> pd.DataFrame:
    """Generate synthetic option chain using Black-Scholes with vol skew.

    Returns DataFrame with columns: strike, option_type, premium, delta, gamma, theta, vega, iv

    Skew: OTM puts get higher IV (put skew), OTM calls get mild smile (30% of put skew).
    Premium is mid-market theoretical. Strategies apply their own bid-ask adjustments.
    """
    interval = _STRIKE_INTERVAL.get(symbol, 50)
    strike_range = _STRIKE_RANGE.get(symbol, 1500)
    base_iv = vix / 100.0

    atm = round(spot / interval) * interval
    strikes = list(range(int(atm - strike_range), int(atm + strike_range) + 1, interval))

    rows = []
    for strike in strikes:
        for opt_type in ("CE", "PE"):
            # Per-strike IV with skew
            if opt_type == "PE" and strike < spot:
                moneyness = (spot - strike) / spot
                iv = base_iv * (1 + skew_factor * moneyness)
            elif opt_type == "CE" and strike > spot:
                moneyness = (strike - spot) / spot
                iv = base_iv * (1 + skew_factor * 0.3 * moneyness)
            else:
                iv = base_iv

            greeks = black_scholes_greeks(
                spot=spot, strike=float(strike), dte=dte,
                risk_free=risk_free, iv=iv, option_type=opt_type,
            )
            premium = greeks["theoretical_price"]

            if premium >= min_premium:
                rows.append({
                    "strike": float(strike),
                    "option_type": opt_type,
                    "premium": premium,
                    "delta": greeks["delta"],
                    "gamma": greeks["gamma"],
                    "theta": greeks["theta"],
                    "vega": greeks["vega"],
                    "iv": iv,
                })

    return pd.DataFrame(rows)


def get_futures_price(
    spot: float, dte: int, risk_free: float = 0.065
) -> float:
    """Synthetic futures price using cost-of-carry model."""
    return spot * math.exp(risk_free * dte / 365.0)


# ---------------------------------------------------------------------------
# Transaction costs (date-aware)
# ---------------------------------------------------------------------------

BROKERAGE_FLAT = 20.0
GST_PCT = 0.18
SEBI_PCT = 0.000001

_OPT_STT_TIERS = [
    (datetime.date(2025, 4, 1), 0.000625),
    (datetime.date(2026, 4, 1), 0.001),
    (datetime.date(2099, 1, 1), 0.0015),
]

_FUT_STT_TIERS = [
    (datetime.date(2025, 4, 1), 0.000125),
    (datetime.date(2026, 4, 1), 0.0002),
    (datetime.date(2099, 1, 1), 0.0005),
]

OPT_EXCHANGE_PCT = 0.000495
OPT_STAMP_DUTY_PCT = 0.00003
FUT_EXCHANGE_PCT = 0.000019
FUT_STAMP_DUTY_PCT = 0.00002
EXERCISE_STT_PCT = 0.00125


def _get_stt_rate(date: datetime.date, tiers: list) -> float:
    for cutoff, rate in tiers:
        if date < cutoff:
            return rate
    return tiers[-1][1]


def calc_options_costs(premium: float, quantity: int, side: str, date: datetime.date) -> float:
    turnover = premium * quantity
    brokerage = min(BROKERAGE_FLAT, turnover * 0.0003)
    stt = turnover * _get_stt_rate(date, _OPT_STT_TIERS) if side == "sell" else 0
    exchange = turnover * OPT_EXCHANGE_PCT
    stamp = turnover * OPT_STAMP_DUTY_PCT if side == "buy" else 0
    sebi = turnover * SEBI_PCT
    gst = (brokerage + exchange) * GST_PCT
    return brokerage + stt + exchange + stamp + sebi + gst


def calc_options_round_trip(entry_premium: float, exit_premium: float, quantity: int, date: datetime.date) -> float:
    return calc_options_costs(entry_premium, quantity, "buy", date) + calc_options_costs(exit_premium, quantity, "sell", date)


def calc_futures_costs(price: float, quantity: int, side: str, date: datetime.date) -> float:
    turnover = price * quantity
    brokerage = min(BROKERAGE_FLAT, turnover * 0.0003)
    stt = turnover * _get_stt_rate(date, _FUT_STT_TIERS) if side == "sell" else 0
    exchange = turnover * FUT_EXCHANGE_PCT
    stamp = turnover * FUT_STAMP_DUTY_PCT if side == "buy" else 0
    sebi = turnover * SEBI_PCT
    gst = (brokerage + exchange) * GST_PCT
    return brokerage + stt + exchange + stamp + sebi + gst


def calc_futures_round_trip(entry_price: float, exit_price: float, quantity: int, date: datetime.date) -> float:
    return calc_futures_costs(entry_price, quantity, "buy", date) + calc_futures_costs(exit_price, quantity, "sell", date)


def calc_exercise_stt(spot: float, quantity: int) -> float:
    return spot * quantity * EXERCISE_STT_PCT


def fetch_real_chain(
    symbol: str, chain_date: datetime.date, data_dir: str = None
) -> pd.DataFrame | None:
    """Load real option chain snapshot from parquet. Returns None if not available."""
    from fo_chain_collector import load_chain
    return load_chain(symbol, chain_date, data_dir=data_dir)
