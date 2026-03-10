# F&O Backtest Engine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a three-layer F&O backtesting engine (data, strategies, engine) that validates spreads, condors, momentum options, and futures strategies on 2 years of synthetic Black-Scholes pricing.

**Architecture:** Data layer generates synthetic option chains from yfinance spot+VIX history. Strategy layer has 4 classes (Futures, Spread, Condor, Momentum) with common entry/exit/P&L interface. Engine layer walks daily bars, manages portfolio constraints, and produces per-strategy metrics + param sweep.

**Tech Stack:** Python 3.11, pandas, numpy, yfinance, existing `greeks.py` (Black-Scholes), existing `backtest_signals.py` (V4 signal computation)

**Design doc:** `docs/plans/2026-03-10-fo-backtest-engine-design.md`

---

## Task 1: Data Layer — Expiry Calendar & Lot Sizes

**Files:**
- Create: `fo_data.py`
- Create: `tests/test_fo_data.py`

**Context:**
- NSE expiry shifted from Thursday to Tuesday in Sep 2025
- Lot sizes changed: Nifty 50→75→65, BankNifty 25→30→28
- NSE holidays are already in `config.py:NSE_HOLIDAYS_2026` and `config.is_trading_day()`
- For 2024/2025 holidays, we'll skip exact holiday matching and just use weekday filtering (close enough for backtest)

**Step 1: Write failing tests for lot sizes and expiry calendar**

```python
# tests/test_fo_data.py
import datetime
import pytest


class TestLotSizes:
    def test_nifty_lot_pre_dec_2024(self):
        from fo_data import get_lot_size
        assert get_lot_size("NIFTY", datetime.date(2024, 6, 15)) == 50

    def test_nifty_lot_dec_2024_to_nov_2025(self):
        from fo_data import get_lot_size
        assert get_lot_size("NIFTY", datetime.date(2025, 3, 15)) == 75

    def test_nifty_lot_dec_2025_onwards(self):
        from fo_data import get_lot_size
        assert get_lot_size("NIFTY", datetime.date(2026, 1, 15)) == 65

    def test_banknifty_lot_pre_dec_2024(self):
        from fo_data import get_lot_size
        assert get_lot_size("BANKNIFTY", datetime.date(2024, 6, 15)) == 25

    def test_banknifty_lot_dec_2024_to_nov_2025(self):
        from fo_data import get_lot_size
        assert get_lot_size("BANKNIFTY", datetime.date(2025, 6, 15)) == 30

    def test_banknifty_lot_dec_2025_onwards(self):
        from fo_data import get_lot_size
        assert get_lot_size("BANKNIFTY", datetime.date(2026, 2, 15)) == 28


class TestExpiryCalendar:
    def test_monthly_expiry_pre_sep_2025_is_thursday(self):
        from fo_data import get_monthly_expiry
        # March 2025 — last Thursday = 27th
        exp = get_monthly_expiry(2025, 3)
        assert exp == datetime.date(2025, 3, 27)
        assert exp.weekday() == 3  # Thursday

    def test_monthly_expiry_post_sep_2025_is_tuesday(self):
        from fo_data import get_monthly_expiry
        # October 2025 — last Tuesday = 28th
        exp = get_monthly_expiry(2025, 10)
        assert exp == datetime.date(2025, 10, 28)
        assert exp.weekday() == 1  # Tuesday

    def test_weekly_expiry_pre_sep_2025_is_thursday(self):
        from fo_data import get_weekly_expiries
        # March 2025 weeks — all should be Thursdays
        expiries = get_weekly_expiries(2025, 3)
        for exp in expiries:
            assert exp.weekday() == 3

    def test_weekly_expiry_post_sep_2025_is_tuesday(self):
        from fo_data import get_weekly_expiries
        expiries = get_weekly_expiries(2025, 10)
        for exp in expiries:
            assert exp.weekday() == 1

    def test_nearest_expiry_respects_min_dte(self):
        from fo_data import get_nearest_expiry
        # From 2025-10-01, min_dte=20 should skip Oct expiry and give Nov
        exp = get_nearest_expiry(datetime.date(2025, 10, 1), min_dte=20)
        assert exp.month >= 10
        assert (exp - datetime.date(2025, 10, 1)).days >= 20
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/aravindms/financial-agent-india && python -m pytest tests/test_fo_data.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'fo_data'"

**Step 3: Implement lot sizes and expiry calendar**

```python
# fo_data.py
"""F&O backtest data layer.

Provides synthetic option chains, futures pricing, lot sizes, and expiry calendar.
Uses yfinance for historical spot/VIX data + greeks.py for Black-Scholes pricing.
"""

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
        (datetime.date(2024, 12, 1), 50),   # pre-Dec 2024
        (datetime.date(2025, 12, 1), 75),   # Dec 2024 - Nov 2025
        (datetime.date(2099, 1, 1), 65),    # Dec 2025 onwards
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

# NSE switched from Thursday to Tuesday expiry in Sep 2025
_EXPIRY_SWITCH_DATE = datetime.date(2025, 9, 1)


def _last_weekday_of_month(year: int, month: int, weekday: int) -> datetime.date:
    """Find last occurrence of weekday (0=Mon..6=Sun) in given month."""
    import calendar
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

    import calendar
    last_day = calendar.monthrange(year, month)[1]
    expiries = []
    d = datetime.date(year, month, 1)
    # Find first target weekday
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

    # Search up to 3 months ahead
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

    # Fallback: 30 days out
    return from_date + datetime.timedelta(days=30)
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/aravindms/financial-agent-india && python -m pytest tests/test_fo_data.py -v`
Expected: All 11 tests PASS

**Step 5: Commit**

```bash
git add fo_data.py tests/test_fo_data.py
git commit -m "feat(fo-backtest): add data layer with lot sizes and expiry calendar"
```

---

## Task 2: Data Layer — Spot/VIX History & Synthetic Option Chains

**Files:**
- Modify: `fo_data.py`
- Modify: `tests/test_fo_data.py`

**Context:**
- yfinance tickers: `^NSEI` (Nifty 50), `^NSEBANK` (Bank Nifty), `^INDIAVIX` (India VIX)
- `greeks.black_scholes_greeks(spot, strike, dte, risk_free, iv, option_type)` returns dict with `delta`, `gamma`, `theta`, `vega`, `iv`, `theoretical_price`
- Strike intervals: 50pt for Nifty, 100pt for BankNifty
- Generate strikes from spot - 1500 to spot + 1500 (Nifty) or spot - 3000 to spot + 3000 (BankNifty)
- VIX is used as the IV proxy for ATM options; no skew in V1

**Step 1: Write failing tests for spot data fetching and chain generation**

```python
# tests/test_fo_data.py — add to existing file

class TestSpotHistory:
    def test_fetch_spot_vix_returns_dataframe(self):
        from fo_data import fetch_spot_vix_history
        # Use a short period to keep test fast
        data = fetch_spot_vix_history("NIFTY", period="1mo")
        assert isinstance(data, pd.DataFrame)
        assert "close" in data.columns
        assert "vix" in data.columns
        assert len(data) > 10

    def test_fetch_banknifty(self):
        from fo_data import fetch_spot_vix_history
        data = fetch_spot_vix_history("BANKNIFTY", period="1mo")
        assert len(data) > 10
        assert data["close"].iloc[-1] > 30000  # BankNifty is above 30k


class TestSyntheticChain:
    def test_chain_has_ce_and_pe(self):
        from fo_data import generate_synthetic_chain
        chain = generate_synthetic_chain(
            spot=23000.0, vix=14.0, dte=30, symbol="NIFTY"
        )
        assert isinstance(chain, pd.DataFrame)
        assert "CE" in chain["option_type"].values
        assert "PE" in chain["option_type"].values

    def test_chain_strike_interval_nifty(self):
        from fo_data import generate_synthetic_chain
        chain = generate_synthetic_chain(
            spot=23000.0, vix=14.0, dte=30, symbol="NIFTY"
        )
        ce_strikes = sorted(chain[chain["option_type"] == "CE"]["strike"].values)
        # Check 50pt intervals
        diffs = [ce_strikes[i+1] - ce_strikes[i] for i in range(len(ce_strikes)-1)]
        assert all(d == 50 for d in diffs)

    def test_chain_strike_interval_banknifty(self):
        from fo_data import generate_synthetic_chain
        chain = generate_synthetic_chain(
            spot=50000.0, vix=16.0, dte=30, symbol="BANKNIFTY"
        )
        ce_strikes = sorted(chain[chain["option_type"] == "CE"]["strike"].values)
        diffs = [ce_strikes[i+1] - ce_strikes[i] for i in range(len(ce_strikes)-1)]
        assert all(d == 100 for d in diffs)

    def test_chain_atm_delta_approximately_0_5(self):
        from fo_data import generate_synthetic_chain
        chain = generate_synthetic_chain(
            spot=23000.0, vix=14.0, dte=30, symbol="NIFTY"
        )
        # ATM CE delta should be ~0.5
        atm_ce = chain[
            (chain["option_type"] == "CE") &
            (chain["strike"] == 23000.0)
        ]
        assert len(atm_ce) == 1
        assert 0.4 < atm_ce.iloc[0]["delta"] < 0.65

    def test_chain_premium_positive(self):
        from fo_data import generate_synthetic_chain
        chain = generate_synthetic_chain(
            spot=23000.0, vix=14.0, dte=30, symbol="NIFTY"
        )
        assert (chain["premium"] > 0).all()

    def test_chain_filters_low_premium(self):
        from fo_data import generate_synthetic_chain
        chain = generate_synthetic_chain(
            spot=23000.0, vix=14.0, dte=30, symbol="NIFTY"
        )
        # All premiums should be >= 5 (liquidity filter)
        assert (chain["premium"] >= 5.0).all()
```

**Step 2: Run tests to verify new tests fail**

Run: `cd /Users/aravindms/financial-agent-india && python -m pytest tests/test_fo_data.py::TestSpotHistory -v`
Expected: FAIL (functions not defined)

**Step 3: Implement spot/VIX fetching and synthetic chain generation**

Add to `fo_data.py`:

```python
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
    Index is DatetimeIndex.
    """
    import yfinance as yf

    spot_ticker = _YFINANCE_TICKERS.get(symbol, "^NSEI")

    # Download spot and VIX together
    tickers = [spot_ticker, _VIX_TICKER]
    raw = yf.download(tickers, period=period, interval="1d", progress=False, threads=True)

    if raw.empty:
        return pd.DataFrame()

    # Extract spot OHLCV
    spot_df = pd.DataFrame({
        "open": raw[("Open", spot_ticker)],
        "high": raw[("High", spot_ticker)],
        "low": raw[("Low", spot_ticker)],
        "close": raw[("Close", spot_ticker)],
        "volume": raw[("Volume", spot_ticker)],
    })

    # Extract VIX close
    vix_close = raw[("Close", _VIX_TICKER)]
    spot_df["vix"] = vix_close

    # Forward-fill VIX gaps, drop rows with no spot data
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
) -> pd.DataFrame:
    """Generate synthetic option chain using Black-Scholes.

    Returns DataFrame with columns:
        strike, option_type, premium, delta, gamma, theta, vega, iv
    """
    interval = _STRIKE_INTERVAL.get(symbol, 50)
    strike_range = _STRIKE_RANGE.get(symbol, 1500)
    iv = vix / 100.0  # VIX is in percentage, BS needs decimal

    # Round spot to nearest interval for ATM
    atm = round(spot / interval) * interval

    strikes = list(range(int(atm - strike_range), int(atm + strike_range) + 1, interval))

    rows = []
    for strike in strikes:
        for opt_type in ("CE", "PE"):
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
```

**Step 4: Run all tests**

Run: `cd /Users/aravindms/financial-agent-india && python -m pytest tests/test_fo_data.py -v`
Expected: All tests PASS (note: TestSpotHistory tests need network — may skip in CI)

**Step 5: Commit**

```bash
git add fo_data.py tests/test_fo_data.py
git commit -m "feat(fo-backtest): add spot/VIX history fetching and synthetic chain generation"
```

---

## Task 3: Data Layer — Transaction Costs (date-aware)

**Files:**
- Modify: `fo_data.py`
- Modify: `tests/test_fo_data.py`

**Context:**
- STT rates changed over time: Budget 2025 (Apr 2025) and Budget 2026 (Apr 2026) both hiked F&O STT
- Brokerage is flat ₹20/order (AngelOne)
- Need separate cost functions for options (premium-based) and futures (notional-based)
- GST is 18% on (brokerage + exchange charges)

**Step 1: Write failing tests for date-aware costs**

```python
# tests/test_fo_data.py — add to existing file

class TestTransactionCosts:
    def test_options_cost_pre_apr_2025(self):
        from fo_data import calc_options_costs
        # STT sell = 0.0625%, lower rate
        cost = calc_options_costs(
            premium=100.0, quantity=75, side="sell",
            date=datetime.date(2025, 3, 15)
        )
        assert cost > 0
        # STT = 100 * 75 * 0.000625 = 4.6875
        assert cost < 50  # reasonable total

    def test_options_cost_post_apr_2026(self):
        from fo_data import calc_options_costs
        # STT sell = 0.15%, higher rate
        cost = calc_options_costs(
            premium=100.0, quantity=75, side="sell",
            date=datetime.date(2026, 5, 15)
        )
        # Higher STT should make this more expensive
        cost_old = calc_options_costs(
            premium=100.0, quantity=75, side="sell",
            date=datetime.date(2025, 3, 15)
        )
        assert cost > cost_old

    def test_futures_cost_round_trip(self):
        from fo_data import calc_futures_round_trip
        cost = calc_futures_round_trip(
            entry_price=23000.0, exit_price=23200.0,
            quantity=75, date=datetime.date(2026, 1, 15)
        )
        assert cost > 0
        assert cost < 500  # reasonable for 1 lot

    def test_options_round_trip(self):
        from fo_data import calc_options_round_trip
        cost = calc_options_round_trip(
            entry_premium=100.0, exit_premium=150.0,
            quantity=75, date=datetime.date(2026, 1, 15)
        )
        assert cost > 0

    def test_exercise_stt_much_higher(self):
        from fo_data import calc_exercise_stt
        # STT on exercise = 0.125% of full notional
        stt = calc_exercise_stt(spot=23000.0, quantity=75)
        # 23000 * 75 * 0.00125 = 2156.25
        assert stt > 2000
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/aravindms/financial-agent-india && python -m pytest tests/test_fo_data.py::TestTransactionCosts -v`
Expected: FAIL

**Step 3: Implement date-aware transaction costs**

Add to `fo_data.py`:

```python
# ---------------------------------------------------------------------------
# Transaction costs (date-aware)
# ---------------------------------------------------------------------------

BROKERAGE_FLAT = 20.0
GST_PCT = 0.18
SEBI_PCT = 0.000001

# Options STT (sell-side only) — changed by budget
_OPT_STT_TIERS = [
    (datetime.date(2025, 4, 1), 0.000625),   # pre-Apr 2025: 0.0625%
    (datetime.date(2026, 4, 1), 0.001),       # Apr 2025 - Mar 2026: 0.1%
    (datetime.date(2099, 1, 1), 0.0015),      # Apr 2026+: 0.15%
]

# Futures STT (sell-side only)
_FUT_STT_TIERS = [
    (datetime.date(2025, 4, 1), 0.000125),    # pre-Apr 2025: 0.0125%
    (datetime.date(2026, 4, 1), 0.0002),      # Apr 2025 - Mar 2026: 0.02%
    (datetime.date(2099, 1, 1), 0.0005),       # Apr 2026+: 0.05%
]

OPT_EXCHANGE_PCT = 0.000495
OPT_STAMP_DUTY_PCT = 0.00003
FUT_EXCHANGE_PCT = 0.000019
FUT_STAMP_DUTY_PCT = 0.00002
EXERCISE_STT_PCT = 0.00125  # 0.125% on full notional


def _get_stt_rate(date: datetime.date, tiers: list) -> float:
    for cutoff, rate in tiers:
        if date < cutoff:
            return rate
    return tiers[-1][1]


def calc_options_costs(
    premium: float, quantity: int, side: str, date: datetime.date
) -> float:
    """Calculate options transaction costs for one leg."""
    turnover = premium * quantity
    brokerage = min(BROKERAGE_FLAT, turnover * 0.0003)
    stt = turnover * _get_stt_rate(date, _OPT_STT_TIERS) if side == "sell" else 0
    exchange = turnover * OPT_EXCHANGE_PCT
    stamp = turnover * OPT_STAMP_DUTY_PCT if side == "buy" else 0
    sebi = turnover * SEBI_PCT
    gst = (brokerage + exchange) * GST_PCT
    return brokerage + stt + exchange + stamp + sebi + gst


def calc_options_round_trip(
    entry_premium: float, exit_premium: float,
    quantity: int, date: datetime.date
) -> float:
    """Total round-trip options costs."""
    return (
        calc_options_costs(entry_premium, quantity, "buy", date) +
        calc_options_costs(exit_premium, quantity, "sell", date)
    )


def calc_futures_costs(
    price: float, quantity: int, side: str, date: datetime.date
) -> float:
    """Calculate futures transaction costs for one leg."""
    turnover = price * quantity
    brokerage = min(BROKERAGE_FLAT, turnover * 0.0003)
    stt = turnover * _get_stt_rate(date, _FUT_STT_TIERS) if side == "sell" else 0
    exchange = turnover * FUT_EXCHANGE_PCT
    stamp = turnover * FUT_STAMP_DUTY_PCT if side == "buy" else 0
    sebi = turnover * SEBI_PCT
    gst = (brokerage + exchange) * GST_PCT
    return brokerage + stt + exchange + stamp + sebi + gst


def calc_futures_round_trip(
    entry_price: float, exit_price: float,
    quantity: int, date: datetime.date
) -> float:
    """Total round-trip futures costs."""
    return (
        calc_futures_costs(entry_price, quantity, "buy", date) +
        calc_futures_costs(exit_price, quantity, "sell", date)
    )


def calc_exercise_stt(spot: float, quantity: int) -> float:
    """STT charged on ITM options that expire (0.125% of notional)."""
    return spot * quantity * EXERCISE_STT_PCT
```

**Step 4: Run tests**

Run: `cd /Users/aravindms/financial-agent-india && python -m pytest tests/test_fo_data.py::TestTransactionCosts -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add fo_data.py tests/test_fo_data.py
git commit -m "feat(fo-backtest): add date-aware F&O transaction costs"
```

---

## Task 4: Strategy Layer — Base Class & Futures Strategy

**Files:**
- Create: `fo_strategies.py`
- Create: `tests/test_fo_strategies.py`

**Context:**
- Futures params from config.py: FUT_TARGET_ATR_MULT, FUT_SL_ATR_MULT (not defined in config — use V8 backtest values: target=1.5x, SL=3.5x)
- Sizing: max 2% loss per trade, 15% margin requirement
- The strategy receives signals from `backtest_signals.compute_signals()` — the `score` and `direction` fields
- Futures P&L is simple: `(exit - entry) * quantity` adjusted for direction

**Step 1: Write failing tests**

```python
# tests/test_fo_strategies.py
import datetime
import pytest
import pandas as pd
import numpy as np


def _make_chain(spot=23000.0, vix=14.0, dte=30):
    """Helper to make a small synthetic chain for testing."""
    from fo_data import generate_synthetic_chain
    return generate_synthetic_chain(spot=spot, vix=vix, dte=dte, symbol="NIFTY")


class TestFuturesStrategy:
    def test_should_enter_bullish_high_score(self):
        from fo_strategies import FuturesStrategy
        strat = FuturesStrategy()
        result = strat.should_enter(
            date=datetime.date(2025, 6, 15),
            spot=23000.0,
            vix=14.0,
            atr=200.0,
            score=5.0,
            direction="bullish",
            available_capital=1000000.0,
            current_positions=[],
        )
        assert result is not None
        assert result["direction"] == "bullish"
        assert result["instrument"] == "FUT"
        assert result["entry_price"] > 0
        assert result["quantity"] > 0
        assert result["target_price"] > result["entry_price"]
        assert result["stoploss_price"] < result["entry_price"]

    def test_should_not_enter_low_score(self):
        from fo_strategies import FuturesStrategy
        strat = FuturesStrategy()
        result = strat.should_enter(
            date=datetime.date(2025, 6, 15),
            spot=23000.0,
            vix=14.0,
            atr=200.0,
            score=2.0,
            direction="bullish",
            available_capital=1000000.0,
            current_positions=[],
        )
        assert result is None

    def test_should_enter_bearish(self):
        from fo_strategies import FuturesStrategy
        strat = FuturesStrategy()
        result = strat.should_enter(
            date=datetime.date(2025, 6, 15),
            spot=23000.0,
            vix=14.0,
            atr=200.0,
            score=5.0,
            direction="bearish",
            available_capital=1000000.0,
            current_positions=[],
        )
        assert result is not None
        assert result["direction"] == "bearish"
        assert result["target_price"] < result["entry_price"]
        assert result["stoploss_price"] > result["entry_price"]

    def test_should_exit_target_hit(self):
        from fo_strategies import FuturesStrategy
        strat = FuturesStrategy()
        position = {
            "instrument": "FUT",
            "direction": "bullish",
            "entry_price": 23000.0,
            "target_price": 23300.0,
            "stoploss_price": 22300.0,
            "peak_price": 23200.0,
            "entry_date": datetime.date(2025, 6, 10),
            "max_hold_date": datetime.date(2025, 6, 25),
        }
        should_exit, reason = strat.should_exit(
            position=position,
            date=datetime.date(2025, 6, 12),
            spot=23350.0,
            high=23400.0,
            low=23200.0,
        )
        assert should_exit is True
        assert reason == "target"

    def test_should_exit_stoploss_hit(self):
        from fo_strategies import FuturesStrategy
        strat = FuturesStrategy()
        position = {
            "instrument": "FUT",
            "direction": "bullish",
            "entry_price": 23000.0,
            "target_price": 23300.0,
            "stoploss_price": 22300.0,
            "peak_price": 23000.0,
            "entry_date": datetime.date(2025, 6, 10),
            "max_hold_date": datetime.date(2025, 6, 25),
        }
        should_exit, reason = strat.should_exit(
            position=position,
            date=datetime.date(2025, 6, 12),
            spot=22200.0,
            high=22500.0,
            low=22100.0,
        )
        assert should_exit is True
        assert reason == "stoploss"

    def test_risk_based_sizing(self):
        from fo_strategies import FuturesStrategy
        strat = FuturesStrategy()
        result = strat.should_enter(
            date=datetime.date(2025, 6, 15),
            spot=23000.0,
            vix=14.0,
            atr=200.0,
            score=5.0,
            direction="bullish",
            available_capital=1000000.0,
            current_positions=[],
        )
        # Max loss should be <= 2% of capital = 20000
        max_loss = abs(result["entry_price"] - result["stoploss_price"]) * result["quantity"]
        assert max_loss <= 20000 * 1.1  # 10% tolerance for rounding
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/aravindms/financial-agent-india && python -m pytest tests/test_fo_strategies.py::TestFuturesStrategy -v`
Expected: FAIL

**Step 3: Implement base class and FuturesStrategy**

```python
# fo_strategies.py
"""F&O backtest strategies.

Four strategy classes with common interface:
- FuturesStrategy: directional futures trades
- SpreadStrategy: vertical spreads (bull call / bear put)
- CondorStrategy: iron condors on index
- MomentumStrategy: ATM options for quick directional bets

Each uses config.py params and V4 signal scores for entry decisions.
"""

import datetime
import math
from dataclasses import dataclass

import pandas as pd

from fo_data import get_lot_size, get_futures_price, get_nearest_expiry


# ---------------------------------------------------------------------------
# Strategy params (matching config.py values)
# ---------------------------------------------------------------------------

FUT_TARGET_ATR_MULT = 1.5
FUT_SL_ATR_MULT = 3.5
FUT_MARGIN_PCT = 0.15
FUT_MAX_RISK_PCT = 0.02     # 2% of capital per trade
FUT_MAX_HOLD_DAYS = 15
FUT_SLIPPAGE_PCT = 0.0005

ENTRY_SCORE_THRESHOLD = 3.5
HIGH_CONVICTION_THRESHOLD = 5.0


# ---------------------------------------------------------------------------
# Base Strategy
# ---------------------------------------------------------------------------

class BaseStrategy:
    """Common interface for all F&O strategies."""

    name: str = "base"

    def should_enter(self, **kwargs) -> dict | None:
        """Check if entry conditions are met. Returns position dict or None."""
        raise NotImplementedError

    def should_exit(self, **kwargs) -> tuple[bool, str]:
        """Check if exit conditions are met. Returns (should_exit, reason)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Futures Strategy
# ---------------------------------------------------------------------------

class FuturesStrategy(BaseStrategy):
    name = "futures"

    def __init__(
        self,
        score_threshold: float = ENTRY_SCORE_THRESHOLD,
        target_atr_mult: float = FUT_TARGET_ATR_MULT,
        sl_atr_mult: float = FUT_SL_ATR_MULT,
        max_risk_pct: float = FUT_MAX_RISK_PCT,
        max_hold_days: int = FUT_MAX_HOLD_DAYS,
    ):
        self.score_threshold = score_threshold
        self.target_atr_mult = target_atr_mult
        self.sl_atr_mult = sl_atr_mult
        self.max_risk_pct = max_risk_pct
        self.max_hold_days = max_hold_days

    def should_enter(
        self,
        date: datetime.date,
        spot: float,
        vix: float,
        atr: float,
        score: float,
        direction: str | None,
        available_capital: float,
        current_positions: list[dict],
        symbol: str = "NIFTY",
    ) -> dict | None:
        """Enter a futures position if score exceeds threshold."""
        if score < self.score_threshold or direction is None:
            return None
        if atr <= 0:
            return None

        # VIX filter — no futures in crisis
        if vix > 28:
            return None

        lot_size = get_lot_size(symbol, date)
        expiry = get_nearest_expiry(date, min_dte=5)
        dte = (expiry - date).days

        # Futures price with cost-of-carry
        fut_price = get_futures_price(spot, dte)
        entry_price = fut_price * (1 + FUT_SLIPPAGE_PCT if direction == "bullish" else 1 - FUT_SLIPPAGE_PCT)

        # Target and stoploss
        if direction == "bullish":
            target = entry_price + self.target_atr_mult * atr
            stoploss = entry_price - self.sl_atr_mult * atr
        else:
            target = entry_price - self.target_atr_mult * atr
            stoploss = entry_price + self.sl_atr_mult * atr

        # Risk-based sizing: max loss <= max_risk_pct of capital
        risk_per_lot = abs(entry_price - stoploss) * lot_size
        max_loss = available_capital * self.max_risk_pct
        num_lots = max(1, int(max_loss / risk_per_lot))

        # Margin check: num_lots * lot_size * fut_price * margin_pct <= available_capital
        margin_needed = num_lots * lot_size * fut_price * FUT_MARGIN_PCT
        while margin_needed > available_capital * 0.5 and num_lots > 1:
            num_lots -= 1
            margin_needed = num_lots * lot_size * fut_price * FUT_MARGIN_PCT

        max_hold_date = date + datetime.timedelta(days=self.max_hold_days)
        # Don't hold past expiry
        if max_hold_date > expiry:
            max_hold_date = expiry - datetime.timedelta(days=1)

        return {
            "instrument": "FUT",
            "symbol": symbol,
            "direction": direction,
            "entry_price": round(entry_price, 2),
            "quantity": num_lots * lot_size,
            "num_lots": num_lots,
            "lot_size": lot_size,
            "target_price": round(target, 2),
            "stoploss_price": round(stoploss, 2),
            "peak_price": entry_price,
            "entry_date": date,
            "max_hold_date": max_hold_date,
            "expiry": expiry,
            "margin_used": round(margin_needed, 2),
            "score": score,
        }

    def should_exit(
        self,
        position: dict,
        date: datetime.date,
        spot: float,
        high: float,
        low: float,
    ) -> tuple[bool, str]:
        """Check exit conditions for futures position."""
        direction = position["direction"]
        entry = position["entry_price"]
        target = position["target_price"]
        stoploss = position["stoploss_price"]

        if direction == "bullish":
            if high >= target:
                return True, "target"
            if low <= stoploss:
                return True, "stoploss"
        else:
            if low <= target:
                return True, "target"
            if high >= stoploss:
                return True, "stoploss"

        # Max hold
        if date >= position["max_hold_date"]:
            return True, "max_hold"

        # Expiry
        if date >= position.get("expiry", date + datetime.timedelta(days=30)):
            return True, "expiry"

        return False, ""
```

**Step 4: Run tests**

Run: `cd /Users/aravindms/financial-agent-india && python -m pytest tests/test_fo_strategies.py::TestFuturesStrategy -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add fo_strategies.py tests/test_fo_strategies.py
git commit -m "feat(fo-backtest): add base strategy class and FuturesStrategy"
```

---

## Task 5: Strategy Layer — Spread Strategy

**Files:**
- Modify: `fo_strategies.py`
- Modify: `tests/test_fo_strategies.py`

**Context:**
- Config params: SPREAD_MIN_DTE=30, SPREAD_MAX_DTE=45, SPREAD_MAX_RISK_PCT=0.02, SPREAD_PROFIT_CAP_PCT=0.80, SPREAD_TIME_EXIT_DAYS=5
- Bull call spread: buy ATM CE, sell OTM CE (bullish direction)
- Bear put spread: buy ATM PE, sell OTM PE (bearish direction)
- Max risk = net debit (what you paid)
- Max profit = strike width - net debit
- Daily P&L: recompute both legs with updated spot/dte using Black-Scholes

**Step 1: Write failing tests**

```python
# tests/test_fo_strategies.py — add to existing file

class TestSpreadStrategy:
    def test_bull_call_spread_entry(self):
        from fo_strategies import SpreadStrategy
        chain = _make_chain(spot=23000.0, vix=14.0, dte=35)
        strat = SpreadStrategy()
        result = strat.should_enter(
            date=datetime.date(2025, 6, 15),
            spot=23000.0,
            vix=14.0,
            atr=200.0,
            score=5.0,
            direction="bullish",
            available_capital=1000000.0,
            current_positions=[],
            chain=chain,
            dte=35,
        )
        assert result is not None
        assert result["instrument"] == "SPREAD"
        assert result["direction"] == "bullish"
        assert result["long_strike"] < result["short_strike"]  # bull call
        assert result["net_debit"] > 0
        assert result["max_profit"] > 0

    def test_bear_put_spread_entry(self):
        from fo_strategies import SpreadStrategy
        chain = _make_chain(spot=23000.0, vix=14.0, dte=35)
        strat = SpreadStrategy()
        result = strat.should_enter(
            date=datetime.date(2025, 6, 15),
            spot=23000.0,
            vix=14.0,
            atr=200.0,
            score=5.0,
            direction="bearish",
            available_capital=1000000.0,
            current_positions=[],
            chain=chain,
            dte=35,
        )
        assert result is not None
        assert result["long_strike"] > result["short_strike"]  # bear put

    def test_spread_not_entered_low_dte(self):
        from fo_strategies import SpreadStrategy
        chain = _make_chain(spot=23000.0, vix=14.0, dte=10)
        strat = SpreadStrategy()
        result = strat.should_enter(
            date=datetime.date(2025, 6, 15),
            spot=23000.0,
            vix=14.0,
            atr=200.0,
            score=5.0,
            direction="bullish",
            available_capital=1000000.0,
            current_positions=[],
            chain=chain,
            dte=10,
        )
        assert result is None  # DTE too low

    def test_spread_exit_profit_cap(self):
        from fo_strategies import SpreadStrategy
        strat = SpreadStrategy()
        position = {
            "instrument": "SPREAD",
            "direction": "bullish",
            "net_debit": 50.0,
            "max_profit": 150.0,
            "entry_date": datetime.date(2025, 6, 10),
            "expiry": datetime.date(2025, 7, 24),
        }
        # Current P&L = 80% of max profit
        should_exit, reason = strat.should_exit(
            position=position,
            date=datetime.date(2025, 6, 20),
            current_spread_value=170.0,  # net_debit was 50, now worth 170 = profit of 120 = 80% of 150
        )
        assert should_exit is True
        assert reason == "profit_cap"

    def test_spread_max_risk_sizing(self):
        from fo_strategies import SpreadStrategy
        chain = _make_chain(spot=23000.0, vix=14.0, dte=35)
        strat = SpreadStrategy()
        result = strat.should_enter(
            date=datetime.date(2025, 6, 15),
            spot=23000.0,
            vix=14.0,
            atr=200.0,
            score=5.0,
            direction="bullish",
            available_capital=1000000.0,
            current_positions=[],
            chain=chain,
            dte=35,
        )
        # Max risk = net_debit * quantity, should be <= 2% of capital
        max_risk = result["net_debit"] * result["quantity"]
        assert max_risk <= 1000000.0 * 0.02 * 1.1
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/aravindms/financial-agent-india && python -m pytest tests/test_fo_strategies.py::TestSpreadStrategy -v`
Expected: FAIL

**Step 3: Implement SpreadStrategy**

Add to `fo_strategies.py`:

```python
# ---------------------------------------------------------------------------
# Spread Strategy
# ---------------------------------------------------------------------------

SPREAD_MIN_DTE = 30
SPREAD_MAX_DTE = 45
SPREAD_MAX_RISK_PCT = 0.02
SPREAD_PROFIT_CAP_PCT = 0.80
SPREAD_TIME_EXIT_DAYS = 5
SPREAD_SL_MULTIPLIER = 2.0  # close if loss > 2x net debit


class SpreadStrategy(BaseStrategy):
    name = "spread"

    def __init__(
        self,
        score_threshold: float = ENTRY_SCORE_THRESHOLD,
        min_dte: int = SPREAD_MIN_DTE,
        max_dte: int = SPREAD_MAX_DTE,
        max_risk_pct: float = SPREAD_MAX_RISK_PCT,
        profit_cap_pct: float = SPREAD_PROFIT_CAP_PCT,
        sl_multiplier: float = SPREAD_SL_MULTIPLIER,
        time_exit_days: int = SPREAD_TIME_EXIT_DAYS,
    ):
        self.score_threshold = score_threshold
        self.min_dte = min_dte
        self.max_dte = max_dte
        self.max_risk_pct = max_risk_pct
        self.profit_cap_pct = profit_cap_pct
        self.sl_multiplier = sl_multiplier
        self.time_exit_days = time_exit_days

    def should_enter(
        self,
        date: datetime.date,
        spot: float,
        vix: float,
        atr: float,
        score: float,
        direction: str | None,
        available_capital: float,
        current_positions: list[dict],
        chain: pd.DataFrame = None,
        dte: int = 0,
        symbol: str = "NIFTY",
    ) -> dict | None:
        if score < self.score_threshold or direction is None:
            return None
        if dte < self.min_dte or dte > self.max_dte:
            return None
        if chain is None or chain.empty:
            return None

        lot_size = get_lot_size(symbol, date)

        # Select strikes based on direction
        atm_strike = round(spot / 50) * 50  # Nifty 50pt intervals
        spread_width = max(100, round(atr))  # at least 100pt wide
        spread_width = round(spread_width / 50) * 50  # round to interval

        if direction == "bullish":
            # Bull call spread: buy ATM CE, sell OTM CE
            long_strike = atm_strike
            short_strike = atm_strike + spread_width
            opt_type = "CE"
        else:
            # Bear put spread: buy ATM PE, sell OTM PE
            long_strike = atm_strike
            short_strike = atm_strike - spread_width
            opt_type = "PE"

        # Find premiums from chain
        long_row = chain[
            (chain["strike"] == long_strike) & (chain["option_type"] == opt_type)
        ]
        short_row = chain[
            (chain["strike"] == short_strike) & (chain["option_type"] == opt_type)
        ]

        if long_row.empty or short_row.empty:
            return None

        long_premium = float(long_row.iloc[0]["premium"])
        short_premium = float(short_row.iloc[0]["premium"])
        net_debit = long_premium - short_premium

        if net_debit <= 0:
            return None  # shouldn't happen for debit spreads

        max_profit = abs(long_strike - short_strike) - net_debit

        if max_profit <= 0:
            return None

        # Risk-based sizing
        max_risk = available_capital * self.max_risk_pct
        num_lots = max(1, int(max_risk / (net_debit * lot_size)))

        expiry = get_nearest_expiry(date, min_dte=self.min_dte)

        return {
            "instrument": "SPREAD",
            "symbol": symbol,
            "direction": direction,
            "option_type": opt_type,
            "long_strike": long_strike,
            "short_strike": short_strike,
            "long_premium": round(long_premium, 2),
            "short_premium": round(short_premium, 2),
            "net_debit": round(net_debit, 2),
            "max_profit": round(max_profit, 2),
            "quantity": num_lots * lot_size,
            "num_lots": num_lots,
            "lot_size": lot_size,
            "entry_date": date,
            "expiry": expiry,
            "score": score,
        }

    def should_exit(
        self,
        position: dict,
        date: datetime.date,
        current_spread_value: float = 0.0,
    ) -> tuple[bool, str]:
        net_debit = position["net_debit"]
        max_profit = position["max_profit"]
        expiry = position["expiry"]

        # Current P&L
        current_pnl = current_spread_value - net_debit

        # Profit cap: close if P&L >= profit_cap_pct of max_profit
        if current_pnl >= self.profit_cap_pct * max_profit:
            return True, "profit_cap"

        # Stop-loss: close if loss >= sl_multiplier * net_debit
        if current_pnl <= -self.sl_multiplier * net_debit:
            return True, "stoploss"

        # Time exit: close N days before expiry
        days_to_expiry = (expiry - date).days
        if days_to_expiry <= self.time_exit_days:
            return True, "time_exit"

        return False, ""
```

**Step 4: Run tests**

Run: `cd /Users/aravindms/financial-agent-india && python -m pytest tests/test_fo_strategies.py::TestSpreadStrategy -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add fo_strategies.py tests/test_fo_strategies.py
git commit -m "feat(fo-backtest): add SpreadStrategy with debit spread entry/exit logic"
```

---

## Task 6: Strategy Layer — Condor & Momentum Strategies

**Files:**
- Modify: `fo_strategies.py`
- Modify: `tests/test_fo_strategies.py`

**Context:**
- Condor: VIX 12-18, monthly expiry, sell 0.25-delta strangles + buy protective wings
  - Nifty: 250pt OTM shorts, 100-200pt wide wings
  - Exit: 50% max credit, 2x credit SL, delta > 0.30, 5 days pre-expiry
- Momentum: high-conviction ATM options, DTE 7-14, 1% risk, +90%/-35% target/SL, 3-day hold
- Both inherit from BaseStrategy

**Step 1: Write failing tests**

```python
# tests/test_fo_strategies.py — add to existing file

class TestCondorStrategy:
    def test_condor_entry_normal_vix(self):
        from fo_strategies import CondorStrategy
        chain = _make_chain(spot=23000.0, vix=15.0, dte=30)
        strat = CondorStrategy()
        result = strat.should_enter(
            date=datetime.date(2025, 6, 15),
            spot=23000.0,
            vix=15.0,
            available_capital=1000000.0,
            current_positions=[],
            chain=chain,
            dte=30,
        )
        assert result is not None
        assert result["instrument"] == "CONDOR"
        assert result["net_credit"] > 0
        assert result["put_short"] < result["call_short"]

    def test_condor_rejected_high_vix(self):
        from fo_strategies import CondorStrategy
        chain = _make_chain(spot=23000.0, vix=25.0, dte=30)
        strat = CondorStrategy()
        result = strat.should_enter(
            date=datetime.date(2025, 6, 15),
            spot=23000.0,
            vix=25.0,
            available_capital=1000000.0,
            current_positions=[],
            chain=chain,
            dte=30,
        )
        assert result is None  # VIX too high

    def test_condor_rejected_low_vix(self):
        from fo_strategies import CondorStrategy
        chain = _make_chain(spot=23000.0, vix=10.0, dte=30)
        strat = CondorStrategy()
        result = strat.should_enter(
            date=datetime.date(2025, 6, 15),
            spot=23000.0,
            vix=10.0,
            available_capital=1000000.0,
            current_positions=[],
            chain=chain,
            dte=30,
        )
        assert result is None  # VIX too low

    def test_condor_exit_profit_target(self):
        from fo_strategies import CondorStrategy
        strat = CondorStrategy()
        position = {
            "instrument": "CONDOR",
            "net_credit": 100.0,
            "entry_date": datetime.date(2025, 6, 10),
            "expiry": datetime.date(2025, 7, 24),
        }
        should_exit, reason = strat.should_exit(
            position=position,
            date=datetime.date(2025, 6, 20),
            current_condor_value=40.0,  # worth 40, paid 100 credit, profit = 60 = 60% of 100
        )
        assert should_exit is True
        assert reason == "profit_target"


class TestMomentumStrategy:
    def test_momentum_entry_high_conviction(self):
        from fo_strategies import MomentumStrategy
        chain = _make_chain(spot=23000.0, vix=14.0, dte=10)
        strat = MomentumStrategy()
        result = strat.should_enter(
            date=datetime.date(2025, 6, 15),
            spot=23000.0,
            vix=14.0,
            score=6.0,
            direction="bullish",
            available_capital=1000000.0,
            current_positions=[],
            chain=chain,
            dte=10,
        )
        assert result is not None
        assert result["instrument"] == "MOMENTUM"
        assert result["option_type"] == "CE"

    def test_momentum_rejected_low_score(self):
        from fo_strategies import MomentumStrategy
        chain = _make_chain(spot=23000.0, vix=14.0, dte=10)
        strat = MomentumStrategy()
        result = strat.should_enter(
            date=datetime.date(2025, 6, 15),
            spot=23000.0,
            vix=14.0,
            score=4.0,
            direction="bullish",
            available_capital=1000000.0,
            current_positions=[],
            chain=chain,
            dte=10,
        )
        assert result is None  # Below high-conviction threshold

    def test_momentum_exit_target(self):
        from fo_strategies import MomentumStrategy
        strat = MomentumStrategy()
        position = {
            "instrument": "MOMENTUM",
            "entry_premium": 100.0,
            "entry_date": datetime.date(2025, 6, 15),
        }
        should_exit, reason = strat.should_exit(
            position=position,
            date=datetime.date(2025, 6, 16),
            current_premium=195.0,  # +95% > 90% target
        )
        assert should_exit is True
        assert reason == "target"

    def test_momentum_exit_stoploss(self):
        from fo_strategies import MomentumStrategy
        strat = MomentumStrategy()
        position = {
            "instrument": "MOMENTUM",
            "entry_premium": 100.0,
            "entry_date": datetime.date(2025, 6, 15),
        }
        should_exit, reason = strat.should_exit(
            position=position,
            date=datetime.date(2025, 6, 16),
            current_premium=60.0,  # -40% > -35% SL
        )
        assert should_exit is True
        assert reason == "stoploss"

    def test_momentum_exit_time(self):
        from fo_strategies import MomentumStrategy
        strat = MomentumStrategy()
        position = {
            "instrument": "MOMENTUM",
            "entry_premium": 100.0,
            "entry_date": datetime.date(2025, 6, 15),
        }
        should_exit, reason = strat.should_exit(
            position=position,
            date=datetime.date(2025, 6, 19),  # 4 days > 3-day max hold
            current_premium=110.0,
        )
        assert should_exit is True
        assert reason == "time_exit"
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/aravindms/financial-agent-india && python -m pytest tests/test_fo_strategies.py::TestCondorStrategy tests/test_fo_strategies.py::TestMomentumStrategy -v`
Expected: FAIL

**Step 3: Implement CondorStrategy and MomentumStrategy**

Add to `fo_strategies.py`:

```python
# ---------------------------------------------------------------------------
# Condor Strategy (Iron Condor)
# ---------------------------------------------------------------------------

CONDOR_MIN_VIX = 12
CONDOR_MAX_VIX = 18
CONDOR_MAX_RISK_PCT = 0.02
CONDOR_TARGET_PCT = 0.50       # take profit at 50% of credit received
CONDOR_SL_MULTIPLIER = 2.0    # close if loss > 2x net credit
CONDOR_TIME_EXIT_DAYS = 5
CONDOR_OTM_POINTS_NIFTY = 250
CONDOR_WING_WIDTH_NIFTY = 150


class CondorStrategy(BaseStrategy):
    name = "condor"

    def __init__(
        self,
        min_vix: float = CONDOR_MIN_VIX,
        max_vix: float = CONDOR_MAX_VIX,
        max_risk_pct: float = CONDOR_MAX_RISK_PCT,
        target_pct: float = CONDOR_TARGET_PCT,
        sl_multiplier: float = CONDOR_SL_MULTIPLIER,
        time_exit_days: int = CONDOR_TIME_EXIT_DAYS,
    ):
        self.min_vix = min_vix
        self.max_vix = max_vix
        self.max_risk_pct = max_risk_pct
        self.target_pct = target_pct
        self.sl_multiplier = sl_multiplier
        self.time_exit_days = time_exit_days

    def should_enter(
        self,
        date: datetime.date,
        spot: float,
        vix: float,
        available_capital: float,
        current_positions: list[dict],
        chain: pd.DataFrame = None,
        dte: int = 0,
        symbol: str = "NIFTY",
    ) -> dict | None:
        # VIX filter
        if vix < self.min_vix or vix > self.max_vix:
            return None
        if chain is None or chain.empty:
            return None
        if dte < 15:
            return None

        lot_size = get_lot_size(symbol, date)
        atm = round(spot / 50) * 50
        otm_pts = CONDOR_OTM_POINTS_NIFTY
        wing_width = CONDOR_WING_WIDTH_NIFTY

        # Short legs
        call_short = atm + otm_pts
        put_short = atm - otm_pts
        # Long legs (protection wings)
        call_long = call_short + wing_width
        put_long = put_short - wing_width

        # Get premiums
        def _get_premium(strike, opt_type):
            row = chain[(chain["strike"] == strike) & (chain["option_type"] == opt_type)]
            return float(row.iloc[0]["premium"]) if not row.empty else 0.0

        call_short_prem = _get_premium(call_short, "CE")
        call_long_prem = _get_premium(call_long, "CE")
        put_short_prem = _get_premium(put_short, "PE")
        put_long_prem = _get_premium(put_long, "PE")

        if call_short_prem == 0 or put_short_prem == 0:
            return None

        # Net credit = (sell short legs) - (buy long legs)
        net_credit = (call_short_prem + put_short_prem) - (call_long_prem + put_long_prem)
        if net_credit <= 0:
            return None

        # Max risk = wing width - net credit (per lot)
        max_risk_per_lot = wing_width - net_credit

        # Sizing
        max_risk = available_capital * self.max_risk_pct
        num_lots = max(1, int(max_risk / (max_risk_per_lot * lot_size)))

        expiry = get_nearest_expiry(date, min_dte=15)

        return {
            "instrument": "CONDOR",
            "symbol": symbol,
            "call_short": call_short,
            "call_long": call_long,
            "put_short": put_short,
            "put_long": put_long,
            "call_short_premium": round(call_short_prem, 2),
            "call_long_premium": round(call_long_prem, 2),
            "put_short_premium": round(put_short_prem, 2),
            "put_long_premium": round(put_long_prem, 2),
            "net_credit": round(net_credit, 2),
            "max_risk": round(max_risk_per_lot, 2),
            "quantity": num_lots * lot_size,
            "num_lots": num_lots,
            "lot_size": lot_size,
            "entry_date": date,
            "expiry": expiry,
        }

    def should_exit(
        self,
        position: dict,
        date: datetime.date,
        current_condor_value: float = 0.0,
    ) -> tuple[bool, str]:
        net_credit = position["net_credit"]
        expiry = position["expiry"]

        # Profit = credit received - current cost to close
        profit = net_credit - current_condor_value

        # Profit target: 50% of net credit
        if profit >= self.target_pct * net_credit:
            return True, "profit_target"

        # Stop-loss: loss > 2x net credit
        if profit <= -self.sl_multiplier * net_credit:
            return True, "stoploss"

        # Time exit
        days_to_expiry = (expiry - date).days
        if days_to_expiry <= self.time_exit_days:
            return True, "time_exit"

        return False, ""


# ---------------------------------------------------------------------------
# Momentum Strategy (ATM options, quick directional)
# ---------------------------------------------------------------------------

MOMENTUM_MAX_RISK_PCT = 0.01
MOMENTUM_SL_PCT = 0.35
MOMENTUM_TARGET_PCT = 0.90
MOMENTUM_TIME_EXIT_DAYS = 3


class MomentumStrategy(BaseStrategy):
    name = "momentum"

    def __init__(
        self,
        score_threshold: float = HIGH_CONVICTION_THRESHOLD,
        max_risk_pct: float = MOMENTUM_MAX_RISK_PCT,
        sl_pct: float = MOMENTUM_SL_PCT,
        target_pct: float = MOMENTUM_TARGET_PCT,
        time_exit_days: int = MOMENTUM_TIME_EXIT_DAYS,
    ):
        self.score_threshold = score_threshold
        self.max_risk_pct = max_risk_pct
        self.sl_pct = sl_pct
        self.target_pct = target_pct
        self.time_exit_days = time_exit_days

    def should_enter(
        self,
        date: datetime.date,
        spot: float,
        vix: float,
        score: float,
        direction: str | None,
        available_capital: float,
        current_positions: list[dict],
        chain: pd.DataFrame = None,
        dte: int = 0,
        symbol: str = "NIFTY",
    ) -> dict | None:
        if score < self.score_threshold or direction is None:
            return None
        if chain is None or chain.empty:
            return None
        if dte < 7 or dte > 14:
            return None

        lot_size = get_lot_size(symbol, date)
        atm = round(spot / 50) * 50
        opt_type = "CE" if direction == "bullish" else "PE"

        # Get ATM premium
        atm_row = chain[
            (chain["strike"] == atm) & (chain["option_type"] == opt_type)
        ]
        if atm_row.empty:
            return None

        premium = float(atm_row.iloc[0]["premium"])

        # Sizing: max risk = max_risk_pct of capital
        max_risk = available_capital * self.max_risk_pct
        num_lots = max(1, int(max_risk / (premium * lot_size)))

        return {
            "instrument": "MOMENTUM",
            "symbol": symbol,
            "direction": direction,
            "option_type": opt_type,
            "strike": atm,
            "entry_premium": round(premium, 2),
            "quantity": num_lots * lot_size,
            "num_lots": num_lots,
            "lot_size": lot_size,
            "entry_date": date,
            "score": score,
        }

    def should_exit(
        self,
        position: dict,
        date: datetime.date,
        current_premium: float = 0.0,
    ) -> tuple[bool, str]:
        entry_premium = position["entry_premium"]
        entry_date = position["entry_date"]

        pnl_pct = (current_premium - entry_premium) / entry_premium

        # Target
        if pnl_pct >= self.target_pct:
            return True, "target"

        # Stop-loss
        if pnl_pct <= -self.sl_pct:
            return True, "stoploss"

        # Time exit
        days_held = (date - entry_date).days
        if days_held >= self.time_exit_days:
            return True, "time_exit"

        return False, ""
```

**Step 4: Run all strategy tests**

Run: `cd /Users/aravindms/financial-agent-india && python -m pytest tests/test_fo_strategies.py -v`
Expected: All tests PASS (16 total across 4 test classes)

**Step 5: Commit**

```bash
git add fo_strategies.py tests/test_fo_strategies.py
git commit -m "feat(fo-backtest): add CondorStrategy and MomentumStrategy"
```

---

## Task 7: Engine Layer — Main Backtest Loop

**Files:**
- Create: `fo_backtest.py`
- Create: `tests/test_fo_backtest.py`

**Context:**
- Engine walks daily bars from `fetch_spot_vix_history()`
- Computes V4 signals using `backtest_signals.compute_signals()`
- Generates synthetic chain per day using `generate_synthetic_chain()`
- Checks exits first, then entries
- Tracks portfolio: positions, capital, daily NAV, margin utilization
- Respects allocation caps and position limits from config.py

**Step 1: Write failing integration test**

```python
# tests/test_fo_backtest.py
import datetime
import pytest
import pandas as pd
import numpy as np


class TestFOBacktestEngine:
    def _make_spot_data(self, days=100):
        """Generate synthetic spot+VIX data for testing without yfinance."""
        dates = pd.bdate_range(start="2025-01-01", periods=days)
        np.random.seed(42)
        # Random walk around 23000
        returns = np.random.normal(0.0005, 0.012, days)
        close = 23000 * np.cumprod(1 + returns)
        high = close * (1 + np.random.uniform(0.002, 0.015, days))
        low = close * (1 - np.random.uniform(0.002, 0.015, days))
        open_ = close * (1 + np.random.normal(0, 0.005, days))

        return pd.DataFrame({
            "Open": open_, "High": high, "Low": low, "Close": close,
            "Volume": np.random.randint(100000, 1000000, days).astype(float),
            "open": open_, "high": high, "low": low, "close": close,
            "volume": np.random.randint(100000, 1000000, days).astype(float),
            "vix": np.random.uniform(12, 18, days),
        }, index=dates)

    def test_engine_runs_without_error(self):
        from fo_backtest import FOBacktestEngine
        data = self._make_spot_data(100)
        engine = FOBacktestEngine(capital=1000000)
        results = engine.run(data, symbol="NIFTY")
        assert "trades" in results
        assert "stats" in results
        assert "daily_nav" in results

    def test_engine_respects_capital_limits(self):
        from fo_backtest import FOBacktestEngine
        data = self._make_spot_data(100)
        engine = FOBacktestEngine(capital=1000000)
        results = engine.run(data, symbol="NIFTY")
        # No NAV should go below 0 (can't lose more than capital)
        assert all(nav >= 0 for nav in results["daily_nav"].values())

    def test_engine_produces_trades(self):
        from fo_backtest import FOBacktestEngine
        data = self._make_spot_data(200)
        engine = FOBacktestEngine(capital=1000000)
        results = engine.run(data, symbol="NIFTY")
        # Should have at least some trades over 200 days
        assert len(results["trades"]) > 0

    def test_engine_stats_valid(self):
        from fo_backtest import FOBacktestEngine
        data = self._make_spot_data(200)
        engine = FOBacktestEngine(capital=1000000)
        results = engine.run(data, symbol="NIFTY")
        stats = results["stats"]
        assert "total_trades" in stats
        assert "win_rate" in stats
        assert "total_pnl" in stats
        assert "max_drawdown" in stats
        if stats["total_trades"] > 0:
            assert 0 <= stats["win_rate"] <= 100

    def test_engine_strategy_breakdown(self):
        from fo_backtest import FOBacktestEngine
        data = self._make_spot_data(200)
        engine = FOBacktestEngine(capital=1000000)
        results = engine.run(data, symbol="NIFTY")
        assert "per_strategy" in results["stats"]
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/aravindms/financial-agent-india && python -m pytest tests/test_fo_backtest.py -v`
Expected: FAIL

**Step 3: Implement the engine**

```python
# fo_backtest.py
"""F&O Backtest Engine.

Walks daily bars, generates synthetic option chains, applies 4 strategies
(futures, spreads, condors, momentum), manages portfolio constraints,
and computes performance metrics.

Usage:
    python fo_backtest.py --symbol NIFTY --period 2y
    python fo_backtest.py --symbol NIFTY --period 1y --score-threshold 5.0
    python fo_backtest.py --symbol NIFTY --sweep
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import compute_atr
from backtest_signals import compute_signals
from fo_data import (
    fetch_spot_vix_history,
    generate_synthetic_chain,
    get_nearest_expiry,
    get_lot_size,
    calc_options_round_trip,
    calc_futures_round_trip,
)
from fo_strategies import (
    FuturesStrategy,
    SpreadStrategy,
    CondorStrategy,
    MomentumStrategy,
)
from greeks import black_scholes_greeks

# Portfolio constraints (from config.py)
MAX_CONCURRENT_POSITIONS = 8
CASH_RESERVE_PCT = 0.20
ALLOC_SPREADS_MAX = 0.35
ALLOC_CONDOR_MAX = 0.15
ALLOC_MOMENTUM_MAX = 0.15
ALLOC_FUTURES_MAX = 0.15

STRATEGY_ALLOC = {
    "futures": ALLOC_FUTURES_MAX,
    "spread": ALLOC_SPREADS_MAX,
    "condor": ALLOC_CONDOR_MAX,
    "momentum": ALLOC_MOMENTUM_MAX,
}


class FOBacktestEngine:
    """Main F&O backtest engine."""

    def __init__(self, capital: int = 1000000):
        self.initial_capital = capital
        self.capital = float(capital)
        self.positions: list[dict] = []
        self.closed_trades: list[dict] = []
        self.daily_nav: dict[str, float] = {}

        # Initialize strategies
        self.strategies = {
            "futures": FuturesStrategy(),
            "spread": SpreadStrategy(),
            "condor": CondorStrategy(),
            "momentum": MomentumStrategy(),
        }

    def _available_capital(self) -> float:
        """Capital available for new trades (after cash reserve)."""
        deployed = sum(p.get("margin_used", p.get("net_debit", p.get("entry_premium", 0)) * p.get("quantity", 0)) for p in self.positions)
        available = self.capital - deployed
        reserve = self.initial_capital * CASH_RESERVE_PCT
        return max(0, available - reserve)

    def _strategy_deployed(self, strategy_name: str) -> float:
        """Total capital deployed in a strategy."""
        total = 0.0
        for p in self.positions:
            if p.get("strategy") == strategy_name:
                if p["instrument"] == "FUT":
                    total += p.get("margin_used", 0)
                elif p["instrument"] == "SPREAD":
                    total += p["net_debit"] * p["quantity"]
                elif p["instrument"] == "CONDOR":
                    total += p["max_risk"] * p["quantity"]
                elif p["instrument"] == "MOMENTUM":
                    total += p["entry_premium"] * p["quantity"]
        return total

    def _can_enter_strategy(self, strategy_name: str) -> bool:
        """Check if strategy allocation cap allows new entry."""
        if len(self.positions) >= MAX_CONCURRENT_POSITIONS:
            return False
        max_alloc = STRATEGY_ALLOC.get(strategy_name, 0.15)
        deployed = self._strategy_deployed(strategy_name)
        return deployed < self.initial_capital * max_alloc

    def _check_exits(self, date, spot_row, chain):
        """Check exit conditions on all open positions."""
        high = float(spot_row.get("High", spot_row.get("high", spot_row["close"] * 1.01)))
        low = float(spot_row.get("Low", spot_row.get("low", spot_row["close"] * 0.99)))
        spot = float(spot_row["close"])
        vix = float(spot_row.get("vix", 14.0))

        to_close = []
        for i, pos in enumerate(self.positions):
            strat = self.strategies.get(pos.get("strategy", ""))
            if strat is None:
                continue

            should_exit = False
            reason = ""

            if pos["instrument"] == "FUT":
                should_exit, reason = strat.should_exit(
                    position=pos, date=date, spot=spot, high=high, low=low,
                )
                if should_exit:
                    # Determine exit price
                    if reason == "target":
                        exit_price = pos["target_price"]
                    elif reason == "stoploss":
                        exit_price = pos["stoploss_price"]
                    else:
                        exit_price = spot
                    pos["exit_price"] = exit_price
                    pos["exit_reason"] = reason

            elif pos["instrument"] == "SPREAD":
                # Recompute spread value with current spot/dte
                expiry = pos["expiry"]
                dte = max(0, (expiry - date).days)
                iv = vix / 100.0
                long_greeks = black_scholes_greeks(spot, pos["long_strike"], dte, iv=iv, option_type=pos["option_type"])
                short_greeks = black_scholes_greeks(spot, pos["short_strike"], dte, iv=iv, option_type=pos["option_type"])
                current_value = long_greeks["theoretical_price"] - short_greeks["theoretical_price"]
                should_exit, reason = strat.should_exit(
                    position=pos, date=date, current_spread_value=current_value,
                )
                if should_exit:
                    pos["exit_spread_value"] = current_value
                    pos["exit_reason"] = reason

            elif pos["instrument"] == "CONDOR":
                expiry = pos["expiry"]
                dte = max(0, (expiry - date).days)
                iv = vix / 100.0
                cs_greeks = black_scholes_greeks(spot, pos["call_short"], dte, iv=iv, option_type="CE")
                cl_greeks = black_scholes_greeks(spot, pos["call_long"], dte, iv=iv, option_type="CE")
                ps_greeks = black_scholes_greeks(spot, pos["put_short"], dte, iv=iv, option_type="PE")
                pl_greeks = black_scholes_greeks(spot, pos["put_long"], dte, iv=iv, option_type="PE")
                current_value = (cs_greeks["theoretical_price"] + ps_greeks["theoretical_price"]) - (cl_greeks["theoretical_price"] + pl_greeks["theoretical_price"])
                should_exit, reason = strat.should_exit(
                    position=pos, date=date, current_condor_value=current_value,
                )
                if should_exit:
                    pos["exit_condor_value"] = current_value
                    pos["exit_reason"] = reason

            elif pos["instrument"] == "MOMENTUM":
                expiry = pos.get("expiry", date + timedelta(days=14))
                dte = max(0, (expiry - date).days)
                iv = vix / 100.0
                greeks = black_scholes_greeks(spot, pos["strike"], dte, iv=iv, option_type=pos["option_type"])
                current_premium = greeks["theoretical_price"]
                should_exit, reason = strat.should_exit(
                    position=pos, date=date, current_premium=current_premium,
                )
                if should_exit:
                    pos["exit_premium"] = current_premium
                    pos["exit_reason"] = reason

            if should_exit:
                to_close.append(i)

        # Close positions (reverse order to preserve indices)
        for i in sorted(to_close, reverse=True):
            pos = self.positions.pop(i)
            self._close_position(pos, date)

    def _close_position(self, pos: dict, date):
        """Record closed trade and update capital."""
        pos["exit_date"] = date
        pnl = 0.0
        costs = 0.0

        if pos["instrument"] == "FUT":
            direction_mult = 1 if pos["direction"] == "bullish" else -1
            pnl = direction_mult * (pos["exit_price"] - pos["entry_price"]) * pos["quantity"]
            costs = calc_futures_round_trip(pos["entry_price"], pos["exit_price"], pos["quantity"], date)

        elif pos["instrument"] == "SPREAD":
            pnl = (pos.get("exit_spread_value", 0) - pos["net_debit"]) * pos["quantity"]
            costs = calc_options_round_trip(pos["net_debit"], pos.get("exit_spread_value", 0), pos["quantity"], date)

        elif pos["instrument"] == "CONDOR":
            # Credit strategy: profit = credit - cost to close
            pnl = (pos["net_credit"] - pos.get("exit_condor_value", 0)) * pos["quantity"]
            costs = calc_options_round_trip(pos["net_credit"], pos.get("exit_condor_value", 0), pos["quantity"], date)

        elif pos["instrument"] == "MOMENTUM":
            pnl = (pos.get("exit_premium", 0) - pos["entry_premium"]) * pos["quantity"]
            costs = calc_options_round_trip(pos["entry_premium"], pos.get("exit_premium", 0), pos["quantity"], date)

        pos["pnl_gross"] = round(pnl, 2)
        pos["costs"] = round(costs, 2)
        pos["pnl_net"] = round(pnl - costs, 2)
        pos["holding_days"] = (date - pos["entry_date"]).days

        self.capital += pos["pnl_net"]
        self.closed_trades.append(pos)

    def _check_entries(self, date, spot_row, chain, signals_row):
        """Check entry conditions across all strategies."""
        spot = float(spot_row["close"])
        vix = float(spot_row.get("vix", 14.0))
        atr_val = float(spot_row.get("atr", spot * 0.01))
        score = float(signals_row.get("score", 0))
        direction = signals_row.get("direction")
        available = self._available_capital()

        if available <= 0:
            return

        # Determine DTE for options strategies
        expiry = get_nearest_expiry(date, min_dte=5)
        dte = (expiry - date).days

        # Monthly expiry for condors
        monthly_expiry = get_nearest_expiry(date, min_dte=15)
        monthly_dte = (monthly_expiry - date).days

        # Try each strategy
        for strat_name, strat in self.strategies.items():
            if not self._can_enter_strategy(strat_name):
                continue

            entry = None
            if strat_name == "futures":
                entry = strat.should_enter(
                    date=date, spot=spot, vix=vix, atr=atr_val,
                    score=score, direction=direction,
                    available_capital=available, current_positions=self.positions,
                )
            elif strat_name == "spread":
                spread_expiry = get_nearest_expiry(date, min_dte=30)
                spread_dte = (spread_expiry - date).days
                entry = strat.should_enter(
                    date=date, spot=spot, vix=vix, atr=atr_val,
                    score=score, direction=direction,
                    available_capital=available, current_positions=self.positions,
                    chain=chain, dte=spread_dte,
                )
            elif strat_name == "condor":
                entry = strat.should_enter(
                    date=date, spot=spot, vix=vix,
                    available_capital=available, current_positions=self.positions,
                    chain=chain, dte=monthly_dte,
                )
            elif strat_name == "momentum":
                entry = strat.should_enter(
                    date=date, spot=spot, vix=vix,
                    score=score, direction=direction,
                    available_capital=available, current_positions=self.positions,
                    chain=chain, dte=dte,
                )

            if entry is not None:
                entry["strategy"] = strat_name
                self.positions.append(entry)

    def run(self, data: pd.DataFrame, symbol: str = "NIFTY") -> dict:
        """Run the backtest on spot+VIX data.

        Args:
            data: DataFrame with columns: Open/open, High/high, Low/low,
                  Close/close, Volume/volume, vix
            symbol: NIFTY or BANKNIFTY

        Returns dict with trades, stats, daily_nav.
        """
        # Ensure we have both Upper and lower case columns for compatibility
        if "Close" not in data.columns and "close" in data.columns:
            data = data.copy()
            data["Open"] = data["open"]
            data["High"] = data["high"]
            data["Low"] = data["low"]
            data["Close"] = data["close"]
            data["Volume"] = data["volume"]

        # Compute signals and ATR
        signals = compute_signals(data)
        atr_series = compute_atr(data)

        for i in range(30, len(data)):  # skip warmup
            row = data.iloc[i]
            date = data.index[i].date() if hasattr(data.index[i], 'date') else data.index[i]
            spot = float(row["close"])
            vix = float(row.get("vix", 14.0))

            # Add ATR to row for strategies
            row_dict = row.to_dict()
            row_dict["atr"] = float(atr_series.iloc[i]) if not pd.isna(atr_series.iloc[i]) else spot * 0.01

            # Generate synthetic chain
            expiry = get_nearest_expiry(date, min_dte=5)
            dte = (expiry - date).days
            chain = generate_synthetic_chain(spot, vix, dte, symbol)

            # Signal data for this bar
            sig = {
                "score": float(signals["score"].iloc[i]) if "score" in signals.columns else 0.0,
                "direction": signals["direction"].iloc[i] if "direction" in signals.columns else None,
            }

            # Check exits first, then entries
            self._check_exits(date, row_dict, chain)
            self._check_entries(date, row_dict, chain, sig)

            # Record daily NAV
            self.daily_nav[str(date)] = round(self.capital, 2)

        # Close any remaining open positions at last bar
        if self.positions:
            last_row = data.iloc[-1]
            last_date = data.index[-1].date() if hasattr(data.index[-1], 'date') else data.index[-1]
            for pos in list(self.positions):
                if pos["instrument"] == "FUT":
                    pos["exit_price"] = float(last_row["close"])
                    pos["exit_reason"] = "backtest_end"
                elif pos["instrument"] == "SPREAD":
                    pos["exit_spread_value"] = pos["net_debit"]  # assume flat
                    pos["exit_reason"] = "backtest_end"
                elif pos["instrument"] == "CONDOR":
                    pos["exit_condor_value"] = pos["net_credit"]  # assume flat
                    pos["exit_reason"] = "backtest_end"
                elif pos["instrument"] == "MOMENTUM":
                    pos["exit_premium"] = pos["entry_premium"]
                    pos["exit_reason"] = "backtest_end"
            for pos in list(self.positions):
                self.positions.remove(pos)
                self._close_position(pos, last_date)

        return {
            "trades": self.closed_trades,
            "stats": self._compute_stats(),
            "daily_nav": self.daily_nav,
        }

    def _compute_stats(self) -> dict:
        """Compute performance statistics."""
        trades = self.closed_trades
        if not trades:
            return {
                "total_trades": 0, "win_rate": 0, "total_pnl": 0,
                "max_drawdown": 0, "per_strategy": {},
            }

        wins = [t for t in trades if t["pnl_net"] >= 0]
        losses = [t for t in trades if t["pnl_net"] < 0]
        total_pnl = sum(t["pnl_net"] for t in trades)
        total_costs = sum(t["costs"] for t in trades)
        win_rate = len(wins) / len(trades) * 100 if trades else 0

        # Sharpe
        pnl_list = [t["pnl_net"] for t in trades]
        mean_pnl = sum(pnl_list) / len(pnl_list)
        var = sum((p - mean_pnl) ** 2 for p in pnl_list) / len(pnl_list)
        std = math.sqrt(var) if var > 0 else 0
        sharpe = round((mean_pnl / std) * math.sqrt(252), 2) if std > 0 else 0

        # Max drawdown from daily NAV
        nav_values = list(self.daily_nav.values())
        max_dd = 0
        peak = nav_values[0] if nav_values else self.initial_capital
        for nav in nav_values:
            if nav > peak:
                peak = nav
            dd = peak - nav
            if dd > max_dd:
                max_dd = dd

        # Profit factor
        win_sum = sum(t["pnl_net"] for t in wins)
        loss_sum = abs(sum(t["pnl_net"] for t in losses))
        pf = round(win_sum / loss_sum, 2) if loss_sum > 0 else float("inf")

        # Per-strategy breakdown
        per_strategy = {}
        for t in trades:
            strat = t.get("strategy", "unknown")
            if strat not in per_strategy:
                per_strategy[strat] = {"trades": 0, "wins": 0, "pnl": 0, "costs": 0}
            per_strategy[strat]["trades"] += 1
            per_strategy[strat]["pnl"] += t["pnl_net"]
            per_strategy[strat]["costs"] += t["costs"]
            if t["pnl_net"] >= 0:
                per_strategy[strat]["wins"] += 1

        # Exit reason breakdown
        exit_reasons = defaultdict(int)
        for t in trades:
            exit_reasons[t.get("exit_reason", "unknown")] += 1

        return {
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "total_costs": round(total_costs, 2),
            "sharpe_ratio": sharpe,
            "max_drawdown": round(max_dd, 2),
            "profit_factor": pf,
            "per_strategy": per_strategy,
            "exit_reasons": dict(exit_reasons),
            "final_capital": round(self.capital, 2),
            "return_pct": round((self.capital - self.initial_capital) / self.initial_capital * 100, 2),
        }


def print_report(results: dict, initial_capital: int = 1000000):
    """Print formatted backtest report."""
    stats = results["stats"]
    if stats["total_trades"] == 0:
        print("No trades generated.")
        return

    border = "=" * 60
    sep = "-" * 58

    print(f"\n{border}")
    print("  F&O BACKTEST RESULTS")
    print(border)

    print(f"\n  Total Trades: {stats['total_trades']}  |  "
          f"Win Rate: {stats['win_rate']:.1f}%  "
          f"({stats['winning_trades']}W / {stats['losing_trades']}L)")
    print(f"  Total P&L: \u20b9{stats['total_pnl']:+,.0f}  |  "
          f"Costs: \u20b9{stats['total_costs']:,.0f}")
    print(f"  Final Capital: \u20b9{stats['final_capital']:,.0f}  |  "
          f"Return: {stats['return_pct']:+.1f}%")

    print(f"\n  {sep}")
    print(f"  RISK METRICS")
    print(f"  {sep}")
    pf = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] != float('inf') else "\u221e"
    print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}  |  Profit Factor: {pf}")
    print(f"  Max Drawdown: \u20b9{stats['max_drawdown']:,.0f}")

    print(f"\n  {sep}")
    print(f"  PER STRATEGY")
    print(f"  {sep}")
    for strat, data in stats.get("per_strategy", {}).items():
        wr = data["wins"] / data["trades"] * 100 if data["trades"] > 0 else 0
        print(f"  {strat:12s}  {data['trades']:3d} trades  "
              f"{wr:.0f}% WR  \u20b9{data['pnl']:+,.0f}")

    print(f"\n  {sep}")
    print(f"  EXIT REASONS")
    print(f"  {sep}")
    for reason, count in sorted(stats.get("exit_reasons", {}).items(), key=lambda x: -x[1]):
        pct = count / stats["total_trades"] * 100
        print(f"  {reason:20s} {count:4d}  ({pct:.0f}%)")

    print(f"\n{border}\n")


def main():
    parser = argparse.ArgumentParser(description="F&O Backtest Engine")
    parser.add_argument("--symbol", type=str, default="NIFTY", choices=["NIFTY", "BANKNIFTY"])
    parser.add_argument("--period", type=str, default="2y", help="yfinance period (1y, 2y)")
    parser.add_argument("--capital", type=int, default=1000000, help="Starting capital (INR)")
    parser.add_argument("--score-threshold", type=float, default=3.5, help="Min signal score for entry")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    print(f"Fetching {args.symbol} data for {args.period}...")
    data = fetch_spot_vix_history(args.symbol, args.period)
    if data.empty:
        print("No data available")
        sys.exit(1)
    print(f"Got {len(data)} daily bars")

    engine = FOBacktestEngine(capital=args.capital)
    # Update score thresholds if specified
    if args.score_threshold != 3.5:
        engine.strategies["futures"].score_threshold = args.score_threshold
        engine.strategies["spread"].score_threshold = args.score_threshold

    results = engine.run(data, symbol=args.symbol)
    print_report(results, args.capital)

    if args.output:
        # Convert dates to strings for JSON serialization
        output = {
            "stats": results["stats"],
            "trades": [{k: str(v) if isinstance(v, (datetime, pd.Timestamp)) else v
                       for k, v in t.items()} for t in results["trades"]],
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2, default=str))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests**

Run: `cd /Users/aravindms/financial-agent-india && python -m pytest tests/test_fo_backtest.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add fo_backtest.py tests/test_fo_backtest.py
git commit -m "feat(fo-backtest): add main engine with portfolio management and reporting"
```

---

## Task 8: Run Full Backtest & Save Results

**Files:**
- No new files. Run existing code.

**Context:**
- This is the validation step — run the engine on real 2-year data and verify it produces meaningful results
- Save output to `data/fo_backtest/results_20260310.json`

**Step 1: Create data directory**

```bash
mkdir -p /Users/aravindms/financial-agent-india/data/fo_backtest
```

**Step 2: Run the backtest**

```bash
cd /Users/aravindms/financial-agent-india
python fo_backtest.py --symbol NIFTY --period 2y --capital 1000000 --output data/fo_backtest/results_20260310.json
```

Expected: Report prints with trade count, win rate, P&L, per-strategy breakdown. Should take 1-3 minutes (generating 500+ synthetic chains).

**Step 3: Verify results file**

```bash
python -c "
import json
from pathlib import Path
r = json.loads(Path('data/fo_backtest/results_20260310.json').read_text())
print(f'Total trades: {r[\"stats\"][\"total_trades\"]}')
print(f'Win rate: {r[\"stats\"][\"win_rate\"]}%')
print(f'P&L: {r[\"stats\"][\"total_pnl\"]}')
for s, d in r['stats'].get('per_strategy', {}).items():
    print(f'  {s}: {d[\"trades\"]} trades, P&L={d[\"pnl\"]}')
"
```

**Step 4: Run all tests to confirm nothing is broken**

```bash
cd /Users/aravindms/financial-agent-india && python -m pytest tests/test_fo_data.py tests/test_fo_strategies.py tests/test_fo_backtest.py -v
```

Expected: All tests PASS

**Step 5: Commit results**

```bash
git add data/fo_backtest/results_20260310.json
git commit -m "feat(fo-backtest): initial 2-year NIFTY backtest results"
```

---

## Summary

| Task | What | Files | Tests |
|------|------|-------|-------|
| 1 | Expiry calendar + lot sizes | fo_data.py | 11 |
| 2 | Spot/VIX history + synthetic chains | fo_data.py | 6 |
| 3 | Date-aware transaction costs | fo_data.py | 5 |
| 4 | Base class + FuturesStrategy | fo_strategies.py | 6 |
| 5 | SpreadStrategy | fo_strategies.py | 5 |
| 6 | Condor + Momentum strategies | fo_strategies.py | 8 |
| 7 | Main engine + reporting | fo_backtest.py | 5 |
| 8 | Full backtest run + results | — | — |
| **Total** | | **3 new files + 3 test files** | **46 tests** |
