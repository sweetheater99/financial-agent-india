# F&O Backtest V1.2 — Real Kite Data Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Calibrate synthetic option pricing against real Kite chains, collect daily chain snapshots for future backtesting, and enable the engine to use real data when available.

**Architecture:** Three components — calibration script compares real vs synthetic, collector script snapshots daily chains to parquet, engine swap-in prefers real data over synthetic. All reuse existing `kite_data.py` (KiteConnect) and `greeks.py` (IV inversion).

**Tech Stack:** Python, kiteconnect, pandas, pyarrow (parquet), existing `kite_data.py`, `fo_data.py`, `greeks.py`

---

### Task 1: Chain Collector — Tests

**Files:**
- Create: `tests/test_fo_chain_collector.py`

**Context:**
- `kite_data.py` has `fetch_option_chain_kite(symbol)` returning SmartAPI format: `[{"strikePrice": 23000, "CE": {"lastTradedPrice": 285, "bidPrice": 284, "askPrice": 286, ...}, "PE": {...}}, ...]`
- `greeks.py` has `implied_volatility(market_price, spot, strike, dte, risk_free, option_type)` returning IV as decimal
- We need to convert Kite's format to our backtest DataFrame format (strike, option_type, premium, bid, ask, oi, volume, iv, delta, gamma, theta, vega, expiry, spot, vix)

**Step 1: Write tests**

```python
"""Tests for fo_chain_collector — daily chain snapshot collection."""
import datetime
import os
import tempfile
import unittest

import pandas as pd


def _make_mock_kite_chain():
    """Mock Kite chain in SmartAPI format."""
    return [
        {
            "strikePrice": 23000,
            "CE": {
                "lastTradedPrice": 285.0,
                "bidPrice": 284.0,
                "askPrice": 286.0,
                "openInterest": 500000,
                "volume": 120000,
                "lotSize": 75,
            },
            "PE": {
                "lastTradedPrice": 270.0,
                "bidPrice": 269.0,
                "askPrice": 271.0,
                "openInterest": 450000,
                "volume": 110000,
                "lotSize": 75,
            },
        },
        {
            "strikePrice": 23050,
            "CE": {
                "lastTradedPrice": 260.0,
                "bidPrice": 259.0,
                "askPrice": 261.0,
                "openInterest": 300000,
                "volume": 80000,
                "lotSize": 75,
            },
            "PE": {
                "lastTradedPrice": 295.0,
                "bidPrice": 294.0,
                "askPrice": 296.0,
                "openInterest": 350000,
                "volume": 90000,
                "lotSize": 75,
            },
        },
    ]


class TestChainCollector(unittest.TestCase):

    def test_convert_kite_to_backtest_format(self):
        """Kite chain should convert to backtest DataFrame format."""
        from fo_chain_collector import convert_kite_chain_to_df
        kite_chain = _make_mock_kite_chain()
        df = convert_kite_chain_to_df(
            kite_chain, spot=23000.0, vix=14.0,
            expiry_str="2026-04-02", symbol="NIFTY",
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4  # 2 strikes x 2 types
        required_cols = ["strike", "option_type", "premium", "bid", "ask",
                         "oi", "volume", "iv", "delta", "gamma", "theta", "vega",
                         "expiry", "spot", "vix"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_convert_premiums_match_kite(self):
        """Premiums should come from Kite lastTradedPrice."""
        from fo_chain_collector import convert_kite_chain_to_df
        kite_chain = _make_mock_kite_chain()
        df = convert_kite_chain_to_df(
            kite_chain, spot=23000.0, vix=14.0,
            expiry_str="2026-04-02", symbol="NIFTY",
        )
        ce_23000 = df[(df["strike"] == 23000) & (df["option_type"] == "CE")]
        assert float(ce_23000.iloc[0]["premium"]) == 285.0
        assert float(ce_23000.iloc[0]["bid"]) == 284.0
        assert float(ce_23000.iloc[0]["ask"]) == 286.0

    def test_convert_computes_iv(self):
        """IV should be computed via BS inversion, not zero."""
        from fo_chain_collector import convert_kite_chain_to_df
        kite_chain = _make_mock_kite_chain()
        df = convert_kite_chain_to_df(
            kite_chain, spot=23000.0, vix=14.0,
            expiry_str="2026-04-02", symbol="NIFTY",
        )
        # IV should be positive for ATM options
        ce_23000 = df[(df["strike"] == 23000) & (df["option_type"] == "CE")]
        assert float(ce_23000.iloc[0]["iv"]) > 0.05, "IV should be computed, not zero"

    def test_save_and_load_parquet(self):
        """Chain should round-trip through parquet."""
        from fo_chain_collector import convert_kite_chain_to_df, save_chain, load_chain
        kite_chain = _make_mock_kite_chain()
        df = convert_kite_chain_to_df(
            kite_chain, spot=23000.0, vix=14.0,
            expiry_str="2026-04-02", symbol="NIFTY",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            save_chain(df, "NIFTY", datetime.date(2026, 3, 10), output_dir=tmpdir)
            loaded = load_chain("NIFTY", datetime.date(2026, 3, 10), data_dir=tmpdir)
            assert loaded is not None
            assert len(loaded) == len(df)
            assert list(loaded.columns) == list(df.columns)

    def test_load_missing_returns_none(self):
        """Loading a non-existent chain should return None."""
        from fo_chain_collector import load_chain
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_chain("NIFTY", datetime.date(2020, 1, 1), data_dir=tmpdir)
            assert result is None


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run tests to verify they fail**

Run: `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m pytest tests/test_fo_chain_collector.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'fo_chain_collector'`

---

### Task 2: Chain Collector — Implementation

**Files:**
- Create: `fo_chain_collector.py`

**Context:**
- `kite_data.py:281` has `fetch_option_chain_kite(symbol, expiry_str)` returning SmartAPI format list
- `kite_data.py:405` has `get_vix_kite()` returning float
- `kite_data.py:174` has `get_ltp_kite(symbol)` returning float
- `kite_data.py:154` has `get_nfo_instruments(symbol)` returning instrument list with expiry info
- `greeks.py:145` has `implied_volatility(market_price, spot, strike, dte, risk_free, option_type)` returning IV
- `greeks.py:59` has `black_scholes_greeks(spot, strike, dte, risk_free, iv, option_type)` returning greeks dict

**Step 1: Write fo_chain_collector.py**

```python
"""F&O Chain Collector — daily option chain snapshots from Kite.

Fetches live option chains for NIFTY and BANKNIFTY, computes IV and greeks,
saves as parquet for use in backtesting.

Usage:
    python fo_chain_collector.py                        # Collect NIFTY + BANKNIFTY
    python fo_chain_collector.py --symbol NIFTY         # NIFTY only
    python fo_chain_collector.py --dry-run               # Show what would be collected
"""

import argparse
import logging
import math
import os
import sys
from datetime import date, datetime

import pandas as pd

from greeks import black_scholes_greeks, implied_volatility

logger = logging.getLogger("fo_chain_collector")

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "fo_chains")


def convert_kite_chain_to_df(
    kite_chain: list,
    spot: float,
    vix: float,
    expiry_str: str,
    symbol: str = "NIFTY",
    risk_free: float = 0.065,
    min_premium: float = 5.0,
) -> pd.DataFrame:
    """Convert Kite SmartAPI-format chain to backtest DataFrame format.

    Computes IV via BS inversion and greeks from the computed IV.
    """
    # Compute DTE from expiry
    try:
        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    except ValueError:
        expiry_date = date.today()
    dte = max(1, (expiry_date - date.today()).days)

    rows = []
    for entry in kite_chain:
        strike = float(entry["strikePrice"])

        for opt_type, kite_key in [("CE", "CE"), ("PE", "PE")]:
            opt_data = entry.get(kite_key)
            if opt_data is None:
                continue

            premium = float(opt_data.get("lastTradedPrice", 0))
            if premium < min_premium:
                continue

            bid = float(opt_data.get("bidPrice", 0))
            ask = float(opt_data.get("askPrice", 0))
            oi = int(opt_data.get("openInterest", 0))
            volume = int(opt_data.get("volume", 0))

            # Compute IV from market price
            iv = implied_volatility(premium, spot, strike, dte, risk_free, opt_type)
            if math.isnan(iv) or iv <= 0:
                iv = vix / 100.0  # fallback to VIX

            # Compute greeks from the IV
            greeks = black_scholes_greeks(
                spot=spot, strike=strike, dte=dte,
                risk_free=risk_free, iv=iv, option_type=opt_type,
            )

            rows.append({
                "strike": strike,
                "option_type": opt_type,
                "premium": premium,
                "bid": bid,
                "ask": ask,
                "oi": oi,
                "volume": volume,
                "iv": iv,
                "delta": greeks["delta"],
                "gamma": greeks["gamma"],
                "theta": greeks["theta"],
                "vega": greeks["vega"],
                "expiry": expiry_str,
                "spot": spot,
                "vix": vix,
            })

    return pd.DataFrame(rows)


def save_chain(df: pd.DataFrame, symbol: str, chain_date: date, output_dir: str = None):
    """Save chain DataFrame as parquet."""
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{symbol}_{chain_date.strftime('%Y%m%d')}.parquet")
    df.to_parquet(path, index=False)
    logger.info("Saved %d rows to %s", len(df), path)
    return path


def load_chain(symbol: str, chain_date: date, data_dir: str = None) -> pd.DataFrame | None:
    """Load chain parquet for a specific date. Returns None if not found."""
    data_dir = data_dir or DEFAULT_OUTPUT_DIR
    path = os.path.join(data_dir, f"{symbol}_{chain_date.strftime('%Y%m%d')}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


def collect_symbol(symbol: str, dry_run: bool = False) -> pd.DataFrame | None:
    """Fetch live chain from Kite and save as parquet."""
    from kite_data import fetch_option_chain_kite, get_ltp_kite, get_vix_kite, get_nfo_instruments

    # Get spot and VIX
    spot = get_ltp_kite(symbol)
    vix = get_vix_kite()

    if spot is None or vix is None:
        logger.error("Failed to fetch spot (%s) or VIX (%s)", spot, vix)
        return None

    # Find nearest 2 expiries
    instruments = get_nfo_instruments(symbol, inst_type="CE")
    expiries = sorted(set(
        inst["expiry"] for inst in instruments
        if inst["expiry"] >= str(date.today())
    ))[:2]

    if not expiries:
        logger.error("No active expiries found for %s", symbol)
        return None

    if dry_run:
        print(f"  {symbol}: spot={spot:.0f}, vix={vix:.1f}")
        print(f"  Expiries: {expiries}")
        print(f"  Would save to: data/fo_chains/{symbol}_{date.today().strftime('%Y%m%d')}.parquet")
        return None

    # Fetch and combine chains for each expiry
    all_dfs = []
    for expiry_str in expiries:
        kite_chain = fetch_option_chain_kite(symbol, expiry_str=expiry_str)
        if kite_chain is None:
            logger.warning("No chain data for %s expiry %s", symbol, expiry_str)
            continue

        df = convert_kite_chain_to_df(kite_chain, spot, vix, expiry_str, symbol)
        all_dfs.append(df)
        logger.info("%s expiry %s: %d rows", symbol, expiry_str, len(df))

    if not all_dfs:
        logger.error("No chain data collected for %s", symbol)
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    path = save_chain(combined, symbol, date.today())
    print(f"  {symbol}: {len(combined)} rows saved to {path}")
    return combined


def main():
    parser = argparse.ArgumentParser(description="F&O Chain Collector")
    parser.add_argument("--symbol", default=None, help="NIFTY or BANKNIFTY (default: both)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be collected")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    symbols = [args.symbol] if args.symbol else ["NIFTY", "BANKNIFTY"]

    print(f"F&O Chain Collector — {date.today()}")
    print("=" * 50)

    for symbol in symbols:
        try:
            collect_symbol(symbol, dry_run=args.dry_run)
        except Exception as e:
            logger.error("Failed to collect %s: %s", symbol, e)

    print("=" * 50)
    print("Done.")


if __name__ == "__main__":
    main()
```

**Step 2: Run tests**

Run: `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m pytest tests/test_fo_chain_collector.py -v`

Expected: All 5 tests PASS.

**Step 3: Run all existing tests to check for regressions**

Run: `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m pytest tests/test_fo_data.py tests/test_fo_strategies.py tests/test_fo_backtest.py tests/test_fo_param_sweep.py tests/test_fo_chain_collector.py -v`

Expected: All 69 tests PASS (64 existing + 5 new).

---

### Task 3: Engine Swap-In — Real Chain Preference

**Files:**
- Modify: `fo_data.py:210+` (add `fetch_real_chain` function)
- Modify: `fo_backtest.py:175-180` (prefer real chain)
- Modify: `tests/test_fo_backtest.py` (add test for real chain preference)

**Context:**
- `fo_backtest.py:178` currently calls `chain = generate_synthetic_chain(spot, vix, dte, symbol=symbol)`
- We want to try loading a real parquet chain first, fall back to synthetic
- `fo_chain_collector.py` has `load_chain(symbol, date, data_dir)` but we should add a wrapper in `fo_data.py` so the import stays clean

**Step 1: Add test for real chain preference**

Add to `tests/test_fo_backtest.py`:

```python
def test_engine_prefers_real_chain(self):
    """Engine should use real chain when available, fall back to synthetic."""
    import tempfile
    import os
    from fo_data import fetch_real_chain
    from fo_chain_collector import convert_kite_chain_to_df, save_chain

    # Create a fake real chain
    kite_chain = [
        {"strikePrice": 23000,
         "CE": {"lastTradedPrice": 200.0, "bidPrice": 199.0, "askPrice": 201.0,
                "openInterest": 100, "volume": 50, "lotSize": 75},
         "PE": {"lastTradedPrice": 180.0, "bidPrice": 179.0, "askPrice": 181.0,
                "openInterest": 100, "volume": 50, "lotSize": 75}},
    ]
    df = convert_kite_chain_to_df(kite_chain, spot=23000.0, vix=14.0,
                                   expiry_str="2026-04-02", symbol="NIFTY")

    with tempfile.TemporaryDirectory() as tmpdir:
        import datetime
        save_chain(df, "NIFTY", datetime.date(2026, 3, 10), output_dir=tmpdir)
        loaded = fetch_real_chain("NIFTY", datetime.date(2026, 3, 10), data_dir=tmpdir)
        assert loaded is not None
        assert len(loaded) == 2  # 1 strike x 2 types
        assert "premium" in loaded.columns

        # Non-existent date returns None (falls back to synthetic)
        missing = fetch_real_chain("NIFTY", datetime.date(2020, 1, 1), data_dir=tmpdir)
        assert missing is None
```

**Step 2: Add `fetch_real_chain` to `fo_data.py`**

Add at the end of `fo_data.py` (after `get_futures_price`):

```python
def fetch_real_chain(
    symbol: str, chain_date: datetime.date, data_dir: str = None
) -> pd.DataFrame | None:
    """Load real option chain snapshot from parquet. Returns None if not available."""
    data_dir = data_dir or os.path.join(os.path.dirname(__file__), "data", "fo_chains")
    path = os.path.join(data_dir, f"{symbol}_{chain_date.strftime('%Y%m%d')}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None
```

Also add `import os` at the top of `fo_data.py` if not already present.

**Step 3: Modify `fo_backtest.py` to prefer real chain**

Change `fo_backtest.py:21-28` imports to add `fetch_real_chain`:

```python
from fo_data import (
    calc_futures_round_trip,
    calc_options_round_trip,
    fetch_real_chain,
    generate_synthetic_chain,
    get_futures_price,
    get_lot_size,
    get_nearest_expiry,
)
```

Change `fo_backtest.py:177-180` from:

```python
            try:
                chain = generate_synthetic_chain(spot, vix, dte, symbol=symbol)
            except Exception:
                chain = pd.DataFrame()
```

To:

```python
            try:
                chain = fetch_real_chain(symbol, current_date)
                if chain is None:
                    chain = generate_synthetic_chain(spot, vix, dte, symbol=symbol)
            except Exception:
                chain = pd.DataFrame()
```

**Step 4: Run all tests**

Run: `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m pytest tests/test_fo_data.py tests/test_fo_strategies.py tests/test_fo_backtest.py tests/test_fo_param_sweep.py tests/test_fo_chain_collector.py -v`

Expected: All 70 tests PASS (64 existing + 5 collector + 1 new engine test).

---

### Task 4: Calibration Script

**Files:**
- Create: `fo_calibrate.py`
- Create: `tests/test_fo_calibrate.py`

**Context:**
- This compares real Kite chain against synthetic chain at matching spot/VIX/DTE
- Uses `fetch_option_chain_kite()` for real data, `generate_synthetic_chain()` for synthetic
- Computes condor credit-to-width ratio for both to validate realism
- This requires a live Kite connection, so tests use mock data

**Step 1: Write tests**

```python
"""Tests for fo_calibrate — synthetic vs real chain comparison."""
import unittest


class TestCalibrate(unittest.TestCase):

    def test_compare_chains_returns_report(self):
        """compare_chains should return a dict with per-strike comparisons."""
        from fo_calibrate import compare_chains
        from fo_data import generate_synthetic_chain

        # Use synthetic as both "real" and "synthetic" for testing
        chain = generate_synthetic_chain(spot=23000.0, vix=14.0, dte=30, symbol="NIFTY")
        report = compare_chains(
            real_chain=chain, synthetic_chain=chain,
            spot=23000.0, symbol="NIFTY",
        )
        assert "comparisons" in report
        assert "condor_real_credit_pct" in report
        assert "condor_synth_credit_pct" in report
        assert len(report["comparisons"]) > 0

    def test_compare_identical_chains_small_gap(self):
        """Identical chains should show ~0% gap."""
        from fo_calibrate import compare_chains
        from fo_data import generate_synthetic_chain

        chain = generate_synthetic_chain(spot=23000.0, vix=14.0, dte=30, symbol="NIFTY")
        report = compare_chains(
            real_chain=chain, synthetic_chain=chain,
            spot=23000.0, symbol="NIFTY",
        )
        # Same chain compared to itself: gap should be ~0
        for comp in report["comparisons"]:
            assert abs(comp["diff_pct"]) < 0.01, f"Gap too large for identical chains: {comp}"


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Write fo_calibrate.py**

```python
"""F&O Calibration — compare real Kite option chains against synthetic pricing.

Validates that skew_factor and bid-ask spread model produce realistic premiums.

Usage:
    python fo_calibrate.py                     # NIFTY calibration (requires Kite login)
    python fo_calibrate.py --symbol BANKNIFTY  # BANKNIFTY calibration
"""

import argparse
import logging
import sys
from datetime import date, datetime

import pandas as pd

from fo_data import generate_synthetic_chain, get_nearest_expiry

logger = logging.getLogger("fo_calibrate")


def compare_chains(
    real_chain: pd.DataFrame,
    synthetic_chain: pd.DataFrame,
    spot: float,
    symbol: str = "NIFTY",
) -> dict:
    """Compare real vs synthetic chains at key strikes.

    Returns dict with per-strike comparisons and condor credit-to-width ratios.
    """
    interval = 50 if symbol == "NIFTY" else 100
    atm = round(spot / interval) * interval

    # Key strikes to compare
    offsets = [0, 250, 500, 750, -250, -500, -750]
    comparisons = []

    for offset in offsets:
        strike = atm + offset
        for opt_type in ["CE", "PE"]:
            real_row = real_chain[
                (real_chain["strike"] == strike) & (real_chain["option_type"] == opt_type)
            ]
            synth_row = synthetic_chain[
                (synthetic_chain["strike"] == strike) & (synthetic_chain["option_type"] == opt_type)
            ]
            if real_row.empty or synth_row.empty:
                continue

            real_prem = float(real_row.iloc[0]["premium"])
            synth_prem = float(synth_row.iloc[0]["premium"])
            real_iv = float(real_row.iloc[0]["iv"])
            synth_iv = float(synth_row.iloc[0]["iv"])

            diff_pct = (synth_prem - real_prem) / real_prem if real_prem > 0 else 0

            comparisons.append({
                "strike": strike,
                "option_type": opt_type,
                "real_premium": round(real_prem, 1),
                "synth_premium": round(synth_prem, 1),
                "diff_pct": round(diff_pct, 4),
                "real_iv": round(real_iv * 100, 1),
                "synth_iv": round(synth_iv * 100, 1),
            })

    # Compute condor credit-to-width for both
    wing_width = 300
    otm_pts = 500

    def _condor_credit(chain):
        def _get(s, t):
            row = chain[(chain["strike"] == s) & (chain["option_type"] == t)]
            return float(row.iloc[0]["premium"]) if not row.empty else 0

        cs = _get(atm + otm_pts, "CE")
        cl = _get(atm + otm_pts + wing_width, "CE")
        ps = _get(atm - otm_pts, "PE")
        pl = _get(atm - otm_pts - wing_width, "PE")
        nc = (cs + ps) - (cl + pl)
        return nc / wing_width if wing_width > 0 else 0

    real_credit_pct = _condor_credit(real_chain)
    synth_credit_pct = _condor_credit(synthetic_chain)

    return {
        "comparisons": comparisons,
        "condor_real_credit_pct": round(real_credit_pct * 100, 1),
        "condor_synth_credit_pct": round(synth_credit_pct * 100, 1),
        "condor_gap_pct": round(abs(real_credit_pct - synth_credit_pct) * 100, 1),
    }


def print_calibration_report(report: dict, spot: float, vix: float, dte: int, symbol: str):
    """Print formatted calibration report."""
    print(f"\nF&O CALIBRATION REPORT — {symbol}")
    print(f"Spot: {spot:.0f}  VIX: {vix:.1f}  DTE: {dte}")
    print("=" * 70)
    print(f"{'Strike':<8} {'Type':<5} {'Real':>8} {'Synth':>8} {'Diff%':>7} {'Real_IV':>8} {'Synth_IV':>9}")
    print("-" * 70)

    for c in sorted(report["comparisons"], key=lambda x: (x["strike"], x["option_type"])):
        print(
            f"{c['strike']:<8} {c['option_type']:<5} "
            f"{c['real_premium']:>8.1f} {c['synth_premium']:>8.1f} "
            f"{c['diff_pct']:>+6.1%} "
            f"{c['real_iv']:>7.1f}% {c['synth_iv']:>8.1f}%"
        )

    print("-" * 70)
    print(f"\nCondor (500pt OTM, 300pt wings):")
    print(f"  Real credit-to-width:      {report['condor_real_credit_pct']:.1f}%")
    print(f"  Synthetic credit-to-width: {report['condor_synth_credit_pct']:.1f}%")
    print(f"  Gap: {report['condor_gap_pct']:.1f}%", end="")
    if report["condor_gap_pct"] < 10:
        print(" — ACCEPTABLE (< 10%)")
    else:
        print(" — NEEDS TUNING (>= 10%)")


def main():
    parser = argparse.ArgumentParser(description="F&O Calibration")
    parser.add_argument("--symbol", default="NIFTY")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # These imports require a live Kite connection
    from kite_data import fetch_option_chain_kite, get_ltp_kite, get_vix_kite

    spot = get_ltp_kite(args.symbol)
    vix = get_vix_kite()

    if spot is None or vix is None:
        print(f"ERROR: Could not fetch spot ({spot}) or VIX ({vix}). Is Kite logged in?")
        sys.exit(1)

    # Fetch real chain
    kite_chain = fetch_option_chain_kite(args.symbol)
    if kite_chain is None:
        print(f"ERROR: Could not fetch option chain for {args.symbol}")
        sys.exit(1)

    # Convert real chain to backtest format
    from fo_chain_collector import convert_kite_chain_to_df
    from kite_data import _find_nearest_expiry

    expiry_str = _find_nearest_expiry(args.symbol)
    expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    dte = max(1, (expiry_date - date.today()).days)

    real_df = convert_kite_chain_to_df(kite_chain, spot, vix, expiry_str, args.symbol)

    # Generate synthetic chain with same params
    synth_df = generate_synthetic_chain(spot, vix, dte, symbol=args.symbol)

    # Compare
    report = compare_chains(real_df, synth_df, spot, args.symbol)
    print_calibration_report(report, spot, vix, dte, args.symbol)


if __name__ == "__main__":
    main()
```

**Step 3: Run tests**

Run: `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m pytest tests/test_fo_calibrate.py tests/test_fo_chain_collector.py -v`

Expected: All 7 tests PASS (5 collector + 2 calibrate).

**Step 4: Run ALL tests**

Run: `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m pytest tests/test_fo_data.py tests/test_fo_strategies.py tests/test_fo_backtest.py tests/test_fo_param_sweep.py tests/test_fo_chain_collector.py tests/test_fo_calibrate.py -v`

Expected: All 72 tests PASS.
