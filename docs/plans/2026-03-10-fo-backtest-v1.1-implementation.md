# F&O Backtest V1.1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add volatility skew to synthetic option pricing and build a parameter sweep tool for optimizing strategy configurations.

**Architecture:** Skew formula added to `generate_synthetic_chain()` in fo_data.py (data layer only — strategies/engine unchanged). Param sweep is a new standalone script `fo_param_sweep.py` that orchestrates multiple backtest runs with varied strategy params.

**Tech Stack:** Python, existing fo_data/fo_strategies/fo_backtest modules, itertools.product for grid generation.

---

### Task 1: Volatility Skew — Tests

**Files:**
- Modify: `tests/test_fo_data.py`

**Context:** Currently `generate_synthetic_chain()` in `fo_data.py:155-193` uses flat IV (`iv = vix / 100.0`) for all strikes. We need tests that verify strike-dependent IV.

**Step 1: Write the failing tests**

Add to `tests/test_fo_data.py` inside `TestSyntheticChain`:

```python
def test_skew_otm_put_iv_higher_than_atm(self):
    """OTM puts should have higher IV than ATM due to put skew."""
    from fo_data import generate_synthetic_chain
    chain = generate_synthetic_chain(spot=23000.0, vix=15.0, dte=30, symbol="NIFTY")
    atm_strike = 23000.0
    otm_put_strike = 22500.0  # 500pt OTM

    atm_row = chain[(chain["strike"] == atm_strike) & (chain["option_type"] == "PE")]
    otm_row = chain[(chain["strike"] == otm_put_strike) & (chain["option_type"] == "PE")]

    assert not atm_row.empty and not otm_row.empty
    assert otm_row.iloc[0]["iv"] > atm_row.iloc[0]["iv"], \
        f"OTM put IV {otm_row.iloc[0]['iv']:.4f} should exceed ATM IV {atm_row.iloc[0]['iv']:.4f}"

def test_skew_otm_call_iv_mildly_higher(self):
    """OTM calls should have mildly higher IV (smile), less than put skew."""
    from fo_data import generate_synthetic_chain
    chain = generate_synthetic_chain(spot=23000.0, vix=15.0, dte=30, symbol="NIFTY")
    atm_strike = 23000.0
    otm_call_strike = 23500.0
    otm_put_strike = 22500.0

    atm_pe = chain[(chain["strike"] == atm_strike) & (chain["option_type"] == "PE")]
    otm_ce = chain[(chain["strike"] == otm_call_strike) & (chain["option_type"] == "CE")]
    otm_pe = chain[(chain["strike"] == otm_put_strike) & (chain["option_type"] == "PE")]

    assert not otm_ce.empty
    call_iv_bump = otm_ce.iloc[0]["iv"] - atm_pe.iloc[0]["iv"]
    put_iv_bump = otm_pe.iloc[0]["iv"] - atm_pe.iloc[0]["iv"]

    assert call_iv_bump > 0, "OTM call IV should be slightly above ATM"
    assert call_iv_bump < put_iv_bump, \
        f"Call IV bump ({call_iv_bump:.4f}) should be less than put IV bump ({put_iv_bump:.4f})"

def test_skew_factor_zero_gives_flat_iv(self):
    """With skew_factor=0, all strikes should have same IV (backward compat)."""
    from fo_data import generate_synthetic_chain
    chain = generate_synthetic_chain(spot=23000.0, vix=15.0, dte=30, symbol="NIFTY", skew_factor=0.0)
    ivs = chain["iv"].unique()
    assert len(ivs) == 1, f"Expected flat IV with skew_factor=0, got {len(ivs)} unique IVs"
    assert abs(ivs[0] - 0.15) < 0.001

def test_skew_increases_with_distance(self):
    """Further OTM puts should have progressively higher IV."""
    from fo_data import generate_synthetic_chain
    chain = generate_synthetic_chain(spot=23000.0, vix=15.0, dte=30, symbol="NIFTY")
    pe_chain = chain[chain["option_type"] == "PE"].sort_values("strike", ascending=False)

    # Get IVs for strikes below ATM (OTM puts)
    otm_puts = pe_chain[pe_chain["strike"] < 23000.0].head(5)
    if len(otm_puts) >= 2:
        ivs = otm_puts["iv"].tolist()
        # IVs should be increasing as strike decreases (further OTM = higher IV)
        for i in range(len(ivs) - 1):
            assert ivs[i] <= ivs[i + 1], \
                f"IV should increase for further OTM puts: {ivs[i]:.4f} vs {ivs[i+1]:.4f}"
```

**Step 2: Run tests to verify they fail**

Run: `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m pytest tests/test_fo_data.py::TestSyntheticChain::test_skew_otm_put_iv_higher_than_atm tests/test_fo_data.py::TestSyntheticChain::test_skew_otm_call_iv_mildly_higher tests/test_fo_data.py::TestSyntheticChain::test_skew_factor_zero_gives_flat_iv tests/test_fo_data.py::TestSyntheticChain::test_skew_increases_with_distance -v`

Expected: 3 FAIL (skew tests), 1 PASS or FAIL depending on whether `skew_factor` param exists yet. The flat IV tests should fail because all IVs are currently identical.

---

### Task 2: Volatility Skew — Implementation

**Files:**
- Modify: `fo_data.py:155-193` (the `generate_synthetic_chain` function)

**Context:** The function currently uses `iv = vix / 100.0` on line 168 for all strikes. We need to compute per-strike IV based on moneyness.

**Step 1: Add skew_factor parameter and per-strike IV computation**

Modify `generate_synthetic_chain()` in `fo_data.py`:

```python
# Change the function signature to add skew_factor
def generate_synthetic_chain(
    spot: float,
    vix: float,
    dte: int,
    symbol: str = "NIFTY",
    risk_free: float = 0.065,
    min_premium: float = 5.0,
    skew_factor: float = 0.8,
) -> pd.DataFrame:
```

Replace the current loop body (lines 168-191) with:

```python
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
```

**Step 2: Run all tests**

Run: `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m pytest tests/test_fo_data.py tests/test_fo_strategies.py tests/test_fo_backtest.py -v`

Expected: All tests PASS including the 4 new skew tests. Existing tests should still pass because `skew_factor=0.8` is the default and doesn't break the existing test expectations (they check for positive premiums, correct option types, etc. — not specific IV values).

**Important:** If `test_condor_entry_normal_vix` fails, it means the skew changed premiums enough to push condor max_risk negative again. If so, the condor OTM/wing params (currently 500pt OTM, 300pt wings in fo_strategies.py) may need adjustment, OR pass `skew_factor` through the engine. Check and fix.

**Step 3: Commit**

```bash
git add fo_data.py tests/test_fo_data.py
git commit -m "feat: add volatility skew to synthetic option chain pricing"
```

---

### Task 3: Skew Validation — Run Backtest & Compare

**Files:**
- No new files — just run the backtest and verify condor P&L is more realistic

**Step 1: Run the full 2-year backtest with skew**

Run: `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 fo_backtest.py --symbol NIFTY --period 2y --capital 1000000 --output data/fo_backtest/results_v1.1_skew_20260310.json`

**Step 2: Compare V1.0 vs V1.1 condor results**

V1.0 baseline (from previous run):
- Condor: 139 trades, 81.3% win rate, ₹38,11,439 P&L (inflated)
- Overall Sharpe: 1.54

V1.1 with skew should show:
- Condor P&L significantly lower (ideally under ₹5L on ₹10L capital)
- Fewer condor entries (higher OTM put premiums → higher net credit → some may still breach wing width guard)
- Other strategies largely unchanged (futures/momentum use spot, not options chain IV)

**Step 3: If condor P&L is still unrealistic (>₹5L)**

Check the specific trades: are condor net_credits still too high? Options:
- Increase `skew_factor` to 1.2 or 1.5
- Increase `CONDOR_OTM_POINTS_NIFTY` further (currently 500)
- Both

**Step 4: Commit results**

```bash
git add data/fo_backtest/results_v1.1_skew_20260310.json
git commit -m "data: backtest results with volatility skew (V1.1)"
```

---

### Task 4: Parameter Sweep — Tests

**Files:**
- Create: `tests/test_fo_param_sweep.py`

**Step 1: Write sweep smoke tests**

```python
"""Tests for fo_param_sweep — parameter grid search."""
import datetime
import json
import os
import sys
import unittest

import numpy as np
import pandas as pd


def _make_short_data(bars: int = 60) -> pd.DataFrame:
    """Create minimal OHLCV+VIX data for fast sweep tests."""
    dates = pd.bdate_range(end=datetime.date(2025, 6, 30), periods=bars)
    rng = np.random.RandomState(42)
    close = 23000 + rng.randn(bars).cumsum() * 50
    return pd.DataFrame({
        "Date": dates,
        "Open": close - rng.rand(bars) * 30,
        "High": close + rng.rand(bars) * 50,
        "Low": close - rng.rand(bars) * 50,
        "Close": close,
        "Volume": rng.randint(100000, 500000, bars),
        "vix": 14 + rng.randn(bars) * 2,
    }).set_index("Date")


class TestParamSweep(unittest.TestCase):

    def test_build_grid_futures(self):
        from fo_param_sweep import build_param_grid
        grid = build_param_grid("futures")
        assert len(grid) == 27  # 3x3x3
        assert all("score_threshold" in g for g in grid)
        assert all("target_atr_mult" in g for g in grid)
        assert all("sl_atr_mult" in g for g in grid)

    def test_build_grid_all_strategies(self):
        from fo_param_sweep import build_param_grid, SWEEP_GRIDS
        for strategy_name in SWEEP_GRIDS:
            grid = build_param_grid(strategy_name)
            assert len(grid) == 27, f"{strategy_name} grid should have 27 combos, got {len(grid)}"

    def test_sweep_single_combo(self):
        """Sweep with 1 combo should return results dict."""
        from fo_param_sweep import run_single_sweep
        data = _make_short_data(60)
        result = run_single_sweep(
            data=data,
            strategy_name="futures",
            params={"score_threshold": 3.5, "target_atr_mult": 1.5, "sl_atr_mult": 3.5},
            symbol="NIFTY",
            capital=1_000_000,
        )
        assert "params" in result
        assert "trades" in result
        assert "sharpe" in result
        assert "win_rate" in result
        assert "total_pnl" in result

    def test_sweep_strategy_returns_ranked(self):
        """Full sweep of one strategy returns results sorted by Sharpe."""
        from fo_param_sweep import sweep_strategy
        data = _make_short_data(60)
        results = sweep_strategy(
            data=data,
            strategy_name="futures",
            symbol="NIFTY",
            capital=1_000_000,
        )
        assert len(results) == 27
        # Should be sorted by Sharpe descending
        sharpes = [r["sharpe"] for r in results]
        assert sharpes == sorted(sharpes, reverse=True)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run tests to verify they fail**

Run: `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m pytest tests/test_fo_param_sweep.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'fo_param_sweep'`

---

### Task 5: Parameter Sweep — Implementation

**Files:**
- Create: `fo_param_sweep.py`

**Context:** The `FOBacktestEngine` in `fo_backtest.py` accepts strategy instances via `self.strategies` dict (line 63-68). Each strategy class accepts params via `__init__`. The sweep needs to create a fresh engine per combo, inject one custom strategy, disable the others, and collect stats.

**Step 1: Write fo_param_sweep.py**

```python
"""F&O Parameter Sweep — grid search over strategy params.

Usage:
    python fo_param_sweep.py --strategy futures --symbol NIFTY --period 2y
    python fo_param_sweep.py --strategy all --symbol NIFTY --period 2y
"""

import argparse
import datetime
import itertools
import json
import sys

import pandas as pd

from fo_backtest import FOBacktestEngine
from fo_data import fetch_spot_vix_history
from fo_strategies import (
    CondorStrategy,
    FuturesStrategy,
    MomentumStrategy,
    SpreadStrategy,
)

# ---------------------------------------------------------------------------
# Sweep grids: param_name -> list of values to try
# ---------------------------------------------------------------------------
SWEEP_GRIDS = {
    "futures": {
        "score_threshold": [3.0, 3.5, 4.0],
        "target_atr_mult": [1.0, 1.5, 2.0],
        "sl_atr_mult": [2.5, 3.5, 4.5],
    },
    "spread": {
        "score_threshold": [3.0, 3.5, 4.0],
        "profit_cap_pct": [0.6, 0.8, 1.0],
        "sl_multiplier": [1.5, 2.0, 3.0],
    },
    "condor": {
        "min_vix": [10, 12, 14],
        "target_pct": [0.3, 0.5, 0.7],
        "sl_multiplier": [1.5, 2.0, 3.0],
    },
    "momentum": {
        "score_threshold": [4.0, 5.0, 6.0],
        "sl_pct": [0.25, 0.35, 0.50],
        "target_pct": [0.6, 0.9, 1.2],
    },
}

STRATEGY_CLASSES = {
    "futures": FuturesStrategy,
    "spread": SpreadStrategy,
    "condor": CondorStrategy,
    "momentum": MomentumStrategy,
}


def build_param_grid(strategy_name: str) -> list[dict]:
    """Build all param combinations for a strategy."""
    grid = SWEEP_GRIDS[strategy_name]
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


class _NullStrategy:
    """Strategy that never enters — used to disable strategies during sweep."""
    name = "null"
    def should_enter(self, **kwargs):
        return None
    def should_exit(self, **kwargs):
        return False, ""


def run_single_sweep(
    data: pd.DataFrame,
    strategy_name: str,
    params: dict,
    symbol: str = "NIFTY",
    capital: float = 1_000_000,
) -> dict:
    """Run one backtest with a specific strategy+params, others disabled."""
    engine = FOBacktestEngine(capital=capital)

    # Disable all strategies, then enable only the target
    for name in engine.strategies:
        engine.strategies[name] = _NullStrategy()

    strategy_cls = STRATEGY_CLASSES[strategy_name]
    engine.strategies[strategy_name] = strategy_cls(**params)

    results = engine.run(data.copy(), symbol=symbol)
    stats = results["stats"]

    return {
        "params": params,
        "trades": stats.get("total_trades", 0),
        "win_rate": stats.get("win_rate", 0),
        "sharpe": stats.get("sharpe_ratio", 0),
        "profit_factor": stats.get("profit_factor", 0),
        "total_pnl": stats.get("total_pnl", 0),
        "max_drawdown": stats.get("max_drawdown_pct", 0),
    }


def sweep_strategy(
    data: pd.DataFrame,
    strategy_name: str,
    symbol: str = "NIFTY",
    capital: float = 1_000_000,
) -> list[dict]:
    """Run full grid sweep for one strategy, return results sorted by Sharpe."""
    grid = build_param_grid(strategy_name)
    results = []

    for i, params in enumerate(grid):
        result = run_single_sweep(data, strategy_name, params, symbol, capital)
        results.append(result)
        # Progress indicator
        sys.stdout.write(f"\r  {strategy_name}: {i+1}/{len(grid)} combos")
        sys.stdout.flush()

    print()  # newline after progress

    # Sort by Sharpe descending
    results.sort(key=lambda r: r["sharpe"], reverse=True)
    return results


def print_top_results(strategy_name: str, results: list[dict], top_n: int = 3):
    """Print top N results for a strategy."""
    print(f"\n  Top {top_n} for {strategy_name}:")
    print(f"  {'Rank':<5} {'Sharpe':>8} {'PF':>6} {'WinR':>6} {'Trades':>7} {'P&L':>12} {'Params'}")
    print(f"  {'-'*5} {'-'*8} {'-'*6} {'-'*6} {'-'*7} {'-'*12} {'-'*30}")

    for i, r in enumerate(results[:top_n]):
        params_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        print(
            f"  {i+1:<5} {r['sharpe']:>8.2f} {r['profit_factor']:>6.2f} "
            f"{r['win_rate']:>5.1%} {r['trades']:>7} "
            f"Rs.{r['total_pnl']:>10,.0f}  {params_str}"
        )


def main():
    parser = argparse.ArgumentParser(description="F&O Parameter Sweep")
    parser.add_argument("--strategy", required=True, choices=list(SWEEP_GRIDS.keys()) + ["all"])
    parser.add_argument("--symbol", default="NIFTY")
    parser.add_argument("--period", default="2y")
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"Fetching {args.symbol} data for {args.period}...")
    data = fetch_spot_vix_history(args.symbol, args.period)

    strategies = list(SWEEP_GRIDS.keys()) if args.strategy == "all" else [args.strategy]

    all_results = {}
    print(f"\nRunning parameter sweep on {len(data)} bars...")
    print("=" * 60)

    for strat_name in strategies:
        results = sweep_strategy(data, strat_name, args.symbol, args.capital)
        all_results[strat_name] = {
            "total_combos": len(results),
            "top_5": results[:5],
            "all_results": results,
        }
        print_top_results(strat_name, results)

    print("\n" + "=" * 60)

    # Save results
    output_path = args.output or f"data/fo_backtest/sweep_results_{datetime.date.today().strftime('%Y%m%d')}.json"

    # Strip all_results for JSON output (keep top_5 only)
    save_data = {
        "sweep_date": str(datetime.date.today()),
        "symbol": args.symbol,
        "period": args.period,
        "capital": args.capital,
        "bars": len(data),
        "strategies": {
            name: {"total_combos": d["total_combos"], "top_5": d["top_5"]}
            for name, d in all_results.items()
        },
    }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
```

**Step 2: Run tests**

Run: `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m pytest tests/test_fo_param_sweep.py -v`

Expected: All 4 tests PASS.

**Step 3: Commit**

```bash
git add fo_param_sweep.py tests/test_fo_param_sweep.py
git commit -m "feat: add parameter sweep tool for F&O strategy optimization"
```

---

### Task 6: Run Full Parameter Sweep & Save Results

**Files:**
- Output: `data/fo_backtest/sweep_results_20260310.json`

**Step 1: Run sweep for all strategies**

Run: `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 fo_param_sweep.py --strategy all --symbol NIFTY --period 2y --capital 1000000`

This runs 108 backtest iterations (27 per strategy × 4 strategies). Should take 2-5 minutes.

**Step 2: Review output**

Check the top 3 per strategy. Key things to look for:
- Futures: Which score threshold works best? Higher ATR target or lower?
- Spread: Does wider DTE range help? What profit cap % is optimal?
- Condor: Lower VIX floor (10) vs conservative (14)? Aggressive vs conservative target?
- Momentum: Is 5.0 score threshold optimal or too restrictive?

**Step 3: Commit results**

```bash
git add data/fo_backtest/sweep_results_20260310.json
git commit -m "data: parameter sweep results for all F&O strategies"
```

**Step 4: Run all tests one final time**

Run: `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m pytest tests/test_fo_data.py tests/test_fo_strategies.py tests/test_fo_backtest.py tests/test_fo_param_sweep.py -v`

Expected: All tests PASS (56 existing + 4 skew + 4 sweep = 64 total).
