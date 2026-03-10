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
        "sharpe": stats.get("sharpe", 0) if stats.get("total_trades", 0) > 0 else float("-inf"),
        "profit_factor": stats.get("profit_factor", 0),
        "total_pnl": stats.get("total_pnl", 0),
        "max_drawdown": stats.get("max_drawdown", 0),
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
            f"{r['win_rate']:>5.1f}% {r['trades']:>7} "
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

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
