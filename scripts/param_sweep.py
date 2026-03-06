"""Parameter sweep for backtest exit optimization.

Tests 384 parameter combinations across Nifty 50 stocks (1-year data)
to find optimal trailing stop and exit parameters.

Usage:
    source venv/bin/activate
    python scripts/param_sweep.py
"""

import itertools
import json
import math
import sys
import time
from pathlib import Path

# Add parent dir to path so we can import backtest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest import BacktestEngine, DEFAULT_PARAMS, NIFTY50_SYMBOLS, fetch_data


# Parameter grid
ATR_TARGET_MULTS = [1.5, 2.0, 2.5, 3.0]
ATR_SL_MULTS = [1.0, 1.5, 2.0]
TRAILING_MULTS = [1.0, 1.5, 2.0, 2.5]
TRAILING_ACTIVATION_PCTS = [2.0, 3.0]
MAX_HOLD_DAYS = [5, 7, 10, 15]
# time_sl fixed to True


def run_sweep(data: dict) -> list[dict]:
    """Run all parameter combinations and return results."""
    combos = list(itertools.product(
        ATR_TARGET_MULTS,
        ATR_SL_MULTS,
        TRAILING_MULTS,
        TRAILING_ACTIVATION_PCTS,
        MAX_HOLD_DAYS,
    ))
    total = len(combos)
    print(f"Total combinations: {total}")

    results = []
    t0 = time.time()

    for idx, (target, sl, trail, act_pct, hold) in enumerate(combos, 1):
        if idx % 20 == 0 or idx == 1:
            elapsed = time.time() - t0
            eta = (elapsed / idx) * (total - idx) if idx > 1 else 0
            print(f"Running combo {idx}/{total}... "
                  f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")

        params = {
            **DEFAULT_PARAMS,
            "atr_target_mult": target,
            "atr_sl_mult": sl,
            "trailing_mult": trail,
            "trailing_activation_pct": act_pct,
            "max_hold_days": hold,
            "time_sl_enabled": True,
        }

        engine = BacktestEngine(params)
        engine.run_multi(data)
        stats = engine.compute_stats()

        if stats["total_trades"] == 0:
            continue

        # Exit reason percentages
        reasons = stats.get("exit_reasons", {})
        n = stats["total_trades"]
        target_pct = reasons.get("target", 0) / n * 100
        sl_pct = reasons.get("stoploss", 0) / n * 100
        trail_pct = reasons.get("trailing_stop", 0) / n * 100
        expiry_pct = reasons.get("expiry", 0) / n * 100

        results.append({
            "atr_target_mult": target,
            "atr_sl_mult": sl,
            "trailing_mult": trail,
            "trailing_activation_pct": act_pct,
            "max_hold_days": hold,
            "total_trades": stats["total_trades"],
            "win_rate": stats["win_rate"],
            "total_pnl": stats["total_pnl"],
            "sharpe_ratio": stats["sharpe_ratio"],
            "profit_factor": stats["profit_factor"],
            "max_drawdown": stats["max_drawdown"],
            "avg_win_pct": stats["avg_win_pct"],
            "avg_loss_pct": stats["avg_loss_pct"],
            "avg_holding_days": stats["avg_holding_days"],
            "target_hit_pct": round(target_pct, 1),
            "stoploss_hit_pct": round(sl_pct, 1),
            "trailing_stop_pct": round(trail_pct, 1),
            "expiry_pct": round(expiry_pct, 1),
        })

    elapsed = time.time() - t0
    print(f"\nCompleted {total} combos in {elapsed:.1f}s "
          f"({elapsed/total:.2f}s per combo)")
    return results


def print_table(results: list[dict], title: str, top_n: int = 20,
                min_win_rate: float = 30.0, sort_key: str = "profit_factor"):
    """Print a ranked results table."""
    # Filter and sort
    filtered = [r for r in results
                if r["win_rate"] >= min_win_rate
                and r["profit_factor"] != float("inf")]
    filtered.sort(key=lambda r: r[sort_key], reverse=True)
    filtered = filtered[:top_n]

    if not filtered:
        print(f"\n{title}: No results with win_rate >= {min_win_rate}%")
        return

    print(f"\n{'=' * 130}")
    print(f"  {title} (top {top_n}, win_rate >= {min_win_rate}%)")
    print(f"{'=' * 130}")

    header = (f"{'Rank':>4} | {'Target':>6} | {'SL':>5} | {'Trail':>5} | "
              f"{'Act%':>5} | {'Hold':>4} | {'WinRate':>7} | {'P&L':>10} | "
              f"{'Sharpe':>6} | {'PF':>6} | {'MaxDD':>8} | "
              f"{'TgtHit%':>7} | {'TrailEx%':>8} | {'Expiry%':>7}")
    print(header)
    print("-" * 130)

    for rank, r in enumerate(filtered, 1):
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "inf"
        print(f"{rank:>4} | {r['atr_target_mult']:>6.1f} | {r['atr_sl_mult']:>5.1f} | "
              f"{r['trailing_mult']:>5.1f} | {r['trailing_activation_pct']:>5.1f} | "
              f"{r['max_hold_days']:>4} | {r['win_rate']:>6.1f}% | "
              f"{r['total_pnl']:>+10,.0f} | {r['sharpe_ratio']:>6.2f} | "
              f"{pf_str:>6} | {r['max_drawdown']:>8,.0f} | "
              f"{r['target_hit_pct']:>6.1f}% | {r['trailing_stop_pct']:>7.1f}% | "
              f"{r['expiry_pct']:>6.1f}%")


def print_recommendations(results: list[dict]):
    """Print analysis and recommendations."""
    print(f"\n{'=' * 80}")
    print("  RECOMMENDATIONS")
    print(f"{'=' * 80}")

    valid = [r for r in results
             if r["profit_factor"] != float("inf") and r["total_trades"] >= 50]

    # Best overall: highest profit factor with win_rate > 35%
    best_overall = [r for r in valid if r["win_rate"] >= 35.0]
    if best_overall:
        best_overall.sort(key=lambda r: r["profit_factor"], reverse=True)
        b = best_overall[0]
        print(f"\n  BEST OVERALL (highest PF, win_rate > 35%):")
        print(f"    Target={b['atr_target_mult']}x  SL={b['atr_sl_mult']}x  "
              f"Trail={b['trailing_mult']}x  Activation={b['trailing_activation_pct']}%  "
              f"Hold={b['max_hold_days']}d")
        print(f"    Win Rate: {b['win_rate']:.1f}%  |  P&L: {b['total_pnl']:+,.0f}  |  "
              f"PF: {b['profit_factor']:.2f}  |  Sharpe: {b['sharpe_ratio']:.2f}")
        print(f"    Exit mix: Target {b['target_hit_pct']:.0f}%  Trail {b['trailing_stop_pct']:.0f}%  "
              f"SL {b['stoploss_hit_pct']:.0f}%  Expiry {b['expiry_pct']:.0f}%")

    # Best conservative: lowest max drawdown with PF > 1.0
    conservative = [r for r in valid if r["profit_factor"] > 1.0]
    if conservative:
        conservative.sort(key=lambda r: r["max_drawdown"])
        c = conservative[0]
        print(f"\n  BEST CONSERVATIVE (lowest max drawdown, PF > 1.0):")
        print(f"    Target={c['atr_target_mult']}x  SL={c['atr_sl_mult']}x  "
              f"Trail={c['trailing_mult']}x  Activation={c['trailing_activation_pct']}%  "
              f"Hold={c['max_hold_days']}d")
        print(f"    Win Rate: {c['win_rate']:.1f}%  |  P&L: {c['total_pnl']:+,.0f}  |  "
              f"PF: {c['profit_factor']:.2f}  |  MaxDD: {c['max_drawdown']:,.0f}")

    # Best P&L
    by_pnl = sorted(valid, key=lambda r: r["total_pnl"], reverse=True)
    if by_pnl:
        p = by_pnl[0]
        print(f"\n  HIGHEST P&L:")
        print(f"    Target={p['atr_target_mult']}x  SL={p['atr_sl_mult']}x  "
              f"Trail={p['trailing_mult']}x  Activation={p['trailing_activation_pct']}%  "
              f"Hold={p['max_hold_days']}d")
        print(f"    Win Rate: {p['win_rate']:.1f}%  |  P&L: {p['total_pnl']:+,.0f}  |  "
              f"PF: {p['profit_factor']:.2f}  |  Sharpe: {p['sharpe_ratio']:.2f}")

    # Current defaults comparison
    current = [r for r in results
               if r["atr_target_mult"] == 2.5
               and r["atr_sl_mult"] == 1.5
               and r["trailing_mult"] == 1.0
               and r["trailing_activation_pct"] == 2.0
               and r["max_hold_days"] == 7]
    if current:
        d = current[0]
        print(f"\n  CURRENT DEFAULTS (for comparison):")
        print(f"    Target=2.5x  SL=1.5x  Trail=1.0x  Activation=2.0%  Hold=7d")
        print(f"    Win Rate: {d['win_rate']:.1f}%  |  P&L: {d['total_pnl']:+,.0f}  |  "
              f"PF: {d['profit_factor']:.2f}  |  Sharpe: {d['sharpe_ratio']:.2f}")
        print(f"    Exit mix: Target {d['target_hit_pct']:.0f}%  Trail {d['trailing_stop_pct']:.0f}%  "
              f"SL {d['stoploss_hit_pct']:.0f}%  Expiry {d['expiry_pct']:.0f}%")

        if best_overall:
            b = best_overall[0]
            pnl_delta = b["total_pnl"] - d["total_pnl"]
            wr_delta = b["win_rate"] - d["win_rate"]
            pf_delta = b["profit_factor"] - d["profit_factor"]
            print(f"\n  IMPROVEMENT (best overall vs current):")
            print(f"    Win Rate: {wr_delta:+.1f}pp  |  P&L: {pnl_delta:+,.0f}  |  "
                  f"PF: {pf_delta:+.2f}")

    print(f"\n{'=' * 80}\n")


def main():
    print("=" * 60)
    print("  PARAMETER SWEEP - Trailing Stop & Exit Optimization")
    print("=" * 60)
    print(f"\nGrid: {len(ATR_TARGET_MULTS)} targets x {len(ATR_SL_MULTS)} SLs x "
          f"{len(TRAILING_MULTS)} trails x {len(TRAILING_ACTIVATION_PCTS)} activations x "
          f"{len(MAX_HOLD_DAYS)} hold periods")

    # Fetch data ONCE
    print(f"\nFetching 1-year data for {len(NIFTY50_SYMBOLS)} Nifty 50 symbols...")
    data = fetch_data(NIFTY50_SYMBOLS, "1y")
    print(f"Got data for {len(data)} symbols ({len(NIFTY50_SYMBOLS) - len(data)} failed/skipped)")

    if not data:
        print("ERROR: No data fetched. Check network connection.")
        sys.exit(1)

    # Run sweep
    results = run_sweep(data)

    if not results:
        print("ERROR: No valid results produced.")
        sys.exit(1)

    # Print tables
    print_table(results, "TOP 20 BY PROFIT FACTOR", top_n=20,
                min_win_rate=30.0, sort_key="profit_factor")

    print_table(results, "TOP 20 BY SHARPE RATIO", top_n=20,
                min_win_rate=30.0, sort_key="sharpe_ratio")

    print_table(results, "TOP 20 BY TOTAL P&L", top_n=20,
                min_win_rate=25.0, sort_key="total_pnl")

    # Recommendations
    print_recommendations(results)

    # Save full results
    out_path = Path(__file__).resolve().parent.parent / "data" / "param_sweep_results.json"
    out_path.parent.mkdir(exist_ok=True)

    # Sort by profit factor for the saved file
    results.sort(key=lambda r: r["profit_factor"]
                 if r["profit_factor"] != float("inf") else 0, reverse=True)

    # Convert inf to string for JSON serialization
    for r in results:
        if r["profit_factor"] == float("inf"):
            r["profit_factor"] = "inf"

    out_path.write_text(json.dumps(results, indent=2))
    print(f"Full results saved to {out_path}")


if __name__ == "__main__":
    main()
