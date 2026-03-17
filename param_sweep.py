#!/usr/bin/env python3
"""Systematic parameter sweep for signal-filtered backtest.

Precomputes expensive data once, then sweeps all param combos fast.
"""

import functools
import itertools
import json
import math
import sys
import time

# Force unbuffered output
print = functools.partial(print, flush=True)
from datetime import datetime

import pandas as pd
import yfinance as yf

from backtest import DEFAULT_PARAMS, fetch_data, NIFTY50_SYMBOLS, calc_round_trip_costs, EQ_SLIPPAGE_PCT
from backtest_signals import SignalBacktestEngine

# ── Parameter grid ──────────────────────────────────────────────────────────
PARAM_GRID = {
    "atr_target_mult": [1.5, 2.0, 2.5, 3.0, 3.5],
    "atr_sl_mult": [1.5, 2.0, 2.5, 3.0, 3.5],
    "max_hold_days": [5, 7, 10, 15, 20, 25],
    "trailing_activation_pct": [3.0],       # fixed — known good
    "trailing_tight_mult": [2.5],           # fixed — known good
    "min_score": [3.5, 4.0, 4.5, 5.0],
    "rs_top_n": [0, 10, 15, 20, 25],  # 0 = no RS filter
}


def precompute(data, nifty_df):
    """Precompute RS returns and Nifty bullish map once."""
    # Nifty EMA50 regime
    nifty_bullish = {}
    if nifty_df is not None and not nifty_df.empty:
        nifty_close = nifty_df["Close"]
        if isinstance(nifty_close, pd.DataFrame):
            nifty_close = nifty_close.iloc[:, 0]
        nifty_ema50 = nifty_close.ewm(span=50, adjust=False).mean()
        for idx in nifty_df.index:
            d = str(idx.date()) if hasattr(idx, "date") else str(idx)
            nifty_bullish[d] = float(nifty_close.loc[idx]) > float(nifty_ema50.loc[idx])

    # RS 20-day returns per symbol per date
    rs_returns = {}
    for symbol, df in data.items():
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        ret20 = close.pct_change(20)
        rs_returns[symbol] = {}
        for idx in df.index:
            d = str(idx.date()) if hasattr(idx, "date") else str(idx)
            v = ret20.loc[idx]
            if not pd.isna(v):
                rs_returns[symbol][d] = float(v)

    return nifty_bullish, rs_returns


def run_sweep_combo(data, nifty_bullish, rs_returns, combo):
    """Run one backtest with precomputed data. Returns stats or None."""
    params = {
        "atr_period": 14,
        "atr_target_mult": combo["atr_target_mult"],
        "atr_sl_mult": combo["atr_sl_mult"],
        "max_hold_days": combo["max_hold_days"],
        "trailing_activation_pct": combo["trailing_activation_pct"],
        "trailing_tight_mult": combo["trailing_tight_mult"],
        "trailing_mult": combo["trailing_tight_mult"],
        "capital_per_trade": 10000,
        "time_sl_enabled": True,
        "time_sl_half_mult": 0.75,
        "time_sl_three_quarter_mult": 0.5,
        "rs_rank_filter": False,  # we'll apply manually
        "stock_wr_filter": False,
    }

    rs_top_n = combo["rs_top_n"]
    min_score = combo["min_score"]

    engine = SignalBacktestEngine(params, min_score=min_score)

    # Run per-symbol backtests (this is the core compute)
    all_trades = []
    for symbol, df in data.items():
        trades = engine.run(symbol, df)
        # Regime filter
        if nifty_bullish:
            trades = [t for t in trades if nifty_bullish.get(t["entry_date"], True)]
        all_trades.extend(trades)

    # RS rank filter (using precomputed data)
    if rs_top_n > 0 and rs_returns:
        all_trades.sort(key=lambda t: t["entry_date"])
        filtered = []
        for t in all_trades:
            d = t["entry_date"]
            sym = t["symbol"]
            day_rs = {s: rs_returns.get(s, {}).get(d, float("-inf")) for s in data}
            ranked = sorted(day_rs, key=day_rs.get, reverse=True)[:rs_top_n]
            if sym in ranked:
                filtered.append(t)
        all_trades = filtered

    if len(all_trades) < 30:
        return None

    all_trades.sort(key=lambda t: t["entry_date"])
    engine.trades = all_trades

    stats = engine.compute_stats()

    return {
        **combo,
        "trades": stats["total_trades"],
        "win_rate": round(stats["win_rate"], 1),
        "pnl": round(stats["total_pnl"]),
        "costs": round(stats["total_costs"]),
        "gross": round(stats["total_pnl"] + stats["total_costs"]),
        "pf": stats["profit_factor"],
        "sharpe": stats["sharpe_ratio"],
        "max_dd": round(stats["max_drawdown"]),
        "avg_hold": round(stats["avg_holding_days"], 1),
    }


def main():
    period = sys.argv[1] if len(sys.argv) > 1 else "1y"

    print(f"Fetching Nifty 50 data (period={period})...")
    data = fetch_data(NIFTY50_SYMBOLS, period)
    print(f"Got data for {len(data)} symbols")

    print("Fetching Nifty index for regime filter...")
    nifty_df = yf.download("^NSEI", period=period, progress=False)
    print(f"Nifty data: {len(nifty_df)} rows")

    print("Precomputing RS returns and regime map...")
    nifty_bullish, rs_returns = precompute(data, nifty_df)

    # Generate all combinations
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    total = len(combos)
    print(f"\nSweeping {total:,} parameter combinations...\n")

    results = []
    start = time.time()
    for i, combo in enumerate(combos):
        if (i + 1) % 1000 == 0 or i == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:>6}/{total}] {rate:.0f}/sec  ETA {eta/60:.1f}min  ({len(results)} valid so far)")

        result = run_sweep_combo(data, nifty_bullish, rs_returns, combo)
        if result:
            results.append(result)

    elapsed = time.time() - start
    print(f"\nDone: {total:,} combos in {elapsed/60:.1f}min ({len(results)} valid)\n")

    # Sort by profit factor
    results.sort(key=lambda r: r["pf"], reverse=True)

    # Print top 40
    print("=" * 130)
    print("  TOP 40 PARAMETER COMBINATIONS (by Profit Factor, min 30 trades)")
    print("=" * 130)
    header = (
        f"{'#':>3}  {'Tgt':>4} {'SL':>4} {'Hold':>4} {'TrAct':>5} "
        f"{'TrMlt':>5} {'MinS':>4} {'RS':>3} | "
        f"{'Trades':>6} {'WR%':>5} {'PnL':>9} {'Gross':>8} {'PF':>5} "
        f"{'Sharpe':>6} {'MaxDD':>7} {'AvgH':>5}"
    )
    print(header)
    print("-" * 130)

    for i, r in enumerate(results[:40]):
        rs_label = f"{r['rs_top_n']}" if r["rs_top_n"] > 0 else "-"
        line = (
            f"{i+1:>3}  {r['atr_target_mult']:>4.1f} {r['atr_sl_mult']:>4.1f} "
            f"{r['max_hold_days']:>4} {r['trailing_activation_pct']:>5.1f} "
            f"{r['trailing_tight_mult']:>5.1f} {r['min_score']:>4.1f} {rs_label:>3} | "
            f"{r['trades']:>6} {r['win_rate']:>4.1f}% {r['pnl']:>+9,} "
            f"{r['gross']:>+8,} {r['pf']:>5.2f} {r['sharpe']:>+6.2f} "
            f"{r['max_dd']:>7,} {r['avg_hold']:>5.1f}"
        )
        print(line)

    # Print bottom 5
    print(f"\n  BOTTOM 5:")
    print("-" * 130)
    for r in results[-5:]:
        rs_label = f"{r['rs_top_n']}" if r["rs_top_n"] > 0 else "-"
        line = (
            f"     {r['atr_target_mult']:>4.1f} {r['atr_sl_mult']:>4.1f} "
            f"{r['max_hold_days']:>4} {r['trailing_activation_pct']:>5.1f} "
            f"{r['trailing_tight_mult']:>5.1f} {r['min_score']:>4.1f} {rs_label:>3} | "
            f"{r['trades']:>6} {r['win_rate']:>4.1f}% {r['pnl']:>+9,} "
            f"{r['gross']:>+8,} {r['pf']:>5.2f} {r['sharpe']:>+6.2f} "
            f"{r['max_dd']:>7,} {r['avg_hold']:>5.1f}"
        )
        print(line)

    # Summary
    profitable = [r for r in results if r["pf"] >= 1.0]
    robust = [r for r in results if r["pf"] >= 1.1 and r["trades"] >= 100]
    print(f"\n{'='*130}")
    print(f"  SUMMARY")
    print(f"{'='*130}")
    print(f"  Total valid combos:         {len(results):,}")
    print(f"  Profitable (PF >= 1.0):     {len(profitable):,} ({100*len(profitable)/max(len(results),1):.1f}%)")
    print(f"  Robust (PF >= 1.1, 100+ t): {len(robust):,}")
    if profitable:
        best = profitable[0]
        print(f"  Best PF:  {best['pf']:.2f} (Tgt={best['atr_target_mult']}, SL={best['atr_sl_mult']}, "
              f"Hold={best['max_hold_days']}, TrAct={best['trailing_activation_pct']}, "
              f"TrMlt={best['trailing_tight_mult']}, MinS={best['min_score']}, RS={best['rs_top_n'] or 'off'})")
        best_pnl = max(profitable, key=lambda r: r["pnl"])
        print(f"  Best PnL: {best_pnl['pnl']:+,} (Tgt={best_pnl['atr_target_mult']}, SL={best_pnl['atr_sl_mult']}, "
              f"Hold={best_pnl['max_hold_days']}, MinS={best_pnl['min_score']}, RS={best_pnl['rs_top_n'] or 'off'})")

    # Param frequency analysis in top 50 profitable
    if profitable:
        top_n = profitable[:min(50, len(profitable))]
        print(f"\n  MOST COMMON PARAMS IN TOP {len(top_n)} PROFITABLE:")
        for key in PARAM_GRID:
            from collections import Counter
            counts = Counter(r[key] for r in top_n)
            top_vals = counts.most_common(3)
            vals_str = ", ".join(f"{v}({c})" for v, c in top_vals)
            print(f"    {key:<25} {vals_str}")

    # Save
    out_path = f"data/param_sweep_{period}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
