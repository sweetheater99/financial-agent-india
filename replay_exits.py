"""
Exit Parameter Replay — simulates different exit strategies on existing trades.

Uses cached candle data from the backtest engine to replay trades under
different ATR multipliers, max hold days, and trailing stop activation thresholds.

Usage:
    python replay_exits.py                    # replay all closed + open trades
    python replay_exits.py --grid             # grid search exit parameters
"""

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import defaultdict

import config
from connect import get_session
from paper_trade import (
    PORTFOLIO_FILE, IST, resolve_token, fetch_daily_candles,
    ATR_PERIOD, calc_round_trip_costs,
)
from indicators import compute_atr

CANDLE_DIR = Path(__file__).parent / "data" / "candles"


def load_trades() -> list[dict]:
    """Load all trades (open + closed) from portfolio."""
    portfolio = json.loads(PORTFOLIO_FILE.read_text())
    trades = []

    for t in portfolio.get("closed_trades", []):
        trades.append({**t, "status": "closed"})
    for p in portfolio.get("positions", []):
        trades.append({**p, "status": "open"})

    return trades


def fetch_and_cache_candles(smart_api, symbol: str, token: str,
                            entry_date: str, days: int = 20) -> list:
    """Fetch candles from entry_date forward, with caching."""
    CANDLE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CANDLE_DIR / f"{symbol}_{entry_date}_replay.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text())

    calendar_days = days * 2 + 5
    start = datetime.strptime(entry_date, "%Y-%m-%d")
    end = start + timedelta(days=calendar_days)

    try:
        resp = smart_api.getCandleData({
            "exchange": "NSE",
            "symboltoken": token,
            "interval": "ONE_DAY",
            "fromdate": start.strftime("%Y-%m-%d 09:15"),
            "todate": end.strftime("%Y-%m-%d 15:30"),
        })
        if resp and resp.get("data"):
            cache_file.write_text(json.dumps(resp["data"], default=str))
            return resp["data"]
    except Exception as e:
        print(f"  Candle fetch failed for {symbol}: {e}")
    return []


def simulate_exit(candles: list, entry_price: float, direction: str,
                  atr: float, target_mult: float, sl_mult: float,
                  max_hold: int, trail_activation_atr: float,
                  trail_mult: float) -> dict:
    """Simulate exit using given parameters on daily candle data.

    Returns dict with exit_day, exit_price, exit_reason, pnl_pct.
    """
    if not candles or not atr or atr <= 0:
        return {"exit_day": 0, "exit_reason": "no_data", "pnl_pct": 0}

    if direction == "bullish":
        target_price = entry_price + target_mult * atr
        fixed_sl = entry_price - sl_mult * atr
    else:
        target_price = entry_price - target_mult * atr
        fixed_sl = entry_price + sl_mult * atr

    peak = entry_price
    trail_active = False
    trail_activation_price = entry_price + trail_activation_atr * atr if direction == "bullish" else entry_price - trail_activation_atr * atr

    trading_day = 0
    for candle in candles:
        # candle: [timestamp, open, high, low, close, volume]
        high = candle[2]
        low = candle[3]
        close = candle[4]
        trading_day += 1

        # Update peak
        if direction == "bullish":
            peak = max(peak, high)
            # Check trail activation
            if not trail_active and peak >= trail_activation_price:
                trail_active = True
        else:
            peak = min(peak, low)
            if not trail_active and peak <= trail_activation_price:
                trail_active = True

        # Check target
        if direction == "bullish" and high >= target_price:
            pnl_pct = (target_price - entry_price) / entry_price * 100
            return {"exit_day": trading_day, "exit_reason": "target",
                    "exit_price": target_price, "pnl_pct": round(pnl_pct, 2)}
        elif direction == "bearish" and low <= target_price:
            pnl_pct = (entry_price - target_price) / entry_price * 100
            return {"exit_day": trading_day, "exit_reason": "target",
                    "exit_price": target_price, "pnl_pct": round(pnl_pct, 2)}

        # Check stop loss (trailing or fixed)
        if trail_active:
            if direction == "bullish":
                trailing_sl = peak - trail_mult * atr
                effective_sl = max(trailing_sl, fixed_sl)
                if low <= effective_sl:
                    exit_price = effective_sl
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                    reason = "trailing_stop" if trailing_sl > fixed_sl else "stoploss"
                    return {"exit_day": trading_day, "exit_reason": reason,
                            "exit_price": round(exit_price, 2), "pnl_pct": round(pnl_pct, 2)}
            else:
                trailing_sl = peak + trail_mult * atr
                effective_sl = min(trailing_sl, fixed_sl)
                if high >= effective_sl:
                    exit_price = effective_sl
                    pnl_pct = (entry_price - exit_price) / entry_price * 100
                    reason = "trailing_stop" if trailing_sl < fixed_sl else "stoploss"
                    return {"exit_day": trading_day, "exit_reason": reason,
                            "exit_price": round(exit_price, 2), "pnl_pct": round(pnl_pct, 2)}
        else:
            # Only fixed SL before trail activates
            if direction == "bullish" and low <= fixed_sl:
                pnl_pct = (fixed_sl - entry_price) / entry_price * 100
                return {"exit_day": trading_day, "exit_reason": "stoploss",
                        "exit_price": round(fixed_sl, 2), "pnl_pct": round(pnl_pct, 2)}
            elif direction == "bearish" and high >= fixed_sl:
                pnl_pct = (entry_price - fixed_sl) / entry_price * 100
                return {"exit_day": trading_day, "exit_reason": "stoploss",
                        "exit_price": round(fixed_sl, 2), "pnl_pct": round(pnl_pct, 2)}

        # Check max hold
        if trading_day >= max_hold:
            if direction == "bullish":
                pnl_pct = (close - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - close) / entry_price * 100
            return {"exit_day": trading_day, "exit_reason": "expiry",
                    "exit_price": round(close, 2), "pnl_pct": round(pnl_pct, 2)}

    # Ran out of candles (still holding)
    last_close = candles[-1][4] if candles else entry_price
    if direction == "bullish":
        pnl_pct = (last_close - entry_price) / entry_price * 100
    else:
        pnl_pct = (entry_price - last_close) / entry_price * 100
    return {"exit_day": trading_day, "exit_reason": "holding",
            "exit_price": round(last_close, 2), "pnl_pct": round(pnl_pct, 2)}


def replay_trades(smart_api, trades: list, params: dict) -> list[dict]:
    """Replay all trades with given exit parameters."""
    results = []

    for t in trades:
        symbol = t["symbol"]
        if t.get("instrument") == "OPT":
            continue  # skip options for now

        entry_price = t["entry_price"]
        direction = t.get("direction", "bullish")
        entry_date = t["entry_date"]

        # Get token
        token = t.get("token")
        if not token:
            resolved = resolve_token(smart_api, symbol)
            if not resolved:
                continue
            _, token = resolved
            time.sleep(config.API_DELAY)

        # Get candles
        candles = fetch_and_cache_candles(smart_api, symbol, token, entry_date)
        time.sleep(config.API_DELAY)
        if not candles:
            continue

        # Get ATR from existing data or compute
        atr = t.get("atr_at_entry")
        if not atr:
            # Compute from candles before entry
            pre_candles = fetch_daily_candles(smart_api, token, days=25)
            time.sleep(config.API_DELAY)
            if pre_candles:
                atr = compute_atr(pre_candles, ATR_PERIOD)

        if not atr:
            continue

        # Skip the first candle (entry day, price already set)
        forward_candles = candles[1:] if len(candles) > 1 else candles

        sim = simulate_exit(
            forward_candles, entry_price, direction, atr,
            target_mult=params["target_mult"],
            sl_mult=params["sl_mult"],
            max_hold=params["max_hold"],
            trail_activation_atr=params["trail_activation_atr"],
            trail_mult=params["trail_mult"],
        )

        results.append({
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "entry_date": entry_date,
            "atr": round(atr, 2),
            "actual_exit": t.get("exit_reason", "open"),
            "actual_pnl_pct": t.get("pnl_pct", None),
            **sim,
        })

    return results


def grid_search(smart_api, trades: list) -> list[dict]:
    """Grid search exit parameters and rank by expectancy."""
    param_grid = {
        "target_mult": [2.0, 2.5, 3.0, 3.5],
        "sl_mult": [1.0, 1.5, 2.0],
        "max_hold": [5, 7, 10, 12],
        "trail_activation_atr": [0.0, 0.5, 1.0, 1.5],
        "trail_mult": [0.75, 1.0, 1.5],
    }

    # Pre-fetch all candles once
    print("Pre-fetching candle data...")
    candle_cache = {}
    for t in trades:
        if t.get("instrument") == "OPT":
            continue
        symbol = t["symbol"]
        token = t.get("token")
        if not token:
            resolved = resolve_token(smart_api, symbol)
            if not resolved:
                continue
            _, token = resolved
            time.sleep(config.API_DELAY)
            t["token"] = token

        key = f"{symbol}_{t['entry_date']}"
        if key not in candle_cache:
            candles = fetch_and_cache_candles(smart_api, symbol, token, t["entry_date"])
            time.sleep(config.API_DELAY)
            candle_cache[key] = candles

        # ATR
        if not t.get("atr_at_entry"):
            pre = fetch_daily_candles(smart_api, token, days=25)
            time.sleep(config.API_DELAY)
            if pre:
                t["atr_at_entry"] = compute_atr(pre, ATR_PERIOD)

    print(f"Cached candles for {len(candle_cache)} trade entries")

    # Grid search (no API calls needed — all from cache)
    combos = []
    from itertools import product
    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)
    print(f"Testing {total_combos} parameter combinations...")

    for tm, sm, mh, ta, trl in product(
        param_grid["target_mult"],
        param_grid["sl_mult"],
        param_grid["max_hold"],
        param_grid["trail_activation_atr"],
        param_grid["trail_mult"],
    ):
        params = {
            "target_mult": tm, "sl_mult": sm, "max_hold": mh,
            "trail_activation_atr": ta, "trail_mult": trl,
        }

        wins = 0
        losses = 0
        total_pnl = 0.0

        for t in trades:
            if t.get("instrument") == "OPT":
                continue
            key = f"{t['symbol']}_{t['entry_date']}"
            candles = candle_cache.get(key, [])
            atr = t.get("atr_at_entry")
            if not candles or not atr:
                continue

            forward = candles[1:] if len(candles) > 1 else candles
            sim = simulate_exit(forward, t["entry_price"], t.get("direction", "bullish"),
                                atr, tm, sm, mh, ta, trl)

            if sim["pnl_pct"] > 0:
                wins += 1
            else:
                losses += 1
            total_pnl += sim["pnl_pct"]

        total = wins + losses
        if total > 0:
            combos.append({
                **params,
                "trades": total,
                "wins": wins,
                "win_rate": round(wins / total * 100, 1),
                "total_pnl_pct": round(total_pnl, 2),
                "avg_pnl_pct": round(total_pnl / total, 2),
            })

    # Sort by total P&L
    combos.sort(key=lambda c: c["total_pnl_pct"], reverse=True)
    return combos


def print_replay_report(results: list[dict], params: dict):
    """Print replay results vs actual outcomes."""
    border = "=" * 80
    print(f"\n{border}")
    print(f"  EXIT PARAMETER REPLAY")
    print(f"  Target: {params['target_mult']}x ATR | SL: {params['sl_mult']}x ATR | Max Hold: {params['max_hold']}d")
    print(f"  Trail Activation: {params['trail_activation_atr']}x ATR | Trail: {params['trail_mult']}x ATR")
    print(border)

    print(f"\n{'Symbol':12s} {'Dir':>7s} {'Actual':>10s} {'Simulated':>10s} {'Diff':>8s} {'SimExit':>12s} {'Days':>5s}")
    print("-" * 70)

    for r in results:
        actual = f"{r['actual_pnl_pct']:+.2f}%" if r['actual_pnl_pct'] is not None else "open"
        simulated = f"{r['pnl_pct']:+.2f}%"
        diff = ""
        if r['actual_pnl_pct'] is not None:
            d = r['pnl_pct'] - r['actual_pnl_pct']
            diff = f"{d:+.2f}%"
        print(f"  {r['symbol']:12s} {r['direction']:>7s} {actual:>10s} {simulated:>10s} {diff:>8s} {r['exit_reason']:>12s} {r['exit_day']:>5d}")

    # Summary
    sim_wins = sum(1 for r in results if r["pnl_pct"] > 0)
    sim_total_pnl = sum(r["pnl_pct"] for r in results)
    print(f"\n  Simulated: {sim_wins}/{len(results)} wins ({sim_wins/len(results)*100:.0f}%), total P&L: {sim_total_pnl:+.2f}%")

    actual_wins = sum(1 for r in results if r.get("actual_pnl_pct") is not None and r["actual_pnl_pct"] > 0)
    actual_total = sum(r["actual_pnl_pct"] for r in results if r.get("actual_pnl_pct") is not None)
    actual_count = sum(1 for r in results if r.get("actual_pnl_pct") is not None)
    if actual_count:
        print(f"  Actual:    {actual_wins}/{actual_count} wins ({actual_wins/actual_count*100:.0f}%), total P&L: {actual_total:+.2f}%")
    print()


def print_grid_report(combos: list[dict]):
    """Print top 20 parameter combos from grid search."""
    border = "=" * 90
    print(f"\n{border}")
    print(f"  GRID SEARCH RESULTS — TOP 20 PARAMETER COMBOS")
    print(border)

    print(f"\n{'Rank':>4s} {'Target':>7s} {'SL':>5s} {'Hold':>5s} {'TrailAct':>9s} {'Trail':>6s} {'WinRate':>8s} {'TotalP&L':>9s} {'Avg':>7s}")
    print("-" * 70)

    for i, c in enumerate(combos[:20]):
        print(f"  {i+1:>3d}  {c['target_mult']:>5.1f}x  {c['sl_mult']:>3.1f}x  {c['max_hold']:>4d}d  {c['trail_activation_atr']:>7.1f}x  {c['trail_mult']:>4.1f}x  {c['win_rate']:>6.1f}%  {c['total_pnl_pct']:>+8.2f}%  {c['avg_pnl_pct']:>+5.2f}%")

    # Current params for comparison
    print(f"\n  Current params: target=2.5x, SL=1.5x, hold=7d, trailAct=0.0x, trail=1.0x")
    current = [c for c in combos if c["target_mult"] == 2.5 and c["sl_mult"] == 1.5
               and c["max_hold"] == 7 and c["trail_activation_atr"] == 0.0 and c["trail_mult"] == 1.0]
    if current:
        c = current[0]
        rank = combos.index(c) + 1
        print(f"  Current rank: #{rank}/{len(combos)} | WinRate: {c['win_rate']}% | Total P&L: {c['total_pnl_pct']:+.2f}%")
    print()


def main():
    parser = argparse.ArgumentParser(description="Replay trades with different exit parameters")
    parser.add_argument("--grid", action="store_true", help="Grid search exit parameters")
    parser.add_argument("--target", type=float, default=2.5, help="ATR target multiplier")
    parser.add_argument("--sl", type=float, default=1.5, help="ATR stop-loss multiplier")
    parser.add_argument("--hold", type=int, default=7, help="Max hold days")
    parser.add_argument("--trail-act", type=float, default=0.0, help="Trail activation (ATR from entry)")
    parser.add_argument("--trail", type=float, default=1.0, help="Trail multiplier (ATR from peak)")
    args = parser.parse_args()

    print("Loading trades from portfolio...")
    trades = load_trades()
    eq_trades = [t for t in trades if t.get("instrument", "EQ") == "EQ"]
    print(f"Found {len(eq_trades)} equity trades ({len(trades)} total)")

    print("Connecting to AngelOne SmartAPI...")
    smart_api = get_session()

    if args.grid:
        combos = grid_search(smart_api, eq_trades)
        print_grid_report(combos)

        # Save results
        results_dir = Path(__file__).parent / "data" / "backtest_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        out = results_dir / f"grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out.write_text(json.dumps(combos[:50], indent=2))
        print(f"Top 50 combos saved: {out}")
    else:
        params = {
            "target_mult": args.target,
            "sl_mult": args.sl,
            "max_hold": args.hold,
            "trail_activation_atr": args.trail_act,
            "trail_mult": args.trail,
        }
        results = replay_trades(smart_api, eq_trades, params)
        print_replay_report(results, params)


if __name__ == "__main__":
    main()
