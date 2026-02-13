"""
Backtest engine for the F&O screener.

Validates screener predictions against forward price data by:
1. Loading daily snapshots saved by screener.py
2. Resolving equity tokens and fetching forward candles via SmartAPI
3. Calculating forward returns at configurable horizons (1d, 3d, Nd)
4. Determining hit/miss (bullish+positive=hit, bearish+negative=hit)
5. Aggregating stats by direction, score tier, and signal type

Usage:
    python backtest.py                        # all snapshots
    python backtest.py --date 2026-02-12      # specific date
    python backtest.py --days 5               # forward horizon (default 5)
    python backtest.py --top 10               # top N per snapshot
    python backtest.py --no-cache             # re-fetch everything
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import config
from connect import get_session
from screener import SNAPSHOT_DIR, BULLISH_CATEGORIES, BEARISH_CATEGORIES, INDEX_SYMBOLS

CANDLE_DIR = Path(__file__).parent / "data" / "candles"
RESULTS_DIR = Path(__file__).parent / "data" / "backtest_results"


# ---------------------------------------------------------------------------
# Snapshot I/O
# ---------------------------------------------------------------------------

def list_snapshots() -> list[str]:
    """List all available snapshot dates (sorted ascending)."""
    if not SNAPSHOT_DIR.exists():
        return []
    dates = []
    for f in SNAPSHOT_DIR.glob("*.json"):
        dates.append(f.stem)  # YYYY-MM-DD
    return sorted(dates)


def load_snapshot(date_str: str) -> dict | None:
    """Load a snapshot by date string (YYYY-MM-DD)."""
    path = SNAPSHOT_DIR / f"{date_str}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Token Resolution
# ---------------------------------------------------------------------------

def resolve_and_cache_token(smart_api, snapshot: dict, symbol: str) -> str | None:
    """Resolve equity token via searchScrip and cache it in the snapshot.

    Writes the resolved token back into the snapshot JSON so subsequent
    runs skip the API call.
    """
    # Check if already cached in snapshot
    for cand in snapshot["candidates"]:
        if cand["symbol"] == symbol and cand.get("equity_token"):
            return cand["equity_token"]

    # Resolve via API
    try:
        search_resp = smart_api.searchScrip("NSE", symbol)
        if not search_resp or not search_resp.get("data"):
            return None

        token = None
        for match in search_resp["data"]:
            if match.get("tradingsymbol", "").endswith("-EQ"):
                token = match.get("symboltoken")
                break
        if not token:
            token = search_resp["data"][0].get("symboltoken")

        if token:
            # Cache back into snapshot
            for cand in snapshot["candidates"]:
                if cand["symbol"] == symbol:
                    cand["equity_token"] = token
                    break
            # Persist to disk
            path = SNAPSHOT_DIR / f"{snapshot['date']}.json"
            path.write_text(json.dumps(snapshot, indent=2, default=str))

        return token
    except Exception as e:
        print(f"    Token resolution failed for {symbol}: {e}")
        return None


# ---------------------------------------------------------------------------
# Candle Fetching & Caching
# ---------------------------------------------------------------------------

def fetch_forward_candles(smart_api, symbol: str, token: str,
                          from_date: str, trading_days: int) -> list | None:
    """Fetch daily candles from from_date forward for enough trading days.

    Requests extra calendar days (2x + 2) to cover weekends and holidays.
    Returns raw candle list [[timestamp, O, H, L, C, V], ...] or None.
    """
    calendar_days = trading_days * 2 + 2
    start = datetime.strptime(from_date, "%Y-%m-%d")
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
            return resp["data"]
    except Exception as e:
        print(f"    Candle fetch failed for {symbol}: {e}")
    return None


def load_or_fetch_candles(smart_api, symbol: str, token: str,
                          from_date: str, trading_days: int,
                          no_cache: bool = False) -> list | None:
    """Cache layer around candle fetching.

    Caches to data/candles/{SYMBOL}_{DATE}.json. Returns candle list or None.
    """
    CANDLE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CANDLE_DIR / f"{symbol}_{from_date}.json"

    if not no_cache and cache_file.exists():
        cached = json.loads(cache_file.read_text())
        # Check if cached data has enough trading days
        if len(cached) >= trading_days:
            return cached

    candles = fetch_forward_candles(smart_api, symbol, token, from_date, trading_days)
    if candles:
        cache_file.write_text(json.dumps(candles, default=str))
    return candles


# ---------------------------------------------------------------------------
# Return Calculation
# ---------------------------------------------------------------------------

def calculate_forward_returns(candles: list, trading_days: int) -> dict | None:
    """Calculate forward returns from candle data.

    Uses the first candle's open as entry price. Returns dict with
    entry_price and return percentages at 1d, 3d, and Nd horizons.
    Returns None if insufficient data.
    """
    if not candles or len(candles) < 2:
        return None

    entry_price = candles[0][1]  # Open of signal day
    if not entry_price or entry_price <= 0:
        return None

    result = {"entry_price": entry_price, "returns": {}}

    for horizon in [1, 3, trading_days]:
        if horizon > len(candles) - 1:
            continue
        # Use close of the horizon-th candle
        exit_price = candles[horizon][4]  # Close
        if exit_price and exit_price > 0:
            ret_pct = ((exit_price - entry_price) / entry_price) * 100
            result["returns"][f"{horizon}d"] = round(ret_pct, 2)

    # Deduplicate: if trading_days is 1 or 3, it's already covered
    return result if result["returns"] else None


def is_hit(direction: str, return_pct: float) -> bool:
    """Determine if a prediction was correct.

    Bullish + positive return = hit.
    Bearish + negative return = hit.
    Zero return = miss.
    """
    if direction == "bullish":
        return return_pct > 0
    elif direction == "bearish":
        return return_pct < 0
    return False


# ---------------------------------------------------------------------------
# Per-Snapshot Backtest
# ---------------------------------------------------------------------------

def backtest_snapshot(smart_api, snapshot: dict, trading_days: int = 5,
                      top_n: int | None = None,
                      no_cache: bool = False) -> list[dict]:
    """Orchestrate backtest for a single snapshot.

    For each candidate: resolve token -> fetch forward candles ->
    calculate returns -> determine hit/miss.

    Returns list of result dicts with symbol, direction, score,
    categories, returns, and hit/miss at each horizon.
    """
    candidates = snapshot["candidates"]
    if top_n:
        candidates = candidates[:top_n]

    results = []
    total = len(candidates)

    for i, cand in enumerate(candidates):
        symbol = cand["symbol"]

        # Skip indices
        if symbol in INDEX_SYMBOLS:
            continue

        print(f"  [{i+1}/{total}] {symbol}...", end=" ", flush=True)

        # Resolve token
        token = resolve_and_cache_token(smart_api, snapshot, symbol)
        if not token:
            print("skip (no token)")
            time.sleep(config.API_DELAY)
            continue

        time.sleep(config.API_DELAY)

        # Fetch forward candles (from the day AFTER the snapshot)
        snapshot_date = snapshot["date"]
        next_day = (datetime.strptime(snapshot_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        candles = load_or_fetch_candles(
            smart_api, symbol, token, next_day, trading_days, no_cache=no_cache
        )

        if not candles:
            print("skip (no candles)")
            time.sleep(config.API_DELAY)
            continue

        time.sleep(config.API_DELAY)

        # Calculate returns
        fwd = calculate_forward_returns(candles, trading_days)
        if not fwd:
            print("skip (insufficient data)")
            continue

        # Determine hit/miss at each horizon
        direction = cand["direction"]
        hits = {}
        for horizon_key, ret_pct in fwd["returns"].items():
            hits[horizon_key] = is_hit(direction, ret_pct)

        result = {
            "symbol": symbol,
            "direction": direction,
            "score": cand["score"],
            "categories": cand["categories"],
            "price_change_pct": cand.get("price_change_pct"),
            "snapshot_date": snapshot["date"],
            "entry_price": fwd["entry_price"],
            "returns": fwd["returns"],
            "hits": hits,
        }
        results.append(result)

        ret_str = " | ".join(f"{k}: {v:+.1f}%" for k, v in fwd["returns"].items())
        print(f"{ret_str}")

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_stats(all_results: list[dict], trading_days: int) -> dict:
    """Aggregate backtest results across snapshots.

    Returns stats dict with hit rates by direction, score tier,
    signal type, and top/worst performers at each horizon.
    """
    if not all_results:
        return {"total": 0}

    # Collect all unique horizons present
    horizons = set()
    for r in all_results:
        horizons.update(r["returns"].keys())
    horizons = sorted(horizons, key=lambda h: int(h.replace("d", "")))

    stats = {"total": len(all_results), "horizons": {}}

    for horizon in horizons:
        h_results = [r for r in all_results if horizon in r.get("hits", {})]
        if not h_results:
            continue

        total = len(h_results)
        hit_count = sum(1 for r in h_results if r["hits"][horizon])

        # By direction
        dir_stats = {}
        for direction in ["bullish", "bearish"]:
            dir_r = [r for r in h_results if r["direction"] == direction]
            if dir_r:
                dir_hits = sum(1 for r in dir_r if r["hits"][horizon])
                dir_stats[direction] = {
                    "total": len(dir_r),
                    "hits": dir_hits,
                    "rate": round(dir_hits / len(dir_r) * 100, 1),
                }

        # By score tier
        tier_stats = {}
        tiers = [(5, ">=5"), (3, ">=3"), (2, ">=2"), (0, "<2")]
        for threshold, label in tiers:
            if label.startswith(">="):
                tier_r = [r for r in h_results if r["score"] >= threshold]
            else:
                tier_r = [r for r in h_results if r["score"] < 2]
            if tier_r:
                tier_hits = sum(1 for r in tier_r if r["hits"][horizon])
                tier_stats[label] = {
                    "total": len(tier_r),
                    "hits": tier_hits,
                    "rate": round(tier_hits / len(tier_r) * 100, 1),
                }

        # By signal type
        signal_stats = {}
        for r in h_results:
            for cat in r["categories"]:
                if cat not in signal_stats:
                    signal_stats[cat] = {"total": 0, "hits": 0}
                signal_stats[cat]["total"] += 1
                if r["hits"][horizon]:
                    signal_stats[cat]["hits"] += 1
        for cat in signal_stats:
            s = signal_stats[cat]
            s["rate"] = round(s["hits"] / s["total"] * 100, 1) if s["total"] > 0 else 0

        # Top/worst performers
        sorted_by_return = sorted(
            h_results,
            key=lambda r: r["returns"].get(horizon, 0),
            reverse=True,
        )
        top_3 = sorted_by_return[:3]
        worst_3 = sorted_by_return[-3:]

        stats["horizons"][horizon] = {
            "total": total,
            "hits": hit_count,
            "rate": round(hit_count / total * 100, 1),
            "by_direction": dir_stats,
            "by_score_tier": tier_stats,
            "by_signal": signal_stats,
            "top_performers": [
                {"symbol": r["symbol"], "return": r["returns"][horizon],
                 "direction": r["direction"], "score": r["score"]}
                for r in top_3
            ],
            "worst_performers": [
                {"symbol": r["symbol"], "return": r["returns"][horizon],
                 "direction": r["direction"], "score": r["score"]}
                for r in worst_3
            ],
        }

    return stats


# ---------------------------------------------------------------------------
# Report Output
# ---------------------------------------------------------------------------

def print_backtest_report(stats: dict, all_results: list[dict]) -> None:
    """Print formatted backtest report to terminal."""
    border = "=" * 60
    sep = "-" * 60

    print(f"\n{border}")
    print(f"  SCREENER BACKTEST REPORT")
    print(f"  {datetime.now().strftime('%d %b %Y, %H:%M')}")
    print(f"  {stats['total']} candidates evaluated")
    print(border)

    if stats["total"] == 0:
        print("\n  No results to report. Candidates may lack forward data")
        print("  (same-day backtest or market holidays).\n")
        return

    # Snapshot dates covered
    dates = sorted(set(r["snapshot_date"] for r in all_results))
    print(f"\n  Snapshots: {', '.join(dates)}")

    for horizon, h_stats in stats.get("horizons", {}).items():
        print(f"\n{sep}")
        print(f"  {horizon.upper()} FORWARD RETURNS")
        print(sep)

        print(f"\n  Overall Hit Rate: {h_stats['rate']}%  ({h_stats['hits']}/{h_stats['total']})")

        # By direction
        dir_parts = []
        for direction, ds in h_stats.get("by_direction", {}).items():
            dir_parts.append(f"{direction.upper()} {ds['rate']}% ({ds['hits']}/{ds['total']})")
        if dir_parts:
            print(f"  By Direction:  {' | '.join(dir_parts)}")

        # By score tier
        tier_parts = []
        for tier, ts in h_stats.get("by_score_tier", {}).items():
            tier_parts.append(f"{tier}: {ts['rate']}%")
        if tier_parts:
            print(f"  By Score Tier: {' | '.join(tier_parts)}")

        # By signal type
        sig_parts = []
        for sig, ss in sorted(h_stats.get("by_signal", {}).items(),
                               key=lambda x: x[1]["rate"], reverse=True):
            sig_parts.append(f"{sig} {ss['rate']}% ({ss['hits']}/{ss['total']})")
        if sig_parts:
            print(f"  By Signal:")
            for part in sig_parts:
                print(f"    {part}")

        # Top performers
        top = h_stats.get("top_performers", [])
        if top:
            print(f"\n  Top Performers:")
            for p in top:
                print(f"    {p['symbol']:<15} {p['return']:+.1f}%  [{p['direction']}] score={p['score']}")

        # Worst performers
        worst = h_stats.get("worst_performers", [])
        if worst:
            print(f"  Worst Performers:")
            for p in worst:
                print(f"    {p['symbol']:<15} {p['return']:+.1f}%  [{p['direction']}] score={p['score']}")

    print(f"\n{border}\n")


def save_backtest_results(stats: dict, all_results: list[dict]) -> Path:
    """Save backtest results to data/backtest_results/backtest_YYYYMMDD_HHMMSS.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"backtest_{timestamp}.json"

    output = {
        "timestamp": datetime.now().isoformat(),
        "stats": stats,
        "results": all_results,
    }
    path.write_text(json.dumps(output, indent=2, default=str))
    print(f"Results saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Signal Win Rates
# ---------------------------------------------------------------------------

WIN_RATE_FILE = RESULTS_DIR / "win_rates.json"
MIN_SAMPLES_FOR_WIN_RATE = 3


def compute_signal_win_rates(all_results: list[dict], horizon: str = "5d") -> dict:
    """
    Group backtest results by (sorted categories + direction) and compute win rates.

    Only includes combos with at least MIN_SAMPLES_FOR_WIN_RATE samples.
    Returns {combo_key: win_rate} where combo_key = "LongBuildUp+PercOIGainers|bullish".
    """
    combos = defaultdict(lambda: {"hits": 0, "total": 0})

    for r in all_results:
        if horizon not in r.get("hits", {}):
            continue

        key = "+".join(sorted(r["categories"])) + "|" + r["direction"]
        combos[key]["total"] += 1
        if r["hits"][horizon]:
            combos[key]["hits"] += 1

    win_rates = {}
    for key, counts in combos.items():
        if counts["total"] >= MIN_SAMPLES_FOR_WIN_RATE:
            win_rates[key] = round(counts["hits"] / counts["total"], 3)

    return win_rates


def save_win_rates(win_rates: dict) -> Path:
    """Write win rates to data/backtest_results/win_rates.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WIN_RATE_FILE.write_text(json.dumps(win_rates, indent=2, sort_keys=True))
    print(f"Win rates saved: {WIN_RATE_FILE} ({len(win_rates)} combos)")
    return WIN_RATE_FILE


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Backtest screener predictions against forward price data")
    parser.add_argument("--date", type=str, help="Specific snapshot date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=5, help="Forward horizon in trading days (default: 5)")
    parser.add_argument("--top", type=int, default=None, help="Top N candidates per snapshot (default: all)")
    parser.add_argument("--no-cache", action="store_true", help="Re-fetch all candle data")
    args = parser.parse_args()

    # Determine which snapshots to process
    if args.date:
        dates = [args.date]
    else:
        dates = list_snapshots()

    if not dates:
        print("No snapshots found. Run the screener first: python screener.py --raw")
        sys.exit(1)

    print(f"Backtesting {len(dates)} snapshot(s) with {args.days}-day forward horizon...")
    print(f"Connecting to AngelOne SmartAPI...")
    smart_api = get_session()

    all_results = []
    for date_str in dates:
        snapshot = load_snapshot(date_str)
        if not snapshot:
            print(f"\nSnapshot {date_str}: not found, skipping")
            continue

        n_cand = len(snapshot["candidates"])
        top_label = f"top {args.top}" if args.top else "all"
        print(f"\nSnapshot {date_str}: {n_cand} candidates ({top_label})")

        results = backtest_snapshot(
            smart_api, snapshot,
            trading_days=args.days,
            top_n=args.top,
            no_cache=args.no_cache,
        )
        all_results.extend(results)

    # Aggregate and report
    stats = aggregate_stats(all_results, args.days)
    print_backtest_report(stats, all_results)
    save_backtest_results(stats, all_results)

    # Compute and save signal win rates for screener scoring
    horizon_key = f"{args.days}d"
    win_rates = compute_signal_win_rates(all_results, horizon=horizon_key)
    if win_rates:
        save_win_rates(win_rates)
    else:
        print(f"No win rates computed (need >= {MIN_SAMPLES_FOR_WIN_RATE} samples per combo)")


if __name__ == "__main__":
    main()
