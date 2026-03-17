"""
Signal Collector — captures full screener output for backtesting.

Runs the screener in raw mode and saves enriched signal data including
candidates that were filtered out (with rejection reasons) for analysis.

Designed to run daily at 9:30 AM IST via Pi cron.

Usage:
    python collect_signals.py              # collect and save
    python collect_signals.py --list       # list saved collections
"""

import argparse
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config
from connect import get_session
from screener import (
    fetch_all_signals, score_signals, ALL_SIGNAL_NAMES,
    SNAPSHOT_DIR,
)
from paper_trade import (
    fetch_market_regime, fetch_daily_candles, resolve_token,
    ATR_PERIOD, IST,
)
from indicators import compute_atr

COLLECT_DIR = Path(__file__).parent / "data" / "signal_collections"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("signal_collector")


def collect_signals(smart_api) -> dict:
    """Run screener and capture comprehensive signal data."""
    today = datetime.now(IST).strftime("%Y-%m-%d")
    timestamp = datetime.now(IST).isoformat()

    # Market regime
    regime_info = fetch_market_regime(smart_api)

    # Phase 1: Fetch all signals
    log.info("Fetching F&O signals...")
    all_signals = fetch_all_signals(smart_api)
    total = sum(len(v) for v in all_signals.values())

    if total == 0:
        log.warning("No signals returned. Market may be closed.")
        return {"date": today, "error": "no_signals"}

    # Phase 2: Score
    log.info("Scoring %d raw signals...", total)
    candidates = score_signals(all_signals)

    # Phase 3: Enrich top candidates with ATR (for backtest simulation)
    log.info("Enriching top 10 candidates with ATR...")
    for cand in candidates[:10]:
        symbol = cand["symbol"]
        resolved = resolve_token(smart_api, symbol)
        if not resolved:
            cand["token"] = None
            cand["atr"] = None
            continue
        time.sleep(config.API_DELAY)

        trading_symbol, token = resolved
        cand["token"] = token

        candles = fetch_daily_candles(smart_api, token)
        time.sleep(config.API_DELAY)
        if candles:
            cand["atr"] = compute_atr(candles, ATR_PERIOD)
            # Also capture last close price for reference
            cand["last_close"] = candles[-1][4] if candles else None
        else:
            cand["atr"] = None

    # Build collection
    collection = {
        "date": today,
        "timestamp": timestamp,
        "market_regime": {
            "regime": regime_info.get("regime"),
            "nifty_ltp": regime_info.get("nifty_ltp"),
            "nifty_sma": regime_info.get("nifty_sma"),
            "vix": regime_info.get("vix"),
        },
        "signal_counts": {dt: len(all_signals.get(dt, [])) for dt in ALL_SIGNAL_NAMES},
        "total_candidates": len(candidates),
        "candidates": [
            {k: v for k, v in c.items() if k not in ("details", "candles")}
            for c in candidates
        ],
    }

    return collection


def save_collection(collection: dict) -> Path:
    """Save signal collection to data/signal_collections/YYYY-MM-DD.json."""
    COLLECT_DIR.mkdir(parents=True, exist_ok=True)
    path = COLLECT_DIR / f"{collection['date']}.json"
    path.write_text(json.dumps(collection, indent=2, default=str))
    log.info("Collection saved: %s (%d candidates)", path, collection.get("total_candidates", 0))
    return path


def list_collections():
    """List all saved collections."""
    if not COLLECT_DIR.exists():
        print("No collections yet.")
        return

    files = sorted(COLLECT_DIR.glob("*.json"))
    if not files:
        print("No collections yet.")
        return

    print(f"\n{'Date':12s} {'Candidates':>12s} {'Regime':>10s}")
    print("-" * 40)
    for f in files:
        data = json.loads(f.read_text())
        regime = data.get("market_regime", {}).get("regime", "?")
        total = data.get("total_candidates", 0)
        print(f"  {data['date']:12s} {total:>10d} {regime:>10s}")
    print(f"\nTotal: {len(files)} collections")


def main():
    parser = argparse.ArgumentParser(description="Collect screener signals for backtesting")
    parser.add_argument("--list", action="store_true", help="List saved collections")
    args = parser.parse_args()

    if args.list:
        list_collections()
        return

    log.info("Connecting to AngelOne SmartAPI...")
    smart_api = get_session()

    collection = collect_signals(smart_api)
    if "error" not in collection:
        save_collection(collection)
        log.info("Done. %d candidates captured with market regime: %s",
                 collection["total_candidates"],
                 collection["market_regime"]["regime"])
    else:
        log.warning("Collection failed: %s", collection.get("error"))


if __name__ == "__main__":
    main()
