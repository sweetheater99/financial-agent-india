"""
Historical OI Pipeline — Phase 3 of financial agent improvements.

Downloads F&O bhavcopies from NSE archives (Jan 2022 - June 2024),
extracts daily OI per stock, reconstructs signal categories
(ShortCovering, LongBuildUp, etc.), and runs a 2-year backtest.

Usage:
    python oi_pipeline.py download    # download bhavcopies (run from Mac)
    python oi_pipeline.py build       # build daily OI database from CSVs
    python oi_pipeline.py signals     # reconstruct signal categories
    python oi_pipeline.py backtest    # run backtest with OI signals
    python oi_pipeline.py summary     # print signal performance summary
"""

import argparse
import csv
import io
import json
import os
import pickle
import sys
import time
import zipfile
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent / "data" / "oi_history"

# F&O stocks to track (top 84 by volume, same as longterm backtest)
FNO_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "WIPRO", "HCLTECH",
    "SUNPHARMA", "BAJFINANCE", "TATAMOTORS", "MARUTI", "NTPC", "ONGC",
    "POWERGRID", "TATASTEEL", "DRREDDY", "BAJAJFINSV", "CIPLA", "BPCL",
    "HINDALCO", "ULTRACEMCO", "NESTLEIND", "TECHM", "COALINDIA", "VEDL",
    "DIVISLAB", "APOLLOHOSP", "JSWSTEEL", "ADANIENT", "ADANIPORTS",
    "GRASIM", "EICHERMOT", "TATACONSUM", "IOC", "BRITANNIA", "GAIL",
    "BAJAJ-AUTO", "HEROMOTOCO", "M&M", "BIOCON", "INDUSINDBK", "DLF",
    "DABUR", "MARICO", "GODREJCP", "PEL", "NMDC", "SAIL", "AMBUJACEM",
    "ACC", "TATAPOWER", "IDEA", "PNB", "BANKBARODA", "FEDERALBNK",
    "MFSL", "CHOLAFIN", "MANAPPURAM", "MUTHOOTFIN", "LICHSGFIN", "RECLTD",
    "PFC", "SHRIRAMFIN", "PERSISTENT", "COFORGE", "MPHASIS", "TATAELXSI",
    "LTTS", "LTIM", "LUPIN", "AUROPHARMA", "TORNTPHARM", "GRANULES",
    "NATIONALUM", "JINDALSTEL", "ABB", "BHARATFORG",
]


def download_bhavcopies():
    """Download F&O bhavcopies from NSE archives (Jan 2022 - June 2024)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = DATA_DIR / "raw"
    raw_dir.mkdir(exist_ok=True)

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': '*/*',
        'Referer': 'https://www.nseindia.com/',
    })

    # Refresh cookies
    session.get('https://www.nseindia.com/', timeout=15)
    time.sleep(2)

    start = date(2022, 1, 3)
    end = date(2024, 6, 28)
    current = start
    downloaded = 0
    skipped = 0
    failed = 0

    while current <= end:
        # Skip weekends
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        date_str = current.strftime("%d%b%Y").upper()
        month_str = current.strftime("%b").upper()
        fname = f"fo{date_str}bhav.csv.zip"
        out_path = raw_dir / fname

        if out_path.exists():
            skipped += 1
            current += timedelta(days=1)
            continue

        url = f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{current.year}/{month_str}/{fname}"

        try:
            r = session.get(url, timeout=15)
            if r.status_code == 200 and len(r.content) > 1000:
                out_path.write_bytes(r.content)
                downloaded += 1
                if downloaded % 50 == 0:
                    print(f"  Downloaded {downloaded} files (at {current})")
            elif r.status_code == 403:
                # Cookie expired, refresh
                print(f"  Cookie refresh at {current}...")
                session.get('https://www.nseindia.com/', timeout=15)
                time.sleep(3)
                r = session.get(url, timeout=15)
                if r.status_code == 200 and len(r.content) > 1000:
                    out_path.write_bytes(r.content)
                    downloaded += 1
                else:
                    failed += 1
            else:
                # Likely a holiday
                failed += 1
        except Exception as e:
            print(f"  Error on {current}: {e}")
            failed += 1
            time.sleep(5)

        # Rate limit: ~1 req/sec
        time.sleep(1.0)
        current += timedelta(days=1)

    print(f"\nDownload complete: {downloaded} new, {skipped} already had, {failed} failed/holidays")


def build_oi_database():
    """Parse bhavcopies and build daily OI database per stock."""
    raw_dir = DATA_DIR / "raw"
    if not raw_dir.exists():
        print("No raw data. Run 'download' first.")
        return

    # Aggregate FUTSTK OI by symbol and date (sum across expiries)
    # Also get close price for signal reconstruction
    daily_data = defaultdict(dict)  # {symbol: {date_str: {oi, close, volume}}}

    zip_files = sorted(raw_dir.glob("fo*bhav.csv.zip"))
    print(f"Processing {len(zip_files)} bhavcopies...")

    for i, zpath in enumerate(zip_files):
        try:
            z = zipfile.ZipFile(zpath)
            csvname = z.namelist()[0]
            with z.open(csvname) as f:
                reader = csv.DictReader(io.TextIOWrapper(f))
                for row in reader:
                    if row.get('INSTRUMENT') != 'FUTSTK':
                        continue
                    symbol = row['SYMBOL']
                    if symbol not in FNO_STOCKS:
                        continue
                    
                    date_str = row['TIMESTAMP'].strip()
                    oi = int(row['OPEN_INT'])
                    close = float(row['CLOSE'])
                    contracts = int(row['CONTRACTS'])

                    if date_str not in daily_data[symbol]:
                        daily_data[symbol][date_str] = {'oi': 0, 'close': close, 'contracts': 0}
                    
                    # Sum OI across all expiries for the same date
                    daily_data[symbol][date_str]['oi'] += oi
                    daily_data[symbol][date_str]['contracts'] += contracts
                    # Use near-month close (first expiry has most liquidity)
                    # The first FUTSTK row for each symbol-date is the near month

        except Exception as e:
            print(f"  Error processing {zpath.name}: {e}")

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(zip_files)} files")

    # Save as pickle for fast loading
    db_path = DATA_DIR / "daily_oi.pkl"
    with open(db_path, 'wb') as f:
        pickle.dump(dict(daily_data), f)

    # Also save summary
    total_records = sum(len(v) for v in daily_data.values())
    symbols_with_data = len(daily_data)
    date_range = set()
    for sym_data in daily_data.values():
        date_range.update(sym_data.keys())
    
    print(f"\nOI database built:")
    print(f"  Symbols: {symbols_with_data}")
    print(f"  Total records: {total_records:,}")
    print(f"  Date range: {min(date_range)} to {max(date_range)}")
    print(f"  Saved to: {db_path}")


def reconstruct_signals():
    """Reconstruct daily signal categories from OI + price changes."""
    db_path = DATA_DIR / "daily_oi.pkl"
    if not db_path.exists():
        print("No OI database. Run 'build' first.")
        return

    with open(db_path, 'rb') as f:
        daily_data = pickle.load(f)

    # For each symbol, compute daily signals
    # Signal types based on price change + OI change:
    # LongBuildUp:   price UP   + OI UP    (new longs entering)
    # ShortCovering: price UP   + OI DOWN  (shorts buying back)
    # ShortBuildUp:  price DOWN + OI UP    (new shorts entering)
    # LongUnwinding: price DOWN + OI DOWN  (longs exiting)

    all_signals = []  # [{date, symbol, signal_type, price_change_pct, oi_change_pct, close, oi}]

    for symbol, dates_data in daily_data.items():
        # Sort dates chronologically
        sorted_dates = sorted(dates_data.keys(), key=lambda d: datetime.strptime(d, "%d-%b-%Y"))

        for i in range(1, len(sorted_dates)):
            prev_date = sorted_dates[i - 1]
            curr_date = sorted_dates[i]

            prev = dates_data[prev_date]
            curr = dates_data[curr_date]

            if prev['close'] == 0 or prev['oi'] == 0:
                continue

            price_change_pct = ((curr['close'] - prev['close']) / prev['close']) * 100
            oi_change_pct = ((curr['oi'] - prev['oi']) / prev['oi']) * 100

            # Classify signal (require meaningful moves)
            signal_type = None
            direction = None

            if price_change_pct > 0.5:  # price up
                if oi_change_pct > 1.0:
                    signal_type = "LongBuildUp"
                    direction = "bullish"
                elif oi_change_pct < -1.0:
                    signal_type = "ShortCovering"
                    direction = "bullish"
            elif price_change_pct < -0.5:  # price down
                if oi_change_pct > 1.0:
                    signal_type = "ShortBuildUp"
                    direction = "bearish"
                elif oi_change_pct < -1.0:
                    signal_type = "LongUnwinding"
                    direction = "bearish"

            if signal_type:
                # Also check if it would appear in PercPriceGainers/Losers
                price_signals = []
                if price_change_pct >= 1.5:
                    price_signals.append("PercPriceGainers")
                elif price_change_pct <= -1.5:
                    price_signals.append("PercPriceLosers")
                if abs(oi_change_pct) >= 3.0:
                    if oi_change_pct > 0:
                        price_signals.append("PercOIGainers")
                    else:
                        price_signals.append("PercOILosers")

                all_signals.append({
                    "date": curr_date,
                    "symbol": symbol,
                    "signal_type": signal_type,
                    "direction": direction,
                    "price_change_pct": round(price_change_pct, 2),
                    "oi_change_pct": round(oi_change_pct, 2),
                    "close": curr['close'],
                    "oi": curr['oi'],
                    "additional_signals": price_signals,
                })

    # Save signals
    signals_path = DATA_DIR / "reconstructed_signals.pkl"
    with open(signals_path, 'wb') as f:
        pickle.dump(all_signals, f)

    # Summary
    signal_counts = defaultdict(int)
    for s in all_signals:
        signal_counts[s['signal_type']] += 1

    print(f"Reconstructed signals:")
    print(f"  Total: {len(all_signals):,}")
    for st, count in sorted(signal_counts.items(), key=lambda x: -x[1]):
        print(f"  {st}: {count:,}")

    # Multi-signal analysis
    multi = [s for s in all_signals if len(s['additional_signals']) > 0]
    print(f"  With additional price/OI signals: {len(multi):,}")

    date_range = set(s['date'] for s in all_signals)
    print(f"  Date range: {min(date_range)} to {max(date_range)}")
    print(f"  Saved to: {signals_path}")


def run_backtest():
    """Backtest OI signals with ATR-based exits using yfinance price data."""
    signals_path = DATA_DIR / "reconstructed_signals.pkl"
    if not signals_path.exists():
        print("No signals. Run 'signals' first.")
        return

    with open(signals_path, 'rb') as f:
        all_signals = pickle.load(f)

    # Load price data from yfinance (cached)
    cache_path = DATA_DIR / "price_cache.pkl"
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            price_data = pickle.load(f)
        print(f"Loaded cached price data for {len(price_data)} stocks")
    else:
        print("Downloading price data from yfinance...")
        try:
            import yfinance as yf
        except ImportError:
            print("yfinance not installed. Run: pip install yfinance")
            return

        price_data = {}
        symbols = list(set(s['symbol'] for s in all_signals))
        for i, sym in enumerate(symbols):
            ticker = f"{sym}.NS"
            try:
                df = yf.download(ticker, start="2022-01-01", end="2024-07-31",
                                progress=False, timeout=10)
                if len(df) > 50:
                    price_data[sym] = df
            except Exception:
                pass
            if (i + 1) % 20 == 0:
                print(f"  Downloaded {i + 1}/{len(symbols)} stocks")

        with open(cache_path, 'wb') as f:
            pickle.dump(price_data, f)
        print(f"Cached price data for {len(price_data)} stocks")

    # ATR computation from price data
    def compute_atr_from_df(df, idx, period=14):
        if idx < period:
            return None
        trs = []
        for i in range(idx - period, idx):
            high = float(df.iloc[i]['High'])
            low = float(df.iloc[i]['Low'])
            prev_close = float(df.iloc[i - 1]['Close']) if i > 0 else high
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
        return sum(trs) / len(trs)

    # Exit parameters to test (reduced grid for speed)
    param_combos = [
        # (target_mult, sl_mult, max_hold, trail_act, trail_mult)
        (2.5, 1.5, 7, 0.0, 1.0),   # current params (pre-Phase 1)
        (2.5, 1.5, 7, 1.5, 1.5),   # Phase 1 params (trail activation)
        (3.5, 1.5, 10, 1.5, 1.5),  # 14-trade "optimal"
        (2.0, 1.0, 12, 1.5, 1.5),  # 2yr best with momentum
        (3.0, 1.5, 10, 1.5, 1.5),  # balanced
    ]

    # Run backtest for each signal type
    signal_types = ["ShortCovering", "LongBuildUp", "ShortBuildUp", "LongUnwinding"]

    # Filter: only bullish signals for now (ShortCovering, LongBuildUp)
    results = {}

    for sig_type in signal_types:
        sig_signals = [s for s in all_signals if s['signal_type'] == sig_type]
        if not sig_signals:
            continue

        direction = sig_signals[0]['direction']

        for params in param_combos:
            target_mult, sl_mult, max_hold, trail_act, trail_mult = params
            param_key = f"T{target_mult}x_SL{sl_mult}x_H{max_hold}_TA{trail_act}x_TR{trail_mult}x"

            wins = 0
            losses = 0
            total_pnl = 0.0
            trades = 0

            for sig in sig_signals:
                symbol = sig['symbol']
                if symbol not in price_data:
                    continue

                df = price_data[symbol]
                entry_date_str = sig['date']

                try:
                    entry_dt = datetime.strptime(entry_date_str, "%d-%b-%Y")
                except:
                    continue

                # Find entry date in dataframe
                entry_idx = None
                for idx in range(len(df)):
                    if df.index[idx].date() == entry_dt.date():
                        entry_idx = idx
                        break

                if entry_idx is None or entry_idx < 15:
                    continue

                entry_price = float(df.iloc[entry_idx]['Close'])
                atr = compute_atr_from_df(df, entry_idx)
                if atr is None or atr <= 0:
                    continue

                target = entry_price + target_mult * atr if direction == "bullish" else entry_price - target_mult * atr
                sl = entry_price - sl_mult * atr if direction == "bullish" else entry_price + sl_mult * atr
                peak = entry_price

                # Simulate exit
                exit_price = None
                for day in range(1, max_hold + 1):
                    check_idx = entry_idx + day
                    if check_idx >= len(df):
                        break

                    price = float(df.iloc[check_idx]['Close'])
                    high = float(df.iloc[check_idx]['High'])
                    low = float(df.iloc[check_idx]['Low'])

                    if direction == "bullish":
                        peak = max(peak, high)
                        if high >= target:
                            exit_price = target
                            break
                        trail_activated = (peak - entry_price) >= trail_act * atr
                        if trail_activated:
                            trailing_sl = peak - trail_mult * atr
                            effective_sl = max(trailing_sl, sl)
                        else:
                            effective_sl = sl
                        if low <= effective_sl:
                            exit_price = effective_sl
                            break
                    else:
                        peak = min(peak, low)
                        if low <= target:
                            exit_price = target
                            break
                        trail_activated = (entry_price - peak) >= trail_act * atr
                        if trail_activated:
                            trailing_sl = peak + trail_mult * atr
                            effective_sl = min(trailing_sl, sl)
                        else:
                            effective_sl = sl
                        if high >= effective_sl:
                            exit_price = effective_sl
                            break

                if exit_price is None:
                    # Max hold expiry
                    last_idx = min(entry_idx + max_hold, len(df) - 1)
                    exit_price = float(df.iloc[last_idx]['Close'])

                if direction == "bullish":
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                else:
                    pnl_pct = ((entry_price - exit_price) / entry_price) * 100

                total_pnl += pnl_pct
                trades += 1
                if pnl_pct > 0:
                    wins += 1
                else:
                    losses += 1

            if trades > 0:
                win_rate = wins / trades * 100
                avg_pnl = total_pnl / trades
                pf_wins = sum(1 for _ in range(wins))  # simplified
                key = f"{sig_type}|{param_key}"
                results[key] = {
                    "signal_type": sig_type,
                    "direction": direction,
                    "params": param_key,
                    "trades": trades,
                    "wins": wins,
                    "losses": losses,
                    "win_rate": round(win_rate, 1),
                    "total_pnl": round(total_pnl, 1),
                    "avg_pnl": round(avg_pnl, 3),
                }

    # Save results
    results_path = DATA_DIR / "oi_backtest_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*90}")
    print(f"  OI SIGNAL BACKTEST RESULTS (Jan 2022 - June 2024)")
    print(f"{'='*90}")
    print(f"{'Signal':<16} {'Params':<35} {'Trades':>6} {'WinRate':>8} {'TotalP&L':>10} {'Avg/Trade':>10}")
    print(f"{'-'*90}")

    for key in sorted(results.keys()):
        r = results[key]
        print(f"{r['signal_type']:<16} {r['params']:<35} {r['trades']:>6} {r['win_rate']:>7.1f}% {r['total_pnl']:>+9.1f}% {r['avg_pnl']:>+9.3f}%")

    print(f"\nResults saved to: {results_path}")


def print_summary():
    """Print a comparative summary of OI signals vs momentum."""
    results_path = DATA_DIR / "oi_backtest_results.json"
    if not results_path.exists():
        print("No backtest results. Run 'backtest' first.")
        return

    with open(results_path) as f:
        results = json.load(f)

    # Group by signal type, show best params for each
    by_signal = defaultdict(list)
    for key, r in results.items():
        by_signal[r['signal_type']].append(r)

    print(f"\n{'='*70}")
    print(f"  SIGNAL TYPE COMPARISON (best params for each)")
    print(f"{'='*70}")

    for sig_type in ["ShortCovering", "LongBuildUp", "ShortBuildUp", "LongUnwinding"]:
        if sig_type not in by_signal:
            continue
        entries = sorted(by_signal[sig_type], key=lambda x: -x['avg_pnl'])
        best = entries[0]
        worst = entries[-1]
        print(f"\n{sig_type} ({best['direction']}):")
        print(f"  Best:  {best['params']}")
        print(f"         Trades={best['trades']}, WinRate={best['win_rate']}%, Avg={best['avg_pnl']:+.3f}%")
        print(f"  Worst: {worst['params']}")
        print(f"         Trades={worst['trades']}, WinRate={worst['win_rate']}%, Avg={worst['avg_pnl']:+.3f}%")

    # Phase 1 params comparison across signal types
    print(f"\n{'='*70}")
    print(f"  PHASE 1 PARAMS (T2.5x SL1.5x H7 TA1.5x TR1.5x) BY SIGNAL")
    print(f"{'='*70}")
    phase1_key = "T2.5x_SL1.5x_H7_TA1.5x_TR1.5x"
    for sig_type in ["ShortCovering", "LongBuildUp", "ShortBuildUp", "LongUnwinding"]:
        key = f"{sig_type}|{phase1_key}"
        if key in results:
            r = results[key]
            print(f"  {sig_type:<16} Trades={r['trades']:>5}  WinRate={r['win_rate']:>5.1f}%  Avg={r['avg_pnl']:>+7.3f}%  Total={r['total_pnl']:>+8.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Historical OI Pipeline")
    parser.add_argument("command", choices=["download", "build", "signals", "backtest", "summary"])
    args = parser.parse_args()

    if args.command == "download":
        download_bhavcopies()
    elif args.command == "build":
        build_oi_database()
    elif args.command == "signals":
        reconstruct_signals()
    elif args.command == "backtest":
        run_backtest()
    elif args.command == "summary":
        print_summary()
