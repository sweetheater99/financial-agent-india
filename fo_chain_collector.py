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

            iv = implied_volatility(premium, spot, strike, dte, risk_free, opt_type)
            if math.isnan(iv) or iv <= 0:
                iv = vix / 100.0

            greeks_data = black_scholes_greeks(
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
                "delta": greeks_data["delta"],
                "gamma": greeks_data["gamma"],
                "theta": greeks_data["theta"],
                "vega": greeks_data["vega"],
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

    spot = get_ltp_kite(symbol)
    vix = get_vix_kite()

    if spot is None or vix is None:
        logger.error("Failed to fetch spot (%s) or VIX (%s)", spot, vix)
        return None

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
