"""
Zerodha Kite Connect data layer.

Provides: authentication, instrument cache, LTP quotes, historical candles,
option chain construction, and Greeks computation.

Usage:
    from kite_data import get_kite, get_ltp_kite, fetch_candles_kite, fetch_option_chain_kite
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional

from kiteconnect import KiteConnect
import config

logger = logging.getLogger("paper_trade")

# ── Module state ──────────────────────────────────────────────────────────
_kite: Optional[KiteConnect] = None
_instruments_cache: dict = {}  # {exchange: {symbol: {token, lot_size, ...}}}
_instruments_loaded: bool = False

# Index symbols that aren't in instruments list — map to Kite quote keys
INDEX_ALIASES = {
    "Nifty 50": "NSE:NIFTY 50",
    "NIFTY 50": "NSE:NIFTY 50",
    "NIFTY": "NSE:NIFTY 50",
    "India VIX": "NSE:INDIA VIX",
    "INDIA VIX": "NSE:INDIA VIX",
    "Nifty Bank": "NSE:NIFTY BANK",
    "BANKNIFTY": "NSE:NIFTY BANK",
}


# ── Authentication ────────────────────────────────────────────────────────

def get_kite() -> KiteConnect:
    """Get authenticated KiteConnect instance. Reuses cached access token if valid."""
    global _kite
    if _kite is not None:
        return _kite

    kite = KiteConnect(api_key=config.KITE_API_KEY)

    # Try cached access token
    token_file = Path(config.KITE_ACCESS_TOKEN_FILE)
    if token_file.exists():
        data = json.loads(token_file.read_text())
        # Token valid for ~24h, check if saved today
        if data.get("date") == str(date.today()):
            kite.set_access_token(data["access_token"])
            _kite = kite
            logger.info("Kite: using cached access token from %s", data["date"])
            return _kite

    raise RuntimeError(
        "Kite access token not found or expired. "
        "Run: python kite_auth.py to login via browser."
    )


def save_access_token(access_token: str) -> None:
    """Save access token to file for reuse."""
    token_file = Path(config.KITE_ACCESS_TOKEN_FILE)
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text(json.dumps({
        "access_token": access_token,
        "date": str(date.today()),
    }))
    logger.info("Kite: access token saved for %s", date.today())


# ── Instrument Cache ──────────────────────────────────────────────────────

def _load_instruments(kite: KiteConnect) -> None:
    """Load and cache all instruments from Kite. Called once per session."""
    global _instruments_cache, _instruments_loaded
    if _instruments_loaded:
        return

    cache_file = Path(os.path.dirname(__file__)) / "data" / ".kite_instruments.json"

    # Use cached file if saved today
    if cache_file.exists():
        data = json.loads(cache_file.read_text())
        if data.get("date") == str(date.today()):
            _instruments_cache = data["instruments"]
            _instruments_loaded = True
            total = sum(len(v) for v in _instruments_cache.values())
            logger.info("Kite: loaded %d instruments from cache", total)
            return

    logger.info("Kite: downloading instruments (first call of the day)...")
    instruments = {}

    for exchange in ["NSE", "NFO", "MCX"]:
        raw = kite.instruments(exchange)
        exch_map = {}
        for inst in raw:
            key = inst["tradingsymbol"]
            exch_map[key] = {
                "token": inst["instrument_token"],
                "symbol": inst.get("name", ""),
                "tradingsymbol": key,
                "lot_size": inst.get("lot_size", 1),
                "instrument_type": inst.get("instrument_type", ""),
                "strike": inst.get("strike", 0),
                "expiry": str(inst.get("expiry", "")),
                "exchange": exchange,
            }
        instruments[exchange] = exch_map
        logger.info("Kite: %s — %d instruments", exchange, len(exch_map))

    # Save to file
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({"date": str(date.today()), "instruments": instruments}))

    _instruments_cache = instruments
    _instruments_loaded = True


def resolve_token(symbol: str, exchange: str = "NSE") -> Optional[int]:
    """Resolve a symbol to its Kite instrument_token."""
    kite = get_kite()
    _load_instruments(kite)

    exch = _instruments_cache.get(exchange, {})

    # Direct match
    if symbol in exch:
        return exch[symbol]["token"]

    # Try common suffixes for NSE equities
    if exchange == "NSE":
        for suffix in ["-EQ", "-BE"]:
            candidate = symbol + suffix
            if candidate in exch:
                return exch[candidate]["token"]

    return None


def resolve_equity_token_kite(symbol: str) -> Optional[int]:
    """Resolve equity symbol to Kite token. Equivalent to utils.resolve_equity_token."""
    return resolve_token(symbol, "NSE")


def get_nfo_instruments(symbol: str, expiry: str = None, inst_type: str = None) -> list:
    """Get NFO instruments for a symbol, optionally filtered by expiry and type (CE/PE/FUT)."""
    kite = get_kite()
    _load_instruments(kite)

    results = []
    for key, inst in _instruments_cache.get("NFO", {}).items():
        if inst["symbol"] != symbol:
            continue
        if expiry and inst["expiry"] != expiry:
            continue
        if inst_type and inst["instrument_type"] != inst_type:
            continue
        results.append(inst)

    return sorted(results, key=lambda x: x.get("strike", 0))


# ── LTP / Quotes ──────────────────────────────────────────────────────────

def get_ltp_kite(symbol: str, exchange: str = "NSE") -> Optional[float]:
    """Fetch LTP for a symbol via Kite. Returns price or None."""
    kite = get_kite()

    # Check index aliases first (Nifty 50, India VIX, etc.)
    alias_key = INDEX_ALIASES.get(symbol)
    if alias_key:
        try:
            resp = kite.ltp([alias_key])
            if resp and alias_key in resp:
                return float(resp[alias_key]["last_price"])
        except Exception as e:
            logger.warning("Kite LTP failed for index %s: %s", symbol, e)
        return None

    _load_instruments(kite)
    token = resolve_token(symbol, exchange)
    if token is None:
        logger.warning("Kite: could not resolve %s:%s", exchange, symbol)
        return None

    try:
        key = f"{exchange}:{symbol}"
        resp = kite.ltp([key])
        if resp and key in resp:
            return float(resp[key]["last_price"])
    except Exception as e:
        logger.warning("Kite LTP failed for %s: %s", symbol, e)
    return None


def get_quote_kite(symbol: str, exchange: str = "NSE") -> Optional[dict]:
    """Fetch full quote (OHLC, volume, OI) for a symbol via Kite."""
    kite = get_kite()
    try:
        key = f"{exchange}:{symbol}"
        resp = kite.quote([key])
        if resp and key in resp:
            return resp[key]
    except Exception as e:
        logger.warning("Kite quote failed for %s: %s", symbol, e)
    return None


# ── Historical Candles ────────────────────────────────────────────────────

KITE_INTERVAL_MAP = {
    "ONE_MINUTE": "minute",
    "FIVE_MINUTE": "5minute",
    "FIFTEEN_MINUTE": "15minute",
    "THIRTY_MINUTE": "30minute",
    "ONE_HOUR": "60minute",
    "ONE_DAY": "day",
}


def fetch_candles_kite(
    symbol: str,
    token: Optional[int] = None,
    exchange: str = "NSE",
    interval: str = "ONE_DAY",
    days: int = 30,
    oi: bool = False,
) -> Optional[list]:
    """Fetch historical candles from Kite. Returns list in SmartAPI format:
    [[timestamp, open, high, low, close, volume], ...]
    """
    kite = get_kite()
    _load_instruments(kite)

    if token is None:
        token = resolve_token(symbol, exchange)
    if token is None:
        logger.warning("Kite: could not resolve %s for candles", symbol)
        return None

    kite_interval = KITE_INTERVAL_MAP.get(interval, "day")
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days)

    try:
        candles = kite.historical_data(
            instrument_token=token,
            from_date=from_date,
            to_date=to_date,
            interval=kite_interval,
            oi=oi,
        )
    except Exception as e:
        logger.warning("Kite candle fetch failed for %s: %s", symbol, e)
        return None

    if not candles:
        return None

    # Convert to SmartAPI format: [timestamp_str, open, high, low, close, volume]
    result = []
    for c in candles:
        ts = c["date"].strftime("%Y-%m-%dT%H:%M:%S+05:30") if hasattr(c["date"], "strftime") else str(c["date"])
        row = [ts, c["open"], c["high"], c["low"], c["close"], c["volume"]]
        result.append(row)

    return result


# ── Option Chain ──────────────────────────────────────────────────────────

def fetch_option_chain_kite(symbol: str, expiry_str: str = None) -> Optional[list]:
    """Build option chain from Kite instruments + quotes.

    Returns data in SmartAPI optionGreek format:
    [{"strikePrice": 2500, "CE": {...}, "PE": {...}}, ...]
    """
    kite = get_kite()
    _load_instruments(kite)

    # Find nearest expiry if not specified
    if expiry_str is None:
        expiry_str = _find_nearest_expiry(symbol)
        if expiry_str is None:
            logger.warning("Kite: no expiry found for %s", symbol)
            return None

    logger.info("Kite: building option chain for %s expiry %s", symbol, expiry_str)

    # Get all CE and PE instruments for this symbol + expiry
    ce_instruments = get_nfo_instruments(symbol, expiry=expiry_str, inst_type="CE")
    pe_instruments = get_nfo_instruments(symbol, expiry=expiry_str, inst_type="PE")

    if not ce_instruments and not pe_instruments:
        logger.warning("Kite: no option instruments for %s expiry %s", symbol, expiry_str)
        return None

    # Build strike map
    strikes = {}
    all_keys = []

    for inst in ce_instruments:
        strike = inst["strike"]
        if strike not in strikes:
            strikes[strike] = {"strikePrice": strike}
        strikes[strike]["_ce_key"] = f"NFO:{inst['tradingsymbol']}"
        strikes[strike]["_ce_token"] = inst["token"]
        strikes[strike]["_ce_lot"] = inst["lot_size"]
        all_keys.append(f"NFO:{inst['tradingsymbol']}")

    for inst in pe_instruments:
        strike = inst["strike"]
        if strike not in strikes:
            strikes[strike] = {"strikePrice": strike}
        strikes[strike]["_pe_key"] = f"NFO:{inst['tradingsymbol']}"
        strikes[strike]["_pe_token"] = inst["token"]
        strikes[strike]["_pe_lot"] = inst["lot_size"]
        all_keys.append(f"NFO:{inst['tradingsymbol']}")

    # Fetch quotes in batches of 500 (Kite limit)
    all_quotes = {}
    for i in range(0, len(all_keys), 500):
        batch = all_keys[i:i + 500]
        try:
            resp = kite.quote(batch)
            all_quotes.update(resp)
        except Exception as e:
            logger.warning("Kite: quote batch failed: %s", e)
        time.sleep(config.KITE_API_DELAY)

    # Build chain in SmartAPI format
    chain = []
    for strike_price in sorted(strikes.keys()):
        entry = {"strikePrice": strike_price}

        ce_key = strikes[strike_price].get("_ce_key")
        if ce_key and ce_key in all_quotes:
            q = all_quotes[ce_key]
            entry["CE"] = _quote_to_option_data(q, strikes[strike_price].get("_ce_lot", 1))

        pe_key = strikes[strike_price].get("_pe_key")
        if pe_key and pe_key in all_quotes:
            q = all_quotes[pe_key]
            entry["PE"] = _quote_to_option_data(q, strikes[strike_price].get("_pe_lot", 1))

        if "CE" in entry or "PE" in entry:
            chain.append(entry)

    logger.info("Kite: built option chain with %d strikes for %s", len(chain), symbol)
    return chain


def _quote_to_option_data(quote: dict, lot_size: int) -> dict:
    """Convert a Kite quote to SmartAPI-compatible option data dict."""
    return {
        "lastTradedPrice": quote.get("last_price", 0),
        "openInterest": quote.get("oi", 0),
        "volume": quote.get("volume", 0),
        "impliedVolatility": 0,  # computed separately if needed
        "delta": 0,
        "gamma": 0,
        "theta": 0,
        "vega": 0,
        "bidPrice": quote.get("depth", {}).get("buy", [{}])[0].get("price", 0),
        "askPrice": quote.get("depth", {}).get("sell", [{}])[0].get("price", 0),
        "lotSize": lot_size,
    }


def _find_nearest_expiry(symbol: str) -> Optional[str]:
    """Find the nearest monthly expiry for a symbol in NFO instruments."""
    kite = get_kite()
    _load_instruments(kite)

    expiries = set()
    for key, inst in _instruments_cache.get("NFO", {}).items():
        if inst["symbol"] == symbol and inst["instrument_type"] in ("CE", "PE"):
            exp = inst["expiry"]
            if exp:
                expiries.add(exp)

    if not expiries:
        return None

    today = str(date.today())
    future_expiries = sorted(e for e in expiries if e >= today)

    if future_expiries:
        return future_expiries[0]

    return None


# ── VIX ───────────────────────────────────────────────────────────────────

def get_vix_kite() -> Optional[float]:
    """Fetch India VIX from Kite."""
    try:
        kite = get_kite()
        resp = kite.quote(["NSE:INDIA VIX"])
        if resp and "NSE:INDIA VIX" in resp:
            return float(resp["NSE:INDIA VIX"]["last_price"])
    except Exception as e:
        logger.warning("Kite VIX fetch failed: %s", e)
    return None
