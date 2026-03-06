"""F&O ban list checker.

Fetches NSE's daily fo_secban.csv and caches it.
Stocks in F&O ban cannot have new F&O positions opened.
"""

import csv
import io
import logging
from datetime import datetime, timedelta
from pathlib import Path

import requests

logger = logging.getLogger("paper_trade")

CACHE_DIR = Path("data/fo_ban_cache")
BAN_URL = "https://nsearchives.nseindia.com/content/fo/fo_secban.csv"

# NSE requires browser-like headers
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.nseindia.com/",
}


def _cache_path() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"fo_ban_{datetime.now().strftime('%Y-%m-%d')}.txt"


def fetch_ban_list() -> set[str]:
    """Fetch today's F&O ban list from NSE. Returns set of banned symbols.

    Uses file-based daily cache. Returns empty set on failure (fail-open
    is safer than blocking all F&O trades on network error).
    """
    cache_file = _cache_path()

    # Check cache first
    if cache_file.exists():
        try:
            symbols = set(cache_file.read_text().strip().split("\n"))
            symbols.discard("")
            return symbols
        except Exception:
            pass

    # Fetch from NSE
    try:
        resp = requests.get(BAN_URL, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            logger.debug("F&O ban fetch failed: HTTP %d", resp.status_code)
            return set()

        # Parse CSV — first column is symbol name
        reader = csv.reader(io.StringIO(resp.text))
        symbols = set()
        for i, row in enumerate(reader):
            if i == 0:
                continue  # skip header
            if row:
                # Clean symbol name (remove whitespace, quotes)
                sym = row[0].strip().strip('"').upper()
                if sym:
                    symbols.add(sym)

        # Cache for today
        cache_file.write_text("\n".join(sorted(symbols)))
        logger.info("F&O ban list: %d symbols banned today", len(symbols))
        return symbols

    except Exception as e:
        logger.debug("F&O ban fetch error: %s", e)
        return set()


def is_in_fo_ban(symbol: str) -> bool:
    """Check if a symbol is in today's F&O ban list."""
    ban_list = fetch_ban_list()
    # Normalize: remove -EQ suffix, uppercase
    clean = symbol.upper().replace("-EQ", "").strip()
    return clean in ban_list
