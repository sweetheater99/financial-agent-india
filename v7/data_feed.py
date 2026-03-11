# v7/data_feed.py
"""Unified data feed for V7.

Kite primary, AngelOne protect-only fallback.
When Kite is unavailable, bot can only monitor LTP for existing positions.
"""
from __future__ import annotations

import logging
import time as time_mod
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))


class DataFeedError(Exception):
    pass


class ProtectOnlyMode(DataFeedError):
    """Raised when trying to do trading operations in protect-only mode."""
    pass


class DataFeed:
    """Abstraction over Kite + AngelOne data sources.

    Modes:
      - "kite": full trading capability
      - "protect_only": AngelOne LTP only, no new trades
      - "offline": no data source (for testing)
    """

    def __init__(self, use_kite: bool = True, use_angelone: bool = True):
        self.kite = None
        self.angelone = None
        self.mode = "offline"
        self._last_ltp: dict[str, float] = {}
        self._last_ltp_time: datetime | None = None

        if use_kite:
            try:
                from kite_data import get_kite
                self.kite = get_kite()
                self.mode = "kite"
                log.info("DataFeed: Kite connected")
            except Exception as e:
                log.warning(f"DataFeed: Kite unavailable — {e}")

        if use_angelone and self.mode != "kite":
            try:
                from connect import get_session
                self.angelone = get_session()
                self.mode = "protect_only"
                log.warning("DataFeed: Protect-only mode (AngelOne LTP only)")
            except Exception as e:
                log.error(f"DataFeed: AngelOne also unavailable — {e}")

    def can_trade(self) -> bool:
        return self.mode == "kite"

    def is_data_stale(self, max_age_seconds: int = 60) -> bool:
        if self._last_ltp_time is None:
            return True
        age = (datetime.now(IST) - self._last_ltp_time).total_seconds()
        return age > max_age_seconds

    # ── LTP ────────────────────────────────────────────────────────

    def get_batch_ltp(self, symbols: list[str]) -> dict[str, float]:
        """Fetch LTP for multiple symbols. Works in all modes."""
        try:
            prices = self._fetch_ltp_batch(symbols)
            self._last_ltp.update(prices)
            self._last_ltp_time = datetime.now(IST)
            return prices
        except Exception as e:
            log.error(f"DataFeed: LTP fetch failed — {e}")
            return self._last_ltp

    def _fetch_ltp_batch(self, symbols: list[str]) -> dict[str, float]:
        if self.mode == "kite" and self.kite:
            from v7.config_v7 import WATCHLIST
            kite_symbols = []
            sym_map = {}
            for sym in symbols:
                wl = next((w for w in WATCHLIST if w["symbol"] == sym), None)
                if wl:
                    key = f"NSE:{sym}"
                    if wl["type"] == "index":
                        key = f"NSE:{sym} 50" if sym == "NIFTY" else f"NSE:{sym}"
                    kite_symbols.append(key)
                    sym_map[key] = sym
            if not kite_symbols:
                return {}
            quotes = self.kite.quote(kite_symbols)
            return {sym_map[k]: v["last_price"] for k, v in quotes.items() if k in sym_map}

        elif self.mode == "protect_only" and self.angelone:
            from v7.config_v7 import WATCHLIST
            prices = {}
            for sym in symbols:
                wl = next((w for w in WATCHLIST if w["symbol"] == sym), None)
                if wl:
                    try:
                        data = self.angelone.ltpData("NSE", sym, wl["token"])
                        if data and data.get("data"):
                            prices[sym] = float(data["data"]["ltp"])
                        time_mod.sleep(0.5)
                    except Exception:
                        pass
            return prices

        return {}

    # ── Candles ────────────────────────────────────────────────────

    def get_candles(self, symbol: str, interval: str = "5minute",
                    days: int = 1) -> list:
        """Fetch OHLCV candles. Only available in Kite mode."""
        if not self.can_trade():
            raise ProtectOnlyMode("Cannot fetch candles in protect-only mode")
        from kite_data import fetch_candles_kite, resolve_token
        from v7.config_v7 import WATCHLIST
        wl = next((w for w in WATCHLIST if w["symbol"] == symbol), None)
        if not wl:
            raise DataFeedError(f"Symbol {symbol} not in watchlist")
        token = resolve_token(symbol)
        return fetch_candles_kite(symbol, token, "NSE", interval, days)

    # ── Option Chain ───────────────────────────────────────────────

    def get_option_chain(self, symbol: str, expiry: str) -> list[dict]:
        """Fetch option chain. Only available in Kite mode."""
        if not self.can_trade():
            raise ProtectOnlyMode("Cannot fetch option chain in protect-only mode")
        from kite_data import fetch_option_chain_kite
        return fetch_option_chain_kite(symbol, expiry)

    # ── VIX ────────────────────────────────────────────────────────

    def get_vix(self) -> float:
        """Fetch India VIX."""
        if self.mode == "kite" and self.kite:
            from kite_data import get_vix_kite
            return get_vix_kite()
        if self.angelone:
            try:
                data = self.angelone.ltpData("NSE", "India VIX", "26017")
                if data and data.get("data"):
                    return float(data["data"]["ltp"])
            except Exception:
                pass
        return 0.0

    # ── Health Check ───────────────────────────────────────────────

    def health_check(self) -> dict:
        """Return current data feed status."""
        return {
            "mode": self.mode,
            "can_trade": self.can_trade(),
            "stale": self.is_data_stale(),
            "last_update": str(self._last_ltp_time) if self._last_ltp_time else None,
            "cached_symbols": len(self._last_ltp),
        }

    def try_reconnect_kite(self) -> bool:
        """Attempt to reconnect to Kite. Returns True if successful."""
        try:
            from kite_data import get_kite
            self.kite = get_kite()
            self.mode = "kite"
            log.info("DataFeed: Kite reconnected")
            return True
        except Exception as e:
            log.warning(f"DataFeed: Kite reconnect failed — {e}")
            return False
