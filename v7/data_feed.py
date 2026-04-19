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

    def is_quote_stale(self) -> bool:
        """Check if Kite quotes are from a previous day (market closed / holiday).
        Returns True if quotes are stale and should NOT be acted on."""
        if self.mode != "kite" or not self.kite:
            return False
        try:
            q = self.kite.quote(["NSE:NIFTY 50"])
            ts = q.get("NSE:NIFTY 50", {}).get("timestamp")
            if ts is None:
                return True
            from datetime import date
            quote_date = ts.date() if hasattr(ts, "date") else date.today()
            if quote_date < date.today():
                log.warning("DataFeed: STALE QUOTES — NIFTY timestamp %s is from previous day. Market likely closed.", ts)
                return True
        except Exception as e:
            log.warning("DataFeed: stale check failed: %s", e)
        return False

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
                    days: int = 1) -> list[dict]:
        """Fetch OHLCV candles as list of dicts. Only available in Kite mode."""
        if not self.can_trade():
            raise ProtectOnlyMode("Cannot fetch candles in protect-only mode")
        from kite_data import fetch_candles_kite, resolve_token
        from v7.config_v7 import WATCHLIST
        wl = next((w for w in WATCHLIST if w["symbol"] == symbol), None)
        if not wl:
            raise DataFeedError(f"Symbol {symbol} not in watchlist")
        token = resolve_token(symbol)
        raw = fetch_candles_kite(symbol, token, "NSE", interval, days)
        if not raw:
            return []
        # Convert [ts, open, high, low, close, volume] → dicts
        return [
            {"timestamp": r[0], "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5]}
            for r in raw
        ]

    # ── Option Chain ───────────────────────────────────────────────

    def get_option_chain(self, symbol: str, expiry: str = None) -> list[dict]:
        """Fetch option chain. Only available in Kite mode.

        Pass expiry=None or "current" for nearest weekly expiry.
        """
        if not self.can_trade():
            raise ProtectOnlyMode("Cannot fetch option chain in protect-only mode")
        from kite_data import fetch_option_chain_kite
        # "current" means auto-detect nearest expiry
        if expiry in (None, "current", ""):
            expiry = None
        return fetch_option_chain_kite(symbol, expiry)

    # ── VIX ────────────────────────────────────────────────────────

    def get_vix(self) -> float:
        """Fetch India VIX."""
        if self.mode == "kite" and self.kite:
            from kite_data import get_vix_kite
            return get_vix_kite()
        # AngelOne VIX removed — stale token 26017 causes errors
        # Fall through to yfinance
        # yfinance fallback
        try:
            import yfinance as yf
            hist = yf.Ticker("^INDIAVIX").history(period="2d")
            if len(hist) >= 1:
                return float(hist.iloc[-1]["Close"])
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
