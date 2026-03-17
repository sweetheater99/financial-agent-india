# v7/ws_monitor.py
"""Real-time WebSocket monitor for V7 SL/target tracking.

Runs as a standalone process during market hours (9:15 AM — 3:30 PM).
Subscribes to Kite WebSocket ticks for all open position instruments.
Checks SL/target levels on every tick and alerts immediately.

Shared state with cron-based executor via JSON files in data/v7/:
- positions.json: read by monitor, written by executor
- ws_state.json: written by monitor (last tick, triggered SLs)
- daily_state.json: updated by monitor on SL/target hits

Usage:
    python -m v7.ws_monitor           # run until 3:30 PM
    python -m v7.ws_monitor --paper   # paper mode (log only, no orders)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, date
from pathlib import Path
from threading import Event
from typing import Any
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger("v7.ws_monitor")

# Market hours
MARKET_OPEN_HOUR, MARKET_OPEN_MIN = 9, 15
MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN = 15, 30


class WSMonitor:
    """WebSocket-based real-time SL/target monitor."""

    def __init__(self, data_dir: str | Path = "data/v7", paper: bool = True):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.paper = paper
        self._stop_event = Event()
        self._ticker = None
        self._subscribed_tokens: set[int] = set()
        # token → position mapping for fast lookup
        self._token_to_positions: dict[int, list[dict]] = {}
        self._last_reload = 0.0
        self._reload_interval = 30  # reload positions every 30s

    def _load_positions(self) -> list[dict]:
        """Load open positions from shared JSON file."""
        path = self.data_dir / "positions.json"
        if not path.exists():
            return []
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def _load_daily_state(self) -> dict:
        """Load daily state."""
        path = self.data_dir / "daily_state.json"
        if not path.exists():
            return {"date": str(date.today())}
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"date": str(date.today())}

    def _save_daily_state(self, state: dict) -> None:
        """Save daily state."""
        path = self.data_dir / "daily_state.json"
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def _save_ws_state(self, ws_state: dict) -> None:
        """Save WebSocket monitor state."""
        path = self.data_dir / "ws_state.json"
        with open(path, "w") as f:
            json.dump(ws_state, f, indent=2, default=str)

    def _resolve_tokens(self, positions: list[dict]) -> dict[int, list[dict]]:
        """Map instrument tokens to positions for fast tick lookup."""
        from kite_data import resolve_token

        token_map: dict[int, list[dict]] = {}
        for pos in positions:
            symbol = pos.get("symbol", "")
            instrument = pos.get("instrument", "FUT")

            # For futures, resolve the trading symbol
            tradingsymbol = pos.get("tradingsymbol", "")
            exchange = pos.get("exchange", "NFO")

            token = None
            if tradingsymbol:
                token = resolve_token(tradingsymbol, exchange)
            elif symbol:
                # Try common patterns
                token = resolve_token(symbol, "NSE")

            if token is not None:
                if token not in token_map:
                    token_map[token] = []
                token_map[token].append(pos)

        return token_map

    def _check_sl_target(self, token: int, ltp: float) -> list[dict]:
        """Check if LTP has hit SL or target for any position mapped to this token.

        Returns list of triggered events.
        """
        positions = self._token_to_positions.get(token, [])
        triggered = []

        for pos in positions:
            sl = pos.get("stoploss", 0)
            target = pos.get("target", 0)
            direction = pos.get("direction", "bullish")
            already_triggered = pos.get("_ws_triggered", False)

            if already_triggered:
                continue

            hit = None
            if direction == "bullish":
                if sl > 0 and ltp <= sl:
                    hit = "stoploss"
                elif target > 0 and ltp >= target:
                    hit = "target"
            else:  # bearish
                if sl > 0 and ltp >= sl:
                    hit = "stoploss"
                elif target > 0 and ltp <= target:
                    hit = "target"

            if hit:
                pos["_ws_triggered"] = True
                triggered.append({
                    "symbol": pos.get("symbol", "?"),
                    "setup_id": pos.get("setup_id", "?"),
                    "type": hit,
                    "ltp": ltp,
                    "sl": sl,
                    "target": target,
                    "direction": direction,
                    "timestamp": datetime.now(IST).isoformat(),
                })

        return triggered

    def _send_alert(self, event: dict) -> None:
        """Send Telegram alert for SL/target hit."""
        try:
            from v7.telegram import TelegramAlerter, AlertLevel
            telegram = TelegramAlerter()

            hit_type = "SL HIT" if event["type"] == "stoploss" else "TARGET HIT"
            mode = " [PAPER]" if self.paper else ""

            msg = "\n".join([
                f"<b>V7 {hit_type}{mode}</b>",
                "",
                f"Symbol: <b>{event['symbol']}</b>",
                f"Setup: {event['setup_id']}",
                f"LTP: {event['ltp']:,.2f}",
                f"SL: {event['sl']:,.2f} | Target: {event['target']:,.2f}",
                f"Direction: {event['direction']}",
            ])
            telegram.send(msg, AlertLevel.HIGH)
        except Exception as e:
            logger.error(f"Alert send failed: {e}")

    def _update_daily_state_on_trigger(self, event: dict) -> None:
        """Update daily state when SL/target is hit."""
        daily = self._load_daily_state()
        triggers = daily.get("ws_triggers", [])
        triggers.append(event)
        daily["ws_triggers"] = triggers

        if event["type"] == "stoploss":
            daily["sl_hits_today"] = daily.get("sl_hits_today", 0) + 1

        self._save_daily_state(daily)

    def _maybe_reload_positions(self) -> None:
        """Reload positions periodically to pick up new trades from executor."""
        now = time.time()
        if now - self._last_reload < self._reload_interval:
            return

        self._last_reload = now
        positions = self._load_positions()

        if not positions:
            # Unsubscribe all if no positions
            if self._subscribed_tokens and self._ticker:
                try:
                    self._ticker.unsubscribe(list(self._subscribed_tokens))
                except Exception:
                    pass
                self._subscribed_tokens.clear()
                self._token_to_positions.clear()
            return

        try:
            new_token_map = self._resolve_tokens(positions)
        except Exception as e:
            logger.error(f"Token resolution failed: {e}")
            return

        new_tokens = set(new_token_map.keys())

        # Subscribe to new tokens, unsubscribe removed ones
        to_add = new_tokens - self._subscribed_tokens
        to_remove = self._subscribed_tokens - new_tokens

        if self._ticker:
            if to_remove:
                try:
                    self._ticker.unsubscribe(list(to_remove))
                except Exception:
                    pass
            if to_add:
                try:
                    self._ticker.subscribe(list(to_add))
                    self._ticker.set_mode(self._ticker.MODE_LTP, list(to_add))
                except Exception as e:
                    logger.error(f"Subscribe failed: {e}")

        self._subscribed_tokens = new_tokens
        self._token_to_positions = new_token_map

    def _on_ticks(self, ws: Any, ticks: list[dict]) -> None:
        """Handle incoming ticks. Check SL/target for each."""
        for tick in ticks:
            token = tick.get("instrument_token")
            ltp = tick.get("last_price", 0)

            if token is None or ltp <= 0:
                continue

            triggered = self._check_sl_target(token, ltp)
            for event in triggered:
                logger.info(f"{event['type'].upper()}: {event['symbol']} at {ltp}")
                self._send_alert(event)
                self._update_daily_state_on_trigger(event)

        # Periodically check for new positions
        self._maybe_reload_positions()

        # Save ws_state with last tick time
        self._save_ws_state({
            "last_tick": datetime.now(IST).isoformat(),
            "subscribed_tokens": len(self._subscribed_tokens),
            "active": True,
        })

    def _on_connect(self, ws: Any, response: Any) -> None:
        """On WebSocket connect, subscribe to position tokens."""
        logger.info("WebSocket connected")
        self._maybe_reload_positions()

    def _on_close(self, ws: Any, code: int, reason: str) -> None:
        """Handle WebSocket close."""
        logger.info(f"WebSocket closed: {code} {reason}")

    def _on_error(self, ws: Any, code: int, reason: str) -> None:
        """Handle WebSocket error."""
        logger.error(f"WebSocket error: {code} {reason}")

    def _is_market_hours(self) -> bool:
        """Check if within market hours."""
        now = datetime.now(IST)
        market_open = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MIN, second=0)
        market_close = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MIN, second=0)
        return market_open <= now <= market_close

    def run(self) -> None:
        """Run the WebSocket monitor until market close or stop signal."""
        from kite_data import get_kite
        from kiteconnect import KiteTicker

        kite = get_kite()
        api_key = os.environ.get("KITE_API_KEY", "")
        access_token = kite.access_token

        if not api_key or not access_token:
            logger.error("Kite API key or access token not available")
            sys.exit(1)

        self._ticker = KiteTicker(api_key, access_token)
        self._ticker.on_ticks = self._on_ticks
        self._ticker.on_connect = self._on_connect
        self._ticker.on_close = self._on_close
        self._ticker.on_error = self._on_error

        # Handle graceful shutdown
        def _signal_handler(sig, frame):
            logger.info("Shutdown signal received")
            self._stop_event.set()
            if self._ticker:
                try:
                    self._ticker.close()
                except Exception:
                    pass

        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)

        logger.info(f"Starting WS monitor (paper={self.paper})")
        self._save_ws_state({"active": True, "started": datetime.now(IST).isoformat()})

        # KiteTicker.connect() blocks — it has its own reconnect loop
        # We use threaded=True so we can check market hours
        self._ticker.connect(threaded=True)

        # Wait until market close or stop signal
        while not self._stop_event.is_set():
            if not self._is_market_hours():
                logger.info("Market closed — shutting down WS monitor")
                break
            self._stop_event.wait(timeout=60)

        # Cleanup
        try:
            self._ticker.close()
        except Exception:
            pass

        self._save_ws_state({"active": False, "stopped": datetime.now(IST).isoformat()})
        logger.info("WS monitor stopped")


def main():
    parser = argparse.ArgumentParser(description="V7 WebSocket SL/Target Monitor")
    parser.add_argument("--paper", action="store_true", default=True,
                        help="Paper mode — alert only, no orders (default: True)")
    parser.add_argument("--live", action="store_true", default=False,
                        help="Live mode — will place exit orders")
    args = parser.parse_args()

    paper = not args.live

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("data/v7/ws_monitor.log"),
        ],
    )

    monitor = WSMonitor(paper=paper)
    monitor.run()


if __name__ == "__main__":
    main()
