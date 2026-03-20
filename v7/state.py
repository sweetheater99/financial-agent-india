# v7/state.py
"""State persistence for V7 trading bot.

All runtime state is persisted to JSON files in data/v7/.
Handles load, save, reset, and stale-data detection.
"""
from __future__ import annotations

import fcntl
import json
import os
import uuid
from datetime import date
from pathlib import Path

from v7.types import Playbook, Position, TradeResult


class StateManager:
    """File-backed state persistence for V7."""

    def __init__(self, state_dir: str | Path):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.dir / name

    def _atomic_write(self, path: Path, data: dict | list) -> None:
        tmp = path.with_suffix(f".tmp.{uuid.uuid4().hex[:8]}")
        f = None
        try:
            f = open(tmp, "w")
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            json.dump(data, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
            os.replace(str(tmp), str(path))
        finally:
            if f is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                f.close()

    def _locked_read(self, name: str) -> dict | list | None:
        path = self._path(name)
        if not path.exists():
            return None
        f = None
        try:
            f = open(path)
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            return json.load(f)
        finally:
            if f is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                f.close()

    def _read_json(self, name: str) -> dict | list | None:
        return self._locked_read(name)

    def _write_json(self, name: str, data: dict | list) -> None:
        self._atomic_write(self._path(name), data)

    # ── Playbook ───────────────────────────────────────────────────

    def save_playbook(self, playbook: Playbook) -> None:
        self._write_json("playbook.json", playbook.to_dict())

    def load_playbook(self, today: date | None = None) -> Playbook | None:
        data = self._read_json("playbook.json")
        if data is None:
            return None
        pb = Playbook.from_dict(data)
        today = today or date.today()
        if pb.date != today:
            return None
        return pb

    # ── Positions ──────────────────────────────────────────────────

    def save_positions(self, positions: list[Position]) -> None:
        self._write_json("positions.json", [p.to_dict() for p in positions])

    def load_positions(self) -> list[Position]:
        data = self._read_json("positions.json")
        if data is None:
            return []
        return [Position.from_dict(d) for d in data]

    # ── Daily State ────────────────────────────────────────────────

    def save_daily_state(self, state: dict) -> None:
        self._write_json("daily_state.json", state)


    # ── Theta Engine State ────────────────────────────────────────────

    def save_theta_state(self, state: dict | None) -> None:
        """Persist theta engine condor state."""
        import json
        path = self.dir / "theta_state.json"
        if state is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(json.dumps(state, indent=2, default=str))

    def load_theta_state(self) -> dict | None:
        """Load persisted theta engine condor state."""
        import json
        path = self.dir / "theta_state.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, Exception):
            return None

    def load_daily_state(self, today: date | None = None) -> dict:
        today = today or date.today()
        data = self._read_json("daily_state.json")
        if data is None or data.get("date") != str(today):
            return {
                "date": str(today),
                "trades_today": 0,
                "sl_hits_today": 0,
                "daily_pnl": 0.0,
                "current_risk": 0.0,
            }
        return data

    # ── Trade History ──────────────────────────────────────────────

    def append_trade(self, trade: TradeResult) -> None:
        history = self._read_json("trade_history.json") or []
        history.append(trade.to_dict())
        self._write_json("trade_history.json", history)

    def load_trade_history(self) -> list[TradeResult]:
        data = self._read_json("trade_history.json")
        if data is None:
            return []
        return [TradeResult.from_dict(d) for d in data]

    # ── Level Memory ───────────────────────────────────────────────

    def save_level_memory(self, levels: dict) -> None:
        self._write_json("level_memory.json", levels)

    def load_level_memory(self) -> dict:
        return self._read_json("level_memory.json") or {}

    # ── Monthly State ──────────────────────────────────────────────

    def save_monthly_state(self, state: dict) -> None:
        self._write_json("monthly_state.json", state)

    def load_monthly_state(self) -> dict:
        data = self._read_json("monthly_state.json")
        if data is None:
            return {
                "month": str(date.today())[:7],
                "mtd_pnl": 0.0,
                "mtd_pnl_pct": 0.0,
                "trades_this_month": 0,
                "survival_mode": False,
            }
        return data

    # ── Edge Tracker ───────────────────────────────────────────────

    def save_edge_tracker(self, data: dict) -> None:
        self._write_json("edge_tracker.json", data)

    def load_edge_tracker(self) -> dict:
        return self._read_json("edge_tracker.json") or {
            "overall_win_rate": 0.0,
            "total_trades": 0,
            "by_strategy": {},
            "by_instrument": {},
            "by_time": {},
        }
