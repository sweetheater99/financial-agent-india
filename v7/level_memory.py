# v7/level_memory.py
"""Persistent key level tracking for V7.

Stores support/resistance levels per symbol with strength scoring.
Levels are strengthened on retests that hold, weakened on breaks,
flipped on clean breaks (resistance → support and vice versa),
and pruned when stale (not tested for N sessions).

Backed by data/v7/level_memory.json via StateManager.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


@dataclass
class Level:
    """A single key price level."""
    price: float
    level_type: str        # "support" or "resistance"
    strength: int          # starts at 1, +1 on retest hold, -1 on break, removed at 0
    source: str            # human-readable origin
    last_tested: str       # ISO date
    created: str           # ISO date

    def to_dict(self) -> dict:
        return {
            "price": self.price,
            "type": self.level_type,
            "strength": self.strength,
            "source": self.source,
            "last_tested": self.last_tested,
            "created": self.created,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Level:
        return cls(
            price=d["price"],
            level_type=d["type"],
            strength=d["strength"],
            source=d["source"],
            last_tested=d["last_tested"],
            created=d.get("created", d["last_tested"]),
        )

    def is_near(self, price: float, threshold_pct: float = 0.1) -> bool:
        """Check if price is within threshold_pct% of this level."""
        if self.price == 0:
            return False
        return abs(price - self.price) / self.price * 100 <= threshold_pct


class LevelMemory:
    """Persistent store for key levels and OI walls."""

    def __init__(self, state_dir: str | Path):
        self._dir = Path(state_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "level_memory.json"
        self._data: dict = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            with open(self._path) as f:
                self._data = json.load(f)
        else:
            self._data = {}

    def save(self) -> None:
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2, default=str)

    def _ensure_symbol(self, symbol: str) -> None:
        if symbol not in self._data:
            self._data[symbol] = {"levels": [], "oi_walls": {}}

    def add_level(
        self, symbol: str, price: float, level_type: str, source: str,
        today: date | None = None,
    ) -> None:
        """Add a new level or strengthen existing nearby level."""
        today_str = str(today or date.today())
        self._ensure_symbol(symbol)

        for lv_dict in self._data[symbol]["levels"]:
            lv = Level.from_dict(lv_dict)
            if lv.is_near(price) and lv.level_type == level_type:
                lv_dict["strength"] += 1
                lv_dict["last_tested"] = today_str
                lv_dict["source"] = f"{lv_dict['source']}; {source}"
                self.save()
                return

        new_level = Level(
            price=price, level_type=level_type, strength=1,
            source=source, last_tested=today_str, created=today_str,
        )
        self._data[symbol]["levels"].append(new_level.to_dict())
        self.save()

    def get_levels(self, symbol: str) -> list[Level]:
        if symbol not in self._data:
            return []
        levels = [Level.from_dict(d) for d in self._data[symbol]["levels"]]
        return sorted(levels, key=lambda lv: lv.price)

    def retest_level(
        self, symbol: str, price: float, held: bool,
        today: date | None = None,
    ) -> None:
        today_str = str(today or date.today())
        if symbol not in self._data:
            return

        to_remove = []
        for i, lv_dict in enumerate(self._data[symbol]["levels"]):
            lv = Level.from_dict(lv_dict)
            if lv.is_near(price):
                if held:
                    lv_dict["strength"] += 1
                    lv_dict["last_tested"] = today_str
                else:
                    lv_dict["strength"] -= 1
                    lv_dict["last_tested"] = today_str
                    if lv_dict["strength"] <= 0:
                        to_remove.append(i)

        for i in reversed(to_remove):
            self._data[symbol]["levels"].pop(i)
        self.save()

    def flip_level(self, symbol: str, price: float, today: date | None = None) -> None:
        today_str = str(today or date.today())
        if symbol not in self._data:
            return

        for lv_dict in self._data[symbol]["levels"]:
            lv = Level.from_dict(lv_dict)
            if lv.is_near(price):
                if lv_dict["type"] == "support":
                    lv_dict["type"] = "resistance"
                else:
                    lv_dict["type"] = "support"
                lv_dict["strength"] = 1
                lv_dict["last_tested"] = today_str
                break
        self.save()

    def remove_stale(self, max_age_days: int = 10, today: date | None = None) -> int:
        today_dt = today or date.today()
        removed = 0

        for symbol in list(self._data.keys()):
            to_remove = []
            for i, lv_dict in enumerate(self._data[symbol]["levels"]):
                last_tested = date.fromisoformat(lv_dict["last_tested"])
                age = (today_dt - last_tested).days
                if age > max_age_days:
                    to_remove.append(i)

            for i in reversed(to_remove):
                self._data[symbol]["levels"].pop(i)
                removed += 1

        if removed:
            self.save()
        return removed

    def update_oi_walls(
        self, symbol: str,
        call_max_oi_strike: float, put_max_oi_strike: float, pcr: float,
    ) -> None:
        self._ensure_symbol(symbol)
        self._data[symbol]["oi_walls"] = {
            "call_max_oi_strike": call_max_oi_strike,
            "put_max_oi_strike": put_max_oi_strike,
            "pcr": pcr,
        }
        self.save()

    def get_oi_walls(self, symbol: str) -> dict:
        if symbol not in self._data:
            return {}
        return self._data[symbol].get("oi_walls", {})

    def to_strategist_context(self, symbols: list[str]) -> dict:
        ctx = {}
        for sym in symbols:
            if sym not in self._data:
                continue
            ctx[sym] = {
                "levels": [Level.from_dict(d).to_dict() for d in self._data[sym]["levels"]],
                "oi_walls": self._data[sym].get("oi_walls", {}),
            }
        return ctx

    def bulk_update(self, levels_dict: dict) -> None:
        for symbol, sym_data in levels_dict.items():
            self._ensure_symbol(symbol)
            for lv_dict in sym_data.get("levels", []):
                exists = False
                for existing in self._data[symbol]["levels"]:
                    existing_lv = Level.from_dict(existing)
                    if existing_lv.is_near(lv_dict["price"]):
                        existing["strength"] = max(existing["strength"], lv_dict.get("strength", 1))
                        existing["last_tested"] = lv_dict.get("last_tested", str(date.today()))
                        exists = True
                        break
                if not exists:
                    self._data[symbol]["levels"].append(lv_dict)

            if "oi_walls" in sym_data:
                self._data[symbol]["oi_walls"] = sym_data["oi_walls"]

        self.save()
