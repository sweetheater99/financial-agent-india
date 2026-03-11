"""Margin estimation and budget tracking for V7.

Tracks margin utilization across theta and directional positions.
Uses conservative estimates (actual SPAN margin via Kite API for live trading).
"""
from __future__ import annotations

from v7.config_v7 import CAPITAL, THETA_LIMITS


class MarginTracker:
    """Track margin utilization and enforce budget limits."""

    def __init__(self, capital: float = CAPITAL["initial"]):
        self.capital = capital
        self._positions: dict[str, float] = {}  # instrument → margin
        self._max_utilization_pct = 70.0  # 30% buffer for MTM

    def add_position(self, instrument: str, margin: float) -> None:
        self._positions[instrument] = margin

    def remove_position(self, instrument: str) -> None:
        self._positions.pop(instrument, None)

    def used_margin(self) -> float:
        return sum(self._positions.values())

    def available_margin(self) -> float:
        return self.capital - self.used_margin()

    def utilization_pct(self) -> float:
        if self.capital == 0:
            return 100.0
        return (self.used_margin() / self.capital) * 100

    def can_add(self, new_margin: float) -> bool:
        """Check if adding this margin stays within 70% utilization."""
        total = self.used_margin() + new_margin
        return (total / self.capital * 100) <= self._max_utilization_pct

    def theta_budget(self) -> float:
        """Max margin available for theta engine."""
        return self.capital * THETA_LIMITS["max_margin_pct"]

    def directional_budget(self) -> float:
        """Max margin available for directional trades."""
        max_deploy = self.capital * (self._max_utilization_pct / 100)
        theta_reserved = self.theta_budget()
        return max_deploy - theta_reserved

    @staticmethod
    def estimate_option_buy_margin(premium: float, lot_size: int) -> float:
        """For bought options, margin = total premium paid."""
        return premium * lot_size

    @staticmethod
    def estimate_spread_margin(strike_width: float, lot_size: int,
                                net_credit: float = 0) -> float:
        """For spreads, margin ≈ max loss = (width - credit) × lots."""
        return (strike_width - net_credit) * lot_size

    def to_dict(self) -> dict:
        return {
            "capital": self.capital,
            "positions": dict(self._positions),
            "used": self.used_margin(),
            "available": self.available_margin(),
            "utilization_pct": round(self.utilization_pct(), 1),
        }
