"""V7 Computed Levels — auto-computed pivot points and VWAP for intraday trading.

Computes:
- Previous day high/low/close (PDH/PDL/PDC)
- Classic pivot points (PP, R1, R2, S1, S2)
- Running VWAP with standard deviation bands
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ComputedLevels:
    """Holds pivot levels and running VWAP for a symbol."""

    symbol: str
    pdh: float  # previous day high
    pdl: float  # previous day low
    pdc: float  # previous day close
    pp: float   # pivot point = (pdh + pdl + pdc) / 3
    r1: float   # 2*pp - pdl
    r2: float   # pp + (pdh - pdl)
    s1: float   # 2*pp - pdh
    s2: float   # pp - (pdh - pdl)
    vwap: float = 0.0
    vwap_upper: float = 0.0  # VWAP + 1 std dev band
    vwap_lower: float = 0.0  # VWAP - 1 std dev band

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ComputedLevels:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    @classmethod
    def from_daily_candle(cls, symbol: str, high: float, low: float, close: float) -> ComputedLevels:
        """Create ComputedLevels from a single daily candle (previous day)."""
        pp = (high + low + close) / 3
        return cls(
            symbol=symbol,
            pdh=high,
            pdl=low,
            pdc=close,
            pp=pp,
            r1=2 * pp - low,
            r2=pp + (high - low),
            s1=2 * pp - high,
            s2=pp - (high - low),
        )


class LevelComputer:
    """Computes pivot levels and running VWAP for multiple symbols."""

    def __init__(self):
        self._levels: dict[str, ComputedLevels] = {}
        # VWAP state per symbol: cumulative TP*V, cumulative V, cumulative (TP-VWAP)^2 * V
        self._vwap_state: dict[str, dict] = {}

    def compute_pivots(self, symbol: str, pdh: float, pdl: float, pdc: float) -> ComputedLevels:
        """Compute PDH/PDL/PDC + classic pivot points."""
        levels = ComputedLevels.from_daily_candle(symbol, pdh, pdl, pdc)
        self._levels[symbol] = levels
        return levels

    def init_vwap(self, symbol: str) -> None:
        """Reset VWAP state for start of day."""
        self._vwap_state[symbol] = {
            "cum_tp_vol": 0.0,
            "cum_vol": 0,
            "cum_var_vol": 0.0,
        }

    def update_vwap(self, symbol: str, high: float, low: float, close: float, volume: int) -> float:
        """Update running VWAP with a new candle, return current VWAP.

        TP = (H + L + C) / 3
        VWAP = sum(TP * V) / sum(V)
        Upper band = VWAP + sqrt(sum((TP - VWAP)^2 * V) / sum(V))
        Lower band = VWAP - same
        """
        if symbol not in self._vwap_state:
            self.init_vwap(symbol)

        state = self._vwap_state[symbol]
        tp = (high + low + close) / 3

        state["cum_tp_vol"] += tp * volume
        state["cum_vol"] += volume

        if state["cum_vol"] == 0:
            return 0.0

        vwap = state["cum_tp_vol"] / state["cum_vol"]

        # Update variance: sum((TP - VWAP)^2 * V)
        # Note: we recompute from scratch would be ideal, but for running calc
        # we accumulate (TP - VWAP)^2 * V using the current VWAP
        state["cum_var_vol"] += (tp - vwap) ** 2 * volume

        std_dev = math.sqrt(state["cum_var_vol"] / state["cum_vol"])

        # Update levels if they exist for this symbol
        if symbol in self._levels:
            self._levels[symbol].vwap = vwap
            self._levels[symbol].vwap_upper = vwap + std_dev
            self._levels[symbol].vwap_lower = vwap - std_dev

        return vwap

    def get_levels(self, symbol: str) -> Optional[ComputedLevels]:
        """Return current levels for a symbol, or None if not computed."""
        return self._levels.get(symbol)

    def format_for_prompt(self, symbols: list[str]) -> dict:
        """Format all levels for strategist prompt."""
        result = {}
        for symbol in symbols:
            levels = self._levels.get(symbol)
            if levels:
                result[symbol] = levels.to_dict()
        return result
