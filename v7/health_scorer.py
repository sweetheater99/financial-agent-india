# v7/health_scorer.py
"""Position Health Scorer — 5-factor, 0-100 composite score.

Used by executor.py to drive graduated exits:
  - score < 30  → exit immediately
  - score < 50  → partial exit / tighten SL
  - score >= 70 → hold / trail

Factors and weights
-------------------
  progress    30%  — how far toward target relative to time elapsed
  momentum    25%  — last 5 candle close direction vs position direction
  premium     20%  — option premium retention (current / entry)
  volume      15%  — last 3 vs prior 3 candle volume ratio
  sl_distance 10%  — buffer between current price and SL
"""
from __future__ import annotations

from datetime import time
from typing import Sequence

from v7.config_v7 import HEALTH_SCORE
from v7.types import Position

# Candle: [timestamp, open, high, low, close, volume]
Candle = Sequence  # [ts, o, h, l, c, v] or dict with named keys

def _close(c) -> float:
    return c["close"] if isinstance(c, dict) else c[4]

def _volume(c) -> float:
    return c["volume"] if isinstance(c, dict) else c[5]

# Expected trade duration (minutes) by setup type
_SETUP_DURATIONS: dict[str, int] = {
    "breakout": 120,
    "bounce": 180,
    "fade": 180,
    "spread": 300,
    "default": 150,
}


# ── Helper ─────────────────────────────────────────────────────────────

def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _setup_duration(setup_type: str) -> int:
    key = setup_type.lower() if setup_type else "default"
    for k, v in _SETUP_DURATIONS.items():
        if k in key:
            return v
    return _SETUP_DURATIONS["default"]


# ── Factor 1: Progress Score (30%) ─────────────────────────────────────

def score_progress(
    pos: Position,
    current_price: float,
    current_time: time,
    setup_type: str = "default",
) -> float:
    """Score how much progress toward target relative to time elapsed.

    progress_ratio = (current - entry) / (target - entry)
    time_ratio     = age_minutes / expected_duration
    raw_score      = progress_ratio / time_ratio * 100

    Negative progress → 0. Time beyond duration → ratio capped at 1.
    """
    entry = pos.entry_price
    target = pos.target

    total_move = abs(target - entry)
    if total_move <= 0:
        return 50.0

    progress_ratio = (current_price - entry) / total_move
    if progress_ratio < 0:
        return 0.0

    expected_duration = _setup_duration(setup_type)
    age = pos.age_minutes(current_time)
    time_ratio = min(age / expected_duration, 1.0) if expected_duration > 0 else 1.0

    if time_ratio <= 0:
        # No time elapsed → can't judge, return neutral
        return 50.0

    raw = (progress_ratio / time_ratio) * 100
    return _clamp(raw)


# ── Factor 2: Momentum Score (25%) ─────────────────────────────────────

def score_momentum(pos: Position, candles: list[Candle]) -> float:
    """Score based on last 5 candle closes alignment with position direction.

    Each consecutive up-move scores positively for bullish, down-move for bearish.
    Recent candles weighted more (weights 1,2,3,4,5 for oldest→newest).
    Returns 50 if fewer than 5 candles (insufficient data).
    """
    if len(candles) < 5:
        return 50.0

    last5 = candles[-5:]
    closes = [_close(c) for c in last5]
    # Weights: oldest=1, newest=5
    weights = [1, 2, 3, 4, 5]
    total_weight = sum(weights[1:])  # 14 — we compare pairs

    bullish = pos.direction.lower() in ("bullish", "long", "buy")

    weighted_score = 0.0
    for i in range(1, 5):
        move = closes[i] - closes[i - 1]
        w = weights[i]
        if (bullish and move > 0) or (not bullish and move < 0):
            weighted_score += w
        elif move == 0:
            weighted_score += w * 0.5  # flat candle = half credit

    raw = (weighted_score / total_weight) * 100
    return _clamp(raw)


# ── Factor 3: Premium Score (20%) ──────────────────────────────────────

def score_premium(pos: Position, current_premium: float) -> float:
    """Score option premium retention.

    Linear scale: health=1.0 → 100, health=0.5 → 0.
    Clamped 0-100.
    Formula: (health - 0.5) / 0.5 * 100
    """
    health = pos.premium_health(current_premium)
    raw = (health - 0.5) / 0.5 * 100
    return _clamp(raw)


# ── Factor 4: Volume Score (15%) ───────────────────────────────────────

def score_volume(candles: list[Candle]) -> float:
    """Compare last 3 vs prior 3 candle volumes.

    ratio > 1 = increasing (good), < 1 = decreasing (bad).
    Score = 50 + (ratio - 1) * 100, clamped 0-100.
    Returns 50 if fewer than 6 candles (insufficient data).
    """
    if len(candles) < 6:
        return 50.0

    prior3 = candles[-6:-3]
    last3 = candles[-3:]

    prior_vol = sum(_volume(c) for c in prior3)
    last_vol = sum(_volume(c) for c in last3)

    if prior_vol <= 0:
        return 50.0

    ratio = last_vol / prior_vol
    raw = 50.0 + (ratio - 1.0) * 100.0
    return _clamp(raw)


# ── Factor 5: SL Distance Score (10%) ──────────────────────────────────

def score_sl_distance(pos: Position, current_price: float) -> float:
    """Score remaining buffer between current price and stoploss.

    At entry → 100. At SL → 0. Linear interpolation.
    For bullish: buffer = current - stoploss, range = entry - stoploss.
    For bearish: buffer = stoploss - current, range = stoploss - entry.
    """
    entry = pos.entry_price
    sl = pos.stoploss
    total_range = abs(entry - sl)

    if total_range == 0:
        return 50.0  # degenerate case

    buffer = current_price - sl
    # For bearish positions, SL is above entry → invert
    if sl > entry:
        buffer = sl - current_price

    raw = (buffer / total_range) * 100
    return _clamp(raw)


# ── Composite Score ─────────────────────────────────────────────────────

def compute_health_score(
    pos: Position,
    current_price: float,
    current_time: time,
    candles: list[Candle],
    current_premium: float,
    setup_type: str = "default",
) -> float:
    """Compute weighted composite health score (0-100, 1 decimal place).

    Factors and weights from HEALTH_SCORE config:
      progress    30%
      momentum    25%
      premium     20%
      volume      15%
      sl_distance 10%
    """
    w = HEALTH_SCORE

    s_progress = score_progress(pos, current_price, current_time, setup_type)
    s_momentum = score_momentum(pos, candles)
    s_premium = score_premium(pos, current_premium)
    s_volume = score_volume(candles)
    s_sl = score_sl_distance(pos, current_price)

    composite = (
        s_progress * w["progress_weight"]
        + s_momentum * w["momentum_weight"]
        + s_premium * w["premium_weight"]
        + s_volume * w["volume_weight"]
        + s_sl * w["sl_distance_weight"]
    )
    return round(_clamp(composite), 1)


# ── Class Wrapper ──────────────────────────────────────────────────────

class PositionHealthScorer:
    """Thin wrapper around compute_health_score for use by executor."""

    def compute(
        self,
        pos: Position,
        underlying_price: float,
        option_price: float,
        current_time: time,
        candles: list[Candle],
        setup_type: str = "default",
    ) -> float:
        return compute_health_score(
            pos, underlying_price, current_time, candles, option_price, setup_type
        )
