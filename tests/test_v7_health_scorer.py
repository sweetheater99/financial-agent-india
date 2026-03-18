# tests/test_v7_health_scorer.py
"""Tests for v7/health_scorer.py — Position Health Scoring System.

Follows TDD: tests written first, then implementation.
"""
import unittest
from datetime import date, time

from v7.types import Position
from v7.health_scorer import (
    score_progress,
    score_momentum,
    score_premium,
    score_volume,
    score_sl_distance,
    compute_health_score,
)


# ── Test helpers ───────────────────────────────────────────────────────

def _make_position(**kwargs) -> Position:
    defaults = dict(
        symbol="RELIANCE", instrument="RELIANCE26MAR1420CE",
        direction="bullish", entry_price=15.75, quantity=500,
        lot_size=250, allocated=7875.0,
        stoploss=1380.0, target=1460.0,
        entry_date=date(2026, 3, 18), setup_id="S3",
        entry_time=time(10, 0),
    )
    defaults.update(kwargs)
    return Position(**defaults)


def _candles(closes: list[float], volumes: list[float] | None = None) -> list[list]:
    """Build minimal candle list: [ts, open, high, low, close, volume]."""
    if volumes is None:
        volumes = [1000.0] * len(closes)
    return [
        [f"ts{i}", c - 0.5, c + 0.5, c - 1.0, c, v]
        for i, (c, v) in enumerate(zip(closes, volumes))
    ]


# ── Factor 1: Progress Score ───────────────────────────────────────────

class TestProgressScore(unittest.TestCase):
    """Progress score = how far toward target vs. time elapsed (weight 30%)."""

    def test_good_progress(self):
        """Position at 60% of target with only 30% of time elapsed → high score."""
        pos = _make_position(
            entry_price=10.0, target=20.0, stoploss=5.0,
            entry_time=time(9, 15),
        )
        # setup_id "S3" → no special type → default=150min
        # 30% of 150min elapsed → current_time = 9:15 + 45min = 10:00
        current_price = 16.0  # (16-10)/(20-10) = 60% progress
        current_time = time(10, 0)  # 45 min elapsed → 30% time
        score = score_progress(pos, current_price, current_time, setup_type="breakout")
        # breakout=120min; 45min elapsed → time_ratio=0.375; progress_ratio=0.6
        # raw = 0.6/0.375 = 1.6 → capped at 100
        self.assertGreater(score, 80)

    def test_no_progress_late(self):
        """Position at 10% of target with 80% of time elapsed → low score."""
        pos = _make_position(
            entry_price=10.0, target=20.0, stoploss=5.0,
            entry_time=time(9, 15),
        )
        # breakout=120min; need 80% elapsed → 96 min → current=10:51
        current_price = 11.0  # 10% progress
        current_time = time(10, 51)  # 96 min elapsed → 80% time
        score = score_progress(pos, current_price, current_time, setup_type="breakout")
        # progress_ratio=0.1, time_ratio=0.8 → raw=0.125 → score=12.5
        self.assertLess(score, 30)

    def test_negative_progress(self):
        """Price below entry (bearish drift for bullish) → very low score."""
        pos = _make_position(
            entry_price=10.0, target=20.0, stoploss=5.0,
            entry_time=time(9, 15),
        )
        current_price = 8.0  # below entry, negative progress
        current_time = time(10, 30)  # 75 min → 62.5% time (120min breakout)
        score = score_progress(pos, current_price, current_time, setup_type="breakout")
        self.assertLess(score, 15)


# ── Factor 2: Momentum Score ──────────────────────────────────────────

class TestMomentumScore(unittest.TestCase):
    """Momentum score = last 5 candle closes alignment with direction (weight 25%)."""

    def test_bullish_with_rising_candles(self):
        """Bullish position + strongly rising closes → high momentum."""
        pos = _make_position(direction="bullish")
        candles = _candles([10.0, 11.0, 12.5, 14.0, 16.0])
        score = score_momentum(pos, candles)
        self.assertGreater(score, 80)

    def test_bullish_with_falling_candles(self):
        """Bullish position + falling closes → low momentum."""
        pos = _make_position(direction="bullish")
        candles = _candles([16.0, 14.0, 12.0, 10.5, 9.0])
        score = score_momentum(pos, candles)
        self.assertLess(score, 30)

    def test_bearish_with_falling_candles(self):
        """Bearish position + falling closes → high momentum."""
        pos = _make_position(direction="bearish")
        candles = _candles([16.0, 14.5, 13.0, 11.0, 9.0])
        score = score_momentum(pos, candles)
        self.assertGreater(score, 70)

    def test_fewer_than_5_candles(self):
        """Handles fewer than 5 candles gracefully (returns neutral 50)."""
        pos = _make_position(direction="bullish")
        candles = _candles([10.0, 11.0])
        score = score_momentum(pos, candles)
        self.assertEqual(score, 50.0)

    def test_empty_candles(self):
        """No candles → returns neutral 50."""
        pos = _make_position(direction="bullish")
        score = score_momentum(pos, [])
        self.assertEqual(score, 50.0)


# ── Factor 3: Premium Score ───────────────────────────────────────────

class TestPremiumScore(unittest.TestCase):
    """Premium score = pos.premium_health() mapped linearly 1.0→100, 0.5→0 (weight 20%)."""

    def test_healthy_premium(self):
        """Current premium close to entry → high score (≥80)."""
        pos = _make_position(entry_price=20.0)
        # premium_health = 18/20 = 0.9 → score = (0.9 - 0.5) / 0.5 * 100 = 80
        score = score_premium(pos, current_premium=18.0)
        self.assertGreaterEqual(score, 80)

    def test_full_premium(self):
        """Current premium == entry → score = 100."""
        pos = _make_position(entry_price=20.0)
        score = score_premium(pos, current_premium=20.0)
        self.assertAlmostEqual(score, 100.0, places=1)

    def test_eroded_premium(self):
        """Current premium at half entry → score near 0."""
        pos = _make_position(entry_price=20.0)
        score = score_premium(pos, current_premium=10.0)
        self.assertLess(score, 10)

    def test_premium_above_entry(self):
        """Premium above entry → score capped at 100."""
        pos = _make_position(entry_price=20.0)
        score = score_premium(pos, current_premium=30.0)
        self.assertAlmostEqual(score, 100.0, places=1)

    def test_zero_premium(self):
        """Premium at 0 → score clamped at 0."""
        pos = _make_position(entry_price=20.0)
        score = score_premium(pos, current_premium=0.0)
        self.assertAlmostEqual(score, 0.0, places=1)


# ── Factor 4: Volume Score ────────────────────────────────────────────

class TestVolumeScore(unittest.TestCase):
    """Volume score compares last 3 candles to prior 3 (weight 15%)."""

    def test_increasing_volume(self):
        """Last 3 candles have double the prior 3 volume → high score."""
        vols = [1000, 1000, 1000, 2000, 2000, 2000]
        candles = _candles([10.0] * 6, volumes=vols)
        score = score_volume(candles)
        # ratio = 2.0 → 50 + (2-1)*100 = 150 → clamped to 100
        self.assertAlmostEqual(score, 100.0, places=1)

    def test_equal_volume(self):
        """Equal volume → score = 50."""
        vols = [1000] * 6
        candles = _candles([10.0] * 6, volumes=vols)
        score = score_volume(candles)
        self.assertAlmostEqual(score, 50.0, places=1)

    def test_decreasing_volume(self):
        """Last 3 candles half the volume of prior 3 → low score."""
        vols = [2000, 2000, 2000, 1000, 1000, 1000]
        candles = _candles([10.0] * 6, volumes=vols)
        score = score_volume(candles)
        # ratio = 0.5 → 50 + (0.5-1)*100 = 0 → clamped to 0
        self.assertAlmostEqual(score, 0.0, places=1)

    def test_fewer_than_6_candles(self):
        """Fewer than 6 candles → returns neutral 50."""
        candles = _candles([10.0] * 4)
        score = score_volume(candles)
        self.assertEqual(score, 50.0)


# ── Factor 5: SL Distance Score ──────────────────────────────────────

class TestSLDistanceScore(unittest.TestCase):
    """SL distance: at entry=100, at SL=0 (weight 10%).

    Uses underlying price for SL (not option premium); stoploss/target
    in Position are underlying-level prices.
    """

    def test_near_entry(self):
        """Current price just above entry → high buffer remaining."""
        pos = _make_position(entry_price=15.75, stoploss=1380.0, target=1460.0)
        # Range = entry - stoploss; use current underlying price
        # Let's use a simplified Position where stoploss is option-level
        pos2 = _make_position(entry_price=15.75, stoploss=5.0, target=25.0)
        # At entry: buffer = 15.75 - 5 = 10.75; range = 10.75
        score = score_sl_distance(pos2, current_price=15.0)
        # buffer = 15.0 - 5.0 = 10; range = 15.75 - 5.0 = 10.75
        # ratio = 10/10.75 ≈ 0.93 → score ≈ 93
        self.assertGreater(score, 70)

    def test_near_stoploss(self):
        """Current price approaching SL → very low score."""
        pos = _make_position(entry_price=15.75, stoploss=5.0, target=25.0)
        # buffer = 5.5 - 5.0 = 0.5; range = 10.75
        score = score_sl_distance(pos, current_price=5.5)
        self.assertLess(score, 20)

    def test_at_entry(self):
        """Current price at entry → score = 100."""
        pos = _make_position(entry_price=15.75, stoploss=5.0, target=25.0)
        score = score_sl_distance(pos, current_price=15.75)
        self.assertAlmostEqual(score, 100.0, places=1)

    def test_at_stoploss(self):
        """Current price at SL → score = 0."""
        pos = _make_position(entry_price=15.75, stoploss=5.0, target=25.0)
        score = score_sl_distance(pos, current_price=5.0)
        self.assertAlmostEqual(score, 0.0, places=1)

    def test_zero_range(self):
        """Entry == SL → returns 50 (degenerate case)."""
        pos = _make_position(entry_price=10.0, stoploss=10.0, target=20.0)
        score = score_sl_distance(pos, current_price=10.0)
        self.assertEqual(score, 50.0)


# ── Composite Score ───────────────────────────────────────────────────

class TestCompositeScore(unittest.TestCase):
    """Weighted sum of all 5 factors, clamped 0-100, rounded to 1 dp."""

    def test_healthy_position(self):
        """All factors healthy → composite > 70."""
        pos = _make_position(
            entry_price=10.0, target=20.0, stoploss=5.0,
            entry_time=time(9, 15), direction="bullish",
        )
        # Good progress (bullish, price near target)
        current_price = 16.0
        current_time = time(9, 45)  # 30min, breakout=120min → 25% elapsed
        candles = _candles(
            [10.5, 11.5, 13.0, 14.5, 16.0],
            volumes=[1000, 1000, 1000, 2000, 2000],
        )
        score = compute_health_score(
            pos, current_price, current_time,
            candles=candles,
            current_premium=9.5,  # entry=10, health=0.95
            setup_type="breakout",
        )
        self.assertGreater(score, 70)

    def test_dying_position(self):
        """All factors bad → composite < 30."""
        pos = _make_position(
            entry_price=15.0, target=25.0, stoploss=5.0,
            entry_time=time(9, 15), direction="bullish",
        )
        # Very late, no progress, falling candles, eroded premium, near SL
        current_price = 6.0  # near SL
        current_time = time(11, 15)  # 120min → 100% of breakout duration
        candles = _candles(
            [14.0, 12.0, 10.0, 8.0, 6.0],
            volumes=[2000, 2000, 2000, 500, 500],
        )
        score = compute_health_score(
            pos, current_price, current_time,
            candles=candles,
            current_premium=6.0,  # entry=15, health=0.4 → score=0
            setup_type="breakout",
        )
        self.assertLess(score, 30)

    def test_score_is_rounded(self):
        """Score is always a float rounded to 1 decimal place."""
        pos = _make_position(entry_price=10.0, target=20.0, stoploss=5.0)
        candles = _candles([10.0] * 5)
        score = compute_health_score(
            pos, 10.0, time(10, 30),
            candles=candles, current_premium=10.0, setup_type="default",
        )
        self.assertEqual(score, round(score, 1))

    def test_score_clamped_0_100(self):
        """Score never goes below 0 or above 100."""
        pos = _make_position(entry_price=10.0, target=20.0, stoploss=5.0)
        candles = _candles([20.0, 25.0, 30.0, 35.0, 40.0])
        score = compute_health_score(
            pos, 20.0, time(9, 20),
            candles=candles, current_premium=25.0, setup_type="breakout",
        )
        self.assertLessEqual(score, 100.0)
        self.assertGreaterEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
