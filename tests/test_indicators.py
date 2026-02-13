"""
Tests for the shared indicators module.

Pure function tests — no API calls. Validates RSI, volume_ratio, and ATR
against known values and edge cases.
"""

import pytest

from indicators import compute_rsi, compute_volume_ratio, compute_atr


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

class TestComputeRSI:
    def _make_candles_from_closes(self, closes: list) -> list:
        """Build minimal candle list from close prices."""
        return [[f"2026-01-{i+1:02d}", c, c, c, c, 100000] for i, c in enumerate(closes)]

    def test_rsi_all_gains(self):
        """All gains → RSI = 100."""
        closes = list(range(100, 120))  # 20 ascending prices
        candles = self._make_candles_from_closes(closes)
        rsi = compute_rsi(candles, 14)
        assert rsi == 100.0

    def test_rsi_all_losses(self):
        """All losses → RSI = 0."""
        closes = list(range(120, 100, -1))  # 20 descending prices
        candles = self._make_candles_from_closes(closes)
        rsi = compute_rsi(candles, 14)
        assert rsi == 0.0

    def test_rsi_range(self):
        """RSI should always be between 0 and 100."""
        closes = [100, 102, 99, 103, 98, 105, 97, 106, 95, 107,
                  94, 108, 93, 109, 92, 110]
        candles = self._make_candles_from_closes(closes)
        rsi = compute_rsi(candles, 14)
        assert rsi is not None
        assert 0 <= rsi <= 100

    def test_rsi_known_values(self):
        """RSI with equal gains and losses → ~50."""
        # Alternating +1, -1 → equal avg gain/loss → RSI ≈ 50
        closes = [100]
        for i in range(20):
            closes.append(closes[-1] + (1 if i % 2 == 0 else -1))
        candles = self._make_candles_from_closes(closes)
        rsi = compute_rsi(candles, 14)
        assert rsi is not None
        assert 45 <= rsi <= 55

    def test_rsi_insufficient_data(self):
        """Less than period+1 candles returns None."""
        candles = self._make_candles_from_closes([100, 101, 102])
        assert compute_rsi(candles, 14) is None

    def test_rsi_empty(self):
        assert compute_rsi([], 14) is None
        assert compute_rsi(None, 14) is None

    def test_rsi_exact_minimum_candles(self):
        """Exactly period+1 candles should work."""
        closes = list(range(100, 116))  # 16 values = 15+1
        candles = self._make_candles_from_closes(closes)
        rsi = compute_rsi(candles, 15)
        assert rsi is not None


# ---------------------------------------------------------------------------
# Volume Ratio
# ---------------------------------------------------------------------------

class TestComputeVolumeRatio:
    def _make_candles_with_volumes(self, volumes: list) -> list:
        return [[f"2026-01-{i+1:02d}", 100, 105, 95, 102, v]
                for i, v in enumerate(volumes)]

    def test_double_volume(self):
        """Today's volume is 2x average → ratio = 2.0."""
        volumes = [100000] * 20 + [200000]
        candles = self._make_candles_with_volumes(volumes)
        ratio = compute_volume_ratio(candles, 20)
        assert ratio == 2.0

    def test_half_volume(self):
        """Today's volume is half average → ratio = 0.5."""
        volumes = [100000] * 20 + [50000]
        candles = self._make_candles_with_volumes(volumes)
        ratio = compute_volume_ratio(candles, 20)
        assert ratio == 0.5

    def test_normal_volume(self):
        """Same volume → ratio = 1.0."""
        volumes = [100000] * 21
        candles = self._make_candles_with_volumes(volumes)
        ratio = compute_volume_ratio(candles, 20)
        assert ratio == 1.0

    def test_insufficient_data(self):
        """Less than lookback+1 candles returns None."""
        volumes = [100000] * 5
        candles = self._make_candles_with_volumes(volumes)
        assert compute_volume_ratio(candles, 20) is None

    def test_empty(self):
        assert compute_volume_ratio([], 20) is None
        assert compute_volume_ratio(None, 20) is None

    def test_zero_avg_volume(self):
        """Zero average volume returns None (avoid division by zero)."""
        volumes = [0] * 20 + [100000]
        candles = self._make_candles_with_volumes(volumes)
        assert compute_volume_ratio(candles, 20) is None


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

class TestComputeATR:
    def _make_candles(self, n=20):
        """Generate synthetic daily candles for ATR testing."""
        candles = []
        price = 100
        for i in range(n):
            o = price
            h = price + 3
            l = price - 2
            c = price + 1
            candles.append([f"2026-01-{i+1:02d}", o, h, l, c, 100000])
            price = c
        return candles

    def test_atr_computation(self):
        """ATR from known candles produces positive value."""
        candles = self._make_candles(20)
        atr = compute_atr(candles, 14)
        assert atr is not None
        assert atr > 0

    def test_insufficient_data(self):
        """Less data than period+1 returns None."""
        candles = self._make_candles(5)
        assert compute_atr(candles, 14) is None

    def test_empty(self):
        assert compute_atr([], 14) is None
        assert compute_atr(None, 14) is None

    def test_known_values(self):
        """ATR with specific candle data."""
        candles = [
            ["2026-01-01", 100, 105, 95, 102, 100000],
            ["2026-01-02", 102, 108, 98, 106, 100000],
            ["2026-01-03", 106, 110, 100, 104, 100000],
            ["2026-01-04", 104, 112, 101, 109, 100000],
            ["2026-01-05", 109, 115, 105, 113, 100000],
            ["2026-01-06", 113, 118, 108, 116, 100000],
        ]
        atr = compute_atr(candles, 5)
        assert atr is not None
        assert atr > 0

    def test_matches_paper_trade_atr(self):
        """indicators.compute_atr should give same result as paper_trade.compute_atr."""
        from paper_trade import compute_atr as pt_atr
        candles = self._make_candles(20)
        assert compute_atr(candles, 14) == pt_atr(candles, 14)


# ---------------------------------------------------------------------------
# Signal Persistence
# ---------------------------------------------------------------------------

class TestSignalPersistence:
    """Tests for load_previous_snapshots and compute_persistence in screener.py."""

    def test_compute_persistence_3_consecutive_days(self):
        """Symbol present same direction in all 3 previous snapshots → persistence_days=3."""
        from screener import compute_persistence
        candidates = [{"symbol": "RELIANCE", "direction": "bullish", "score": 5.0}]
        prev_snapshots = [
            {"candidates": [{"symbol": "RELIANCE", "direction": "bullish"}]},
            {"candidates": [{"symbol": "RELIANCE", "direction": "bullish"}]},
            {"candidates": [{"symbol": "RELIANCE", "direction": "bullish"}]},
        ]
        compute_persistence(candidates, prev_snapshots)
        assert candidates[0]["persistence_days"] == 3

    def test_compute_persistence_2_consecutive_days(self):
        """Symbol present 2 days then absent → persistence_days=2."""
        from screener import compute_persistence
        candidates = [{"symbol": "TCS", "direction": "bearish", "score": 4.0}]
        prev_snapshots = [
            {"candidates": [{"symbol": "TCS", "direction": "bearish"}]},
            {"candidates": [{"symbol": "TCS", "direction": "bearish"}]},
            {"candidates": [{"symbol": "OTHER", "direction": "bullish"}]},
        ]
        compute_persistence(candidates, prev_snapshots)
        assert candidates[0]["persistence_days"] == 2

    def test_compute_persistence_direction_change_breaks_streak(self):
        """Direction flip on day 2 → only 1 day of persistence."""
        from screener import compute_persistence
        candidates = [{"symbol": "HDFCBANK", "direction": "bullish", "score": 3.0}]
        prev_snapshots = [
            {"candidates": [{"symbol": "HDFCBANK", "direction": "bullish"}]},
            {"candidates": [{"symbol": "HDFCBANK", "direction": "bearish"}]},
            {"candidates": [{"symbol": "HDFCBANK", "direction": "bullish"}]},
        ]
        compute_persistence(candidates, prev_snapshots)
        assert candidates[0]["persistence_days"] == 1

    def test_compute_persistence_not_found(self):
        """Symbol not in any previous snapshot → persistence_days=0."""
        from screener import compute_persistence
        candidates = [{"symbol": "NEWSTOCK", "direction": "bullish", "score": 2.0}]
        prev_snapshots = [
            {"candidates": [{"symbol": "OTHER", "direction": "bullish"}]},
        ]
        compute_persistence(candidates, prev_snapshots)
        assert candidates[0]["persistence_days"] == 0

    def test_compute_persistence_empty_snapshots(self):
        """No previous snapshots → persistence_days=0."""
        from screener import compute_persistence
        candidates = [{"symbol": "RELIANCE", "direction": "bullish", "score": 5.0}]
        compute_persistence(candidates, [])
        assert candidates[0]["persistence_days"] == 0

    def test_load_previous_snapshots_excludes_today(self, tmp_path, monkeypatch):
        """Today's snapshot should not be included."""
        import screener
        from screener import load_previous_snapshots
        from datetime import datetime
        monkeypatch.setattr(screener, "SNAPSHOT_DIR", tmp_path)

        today = datetime.now().strftime("%Y-%m-%d")
        # Create today's snapshot
        (tmp_path / f"{today}.json").write_text('{"candidates": []}')
        # Create yesterday's snapshot
        from datetime import timedelta
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        (tmp_path / f"{yesterday}.json").write_text('{"candidates": [{"symbol": "A", "direction": "bullish"}]}')

        result = load_previous_snapshots()
        assert len(result) == 1
        assert result[0]["candidates"][0]["symbol"] == "A"
