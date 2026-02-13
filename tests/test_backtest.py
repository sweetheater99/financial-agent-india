"""Tests for backtest engine pure functions (no API calls)."""

import json
from pathlib import Path

import pytest

# Add parent to path so imports work
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest import (
    is_hit,
    calculate_forward_returns,
    aggregate_stats,
    load_snapshot,
    list_snapshots,
)
from screener import save_snapshot, SNAPSHOT_DIR, ALL_SIGNAL_NAMES


# ---------------------------------------------------------------------------
# is_hit()
# ---------------------------------------------------------------------------

class TestIsHit:
    def test_bullish_positive_return(self):
        assert is_hit("bullish", 2.5) is True

    def test_bullish_negative_return(self):
        assert is_hit("bullish", -1.3) is False

    def test_bearish_negative_return(self):
        assert is_hit("bearish", -3.0) is True

    def test_bearish_positive_return(self):
        assert is_hit("bearish", 1.5) is False

    def test_zero_return_bullish(self):
        assert is_hit("bullish", 0.0) is False

    def test_zero_return_bearish(self):
        assert is_hit("bearish", 0.0) is False

    def test_unknown_direction(self):
        assert is_hit("sideways", 5.0) is False


# ---------------------------------------------------------------------------
# calculate_forward_returns()
# ---------------------------------------------------------------------------

class TestCalculateForwardReturns:
    def test_basic_returns(self):
        # candles: [timestamp, O, H, L, C, V]
        candles = [
            ["2026-02-12T00:00:00", 100, 105, 98, 102, 1000],  # day 0 (entry)
            ["2026-02-13T00:00:00", 102, 108, 101, 106, 1200],  # day 1
            ["2026-02-14T00:00:00", 106, 110, 104, 108, 900],   # day 2
            ["2026-02-17T00:00:00", 108, 112, 107, 111, 1100],  # day 3
            ["2026-02-18T00:00:00", 111, 115, 110, 113, 800],   # day 4
            ["2026-02-19T00:00:00", 113, 116, 112, 115, 950],   # day 5
        ]
        result = calculate_forward_returns(candles, trading_days=5)
        assert result is not None
        assert result["entry_price"] == 100
        # 1d: close of candle[1] = 106 -> (106-100)/100 * 100 = 6.0%
        assert result["returns"]["1d"] == 6.0
        # 3d: close of candle[3] = 111 -> 11.0%
        assert result["returns"]["3d"] == 11.0
        # 5d: close of candle[5] = 115 -> 15.0%
        assert result["returns"]["5d"] == 15.0

    def test_insufficient_data(self):
        # Only 1 candle, can't compute any forward return
        candles = [
            ["2026-02-12T00:00:00", 100, 105, 98, 102, 1000],
        ]
        result = calculate_forward_returns(candles, trading_days=5)
        assert result is None

    def test_empty_candles(self):
        assert calculate_forward_returns([], trading_days=5) is None

    def test_none_candles(self):
        assert calculate_forward_returns(None, trading_days=5) is None

    def test_two_candles_only_1d(self):
        candles = [
            ["2026-02-12T00:00:00", 100, 105, 98, 102, 1000],
            ["2026-02-13T00:00:00", 102, 108, 101, 110, 1200],
        ]
        result = calculate_forward_returns(candles, trading_days=5)
        assert result is not None
        assert "1d" in result["returns"]
        assert result["returns"]["1d"] == 10.0
        assert "3d" not in result["returns"]
        assert "5d" not in result["returns"]

    def test_negative_returns(self):
        candles = [
            ["2026-02-12T00:00:00", 100, 105, 98, 102, 1000],
            ["2026-02-13T00:00:00", 102, 103, 90, 92, 1200],
        ]
        result = calculate_forward_returns(candles, trading_days=5)
        assert result["returns"]["1d"] == -8.0

    def test_zero_entry_price(self):
        candles = [
            ["2026-02-12T00:00:00", 0, 105, 98, 102, 1000],
            ["2026-02-13T00:00:00", 102, 108, 101, 110, 1200],
        ]
        assert calculate_forward_returns(candles, trading_days=5) is None


# ---------------------------------------------------------------------------
# aggregate_stats()
# ---------------------------------------------------------------------------

class TestAggregateStats:
    def _make_result(self, symbol, direction, score, categories,
                     returns, hits):
        return {
            "symbol": symbol,
            "direction": direction,
            "score": score,
            "categories": categories,
            "price_change_pct": 1.0,
            "snapshot_date": "2026-02-12",
            "entry_price": 100,
            "returns": returns,
            "hits": hits,
        }

    def test_basic_aggregation(self):
        results = [
            self._make_result("A", "bullish", 5.0, ["LongBuildUp"],
                              {"1d": 2.0}, {"1d": True}),
            self._make_result("B", "bearish", 3.0, ["ShortBuildUp"],
                              {"1d": -1.5}, {"1d": True}),
            self._make_result("C", "bullish", 2.0, ["PercPriceGainers"],
                              {"1d": -0.5}, {"1d": False}),
        ]
        stats = aggregate_stats(results, trading_days=5)
        assert stats["total"] == 3

        h = stats["horizons"]["1d"]
        assert h["total"] == 3
        assert h["hits"] == 2
        assert h["rate"] == pytest.approx(66.7, abs=0.1)

    def test_direction_breakdown(self):
        results = [
            self._make_result("A", "bullish", 5.0, ["LongBuildUp"],
                              {"1d": 2.0}, {"1d": True}),
            self._make_result("B", "bullish", 3.0, ["PercPriceGainers"],
                              {"1d": -1.0}, {"1d": False}),
            self._make_result("C", "bearish", 4.0, ["ShortBuildUp"],
                              {"1d": -3.0}, {"1d": True}),
        ]
        stats = aggregate_stats(results, trading_days=5)
        h = stats["horizons"]["1d"]

        assert h["by_direction"]["bullish"]["rate"] == 50.0
        assert h["by_direction"]["bearish"]["rate"] == 100.0

    def test_score_tier_breakdown(self):
        results = [
            self._make_result("A", "bullish", 6.0, ["LongBuildUp"],
                              {"1d": 2.0}, {"1d": True}),
            self._make_result("B", "bullish", 1.0, ["PercPriceGainers"],
                              {"1d": -1.0}, {"1d": False}),
        ]
        stats = aggregate_stats(results, trading_days=5)
        h = stats["horizons"]["1d"]

        assert h["by_score_tier"][">=5"]["rate"] == 100.0
        assert h["by_score_tier"]["<2"]["rate"] == 0.0

    def test_empty_results(self):
        stats = aggregate_stats([], trading_days=5)
        assert stats["total"] == 0

    def test_signal_type_breakdown(self):
        results = [
            self._make_result("A", "bullish", 5.0, ["LongBuildUp", "PercPriceGainers"],
                              {"1d": 2.0}, {"1d": True}),
            self._make_result("B", "bullish", 3.0, ["LongBuildUp"],
                              {"1d": -1.0}, {"1d": False}),
        ]
        stats = aggregate_stats(results, trading_days=5)
        h = stats["horizons"]["1d"]

        assert h["by_signal"]["LongBuildUp"]["total"] == 2
        assert h["by_signal"]["LongBuildUp"]["hits"] == 1
        assert h["by_signal"]["LongBuildUp"]["rate"] == 50.0
        assert h["by_signal"]["PercPriceGainers"]["total"] == 1
        assert h["by_signal"]["PercPriceGainers"]["hits"] == 1


# ---------------------------------------------------------------------------
# save_snapshot()
# ---------------------------------------------------------------------------

class TestSaveSnapshot:
    def test_strips_details_and_candles(self, tmp_path, monkeypatch):
        monkeypatch.setattr("screener.SNAPSHOT_DIR", tmp_path)

        candidates = [
            {
                "symbol": "RELIANCE",
                "categories": ["LongBuildUp", "PercPriceGainers"],
                "score": 5.0,
                "direction": "bullish",
                "bullish_score": 5.0,
                "bearish_score": 0.0,
                "price_change_pct": 2.5,
                "details": {"LongBuildUp": {"ltp": 2800}},
                "candles": [[1, 2, 3, 4, 5, 6]],
            }
        ]
        signals = {"LongBuildUp": [{"tradingSymbol": "RELIANCE26FEB26FUT"}]}

        path = save_snapshot(candidates, signals)

        data = json.loads(path.read_text())
        assert "date" in data
        assert "timestamp" in data
        assert "signal_counts" in data
        assert "total_candidates" in data
        assert data["total_candidates"] == 1

        cand = data["candidates"][0]
        assert "details" not in cand
        assert "candles" not in cand
        assert cand["symbol"] == "RELIANCE"
        assert cand["score"] == 5.0

    def test_correct_schema_keys(self, tmp_path, monkeypatch):
        monkeypatch.setattr("screener.SNAPSHOT_DIR", tmp_path)

        candidates = [
            {
                "symbol": "TCS",
                "categories": ["ShortBuildUp"],
                "score": 2.0,
                "direction": "bearish",
                "bullish_score": 0.0,
                "bearish_score": 2.0,
                "price_change_pct": -1.5,
                "details": {},
            }
        ]
        signals = {name: [] for name in ALL_SIGNAL_NAMES}
        signals["ShortBuildUp"] = [{"tradingSymbol": "TCS26FEB26FUT"}]

        path = save_snapshot(candidates, signals)
        data = json.loads(path.read_text())

        assert set(data.keys()) == {"date", "timestamp", "signal_counts", "total_candidates", "candidates"}
        # Signal counts should have all signal types
        for name in ALL_SIGNAL_NAMES:
            assert name in data["signal_counts"]


# ---------------------------------------------------------------------------
# load_snapshot() / list_snapshots()
# ---------------------------------------------------------------------------

class TestSnapshotIO:
    def test_load_existing_snapshot(self, tmp_path, monkeypatch):
        monkeypatch.setattr("backtest.SNAPSHOT_DIR", tmp_path)

        snapshot = {
            "date": "2026-02-12",
            "timestamp": "2026-02-12T10:00:00",
            "signal_counts": {},
            "total_candidates": 1,
            "candidates": [{"symbol": "RELIANCE", "score": 5.0, "direction": "bullish",
                            "categories": ["LongBuildUp"]}],
        }
        (tmp_path / "2026-02-12.json").write_text(json.dumps(snapshot))

        loaded = load_snapshot("2026-02-12")
        assert loaded is not None
        assert loaded["date"] == "2026-02-12"
        assert len(loaded["candidates"]) == 1

    def test_load_nonexistent_snapshot(self, tmp_path, monkeypatch):
        monkeypatch.setattr("backtest.SNAPSHOT_DIR", tmp_path)
        assert load_snapshot("2099-01-01") is None

    def test_list_snapshots_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("backtest.SNAPSHOT_DIR", tmp_path)
        assert list_snapshots() == []

    def test_list_snapshots_sorted(self, tmp_path, monkeypatch):
        monkeypatch.setattr("backtest.SNAPSHOT_DIR", tmp_path)

        for date in ["2026-02-14", "2026-02-12", "2026-02-13"]:
            (tmp_path / f"{date}.json").write_text("{}")

        dates = list_snapshots()
        assert dates == ["2026-02-12", "2026-02-13", "2026-02-14"]
