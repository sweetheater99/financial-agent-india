"""Tests for expected_move.py -- uses synthetic data, no API calls."""

import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from expected_move import (
    compute_expected_move,
    record_daily_move,
    get_move_accuracy,
    generate_move_report,
    _load_moves,
    _save_moves,
    MOVES_FILE,
)


class TestComputeExpectedMove:
    """Test 1: Verify expected move math."""

    def test_basic_calculation(self):
        # VIX=15, spot=22500, 1 day
        # Expected = 22500 * (15/100) / sqrt(252) * sqrt(1)
        expected = 22500 * 0.15 / math.sqrt(252)
        result = compute_expected_move(15.0, 22500.0, days=1)
        assert abs(result - expected) < 0.01

    def test_multi_day(self):
        # 5-day expected move should be sqrt(5) times 1-day
        one_day = compute_expected_move(15.0, 22500.0, days=1)
        five_day = compute_expected_move(15.0, 22500.0, days=5)
        assert abs(five_day - one_day * math.sqrt(5)) < 0.01

    def test_zero_vix(self):
        assert compute_expected_move(0.0, 22500.0) == 0.0

    def test_zero_spot(self):
        assert compute_expected_move(15.0, 0.0) == 0.0

    def test_negative_days(self):
        assert compute_expected_move(15.0, 22500.0, days=-1) == 0.0


class TestRecordDailyMove:
    """Test 2: Verify record saving."""

    def test_record_saved(self, tmp_path):
        mock_file = tmp_path / "expected_moves.json"
        with patch("expected_move.MOVES_FILE", mock_file), \
             patch("expected_move.DATA_DIR", tmp_path):
            entry = record_daily_move(22500.0, 22580.0, 14.2, "2026-03-07")
            assert entry is not None
            assert entry["date"] == "2026-03-07"
            assert entry["actual_move"] == 80.0
            assert entry["vix"] == 14.2
            assert entry["expected_move"] > 0

            # Verify file was written
            data = json.loads(mock_file.read_text())
            assert len(data) == 1
            assert data[0]["date"] == "2026-03-07"

    def test_duplicate_date_overwrites(self, tmp_path):
        mock_file = tmp_path / "expected_moves.json"
        with patch("expected_move.MOVES_FILE", mock_file), \
             patch("expected_move.DATA_DIR", tmp_path):
            record_daily_move(22500.0, 22580.0, 14.2, "2026-03-07")
            record_daily_move(22500.0, 22600.0, 14.5, "2026-03-07")
            data = json.loads(mock_file.read_text())
            assert len(data) == 1
            assert data[0]["actual_move"] == 100.0

    def test_invalid_inputs(self):
        result = record_daily_move(0.0, 22500.0, 14.2)
        assert result is None


class TestAccuracyStrongEdge:
    """Test 3: Strong seller edge when actual < expected consistently."""

    def test_strong_edge(self, tmp_path):
        mock_file = tmp_path / "expected_moves.json"
        # Create data where actual moves are ~60% of expected (strong edge)
        moves = []
        for i in range(30):
            moves.append({
                "date": f"2026-02-{i+1:02d}",
                "spot_open": 22500.0,
                "spot_close": 22560.0,
                "actual_move": 60.0,
                "expected_move": 100.0,
                "vix": 14.0,
                "ratio": 0.60,
            })
        mock_file.write_text(json.dumps(moves))

        with patch("expected_move.MOVES_FILE", mock_file):
            accuracy = get_move_accuracy(30)
            assert accuracy["days_analyzed"] == 30
            assert accuracy["avg_ratio"] == 0.6
            assert accuracy["seller_edge"] is True
            assert accuracy["pct_within_expected"] == 100
            assert accuracy["avg_expected"] == 100.0
            assert accuracy["avg_actual"] == 60.0


class TestAccuracyNoEdge:
    """Test 4: No seller edge when actual ~ expected."""

    def test_no_edge(self, tmp_path):
        mock_file = tmp_path / "expected_moves.json"
        moves = []
        for i in range(30):
            moves.append({
                "date": f"2026-02-{i+1:02d}",
                "spot_open": 22500.0,
                "spot_close": 22600.0,
                "actual_move": 100.0,
                "expected_move": 100.0,
                "vix": 14.0,
                "ratio": 1.0,
            })
        mock_file.write_text(json.dumps(moves))

        with patch("expected_move.MOVES_FILE", mock_file):
            accuracy = get_move_accuracy(30)
            assert accuracy["seller_edge"] is False
            assert accuracy["avg_ratio"] == 1.0


class TestAccuracyEmptyData:
    """Test 5: No data file, verify graceful handling."""

    def test_empty(self, tmp_path):
        mock_file = tmp_path / "nonexistent.json"
        with patch("expected_move.MOVES_FILE", mock_file):
            accuracy = get_move_accuracy(30)
            assert accuracy["days_analyzed"] == 0
            assert accuracy["seller_edge"] is False
            assert accuracy["avg_ratio"] == 0.0


class TestReportFormat:
    """Test 6: Verify HTML report contains key sections."""

    def test_report_with_data(self, tmp_path):
        mock_file = tmp_path / "expected_moves.json"
        moves = []
        for i in range(30):
            moves.append({
                "date": f"2026-02-{i+1:02d}",
                "spot_open": 22500.0,
                "spot_close": 22560.0,
                "actual_move": 60.0,
                "expected_move": 100.0,
                "vix": 14.0,
                "ratio": 0.60,
            })
        mock_file.write_text(json.dumps(moves))

        with patch("expected_move.MOVES_FILE", mock_file):
            report = generate_move_report()
            assert "Expected Move Accuracy" in report
            assert "Avg Expected" in report
            assert "Avg Actual" in report
            assert "IV Overpricing" in report
            assert "Days Within Expected" in report
            assert "Verdict" in report

    def test_report_no_data(self, tmp_path):
        mock_file = tmp_path / "nonexistent.json"
        with patch("expected_move.MOVES_FILE", mock_file):
            report = generate_move_report()
            assert "No data recorded" in report
