# tests/test_v7_edge_tracker.py
"""Tests for V7 edge tracker."""
import json
import pytest
from datetime import date
from pathlib import Path
from v7.edge_tracker import EdgeTracker
from v7.types import TradeResult, SetupType


@pytest.fixture
def tmp_tracker(tmp_path):
    return EdgeTracker(data_dir=tmp_path)


@pytest.fixture
def sample_trade():
    return TradeResult(
        symbol="NIFTY", instrument="NIFTY CE 24400",
        direction="bullish", entry_price=100.0, exit_price=150.0,
        quantity=75, entry_date=date(2026, 3, 11),
        exit_date=date(2026, 3, 11), exit_reason="target",
        pnl=3750.0, pnl_pct=50.0, costs=120.0,
        setup_id="N1", setup_type=SetupType.BREAKOUT_LONG,
        entry_grade="A", exit_grade="A",
    )


def test_record_trade(tmp_tracker, sample_trade):
    tmp_tracker.record(sample_trade, strategy="momentum", time_bucket="9:45-11:00")
    stats = tmp_tracker.get_stats()
    assert stats["overall"]["trades"] == 1
    assert stats["overall"]["wins"] == 1


def test_record_loss(tmp_tracker):
    trade = TradeResult(
        symbol="HDFCBANK", instrument="HDFCBANK CE 1600",
        direction="bullish", entry_price=100.0, exit_price=70.0,
        quantity=550, entry_date=date(2026, 3, 11),
        exit_date=date(2026, 3, 11), exit_reason="stoploss",
        pnl=-16500.0, pnl_pct=-30.0, costs=80.0,
        setup_id="H1", setup_type=SetupType.SUPPORT_BOUNCE,
        entry_grade="B", exit_grade="C",
    )
    tmp_tracker.record(trade, strategy="mean_reversion", time_bucket="11:00-13:00")
    stats = tmp_tracker.get_stats()
    assert stats["overall"]["losses"] == 1
    assert stats["by_strategy"]["mean_reversion"]["win_rate"] == 0.0


def test_by_instrument(tmp_tracker, sample_trade):
    tmp_tracker.record(sample_trade, strategy="momentum", time_bucket="9:45-11:00")
    stats = tmp_tracker.get_stats()
    assert stats["by_instrument"]["NIFTY"]["net_pnl"] == 3750.0


def test_by_time(tmp_tracker, sample_trade):
    tmp_tracker.record(sample_trade, strategy="momentum", time_bucket="9:45-11:00")
    stats = tmp_tracker.get_stats()
    assert stats["by_time"]["9:45-11:00"]["trades"] == 1
    assert stats["by_time"]["9:45-11:00"]["win_rate"] == 1.0


def test_by_setup_type(tmp_tracker, sample_trade):
    tmp_tracker.record(sample_trade, strategy="momentum", time_bucket="9:45-11:00")
    stats = tmp_tracker.get_stats()
    assert stats["by_setup_type"]["breakout_long"]["trades"] == 1


def test_persistence(tmp_path, sample_trade):
    tracker1 = EdgeTracker(data_dir=tmp_path)
    tracker1.record(sample_trade, strategy="momentum", time_bucket="9:45-11:00")
    # New instance loads from disk
    tracker2 = EdgeTracker(data_dir=tmp_path)
    stats = tracker2.get_stats()
    assert stats["overall"]["trades"] == 1


def test_kill_candidates_not_enough_trades(tmp_tracker, sample_trade):
    # Less than 30 trades — no kill candidates
    tmp_tracker.record(sample_trade, strategy="momentum", time_bucket="9:45-11:00")
    assert tmp_tracker.kill_candidates(min_trades=30) == []


def test_kill_candidates_low_win_rate(tmp_path):
    tracker = EdgeTracker(data_dir=tmp_path)
    # Record 31 trades: 10 wins, 21 losses (32% win rate)
    for i in range(10):
        t = TradeResult(
            symbol="NIFTY", instrument="NIFTY CE", direction="bullish",
            entry_price=100.0, exit_price=120.0, quantity=75,
            entry_date=date(2026, 3, 1), exit_date=date(2026, 3, 1),
            exit_reason="target", pnl=1500.0, pnl_pct=20.0, costs=80.0,
            setup_id=f"N{i}", setup_type=SetupType.BREAKOUT_LONG,
        )
        tracker.record(t, strategy="momentum", time_bucket="9:45-11:00")
    for i in range(21):
        t = TradeResult(
            symbol="NIFTY", instrument="NIFTY CE", direction="bullish",
            entry_price=100.0, exit_price=70.0, quantity=75,
            entry_date=date(2026, 3, 1), exit_date=date(2026, 3, 1),
            exit_reason="stoploss", pnl=-2250.0, pnl_pct=-30.0, costs=80.0,
            setup_id=f"NL{i}", setup_type=SetupType.BREAKOUT_LONG,
        )
        tracker.record(t, strategy="momentum", time_bucket="9:45-11:00")
    kills = tracker.kill_candidates(min_trades=30, min_win_rate=0.40)
    assert "momentum" in kills


def test_avg_rr(tmp_path):
    tracker = EdgeTracker(data_dir=tmp_path)
    # Win: risk 30, reward 50 → R:R 1.67
    t1 = TradeResult(
        symbol="NIFTY", instrument="NIFTY CE", direction="bullish",
        entry_price=100.0, exit_price=150.0, quantity=75,
        entry_date=date(2026, 3, 1), exit_date=date(2026, 3, 1),
        exit_reason="target", pnl=3750.0, pnl_pct=50.0, costs=80.0,
        setup_id="N1", setup_type=SetupType.BREAKOUT_LONG,
    )
    # Loss: risk 30, lost 30
    t2 = TradeResult(
        symbol="NIFTY", instrument="NIFTY CE", direction="bullish",
        entry_price=100.0, exit_price=70.0, quantity=75,
        entry_date=date(2026, 3, 1), exit_date=date(2026, 3, 1),
        exit_reason="stoploss", pnl=-2250.0, pnl_pct=-30.0, costs=80.0,
        setup_id="N2", setup_type=SetupType.BREAKOUT_LONG,
    )
    tracker.record(t1, strategy="momentum", time_bucket="9:45-11:00")
    tracker.record(t2, strategy="momentum", time_bucket="9:45-11:00")
    stats = tracker.get_stats()
    assert stats["by_strategy"]["momentum"]["avg_rr"] > 0


def test_edge_summary_for_prompt(tmp_tracker, sample_trade):
    tmp_tracker.record(sample_trade, strategy="momentum", time_bucket="9:45-11:00")
    summary = tmp_tracker.summary_for_prompt()
    assert "momentum" in summary
    assert "NIFTY" in summary
