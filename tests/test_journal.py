"""Tests for trade journal: entry thesis, exit review, and analytics."""

import sys
import json
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import journal


def _use_tmp_journal(tmp_path):
    """Redirect journal storage to a temp directory."""
    journal.JOURNAL_DIR = tmp_path
    journal.JOURNAL_FILE = tmp_path / "trades.json"


def test_record_entry(tmp_path):
    _use_tmp_journal(tmp_path)
    entry = journal.record_entry(
        symbol="RELIANCE", direction="bullish", instrument="EQ",
        entry_price=2500, entry_date="2026-03-01",
        regime="trending_up", score=7.5, allocation=10000,
    )
    assert entry["symbol"] == "RELIANCE"
    assert entry["thesis"]["regime"] == "trending_up"
    assert entry["thesis"]["score"] == 7.5
    assert entry["exit"] is None

    # Verify persisted
    data = json.loads(journal.JOURNAL_FILE.read_text())
    assert len(data) == 1


def test_record_exit_found(tmp_path):
    _use_tmp_journal(tmp_path)
    journal.record_entry(
        symbol="TCS", direction="bullish", instrument="EQ",
        entry_price=3500, entry_date="2026-03-01",
    )
    result = journal.record_exit(
        symbol="TCS", entry_date="2026-03-01", direction="bullish",
        exit_price=3600, exit_date="2026-03-05", exit_reason="target",
        pnl=400, pnl_pct=2.86, costs=20, holding_days=4,
    )
    assert result is not None
    assert result["exit"]["exit_reason"] == "target"
    assert result["exit"]["thesis_played_out"] is True


def test_record_exit_stoploss_thesis_wrong(tmp_path):
    _use_tmp_journal(tmp_path)
    journal.record_entry(
        symbol="INFY", direction="bullish", instrument="EQ",
        entry_price=1500, entry_date="2026-03-01",
    )
    result = journal.record_exit(
        symbol="INFY", entry_date="2026-03-01", direction="bullish",
        exit_price=1400, exit_date="2026-03-03", exit_reason="stoploss",
        pnl=-400, pnl_pct=-6.67,
    )
    assert result["exit"]["thesis_played_out"] is False


def test_record_exit_not_found(tmp_path):
    _use_tmp_journal(tmp_path)
    result = journal.record_exit(
        symbol="GHOST", entry_date="2026-01-01", direction="bullish",
        exit_price=100, exit_date="2026-03-01", exit_reason="manual",
        pnl=0, pnl_pct=0,
    )
    assert result is None


def test_journal_stats_empty(tmp_path):
    _use_tmp_journal(tmp_path)
    stats = journal.get_journal_stats()
    assert stats["total_entries"] == 0
    assert stats["closed"] == 0


def test_journal_stats_with_trades(tmp_path):
    _use_tmp_journal(tmp_path)
    # Two entries, one closed with profit, one still open
    journal.record_entry(symbol="A", direction="bullish", instrument="EQ",
                         entry_price=100, entry_date="2026-03-01", regime="trending_up")
    journal.record_entry(symbol="B", direction="bullish", instrument="EQ",
                         entry_price=200, entry_date="2026-03-01", regime="ranging")
    journal.record_exit(symbol="A", entry_date="2026-03-01", direction="bullish",
                        exit_price=110, exit_date="2026-03-05", exit_reason="target",
                        pnl=100, pnl_pct=10.0)

    stats = journal.get_journal_stats()
    assert stats["total_entries"] == 2
    assert stats["closed"] == 1
    assert stats["open"] == 1
    assert stats["thesis_accuracy"] == 100.0
    assert "trending_up" in stats["per_regime"]


def test_journal_report_no_trades(tmp_path):
    _use_tmp_journal(tmp_path)
    report = journal.generate_journal_report()
    assert "No closed trades" in report


def test_journal_report_with_trades(tmp_path):
    _use_tmp_journal(tmp_path)
    journal.record_entry(symbol="X", direction="bullish", instrument="EQ",
                         entry_price=500, entry_date="2026-03-01")
    journal.record_exit(symbol="X", entry_date="2026-03-01", direction="bullish",
                        exit_price=520, exit_date="2026-03-04", exit_reason="target",
                        pnl=80, pnl_pct=4.0)
    report = journal.generate_journal_report()
    assert "Thesis Accuracy" in report
    assert "Per Regime" in report or "Per Exit Reason" in report
