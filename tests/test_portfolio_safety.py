"""Tests for portfolio backup, corruption recovery, and state cleanup."""

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def portfolio_dir(tmp_path, monkeypatch):
    """Point PORTFOLIO_FILE and PORTFOLIO_DIR to a temp directory."""
    import paper_trade

    pf = tmp_path / "portfolio.json"
    monkeypatch.setattr(paper_trade, "PORTFOLIO_DIR", tmp_path)
    monkeypatch.setattr(paper_trade, "PORTFOLIO_FILE", pf)
    return tmp_path


@pytest.fixture()
def state_dir(tmp_path, monkeypatch):
    """Point STATE_FILE to a temp directory."""
    import smart_monitor

    sf = tmp_path / "monitor_state.json"
    monkeypatch.setattr(smart_monitor, "STATE_FILE", sf)
    return tmp_path


# ---------------------------------------------------------------------------
# save_portfolio tests
# ---------------------------------------------------------------------------

class TestSavePortfolio:
    def test_save_creates_backup(self, portfolio_dir):
        """Saving should copy existing file to .bak before writing."""
        import paper_trade

        pf = portfolio_dir / "portfolio.json"
        bak = portfolio_dir / "portfolio.json.bak"

        # Write an initial portfolio
        original = {"capital": 100_000, "version": 1}
        pf.write_text(json.dumps(original))

        # Save a new one
        updated = {"capital": 100_000, "version": 2}
        paper_trade.save_portfolio(updated)

        # .bak should contain the original
        assert bak.exists()
        assert json.loads(bak.read_text()) == original

        # Main file should have updated data
        assert json.loads(pf.read_text()) == updated

    def test_save_atomic_write(self, portfolio_dir):
        """Save should write to .tmp first, then rename (no .tmp left over)."""
        import paper_trade

        pf = portfolio_dir / "portfolio.json"
        tmp_file = portfolio_dir / "portfolio.json.tmp"

        data = {"capital": 100_000, "positions": []}
        paper_trade.save_portfolio(data)

        # .tmp should not remain after successful save
        assert not tmp_file.exists()

        # Main file should have the data
        assert json.loads(pf.read_text()) == data

    def test_save_first_time_no_backup(self, portfolio_dir):
        """First save (no existing file) should not create .bak."""
        import paper_trade

        bak = portfolio_dir / "portfolio.json.bak"
        paper_trade.save_portfolio({"capital": 100_000})
        assert not bak.exists()


# ---------------------------------------------------------------------------
# load_portfolio tests
# ---------------------------------------------------------------------------

class TestLoadPortfolio:
    def test_load_recovers_from_corruption(self, portfolio_dir):
        """If main file is corrupt, load should fall back to .bak."""
        import paper_trade

        pf = portfolio_dir / "portfolio.json"
        bak = portfolio_dir / "portfolio.json.bak"

        good = {"capital": 100_000, "positions": [], "recovered": True}
        pf.write_text("{corrupt json!!!")
        bak.write_text(json.dumps(good))

        result = paper_trade.load_portfolio()
        assert result == good

    def test_load_returns_empty_if_both_corrupted(self, portfolio_dir):
        """If both main and .bak are corrupt, return _empty_portfolio()."""
        import paper_trade

        pf = portfolio_dir / "portfolio.json"
        bak = portfolio_dir / "portfolio.json.bak"

        pf.write_text("{bad")
        bak.write_text("{also bad")

        result = paper_trade.load_portfolio()
        assert result["capital"] == paper_trade.TOTAL_CAPITAL
        assert result["positions"] == []
        assert result["closed_trades"] == []

    def test_load_normal(self, portfolio_dir):
        """Normal load from a valid file."""
        import paper_trade

        pf = portfolio_dir / "portfolio.json"
        data = {"capital": 200_000, "positions": [{"symbol": "RELIANCE"}]}
        pf.write_text(json.dumps(data))

        result = paper_trade.load_portfolio()
        assert result == data

    def test_load_missing_file(self, portfolio_dir):
        """Missing file should return empty portfolio."""
        import paper_trade

        result = paper_trade.load_portfolio()
        assert result["capital"] == paper_trade.TOTAL_CAPITAL


# ---------------------------------------------------------------------------
# _save_state / _load_state tests
# ---------------------------------------------------------------------------

class TestStateSafety:
    def test_save_state_creates_backup(self, state_dir):
        """_save_state should back up existing state file."""
        import smart_monitor

        sf = state_dir / "monitor_state.json"
        bak = state_dir / "monitor_state.json.bak"

        original = {"last_check": {"RELIANCE": "2026-01-01"}}
        sf.write_text(json.dumps(original))

        updated = {"last_check": {"TCS": "2026-01-02"}}
        smart_monitor._save_state(updated)

        assert json.loads(bak.read_text()) == original
        assert json.loads(sf.read_text()) == updated

    def test_save_state_atomic(self, state_dir):
        """No .tmp file should remain after save."""
        import smart_monitor

        tmp_file = state_dir / "monitor_state.json.tmp"
        smart_monitor._save_state({"last_check": {}})
        assert not tmp_file.exists()

    def test_load_state_recovers_from_corruption(self, state_dir):
        """Corrupt state file should fall back to .bak."""
        import smart_monitor

        sf = state_dir / "monitor_state.json"
        bak = state_dir / "monitor_state.json.bak"

        good = {"last_check": {"INFY": "2026-01-01"}, "recovered": True}
        sf.write_text("not json")
        bak.write_text(json.dumps(good))

        result = smart_monitor._load_state()
        assert result == good

    def test_load_state_returns_defaults_if_both_corrupt(self, state_dir):
        """Both corrupt → return defaults."""
        import smart_monitor

        sf = state_dir / "monitor_state.json"
        bak = state_dir / "monitor_state.json.bak"

        sf.write_text("{bad")
        bak.write_text("{bad")

        result = smart_monitor._load_state()
        assert result["last_check"] == {}
        assert result["circuit_breaker_active"] is False


# ---------------------------------------------------------------------------
# cleanup_closed_positions tests
# ---------------------------------------------------------------------------

class TestCleanupClosedPositions:
    def test_cleanup_closed_positions(self):
        """Cleanup should remove closed symbols from state tracking."""
        from smart_monitor import cleanup_closed_positions

        portfolio = {
            "positions": [
                {"symbol": "RELIANCE", "status": "open"},
                {"symbol": "TCS", "status": "closed"},
                {"symbol": "INFY", "status": "open"},
            ]
        }
        state = {
            "last_check": {
                "RELIANCE": "2026-01-01",
                "TCS": "2026-01-01",
                "INFY": "2026-01-01",
                "HDFCBANK": "2026-01-01",  # stale, not in positions at all
            },
            "position_assessments": {
                "RELIANCE": {"grade": "A"},
                "TCS": {"grade": "B"},
            },
        }

        cleanup_closed_positions(portfolio, state)

        # Only open symbols should remain
        assert "RELIANCE" in state["last_check"]
        assert "INFY" in state["last_check"]
        assert "TCS" not in state["last_check"]
        assert "HDFCBANK" not in state["last_check"]

        assert "RELIANCE" in state["position_assessments"]
        assert "TCS" not in state["position_assessments"]

    def test_cleanup_handles_missing_keys(self):
        """Cleanup should not fail if state lacks expected keys."""
        from smart_monitor import cleanup_closed_positions

        portfolio = {"positions": [{"symbol": "RELIANCE", "status": "open"}]}
        state = {}

        # Should not raise
        cleanup_closed_positions(portfolio, state)
