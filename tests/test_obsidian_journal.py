import os, sys, json
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_format_daily_journal():
    from smart_monitor import format_daily_journal

    portfolio = {
        "positions": [
            {"symbol": "MCX", "direction": "bullish", "instrument": "EQ",
             "entry_price": 2557, "entry_date": "2026-03-09", "status": "open",
             "entry_thesis": {"reasoning": "Strong OI buildup with volume breakout"},
             "_last_ltp": 2590, "allocated": 5000},
        ],
        "closed_trades": [
            {"symbol": "BPCL", "direction": "bearish", "instrument": "PE",
             "entry_price": 42.50, "exit_price": 38.00, "exit_date": "2026-03-09",
             "pnl": -380, "pnl_pct": -3.2, "exit_reason": "stoploss"},
        ],
        "stats": {"total_pnl": -380},
    }
    context = {"regime": "TRENDING_UP", "vix": 16.2, "fii_net": -2300, "pcr": 0.85}

    md = format_daily_journal(portfolio, context, "2026-03-09")
    assert "## Trading Journal" in md
    assert "MCX" in md
    assert "BPCL" in md
    assert "TRENDING_UP" in md

def test_write_daily_journal_creates_file(tmp_path):
    import smart_monitor
    orig_vault = smart_monitor.VAULT_DIR
    smart_monitor.VAULT_DIR = tmp_path
    try:
        portfolio = {
            "positions": [
                {"symbol": "TEST", "direction": "bullish", "instrument": "EQ",
                 "entry_price": 100, "entry_date": "2026-03-09", "status": "open",
                 "entry_thesis": {"reasoning": "Test thesis"}, "allocated": 1000},
            ],
            "closed_trades": [],
            "stats": {},
        }
        result = smart_monitor.write_daily_journal(portfolio, {}, today="2026-03-09")
        assert result is not None
        daily_file = tmp_path / "daily" / "2026-03-09.md"
        assert daily_file.exists()
        content = daily_file.read_text()
        assert "## Trading Journal" in content
        assert "TEST" in content
    finally:
        smart_monitor.VAULT_DIR = orig_vault

def test_write_journal_appends_to_existing(tmp_path):
    import smart_monitor
    orig_vault = smart_monitor.VAULT_DIR
    smart_monitor.VAULT_DIR = tmp_path
    try:
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir()
        (daily_dir / "2026-03-09.md").write_text("# 2026-03-09\nSome notes here\n")

        portfolio = {"positions": [], "closed_trades": [], "stats": {}}
        result = smart_monitor.write_daily_journal(portfolio, {}, today="2026-03-09")
        content = (daily_dir / "2026-03-09.md").read_text()
        assert "Some notes here" in content  # Original content preserved
        assert "## Trading Journal" in content  # Journal appended
    finally:
        smart_monitor.VAULT_DIR = orig_vault

def test_read_recent_journal(tmp_path):
    from smart_monitor import read_recent_journal

    daily = tmp_path / "daily"
    daily.mkdir()
    (daily / "2026-03-07.md").write_text("# March 7\n## Trading Journal\nBought MCX at 2500\n")
    (daily / "2026-03-08.md").write_text("# March 8\n## Trading Journal\nSold BPCL at loss\n")
    (daily / "2026-03-09.md").write_text("# March 9\nNo trading section\n")

    result = read_recent_journal(vault_dir=tmp_path, days=3, today="2026-03-09")
    assert len(result) >= 1
    assert any("MCX" in j or "BPCL" in j for j in result)
