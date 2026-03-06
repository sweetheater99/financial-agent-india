"""Trade journal — structured entry thesis + exit review for every trade.

Records WHY a trade was entered (signals, regime, confluence), and on exit,
captures whether the thesis played out. Stores as JSON for analysis.

Integrated into paper_trade.py open_positions() and close_position().
"""

import json
from datetime import datetime
from pathlib import Path

JOURNAL_DIR = Path(__file__).parent / "data" / "journal"
JOURNAL_FILE = JOURNAL_DIR / "trades.json"


def _load_journal() -> list[dict]:
    """Load journal entries from disk."""
    if JOURNAL_FILE.exists():
        try:
            return json.loads(JOURNAL_FILE.read_text())
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def _save_journal(entries: list[dict]) -> None:
    """Save journal entries to disk."""
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    JOURNAL_FILE.write_text(json.dumps(entries, indent=2, default=str))


def record_entry(
    symbol: str,
    direction: str,
    instrument: str,
    entry_price: float,
    entry_date: str,
    signals: dict | None = None,
    regime: str = "",
    macro_gate: str = "",
    supertrend: str = "",
    pcr: float = 0,
    breadth: str = "",
    sector_rotation: str = "",
    unusual_oi: str = "",
    x_sentiment: str = "",
    yt_sentiment: str = "",
    score: float = 0,
    allocation: float = 0,
    vol_mult: float = 1.0,
    notes: str = "",
) -> dict:
    """Record entry thesis for a trade.

    Args:
        symbol: stock symbol
        direction: "bullish" or "bearish"
        instrument: "EQ", "OPT", "SPREAD", etc.
        entry_price: actual entry price after slippage
        entry_date: "YYYY-MM-DD"
        signals: dict of screener signals that triggered this trade
        regime: market regime at entry
        macro_gate: macro hard gate status
        supertrend: supertrend signal
        pcr: put-call ratio
        breadth: market breadth signal
        sector_rotation: sector momentum status
        unusual_oi: unusual OI signal
        x_sentiment: X/Twitter sentiment
        yt_sentiment: YouTube sentiment
        score: screener score
        allocation: capital allocated
        vol_mult: volatility sizing multiplier
        notes: free-text notes

    Returns:
        journal entry dict
    """
    entry = {
        "id": f"{symbol}_{entry_date}_{direction}",
        "symbol": symbol,
        "direction": direction,
        "instrument": instrument,
        "entry_price": entry_price,
        "entry_date": entry_date,
        "entry_timestamp": datetime.now().isoformat(),
        "thesis": {
            "score": score,
            "regime": regime,
            "macro_gate": macro_gate,
            "supertrend": supertrend,
            "pcr": round(pcr, 2),
            "breadth": breadth,
            "sector_rotation": sector_rotation,
            "unusual_oi": unusual_oi,
            "x_sentiment": x_sentiment,
            "yt_sentiment": yt_sentiment,
            "signals": signals or {},
            "vol_mult": round(vol_mult, 2),
            "allocation": round(allocation, 2),
        },
        "notes": notes,
        "exit": None,
    }

    journal = _load_journal()
    journal.append(entry)
    _save_journal(journal)
    return entry


def record_exit(
    symbol: str,
    entry_date: str,
    direction: str,
    exit_price: float,
    exit_date: str,
    exit_reason: str,
    pnl: float,
    pnl_pct: float,
    costs: float = 0,
    holding_days: int = 0,
    thesis_played_out: bool | None = None,
    exit_notes: str = "",
) -> dict | None:
    """Record exit review for a trade.

    Finds the matching entry and updates it with exit data.

    Args:
        thesis_played_out: True if the original thesis was correct
            (even if P&L is negative due to timing). None = auto-classify.

    Returns:
        updated journal entry or None if not found.
    """
    journal = _load_journal()
    target_id = f"{symbol}_{entry_date}_{direction}"

    for entry in reversed(journal):
        if entry["id"] == target_id and entry["exit"] is None:
            # Auto-classify thesis if not provided
            if thesis_played_out is None:
                if exit_reason in ("target", "partial_target", "opt_partial_profit"):
                    thesis_played_out = True
                elif exit_reason in ("stoploss", "gap_through_sl"):
                    thesis_played_out = False
                elif exit_reason in ("trailing_stop", "time_tightened_sl"):
                    thesis_played_out = pnl >= 0
                else:
                    thesis_played_out = pnl >= 0

            entry["exit"] = {
                "exit_price": exit_price,
                "exit_date": exit_date,
                "exit_timestamp": datetime.now().isoformat(),
                "exit_reason": exit_reason,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "costs": round(costs, 2),
                "holding_days": holding_days,
                "thesis_played_out": thesis_played_out,
                "notes": exit_notes,
            }
            _save_journal(journal)
            return entry

    return None


def get_journal_stats() -> dict:
    """Compute journal analytics.

    Returns stats on thesis accuracy, regime performance, etc.
    """
    journal = _load_journal()
    closed = [e for e in journal if e.get("exit") is not None]

    if not closed:
        return {"total_entries": len(journal), "closed": 0}

    thesis_correct = sum(1 for e in closed if e["exit"].get("thesis_played_out"))
    thesis_wrong = sum(1 for e in closed if not e["exit"].get("thesis_played_out"))

    # Per-regime stats
    regime_stats = {}
    for e in closed:
        regime = e["thesis"].get("regime", "unknown")
        if regime not in regime_stats:
            regime_stats[regime] = {"trades": 0, "wins": 0, "pnl": 0}
        regime_stats[regime]["trades"] += 1
        regime_stats[regime]["pnl"] += e["exit"]["pnl"]
        if e["exit"]["pnl"] >= 0:
            regime_stats[regime]["wins"] += 1

    # Per-exit-reason stats
    reason_stats = {}
    for e in closed:
        reason = e["exit"]["exit_reason"]
        if reason not in reason_stats:
            reason_stats[reason] = {"count": 0, "pnl": 0}
        reason_stats[reason]["count"] += 1
        reason_stats[reason]["pnl"] += e["exit"]["pnl"]

    # Signals that led to winning vs losing trades
    winning_signals = {}
    losing_signals = {}
    for e in closed:
        categories = e["thesis"].get("signals", {}).get("categories", [])
        bucket = winning_signals if e["exit"]["pnl"] >= 0 else losing_signals
        for cat in categories:
            bucket[cat] = bucket.get(cat, 0) + 1

    return {
        "total_entries": len(journal),
        "closed": len(closed),
        "open": len(journal) - len(closed),
        "thesis_accuracy": round(thesis_correct / len(closed) * 100, 1) if closed else 0,
        "thesis_correct": thesis_correct,
        "thesis_wrong": thesis_wrong,
        "per_regime": regime_stats,
        "per_exit_reason": reason_stats,
        "winning_signals": winning_signals,
        "losing_signals": losing_signals,
    }


def generate_journal_report() -> str:
    """Generate Telegram-formatted journal report."""
    stats = get_journal_stats()

    if stats["closed"] == 0:
        return "<b>Trade Journal</b>\nNo closed trades yet."

    lines = ["<b>Trade Journal Report</b>\n"]
    lines.append(f"Entries: {stats['total_entries']}  |  Closed: {stats['closed']}  |  Open: {stats['open']}")
    lines.append(f"Thesis Accuracy: {stats['thesis_accuracy']:.0f}% "
                 f"({stats['thesis_correct']} correct / {stats['thesis_wrong']} wrong)")

    if stats["per_regime"]:
        lines.append(f"\n<b>Per Regime:</b>")
        for regime, data in sorted(stats["per_regime"].items(), key=lambda x: -x[1]["pnl"]):
            wr = data["wins"] / data["trades"] * 100 if data["trades"] > 0 else 0
            lines.append(f"  {regime}: {data['trades']} trades, {wr:.0f}% WR, ₹{data['pnl']:+,.0f}")

    if stats["per_exit_reason"]:
        lines.append(f"\n<b>Per Exit Reason:</b>")
        for reason, data in sorted(stats["per_exit_reason"].items(), key=lambda x: -x[1]["count"]):
            avg = data["pnl"] / data["count"] if data["count"] > 0 else 0
            lines.append(f"  {reason}: {data['count']}x, avg ₹{avg:+,.0f}")

    return "\n".join(lines)
