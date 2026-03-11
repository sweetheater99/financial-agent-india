# v7/edge_tracker.py
"""Edge Tracker — strategy performance persistence and kill decisions.

Tracks every closed trade across multiple dimensions:
- By strategy (momentum, mean_reversion, theta)
- By instrument (NIFTY, HDFCBANK, etc.)
- By time of day (9:45-11:00, 11:00-13:00, 13:00-14:30)
- By setup type (breakout_long, support_bounce, etc.)

Feeds back into Strategist prompts so Claude weights the playbook
toward what actually works.
"""
from __future__ import annotations

import json
from pathlib import Path

from v7.types import TradeResult


class EdgeTracker:
    """Persistent edge tracker. Stores raw trade records and computes stats on demand."""

    def __init__(self, data_dir: Path | str = Path("data/v7")):
        self._data_dir = Path(data_dir)
        self._file = self._data_dir / "edge_tracker.json"
        self._trades: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self._file.exists():
            try:
                self._trades = json.loads(self._file.read_text())
            except (json.JSONDecodeError, ValueError):
                self._trades = []

    def _save(self) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._file.write_text(json.dumps(self._trades, indent=2, default=str))

    def record(self, trade: TradeResult, strategy: str, time_bucket: str) -> None:
        """Record a completed trade into the edge tracker."""
        entry = {
            "symbol": trade.symbol,
            "instrument": trade.instrument,
            "setup_type": trade.setup_type.value,
            "strategy": strategy,
            "time_bucket": time_bucket,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "pnl": trade.pnl,
            "pnl_pct": trade.pnl_pct,
            "costs": trade.costs,
            "entry_date": str(trade.entry_date),
            "exit_date": str(trade.exit_date),
            "exit_reason": trade.exit_reason,
            "entry_grade": trade.entry_grade,
            "exit_grade": trade.exit_grade,
            "is_win": trade.pnl > 0,
        }
        self._trades.append(entry)
        self._save()

    def get_stats(self) -> dict:
        """Compute aggregated stats across all dimensions."""
        trades = self._trades
        if not trades:
            return {
                "overall": {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "net_pnl": 0.0},
                "by_strategy": {},
                "by_instrument": {},
                "by_time": {},
                "by_setup_type": {},
            }

        wins = [t for t in trades if t["is_win"]]
        losses = [t for t in trades if not t["is_win"]]

        return {
            "overall": {
                "trades": len(trades),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": len(wins) / len(trades) if trades else 0.0,
                "net_pnl": sum(t["pnl"] for t in trades),
            },
            "by_strategy": self._group_stats(trades, "strategy"),
            "by_instrument": self._instrument_stats(trades),
            "by_time": self._group_stats(trades, "time_bucket"),
            "by_setup_type": self._group_stats(trades, "setup_type"),
        }

    def _group_stats(self, trades: list[dict], key: str) -> dict:
        """Compute win rate, avg R:R, net P&L per group."""
        groups: dict[str, list[dict]] = {}
        for t in trades:
            g = t.get(key, "unknown")
            groups.setdefault(g, []).append(t)

        result = {}
        for name, group in groups.items():
            wins = [t for t in group if t["is_win"]]
            losses = [t for t in group if not t["is_win"]]
            avg_win = sum(t["pnl_pct"] for t in wins) / len(wins) if wins else 0.0
            avg_loss = abs(sum(t["pnl_pct"] for t in losses) / len(losses)) if losses else 1.0
            result[name] = {
                "trades": len(group),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": len(wins) / len(group) if group else 0.0,
                "avg_rr": avg_win / avg_loss if avg_loss > 0 else 0.0,
                "net_pnl": sum(t["pnl"] for t in group),
            }
        return result

    def _instrument_stats(self, trades: list[dict]) -> dict:
        """Per-instrument stats: trades, net P&L."""
        groups: dict[str, list[dict]] = {}
        for t in trades:
            groups.setdefault(t["symbol"], []).append(t)

        result = {}
        for name, group in groups.items():
            wins = [t for t in group if t["is_win"]]
            result[name] = {
                "trades": len(group),
                "wins": len(wins),
                "win_rate": len(wins) / len(group) if group else 0.0,
                "net_pnl": sum(t["pnl"] for t in group),
            }
        return result

    def kill_candidates(self, min_trades: int = 30, min_win_rate: float = 0.40) -> list[str]:
        """Return strategy names that should be disabled (< min_win_rate after min_trades)."""
        stats = self.get_stats()
        kills = []
        for strategy, data in stats["by_strategy"].items():
            if data["trades"] >= min_trades and data["win_rate"] < min_win_rate:
                kills.append(strategy)
        return kills

    def summary_for_prompt(self) -> str:
        """Generate a concise text summary for inclusion in Strategist prompts."""
        stats = self.get_stats()
        if stats["overall"]["trades"] == 0:
            return "No trades recorded yet."

        lines = []
        o = stats["overall"]
        lines.append(
            f"Overall: {o['trades']} trades, {o['win_rate']:.0%} win rate, "
            f"net P&L: {o['net_pnl']:+,.0f}"
        )

        if stats["by_strategy"]:
            lines.append("By strategy:")
            for name, s in stats["by_strategy"].items():
                lines.append(
                    f"  {name}: {s['trades']} trades, {s['win_rate']:.0%} WR, "
                    f"avg R:R {s['avg_rr']:.1f}, net {s['net_pnl']:+,.0f}"
                )

        if stats["by_instrument"]:
            lines.append("By instrument:")
            for name, s in stats["by_instrument"].items():
                lines.append(
                    f"  {name}: {s['trades']} trades, {s['win_rate']:.0%} WR, "
                    f"net {s['net_pnl']:+,.0f}"
                )

        if stats["by_time"]:
            lines.append("By time:")
            for name, s in stats["by_time"].items():
                lines.append(
                    f"  {name}: {s['trades']} trades, {s['win_rate']:.0%} WR"
                )

        kills = self.kill_candidates()
        if kills:
            lines.append(f"KILL CANDIDATES (< 40% WR after 30+ trades): {', '.join(kills)}")

        return "\n".join(lines)
