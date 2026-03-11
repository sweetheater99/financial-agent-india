"""V7 Trade Journal — grading, Obsidian persistence, weekly/monthly reviews.

Daily journal (3:30 PM, Haiku):
  - Grade every trade: entry quality (A/B/C), exit quality (A/B/C)
  - One-sentence lesson per trade
  - Day summary: wins/losses, P&L breakdown, day type accuracy

Weekly review (Sunday, Sonnet):
  - Performance attribution by strategy/instrument/time/setup
  - Watchlist rotation decisions
  - Level memory updates

Monthly report (1st of month, Sonnet):
  - P&L report, transaction costs, net return %
  - Tax estimate, drawdown analysis
  - Strategy allocation, capital recommendation
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import date

from v7.types import TradeResult, DayClassification, SetupType


class Journal:
    """Manages trade journal, Obsidian vault writes, and review prompts."""

    def __init__(
        self,
        vault_dir: Path | str = Path.home() / "Documents" / "Obsidian" / "trading-journal",
        data_dir: Path | str = Path("data/v7"),
    ):
        self._vault_dir = Path(vault_dir)
        self._data_dir = Path(data_dir)

    def save_daily(
        self,
        date_str: str,
        trades: list[TradeResult],
        grading: dict,
        day_classification: str,
        directional_pnl: float,
        theta_pnl: float,
        total_pnl: float,
    ) -> Path:
        """Save daily journal to Obsidian vault. Returns path."""
        md = format_obsidian_journal(
            date_str=date_str,
            trades=trades,
            grading=grading,
            day_classification=day_classification,
            directional_pnl=directional_pnl,
            theta_pnl=theta_pnl,
            total_pnl=total_pnl,
        )
        self._vault_dir.mkdir(parents=True, exist_ok=True)
        path = self._vault_dir / f"{date_str}.md"
        path.write_text(md)
        return path

    def load_recent_lessons(self, days: int = 5) -> list[str]:
        """Load lessons from recent journal entries for Strategist context."""
        lessons = []
        if not self._vault_dir.exists():
            return lessons
        files = sorted(self._vault_dir.glob("*.md"), reverse=True)[:days]
        for f in files:
            content = f.read_text()
            # Extract lines that start with "Lesson:" or are in the lessons section
            for line in content.split("\n"):
                stripped = line.strip()
                if stripped.startswith("- Lesson:") or stripped.startswith("Lesson:"):
                    lessons.append(stripped)
        return lessons


def grade_trades_prompt(
    trades: list[TradeResult],
    day_classification: DayClassification | str,
    playbook_setups: int,
    day_pnl: float,
) -> str:
    """Generate the prompt for Claude Haiku to grade today's trades.

    Returns the full prompt string. Caller sends to Claude API.
    """
    dc = day_classification.value if isinstance(day_classification, DayClassification) else day_classification

    trade_lines = []
    for t in trades:
        trade_lines.append(
            f"- {t.symbol} ({t.instrument}): {t.direction}, "
            f"entry {t.entry_price:.2f} -> exit {t.exit_price:.2f}, "
            f"P&L {t.pnl:+,.0f} ({t.pnl_pct:+.1f}%), "
            f"exit reason: {t.exit_reason}, "
            f"setup: {t.setup_id} ({t.setup_type.value})"
        )

    trades_text = "\n".join(trade_lines) if trade_lines else "No trades today."

    return f"""Grade today's trades. For each trade, provide:
- entry_grade: A (trigger + confirmation aligned) / B (trigger fired, weak confirmation) / C (FOMO or forced entry)
- exit_grade: A (plan followed exactly) / B (minor deviation from plan) / C (panic exit or held too long)
- lesson: one sentence — what to remember for next time

Also provide a day_summary with:
- day_type_accuracy: was the morning classification correct? ("correct", "partially correct", "wrong")
- best_trade: setup_id of best executed trade
- worst_trade: setup_id of worst executed trade
- overall_lesson: one sentence for the day

Context:
- Day classification: {dc}
- Playbook had {playbook_setups} setups
- Day P&L: {day_pnl:+,.0f}

Trades:
{trades_text}

Respond in JSON:
{{
  "trades": [
    {{"setup_id": "...", "entry_grade": "A/B/C", "exit_grade": "A/B/C", "lesson": "..."}}
  ],
  "day_summary": {{
    "day_type_accuracy": "correct/partially correct/wrong",
    "best_trade": "setup_id",
    "worst_trade": "setup_id",
    "overall_lesson": "..."
  }}
}}"""


def parse_journal_response(raw: str) -> dict:
    """Parse Claude's journal grading response."""
    from utils import parse_claude_json
    return parse_claude_json(raw)


def format_obsidian_journal(
    date_str: str,
    trades: list[TradeResult],
    grading: dict,
    day_classification: str,
    directional_pnl: float,
    theta_pnl: float,
    total_pnl: float,
) -> str:
    """Format the daily journal as Obsidian-compatible Markdown."""
    grade_map = {}
    for tg in grading.get("trades", []):
        grade_map[tg["setup_id"]] = tg

    wins = sum(1 for t in trades if t.pnl > 0)
    losses = sum(1 for t in trades if t.pnl <= 0)
    day_summary = grading.get("day_summary", {})

    lines = [
        f"# Trading Journal — {date_str}",
        "",
        f"**Day classification:** {day_classification}",
        f"**Day type accuracy:** {day_summary.get('day_type_accuracy', 'N/A')}",
        f"**Trades:** {len(trades)} ({wins}W / {losses}L)",
        f"**P&L:** Directional {directional_pnl:+,.0f} + Theta {theta_pnl:+,.0f} = **{total_pnl:+,.0f}**",
        "",
        "---",
        "",
        "## Trades",
        "",
    ]

    for t in trades:
        g = grade_map.get(t.setup_id, {})
        lines.extend([
            f"### {t.symbol} — {t.instrument}",
            f"- Direction: {t.direction}",
            f"- Entry: {t.entry_price:.2f} | Exit: {t.exit_price:.2f}",
            f"- P&L: {t.pnl:+,.0f} ({t.pnl_pct:+.1f}%)",
            f"- Exit reason: {t.exit_reason}",
            f"- Setup: {t.setup_id} ({t.setup_type.value})",
            f"- Entry grade: **{g.get('entry_grade', '?')}** | Exit grade: **{g.get('exit_grade', '?')}**",
            f"- Lesson: {g.get('lesson', '')}",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## Day Summary",
        "",
        f"- Best trade: {day_summary.get('best_trade', 'N/A')}",
        f"- Worst trade: {day_summary.get('worst_trade', 'N/A')}",
        f"- Overall lesson: {day_summary.get('overall_lesson', '')}",
    ])

    return "\n".join(lines)


def generate_weekly_review_prompt(
    trades_this_week: list[TradeResult],
    edge_summary: str,
    level_memory: dict,
    watchlist_performance: dict[str, float],
) -> str:
    """Generate prompt for Sonnet weekly review.

    Returns the full prompt string. Caller sends to Claude API.
    """
    trade_lines = []
    for t in trades_this_week:
        trade_lines.append(
            f"- {t.symbol}: {t.setup_type.value}, P&L {t.pnl:+,.0f}, "
            f"grades {t.entry_grade}/{t.exit_grade}"
        )
    trades_text = "\n".join(trade_lines) if trade_lines else "No trades this week."

    perf_lines = []
    for sym, pnl in sorted(watchlist_performance.items(), key=lambda x: x[1]):
        perf_lines.append(f"  {sym}: {pnl:+,.0f}")
    perf_text = "\n".join(perf_lines) if perf_lines else "  No data."

    levels_text = json.dumps(level_memory, indent=2, default=str)[:2000]

    return f"""Weekly performance review. Analyze and provide recommendations.

EDGE TRACKER:
{edge_summary}

THIS WEEK'S TRADES:
{trades_text}

WATCHLIST PERFORMANCE (net P&L per instrument):
{perf_text}

CURRENT LEVEL MEMORY (truncated):
{levels_text}

Provide:
1. Performance attribution by strategy, instrument, time of day, and setup type.
2. Strategies to disable (< 40% win rate after 30+ trades).
3. Instruments to drop from active watchlist (consistently losing) and replacements.
4. Time slots to avoid.
5. Watchlist rotation: which instrument to drop, which to add.
6. Level memory updates:
   - Levels that held 2+ times this week → strengthen
   - Levels that broke cleanly → remove or flip
   - New levels from this week's price action

Respond in JSON:
{{
  "attribution_summary": "...",
  "strategies_to_disable": [],
  "instruments_to_drop": [],
  "instruments_to_add": [],
  "time_slots_to_avoid": [],
  "level_updates": {{
    "strengthen": [{{"symbol": "...", "price": 0, "reason": "..."}}],
    "remove": [{{"symbol": "...", "price": 0, "reason": "..."}}],
    "flip": [{{"symbol": "...", "price": 0, "old_type": "...", "new_type": "..."}}],
    "add": [{{"symbol": "...", "price": 0, "type": "...", "source": "..."}}]
  }},
  "watchlist_rotation": {{
    "drop": {{"symbol": "...", "reason": "..."}},
    "add": {{"symbol": "...", "reason": "..."}}
  }},
  "next_week_focus": "..."
}}"""


def generate_monthly_report_prompt(
    month: str,
    total_pnl: float,
    total_costs: float,
    capital: float,
    trades_count: int,
    win_rate: float,
    max_drawdown_pct: float,
    edge_summary: str,
    theta_pnl: float,
    directional_pnl: float,
) -> str:
    """Generate prompt for Sonnet monthly report.

    Returns the full prompt string. Caller sends to Claude API.
    """
    net_pnl = total_pnl - total_costs
    net_return_pct = (net_pnl / capital * 100) if capital > 0 else 0.0

    # F&O turnover for tax: sum of absolute trade values
    # This is an estimate — actual turnover is trade_value * quantity for each leg
    estimated_turnover = abs(total_pnl) * 10  # rough: P&L ~ 10% of turnover

    return f"""Monthly performance report for {month}.

PERFORMANCE:
- Gross P&L: {total_pnl:+,.0f}
- Transaction costs: {total_costs:,.0f}
- Net P&L: {net_pnl:+,.0f}
- Net return: {net_return_pct:+.1f}% on {capital:,.0f} capital
- Directional P&L: {directional_pnl:+,.0f}
- Theta P&L: {theta_pnl:+,.0f}
- Total trades: {trades_count}
- Win rate: {win_rate:.0%}
- Max drawdown: {max_drawdown_pct:.1f}%

EDGE TRACKER:
{edge_summary}

ESTIMATED TURNOVER: {estimated_turnover:,.0f}

Provide a monthly report including:
1. P&L report with breakdown (directional vs theta, transaction costs, net return %)
2. Tax estimate:
   - F&O turnover and classification (speculative vs non-speculative)
   - Estimated advance tax due (30% bracket assumed)
   - STT already paid (included in transaction costs)
3. Drawdown analysis: max drawdown, recovery time, risk events
4. Strategy allocation for next month: shift toward what's working
5. Capital recommendation: grow (add funds) / maintain / reduce (withdraw)
6. Withdrawal recommendation: suggest withdrawing 50% of net profit
7. Key lessons from the month

Respond in JSON:
{{
  "pnl_report": {{
    "gross_pnl": 0,
    "costs": 0,
    "net_pnl": 0,
    "return_pct": 0,
    "directional_pnl": 0,
    "theta_pnl": 0
  }},
  "tax_estimate": {{
    "turnover": 0,
    "estimated_tax": 0,
    "advance_tax_due": 0,
    "notes": "..."
  }},
  "drawdown_analysis": {{
    "max_drawdown_pct": 0,
    "recovery_days": 0,
    "risk_events": []
  }},
  "strategy_allocation": {{
    "directional_pct": 60,
    "theta_pct": 40,
    "changes": "..."
  }},
  "capital_recommendation": "grow/maintain/reduce",
  "withdrawal": {{
    "amount": 0,
    "reasoning": "..."
  }},
  "key_lessons": ["..."]
}}"""
