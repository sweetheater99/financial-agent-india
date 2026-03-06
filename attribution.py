"""Performance attribution -- which strategies, signals, and regimes make money."""

import math
from datetime import datetime

# Import SYMBOL_SECTOR for sector lookups
try:
    from paper_trade import SYMBOL_SECTOR
except ImportError:
    SYMBOL_SECTOR = {}


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Safe division returning default on zero denominator."""
    return a / b if b != 0 else default


def _parse_date(d: str) -> datetime | None:
    """Parse YYYY-MM-DD date string, return None on failure."""
    if not d:
        return None
    try:
        return datetime.strptime(d, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def _holding_days(entry_date: str, exit_date: str) -> int | None:
    """Compute calendar days between entry and exit."""
    ed = _parse_date(entry_date)
    xd = _parse_date(exit_date)
    if ed and xd:
        return (xd - ed).days
    return None


def _compute_sharpe(pnls: list[float]) -> float:
    """Compute Sharpe-like ratio (mean/std) of P&L list."""
    if len(pnls) < 2:
        return 0.0
    mean = sum(pnls) / len(pnls)
    variance = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
    std = math.sqrt(variance) if variance > 0 else 0.0
    return _safe_div(mean, std)


def compute_attribution(journal_entries: list[dict], closed_trades: list[dict]) -> dict:
    """Full attribution analysis.

    Args:
        journal_entries: list of journal entry dicts (from journal.py _load_journal)
        closed_trades: list of closed trade dicts (from portfolio["closed_trades"])

    Returns attribution breakdown by strategy, signal, regime, direction, sector, and time.
    """
    # Build journal lookup by ID for enrichment
    journal_by_id = {}
    for je in journal_entries:
        jid = je.get("id", "")
        if jid:
            journal_by_id[jid] = je

    # --- Per Strategy (instrument type) ---
    per_strategy: dict[str, dict] = {}
    for t in closed_trades:
        instrument = t.get("instrument", "EQ")
        if instrument not in per_strategy:
            per_strategy[instrument] = {
                "trades": 0, "wins": 0, "pnl": 0.0, "pnls": [],
            }
        s = per_strategy[instrument]
        s["trades"] += 1
        pnl = t.get("pnl", 0)
        s["pnl"] += pnl
        s["pnls"].append(pnl)
        if pnl >= 0:
            s["wins"] += 1

    # Finalize per-strategy stats
    for key, s in per_strategy.items():
        s["avg_pnl"] = round(_safe_div(s["pnl"], s["trades"]), 2)
        s["win_rate"] = round(_safe_div(s["wins"], s["trades"]) * 100, 1)
        s["sharpe"] = round(_compute_sharpe(s["pnls"]), 2)
        del s["pnls"]  # clean up internal field

    # --- Per Signal (from journal) ---
    per_signal: dict[str, dict] = {}
    for t in closed_trades:
        symbol = t.get("symbol", "")
        entry_date = t.get("entry_date", "")
        direction = t.get("direction", "")
        jid = f"{symbol}_{entry_date}_{direction}"
        je = journal_by_id.get(jid)
        if not je:
            continue
        categories = je.get("thesis", {}).get("signals", {}).get("categories", [])
        pnl = t.get("pnl", 0)
        for cat in categories:
            if cat not in per_signal:
                per_signal[cat] = {
                    "appeared_in": 0, "wins": 0, "losses": 0,
                    "total_pnl": 0.0,
                }
            ps = per_signal[cat]
            ps["appeared_in"] += 1
            ps["total_pnl"] += pnl
            if pnl >= 0:
                ps["wins"] += 1
            else:
                ps["losses"] += 1

    for cat, ps in per_signal.items():
        ps["avg_pnl_when_present"] = round(
            _safe_div(ps["total_pnl"], ps["appeared_in"]), 2
        )
        ps["win_rate"] = round(
            _safe_div(ps["wins"], ps["appeared_in"]) * 100, 1
        )

    # --- Per Regime (from journal) ---
    per_regime: dict[str, dict] = {}
    for t in closed_trades:
        symbol = t.get("symbol", "")
        entry_date = t.get("entry_date", "")
        direction = t.get("direction", "")
        jid = f"{symbol}_{entry_date}_{direction}"
        je = journal_by_id.get(jid)
        regime = je.get("thesis", {}).get("regime", "unknown") if je else "unknown"
        if regime not in per_regime:
            per_regime[regime] = {"trades": 0, "wins": 0, "pnl": 0.0}
        r = per_regime[regime]
        r["trades"] += 1
        pnl = t.get("pnl", 0)
        r["pnl"] += pnl
        if pnl >= 0:
            r["wins"] += 1

    for regime, r in per_regime.items():
        r["win_rate"] = round(_safe_div(r["wins"], r["trades"]) * 100, 1)
        r["pnl"] = round(r["pnl"], 2)

    # --- Per Direction ---
    per_direction: dict[str, dict] = {}
    for t in closed_trades:
        direction = t.get("direction", "unknown")
        if direction not in per_direction:
            per_direction[direction] = {"trades": 0, "wins": 0, "pnl": 0.0}
        d = per_direction[direction]
        d["trades"] += 1
        pnl = t.get("pnl", 0)
        d["pnl"] += pnl
        if pnl >= 0:
            d["wins"] += 1
    for direction, d in per_direction.items():
        d["win_rate"] = round(_safe_div(d["wins"], d["trades"]) * 100, 1)
        d["pnl"] = round(d["pnl"], 2)

    # --- Per Sector ---
    per_sector: dict[str, dict] = {}
    for t in closed_trades:
        symbol = t.get("symbol", "")
        sector = SYMBOL_SECTOR.get(symbol, "Other")
        if sector not in per_sector:
            per_sector[sector] = {"trades": 0, "wins": 0, "pnl": 0.0}
        sec = per_sector[sector]
        sec["trades"] += 1
        pnl = t.get("pnl", 0)
        sec["pnl"] += pnl
        if pnl >= 0:
            sec["wins"] += 1
    for sector, sec in per_sector.items():
        sec["win_rate"] = round(_safe_div(sec["wins"], sec["trades"]) * 100, 1)
        sec["pnl"] = round(sec["pnl"], 2)

    # --- Time Analysis ---
    winner_hold_days = []
    loser_hold_days = []
    day_of_week_pnl: dict[int, list[float]] = {i: [] for i in range(7)}

    for t in closed_trades:
        entry_date = t.get("entry_date", "")
        exit_date = t.get("exit_date", "")
        pnl = t.get("pnl", 0)
        hold = _holding_days(entry_date, exit_date)
        if hold is not None:
            if pnl >= 0:
                winner_hold_days.append(hold)
            else:
                loser_hold_days.append(hold)

        ed = _parse_date(entry_date)
        if ed:
            day_of_week_pnl[ed.weekday()].append(pnl)

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_avg = {
        day_names[i]: round(_safe_div(sum(pnls), len(pnls)), 2)
        for i, pnls in day_of_week_pnl.items() if pnls
    }
    best_day = max(day_avg, key=day_avg.get) if day_avg else "N/A"
    worst_day = min(day_avg, key=day_avg.get) if day_avg else "N/A"

    time_analysis = {
        "avg_winner_hold_days": round(
            _safe_div(sum(winner_hold_days), len(winner_hold_days)), 1
        ) if winner_hold_days else 0,
        "avg_loser_hold_days": round(
            _safe_div(sum(loser_hold_days), len(loser_hold_days)), 1
        ) if loser_hold_days else 0,
        "best_day_of_week": best_day,
        "worst_day_of_week": worst_day,
    }

    # --- Recommendations ---
    recommendations = []
    for instrument, s in per_strategy.items():
        if s["trades"] >= 3 and s["pnl"] < 0:
            recommendations.append(
                f"Kill {instrument} strategy: Rs.{s['pnl']:+,.0f} total P&L "
                f"over {s['trades']} trades ({s['win_rate']:.0f}% WR)"
            )
        if s["trades"] >= 5 and s["win_rate"] < 35:
            recommendations.append(
                f"Review {instrument}: only {s['win_rate']:.0f}% win rate"
            )

    for cat, ps in per_signal.items():
        if ps["appeared_in"] >= 3 and ps["total_pnl"] < 0:
            recommendations.append(
                f"Signal '{cat}' is net negative: Rs.{ps['total_pnl']:+,.0f} "
                f"across {ps['appeared_in']} trades"
            )

    for direction, d in per_direction.items():
        if d["trades"] >= 3 and d["pnl"] < -500:
            recommendations.append(
                f"Reduce {direction} trades: Rs.{d['pnl']:+,.0f} total P&L"
            )

    if time_analysis["avg_loser_hold_days"] > 0 and time_analysis["avg_winner_hold_days"] > 0:
        if time_analysis["avg_loser_hold_days"] > time_analysis["avg_winner_hold_days"] * 1.5:
            recommendations.append(
                f"Cut losers faster: avg loser held {time_analysis['avg_loser_hold_days']} days "
                f"vs winner {time_analysis['avg_winner_hold_days']} days"
            )

    return {
        "per_strategy": per_strategy,
        "per_signal": per_signal,
        "per_regime": per_regime,
        "per_direction": per_direction,
        "per_sector": per_sector,
        "time_analysis": time_analysis,
        "recommendations": recommendations,
    }


def generate_attribution_report(attribution: dict) -> str:
    """Telegram-formatted attribution report (HTML parse_mode)."""
    lines = ["<b>Performance Attribution</b>\n"]

    # Per strategy
    ps = attribution.get("per_strategy", {})
    if ps:
        lines.append("<b>By Strategy:</b>")
        for inst, s in sorted(ps.items(), key=lambda x: -x[1]["pnl"]):
            lines.append(
                f"  {inst}: {s['trades']} trades, "
                f"{s['win_rate']:.0f}% WR, "
                f"Rs.{s['pnl']:+,.0f} "
                f"(Sharpe {s['sharpe']:.1f})"
            )

    # Per direction
    pd_ = attribution.get("per_direction", {})
    if pd_:
        lines.append("\n<b>By Direction:</b>")
        for d, data in pd_.items():
            lines.append(
                f"  {d}: {data['trades']} trades, "
                f"{data['win_rate']:.0f}% WR, "
                f"Rs.{data['pnl']:+,.0f}"
            )

    # Per regime
    pr = attribution.get("per_regime", {})
    if pr:
        lines.append("\n<b>By Regime:</b>")
        for regime, data in sorted(pr.items(), key=lambda x: -x[1]["pnl"]):
            lines.append(
                f"  {regime}: {data['trades']} trades, "
                f"{data['win_rate']:.0f}% WR, "
                f"Rs.{data['pnl']:+,.0f}"
            )

    # Per sector (top 5)
    psec = attribution.get("per_sector", {})
    if psec:
        lines.append("\n<b>By Sector (top 5):</b>")
        for sector, data in sorted(psec.items(), key=lambda x: -x[1]["pnl"])[:5]:
            lines.append(
                f"  {sector}: {data['trades']} trades, "
                f"{data['win_rate']:.0f}% WR, "
                f"Rs.{data['pnl']:+,.0f}"
            )

    # Top signals
    psig = attribution.get("per_signal", {})
    if psig:
        lines.append("\n<b>Top Signals:</b>")
        for sig, data in sorted(psig.items(), key=lambda x: -x[1]["total_pnl"])[:5]:
            lines.append(
                f"  {sig}: {data['appeared_in']}x, "
                f"{data['win_rate']:.0f}% WR, "
                f"avg Rs.{data['avg_pnl_when_present']:+,.0f}"
            )

    # Time
    ta = attribution.get("time_analysis", {})
    if ta:
        lines.append("\n<b>Time Analysis:</b>")
        lines.append(
            f"  Winner hold: {ta.get('avg_winner_hold_days', 0):.0f}d  |  "
            f"Loser hold: {ta.get('avg_loser_hold_days', 0):.0f}d"
        )
        lines.append(
            f"  Best day: {ta.get('best_day_of_week', 'N/A')}  |  "
            f"Worst day: {ta.get('worst_day_of_week', 'N/A')}"
        )

    # Recommendations
    recs = attribution.get("recommendations", [])
    if recs:
        lines.append("\n<b>Recommendations:</b>")
        for r in recs:
            lines.append(f"  - {r}")

    return "\n".join(lines)
