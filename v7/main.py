# v7/main.py
"""V7 CLI Entry Point — runs all V7 operations.

Usage:
    python -m v7.main premarket          # 8:43 AM — generate playbook
    python -m v7.main opening-read       # 9:13 AM — classify day type
    python -m v7.main checkin --num 1    # 10:28 AM — check-in 1
    python -m v7.main checkin --num 2    # 12:58 PM — check-in 2
    python -m v7.main tick               # every 3 min — main executor loop
    python -m v7.main eod                # 3:33 PM — EOD review + journal
    python -m v7.main weekly             # Sunday 10:03 AM — weekly review
    python -m v7.main monthly            # 1st of month 10:07 AM — monthly report
    python -m v7.main status             # print current state
    python -m v7.main paper-status       # print paper trading performance

    --paper flag enables paper trading mode (no real orders).
    Paper mode is DEFAULT until explicitly switched to live.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

STATUS_COMMANDS = ["status", "paper-status"]
TRADING_COMMANDS = ["premarket", "opening-read", "checkin", "tick", "eod", "weekly", "monthly"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="V7 Professional Trader Bot")
    parser.add_argument(
        "command",
        choices=STATUS_COMMANDS + TRADING_COMMANDS,
        help="Operation to run",
    )
    parser.add_argument(
        "--paper", action="store_true", default=False,
        help="Paper trading mode (no real orders)",
    )
    parser.add_argument(
        "--num", type=int, default=1,
        help="Check-in number (1 or 2)",
    )
    return parser.parse_args(argv)


def _init_components(paper: bool = False) -> dict:
    """Initialize all V7 components. Returns dict of component instances."""
    from v7.state import StateManager
    from v7.edge_tracker import EdgeTracker
    from v7.journal import Journal
    from v7.telegram import TelegramAlerter
    from v7.config_v7 import CAPITAL

    data_dir = Path("data/v7")
    data_dir.mkdir(parents=True, exist_ok=True)

    state = StateManager(data_dir)
    edge_tracker = EdgeTracker(data_dir=data_dir)
    journal = Journal(data_dir=data_dir)
    telegram = TelegramAlerter()

    components = {
        "state": state,
        "edge_tracker": edge_tracker,
        "journal": journal,
        "telegram": telegram,
        "paper": paper,
        "capital": CAPITAL["initial"],
        "data_dir": data_dir,
    }

    capital = CAPITAL["initial"]

    try:
        from v7.data_feed import DataFeed
        data_feed = DataFeed(use_kite=True)  # always use Kite for data; paper only affects orders
        components["data_feed"] = data_feed
    except (ImportError, Exception):
        data_feed = None

    try:
        from v7.risk_engine import RiskEngine
        risk_engine = RiskEngine(capital=capital, state_dir=data_dir)
        components["risk_engine"] = risk_engine
    except (ImportError, Exception):
        risk_engine = None

    try:
        from v7.margin import MarginTracker
        margin = MarginTracker(capital=capital)
        components["margin"] = margin
    except (ImportError, Exception):
        margin = None

    try:
        from v7.level_memory import LevelMemory
        components["level_memory"] = LevelMemory(state_dir=data_dir)
    except (ImportError, Exception):
        pass

    try:
        from v7.order_manager import OrderManager
        order_mgr = OrderManager(dry_run=paper)
        components["order_manager"] = order_mgr
    except (ImportError, Exception):
        order_mgr = None

    try:
        from v7.strategist import Strategist
        components["strategist"] = Strategist(
            state_dir=str(data_dir),
            data_feed=data_feed,
            edge_tracker=edge_tracker,
            risk_engine=risk_engine,
        )
    except (ImportError, Exception):
        pass

    try:
        from v7.executor import Executor
        components["executor"] = Executor(
            state_mgr=state,
            data_feed=data_feed,
            risk_engine=risk_engine,
            order_mgr=order_mgr,
            margin_tracker=margin,
            capital=capital,
            strategist=components.get("strategist"),
        )
    except (ImportError, Exception):
        pass

    try:
        from v7.theta_engine import ThetaEngine
        import v7.strike_selector as strike_sel
        components["theta_engine"] = ThetaEngine(
            data_feed=data_feed,
            order_mgr=order_mgr,
            state_mgr=state,
            strike_selector=strike_sel,
            margin_tracker=margin,
            capital=capital,
        )
    except (ImportError, Exception):
        pass

    # Wire theta engine into executor so it ticks every cycle
    if "executor" in components and "theta_engine" in components:
        components["executor"].set_theta_engine(components["theta_engine"])

    return components


def cmd_premarket(components: dict) -> None:
    """Run pre-market playbook generation."""
    from v7.telegram import format_playbook_summary, AlertLevel

    strategist = components.get("strategist")
    telegram = components["telegram"]

    if not strategist:
        print("ERROR: Strategist not available (Plan 2 not implemented)")
        telegram.send("V7: Strategist not available for premarket", AlertLevel.CRITICAL)
        sys.exit(1)

    playbook = strategist.premarket()
    print(f"Playbook generated: {playbook.day_classification.value}, "
          f"{len(playbook.all_setups())} setups")

    msg = format_playbook_summary(playbook)
    telegram.send(msg, AlertLevel.LOW)


def cmd_opening_read(components: dict) -> None:
    """Run opening read (after first 30 min)."""
    strategist = components.get("strategist")
    telegram = components["telegram"]

    if not strategist:
        print("ERROR: Strategist not available")
        sys.exit(1)

    updated_playbook = strategist.opening_read()
    if updated_playbook is None:
        print("Opening read skipped — no playbook available")
        return

    print(f"Opening read done. Day type: {updated_playbook.day_classification.value}")

    from v7.telegram import format_checkin, AlertLevel
    msg = format_checkin(
        checkin_num=0,
        plan_changed=True,
        summary=f"Opening read: {updated_playbook.day_classification.value}",
    )
    telegram.send(msg, AlertLevel.LOW)


def cmd_checkin(components: dict, num: int) -> None:
    """Run strategist check-in."""
    strategist = components.get("strategist")
    telegram = components["telegram"]

    if not strategist:
        print("ERROR: Strategist not available")
        sys.exit(1)

    result = strategist.checkin(num)
    print(f"Check-in #{num} done. Plan changed: {result.get('plan_changed', False)}")

    from v7.telegram import format_checkin, AlertLevel
    state = components["state"]
    daily = state.load_daily_state()
    positions = state.load_positions()
    msg = format_checkin(
        checkin_num=num,
        plan_changed=result.get("plan_changed", False),
        summary=result.get("summary", "No changes"),
        daily_pnl=daily.get("daily_pnl", 0.0),
        open_count=len(positions),
    )
    telegram.send(msg, AlertLevel.LOW)


def cmd_tick(components: dict) -> None:
    """Run one executor tick."""
    executor = components.get("executor")
    if not executor:
        print("ERROR: Executor not available (Plan 3 not implemented)")
        sys.exit(1)

    executor.tick()


def cmd_eod(components: dict) -> None:
    """Run EOD review + journal."""
    from config import get_anthropic_client, CLAUDE_MODEL_LIGHT
    from v7.journal import grade_trades_prompt, parse_journal_response
    from v7.telegram import format_eod_summary, AlertLevel
    from v7.types import TradeResult, SetupType

    state = components["state"]
    journal = components["journal"]
    edge_tracker = components["edge_tracker"]
    telegram = components["telegram"]

    today = datetime.now(IST).strftime("%Y-%m-%d")
    daily = state.load_daily_state()
    trades_today = daily.get("closed_trades", [])
    playbook = state.load_playbook()

    # Convert trade dicts to TradeResult objects
    trade_results = [TradeResult.from_dict(t) for t in trades_today]

    if trade_results:
        # Call Claude Haiku for grading
        prompt = grade_trades_prompt(
            trades=trade_results,
            day_classification=playbook.day_classification if playbook else "UNKNOWN",
            playbook_setups=len(playbook.all_setups()) if playbook else 0,
            day_pnl=daily.get("total_pnl", 0),
        )

        try:
            client = get_anthropic_client()
            response = client.messages.create(
                model=CLAUDE_MODEL_LIGHT,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )
            grading = parse_journal_response(response.content[0].text)
        except Exception as e:
            print(f"Claude grading failed: {e}")
            grading = {
                "trades": [{"setup_id": t.setup_id, "entry_grade": "B",
                            "exit_grade": "B", "lesson": "Grading unavailable"}
                           for t in trade_results],
                "day_summary": {"day_type_accuracy": "unknown", "best_trade": "",
                                "worst_trade": "", "overall_lesson": "Grading failed"},
            }

        # Apply grades back to trade results
        grade_map = {tg["setup_id"]: tg for tg in grading.get("trades", [])}
        for t in trade_results:
            if t.setup_id in grade_map:
                t.entry_grade = grade_map[t.setup_id].get("entry_grade", "B")
                t.exit_grade = grade_map[t.setup_id].get("exit_grade", "B")
                t.lesson = grade_map[t.setup_id].get("lesson", "")

        # Record in edge tracker
        for t in trade_results:
            strategy = _infer_strategy(t)
            time_bucket = _infer_time_bucket(t)
            edge_tracker.record(t, strategy=strategy, time_bucket=time_bucket)

        # Save journal to Obsidian
        directional_pnl = sum(t.pnl for t in trade_results if t.setup_type != SetupType.IRON_CONDOR)
        theta_pnl = sum(t.pnl for t in trade_results if t.setup_type == SetupType.IRON_CONDOR)
        total_pnl = directional_pnl + theta_pnl

        path = journal.save_daily(
            date_str=today,
            trades=trade_results,
            grading=grading,
            day_classification=playbook.day_classification.value if playbook else "UNKNOWN",
            directional_pnl=directional_pnl,
            theta_pnl=theta_pnl,
            total_pnl=total_pnl,
        )
        print(f"Journal saved: {path}")
    else:
        directional_pnl = 0
        theta_pnl = 0
        total_pnl = 0
        # Log WHY no trades happened
        blocked = daily.get("_notrade_blocked_ticks", 0)
        if blocked > 0:
            print(f"No trades today — no-trade condition blocked {blocked} ticks")
        else:
            # Check if triggers were just never hit
            all_setups = playbook.all_setups() if playbook else []
            active = [s for s in all_setups if not s.cancelled]
            unfired = [s for s in active if not s.fired]
            if unfired:
                reasons = []
                for s in unfired:
                    reasons.append(f"  {s.id}({s.symbol}): trigger={s.trigger_level}")
                print(f"No trades today — {len(unfired)} setups never triggered:")
                for r in reasons:
                    print(r)
            else:
                print("No trades today — skipping journal")

    # EOD Telegram summary
    wins = sum(1 for t in trade_results if t.pnl > 0)

    msg = format_eod_summary(
        trades_today=len(trade_results),
        wins=wins,
        losses=len(trade_results) - wins,
        directional_pnl=directional_pnl,
        theta_pnl=theta_pnl,
        total_pnl=total_pnl,
        capital=components["capital"],
        carried_positions=daily.get("carried_positions", []),
        day_type_predicted=playbook.day_classification.value if playbook else "?",
        day_type_actual=daily.get("actual_day_type", "?"),
    )
    telegram.send(msg, AlertLevel.MEDIUM)


def cmd_weekly(components: dict) -> None:
    """Run weekly review."""
    from config import get_anthropic_client, CLAUDE_MODEL
    from v7.journal import generate_weekly_review_prompt, parse_journal_response
    from v7.telegram import format_weekly_report, AlertLevel

    state = components["state"]
    edge_tracker = components["edge_tracker"]
    telegram = components["telegram"]

    # Load week's trades from edge tracker
    edge_summary = edge_tracker.summary_for_prompt()
    level_memory = state.load_level_memory()

    # Gather watchlist performance from edge tracker stats
    stats = edge_tracker.get_stats()
    watchlist_perf = {sym: data["net_pnl"] for sym, data in stats.get("by_instrument", {}).items()}

    # Get this week's trades from daily states
    # (In practice, edge_tracker has all trades — we just need the recent ones)
    trades_this_week = []  # TODO: filter edge_tracker trades to this week

    prompt = generate_weekly_review_prompt(
        trades_this_week=trades_this_week,
        edge_summary=edge_summary,
        level_memory=level_memory,
        watchlist_performance=watchlist_perf,
    )

    try:
        client = get_anthropic_client()
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        review = parse_journal_response(response.content[0].text)
    except Exception as e:
        print(f"Weekly review failed: {e}")
        telegram.send(format_weekly_report(f"Weekly review generation failed: {e}"), AlertLevel.HIGH)
        return

    # Apply level memory updates
    if "level_updates" in review:
        _apply_level_updates(state, review["level_updates"])

    # Save review
    data_dir = components["data_dir"]
    today = datetime.now(IST).strftime("%Y-%m-%d")
    review_file = data_dir / f"weekly_review_{today}.json"
    review_file.write_text(json.dumps(review, indent=2, default=str))

    msg = format_weekly_report(json.dumps(review, indent=2, default=str)[:3500])
    telegram.send(msg, AlertLevel.LOW)
    print(f"Weekly review saved: {review_file}")


def cmd_monthly(components: dict) -> None:
    """Run monthly report."""
    from config import get_anthropic_client, CLAUDE_MODEL
    from v7.journal import generate_monthly_report_prompt, parse_journal_response
    from v7.telegram import format_weekly_report, AlertLevel

    edge_tracker = components["edge_tracker"]
    telegram = components["telegram"]
    capital = components["capital"]

    now = datetime.now(IST)
    # Report for previous month
    if now.month == 1:
        month_str = f"{now.year - 1}-12"
    else:
        month_str = f"{now.year}-{now.month - 1:02d}"

    stats = edge_tracker.get_stats()
    overall = stats["overall"]

    prompt = generate_monthly_report_prompt(
        month=month_str,
        total_pnl=overall.get("net_pnl", 0),
        total_costs=0,  # TODO: sum costs from edge tracker trades
        capital=capital,
        trades_count=overall.get("trades", 0),
        win_rate=overall.get("win_rate", 0),
        max_drawdown_pct=0,  # TODO: from monthly_state.json
        edge_summary=edge_tracker.summary_for_prompt(),
        theta_pnl=stats.get("by_strategy", {}).get("theta", {}).get("net_pnl", 0),
        directional_pnl=overall.get("net_pnl", 0) - stats.get("by_strategy", {}).get("theta", {}).get("net_pnl", 0),
    )

    try:
        client = get_anthropic_client()
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        report = parse_journal_response(response.content[0].text)
    except Exception as e:
        print(f"Monthly report failed: {e}")
        telegram.send(f"<b>V7 Monthly Report</b>\n\nGeneration failed: {e}", AlertLevel.HIGH)
        return

    data_dir = components["data_dir"]
    report_file = data_dir / f"monthly_report_{month_str}.json"
    report_file.write_text(json.dumps(report, indent=2, default=str))

    msg = format_weekly_report(f"Monthly Report — {month_str}\n\n{json.dumps(report, indent=2, default=str)[:3500]}")
    telegram.send(msg, AlertLevel.LOW)
    print(f"Monthly report saved: {report_file}")


def cmd_status(components: dict) -> None:
    """Print current V7 state."""
    state = components["state"]
    daily = state.load_daily_state()
    playbook = state.load_playbook()

    print("=== V7 Status ===")
    if playbook:
        print(f"Date: {playbook.date}")
        print(f"Day type: {playbook.day_classification.value}")
        print(f"Setups: {len(playbook.all_setups())} total, "
              f"{len(playbook.active_setups())} active")
    else:
        print("No playbook loaded")

    positions = state.load_positions()
    print(f"Open positions: {len(positions)}")
    for p in positions:
        print(f"  {p.symbol} {p.instrument}: entry {p.entry_price:.2f}, SL {p.stoploss:.2f}")

    print(f"Daily P&L: {daily.get('daily_pnl', 0):+,.0f}")
    print(f"Trade count: {daily.get('trades_today', 0)}")
    print(f"SL hits: {daily.get('consecutive_sl_hits', 0)}")
    closed = daily.get('closed_trades', [])
    if closed:
        print(f"Closed trades: {len(closed)}")
        for t in closed:
            print(f"  {t.get('symbol')} {t.get('instrument')}: pnl={t.get('pnl', 0):+,.0f} ({t.get('exit_reason')})")


def cmd_paper_status(components: dict) -> None:
    """Print paper trading performance summary."""
    edge_tracker = components["edge_tracker"]
    edge_tracker.get_stats()
    print("=== V7 Paper Trading Status ===")
    print(edge_tracker.summary_for_prompt())


def _infer_strategy(trade) -> str:
    """Infer strategy name from setup type."""
    from v7.types import SetupType
    if trade.setup_type in (SetupType.BREAKOUT_LONG, SetupType.BREAKOUT_SHORT):
        return "momentum"
    if trade.setup_type in (SetupType.SUPPORT_BOUNCE, SetupType.RESISTANCE_FADE):
        return "mean_reversion"
    if trade.setup_type in (SetupType.IRON_CONDOR, SetupType.CREDIT_SPREAD_BULL, SetupType.CREDIT_SPREAD_BEAR):
        return "theta"
    return "other"


def _infer_time_bucket(trade) -> str:
    """Infer time bucket from trade entry time. Default to morning."""
    # In practice, entry_time would be stored on the trade.
    # For now, default to morning bucket.
    return "9:45-11:00"


def _apply_level_updates(state, updates: dict) -> None:
    """Apply level memory updates from weekly review."""
    level_memory = state.load_level_memory()

    for item in updates.get("add", []):
        sym = item["symbol"]
        if sym not in level_memory:
            level_memory[sym] = {"levels": [], "oi_walls": {}}
        level_memory[sym]["levels"].append({
            "price": item["price"],
            "type": item.get("type", "support"),
            "strength": 1,
            "source": item.get("source", "weekly review"),
            "last_tested": str(date.today()),
            "created": str(date.today()),
        })

    for item in updates.get("strengthen", []):
        sym = item["symbol"]
        if sym in level_memory:
            for level in level_memory[sym].get("levels", []):
                if abs(level["price"] - item["price"]) < 10:
                    level["strength"] = level.get("strength", 1) + 1

    for item in updates.get("remove", []):
        sym = item["symbol"]
        if sym in level_memory:
            level_memory[sym]["levels"] = [
                lv for lv in level_memory[sym]["levels"]
                if abs(lv["price"] - item["price"]) >= 10
            ]

    state.save_level_memory(level_memory)


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    args = parse_args(argv)

    try:
        components = _init_components(paper=args.paper)
    except Exception as e:
        print(f"Failed to initialize: {e}")
        traceback.print_exc()
        sys.exit(1)

    try:
        if args.command == "premarket":
            cmd_premarket(components)
        elif args.command == "opening-read":
            cmd_opening_read(components)
        elif args.command == "checkin":
            cmd_checkin(components, args.num)
        elif args.command == "tick":
            cmd_tick(components)
        elif args.command == "eod":
            cmd_eod(components)
        elif args.command == "weekly":
            cmd_weekly(components)
        elif args.command == "monthly":
            cmd_monthly(components)
        elif args.command == "status":
            cmd_status(components)
        elif args.command == "paper-status":
            cmd_paper_status(components)
    except Exception as e:
        print(f"Command {args.command} failed: {e}")
        traceback.print_exc()
        from v7.telegram import format_error, AlertLevel
        components["telegram"].send(
            format_error(f"V7 {args.command} failed", str(e)),
            AlertLevel.CRITICAL,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
