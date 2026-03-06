"""Tax-ready P&L reports for Indian income tax filing.

Indian tax rules:
- Equity delivery (STCG <12 months): 15% of gains (Section 111A)
- Equity delivery (LTCG >12 months): 10% of gains above Rs.1L (Section 112A)
- F&O (speculative business income): taxed at slab rate (~30% for most traders)
- Intraday equity: speculative income, slab rate
- STT paid is deductible for F&O
- Losses: equity STCG can offset any capital gains, F&O losses can carry forward 8 years
"""

from datetime import datetime


def _parse_date(d: str) -> datetime | None:
    """Parse YYYY-MM-DD date string."""
    if not d:
        return None
    try:
        return datetime.strptime(d, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def _is_in_fy(date_str: str, fy: str) -> bool:
    """Check if a date falls in the given financial year (e.g., '2025-26').

    FY 2025-26 runs from 2025-04-01 to 2026-03-31.
    """
    if not date_str or not fy:
        return True  # if no FY filter, include all
    try:
        parts = fy.split("-")
        start_year = int(parts[0])
        fy_start = datetime(start_year, 4, 1)
        fy_end = datetime(start_year + 1, 3, 31, 23, 59, 59)
        dt = _parse_date(date_str)
        if dt is None:
            return True
        return fy_start <= dt <= fy_end
    except (ValueError, IndexError):
        return True


def _classify_trade(trade: dict) -> str:
    """Classify a trade for tax purposes.

    Returns: "equity_stcg", "equity_ltcg", "fo_business", or "equity_intraday"
    """
    instrument = trade.get("instrument", "EQ")

    # F&O trades (options, spreads, futures)
    if instrument in ("OPT", "SPREAD", "IRON_CONDOR", "MOMENTUM_OPT",
                      "WEEKLY_THETA", "FUT"):
        return "fo_business"

    # Equity trades
    entry_date = _parse_date(trade.get("entry_date", ""))
    exit_date = _parse_date(trade.get("exit_date", ""))

    if entry_date and exit_date:
        # Same day = intraday
        if entry_date.date() == exit_date.date():
            return "equity_intraday"
        # > 12 months = LTCG
        delta = (exit_date - entry_date).days
        if delta > 365:
            return "equity_ltcg"

    return "equity_stcg"


def _estimate_stt(trade: dict) -> float:
    """Estimate STT paid on a trade.

    Equity delivery: 0.1% on both buy and sell sides (on turnover)
    Options: 0.0625% on sell side only (on premium)
    """
    instrument = trade.get("instrument", "EQ")
    entry_price = trade.get("entry_price", 0)
    exit_price = trade.get("exit_price", 0)
    quantity = abs(trade.get("quantity", 1))

    if instrument in ("OPT", "SPREAD", "IRON_CONDOR", "MOMENTUM_OPT",
                      "WEEKLY_THETA"):
        # STT on sell side only for options
        return exit_price * quantity * 0.000625
    else:
        # Equity: STT on both buy and sell
        buy_stt = entry_price * quantity * 0.001
        sell_stt = exit_price * quantity * 0.001
        return buy_stt + sell_stt


def _estimate_other_charges(trade: dict) -> float:
    """Estimate exchange charges, SEBI fees, stamp duty, GST, brokerage."""
    instrument = trade.get("instrument", "EQ")
    entry_price = trade.get("entry_price", 0)
    exit_price = trade.get("exit_price", 0)
    quantity = abs(trade.get("quantity", 1))
    turnover = (entry_price + exit_price) * quantity

    brokerage = 40.0  # Rs.20 per order x 2 (buy+sell)

    if instrument in ("OPT", "SPREAD", "IRON_CONDOR", "MOMENTUM_OPT",
                      "WEEKLY_THETA"):
        exchange = turnover * 0.000495
        stamp = entry_price * quantity * 0.00003
        sebi = turnover * 0.000001
    else:
        exchange = turnover * 0.0000345
        stamp = entry_price * quantity * 0.00015
        sebi = turnover * 0.000001

    gst = (brokerage + exchange) * 0.18
    return brokerage + exchange + stamp + sebi + gst


def generate_tax_report(
    closed_trades: list[dict],
    financial_year: str = "2025-26",
) -> dict:
    """Compute tax-ready P&L summary.

    Args:
        closed_trades: list of closed trade dicts
        financial_year: "2025-26" format

    Returns tax breakdown by category with estimated taxes.
    """
    equity_stcg = {"gains": 0.0, "losses": 0.0, "count": 0}
    equity_ltcg = {"gains": 0.0, "losses": 0.0, "count": 0}
    fo_business = {"gains": 0.0, "losses": 0.0, "count": 0}
    equity_intraday = {"gains": 0.0, "losses": 0.0, "count": 0}

    total_stt = 0.0
    total_other_charges = 0.0
    total_turnover = 0.0
    fo_turnover = 0.0
    eq_turnover = 0.0
    trade_wise = []

    buckets = {
        "equity_stcg": equity_stcg,
        "equity_ltcg": equity_ltcg,
        "fo_business": fo_business,
        "equity_intraday": equity_intraday,
    }

    for t in closed_trades:
        exit_date = t.get("exit_date", "")
        if not _is_in_fy(exit_date, financial_year):
            continue

        category = _classify_trade(t)
        bucket = buckets[category]
        pnl = t.get("pnl", 0)
        bucket["count"] += 1

        if pnl >= 0:
            bucket["gains"] += pnl
        else:
            bucket["losses"] += pnl  # negative value

        stt = _estimate_stt(t)
        other = _estimate_other_charges(t)
        # Use actual transaction_costs if available
        actual_costs = t.get("transaction_costs", 0)
        if actual_costs > 0:
            total_stt += stt
            total_other_charges += actual_costs - stt
        else:
            total_stt += stt
            total_other_charges += other

        entry_price = t.get("entry_price", 0)
        exit_price = t.get("exit_price", 0)
        quantity = abs(t.get("quantity", 1))
        trade_turnover = (entry_price + exit_price) * quantity

        total_turnover += trade_turnover
        if category == "fo_business":
            # F&O turnover = absolute profit + absolute loss (for audit purposes)
            fo_turnover += abs(pnl)
        else:
            eq_turnover += trade_turnover

        trade_wise.append({
            "symbol": t.get("symbol", ""),
            "instrument": t.get("instrument", "EQ"),
            "direction": t.get("direction", ""),
            "entry_date": t.get("entry_date", ""),
            "exit_date": exit_date,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
            "pnl_gross": t.get("pnl_gross", pnl),
            "pnl_net": pnl,
            "stt": round(stt, 2),
            "other_charges": round(other, 2),
            "tax_category": category,
        })

    # Compute net and taxes
    stcg_net = equity_stcg["gains"] + equity_stcg["losses"]
    ltcg_net = equity_ltcg["gains"] + equity_ltcg["losses"]
    fo_net = fo_business["gains"] + fo_business["losses"]
    intraday_net = equity_intraday["gains"] + equity_intraday["losses"]

    # LTCG: 10% on gains above Rs.1L (Section 112A)
    ltcg_taxable = max(0, equity_ltcg["gains"] - 100000)
    ltcg_tax = ltcg_taxable * 0.10

    result = {
        "financial_year": financial_year,
        "equity_stcg": {
            "gains": round(equity_stcg["gains"], 2),
            "losses": round(equity_stcg["losses"], 2),
            "net": round(stcg_net, 2),
            "trades": equity_stcg["count"],
            "tax_at_15pct": round(max(0, stcg_net) * 0.15, 2),
        },
        "equity_ltcg": {
            "gains": round(equity_ltcg["gains"], 2),
            "losses": round(equity_ltcg["losses"], 2),
            "net": round(ltcg_net, 2),
            "trades": equity_ltcg["count"],
            "taxable_above_1l": round(ltcg_taxable, 2),
            "tax_at_10pct": round(ltcg_tax, 2),
        },
        "fo_business": {
            "gains": round(fo_business["gains"], 2),
            "losses": round(fo_business["losses"], 2),
            "net": round(fo_net, 2),
            "trades": fo_business["count"],
            "estimated_tax_at_30pct": round(max(0, fo_net) * 0.30, 2),
        },
        "equity_intraday": {
            "gains": round(equity_intraday["gains"], 2),
            "losses": round(equity_intraday["losses"], 2),
            "net": round(intraday_net, 2),
            "trades": equity_intraday["count"],
            "estimated_tax_at_30pct": round(max(0, intraday_net) * 0.30, 2),
        },
        "total_turnover": round(total_turnover, 2),
        "fo_turnover": round(fo_turnover, 2),
        "eq_turnover": round(eq_turnover, 2),
        "audit_required_fo": fo_turnover > 20000000,  # Rs.2cr F&O
        "audit_required_eq": eq_turnover > 60000000,  # Rs.6cr equity
        "total_stt_paid": round(total_stt, 2),
        "total_transaction_costs": round(total_stt + total_other_charges, 2),
        "trade_wise": trade_wise,
        "summary": "",  # filled below
    }

    # Generate summary
    total_tax = (
        result["equity_stcg"]["tax_at_15pct"]
        + result["equity_ltcg"]["tax_at_10pct"]
        + result["fo_business"]["estimated_tax_at_30pct"]
        + result["equity_intraday"]["estimated_tax_at_30pct"]
    )
    total_net = stcg_net + ltcg_net + fo_net + intraday_net
    total_trades = sum(b["count"] for b in buckets.values())

    summary_lines = [
        f"FY {financial_year} Tax Summary",
        f"Total Trades: {total_trades}",
        f"Total Net P&L: Rs.{total_net:+,.2f}",
        f"",
        f"Equity STCG: Rs.{stcg_net:+,.2f} (tax ~Rs.{result['equity_stcg']['tax_at_15pct']:,.2f} @ 15%)",
        f"Equity LTCG: Rs.{ltcg_net:+,.2f} (tax ~Rs.{result['equity_ltcg']['tax_at_10pct']:,.2f} @ 10% above 1L)",
        f"F&O Business: Rs.{fo_net:+,.2f} (tax ~Rs.{result['fo_business']['estimated_tax_at_30pct']:,.2f} @ 30%)",
        f"Intraday: Rs.{intraday_net:+,.2f} (tax ~Rs.{result['equity_intraday']['estimated_tax_at_30pct']:,.2f} @ 30%)",
        f"",
        f"Estimated Total Tax: Rs.{total_tax:,.2f}",
        f"Total STT Paid: Rs.{total_stt:,.2f}",
        f"Total Transaction Costs: Rs.{result['total_transaction_costs']:,.2f}",
        f"F&O Turnover: Rs.{fo_turnover:,.2f}" + (" (AUDIT REQUIRED >2Cr)" if result["audit_required_fo"] else ""),
        f"Equity Turnover: Rs.{eq_turnover:,.2f}" + (" (AUDIT REQUIRED >6Cr)" if result["audit_required_eq"] else ""),
    ]
    result["summary"] = "\n".join(summary_lines)

    return result


def generate_tax_pnl_statement(
    closed_trades: list[dict],
    financial_year: str = "2025-26",
) -> str:
    """Generate detailed P&L statement in tabular format suitable for CA filing.

    Columns: Date, Symbol, Buy/Sell, Qty, Buy Price, Sell Price,
             Gross P&L, STT, Other Charges, Net P&L, Tax Category
    """
    tax_data = generate_tax_report(closed_trades, financial_year)
    trade_wise = tax_data["trade_wise"]

    if not trade_wise:
        return f"FY {financial_year} P&L Statement\nNo trades in this financial year."

    lines = [
        f"P&L STATEMENT - FY {financial_year}",
        "=" * 120,
        f"{'Date':<12} {'Symbol':<12} {'Dir':<8} {'Qty':>6} {'Entry':>10} "
        f"{'Exit':>10} {'Gross P&L':>12} {'STT':>8} {'Charges':>10} "
        f"{'Net P&L':>12} {'Category':<15}",
        "-" * 120,
    ]

    for tw in sorted(trade_wise, key=lambda x: x.get("exit_date", "")):
        lines.append(
            f"{tw['exit_date']:<12} {tw['symbol']:<12} {tw['direction']:<8} "
            f"{tw['quantity']:>6} {tw['entry_price']:>10.2f} "
            f"{tw['exit_price']:>10.2f} {tw['pnl_gross']:>12.2f} "
            f"{tw['stt']:>8.2f} {tw['other_charges']:>10.2f} "
            f"{tw['pnl_net']:>12.2f} {tw['tax_category']:<15}"
        )

    lines.append("-" * 120)
    lines.append("")
    lines.append(tax_data["summary"])

    return "\n".join(lines)


def generate_telegram_tax_summary(tax_data: dict) -> str:
    """Brief Telegram-formatted tax summary (HTML parse_mode)."""
    lines = [f"<b>Tax Report FY {tax_data.get('financial_year', '')}</b>\n"]

    stcg = tax_data.get("equity_stcg", {})
    fo = tax_data.get("fo_business", {})
    ltcg = tax_data.get("equity_ltcg", {})

    if stcg.get("trades", 0) > 0:
        lines.append(
            f"<b>Equity STCG:</b> Rs.{stcg['net']:+,.0f} "
            f"({stcg['trades']} trades, ~Rs.{stcg['tax_at_15pct']:,.0f} tax)"
        )

    if ltcg.get("trades", 0) > 0:
        lines.append(
            f"<b>Equity LTCG:</b> Rs.{ltcg['net']:+,.0f} "
            f"({ltcg['trades']} trades, ~Rs.{ltcg['tax_at_10pct']:,.0f} tax)"
        )

    if fo.get("trades", 0) > 0:
        lines.append(
            f"<b>F&amp;O Business:</b> Rs.{fo['net']:+,.0f} "
            f"({fo['trades']} trades, ~Rs.{fo['estimated_tax_at_30pct']:,.0f} tax)"
        )

    total_tax = (
        stcg.get("tax_at_15pct", 0)
        + ltcg.get("tax_at_10pct", 0)
        + fo.get("estimated_tax_at_30pct", 0)
        + tax_data.get("equity_intraday", {}).get("estimated_tax_at_30pct", 0)
    )

    lines.append(f"\nEstimated Tax: Rs.{total_tax:,.0f}")
    lines.append(f"STT Paid: Rs.{tax_data.get('total_stt_paid', 0):,.0f}")

    if tax_data.get("audit_required_fo") or tax_data.get("audit_required_eq"):
        lines.append("\nTax audit may be required!")

    return "\n".join(lines)
