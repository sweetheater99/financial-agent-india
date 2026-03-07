"""Backtesting engine for equity strategy.

Replays historical daily OHLCV data from yfinance and simulates
the equity long strategy with ATR-based exits, trailing stops,
and transaction costs.

Usage:
    python backtest.py --symbols RELIANCE,TCS,HDFCBANK --period 1y
    python backtest.py --nifty50 --period 2y --atr-target 3.0 --atr-sl 2.0
    python backtest.py --symbols RELIANCE --period 6mo --output results.json
"""

import argparse
import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Transaction cost rates (same as paper_trade.py)
BROKERAGE_FLAT = 20.0
EQ_STT_PCT = 0.001
EQ_EXCHANGE_PCT = 0.0000345
EQ_STAMP_DUTY_PCT = 0.00015
EQ_SEBI_PCT = 0.000001
GST_PCT = 0.18
EQ_SLIPPAGE_PCT = 0.001

# Default strategy params
DEFAULT_PARAMS = {
    "atr_period": 14,
    "atr_target_mult": 2.0,
    "atr_sl_mult": 2.0,
    "trailing_mult": 2.5,
    "trailing_activation_pct": 3.0,
    "trailing_tight_mult": 2.5,
    "max_hold_days": 15,
    "capital_per_trade": 10000,
    "time_sl_enabled": True,
    "time_sl_half_mult": 0.75,
    "time_sl_three_quarter_mult": 0.5,
}

# Nifty 50 symbols for --nifty50 mode
NIFTY50_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR",
    "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK",
    "WIPRO", "HCLTECH", "ASIANPAINT", "MARUTI", "SUNPHARMA", "TATAMOTORS",
    "NTPC", "ULTRACEMCO", "POWERGRID", "TITAN", "BAJFINANCE", "NESTLEIND",
    "TECHM", "TATASTEEL", "INDUSINDBK", "BAJAJFINSV", "ONGC", "JSWSTEEL",
    "COALINDIA", "ADANIENT", "M&M", "ADANIPORTS", "GRASIM", "CIPLA",
    "BPCL", "DRREDDY", "TATACONSUM", "APOLLOHOSP", "EICHERMOT", "DIVISLAB",
    "BRITANNIA", "HINDALCO", "HEROMOTOCO", "BAJAJ-AUTO", "SBILIFE",
    "HDFCLIFE", "SHRIRAMFIN", "TRENT",
]


def calc_eq_costs(price: float, quantity: int, side: str) -> float:
    """Calculate equity transaction costs for one leg."""
    turnover = price * quantity
    brokerage = min(BROKERAGE_FLAT, turnover * 0.0003)
    stt = turnover * EQ_STT_PCT
    exchange = turnover * EQ_EXCHANGE_PCT
    stamp = turnover * EQ_STAMP_DUTY_PCT if side == "buy" else 0
    sebi = turnover * EQ_SEBI_PCT
    gst = (brokerage + exchange) * GST_PCT
    return brokerage + stt + exchange + stamp + sebi + gst


def calc_round_trip_costs(entry_price, exit_price, quantity):
    """Total round-trip costs."""
    return calc_eq_costs(entry_price, quantity, "buy") + calc_eq_costs(exit_price, quantity, "sell")


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ATR from OHLCV DataFrame."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.rolling(window=period, min_periods=period).mean()


def fetch_data(symbols: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
    """Fetch historical daily OHLCV from yfinance.

    Returns {symbol: DataFrame} with columns: Open, High, Low, Close, Volume
    """
    import yfinance as yf

    result = {}
    tickers = [f"{s}.NS" for s in symbols]

    data = yf.download(tickers, period=period, interval="1d", progress=False, threads=True)

    if data.empty:
        return {}

    # Handle single vs multi-ticker download format
    if len(tickers) == 1:
        sym = symbols[0]
        df = data[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if len(df) >= 20:
            result[sym] = df
    else:
        for sym, ticker in zip(symbols, tickers):
            try:
                df = pd.DataFrame({
                    "Open": data[("Open", ticker)],
                    "High": data[("High", ticker)],
                    "Low": data[("Low", ticker)],
                    "Close": data[("Close", ticker)],
                    "Volume": data[("Volume", ticker)],
                }).dropna()
                if len(df) >= 20:
                    result[sym] = df
            except (KeyError, TypeError):
                continue

    return result


class BacktestEngine:
    """Simulates equity long strategy on historical data."""

    def __init__(self, params: dict = None):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.trades = []
        self.equity_curve = []

    def run(self, symbol: str, df: pd.DataFrame) -> list[dict]:
        """Run backtest on a single symbol.

        Strategy: Buy at each bar's close, manage with ATR-based exits.
        This simulates "what if we entered this stock on day X".

        Returns list of trade dicts.
        """
        p = self.params
        atr_series = compute_atr(df, p["atr_period"])

        trades = []

        # Slide through the data, entering a trade every max_hold_days bars
        # (simulate non-overlapping trades)
        i = p["atr_period"]  # start after ATR is available

        while i < len(df) - 1:
            entry_idx = i
            entry_row = df.iloc[entry_idx]
            entry_price = float(entry_row["Close"]) * (1 + EQ_SLIPPAGE_PCT)  # buy slippage
            atr = float(atr_series.iloc[entry_idx])

            if atr <= 0 or entry_price <= 0:
                i += 1
                continue

            target = entry_price + p["atr_target_mult"] * atr
            stoploss = entry_price - p["atr_sl_mult"] * atr
            quantity = max(1, int(p["capital_per_trade"] / entry_price))

            peak = entry_price
            exit_price = None
            exit_reason = None
            exit_idx = None

            for j in range(1, p["max_hold_days"] + 2):
                day_idx = entry_idx + j
                if day_idx >= len(df):
                    # Force exit at last available bar
                    exit_price = float(df.iloc[-1]["Close"]) * (1 - EQ_SLIPPAGE_PCT)
                    exit_reason = "data_end"
                    exit_idx = len(df) - 1
                    break

                day = df.iloc[day_idx]
                high = float(day["High"])
                low = float(day["Low"])
                close = float(day["Close"])

                # Update peak
                if high > peak:
                    peak = high

                # Check target (intraday)
                if high >= target:
                    exit_price = target * (1 - EQ_SLIPPAGE_PCT)
                    exit_reason = "target"
                    exit_idx = day_idx
                    break

                # Check stoploss (intraday)
                if low <= stoploss:
                    exit_price = stoploss * (1 - EQ_SLIPPAGE_PCT)
                    exit_reason = "stoploss"
                    exit_idx = day_idx
                    break

                # Trailing stop — only activates after threshold
                progress = j / p["max_hold_days"]
                unrealized_pct = (peak - entry_price) / entry_price * 100

                if unrealized_pct >= p["trailing_activation_pct"]:
                    trail_sl = peak - p["trailing_tight_mult"] * atr
                    trail_sl = max(trail_sl, entry_price)  # never below entry
                    effective_sl = max(trail_sl, stoploss)
                else:
                    # Before activation: only fixed SL, no trailing
                    effective_sl = stoploss

                # Time-based SL tightening
                if p["time_sl_enabled"]:
                    if progress >= 0.75:
                        time_sl = max(entry_price, peak - p["time_sl_three_quarter_mult"] * atr)
                        effective_sl = max(effective_sl, time_sl)
                    elif progress >= 0.5:
                        time_sl = peak - p["time_sl_half_mult"] * atr
                        effective_sl = max(effective_sl, time_sl)

                if low <= effective_sl:
                    exit_price = effective_sl * (1 - EQ_SLIPPAGE_PCT)
                    is_trailing = unrealized_pct >= p["trailing_activation_pct"]
                    exit_reason = "trailing_stop" if is_trailing else "stoploss"
                    exit_idx = day_idx
                    break

                # Max hold expiry
                if j >= p["max_hold_days"]:
                    exit_price = close * (1 - EQ_SLIPPAGE_PCT)
                    exit_reason = "expiry"
                    exit_idx = day_idx
                    break

            if exit_price is None:
                i += 1
                continue

            # Calculate P&L
            pnl_gross = (exit_price - entry_price) * quantity
            costs = calc_round_trip_costs(entry_price, exit_price, quantity)
            pnl_net = pnl_gross - costs
            pnl_pct = (pnl_net / (entry_price * quantity)) * 100

            trade = {
                "symbol": symbol,
                "entry_date": str(df.index[entry_idx].date()),
                "exit_date": str(df.index[exit_idx].date()),
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "quantity": quantity,
                "atr": round(atr, 2),
                "target": round(target, 2),
                "stoploss": round(stoploss, 2),
                "pnl_gross": round(pnl_gross, 2),
                "costs": round(costs, 2),
                "pnl_net": round(pnl_net, 2),
                "pnl_pct": round(pnl_pct, 2),
                "exit_reason": exit_reason,
                "holding_days": exit_idx - entry_idx,
            }
            trades.append(trade)

            # Skip to after this trade exits
            i = exit_idx + 1

        return trades

    def run_multi(self, data: dict[str, pd.DataFrame]) -> list[dict]:
        """Run backtest across multiple symbols."""
        all_trades = []
        for symbol, df in data.items():
            trades = self.run(symbol, df)
            all_trades.extend(trades)

        # Sort by entry date
        all_trades.sort(key=lambda t: t["entry_date"])
        self.trades = all_trades
        return all_trades

    def compute_stats(self) -> dict:
        """Compute performance statistics from completed trades."""
        trades = self.trades
        if not trades:
            return {"total_trades": 0}

        pnls = [t["pnl_net"] for t in trades]
        pnl_pcts = [t["pnl_pct"] for t in trades]
        wins = [t for t in trades if t["pnl_net"] >= 0]
        losses = [t for t in trades if t["pnl_net"] < 0]

        total_pnl = sum(pnls)
        total_costs = sum(t["costs"] for t in trades)
        win_rate = len(wins) / len(trades) * 100

        # Averages
        avg_win_pct = sum(t["pnl_pct"] for t in wins) / len(wins) if wins else 0
        avg_loss_pct = sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0
        avg_hold = sum(t["holding_days"] for t in trades) / len(trades)

        # Sharpe
        mean_ret = sum(pnl_pcts) / len(pnl_pcts)
        var = sum((r - mean_ret) ** 2 for r in pnl_pcts) / len(pnl_pcts)
        std = math.sqrt(var) if var > 0 else 0
        sharpe = round((mean_ret / std) * math.sqrt(252), 2) if std > 0 else 0

        # Max drawdown
        cumulative = 0
        peak_cum = 0
        max_dd = 0
        for p in pnls:
            cumulative += p
            if cumulative > peak_cum:
                peak_cum = cumulative
            dd = peak_cum - cumulative
            if dd > max_dd:
                max_dd = dd

        # Profit factor
        win_sum = sum(t["pnl_net"] for t in wins)
        loss_sum = abs(sum(t["pnl_net"] for t in losses))
        pf = round(win_sum / loss_sum, 2) if loss_sum > 0 else float("inf")

        # Expectancy
        expectancy = mean_ret * (len(wins) / len(trades)) * avg_win_pct - (len(losses) / len(trades)) * abs(avg_loss_pct) if trades else 0

        # Exit reason breakdown
        reasons = {}
        for t in trades:
            r = t["exit_reason"]
            reasons[r] = reasons.get(r, 0) + 1

        # Per-symbol breakdown
        symbols = {}
        for t in trades:
            s = t["symbol"]
            if s not in symbols:
                symbols[s] = {"trades": 0, "wins": 0, "pnl": 0}
            symbols[s]["trades"] += 1
            symbols[s]["pnl"] += t["pnl_net"]
            if t["pnl_net"] >= 0:
                symbols[s]["wins"] += 1

        # Tax estimates (Indian)
        stcg_equity = round(max(0, total_pnl) * 0.15, 2)  # 15% STCG on equity

        return {
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "total_costs": round(total_costs, 2),
            "avg_win_pct": round(avg_win_pct, 2),
            "avg_loss_pct": round(avg_loss_pct, 2),
            "avg_holding_days": round(avg_hold, 1),
            "sharpe_ratio": sharpe,
            "max_drawdown": round(max_dd, 2),
            "profit_factor": pf,
            "best_trade": max(trades, key=lambda t: t["pnl_pct"]),
            "worst_trade": min(trades, key=lambda t: t["pnl_pct"]),
            "exit_reasons": reasons,
            "per_symbol": symbols,
            "estimated_stcg_tax": stcg_equity,
        }

    def print_report(self, stats: dict = None):
        """Print formatted backtest report to terminal."""
        if stats is None:
            stats = self.compute_stats()

        if stats["total_trades"] == 0:
            print("No trades generated.")
            return

        border = "=" * 60
        sep = "-" * 58

        print(f"\n{border}")
        print("  BACKTEST RESULTS")
        print(f"  Params: ATR target={self.params['atr_target_mult']}x, "
              f"SL={self.params['atr_sl_mult']}x, "
              f"hold={self.params['max_hold_days']}d, "
              f"capital/trade=\u20b9{self.params['capital_per_trade']:,}")
        print(border)

        print(f"\n  Total Trades: {stats['total_trades']}  |  "
              f"Win Rate: {stats['win_rate']:.1f}%  "
              f"({stats['winning_trades']}W / {stats['losing_trades']}L)")
        print(f"  Total P&L: \u20b9{stats['total_pnl']:+,.0f}  |  "
              f"Costs: \u20b9{stats['total_costs']:,.0f}")
        print(f"  Avg Win: {stats['avg_win_pct']:+.2f}%  |  "
              f"Avg Loss: {stats['avg_loss_pct']:+.2f}%")
        print(f"  Avg Holding: {stats['avg_holding_days']:.1f} days")

        print(f"\n  {sep}")
        print(f"  RISK METRICS")
        print(f"  {sep}")
        pf = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] != float('inf') else "\u221e"
        print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}  |  "
              f"Profit Factor: {pf}")
        print(f"  Max Drawdown: \u20b9{stats['max_drawdown']:,.0f}")

        print(f"\n  Best:  {stats['best_trade']['symbol']} "
              f"({stats['best_trade']['pnl_pct']:+.1f}%, "
              f"\u20b9{stats['best_trade']['pnl_net']:+,.0f})")
        print(f"  Worst: {stats['worst_trade']['symbol']} "
              f"({stats['worst_trade']['pnl_pct']:+.1f}%, "
              f"\u20b9{stats['worst_trade']['pnl_net']:+,.0f})")

        print(f"\n  {sep}")
        print(f"  EXIT REASONS")
        print(f"  {sep}")
        for reason, count in sorted(stats["exit_reasons"].items(), key=lambda x: -x[1]):
            pct = count / stats["total_trades"] * 100
            print(f"  {reason:20s} {count:4d}  ({pct:.0f}%)")

        print(f"\n  {sep}")
        print(f"  TAX ESTIMATE")
        print(f"  {sep}")
        print(f"  STCG (15% on equity gains): \u20b9{stats['estimated_stcg_tax']:,.0f}")
        print(f"  Net after tax: \u20b9{stats['total_pnl'] - stats['estimated_stcg_tax']:+,.0f}")

        # Top 5 symbols
        if stats["per_symbol"]:
            sorted_syms = sorted(stats["per_symbol"].items(), key=lambda x: -x[1]["pnl"])
            print(f"\n  {sep}")
            print(f"  TOP SYMBOLS")
            print(f"  {sep}")
            for sym, data in sorted_syms[:5]:
                wr = data["wins"] / data["trades"] * 100 if data["trades"] > 0 else 0
                print(f"  {sym:15s} {data['trades']:3d} trades  "
                      f"{wr:.0f}% WR  \u20b9{data['pnl']:+,.0f}")
            if len(sorted_syms) > 5:
                print(f"\n  WORST SYMBOLS")
                for sym, data in sorted_syms[-3:]:
                    wr = data["wins"] / data["trades"] * 100 if data["trades"] > 0 else 0
                    print(f"  {sym:15s} {data['trades']:3d} trades  "
                          f"{wr:.0f}% WR  \u20b9{data['pnl']:+,.0f}")

        print(f"\n{border}\n")


def main():
    parser = argparse.ArgumentParser(description="Backtest equity strategy on historical data")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols (e.g. RELIANCE,TCS)")
    parser.add_argument("--nifty50", action="store_true", help="Run on all Nifty 50 stocks")
    parser.add_argument("--period", type=str, default="1y", help="yfinance period (1mo, 3mo, 6mo, 1y, 2y)")
    parser.add_argument("--atr-target", type=float, default=2.5, help="ATR target multiplier")
    parser.add_argument("--atr-sl", type=float, default=1.5, help="ATR stoploss multiplier")
    parser.add_argument("--max-hold", type=int, default=7, help="Max holding days")
    parser.add_argument("--capital", type=float, default=10000, help="Capital per trade")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--no-time-sl", action="store_true", help="Disable time-based SL tightening")
    args = parser.parse_args()

    if not args.symbols and not args.nifty50:
        print("Error: specify --symbols or --nifty50")
        sys.exit(1)

    symbols = NIFTY50_SYMBOLS if args.nifty50 else [s.strip() for s in args.symbols.split(",")]

    params = {
        **DEFAULT_PARAMS,
        "atr_target_mult": args.atr_target,
        "atr_sl_mult": args.atr_sl,
        "max_hold_days": args.max_hold,
        "capital_per_trade": args.capital,
        "time_sl_enabled": not args.no_time_sl,
    }

    print(f"Fetching data for {len(symbols)} symbol(s), period={args.period}...")
    data = fetch_data(symbols, args.period)
    print(f"Got data for {len(data)} symbols")

    if not data:
        print("No data available")
        sys.exit(1)

    engine = BacktestEngine(params)
    engine.run_multi(data)
    stats = engine.compute_stats()
    engine.print_report(stats)

    if args.output:
        output = {
            "params": params,
            "stats": stats,
            "trades": engine.trades,
        }
        Path(args.output).write_text(json.dumps(output, indent=2, default=str))
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
