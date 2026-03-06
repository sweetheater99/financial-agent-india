"""Signal-filtered backtester — replays technical-indicator proxies for screener
confluence on historical data, entering only when multiple signals align.

Unlike backtest.py (which enters at every bar), this module computes RSI,
volume spikes, EMA momentum, ATR breakouts, and trend confirmation as proxies
for the real F&O screener signals, and only enters when the combined score
exceeds a configurable threshold.

Exit logic, transaction costs, and report format mirror backtest.py.

Usage:
    python backtest_signals.py --symbols RELIANCE,TCS --period 1y
    python backtest_signals.py --nifty50 --period 1y
    python backtest_signals.py --nifty50 --period 2y --min-score 4.0 --atr-target 2.0
    python backtest_signals.py --nifty50 --period 1y --output results.json
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from backtest import (
    NIFTY50_SYMBOLS,
    DEFAULT_PARAMS,
    EQ_SLIPPAGE_PCT,
    calc_round_trip_costs,
    compute_atr,
    fetch_data,
)

# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

# Signal weights (proxy for screener weights)
SIGNAL_WEIGHTS = {
    "momentum": 1.0,
    "volume": 1.0,
    "rsi": 1.0,
    "atr_breakout": 1.5,
    "trend_confirm": 2.0,
}

DEFAULT_MIN_SCORE = 3.5


def compute_signals(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """Compute all technical-indicator signals for every bar.

    Returns a DataFrame aligned to *df* with columns:
        direction    : 'bullish' | 'bearish' | None
        score        : float (sum of weighted signals that fired)
        signals_fired: list[str]
    plus individual boolean columns for each signal.
    """
    # Flatten MultiIndex columns from yfinance (e.g. ("Close", "RELIANCE.NS") -> scalar)
    def _col(name):
        col = df[name]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        return col

    close = _col("Close")
    high = _col("High")
    low = _col("Low")
    volume = _col("Volume")

    # --- Indicators ---
    ema20 = close.ewm(span=20, adjust=False).mean()
    atr = compute_atr(df, atr_period)
    day_range = high - low

    # RSI (14-period Wilder)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    # Handle zero/negative-zero avg_loss: when no losses, RSI = 100
    avg_loss_safe = avg_loss.where(avg_loss.abs() > 1e-10, float("nan"))
    rs = avg_gain / avg_loss_safe
    rsi = 100 - (100 / (1 + rs))
    # Where avg_loss is ~0 (all gains), RSI should be 100
    rsi = rsi.fillna(100.0)

    # Volume ratio (current / 20-day SMA)
    vol_sma20 = volume.rolling(20, min_periods=20).mean()
    vol_ratio = volume / vol_sma20.replace(0, float("nan"))

    # 3-day price trend + volume increasing (OI proxy)
    price_up_3d = close > close.shift(3)
    price_down_3d = close < close.shift(3)
    vol_increasing = (volume > volume.shift(1)) & (volume.shift(1) > volume.shift(2))

    # --- Per-bar signal evaluation ---
    n = len(df)
    directions = [None] * n
    scores = [0.0] * n
    signals_lists = [[] for _ in range(n)]

    sig_momentum = [False] * n
    sig_volume = [False] * n
    sig_rsi = [False] * n
    sig_atr_breakout = [False] * n
    sig_trend_confirm = [False] * n

    for i in range(n):
        # Need at least 20 bars of history for all indicators
        if pd.isna(ema20.iloc[i]) or pd.isna(atr.iloc[i]) or pd.isna(rsi.iloc[i]) or pd.isna(vol_sma20.iloc[i]):
            continue

        c = float(close.iloc[i])
        e = float(ema20.iloc[i])
        r = float(rsi.iloc[i])
        vr = float(vol_ratio.iloc[i]) if not pd.isna(vol_ratio.iloc[i]) else 0
        dr = float(day_range.iloc[i])
        a = float(atr.iloc[i])

        if a <= 0 or c <= 0:
            continue

        # Determine raw direction from price vs EMA + volume
        above_ema = c > e
        below_ema = c < e
        vol_spike = vr > 1.3

        # Direction: bullish if close > EMA20 + volume spike; bearish if close < EMA20 + volume spike
        # If no volume spike, we still allow direction but volume signal won't fire
        if above_ema:
            direction = "bullish"
        elif below_ema:
            direction = "bearish"
        else:
            continue

        fired = []
        score = 0.0

        # 1. Price momentum: close vs 20-day EMA
        if (direction == "bullish" and above_ema) or (direction == "bearish" and below_ema):
            fired.append("momentum")
            score += SIGNAL_WEIGHTS["momentum"]
            sig_momentum[i] = True

        # 2. Volume spike: volume > 1.3x 20-day average
        if vol_spike:
            fired.append("volume")
            score += SIGNAL_WEIGHTS["volume"]
            sig_volume[i] = True

        # 3. RSI filter
        #    Bullish: RSI 40-65 (oversold recovery + momentum, not overbought)
        #    Bearish: RSI 35-60 (not yet oversold bounce territory)
        if direction == "bullish" and 40 <= r <= 65:
            fired.append("rsi")
            score += SIGNAL_WEIGHTS["rsi"]
            sig_rsi[i] = True
        elif direction == "bearish" and 35 <= r <= 60:
            fired.append("rsi")
            score += SIGNAL_WEIGHTS["rsi"]
            sig_rsi[i] = True

        # 4. ATR breakout: today's range > 1.2x ATR
        if dr > 1.2 * a:
            fired.append("atr_breakout")
            score += SIGNAL_WEIGHTS["atr_breakout"]
            sig_atr_breakout[i] = True

        # 5. Trend confirmation (OI proxy): 3-day price trend + volume increasing
        if direction == "bullish" and not pd.isna(price_up_3d.iloc[i]) and price_up_3d.iloc[i]:
            if not pd.isna(vol_increasing.iloc[i]) and vol_increasing.iloc[i]:
                fired.append("trend_confirm")
                score += SIGNAL_WEIGHTS["trend_confirm"]
                sig_trend_confirm[i] = True
        elif direction == "bearish" and not pd.isna(price_down_3d.iloc[i]) and price_down_3d.iloc[i]:
            if not pd.isna(vol_increasing.iloc[i]) and vol_increasing.iloc[i]:
                fired.append("trend_confirm")
                score += SIGNAL_WEIGHTS["trend_confirm"]
                sig_trend_confirm[i] = True

        directions[i] = direction
        scores[i] = score
        signals_lists[i] = fired

    result = pd.DataFrame({
        "direction": directions,
        "score": scores,
        "sig_momentum": sig_momentum,
        "sig_volume": sig_volume,
        "sig_rsi": sig_rsi,
        "sig_atr_breakout": sig_atr_breakout,
        "sig_trend_confirm": sig_trend_confirm,
    }, index=df.index)
    result["signals_fired"] = signals_lists
    return result


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

class SignalBacktestEngine:
    """Backtests equity strategy with signal-filtered entries."""

    def __init__(self, params: dict | None = None, min_score: float = DEFAULT_MIN_SCORE):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.min_score = min_score
        self.trades: list[dict] = []
        self.total_bars = 0
        self.entry_bars = 0
        self.signal_counts: dict[str, int] = {
            "momentum": 0,
            "volume": 0,
            "rsi": 0,
            "atr_breakout": 0,
            "trend_confirm": 0,
        }

    def run(self, symbol: str, df: pd.DataFrame) -> list[dict]:
        """Run signal-filtered backtest on a single symbol."""
        # Flatten MultiIndex columns from yfinance if present
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = [c[0] for c in df.columns]

        p = self.params
        atr_series = compute_atr(df, p["atr_period"])
        signals_df = compute_signals(df, p["atr_period"])

        trades = []
        self.total_bars += len(df)

        i = max(20, p["atr_period"])  # need enough history for all indicators

        while i < len(df) - 1:
            # Check signal score at bar i
            score = signals_df["score"].iloc[i]
            direction = signals_df["direction"].iloc[i]

            if score < self.min_score or direction is None:
                i += 1
                continue

            self.entry_bars += 1
            fired = signals_df["signals_fired"].iloc[i]
            for sig in fired:
                self.signal_counts[sig] = self.signal_counts.get(sig, 0) + 1

            entry_idx = i
            entry_row = df.iloc[entry_idx]
            entry_price = float(entry_row["Close"]) * (1 + EQ_SLIPPAGE_PCT)
            atr = float(atr_series.iloc[entry_idx])

            if atr <= 0 or entry_price <= 0:
                i += 1
                continue

            # For bearish signals we would short — but since we're equity-only,
            # skip bearish entries (same as the real system which picks direction).
            # If you want to include bearish as short, flip target/SL.
            if direction == "bearish":
                # Skip bearish for equity-long-only backtest
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
                    exit_price = float(df.iloc[-1]["Close"]) * (1 - EQ_SLIPPAGE_PCT)
                    exit_reason = "data_end"
                    exit_idx = len(df) - 1
                    break

                day = df.iloc[day_idx]
                high = float(day["High"])
                low = float(day["Low"])
                close = float(day["Close"])

                if high > peak:
                    peak = high

                # Target hit
                if high >= target:
                    exit_price = target * (1 - EQ_SLIPPAGE_PCT)
                    exit_reason = "target"
                    exit_idx = day_idx
                    break

                # Stoploss hit
                if low <= stoploss:
                    exit_price = stoploss * (1 - EQ_SLIPPAGE_PCT)
                    exit_reason = "stoploss"
                    exit_idx = day_idx
                    break

                # Trailing stop
                progress = j / p["max_hold_days"]
                unrealized_pct = (close - entry_price) / entry_price * 100

                if unrealized_pct >= p["trailing_activation_pct"]:
                    trail_sl = peak - p["trailing_tight_mult"] * atr
                    trail_sl = max(trail_sl, entry_price)
                else:
                    trail_sl = peak - p["trailing_mult"] * atr

                effective_sl = max(trail_sl, stoploss)

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
                    exit_reason = "trailing_stop" if trail_sl > stoploss else "stoploss"
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
                "signal_score": round(score, 1),
                "signals_fired": fired,
                "direction": direction,
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

        # Tax estimates
        stcg_equity = round(max(0, total_pnl) * 0.15, 2)

        # Entry rate
        entry_rate = (self.entry_bars / self.total_bars * 100) if self.total_bars > 0 else 0

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
            "signal_distribution": dict(self.signal_counts),
            "entry_rate_pct": round(entry_rate, 2),
            "total_bars_scanned": self.total_bars,
            "entry_bars": self.entry_bars,
        }

    def print_report(self, stats: dict | None = None):
        """Print formatted backtest report."""
        if stats is None:
            stats = self.compute_stats()

        if stats["total_trades"] == 0:
            print("No trades generated.")
            return

        border = "=" * 60
        sep = "-" * 58

        print(f"\n{border}")
        print("  SIGNAL-FILTERED BACKTEST RESULTS")
        print(f"  Params: ATR target={self.params['atr_target_mult']}x, "
              f"SL={self.params['atr_sl_mult']}x, "
              f"hold={self.params['max_hold_days']}d, "
              f"capital/trade=\u20b9{self.params['capital_per_trade']:,}")
        print(f"  Min signal score: {self.min_score}")
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
        print(f"  SIGNAL DISTRIBUTION")
        print(f"  {sep}")
        print(f"  Entry rate: {stats['entry_rate_pct']:.2f}% "
              f"({stats['entry_bars']} entries / {stats['total_bars_scanned']} bars)")
        for sig, count in sorted(stats["signal_distribution"].items(), key=lambda x: -x[1]):
            if count > 0:
                pct = count / stats["entry_bars"] * 100 if stats["entry_bars"] > 0 else 0
                print(f"  {sig:20s} {count:4d}  ({pct:.0f}% of entries)")

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

        # Top/worst symbols
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Signal-filtered backtest — enters only on multi-signal confluence"
    )
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols (e.g. RELIANCE,TCS)")
    parser.add_argument("--nifty50", action="store_true", help="Run on all Nifty 50 stocks")
    parser.add_argument("--period", type=str, default="1y", help="yfinance period (1mo, 3mo, 6mo, 1y, 2y)")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
                        help=f"Minimum signal score to enter (default: {DEFAULT_MIN_SCORE})")
    parser.add_argument("--atr-target", type=float, default=2.5, help="ATR target multiplier")
    parser.add_argument("--atr-sl", type=float, default=1.5, help="ATR stoploss multiplier")
    parser.add_argument("--max-hold", type=int, default=7, help="Max holding days")
    parser.add_argument("--trail", type=float, default=1.0, help="Trailing stop ATR multiplier")
    parser.add_argument("--trail-act", type=float, default=2.0, help="Trailing activation %% profit")
    parser.add_argument("--capital", type=float, default=10000, help="Capital per trade")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
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
        "trailing_mult": args.trail,
        "trailing_activation_pct": args.trail_act,
        "capital_per_trade": args.capital,
    }

    print(f"Fetching data for {len(symbols)} symbol(s), period={args.period}...")
    data = fetch_data(symbols, args.period)
    print(f"Got data for {len(data)} symbols")

    if not data:
        print("No data available")
        sys.exit(1)

    engine = SignalBacktestEngine(params, min_score=args.min_score)
    engine.run_multi(data)
    stats = engine.compute_stats()
    engine.print_report(stats)

    if args.output:
        output = {
            "params": {**params, "min_score": args.min_score},
            "stats": stats,
            "trades": engine.trades,
        }
        Path(args.output).write_text(json.dumps(output, indent=2, default=str))
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
