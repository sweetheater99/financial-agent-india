"""F&O Backtest Engine — walks daily bars, generates synthetic chains,
checks exits then entries, tracks portfolio, computes stats.

Usage:
    python fo_backtest.py --symbol NIFTY --period 2y --capital 1000000
    python fo_backtest.py --symbol NIFTY --period 1y --output results.json
"""

import argparse
import datetime
import json
import logging
import math
import sys
from collections import defaultdict

logger = logging.getLogger("fo_backtest")

import numpy as np
import pandas as pd

from backtest import compute_atr
from backtest_signals import compute_signals
from fo_data import (
    calc_futures_round_trip,
    calc_options_round_trip,
    fetch_real_chain,
    generate_synthetic_chain,
    get_futures_price,
    get_lot_size,
    get_nearest_expiry,
)
from fo_strategies import (
    CondorStrategy,
    FuturesStrategy,
    MomentumStrategy,
    SpreadStrategy,
)
from greeks import black_scholes_greeks

# ---------------------------------------------------------------------------
# Allocation caps
# ---------------------------------------------------------------------------
MAX_CONCURRENT_POSITIONS = 8
CASH_RESERVE_PCT = 0.20

STRATEGY_CAPS = {
    "futures": 0.15,
    "spread": 0.35,
    "condor": 0.15,
    "momentum": 0.15,
}


class FOBacktestEngine:
    """Main backtest engine for F&O strategies."""

    def __init__(self, capital: float = 1_000_000):
        self.initial_capital = capital
        self.capital = capital
        self.positions: list[dict] = []
        self.closed_trades: list[dict] = []
        self.daily_nav: dict[str, float] = {}
        self.exit_reasons: dict[str, int] = defaultdict(int)

        # Strategy instances
        self.strategies = {
            "futures": FuturesStrategy(),
            "spread": SpreadStrategy(),
            "condor": CondorStrategy(),
            "momentum": MomentumStrategy(),
        }

    # ------------------------------------------------------------------
    # Capital tracking
    # ------------------------------------------------------------------

    def _deployed_capital(self, position: dict) -> float:
        """Capital deployed by a single position."""
        inst = position.get("instrument", "")
        if inst == "FUT":
            return position.get("margin_used", 0)
        elif inst == "SPREAD":
            return position.get("net_debit", 0) * position.get("quantity", 0)
        elif inst == "CONDOR":
            return position.get("max_risk", 0) * position.get("quantity", 0)
        elif inst == "MOMENTUM":
            return position.get("entry_premium", 0) * position.get("quantity", 0)
        return 0

    def _total_deployed(self) -> float:
        return sum(self._deployed_capital(p) for p in self.positions)

    def _available_capital(self) -> float:
        reserve = self.initial_capital * CASH_RESERVE_PCT
        return max(0, self.capital - self._total_deployed() - reserve)

    def _strategy_deployed(self, strategy_name: str) -> float:
        return sum(
            self._deployed_capital(p)
            for p in self.positions
            if p.get("instrument", "").lower() == strategy_name
            or (strategy_name == "futures" and p.get("instrument") == "FUT")
            or (strategy_name == "spread" and p.get("instrument") == "SPREAD")
            or (strategy_name == "condor" and p.get("instrument") == "CONDOR")
            or (strategy_name == "momentum" and p.get("instrument") == "MOMENTUM")
        )

    def _strategy_within_cap(self, strategy_name: str) -> bool:
        cap = STRATEGY_CAPS.get(strategy_name, 0.15)
        return self._strategy_deployed(strategy_name) < self.capital * cap

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self, data: pd.DataFrame, symbol: str = "NIFTY") -> dict:
        """Run backtest on provided OHLCV+VIX data."""
        # Ensure both uppercase and lowercase columns exist
        col_map_up = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        col_map_down = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}

        for lc, uc in col_map_up.items():
            if uc not in data.columns and lc in data.columns:
                data[uc] = data[lc]
        for uc, lc in col_map_down.items():
            if lc not in data.columns and uc in data.columns:
                data[lc] = data[uc]

        # Ensure vix column exists
        if "vix" not in data.columns:
            data["vix"] = 15.0  # default VIX

        # Compute signals
        try:
            signals = compute_signals(data)
        except Exception:
            signals = pd.DataFrame(
                {"score": 0.0, "direction": None}, index=data.index
            )

        if "score" not in signals.columns:
            signals["score"] = 0.0
        if "direction" not in signals.columns:
            signals["direction"] = None

        # Compute ATR
        atr_series = compute_atr(data)

        # Reset state
        self.capital = self.initial_capital
        self.positions = []
        self.closed_trades = []
        self.daily_nav = {}
        self.exit_reasons = defaultdict(int)

        # Walk daily bars (skip first 30 for warmup)
        start_idx = min(30, len(data) - 1)
        for i in range(start_idx, len(data)):
            row = data.iloc[i]
            date = data.index[i]
            if isinstance(date, pd.Timestamp):
                date_d = date.date()
            else:
                date_d = date

            spot = float(row["Close"])
            high = float(row["High"])
            low = float(row["Low"])
            vix = float(row.get("vix", 15.0))
            atr_val = float(atr_series.iloc[i]) if not pd.isna(atr_series.iloc[i]) else 0.0
            score = float(signals.iloc[i].get("score", 0)) if i < len(signals) else 0.0
            direction = signals.iloc[i].get("direction", None) if i < len(signals) else None

            # Generate synthetic chain
            expiry = get_nearest_expiry(date_d, min_dte=7)
            dte = (expiry - date_d).days
            if dte <= 0:
                dte = 1

            try:
                chain = fetch_real_chain(symbol, date_d)
                if chain is not None:
                    logger.debug("%s: using real chain", date_d)
                else:
                    chain = generate_synthetic_chain(spot, vix, dte, symbol=symbol)
                    logger.debug("%s: using synthetic chain", date_d)
            except Exception:
                logger.warning("Failed to load chain for %s, using empty", date_d, exc_info=True)
                chain = pd.DataFrame()

            # 1. Check exits first
            self._check_exits(date_d, spot, high, low, vix, dte, chain, symbol)

            # 2. Check entries
            self._check_entries(
                date_d, spot, vix, atr_val, score, direction,
                chain, dte, symbol,
            )

            # Record daily NAV
            nav = self.capital + self._total_deployed()
            self.daily_nav[str(date_d)] = round(nav, 2)

        # Close all remaining positions at last close
        if len(data) > 0:
            last_row = data.iloc[-1]
            last_date = data.index[-1]
            if isinstance(last_date, pd.Timestamp):
                last_date = last_date.date()
            last_close = float(last_row["Close"])
            self._close_all_remaining(last_date, last_close, symbol)

        return {
            "trades": self.closed_trades,
            "stats": self._compute_stats(),
            "daily_nav": self.daily_nav,
        }

    # ------------------------------------------------------------------
    # Exit checking
    # ------------------------------------------------------------------

    def _check_exits(
        self,
        date: datetime.date,
        spot: float,
        high: float,
        low: float,
        vix: float,
        dte: int,
        chain: pd.DataFrame,
        symbol: str,
    ):
        """Check all open positions for exit signals."""
        to_close = []
        iv = vix / 100.0

        for idx, pos in enumerate(self.positions):
            inst = pos.get("instrument", "")
            should_exit = False
            reason = ""
            exit_price = spot

            if inst == "FUT":
                strategy = self.strategies["futures"]
                should_exit, reason = strategy.should_exit(
                    position=pos, date=date, spot=spot, high=high, low=low,
                )
                if should_exit:
                    if reason == "target":
                        exit_price = pos["target_price"]
                    elif reason == "stoploss":
                        exit_price = pos["stoploss_price"]
                    else:
                        exit_price = spot

            elif inst == "SPREAD":
                strategy = self.strategies["spread"]
                # Recompute both legs
                pos_expiry = pos.get("expiry", date + datetime.timedelta(days=30))
                spread_dte = max(1, (pos_expiry - date).days)
                long_greeks = black_scholes_greeks(
                    spot, pos["long_strike"], spread_dte, iv=iv,
                    option_type=pos["option_type"],
                )
                short_greeks = black_scholes_greeks(
                    spot, pos["short_strike"], spread_dte, iv=iv,
                    option_type=pos["option_type"],
                )
                current_spread_value = long_greeks["theoretical_price"] - short_greeks["theoretical_price"]
                current_spread_value = max(0, current_spread_value)

                should_exit, reason = strategy.should_exit(
                    position=pos, date=date, current_spread_value=current_spread_value,
                )
                if should_exit:
                    exit_price = current_spread_value

            elif inst == "CONDOR":
                strategy = self.strategies["condor"]
                pos_expiry = pos.get("expiry", date + datetime.timedelta(days=30))
                condor_dte = max(1, (pos_expiry - date).days)

                cs_greeks = black_scholes_greeks(spot, pos["call_short"], condor_dte, iv=iv, option_type="CE")
                cl_greeks = black_scholes_greeks(spot, pos["call_long"], condor_dte, iv=iv, option_type="CE")
                ps_greeks = black_scholes_greeks(spot, pos["put_short"], condor_dte, iv=iv, option_type="PE")
                pl_greeks = black_scholes_greeks(spot, pos["put_long"], condor_dte, iv=iv, option_type="PE")

                current_condor_value = (
                    (cs_greeks["theoretical_price"] + ps_greeks["theoretical_price"])
                    - (cl_greeks["theoretical_price"] + pl_greeks["theoretical_price"])
                )

                should_exit, reason = strategy.should_exit(
                    position=pos, date=date, current_condor_value=current_condor_value,
                )
                if should_exit:
                    exit_price = current_condor_value

            elif inst == "MOMENTUM":
                strategy = self.strategies["momentum"]
                pos_entry_date = pos.get("entry_date", date)
                # Recompute ATM option premium
                # Use a rough DTE: original entry might have had a specific expiry
                mom_dte = max(1, 10 - (date - pos_entry_date).days)
                atm_greeks = black_scholes_greeks(
                    spot, pos["strike"], mom_dte, iv=iv,
                    option_type=pos["option_type"],
                )
                current_premium = atm_greeks["theoretical_price"]

                should_exit, reason = strategy.should_exit(
                    position=pos, date=date, current_premium=current_premium,
                )
                if should_exit:
                    exit_price = current_premium

            if should_exit:
                to_close.append((idx, reason, exit_price))

        # Close in reverse order to preserve indices
        for idx, reason, exit_price in sorted(to_close, reverse=True):
            self._close_position(idx, date, exit_price, reason, symbol)

    # ------------------------------------------------------------------
    # Close position
    # ------------------------------------------------------------------

    def _close_position(
        self,
        idx: int,
        date: datetime.date,
        exit_price: float,
        reason: str,
        symbol: str,
    ):
        """Close a position, compute P&L and transaction costs."""
        pos = self.positions.pop(idx)
        inst = pos.get("instrument", "")
        quantity = pos.get("quantity", 0)
        pnl = 0.0
        costs = 0.0

        if inst == "FUT":
            entry_price = pos["entry_price"]
            if pos["direction"] == "bullish":
                pnl = (exit_price - entry_price) * quantity
            else:
                pnl = (entry_price - exit_price) * quantity
            costs = calc_futures_round_trip(entry_price, exit_price, quantity, date)

        elif inst == "SPREAD":
            net_debit = pos["net_debit"]
            # exit_price is current spread value
            pnl = (exit_price - net_debit) * quantity
            # Approx costs: entry + exit on both legs
            costs = calc_options_round_trip(
                pos["long_premium"], exit_price + pos["short_premium"], quantity, date
            )

        elif inst == "CONDOR":
            net_credit = pos["net_credit"]
            # exit_price is current condor value (what we'd pay to close)
            pnl = (net_credit - exit_price) * quantity
            total_entry = pos["call_short_premium"] + pos["put_short_premium"] + pos["call_long_premium"] + pos["put_long_premium"]
            costs = calc_options_round_trip(total_entry, max(0, exit_price), quantity, date)

        elif inst == "MOMENTUM":
            entry_premium = pos["entry_premium"]
            pnl = (exit_price - entry_premium) * quantity
            costs = calc_options_round_trip(entry_premium, exit_price, quantity, date)

        net_pnl = pnl - costs
        self.capital += net_pnl

        # Release deployed capital back
        # (deployed capital was already subtracted from available; closing releases it)

        trade_record = {
            **pos,
            "exit_date": date,
            "exit_price": round(exit_price, 2),
            "gross_pnl": round(pnl, 2),
            "costs": round(costs, 2),
            "net_pnl": round(net_pnl, 2),
            "exit_reason": reason,
        }
        self.closed_trades.append(trade_record)
        self.exit_reasons[reason] += 1

    # ------------------------------------------------------------------
    # Entry checking
    # ------------------------------------------------------------------

    def _check_entries(
        self,
        date: datetime.date,
        spot: float,
        vix: float,
        atr: float,
        score: float,
        direction: str | None,
        chain: pd.DataFrame,
        dte: int,
        symbol: str,
    ):
        """Try each strategy for entry, respecting allocation caps."""
        if len(self.positions) >= MAX_CONCURRENT_POSITIONS:
            return

        available = self._available_capital()
        if available <= 0:
            return

        # Try futures
        if self._strategy_within_cap("futures") and len(self.positions) < MAX_CONCURRENT_POSITIONS:
            entry = self.strategies["futures"].should_enter(
                date=date, spot=spot, vix=vix, atr=atr,
                score=score, direction=direction,
                available_capital=available,
                current_positions=self.positions,
                symbol=symbol,
            )
            if entry and self._deployed_capital(entry) <= available:
                self.positions.append(entry)
                available = self._available_capital()

        # Try spread
        if self._strategy_within_cap("spread") and len(self.positions) < MAX_CONCURRENT_POSITIONS:
            entry = self.strategies["spread"].should_enter(
                date=date, spot=spot, vix=vix, atr=atr,
                score=score, direction=direction,
                available_capital=available,
                current_positions=self.positions,
                chain=chain, dte=dte, symbol=symbol,
            )
            if entry and self._deployed_capital(entry) <= available:
                self.positions.append(entry)
                available = self._available_capital()

        # Try condor
        if self._strategy_within_cap("condor") and len(self.positions) < MAX_CONCURRENT_POSITIONS:
            entry = self.strategies["condor"].should_enter(
                date=date, spot=spot, vix=vix,
                available_capital=available,
                current_positions=self.positions,
                chain=chain, dte=dte, symbol=symbol,
            )
            if entry and self._deployed_capital(entry) <= available:
                self.positions.append(entry)
                available = self._available_capital()

        # Try momentum
        if self._strategy_within_cap("momentum") and len(self.positions) < MAX_CONCURRENT_POSITIONS:
            entry = self.strategies["momentum"].should_enter(
                date=date, spot=spot, vix=vix,
                score=score, direction=direction,
                available_capital=available,
                current_positions=self.positions,
                chain=chain, dte=dte, symbol=symbol,
            )
            if entry and self._deployed_capital(entry) <= available:
                self.positions.append(entry)

    # ------------------------------------------------------------------
    # Close all remaining at backtest end
    # ------------------------------------------------------------------

    def _close_all_remaining(self, date: datetime.date, last_close: float, symbol: str):
        """Force-close all open positions at last close."""
        while self.positions:
            pos = self.positions[0]
            inst = pos.get("instrument", "")

            if inst == "FUT":
                exit_price = last_close
            elif inst == "SPREAD":
                # Flatten: exit at entry cost (net_debit)
                exit_price = pos["net_debit"]
            elif inst == "CONDOR":
                # Flatten: exit at entry credit
                exit_price = pos["net_credit"]
            elif inst == "MOMENTUM":
                # Flatten: exit at entry premium
                exit_price = pos["entry_premium"]
            else:
                exit_price = last_close

            self._close_position(0, date, exit_price, "backtest_end", symbol)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _compute_stats(self) -> dict:
        """Compute backtest statistics."""
        trades = self.closed_trades
        total_trades = len(trades)

        if total_trades == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "profit_factor": 0.0,
                "per_strategy": {},
                "exit_reasons": dict(self.exit_reasons),
            }

        pnls = [t["net_pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_pnl = sum(pnls)
        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0.0

        # Sharpe ratio (annualized, using daily NAV returns)
        nav_values = list(self.daily_nav.values())
        if len(nav_values) > 1:
            nav_arr = np.array(nav_values, dtype=float)
            daily_returns = np.diff(nav_arr) / nav_arr[:-1]
            daily_returns = daily_returns[~np.isnan(daily_returns)]
            if len(daily_returns) > 0 and np.std(daily_returns) > 0:
                sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        # Max drawdown
        if len(nav_values) > 0:
            nav_arr = np.array(nav_values, dtype=float)
            peak = np.maximum.accumulate(nav_arr)
            drawdown = (nav_arr - peak) / peak
            max_drawdown = float(np.min(drawdown)) * 100  # as percentage
        else:
            max_drawdown = 0.0

        # Profit factor
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Per-strategy breakdown
        per_strategy = {}
        strategy_map = {
            "FUT": "futures",
            "SPREAD": "spread",
            "CONDOR": "condor",
            "MOMENTUM": "momentum",
        }
        for inst, name in strategy_map.items():
            strat_trades = [t for t in trades if t.get("instrument") == inst]
            if strat_trades:
                strat_pnls = [t["net_pnl"] for t in strat_trades]
                strat_wins = [p for p in strat_pnls if p > 0]
                per_strategy[name] = {
                    "trades": len(strat_trades),
                    "win_rate": round((len(strat_wins) / len(strat_trades)) * 100, 1),
                    "total_pnl": round(sum(strat_pnls), 2),
                }

        return {
            "total_trades": total_trades,
            "win_rate": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "sharpe": round(sharpe, 2),
            "max_drawdown": round(max_drawdown, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 999.99,
            "per_strategy": per_strategy,
            "exit_reasons": dict(self.exit_reasons),
        }


# ---------------------------------------------------------------------------
# Report & CLI
# ---------------------------------------------------------------------------


def print_report(results: dict):
    """Print formatted backtest report."""
    stats = results["stats"]
    print("\n" + "=" * 60)
    print("F&O BACKTEST REPORT")
    print("=" * 60)
    print(f"  Total Trades:   {stats['total_trades']}")
    print(f"  Win Rate:       {stats['win_rate']:.1f}%")
    print(f"  Total P&L:      Rs.{stats['total_pnl']:,.2f}")
    print(f"  Sharpe Ratio:   {stats['sharpe']:.2f}")
    print(f"  Max Drawdown:   {stats['max_drawdown']:.2f}%")
    print(f"  Profit Factor:  {stats['profit_factor']:.2f}")

    if stats.get("per_strategy"):
        print("\n  Per-Strategy Breakdown:")
        for name, sdata in stats["per_strategy"].items():
            print(f"    {name:12s}  trades={sdata['trades']:3d}  "
                  f"win={sdata['win_rate']:5.1f}%  "
                  f"pnl=Rs.{sdata['total_pnl']:,.2f}")

    if stats.get("exit_reasons"):
        print("\n  Exit Reasons:")
        for reason, count in sorted(stats["exit_reasons"].items(), key=lambda x: -x[1]):
            print(f"    {reason:15s}  {count}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="F&O Backtest Engine")
    parser.add_argument("--symbol", default="NIFTY", help="Symbol to backtest")
    parser.add_argument("--period", default="2y", help="Data period (e.g., 1y, 2y)")
    parser.add_argument("--capital", type=float, default=1_000_000, help="Starting capital")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    from fo_data import fetch_spot_vix_history

    print(f"Fetching {args.symbol} data for {args.period}...")
    data = fetch_spot_vix_history(symbol=args.symbol, period=args.period)
    if data.empty:
        print("ERROR: No data fetched. Exiting.")
        sys.exit(1)

    print(f"Running backtest on {len(data)} bars...")
    engine = FOBacktestEngine(capital=args.capital)
    results = engine.run(data, symbol=args.symbol)

    print_report(results)

    if args.output:
        # Convert for JSON serialization
        output = {
            "stats": results["stats"],
            "daily_nav": results["daily_nav"],
            "trades": [
                {k: str(v) if isinstance(v, (datetime.date, datetime.datetime)) else v
                 for k, v in t.items()}
                for t in results["trades"]
            ],
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
