"""
evaluate.py — Autoresearch backtest runner and guardrail judge.

THIS FILE IS THE INCORRUPTIBLE JUDGE.
Do NOT modify strategy logic, guardrail thresholds, or scoring weights.
The agent may only modify baseline.py and experiment config files.

Usage:
    python3 autoresearch/evaluate.py --dry-run
    python3 autoresearch/evaluate.py --evaluate --experiment path/to/exp.py
    python3 autoresearch/evaluate.py --benchmark
"""

from __future__ import annotations

import argparse
import datetime
import difflib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — allow running from project root or autoresearch/
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from fo_backtest import FOBacktestEngine
from fo_strategies import (
    CondorStrategy,
    FuturesStrategy,
    MomentumStrategy,
    SpreadStrategy,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_TSV = _HERE / "results.tsv"
CACHE_PATH = _ROOT / "data" / "backtest_cache.csv"
CAPITAL = 1_000_000
HOLDOUT_MONTHS = 6
NIGHTLY_DIFF_CAP = 150  # cumulative KEPT diff lines per night

# ---------------------------------------------------------------------------
# 1. Composite score
# ---------------------------------------------------------------------------


def composite_score(stats: dict) -> float:
    """Normalized composite score in [0, 1].

    Inputs expected (from run_backtest stats):
        sharpe        — float (annualized)
        profit_factor — float (>= 0)
        max_drawdown  — positive decimal (e.g. 0.125 for 12.5% drawdown)
    """
    sharpe = float(stats.get("sharpe", 0.0))
    pf = float(stats.get("profit_factor", 0.0))
    dd = float(stats.get("max_drawdown", 0.0))  # positive decimal

    norm_sharpe = min(max(sharpe, 0) / 3.0, 1.0)
    norm_pf = min(max(pf - 1.0, 0) / 2.0, 1.0)
    norm_dd = 1.0 - min(dd / 0.15, 1.0)

    return norm_sharpe * 0.4 + norm_pf * 0.3 + norm_dd * 0.3


# ---------------------------------------------------------------------------
# 2. Strategy construction
# ---------------------------------------------------------------------------


def build_strategies(config: dict) -> dict:
    """Construct strategy instances from a config dict.

    Returns a dict keyed by strategy name, matching the engine's
    self.strategies dict structure: {"futures": ..., "spread": ...,
    "condor": ..., "momentum": ...}.
    """
    futures = FuturesStrategy(
        score_threshold=config.get("futures_score_threshold", 3.5),
        target_atr_mult=config.get("futures_target_atr_mult", 1.5),
        sl_atr_mult=config.get("futures_sl_atr_mult", 3.5),
        max_risk_pct=config.get("max_risk_pct", 0.02),
        max_hold_days=int(config.get("max_hold_days", 15)),
    )

    spread = SpreadStrategy(
        score_threshold=config.get("futures_score_threshold", 3.5),
        profit_cap_pct=config.get("spread_profit_cap_pct", 0.80),
        sl_multiplier=config.get("spread_sl_multiplier", 2.0),
    )

    condor = CondorStrategy(
        min_vix=config.get("condor_min_vix", 12),
        max_vix=config.get("condor_max_vix", 18),
        target_pct=config.get("condor_target_pct", 0.50),
        sl_multiplier=config.get("condor_sl_multiplier", 2.0),
    )

    momentum = MomentumStrategy(
        score_threshold=config.get("momentum_score_threshold", 5.0),
        sl_pct=config.get("momentum_sl_pct", 0.35),
        target_pct=config.get("momentum_target_pct", 0.90),
        time_exit_days=int(config.get("momentum_time_exit_days", 3)),
    )

    return {
        "futures": futures,
        "spread": spread,
        "condor": condor,
        "momentum": momentum,
    }


# ---------------------------------------------------------------------------
# 3. run_backtest
# ---------------------------------------------------------------------------


def run_backtest(config: dict, data: pd.DataFrame, symbol: str = "NIFTY") -> dict:
    """Run a single backtest with config injected into the engine.

    Returns a stats dict with max_drawdown normalized to positive decimal
    (e.g. 0.125 for 12.5% drawdown), ready for composite_score().
    """
    engine = FOBacktestEngine(capital=CAPITAL)

    # Inject strategies built from config
    engine.strategies = build_strategies(config)

    result = engine.run(data.copy(), symbol=symbol)
    stats = result["stats"]

    # fo_backtest._compute_stats() returns max_drawdown as a negative percentage
    # e.g. -12.5 for a 12.5% drawdown. Normalize to positive decimal.
    raw_dd = stats.get("max_drawdown", 0.0)  # e.g. -12.5
    stats["max_drawdown"] = abs(raw_dd) / 100.0  # 0.125

    # Expose composite score inline for convenience
    stats["composite_score"] = composite_score(stats)

    return stats


# ---------------------------------------------------------------------------
# 4. Data loading
# ---------------------------------------------------------------------------


def load_cached_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load parquet cache and split into train / holdout.

    Returns:
        train   — everything before 6 months ago
        holdout — last 6 months

    Raises:
        FileNotFoundError if cache doesn't exist.
    """
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Backtest cache not found at {CACHE_PATH}.\n"
            "Run: python3 autoresearch/build_cache.py --symbol NIFTY --period 3y\n"
            "to generate it. This requires Kite Connect auth (kite_auth.py)."
        )

    df = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    cutoff = pd.Timestamp.now() - pd.DateOffset(months=HOLDOUT_MONTHS)
    train = df[df.index < cutoff].copy()
    holdout = df[df.index >= cutoff].copy()

    return train, holdout


# ---------------------------------------------------------------------------
# 5. Guardrails
# ---------------------------------------------------------------------------


def _count_holdout_months(holdout: pd.DataFrame) -> float:
    """Number of calendar months covered by holdout data."""
    if holdout.empty:
        return 0.0
    span_days = (holdout.index[-1] - holdout.index[0]).days
    return max(1.0, span_days / 30.44)


def check_guardrails(
    train_stats: dict,
    holdout_stats: dict,
    baseline_holdout_score: float,
    diff_metrics: dict,
    config: dict,
    holdout_df: pd.DataFrame,
) -> tuple[bool, list[str]]:
    """Apply all guardrails. Returns (passed: bool, failures: list[str])."""
    failures = []

    train_trades = train_stats.get("total_trades", 0)
    holdout_trades = holdout_stats.get("total_trades", 0)
    holdout_sharpe = holdout_stats.get("sharpe", 0.0)
    holdout_pf = holdout_stats.get("profit_factor", 0.0)
    holdout_dd = holdout_stats.get("max_drawdown", 0.0)  # positive decimal
    holdout_wr = holdout_stats.get("win_rate", 0.0)  # 0-100
    holdout_score = holdout_stats.get("composite_score", 0.0)

    # Train liquidity
    if train_trades < 20:
        failures.append(f"train_trades={train_trades} < 20 minimum")

    # Holdout liquidity
    holdout_months = _count_holdout_months(holdout_df)
    min_holdout_by_month = max(20, int(holdout_months * 3))
    if holdout_trades < 20:
        failures.append(f"holdout_trades={holdout_trades} < 20 minimum")
    elif holdout_trades < min_holdout_by_month:
        avg = holdout_trades / holdout_months
        failures.append(
            f"holdout_trades/month={avg:.1f} < 3.0 minimum "
            f"(need {min_holdout_by_month} over {holdout_months:.1f} months)"
        )

    # Holdout quality
    if holdout_sharpe <= 0:
        failures.append(f"holdout_sharpe={holdout_sharpe:.3f} <= 0")

    if holdout_pf <= 1.0:
        failures.append(f"holdout_profit_factor={holdout_pf:.3f} <= 1.0")

    # Score improvement over baseline
    delta = holdout_score - baseline_holdout_score
    if delta < 0.05:
        failures.append(
            f"holdout_score delta={delta:.4f} < 0.05 minimum "
            f"(baseline={baseline_holdout_score:.4f}, experiment={holdout_score:.4f})"
        )

    # Win rate cap (avoid overfit)
    if holdout_wr > 85.0:
        failures.append(f"win_rate={holdout_wr:.1f}% > 85% cap")

    # Drawdown limit
    if holdout_dd > 0.15:
        failures.append(f"max_drawdown={holdout_dd:.3f} > 0.15 (15%) limit")

    # Complexity: net line change
    net_lines = diff_metrics.get("net_lines", 0)
    if abs(net_lines) > 50:
        failures.append(f"net_line_change={net_lines} > 50 limit")

    # Diff size: gross changed lines
    gross_lines = diff_metrics.get("gross_lines", 0)
    if gross_lines > 60:
        failures.append(f"gross_changed_lines={gross_lines} > 60 limit")

    # Regression check: no sub-metric degrades > 30%
    baseline_sharpe = train_stats.get("sharpe", 0.0)  # using train as reference
    # Compare holdout sub-metrics against baseline holdout where available
    # (baseline holdout stats passed in as separate key if available)
    baseline_hstats = diff_metrics.get("baseline_holdout_stats", {})
    if baseline_hstats:
        b_sharpe = baseline_hstats.get("sharpe", 0.0)
        b_pf = baseline_hstats.get("profit_factor", 0.0)
        b_dd = baseline_hstats.get("max_drawdown", 0.0)

        if b_sharpe > 0 and holdout_sharpe < b_sharpe * 0.70:
            failures.append(
                f"regression: holdout_sharpe degraded "
                f"{(1 - holdout_sharpe/b_sharpe)*100:.1f}% > 30%"
            )
        if b_pf > 0 and holdout_pf < b_pf * 0.70:
            failures.append(
                f"regression: holdout_profit_factor degraded "
                f"{(1 - holdout_pf/b_pf)*100:.1f}% > 30%"
            )
        if b_dd > 0 and holdout_dd > b_dd * 1.30:
            failures.append(
                f"regression: holdout_max_drawdown degraded "
                f"{(holdout_dd/b_dd - 1)*100:.1f}% > 30%"
            )

    # Tier 4 gate: custom_* hooks require >= 5 KEPT results in results.tsv
    has_custom = any(
        config.get(k) is not None
        for k in ("custom_signal_fn", "custom_entry_filter_fn", "custom_exit_fn")
    )
    if has_custom:
        kept_count = _count_kept_results()
        if kept_count < 5:
            failures.append(
                f"tier4_gate: custom hooks present but only {kept_count} KEPT "
                "results in results.tsv (need >= 5)"
            )

    return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# 6. Nightly drift check
# ---------------------------------------------------------------------------


def check_nightly_drift_cap() -> tuple[bool, int]:
    """Return (within_cap, cumulative_diff_lines) for KEPT experiments tonight.

    TSV columns (0-indexed):
        0=timestamp  1=experiment_num  2=baseline_score  3=train_score
        4=holdout_score  5=delta  6=status  7=train_trades  8=holdout_trades
        9=diff_lines  10=description
    """
    if not RESULTS_TSV.exists():
        return True, 0

    today = datetime.date.today().isoformat()
    total_diff = 0

    with open(RESULTS_TSV) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 10:
                continue
            ts = parts[0]
            status = parts[6] if len(parts) > 6 else ""
            diff_lines_raw = parts[9] if len(parts) > 9 else "0"
            if ts.startswith(today) and status == "KEPT":
                try:
                    total_diff += int(diff_lines_raw)
                except ValueError:
                    pass

    return total_diff <= NIGHTLY_DIFF_CAP, total_diff


def _count_kept_results() -> int:
    """Count total KEPT entries in results.tsv."""
    if not RESULTS_TSV.exists():
        return 0
    count = 0
    with open(RESULTS_TSV) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 6 and parts[6] == "KEPT":
                count += 1
    return count


# ---------------------------------------------------------------------------
# 7. Diff metrics
# ---------------------------------------------------------------------------


def compute_diff_metrics(baseline_path: str, experiment_path: str) -> dict:
    """Count additions/deletions between two files using unified diff.

    Returns:
        additions   — lines added (+ lines)
        deletions   — lines removed (- lines)
        net_lines   — additions - deletions
        gross_lines — additions + deletions
    """
    try:
        with open(baseline_path) as f:
            baseline_lines = f.readlines()
        with open(experiment_path) as f:
            experiment_lines = f.readlines()
    except FileNotFoundError as e:
        return {"additions": 0, "deletions": 0, "net_lines": 0, "gross_lines": 0, "error": str(e)}

    diff = list(difflib.unified_diff(baseline_lines, experiment_lines, lineterm=""))
    additions = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
    deletions = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))

    return {
        "additions": additions,
        "deletions": deletions,
        "net_lines": additions - deletions,
        "gross_lines": additions + deletions,
    }


# ---------------------------------------------------------------------------
# 8. Config loader helper
# ---------------------------------------------------------------------------


def _load_config_from_file(path: str) -> dict:
    """Import get_strategy_config() from an arbitrary .py file."""
    spec = importlib.util.spec_from_file_location("_config_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "get_strategy_config"):
        raise AttributeError(f"{path} must define get_strategy_config() -> dict")
    return mod.get_strategy_config()


# ---------------------------------------------------------------------------
# 9. Main evaluate() function
# ---------------------------------------------------------------------------


def evaluate(
    experiment_config: dict,
    baseline_config: dict | None = None,
    experiment_file: str | None = None,
    baseline_file: str | None = None,
    symbol: str = "NIFTY",
) -> dict:
    """Compare experiment config against baseline on train+holdout data.

    Args:
        experiment_config: dict of strategy parameters for the experiment
        baseline_config:   dict of strategy parameters for the baseline
                           (defaults to autoresearch.baseline.get_strategy_config())
        experiment_file:   path to experiment .py file (for diff metrics)
        baseline_file:     path to baseline .py file (for diff metrics)
        symbol:            index symbol (default NIFTY)

    Returns a result dict with keys:
        passed, failures, train_stats, holdout_stats,
        baseline_train_stats, baseline_holdout_stats,
        train_score, holdout_score, baseline_score, delta,
        diff_metrics, guardrail_details
    """
    if baseline_config is None:
        from autoresearch.baseline import get_strategy_config
        baseline_config = get_strategy_config()

    train_data, holdout_data = load_cached_data()

    # Run baseline on both splits
    baseline_train_stats = run_backtest(baseline_config, train_data, symbol)
    baseline_holdout_stats = run_backtest(baseline_config, holdout_data, symbol)
    baseline_holdout_score = baseline_holdout_stats["composite_score"]

    # Run experiment on both splits
    exp_train_stats = run_backtest(experiment_config, train_data, symbol)
    exp_holdout_stats = run_backtest(experiment_config, holdout_data, symbol)

    # Diff metrics
    diff_metrics: dict = {"additions": 0, "deletions": 0, "net_lines": 0, "gross_lines": 0}
    if experiment_file and baseline_file:
        diff_metrics = compute_diff_metrics(baseline_file, experiment_file)
    elif experiment_file:
        default_baseline = str(_HERE / "baseline.py")
        diff_metrics = compute_diff_metrics(default_baseline, experiment_file)

    # Attach baseline holdout stats for regression checks
    diff_metrics["baseline_holdout_stats"] = baseline_holdout_stats

    # Apply guardrails
    passed, failures = check_guardrails(
        train_stats=exp_train_stats,
        holdout_stats=exp_holdout_stats,
        baseline_holdout_score=baseline_holdout_score,
        diff_metrics=diff_metrics,
        config=experiment_config,
        holdout_df=holdout_data,
    )

    delta = exp_holdout_stats["composite_score"] - baseline_holdout_score

    return {
        "passed": passed,
        "failures": failures,
        "train_stats": exp_train_stats,
        "holdout_stats": exp_holdout_stats,
        "baseline_train_stats": baseline_train_stats,
        "baseline_holdout_stats": baseline_holdout_stats,
        "train_score": exp_train_stats["composite_score"],
        "holdout_score": exp_holdout_stats["composite_score"],
        "baseline_score": baseline_holdout_score,
        "delta": delta,
        "diff_metrics": {k: v for k, v in diff_metrics.items() if k != "baseline_holdout_stats"},
        "guardrail_details": failures,
    }


# ---------------------------------------------------------------------------
# 10. CLI
# ---------------------------------------------------------------------------


def _fmt_stats(label: str, stats: dict) -> str:
    dd_pct = stats.get("max_drawdown", 0.0) * 100
    return (
        f"{label}:\n"
        f"  trades={stats.get('total_trades', 0)}  "
        f"sharpe={stats.get('sharpe', 0):.3f}  "
        f"pf={stats.get('profit_factor', 0):.3f}  "
        f"dd={dd_pct:.1f}%  "
        f"wr={stats.get('win_rate', 0):.1f}%  "
        f"score={stats.get('composite_score', 0):.4f}"
    )


def cli_dry_run(symbol: str = "NIFTY"):
    """Validate baseline only — print score, stats, timing, estimated experiments/night."""
    from autoresearch.baseline import get_strategy_config
    config = get_strategy_config()

    print("Loading cached data...")
    train_data, holdout_data = load_cached_data()
    print(f"  Train bars:   {len(train_data)}")
    print(f"  Holdout bars: {len(holdout_data)}")

    print("\nRunning baseline backtest on train...")
    t0 = time.perf_counter()
    train_stats = run_backtest(config, train_data, symbol)
    train_elapsed = time.perf_counter() - t0

    print("Running baseline backtest on holdout...")
    t1 = time.perf_counter()
    holdout_stats = run_backtest(config, holdout_data, symbol)
    holdout_elapsed = time.perf_counter() - t1

    total_elapsed = train_elapsed + holdout_elapsed

    print(f"\n{_fmt_stats('Train   ', train_stats)}")
    print(f"{_fmt_stats('Holdout ', holdout_stats)}")

    print(f"\nTiming:")
    print(f"  Train backtest:   {train_elapsed:.2f}s")
    print(f"  Holdout backtest: {holdout_elapsed:.2f}s")
    print(f"  Total per eval:   {total_elapsed:.2f}s")

    night_seconds = 6 * 3600  # 6-hour window
    est_exps = int(night_seconds / total_elapsed) if total_elapsed > 0 else 0
    # Account for ~2x overhead (Claude thinking, file I/O, TSV writes)
    est_exps_adjusted = est_exps // 2
    print(f"  Est experiments/night (6h): ~{est_exps_adjusted}")

    # Drift check
    within_cap, cumulative_diff = check_nightly_drift_cap()
    print(f"\nNightly drift: {cumulative_diff} / {NIGHTLY_DIFF_CAP} KEPT diff lines today")

    print("\nDry-run PASSED — baseline validated." if train_stats["total_trades"] >= 20
          else "\nWARNING: baseline produced fewer than 20 trades on train set.")


def cli_evaluate(experiment_file: str, symbol: str = "NIFTY"):
    """Compare experiment vs baseline, output JSON result."""
    print(f"Loading experiment: {experiment_file}")
    exp_config = _load_config_from_file(experiment_file)

    from autoresearch.baseline import get_strategy_config
    baseline_config = get_strategy_config()

    baseline_file = str(_HERE / "baseline.py")

    print("Running evaluation...")
    t0 = time.perf_counter()
    result = evaluate(
        experiment_config=exp_config,
        baseline_config=baseline_config,
        experiment_file=experiment_file,
        baseline_file=baseline_file,
        symbol=symbol,
    )
    elapsed = time.perf_counter() - t0

    result["elapsed_seconds"] = round(elapsed, 2)

    # Pretty print key stats
    print(f"\n{_fmt_stats('Train   ', result['train_stats'])}")
    print(f"{_fmt_stats('Holdout ', result['holdout_stats'])}")
    print(f"\nBaseline holdout score: {result['baseline_score']:.4f}")
    print(f"Experiment holdout score: {result['holdout_score']:.4f}")
    print(f"Delta: {result['delta']:+.4f}")
    print(f"\nDiff: +{result['diff_metrics']['additions']} -{result['diff_metrics']['deletions']} lines")

    if result["passed"]:
        print("\nRESULT: PASSED — all guardrails satisfied")
    else:
        print(f"\nRESULT: FAILED — {len(result['failures'])} guardrail(s) failed:")
        for f in result["failures"]:
            print(f"  - {f}")

    print(f"\nElapsed: {elapsed:.2f}s")

    # Output JSON to stdout (caller can capture)
    print("\n--- JSON ---")
    serializable = {k: v for k, v in result.items() if k != "guardrail_details"}
    print(json.dumps(serializable, indent=2, default=str))


def cli_benchmark(symbol: str = "NIFTY"):
    """Time a single backtest and print timing estimates."""
    from autoresearch.baseline import get_strategy_config
    config = get_strategy_config()

    print("Loading cached data...")
    train_data, holdout_data = load_cached_data()

    print(f"Benchmarking single backtest ({len(train_data)} bars)...")
    times = []
    for i in range(3):
        t0 = time.perf_counter()
        run_backtest(config, train_data, symbol)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s")

    avg = sum(times) / len(times)
    print(f"\nAverage: {avg:.3f}s")
    print(f"Per-eval (train+holdout): ~{avg * 2:.1f}s")
    holdout_frac = len(holdout_data) / max(len(train_data), 1)
    est_holdout = avg * holdout_frac
    print(f"Est holdout backtest: ~{est_holdout:.2f}s")
    full_eval = avg + est_holdout
    print(f"Full eval (train+holdout): ~{full_eval:.2f}s")
    night_seconds = 6 * 3600
    print(f"Est experiments in 6h (2x overhead): ~{int(night_seconds / full_eval / 2)}")


def main():
    parser = argparse.ArgumentParser(description="Autoresearch evaluate.py — backtest judge")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Validate baseline only")
    mode.add_argument("--evaluate", action="store_true", help="Compare experiment vs baseline")
    mode.add_argument("--benchmark", action="store_true", help="Time a single backtest")

    parser.add_argument("--experiment", type=str, help="Path to experiment .py file (for --evaluate)")
    parser.add_argument("--symbol", default="NIFTY", help="Symbol (default: NIFTY)")

    args = parser.parse_args()

    if args.dry_run:
        cli_dry_run(symbol=args.symbol)
    elif args.evaluate:
        if not args.experiment:
            parser.error("--evaluate requires --experiment <path>")
        cli_evaluate(experiment_file=args.experiment, symbol=args.symbol)
    elif args.benchmark:
        cli_benchmark(symbol=args.symbol)


if __name__ == "__main__":
    main()
