"""
cache_data.py — Weekly data refresh for autoresearch backtests.

Downloads 2.5 years of Nifty OHLCV + India VIX via yfinance,
merges them, saves to data/backtest_cache.parquet, then re-validates
the baseline strategy against the new holdout window.

Usage:
    python3 autoresearch/cache_data.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — allow running from project root or autoresearch/
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datetime import datetime, timedelta

import pandas as pd

CACHE_PATH = _ROOT / "data" / "backtest_cache.parquet"
NIFTY_TICKER = "^NSEI"
VIX_TICKER = "^INDIAVIX"
DOWNLOAD_INTERVAL = "1d"


# ---------------------------------------------------------------------------
# 1. Telegram alert
# ---------------------------------------------------------------------------

def _telegram_send(text: str) -> None:
    """Send an HTML-formatted Telegram message. Fails silently."""
    token = os.environ.get("DEAL_BOT_TOKEN", "")
    forum_chat_id = os.environ.get("TELEGRAM_FORUM_CHAT_ID", "")
    topic_id = os.environ.get("TELEGRAM_TOPIC_STOCKS", "")
    if not token or not forum_chat_id:
        print("[telegram] env vars missing — skipping alert")
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload: dict = {
            "chat_id": forum_chat_id,
            "text": text,
            "parse_mode": "HTML",
        }
        if topic_id:
            payload["message_thread_id"] = topic_id
        # Write to temp file and POST — avoids shell escaping issues
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as fh:
            tmp_path = fh.name
            fh.write(urllib.parse.urlencode(payload))
        with open(tmp_path, "rb") as fh:
            req = urllib.request.Request(
                url,
                data=fh.read(),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            urllib.request.urlopen(req, timeout=10)
    except Exception as exc:
        print(f"[telegram] send failed: {exc}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 2. Data download and cache
# ---------------------------------------------------------------------------

def refresh_cache() -> pd.DataFrame:
    """Download Nifty + VIX, merge, ffill VIX, save parquet. Returns merged df."""
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("yfinance not installed — run: pip install yfinance")

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365 * 2 + 210)).strftime("%Y-%m-%d")

    print(f"[cache_data] Downloading {NIFTY_TICKER} ({start_date} to {end_date}) ...")
    nifty_raw = yf.download(
        NIFTY_TICKER,
        start=start_date,
        end=end_date,
        interval=DOWNLOAD_INTERVAL,
        auto_adjust=True,
        progress=False,
    )
    if nifty_raw.empty:
        raise ValueError(f"No data returned for {NIFTY_TICKER}")

    print(f"[cache_data] Downloading {VIX_TICKER} ({start_date} to {end_date}) ...")
    vix_raw = yf.download(
        VIX_TICKER,
        start=start_date,
        end=end_date,
        interval=DOWNLOAD_INTERVAL,
        auto_adjust=True,
        progress=False,
    )
    if vix_raw.empty:
        raise ValueError(f"No data returned for {VIX_TICKER}")

    # Flatten MultiIndex columns if present (yfinance >= 0.2 returns them)
    if isinstance(nifty_raw.columns, pd.MultiIndex):
        nifty_raw.columns = nifty_raw.columns.get_level_values(0)
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_raw.columns = vix_raw.columns.get_level_values(0)

    # Normalise index to date-only (no timezone)
    nifty_raw.index = pd.to_datetime(nifty_raw.index).normalize().tz_localize(None)
    vix_raw.index = pd.to_datetime(vix_raw.index).normalize().tz_localize(None)

    # Keep only OHLCV from Nifty
    ohlcv_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in nifty_raw.columns]
    nifty = nifty_raw[ohlcv_cols].copy()

    # Extract VIX close
    vix_close = vix_raw["Close"].rename("VIX")

    # Merge on date index; left-join keeps all Nifty trading days
    merged = nifty.join(vix_close, how="left")

    # Forward-fill VIX for Nifty days when VIX data is missing (holidays, gaps)
    merged["VIX"] = merged["VIX"].ffill()

    # Drop any rows where Nifty OHLCV is entirely NaN
    merged.dropna(subset=["Close"], inplace=True)

    # Sort by date ascending
    merged.sort_index(inplace=True)

    # Save
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(CACHE_PATH)

    rows = len(merged)
    start = merged.index[0].date()
    end = merged.index[-1].date()
    print(f"[cache_data] Saved {rows} rows ({start} → {end}) to {CACHE_PATH}")
    return merged


# ---------------------------------------------------------------------------
# 3. Baseline validation against new holdout
# ---------------------------------------------------------------------------

def validate_baseline() -> dict:
    """Re-run baseline on fresh holdout; alert if degraded. Returns stats dict."""
    from autoresearch.evaluate import load_cached_data, run_backtest, composite_score
    from autoresearch.baseline import get_strategy_config

    print("[cache_data] Loading cached data for validation ...")
    _train, holdout = load_cached_data()
    print(f"[cache_data] Holdout window: {len(holdout)} bars")

    config = get_strategy_config()
    print("[cache_data] Running baseline on holdout ...")
    stats = run_backtest(config, holdout, symbol="NIFTY")

    sharpe = stats.get("sharpe", 0.0)
    pf = stats.get("profit_factor", 0.0)
    score = stats.get("composite_score", 0.0)
    trades = stats.get("total_trades", 0)
    dd_pct = stats.get("max_drawdown", 0.0) * 100

    print(
        f"[cache_data] Holdout results — "
        f"trades={trades}  sharpe={sharpe:.3f}  pf={pf:.3f}  "
        f"dd={dd_pct:.1f}%  score={score:.4f}"
    )

    degraded = sharpe <= 0 or pf <= 1.0

    if degraded:
        msg = (
            "⚠️ <b>Autoresearch: Baseline degraded on new data</b>\n\n"
            f"Holdout trades: {trades}\n"
            f"Sharpe: {sharpe:.3f} (need &gt; 0)\n"
            f"Profit factor: {pf:.3f} (need &gt; 1.0)\n"
            f"Max drawdown: {dd_pct:.1f}%\n"
            f"Composite score: {score:.4f}"
        )
        print(f"[cache_data] DEGRADED — sending Telegram alert")
        _telegram_send(msg)
    else:
        print(f"[cache_data] Baseline healthy on new holdout (sharpe={sharpe:.3f}, pf={pf:.3f})")

    return {
        "degraded": degraded,
        "sharpe": sharpe,
        "profit_factor": pf,
        "composite_score": score,
        "total_trades": trades,
        "max_drawdown_pct": dd_pct,
    }


# ---------------------------------------------------------------------------
# 4. CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="cache_data.py — refresh backtest cache + validate baseline"
    )
    parser.add_argument(
        "--refresh-only",
        action="store_true",
        help="Only download data, skip baseline validation",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate baseline against existing cache, skip download",
    )
    args = parser.parse_args()

    if args.validate_only:
        result = validate_baseline()
    elif args.refresh_only:
        refresh_cache()
    else:
        # Default: both refresh and validate
        refresh_cache()
        result = validate_baseline()
        if result["degraded"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
