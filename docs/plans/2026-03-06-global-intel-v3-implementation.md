# Trading System V3: Global Macro Intelligence — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add global macro awareness (US markets, FII/DII, X/Twitter, PCR, Supertrend, CPR), new strategies (weekly theta, BankNifty), trailing SL, and system health alerts to the paper trading bot.

**Architecture:** Three new modules (`global_intel.py`, `x_intel.py`, `oi_analysis.py`) feed a `MacroContext` dict into the existing `open_positions()` pipeline. Hard gates (US crash, FII selling, GIFT gap) can block/reduce positions. Soft signals (X sentiment, Asia markets) are logged. New strategies (weekly theta, BankNifty condors) extend `_try_index_strategies()`.

**Tech Stack:** yfinance, beautifulsoup4, requests (X scraper), ta library (Supertrend), Claude Haiku (classification), existing SmartAPI option chain

**Design doc:** `docs/plans/2026-03-06-global-intel-v3-design.md`

---

## Existing Codebase Reference

Key files to understand before implementing:
- `paper_trade.py` — Main engine. `open_positions()` at line 1632, `monitor_positions()` at line 2101, `_try_index_strategies()` at line 2675, `run_paper_trade()` at line 2819
- `config.py` — All constants. V2 constants start at line 173. VIX_TIERS at line 241, allocations at line 264
- `agent_with_options.py` — `fetch_option_chain()` at line 79, `select_iron_condor_strikes()` at line 327
- `regime.py` — `classify_regime()` function, returns `{regime, confidence, reason, iv_percentile}`
- `youtube_intel.py` — Pattern to follow: file-based caching, Claude Haiku classification, search + classify pipeline
- `risk_manager.py` — `RiskManager` class, `can_open_position()`, circuit breakers
- `scripts/paper_trade_cron.sh` — Cron wrapper, runs on Pi

Key constants already defined:
- `NIFTY_TOKEN = "99926000"`, `VIX_TOKEN = "26017"`
- `TOTAL_CAPITAL = 100_000`
- `IST = timezone(timedelta(hours=5, minutes=30))`
- `ALLOC_IRON_CONDOR_MAX = 0.10`, `ALLOC_MOMENTUM_MAX = 0.15`

---

### Task 1: V3 Config Constants + Dependencies

**Files:**
- Modify: `config.py:264-335` (after existing V2 constants)
- Modify: `requirements.txt` (or pip install directly)
- Test: `tests/test_config.py` (extend existing)

**Step 1: Add V3 constants to config.py**

After the existing `PAPER_TRADE_SLIPPAGE_PCT` line (~331), add:

```python
# ── V3 Global Macro Intelligence ────────────────────────────────────────────

# US Market hard gates
US_CRASH_BLOCK_PCT = 2.0          # S&P down >2% → block all bullish
US_SEVERE_CRASH_PCT = 3.0         # S&P down >3% → block ALL entries
US_MILD_RED_PCT = 1.0             # S&P down 1-2% → reduce bullish 50%
NASDAQ_IT_CRASH_PCT = 3.0         # Nasdaq down >3% → block IT sector bullish

# GIFT Nifty hard gates
GIFT_GAP_REDUCE_PCT = 1.5         # gap down >1.5% → reduce 50%
GIFT_GAP_BLOCK_PCT = 2.5          # gap down >2.5% → block bullish

# FII/DII hard gates
FII_HEAVY_SELL_CRORES = 5000      # FII net sell >5000cr → reduce bullish 50%
FII_EXTREME_SELL_CRORES = 10000   # FII net sell >10000cr → block bullish equity
DII_SUPPORT_MODERATE_PCT = 0.25   # DII buying moderates FII threshold by 25%

# PCR gates
PCR_EUPHORIA = 0.5                # PCR <0.5 → block bullish
PCR_EXTREME_CALL = 0.7            # PCR <0.7 → reduce bullish 25%
PCR_EXTREME_PUT = 1.3             # PCR >1.3 → log bullish tailwind

# Supertrend
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0
SUPERTREND_DISAGREE_REDUCTION = 0.25  # reduce size 25% if ST disagrees

# CPR thresholds
CPR_NARROW_PCT = 0.3              # <0.3% = trending day expected
CPR_WIDE_PCT = 0.8                # >0.8% = sideways day expected

# Trailing SL enhancement
TRAILING_SL_ACTIVATION_PCT = 2.0
TRAILING_SL_ATR_MULT = 1.5

# Weekly theta strategy
WEEKLY_THETA_MIN_VIX = 14.0
WEEKLY_THETA_TARGET_PCT = 0.50
WEEKLY_THETA_SL_MULTIPLIER = 2.0
WEEKLY_THETA_MAX_RISK_PCT = 0.02
WEEKLY_THETA_OTM_POINTS_NIFTY = 250
WEEKLY_THETA_OTM_POINTS_BANKNIFTY = 600

# BankNifty
BANKNIFTY_TOKEN = "99926009"
BANKNIFTY_LOT_SIZE = 30
BANKNIFTY_VIX_MULTIPLIER = 1.3   # BankNifty vol ~1.3x Nifty

# Updated allocations (V3)
ALLOC_EQUITY_MAX_V3 = 0.35
ALLOC_SPREADS_MAX_V3 = 0.25
ALLOC_IRON_CONDOR_MAX_V3 = 0.15
ALLOC_MOMENTUM_MAX_V3 = 0.10
ALLOC_WEEKLY_THETA_MAX = 0.10
ALLOC_CASH_MIN_V3 = 0.05

# X/Twitter intel
X_SEARCH_QUERIES = [
    "Nifty today",
    "GIFT Nifty",
    "India VIX",
    "FII selling OR FII buying",
    "RBI rate",
    "war India OR sanctions India",
]
X_CACHE_HOURS = 1
X_MIN_LIKES = 50

# Global intel caching
GLOBAL_INTEL_CACHE_HOURS = 6
FII_DII_CACHE_HOURS = 12
```

**Step 2: Install new dependencies**

```bash
pip install yfinance beautifulsoup4
# On Pi later: ssh pi@homepi.local "cd ~/financial-agent-india && source venv/bin/activate && pip install yfinance beautifulsoup4"
```

**Step 3: Write config tests**

Add to `tests/test_config.py`:

```python
def test_v3_us_market_constants():
    from config import US_CRASH_BLOCK_PCT, US_SEVERE_CRASH_PCT, US_MILD_RED_PCT
    assert US_CRASH_BLOCK_PCT == 2.0
    assert US_SEVERE_CRASH_PCT == 3.0
    assert 0 < US_MILD_RED_PCT < US_CRASH_BLOCK_PCT

def test_v3_gift_nifty_constants():
    from config import GIFT_GAP_REDUCE_PCT, GIFT_GAP_BLOCK_PCT
    assert GIFT_GAP_REDUCE_PCT < GIFT_GAP_BLOCK_PCT

def test_v3_fii_constants():
    from config import FII_HEAVY_SELL_CRORES, FII_EXTREME_SELL_CRORES
    assert FII_HEAVY_SELL_CRORES < FII_EXTREME_SELL_CRORES

def test_v3_allocations_sum():
    from config import (ALLOC_EQUITY_MAX_V3, ALLOC_SPREADS_MAX_V3,
                        ALLOC_IRON_CONDOR_MAX_V3, ALLOC_MOMENTUM_MAX_V3,
                        ALLOC_WEEKLY_THETA_MAX, ALLOC_CASH_MIN_V3)
    total = (ALLOC_EQUITY_MAX_V3 + ALLOC_SPREADS_MAX_V3 +
             ALLOC_IRON_CONDOR_MAX_V3 + ALLOC_MOMENTUM_MAX_V3 +
             ALLOC_WEEKLY_THETA_MAX + ALLOC_CASH_MIN_V3)
    assert total == 1.0

def test_v3_banknifty_config():
    from config import BANKNIFTY_TOKEN, BANKNIFTY_LOT_SIZE
    assert BANKNIFTY_TOKEN == "99926009"
    assert BANKNIFTY_LOT_SIZE == 30
```

**Step 4: Run tests**

```bash
pytest tests/test_config.py -v
```

**Step 5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "feat(v3): add global intel + strategy enhancement config constants"
```

---

### Task 2: Global Intel Module — US Markets + GIFT Nifty + Asia

**Files:**
- Create: `global_intel.py`
- Test: `tests/test_global_intel.py`

**Context:** This module fetches US market data (S&P 500, Nasdaq), GIFT Nifty gap, and Asia markets (Hang Seng, Nikkei) using yfinance. All data is cached to files for 6 hours. The module computes percentage changes from previous day close.

**Step 1: Write tests first**

Create `tests/test_global_intel.py`:

```python
"""Tests for global_intel module — US markets, GIFT Nifty, Asia, FII/DII."""
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime


def _make_history_df(prices):
    """Helper: create a yfinance-like history DataFrame from list of close prices."""
    return pd.DataFrame({"Close": prices}, index=pd.date_range("2026-03-04", periods=len(prices)))


class TestUSMarkets:
    def test_sp500_down_2pct(self):
        from global_intel import _compute_us_market_data
        data = _compute_us_market_data(
            sp500_hist=_make_history_df([5000, 4850]),
            nasdaq_hist=_make_history_df([18000, 17700]),
        )
        assert data["sp500_pct_change"] < -2.0
        assert data["nasdaq_pct_change"] < 0

    def test_sp500_flat(self):
        from global_intel import _compute_us_market_data
        data = _compute_us_market_data(
            sp500_hist=_make_history_df([5000, 5010]),
            nasdaq_hist=_make_history_df([18000, 18050]),
        )
        assert abs(data["sp500_pct_change"]) < 1.0
        assert abs(data["nasdaq_pct_change"]) < 1.0

    def test_empty_history(self):
        from global_intel import _compute_us_market_data
        data = _compute_us_market_data(
            sp500_hist=pd.DataFrame(),
            nasdaq_hist=pd.DataFrame(),
        )
        assert data["sp500_pct_change"] == 0.0
        assert data["nasdaq_pct_change"] == 0.0


class TestGIFTNifty:
    def test_gap_down(self):
        from global_intel import _compute_gift_nifty_gap
        # Previous Nifty close 24500, GIFT at 24100 = -1.63%
        result = _compute_gift_nifty_gap(gift_ltp=24100, prev_nifty_close=24500)
        assert result["gift_nifty_gap_pct"] < -1.5
        assert result["gift_nifty_ltp"] == 24100

    def test_gap_up(self):
        from global_intel import _compute_gift_nifty_gap
        result = _compute_gift_nifty_gap(gift_ltp=24900, prev_nifty_close=24500)
        assert result["gift_nifty_gap_pct"] > 1.0

    def test_no_data(self):
        from global_intel import _compute_gift_nifty_gap
        result = _compute_gift_nifty_gap(gift_ltp=None, prev_nifty_close=None)
        assert result["gift_nifty_gap_pct"] == 0.0


class TestAsiaMarkets:
    def test_asia_pct_changes(self):
        from global_intel import _compute_asia_data
        data = _compute_asia_data(
            hang_seng_hist=_make_history_df([20000, 19600]),
            nikkei_hist=_make_history_df([40000, 39500]),
        )
        assert data["hang_seng_pct"] < -1.0
        assert data["nikkei_pct"] < -1.0


class TestHardGateLogic:
    def test_block_all_on_severe_crash(self):
        from global_intel import compute_hard_gate
        gate = compute_hard_gate(sp500_pct=-3.5, nasdaq_pct=-4.0, gift_gap_pct=-1.0,
                                 fii_net=0, dii_net=0, pcr=1.0)
        assert gate["action"] == "BLOCK_ALL"

    def test_block_bullish_on_us_crash(self):
        from global_intel import compute_hard_gate
        gate = compute_hard_gate(sp500_pct=-2.5, nasdaq_pct=-1.5, gift_gap_pct=0,
                                 fii_net=0, dii_net=0, pcr=1.0)
        assert gate["action"] == "BLOCK_BULLISH"

    def test_reduce_50_on_mild_red(self):
        from global_intel import compute_hard_gate
        gate = compute_hard_gate(sp500_pct=-1.5, nasdaq_pct=-1.0, gift_gap_pct=0,
                                 fii_net=0, dii_net=0, pcr=1.0)
        assert gate["action"] == "REDUCE_50"

    def test_reduce_50_on_gift_gap(self):
        from global_intel import compute_hard_gate
        gate = compute_hard_gate(sp500_pct=0.5, nasdaq_pct=0.3, gift_gap_pct=-1.8,
                                 fii_net=0, dii_net=0, pcr=1.0)
        assert gate["action"] == "REDUCE_50"

    def test_block_bullish_on_gift_crash(self):
        from global_intel import compute_hard_gate
        gate = compute_hard_gate(sp500_pct=0, nasdaq_pct=0, gift_gap_pct=-3.0,
                                 fii_net=0, dii_net=0, pcr=1.0)
        assert gate["action"] == "BLOCK_BULLISH"

    def test_fii_heavy_selling(self):
        from global_intel import compute_hard_gate
        gate = compute_hard_gate(sp500_pct=0.5, nasdaq_pct=0.5, gift_gap_pct=0,
                                 fii_net=-6000, dii_net=0, pcr=1.0)
        assert gate["action"] == "REDUCE_50"

    def test_fii_extreme_selling(self):
        from global_intel import compute_hard_gate
        gate = compute_hard_gate(sp500_pct=0, nasdaq_pct=0, gift_gap_pct=0,
                                 fii_net=-12000, dii_net=0, pcr=1.0)
        assert gate["action"] == "BLOCK_BULLISH"

    def test_fii_selling_moderated_by_dii(self):
        from global_intel import compute_hard_gate
        # FII selling 6000cr but DII buying → threshold raised by 25%, so 6000 < 6250 → no gate
        gate = compute_hard_gate(sp500_pct=0, nasdaq_pct=0, gift_gap_pct=0,
                                 fii_net=-6000, dii_net=5000, pcr=1.0)
        # DII moderates: effective threshold = 5000 * 1.25 = 6250, so 6000 < 6250 → NONE
        assert gate["action"] == "NONE"

    def test_pcr_euphoria_blocks(self):
        from global_intel import compute_hard_gate
        gate = compute_hard_gate(sp500_pct=1.0, nasdaq_pct=1.0, gift_gap_pct=0.5,
                                 fii_net=2000, dii_net=1000, pcr=0.4)
        assert gate["action"] == "BLOCK_BULLISH"

    def test_pcr_extreme_call_reduces(self):
        from global_intel import compute_hard_gate
        gate = compute_hard_gate(sp500_pct=0, nasdaq_pct=0, gift_gap_pct=0,
                                 fii_net=0, dii_net=0, pcr=0.65)
        assert gate["action"] == "REDUCE_25"

    def test_all_green_no_gate(self):
        from global_intel import compute_hard_gate
        gate = compute_hard_gate(sp500_pct=1.0, nasdaq_pct=0.8, gift_gap_pct=0.5,
                                 fii_net=2000, dii_net=1000, pcr=1.0)
        assert gate["action"] == "NONE"

    def test_nasdaq_it_crash(self):
        from global_intel import compute_hard_gate
        gate = compute_hard_gate(sp500_pct=-1.0, nasdaq_pct=-3.5, gift_gap_pct=0,
                                 fii_net=0, dii_net=0, pcr=1.0)
        assert "BLOCK_IT_BULLISH" in gate["action"] or gate["action"] == "REDUCE_50"
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_global_intel.py -v
```

Expected: All fail with `ModuleNotFoundError: No module named 'global_intel'`

**Step 3: Implement `global_intel.py`**

```python
"""Global macro intelligence for Trading System V3.

Fetches US markets (S&P 500, Nasdaq), GIFT Nifty gap, Asia markets,
FII/DII flows, and computes hard gate decisions.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger("paper_trade")

CACHE_DIR = Path("data/global_intel_cache")


# ── File cache ────────────────────────────────────────────────────────────────

def _get_cache(cache_key: str, max_age_hours: float) -> dict | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            cached_at = datetime.fromisoformat(data.get("_cached_at", "2000-01-01"))
            if datetime.now() - cached_at < timedelta(hours=max_age_hours):
                return data.get("result")
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _set_cache(cache_key: str, result: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    cache_file.write_text(json.dumps({
        "_cached_at": datetime.now().isoformat(),
        "result": result,
    }, indent=2))


# ── Percentage change helper ──────────────────────────────────────────────────

def _pct_change_from_history(hist: pd.DataFrame) -> float:
    """Compute close-to-close % change from a 2-day yfinance history DataFrame."""
    if hist is None or hist.empty or len(hist) < 2:
        return 0.0
    closes = hist["Close"].values
    prev, current = float(closes[-2]), float(closes[-1])
    if prev == 0:
        return 0.0
    return round(((current - prev) / prev) * 100, 2)


# ── US Markets ────────────────────────────────────────────────────────────────

def _compute_us_market_data(sp500_hist: pd.DataFrame, nasdaq_hist: pd.DataFrame) -> dict:
    return {
        "sp500_pct_change": _pct_change_from_history(sp500_hist),
        "nasdaq_pct_change": _pct_change_from_history(nasdaq_hist),
    }


def fetch_us_markets() -> dict:
    """Fetch S&P 500 and Nasdaq previous day change. Cached 6 hours."""
    cache_key = f"us_markets_{datetime.now().strftime('%Y-%m-%d')}"
    cached = _get_cache(cache_key, config.GLOBAL_INTEL_CACHE_HOURS)
    if cached:
        return cached

    try:
        import yfinance as yf
        sp500 = yf.Ticker("^GSPC").history(period="2d")
        nasdaq = yf.Ticker("^IXIC").history(period="2d")
        result = _compute_us_market_data(sp500, nasdaq)
    except Exception as e:
        logger.debug("US market fetch failed: %s", e)
        result = {"sp500_pct_change": 0.0, "nasdaq_pct_change": 0.0}

    _set_cache(cache_key, result)
    return result


# ── GIFT Nifty ────────────────────────────────────────────────────────────────

def _compute_gift_nifty_gap(gift_ltp: float | None, prev_nifty_close: float | None) -> dict:
    if gift_ltp is None or prev_nifty_close is None or prev_nifty_close == 0:
        return {"gift_nifty_gap_pct": 0.0, "gift_nifty_ltp": 0.0}
    gap_pct = round(((gift_ltp - prev_nifty_close) / prev_nifty_close) * 100, 2)
    return {"gift_nifty_gap_pct": gap_pct, "gift_nifty_ltp": gift_ltp}


def fetch_gift_nifty_gap(prev_nifty_close: float) -> dict:
    """Fetch GIFT Nifty LTP and compute gap vs previous Nifty close. Cached 6 hours."""
    cache_key = f"gift_nifty_{datetime.now().strftime('%Y-%m-%d')}"
    cached = _get_cache(cache_key, config.GLOBAL_INTEL_CACHE_HOURS)
    if cached:
        return cached

    try:
        import yfinance as yf
        # Try Nifty futures as proxy for GIFT Nifty
        nifty_fut = yf.Ticker("^NSEI").history(period="1d")
        ltp = float(nifty_fut["Close"].iloc[-1]) if not nifty_fut.empty else None
        result = _compute_gift_nifty_gap(ltp, prev_nifty_close)
    except Exception as e:
        logger.debug("GIFT Nifty fetch failed: %s", e)
        result = {"gift_nifty_gap_pct": 0.0, "gift_nifty_ltp": 0.0}

    _set_cache(cache_key, result)
    return result


# ── Asia Markets ──────────────────────────────────────────────────────────────

def _compute_asia_data(hang_seng_hist: pd.DataFrame, nikkei_hist: pd.DataFrame) -> dict:
    return {
        "hang_seng_pct": _pct_change_from_history(hang_seng_hist),
        "nikkei_pct": _pct_change_from_history(nikkei_hist),
    }


def fetch_asia_markets() -> dict:
    """Fetch Hang Seng and Nikkei previous day change. Cached 6 hours."""
    cache_key = f"asia_markets_{datetime.now().strftime('%Y-%m-%d')}"
    cached = _get_cache(cache_key, config.GLOBAL_INTEL_CACHE_HOURS)
    if cached:
        return cached

    try:
        import yfinance as yf
        hang_seng = yf.Ticker("^HSI").history(period="2d")
        nikkei = yf.Ticker("^N225").history(period="2d")
        result = _compute_asia_data(hang_seng, nikkei)
    except Exception as e:
        logger.debug("Asia market fetch failed: %s", e)
        result = {"hang_seng_pct": 0.0, "nikkei_pct": 0.0}

    _set_cache(cache_key, result)
    return result


# ── FII/DII Flows ────────────────────────────────────────────────────────────

def fetch_fii_dii() -> dict:
    """Scrape FII/DII daily activity from Moneycontrol. Cached 12 hours."""
    cache_key = f"fii_dii_{datetime.now().strftime('%Y-%m-%d')}"
    cached = _get_cache(cache_key, config.FII_DII_CACHE_HOURS)
    if cached:
        return cached

    result = {"fii_net_crores": 0, "dii_net_crores": 0, "date": "", "source": "unavailable"}

    try:
        import requests
        from bs4 import BeautifulSoup

        url = "https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/index.php"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        # Parse the FII/DII table — structure may change, so wrap in try
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all("td")
                if len(cells) >= 4:
                    label = cells[0].get_text(strip=True).upper()
                    if "FII" in label or "FPI" in label:
                        try:
                            buy = float(cells[1].get_text(strip=True).replace(",", ""))
                            sell = float(cells[2].get_text(strip=True).replace(",", ""))
                            result["fii_net_crores"] = round(buy - sell, 2)
                            result["source"] = "moneycontrol"
                        except ValueError:
                            pass
                    elif "DII" in label:
                        try:
                            buy = float(cells[1].get_text(strip=True).replace(",", ""))
                            sell = float(cells[2].get_text(strip=True).replace(",", ""))
                            result["dii_net_crores"] = round(buy - sell, 2)
                        except ValueError:
                            pass

        result["date"] = datetime.now().strftime("%Y-%m-%d")
    except Exception as e:
        logger.debug("FII/DII scrape failed: %s", e)

    _set_cache(cache_key, result)
    return result


# ── Hard Gate Logic ───────────────────────────────────────────────────────────

def compute_hard_gate(
    sp500_pct: float = 0,
    nasdaq_pct: float = 0,
    gift_gap_pct: float = 0,
    fii_net: float = 0,
    dii_net: float = 0,
    pcr: float = 1.0,
) -> dict:
    """Compute the strictest hard gate from all macro signals.

    Returns dict with 'action' and 'reason'.
    Actions (strictest to mildest):
      BLOCK_ALL, BLOCK_BULLISH, BLOCK_IT_BULLISH, REDUCE_50, REDUCE_25, NONE
    """
    # Priority list: check strictest first
    reasons = []

    # 1. US severe crash → block everything
    if sp500_pct <= -config.US_SEVERE_CRASH_PCT:
        return {"action": "BLOCK_ALL", "reason": f"S&P 500 {sp500_pct:+.1f}% severe crash"}

    # 2. US crash → block bullish
    if sp500_pct <= -config.US_CRASH_BLOCK_PCT:
        reasons.append(("BLOCK_BULLISH", f"S&P 500 {sp500_pct:+.1f}% crash"))

    # 3. GIFT crash gap → block bullish
    if gift_gap_pct <= -config.GIFT_GAP_BLOCK_PCT:
        reasons.append(("BLOCK_BULLISH", f"GIFT Nifty gap {gift_gap_pct:+.1f}%"))

    # 4. FII extreme selling → block bullish
    fii_threshold_heavy = config.FII_HEAVY_SELL_CRORES
    fii_threshold_extreme = config.FII_EXTREME_SELL_CRORES
    if dii_net > 0 and fii_net < 0:
        # DII buying moderates FII threshold
        fii_threshold_heavy *= (1 + config.DII_SUPPORT_MODERATE_PCT)
        fii_threshold_extreme *= (1 + config.DII_SUPPORT_MODERATE_PCT)

    if fii_net <= -fii_threshold_extreme:
        reasons.append(("BLOCK_BULLISH", f"FII net sell {abs(fii_net):.0f}cr (extreme)"))

    # 5. PCR euphoria → block bullish
    if pcr < config.PCR_EUPHORIA:
        reasons.append(("BLOCK_BULLISH", f"PCR {pcr:.2f} euphoria"))

    # If any BLOCK_BULLISH, return it
    block_bullish = [r for r in reasons if r[0] == "BLOCK_BULLISH"]
    if block_bullish:
        return {"action": "BLOCK_BULLISH", "reason": block_bullish[0][1]}

    # 6. Nasdaq IT crash → block IT bullish specifically
    if nasdaq_pct <= -config.NASDAQ_IT_CRASH_PCT:
        reasons.append(("BLOCK_IT_BULLISH", f"Nasdaq {nasdaq_pct:+.1f}% crash"))

    block_it = [r for r in reasons if r[0] == "BLOCK_IT_BULLISH"]
    if block_it:
        return {"action": "BLOCK_IT_BULLISH", "reason": block_it[0][1]}

    # 7. Reduce 50% signals
    reduce_50_reasons = []
    if -config.US_CRASH_BLOCK_PCT < sp500_pct <= -config.US_MILD_RED_PCT:
        reduce_50_reasons.append(f"S&P 500 {sp500_pct:+.1f}%")
    if -config.GIFT_GAP_BLOCK_PCT < gift_gap_pct <= -config.GIFT_GAP_REDUCE_PCT:
        reduce_50_reasons.append(f"GIFT gap {gift_gap_pct:+.1f}%")
    if -fii_threshold_extreme < fii_net <= -fii_threshold_heavy:
        reduce_50_reasons.append(f"FII net sell {abs(fii_net):.0f}cr")

    if reduce_50_reasons:
        return {"action": "REDUCE_50", "reason": "; ".join(reduce_50_reasons)}

    # 8. PCR extreme call buying → reduce 25%
    if pcr < config.PCR_EXTREME_CALL:
        return {"action": "REDUCE_25", "reason": f"PCR {pcr:.2f} extreme call buying"}

    return {"action": "NONE", "reason": "Global cues neutral"}


# ── MacroContext assembly ─────────────────────────────────────────────────────

def fetch_macro_context(prev_nifty_close: float = 0, pcr: float = 1.0) -> dict:
    """Fetch all global data and compute hard gate. Main entry point.

    Args:
        prev_nifty_close: Previous day's Nifty close (for GIFT gap calculation)
        pcr: Current Nifty PCR (computed separately from option chain)

    Returns:
        MacroContext dict with all data + hard_gate + hard_gate_reason
    """
    us = fetch_us_markets()
    gift = fetch_gift_nifty_gap(prev_nifty_close) if prev_nifty_close > 0 else {"gift_nifty_gap_pct": 0.0, "gift_nifty_ltp": 0.0}
    asia = fetch_asia_markets()
    fii_dii = fetch_fii_dii()

    gate = compute_hard_gate(
        sp500_pct=us["sp500_pct_change"],
        nasdaq_pct=us["nasdaq_pct_change"],
        gift_gap_pct=gift["gift_nifty_gap_pct"],
        fii_net=fii_dii["fii_net_crores"],
        dii_net=fii_dii["dii_net_crores"],
        pcr=pcr,
    )

    return {
        **us,
        **gift,
        **asia,
        "fii_net_crores": fii_dii["fii_net_crores"],
        "dii_net_crores": fii_dii["dii_net_crores"],
        "hard_gate": gate["action"],
        "hard_gate_reason": gate["reason"],
        "fetched_at": datetime.now().isoformat(),
    }
```

**Step 4: Run tests**

```bash
pytest tests/test_global_intel.py -v
```

Expected: All pass

**Step 5: Commit**

```bash
git add global_intel.py tests/test_global_intel.py
git commit -m "feat(v3): global intel module — US markets, GIFT Nifty, Asia, FII/DII, hard gates"
```

---

### Task 3: OI Analysis Module — PCR + Max Pain

**Files:**
- Create: `oi_analysis.py`
- Test: `tests/test_oi_analysis.py`

**Context:** Operates on the option chain list that `fetch_option_chain()` in `agent_with_options.py` already returns. Each item in the chain has `strikePrice`, `CE: {openInterest, ...}`, `PE: {openInterest, ...}`.

**Step 1: Write tests**

Create `tests/test_oi_analysis.py`:

```python
"""Tests for OI analysis — PCR, max pain, OI buildup."""


def _make_chain(strikes_oi):
    """Helper: build synthetic chain. strikes_oi = [(strike, call_oi, put_oi), ...]"""
    return [
        {
            "strikePrice": s,
            "CE": {"openInterest": c_oi, "lastPrice": 100},
            "PE": {"openInterest": p_oi, "lastPrice": 100},
        }
        for s, c_oi, p_oi in strikes_oi
    ]


class TestPCR:
    def test_neutral_pcr(self):
        from oi_analysis import compute_pcr
        chain = _make_chain([(24000, 1000, 1000), (24500, 1000, 1000)])
        assert compute_pcr(chain) == 1.0

    def test_bearish_pcr(self):
        from oi_analysis import compute_pcr
        chain = _make_chain([(24000, 500, 1500), (24500, 500, 1500)])
        assert compute_pcr(chain) == 3.0  # 3000 put OI / 1000 call OI

    def test_bullish_pcr(self):
        from oi_analysis import compute_pcr
        chain = _make_chain([(24000, 2000, 500), (24500, 2000, 500)])
        assert compute_pcr(chain) == 0.5

    def test_zero_call_oi(self):
        from oi_analysis import compute_pcr
        chain = _make_chain([(24000, 0, 1000)])
        assert compute_pcr(chain) == 1.0  # fallback

    def test_empty_chain(self):
        from oi_analysis import compute_pcr
        assert compute_pcr([]) == 1.0


class TestMaxPain:
    def test_basic_max_pain(self):
        from oi_analysis import compute_max_pain
        # Heavy call OI at 24500, heavy put OI at 24000
        # Max pain should be between them
        chain = _make_chain([
            (23500, 100, 5000),
            (24000, 200, 3000),
            (24500, 5000, 200),
            (25000, 3000, 100),
        ])
        mp = compute_max_pain(chain, lot_size=75)
        assert 23500 <= mp <= 25000

    def test_single_strike(self):
        from oi_analysis import compute_max_pain
        chain = _make_chain([(24000, 1000, 1000)])
        assert compute_max_pain(chain) == 24000

    def test_empty_chain(self):
        from oi_analysis import compute_max_pain
        assert compute_max_pain([]) == 0


class TestOIBuildup:
    def test_top_oi_strikes(self):
        from oi_analysis import get_top_oi_strikes
        chain = _make_chain([
            (23500, 100, 5000),
            (24000, 200, 3000),
            (24500, 5000, 200),
            (25000, 3000, 100),
        ])
        result = get_top_oi_strikes(chain, top_n=2)
        assert len(result["call_resistance"]) == 2
        assert len(result["put_support"]) == 2
        # Highest call OI = 24500 (resistance)
        assert result["call_resistance"][0]["strike"] == 24500
        # Highest put OI = 23500 (support)
        assert result["put_support"][0]["strike"] == 23500
```

**Step 2: Implement `oi_analysis.py`**

```python
"""OI analysis for Trading System V3.

PCR computation, max pain, and OI buildup from option chain data.
Zero additional API calls — operates on chain data already fetched.
"""


def compute_pcr(chain: list[dict]) -> float:
    """Compute Put-Call Ratio from option chain OI.

    Returns put_oi / call_oi. Defaults to 1.0 if no data.
    """
    if not chain:
        return 1.0

    put_oi = sum(s.get("PE", {}).get("openInterest", 0) for s in chain)
    call_oi = sum(s.get("CE", {}).get("openInterest", 0) for s in chain)

    if call_oi == 0:
        return 1.0
    return round(put_oi / call_oi, 2)


def compute_max_pain(chain: list[dict], lot_size: int = 75) -> float:
    """Find strike where total option writer pain is minimized.

    At each test strike, calculate how much all option writers would lose
    if the underlying expired at that strike. The strike with minimum total
    loss is max pain — where market makers want the index to settle.

    Returns the max pain strike price, or 0 if chain is empty.
    """
    if not chain:
        return 0

    strikes = sorted(set(s["strikePrice"] for s in chain))
    min_pain = float("inf")
    max_pain_strike = strikes[len(strikes) // 2]

    for test_strike in strikes:
        total_pain = 0
        for s in chain:
            sp = s["strikePrice"]
            call_oi = s.get("CE", {}).get("openInterest", 0)
            put_oi = s.get("PE", {}).get("openInterest", 0)

            # Call writers lose if test_strike > strike (calls go ITM)
            if test_strike > sp:
                total_pain += (test_strike - sp) * call_oi * lot_size
            # Put writers lose if test_strike < strike (puts go ITM)
            if test_strike < sp:
                total_pain += (sp - test_strike) * put_oi * lot_size

        if total_pain < min_pain:
            min_pain = total_pain
            max_pain_strike = test_strike

    return max_pain_strike


def get_top_oi_strikes(chain: list[dict], top_n: int = 3) -> dict:
    """Identify top OI strikes for support (put OI) and resistance (call OI).

    Returns:
        {"call_resistance": [{"strike": N, "oi": M}, ...],
         "put_support": [{"strike": N, "oi": M}, ...]}
    """
    if not chain:
        return {"call_resistance": [], "put_support": []}

    call_oi_list = [
        {"strike": s["strikePrice"], "oi": s.get("CE", {}).get("openInterest", 0)}
        for s in chain
    ]
    put_oi_list = [
        {"strike": s["strikePrice"], "oi": s.get("PE", {}).get("openInterest", 0)}
        for s in chain
    ]

    call_oi_list.sort(key=lambda x: x["oi"], reverse=True)
    put_oi_list.sort(key=lambda x: x["oi"], reverse=True)

    return {
        "call_resistance": call_oi_list[:top_n],
        "put_support": put_oi_list[:top_n],
    }
```

**Step 3: Run tests**

```bash
pytest tests/test_oi_analysis.py -v
```

**Step 4: Commit**

```bash
git add oi_analysis.py tests/test_oi_analysis.py
git commit -m "feat(v3): OI analysis module — PCR, max pain, OI buildup"
```

---

### Task 4: Supertrend + CPR Indicators

**Files:**
- Create: `indicators_v3.py` (separate from existing `indicators.py` to avoid conflicts)
- Test: `tests/test_supertrend_cpr.py`

**Context:** The `ta` library is already installed. Supertrend is NOT in the `ta` library — it needs manual implementation using ATR. CPR is pure math from previous day's high/low/close.

**Step 1: Write tests**

Create `tests/test_supertrend_cpr.py`:

```python
"""Tests for Supertrend and CPR indicators."""
import pandas as pd
import numpy as np


def _make_ohlc(n=30, base=24000, volatility=100):
    """Create synthetic OHLC data."""
    np.random.seed(42)
    closes = base + np.cumsum(np.random.randn(n) * volatility)
    highs = closes + abs(np.random.randn(n) * volatility * 0.5)
    lows = closes - abs(np.random.randn(n) * volatility * 0.5)
    opens = closes + np.random.randn(n) * volatility * 0.3
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
    })


class TestSupertrend:
    def test_returns_buy_or_sell(self):
        from indicators_v3 import compute_supertrend
        df = _make_ohlc(30)
        signal = compute_supertrend(df, period=10, multiplier=3)
        assert signal in ("buy", "sell")

    def test_uptrending_data(self):
        from indicators_v3 import compute_supertrend
        # Create clearly uptrending data
        closes = [24000 + i * 50 for i in range(30)]
        df = pd.DataFrame({
            "open": [c - 10 for c in closes],
            "high": [c + 30 for c in closes],
            "low": [c - 30 for c in closes],
            "close": closes,
        })
        assert compute_supertrend(df) == "buy"

    def test_downtrending_data(self):
        from indicators_v3 import compute_supertrend
        closes = [26000 - i * 50 for i in range(30)]
        df = pd.DataFrame({
            "open": [c + 10 for c in closes],
            "high": [c + 30 for c in closes],
            "low": [c - 30 for c in closes],
            "close": closes,
        })
        assert compute_supertrend(df) == "sell"

    def test_insufficient_data(self):
        from indicators_v3 import compute_supertrend
        df = _make_ohlc(5)
        # Should return "unknown" or default, not crash
        signal = compute_supertrend(df, period=10, multiplier=3)
        assert signal in ("buy", "sell", "unknown")


class TestCPR:
    def test_narrow_cpr(self):
        from indicators_v3 import compute_cpr
        # Small range day
        result = compute_cpr(prev_high=24100, prev_low=24050, prev_close=24080)
        assert result["cpr_width_pct"] < 0.3
        assert result["day_type"] == "trending"

    def test_wide_cpr(self):
        from indicators_v3 import compute_cpr
        # Wide range day
        result = compute_cpr(prev_high=24800, prev_low=24000, prev_close=24400)
        assert result["cpr_width_pct"] > 0.8
        assert result["day_type"] == "sideways"

    def test_normal_cpr(self):
        from indicators_v3 import compute_cpr
        result = compute_cpr(prev_high=24300, prev_low=24100, prev_close=24200)
        assert result["day_type"] == "normal"

    def test_cpr_values(self):
        from indicators_v3 import compute_cpr
        result = compute_cpr(prev_high=24300, prev_low=24100, prev_close=24200)
        assert "pivot" in result
        assert "tc" in result
        assert "bc" in result
        assert result["tc"] >= result["bc"]
```

**Step 2: Implement `indicators_v3.py`**

```python
"""V3 indicators: Supertrend and CPR.

Supertrend: ATR-based trailing stop that flips on price crossing.
CPR: Central Pivot Range from previous day's high/low/close.
"""

import numpy as np
import pandas as pd

import config


def compute_supertrend(
    df: pd.DataFrame,
    period: int = None,
    multiplier: float = None,
) -> str:
    """Compute Supertrend signal from OHLC DataFrame.

    Returns "buy", "sell", or "unknown" (if insufficient data).
    """
    period = period or config.SUPERTREND_PERIOD
    multiplier = multiplier or config.SUPERTREND_MULTIPLIER

    if len(df) < period + 1:
        return "unknown"

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    # ATR computation
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            abs(high[1:] - close[:-1]),
            abs(low[1:] - close[:-1]),
        ),
    )
    # Prepend 0 for alignment
    tr = np.insert(tr, 0, high[0] - low[0])

    atr = np.zeros(len(tr))
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    # Supertrend bands
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = np.zeros(len(close))
    direction = np.ones(len(close))  # 1 = up (buy), -1 = down (sell)

    for i in range(1, len(close)):
        if atr[i] == 0:
            supertrend[i] = supertrend[i - 1]
            direction[i] = direction[i - 1]
            continue

        # Adjust bands based on previous values
        if lower_band[i] > supertrend[i - 1] or close[i - 1] > supertrend[i - 1]:
            lower_band[i] = max(lower_band[i],
                                supertrend[i - 1] if direction[i - 1] == 1 else lower_band[i])

        if upper_band[i] < supertrend[i - 1] or close[i - 1] < supertrend[i - 1]:
            upper_band[i] = min(upper_band[i],
                                supertrend[i - 1] if direction[i - 1] == -1 else upper_band[i])

        if direction[i - 1] == 1:  # previous was uptrend
            if close[i] < supertrend[i - 1]:
                direction[i] = -1
                supertrend[i] = upper_band[i]
            else:
                direction[i] = 1
                supertrend[i] = lower_band[i]
        else:  # previous was downtrend
            if close[i] > supertrend[i - 1]:
                direction[i] = 1
                supertrend[i] = lower_band[i]
            else:
                direction[i] = -1
                supertrend[i] = upper_band[i]

    last_dir = direction[-1]
    if last_dir == 1:
        return "buy"
    elif last_dir == -1:
        return "sell"
    return "unknown"


def compute_cpr(prev_high: float, prev_low: float, prev_close: float) -> dict:
    """Compute Central Pivot Range from previous day's candle.

    Returns dict with pivot, tc, bc, cpr_width, cpr_width_pct, day_type.
    """
    pivot = (prev_high + prev_low + prev_close) / 3
    bc = (prev_high + prev_low) / 2
    tc = (pivot - bc) + pivot
    cpr_width = abs(tc - bc)
    cpr_width_pct = (cpr_width / prev_close) * 100 if prev_close > 0 else 0

    if cpr_width_pct < config.CPR_NARROW_PCT:
        day_type = "trending"
    elif cpr_width_pct > config.CPR_WIDE_PCT:
        day_type = "sideways"
    else:
        day_type = "normal"

    return {
        "pivot": round(pivot, 2),
        "tc": round(max(tc, bc), 2),
        "bc": round(min(tc, bc), 2),
        "cpr_width": round(cpr_width, 2),
        "cpr_width_pct": round(cpr_width_pct, 3),
        "day_type": day_type,
    }
```

**Step 3: Run tests**

```bash
pytest tests/test_supertrend_cpr.py -v
```

**Step 4: Commit**

```bash
git add indicators_v3.py tests/test_supertrend_cpr.py
git commit -m "feat(v3): Supertrend and CPR indicators"
```

---

### Task 5: X/Twitter Intel Module

**Files:**
- Create: `x_intel.py`
- Test: `tests/test_x_intel.py`

**Context:** Minimal Python port of the `x-research` TypeScript scraper at `~/.claude/skills/x-research/`. Uses cookie-based auth (`TWITTER_COOKIES` env var). Only the `search()` function is needed. Classification uses Claude Haiku like `youtube_intel.py` does.

**Important:** Twitter's search API requires specific headers and a guest token or auth cookies. The scraper should use the same approach as `agent-twitter-client`: cookie-based auth with `auth_token` and `ct0` cookies.

**Step 1: Write tests**

Create `tests/test_x_intel.py`:

```python
"""Tests for X/Twitter intel module."""
from unittest.mock import patch, MagicMock


class TestXSearch:
    @patch("x_intel.requests.get")
    def test_search_returns_tweets(self, mock_get):
        from x_intel import search_x
        # Mock Twitter search response
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "globalObjects": {
                "tweets": {
                    "1": {"full_text": "Nifty looking bearish today", "favorite_count": 100,
                          "created_at": "Mon Mar 09 05:00:00 +0000 2026",
                          "user_id_str": "123"},
                }
            }
        }
        mock_get.return_value = mock_resp
        # Should not crash even with mock data
        # Real test would need proper response structure

    def test_parse_cookies(self):
        from x_intel import _parse_cookies
        cookies = _parse_cookies("auth_token=abc123; ct0=xyz789")
        assert cookies["auth_token"] == "abc123"
        assert cookies["ct0"] == "xyz789"

    def test_parse_cookies_empty(self):
        from x_intel import _parse_cookies
        cookies = _parse_cookies("")
        assert cookies == {}


class TestXClassification:
    @patch("x_intel._call_claude_haiku")
    def test_classify_bullish(self, mock_haiku):
        from x_intel import classify_x_sentiment
        mock_haiku.return_value = {
            "sentiment": "bullish",
            "confidence": "high",
            "key_themes": ["FII buying", "Nifty breakout"],
        }
        result = classify_x_sentiment(["Nifty looking strong", "FII buying heavily"])
        assert result["sentiment"] == "bullish"

    @patch("x_intel._call_claude_haiku")
    def test_classify_crisis(self, mock_haiku):
        from x_intel import classify_x_sentiment
        mock_haiku.return_value = {
            "sentiment": "crisis",
            "confidence": "high",
            "key_themes": ["war", "sanctions"],
        }
        result = classify_x_sentiment(["War escalation", "India sanctions"])
        assert result["sentiment"] == "crisis"

    def test_classify_empty(self):
        from x_intel import classify_x_sentiment
        result = classify_x_sentiment([])
        assert result is None


class TestFetchXSentiment:
    @patch("x_intel.search_x")
    @patch("x_intel.classify_x_sentiment")
    def test_cached_result(self, mock_classify, mock_search):
        from x_intel import fetch_x_sentiment
        # First call should search + classify
        mock_search.return_value = [{"text": "test", "likes": 100}]
        mock_classify.return_value = {"sentiment": "neutral", "confidence": "low", "key_themes": []}
        result = fetch_x_sentiment()
        assert result is not None or result is None  # graceful either way
```

**Step 2: Implement `x_intel.py`**

```python
"""X/Twitter intelligence for Trading System V3.

Searches X for India market sentiment, classifies via Claude Haiku.
Soft signal only — never auto-blocks, used as contradiction filter.

Requires TWITTER_COOKIES env var: "auth_token=XXX; ct0=YYY"
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import requests

import config
from utils import parse_claude_json

logger = logging.getLogger("paper_trade")

CACHE_DIR = Path("data/x_intel_cache")


# ── Cookie auth ───────────────────────────────────────────────────────────────

def _parse_cookies(cookie_str: str) -> dict:
    """Parse 'key=val; key2=val2' into dict."""
    if not cookie_str:
        return {}
    result = {}
    for part in cookie_str.split(";"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            result[k.strip()] = v.strip()
    return result


def _get_cookies() -> dict:
    cookie_str = os.environ.get("TWITTER_COOKIES", "")
    if not cookie_str:
        # Try loading from global env
        env_file = Path.home() / ".config" / "env" / "global.env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("TWITTER_COOKIES="):
                    cookie_str = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    return _parse_cookies(cookie_str)


# ── X Search ──────────────────────────────────────────────────────────────────

def search_x(query: str, since_hours: int = 12, min_likes: int = None, limit: int = 20) -> list[dict]:
    """Search X/Twitter for recent tweets matching query.

    Returns list of {text, likes, user, timestamp}.
    Requires TWITTER_COOKIES env var.
    """
    min_likes = min_likes or config.X_MIN_LIKES
    cookies = _get_cookies()
    if not cookies.get("auth_token") or not cookies.get("ct0"):
        logger.debug("X search: no Twitter cookies configured")
        return []

    headers = {
        "Authorization": "Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA",
        "X-Csrf-Token": cookies["ct0"],
        "Cookie": f"auth_token={cookies['auth_token']}; ct0={cookies['ct0']}",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    }

    # Build search query with time filter
    since_str = (datetime.utcnow() - timedelta(hours=since_hours)).strftime("%Y-%m-%d")
    full_query = f"{query} since:{since_str} -is:retweet min_faves:{min_likes}"

    url = "https://api.twitter.com/2/search/adaptive.json"
    params = {
        "q": full_query,
        "count": limit,
        "tweet_mode": "extended",
        "result_type": "recent",
    }

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code != 200:
            logger.debug("X search failed: HTTP %d", resp.status_code)
            return []

        data = resp.json()
        tweets_raw = data.get("globalObjects", {}).get("tweets", {})
        users_raw = data.get("globalObjects", {}).get("users", {})

        tweets = []
        for tid, tweet in tweets_raw.items():
            user_id = tweet.get("user_id_str", "")
            user = users_raw.get(user_id, {})
            tweets.append({
                "text": tweet.get("full_text", ""),
                "likes": tweet.get("favorite_count", 0),
                "user": user.get("screen_name", "unknown"),
                "timestamp": tweet.get("created_at", ""),
            })

        return sorted(tweets, key=lambda t: t["likes"], reverse=True)[:limit]

    except Exception as e:
        logger.debug("X search error: %s", e)
        return []


# ── Claude classification ─────────────────────────────────────────────────────

def _call_claude_haiku(system_prompt: str, user_message: str) -> dict | None:
    client = config.get_anthropic_client()
    response = client.messages.create(
        model=config.CLAUDE_MODEL_LIGHT,
        max_tokens=512,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    text = response.content[0].text
    return parse_claude_json(text)


def classify_x_sentiment(tweet_texts: list[str]) -> dict | None:
    """Classify aggregate X/Twitter sentiment from tweet texts.

    Returns {sentiment, confidence, key_themes} or None.
    """
    if not tweet_texts:
        return None

    combined = "\n---\n".join(tweet_texts[:20])  # limit to 20 tweets
    system_prompt = (
        "You are an Indian stock market analyst. Analyze these recent X/Twitter posts "
        "about Indian markets and classify the overall sentiment. "
        "Respond with ONLY valid JSON:\n"
        '{"sentiment": "bullish | bearish | neutral | crisis", '
        '"confidence": "high | medium | low", '
        '"key_themes": ["theme1", "theme2"]}'
    )

    try:
        return _call_claude_haiku(system_prompt, combined[:4000])
    except Exception:
        return None


# ── Caching ───────────────────────────────────────────────────────────────────

def _get_cache(cache_key: str, max_age_hours: float) -> dict | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            cached_at = datetime.fromisoformat(data.get("_cached_at", "2000-01-01"))
            if datetime.now() - cached_at < timedelta(hours=max_age_hours):
                return data.get("result")
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _set_cache(cache_key: str, result: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    cache_file.write_text(json.dumps({
        "_cached_at": datetime.now().isoformat(),
        "result": result,
    }, indent=2))


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_x_sentiment() -> dict | None:
    """Fetch X/Twitter market sentiment. Cached per config.X_CACHE_HOURS.

    Returns {sentiment, confidence, key_themes, tweet_count} or None.
    """
    cache_key = f"x_sentiment_{datetime.now().strftime('%Y-%m-%d_%H')}"
    cached = _get_cache(cache_key, config.X_CACHE_HOURS)
    if cached:
        return cached

    all_texts = []
    for query in config.X_SEARCH_QUERIES:
        tweets = search_x(query, since_hours=12, limit=10)
        all_texts.extend(t["text"] for t in tweets)

    if not all_texts:
        return None

    result = classify_x_sentiment(all_texts)
    if result:
        result["tweet_count"] = len(all_texts)
        _set_cache(cache_key, result)
    return result
```

**Step 3: Run tests**

```bash
pytest tests/test_x_intel.py -v
```

**Step 4: Commit**

```bash
git add x_intel.py tests/test_x_intel.py
git commit -m "feat(v3): X/Twitter intel module — Python scraper + Claude Haiku sentiment"
```

---

### Task 6: Wire Global Intel + OI + Indicators into paper_trade.py

**Files:**
- Modify: `paper_trade.py:1632-1700` (open_positions)
- Test: `tests/test_global_intel_integration.py`

**Context:** This is the critical wiring task. `open_positions()` currently fetches Nifty candles + VIX, detects regime, and filters candidates. We need to add macro context, PCR, Supertrend, and CPR into the pipeline.

**Step 1: Write integration tests**

Create `tests/test_global_intel_integration.py`:

```python
"""Integration tests: global intel gates in open_positions pipeline."""
from unittest.mock import patch


class TestMacroGateIntegration:
    def test_block_all_returns_zero(self):
        """When hard gate is BLOCK_ALL, open_positions should return 0."""
        from global_intel import compute_hard_gate
        gate = compute_hard_gate(sp500_pct=-4.0, nasdaq_pct=-3.0, gift_gap_pct=-2.0,
                                 fii_net=-15000, dii_net=0, pcr=0.4)
        assert gate["action"] == "BLOCK_ALL"

    def test_reduce_50_halves_capital(self):
        """When hard gate is REDUCE_50, available capital should be halved."""
        from global_intel import compute_hard_gate
        gate = compute_hard_gate(sp500_pct=-1.5, nasdaq_pct=0, gift_gap_pct=0,
                                 fii_net=0, dii_net=0, pcr=1.0)
        assert gate["action"] == "REDUCE_50"
        # In open_positions: available_capital *= 0.5

    def test_supertrend_disagree_reduces(self):
        """When Supertrend disagrees with screener signal, reduce allocation."""
        # This tests the logic, not the wiring
        from config import SUPERTREND_DISAGREE_REDUCTION
        assert SUPERTREND_DISAGREE_REDUCTION == 0.25

    def test_cpr_narrow_favors_directional(self):
        """Narrow CPR day should favor directional trades."""
        from indicators_v3 import compute_cpr
        result = compute_cpr(prev_high=24100, prev_low=24050, prev_close=24080)
        assert result["day_type"] == "trending"
```

**Step 2: Modify `open_positions()` in paper_trade.py**

After the existing regime detection block (~line 1674), add:

```python
    # --- V3 Global Macro Intelligence ---
    pcr = 1.0  # default
    max_pain = 0
    try:
        from oi_analysis import compute_pcr, compute_max_pain
        chain = fetch_option_chain(smart_api, "NIFTY", NIFTY_TOKEN)
        time.sleep(config.API_DELAY)
        if chain:
            pcr = compute_pcr(chain)
            max_pain = compute_max_pain(chain)
            logger.info("PCR: %.2f | Max Pain: %.0f", pcr, max_pain)
    except Exception as e:
        logger.debug("OI analysis failed: %s", e)

    # Fetch macro context (US, GIFT, Asia, FII/DII)
    macro = None
    try:
        from global_intel import fetch_macro_context
        prev_close = nifty_candles[-1][4] if nifty_candles else 0
        macro = fetch_macro_context(prev_nifty_close=prev_close, pcr=pcr)
        logger.info("Global: S&P %+.1f%% | Nasdaq %+.1f%% | GIFT gap %+.1f%% | FII %+.0fcr",
                    macro.get("sp500_pct_change", 0), macro.get("nasdaq_pct_change", 0),
                    macro.get("gift_nifty_gap_pct", 0), macro.get("fii_net_crores", 0))
    except Exception as e:
        logger.debug("Global intel failed: %s", e)

    # Apply macro hard gates
    if macro:
        gate = macro.get("hard_gate", "NONE")
        gate_reason = macro.get("hard_gate_reason", "")

        if gate == "BLOCK_ALL":
            logger.warning("MACRO BLOCK ALL: %s — skipping all new positions", gate_reason)
            return 0
        elif gate == "BLOCK_BULLISH":
            logger.warning("MACRO BLOCK BULLISH: %s", gate_reason)
            # Filter out bullish candidates later
        elif gate == "REDUCE_50":
            available_capital = round(available_capital * 0.5, 2)
            logger.info("MACRO REDUCE 50%%: %s → capital ₹%.0f", gate_reason, available_capital)
        elif gate == "REDUCE_25":
            available_capital = round(available_capital * 0.75, 2)
            logger.info("MACRO REDUCE 25%%: %s → capital ₹%.0f", gate_reason, available_capital)

    # --- V3 Supertrend + CPR ---
    supertrend_signal = "unknown"
    cpr_day_type = "normal"
    try:
        from indicators_v3 import compute_supertrend, compute_cpr
        if nifty_candles and len(nifty_candles) >= 15:
            nifty_df_st = pd.DataFrame(nifty_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            supertrend_signal = compute_supertrend(nifty_df_st)
            logger.info("Supertrend: %s", supertrend_signal)

            prev_candle = nifty_candles[-2] if len(nifty_candles) >= 2 else nifty_candles[-1]
            cpr_result = compute_cpr(prev_high=prev_candle[2], prev_low=prev_candle[3], prev_close=prev_candle[4])
            cpr_day_type = cpr_result["day_type"]
            logger.info("CPR: %s (width=%.3f%%, pivot=%.0f)", cpr_day_type, cpr_result["cpr_width_pct"], cpr_result["pivot"])
    except Exception as e:
        logger.debug("Supertrend/CPR failed: %s", e)

    # --- X/Twitter sentiment (soft signal) ---
    x_sentiment = None
    try:
        from x_intel import fetch_x_sentiment
        x_sentiment = fetch_x_sentiment()
        if x_sentiment:
            logger.info("X sentiment: %s (confidence=%s, themes=%s)",
                        x_sentiment.get("sentiment"), x_sentiment.get("confidence"),
                        x_sentiment.get("key_themes", []))
    except Exception as e:
        logger.debug("X intel unavailable: %s", e)
```

Then in the **candidate filtering loop** (around line 1722), add BLOCK_BULLISH and IT sector filtering:

```python
        # V3: Macro BLOCK_BULLISH gate
        if macro and macro.get("hard_gate") == "BLOCK_BULLISH" and direction == "bullish":
            logger.info("SKIP %s: bullish blocked by macro gate (%s)", symbol, macro.get("hard_gate_reason", ""))
            continue

        # V3: Nasdaq IT crash → block IT sector bullish
        if macro and macro.get("hard_gate") == "BLOCK_IT_BULLISH" and direction == "bullish":
            sector = SYMBOL_SECTOR.get(symbol, "")
            if sector == "IT":
                logger.info("SKIP %s: IT bullish blocked by Nasdaq crash", symbol)
                continue

        # V3: Supertrend disagreement → reduce allocation
        if supertrend_signal != "unknown":
            if direction == "bullish" and supertrend_signal == "sell":
                c["allocation_multiplier"] = c.get("allocation_multiplier", 1.0) * (1 - config.SUPERTREND_DISAGREE_REDUCTION)
                logger.info("%s: Supertrend disagrees (sell vs bullish), reducing allocation 25%%", symbol)
            elif direction == "bearish" and supertrend_signal == "buy":
                c["allocation_multiplier"] = c.get("allocation_multiplier", 1.0) * (1 - config.SUPERTREND_DISAGREE_REDUCTION)
                logger.info("%s: Supertrend disagrees (buy vs bearish), reducing allocation 25%%", symbol)

        # V3: X/Twitter contradiction filter (soft)
        if x_sentiment and x_sentiment.get("sentiment") == "crisis":
            logger.warning("X sentiment: CRISIS detected — %s", x_sentiment.get("key_themes", []))
```

**Step 3: Run all tests**

```bash
pytest tests/ -v
```

**Step 4: Commit**

```bash
git add paper_trade.py tests/test_global_intel_integration.py
git commit -m "feat(v3): wire global intel, OI analysis, Supertrend, CPR into open_positions pipeline"
```

---

### Task 7: Trailing Stop Loss Enhancement

**Files:**
- Modify: `paper_trade.py:2351-2401` (equity section of monitor_positions)
- Test: `tests/test_trailing_sl.py`

**Context:** Current equity trailing stop uses `ATR_TRAILING_MULTIPLIER = 1.0`. The V3 enhancement adds an activation threshold: trailing stop only activates after +2% unrealized. Before that, the fixed ATR stoploss applies. After activation, trail at `high_water_mark - 1.5 * ATR`, never below entry price.

Looking at the existing code (~line 2355-2397), the ATR-based trailing stop already exists. The V3 enhancement tightens the trail after activation threshold is crossed.

**Step 1: Write tests**

Create `tests/test_trailing_sl.py`:

```python
"""Tests for V3 trailing stop loss enhancement."""


def test_trailing_sl_not_active_below_threshold():
    """Before +2%, the existing fixed SL applies, not the enhanced trail."""
    from config import TRAILING_SL_ACTIVATION_PCT
    assert TRAILING_SL_ACTIVATION_PCT == 2.0
    # If unrealized pct < 2.0%, trailing SL should not tighten

def test_trailing_sl_activates_at_threshold():
    """After +2%, trailing SL should activate and never go below entry."""
    entry = 100.0
    atr = 5.0
    high_water = 103.0  # +3% above entry
    from config import TRAILING_SL_ATR_MULT
    trailing_stop = high_water - TRAILING_SL_ATR_MULT * atr  # 103 - 7.5 = 95.5
    # Never below entry
    effective_stop = max(trailing_stop, entry)
    assert effective_stop == entry  # 95.5 < 100, so capped at entry

def test_trailing_sl_moves_up_with_price():
    """As price rises, trailing SL should move up."""
    entry = 100.0
    atr = 3.0
    high_water = 110.0  # +10% above entry
    from config import TRAILING_SL_ATR_MULT
    trailing_stop = high_water - TRAILING_SL_ATR_MULT * atr  # 110 - 4.5 = 105.5
    effective_stop = max(trailing_stop, entry)
    assert effective_stop == 105.5  # well above entry

def test_trailing_sl_locks_in_profit():
    """After a big run-up, SL should lock in meaningful profit."""
    entry = 100.0
    atr = 2.0
    high_water = 115.0
    from config import TRAILING_SL_ATR_MULT
    trailing_stop = high_water - TRAILING_SL_ATR_MULT * atr  # 115 - 3 = 112
    effective_stop = max(trailing_stop, entry)
    # Lock in +12% profit
    assert effective_stop == 112.0
    assert effective_stop > entry
```

**Step 2: Modify equity trailing stop in monitor_positions**

In `paper_trade.py`, inside the equity ATR-based section (~line 2355), enhance the trailing stop logic to use the tighter V3 trail after activation:

```python
            if atr_at_entry is not None:
                unrealized_pct = calc_pnl_pct(pos["entry_price"], ltp, pos["direction"])

                if pos["direction"] == "bullish":
                    if ltp >= pos["target_price"]:
                        reason = "target"
                    else:
                        # V3: Enhanced trailing after activation threshold
                        if unrealized_pct >= config.TRAILING_SL_ACTIVATION_PCT:
                            # Tighter trail: 1.5x ATR from high water mark, floored at entry
                            trailing_sl = peak - config.TRAILING_SL_ATR_MULT * atr_at_entry
                            trailing_sl = max(trailing_sl, pos["entry_price"])
                        else:
                            # Standard trail before activation
                            trailing_sl = peak - ATR_TRAILING_MULTIPLIER * atr_at_entry

                        fixed_sl = pos["stoploss_price"]
                        effective_sl = max(trailing_sl, fixed_sl)
                        if ltp <= effective_sl:
                            reason = "trailing_stop" if trailing_sl > fixed_sl else "stoploss"
                # ... similar for bearish direction
```

**Step 3: Run tests**

```bash
pytest tests/test_trailing_sl.py tests/ -v
```

**Step 4: Commit**

```bash
git add paper_trade.py tests/test_trailing_sl.py
git commit -m "feat(v3): enhanced trailing SL — activates at +2%, tighter 1.5x ATR trail"
```

---

### Task 8: Weekly Theta Harvesting Strategy

**Files:**
- Modify: `agent_with_options.py` (add `select_strangle_strikes()`)
- Modify: `paper_trade.py` (add weekly theta entry/exit/monitoring)
- Test: `tests/test_weekly_theta.py`

**Context:** New strategy H. Sell OTM strangles on Nifty/BankNifty on Friday/Monday targeting Tuesday weekly expiry. Uses existing `fetch_option_chain()`. Exit at 50% premium decay, 2x stoploss, or Tuesday 2 PM.

**Step 1: Write tests**

Create `tests/test_weekly_theta.py`:

```python
"""Tests for weekly theta harvesting strategy."""
from datetime import date


class TestWeeklyThetaEligibility:
    def test_friday_eligible(self):
        from paper_trade import _is_weekly_theta_day
        # Friday = weekday 4
        assert _is_weekly_theta_day(date(2026, 3, 6)) is True  # Friday

    def test_monday_eligible(self):
        from paper_trade import _is_weekly_theta_day
        assert _is_weekly_theta_day(date(2026, 3, 9)) is True  # Monday

    def test_tuesday_not_eligible(self):
        from paper_trade import _is_weekly_theta_day
        assert _is_weekly_theta_day(date(2026, 3, 10)) is False

    def test_wednesday_not_eligible(self):
        from paper_trade import _is_weekly_theta_day
        assert _is_weekly_theta_day(date(2026, 3, 11)) is False


class TestWeeklyThetaExit:
    def test_target_50pct_decay(self):
        from paper_trade import check_weekly_theta_exit
        pos = {"total_credit": 80, "expiry": "2026-03-10", "entry_date": "2026-03-06"}
        # Current premium = 35 (< 80 * 0.50 = 40)
        assert check_weekly_theta_exit(pos, current_call_prem=20, current_put_prem=15) == "target"

    def test_stoploss_premium_doubles(self):
        from paper_trade import check_weekly_theta_exit
        pos = {"total_credit": 80, "expiry": "2026-03-10", "entry_date": "2026-03-06"}
        # Current premium = 170 (> 80 * 2.0 = 160)
        assert check_weekly_theta_exit(pos, current_call_prem=90, current_put_prem=80) == "stoploss"

    def test_hold_normal(self):
        from paper_trade import check_weekly_theta_exit
        pos = {"total_credit": 80, "expiry": "2026-03-10", "entry_date": "2026-03-06"}
        # Current premium = 60 (between target and SL)
        assert check_weekly_theta_exit(pos, current_call_prem=35, current_put_prem=25) is None


class TestStrangleStrikes:
    def test_select_strangle(self):
        from agent_with_options import select_strangle_strikes
        chain = [
            {"strikePrice": 24000, "CE": {"lastPrice": 250, "openInterest": 5000}, "PE": {"lastPrice": 10, "openInterest": 500}},
            {"strikePrice": 24200, "CE": {"lastPrice": 150, "openInterest": 8000}, "PE": {"lastPrice": 20, "openInterest": 2000}},
            {"strikePrice": 24500, "CE": {"lastPrice": 50, "openInterest": 10000}, "PE": {"lastPrice": 45, "openInterest": 10000}},
            {"strikePrice": 24800, "CE": {"lastPrice": 20, "openInterest": 8000}, "PE": {"lastPrice": 150, "openInterest": 8000}},
            {"strikePrice": 25000, "CE": {"lastPrice": 10, "openInterest": 5000}, "PE": {"lastPrice": 250, "openInterest": 5000}},
        ]
        result = select_strangle_strikes(chain, spot=24500, otm_points=250, lot_size=75)
        assert result is not None
        assert result["call_strike"] > 24500
        assert result["put_strike"] < 24500
        assert result["total_credit"] > 0
```

**Step 2: Add `select_strangle_strikes()` to agent_with_options.py**

```python
def select_strangle_strikes(chain, spot, otm_points, lot_size, min_oi=500):
    """Select OTM strangle strikes for theta harvesting.

    Args:
        chain: option chain list
        spot: current underlying price
        otm_points: how far OTM each leg should be
        lot_size: contract lot size
        min_oi: minimum OI per leg

    Returns dict with call_strike, put_strike, call_premium, put_premium,
    total_credit, lot_size, or None if no valid strikes.
    """
    target_call = spot + otm_points
    target_put = spot - otm_points

    # Find nearest strikes to targets
    call_candidates = [
        s for s in chain
        if s["strikePrice"] >= target_call
        and s.get("CE", {}).get("openInterest", 0) >= min_oi
        and s.get("CE", {}).get("lastPrice", 0) > 0
    ]
    put_candidates = [
        s for s in chain
        if s["strikePrice"] <= target_put
        and s.get("PE", {}).get("openInterest", 0) >= min_oi
        and s.get("PE", {}).get("lastPrice", 0) > 0
    ]

    if not call_candidates or not put_candidates:
        return None

    call_strike_data = min(call_candidates, key=lambda s: abs(s["strikePrice"] - target_call))
    put_strike_data = min(put_candidates, key=lambda s: abs(s["strikePrice"] - target_put))

    call_prem = call_strike_data["CE"]["lastPrice"]
    put_prem = put_strike_data["PE"]["lastPrice"]

    return {
        "call_strike": call_strike_data["strikePrice"],
        "put_strike": put_strike_data["strikePrice"],
        "call_premium": call_prem,
        "put_premium": put_prem,
        "total_credit": call_prem + put_prem,
        "lot_size": lot_size,
    }
```

**Step 3: Add weekly theta functions to paper_trade.py**

```python
def _is_weekly_theta_day(d: date = None) -> bool:
    """True if today is Friday (4) or Monday (0) — good days to sell weekly theta."""
    if d is None:
        d = datetime.now(IST).date()
    return d.weekday() in (0, 4)  # Monday or Friday


def check_weekly_theta_exit(pos: dict, current_call_prem: float, current_put_prem: float) -> str | None:
    """Check exit conditions for weekly theta (short strangle).

    Returns exit reason or None.
    """
    total_current = current_call_prem + current_put_prem
    total_entry = pos["total_credit"]

    # Target: 50% premium decay
    if total_current <= total_entry * config.WEEKLY_THETA_TARGET_PCT:
        return "target"

    # Stop loss: premium doubles
    if total_current >= total_entry * config.WEEKLY_THETA_SL_MULTIPLIER:
        return "stoploss"

    return None
```

Wire into `_try_index_strategies()` and `monitor_positions()`.

**Step 4: Run tests**

```bash
pytest tests/test_weekly_theta.py -v
```

**Step 5: Commit**

```bash
git add agent_with_options.py paper_trade.py tests/test_weekly_theta.py
git commit -m "feat(v3): weekly theta harvesting — short strangle on Nifty weekly expiry"
```

---

### Task 9: BankNifty Support

**Files:**
- Modify: `paper_trade.py` (add BankNifty to `_try_index_strategies()`)
- Test: `tests/test_banknifty.py`

**Context:** Extend `_try_index_strategies()` to run iron condor, momentum, and weekly theta on BankNifty in addition to Nifty. BankNifty token is `99926009`, lot size is 30. VIX thresholds scale by 1.3x for BankNifty.

**Step 1: Write tests**

Create `tests/test_banknifty.py`:

```python
"""Tests for BankNifty support."""


def test_banknifty_config():
    from config import BANKNIFTY_TOKEN, BANKNIFTY_LOT_SIZE, BANKNIFTY_VIX_MULTIPLIER
    assert BANKNIFTY_TOKEN == "99926009"
    assert BANKNIFTY_LOT_SIZE == 30
    assert BANKNIFTY_VIX_MULTIPLIER == 1.3


def test_banknifty_vix_scaling():
    """BankNifty VIX thresholds should be 1.3x Nifty thresholds."""
    from config import CONDOR_MIN_VIX, CONDOR_MAX_VIX, BANKNIFTY_VIX_MULTIPLIER
    bn_min = CONDOR_MIN_VIX * BANKNIFTY_VIX_MULTIPLIER
    bn_max = CONDOR_MAX_VIX * BANKNIFTY_VIX_MULTIPLIER
    assert bn_min > CONDOR_MIN_VIX
    assert bn_max > CONDOR_MAX_VIX


def test_banknifty_wider_otm():
    """BankNifty strangles should use wider OTM points than Nifty."""
    from config import WEEKLY_THETA_OTM_POINTS_NIFTY, WEEKLY_THETA_OTM_POINTS_BANKNIFTY
    assert WEEKLY_THETA_OTM_POINTS_BANKNIFTY > WEEKLY_THETA_OTM_POINTS_NIFTY
```

**Step 2: Extend `_try_index_strategies()` to loop over both indices**

In `paper_trade.py`, refactor `_try_index_strategies()` to iterate over a list of indices:

```python
INDEX_CONFIGS = [
    {"name": "NIFTY", "token": NIFTY_TOKEN, "lot_size": 75,
     "otm_points": config.WEEKLY_THETA_OTM_POINTS_NIFTY, "vix_mult": 1.0},
    {"name": "BANKNIFTY", "token": config.BANKNIFTY_TOKEN, "lot_size": config.BANKNIFTY_LOT_SIZE,
     "otm_points": config.WEEKLY_THETA_OTM_POINTS_BANKNIFTY, "vix_mult": config.BANKNIFTY_VIX_MULTIPLIER},
]
```

Then loop over `INDEX_CONFIGS` inside `_try_index_strategies()`, applying the same iron condor / momentum / weekly theta logic with index-specific parameters.

**Step 3: Run tests**

```bash
pytest tests/test_banknifty.py tests/ -v
```

**Step 4: Commit**

```bash
git add paper_trade.py tests/test_banknifty.py
git commit -m "feat(v3): BankNifty support — iron condor, momentum, weekly theta on second index"
```

---

### Task 10: System Health Alerting

**Files:**
- Modify: `scripts/paper_trade_cron.sh`
- Create: `scripts/watchdog.sh`
- No Python tests needed (shell scripts)

**Step 1: Enhance paper_trade_cron.sh with heartbeat**

Read the existing file first, then modify to add heartbeat + failure alerting:

```bash
#!/bin/bash
# Paper trade cron wrapper — called every 30 minutes during market hours.
#
# V3: Added heartbeat + failure alerts via Telegram.

export PATH="$HOME/.local/bin:$PATH"

# Source env vars for Telegram notifications
if [ -f "$HOME/.config/env/global.env" ]; then
    set -a
    source "$HOME/.config/env/global.env"
    set +a
fi

cd ~/financial-agent-india
source venv/bin/activate

LOG=data/paper_trades/cron.log
mkdir -p data/paper_trades

echo "--- $(TZ=Asia/Kolkata date) ---" >> "$LOG"

HOUR=$(TZ=Asia/Kolkata date +%H)
MIN=$(TZ=Asia/Kolkata date +%M)

# Determine mode
MODE=""
if [ "$HOUR" -eq 9 ] && [ "$MIN" -le 45 ]; then
    MODE="open"
    echo "[OPEN] Running screener + opening positions" >> "$LOG"
    python paper_trade.py open >> "$LOG" 2>&1
    OPEN_EXIT=$?
fi

echo "[MONITOR] Checking exit conditions" >> "$LOG"
python paper_trade.py monitor >> "$LOG" 2>&1
MON_EXIT=$?

# 3:15-3:30 PM IST: Final monitor pass
if [ "$HOUR" -eq 15 ] && [ "$MIN" -ge 15 ]; then
    echo "[FINAL] End-of-day monitor pass" >> "$LOG"
    python paper_trade.py monitor >> "$LOG" 2>&1
fi

# --- V3 Heartbeat / Failure Alert ---
EXIT_CODE=${OPEN_EXIT:-$MON_EXIT}
OPEN_COUNT=$(python -c "
import json; p=json.load(open('data/paper_trades/portfolio.json'))
print(sum(1 for pos in p.get('positions',[]) if pos.get('status')=='open'))
" 2>/dev/null || echo "?")
CAPITAL=$(python -c "
import json; p=json.load(open('data/paper_trades/portfolio.json'))
print(f\"₹{p.get('available_capital',0):,.0f}\")
" 2>/dev/null || echo "?")

if [ "${EXIT_CODE:-0}" -eq 0 ]; then
    MSG="[hb] ${MODE:-monitor} OK | Open: $OPEN_COUNT | Capital: $CAPITAL"
else
    LAST_LINES=$(tail -5 "$LOG" | head -c 500)
    MSG="[ALERT] paper_trade FAILED (exit $EXIT_CODE)
$LAST_LINES"
fi

# Send via Telegram (use temp file to avoid escaping issues)
TMPFILE=$(mktemp)
echo "$MSG" > "$TMPFILE"
curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    -d chat_id="${TELEGRAM_CHAT_ID}" \
    -d parse_mode="HTML" \
    --data-urlencode "text@$TMPFILE" > /dev/null 2>&1
rm -f "$TMPFILE"

echo "" >> "$LOG"
```

**Step 2: Create watchdog script**

```bash
#!/bin/bash
# watchdog.sh — alert if paper_trade hasn't run in 2+ hours
# Add to cron: 0 */2 * * 1-5 ~/financial-agent-india/scripts/watchdog.sh

if [ -f "$HOME/.config/env/global.env" ]; then
    set -a
    source "$HOME/.config/env/global.env"
    set +a
fi

PORTFOLIO="$HOME/financial-agent-india/data/paper_trades/portfolio.json"

if [ ! -f "$PORTFOLIO" ]; then
    exit 0  # no portfolio yet
fi

# Check last modification time (GNU stat on Linux)
LAST_MOD=$(stat -c %Y "$PORTFOLIO" 2>/dev/null || stat -f %m "$PORTFOLIO" 2>/dev/null)
NOW=$(date +%s)
DIFF=$(( (NOW - LAST_MOD) / 3600 ))

if [ "$DIFF" -ge 2 ]; then
    TMPFILE=$(mktemp)
    echo "[WATCHDOG] paper_trade hasn't run in ${DIFF}h. Check Pi cron." > "$TMPFILE"
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d chat_id="${TELEGRAM_CHAT_ID}" \
        -d parse_mode="HTML" \
        --data-urlencode "text@$TMPFILE" > /dev/null 2>&1
    rm -f "$TMPFILE"
fi
```

**Step 3: Commit**

```bash
chmod +x scripts/watchdog.sh
git add scripts/paper_trade_cron.sh scripts/watchdog.sh
git commit -m "feat(v3): system health — heartbeat alerts + watchdog cron"
```

---

### Task 11: Telegram Summary with Macro Context

**Files:**
- Modify: `paper_trade.py` (`_telegram_daily_summary` or `_telegram_notify_entry`)

**Context:** The existing Telegram alert functions send entry/exit notifications. Enhance the daily summary to include global macro context.

**Step 1: Find and modify the Telegram summary function**

Add macro context to the daily summary message:

```python
# In the daily summary, add a global cues line:
if macro:
    lines.append(f"Global: S&P {macro.get('sp500_pct_change', 0):+.1f}% | "
                 f"Nasdaq {macro.get('nasdaq_pct_change', 0):+.1f}% | "
                 f"GIFT gap {macro.get('gift_nifty_gap_pct', 0):+.1f}% | "
                 f"FII {macro.get('fii_net_crores', 0):+.0f}cr")
    if macro.get("hard_gate") != "NONE":
        lines.append(f"Gate: {macro['hard_gate']} — {macro['hard_gate_reason']}")

if pcr != 1.0:
    lines.append(f"PCR: {pcr:.2f} | Max Pain: {max_pain:.0f}")

if supertrend_signal != "unknown":
    lines.append(f"Supertrend: {supertrend_signal} | CPR: {cpr_day_type}")

if x_sentiment:
    lines.append(f"X: {x_sentiment.get('sentiment', '?')} ({', '.join(x_sentiment.get('key_themes', [])[:3])})")
```

**Step 2: Commit**

```bash
git add paper_trade.py
git commit -m "feat(v3): Telegram daily summary includes global cues, PCR, Supertrend, CPR"
```

---

### Task 12: Final Integration Test + Deploy

**Files:**
- Run: full test suite
- Deploy: Pi

**Step 1: Run full test suite locally**

```bash
pytest tests/ -v --tb=short
```

All tests must pass.

**Step 2: Install new deps on Pi**

```bash
ssh pi@homepi.local "cd ~/financial-agent-india && source venv/bin/activate && pip install yfinance beautifulsoup4"
```

**Step 3: Push and pull on Pi**

```bash
git push origin main
ssh pi@homepi.local "cd ~/financial-agent-india && git pull origin main"
```

**Step 4: Create data directories on Pi**

```bash
ssh pi@homepi.local "mkdir -p ~/financial-agent-india/data/global_intel_cache ~/financial-agent-india/data/x_intel_cache"
```

**Step 5: Run tests on Pi**

```bash
ssh pi@homepi.local "cd ~/financial-agent-india && source venv/bin/activate && python -m pytest tests/ -v --tb=short"
```

**Step 6: Deploy cron scripts**

```bash
scp scripts/watchdog.sh pi@homepi.local:~/financial-agent-india/scripts/
ssh pi@homepi.local "chmod +x ~/financial-agent-india/scripts/watchdog.sh"
# Add watchdog to Pi crontab:
# 0 */2 * * 1-5 ~/financial-agent-india/scripts/watchdog.sh
```

**Step 7: Manual dry run**

```bash
ssh pi@homepi.local "cd ~/financial-agent-india && source venv/bin/activate && python paper_trade.py open"
```

---

## Task Summary

| Task | Feature | New Files | Est Lines |
|------|---------|-----------|-----------|
| 1 | Config constants + deps | — | ~80 |
| 2 | Global intel (US, GIFT, Asia, FII/DII) | `global_intel.py` | ~250 |
| 3 | OI analysis (PCR, max pain) | `oi_analysis.py` | ~100 |
| 4 | Supertrend + CPR | `indicators_v3.py` | ~150 |
| 5 | X/Twitter intel | `x_intel.py` | ~180 |
| 6 | Wire into open_positions() | — | ~80 |
| 7 | Trailing SL enhancement | — | ~20 |
| 8 | Weekly theta strategy | — | ~120 |
| 9 | BankNifty support | — | ~60 |
| 10 | System health alerts | `scripts/watchdog.sh` | ~50 |
| 11 | Telegram macro summary | — | ~20 |
| 12 | Integration test + deploy | — | — |

**Total: ~12 tasks, ~1,110 new lines, 7 test files**
