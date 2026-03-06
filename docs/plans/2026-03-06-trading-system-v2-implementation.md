# Trading System V2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform the paper trading bot from equity-only into a multi-strategy, regime-aware system with debit spreads, credit spreads, iron condors, momentum buying, and YouTube intel.

**Architecture:** Three new modules (regime.py, risk_manager.py, youtube_intel.py) plus major extensions to paper_trade.py and config.py. Each strategy is a method on the existing paper_trade flow. Regime detection runs first and gates which strategies are active.

**Tech Stack:** Python 3.11+, AngelOne SmartAPI, Anthropic Claude (haiku for classification), yt-dlp, youtube-transcript-api, existing indicator functions.

**Design doc:** `docs/plans/2026-03-06-trading-system-v2-design.md`

---

## Phase 0: Pre-Implementation Setup

### Task 0: Add dependencies and create data scaffolding

**Files:**
- Modify: `requirements.txt`
- Create: `data/iv_history.json` (empty list `[]`)
- Create: `data/vix_history.json` (empty list `[]`)
- Create: `data/nifty_candles.json` (empty list `[]`)
- Create: `data/earnings_calendar.json` (empty dict with top 20 stocks)
- Create: `data/fo_ban_cache.json` (empty dict `{}`)

**Step 1: Add missing dependencies**

```bash
echo "pandas-ta>=0.3.14b0" >> requirements.txt
echo "yt-dlp" >> requirements.txt
echo "youtube-transcript-api" >> requirements.txt
pip install --user pandas-ta yt-dlp youtube-transcript-api
```

**Step 2: Create empty data files**

```python
# data/earnings_calendar.json — manually maintained, quarterly update
{
    "RELIANCE": ["2026-04-18", "2026-07-17", "2026-10-16", "2027-01-15"],
    "HDFCBANK": ["2026-04-19", "2026-07-18", "2026-10-17", "2027-01-16"],
    "TCS": ["2026-04-10", "2026-07-10", "2026-10-09", "2027-01-09"],
    "INFY": ["2026-04-15", "2026-07-14", "2026-10-13", "2027-01-13"],
    "_note": "Approximate dates. Update quarterly from BSE corporate announcements."
}
```

**Step 3: Verify pandas-ta works**

```bash
python -c "import pandas_ta; print('pandas_ta', pandas_ta.version)"
python -c "from scipy.stats import norm; print('scipy OK')"
```

**Step 4: Commit**

```bash
git add requirements.txt data/
git commit -m "chore: add dependencies and data scaffolding for trading system v2"
```

---

## Phase 1: Foundation (config, holidays, regime detection, risk manager)

### Task 1: Add NSE holidays and constants to config.py

**Files:**
- Modify: `config.py`
- Test: `tests/test_config.py`

**Step 1: Write failing tests**

Add to `tests/test_config.py`:
```python
from datetime import date

def test_nse_holidays_2026_count():
    from config import NSE_HOLIDAYS_2026
    assert len(NSE_HOLIDAYS_2026) == 16

def test_republic_day_is_holiday():
    from config import NSE_HOLIDAYS_2026
    assert date(2026, 1, 26) in NSE_HOLIDAYS_2026

def test_is_trading_day_weekend():
    from config import is_trading_day
    assert is_trading_day(date(2026, 3, 7)) is False  # Saturday

def test_is_trading_day_holiday():
    from config import is_trading_day
    assert is_trading_day(date(2026, 1, 26)) is False  # Republic Day

def test_is_trading_day_normal():
    from config import is_trading_day
    assert is_trading_day(date(2026, 3, 9)) is True  # Monday, not holiday

def test_vix_tiers():
    from config import VIX_TIERS
    assert VIX_TIERS[0] == (12, "extreme_complacency")
    assert VIX_TIERS[-1] == (float("inf"), "crisis")
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/aravindms/financial-agent-india && python -m pytest tests/test_config.py -v -k "holiday or trading_day or vix_tier"`
Expected: FAIL (ImportError — NSE_HOLIDAYS_2026, is_trading_day, VIX_TIERS don't exist)

**Step 3: Implement in config.py**

Add after existing constants (~line 30):
```python
from datetime import date

NSE_HOLIDAYS_2026 = {
    date(2026, 1, 26),   # Republic Day
    date(2026, 2, 26),   # Maha Shivaratri
    date(2026, 3, 10),   # Holi
    date(2026, 3, 30),   # Id-Ul-Fitr
    date(2026, 4, 2),    # Ram Navami
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 6, 5),    # Id-Ul-Adha
    date(2026, 7, 6),    # Muharram
    date(2026, 8, 15),   # Independence Day
    date(2026, 8, 19),   # Janmashtami
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    date(2026, 10, 20),  # Dussehra
    date(2026, 11, 9),   # Diwali Laxmi Pujan
    date(2026, 11, 30),  # Guru Nanak Jayanti
}

def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in NSE_HOLIDAYS_2026

VIX_TIERS = [
    (12, "extreme_complacency"),
    (18, "normal"),
    (22, "elevated"),
    (28, "high"),
    (float("inf"), "crisis"),
]

# Strategy constants
MAX_RISK_PER_TRADE = 0.02          # 2% of capital
MAX_PORTFOLIO_LOSS_CORRELATION = 0.10  # 10% assuming all lose
DAILY_DRAWDOWN_LIMIT = 0.03        # 3%
WEEKLY_DRAWDOWN_LIMIT = 0.05       # 5%
MONTHLY_DRAWDOWN_LIMIT = 0.08      # 8%
TOTAL_DRAWDOWN_HALT = 0.15         # 15% -> full stop
CONSECUTIVE_LOSS_PAUSE = 3         # pause strategy after 3 losses
SPREAD_TIME_EXIT_DAYS = 5          # close spreads after 5 trading days
SPREAD_PROFIT_CAP = 0.80           # close at 80% of max profit
CONDOR_PROFIT_TARGET = 0.50        # close at 50% of max profit
CONDOR_LOSS_LIMIT = 2.0            # close at 2x credit received
MOMENTUM_STOP_PCT = 0.35           # 35% of premium
MOMENTUM_TARGET_PCT = 0.90         # 90% of premium (near double)
MOMENTUM_TIME_EXIT_DAYS = 3        # close after 3 days if < 15% profit
EXPIRY_SAFETY_DAYS = 3             # close stock options E-3
IV_PERCENTILE_MAX_DEBIT = 70       # no debit spreads above 70th pctl
MIN_OI_THRESHOLD = 500             # both legs need OI > 500
MIN_BID_ASK_RATIO = 0.05           # max 5% bid-ask spread
CONDOR_VIX_MIN = 12
CONDOR_VIX_MAX = 18
CONDOR_ADX_MAX = 20
CONDOR_RANGE_DAYS = 7
CONDOR_RANGE_POINTS = 400

# YouTube Intel
YT_CHANNEL_ALLOWLIST = [
    "PR Sundar", "CA Rachana Ranade", "Power of Stocks",
    "Pranjal Kamra", "Nitin Bhatia",
]
YT_MARKET_CACHE_HOURS = 12
YT_STOCK_CACHE_HOURS = 24

# Capital allocation limits
ALLOC_EQUITY_MAX = 0.40
ALLOC_SPREADS_MAX = 0.30
ALLOC_CONDOR_MAX = 0.10
ALLOC_MOMENTUM_MAX = 0.15
ALLOC_CASH_MIN = 0.05
```

Also update `is_market_open()` to use `is_trading_day()`:
```python
def is_market_open(_now=None):
    # ... existing time logic ...
    today = _now.date() if hasattr(_now, 'date') else date.today()
    if not is_trading_day(today):
        return False, "Market closed (holiday or weekend)"
    # ... rest of existing logic ...
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_config.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "feat: add NSE holidays, VIX tiers, strategy constants"
```

---

### Task 2: Update trading day helpers in paper_trade.py

**Files:**
- Modify: `paper_trade.py:728-760`
- Test: `tests/test_paper_trade.py`

**Step 1: Write failing tests**

```python
def test_trading_days_between_skips_holidays():
    from paper_trade import _trading_days_between
    # Republic Day (Jan 26, 2026) is Monday - should be skipped
    result = _trading_days_between("2026-01-23", "2026-01-27")
    assert result == 1  # Fri 23 -> Mon 27, but Mon 26 is holiday, so just Fri

def test_add_trading_days_skips_holidays():
    from paper_trade import _add_trading_days
    # Starting Jan 23 (Fri), add 1 trading day
    # Jan 24 Sat, Jan 25 Sun, Jan 26 Mon (holiday), Jan 27 Tue
    result = _add_trading_days("2026-01-23", 1)
    assert result == "2026-01-27"
```

**Step 2: Run to verify failure**

Run: `python -m pytest tests/test_paper_trade.py -v -k "holiday"`
Expected: FAIL (current code doesn't check holidays)

**Step 3: Update _trading_days_between and _add_trading_days**

In `paper_trade.py`, modify both functions to use `config.is_trading_day()`:

```python
def _trading_days_between(start_date: str, end_date: str) -> int:
    from config import is_trading_day
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    count = 0
    d = start
    while d <= end:
        if is_trading_day(d):
            count += 1
        d += timedelta(days=1)
    return count

def _add_trading_days(start_date: str, n_days: int) -> str:
    from config import is_trading_day
    d = datetime.strptime(start_date, "%Y-%m-%d").date()
    added = 0
    while added < n_days:
        d += timedelta(days=1)
        if is_trading_day(d):
            added += 1
    return d.strftime("%Y-%m-%d")
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_paper_trade.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add paper_trade.py tests/test_paper_trade.py
git commit -m "feat: trading day helpers now skip NSE holidays"
```

---

### Task 3: Create regime.py — Market regime detection

**Files:**
- Create: `regime.py`
- Test: `tests/test_regime.py`

**Step 1: Write failing tests**

Create `tests/test_regime.py`:
```python
import pytest
from regime import classify_regime, RegimeResult

def test_trending_up():
    result = classify_regime(
        adx_values=[22, 23, 25, 26],      # ADX > 20 for 3+ bars
        price_vs_ema="above",              # price > 20 EMA
        price_trend="rising",              # rising 5+ days
        vix=15.0,
        nifty_range_7d=600,                # not sideways
        bb_width_pctl=50,
    )
    assert result.regime == "TRENDING_UP"
    assert result.confidence > 0.7

def test_trending_down():
    result = classify_regime(
        adx_values=[22, 24, 26, 27],
        price_vs_ema="below",
        price_trend="falling",
        vix=16.0,
        nifty_range_7d=500,
        bb_width_pctl=50,
    )
    assert result.regime == "TRENDING_DOWN"
    assert result.confidence > 0.7

def test_sideways():
    result = classify_regime(
        adx_values=[15, 16, 17, 18],       # ADX < 20 for 3+ bars
        price_vs_ema="near",
        price_trend="flat",
        vix=14.0,
        nifty_range_7d=350,                # < 400 pts
        bb_width_pctl=15,                  # BB width near 20-day low
    )
    assert result.regime == "SIDEWAYS"
    assert result.confidence > 0.7

def test_volatile():
    result = classify_regime(
        adx_values=[30, 32, 35, 38],
        price_vs_ema="below",
        price_trend="falling",
        vix=25.0,                          # VIX > 20
        nifty_range_7d=800,
        bb_width_pctl=90,
    )
    assert result.regime == "VOLATILE"

def test_uncertain_when_mixed_signals():
    result = classify_regime(
        adx_values=[19, 20, 21, 19],       # ADX hovering around 20
        price_vs_ema="above",
        price_trend="flat",                # mixed: above EMA but flat
        vix=17.0,
        nifty_range_7d=450,
        bb_width_pctl=50,
    )
    assert result.regime == "UNCERTAIN"
    assert result.confidence < 0.7

def test_vix_spike_override():
    from regime import check_vix_spike
    # VIX jumped 25% in one day
    assert check_vix_spike(20.0, 15.0) is True
    # Normal VIX change
    assert check_vix_spike(15.5, 15.0) is False

def test_get_vix_tier():
    from regime import get_vix_tier
    assert get_vix_tier(10.0) == "extreme_complacency"
    assert get_vix_tier(15.0) == "normal"
    assert get_vix_tier(20.0) == "elevated"
    assert get_vix_tier(25.0) == "high"
    assert get_vix_tier(35.0) == "crisis"

def test_strategy_allowed():
    from regime import is_strategy_allowed
    result = RegimeResult(regime="TRENDING_UP", confidence=0.8, detail="")
    assert is_strategy_allowed(result, "EQUITY_LONG") is True
    assert is_strategy_allowed(result, "BULL_CALL_SPREAD") is True
    assert is_strategy_allowed(result, "BEAR_PUT_SPREAD") is False
    assert is_strategy_allowed(result, "IRON_CONDOR") is False

def test_strategy_allowed_sideways():
    result = RegimeResult(regime="SIDEWAYS", confidence=0.8, detail="")
    assert is_strategy_allowed(result, "IRON_CONDOR") is True
    assert is_strategy_allowed(result, "BULL_CALL_SPREAD") is False
    assert is_strategy_allowed(result, "BEAR_PUT_SPREAD") is False

def test_strategy_allowed_uncertain_reduces():
    result = RegimeResult(regime="UNCERTAIN", confidence=0.5, detail="")
    assert is_strategy_allowed(result, "EQUITY_LONG") is True  # allowed but at 50% size
    assert is_strategy_allowed(result, "IRON_CONDOR") is False
```

**Step 2: Run to verify failure**

Run: `python -m pytest tests/test_regime.py -v`
Expected: FAIL (regime module doesn't exist)

**Step 3: Create regime.py**

```python
"""Market regime detection for the trading bot.

Classifies market into: TRENDING_UP, TRENDING_DOWN, SIDEWAYS, VOLATILE, UNCERTAIN.
Each regime gates which strategies are allowed.
"""

from dataclasses import dataclass
import config


@dataclass
class RegimeResult:
    regime: str          # TRENDING_UP, TRENDING_DOWN, SIDEWAYS, VOLATILE, UNCERTAIN
    confidence: float    # 0.0 - 1.0
    detail: str          # human-readable explanation


# Regime -> allowed strategies
STRATEGY_MAP = {
    "TRENDING_UP": {"EQUITY_LONG", "BULL_CALL_SPREAD", "MOMENTUM_CALL"},
    "TRENDING_DOWN": {"BEAR_PUT_SPREAD", "MOMENTUM_PUT"},
    "SIDEWAYS": {"EQUITY_LONG", "IRON_CONDOR"},
    "VOLATILE": {"MOMENTUM_CALL", "MOMENTUM_PUT"},
    "UNCERTAIN": {"EQUITY_LONG"},  # at reduced size
}


def get_vix_tier(vix: float) -> str:
    for threshold, tier_name in config.VIX_TIERS:
        if vix < threshold:
            return tier_name
    return "crisis"


def check_vix_spike(current_vix: float, previous_vix: float) -> bool:
    if previous_vix <= 0:
        return False
    change_pct = (current_vix - previous_vix) / previous_vix
    return change_pct >= 0.20


def classify_regime(
    adx_values: list[float],
    price_vs_ema: str,       # "above", "below", "near"
    price_trend: str,        # "rising", "falling", "flat"
    vix: float,
    nifty_range_7d: float,   # high-low range over 7 days in points
    bb_width_pctl: float,    # Bollinger Band width percentile (0-100)
) -> RegimeResult:
    scores = {"TRENDING_UP": 0, "TRENDING_DOWN": 0, "SIDEWAYS": 0, "VOLATILE": 0}

    # VIX override: high VIX = volatile regardless
    if vix > 20:
        scores["VOLATILE"] += 3
    elif vix > 18:
        scores["VOLATILE"] += 1

    # ADX signals
    recent_adx = adx_values[-3:] if len(adx_values) >= 3 else adx_values
    adx_above_20 = sum(1 for a in recent_adx if a > 20)
    adx_below_20 = sum(1 for a in recent_adx if a < 20)

    if adx_above_20 >= 3:
        if price_vs_ema == "above" and price_trend == "rising":
            scores["TRENDING_UP"] += 3
        elif price_vs_ema == "below" and price_trend == "falling":
            scores["TRENDING_DOWN"] += 3
        else:
            scores["TRENDING_UP" if price_vs_ema == "above" else "TRENDING_DOWN"] += 1
    elif adx_below_20 >= 3:
        scores["SIDEWAYS"] += 2

    # Range check for sideways
    if nifty_range_7d < config.CONDOR_RANGE_POINTS:
        scores["SIDEWAYS"] += 1
    else:
        scores["SIDEWAYS"] -= 1

    # Bollinger Band width
    if bb_width_pctl < 25:
        scores["SIDEWAYS"] += 1
    elif bb_width_pctl > 75:
        scores["VOLATILE"] += 1

    # Price trend
    if price_trend == "rising":
        scores["TRENDING_UP"] += 1
    elif price_trend == "falling":
        scores["TRENDING_DOWN"] += 1
    elif price_trend == "flat":
        scores["SIDEWAYS"] += 1

    # Find winner
    max_score = max(scores.values())
    total_score = sum(max(0, s) for s in scores.values()) or 1
    winner = max(scores, key=scores.get)
    confidence = max_score / total_score if total_score > 0 else 0

    if confidence < 0.7 or max_score < 2:
        regime = "UNCERTAIN"
        confidence = min(confidence, 0.5)
    else:
        regime = winner

    detail = f"{regime} (conf={confidence:.0%}): ADX={recent_adx[-1]:.0f}, VIX={vix:.1f}, range={nifty_range_7d:.0f}pts, BB_pctl={bb_width_pctl:.0f}"

    return RegimeResult(regime=regime, confidence=confidence, detail=detail)


def is_strategy_allowed(regime_result: RegimeResult, strategy: str) -> bool:
    return strategy in STRATEGY_MAP.get(regime_result.regime, set())
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_regime.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add regime.py tests/test_regime.py
git commit -m "feat: add regime detection module"
```

---

### Task 4: Create risk_manager.py — Drawdown limits and portfolio guards

**Files:**
- Create: `risk_manager.py`
- Test: `tests/test_risk_manager.py`

**Step 1: Write failing tests**

Create `tests/test_risk_manager.py`:
```python
import pytest
from risk_manager import RiskManager

def make_portfolio(capital=100000, available=50000, positions=None, closed_trades=None):
    return {
        "capital": capital,
        "available_capital": available,
        "positions": positions or [],
        "closed_trades": closed_trades or [],
        "stats": {"total_pnl": 0},
    }

def test_max_risk_per_trade():
    rm = RiskManager(capital=100000)
    assert rm.max_risk_amount() == 2000  # 2% of 100k

def test_daily_drawdown_halt():
    rm = RiskManager(capital=100000)
    rm.record_daily_pnl(-3500)  # -3.5% > 3% limit
    assert rm.should_halt_daily() is True

def test_daily_drawdown_ok():
    rm = RiskManager(capital=100000)
    rm.record_daily_pnl(-2000)  # -2% < 3% limit
    assert rm.should_halt_daily() is False

def test_weekly_drawdown_reduce():
    rm = RiskManager(capital=100000)
    rm.record_weekly_pnl(-5500)  # -5.5% > 5% limit
    assert rm.should_reduce_weekly() is True
    assert rm.weekly_size_multiplier() == 0.5

def test_total_drawdown_full_stop():
    rm = RiskManager(capital=100000)
    rm.set_current_capital(84000)  # 16% drawdown > 15% limit
    assert rm.should_full_stop() is True

def test_total_drawdown_ok():
    rm = RiskManager(capital=100000)
    rm.set_current_capital(90000)  # 10% drawdown < 15%
    assert rm.should_full_stop() is False

def test_correlation_guard_ok():
    rm = RiskManager(capital=100000)
    positions = [
        {"allocated": 5000},  # max loss 5k if all go wrong
        {"allocated": 4000},
    ]
    assert rm.passes_correlation_guard(positions, new_risk=2000) is True  # 11k < 10k? No, 11% > 10%

def test_correlation_guard_blocks():
    rm = RiskManager(capital=100000)
    positions = [
        {"allocated": 5000},
        {"allocated": 4000},
    ]
    # Adding 3000 risk = total 12k = 12% > 10% limit
    assert rm.passes_correlation_guard(positions, new_risk=3000) is False

def test_consecutive_loss_pause():
    rm = RiskManager(capital=100000)
    closed = [
        {"exit_reason": "stoploss", "instrument": "EQ", "exit_date": "2026-03-05"},
        {"exit_reason": "trailing_stop", "instrument": "EQ", "exit_date": "2026-03-05"},
        {"exit_reason": "stoploss", "instrument": "EQ", "exit_date": "2026-03-06"},
    ]
    assert rm.should_pause_strategy("EQ", closed) is True

def test_consecutive_loss_no_pause_after_win():
    rm = RiskManager(capital=100000)
    closed = [
        {"exit_reason": "stoploss", "instrument": "EQ", "exit_date": "2026-03-04"},
        {"exit_reason": "target", "instrument": "EQ", "exit_date": "2026-03-05"},
        {"exit_reason": "stoploss", "instrument": "EQ", "exit_date": "2026-03-06"},
    ]
    assert rm.should_pause_strategy("EQ", closed) is False

def test_vix_size_multiplier():
    rm = RiskManager(capital=100000)
    assert rm.vix_size_multiplier(10) == 0.5   # extreme complacency
    assert rm.vix_size_multiplier(15) == 1.0   # normal
    assert rm.vix_size_multiplier(20) == 0.75  # elevated
    assert rm.vix_size_multiplier(25) == 0.5   # high
    assert rm.vix_size_multiplier(35) == 0.0   # crisis
```

**Step 2: Run to verify failure**

Run: `python -m pytest tests/test_risk_manager.py -v`
Expected: FAIL

**Step 3: Create risk_manager.py**

```python
"""Risk management module.

Enforces drawdown limits, position sizing, correlation guards,
and strategy pause rules.
"""

import config
from regime import get_vix_tier


class RiskManager:
    def __init__(self, capital: float):
        self.initial_capital = capital
        self.current_capital = capital
        self._daily_pnl = 0.0
        self._weekly_pnl = 0.0

    def max_risk_amount(self) -> float:
        return self.current_capital * config.MAX_RISK_PER_TRADE

    def set_current_capital(self, capital: float):
        self.current_capital = capital

    # --- Daily drawdown ---
    def record_daily_pnl(self, pnl: float):
        self._daily_pnl = pnl

    def should_halt_daily(self) -> bool:
        return self._daily_pnl < -(self.current_capital * config.DAILY_DRAWDOWN_LIMIT)

    # --- Weekly drawdown ---
    def record_weekly_pnl(self, pnl: float):
        self._weekly_pnl = pnl

    def should_reduce_weekly(self) -> bool:
        return self._weekly_pnl < -(self.current_capital * config.WEEKLY_DRAWDOWN_LIMIT)

    def weekly_size_multiplier(self) -> float:
        return 0.5 if self.should_reduce_weekly() else 1.0

    # --- Total drawdown ---
    def should_full_stop(self) -> bool:
        drawdown = (self.initial_capital - self.current_capital) / self.initial_capital
        return drawdown >= config.TOTAL_DRAWDOWN_HALT

    # --- Correlation guard ---
    def passes_correlation_guard(self, positions: list, new_risk: float) -> bool:
        total_risk = sum(p.get("allocated", 0) for p in positions) + new_risk
        max_allowed = self.current_capital * config.MAX_PORTFOLIO_LOSS_CORRELATION
        return total_risk <= max_allowed

    # --- Consecutive loss pause ---
    def should_pause_strategy(self, instrument: str, closed_trades: list) -> bool:
        loss_reasons = {"stoploss", "trailing_stop"}
        relevant = [t for t in closed_trades if t.get("instrument", "EQ") == instrument]
        relevant.sort(key=lambda t: t.get("exit_date", ""))
        recent = relevant[-config.CONSECUTIVE_LOSS_PAUSE:] if relevant else []
        if len(recent) < config.CONSECUTIVE_LOSS_PAUSE:
            return False
        return all(t.get("exit_reason") in loss_reasons for t in recent)

    # --- VIX-based size adjustment ---
    def vix_size_multiplier(self, vix: float) -> float:
        tier = get_vix_tier(vix)
        multipliers = {
            "extreme_complacency": 0.5,
            "normal": 1.0,
            "elevated": 0.75,
            "high": 0.5,
            "crisis": 0.0,
        }
        return multipliers.get(tier, 1.0)
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_risk_manager.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add risk_manager.py tests/test_risk_manager.py
git commit -m "feat: add risk manager with drawdown limits and guards"
```

---

### Task 5: Add regime data fetching to regime.py

**Files:**
- Modify: `regime.py`
- Test: `tests/test_regime.py`

This task adds `fetch_regime_data(smart_api) -> RegimeResult` which fetches live Nifty data and calls `classify_regime()`. This replaces the existing `classify_regime()` + `fetch_market_regime()` in paper_trade.py.

**Step 1: Write test**

Add to `tests/test_regime.py`:
```python
def test_compute_adx_from_candles():
    from regime import compute_adx_series
    # Need 14+ candles for ADX(10)
    candles = [
        ["2026-01-01", 100, 105, 98, 103, 1000],
        ["2026-01-02", 103, 108, 101, 107, 1200],
        ["2026-01-03", 107, 110, 104, 106, 1100],
        ["2026-01-04", 106, 109, 103, 108, 1300],
        ["2026-01-05", 108, 112, 106, 111, 1400],
        ["2026-01-06", 111, 115, 109, 114, 1500],
        ["2026-01-07", 114, 117, 112, 116, 1300],
        ["2026-01-08", 116, 119, 113, 115, 1100],
        ["2026-01-09", 115, 118, 112, 117, 1200],
        ["2026-01-10", 117, 121, 115, 120, 1600],
        ["2026-01-11", 120, 123, 118, 122, 1400],
        ["2026-01-12", 122, 125, 119, 121, 1300],
        ["2026-01-13", 121, 124, 118, 123, 1500],
        ["2026-01-14", 123, 127, 121, 126, 1700],
        ["2026-01-15", 126, 129, 124, 128, 1600],
        ["2026-01-16", 128, 131, 125, 127, 1400],
        ["2026-01-17", 127, 130, 124, 129, 1500],
        ["2026-01-18", 129, 133, 127, 132, 1800],
        ["2026-01-19", 132, 135, 130, 134, 1700],
        ["2026-01-20", 134, 137, 131, 136, 1900],
    ]
    result = compute_adx_series(candles, period=10)
    assert len(result) >= 4
    assert all(isinstance(v, float) for v in result)

def test_compute_price_vs_ema():
    from regime import compute_price_vs_ema
    candles_up = [[f"2026-01-{i:02d}", i*10, i*10+5, i*10-2, i*10+3, 1000] for i in range(1, 25)]
    result = compute_price_vs_ema(candles_up, period=20)
    assert result in ("above", "below", "near")

def test_compute_nifty_range():
    from regime import compute_nifty_range
    candles = [
        ["2026-01-01", 22000, 22200, 21900, 22100, 1000],
        ["2026-01-02", 22100, 22300, 22000, 22250, 1200],
        ["2026-01-03", 22250, 22350, 22100, 22200, 1100],
    ]
    result = compute_nifty_range(candles, days=3)
    assert result == 22350 - 21900  # 450

def test_compute_bb_width_percentile():
    from regime import compute_bb_width_percentile
    # Needs 20+ candles
    candles = [[f"2026-01-{i:02d}", 100+i, 102+i, 98+i, 101+i, 1000] for i in range(1, 30)]
    result = compute_bb_width_percentile(candles, period=20)
    assert 0 <= result <= 100
```

**Step 2: Run to verify failure**

Run: `python -m pytest tests/test_regime.py -v -k "compute"`
Expected: FAIL

**Step 3: Add helper functions to regime.py**

Add these functions to `regime.py`:

```python
def compute_adx_series(candles: list, period: int = 10) -> list[float]:
    """Compute ADX values from candles. Returns last 4+ values."""
    if len(candles) < period * 2:
        return []

    # True Range, +DM, -DM
    tr_list, plus_dm_list, minus_dm_list = [], [], []
    for i in range(1, len(candles)):
        high, low = candles[i][2], candles[i][3]
        prev_high, prev_low = candles[i-1][2], candles[i-1][3]
        prev_close = candles[i-1][4]

        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        plus_dm = max(high - prev_high, 0) if (high - prev_high) > (prev_low - low) else 0
        minus_dm = max(prev_low - low, 0) if (prev_low - low) > (high - prev_high) else 0

        tr_list.append(tr)
        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)

    if len(tr_list) < period:
        return []

    # Smoothed values (Wilder's)
    atr = sum(tr_list[:period]) / period
    plus_di_smooth = sum(plus_dm_list[:period]) / period
    minus_di_smooth = sum(minus_dm_list[:period]) / period

    dx_values = []
    for i in range(period, len(tr_list)):
        atr = (atr * (period - 1) + tr_list[i]) / period
        plus_di_smooth = (plus_di_smooth * (period - 1) + plus_dm_list[i]) / period
        minus_di_smooth = (minus_di_smooth * (period - 1) + minus_dm_list[i]) / period

        if atr == 0:
            continue
        plus_di = 100 * plus_di_smooth / atr
        minus_di = 100 * minus_di_smooth / atr

        di_sum = plus_di + minus_di
        if di_sum == 0:
            dx_values.append(0.0)
        else:
            dx_values.append(100 * abs(plus_di - minus_di) / di_sum)

    if len(dx_values) < period:
        return dx_values[-4:] if dx_values else []

    # ADX = smoothed DX
    adx = sum(dx_values[:period]) / period
    adx_series = [adx]
    for i in range(period, len(dx_values)):
        adx = (adx * (period - 1) + dx_values[i]) / period
        adx_series.append(adx)

    return adx_series[-4:]


def compute_price_vs_ema(candles: list, period: int = 20) -> str:
    """Returns 'above', 'below', or 'near' relative to EMA."""
    if len(candles) < period:
        return "near"
    closes = [c[4] for c in candles]
    multiplier = 2 / (period + 1)
    ema = sum(closes[:period]) / period
    for close in closes[period:]:
        ema = (close - ema) * multiplier + ema

    current = closes[-1]
    pct_diff = (current - ema) / ema * 100
    if pct_diff > 0.5:
        return "above"
    elif pct_diff < -0.5:
        return "below"
    return "near"


def compute_price_trend(candles: list, days: int = 5) -> str:
    """Returns 'rising', 'falling', or 'flat' based on last N days."""
    if len(candles) < days:
        return "flat"
    recent_closes = [c[4] for c in candles[-days:]]
    up_days = sum(1 for i in range(1, len(recent_closes)) if recent_closes[i] > recent_closes[i-1])
    down_days = sum(1 for i in range(1, len(recent_closes)) if recent_closes[i] < recent_closes[i-1])
    if up_days >= days - 1:
        return "rising"
    elif down_days >= days - 1:
        return "falling"
    return "flat"


def compute_nifty_range(candles: list, days: int = 7) -> float:
    """High-low range over last N days in points."""
    recent = candles[-days:] if len(candles) >= days else candles
    highs = [c[2] for c in recent]
    lows = [c[3] for c in recent]
    return max(highs) - min(lows)


def compute_bb_width_percentile(candles: list, period: int = 20) -> float:
    """Bollinger Band width percentile (0-100) over lookback."""
    if len(candles) < period + 5:
        return 50.0
    closes = [c[4] for c in candles]

    widths = []
    for i in range(period, len(closes)):
        window = closes[i-period:i]
        sma = sum(window) / period
        std = (sum((x - sma)**2 for x in window) / period) ** 0.5
        if sma > 0:
            widths.append(2 * std / sma * 100)

    if not widths:
        return 50.0
    current_width = widths[-1]
    rank = sum(1 for w in widths if w <= current_width) / len(widths) * 100
    return rank
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_regime.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add regime.py tests/test_regime.py
git commit -m "feat: add regime data computation helpers (ADX, EMA, BB, range)"
```

---

## Phase 2: Vertical Spreads (Bull Call + Bear Put)

### Task 6: Add spread strike selection to agent_with_options.py

**Files:**
- Modify: `agent_with_options.py`
- Test: `tests/test_agent_with_options.py`

**Step 1: Write failing tests**

Add to `tests/test_agent_with_options.py`:
```python
def test_select_spread_strikes_bull_call():
    from agent_with_options import select_spread_strikes
    # Mock option chain data (list of dicts with strikePrice, CE, PE)
    chain = [
        {"strikePrice": 1280, "CE": {"openInterest": 2000, "impliedVolatility": 25, "delta": 0.7, "lastPrice": 55}, "PE": {}},
        {"strikePrice": 1300, "CE": {"openInterest": 5000, "impliedVolatility": 22, "delta": 0.5, "lastPrice": 40}, "PE": {}},
        {"strikePrice": 1320, "CE": {"openInterest": 3000, "impliedVolatility": 20, "delta": 0.35, "lastPrice": 28}, "PE": {}},
        {"strikePrice": 1340, "CE": {"openInterest": 1500, "impliedVolatility": 19, "delta": 0.22, "lastPrice": 18}, "PE": {}},
        {"strikePrice": 1360, "CE": {"openInterest": 800, "impliedVolatility": 18, "delta": 0.12, "lastPrice": 10}, "PE": {}},
    ]
    result = select_spread_strikes(chain, spot=1305, direction="bullish", atr=30, budget=5000, lot_size=250)
    assert result is not None
    assert result["long_strike"] <= result["short_strike"]  # bull call: buy lower, sell higher
    assert result["net_debit"] > 0
    assert result["max_profit"] > 0
    assert result["rr_ratio"] >= 1.5  # R:R check

def test_select_spread_strikes_bear_put():
    from agent_with_options import select_spread_strikes
    chain = [
        {"strikePrice": 1260, "PE": {"openInterest": 800, "impliedVolatility": 20, "delta": -0.15, "lastPrice": 12}, "CE": {}},
        {"strikePrice": 1280, "PE": {"openInterest": 2000, "impliedVolatility": 22, "delta": -0.25, "lastPrice": 20}, "CE": {}},
        {"strikePrice": 1300, "PE": {"openInterest": 5000, "impliedVolatility": 24, "delta": -0.5, "lastPrice": 38}, "CE": {}},
        {"strikePrice": 1320, "PE": {"openInterest": 3000, "impliedVolatility": 26, "delta": -0.65, "lastPrice": 52}, "CE": {}},
    ]
    result = select_spread_strikes(chain, spot=1305, direction="bearish", atr=30, budget=5000, lot_size=250)
    assert result is not None
    assert result["long_strike"] >= result["short_strike"]  # bear put: buy higher, sell lower

def test_select_spread_strikes_no_liquidity():
    from agent_with_options import select_spread_strikes
    chain = [
        {"strikePrice": 1300, "CE": {"openInterest": 100, "impliedVolatility": 22, "delta": 0.5, "lastPrice": 40}, "PE": {}},
        {"strikePrice": 1320, "CE": {"openInterest": 50, "impliedVolatility": 20, "delta": 0.35, "lastPrice": 28}, "PE": {}},
    ]
    result = select_spread_strikes(chain, spot=1305, direction="bullish", atr=30, budget=5000, lot_size=250)
    assert result is None  # OI too low

def test_select_spread_strikes_budget_exceeded():
    from agent_with_options import select_spread_strikes
    chain = [
        {"strikePrice": 1300, "CE": {"openInterest": 5000, "impliedVolatility": 22, "delta": 0.5, "lastPrice": 400}, "PE": {}},
        {"strikePrice": 1320, "CE": {"openInterest": 3000, "impliedVolatility": 20, "delta": 0.35, "lastPrice": 380}, "PE": {}},
    ]
    result = select_spread_strikes(chain, spot=1305, direction="bullish", atr=30, budget=2000, lot_size=250)
    assert result is None  # debit * lot_size > budget
```

**Step 2: Run to verify failure**

Run: `python -m pytest tests/test_agent_with_options.py -v -k "spread"`
Expected: FAIL

**Step 3: Implement select_spread_strikes in agent_with_options.py**

Add to `agent_with_options.py`:
```python
import config

def select_spread_strikes(
    option_chain: list,
    spot: float,
    direction: str,        # "bullish" or "bearish"
    atr: float,
    budget: float,
    lot_size: int,
) -> dict | None:
    """Select optimal strikes for a vertical spread.

    Bull call spread: buy ATM call, sell OTM call.
    Bear put spread: buy ATM put, sell OTM put.

    Returns dict with long_strike, short_strike, net_debit, max_profit, rr_ratio,
    or None if no valid spread found.
    """
    option_type = "CE" if direction == "bullish" else "PE"

    # Sort strikes
    strikes = sorted(option_chain, key=lambda x: x["strikePrice"])

    # Find ATM strike (closest to spot)
    atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i]["strikePrice"] - spot))

    if direction == "bullish":
        # Long leg: ATM or 1-strike ITM (lower strike for calls)
        long_idx = max(0, atm_idx - 1) if atm_idx > 0 else atm_idx
        # Try short legs: 2-4 strikes above long
        candidates = range(long_idx + 2, min(long_idx + 5, len(strikes)))
    else:
        # Long leg: ATM or 1-strike ITM (higher strike for puts)
        long_idx = min(len(strikes) - 1, atm_idx + 1) if atm_idx < len(strikes) - 1 else atm_idx
        # Try short legs: 2-4 strikes below long
        candidates = range(max(0, long_idx - 4), long_idx - 1)

    best_spread = None

    for short_idx in candidates:
        long_data = strikes[long_idx].get(option_type, {})
        short_data = strikes[short_idx].get(option_type, {})

        if not long_data or not short_data:
            continue

        # Liquidity check
        long_oi = long_data.get("openInterest", 0) or 0
        short_oi = short_data.get("openInterest", 0) or 0
        if long_oi < config.MIN_OI_THRESHOLD or short_oi < config.MIN_OI_THRESHOLD:
            continue

        long_premium = long_data.get("lastPrice", 0) or 0
        short_premium = short_data.get("lastPrice", 0) or 0

        if long_premium <= 0 or short_premium <= 0:
            continue

        long_strike = strikes[long_idx]["strikePrice"]
        short_strike = strikes[short_idx]["strikePrice"]
        spread_width = abs(long_strike - short_strike)
        net_debit = long_premium - short_premium

        if net_debit <= 0:
            continue

        max_profit_per_unit = spread_width - net_debit
        if max_profit_per_unit <= 0:
            continue

        total_debit = net_debit * lot_size
        total_max_profit = max_profit_per_unit * lot_size
        rr_ratio = max_profit_per_unit / net_debit

        # Budget check
        if total_debit > budget:
            continue

        # R:R check (at least 1.5:1)
        if rr_ratio < 1.5:
            continue

        # Spread width should be meaningful relative to ATR
        if spread_width < atr:
            continue

        if best_spread is None or rr_ratio > best_spread["rr_ratio"]:
            best_spread = {
                "long_strike": long_strike,
                "short_strike": short_strike,
                "long_premium": long_premium,
                "short_premium": short_premium,
                "spread_width": spread_width,
                "net_debit": net_debit,
                "total_debit": total_debit,
                "max_profit": total_max_profit,
                "max_profit_per_unit": max_profit_per_unit,
                "rr_ratio": rr_ratio,
                "long_oi": long_oi,
                "short_oi": short_oi,
                "option_type": option_type,
            }

    return best_spread
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_agent_with_options.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add agent_with_options.py tests/test_agent_with_options.py
git commit -m "feat: add spread strike selection with liquidity and R:R checks"
```

---

### Task 7: Add _open_spread_position() to paper_trade.py

**Files:**
- Modify: `paper_trade.py`
- Test: `tests/test_paper_trade.py`

This is the core spread opening logic. It handles both bull call and bear put spreads.

**Step 1: Write failing tests**

Add to `tests/test_paper_trade.py`:
```python
def test_spread_position_structure():
    """Verify spread position has all required fields."""
    pos = {
        "symbol": "RELIANCE",
        "instrument": "SPREAD",
        "spread_type": "BULL_CALL",
        "direction": "bullish",
        "long_leg": {"strike": 1300, "option_type": "CE", "entry_premium": 40.0},
        "short_leg": {"strike": 1340, "option_type": "CE", "entry_premium": 18.0},
        "spread_width": 40,
        "net_debit": 22.0,
        "lot_size": 250,
        "num_lots": 1,
        "quantity": 250,
        "allocated": 5500.0,
        "max_profit": 4500.0,
        "status": "open",
    }
    assert pos["instrument"] == "SPREAD"
    assert pos["allocated"] == pos["net_debit"] * pos["quantity"]
    assert pos["max_profit"] == (pos["spread_width"] - pos["net_debit"]) * pos["quantity"]

def test_spread_pnl_at_max_profit():
    from paper_trade import calc_spread_pnl
    # Bull call spread: bought 1300 CE at 40, sold 1340 CE at 18
    # Stock at expiry: 1350 (above both strikes)
    # Long leg worth: 50 (1350-1300), Short leg worth: 10 (1350-1340)
    # P&L per unit: (50-40) - (10-18) = 10 + 8 = 18
    # Alternatively: spread_width - net_debit = 40 - 22 = 18
    pnl = calc_spread_pnl(
        long_entry=40, long_exit=50,
        short_entry=18, short_exit=10,
        quantity=250
    )
    assert pnl == (50 - 40 - 10 + 18) * 250  # 4500

def test_spread_pnl_at_max_loss():
    from paper_trade import calc_spread_pnl
    # Stock below both strikes, both expire worthless
    pnl = calc_spread_pnl(
        long_entry=40, long_exit=2,
        short_entry=18, short_exit=1,
        quantity=250
    )
    assert pnl == (2 - 40 - 1 + 18) * 250  # -5250
```

**Step 2: Run to verify failure**

Run: `python -m pytest tests/test_paper_trade.py -v -k "spread"`
Expected: FAIL

**Step 3: Implement**

Add to `paper_trade.py`:

```python
def calc_spread_pnl(long_entry, long_exit, short_entry, short_exit, quantity):
    """Calculate P&L for a vertical spread.

    P&L = (long_exit - long_entry) - (short_exit - short_entry) * quantity
    For the short leg, we sold at entry and buy back at exit.
    """
    long_pnl = (long_exit - long_entry) * quantity
    short_pnl = (short_entry - short_exit) * quantity  # sold high, buy back low = profit
    return long_pnl + short_pnl
```

Then add `_open_spread_position()`:

```python
def _open_spread_position(
    smart_api, symbol, eq_token, spot_price, allocation, direction, atr, candles, alloc
) -> dict | None:
    """Open a vertical spread (bull call or bear put).

    Returns position dict or None if no valid spread found.
    """
    from agent_with_options import get_nearest_expiry, fetch_option_chain, select_spread_strikes

    expiry = get_nearest_expiry(min_dte=config.EXPIRY_SAFETY_DAYS + config.SPREAD_TIME_EXIT_DAYS)
    if not expiry:
        logger.info("SKIP %s: no expiry with sufficient DTE", symbol)
        return None

    dte = days_to_expiry(expiry)
    logger.info("  %s: using expiry %s (%dd DTE) for %s spread", symbol, expiry, dte, direction)

    # Fetch option chain
    option_chain = fetch_option_chain(smart_api, symbol, eq_token)
    time.sleep(config.API_DELAY)
    if not option_chain:
        logger.info("  %s: option chain fetch failed, retrying...", symbol)
        smart_api = refresh_session()
        time.sleep(config.API_DELAY)
        option_chain = fetch_option_chain(smart_api, symbol, eq_token)
        time.sleep(config.API_DELAY)
    if not option_chain:
        logger.info("SKIP %s: option chain unavailable", symbol)
        return None

    # Get lot size from any contract
    lot_size = None
    for row in option_chain:
        opt_type = "CE" if direction == "bullish" else "PE"
        opt_data = row.get(opt_type, {})
        if opt_data and opt_data.get("lotSize"):
            lot_size = int(opt_data["lotSize"])
            break
    if not lot_size:
        logger.info("SKIP %s: could not determine lot size", symbol)
        return None

    # Select strikes
    spread = select_spread_strikes(
        option_chain, spot=spot_price, direction=direction,
        atr=atr, budget=allocation, lot_size=lot_size,
    )
    if not spread:
        logger.info("SKIP %s: no valid spread found (liquidity/budget/R:R)", symbol)
        return None

    spread_type = "BULL_CALL" if direction == "bullish" else "BEAR_PUT"
    opt_type = spread["option_type"]

    logger.info(
        "  %s %s: long %s@%.1f, short %s@%.1f, debit=%.1f, max_profit=%.0f, R:R=%.1f",
        symbol, spread_type, spread["long_strike"], spread["long_premium"],
        spread["short_strike"], spread["short_premium"],
        spread["net_debit"], spread["max_profit"], spread["rr_ratio"],
    )

    position = {
        "symbol": symbol,
        "token": eq_token,
        "direction": direction,
        "instrument": "SPREAD",
        "spread_type": spread_type,
        "long_leg": {
            "strike": spread["long_strike"],
            "option_type": opt_type,
            "entry_premium": apply_slippage(spread["long_premium"], "OPT", "buy"),
            "token": None,  # resolved at execution time for live trading
        },
        "short_leg": {
            "strike": spread["short_strike"],
            "option_type": opt_type,
            "entry_premium": apply_slippage(spread["short_premium"], "OPT", "sell"),
            "token": None,
        },
        "spread_width": spread["spread_width"],
        "net_debit": spread["net_debit"],
        "lot_size": lot_size,
        "num_lots": 1,
        "quantity": lot_size,
        "allocated": spread["total_debit"],
        "max_profit": spread["max_profit"],
        "rr_ratio": spread["rr_ratio"],
        "underlying_at_entry": spot_price,
        "atr_at_entry": atr,
        "expiry": expiry,
        "entry_date": _today_ist(),
        "max_hold_date": _add_trading_days(_today_ist(), config.SPREAD_TIME_EXIT_DAYS),
        "score": alloc.get("score", 0),
        "categories": alloc.get("categories", []),
        "market_regime": alloc.get("market_regime", "normal"),
        "status": "open",
    }

    return position
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_paper_trade.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add paper_trade.py tests/test_paper_trade.py
git commit -m "feat: add spread position opening with strike selection"
```

---

### Task 8: Add spread exit logic to monitor_positions

**Files:**
- Modify: `paper_trade.py` (monitor_positions function, ~line 1360)
- Test: `tests/test_paper_trade.py`

**Step 1: Write failing tests**

```python
def test_spread_exit_target_hit():
    from paper_trade import check_spread_exit
    pos = {
        "instrument": "SPREAD", "spread_type": "BEAR_PUT", "direction": "bearish",
        "underlying_at_entry": 1300, "atr_at_entry": 30,
        "max_profit": 4500, "allocated": 5500, "net_debit": 22,
        "spread_width": 40, "quantity": 250,
        "entry_date": "2026-03-01", "max_hold_date": "2026-03-08",
        "expiry": "26MAR2026",
    }
    # Underlying dropped 2x ATR (target for bearish)
    reason = check_spread_exit(pos, underlying_ltp=1240, today="2026-03-04",
                                long_premium=35, short_premium=5)
    assert reason == "target"

def test_spread_exit_stoploss():
    from paper_trade import check_spread_exit
    pos = {
        "instrument": "SPREAD", "spread_type": "BEAR_PUT", "direction": "bearish",
        "underlying_at_entry": 1300, "atr_at_entry": 30,
        "max_profit": 4500, "allocated": 5500, "net_debit": 22,
        "spread_width": 40, "quantity": 250,
        "entry_date": "2026-03-01", "max_hold_date": "2026-03-08",
        "expiry": "26MAR2026",
    }
    # Underlying rose 1.5x ATR (stoploss for bearish)
    reason = check_spread_exit(pos, underlying_ltp=1350, today="2026-03-04",
                                long_premium=8, short_premium=3)
    assert reason == "stoploss"

def test_spread_exit_profit_cap():
    from paper_trade import check_spread_exit
    pos = {
        "instrument": "SPREAD", "spread_type": "BULL_CALL", "direction": "bullish",
        "underlying_at_entry": 1300, "atr_at_entry": 30,
        "max_profit": 4500, "allocated": 5500, "net_debit": 22,
        "spread_width": 40, "quantity": 250,
        "long_leg": {"entry_premium": 40},
        "short_leg": {"entry_premium": 18},
        "entry_date": "2026-03-01", "max_hold_date": "2026-03-08",
        "expiry": "26MAR2026",
    }
    # Current spread value near max profit (long=39, short=1 -> spread=38 vs max 40)
    reason = check_spread_exit(pos, underlying_ltp=1345, today="2026-03-04",
                                long_premium=39, short_premium=1)
    assert reason == "profit_cap"

def test_spread_exit_time():
    from paper_trade import check_spread_exit
    pos = {
        "instrument": "SPREAD", "spread_type": "BULL_CALL", "direction": "bullish",
        "underlying_at_entry": 1300, "atr_at_entry": 30,
        "max_profit": 4500, "allocated": 5500, "net_debit": 22,
        "spread_width": 40, "quantity": 250,
        "long_leg": {"entry_premium": 40},
        "short_leg": {"entry_premium": 18},
        "entry_date": "2026-03-01", "max_hold_date": "2026-03-06",
        "expiry": "26MAR2026",
    }
    # Past max hold date
    reason = check_spread_exit(pos, underlying_ltp=1310, today="2026-03-07",
                                long_premium=30, short_premium=15)
    assert reason == "expiry"
```

**Step 2: Run to verify failure**

Run: `python -m pytest tests/test_paper_trade.py -v -k "spread_exit"`
Expected: FAIL

**Step 3: Implement check_spread_exit**

```python
def check_spread_exit(pos, underlying_ltp, today, long_premium, short_premium):
    """Determine if a spread position should be exited.

    Returns exit reason string or None to hold.
    """
    entry = pos["underlying_at_entry"]
    atr = pos["atr_at_entry"]
    direction = pos["direction"]

    # Time exit
    if today >= pos["max_hold_date"]:
        return "expiry"

    # DTE check (E-3 safety for stock options)
    dte = days_to_expiry(pos["expiry"])
    if dte <= config.EXPIRY_SAFETY_DAYS:
        return "expiry"

    # Target hit (underlying moved favorably by 2x ATR)
    if direction == "bullish" and underlying_ltp >= entry + 2.0 * atr:
        return "target"
    if direction == "bearish" and underlying_ltp <= entry - 2.0 * atr:
        return "target"

    # Stoploss hit (underlying moved adversely by 1.5x ATR)
    if direction == "bullish" and underlying_ltp <= entry - 1.5 * atr:
        return "stoploss"
    if direction == "bearish" and underlying_ltp >= entry + 1.5 * atr:
        return "stoploss"

    # Profit cap: spread reached 80% of max profit
    current_spread_value = long_premium - short_premium
    entry_debit = pos["net_debit"]
    current_profit_per_unit = current_spread_value - entry_debit
    max_profit_per_unit = pos["spread_width"] - entry_debit
    if max_profit_per_unit > 0:
        profit_pct = current_profit_per_unit / max_profit_per_unit
        if profit_pct >= config.SPREAD_PROFIT_CAP:
            return "profit_cap"

    return None
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_paper_trade.py -v -k "spread"`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add paper_trade.py tests/test_paper_trade.py
git commit -m "feat: add spread exit logic (target, stoploss, profit cap, time)"
```

---

## Phase 3: YouTube Intel

### Task 9: Create youtube_intel.py

**Files:**
- Create: `youtube_intel.py`
- Test: `tests/test_youtube_intel.py`

**Step 1: Write failing tests**

Create `tests/test_youtube_intel.py`:
```python
import pytest
from unittest.mock import patch, MagicMock
from youtube_intel import (
    search_youtube_videos, extract_transcript,
    classify_market_intel, classify_stock_intel,
    fetch_market_intel, fetch_stock_intel,
)

def test_search_youtube_returns_list():
    with patch("youtube_intel._run_yt_dlp_search") as mock:
        mock.return_value = [
            {"id": "abc123", "title": "Nifty Analysis Today", "upload_date": "20260306"},
        ]
        results = search_youtube_videos("Nifty analysis today", max_results=3)
        assert len(results) == 1
        assert results[0]["id"] == "abc123"

def test_extract_transcript_returns_text():
    with patch("youtube_intel._get_transcript") as mock:
        mock.return_value = "Market is looking bullish today. Nifty support at 22000."
        text = extract_transcript("abc123")
        assert "bullish" in text.lower()

def test_classify_market_intel_structure():
    with patch("youtube_intel._call_claude_haiku") as mock:
        mock.return_value = {
            "market_bias": "bullish",
            "key_levels": {"nifty_support": 22000, "nifty_resistance": 22500},
            "sectors_bullish": ["Banking"],
            "sectors_bearish": ["Pharma"],
            "events_today": [],
            "confidence": "medium",
            "summary": "Market looking positive",
        }
        result = classify_market_intel("Some transcript text")
        assert result["market_bias"] in ("bullish", "bearish", "sideways")
        assert "nifty_support" in result["key_levels"]

def test_classify_stock_intel_structure():
    with patch("youtube_intel._call_claude_haiku") as mock:
        mock.return_value = {
            "sentiment": "bullish",
            "key_levels": {"support": 1250, "resistance": 1350},
            "red_flags": [],
            "catalyst": "sector rotation",
            "confidence": "high",
        }
        result = classify_stock_intel("RELIANCE", "Some transcript about Reliance")
        assert result["sentiment"] in ("bullish", "bearish", "neutral")

def test_cache_prevents_duplicate_calls(tmp_path):
    with patch("youtube_intel.CACHE_DIR", tmp_path):
        with patch("youtube_intel._fetch_market_intel_uncached") as mock:
            mock.return_value = {"market_bias": "bullish", "confidence": "medium"}
            # First call
            r1 = fetch_market_intel()
            # Second call should use cache
            r2 = fetch_market_intel()
            assert mock.call_count == 1
            assert r1 == r2
```

**Step 2: Run to verify failure**

Run: `python -m pytest tests/test_youtube_intel.py -v`
Expected: FAIL

**Step 3: Create youtube_intel.py**

```python
"""YouTube intelligence for market and stock analysis.

Mode A: Daily market intel (pre-market)
Mode B: Pre-trade stock research
"""

import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

import config
from utils import parse_claude_json

CACHE_DIR = Path("data/youtube_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _run_yt_dlp_search(query: str, max_results: int = 3) -> list[dict]:
    """Search YouTube via yt-dlp. Returns list of video metadata."""
    try:
        cmd = [
            "yt-dlp", f"ytsearch{max_results}:{query}",
            "--dump-json", "--no-download", "--flat-playlist",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return []
        videos = []
        for line in result.stdout.strip().split("\n"):
            if line:
                try:
                    videos.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return videos
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def _get_transcript(video_id: str) -> str | None:
    """Get transcript for a YouTube video."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Try English first, then Hindi, then auto-generated
        for lang in ["en", "hi"]:
            try:
                transcript = transcript_list.find_transcript([lang])
                entries = transcript.fetch()
                return " ".join(e["text"] for e in entries)
            except Exception:
                continue
        # Try auto-generated
        try:
            transcript = transcript_list.find_generated_transcript(["en", "hi"])
            entries = transcript.fetch()
            return " ".join(e["text"] for e in entries)
        except Exception:
            pass
    except Exception:
        pass
    return None


def _call_claude_haiku(prompt: str) -> dict | None:
    """Call Claude haiku for classification."""
    client = config.get_anthropic_client()
    try:
        response = client.messages.create(
            model=config.CLAUDE_MODEL_LIGHT,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return parse_claude_json(response.content[0].text)
    except Exception:
        return None


def search_youtube_videos(query: str, max_results: int = 3) -> list[dict]:
    return _run_yt_dlp_search(query, max_results)


def extract_transcript(video_id: str) -> str | None:
    return _get_transcript(video_id)


def classify_market_intel(transcript_text: str) -> dict | None:
    prompt = f"""Analyze this market commentary transcript and extract structured intelligence.
Respond with ONLY valid JSON:
{{
    "market_bias": "bullish" | "bearish" | "sideways",
    "key_levels": {{"nifty_support": <number>, "nifty_resistance": <number>}},
    "sectors_bullish": [<list of sector names>],
    "sectors_bearish": [<list of sector names>],
    "events_today": [<list of events mentioned>],
    "confidence": "high" | "medium" | "low",
    "summary": "<2-3 sentence summary>"
}}

Transcript:
{transcript_text[:3000]}"""
    return _call_claude_haiku(prompt)


def classify_stock_intel(symbol: str, transcript_text: str) -> dict | None:
    prompt = f"""Analyze this transcript about {symbol} stock and extract trading intelligence.
Respond with ONLY valid JSON:
{{
    "sentiment": "bullish" | "bearish" | "neutral",
    "key_levels": {{"support": <number or null>, "resistance": <number or null>}},
    "red_flags": [<list of concerns like "earnings next week", "promoter selling">],
    "catalyst": "<main catalyst or null>",
    "confidence": "high" | "medium" | "low"
}}

Transcript:
{transcript_text[:3000]}"""
    return _call_claude_haiku(prompt)


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _read_cache(key: str, max_age_hours: float) -> dict | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        cached_at = datetime.fromisoformat(data.get("_cached_at", "2000-01-01"))
        if datetime.now() - cached_at > timedelta(hours=max_age_hours):
            return None
        return data.get("result")
    except Exception:
        return None


def _write_cache(key: str, result: dict):
    path = _cache_path(key)
    path.write_text(json.dumps({
        "_cached_at": datetime.now().isoformat(),
        "result": result,
    }, indent=2))


def _fetch_market_intel_uncached() -> dict | None:
    """Fetch and classify market intel from YouTube."""
    queries = ["Nifty analysis today", "Indian market outlook today"]
    all_transcripts = []

    for query in queries:
        videos = search_youtube_videos(query, max_results=2)
        for video in videos:
            vid_id = video.get("id")
            if not vid_id:
                continue
            transcript = extract_transcript(vid_id)
            if transcript:
                all_transcripts.append(transcript[:2000])
            if len(all_transcripts) >= 3:
                break
        if len(all_transcripts) >= 3:
            break

    if not all_transcripts:
        return None

    combined = "\n---\n".join(all_transcripts)
    return classify_market_intel(combined)


def fetch_market_intel() -> dict | None:
    """Mode A: Daily market intel with caching."""
    today = datetime.now().strftime("%Y-%m-%d")
    cache_key = f"market_intel_{today}"

    cached = _read_cache(cache_key, config.YT_MARKET_CACHE_HOURS)
    if cached:
        return cached

    result = _fetch_market_intel_uncached()
    if result:
        _write_cache(cache_key, result)
    return result


def fetch_stock_intel(symbol: str) -> dict | None:
    """Mode B: Pre-trade stock research with caching."""
    today = datetime.now().strftime("%Y-%m-%d")
    cache_key = f"stock_intel_{symbol}_{today}"

    cached = _read_cache(cache_key, config.YT_STOCK_CACHE_HOURS)
    if cached:
        return cached

    videos = search_youtube_videos(f"{symbol} stock analysis", max_results=2)
    for video in videos:
        vid_id = video.get("id")
        if not vid_id:
            continue
        transcript = extract_transcript(vid_id)
        if transcript:
            result = classify_stock_intel(symbol, transcript)
            if result:
                _write_cache(cache_key, result)
                return result

    return None
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_youtube_intel.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add youtube_intel.py tests/test_youtube_intel.py
git commit -m "feat: add YouTube intel module (market + stock research)"
```

---

## Phase 4: Integration — Wire Everything Into paper_trade.py

### Task 10: Integrate regime detection into open_positions flow

**Files:**
- Modify: `paper_trade.py` (open_positions function, ~line 999)

**Step 1:** Replace existing `classify_regime()` and `fetch_market_regime()` calls with new `regime.py` module. The existing functions at lines 861-928 in paper_trade.py are replaced by the new regime module.

In `open_positions()`:
```python
# Replace:
regime_info = fetch_market_regime(smart_api)
regime = regime_info["regime"]

# With:
from regime import classify_regime, compute_adx_series, compute_price_vs_ema, \
    compute_price_trend, compute_nifty_range, compute_bb_width_percentile, \
    is_strategy_allowed, RegimeResult, check_vix_spike
from risk_manager import RiskManager

# Fetch Nifty candles for regime detection
nifty_candles = fetch_daily_candles(smart_api, NIFTY_TOKEN, days=30)
vix_ltp = get_ltp(smart_api, "INDIA VIX", VIX_TOKEN)

if nifty_candles and vix_ltp:
    adx_values = compute_adx_series(nifty_candles, period=10)
    price_vs_ema = compute_price_vs_ema(nifty_candles, period=20)
    price_trend = compute_price_trend(nifty_candles, days=5)
    nifty_range = compute_nifty_range(nifty_candles, days=7)
    bb_pctl = compute_bb_width_percentile(nifty_candles, period=20)

    regime_result = classify_regime(
        adx_values=adx_values, price_vs_ema=price_vs_ema,
        price_trend=price_trend, vix=vix_ltp,
        nifty_range_7d=nifty_range, bb_width_pctl=bb_pctl,
    )
else:
    regime_result = RegimeResult(regime="UNCERTAIN", confidence=0.3, detail="data unavailable")

logger.info("Regime: %s", regime_result.detail)
```

Then for each candidate, check strategy allowance:
```python
# Determine strategy type based on direction + regime
if c["direction"] == "bullish":
    if is_strategy_allowed(regime_result, "BULL_CALL_SPREAD") and has_liquid_options:
        strategy = "BULL_CALL_SPREAD"
    elif is_strategy_allowed(regime_result, "EQUITY_LONG"):
        strategy = "EQUITY_LONG"
    else:
        logger.info("SKIP %s: no bullish strategy allowed in %s regime", symbol, regime_result.regime)
        continue
elif c["direction"] == "bearish":
    if is_strategy_allowed(regime_result, "BEAR_PUT_SPREAD"):
        strategy = "BEAR_PUT_SPREAD"
    else:
        logger.info("SKIP %s: no bearish strategy allowed in %s regime", symbol, regime_result.regime)
        continue
```

**Step 2:** Run existing tests to verify nothing is broken

Run: `python -m pytest tests/test_paper_trade.py -v`

**Step 3: Commit**

```bash
git add paper_trade.py
git commit -m "feat: integrate regime detection into position opening flow"
```

---

### Task 11: Integrate risk manager into paper_trade.py

**Files:**
- Modify: `paper_trade.py`

Wire `RiskManager` into the main flow:

In `run_paper_trade()`:
```python
rm = RiskManager(capital=portfolio["capital"])
rm.set_current_capital(portfolio["capital"] + portfolio["stats"].get("total_pnl", 0))

if rm.should_full_stop():
    logger.warning("FULL STOP: drawdown exceeds %.0f%%. Paper trade for 30 days.",
                    config.TOTAL_DRAWDOWN_HALT * 100)
    _telegram_send("FULL STOP: Drawdown limit reached. Bot halted.")
    return

# Apply VIX multiplier to available capital
vix_mult = rm.vix_size_multiplier(vix_ltp or 15)
effective_capital = portfolio["available_capital"] * vix_mult * rm.weekly_size_multiplier()
```

In `open_positions()`, before opening any position:
```python
# Check correlation guard
if not rm.passes_correlation_guard(portfolio["positions"], new_risk=allocation):
    logger.info("SKIP %s: correlation guard (total risk would exceed %.0f%%)",
                symbol, config.MAX_PORTFOLIO_LOSS_CORRELATION * 100)
    continue

# Check consecutive loss pause
if rm.should_pause_strategy(strategy_instrument, portfolio["closed_trades"]):
    logger.info("SKIP %s: strategy paused after %d consecutive losses",
                symbol, config.CONSECUTIVE_LOSS_PAUSE)
    continue
```

**Step 1:** Implement the integration

**Step 2:** Run tests

Run: `python -m pytest tests/ -v`

**Step 3: Commit**

```bash
git add paper_trade.py
git commit -m "feat: integrate risk manager (drawdown limits, correlation guard)"
```

---

### Task 12: Integrate YouTube intel into screener and paper_trade

**Files:**
- Modify: `screener.py`
- Modify: `paper_trade.py`

In `screener.py`, add to `run_screener()`:
```python
# Before Claude analysis, fetch YouTube market intel
try:
    from youtube_intel import fetch_market_intel
    yt_intel = fetch_market_intel()
    if yt_intel:
        # Include in Claude prompt as additional context
        yt_context = f"\nYouTube Market Intel: {json.dumps(yt_intel)}"
except Exception as e:
    logger.debug("YouTube intel unavailable: %s", e)
    yt_intel = None
```

In `paper_trade.py open_positions()`, add stock-level YouTube check:
```python
# YouTube stock intel (gate, not signal)
try:
    from youtube_intel import fetch_stock_intel
    yt_stock = fetch_stock_intel(symbol)
    if yt_stock and yt_stock.get("sentiment"):
        if (c["direction"] == "bullish" and yt_stock["sentiment"] == "bearish" and
            yt_stock.get("confidence") == "high"):
            logger.info("SKIP %s: YouTube strongly bearish (contradicts bullish signal)", symbol)
            continue
        if (c["direction"] == "bearish" and yt_stock["sentiment"] == "bullish" and
            yt_stock.get("confidence") == "high"):
            logger.info("SKIP %s: YouTube strongly bullish (contradicts bearish signal)", symbol)
            continue
except Exception:
    pass  # YouTube intel is optional
```

**Step 1:** Implement

**Step 2:** Run tests

**Step 3: Commit**

```bash
git add screener.py paper_trade.py
git commit -m "feat: integrate YouTube intel as pre-trade gate"
```

---

### Task 13: Add spread monitoring to monitor_positions

**Files:**
- Modify: `paper_trade.py` (monitor_positions, ~line 1360)

Add spread handling to the existing monitor loop:

```python
# In monitor_positions, add spread-specific monitoring:
if pos["instrument"] == "SPREAD":
    # Get underlying price
    underlying_ltp = get_ltp(smart_api, pos["symbol"], pos["token"])
    if not underlying_ltp:
        continue

    # For paper trading, estimate current premium from underlying move
    # (In live trading, we'd fetch actual option premiums)
    long_premium = estimate_premium_from_underlying(pos, "long", underlying_ltp)
    short_premium = estimate_premium_from_underlying(pos, "short", underlying_ltp)

    exit_reason = check_spread_exit(
        pos, underlying_ltp=underlying_ltp, today=_today_ist(),
        long_premium=long_premium, short_premium=short_premium,
    )

    if exit_reason:
        # Calculate spread P&L
        spread_pnl = calc_spread_pnl(
            pos["long_leg"]["entry_premium"], long_premium,
            pos["short_leg"]["entry_premium"], short_premium,
            pos["quantity"],
        )
        # Use underlying price as "exit price" for tracking
        close_position(portfolio, pos, underlying_ltp, exit_reason)
```

Also add `estimate_premium_from_underlying()`:
```python
def estimate_premium_from_underlying(pos, leg, underlying_ltp):
    """Estimate current option premium based on underlying price movement.

    For paper trading only — in live trading, fetch actual LTP.
    Uses simple intrinsic + time value approximation.
    """
    leg_data = pos["long_leg"] if leg == "long" else pos["short_leg"]
    strike = leg_data["strike"]
    entry_premium = leg_data["entry_premium"]
    opt_type = leg_data["option_type"]

    # Intrinsic value
    if opt_type == "CE":
        intrinsic = max(0, underlying_ltp - strike)
    else:  # PE
        intrinsic = max(0, strike - underlying_ltp)

    # Time value decays linearly (rough approximation)
    dte = days_to_expiry(pos["expiry"])
    entry_dte = days_to_expiry(pos["expiry"])  # approximate
    time_value = max(0, entry_premium - max(0,
        pos["underlying_at_entry"] - strike if opt_type == "CE"
        else strike - pos["underlying_at_entry"]))

    # Simple model: intrinsic + fraction of original time value
    if entry_dte > 0:
        tv_remaining = time_value * (dte / max(entry_dte, 1)) * 0.7
    else:
        tv_remaining = 0

    return max(0.05, intrinsic + tv_remaining)  # floor at 0.05
```

**Step 1:** Implement

**Step 2:** Run tests

**Step 3: Commit**

```bash
git add paper_trade.py
git commit -m "feat: add spread monitoring and exit logic to monitor loop"
```

---

## Phase 2.5: Credit Spreads (Bull Put / Bear Call)

Credit spreads reuse the vertical spread infrastructure from Phase 2 but with reversed economics: sell the closer-to-money leg, buy the further OTM leg for protection. Net credit received upfront, profit from time decay and IV crush.

### Task 14: Add credit spread strike selection

**Files:**
- Modify: `agent_with_options.py`
- Test: `tests/test_agent_with_options.py`

**Step 1: Write failing tests**

Add to `tests/test_agent_with_options.py`:
```python
def test_select_credit_spread_bull_put():
    from agent_with_options import select_credit_spread_strikes
    chain = [
        {"strikePrice": 22000, "PE": {"openInterest": 50000, "impliedVolatility": 18, "delta": -0.10, "lastPrice": 45}, "CE": {}},
        {"strikePrice": 22100, "PE": {"openInterest": 80000, "impliedVolatility": 19, "delta": -0.15, "lastPrice": 65}, "CE": {}},
        {"strikePrice": 22200, "PE": {"openInterest": 120000, "impliedVolatility": 20, "delta": -0.22, "lastPrice": 95}, "CE": {}},
        {"strikePrice": 22300, "PE": {"openInterest": 90000, "impliedVolatility": 21, "delta": -0.28, "lastPrice": 130}, "CE": {}},
        {"strikePrice": 22400, "PE": {"openInterest": 60000, "impliedVolatility": 22, "delta": -0.35, "lastPrice": 175}, "CE": {}},
    ]
    result = select_credit_spread_strikes(
        chain, spot=22500, direction="bullish", max_loss_budget=2000, lot_size=75
    )
    assert result is not None
    assert result["short_strike"] > result["long_strike"]  # bull put: sell higher put, buy lower put
    assert result["net_credit"] > 0
    assert result["max_loss"] <= 2000
    assert result["max_loss"] == (result["short_strike"] - result["long_strike"]) * 75 - result["net_credit"]

def test_select_credit_spread_bear_call():
    from agent_with_options import select_credit_spread_strikes
    chain = [
        {"strikePrice": 22600, "CE": {"openInterest": 60000, "impliedVolatility": 22, "delta": 0.35, "lastPrice": 170}, "PE": {}},
        {"strikePrice": 22700, "CE": {"openInterest": 90000, "impliedVolatility": 21, "delta": 0.28, "lastPrice": 125}, "PE": {}},
        {"strikePrice": 22800, "CE": {"openInterest": 120000, "impliedVolatility": 20, "delta": 0.22, "lastPrice": 90}, "PE": {}},
        {"strikePrice": 22900, "CE": {"openInterest": 80000, "impliedVolatility": 19, "delta": 0.15, "lastPrice": 60}, "PE": {}},
        {"strikePrice": 23000, "CE": {"openInterest": 50000, "impliedVolatility": 18, "delta": 0.10, "lastPrice": 40}, "PE": {}},
    ]
    result = select_credit_spread_strikes(
        chain, spot=22500, direction="bearish", max_loss_budget=2000, lot_size=75
    )
    assert result is not None
    assert result["short_strike"] < result["long_strike"]  # bear call: sell lower call, buy higher call
    assert result["net_credit"] > 0
    assert result["max_loss"] <= 2000

def test_select_credit_spread_none_if_credit_too_low():
    from agent_with_options import select_credit_spread_strikes
    chain = [
        {"strikePrice": 21000, "PE": {"openInterest": 50000, "impliedVolatility": 12, "delta": -0.02, "lastPrice": 2}, "CE": {}},
        {"strikePrice": 21100, "PE": {"openInterest": 40000, "impliedVolatility": 13, "delta": -0.03, "lastPrice": 5}, "CE": {}},
    ]
    result = select_credit_spread_strikes(
        chain, spot=22500, direction="bullish", max_loss_budget=2000, lot_size=75
    )
    assert result is None  # credit too low (strikes too far OTM)

def test_select_credit_spread_none_if_low_oi():
    from agent_with_options import select_credit_spread_strikes
    chain = [
        {"strikePrice": 22200, "PE": {"openInterest": 100, "impliedVolatility": 20, "delta": -0.22, "lastPrice": 95}, "CE": {}},
        {"strikePrice": 22300, "PE": {"openInterest": 200, "impliedVolatility": 21, "delta": -0.28, "lastPrice": 130}, "CE": {}},
    ]
    result = select_credit_spread_strikes(
        chain, spot=22500, direction="bullish", max_loss_budget=2000, lot_size=75
    )
    assert result is None  # OI too low for Nifty
```

**Step 2: Run to verify failure**

Run: `python -m pytest tests/test_agent_with_options.py -k "credit" -v`
Expected: FAIL — `select_credit_spread_strikes` not defined

**Step 3: Implement select_credit_spread_strikes()**

Add to `agent_with_options.py`:
```python
def select_credit_spread_strikes(chain, spot, direction, max_loss_budget, lot_size,
                                  min_oi=500, min_credit_per_lot=30, target_delta=0.25):
    """Select strikes for a credit spread (bull put or bear call).

    Bull put: sell OTM put (delta ~0.25), buy further OTM put for protection.
    Bear call: sell OTM call (delta ~0.25), buy further OTM call for protection.
    Net credit received upfront. Max loss = width - credit.

    Returns dict with short_strike, long_strike, net_credit, max_loss, max_profit, rr_ratio
    or None if no valid spread found.
    """
    if direction == "bullish":
        # Bull put spread: sell higher put, buy lower put (both below spot)
        opt_key = "PE"
        # Filter strikes below spot
        candidates = [s for s in chain if s["strikePrice"] < spot and s.get(opt_key, {}).get("openInterest", 0) >= min_oi]
        candidates.sort(key=lambda s: s["strikePrice"], reverse=True)  # highest first (closest to spot)
    else:
        # Bear call spread: sell lower call, buy higher call (both above spot)
        opt_key = "CE"
        candidates = [s for s in chain if s["strikePrice"] > spot and s.get(opt_key, {}).get("openInterest", 0) >= min_oi]
        candidates.sort(key=lambda s: s["strikePrice"])  # lowest first (closest to spot)

    if len(candidates) < 2:
        return None

    # Find short leg: closest to target delta
    best_short = None
    best_delta_diff = float("inf")
    for c in candidates:
        delta = abs(c[opt_key].get("delta", 0))
        diff = abs(delta - target_delta)
        if diff < best_delta_diff:
            best_delta_diff = diff
            best_short = c

    if not best_short:
        return None

    short_premium = best_short[opt_key]["lastPrice"]
    short_strike = best_short["strikePrice"]

    # Find long leg: 1-3 strikes further OTM from short
    if direction == "bullish":
        long_candidates = [c for c in candidates if c["strikePrice"] < short_strike]
    else:
        long_candidates = [c for c in candidates if c["strikePrice"] > short_strike]

    if not long_candidates:
        return None

    # Try each long candidate, pick best risk/reward within budget
    best_spread = None
    for lc in long_candidates[:3]:  # try up to 3 strikes away
        long_premium = lc[opt_key]["lastPrice"]
        long_strike = lc["strikePrice"]
        width = abs(short_strike - long_strike)

        net_credit = (short_premium - long_premium) * lot_size
        max_loss = (width * lot_size) - net_credit
        max_profit = net_credit

        if net_credit <= 0 or net_credit < min_credit_per_lot * lot_size:
            continue
        if max_loss > max_loss_budget:
            continue
        if max_loss <= 0:
            continue

        rr = max_profit / max_loss

        if best_spread is None or rr > best_spread["rr_ratio"]:
            best_spread = {
                "short_strike": short_strike,
                "long_strike": long_strike,
                "short_premium": short_premium,
                "long_premium": long_premium,
                "net_credit": net_credit,
                "max_loss": max_loss,
                "max_profit": max_profit,
                "rr_ratio": round(rr, 2),
                "width": width,
            }

    return best_spread
```

**Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_agent_with_options.py -k "credit" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add agent_with_options.py tests/test_agent_with_options.py
git commit -m "feat: add credit spread strike selection (bull put / bear call)"
```

---

### Task 15: Add IV routing and credit spread opening to paper_trade.py

**Files:**
- Modify: `paper_trade.py`
- Modify: `config.py`
- Test: `tests/test_paper_trade.py`

**Step 1: Add credit spread config constants**

Add to `config.py`:
```python
# Credit spread config (Strategy F)
CREDIT_SPREAD_MIN_IV_PERCENTILE = 50      # only enter when IV elevated
CREDIT_SPREAD_PREFER_IV_PERCENTILE = 70   # strong preference above this
DEBIT_SPREAD_MAX_IV_PERCENTILE = 70       # block debit spreads above this
CREDIT_SPREAD_TARGET_PCT = 0.50           # take profit at 50% of max
CREDIT_SPREAD_SL_MULTIPLIER = 2.0         # stop at 2x credit received
CREDIT_SPREAD_TIME_EXIT_DAYS = 5          # close 5 days before expiry
CREDIT_SPREAD_MAX_DELTA = 0.35            # close if short delta exceeds this
CREDIT_SPREAD_MIN_VIX = 15                # don't sell premium below VIX 15
CREDIT_SPREAD_MAX_VIX = 25                # don't sell premium above VIX 25
```

**Step 2: Write failing tests**

Add to `tests/test_paper_trade.py`:
```python
def test_iv_routing_high_iv_blocks_debit():
    """When IV > 70%, debit spreads should be blocked."""
    from config import DEBIT_SPREAD_MAX_IV_PERCENTILE
    iv_percentile = 75
    assert iv_percentile > DEBIT_SPREAD_MAX_IV_PERCENTILE
    # The routing logic should return "credit" not "debit"

def test_iv_routing_low_iv_prefers_debit():
    """When IV < 50%, credit spreads should be blocked."""
    from config import CREDIT_SPREAD_MIN_IV_PERCENTILE
    iv_percentile = 40
    assert iv_percentile < CREDIT_SPREAD_MIN_IV_PERCENTILE
    # The routing logic should return "debit" not "credit"

def test_open_credit_spread_position_structure():
    """Credit spread position should have spread_type='credit' and net_credit field."""
    # Will test _open_spread_position with spread_type="credit"
    pass  # Filled in Step 3

def test_credit_spread_exit_target():
    """Credit spread should exit when P&L reaches 50% of max profit."""
    from paper_trade import check_spread_exit
    pos = {
        "spread_type": "credit",
        "spread_direction": "bullish",
        "net_credit": 3000,          # credit received
        "max_profit": 3000,          # max profit = credit
        "max_loss": 4500,            # width*qty - credit
        "long_leg": {"strike": 22100, "entry_premium": 65, "option_type": "PE"},
        "short_leg": {"strike": 22300, "entry_premium": 130, "option_type": "PE"},
        "underlying_at_entry": 22500,
        "entry_date": "2026-03-01",
        "expiry": "2026-03-26",
        "quantity": 75,
    }
    # Premium decayed — short leg worth 70 (from 130), long leg worth 30 (from 65)
    # Current net debit to close = (70 - 30) * 75 = 3000... but we received 3000 credit
    # P&L = credit - cost_to_close = 3000 - 3000 = 0 (no profit yet)
    # If short drops to 50, long drops to 20: cost_to_close = (50-20)*75 = 2250
    # P&L = 3000 - 2250 = 750 = 25% of max profit (not enough)
    # If short drops to 30, long drops to 10: cost_to_close = (30-10)*75 = 1500
    # P&L = 3000 - 1500 = 1500 = 50% of max profit -> EXIT
    exit_reason = check_spread_exit(
        pos, underlying_ltp=22550, today=date(2026, 3, 10),
        long_premium=10, short_premium=30,
    )
    assert exit_reason == "credit_target_50pct"

def test_credit_spread_exit_stoploss():
    """Credit spread should exit when loss reaches 2x credit."""
    from paper_trade import check_spread_exit
    pos = {
        "spread_type": "credit",
        "spread_direction": "bullish",
        "net_credit": 3000,
        "max_profit": 3000,
        "max_loss": 4500,
        "long_leg": {"strike": 22100, "entry_premium": 65, "option_type": "PE"},
        "short_leg": {"strike": 22300, "entry_premium": 130, "option_type": "PE"},
        "underlying_at_entry": 22500,
        "entry_date": "2026-03-01",
        "expiry": "2026-03-26",
        "quantity": 75,
    }
    # Underlying dropped, short put went deep ITM
    # Short worth 300, long worth 180: cost_to_close = (300-180)*75 = 9000
    # P&L = 3000 - 9000 = -6000 = 2x credit -> STOP
    exit_reason = check_spread_exit(
        pos, underlying_ltp=22100, today=date(2026, 3, 10),
        long_premium=180, short_premium=300,
    )
    assert exit_reason == "credit_stoploss_2x"
```

**Step 3: Implement IV routing and credit spread support**

In `paper_trade.py`, modify the spread opening logic to route between debit and credit:
```python
def _select_spread_type(direction, iv_percentile, vix):
    """Route between debit and credit spread based on IV environment.

    Returns 'debit', 'credit', or None if neither is viable.
    """
    from config import (
        DEBIT_SPREAD_MAX_IV_PERCENTILE,
        CREDIT_SPREAD_MIN_IV_PERCENTILE,
        CREDIT_SPREAD_MIN_VIX,
        CREDIT_SPREAD_MAX_VIX,
    )

    can_debit = iv_percentile <= DEBIT_SPREAD_MAX_IV_PERCENTILE
    can_credit = (
        iv_percentile >= CREDIT_SPREAD_MIN_IV_PERCENTILE
        and CREDIT_SPREAD_MIN_VIX <= vix <= CREDIT_SPREAD_MAX_VIX
    )

    if can_credit and not can_debit:
        return "credit"
    if can_debit and not can_credit:
        return "debit"
    if can_debit and can_credit:
        # In the overlap zone (50-70%), prefer debit (known edge from design)
        return "debit"
    return None  # Neither viable (VIX out of range for credit, IV too high for debit)
```

Extend `_open_spread_position()` to handle `spread_type="credit"`:
```python
# In _open_spread_position, add spread_type parameter:
# For credit spreads:
#   - short leg is the closer-to-money leg (higher premium)
#   - long leg is further OTM (lower premium, protection)
#   - Record net_credit instead of net_debit
#   - Position structure gets: "spread_type": "credit", "net_credit": amount
```

Extend `check_spread_exit()` to handle credit-specific exits:
```python
# Add to check_spread_exit():
if pos.get("spread_type") == "credit":
    net_credit = pos["net_credit"]
    cost_to_close = abs(short_premium - long_premium) * pos["quantity"]
    current_pnl = net_credit - cost_to_close

    # Target: 50% of max profit (credit received)
    if current_pnl >= net_credit * CREDIT_SPREAD_TARGET_PCT:
        return "credit_target_50pct"

    # Stoploss: loss exceeds 2x credit
    if current_pnl <= -(net_credit * CREDIT_SPREAD_SL_MULTIPLIER):
        return "credit_stoploss_2x"

    # Short strike breach
    if pos["spread_direction"] == "bullish" and underlying_ltp <= pos["short_leg"]["strike"]:
        return "credit_short_strike_breach"
    if pos["spread_direction"] == "bearish" and underlying_ltp >= pos["short_leg"]["strike"]:
        return "credit_short_strike_breach"
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_paper_trade.py -k "credit" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add paper_trade.py config.py tests/test_paper_trade.py
git commit -m "feat: add credit spread support with IV routing (bull put / bear call)"
```

---

### Task 16: Add CASH regime and 100% cash triggers

**Files:**
- Modify: `regime.py`
- Modify: `risk_manager.py`
- Modify: `config.py`
- Test: `tests/test_regime.py`, `tests/test_risk_manager.py`

**Step 1: Write failing tests**

Add to `tests/test_regime.py`:
```python
def test_cash_regime_on_monthly_drawdown():
    from regime import classify_regime
    # When risk_manager reports monthly drawdown > 8%, regime should be CASH
    risk_state = {"monthly_drawdown_pct": -9.0, "consecutive_losses": 2}
    result = classify_regime(nifty_data={}, vix=16, risk_state=risk_state)
    assert result["regime"] == "CASH"

def test_cash_regime_on_consecutive_losses():
    from regime import classify_regime
    risk_state = {"monthly_drawdown_pct": -2.0, "consecutive_losses": 5}
    result = classify_regime(nifty_data={}, vix=16, risk_state=risk_state)
    assert result["regime"] == "CASH"

def test_cash_regime_on_high_vix():
    from regime import classify_regime
    risk_state = {"monthly_drawdown_pct": 0, "consecutive_losses": 0}
    result = classify_regime(nifty_data={}, vix=26, risk_state=risk_state)
    assert result["regime"] == "CASH"
```

Add to `tests/test_risk_manager.py`:
```python
def test_cash_duration_8pct_drawdown():
    from risk_manager import RiskManager
    rm = RiskManager(initial_capital=100000)
    rm.record_pnl(-8500)  # 8.5% monthly drawdown
    assert rm.is_cash_mode() is True
    assert rm.cash_days_remaining() == 5

def test_cash_duration_12pct_drawdown():
    from risk_manager import RiskManager
    rm = RiskManager(initial_capital=100000)
    rm.record_pnl(-12500)  # 12.5% monthly drawdown
    assert rm.is_cash_mode() is True
    assert rm.cash_days_remaining() == 10

def test_cash_duration_consecutive_losses():
    from risk_manager import RiskManager
    rm = RiskManager(initial_capital=100000)
    for _ in range(5):
        rm.record_loss()
    assert rm.is_cash_mode() is True
    assert rm.cash_days_remaining() == 5  # 1 week
```

**Step 2: Run to verify failure**

Run: `python -m pytest tests/test_regime.py tests/test_risk_manager.py -k "cash" -v`
Expected: FAIL

**Step 3: Implement CASH regime and cash mode tracking**

Add to `config.py`:
```python
# CASH regime triggers
CASH_TRIGGER_MONTHLY_DRAWDOWN_PCT = 8.0   # Go 100% cash if monthly drawdown > 8%
CASH_TRIGGER_SEVERE_DRAWDOWN_PCT = 12.0   # Extended cash if monthly drawdown > 12%
CASH_TRIGGER_CONSECUTIVE_LOSSES = 5        # Go cash after 5 consecutive losses
CASH_TRIGGER_VIX = 25                      # Go cash when VIX > 25
CASH_DURATION_NORMAL = 5                   # 5 trading days for 8% drawdown
CASH_DURATION_SEVERE = 10                  # 10 trading days for 12% drawdown
CASH_DURATION_LOSSES = 5                   # 5 trading days (1 week) for consecutive losses
```

In `regime.py`, add CASH check as the first thing in `classify_regime()`:
```python
def classify_regime(nifty_data, vix, risk_state=None):
    # CASH regime overrides everything
    if risk_state:
        if risk_state.get("monthly_drawdown_pct", 0) <= -CASH_TRIGGER_MONTHLY_DRAWDOWN_PCT:
            return {"regime": "CASH", "confidence": 1.0, "reason": f"Monthly drawdown {risk_state['monthly_drawdown_pct']:.1f}%"}
        if risk_state.get("consecutive_losses", 0) >= CASH_TRIGGER_CONSECUTIVE_LOSSES:
            return {"regime": "CASH", "confidence": 1.0, "reason": f"{risk_state['consecutive_losses']} consecutive losses"}
    if vix and vix > CASH_TRIGGER_VIX:
        return {"regime": "CASH", "confidence": 1.0, "reason": f"VIX {vix} > {CASH_TRIGGER_VIX}"}

    # ... rest of regime classification
```

In `risk_manager.py`, add cash mode tracking:
```python
def is_cash_mode(self):
    if self._cash_until and _today_ist() < self._cash_until:
        return True
    return False

def cash_days_remaining(self):
    if not self._cash_until:
        return 0
    delta = (self._cash_until - _today_ist()).days
    return max(0, delta)
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_regime.py tests/test_risk_manager.py -k "cash" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add regime.py risk_manager.py config.py tests/test_regime.py tests/test_risk_manager.py
git commit -m "feat: add CASH regime with 100% cash triggers (drawdown, losses, VIX)"
```

---

## Phase 5: Iron Condor and Momentum (Optional — implement after Phase 1-4 validated)

### Task 17: Add iron condor opening logic

Add `_open_iron_condor()` to paper_trade.py. Only activates when regime = SIDEWAYS and all condor checks pass. Uses Nifty index options (not stock options — always liquid, no delivery risk).

### Task 18: Add momentum options buying logic

Add `_open_momentum_position()` to paper_trade.py. Buys Nifty CE/PE on breakout signals with 1% risk cap, 35% stop, 90% target.

### Task 19: Add earnings calendar and event checking

Add event calendar data to config.py. Check before every trade for earnings/RBI/budget dates.

---

## Execution Summary

| Phase | Tasks | Priority | Est. Complexity |
|-------|-------|----------|----------------|
| Phase 0: Setup | Task 0 | Prerequisite | Low |
| Phase 1: Foundation | Tasks 1-5 | Critical | Medium |
| Phase 2: Debit Spreads | Tasks 6-8 | Critical | High |
| Phase 2.5: Credit Spreads | Tasks 14-16 | Critical | Medium |
| Phase 3: YouTube Intel | Task 9 | Medium | Medium |
| Phase 4: Integration | Tasks 10-13 | Critical | High |
| Phase 5: Condor + Momentum | Tasks 17-19 | Low (after validation) | Medium |

**Total: 20 tasks. Phases 0-4 + 2.5 (17 tasks) are the MVP.**

### Key Design References for Implementation

Developers implementing this plan should read these design doc sections before starting:
- **Part 7.5: Data Sources** — IV percentile (VIX proxy), delta calculation (Black-Scholes), slippage model, F&O ban source, earnings calendar
- **Part 7.5: First-Run Warmup** — cold start handling, fallback to UNCERTAIN regime
- **Part 7.5: Spread Position Schema** — position dict structure for spreads vs equity
- **Part 7.5: Signal → Regime → Strategy Decision Tree** — complete routing logic
- **Part 7.5: Capital Allocation Constants** — all constants to add to config.py
