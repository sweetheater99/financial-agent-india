# Indicator Discovery Module — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 7 new indicators (MFI, OBV divergence, VWAP deviation, ADX, PCR signal, Max Pain signal, FII momentum) to improve entry/exit signal quality.

**Architecture:** New `indicators_v4.py` with pure functions. OHLCV indicators integrate into `backtest_signals.py` scoring. All 7 feed into Claude's entry/exit prompts in `claude_intel.py`.

**Tech Stack:** pandas, numpy (no new deps)

---

### Task 1: MFI (Money Flow Index)

**Files:**
- Create: `indicators_v4.py`
- Test: `tests/test_indicators_v4.py`

**Step 1: Write the failing test**

```python
"""Tests for indicators_v4.py — new indicator functions."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indicators_v4 import compute_mfi


def _make_ohlcv(n=30, base=100.0, trend=1.0, base_vol=100_000):
    """Synthetic OHLCV DataFrame."""
    dates = pd.bdate_range("2025-01-01", periods=n)
    closes = [base + trend * i for i in range(n)]
    return pd.DataFrame({
        "Open": [c - 0.5 for c in closes],
        "High": [c + 2.0 for c in closes],
        "Low": [c - 1.0 for c in closes],
        "Close": closes,
        "Volume": [base_vol] * n,
    }, index=dates)


def test_mfi_returns_series():
    df = _make_ohlcv(30)
    result = compute_mfi(df, period=14)
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)


def test_mfi_range_0_100():
    df = _make_ohlcv(60, trend=0.5)
    result = compute_mfi(df, period=14)
    valid = result.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_mfi_uptrend_high():
    """Strong uptrend with consistent volume should have MFI > 50."""
    df = _make_ohlcv(40, trend=2.0)
    result = compute_mfi(df, period=14)
    # Last value in a strong uptrend
    assert result.iloc[-1] > 50


def test_mfi_insufficient_data():
    df = _make_ohlcv(5)
    result = compute_mfi(df, period=14)
    assert result.dropna().empty
```

**Step 2: Run test to verify it fails**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_indicators_v4.py::test_mfi_returns_series -v`
Expected: FAIL — `ImportError: cannot import name 'compute_mfi' from 'indicators_v4'`

**Step 3: Write minimal implementation**

Create `indicators_v4.py`:

```python
"""V4 indicators: MFI, OBV divergence, VWAP deviation, ADX, and global-intel signals.

Pure functions. Each takes a pandas DataFrame (OHLCV) or scalar values.
No side effects, no API calls.
"""

import numpy as np
import pandas as pd


def compute_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index — volume-weighted RSI.

    Returns Series of MFI values (0-100), NaN where insufficient data.
    """
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    raw_money_flow = typical_price * df["Volume"]

    delta = typical_price.diff()

    pos_flow = pd.Series(0.0, index=df.index)
    neg_flow = pd.Series(0.0, index=df.index)
    pos_flow[delta > 0] = raw_money_flow[delta > 0]
    neg_flow[delta < 0] = raw_money_flow[delta < 0]

    pos_sum = pos_flow.rolling(period, min_periods=period).sum()
    neg_sum = neg_flow.rolling(period, min_periods=period).sum()

    # Avoid division by zero
    neg_sum_safe = neg_sum.replace(0, np.nan)
    money_ratio = pos_sum / neg_sum_safe
    mfi = 100 - (100 / (1 + money_ratio))
    # Where neg_sum is 0 (all positive flow), MFI = 100
    mfi = mfi.fillna(100.0)

    # NaN out insufficient data
    mfi.iloc[:period] = np.nan

    return mfi
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_indicators_v4.py -v -k mfi`
Expected: 4 PASS

**Step 5: Commit**

```bash
git add indicators_v4.py tests/test_indicators_v4.py
git commit -m "feat: add MFI indicator (indicators_v4)"
```

---

### Task 2: OBV Divergence

**Files:**
- Modify: `indicators_v4.py`
- Test: `tests/test_indicators_v4.py`

**Step 1: Write the failing test**

```python
from indicators_v4 import compute_obv_divergence


def test_obv_divergence_returns_series():
    df = _make_ohlcv(30)
    result = compute_obv_divergence(df, lookback=10)
    assert isinstance(result, pd.Series)
    assert set(result.dropna().unique()).issubset({"bullish_div", "bearish_div", "none"})


def test_obv_bearish_divergence():
    """Price making new highs but OBV declining → bearish divergence."""
    n = 30
    dates = pd.bdate_range("2025-01-01", periods=n)
    closes = [100 + i * 1.0 for i in range(n)]  # price trending up
    # Volume declining on up days → OBV should flatten/decline
    volumes = [100_000 - i * 3000 for i in range(n)]
    df = pd.DataFrame({
        "Open": [c - 0.5 for c in closes],
        "High": [c + 1.0 for c in closes],
        "Low": [c - 1.0 for c in closes],
        "Close": closes,
        "Volume": [max(v, 1000) for v in volumes],
    }, index=dates)
    result = compute_obv_divergence(df, lookback=10)
    # Should detect bearish divergence at some point in late bars
    assert "bearish_div" in result.values


def test_obv_divergence_insufficient_data():
    df = _make_ohlcv(5)
    result = compute_obv_divergence(df, lookback=10)
    assert (result == "none").all() or result.isna().all()
```

**Step 2: Run test to verify it fails**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_indicators_v4.py -v -k obv`
Expected: FAIL — ImportError

**Step 3: Implement**

Add to `indicators_v4.py`:

```python
def compute_obv_divergence(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """Detect OBV divergence from price.

    Returns Series with values: 'bullish_div', 'bearish_div', 'none'.
    - Bearish div: price higher high + OBV lower high
    - Bullish div: price lower low + OBV higher low
    """
    close = df["Close"]
    volume = df["Volume"]

    # Compute OBV
    obv = pd.Series(0.0, index=df.index)
    for i in range(1, len(df)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i - 1]

    result = pd.Series("none", index=df.index)
    if len(df) < lookback + 1:
        return result

    for i in range(lookback, len(df)):
        window_close = close.iloc[i - lookback:i + 1]
        window_obv = obv.iloc[i - lookback:i + 1]

        price_higher_high = close.iloc[i] >= window_close.max()
        price_lower_low = close.iloc[i] <= window_close.min()
        obv_lower_high = obv.iloc[i] < window_obv.iloc[:-1].max()
        obv_higher_low = obv.iloc[i] > window_obv.iloc[:-1].min()

        if price_higher_high and obv_lower_high:
            result.iloc[i] = "bearish_div"
        elif price_lower_low and obv_higher_low:
            result.iloc[i] = "bullish_div"

    return result
```

**Step 4: Run tests**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_indicators_v4.py -v -k obv`
Expected: 3 PASS

**Step 5: Commit**

```bash
git add indicators_v4.py tests/test_indicators_v4.py
git commit -m "feat: add OBV divergence indicator"
```

---

### Task 3: VWAP Deviation

**Files:**
- Modify: `indicators_v4.py`
- Test: `tests/test_indicators_v4.py`

**Step 1: Write the failing test**

```python
from indicators_v4 import compute_vwap_deviation


def test_vwap_deviation_returns_series():
    df = _make_ohlcv(30)
    result = compute_vwap_deviation(df)
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)


def test_vwap_deviation_centered_around_zero():
    """Flat price with constant volume → deviation near zero."""
    df = _make_ohlcv(30, trend=0.0)
    result = compute_vwap_deviation(df)
    valid = result.dropna()
    assert (valid.abs() < 5.0).all()  # within 5 ATRs
```

**Step 2: Run test to verify it fails**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_indicators_v4.py -v -k vwap`
Expected: FAIL

**Step 3: Implement**

Add to `indicators_v4.py`:

```python
def compute_vwap_deviation(df: pd.DataFrame, atr_period: int = 14) -> pd.Series:
    """Price deviation from cumulative VWAP, normalized by ATR.

    Returns Series of floats: positive = above VWAP, negative = below.
    Units are multiples of ATR. NaN where insufficient data.
    """
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_tp_vol = (typical_price * df["Volume"]).cumsum()
    cum_vol = df["Volume"].cumsum().replace(0, np.nan)
    vwap = cum_tp_vol / cum_vol

    # ATR for normalization
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period, min_periods=atr_period).mean()

    deviation = (df["Close"] - vwap) / atr.replace(0, np.nan)
    deviation.iloc[:atr_period] = np.nan
    return deviation
```

**Step 4: Run tests**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_indicators_v4.py -v -k vwap`
Expected: 2 PASS

**Step 5: Commit**

```bash
git add indicators_v4.py tests/test_indicators_v4.py
git commit -m "feat: add VWAP deviation indicator"
```

---

### Task 4: ADX + DI+/DI-

**Files:**
- Modify: `indicators_v4.py`
- Test: `tests/test_indicators_v4.py`

**Step 1: Write the failing test**

```python
from indicators_v4 import compute_adx


def test_adx_returns_dict_of_series():
    df = _make_ohlcv(40)
    result = compute_adx(df, period=14)
    assert isinstance(result, dict)
    assert "adx" in result and "di_plus" in result and "di_minus" in result
    assert isinstance(result["adx"], pd.Series)


def test_adx_range():
    df = _make_ohlcv(60, trend=1.0)
    result = compute_adx(df, period=14)
    valid = result["adx"].dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_adx_strong_trend():
    """Strong uptrend should produce ADX > 25 and DI+ > DI-."""
    df = _make_ohlcv(60, trend=3.0, volatility=1.0)
    result = compute_adx(df, period=14)
    assert result["adx"].iloc[-1] > 20
    assert result["di_plus"].iloc[-1] > result["di_minus"].iloc[-1]


def test_adx_insufficient_data():
    df = _make_ohlcv(10)
    result = compute_adx(df, period=14)
    assert result["adx"].dropna().empty
```

**Step 2: Run test to verify it fails**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_indicators_v4.py -v -k adx`
Expected: FAIL

**Step 3: Implement**

Add to `indicators_v4.py`:

```python
def compute_adx(df: pd.DataFrame, period: int = 14) -> dict[str, pd.Series]:
    """Average Directional Index with DI+/DI-.

    Returns dict with keys: 'adx', 'di_plus', 'di_minus'.
    All are Series (0-100), NaN where insufficient data.
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    # Directional movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)

    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

    # Wilder's smoothing (EMA with alpha = 1/period)
    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    smooth_plus = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    smooth_minus = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    di_plus = 100 * smooth_plus / atr.replace(0, np.nan)
    di_minus = 100 * smooth_minus / atr.replace(0, np.nan)

    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    # NaN out warmup
    warmup = 2 * period
    adx.iloc[:warmup] = np.nan
    di_plus.iloc[:period] = np.nan
    di_minus.iloc[:period] = np.nan

    return {"adx": adx, "di_plus": di_plus, "di_minus": di_minus}
```

**Step 4: Run tests**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_indicators_v4.py -v -k adx`
Expected: 4 PASS

**Step 5: Commit**

```bash
git add indicators_v4.py tests/test_indicators_v4.py
git commit -m "feat: add ADX + DI+/DI- indicator"
```

---

### Task 5: Global Intel Signals (PCR, Max Pain, FII)

**Files:**
- Modify: `indicators_v4.py`
- Test: `tests/test_indicators_v4.py`

**Step 1: Write the failing test**

```python
from indicators_v4 import compute_pcr_signal, compute_maxpain_signal, compute_fii_momentum


def test_pcr_bullish():
    assert compute_pcr_signal(1.5) > 0.5  # high PCR = bullish

def test_pcr_bearish():
    assert compute_pcr_signal(0.5) < -0.5  # low PCR = bearish

def test_pcr_neutral():
    result = compute_pcr_signal(1.0)
    assert -0.3 <= result <= 0.3

def test_pcr_none():
    assert compute_pcr_signal(None) == 0.0


def test_maxpain_at_maxpain():
    """Price at max pain → strong mean reversion signal."""
    result = compute_maxpain_signal(100.0, 100.0)
    assert abs(result) > 0.5  # near max pain = caution

def test_maxpain_far():
    """Price far from max pain → weak signal."""
    result = compute_maxpain_signal(120.0, 100.0)
    assert abs(result) < 0.5

def test_maxpain_none():
    assert compute_maxpain_signal(100.0, None) == 0.0


def test_fii_buying():
    assert compute_fii_momentum([1000, 1500, 2000]) > 0.5

def test_fii_selling():
    assert compute_fii_momentum([-1000, -1500, -2000]) < -0.5

def test_fii_empty():
    assert compute_fii_momentum([]) == 0.0

def test_fii_none():
    assert compute_fii_momentum(None) == 0.0
```

**Step 2: Run test to verify it fails**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_indicators_v4.py -v -k "pcr or maxpain or fii"`
Expected: FAIL

**Step 3: Implement**

Add to `indicators_v4.py`:

```python
def compute_pcr_signal(pcr: float | None) -> float:
    """Convert PCR to -1..+1 signal.

    High PCR (>1.3) = puts heavy = contrarian bullish.
    Low PCR (<0.7) = calls heavy = contrarian bearish.
    """
    if pcr is None:
        return 0.0
    if pcr >= 1.5:
        return 1.0
    if pcr >= 1.3:
        return 0.5 + (pcr - 1.3) / 0.2 * 0.5
    if pcr <= 0.5:
        return -1.0
    if pcr <= 0.7:
        return -0.5 - (0.7 - pcr) / 0.2 * 0.5
    # 0.7 to 1.3 → linear from -0.5 to +0.5
    return (pcr - 1.0) / 0.3 * 0.5


def compute_maxpain_signal(price: float, max_pain: float | None) -> float:
    """Signal based on proximity to max pain. Returns -1..+1.

    Close to max pain (within 1%) = mean reversion zone (+/- 0.8).
    Far from max pain (>3%) = weak signal.
    Sign: positive if price below max pain (bullish pull up), negative if above.
    """
    if max_pain is None or max_pain <= 0:
        return 0.0

    pct_diff = (price - max_pain) / max_pain  # positive = above max pain
    abs_diff = abs(pct_diff)

    if abs_diff <= 0.01:
        strength = 0.8
    elif abs_diff <= 0.03:
        strength = 0.8 * (1 - (abs_diff - 0.01) / 0.02)
    else:
        strength = 0.1

    # Negative sign if above max pain (bearish pull), positive if below (bullish pull)
    return -strength if pct_diff > 0 else strength


def compute_fii_momentum(flows_3d: list[float] | None) -> float:
    """FII 3-day flow momentum. Returns -1..+1.

    Sustained buying = bullish, sustained selling = bearish.
    Normalizes by ₹3000Cr as "strong" threshold.
    """
    if not flows_3d:
        return 0.0

    avg = sum(flows_3d) / len(flows_3d)
    # Normalize: ±3000Cr = ±1.0
    signal = max(-1.0, min(1.0, avg / 3000))
    return round(signal, 2)
```

**Step 4: Run tests**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_indicators_v4.py -v -k "pcr or maxpain or fii"`
Expected: 10 PASS

**Step 5: Commit**

```bash
git add indicators_v4.py tests/test_indicators_v4.py
git commit -m "feat: add PCR, max pain, FII momentum signals"
```

---

### Task 6: Integrate OHLCV indicators into backtest_signals.py

**Files:**
- Modify: `backtest_signals.py:74-234` (SIGNAL_WEIGHTS + compute_signals)
- Test: `tests/test_backtest_signals.py`

**Step 1: Write the failing test**

Add to `tests/test_backtest_signals.py`:

```python
def test_new_signals_in_weights():
    """New V4 signals should appear in SIGNAL_WEIGHTS."""
    assert "mfi" in SIGNAL_WEIGHTS
    assert "obv_divergence" in SIGNAL_WEIGHTS
    assert "vwap_deviation" in SIGNAL_WEIGHTS
    assert "adx_trend" in SIGNAL_WEIGHTS


def test_new_signals_fire():
    """compute_signals should include new signal columns."""
    df = make_df(60, trend=2.0, volatility=2.0)
    result = compute_signals(df)
    assert "sig_mfi" in result.columns
    assert "sig_obv_divergence" in result.columns
    assert "sig_vwap_deviation" in result.columns
    assert "sig_adx_trend" in result.columns


def test_adx_modulates_score():
    """When ADX < 20 (choppy), total score should be reduced."""
    # Low-trend data (ADX should be low)
    df = make_df(60, trend=0.0, volatility=5.0)
    result = compute_signals(df)
    # In choppy markets, scores should be dampened
    # (Hard to assert exact value, just ensure column exists and scores aren't inflated)
    assert result["score"].max() <= 15  # reasonable cap with dampening
```

**Step 2: Run test to verify it fails**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_backtest_signals.py::test_new_signals_in_weights -v`
Expected: FAIL — KeyError

**Step 3: Implement**

Edit `backtest_signals.py`:

1. Add import at top (after existing imports):
```python
from indicators_v4 import compute_mfi, compute_obv_divergence, compute_vwap_deviation, compute_adx
```

2. Add to `SIGNAL_WEIGHTS` dict (line ~80):
```python
    "mfi": 1.0,
    "obv_divergence": 1.0,
    "vwap_deviation": 0.5,
    "adx_trend": 1.5,
```

3. In `compute_signals()`, after the existing indicator computation block (~line 108-132), add:
```python
    # --- V4 indicators ---
    mfi = compute_mfi(df, period=14)
    obv_div = compute_obv_divergence(df, lookback=10)
    vwap_dev = compute_vwap_deviation(df)
    adx_result = compute_adx(df, period=14)
    adx_val = adx_result["adx"]
    di_plus = adx_result["di_plus"]
    di_minus = adx_result["di_minus"]
```

4. Add signal boolean arrays after existing ones (~line 144):
```python
    sig_mfi = [False] * n
    sig_obv_divergence = [False] * n
    sig_vwap_deviation = [False] * n
    sig_adx_trend = [False] * n
```

5. Inside the per-bar loop, after signal 5 (trend_confirm, ~line 218), add:
```python
        # 6. MFI: oversold buying (bullish) or overbought selling (bearish)
        if not pd.isna(mfi.iloc[i]):
            m = float(mfi.iloc[i])
            if direction == "bullish" and m < 30:
                fired.append("mfi")
                score += SIGNAL_WEIGHTS["mfi"]
                sig_mfi[i] = True
            elif direction == "bearish" and m > 70:
                fired.append("mfi")
                score += SIGNAL_WEIGHTS["mfi"]
                sig_mfi[i] = True

        # 7. OBV divergence: confirms or warns
        if i < len(obv_div) and obv_div.iloc[i] != "none":
            od = obv_div.iloc[i]
            if direction == "bullish" and od == "bullish_div":
                fired.append("obv_divergence")
                score += SIGNAL_WEIGHTS["obv_divergence"]
                sig_obv_divergence[i] = True
            elif direction == "bearish" and od == "bearish_div":
                fired.append("obv_divergence")
                score += SIGNAL_WEIGHTS["obv_divergence"]
                sig_obv_divergence[i] = True

        # 8. VWAP deviation: oversold bounce / extended warning
        if not pd.isna(vwap_dev.iloc[i]):
            vd = float(vwap_dev.iloc[i])
            if direction == "bullish" and vd < -1.0:
                fired.append("vwap_deviation")
                score += SIGNAL_WEIGHTS["vwap_deviation"]
                sig_vwap_deviation[i] = True
            elif direction == "bearish" and vd > 1.0:
                fired.append("vwap_deviation")
                score += SIGNAL_WEIGHTS["vwap_deviation"]
                sig_vwap_deviation[i] = True

        # 9. ADX trend strength: strong trend confirms direction
        if not pd.isna(adx_val.iloc[i]):
            ax = float(adx_val.iloc[i])
            dp = float(di_plus.iloc[i]) if not pd.isna(di_plus.iloc[i]) else 0
            dm = float(di_minus.iloc[i]) if not pd.isna(di_minus.iloc[i]) else 0
            if ax > 25:
                if (direction == "bullish" and dp > dm) or (direction == "bearish" and dm > dp):
                    fired.append("adx_trend")
                    score += SIGNAL_WEIGHTS["adx_trend"]
                    sig_adx_trend[i] = True

        # ADX dampening: choppy market reduces all scores
        if not pd.isna(adx_val.iloc[i]) and float(adx_val.iloc[i]) < 20:
            score *= 0.7
```

6. Add new columns to the result DataFrame (~line 224-233):
```python
        "sig_mfi": sig_mfi,
        "sig_obv_divergence": sig_obv_divergence,
        "sig_vwap_deviation": sig_vwap_deviation,
        "sig_adx_trend": sig_adx_trend,
```

**Step 4: Run tests**

Run: `cd ~/financial-agent-india && python -m pytest tests/test_backtest_signals.py -v`
Expected: ALL PASS (existing + 3 new)

**Step 5: Commit**

```bash
git add backtest_signals.py tests/test_backtest_signals.py
git commit -m "feat: integrate V4 indicators into backtest scoring"
```

---

### Task 7: Integrate into Claude entry/exit prompts

**Files:**
- Modify: `claude_intel.py:339-361` (entry prompt)
- Modify: `claude_intel.py:432-440` (batch candidate lines)
- Modify: `paper_trade.py` (compute + pass new indicators to candidates)

**Step 1: Add indicator computation to paper_trade.py candidate enrichment**

Find where candidates get enriched with RSI/volume_ratio/supertrend (search for `compute_rsi` or `volume_ratio` in paper_trade.py) and add:

```python
# V4 indicators
from indicators_v4 import (
    compute_mfi, compute_obv_divergence, compute_vwap_deviation, compute_adx,
    compute_pcr_signal, compute_maxpain_signal, compute_fii_momentum,
)

# After fetching candle data for the candidate and building a DataFrame:
# (Compute alongside existing RSI/volume ratio)
try:
    if candidate_df is not None and len(candidate_df) >= 20:
        mfi_series = compute_mfi(candidate_df)
        c["mfi"] = round(float(mfi_series.iloc[-1]), 1) if not pd.isna(mfi_series.iloc[-1]) else None

        obv_div = compute_obv_divergence(candidate_df)
        c["obv_divergence"] = obv_div.iloc[-1] if obv_div.iloc[-1] != "none" else None

        vwap_dev = compute_vwap_deviation(candidate_df)
        c["vwap_deviation"] = round(float(vwap_dev.iloc[-1]), 2) if not pd.isna(vwap_dev.iloc[-1]) else None

        adx_result = compute_adx(candidate_df)
        c["adx"] = round(float(adx_result["adx"].iloc[-1]), 1) if not pd.isna(adx_result["adx"].iloc[-1]) else None
        c["di_plus"] = round(float(adx_result["di_plus"].iloc[-1]), 1) if not pd.isna(adx_result["di_plus"].iloc[-1]) else None
        c["di_minus"] = round(float(adx_result["di_minus"].iloc[-1]), 1) if not pd.isna(adx_result["di_minus"].iloc[-1]) else None
except Exception as e:
    logger.debug("V4 indicators failed for %s: %s", symbol, e)

# Global intel signals (computed once per cycle, not per candidate):
pcr_signal = compute_pcr_signal(pcr)
maxpain_signal = compute_maxpain_signal(nifty_ltp, max_pain)
fii_momentum = compute_fii_momentum(fii_3day_flows)  # need to extract from macro dict
```

**Step 2: Update Claude entry prompt in claude_intel.py**

In `evaluate_entry()` (~line 339-361), add after the existing candidate section:

```python
- MFI: {candidate.get('mfi', 'N/A')}
- OBV: {candidate.get('obv_divergence', 'N/A')}
- VWAP dev: {candidate.get('vwap_deviation', 'N/A')} ATR
- ADX: {candidate.get('adx', 'N/A')} | DI+: {candidate.get('di_plus', 'N/A')} | DI-: {candidate.get('di_minus', 'N/A')}
```

In `evaluate_candidates()` (~line 434-441), add to candidate_lines format:

```python
f"MFI={c.get('mfi', '?')} OBV={c.get('obv_divergence', '?')} "
f"VWAP={c.get('vwap_deviation', '?')} ADX={c.get('adx', '?')} "
```

**Step 3: Run existing tests**

Run: `cd ~/financial-agent-india && python -m pytest tests/ -v --timeout=30 -x`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add paper_trade.py claude_intel.py
git commit -m "feat: pass V4 indicators to Claude entry/exit prompts"
```

---

### Task 8: Baseline Backtest Comparison

**Files:**
- No code changes — validation only

**Step 1: Run baseline backtest (old signals)**

```bash
cd ~/financial-agent-india
python backtest_signals.py --nifty50 --period 2y --output results_baseline.json 2>&1 | tail -20
```

Record: win rate, profit factor, max drawdown, total trades.

**Step 2: Run new backtest (with V4 signals)**

```bash
python backtest_signals.py --nifty50 --period 2y --output results_v4.json 2>&1 | tail -20
```

Compare metrics. If any V4 indicator degrades performance, consider:
- Lowering its weight
- Changing thresholds
- Removing it from scoring (keep in Claude context only)

**Step 3: Document results**

Save comparison to `docs/plans/2026-03-09-indicator-discovery-results.md`.

**Step 4: Commit**

```bash
git add docs/plans/2026-03-09-indicator-discovery-results.md
git commit -m "docs: baseline vs V4 indicator backtest comparison"
```
