# Indicator Discovery Module — Design

**Date:** 2026-03-09
**Goal:** Add 7 new technical indicators to fill gaps in the financial agent's signal coverage, improving entry/exit quality.

## Motivation

Current signals: RSI, volume ratio, EMA, ATR breakout, Supertrend, CPR. Gaps in volume-price analysis, trend strength, and quantified use of already-fetched India-specific data (PCR, max pain, FII/DII).

## Indicators

### OHLCV-based (backtestable)

| Indicator | Function | Output | Scoring weight | Signal logic |
|---|---|---|---|---|
| MFI(14) | `compute_mfi(df, period=14)` | 0-100 | 1.0 | Bullish: MFI < 30 + price rising. Bearish: MFI > 70 + price falling |
| OBV divergence | `compute_obv_divergence(df, lookback=10)` | `bullish_div` / `bearish_div` / `none` | 1.0 | Price new high + OBV not = bearish div (exit warning). Price new low + OBV not = bullish div (entry boost) |
| VWAP deviation | `compute_vwap_deviation(df)` | % of ATR from VWAP | 0.5 | < -1 ATR = oversold bounce candidate. > +1 ATR = extended, avoid |
| ADX + DI+/DI- | `compute_adx(df, period=14)` | ADX float + DI+/DI- | 1.5 | ADX > 25 = strong trend (size up, trust signals). ADX < 20 = choppy (reduce all signal weights) |

### Global intel (live only, not backtestable)

| Indicator | Function | Output | Signal logic |
|---|---|---|---|
| PCR signal | `compute_pcr_signal(pcr)` | -1 to +1 | PCR > 1.3 = bullish, < 0.7 = bearish, between = neutral |
| Max Pain proximity | `compute_maxpain_signal(price, max_pain)` | -1 to +1 | Within 1% = mean reversion zone (caution on directional entries) |
| FII flow momentum | `compute_fii_momentum(flows_3d)` | -1 to +1 | 3-day net buy trend = bullish context, net sell = bearish |

## Architecture

```
indicators_v4.py              ← NEW: 7 indicator functions
    ↓
backtest_signals.py           ← EDIT: add 4 OHLCV signals to compute_signals()
    ↓
paper_trade.py                ← EDIT: pass all 7 values in Claude context prompt
```

### indicators_v4.py

Pure functions, same interface as indicators.py/indicators_v3.py. Each takes a pandas DataFrame (OHLCV) or scalar values, returns computed result. No side effects, no API calls.

### backtest_signals.py changes

- Import 4 new functions from indicators_v4
- Add to `SIGNAL_WEIGHTS`: mfi (1.0), obv_divergence (1.0), vwap_deviation (0.5), adx_trend (1.5)
- Add signal computation in `compute_signals()` loop
- ADX modulates other signals: if ADX < 20, multiply total score by 0.7

### paper_trade.py changes

- Import all 7 functions
- Compute OHLCV indicators from candle data already fetched
- Compute global intel signals from data already in context (PCR, max pain, FII)
- Add "Additional Signals" section to Claude's entry/exit prompt

## Validation

1. Backtest Nifty 50, 2-year period with old signals (baseline)
2. Backtest same with new 9-signal composite
3. Compare: win rate, profit factor, max drawdown, Sharpe proxy
4. Drop any indicator that degrades the composite
5. Run paper trading for 1 week with new signals before trusting weights

## Non-goals

- No TradingView scraping or Pine Script conversion
- No indicator factory automation
- No new dependencies (all indicators computed with pandas/numpy)
- No changes to risk guardrails, safety layer, or exit mechanics
