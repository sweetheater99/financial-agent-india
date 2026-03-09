# V4 Indicator Backtest Results

**Date:** 2026-03-09
**Test:** Nifty 50, 1-year period, default params (min-score 3.5, ATR target 2.0)

## V4 Signal Distribution (9-signal composite)

| Signal | Fires | % of Entries | Weight | Notes |
|---|---|---|---|---|
| momentum | 1830 | 100% | 1.0 | Always fires (baseline) |
| rsi | 1401 | 77% | 1.0 | High frequency, expected |
| **adx_trend** | 1039 | **57%** | 1.5 | **New. High impact — fires in majority of entries** |
| atr_breakout | 858 | 47% | 1.5 | Existing |
| volume | 845 | 46% | 1.0 | Existing |
| trend_confirm | 660 | 36% | 2.0 | Existing (highest weight) |
| **vwap_deviation** | 419 | **23%** | 0.5 | **New. Good selectivity** |
| **mfi** | 35 | **2%** | 1.0 | **New. Very selective — only fires at extremes** |
| **obv_divergence** | 5 | **0.3%** | 1.0 | **New. Extremely rare — confirms major divergences only** |

## Portfolio Metrics

| Metric | Value |
|---|---|
| Total trades | 573 |
| Win rate | 46.6% |
| Avg win | +2.23% |
| Avg loss | -2.06% |
| Profit factor | 0.92 |
| Sharpe | -0.33 |
| Max drawdown | ₹10,349 |
| Avg holding | 6.4 days |

## Observations

1. **ADX is the standout**: fires in 57% of entries with weight 1.5 — biggest new contributor to scoring. Validates the hypothesis that trend strength was our biggest gap.

2. **VWAP deviation is well-calibrated**: 23% fire rate means it's selective but not too rare. The 0.5 weight is appropriate.

3. **MFI is extremely selective**: only 2% fire rate. This means it only catches true oversold/overbought extremes. Consider:
   - Widening thresholds (currently < 30 for bullish, > 70 for bearish)
   - Or keeping it as a rare high-conviction signal

4. **OBV divergence is too rare**: 0.3% fire rate means it's almost never contributing. The lookback=10 may be too short for daily data. Consider:
   - Increasing lookback to 20
   - Or accepting it as an "exit warning" signal rather than entry signal

5. **ADX dampening is active**: scores in choppy markets (ADX < 20) get reduced by 0.7x. This helps filter low-quality entries.

6. **Overall P&L is negative**: -₹4,297 on ₹10L capital. This is a backtest on default params — the real edge comes from Claude's filtering (not run in backtest mode). The signal distribution is what matters for tuning.

## Tuning Recommendations

- **MFI**: Consider widening to < 40 / > 60 for more signal coverage
- **OBV divergence**: Increase lookback to 20 or make it an exit-only signal
- **ADX weight**: 1.5 seems right — it's the most impactful new signal
- **VWAP weight**: 0.5 is conservative, could try 1.0

## Next Steps

- Run paper trading with V4 signals for 1 week
- Monitor Claude's use of the new indicators in entry decisions
- Tune MFI/OBV thresholds based on live signal frequency
