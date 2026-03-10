# F&O Backtest Engine V1.1 Design

**Date:** 2026-03-10

**Goal:** Fix synthetic pricing accuracy (volatility skew) and add parameter optimization (grid sweep) to produce realistic backtest results and optimal strategy configurations.

**Architecture:** Two enhancements layered on V1.0 — skew formula in data layer, sweep script as standalone orchestrator.

**Tech Stack:** Python, existing fo_data/fo_strategies/fo_backtest modules, itertools for grid generation.

---

## Enhancement 1: Volatility Skew

**Problem:** V1.0 uses flat IV (VIX/100) for all strikes. This overprices OTM options, inflating condor credits beyond wing width — a pricing artifact impossible in real markets.

**Solution:** Strike-dependent IV in `generate_synthetic_chain()`.

### Formula

```python
base_iv = vix / 100.0

# OTM puts get higher IV (put skew — demand for downside protection)
if option_type == "PE" and strike < spot:
    moneyness = (spot - strike) / spot
    iv = base_iv * (1 + skew_factor * moneyness)

# OTM calls get mildly higher IV (smile effect, 30% of put skew)
elif option_type == "CE" and strike > spot:
    moneyness = (strike - spot) / spot
    iv = base_iv * (1 + skew_factor * 0.3 * moneyness)

# ATM stays at base_iv
else:
    iv = base_iv
```

### Parameters

| Param | Default | Range | Effect |
|-------|---------|-------|--------|
| `skew_factor` | 0.8 | 0.3–1.5 | Controls steepness of put skew |
| Call skew ratio | 0.3 (hardcoded) | — | Calls get 30% of put skew |

### Calibration

With `skew_factor=0.8`, spot=23000, VIX=14 (base_iv=0.14):
- ATM (23000): IV = 14.0%
- 500pt OTM put (22500): moneyness = 0.0217, IV = 14.0% × (1 + 0.8 × 0.0217) = 14.24%
- 1000pt OTM put (22000): moneyness = 0.0435, IV = 14.0% × (1 + 0.8 × 0.0435) = 14.49%
- 500pt OTM call (23500): moneyness = 0.0217, IV = 14.0% × (1 + 0.8 × 0.3 × 0.0217) = 14.07%

This produces a realistic ~2-3% IV bump for OTM puts at typical condor distances.

### Files Changed

- `fo_data.py`: Add `skew_factor` param to `generate_synthetic_chain()`, apply per-strike IV
- `tests/test_fo_data.py`: Add skew tests (OTM put IV > ATM IV > OTM call IV ordering)
- No changes to strategies or engine — they consume chain DataFrames unchanged

---

## Enhancement 2: Parameter Sweep

**Problem:** V1.0 uses hardcoded strategy params. Need to find optimal configurations across the 2-year backtest period.

**Solution:** Grid search over key params per strategy, isolated single-strategy runs.

### Sweep Grids

**Futures:**
| Param | Values |
|-------|--------|
| score_threshold | 3.0, 3.5, 4.0 |
| target_atr_mult | 1.0, 1.5, 2.0 |
| sl_atr_mult | 2.5, 3.5, 4.5 |
| **Total** | **27 combos** |

**Spread:**
| Param | Values |
|-------|--------|
| dte_min | 20, 30, 40 |
| profit_cap_pct | 0.6, 0.8, 1.0 |
| sl_multiplier | 1.5, 2.0, 3.0 |
| **Total** | **27 combos** |

**Condor:**
| Param | Values |
|-------|--------|
| min_vix | 10, 12, 14 |
| target_pct | 0.3, 0.5, 0.7 |
| sl_multiplier | 1.5, 2.0, 3.0 |
| **Total** | **27 combos** |

**Momentum:**
| Param | Values |
|-------|--------|
| score_threshold | 4.0, 5.0, 6.0 |
| sl_pct | 0.25, 0.35, 0.50 |
| target_pct | 0.6, 0.9, 1.2 |
| **Total** | **27 combos** |

### Design Choices

1. **Single-strategy isolation:** Each sweep run activates only one strategy. This isolates param effects without cross-strategy interference.

2. **Ranking metric:** Primary = Sharpe ratio (risk-adjusted). Secondary = profit factor. Minimum 10 trades required to rank.

3. **Final validation:** After finding optimal params per strategy, one combined run with all four strategies validates they work together.

### Interface

```bash
# Sweep one strategy
python fo_param_sweep.py --strategy futures --symbol NIFTY --period 2y

# Sweep all strategies sequentially
python fo_param_sweep.py --strategy all --symbol NIFTY --period 2y

# Custom capital
python fo_param_sweep.py --strategy condor --symbol NIFTY --period 2y --capital 500000
```

### Output

`data/fo_backtest/sweep_results_YYYYMMDD.json`:
```json
{
  "sweep_date": "2026-03-10",
  "symbol": "NIFTY",
  "period": "2y",
  "strategies": {
    "futures": {
      "total_combos": 27,
      "top_5": [
        {
          "params": {"score_threshold": 4.0, "target_atr_mult": 1.5, "sl_atr_mult": 3.5},
          "trades": 42,
          "win_rate": 0.67,
          "sharpe": 1.82,
          "profit_factor": 2.1,
          "total_pnl": 85000,
          "max_drawdown": -0.12
        }
      ]
    }
  }
}
```

Console prints top 3 per strategy with key metrics during sweep.

### Files Created

- `fo_param_sweep.py`: Sweep orchestrator (~200 lines)
- `tests/test_fo_param_sweep.py`: Sweep smoke tests (~80 lines)

---

## Files Summary

| File | Action | Est. Lines |
|------|--------|------------|
| `fo_data.py` | Modify — add skew to `generate_synthetic_chain()` | +15 |
| `tests/test_fo_data.py` | Modify — add skew tests | +30 |
| `fo_param_sweep.py` | Create — sweep orchestrator | ~200 |
| `tests/test_fo_param_sweep.py` | Create — sweep smoke tests | ~80 |
