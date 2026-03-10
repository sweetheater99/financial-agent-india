# F&O Backtest V1.2 Design — Real Kite Data Integration

**Date:** 2026-03-10

**Goal:** Calibrate synthetic pricing against real Kite option chains, start daily forward collection, and enable the backtest engine to use real data when available.

**Architecture:** Three parts — calibration script (one-time), daily chain collector (cron), engine swap-in (hybrid real/synthetic).

**Tech Stack:** Python, existing kite_data.py (KiteConnect), existing fo_data.py, parquet for storage.

---

## Part 1: Calibration (`fo_calibrate.py`)

**Purpose:** Compare real Kite chain against synthetic chain to validate skew_factor and spread_coeff defaults.

### Flow

1. Authenticate via `get_kite()` (requires valid access token)
2. Fetch live spot + VIX via `get_ltp_kite("NIFTY")` and `get_vix_kite()`
3. Fetch real option chain via `fetch_option_chain_kite("NIFTY")`
4. Compute DTE from nearest expiry
5. Generate synthetic chain with same spot/VIX/DTE using `generate_synthetic_chain()`
6. Compare at key strikes: ATM, ±250, ±500, ±750pt
7. Print calibration report: real vs synthetic premium, IV, bid-ask spread
8. Compute condor credit-to-width for both real and synthetic
9. Suggest adjusted skew_factor/spread_coeff if gap > 10%

### Interface

```bash
python fo_calibrate.py                    # NIFTY calibration
python fo_calibrate.py --symbol BANKNIFTY # BANKNIFTY calibration
```

### Output

```
F&O CALIBRATION REPORT — NIFTY
Spot: 23150  VIX: 13.8  DTE: 22  Expiry: 2026-04-02

Strike   Type  Real   Synth  Diff%  Real_IV  Synth_IV
23150    CE    285.4  292.1  +2.3%  13.8%    13.8%
23150    PE    270.1  278.5  +3.1%  13.8%    13.8%
22650    PE    95.2   110.8  +16.4% 17.1%    15.5%
23650    CE    88.7   98.2   +10.7% 14.9%    14.3%
...

Condor (500pt OTM, 300pt wings):
  Real credit-to-width:      28.3%
  Synthetic credit-to-width: 31.5%
  Gap: 3.2% — ACCEPTABLE (< 10%)

Current: skew_factor=8.0, spread_coeff=3.0
Recommendation: No change needed
```

---

## Part 2: Daily Chain Collector (`fo_chain_collector.py`)

**Purpose:** Snapshot real option chains daily at close for future backtesting.

### What it collects

Per symbol (NIFTY, BANKNIFTY):
- Full option chain for nearest 2 expiries (weekly + monthly)
- Spot price, VIX
- All strikes with premium > 5

### Storage

- Path: `data/fo_chains/{SYMBOL}_{YYYYMMDD}.parquet`
- Schema (matches `generate_synthetic_chain()` output + extras):

```
strike       float64   — strike price
option_type  str       — CE/PE
premium      float64   — last traded price
bid          float64   — best bid
ask          float64   — best ask
oi           int64     — open interest
volume       int64     — volume
iv           float64   — implied volatility (computed via BS inversion)
delta        float64   — computed from greeks.py
gamma        float64
theta        float64
vega         float64
expiry       str       — expiry date
spot         float64   — spot at snapshot time
vix          float64   — VIX at snapshot time
```

### Interface

```bash
python fo_chain_collector.py                       # Collect NIFTY + BANKNIFTY
python fo_chain_collector.py --symbol NIFTY        # NIFTY only
python fo_chain_collector.py --dry-run              # Show what would be collected
```

### Cron (Pi)

```
25 15 * * 1-5  cd ~/financial-agent-india && /path/to/python fo_chain_collector.py >> data/fo_chains/collector.log 2>&1
```

### IV Computation

Real chains from Kite don't include IV. Compute it by inverting BS:
- Use `greeks.py` bisection: find IV where `BS_price(spot, strike, dte, r, iv) ≈ market_premium`
- Store computed IV in the parquet for consistency with synthetic chain schema

---

## Part 3: Engine Swap-In

**Purpose:** Let the backtest engine use real chains when available, fall back to synthetic.

### Changes to `fo_data.py`

```python
def fetch_real_chain(symbol: str, date: datetime.date) -> pd.DataFrame | None:
    """Load real option chain snapshot for a specific date. Returns None if not available."""
    path = f"data/fo_chains/{symbol}_{date.strftime('%Y%m%d')}.parquet"
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None
```

### Changes to `fo_backtest.py`

In the daily loop where `generate_synthetic_chain()` is called, add:

```python
# Prefer real chain if available
chain = fetch_real_chain(symbol, current_date)
if chain is None:
    chain = generate_synthetic_chain(spot, vix, dte, symbol)
```

### Backward Compatibility

- If no real data exists (historical dates), synthetic chain is used — same as V1.1
- As daily collection accumulates, more dates get real data
- Mixed mode is fine — engine logs which dates used real vs synthetic

---

## Files Summary

| File | Action | Est. Lines |
|------|--------|------------|
| `fo_calibrate.py` | Create | ~100 |
| `fo_chain_collector.py` | Create | ~120 |
| `fo_data.py` | Modify — add `fetch_real_chain()` | +15 |
| `fo_backtest.py` | Modify — prefer real chain | +5 |
| `tests/test_fo_calibrate.py` | Create | ~50 |
| `tests/test_fo_chain_collector.py` | Create | ~60 |

---

## Constraints

- **Kite cannot fetch expired option data** — only active contracts. This is why we need forward collection.
- **Kite access token expires daily** — cron must run after daily `kite_auth.py` login or token refresh.
- **API rate limit** — `KITE_API_DELAY = 0.35s` between calls. ~240 instruments per symbol = ~2 quote batches = ~1 second per symbol.
- **IV inversion** — bisection method has edge cases (deep ITM/OTM). Fallback to VIX if inversion fails.
