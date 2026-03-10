# F&O Backtest Engine Design

**Date:** 2026-03-10

**Goal:** Build a layered F&O backtesting engine that validates and optimizes all four trading strategies (spreads, condors, momentum options, futures) using synthetic Black-Scholes pricing on 2 years of historical data, with a path to swap in real AngelOne candle data.

**Architecture:** Three-layer system — Data (synthetic option chains) → Strategies (4 strategy classes) → Engine (daily simulation + portfolio tracking). Param sweep built in.

**Tech Stack:** Python, pandas, numpy, yfinance (spot/VIX history), existing `greeks.py` (Black-Scholes), existing `backtest_signals.py` (V4 signals)

---

## Layer 1: Data (`fo_data.py`)

**Purpose:** Provide option chains and futures prices for any symbol on any historical date.

### Synthetic Mode (default)

- Spot OHLCV + India VIX from yfinance (^NSEI, ^NSEBANK, ^INDIAVIX) — 2 years
- Generate strikes at 50pt intervals (Nifty) / 100pt (BankNifty) around spot
- Price each strike using `greeks.black_scholes_greeks()` with VIX as IV proxy
- Futures price: `spot * e^(r * dte/365)`, r = 6.5%

### Lot Size Lookup (date-aware)

| Symbol | Pre-Dec 2024 | Dec 2024 - Nov 2025 | Dec 2025+ |
|--------|---|---|---|
| Nifty | 50 | 75 | 65 |
| BankNifty | 25 | 30 | 28 |

### Expiry Calendar

- Pre-computed monthly + weekly expiries
- Thursday expiry pre-Sep 2025, Tuesday post-Sep 2025
- Handles holidays (NSE holiday list)

### Interface

```python
def get_option_chain(symbol: str, date: datetime.date, spot: float, vix: float, risk_free: float = 0.065) -> pd.DataFrame
# Returns: strike, option_type(CE/PE), premium, delta, gamma, theta, vega, iv, dte

def get_futures_price(symbol: str, date: datetime.date, spot: float, dte: int, risk_free: float = 0.065) -> float

def get_lot_size(symbol: str, date: datetime.date) -> int

def get_nearest_expiry(date: datetime.date, min_dte: int = 0, weekly: bool = False) -> datetime.date
```

### Real Mode (future swap-in)

- Replace synthetic pricing with AngelOne `getCandleData(exchange="NFO")` for active contracts
- Same interface — engine and strategies don't change

### Caching

- Parquet files in `data/fo_backtest/`
- Spot/VIX history cached on first fetch

---

## Layer 2: Strategies (`fo_strategies.py`)

Four strategy classes with common interface:

```python
class BaseStrategy:
    def should_enter(self, date, spot, vix, signals, chain, portfolio) -> dict | None
    def should_exit(self, position, date, spot, chain) -> tuple[bool, str]
    def compute_pnl(self, position, exit_prices, costs) -> float
```

### A. FuturesStrategy

- **Entry:** V4 composite score > threshold, direction from signal consensus
- **Sizing:** Risk-based — max 2% capital loss per trade, 15% margin (FUT_MARGIN_PCT)
- **Target:** entry +/- 1.5x ATR (FUT_TARGET_ATR_MULT)
- **Stop-loss:** entry -/+ 3.5x ATR (FUT_SL_ATR_MULT)
- **Exit triggers:** target, SL, trailing stop, max hold days
- **P&L:** `(exit - entry) * lot_size * num_lots - costs`

### B. SpreadStrategy

- **Entry:** Score > threshold, DTE 30-45, direction from signals
- **Strike selection:** Long leg ATM/near-ATM, short leg OTM by 1-2x ATR
- **Max risk:** Net debit (debit) or width - credit (credit)
- **Exit triggers:** 80% max profit, 2x credit SL, 5 days pre-expiry
- **P&L:** Leg-by-leg premium change, daily greeks recomputation

### C. CondorStrategy

- **Entry:** VIX 12-18, monthly expiry, delta 0.25 short legs
- **Structure:** Nifty 250pt OTM shorts, 100-200pt protection wings
- **Exit triggers:** 50% max credit, 2x credit SL, delta > 0.30 any leg, 5 days pre-expiry
- **P&L:** 4-leg net premium tracking

### D. MomentumStrategy

- **Entry:** High-conviction signal (score > 5.0), ATM options, DTE 7-14
- **Max risk:** 1% capital per trade
- **Exit triggers:** +90% target, -35% SL, 3-day max hold
- **P&L:** Option premium change

### Common Features

- Daily greeks recomputation (theta decay, delta shift as spot moves)
- Auto square-off on expiry day (avoid STT exercise trap — 0.125% on notional)
- Date-aware transaction costs
- Liquidity filter: skip strikes with premium < 5

---

## Layer 3: Engine (`fo_backtest.py`)

### Main Loop (daily)

```
for each trading day in 2-year range:
    1. Load spot, VIX for the day
    2. Compute V4 signals (reuse backtest_signals.py logic)
    3. Generate synthetic option chain
    4. CHECK EXITS: for each open position, evaluate exit triggers
       - Target hit, SL hit, trailing stop, expiry approaching, theta decay
       - Gap handling: if overnight gap > expected move, SL at open (not SL price)
    5. CHECK ENTRIES: for each strategy, call should_enter()
       - Respect allocation caps, position limits, margin availability
    6. Record: daily NAV, margin utilization, position count
```

### Portfolio Constraints (from config.py)

- Max 8 concurrent positions (MAX_CONCURRENT_POSITIONS)
- Max 2 same-sector (MAX_SAME_SECTOR)
- 20% cash reserve (CASH_RESERVE_PCT)
- Strategy caps: spreads 35%, condors 15%, momentum 15%, futures 15%
- Weekly portfolio SL: 4% (WEEKLY_PORTFOLIO_SL_PCT)
- Max simultaneous open loss: 20% (MAX_SIMULTANEOUS_LOSS_PCT)

### Anti-Bias Guards

- **No look-ahead:** Signals computed from data available at bar open only
- **No survivorship bias:** Use index constituents (Nifty 50 is rebalanced but stable)
- **Auto square-off ITM before expiry:** Avoid exercise STT trap
- **Realistic fills:** Gap-adjusted SL (open price, not SL level) when overnight gap > expected move
- **Entry on next bar:** Signal on day N, entry on day N+1 open

### Entry Signal Modes

- **Mechanical:** V4 composite score > configurable threshold (default 3.5)
- **High-conviction:** Score > 5.0 (simulates Claude's filtering)
- Both modes configurable via param sweep

---

## Transaction Costs (date-aware)

| Cost | Pre-Apr 2025 | Apr 2025-Mar 2026 | Apr 2026+ |
|------|---|---|---|
| Options STT (sell) | 0.0625% | 0.1% | 0.15% |
| Futures STT (sell) | 0.0125% | 0.02% | 0.05% |
| Exercise STT (ITM expiry) | 0.125% | 0.125% | 0.125% |
| Brokerage | 20/order | 20/order | 20/order |

Plus: exchange charges, stamp duty, SEBI fees, 18% GST — all from existing config constants.

---

## Reporting & Param Sweep

### Metrics (per strategy + aggregate)

- Total P&L, win rate, profit factor, Sharpe ratio, max drawdown, Calmar ratio
- Avg theta collected per day (credit strategies)
- Avg DTE at entry and exit
- Margin utilization %
- Monthly P&L breakdown
- Strategy-level P&L attribution

### Param Sweep

- Independent sweep per strategy
- Key params per strategy:
  - **Futures:** score threshold, ATR target/SL multipliers, max hold days
  - **Spreads:** DTE range, ATR distance, profit cap %, SL multiplier
  - **Condors:** VIX range, delta targets, credit target %, SL multiplier
  - **Momentum:** score threshold, SL %, target %, max hold days
- Output: `data/fo_backtest/sweep_results_YYYYMMDD.json`

### Output Files

- `data/fo_backtest/results_YYYYMMDD.json` — full backtest results
- `data/fo_backtest/trades_YYYYMMDD.csv` — trade log
- `data/fo_backtest/daily_nav_YYYYMMDD.csv` — daily portfolio NAV

---

## Files to Create

| File | Purpose | Est. Lines |
|------|---------|------------|
| `fo_data.py` | Data layer — synthetic chains, lot sizes, expiry calendar | ~300 |
| `fo_strategies.py` | 4 strategy classes with entry/exit/P&L logic | ~500 |
| `fo_backtest.py` | Engine, portfolio tracking, reporting, param sweep | ~400 |
| `tests/test_fo_data.py` | Synthetic pricing, lot sizes, expiry tests | ~200 |
| `tests/test_fo_strategies.py` | Strategy entry/exit logic tests | ~300 |
| `tests/test_fo_backtest.py` | Integration tests | ~150 |

---

## V1.1 Enhancements (after V1 validated)

- **Volatility skew:** `IV(strike) = VIX * (1 + 0.1 * (ATM - strike) / ATM)` for puts
- **Physical settlement handling:** For stock F&O (not index)
- **Real AngelOne data swap-in:** Replace synthetic with actual candle data
- **Expiry day shift:** Thursday → Tuesday transition (Sep 2025)
- **Stock-specific IV:** Per-stock IV instead of VIX proxy
