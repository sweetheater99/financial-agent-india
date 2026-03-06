# Design: Bear Put Spread + YouTube Intel Integration

Date: 2026-03-05
Status: Approved

## Context

The paper trading bot has never opened a single bearish position despite having put-buying code. Research across 20+ YouTube videos, SEBI data, and broker sources unanimously shows:

- 93% of individual F&O traders lose money (SEBI FY22-24)
- Naked put buying has ~20-30% win rate for swing trades
- Theta decay eats 15-35% of premium over 7-day holds
- Bear put spreads are the only viable bearish strategy at 1L capital

## Part 1: Bear Put Spread Strategy

### Replace naked put buying with bear put spreads

**What is a bear put spread:**
- Buy 1 ATM put (higher strike)
- Sell 1 OTM put (lower strike, same expiry, same lot)
- Net debit = max loss. Profit capped at strike width minus debit.

**Parameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Long leg | ATM or 1-strike ITM put | Higher delta, better directionality |
| Short leg | 2-4 strikes OTM from long leg | Balances cost reduction vs profit cap |
| Spread width | Stock-dependent (see selection logic) | Target R:R of 1:2 minimum |
| Expiry | Next month (30-45 DTE minimum) | Kills theta decay |
| Entry trigger | Screener bearish signal (ShortBuildUp, LongUnwinding, etc.) | Same as current |
| Exit: target | Underlying drops by 2.0x ATR | ATR-based, mirrors equity logic |
| Exit: stoploss | Underlying rises by 1.5x ATR | ATR-based, mirrors equity logic |
| Exit: time | 7 trading days OR DTE < 7 | Whichever comes first |
| Exit: profit cap | Spread reaches 80% of max profit | Don't hold for last 20% |
| Max risk per trade | 10k (debit paid) | Keeps 2-3 spreads affordable |
| Max concurrent | 2-3 bear spreads | Alongside 3-4 equity longs |

**Strike selection logic:**
1. Fetch option chain for the symbol + next month expiry
2. Long leg: ATM strike (closest to spot) or 1 strike ITM
3. Short leg: Pick OTM strike such that:
   - Spread width >= 1.5x ATR (enough room for profit)
   - Net debit <= allocation budget
   - Short leg has OI > 1000 (liquidity check)
   - R:R >= 1:2 (max_profit / debit >= 2)
4. If no valid spread found, skip the trade

**Capital allocation:**
```
1,00,000 total
  Max 60% equity longs:       ~60k (3-4 positions @ 15-20k)
  Max 30% bear put spreads:   ~30k (2-3 spreads @ 5-10k)
  Min 10% cash reserve:       ~10k (always)
```

**VIX guard:**
- VIX > 25: Skip new bearish spreads (premiums too expensive, whipsaw risk)
- VIX 20-25: Allow but reduce allocation by 50%
- VIX < 20: Normal allocation

### Position tracking

```python
position = {
    "symbol": "RELIANCE",
    "instrument": "SPREAD",
    "spread_type": "BEAR_PUT",
    "direction": "bearish",
    "long_leg": {
        "strike": 1300,
        "option_type": "PE",
        "token": "...",
        "contract_symbol": "RELIANCE27MAR26PE1300",
        "entry_premium": 45.0,
    },
    "short_leg": {
        "strike": 1260,
        "option_type": "PE",
        "token": "...",
        "contract_symbol": "RELIANCE27MAR26PE1260",
        "entry_premium": 28.0,
    },
    "spread_width": 40,        # strike difference
    "net_debit": 17.0,         # per unit
    "lot_size": 250,
    "num_lots": 1,
    "quantity": 250,
    "allocated": 4250.0,       # net_debit * quantity (= max loss)
    "max_profit": 5750.0,      # (spread_width - net_debit) * quantity
    "underlying_at_entry": 1305.0,
    "atr_at_entry": 30.0,
    "expiry": "26MAR2026",
    "entry_date": "2026-03-05",
    "max_hold_date": "2026-03-14",
    "status": "open",
}
```

### Exit logic

Monitor the underlying stock price (not option premiums):
- **Target hit**: underlying <= entry - 2.0 * ATR -> close both legs
- **Stoploss hit**: underlying >= entry + 1.5 * ATR -> close both legs
- **Profit cap**: spread value >= 80% of max profit -> close (diminishing returns)
- **Time exit**: 7 trading days or DTE < 7 -> close
- **Close both legs simultaneously** — sell long put, buy back short put

P&L = (long_leg_exit - long_leg_entry) - (short_leg_exit - short_leg_entry) * quantity
     minus transaction costs for 4 legs (2 entry + 2 exit)

### Transaction costs (4 legs)

Each leg: brokerage (20) + STT + exchange + GST
Total per round trip: ~160-200 for the spread
Factor this into the minimum spread width — don't open spreads where costs > 5% of debit.

---

## Part 2: YouTube Intel Integration

### Mode A: Daily Market Intel (pre-market)

**When:** 8:45 AM IST, before screener runs
**What:** Search YouTube for market outlook published in last 24h

**Process:**
1. Search: "Nifty analysis today", "market outlook India today"
2. Filter: videos from last 24h, prefer known channels
3. Pull transcript from top 2-3 results
4. Claude (haiku) classifies into:
```json
{
    "market_bias": "bullish | bearish | sideways",
    "key_levels": {"nifty_support": 22000, "nifty_resistance": 22500},
    "sectors_bullish": ["Banking", "IT"],
    "sectors_bearish": ["Auto", "Pharma"],
    "events_today": ["RBI policy", "US Fed minutes"],
    "confidence": "high | medium | low",
    "summary": "..."
}
```
5. Feed into regime classification as additional signal
6. Cache for 12 hours

**Channel allowlist (high quality, frequent uploads):**
- P R Sundar, CA Rachana Ranade, Power of Stocks
- Pranjal Kamra, Nitin Bhatia, Market Guru
- Configurable in config.py

### Mode B: Pre-Trade Stock Research (before opening positions)

**When:** After screener picks candidates, before opening any position
**What:** Search YouTube for recent analysis on that specific stock

**Process:**
1. Search: "{SYMBOL} stock analysis" (last 7 days)
2. Pull transcript from top 1-2 results (if available)
3. Claude (haiku) extracts:
```json
{
    "sentiment": "bullish | bearish | neutral",
    "key_levels": {"support": 1250, "resistance": 1350},
    "red_flags": ["earnings next week", "promoter selling"],
    "catalyst": "sector rotation into pharma",
    "confidence": "high | medium | low"
}
```
4. Integration with entry filters:
   - YouTube strongly contradicts signal -> skip or reduce size by 50%
   - YouTube confirms signal -> no change (don't over-weight)
   - No YouTube data found -> proceed normally (most stocks won't have recent videos)
5. Cache for 24 hours per symbol

### Implementation

New file: `youtube_intel.py`
- `fetch_market_intel() -> dict` — Mode A
- `fetch_stock_intel(symbol: str) -> dict | None` — Mode B
- Uses yt-dlp for search + youtube-transcript-api for transcripts
- Claude haiku for classification
- File-based cache (data/youtube_cache/)

Wire into:
- `screener.py` — call fetch_market_intel() in morning run, include in Claude briefing prompt
- `paper_trade.py` — call fetch_stock_intel() in open_positions() loop, add as entry filter

### Dependencies
- yt-dlp (already installed on Pi)
- youtube-transcript-api (already installed on Pi)
- No new API keys needed

### Cost estimate
- ~2-5 haiku calls per morning (market intel)
- ~1 haiku call per candidate stock (stock intel, 5-10 per day)
- Total: ~10-15 haiku calls/day = negligible cost

---

## Files Changed

| File | Change |
|------|--------|
| `paper_trade.py` | Replace `_open_put_position()` with `_open_bear_put_spread()`, new spread tracking, exit logic for spreads, capital allocation constants |
| `agent_with_options.py` | Update `select_put_strike()` to `select_spread_strikes()` with delta/OI/R:R logic |
| `youtube_intel.py` | **New** — market intel + stock intel + caching |
| `screener.py` | Wire YouTube market intel into morning briefing |
| `config.py` | YouTube channel allowlist, cache TTLs, spread constants |
| `tests/test_paper_trade.py` | Update tests for spread logic, add spread-specific tests |
| `tests/test_youtube_intel.py` | **New** — tests for YouTube intel module |

## Risk / Unknowns

1. **Spread margin for live trading**: Paper trading ignores margin. For live, need to verify AngelOne's margin calculator accepts spread orders. May need to place as two separate orders.
2. **Spread liquidity on stock options**: OTM puts on individual stocks may have low OI. Need the liquidity check (OI > 1000) to be strict.
3. **YouTube transcript availability**: Many Indian finance videos are in Hindi without English transcripts. May need to handle Hindi transcripts or skip gracefully.
4. **YouTube rate limiting**: yt-dlp can get rate limited. Cache aggressively, retry with backoff.
