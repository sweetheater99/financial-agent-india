# Indian F&O Trading Bot - Research Report
**Date**: March 6, 2026
**Purpose**: Research for paper trading bot with potential commercialization

---

## 1. F&O Taxation in India (2025-2026)

### STT Rates (Effective April 1, 2026 - Budget 2026)

| Instrument | Transaction | Old Rate | New Rate (Apr 2026) |
|------------|------------|----------|---------------------|
| Futures | Sell side | 0.02% | **0.05%** |
| Options (premium) | Sell side | 0.10% | **0.15%** |
| Options (exercised) | Buy side | 0.125% | **0.15%** |

This is the largest STT hike in 20 years, explicitly aimed at curbing retail speculation. SEBI data shows 90-93% of individual F&O traders incur losses.

**Note**: The October 2024 Budget had already hiked STT (futures: 0.0125% -> 0.02%, options: 0.0625% -> 0.10%), so this is the second hike in two years.

### Income Tax Treatment

- F&O income is classified as **non-speculative business income** (not capital gains, not speculative).
- File using **ITR-3** (mandatory for F&O traders).
- Taxed at applicable slab rates (no special rate like LTCG/STCG).
- Losses can be carried forward for 8 years and set off against any business income (not salary).
- F&O losses CANNOT be set off against salary or capital gains income.

### Tax Audit Requirements (Section 44AB)

| Condition | Audit Required? |
|-----------|----------------|
| Turnover > Rs 10 crore (95%+ digital transactions) | Yes |
| Turnover > Rs 1 crore (cash > 5%) | Yes |
| Turnover < Rs 2 crore, profit >= 6% of turnover (Section 44AD) | No (presumptive) |
| Turnover < Rs 3 crore (95%+ digital), profit >= 6% | No (presumptive) |
| Any turnover + showing a loss + not under 44AD | Yes, if turnover > threshold |

**F&O Turnover Calculation**: Sum of absolute values of profit and loss from each squared-off trade. This is NOT the total traded value -- it's |profit| + |loss| from each trade.

Filing deadline for tax audit: **September 30** of the assessment year.

### GST on Brokerage

- **18% GST** on brokerage, exchange transaction charges, SEBI charges, and IPFT charges.
- GST is NOT applied on STT or stamp duty (those are separate taxes).
- Example: Rs 100 brokerage + Rs 20 exchange charges = Rs 21.60 GST.

### Stamp Duty

| Instrument | Rate | Side |
|------------|------|------|
| Futures | 0.002% (Rs 200/crore) | Buy side |
| Options | 0.003% (Rs 300/crore) | Buy side |

---

## 2. SEBI F&O Regulation Changes (2024-2025)

### Weekly Expiry Restrictions (Effective November 20, 2024)

- Each exchange can only offer **one benchmark index** for weekly expiry.
- **NSE**: Only Nifty 50 weekly expiry retained. Bank Nifty, Nifty Financial Services, Nifty Midcap Select, Nifty Next 50 weeklies **discontinued**.
- **BSE**: Only Sensex weekly expiry retained.
- Monthly expiries for all indices continue as before.

### Lot Size Increases (Effective November 21, 2024+)

SEBI mandated contract value between **Rs 15-20 lakhs** (up from Rs 5-10 lakhs).

| Index | Old Lot Size | New Lot Size | Effective Date |
|-------|-------------|-------------|----------------|
| Nifty 50 | 25 | **75** | Dec 26, 2024 (monthly), Jan 2, 2025 (weekly) |
| Bank Nifty | 15 | **30** | Dec 24, 2024 (monthly), Jan 2025 (weekly) |
| Sensex | 10 | **20** | Jan 7, 2025 (weekly) |

**Impact**: Margin for Nifty 50 jumped from ~Rs 73,200 (lot of 25) to ~Rs 2,34,000 (lot of 75).

### Margin Requirement Changes

1. **2% Extreme Loss Margin (ELM)** on short index options on expiry day (effective Nov 20, 2024). Increases cost of selling options on expiry.
2. **No calendar spread margin benefit** on expiry day for contracts expiring that day (effective Feb 10, 2025).
3. **Full upfront premium** required for option buyers (no leverage).

### Position Limit Monitoring

- Exchanges monitor intraday position limits **multiple times** throughout the day (not just end-of-day). Effective April 1, 2025.

### Lot Size Revisions for Stock F&O (January 2026)

NSE revised lot sizes for Bank Nifty and Nifty Midcap Select derivatives again in March 2025, effective from new contract series. Existing contracts are NOT affected -- old lot sizes continue until those contracts expire.

---

## 3. AngelOne SmartAPI Reliability

### Rate Limits

| API Type | Limit |
|----------|-------|
| Order APIs (place/modify/cancel/GTT) | **9-10 orders per second** |
| Data APIs (getTradeBook, etc.) | ~1 request per second |
| Historical data | Varies, throttled |

**Known issue**: Users report random "Access denied because of exceeding access rate" errors even when well within limits. One user hit getTradeBook every 15 seconds and got rate limit errors every 4-5 minutes.

### Static IP Mandate (April 1, 2026) -- CRITICAL

Effective **April 1, 2026**, AngelOne (and all brokers per SEBI/NSE circular) will:
- **Only accept API orders from a registered static IP**.
- You can update your mapped static IP **at most once per calendar week**.
- Standard TOPS (Threshold Orders Per Second): **10 OPS per exchange/segment**.
- NSE mandates retail algos be hosted on the **Trading Member's (broker's) server** -- unclear enforcement for retail API users.

**Implication for Pi-based bot**: A Raspberry Pi on home broadband (dynamic IP) will NOT work after April 1, 2026 unless you get a static IP from your ISP or route through a cloud proxy with a static IP.

### Comparison with Other Broker APIs

| Feature | AngelOne SmartAPI | Zerodha Kite Connect | Upstox API |
|---------|------------------|---------------------|------------|
| Cost | Free | Rs 2,000/month | Free |
| Languages | Python, Java, Node | Python, Java, Go, PHP | Python, Node, Java |
| Documentation | Decent | Best in class | Good |
| Reliability | Intermittent rate limit issues | Slow during market open/close (15s order delays reported) | Stable, widely trusted |
| Community | Active forum | Large community (TradingQ&A) | Growing |
| WebSocket | Available | Available | Available |
| Historical Data | Available | Available | Available |

**Key tradeoff**: Zerodha has the best ecosystem but costs Rs 2,000/month. AngelOne is free but has quirks. Upstox is free and stable but less community support.

---

## 4. Operational Risks for Automated F&O Trading

### F&O Ban Stocks (MWPL)

- A stock enters F&O ban when total open interest exceeds **95% of Market Wide Position Limit (MWPL)**.
- Ban is lifted when OI drops below **80% of MWPL**.
- During ban: **No new positions** allowed. Can only **square off** existing positions.
- Violation penalty: Exchange levy of **1% of increased position value** on the day of violation.

**Bot handling**: Monitor MWPL data via NSE API. Before placing any stock F&O order, check if the stock is in ban list. Index derivatives (Nifty, Bank Nifty) are **never** in F&O ban.

**Position limits for stocks in ban period** were updated effective December 8, 2025 (AngelOne circular).

### Circuit Breaker Halts

Index-wide circuit breakers (BSE Sensex or Nifty 50, whichever breaches first):

| Level | Before 1:00 PM | 1:00-2:00 PM | 2:00-2:30 PM | After 2:30 PM |
|-------|---------------|-------------|-------------|--------------|
| **10%** | 45 min halt | 15 min halt | No halt | No halt |
| **15%** | 1h 45m halt | 45 min halt | Halt for day | Halt for day |
| **20%** | Halt for day | Halt for day | Halt for day | Halt for day |

**Individual stock circuit filters**: 2%, 5%, 10%, or 20% (set by exchange based on stock's volatility). Stocks in F&O segment typically have **no circuit limits** (they have price bands removed), but index derivatives follow index circuit breakers.

**Bot handling**: If a halt triggers, all pending orders become invalid. Bot must detect halt via WebSocket disconnection or order rejection and NOT retry orders blindly.

### Lot Size Changes Mid-Position

- **Existing contracts are NOT affected**. Old lot sizes continue until those specific contracts expire.
- New lot sizes apply only to newly introduced contract series.
- No forced position adjustment or conversion mid-contract.
- Bot must track lot size per contract expiry, not assume a single lot size for an instrument.

### Exchange Holidays and Special Sessions

- **15 scheduled market closures** in 2026 (weekday holidays).
- Normal trading hours: **9:15 AM to 3:30 PM IST**, Monday-Friday.
- **Muhurat Trading**: Special 1-hour session on Diwali (Nov 8, 2026 -- falls on Sunday). Timings announced separately.
- **No early closures** on regular trading days (unlike some global markets).
- Pre-open session: 9:00-9:15 AM (order matching, no F&O).

**Bot handling**: Maintain a holiday calendar. Fetch updated list from NSE periodically. Do not place orders on holidays. Handle Muhurat trading as a special case if desired.

---

## 5. Running a Trading Bot: Raspberry Pi vs Cloud

### Raspberry Pi

**Pros**:
- Zero recurring cost (one-time ~Rs 5,000-8,000 for Pi 4/5).
- Full control, no vendor lock-in.
- Already running financial-agent on Pi (proven setup).

**Cons**:
- **Static IP problem**: After April 1, 2026, SEBI mandates static IP for API trading. Home broadband typically has dynamic IP. Options: (a) get static IP from ISP (~Rs 200-500/month), (b) route through cloud VPN/proxy with static IP.
- **Power reliability**: UPS needed. Power outage during market hours = missed trades.
- **Network reliability**: ISP outage = catastrophic for live trading. No failover.
- **SD card corruption**: Known Pi failure mode. Use SSD boot or quality SD with regular backups.
- **WiFi instability**: Must use Ethernet + disable WiFi power save (already known issue from MEMORY.md).
- **Single point of failure**: No redundancy.

### Cloud (AWS/GCP/DigitalOcean)

**Pros**:
- **Static IP included** (Elastic IP on AWS, Reserved IP on DO).
- High availability, redundancy.
- Easy to get Mumbai region (ap-south-1) for low latency to NSE.
- Professional monitoring and alerting.

**Cons**:
- Monthly cost: Rs 500-2,000/month for a small VM.
- Vendor dependency.
- Need to secure access (SSH keys, firewall).

### Latency Considerations for India

- NSE servers are in Mumbai (colocation at BSE/NSE data centers).
- **Cloud in Mumbai (ap-south-1)**: ~1-5ms to NSE. Best for retail.
- **Pi on home broadband**: ~10-50ms depending on ISP and location. Acceptable for positional/swing trading, risky for scalping.
- **For F&O strategies with hold times > 5 minutes**: Latency difference is irrelevant.
- **For expiry-day options trading**: Every millisecond matters. Cloud wins.

### Recommendation for This Project

| Phase | Infrastructure | Rationale |
|-------|---------------|-----------|
| Paper trading | Raspberry Pi | No real money at risk, free, already set up |
| Live trading (initial) | Cloud VM (Mumbai) + static IP | SEBI compliance, reliability |
| Live trading (scaled) | Cloud VM + Pi as backup monitor | Redundancy |

**Cost-effective cloud option**: DigitalOcean droplet in Bangalore (SGP1 closest) or AWS Lightsail Mumbai -- Rs 400-800/month for a basic instance.

---

## 6. Key Regulatory Timeline (2026)

| Date | Change | Impact |
|------|--------|--------|
| **April 1, 2026** | STT hike (futures 0.05%, options 0.15%) | Higher transaction costs |
| **April 1, 2026** | Static IP mandatory for API trading | Must register static IP with broker |
| **April 1, 2026** | SEBI algo trading framework fully binding | Registration, audit trail requirements |
| **Ongoing** | Position limit monitoring intraday | More frequent OI checks needed |

---

## 7. Cost Modeling for a Single Nifty Options Trade (Post April 2026)

Assumptions: Buy 1 lot Nifty CE (75 qty), premium Rs 200, sell at Rs 250.

| Charge | Buy Side | Sell Side |
|--------|----------|-----------|
| Premium paid/received | Rs 15,000 | Rs 18,750 |
| Brokerage (Rs 20 flat) | Rs 20 | Rs 20 |
| STT (0.15% on sell premium) | -- | Rs 28.13 |
| Exchange txn charges (~0.05%) | Rs 7.50 | Rs 9.38 |
| GST (18% on brokerage + txn) | Rs 4.95 | Rs 5.29 |
| SEBI charges (Rs 10/crore) | Rs 0.02 | Rs 0.02 |
| Stamp duty (0.003% on buy) | Rs 0.45 | -- |
| **Total charges** | **~Rs 33** | **~Rs 63** |
| **Total round-trip cost** | | **~Rs 96** |
| **Profit before charges** | | **Rs 3,750** |
| **Profit after charges** | | **~Rs 3,654** |

For a Rs 3,750 gross profit trade, charges eat ~2.6%. For smaller profits (e.g., Rs 500 on a scalp), charges become 19% -- significant.

---

## Sources

- [ClearTax - STT Latest Updates](https://cleartax.in/s/securities-transaction-tax-stt)
- [Bajaj Finserv - STT India 2026](https://www.bajajfinserv.in/securities-transaction-tax)
- [Finnovate - Budget 2026 STT Hike](https://www.finnovate.in/learn/blog/budget-2026-stt-hike-fno-trades-explained)
- [1Finance - STT Futures Options Budget 2026](https://1finance.co.in/blog/stt-futures-options-increased-budget-2026-for-fno-investors/)
- [Zerodha Z-Connect - SEBI New Rules](https://zerodha.com/z-connect/business-updates/sebis-new-rules-for-index-derivatives-heres-whats-changing)
- [ICICI Direct - SEBI New F&O Rules](https://www.icicidirect.com/research/equity/finace/sebi-introduces-new-rules-for-restricted-entry-in-futures-and-options-trading)
- [Business Standard - SEBI F&O Measures](https://www.business-standard.com/markets/news/sebi-announces-six-key-changes-to-curb-speculation-in-derivatives-trading-124100101316_1.html)
- [AngelOne - SmartAPI Changes April 2026](https://www.angelone.in/news/market-updates/what-s-changing-in-angel-one-s-smartapi-access-from-april-1-2026)
- [SmartAPI Forum - Rate Limits](https://smartapi.angelone.in/smartapi/forum/topic/4387/changes-in-api-rate-limit)
- [SmartAPI Forum - Static IP Keys](https://smartapi.angelone.in/smartapi/forum/topic/5352/static-ip-based-api-keys-now-live-old-flow-still-supported-temporarily)
- [Fintrens - NSE Static IP Mandate](https://blogs.fintrens.com/urgent-your-algo-trading-may-stop-on-october-1st-nse-static-ip-mandate-explained/)
- [Bajaj Finserv - F&O Ban](https://www.bajajfinserv.in/what-is-f-and-o-ban)
- [Paytm Money - F&O Ban MWPL](https://www.paytmmoney.com/blog/fo-ban-list-meaning-rules-impact/)
- [NSE India - Circuit Breakers](https://www.nseindia.com/products-services/equity-market-circuit-breakers)
- [Groww - NSE Holidays 2026](https://groww.in/p/nse-holidays)
- [NSE India - Market Holidays](https://www.nseindia.com/resources/exchange-communication-holidays)
- [VakilAdda - F&O Tax Audit FY 2025-26](https://vakiladda.com/complete-guide-to-fo-tax-audit-for-financial-year-2025-26-ay-2026-27/)
- [ClearTax - Tax Audit 44AB](https://cleartax.in/s/tax-audit-section-44ab)
- [Zerodha Charges](https://zerodha.com/charges/)
- [Motilal Oswal - GST in Stock Broking](https://www.motilaloswal.com/learning-centre/2025/8/everything-you-need-to-know-about-gst-in-stock-broking)
- [TradingQ&A - Best API Comparison](https://tradingqna.com/t/best-api-for-trading-other-than-kite-in-terms-of-performance-and-reliability/119934)
- [Ventura - F&O Lot Size Changes Jan 2026](https://www.venturasecurities.com/blog/fo-lot-size-changes-in-india-what-traders-need-to-know-effective-jan-2026/)
