"""
Tool definitions and handlers for the agentic financial assistant.

Each tool has:
  - A JSON schema (passed to the Anthropic tool-use API)
  - A handler function that calls SmartAPI or DuckDuckGo and returns JSON

dispatch_tool() routes tool calls by name.
"""

import json
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Tool schemas (Anthropic tool-use format)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "search_stock",
        "description": (
            "Search for a stock by name or symbol to find its trading symbol and token. "
            "Use this first when the user mentions a stock name and you don't know the token."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Stock name or symbol to search for (e.g. 'RELIANCE', 'TCS', 'Infosys')",
                },
                "exchange": {
                    "type": "string",
                    "enum": ["NSE", "BSE", "NFO"],
                    "description": "Exchange to search on. Default NSE.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_quote",
        "description": (
            "Get the live quote for a stock — LTP, bid/ask, volume, open, high, low, close, "
            "and percentage change. Requires the exchange, symbol, and token."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "exchange": {"type": "string", "description": "Exchange (NSE or BSE)"},
                "symbol": {"type": "string", "description": "Trading symbol (e.g. 'RELIANCE-EQ')"},
                "token": {"type": "string", "description": "SmartAPI symbol token (e.g. '2885')"},
            },
            "required": ["exchange", "symbol", "token"],
        },
    },
    {
        "name": "get_historical_data",
        "description": (
            "Fetch historical OHLCV candle data for a stock. "
            "Returns date, open, high, low, close, volume for each candle."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "exchange": {"type": "string", "description": "Exchange (NSE or BSE)"},
                "token": {"type": "string", "description": "SmartAPI symbol token"},
                "interval": {
                    "type": "string",
                    "enum": ["ONE_MINUTE", "FIVE_MINUTE", "FIFTEEN_MINUTE",
                             "THIRTY_MINUTE", "ONE_HOUR", "ONE_DAY"],
                    "description": "Candle interval. Default ONE_DAY.",
                },
                "days": {
                    "type": "integer",
                    "description": "Number of days of history. Default 30.",
                },
            },
            "required": ["exchange", "token"],
        },
    },
    {
        "name": "get_option_chain",
        "description": (
            "Fetch the option chain for a stock with Greeks (delta, gamma, theta, vega), "
            "IV, and open interest for each strike. Only works for F&O stocks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol (e.g. 'RELIANCE', 'NIFTY', 'BANKNIFTY')",
                },
                "expiry": {
                    "type": "string",
                    "description": "Expiry date (DD-Mon-YYYY, e.g. '27-Feb-2025'). Empty for nearest.",
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_gainers_losers",
        "description": "Get the top gainers or losers on NSE/BSE for a given data type.",
        "input_schema": {
            "type": "object",
            "properties": {
                "data_type": {
                    "type": "string",
                    "enum": ["PercOIGainers", "PercOILosers", "PercPriceGainers", "PercPriceLosers",
                             "LongBuildUp", "ShortBuildUp", "LongUnwinding", "ShortCovering"],
                    "description": "Type of data to fetch.",
                },
                "exchange": {
                    "type": "string",
                    "enum": ["NSE", "BSE"],
                    "description": "Exchange. Default NSE.",
                },
            },
            "required": ["data_type"],
        },
    },
    {
        "name": "get_pcr",
        "description": "Get the put-call ratio for a stock or index.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Symbol (e.g. 'RELIANCE', 'NIFTY', 'BANKNIFTY')",
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_holdings",
        "description": "Get all stocks in the user's portfolio (delivery holdings) with current value and P&L.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_positions",
        "description": "Get all open intraday and F&O positions with current P&L.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_order_book",
        "description": "Get today's order book — all orders placed today with their status.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_margin",
        "description": "Get available margin/funds — cash, collateral, and used margin.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "place_order",
        "description": (
            "Place a buy or sell order. In DRY_RUN mode (default), this only simulates the order. "
            "Use this when the user explicitly asks to buy or sell."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Trading symbol (e.g. 'RELIANCE-EQ')"},
                "token": {"type": "string", "description": "SmartAPI symbol token"},
                "transaction_type": {
                    "type": "string",
                    "enum": ["BUY", "SELL"],
                    "description": "Buy or sell",
                },
                "quantity": {"type": "integer", "description": "Number of shares/lots"},
                "order_type": {
                    "type": "string",
                    "enum": ["MARKET", "LIMIT"],
                    "description": "Order type. Default MARKET.",
                },
                "price": {
                    "type": "number",
                    "description": "Limit price. Required for LIMIT orders, ignored for MARKET.",
                },
                "exchange": {"type": "string", "description": "Exchange (NSE or BSE). Default NSE."},
                "product_type": {
                    "type": "string",
                    "enum": ["DELIVERY", "INTRADAY"],
                    "description": "Product type. Default DELIVERY.",
                },
            },
            "required": ["symbol", "token", "transaction_type", "quantity"],
        },
    },
    {
        "name": "search_news",
        "description": (
            "Search for recent financial news using DuckDuckGo. "
            "Use this when the user asks about news, events, or recent developments for a stock or market."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (e.g. 'RELIANCE quarterly results', 'NIFTY today')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return. Default 5.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "run_screener",
        "description": (
            "Run the morning F&O screener — scans all gainers/losers across price, OI, "
            "and buildup categories, cross-references stocks, and produces an actionable "
            "watchlist with Claude analysis. Use when the user asks what to watch today, "
            "wants a market scan, or says 'run the screener'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "top_n": {
                    "type": "integer",
                    "description": "Number of top candidates to analyze in depth. Default 5.",
                },
                "raw_only": {
                    "type": "boolean",
                    "description": "If true, return raw signal data without Claude analysis. Default false.",
                },
            },
        },
    },
    {
        "name": "portfolio_watchdog",
        "description": (
            "Analyze portfolio holdings against today's F&O signals. Cross-references "
            "your holdings and positions with OI buildups, price/OI movers to flag stocks "
            "that need action — EXIT/REDUCE, HOLD/ADD, or MONITOR with urgency levels. "
            "Use when the user asks to check their portfolio, wants a risk check, or says "
            "'check my holdings'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "detailed": {
                    "type": "boolean",
                    "description": "Include candle enrichment for flagged stocks. Default false.",
                },
            },
        },
    },
    {
        "name": "deep_dive_stock",
        "description": (
            "Comprehensive single-stock analysis with trade plan. Combines 30-day candles, "
            "option chain (IV, OI, Greeks), OI buildup signals, and PCR into an actionable "
            "trade plan with entry, targets, stop-loss, and risk:reward. Use when the user "
            "asks for a deep dive, detailed analysis, or trade plan for a specific stock."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol (e.g. 'RELIANCE', 'BHARATFORG')",
                },
                "token": {
                    "type": "string",
                    "description": "SmartAPI token. If omitted, auto-resolved via search.",
                },
            },
            "required": ["symbol"],
        },
    },
]


# ---------------------------------------------------------------------------
# Handler functions
# ---------------------------------------------------------------------------

def handle_search_stock(smart_api, params):
    query = params["query"]
    exchange = params.get("exchange", "NSE")
    try:
        results = smart_api.searchScrip(exchange, query)
        if results and results.get("data"):
            # Return top 5 matches
            matches = results["data"][:5]
            return {"matches": matches, "count": len(matches)}
        return {"matches": [], "count": 0, "message": f"No results for '{query}' on {exchange}"}
    except Exception as e:
        return {"error": str(e)}


def handle_get_quote(smart_api, params):
    exchange = params["exchange"]
    symbol = params["symbol"]
    token = params["token"]
    try:
        # Try ltpData first for a quick quote
        ltp_resp = smart_api.ltpData(exchange, symbol, token)
        # Also get full market data for more details
        try:
            market_resp = smart_api.getMarketData({
                "mode": "FULL",
                "exchangeTokens": {exchange: [token]},
            })
            if market_resp and market_resp.get("data") and market_resp["data"].get("fetched"):
                full = market_resp["data"]["fetched"][0]
                return {
                    "symbol": symbol,
                    "token": token,
                    "exchange": exchange,
                    "ltp": full.get("ltp"),
                    "open": full.get("open"),
                    "high": full.get("high"),
                    "low": full.get("low"),
                    "close": full.get("close"),
                    "volume": full.get("tradeVolume"),
                    "bid": full.get("bestBidPrice"),
                    "ask": full.get("bestAskPrice"),
                    "change_pct": full.get("percentChange"),
                    "net_change": full.get("netChange"),
                }
        except Exception:
            pass

        # Fall back to LTP data
        if ltp_resp and ltp_resp.get("data"):
            return {
                "symbol": symbol,
                "token": token,
                "exchange": exchange,
                "ltp": ltp_resp["data"].get("ltp"),
                "open": ltp_resp["data"].get("open"),
                "high": ltp_resp["data"].get("high"),
                "low": ltp_resp["data"].get("low"),
                "close": ltp_resp["data"].get("close"),
            }
        return {"error": "No quote data returned"}
    except Exception as e:
        return {"error": str(e)}


def handle_get_historical_data(smart_api, params):
    exchange = params.get("exchange", "NSE")
    token = params["token"]
    interval = params.get("interval", "ONE_DAY")
    days = params.get("days", 30)

    to_date = datetime.now()
    from_date = to_date - timedelta(days=days)

    try:
        response = smart_api.getCandleData({
            "exchange": exchange,
            "symboltoken": token,
            "interval": interval,
            "fromdate": from_date.strftime("%Y-%m-%d 09:15"),
            "todate": to_date.strftime("%Y-%m-%d 15:30"),
        })

        if response and response.get("data"):
            candles = response["data"]
            # Format as list of dicts for readability
            formatted = []
            for c in candles:
                formatted.append({
                    "date": c[0][:10] if isinstance(c[0], str) else str(c[0])[:10],
                    "open": c[1], "high": c[2], "low": c[3], "close": c[4],
                    "volume": c[5],
                })
            return {"candles": formatted, "count": len(formatted), "interval": interval}
        return {"candles": [], "count": 0, "message": "No data returned"}
    except Exception as e:
        return {"error": str(e)}


def get_nearest_expiry() -> str:
    """Get the nearest monthly options expiry (last Thursday of the month) in dd-Mon-yyyy format."""
    from datetime import date, timedelta
    import calendar

    today = date.today()
    for month_offset in range(3):
        y = today.year + (today.month + month_offset - 1) // 12
        m = (today.month + month_offset - 1) % 12 + 1
        last_day = calendar.monthrange(y, m)[1]
        d = date(y, m, last_day)
        while d.weekday() != 3:
            d -= timedelta(days=1)
        if d >= today:
            return d.strftime("%d%b%Y").upper()
    return ""


def handle_get_option_chain(smart_api, params):
    symbol = params["symbol"]
    expiry = params.get("expiry", "") or get_nearest_expiry()
    try:
        response = smart_api.optionGreek({
            "name": symbol,
            "expirydate": expiry,
        })
        if response and response.get("data"):
            data = response["data"]
            # Summarize to keep token usage reasonable
            summary = []
            for row in data:
                strike = row.get("strikePrice", 0)
                ce = row.get("CE", {}) or {}
                pe = row.get("PE", {}) or {}
                summary.append({
                    "strike": strike,
                    "ce_oi": ce.get("openInterest", 0),
                    "ce_iv": ce.get("impliedVolatility", 0),
                    "ce_delta": ce.get("delta", 0),
                    "ce_ltp": ce.get("ltp", 0),
                    "pe_oi": pe.get("openInterest", 0),
                    "pe_iv": pe.get("impliedVolatility", 0),
                    "pe_delta": pe.get("delta", 0),
                    "pe_ltp": pe.get("ltp", 0),
                })
            return {"options": summary, "count": len(summary), "symbol": symbol}
        return {"options": [], "count": 0, "message": "No option chain data returned"}
    except Exception as e:
        return {"error": str(e)}


def handle_get_gainers_losers(smart_api, params):
    data_type = params["data_type"]
    exchange = params.get("exchange", "NSE")

    # Buildup types use oIBuildup endpoint with space-separated names
    OI_BUILDUP_MAP = {
        "LongBuildUp": "Long Built Up",
        "ShortBuildUp": "Short Built Up",
        "LongUnwinding": "Long Unwinding",
        "ShortCovering": "Short Covering",
    }

    try:
        if data_type in OI_BUILDUP_MAP:
            response = smart_api.oIBuildup({
                "expirytype": "NEAR",
                "datatype": OI_BUILDUP_MAP[data_type],
            })
        else:
            response = smart_api.gainersLosers({
                "datatype": data_type,
                "expirytype": "NEAR",
            })
        if response and response.get("data"):
            return {"data": response["data"][:10], "type": data_type}
        return {"data": [], "message": "No data returned"}
    except Exception as e:
        return {"error": str(e)}


def handle_get_pcr(smart_api, params):
    symbol = params["symbol"].upper()
    try:
        response = smart_api.putCallRatio()
        if response and response.get("data"):
            # Find the requested symbol in the PCR list
            for item in response["data"]:
                if symbol in item.get("tradingSymbol", "").upper():
                    return {"pcr_data": item, "symbol": symbol}
            # Symbol not found — return all available
            return {"pcr_data": response["data"][:5], "symbol": symbol,
                    "message": f"'{symbol}' not found, showing top results"}
        return {"pcr_data": None, "message": "No PCR data returned"}
    except Exception as e:
        return {"error": str(e)}


def handle_get_holdings(smart_api, _params):
    try:
        response = smart_api.allholding()
        if response and response.get("data"):
            holdings = response["data"].get("holdings", response["data"])
            summary = []
            for h in (holdings if isinstance(holdings, list) else []):
                summary.append({
                    "symbol": h.get("tradingsymbol", h.get("symbolname", "")),
                    "quantity": h.get("quantity", 0),
                    "avg_price": h.get("averageprice", 0),
                    "ltp": h.get("ltp", 0),
                    "pnl": h.get("profitandloss", 0),
                    "pnl_pct": h.get("pnlpercentage", 0),
                })
            total_value = response["data"].get("totalholding", {})
            return {"holdings": summary, "count": len(summary), "totals": total_value}
        return {"holdings": [], "count": 0, "message": "No holdings found"}
    except Exception as e:
        return {"error": str(e)}


def handle_get_positions(smart_api, _params):
    try:
        response = smart_api.position()
        if response and response.get("data"):
            return {"positions": response["data"], "count": len(response["data"])}
        return {"positions": [], "count": 0, "message": "No open positions"}
    except Exception as e:
        return {"error": str(e)}


def handle_get_order_book(smart_api, _params):
    try:
        response = smart_api.orderBook()
        if response and response.get("data"):
            orders = []
            for o in response["data"]:
                orders.append({
                    "order_id": o.get("orderid"),
                    "symbol": o.get("tradingsymbol"),
                    "type": o.get("transactiontype"),
                    "qty": o.get("quantity"),
                    "price": o.get("price"),
                    "status": o.get("orderstatus"),
                    "time": o.get("updatetime"),
                })
            return {"orders": orders, "count": len(orders)}
        return {"orders": [], "count": 0, "message": "No orders today"}
    except Exception as e:
        return {"error": str(e)}


def handle_get_margin(smart_api, _params):
    try:
        response = smart_api.rmsLimit()
        if response and response.get("data"):
            d = response["data"]
            return {
                "available_cash": d.get("availablecash"),
                "available_margin": d.get("availableintradaypayin"),
                "collateral": d.get("collateral"),
                "used_margin": d.get("utiliseddebits"),
                "net": d.get("net"),
            }
        return {"error": "No margin data returned"}
    except Exception as e:
        return {"error": str(e)}


def handle_place_order(smart_api, params, dry_run=True):
    symbol = params["symbol"]
    token = params["token"]
    txn_type = params["transaction_type"]
    qty = params["quantity"]
    order_type = params.get("order_type", "MARKET")
    price = params.get("price", 0)
    exchange = params.get("exchange", "NSE")
    product_type = params.get("product_type", "DELIVERY")

    order_params = {
        "variety": "NORMAL",
        "tradingsymbol": symbol,
        "symboltoken": token,
        "transactiontype": txn_type,
        "exchange": exchange,
        "ordertype": order_type,
        "producttype": product_type,
        "duration": "DAY",
        "quantity": qty,
        "price": price if order_type == "LIMIT" else 0,
        "squareoff": 0,
        "stoploss": 0,
        "triggerprice": 0,
    }

    if dry_run:
        return {
            "status": "DRY_RUN",
            "message": f"[SIMULATED] Would {txn_type} {qty}x {symbol} @ {order_type}"
                       + (f" ₹{price}" if order_type == "LIMIT" else " MARKET"),
            "order_params": order_params,
        }

    try:
        order_id = smart_api.placeOrder(order_params)
        return {
            "status": "PLACED",
            "order_id": order_id,
            "message": f"Order placed: {txn_type} {qty}x {symbol}",
        }
    except Exception as e:
        return {"status": "FAILED", "error": str(e)}


def handle_search_news(_smart_api, params):
    query = params["query"]
    max_results = params.get("max_results", 5)
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.news(query, max_results=max_results):
                results.append({
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "body": r.get("body", "")[:200],
                    "date": r.get("date"),
                    "source": r.get("source"),
                })
        return {"results": results, "count": len(results), "query": query}
    except ImportError:
        return {"error": "duckduckgo-search not installed. Run: pip install duckduckgo-search"}
    except Exception as e:
        return {"error": str(e)}


def handle_run_screener(smart_api, params):
    top_n = params.get("top_n", 5)
    raw_only = params.get("raw_only", False)
    try:
        from screener import run_screener
        result = run_screener(smart_api, top_n=top_n, raw_only=raw_only)
        return result
    except Exception as e:
        return {"error": str(e)}


def handle_portfolio_watchdog(smart_api, params):
    detailed = params.get("detailed", False)
    try:
        from portfolio_analysis import run_watchdog
        result = run_watchdog(smart_api, detailed=detailed)
        return result
    except Exception as e:
        return {"error": str(e)}


def handle_deep_dive_stock(smart_api, params):
    symbol = params["symbol"]
    token = params.get("token")
    try:
        from portfolio_analysis import run_deep_dive
        result = run_deep_dive(smart_api, symbol, token=token)
        return result
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_HANDLERS = {
    "search_stock": handle_search_stock,
    "get_quote": handle_get_quote,
    "get_historical_data": handle_get_historical_data,
    "get_option_chain": handle_get_option_chain,
    "get_gainers_losers": handle_get_gainers_losers,
    "get_pcr": handle_get_pcr,
    "get_holdings": handle_get_holdings,
    "get_positions": handle_get_positions,
    "get_order_book": handle_get_order_book,
    "get_margin": handle_get_margin,
    "place_order": None,  # special-cased for dry_run
    "search_news": handle_search_news,
    "run_screener": handle_run_screener,
    "portfolio_watchdog": handle_portfolio_watchdog,
    "deep_dive_stock": handle_deep_dive_stock,
}


def dispatch_tool(name: str, tool_input: dict, smart_api, dry_run: bool = True) -> str:
    """
    Execute a tool by name and return the JSON result as a string.
    """
    if name == "place_order":
        result = handle_place_order(smart_api, tool_input, dry_run=dry_run)
    elif name in _HANDLERS:
        handler = _HANDLERS[name]
        result = handler(smart_api, tool_input)
    else:
        result = {"error": f"Unknown tool: {name}"}

    return json.dumps(result, default=str)
