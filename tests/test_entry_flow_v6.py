import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_v6_function_exists():
    """_v6_pre_filter_candidates exists and is callable."""
    from paper_trade import _v6_pre_filter_candidates
    assert callable(_v6_pre_filter_candidates)


def test_v6_pre_filter_allows_all_directions():
    """V6 safety filter does NOT block by regime or direction."""
    from paper_trade import _v6_pre_filter_candidates
    candidates = [
        {"symbol": "RELIANCE", "direction": "bullish", "score": 6.0},
        {"symbol": "HDFC", "direction": "bearish", "score": 7.0},
    ]
    portfolio = {"positions": [], "closed_trades": []}
    result = _v6_pre_filter_candidates(candidates, portfolio)
    assert len(result) == 2, "V6 should pass all candidates regardless of direction"


def test_v6_pre_filter_blocks_duplicates():
    """V6 still blocks duplicate positions."""
    from paper_trade import _v6_pre_filter_candidates
    candidates = [{"symbol": "RELIANCE", "direction": "bullish"}]
    portfolio = {
        "positions": [{"symbol": "RELIANCE", "status": "open"}],
        "closed_trades": []
    }
    result = _v6_pre_filter_candidates(candidates, portfolio)
    assert len(result) == 0, "Should block duplicate positions"


def test_v6_pre_filter_blocks_sector_limit():
    """V6 still enforces sector limits."""
    from paper_trade import _v6_pre_filter_candidates, SYMBOL_SECTOR
    # Need a symbol with a known sector that has 2 open positions
    sectors = {}
    for sym, sec in SYMBOL_SECTOR.items():
        if sec not in sectors:
            sectors[sec] = []
        sectors[sec].append(sym)

    # Find a sector with 3+ symbols
    test_sector = None
    test_symbols = None
    for sec, syms in sectors.items():
        if len(syms) >= 3:
            test_sector = sec
            test_symbols = syms[:3]
            break

    if test_symbols:
        candidates = [{"symbol": test_symbols[2], "direction": "bullish"}]
        portfolio = {
            "positions": [
                {"symbol": test_symbols[0], "status": "open"},
                {"symbol": test_symbols[1], "status": "open"},
            ],
            "closed_trades": []
        }
        result = _v6_pre_filter_candidates(candidates, portfolio)
        assert len(result) == 0, f"Should block {test_symbols[2]} — sector {test_sector} has 2 open positions"
