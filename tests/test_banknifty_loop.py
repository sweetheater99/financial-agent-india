"""Tests for multi-index strategy loop."""


def test_index_configs_has_both():
    from paper_trade import INDEX_CONFIGS
    names = [c["name"] for c in INDEX_CONFIGS]
    assert "NIFTY" in names
    assert "BANKNIFTY" in names


def test_index_configs_lot_sizes():
    from paper_trade import INDEX_CONFIGS
    for cfg in INDEX_CONFIGS:
        if cfg["name"] == "NIFTY":
            assert cfg["lot_size"] == 75
        elif cfg["name"] == "BANKNIFTY":
            assert cfg["lot_size"] == 30


def test_banknifty_vix_mult():
    from paper_trade import INDEX_CONFIGS
    for cfg in INDEX_CONFIGS:
        if cfg["name"] == "BANKNIFTY":
            assert cfg["vix_mult"] == 1.3
