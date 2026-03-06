"""Tests for F&O ban list checker."""
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestParseBanList:
    @patch("fo_ban.requests.get")
    def test_fetch_parses_csv(self, mock_get):
        from fo_ban import fetch_ban_list
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "Symbol\nBANDANBNK\nINDIANB\nGMRINFRA\n"
        mock_get.return_value = mock_resp

        # Clear any cached file for today
        from fo_ban import _cache_path
        cache = _cache_path()
        if cache.exists():
            cache.unlink()

        result = fetch_ban_list()
        assert "BANDANBNK" in result
        assert "INDIANB" in result
        assert "GMRINFRA" in result
        assert len(result) == 3

    @patch("fo_ban.requests.get")
    def test_fetch_failure_returns_empty(self, mock_get):
        from fo_ban import fetch_ban_list, _cache_path
        cache = _cache_path()
        if cache.exists():
            cache.unlink()

        mock_get.side_effect = Exception("network error")
        result = fetch_ban_list()
        assert result == set()

    def test_is_in_fo_ban_with_suffix(self):
        from fo_ban import is_in_fo_ban
        with patch("fo_ban.fetch_ban_list", return_value={"BANDANBNK", "INDIANB"}):
            assert is_in_fo_ban("BANDANBNK") is True
            assert is_in_fo_ban("BANDANBNK-EQ") is True
            assert is_in_fo_ban("RELIANCE") is False

    def test_is_in_fo_ban_case_insensitive(self):
        from fo_ban import is_in_fo_ban
        with patch("fo_ban.fetch_ban_list", return_value={"BANDANBNK"}):
            assert is_in_fo_ban("bandanbnk") is True
