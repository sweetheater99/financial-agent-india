# tests/test_v7_main.py
"""Tests for V7 main CLI entry point."""
import pytest
import sys
from unittest.mock import patch, MagicMock
from v7.main import parse_args, STATUS_COMMANDS, TRADING_COMMANDS


def test_parse_args_premarket():
    args = parse_args(["premarket"])
    assert args.command == "premarket"


def test_parse_args_tick():
    args = parse_args(["tick"])
    assert args.command == "tick"


def test_parse_args_eod():
    args = parse_args(["eod"])
    assert args.command == "eod"


def test_parse_args_status():
    args = parse_args(["status"])
    assert args.command == "status"


def test_parse_args_weekly():
    args = parse_args(["weekly"])
    assert args.command == "weekly"


def test_parse_args_monthly():
    args = parse_args(["monthly"])
    assert args.command == "monthly"


def test_parse_args_checkin():
    args = parse_args(["checkin", "--num", "1"])
    assert args.command == "checkin"
    assert args.num == 1


def test_parse_args_paper_flag():
    args = parse_args(["--paper", "tick"])
    assert args.paper is True


def test_parse_args_live_flag():
    args = parse_args(["tick"])
    assert args.paper is False


def test_all_commands_defined():
    all_cmds = STATUS_COMMANDS + TRADING_COMMANDS
    for cmd in ["premarket", "opening-read", "checkin", "tick", "eod",
                 "weekly", "monthly", "status", "paper-status"]:
        assert cmd in all_cmds


def test_parse_args_opening_read():
    args = parse_args(["opening-read"])
    assert args.command == "opening-read"


def test_parse_args_paper_status():
    args = parse_args(["paper-status"])
    assert args.command == "paper-status"
