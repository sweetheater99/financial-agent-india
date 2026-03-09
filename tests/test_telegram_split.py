"""Tests for Telegram message splitting in paper_trade._telegram_send()."""

import sys
import types


def _make_paper_trade():
    """Import paper_trade with minimal stubs to avoid heavy deps."""
    import importlib
    # We need the actual module — just mock external calls
    sys.path.insert(0, "/Users/aravindms/financial-agent-india")
    import paper_trade
    return paper_trade


def test_short_message_not_split(monkeypatch):
    pt = _make_paper_trade()
    sent = []

    def fake_send(text, **kwargs):
        sent.append(text)

    monkeypatch.setattr(pt, "_send_single_telegram", fake_send)

    pt._telegram_send("Hello world")
    assert len(sent) == 1
    assert sent[0] == "Hello world"


def test_exactly_4000_not_split(monkeypatch):
    pt = _make_paper_trade()
    sent = []
    monkeypatch.setattr(pt, "_send_single_telegram", lambda text, **kw: sent.append(text))

    msg = "x" * 4000
    pt._telegram_send(msg)
    assert len(sent) == 1


def test_long_message_split_at_newline(monkeypatch):
    pt = _make_paper_trade()
    sent = []
    monkeypatch.setattr(pt, "_send_single_telegram", lambda text, **kw: sent.append(text))

    # Build a message with newlines, over 4000 chars
    lines = ["Line " + str(i).zfill(4) + " " + "x" * 80 for i in range(60)]
    msg = "\n".join(lines)
    assert len(msg) > 4000

    pt._telegram_send(msg)

    # Should be split into multiple chunks
    assert len(sent) >= 2


def test_chunk_headers_added(monkeypatch):
    pt = _make_paper_trade()
    sent = []
    monkeypatch.setattr(pt, "_send_single_telegram", lambda text, **kw: sent.append(text))

    lines = ["Line " + "x" * 90 for _ in range(60)]
    msg = "\n".join(lines)
    pt._telegram_send(msg)

    total = len(sent)
    assert total >= 2
    for i, chunk in enumerate(sent, 1):
        assert chunk.startswith(f"({i}/{total}) ")


def test_all_chunks_under_4100(monkeypatch):
    pt = _make_paper_trade()
    sent = []
    monkeypatch.setattr(pt, "_send_single_telegram", lambda text, **kw: sent.append(text))

    # Very long message
    lines = ["A" * 95 + "\n" for _ in range(100)]
    msg = "".join(lines)
    pt._telegram_send(msg)

    for chunk in sent:
        assert len(chunk) < 4100, f"Chunk too long: {len(chunk)} chars"


def test_kwargs_preserved(monkeypatch):
    pt = _make_paper_trade()
    calls = []

    def fake_send(text, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(pt, "_send_single_telegram", fake_send)

    # Short message with kwargs
    pt._telegram_send("hi", topic_id="123")
    assert calls[0] == {"topic_id": "123"}


def test_no_newline_hard_split(monkeypatch):
    pt = _make_paper_trade()
    sent = []
    monkeypatch.setattr(pt, "_send_single_telegram", lambda text, **kw: sent.append(text))

    # A single long line with no newlines
    msg = "x" * 8500
    pt._telegram_send(msg)
    assert len(sent) >= 2
    for chunk in sent:
        assert len(chunk) < 4100
