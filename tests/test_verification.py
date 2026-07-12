"""The verification gate that Stage-1 regex rules run their matches through."""
from __future__ import annotations

import pytest

from privyscope._core import verification
from privyscope._core.bioes import Span
from privyscope._core.regex_filter import RegexFilter, RegexRule


def test_vendored_functions_are_loaded():
    # 0 means setup.py never vendored the submodule, so every rule would run unverified.
    assert verification.available() > 0


@pytest.mark.parametrize("name,value,expected", [
    ("kr_rrn_valid", "900101-1234568", True),
    ("kr_rrn_valid", "900101-1234567", False),   # bad checksum
    ("luhn", "4532015112830366", True),
    ("luhn", "4532015112830367", False),         # bad check digit
    ("luhn", "4532-0151-1283-0366", True),       # separators are ignored
])
def test_verify_known_functions(name, value, expected):
    assert verification.verify(name, value) is expected


def test_resolve_returns_none_for_unknown():
    assert verification.resolve("no_such_verifier") is None


def test_unknown_name_fails_open():
    """A missing verifier must keep the match: for a redaction engine, over-redacting is
    safer than silently letting PII through."""
    assert verification.verify("no_such_verifier", "anything") is True


def test_raising_verifier_fails_open(monkeypatch):
    def boom(_value):
        raise RuntimeError("validator is broken")

    monkeypatch.setattr(verification, "resolve", lambda _n: boom)
    assert verification.verify("boom", "900101-1234568") is True


def test_regex_filter_drops_unverified_matches():
    import re

    rules = [RegexRule(
        label="ID_NUM",
        pattern=re.compile(r"[0-9]{6}-[0-9]{7}"),
        priority=100,
        verify="kr_rrn_valid",
        rule_id="rrn_01",
    )]
    rf = RegexFilter(rules)
    assert rf.find("주민 900101-1234568 입니다") == [Span("ID_NUM", 3, 17)]
    assert rf.find("주민 900101-1234567 입니다") == []   # checksum fails -> no span


def test_regex_filter_without_verify_keeps_every_match():
    import re

    rf = RegexFilter([RegexRule(label="X", pattern=re.compile(r"\d+"))])
    assert [s.label for s in rf.find("abc 123 def 456")] == ["X", "X"]
