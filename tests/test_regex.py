"""Stage 1 regex filter tests over the language-neutral fixture ruleset."""
from privyscope._core.regex_filter import RegexFilter


def _filter(fixture_rules):
    return RegexFilter.from_yaml(fixture_rules)


def test_phone_email(fixture_rules):
    rf = _filter(fixture_rules)
    text = "call 555-123-4567, email user@example.com"
    spans = {(s.label, text[s.start:s.end]) for s in rf.find(text)}
    assert ("PHONE", "555-123-4567") in spans
    assert ("EMAIL", "user@example.com") in spans


def test_secret_beats_generic(fixture_rules):
    rf = _filter(fixture_rules)
    spans = rf.find("key is sk-abcdefghijklmnopqrstuvwxyz0123456789")
    assert any(s.label == "SECRET" for s in spans)


def test_card_number_is_bank_not_phone(fixture_rules):
    rf = _filter(fixture_rules)
    spans = {(s.label, s.start, s.end) for s in rf.find("card 1234-5678-9012-3456 charged")}
    assert any(lab == "BANK" for lab, _s, _e in spans)
    assert not any(lab == "PHONE" for lab, _s, _e in spans)


def test_no_overlapping_spans(fixture_rules):
    rf = _filter(fixture_rules)
    spans = sorted(rf.find("555-123-4567 user@x.com 1111-2222-3333-4444"), key=lambda s: s.start)
    for a, b in zip(spans, spans[1:]):
        assert a.end <= b.start
