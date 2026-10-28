"""Plugin-declared NER-over-regex precedence (``ner_precedence:`` in entity_config.yaml).

A regex span wins every cross-label overlap by default. A plugin can list label pairs whose
meaning comes from context, not shape (a 9-digit routing number matches ID_NUM, a dashed account
number matches PHONE, but the model reads "routing number" and says BANK).
"""
from privyscope._core.bioes import Span
from privyscope._core.pipeline import NerPrecedence, apply_ner_precedence, merge_union
from privyscope._core.plugins import parse_ner_precedence

BANK_OVER = NerPrecedence("BANK", ("ID_NUM", "PHONE"))
TEXT = "routing number 301225349 for the account"
START = TEXT.index("3012")
END = START + 9


def test_ner_bank_beats_overlapping_regex_id_num():
    regex, ner = [Span("ID_NUM", START, END)], [Span("BANK", START, END)]
    assert merge_union(apply_ner_precedence(regex, ner, [BANK_OVER]), ner) == [Span("BANK", START, END)]


def test_default_still_lets_regex_win_without_a_rule():
    regex, ner = [Span("ID_NUM", START, END)], [Span("BANK", START, END)]
    assert merge_union(apply_ner_precedence(regex, ner, []), ner) == [Span("ID_NUM", START, END)]


def test_partial_overlap_also_drops_the_regex_span():
    regex, ner = [Span("PHONE", START + 3, END)], [Span("BANK", START, END)]
    assert apply_ner_precedence(regex, ner, [BANK_OVER]) == []


def test_regex_without_an_overlapping_ner_span_is_kept():
    regex = [Span("ID_NUM", START, END)]
    assert apply_ner_precedence(regex, [Span("BANK", 0, 5)], [BANK_OVER]) == regex
    assert apply_ner_precedence(regex, [], [BANK_OVER]) == regex


def test_labels_outside_over_are_untouched():
    regex, ner = [Span("EMAIL", START, END)], [Span("BANK", START, END)]
    assert apply_ner_precedence(regex, ner, [BANK_OVER]) == regex


def test_rule_only_fires_for_its_own_ner_label():
    regex, ner = [Span("ID_NUM", START, END)], [Span("PER", START, END)]
    assert apply_ner_precedence(regex, ner, [BANK_OVER]) == regex


def test_parse_reads_valid_entries_and_skips_malformed_ones():
    data = {"ner_precedence": [
        {"label": "BANK", "over": ["ID_NUM", "PHONE"]},
        {"label": "BANK"},                                  # no `over`
        {"label": "", "over": ["ID_NUM"]},                  # empty label
        {"label": "X", "over": ["X", 3]},                   # only itself / non-strings
        "BANK",                                             # not a mapping
    ]}
    assert parse_ner_precedence(data) == (BANK_OVER,)
    assert parse_ner_precedence({}) == () and parse_ner_precedence(None) == ()
