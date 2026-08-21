"""Span coalescing + tokenizer round-trip (decode-mismatch) edge cases."""
from types import SimpleNamespace

from privyscope._core.bioes import Span, coalesce_adjacent
from privyscope._core.runtime import NerRuntime


# --- coalesce_adjacent ----------------------------------------------------
# A WordPiece split (``2024`` -> ``202`` + ``##4``) can make the model emit two
# abutting same-label spans for one entity; they must be healed back into one.

def test_coalesce_merges_touching_same_label():
    spans = [Span("DATE", 0, 3), Span("DATE", 3, 10)]
    assert coalesce_adjacent(spans) == [Span("DATE", 0, 10)]


def test_coalesce_merges_overlapping_same_label():
    spans = [Span("PER", 0, 4), Span("PER", 2, 6)]
    assert coalesce_adjacent(spans) == [Span("PER", 0, 6)]


def test_coalesce_keeps_gap_separated_spans():
    # A delimiter between two entities leaves a char gap -> distinct entities.
    spans = [Span("DATE", 0, 3), Span("DATE", 4, 7)]
    assert coalesce_adjacent(spans) == spans


def test_coalesce_keeps_touching_different_labels():
    spans = [Span("PER", 0, 3), Span("PHONE", 3, 10)]
    assert coalesce_adjacent(spans) == spans


def test_coalesce_sorts_and_is_stable_on_empty():
    assert coalesce_adjacent([]) == []
    unordered = [Span("DATE", 3, 10), Span("DATE", 0, 3)]
    assert coalesce_adjacent(unordered) == [Span("DATE", 0, 10)]


def test_coalesce_nested_span_absorbed():
    spans = [Span("LOC", 0, 12), Span("LOC", 4, 8)]
    assert coalesce_adjacent(spans) == [Span("LOC", 0, 12)]


# --- _decode_mismatch -----------------------------------------------------
# Only genuine information loss should flag; the tokenizer's by-design folds
# (WordPiece spacing, do_lower_case, NFKC) must not.

def _mismatch(text, decoded):
    stub = SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda *a, **k: decoded))
    return NerRuntime._decode_mismatch(stub, text, [1, 2, 3])


def test_decode_mismatch_ignores_wordpiece_spacing_and_lowercasing():
    # bert-base-chinese: CJK chars re-spaced, Latin lowercased -> not a mismatch.
    assert _mismatch("我在Kakao联系他", "我 在 kakao 联 系 他") is False


def test_decode_mismatch_ignores_fullwidth_nfkc_fold():
    assert _mismatch("ＡＢＣ１２３", "abc123") is False


def test_decode_mismatch_flags_real_loss():
    # An [UNK] swallowing a rare glyph is genuine loss -> must flag.
    assert _mismatch("我叫𠮷田", "我 叫 [UNK] 田") is True
