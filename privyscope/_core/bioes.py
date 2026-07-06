"""BIOES span <-> token-tag conversion (SRS §10.1).

Conversions are offset-driven so they work with any Hugging Face fast tokenizer
that returns ``offset_mapping``. The same routines are used to build training
labels (spans -> BIOES) and to decode model output back to character spans
(BIOES -> spans).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .entities import OUTSIDE


@dataclass(frozen=True)
class Span:
    """A character-offset PII span. ``end`` is exclusive (SRS §10.3)."""

    label: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid span offsets: start={self.start} end={self.end}")


Offset = Tuple[int, int]  # (char_start, char_end) for a single token


def spans_to_bioes(spans: Sequence[Span], offsets: Sequence[Offset]) -> List[str]:
    """Project character spans onto token offsets, producing one BIOES tag per token.

    Special tokens are signalled by a ``(0, 0)`` offset (the convention used by HF
    fast tokenizers) and are always labelled ``O``. Tokens fully covered by a span
    receive the span's tag; a span covering a single token becomes ``S-``.
    """
    tags = [OUTSIDE] * len(offsets)
    for span in spans:
        member_idx = [
            i
            for i, (s, e) in enumerate(offsets)
            if not (s == 0 and e == 0)  # skip special tokens
            and s >= span.start
            and e <= span.end
            and e > s
        ]
        if not member_idx:
            continue
        if len(member_idx) == 1:
            tags[member_idx[0]] = f"S-{span.label}"
            continue
        first, last = member_idx[0], member_idx[-1]
        tags[first] = f"B-{span.label}"
        tags[last] = f"E-{span.label}"
        for i in member_idx[1:-1]:
            tags[i] = f"I-{span.label}"
    return tags


def bioes_to_spans(tags: Sequence[str], offsets: Sequence[Offset]) -> List[Span]:
    """Decode a BIOES tag sequence back into character spans.

    Tolerant of mildly malformed sequences (e.g. an ``I-`` with no opening ``B-``):
    a run is closed whenever the entity type changes, an ``S-``/``E-`` is seen, or
    an ``O`` is reached. This mirrors the lenient post-Viterbi reconstruction used
    by the inference engine.
    """
    spans: List[Span] = []
    cur_label: str | None = None
    cur_start: int | None = None
    cur_end: int | None = None

    def flush() -> None:
        nonlocal cur_label, cur_start, cur_end
        if cur_label is not None and cur_start is not None and cur_end is not None:
            spans.append(Span(cur_label, cur_start, cur_end))
        cur_label = cur_start = cur_end = None

    for tag, (s, e) in zip(tags, offsets):
        if tag == OUTSIDE or (s == 0 and e == 0):
            flush()
            continue
        prefix, _, label = tag.partition("-")
        if prefix == "S":
            flush()
            spans.append(Span(label, s, e))
            continue
        if prefix == "B":
            flush()
            cur_label, cur_start, cur_end = label, s, e
            continue
        if prefix in ("I", "E"):
            if cur_label == label:
                cur_end = e
            else:  # orphan I-/E-: start a fresh span rather than drop it
                flush()
                cur_label, cur_start, cur_end = label, s, e
            if prefix == "E":
                flush()
    flush()
    return spans


# Valid BIOES transitions, used both to constrain the Viterbi decoder and to
# validate hand-authored / synthetic label sequences in tests.
def is_valid_transition(prev: str, nxt: str) -> bool:
    """Return whether ``prev -> nxt`` is a legal BIOES transition."""
    p_pref, _, p_lab = prev.partition("-")
    n_pref, _, n_lab = nxt.partition("-")
    prev_inside = prev != OUTSIDE and p_pref in ("B", "I")
    if prev == OUTSIDE or p_pref in ("E", "S"):
        # Outside a span: may only stay O, start a new B-, or emit a single S-.
        return nxt == OUTSIDE or n_pref in ("B", "S")
    if prev_inside:
        # Inside a span: must continue (I-/E-) with the *same* entity type.
        return n_pref in ("I", "E") and n_lab == p_lab
    return False


def validate_bioes_sequence(tags: Sequence[str]) -> bool:
    """Return True iff every adjacent pair in ``tags`` is a legal transition."""
    for prev, nxt in zip(tags, tags[1:]):
        if not is_valid_transition(prev, nxt):
            return False
    # A sequence ending mid-span (B-/I- with no closing E-) is invalid.
    if tags:
        last_pref = tags[-1].partition("-")[0]
        if last_pref in ("B", "I"):
            return False
    return True
