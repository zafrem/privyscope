"""Two-stage hybrid pipeline + Union merge (SRS §3.4).

Stage 1 (regex) and Stage 2 (NER) spans are combined with a Union strategy.
On overlap, the structural regex span wins (higher precision for fixed formats);
otherwise the longer span is kept. Output assembly (placeholders, redacted text,
output modes, entity filtering) lives here.
"""
from __future__ import annotations

from typing import AbstractSet, Iterable, List, Optional, Sequence

from .bioes import Span
from .schema import DetectedSpan, RedactionResult, build_redacted_text

REDACTED_PLACEHOLDER = "<REDACTED>"


def filter_stopwords(
    spans: Sequence[Span], text: str, stopwords: AbstractSet[str]
) -> List[Span]:
    """Drop spans whose surface text is exactly a denylisted common word.

    Offset-preserving: reads ``text`` but never mutates it, so surviving spans keep
    their original character offsets. Intended for the contextual NER stage, which
    can over-fire on ordinary words (e.g. 주민/문서 mis-tagged ``PER``); structural
    regex spans are validated upstream and are not passed through here.
    """
    if not stopwords:
        return list(spans)
    return [s for s in spans if text[s.start : s.end].strip() not in stopwords]


def _overlaps(a: Span, b: Span) -> bool:
    return not (a.end <= b.start or a.start >= b.end)


def merge_union(regex_spans: Sequence[Span], ner_spans: Sequence[Span]) -> List[Span]:
    """Union-merge two span sets, resolving overlaps in favour of regex/longer.

    A regex span keeps its (higher-precision) label on overlap, but when the
    overlapping NER span carries the *same* label the regex span is **extended**
    to the union of both ranges rather than suppressing the NER span. This stops a
    coarse structural matcher from truncating the model's wider extent — e.g. an
    address regex that stops at the road name (``…판교역로``) must not drop the NER
    tail (``…판교역로 166 판교푸르지오 101동 1204호``) and leak the building/unit.
    Cross-label overlaps still resolve to the regex span (structural precision).
    """
    regex_list = list(regex_spans)
    ner_list = list(ner_spans)
    absorbed: set[int] = set()  # NER spans merged into a same-label regex span
    extended: List[Span] = []
    for r in regex_list:
        start, end = r.start, r.end
        changed = True
        while changed:  # re-scan: an extension can reach a further same-label span
            changed = False
            for i, n in enumerate(ner_list):
                if i in absorbed or n.label != r.label:
                    continue
                if start < n.end and end > n.start:  # overlap
                    ns, ne = min(start, n.start), max(end, n.end)
                    if (ns, ne) != (start, end):
                        start, end, changed = ns, ne, True
                    absorbed.add(i)
        extended.append(r if (start, end) == (r.start, r.end) else Span(r.label, start, end))
    remaining_ner = [n for i, n in enumerate(ner_list) if i not in absorbed]

    # Resolve any residual overlaps: regex (structural) first, then longer, then earlier.
    tagged = [(s, 1) for s in extended] + [(s, 0) for s in remaining_ner]
    tagged.sort(key=lambda t: (-t[1], -(t[0].end - t[0].start), t[0].start))
    kept: List[Span] = []
    for span, _origin in tagged:
        if any(_overlaps(span, k) for k in kept):
            continue
        kept.append(span)
    kept.sort(key=lambda s: s.start)
    return kept


def assemble(
    text: str,
    spans: Sequence[Span],
    output_mode: str = "typed",
    entity_types: Optional[Iterable[str]] = None,
    decoded_mismatch: bool = False,
) -> RedactionResult:
    """Build the FR-2.2 result object from merged spans."""
    allowed = set(entity_types) if entity_types is not None else None
    detected: List[DetectedSpan] = []
    for s in spans:
        if allowed is not None and s.label not in allowed:
            continue
        if output_mode == "redacted":
            label, placeholder = "redacted", REDACTED_PLACEHOLDER
        else:
            label, placeholder = s.label, f"<{s.label}>"
        detected.append(DetectedSpan(label, s.start, s.end, text[s.start : s.end], placeholder))
    detected.sort(key=lambda d: d.start)
    return RedactionResult(
        text=text,
        detected_spans=detected,
        redacted_text=build_redacted_text(text, detected),
        output_mode=output_mode,
        decoded_mismatch=decoded_mismatch,
    )
