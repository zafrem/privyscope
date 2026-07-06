"""Two-stage hybrid pipeline + Union merge (SRS §3.4).

Stage 1 (regex) and Stage 2 (NER) spans are combined with a Union strategy.
On overlap, the structural regex span wins (higher precision for fixed formats);
otherwise the longer span is kept. Output assembly (placeholders, redacted text,
output modes, entity filtering) lives here.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from .bioes import Span
from .schema import DetectedSpan, RedactionResult, build_redacted_text

REDACTED_PLACEHOLDER = "<REDACTED>"


def merge_union(regex_spans: Sequence[Span], ner_spans: Sequence[Span]) -> List[Span]:
    """Union-merge two span sets, resolving overlaps in favour of regex/longer."""
    # Tag origin so regex (structural) wins ties on overlap.
    tagged = [(s, 1) for s in regex_spans] + [(s, 0) for s in ner_spans]
    # Resolve: higher origin first, then longer, then earlier.
    tagged.sort(key=lambda t: (-t[1], -(t[0].end - t[0].start), t[0].start))
    kept: List[Span] = []
    for span, _origin in tagged:
        if any(not (span.end <= k.start or span.start >= k.end) for k in kept):
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
