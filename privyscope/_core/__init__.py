"""Runtime core: loading, regex/NER stages, decoding, span assembly."""
from .bioes import Span, bioes_to_spans, spans_to_bioes
from .pipeline import assemble, merge_union
from .regex_filter import RegexFilter
from .schema import DetectedSpan, RedactionResult, SCHEMA_VERSION
from .transitions import OPERATING_POINTS, resolve_operating_point

__all__ = [
    "Span",
    "bioes_to_spans",
    "spans_to_bioes",
    "assemble",
    "merge_union",
    "RegexFilter",
    "DetectedSpan",
    "RedactionResult",
    "SCHEMA_VERSION",
    "OPERATING_POINTS",
    "resolve_operating_point",
]
