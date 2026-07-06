"""Structured output schema (SRS FR-2.2, §10.3).

``schema_version`` MUST increment on any breaking change to this structure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

SCHEMA_VERSION = 1


@dataclass
class DetectedSpan:
    label: str
    start: int          # inclusive char offset
    end: int            # exclusive char offset
    text: str
    placeholder: str

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "placeholder": self.placeholder,
        }


@dataclass
class RedactionResult:
    """Result of a single ``redact`` call. Mirrors §10.3 exactly."""

    text: str
    detected_spans: List[DetectedSpan]
    redacted_text: str
    output_mode: str = "typed"
    decoded_mismatch: bool = False
    schema_version: int = SCHEMA_VERSION

    # -- convenience accessors (FR-2.7) ------------------------------------
    @property
    def masked_text(self) -> str:
        return self.redacted_text

    @property
    def summary(self) -> dict:
        by_label: Dict[str, int] = {}
        for s in self.detected_spans:
            by_label[s.label] = by_label.get(s.label, 0) + 1
        return {
            "output_mode": self.output_mode,
            "span_count": len(self.detected_spans),
            "by_label": dict(sorted(by_label.items())),
            "decoded_mismatch": self.decoded_mismatch,
        }

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "summary": self.summary,
            "text": self.text,
            "detected_spans": [s.to_dict() for s in self.detected_spans],
            "redacted_text": self.redacted_text,
        }


def build_redacted_text(text: str, spans: List[DetectedSpan]) -> str:
    """Replace each span with its placeholder, working right-to-left."""
    out = text
    for s in sorted(spans, key=lambda x: x.start, reverse=True):
        out = out[: s.start] + s.placeholder + out[s.end :]
    return out
