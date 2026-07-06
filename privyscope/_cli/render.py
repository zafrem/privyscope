"""Terminal rendering for the CLI (SRS FR-2.1 ANSI span previews)."""
from __future__ import annotations

import json
import sys
from typing import List

from .._core.schema import RedactionResult

# Per-label ANSI background colours; falls back to a default for custom labels.
_COLORS = {
    "PER": "\033[48;5;31m", "PHONE": "\033[48;5;130m", "ID_NUM": "\033[48;5;90m",
    "EMAIL": "\033[48;5;28m", "LOC": "\033[48;5;24m", "BANK": "\033[48;5;94m",
    "DATE": "\033[48;5;55m", "SECRET": "\033[48;5;124m",
}
_DEFAULT_COLOR = "\033[48;5;240m"
_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"


def supports_color(stream=sys.stdout) -> bool:
    return hasattr(stream, "isatty") and stream.isatty()


def highlight(result: RedactionResult, color: bool = True) -> str:
    """Return the input with detected spans visually marked."""
    text = result.text
    out: List[str] = []
    cursor = 0
    for s in sorted(result.detected_spans, key=lambda d: d.start):
        out.append(text[cursor : s.start])
        chunk = text[s.start : s.end]
        if color:
            c = _COLORS.get(s.label, _DEFAULT_COLOR)
            out.append(f"{c}{chunk}{_RESET}{_DIM}[{s.label}]{_RESET}")
        else:
            out.append(f"[{chunk}]({s.label})")
        cursor = s.end
    out.append(text[cursor:])
    return "".join(out)


def render_human(result: RedactionResult, color: bool = True) -> str:
    lines = [
        highlight(result, color),
        "",
        f"{_BOLD if color else ''}redacted:{_RESET if color else ''} {result.redacted_text}",
        f"{_DIM if color else ''}spans: {result.summary['span_count']}  "
        f"by_label: {result.summary['by_label']}"
        + ("  (decode mismatch)" if result.decoded_mismatch else "")
        + (_RESET if color else ""),
    ]
    return "\n".join(lines)


def render_json(result: RedactionResult) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)


def print_result(result: RedactionResult, as_json: bool) -> None:
    if as_json:
        print(render_json(result))
    else:
        print(render_human(result, color=supports_color()))
