"""Evaluation dataset loading (SRS FR-2.4). Standalone JSONL reader.

Record schema matches the synthesis / fine-tuning format:
    {"text": "...", "spans": [{"label": "PER", "start": 0, "end": 3}, ...]}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from .._core.bioes import Span


def read_eval_jsonl(
    path: str | Path, limit: Optional[int] = None
) -> Iterator[Tuple[str, List[Span]]]:
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            if limit is not None and i >= limit:
                break
            obj = json.loads(line)
            text = obj["text"]
            spans = [
                Span(s["label"], int(s["start"]), int(s["end"]))
                for s in obj.get("spans", [])
            ]
            yield text, spans
