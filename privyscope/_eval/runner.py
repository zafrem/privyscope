"""Evaluation runner (SRS FR-2.4)."""
from __future__ import annotations

from typing import Optional

from .._api import Privyscope
from .._core.bioes import Span
from .dataset import read_eval_jsonl
from .metrics import TypedAccumulator, UntypedAccumulator


def run_eval(
    engine: Privyscope,
    dataset_path: str,
    eval_mode: str = "typed",
    limit: Optional[int] = None,
) -> dict:
    acc = TypedAccumulator() if eval_mode == "typed" else UntypedAccumulator()
    n = 0
    for text, gold in read_eval_jsonl(dataset_path, limit=limit):
        result = engine.redact(text)
        pred = [Span(d.label, d.start, d.end) for d in result.detected_spans]
        acc.update(gold, pred)
        n += 1
    report = {"dataset": dataset_path, "eval_mode": eval_mode, "examples": n, "ner_enabled": engine.has_ner}
    report.update(acc.result())
    return report
