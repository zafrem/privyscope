"""Strict entity-level and span-level metrics (SRS FR-2.4, NFR-3).

* typed   -> per-label and micro Strict-F1 (offset + label must match).
* untyped -> span-level F1 (offset only) plus ``ground_truth_label_recall``
             per source label, for datasets using a different taxonomy.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from .._core.bioes import Span

Key = Tuple[str, int, int]


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn}


class TypedAccumulator:
    """Accumulates strict (label, start, end) matches for typed eval."""

    def __init__(self) -> None:
        self.per_label: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])  # tp, fp, fn

    def update(self, gold: Sequence[Span], pred: Sequence[Span]) -> None:
        g = {(s.label, s.start, s.end) for s in gold}
        p = {(s.label, s.start, s.end) for s in pred}
        for key in p & g:
            self.per_label[key[0]][0] += 1
        for key in p - g:
            self.per_label[key[0]][1] += 1
        for key in g - p:
            self.per_label[key[0]][2] += 1

    def result(self) -> Dict[str, object]:
        labels = {}
        TP = FP = FN = 0
        for label, (tp, fp, fn) in sorted(self.per_label.items()):
            labels[label] = _prf(tp, fp, fn)
            TP, FP, FN = TP + tp, FP + fp, FN + fn
        return {"micro": _prf(TP, FP, FN), "per_label": labels}


class UntypedAccumulator:
    """Span-level (label-agnostic) match plus per-source-label recall."""

    def __init__(self) -> None:
        self.tp = self.fp = self.fn = 0
        self.src_total: Dict[str, int] = defaultdict(int)
        self.src_hit: Dict[str, int] = defaultdict(int)

    def update(self, gold: Sequence[Span], pred: Sequence[Span]) -> None:
        g = {(s.start, s.end) for s in gold}
        p = {(s.start, s.end) for s in pred}
        self.tp += len(p & g)
        self.fp += len(p - g)
        self.fn += len(g - p)
        for s in gold:
            self.src_total[s.label] += 1
            if (s.start, s.end) in p:
                self.src_hit[s.label] += 1

    def result(self) -> Dict[str, object]:
        recall = {
            label: round(self.src_hit[label] / total, 4)
            for label, total in sorted(self.src_total.items())
            if total
        }
        return {"span_level": _prf(self.tp, self.fp, self.fn), "ground_truth_label_recall": recall}
