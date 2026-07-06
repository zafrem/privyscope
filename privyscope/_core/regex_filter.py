"""Stage 1 — Regex Filter (SRS §3.4).

Loads ``regex_rules.yaml`` and returns structurally-obvious PII spans. Overlaps
between rules are resolved by ``priority`` (then by longer match). The rule file
is user-extensible without code changes (SRS §3.4).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import yaml

from .bioes import Span


@dataclass
class RegexRule:
    label: str
    pattern: "re.Pattern[str]"
    priority: int = 0


class RegexFilter:
    def __init__(self, rules: Sequence[RegexRule]) -> None:
        self.rules = list(rules)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RegexFilter":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        rules = [
            RegexRule(r["label"], re.compile(r["pattern"]), int(r.get("priority", 0)))
            for r in data.get("rules", [])
        ]
        return cls(rules)

    def find(self, text: str) -> List[Span]:
        """Return non-overlapping spans, preferring higher priority / longer match."""
        candidates: List[tuple[int, int, int, str]] = []  # (start, end, priority, label)
        for rule in self.rules:
            for m in rule.pattern.finditer(text):
                if m.end() > m.start():
                    candidates.append((m.start(), m.end(), rule.priority, rule.label))
        # Greedy resolution: sort by priority desc, then length desc, then position.
        candidates.sort(key=lambda c: (-c[2], -(c[1] - c[0]), c[0]))
        chosen: List[Span] = []
        taken: List[tuple[int, int]] = []
        for start, end, _prio, label in candidates:
            if any(not (end <= s or start >= e) for s, e in taken):
                continue  # overlaps an already-chosen, higher-priority span
            chosen.append(Span(label, start, end))
            taken.append((start, end))
        chosen.sort(key=lambda s: s.start)
        return chosen
