"""Stage 1 — Regex Filter (SRS §3.4).

Loads ``regex_rules.yaml`` and returns structurally-obvious PII spans. Overlaps
between rules are resolved by ``priority`` (then by longer match). The rule file
is user-extensible without code changes (SRS §3.4).

Rules are compiled from the pii-pattern-engine ruleset by each language pack's
``scripts/gen_regex_rules.py``. Beyond a pattern, a compiled rule may carry a
``verify:`` function name; the match is only emitted if that validator accepts it
(see :mod:`privyscope._core.verification`). Hand-written rule files that predate
the pattern engine omit ``verify:`` and still load unchanged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import yaml

from . import verification
from .bioes import Span


@dataclass
class RegexRule:
    label: str
    pattern: "re.Pattern[str]"
    priority: int = 0
    verify: Optional[str] = None   # verification fn name; None = accept every match
    rule_id: str = ""              # upstream pattern id, for debugging


class RegexFilter:
    def __init__(self, rules: Sequence[RegexRule]) -> None:
        self.rules = list(rules)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RegexFilter":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        rules = [
            RegexRule(
                label=r["label"],
                pattern=re.compile(r["pattern"]),
                priority=int(r.get("priority", 0)),
                verify=r.get("verify"),
                rule_id=r.get("id", ""),
            )
            for r in data.get("rules", [])
        ]
        return cls(rules)

    def find(self, text: str) -> List[Span]:
        """Return non-overlapping spans, preferring higher priority / longer match."""
        candidates: List[tuple[int, int, int, str]] = []  # (start, end, priority, label)
        for rule in self.rules:
            for m in rule.pattern.finditer(text):
                if m.end() <= m.start():
                    continue
                # Checksum/dictionary gate: drops matches that fit the shape but are
                # not real (bad RRN checksum, non-surname Hangul, etc.).
                if rule.verify and not verification.verify(rule.verify, m.group()):
                    continue
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
