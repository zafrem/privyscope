"""Constrained-BIOES transition scheme with 6 learnable bias parameters.

SRS §3.5 / FR-2.5: the span decoder is a constrained Viterbi whose only free
parameters are six transition biases, each covering one *category* of legal
BIOES transition. Illegal transitions are hard-masked to ``-inf``. Operating
points (high_recall / balanced / high_precision) are just different settings of
these six biases — no retraining needed (FR-2.5).

This module is pure-Python (no numpy/torch) so it can be shared verbatim by the
torch training code and copied into the dependency-light public inference
engine.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .entities import OUTSIDE

NEG_INF = -1.0e4  # finite sentinel keeps Viterbi numerically stable

# The six transition categories, indexed 0..5.
CATEGORIES = ("outside", "begin", "single", "inside", "end", "exit")

# Default ("balanced") biases. Higher -> that transition is more encouraged.
DEFAULT_BIASES: Dict[str, float] = {
    "outside": 0.0,
    "begin": 0.0,
    "single": 0.0,
    "inside": 0.0,
    "end": 0.0,
    "exit": 0.0,
}

# Operating-point presets (FR-2.5). high_recall encourages entering and
# continuing spans; high_precision is conservative about span entry.
OPERATING_POINTS: Dict[str, Dict[str, float]] = {
    "high_recall": {"outside": -0.5, "begin": 1.0, "single": 1.0, "inside": 0.8, "end": 0.3, "exit": -0.3},
    "balanced": dict(DEFAULT_BIASES),
    "high_precision": {"outside": 0.5, "begin": -1.0, "single": -1.0, "inside": 0.2, "end": 0.5, "exit": 0.5},
}


def _prefix(label: str) -> str:
    return OUTSIDE if label == OUTSIDE else label.partition("-")[0]


def _entity(label: str) -> Optional[str]:
    return None if label == OUTSIDE else label.partition("-")[2]


def categorize(prev: str, nxt: str) -> Optional[int]:
    """Return the category index (0..5) for ``prev -> nxt``, or None if illegal."""
    pp, np_ = _prefix(prev), _prefix(nxt)
    prev_outside = pp in (OUTSIDE, "E", "S")
    if prev_outside:
        if np_ == OUTSIDE:
            return CATEGORIES.index("exit") if pp in ("E", "S") else CATEGORIES.index("outside")
        if np_ == "B":
            return CATEGORIES.index("begin")
        if np_ == "S":
            return CATEGORIES.index("single")
        return None  # O/E/S may not be followed by I- or E-
    # prev is inside a span (B- or I-): only same-entity I-/E- may follow.
    if np_ in ("I", "E") and _entity(prev) == _entity(nxt):
        return CATEGORIES.index("inside" if np_ == "I" else "end")
    return None


def category_grid(labels: Sequence[str]) -> List[List[Optional[int]]]:
    """Precompute the category index (or None) for every (prev, next) label pair."""
    return [[categorize(p, n) for n in labels] for p in labels]


def build_transition_matrix(
    labels: Sequence[str], biases: Sequence[float], neg_inf: float = NEG_INF
) -> List[List[float]]:
    """Assemble the dense ``[L][L]`` additive transition-score matrix.

    ``biases`` is the six category biases in :data:`CATEGORIES` order.
    """
    if len(biases) != len(CATEGORIES):
        raise ValueError(f"expected {len(CATEGORIES)} biases, got {len(biases)}")
    grid = category_grid(labels)
    return [
        [neg_inf if cat is None else float(biases[cat]) for cat in row]
        for row in grid
    ]


def biases_dict_to_list(biases: Dict[str, float]) -> List[float]:
    """Convert a category->bias dict into the canonical six-element ordering."""
    return [float(biases.get(c, DEFAULT_BIASES[c])) for c in CATEGORIES]


def resolve_operating_point(name_or_biases) -> List[float]:
    """Resolve an operating point name or a raw biases mapping/sequence to a list."""
    if name_or_biases is None:
        return biases_dict_to_list(DEFAULT_BIASES)
    if isinstance(name_or_biases, str):
        if name_or_biases not in OPERATING_POINTS:
            raise KeyError(
                f"unknown operating point {name_or_biases!r}; "
                f"choose from {sorted(OPERATING_POINTS)} or pass raw biases"
            )
        return biases_dict_to_list(OPERATING_POINTS[name_or_biases])
    if isinstance(name_or_biases, dict):
        return biases_dict_to_list(name_or_biases)
    biases = list(name_or_biases)
    if len(biases) != len(CATEGORIES):
        raise ValueError(f"expected {len(CATEGORIES)} biases, got {len(biases)}")
    return [float(b) for b in biases]
