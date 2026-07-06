"""Canonical PII entity codes and BIOES label-space construction.

Source of truth: PRIVYSCOPE-SRS-001 §6 and PRIVYSCOPE-ENTITY-SPEC-001 §2.
This module is shared by the synthesis engine, the domain-adaptive MLM probes,
and the NER fine-tuning label schema so that the same taxonomy is used end to end.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# The eight base entity codes (SRS §6). Order is significant: it fixes the index
# layout of the BIOES classification head ([T, 1 + N*4], SRS §3.5).
BASE_ENTITIES: Tuple[str, ...] = (
    "PER",      # Person name            (contextual)
    "PHONE",    # Phone number           (structural)
    "ID_NUM",   # National / govt ID     (structural + hybrid)
    "EMAIL",    # Email address          (structural)
    "LOC",      # Detailed address       (contextual + hybrid)
    "BANK",     # Financial information  (structural)
    "DATE",     # Private date           (hybrid)
    "SECRET",   # Credentials / secrets  (structural + hybrid)
)

# Detection strategy per entity (ENTITY_SPEC §2). Drives which pipeline layer
# owns the entity at inference time.
DETECTION_STRATEGY: Dict[str, str] = {
    "PER": "contextual",
    "PHONE": "structural",
    "ID_NUM": "structural",
    "EMAIL": "structural",
    "LOC": "contextual",
    "BANK": "structural",
    "DATE": "hybrid",
    "SECRET": "hybrid",
}

# BIOES prefixes. "O" is the single background tag; the four affix prefixes are
# applied per entity class.
OUTSIDE = "O"
BIOES_PREFIXES: Tuple[str, ...] = ("B", "I", "E", "S")


def build_label_list(entities: Tuple[str, ...] | List[str] = BASE_ENTITIES) -> List[str]:
    """Return the ordered BIOES label list: ['O', 'B-PER', 'I-PER', ...].

    Layout matches the head output shape ``[T, 1 + N*4]`` from SRS §3.5:
    index 0 is ``O``; entity ``e`` (0-based) occupies indices
    ``1 + 4*e .. 1 + 4*e + 3`` in (B, I, E, S) order.
    """
    labels: List[str] = [OUTSIDE]
    for ent in entities:
        for prefix in BIOES_PREFIXES:
            labels.append(f"{prefix}-{ent}")
    return labels


def build_label_maps(
    entities: Tuple[str, ...] | List[str] = BASE_ENTITIES,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Return (label2id, id2label) for the BIOES label space."""
    labels = build_label_list(entities)
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for i, lab in enumerate(labels)}
    return label2id, id2label


def entity_of(label: str) -> str | None:
    """Return the entity code for a BIOES label, or ``None`` for ``O``."""
    if label == OUTSIDE:
        return None
    return label.split("-", 1)[1]


def prefix_of(label: str) -> str | None:
    """Return the BIOES prefix (B/I/E/S) for a label, or ``None`` for ``O``."""
    if label == OUTSIDE:
        return None
    return label.split("-", 1)[0]
