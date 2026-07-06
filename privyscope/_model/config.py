"""Label-space and checkpoint configuration (SRS FR-2.6, §8.2).

Supports custom label spaces so organisations can fine-tune on their own PII
taxonomy without touching model code: a ``--label-space-json`` file maps source
labels to privyscope (or custom) entity codes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .._core.entities import BASE_ENTITIES, build_label_maps

# Last-resort base checkpoint when neither --base-model nor the active language
# plugin's ``default_base_model`` is set. Language plugins supply the right one
# (e.g. klue/roberta-base for ko, roberta-base for en) (SRS §1.4).
DEFAULT_BASE_MODEL = "bert-base-multilingual-cased"


def load_label_space(
    label_space_json: Optional[str] = None,
) -> Tuple[List[str], Dict[str, int], Dict[int, str], Dict[str, str]]:
    """Return (bioes_labels, label2id, id2label, source->target label_map).

    Without a custom file, uses the eight base entities (SRS §6). With one, the
    target entity set is the *values* of ``label_map`` (de-duplicated, ordered).
    """
    if not label_space_json:
        label2id, id2label = build_label_maps(BASE_ENTITIES)
        from .._core.entities import build_label_list

        return build_label_list(BASE_ENTITIES), label2id, id2label, {}

    data = json.loads(Path(label_space_json).read_text(encoding="utf-8"))
    label_map: Dict[str, str] = data["label_map"]
    # Preserve first-seen order of target codes for a stable head layout.
    targets: List[str] = []
    for tgt in label_map.values():
        if tgt not in targets:
            targets.append(tgt)
    from .._core.entities import build_label_list

    labels = build_label_list(tuple(targets))
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for i, lab in enumerate(labels)}
    return labels, label2id, id2label, label_map


def remap_label(label: str, label_map: Dict[str, str]) -> Optional[str]:
    """Apply a custom label map; return None to drop unmapped source labels."""
    if not label_map:
        return label
    return label_map.get(label)
