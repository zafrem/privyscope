"""Stage 2 — ONNX NER runtime (SRS §3.4, §8.1).

Loads the INT8 ONNX graph + tokenizer + engine metadata, runs a single forward
pass to obtain BIOES emissions, then constrained-Viterbi decodes them into
character spans. No PyTorch at runtime (NFR / §2.4).
"""
from __future__ import annotations

import json
import os
import unicodedata
from pathlib import Path
from typing import List, Sequence, Tuple

# privyscope runs torch-free (ONNX Runtime + fast tokenizers). Tell transformers
# not to probe for a PyTorch/TF backend so it stays silent on import; otherwise
# it prints "[transformers] PyTorch was not found / Disabling PyTorch ..." noise.
os.environ.setdefault("USE_TORCH", "0")
os.environ.setdefault("USE_TF", "0")

import numpy as np

from .bioes import Span, bioes_to_spans, coalesce_adjacent
from .decoder import viterbi_decode


def _reconstruct_offsets(text, tokenizer, input_ids) -> List[Tuple[int, int]]:
    """Best-effort char offsets for a *slow* tokenizer that omits offset_mapping.

    The Japanese ``BertJapaneseTokenizer`` (MeCab) has no fast variant, so it never
    returns ``offset_mapping`` and ``predict`` would ``KeyError`` — the whole
    ``redact()`` path crashes for ja. It also NFKC-normalizes text before tokenizing
    (``MecabTokenizer(normalize_text=True)``): full-width digits/latin/punct fold to
    half-width (``０９０`` -> ``09``, ``＠`` -> ``@``, ``－`` -> ``-``), so token surfaces
    do NOT appear verbatim in the raw text. We therefore search in an NFKC-normalized
    copy and map each match back to original character indices via ``norm_to_orig``.
    A naive ``text.find`` on the raw text drops every full-width span to ``(0, 0)``,
    silently losing full-width IDs/phones — common in real Japanese input. Special or
    unalignable tokens map to ``(0, 0)`` and decode as ``O``. Mirrors the MLM training
    path (``domain_mlm._reconstruct_offsets``) so training and inference align.
    """
    special = set(tokenizer.all_special_tokens)
    tokens = tokenizer.convert_ids_to_tokens(list(input_ids))
    # Normalize per original char so normalized positions map back to the raw text.
    # NFKC is mostly 1:1 but a few chars expand to several; norm_to_orig records, for
    # each normalized char, which original char index it came from.
    norm_parts: List[str] = []
    norm_to_orig: List[int] = []
    for oi, ch in enumerate(text):
        for nc in unicodedata.normalize("NFKC", ch):
            norm_parts.append(nc)
            norm_to_orig.append(oi)
    norm = "".join(norm_parts)
    offsets: List[Tuple[int, int]] = []
    cursor = 0  # position within the normalized string
    for tok in tokens:
        if tok in special:
            offsets.append((0, 0))
            continue
        surface = tok[2:] if tok.startswith("##") else tok
        idx = norm.find(surface, cursor) if surface else -1
        if idx == -1:
            offsets.append((0, 0))
            continue
        start = norm_to_orig[idx]
        end = norm_to_orig[idx + len(surface) - 1] + 1
        offsets.append((start, end))
        cursor = idx + len(surface)
    return offsets


class NerRuntime:
    """ONNX-backed BIOES NER engine for one language."""

    def __init__(self, bundle_dir: str | Path, providers: Sequence[str] | None = None) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.dir = Path(bundle_dir)
        meta = json.loads((self.dir / "privyscope_meta.json").read_text(encoding="utf-8"))
        self.labels: List[str] = meta["labels"]
        self.default_biases: List[float] = meta["transition_biases"]
        self.max_length: int = int(meta.get("max_length", 256))
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.dir), use_fast=True)
        onnx_path = self.dir / meta["onnx_file"]
        self.session = ort.InferenceSession(
            str(onnx_path), providers=list(providers) if providers else ["CPUExecutionProvider"]
        )

    @staticmethod
    def available_providers() -> List[str]:
        import onnxruntime as ort

        return list(ort.get_available_providers())

    def predict(self, text: str, biases: Sequence[float]) -> Tuple[List[Span], bool]:
        """Return (spans, decoded_mismatch) for one input string."""
        enc = self.tokenizer(
            text,
            return_offsets_mapping=self.tokenizer.is_fast,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        input_ids = enc["input_ids"][0]
        emissions = self.session.run(
            None,
            {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            },
        )[0][0]  # [T, L]
        path = viterbi_decode(emissions, self.labels, biases)
        tags = [self.labels[i] for i in path]
        if self.tokenizer.is_fast:  # fast tokenizers give exact char offsets
            offset_pairs = [(int(s), int(e)) for s, e in enc["offset_mapping"][0]]
        else:  # slow (ja MeCab): rebuild offsets so redact() works instead of crashing
            offset_pairs = _reconstruct_offsets(text, self.tokenizer, input_ids.tolist())
        spans = coalesce_adjacent(bioes_to_spans(tags, offset_pairs))
        return spans, self._decode_mismatch(text, enc["input_ids"][0])

    def _decode_mismatch(self, text: str, input_ids) -> bool:
        """True when the tokenizer cannot round-trip the input (FR-2.2).

        The flag exists to catch genuine information loss (an ``[UNK]`` swallowing a
        rare glyph) that would make offsets unreliable. It must NOT fire on the
        transformations the tokenizer applies *by design*: WordPiece re-inserts spaces
        between CJK chars, ``do_lower_case`` folds ``Kakao`` -> ``kakao``, and NFKC
        folds full-width forms. Those leave character offsets intact, so normalize them
        out (NFKC + casefold + whitespace) before comparing — otherwise ordinary
        Latin-in-CJK text (a brand name inside a Chinese sentence) false-positives.
        """
        decoded = self.tokenizer.decode(input_ids, skip_special_tokens=True)

        def norm(s: str) -> str:
            return "".join(unicodedata.normalize("NFKC", s).casefold().split())

        return norm(decoded) != norm(text)
