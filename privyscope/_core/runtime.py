"""Stage 2 — ONNX NER runtime (SRS §3.4, §8.1).

Loads the INT8 ONNX graph + tokenizer + engine metadata, runs a single forward
pass to obtain BIOES emissions, then constrained-Viterbi decodes them into
character spans. No PyTorch at runtime (NFR / §2.4).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Sequence, Tuple

# privyscope runs torch-free (ONNX Runtime + fast tokenizers). Tell transformers
# not to probe for a PyTorch/TF backend so it stays silent on import; otherwise
# it prints "[transformers] PyTorch was not found / Disabling PyTorch ..." noise.
os.environ.setdefault("USE_TORCH", "0")
os.environ.setdefault("USE_TF", "0")

import numpy as np

from .bioes import Span, bioes_to_spans
from .decoder import viterbi_decode


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
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        offsets = enc["offset_mapping"][0]
        emissions = self.session.run(
            None,
            {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            },
        )[0][0]  # [T, L]
        path = viterbi_decode(emissions, self.labels, biases)
        tags = [self.labels[i] for i in path]
        offset_pairs = [(int(s), int(e)) for s, e in offsets]
        spans = bioes_to_spans(tags, offset_pairs)
        return spans, self._decode_mismatch(text, enc["input_ids"][0])

    def _decode_mismatch(self, text: str, input_ids) -> bool:
        """True when the tokenizer cannot round-trip the input (FR-2.2)."""
        decoded = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        norm = lambda s: "".join(s.split())  # noqa: E731  (ignore whitespace)
        return norm(decoded) != norm(text)
