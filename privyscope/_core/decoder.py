"""Constrained Viterbi span decoder (SRS §3.4 Stage 2, §3.5).

Numpy-only — no torch dependency at inference time. Operates on ONNX emission
logits ``[T, L]`` plus the six transition biases (the operating point). Illegal
BIOES transitions are masked by :func:`build_transition_matrix`.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .transitions import build_transition_matrix


def viterbi_decode(
    emissions: np.ndarray, labels: Sequence[str], biases: Sequence[float]
) -> List[int]:
    """Return the globally optimal label-id path under BIOES constraints.

    ``emissions``: ``[T, L]`` float array; ``biases``: six category biases.
    """
    T, L = emissions.shape
    if T == 0:
        return []
    trans = np.asarray(build_transition_matrix(labels, biases), dtype=np.float64)  # [L, L]
    score = emissions[0].astype(np.float64).copy()
    back = np.zeros((T, L), dtype=np.int64)
    for t in range(1, T):
        # all_scores[j, k] = score[j] + trans[j, k]
        all_scores = score[:, None] + trans
        best_prev = all_scores.argmax(axis=0)          # [L]
        score = all_scores[best_prev, np.arange(L)] + emissions[t]
        back[t] = best_prev
    last = int(score.argmax())
    path = [last]
    for t in range(T - 1, 0, -1):
        last = int(back[t, last])
        path.append(last)
    path.reverse()
    return path
