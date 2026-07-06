"""Constrained Viterbi decoder + transition scheme tests."""
import numpy as np

from privyscope._core.bioes import bioes_to_spans
from privyscope._core.decoder import viterbi_decode
from privyscope._core.entities import build_label_list
from privyscope._core.transitions import (
    OPERATING_POINTS,
    build_transition_matrix,
    categorize,
    resolve_operating_point,
)

LABELS = build_label_list()


def _emissions_for(tag_seq):
    """One-hot-ish emissions that strongly favour the given gold tags."""
    idx = {l: i for i, l in enumerate(LABELS)}
    emis = np.full((len(tag_seq), len(LABELS)), -5.0)
    for t, tag in enumerate(tag_seq):
        emis[t, idx[tag]] = 5.0
    return emis


def test_viterbi_recovers_clean_path():
    gold = ["O", "B-PER", "I-PER", "E-PER", "O"]
    path = viterbi_decode(_emissions_for(gold), LABELS, resolve_operating_point("balanced"))
    assert [LABELS[i] for i in path] == gold


def test_viterbi_respects_constraints():
    # Emissions favour an illegal O -> I-PER jump; decoder must avoid it.
    emis = _emissions_for(["O", "I-PER", "O"])
    path = [LABELS[i] for i in viterbi_decode(emis, LABELS, resolve_operating_point("balanced"))]
    for prev, nxt in zip(path, path[1:]):
        assert categorize(prev, nxt) is not None


def test_high_recall_enters_more_than_high_precision():
    # A leading O (like [CLS]) then an ambiguous token mildly preferring S-PHONE.
    # The operating point biases the O -> {O, S-PHONE} transition into the token.
    idx = {l: i for i, l in enumerate(LABELS)}
    emis = np.zeros((2, len(LABELS)))
    emis[0, idx["O"]] = 5.0
    emis[1, idx["S-PHONE"]] = 0.2
    hr = viterbi_decode(emis, LABELS, resolve_operating_point("high_recall"))
    hp = viterbi_decode(emis, LABELS, resolve_operating_point("high_precision"))
    assert LABELS[hr[1]].startswith("S-")        # recall enters the span
    assert LABELS[hp[1]] == "O"                  # precision stays out


def test_all_operating_points_resolve():
    for name in OPERATING_POINTS:
        assert len(resolve_operating_point(name)) == 6
    assert len(resolve_operating_point([0, 0, 0, 0, 0, 0])) == 6


def test_decode_to_spans_roundtrip():
    gold = ["O", "S-EMAIL", "O"]
    offsets = [(0, 1), (1, 16), (16, 17)]
    path = viterbi_decode(_emissions_for(gold), LABELS, resolve_operating_point("balanced"))
    spans = bioes_to_spans([LABELS[i] for i in path], offsets)
    assert spans and spans[0].label == "EMAIL" and (spans[0].start, spans[0].end) == (1, 16)
