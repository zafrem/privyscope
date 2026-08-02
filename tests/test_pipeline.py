"""End-to-end pipeline tests in regex-only mode (fixture ruleset)."""
from privyscope._core.bioes import Span
from privyscope._core.pipeline import merge_union


def test_redact_basic(engine):
    r = engine.redact("call 555-123-4567 email user@example.com")
    assert r.redacted_text == "call <PHONE> email <EMAIL>"
    assert r.summary["by_label"] == {"EMAIL": 1, "PHONE": 1}
    assert r.schema_version == 1


def test_output_mode_redacted(engine):
    r = engine.redact("555-123-4567", output_mode="redacted")
    assert r.redacted_text == "<REDACTED>"
    assert r.detected_spans[0].label == "redacted"


def test_entity_filter(engine):
    r = engine.redact("555-123-4567 user@example.com", entity_types=["PHONE"])
    assert r.summary["by_label"] == {"PHONE": 1}


def test_batch(engine):
    rs = engine.batch_redact(["555-123-4567", "user@example.com"])
    assert [r.summary["span_count"] for r in rs] == [1, 1]


def test_merge_union_prefers_regex_on_overlap():
    regex = [Span("PHONE", 0, 12)]
    ner = [Span("PER", 5, 12)]  # overlaps -> dropped in favour of regex
    merged = merge_union(regex, ner)
    assert merged == [Span("PHONE", 0, 12)]


def test_merge_union_keeps_disjoint():
    merged = merge_union([Span("PHONE", 0, 5)], [Span("PER", 6, 9)])
    assert len(merged) == 2


def test_merge_union_extends_same_label_regex_to_ner_tail():
    # A coarse address regex stops early; the same-label NER span extends further.
    # The regex must not truncate the NER tail (else the building/unit leaks).
    regex = [Span("LOC", 0, 16)]        # "…판교역로"
    ner = [Span("LOC", 0, 20)]          # "…판교역로 166"
    assert merge_union(regex, ner) == [Span("LOC", 0, 20)]


def test_merge_union_cross_label_overlap_still_prefers_regex():
    # Different labels must NOT be merged — regex keeps its label and extent.
    regex = [Span("PHONE", 0, 12)]
    ner = [Span("BANK", 0, 20)]
    assert merge_union(regex, ner) == [Span("PHONE", 0, 12)]
