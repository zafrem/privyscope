"""Output schema + eval metric tests (fixture ruleset)."""
from privyscope._core.bioes import Span
from privyscope._core.schema import DetectedSpan, build_redacted_text
from privyscope._eval.metrics import TypedAccumulator, UntypedAccumulator


def test_schema_shape(engine):
    d = engine.redact("555-123-4567 user@example.com").to_dict()
    assert set(d) == {"schema_version", "summary", "text", "detected_spans", "redacted_text"}
    assert set(d["summary"]) == {"output_mode", "span_count", "by_label", "decoded_mismatch"}
    for s in d["detected_spans"]:
        assert set(s) == {"label", "start", "end", "text", "placeholder"}


def test_build_redacted_text_right_to_left():
    text = "ab CDE fg"
    spans = [DetectedSpan("X", 0, 2, "ab", "<X>"), DetectedSpan("Y", 3, 6, "CDE", "<Y>")]
    assert build_redacted_text(text, spans) == "<X> <Y> fg"


def test_typed_metrics_strict():
    acc = TypedAccumulator()
    gold = [Span("PER", 0, 3), Span("PHONE", 4, 17)]
    pred = [Span("PHONE", 4, 17)]  # missed PER
    acc.update(gold, pred)
    res = acc.result()
    assert res["micro"]["tp"] == 1 and res["micro"]["fn"] == 1
    assert res["per_label"]["PER"]["recall"] == 0.0


def test_untyped_label_recall():
    acc = UntypedAccumulator()
    gold = [Span("given_name", 0, 3), Span("mobile", 4, 17)]
    pred = [Span("PER", 0, 3)]  # offsets match given_name only
    acc.update(gold, pred)
    res = acc.result()
    assert res["ground_truth_label_recall"]["given_name"] == 1.0
    assert res["ground_truth_label_recall"]["mobile"] == 0.0
