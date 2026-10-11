"""Plugin-declared trailing-particle trim for NER spans.

The Korean tokenizer has single tokens such as ``##일이`` / ``##일에`` (day + particle). The
model tags whole tokens, so a DATE ending in ``20일`` can only be emitted as ``20일이`` — the
true boundary is inside a token. Plugins declare the characters to trim in
``entity_config.yaml``; the core applies them at character level to NER spans.
"""
from privyscope._api import Privyscope
from privyscope._core.bioes import Span
from privyscope._core.pipeline import SpanTrim, trim_trailing
from privyscope._core.plugins import parse_span_trims
from privyscope._core.regex_filter import RegexFilter
from privyscope._core.transitions import CATEGORIES

DAY = SpanTrim(label="DATE", after="일", suffixes=("이", "에", "은", "을", "도", "로", "만"))
TEXT = "제 생일은 1990년 12월 20일이에요"
START = TEXT.index("1990")


def _span(text, label="DATE", end_after="일이"):
    return Span(label, START, text.index(end_after, START) + len(end_after))


def test_trims_particle_fused_after_day():
    out = trim_trailing([_span(TEXT)], TEXT, [DAY])
    assert [TEXT[s.start:s.end] for s in out] == ["1990년 12월 20일"]


def test_each_declared_particle_is_trimmed():
    for p in DAY.suffixes:
        text = f"생일은 1990년 12월 20일{p} 좋다"
        start = text.index("1990")
        s = Span("DATE", start, text.index("일" + p, start) + 2)
        assert text[trim_trailing([s], text, [DAY])[0].end - 1] == "일", p


def test_span_already_ending_at_the_day_is_untouched():
    s = Span("DATE", START, TEXT.index("일이") + 1)
    assert trim_trailing([s], TEXT, [DAY]) == [s]


def test_other_labels_are_untouched():
    s = _span(TEXT, label="PER")
    assert trim_trailing([s], TEXT, [DAY]) == [s]


def test_particle_not_preceded_by_the_anchor_is_untouched():
    text = "회의는 3시에 시작"          # DATE-labelled span ending in 에, but not 일+에
    s = Span("DATE", text.index("3"), text.index("에") + 1)
    assert trim_trailing([s], text, [DAY]) == [s]


def test_trims_once_and_never_empties_a_span():
    text = "일이"
    s = Span("DATE", 0, 2)
    out = trim_trailing([s], text, [DAY])
    assert out == [Span("DATE", 0, 1)]
    assert trim_trailing(out, text, [DAY]) == out


def test_no_trims_returns_spans_unchanged():
    s = _span(TEXT)
    assert trim_trailing([s], TEXT, []) == [s]


def test_parse_span_trims():
    data = {"span_trim": [
        {"label": "DATE", "after": "일", "suffixes": ["이", "에"]},
        {"label": "DATE", "after": "", "suffixes": ["이"]},          # malformed: no anchor
        {"label": "DATE", "after": "월"},                            # malformed: no suffixes
        "not-a-mapping",
    ]}
    assert parse_span_trims(data) == (SpanTrim("DATE", "일", ("이", "에")),)
    assert parse_span_trims({}) == ()


NAME = SpanTrim(label="PER", after="", suffixes=("가", "의"), min_keep=2)


def test_parse_anchorless_trim_needs_min_keep():
    ok = {"span_trim": [{"label": "PER", "suffixes": ["가"], "min_keep": 2}]}
    assert parse_span_trims(ok) == (SpanTrim("PER", "", ("가",), 2),)
    for bad in ({"label": "PER", "suffixes": ["가"]},                       # nothing anchors the cut
                {"label": "PER", "suffixes": ["가"], "min_keep": 1},
                {"label": "PER", "suffixes": ["가"], "min_keep": "2"},
                {"label": "PER", "suffixes": ["가"], "min_keep": True}):
        assert parse_span_trims({"span_trim": [bad]}) == ()


def test_anchorless_trim_removes_particle_after_a_name():
    text = "담당은 홍희지가 맡습니다"
    s = Span("PER", text.index("홍"), text.index("가") + 1)
    assert [text[x.start:x.end] for x in trim_trailing([s], text, [NAME])] == ["홍희지"]


def test_anchorless_trim_keeps_min_chars():
    text = "정가 안내"           # 2-char span: cutting 가 would leave one syllable
    s = Span("PER", 0, 2)
    assert trim_trailing([s], text, [NAME]) == [s]
    text = "신하의 책"           # 3 chars -> 2 remain, allowed
    assert trim_trailing([Span("PER", 0, 3)], text, [NAME]) == [Span("PER", 0, 2)]


class _FakeRuntime:
    """Stands in for the ONNX runtime: always tags the day token plus its particle."""
    default_biases = [0.0] * len(CATEGORIES)

    def predict(self, text, biases):
        end = text.index("일이") + 2
        return [Span("DATE", text.index("1990"), end)], False


def test_engine_applies_declared_trim_before_masking():
    rules = {"version": 2, "language": "ko", "rules": []}
    import tempfile, yaml, pathlib
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "r.yaml"
        p.write_text(yaml.safe_dump(rules), encoding="utf-8")
        regex = RegexFilter.from_yaml(p)
    plain = Privyscope(regex, _FakeRuntime())
    trimmed = Privyscope(regex, _FakeRuntime(), span_trims=[DAY])
    assert plain.redact(TEXT).redacted_text == "제 생일은 <DATE>에요"      # particle eaten
    assert trimmed.redact(TEXT).redacted_text == "제 생일은 <DATE>이에요"  # particle kept
