"""Per-text language auto-detection heuristics."""
import pytest

from privyscope._core.detect import detect_language

ALL = ["ko", "en", "ja", "zh"]


def test_hangul_routes_ko():
    assert detect_language("홍길동 010-1234-5678", ALL) == "ko"


def test_latin_routes_en():
    assert detect_language("John Smith 555-123-4567", ALL) == "en"


def test_kana_routes_ja():
    assert detect_language("山田たろう です", ALL) == "ja"


def test_han_without_kana_routes_zh():
    assert detect_language("张伟在北京", ALL) == "zh"


def test_restricted_to_available():
    # Hangul text, but ko not installed → falls back, not ko.
    assert detect_language("홍길동", ["en"]) == "en"  # sole available
    assert detect_language("홍길동", ["en", "ja"]) is None  # ambiguous, ko absent


def test_env_fallback(monkeypatch):
    monkeypatch.setenv("PRIVYSCOPE_LANG", "en")
    assert detect_language("12345 !!!", ["en", "ja"]) == "en"


def test_no_signal_no_env_returns_none(monkeypatch):
    monkeypatch.delenv("PRIVYSCOPE_LANG", raising=False)
    assert detect_language("12345 !!!", ["en", "ja"]) is None


def test_distinctive_script_beats_latin():
    # Korean text with an embedded email/URL stays ko: Latin is the weak signal.
    assert detect_language("홍길동 010-1234-5678 a@b.com", ["ko", "en"]) == "ko"
    # Pure Latin (no distinctive script) → en.
    assert detect_language("Meet John at 555-123-4567 tomorrow", ["ko", "en"]) == "en"


@pytest.mark.parametrize("text", ["", "   ", "@@@ ###"])
def test_sole_available_default(text):
    assert detect_language(text, ["ko"]) == "ko"


# -- Simplified vs Traditional Chinese (zh-Hans / zh-Hant) ----------------

ZH = ["zh-Hans", "zh-Hant", "en", "ko"]


def test_simplified_routes_zh_hans():
    assert detect_language("张伟说这个国家很现代", ZH) == "zh-Hans"
    assert detect_language("广州市浦东新区，电话13812345678", ZH) == "zh-Hans"


def test_traditional_routes_zh_hant():
    assert detect_language("張偉說這個國家很現代", ZH) == "zh-Hant"
    assert detect_language("臺北市中正區，電話0912345678", ZH) == "zh-Hant"


def test_zh_tie_defaults_to_simplified():
    # Only characters shared by both forms → no signal → default Simplified.
    assert detect_language("中文", ZH) == "zh-Hans"


def test_zh_single_form_installed_wins():
    # Traditional text but only Simplified installed → Simplified (sole zh form).
    assert detect_language("張偉", ["zh-Hans", "en"]) == "zh-Hans"
    assert detect_language("张伟", ["zh-Hant", "en"]) == "zh-Hant"


def test_legacy_zh_still_supported():
    assert detect_language("张伟在北京", ["zh", "en"]) == "zh"
