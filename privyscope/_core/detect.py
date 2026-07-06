"""Per-text language auto-detection via Unicode-script heuristics.

Pure stdlib, no new dependencies. Detection is **restricted to the set of
installed/available language codes**, so the router never picks a language the
user has not installed. Used by :meth:`privyscope.Privyscope.auto` and the CLI when
multiple language plugins are present and no ``--lang`` was given.
"""
from __future__ import annotations

import os
import unicodedata
from typing import Dict, Iterable, List, Optional, Tuple


def _script_counts(text: str) -> Dict[str, int]:
    counts = {"Hangul": 0, "Hiragana": 0, "Katakana": 0, "Han": 0, "Latin": 0}
    for ch in text:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:  # unnamed char
            continue
        if "HANGUL" in name:
            counts["Hangul"] += 1
        elif "HIRAGANA" in name:
            counts["Hiragana"] += 1
        elif "KATAKANA" in name:
            counts["Katakana"] += 1
        elif "CJK" in name:
            counts["Han"] += 1
        elif "LATIN" in name:
            counts["Latin"] += 1
    return counts


# Characters that occur only in Simplified or only in Traditional Chinese, used
# to disambiguate zh-Hans vs zh-Hant — both are Han script, so the script scan
# alone cannot tell them apart. Not exhaustive: text built only from characters
# shared by both forms yields no signal and falls back to $PRIVYSCOPE_LANG /
# --lang (or, when only one zh form is installed, that one).
_SIMP_ONLY = frozenset(
    "们这国时还会个东车门长马问说见证过当后样经现边师应关习书实战报"
    "觉让认识语务写单价众节约级纪线练组细给继续维绿罗职专业达迁运"
    "电话来开学区湾体义医难头买卖红钱银铁极标楼龙岛县镇党团设计议"
    "讯询谁课读资际处备杂台湾湾们对发欢爱乐离开关闭历题点击网络营"
)
_TRAD_ONLY = frozenset(
    "們這國時還會個東車門長馬問說見證過當後樣經現邊師應關習書實戰報"
    "覺讓認識語務寫單價眾節約級紀線練組細給繼續維綠羅職專業達遷運"
    "電話來開學區灣體義醫難頭買賣紅錢銀鐵極標樓龍島縣鎮黨團設計議"
    "訊詢誰課讀資際處備雜臺灣彎們對發歡愛樂離開關閉歷題點擊網絡營"
)


def _simp_trad_counts(text: str) -> Tuple[int, int]:
    """Count Simplified-only vs Traditional-only characters in ``text``."""
    simp = trad = 0
    for ch in text:
        if ch in _SIMP_ONLY:
            simp += 1
        elif ch in _TRAD_ONLY:
            trad += 1
    return simp, trad


def _resolve_zh(text: str, avail: set) -> Optional[str]:
    """Pick the installed Chinese code (zh-Hans / zh-Hant, or legacy zh)."""
    has_hans = "zh-Hans" in avail
    has_hant = "zh-Hant" in avail
    if not (has_hans or has_hant):
        return "zh" if "zh" in avail else None
    if has_hans and has_hant:
        simp, trad = _simp_trad_counts(text)
        # Default to Simplified on a tie / no distinctive characters.
        return "zh-Hant" if trad > simp else "zh-Hans"
    return "zh-Hans" if has_hans else "zh-Hant"


def detect_language(text: str, available: Iterable[str]) -> Optional[str]:
    """Return the best language code in ``available`` for ``text``, or ``None``.

    Heuristic: Hangul → ko, kana → ja, Han-without-kana → Chinese, otherwise
    Latin → en. For Chinese the code is resolved to ``zh-Hans`` / ``zh-Hant``
    (or legacy ``zh``) by a Simplified-vs-Traditional character heuristic.
    Distinctive scripts (Hangul/kana/Han) outrank Latin — Latin appears in every
    language's text (emails, URLs), so any Hangul beats an embedded ``a@b.com``.
    Falls back to ``$PRIVYSCOPE_LANG``, then the sole available language, else
    ``None``.
    """
    avail = set(available)
    c = _script_counts(text)
    has_kana = bool(c["Hiragana"] or c["Katakana"])

    # Distinctive scripts are strong language signals and outrank Latin.
    distinctive: List[Tuple[str, int]] = []
    if c["Hangul"]:
        distinctive.append(("ko", c["Hangul"]))
    if has_kana:
        distinctive.append(("ja", c["Hiragana"] + c["Katakana"]))
    if c["Han"]:
        # Han mixed with kana is Japanese; Han alone leans Chinese.
        if has_kana:
            distinctive.append(("ja", c["Han"]))
        else:
            zh = _resolve_zh(text, avail)
            if zh is not None:
                distinctive.append((zh, c["Han"]))

    for code, _score in sorted(distinctive, key=lambda kv: kv[1], reverse=True):
        if code in avail:
            return code
    if c["Latin"] and "en" in avail:
        return "en"

    env = os.environ.get("PRIVYSCOPE_LANG")
    if env and env in avail:
        return env
    if len(avail) == 1:
        return next(iter(avail))
    return None
