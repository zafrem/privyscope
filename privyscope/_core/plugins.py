"""Language-plugin contract + discovery (core+plugin architecture).

Each language ships as a thin plugin package (``privyscope_ko``, ``privyscope_en``,
…) that registers a :class:`LanguagePlugin` under the ``privyscope.languages``
entry-point group. The core engine discovers installed languages at runtime and
loads their packaged YAML data + default weights repo. This lets a single
``privyscope`` command serve any combination of co-installed languages without the
file collisions that a self-contained per-language package would cause.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

from .pipeline import NerPrecedence, SpanTrim


@dataclass(frozen=True)
class LanguagePlugin:
    """Registration record a language plugin exposes as its ``LANG`` symbol.

    ``scripts`` are Unicode script hints used by per-text auto-detection
    (e.g. ``("Hangul",)`` for Korean). ``package`` is the plugin's own import
    name, used to locate its packaged ``regex_rules.yaml`` / ``entity_config.yaml``.
    """

    code: str                          # ISO code, e.g. "ko"
    display_name: str                  # human-readable, e.g. "Korean"
    default_repo: str                  # default HF Hub weights repo
    package: str                       # plugin import name, e.g. "privyscope_ko"
    scripts: Tuple[str, ...] = ()      # script hints for auto-detection
    default_base_model: str = ""       # base checkpoint for fine-tuning

    def _resource(self, name: str) -> Path:
        from importlib.resources import files

        return Path(str(files(self.package) / name))

    def regex_rules_path(self) -> Path:
        return self._resource("regex_rules.yaml")

    def entity_config_path(self) -> Path:
        return self._resource("entity_config.yaml")

    def stopwords(self) -> frozenset:
        """Common-word denylist from ``entity_config.yaml`` (empty if absent).

        Surfaces a contextual NER span must never equal — an offset-preserving
        false-positive filter applied to *output spans*, never to the input text.
        """
        import yaml

        try:
            data = yaml.safe_load(self.entity_config_path().read_text(encoding="utf-8")) or {}
        except (FileNotFoundError, OSError):
            return frozenset()
        return frozenset(
            w.strip() for w in (data.get("stopwords") or []) if isinstance(w, str) and w.strip()
        )

    def span_trims(self) -> Tuple[SpanTrim, ...]:
        """Boundary-trim rules from ``entity_config.yaml`` (empty if absent).

        For languages whose tokenizer fuses a span-final character with a following
        particle (Korean ``20일`` + ``이``); see :class:`SpanTrim`.
        """
        import yaml

        try:
            data = yaml.safe_load(self.entity_config_path().read_text(encoding="utf-8")) or {}
        except (FileNotFoundError, OSError):
            return ()
        return parse_span_trims(data)


    def ner_precedence(self) -> Tuple[NerPrecedence, ...]:
        """NER-over-regex precedence rules from ``entity_config.yaml`` (empty if absent).

        See :class:`NerPrecedence`; the default is that a regex span wins every cross-label overlap.
        """
        import yaml

        try:
            data = yaml.safe_load(self.entity_config_path().read_text(encoding="utf-8")) or {}
        except (FileNotFoundError, OSError):
            return ()
        return parse_ner_precedence(data)


def parse_ner_precedence(data: dict) -> Tuple[NerPrecedence, ...]:
    """Read the optional ``ner_precedence:`` list (``{label, over: [labels]}``) from an entity config.

    Malformed entries are skipped rather than failing the load, as with ``span_trim``.
    """
    rules = []
    for entry in (data or {}).get("ner_precedence") or []:
        if not isinstance(entry, dict):
            continue
        label, over = entry.get("label"), entry.get("over")
        if not (isinstance(label, str) and label and isinstance(over, list)):
            continue
        over = tuple(x for x in over if isinstance(x, str) and x and x != label)
        if over:
            rules.append(NerPrecedence(label, over))
    return tuple(rules)


def parse_span_trims(data: dict) -> Tuple[SpanTrim, ...]:
    """Read the optional ``span_trim:`` list from an ``entity_config.yaml`` mapping.

    Each entry is ``{label, suffixes}`` plus either ``after`` (the character the span must
    end on once the suffix is cut) or ``min_keep >= 2`` (any character may precede the
    suffix, but at least that many must remain). Malformed entries are skipped rather than
    failing the load, since the section only refines span boundaries.
    """
    trims = []
    for entry in (data or {}).get("span_trim") or []:
        if not isinstance(entry, dict):
            continue
        label, suffixes = entry.get("label"), entry.get("suffixes")
        after, min_keep = entry.get("after", ""), entry.get("min_keep", 1)
        if not (isinstance(label, str) and label and isinstance(after, str)):
            continue
        if not (isinstance(min_keep, int) and not isinstance(min_keep, bool) and min_keep >= 1):
            continue
        if not after and min_keep < 2:   # nothing anchors the cut: refuse rather than guess
            continue
        if not isinstance(suffixes, list):
            continue
        suffixes = tuple(x for x in suffixes if isinstance(x, str) and x)
        if suffixes:
            trims.append(SpanTrim(label, after, suffixes, min_keep))
    return tuple(trims)


@lru_cache(maxsize=1)
def installed_languages() -> Dict[str, LanguagePlugin]:
    """Discover every installed language plugin via entry points (cached)."""
    from importlib.metadata import entry_points

    try:  # Python 3.10+ selectable API
        eps = entry_points(group="privyscope.languages")
    except TypeError:  # pragma: no cover - Python 3.9 fallback
        eps = entry_points().get("privyscope.languages", [])

    found: Dict[str, LanguagePlugin] = {}
    for ep in eps:
        plugin = ep.load()
        if isinstance(plugin, LanguagePlugin):
            found[plugin.code] = plugin
    return found


def get_plugin(code: str) -> LanguagePlugin:
    """Return the plugin registered for ``code`` or raise a helpful error."""
    langs = installed_languages()
    try:
        return langs[code]
    except KeyError:
        avail = ", ".join(sorted(langs)) or "(none installed)"
        raise ValueError(
            f"no privyscope language plugin for {code!r}; installed: {avail}. "
            f"Install one, e.g. pip install privyscope-{code}"
        ) from None


def resolve_plugin(lang: str | None) -> LanguagePlugin:
    """Resolve a plugin from an explicit ``lang`` or the sole-installed default.

    Raises with guidance when no language is installed, or when several are and
    none was named (the caller should pass ``lang=`` / ``--lang`` or use
    ``Privyscope.auto()``).
    """
    if lang is not None:
        return get_plugin(lang)
    langs = installed_languages()
    if not langs:
        raise ValueError(
            "no privyscope language plugin installed; install one, "
            "e.g. pip install privyscope-ko"
        )
    if len(langs) == 1:
        return next(iter(langs.values()))
    avail = ", ".join(sorted(langs))
    raise ValueError(
        f"multiple language plugins installed ({avail}); pass lang=... (API) "
        f"or --lang (CLI) to choose, or use Privyscope.auto() for per-text routing"
    )
