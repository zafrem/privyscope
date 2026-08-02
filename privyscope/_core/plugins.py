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
