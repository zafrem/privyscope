"""Public Python API (SRS FR-2.7).

    from privyscope import Privyscope            # or: from privyscope_ko import Privyscope

    # one language (explicit, or the sole one installed)
    engine = Privyscope.from_pretrained(lang="ko")
    result = engine.redact("홍길동의 전화번호는 010-1234-5678")
    result.masked_text      # "<PER>의 전화번호는 <PHONE>"

    # multiple languages installed → route each text automatically
    auto = Privyscope.auto()
    auto.redact("홍길동 010-1234-5678")          # → Korean engine
    auto.redact("John Smith 555-123-4567")       # → English engine

The engine is a two-stage hybrid pipeline (SRS §3.4): a regex filter (always
available) plus an optional ONNX NER stage. ``regex_only=True`` runs the engine
with no model weights — useful offline and for structurally-obvious PII.

Language-specific data (regex rules, entity config, default weights repo) comes
from an installed language plugin discovered via the ``privyscope.languages``
entry-point group; the core hardcodes nothing language-specific.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

from ._core.bioes import Span
from ._core.detect import detect_language
from ._core.download import resolve_weights
from ._core.pipeline import assemble, filter_stopwords, merge_union
from ._core.plugins import LanguagePlugin, installed_languages, resolve_plugin
from ._core.regex_filter import RegexFilter
from ._core.schema import RedactionResult
from ._core.transitions import CATEGORIES, resolve_operating_point

OperatingPoint = Union[str, Sequence[float], dict, None]


def _regex_rules_path(plugin: LanguagePlugin) -> Path:
    """Locate the active language's regex ruleset (env override wins)."""
    env = os.environ.get("PRIVYSCOPE_REGEX_RULES")
    if env:
        return Path(env)
    return plugin.regex_rules_path()


class Privyscope:
    """Two-stage PII redaction engine for a single language."""

    def __init__(
        self,
        regex_filter: RegexFilter,
        runtime=None,
        operating_point: OperatingPoint = "balanced",
        stopwords: Optional[Iterable[str]] = None,
    ) -> None:
        self._regex = regex_filter
        self._runtime = runtime
        self._operating_point = operating_point
        self._stopwords = frozenset(stopwords) if stopwords else frozenset()

    # -- construction -------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        operating_point: OperatingPoint = "balanced",
        *,
        lang: Optional[str] = None,
        repo_id: Optional[str] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        regex_rules: Optional[str] = None,
        providers: Optional[Sequence[str]] = None,
        regex_only: bool = False,
    ) -> "Privyscope":
        """Load the engine for one language, fetching ONNX weights on first use.

        ``lang`` selects the language plugin; if omitted, the sole installed
        plugin is used (raises if several are installed — pass ``lang`` or use
        :meth:`auto`). Pass ``regex_only=True`` to skip the NER stage (no weights
        required). ``cache_dir`` pointing at a local bundle enables offline use.
        """
        plugin = resolve_plugin(lang)
        rules_path = regex_rules or _regex_rules_path(plugin)
        regex = RegexFilter.from_yaml(rules_path)

        runtime = None
        stopwords: Optional[Iterable[str]] = None
        if not regex_only:
            from ._core.runtime import NerRuntime

            repo = repo_id or plugin.default_repo
            bundle = resolve_weights(repo_id=repo, revision=revision, cache_dir=cache_dir)
            runtime = NerRuntime(bundle, providers=providers)
            stopwords = plugin.stopwords()  # denylist only matters for NER spans
        return cls(regex, runtime, operating_point, stopwords)

    @classmethod
    def regex_only(
        cls, *, lang: Optional[str] = None, regex_rules: Optional[str] = None
    ) -> "Privyscope":
        """Convenience constructor for a Stage-1-only engine (no weights)."""
        return cls.from_pretrained(regex_only=True, lang=lang, regex_rules=regex_rules)

    @classmethod
    def auto(
        cls,
        operating_point: OperatingPoint = "balanced",
        *,
        cache_dir: Optional[str] = None,
        providers: Optional[Sequence[str]] = None,
        regex_only: bool = False,
    ) -> "AutoPrivyscope":
        """Return a dispatcher that routes each text to its language engine.

        Languages are detected per text (Unicode-script heuristic) and restricted
        to the installed plugins. Per-language engines are built lazily and cached.
        """
        return AutoPrivyscope(
            operating_point=operating_point,
            cache_dir=cache_dir,
            providers=providers,
            regex_only=regex_only,
        )

    # -- inference ----------------------------------------------------------
    def _effective_biases(self, operating_point: OperatingPoint) -> List[float]:
        """Learned baseline biases shifted by the operating-point offset (FR-2.5)."""
        offset = resolve_operating_point(
            operating_point if operating_point is not None else self._operating_point
        )
        base = self._runtime.default_biases if self._runtime else [0.0] * len(CATEGORIES)
        return [b + o for b, o in zip(base, offset)]

    def redact(
        self,
        text: str,
        entity_types: Optional[Iterable[str]] = None,
        output_mode: str = "typed",
        operating_point: OperatingPoint = None,
    ) -> RedactionResult:
        """Detect and mask PII in a single string (FR-2.2)."""
        regex_spans: List[Span] = self._regex.find(text)
        ner_spans: List[Span] = []
        mismatch = False
        if self._runtime is not None:
            ner_spans, mismatch = self._runtime.predict(text, self._effective_biases(operating_point))
            ner_spans = filter_stopwords(ner_spans, text, self._stopwords)
        merged = merge_union(regex_spans, ner_spans)
        return assemble(text, merged, output_mode, entity_types, mismatch)

    def batch_redact(
        self,
        texts: Sequence[str],
        entity_types: Optional[Iterable[str]] = None,
        output_mode: str = "typed",
        operating_point: OperatingPoint = None,
    ) -> List[RedactionResult]:
        return [self.redact(t, entity_types, output_mode, operating_point) for t in texts]

    @property
    def has_ner(self) -> bool:
        return self._runtime is not None


class AutoPrivyscope:
    """Multi-language dispatcher (see :meth:`Privyscope.auto`).

    Mirrors the :class:`Privyscope` inference surface (``redact`` /
    ``batch_redact`` / ``has_ner``) and routes each call to a lazily-built,
    cached per-language engine chosen by :func:`detect_language`.

    **Thread-safety:** per-language engine construction is guarded by a lock, so
    concurrent callers (e.g. a multi-threaded web server) never build or download
    the same engine twice. Once built, ONNX Runtime sessions are safe for
    concurrent inference. Call :meth:`preload` at startup to build engines
    eagerly and avoid a first-request latency spike.
    """

    def __init__(
        self,
        *,
        operating_point: OperatingPoint,
        cache_dir: Optional[str],
        providers: Optional[Sequence[str]],
        regex_only: bool,
    ) -> None:
        self._available = sorted(installed_languages())
        if not self._available:
            raise ValueError(
                "no privyscope language plugin installed; install one, "
                "e.g. pip install privyscope-ko"
            )
        self._kwargs = dict(
            operating_point=operating_point,
            cache_dir=cache_dir,
            providers=providers,
            regex_only=regex_only,
        )
        self._regex_only = regex_only
        self._engines: Dict[str, Privyscope] = {}
        self._lock = threading.Lock()

    @property
    def available_languages(self) -> List[str]:
        return list(self._available)

    def _engine_for(self, lang: str) -> Privyscope:
        # Fast path: already built (dict reads are atomic under the GIL).
        engine = self._engines.get(lang)
        if engine is not None:
            return engine
        # Slow path: build under lock with a double-check so concurrent callers
        # for the same language share one engine (and one weights download).
        with self._lock:
            engine = self._engines.get(lang)
            if engine is None:
                engine = Privyscope.from_pretrained(lang=lang, **self._kwargs)
                self._engines[lang] = engine
            return engine

    def preload(self, langs: Optional[Iterable[str]] = None) -> "AutoPrivyscope":
        """Eagerly build (and cache) engines so the first request is warm.

        Builds every installed language by default, or just ``langs``. Returns
        ``self`` for chaining, e.g. ``engine = Privyscope.auto().preload()``.
        Ideal at web-server startup: fail fast on missing weights and avoid a
        cold-start spike on the first redact call.
        """
        for lang in (langs if langs is not None else self._available):
            self._engine_for(lang)
        return self

    def route(self, text: str) -> str:
        """Return the language code chosen for ``text``."""
        return detect_language(text, self._available) or self._available[0]

    def redact(
        self,
        text: str,
        entity_types: Optional[Iterable[str]] = None,
        output_mode: str = "typed",
        operating_point: OperatingPoint = None,
    ) -> RedactionResult:
        return self._engine_for(self.route(text)).redact(
            text, entity_types, output_mode, operating_point
        )

    def batch_redact(
        self,
        texts: Sequence[str],
        entity_types: Optional[Iterable[str]] = None,
        output_mode: str = "typed",
        operating_point: OperatingPoint = None,
    ) -> List[RedactionResult]:
        return [self.redact(t, entity_types, output_mode, operating_point) for t in texts]

    @property
    def has_ner(self) -> bool:
        return not self._regex_only
