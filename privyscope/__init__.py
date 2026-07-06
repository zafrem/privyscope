"""privyscope — multilingual PII detection & masking engine (Apache-2.0).

This is the language-neutral **core**: the two-stage hybrid engine (regex filter
+ ONNX BIOES/Viterbi NER), the ``privyscope`` console command, and the plugin
machinery that discovers installed language packs (``privyscope-ko``,
``privyscope-en``, …). Install at least one language plugin to do real work::

    pip install privyscope-ko            # pulls in this core automatically

Public surface::

    from privyscope import Privyscope
    engine = Privyscope.from_pretrained(lang="ko")   # one language
    auto   = Privyscope.auto()                        # route per text
"""
from ._api import Privyscope
from ._core.schema import RedactionResult, DetectedSpan, SCHEMA_VERSION
from ._core.plugins import LanguagePlugin, installed_languages

__version__ = "0.1.0"

__all__ = [
    "Privyscope",
    "RedactionResult",
    "DetectedSpan",
    "SCHEMA_VERSION",
    "LanguagePlugin",
    "installed_languages",
    "__version__",
]
