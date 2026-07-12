"""Verification gate for Stage-1 regex matches (SRS §3.4).

Compiled rules carry an optional ``verify:`` function name sourced from the
pii-pattern-engine ruleset (``kr_rrn_valid``, ``luhn``, ``korean_name_valid``, …).
These are checksum and dictionary validators that reject structurally plausible
but invalid matches. They are what makes high-recall patterns usable at all: the
Korean name rule is ``[가-힣]{2,5}``, which without ``korean_name_valid`` fires on
almost every Korean word.

The functions themselves are vendored from the submodule at build time into
``_vendor/`` (see ``scripts/vendor_verification.py``).

Failure policy is **fail-open**: an unknown name, or a validator that raises, keeps
the match. For a redaction engine, over-redacting is safer than silently dropping
PII, so a broken validator must never turn into missed PII. Both cases warn once.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

Verifier = Callable[[str], bool]


@lru_cache(maxsize=1)
def _registry() -> Dict[str, Verifier]:
    """Public ``(str) -> bool`` functions defined in the vendored module."""
    import inspect

    try:
        from ._vendor.verification.python import verification as mod
    except ImportError:  # pragma: no cover - build hook did not run
        logger.warning(
            "privyscope: verification functions unavailable (the _vendor/ tree was not "
            "built). Regex rules will match without checksum/dictionary validation, "
            "which raises false positives. Rebuild with: python scripts/vendor_verification.py"
        )
        return {}

    return {
        name: fn
        for name, fn in vars(mod).items()
        if not name.startswith("_")
        and inspect.isfunction(fn)
        and fn.__module__ == mod.__name__
    }


@lru_cache(maxsize=None)
def resolve(name: str) -> Optional[Verifier]:
    """Return the verifier registered as ``name``, or None if it does not exist."""
    fn = _registry().get(name)
    if fn is None:
        logger.warning(
            "privyscope: unknown verification function %r; matches for this rule are "
            "kept unverified (fail-open)", name
        )
    return fn


def verify(name: str, value: str) -> bool:
    """True if ``value`` passes verifier ``name``. Fail-open on error/unknown name."""
    fn = resolve(name)
    if fn is None:
        return True
    try:
        return bool(fn(value))
    except Exception:  # noqa: BLE001 - a broken validator must not drop PII
        logger.warning(
            "privyscope: verification %r raised on a match; keeping it (fail-open)",
            name, exc_info=True,
        )
        return True


def available() -> int:
    """Number of verifiers loaded (0 means the vendored tree is missing)."""
    return len(_registry())
