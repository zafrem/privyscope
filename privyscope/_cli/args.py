"""CLI argument parsing (SRS FR-2.1)."""
from __future__ import annotations

import argparse

from .. import __version__


def _version_string() -> str:
    """Core version plus the language plugins discovered at runtime."""
    from .._core.plugins import installed_languages

    langs = installed_languages()
    if langs:
        detail = ", ".join(f"{c} ({p.display_name})" for c, p in sorted(langs.items()))
    else:
        detail = "none — install one, e.g. pip install privyscope-ko"
    return f"privyscope {__version__} (languages: {detail})"


def _add_lang(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--lang",
        help="language code (e.g. ko, en); omit to auto-detect per text when "
        "several are installed, or use the sole installed language",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="privyscope",
        description="privyscope — multilingual PII detection & masking (redact / eval / train)",
    )
    p.add_argument("--version", action="version", version=_version_string())
    sub = p.add_subparsers(dest="command", required=True)

    # -- redact -------------------------------------------------------------
    r = sub.add_parser("redact", help="detect and mask PII in text")
    r.add_argument("text", nargs="?", help="text to redact; omit to read stdin / interactive")
    r.add_argument("-f", "--file", help="read input from a file")
    _add_lang(r)
    r.add_argument("--device", choices=["cpu", "cuda", "coreml"], default="cpu")
    r.add_argument(
        "--operating-point", default="balanced",
        help="high_recall | balanced | high_precision, or comma-separated raw biases",
    )
    r.add_argument("--output-mode", choices=["typed", "redacted"], default="typed")
    r.add_argument("--entity-types", help="comma-separated subset, e.g. PER,PHONE")
    r.add_argument("--regex-only", action="store_true", help="skip the NER stage (no weights)")
    r.add_argument("--cache-dir", help="local weight bundle directory (offline)")
    r.add_argument("--revision", help="pin a HF Hub weight revision")
    r.add_argument("--json", action="store_true", help="force JSON output")

    # -- eval ---------------------------------------------------------------
    e = sub.add_parser("eval", help="evaluate against a labelled JSONL dataset")
    e.add_argument("dataset", help="JSONL with text + spans")
    _add_lang(e)
    e.add_argument("--eval-mode", choices=["typed", "untyped"], default="typed")
    e.add_argument("--operating-point", default="balanced")
    e.add_argument("--regex-only", action="store_true")
    e.add_argument("--cache-dir")
    e.add_argument("--limit", type=int, default=None)

    # -- train --------------------------------------------------------------
    t = sub.add_parser("train", help="fine-tune the NER model on custom JSONL")
    t.add_argument("dataset", help="training JSONL")
    _add_lang(t)
    t.add_argument("--validation-dataset")
    t.add_argument("--label-space-json", help="custom label map (FR-2.6 / §8.2)")
    t.add_argument("--output-dir", required=True)
    t.add_argument("--base-model", default=None)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--learning-rate", type=float, default=3e-5)
    return p
