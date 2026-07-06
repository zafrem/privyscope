"""Unified CLI entrypoint: ``privyscope redact | eval | train`` (SRS FR-2.1)."""
from __future__ import annotations

import sys
from typing import List, Optional, Sequence

from ._api import AutoPrivyscope, Privyscope
from ._cli.args import build_parser
from ._cli.render import print_result, supports_color
from ._core.plugins import installed_languages, resolve_plugin

_PROVIDERS = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
}


def _parse_operating_point(value: str):
    """A preset name, or comma-separated raw transition biases (FR-2.5)."""
    if value and "," in value:
        return [float(x) for x in value.split(",")]
    return value


def _read_input(args) -> Optional[str]:
    """Resolve redact input from arg / file / stdin; None triggers interactive mode."""
    if args.text is not None:
        return args.text
    if args.file:
        with open(args.file, "r", encoding="utf-8") as fh:
            return fh.read().rstrip("\n")
    if not sys.stdin.isatty():
        data = sys.stdin.read()
        return data.rstrip("\n") if data else None
    return None


def _build_engine(args):
    """Build the inference engine for redact/eval.

    ``--lang X`` → that language. No ``--lang`` with one plugin installed → that
    plugin. No ``--lang`` with several installed → an auto-routing dispatcher.
    Returns an ``Privyscope`` or an ``AutoPrivyscope`` (same redact surface).
    """
    operating_point = _parse_operating_point(getattr(args, "operating_point", "balanced"))
    cache_dir = getattr(args, "cache_dir", None)
    regex_only = getattr(args, "regex_only", False)
    providers = _PROVIDERS.get(getattr(args, "device", "cpu"), None)
    lang = getattr(args, "lang", None)

    if lang is None and len(installed_languages()) > 1:
        return AutoPrivyscope(
            operating_point=operating_point,
            cache_dir=cache_dir,
            providers=providers,
            regex_only=regex_only,
        )
    return Privyscope.from_pretrained(
        operating_point=operating_point,
        lang=lang,
        cache_dir=cache_dir,
        revision=getattr(args, "revision", None),
        regex_only=regex_only,
        providers=providers,
    )


def _cmd_redact(args) -> int:
    engine = _build_engine(args)
    entity_types = args.entity_types.split(",") if args.entity_types else None
    text = _read_input(args)
    as_json = args.json or not supports_color()

    if text is None:  # interactive mode (FR-2.1)
        print("privyscope interactive redact — enter text (Ctrl-D to exit)", file=sys.stderr)
        for line in sys.stdin:
            line = line.rstrip("\n")
            if not line:
                continue
            print_result(engine.redact(line, entity_types, args.output_mode), as_json)
        return 0

    print_result(engine.redact(text, entity_types, args.output_mode), as_json)
    return 0


def _cmd_eval(args) -> int:
    from ._eval.runner import run_eval

    engine = _build_engine(args)
    report = run_eval(engine, args.dataset, eval_mode=args.eval_mode, limit=args.limit)
    import json

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _cmd_train(args) -> int:
    try:
        from ._train.runner import run_train
    except ImportError as exc:
        print(
            f"training requires the optional extra: pip install 'privyscope[train]' ({exc})",
            file=sys.stderr,
        )
        return 2
    # Default the base checkpoint from the selected language plugin (FR-2.6).
    if not getattr(args, "base_model", None):
        plugin = resolve_plugin(getattr(args, "lang", None))
        if plugin.default_base_model:
            args.base_model = plugin.default_base_model
    summary = run_train(args)
    import json

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return {
        "redact": _cmd_redact,
        "eval": _cmd_eval,
        "train": _cmd_train,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
