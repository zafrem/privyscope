#!/usr/bin/env python3
"""Vendor the pii-pattern-engine verification functions into the core package.

Stage-1 regex rules carry a ``verify:`` function name (``kr_rrn_valid``, ``luhn``,
…). Those validators live in the ``pii-pattern-engine`` submodule; the wheel must
carry them, so this script copies them into ``privyscope/_core/_vendor/``.

Run automatically by the ``build_py`` hook in setup.py, so any build path
(``pip install .``, ``python -m build``, editable installs) picks it up. It can
also be run by hand after bumping the submodule:

    git submodule update --init --recursive
    python scripts/vendor_verification.py

The upstream ``verification.py`` locates its CSVs as
``Path(__file__).parent.parent.parent / "datas"``. Mirroring the upstream layout
under ``_vendor/`` keeps that math correct, so the file is copied byte-for-byte
with no patching.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = ROOT / "pii-pattern-engine"
VENDOR = ROOT / "privyscope" / "_core" / "_vendor"

# Keeps `_get_data_path()` (…/parent.parent.parent/"datas") resolving inside _vendor.
_LAYOUT = {
    UPSTREAM / "verification" / "python" / "verification.py":
        VENDOR / "verification" / "python" / "verification.py",
}


def main() -> int:
    if not UPSTREAM.exists():
        print(
            f"error: submodule missing at {UPSTREAM}\n"
            "       run: git submodule update --init --recursive",
            file=sys.stderr,
        )
        return 1

    if VENDOR.exists():
        shutil.rmtree(VENDOR)

    for src, dst in _LAYOUT.items():
        if not src.exists():
            print(f"error: expected upstream file missing: {src}", file=sys.stderr)
            return 1
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    # Data files backing the dictionary validators (surname/address lists). Without
    # them the loaders return empty sets and every name/address match is rejected.
    src_datas = UPSTREAM / "datas"
    if not src_datas.is_dir():
        print(f"error: expected upstream datas/ missing: {src_datas}", file=sys.stderr)
        return 1
    shutil.copytree(src_datas, VENDOR / "datas")

    for pkg in (VENDOR, VENDOR / "verification", VENDOR / "verification" / "python"):
        (pkg / "__init__.py").write_text(
            '"""Vendored from the pii-pattern-engine submodule; do not edit by hand."""\n',
            encoding="utf-8",
        )

    n_data = len(list((VENDOR / "datas").glob("*")))
    print(f"[vendor] verification.py + {n_data} data files -> {VENDOR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
