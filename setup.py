"""Build shim: vendor the pii-pattern-engine verification functions before packaging.

Runs at import, i.e. before ``setup()`` evaluates ``[tool.setuptools.packages.find]``, so
the generated ``_core/_vendor/`` tree exists in time to be discovered and packaged. That
covers every build path — ``pip install .``, ``python -m build``, editable installs.

Three cases, in order:
  submodule present            -> (re)generate, so a build always tracks the checkout
  absent but _vendor/ present  -> reuse it; this is an sdist, which ships the generated
                                  tree but not the submodule
  neither                      -> fail loudly rather than silently publish a wheel whose
                                  regex rules would run with no checksum validation
"""
from pathlib import Path

from setuptools import setup

HERE = Path(__file__).parent
VENDOR = HERE / "privyscope" / "_core" / "_vendor"
UPSTREAM = HERE / "pii-pattern-engine"


def _vendor() -> None:
    if UPSTREAM.is_dir():
        import subprocess
        import sys

        subprocess.run(
            [sys.executable, str(HERE / "scripts" / "vendor_verification.py")], check=True
        )
    elif not (VENDOR / "verification" / "python" / "verification.py").exists():
        raise SystemExit(
            "privyscope: cannot build — the pii-pattern-engine submodule is missing and no\n"
            "vendored copy is present. Verification functions gate the Stage-1 regex rules;\n"
            "without them the engine reports false positives.\n"
            "  run: git submodule update --init --recursive"
        )


_vendor()
setup()
