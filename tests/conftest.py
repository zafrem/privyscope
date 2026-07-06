import sys
from pathlib import Path

# Allow `import privyscope` when running from a source checkout without install:
# add the project root (which contains the `privyscope` package) to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from privyscope._api import Privyscope  # noqa: E402
from privyscope._core.regex_filter import RegexFilter  # noqa: E402

FIXTURE_RULES = Path(__file__).resolve().parent / "fixtures" / "regex_rules.min.yaml"


@pytest.fixture(scope="session")
def fixture_rules() -> Path:
    return FIXTURE_RULES


@pytest.fixture(scope="module")
def engine() -> Privyscope:
    """A regex-only engine built straight from the fixture ruleset.

    The core ships no language data, so engine tests bypass plugin resolution and
    construct the engine directly from the bundled fixture.
    """
    return Privyscope(RegexFilter.from_yaml(FIXTURE_RULES))
