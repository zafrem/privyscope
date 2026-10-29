"""Plugin-declared default operating point (``default_operating_point:`` in entity_config.yaml)."""
from pathlib import Path

from privyscope._core.plugins import LanguagePlugin
from privyscope._core.transitions import OPERATING_POINTS


class _Plugin(LanguagePlugin):
    cfg: Path = Path()

    def entity_config_path(self) -> Path:
        return self.cfg


def _plugin(tmp_path: Path, body: str) -> LanguagePlugin:
    _Plugin.cfg = tmp_path / "entity_config.yaml"
    _Plugin.cfg.write_text(body, encoding="utf-8")
    return _Plugin("xx", "X", "o/x", "pkg")


def test_reads_known_preset(tmp_path):
    assert _plugin(tmp_path, "default_operating_point: high_recall\n").default_operating_point() == "high_recall"


def test_absent_or_unknown_preset_is_none(tmp_path):
    assert _plugin(tmp_path, "language: xx\n").default_operating_point() is None
    assert _plugin(tmp_path, "default_operating_point: turbo\n").default_operating_point() is None
    assert "turbo" not in OPERATING_POINTS
