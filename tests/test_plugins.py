"""Language-plugin contract + resolution logic."""
import pytest

from privyscope._core import plugins
from privyscope._core.plugins import LanguagePlugin, get_plugin, resolve_plugin

KO = LanguagePlugin(
    code="ko", display_name="Korean", default_repo="zafrem/privyscope-ko",
    package="privyscope_ko", scripts=("Hangul",), default_base_model="klue/roberta-base",
)
EN = LanguagePlugin(
    code="en", display_name="English", default_repo="zafrem/privyscope-en",
    package="privyscope_en", scripts=("Latin",), default_base_model="roberta-base",
)


@pytest.fixture
def installed(monkeypatch):
    """Patch the (otherwise cached) plugin registry with a controlled set."""
    def _set(mapping):
        monkeypatch.setattr(plugins, "installed_languages", lambda: mapping)
    return _set


def test_resolve_explicit(installed):
    installed({"ko": KO, "en": EN})
    assert resolve_plugin("ko") is KO
    assert resolve_plugin("en") is EN


def test_resolve_sole_installed(installed):
    installed({"ko": KO})
    assert resolve_plugin(None) is KO


def test_resolve_none_installed_errors(installed):
    installed({})
    with pytest.raises(ValueError, match="no privyscope language plugin installed"):
        resolve_plugin(None)


def test_resolve_ambiguous_errors(installed):
    installed({"ko": KO, "en": EN})
    with pytest.raises(ValueError, match="multiple language plugins"):
        resolve_plugin(None)


def test_get_plugin_unknown_lists_installed(installed):
    installed({"ko": KO})
    with pytest.raises(ValueError, match="installed: ko"):
        get_plugin("en")


def test_plugin_locates_packaged_yaml(tmp_path, monkeypatch):
    # A plugin's resource paths resolve under its package via importlib.resources.
    import importlib
    import sys

    pkg_dir = tmp_path / "dummy_plugin"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "regex_rules.yaml").write_text("version: 1\nrules: []\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    sys.modules.pop("dummy_plugin", None)

    plugin = LanguagePlugin(
        code="xx", display_name="X", default_repo="r", package="dummy_plugin",
    )
    assert plugin.regex_rules_path().name == "regex_rules.yaml"
    assert plugin.regex_rules_path().exists()
