"""AutoPrivyscope dispatcher: caching, preload, and thread-safe construction."""
import threading
import time

import pytest

from privyscope import _api
from privyscope._api import AutoPrivyscope, Privyscope


@pytest.fixture
def counting_auto(monkeypatch):
    """An AutoPrivyscope whose engine construction is counted (no real weights).

    Patches the installed-language set and ``Privyscope.from_pretrained`` so each
    build is a cheap sentinel; a small sleep widens the race window so the
    double-checked lock is actually exercised.
    """
    monkeypatch.setattr(_api, "installed_languages", lambda: {"en": object(), "ko": object()})

    builds = {}

    def fake_from_pretrained(*args, lang=None, **kwargs):
        time.sleep(0.01)  # widen the window for concurrent first-hits
        builds[lang] = builds.get(lang, 0) + 1
        return f"engine::{lang}"

    monkeypatch.setattr(Privyscope, "from_pretrained", classmethod(
        lambda cls, *a, lang=None, **k: fake_from_pretrained(lang=lang)
    ))
    auto = AutoPrivyscope(operating_point="balanced", cache_dir=None, providers=None, regex_only=True)
    return auto, builds


def test_engine_cached_per_language(counting_auto):
    auto, builds = counting_auto
    e1 = auto._engine_for("en")
    e2 = auto._engine_for("en")
    assert e1 == e2 == "engine::en"
    assert builds["en"] == 1  # built once, then cached


def test_preload_builds_all_once(counting_auto):
    auto, builds = counting_auto
    assert auto.preload() is auto                 # returns self for chaining
    assert builds == {"en": 1, "ko": 1}
    auto.preload()                                # idempotent — no rebuilds
    assert builds == {"en": 1, "ko": 1}


def test_preload_subset(counting_auto):
    auto, builds = counting_auto
    auto.preload(["en"])
    assert builds == {"en": 1}


def test_concurrent_first_hit_builds_once(counting_auto):
    auto, builds = counting_auto
    barrier = threading.Barrier(8)

    def hammer():
        barrier.wait()              # release all threads at once
        auto._engine_for("en")

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert builds["en"] == 1        # the lock prevented duplicate construction
