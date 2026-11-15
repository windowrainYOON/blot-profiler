"""Korean / English interface text (1.4.0).

The source text is Korean; ``_t(text)`` returns the English text from ``i18n_en.EN`` when
the interface language is English. f-strings became ``_t("…{0}…").format(…)``.

The language is chosen once at start-up (File → Settings → General, or View → Language) and
kept in the per-computer prefs file; changing it takes effect after a restart.
``BLOT_PROFILER_LANG`` (ko · en) overrides it, e.g. for tests.
"""

from __future__ import annotations

import json
import locale
import os
import sys

LANGS = (("system", "시스템 언어 따르기 · System"), ("ko", "한국어"), ("en", "English"))


def system_language() -> str:
    """ko when the computer's language is Korean, otherwise en."""
    names = []
    try:
        from PySide6.QtCore import QLocale

        names += list(QLocale.system().uiLanguages()) + [QLocale.system().name()]
    except Exception:
        pass
    for var in ("LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"):
        names.append(os.environ.get(var, ""))
    try:
        names.append(locale.getlocale()[0] or "")
    except Exception:
        pass
    for n in names:
        if n:
            return "ko" if n.lower().startswith("ko") else "en"
    return "en"


def prefs_path() -> str:
    """The per-computer prefs file (prefs.path uses this too)."""
    env = os.environ.get("BLOT_PROFILER_PREFS")
    if env:
        return env
    if sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support/BlotProfiler")
    elif sys.platform.startswith("win"):
        base = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "BlotProfiler")
    else:
        base = os.path.expanduser("~/.config/blotprofiler")
    return os.path.join(base, "prefs.json")


def setting() -> str:
    """The saved choice: system · ko · en."""
    try:
        with open(prefs_path(), encoding="utf-8") as f:
            v = json.load(f).get("general", {}).get("language", "system")
    except Exception:
        v = "system"
    return v if v in ("system", "ko", "en") else "system"


def set_setting(value: str):
    from . import prefs

    p = prefs.load()
    p.setdefault("general", {})["language"] = value
    prefs.save(p)


def resolve(value: str) -> str:
    return system_language() if value == "system" else value


def _start_language() -> str:
    env = os.environ.get("BLOT_PROFILER_LANG", "").lower()
    if env in ("ko", "en"):
        return env
    return resolve(setting())


LANG = _start_language()
_EN = None


def _t(text: str) -> str:
    global _EN
    if LANG == "ko":
        return text
    if _EN is None:
        from .i18n_en import EN

        _EN = EN
    return _EN.get(text, text)


def set_language(lang: str):
    """Switch the running language (tests and offscreen renders; the app restarts instead)."""
    global LANG
    LANG = lang
