"""1.4.0: Korean / English interface text."""
import ast
import os
import re

from blot_profiler import i18n
from blot_profiler.i18n_en import EN

ROOT = os.path.join(os.path.dirname(__file__), "..", "blot_profiler")
KO = re.compile("[가-힣]")


def _t_literals():
    for dirpath, _, files in os.walk(ROOT):
        for f in files:
            if f.endswith(".py") and f != "i18n_en.py":
                tree = ast.parse(open(os.path.join(dirpath, f), encoding="utf-8").read())
                for n in ast.walk(tree):
                    if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_t"
                            and n.args and isinstance(n.args[0], ast.Constant)):
                        yield f, n.args[0].value


def test_every_text_has_english():
    missing = [(f, s) for f, s in _t_literals() if s not in EN]
    assert not missing, missing[:5]


def test_english_keeps_format_fields_and_has_no_korean():
    fields = lambda s: sorted(re.findall(r"\{\d+[^}]*\}", s))  # noqa: E731
    for k, v in EN.items():
        assert fields(k) == fields(v), k
        if "Language" not in v and "한국어" not in v:
            assert not KO.search(v), v


def test_switch_language():
    old = i18n.LANG
    try:
        i18n.set_language("en")
        assert i18n._t("밴드 찾기") == "Find bands"
        assert i18n._t("레인 {0}개, 박스 {1}개, 줄 {2}개").format(3, 4, 5) == "3 lanes, 4 boxes, 5 rows"
        assert i18n._t("아직 번역 없는 글") == "아직 번역 없는 글"
        i18n.set_language("ko")
        assert i18n._t("밴드 찾기") == "밴드 찾기"
    finally:
        i18n.set_language(old)


def test_language_setting_round_trip():
    assert i18n.setting() in ("system", "ko", "en")
    i18n.set_setting("en")
    assert i18n.setting() == "en"
    i18n.set_setting("system")
    assert i18n.resolve("system") in ("ko", "en")
