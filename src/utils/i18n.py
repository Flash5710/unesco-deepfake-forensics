import json
import os

_STRINGS = None
_DEFAULT_LANG = "es"

_I18N_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "i18n")


def _load():
    global _STRINGS
    if _STRINGS is not None:
        return _STRINGS
    path = os.path.join(_I18N_DIR, "strings.json")
    if not os.path.exists(path):
        _STRINGS = {}
        return _STRINGS
    with open(path, "r", encoding="utf-8") as f:
        _STRINGS = json.load(f)
    return _STRINGS


def t(section: str, key: str, lang: str = None, **params) -> str:
    data = _load()
    if lang is None:
        lang = _DEFAULT_LANG
    if lang not in ("es", "en"):
        lang = "es"

    entry = data.get(section, {}).get(key, None)
    if entry is None:
        return key

    val = entry.get(lang) or entry.get("es") or key
    if isinstance(val, list):
        val = val[0] if len(val) == 1 else str(val)

    if params:
        for k, v in params.items():
            val = val.replace("{" + k + "}", str(v))

    return val


def available_langs() -> list:
    return ["es", "en"]


def set_default_lang(lang: str):
    global _DEFAULT_LANG
    if lang in ("es", "en"):
        _DEFAULT_LANG = lang
