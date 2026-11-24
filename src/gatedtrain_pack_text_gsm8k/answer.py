"""GSM8K final-answer scorer. This file imports only the standard library, so its hash can be pinned as the scorer's
code hash."""

from __future__ import annotations

import re
from decimal import Context, Decimal
from typing import Any

# A hyphen after a word character, a dot or another hyphen is not a minus sign ("10-5", "2023-05", "--5"). A number
# body does not start right after a digit or a dot ("1.2.3" yields "1.2"). ASCII digits only.
_NUMBER = re.compile(r"(?:(?<![A-Za-z0-9_.\-])-)?(?<![0-9.])(?:[0-9][0-9,]*(?:\.[0-9]+)?|\.[0-9]+)")
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_PHRASE = "the answer is"
_PLAIN = re.compile(r"-?(?:[0-9]+\.?[0-9]*|\.[0-9]+)")


def normalise(text: str) -> str | None:
    """The canonical decimal string of `text` (commas and spaces stripped, no trailing zeros), or None."""
    if not isinstance(text, str):
        return None
    s = text.replace(",", "").replace(" ", "").strip()
    if not _PLAIN.fullmatch(s):
        return None
    try:
        d = Decimal(s)
        if d == 0:
            return "0"
        # Exact: the default 28-digit context would round long inputs.
        return format(d.normalize(Context(prec=len(s) + 2)), "f")
    except ArithmeticError:  # InvalidOperation, Overflow on absurdly long inputs
        return None


def _first_number(text: str) -> str | None:
    m = _NUMBER.search(text)
    return normalise(m.group(0)) if m else None


def _last_number(text: str) -> str | None:
    found = _NUMBER.findall(text)
    return normalise(found[-1]) if found else None


def extract(response: str) -> str | None:
    """The normalised number of the first `\\boxed{...}`, else the first number after the last "The answer is"
    (case-insensitive), else the last number in the response."""
    boxed = _BOXED.search(response)
    if boxed:
        value = _first_number(boxed.group(1))
        if value is not None:
            return value
    idx = response.lower().rfind(_PHRASE)
    if idx >= 0:
        value = _first_number(response[idx + len(_PHRASE) :])
        if value is not None:
            return value
    return _last_number(response)


def label_ok(label: Any) -> bool:
    """True when the label is a string that normalises to a number."""
    return isinstance(label, str) and normalise(label) is not None


def score(prediction: Any, label: Any) -> float:
    """1.0 when the number extracted from the prediction equals the normalised label, else 0.0."""
    if not isinstance(prediction, str) or not label_ok(label):
        return 0.0
    got = extract(prediction)
    return 1.0 if got is not None and got == normalise(label) else 0.0
