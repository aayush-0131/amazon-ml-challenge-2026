"""Conservative, country-agnostic text representations for entity resolution."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_WHITESPACE = re.compile(r"\s+")


def normalize_basic(text: object | None) -> str:
    """NFKC-normalize, casefold, replace punctuation with spaces, and trim.

    Letters and digits from every script are preserved. Legal suffixes and address
    components are deliberately retained; this is a representation, not a rewrite
    of the source field.
    """

    if text is None:
        return ""
    value = unicodedata.normalize("NFKC", str(text)).casefold()
    value = "".join(
        " " if (character.isspace() or unicodedata.category(character).startswith("P"))
        else character
        for character in value
    )
    return _WHITESPACE.sub(" ", value).strip()


def normalize_name(text: object | None) -> str:
    """Return the conservative normalized business-name representation."""

    return normalize_basic(text)


def normalize_address(text: object | None) -> str:
    """Return the conservative normalized address representation."""

    return normalize_basic(text)


def tokens(text: str) -> tuple[str, ...]:
    return tuple(text.split()) if text else ()


def digit_tokens(text_tokens: tuple[str, ...]) -> tuple[str, ...]:
    """Return tokens made entirely of Unicode decimal digits."""

    return tuple(token for token in text_tokens if token.isdecimal())


def postal_like_tokens(text_tokens: tuple[str, ...]) -> tuple[str, ...]:
    """Return generic numeric tokens that could encode postal/area identifiers."""

    return tuple(
        token for token in text_tokens if token.isdecimal() and 4 <= len(token) <= 10
    )


@dataclass(frozen=True)
class TextRepresentations:
    raw: str
    normalized: str
    tokens: tuple[str, ...]
    unique_tokens: frozenset[str]
    digit_tokens: frozenset[str]
    postal_like_tokens: frozenset[str]


def represent_name(text: object | None) -> TextRepresentations:
    raw = "" if text is None else str(text)
    normalized = normalize_name(raw)
    token_values = tokens(normalized)
    return TextRepresentations(
        raw=raw,
        normalized=normalized,
        tokens=token_values,
        unique_tokens=frozenset(token_values),
        digit_tokens=frozenset(digit_tokens(token_values)),
        postal_like_tokens=frozenset(postal_like_tokens(token_values)),
    )


def represent_address(text: object | None) -> TextRepresentations:
    raw = "" if text is None else str(text)
    normalized = normalize_address(raw)
    token_values = tokens(normalized)
    return TextRepresentations(
        raw=raw,
        normalized=normalized,
        tokens=token_values,
        unique_tokens=frozenset(token_values),
        digit_tokens=frozenset(digit_tokens(token_values)),
        postal_like_tokens=frozenset(postal_like_tokens(token_values)),
    )
