from __future__ import annotations

from business_entity_resolution.normalize import (
    normalize_address,
    normalize_basic,
    normalize_name,
    represent_address,
)


def test_basic_normalization_is_conservative_and_unicode_aware() -> None:
    assert normalize_basic("  ACME, Inc.\t１２３  ") == "acme inc 123"
    assert normalize_basic("राम-मार्केटिंग Pvt. Ltd.") == "राम मार्केटिंग pvt ltd"


def test_name_does_not_strip_legal_suffixes() -> None:
    assert normalize_name("Example LLC") == "example llc"


def test_address_preserves_digits_and_exposes_representations() -> None:
    assert normalize_address("12-A Main Rd.") == "12 a main rd"
    represented = represent_address("ZIP 75001 / 42")
    assert represented.raw == "ZIP 75001 / 42"
    assert represented.normalized == "zip 75001 42"
    assert represented.digit_tokens == frozenset({"75001", "42"})
    assert represented.postal_like_tokens == frozenset({"75001"})
