"""Inexpensive lexical pair features used by EXP001."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rapidfuzz import fuzz

from .normalize import TextRepresentations, represent_address, represent_name


def token_jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def token_overlap(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def _fuzzy(function: object, left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return float(function(left, right))


@dataclass(frozen=True)
class PairFeatures:
    raw_name_exact: bool
    normalized_name_exact: bool
    raw_address_exact: bool
    normalized_address_exact: bool
    name_ratio: float
    name_token_set_ratio: float
    name_token_sort_ratio: float
    address_ratio: float
    address_token_set_ratio: float
    address_token_sort_ratio: float
    name_token_jaccard: float
    address_token_jaccard: float
    digit_token_overlap: float
    postal_like_agreement: float
    left_address_missing: bool
    right_address_missing: bool

    def to_dict(self) -> dict[str, bool | float]:
        return asdict(self)


def pair_features_from_representations(
    left_name: TextRepresentations,
    left_address: TextRepresentations,
    right_name: TextRepresentations,
    right_address: TextRepresentations,
) -> PairFeatures:
    digits_left = left_address.digit_tokens
    digits_right = right_address.digit_tokens
    postal_left = left_address.postal_like_tokens
    postal_right = right_address.postal_like_tokens
    return PairFeatures(
        raw_name_exact=bool(left_name.raw and left_name.raw == right_name.raw),
        normalized_name_exact=bool(
            left_name.normalized and left_name.normalized == right_name.normalized
        ),
        raw_address_exact=bool(
            left_address.raw and left_address.raw == right_address.raw
        ),
        normalized_address_exact=bool(
            left_address.normalized
            and left_address.normalized == right_address.normalized
        ),
        name_ratio=_fuzzy(fuzz.ratio, left_name.normalized, right_name.normalized),
        name_token_set_ratio=_fuzzy(
            fuzz.token_set_ratio, left_name.normalized, right_name.normalized
        ),
        name_token_sort_ratio=_fuzzy(
            fuzz.token_sort_ratio, left_name.normalized, right_name.normalized
        ),
        address_ratio=_fuzzy(
            fuzz.ratio, left_address.normalized, right_address.normalized
        ),
        address_token_set_ratio=_fuzzy(
            fuzz.token_set_ratio, left_address.normalized, right_address.normalized
        ),
        address_token_sort_ratio=_fuzzy(
            fuzz.token_sort_ratio, left_address.normalized, right_address.normalized
        ),
        name_token_jaccard=token_jaccard(
            left_name.unique_tokens, right_name.unique_tokens
        ),
        address_token_jaccard=token_jaccard(
            left_address.unique_tokens, right_address.unique_tokens
        ),
        digit_token_overlap=token_overlap(digits_left, digits_right),
        postal_like_agreement=float(bool(postal_left & postal_right)),
        left_address_missing=not left_address.normalized,
        right_address_missing=not right_address.normalized,
    )


def pair_features(
    left_name: object | None,
    left_address: object | None,
    right_name: object | None,
    right_address: object | None,
) -> PairFeatures:
    return pair_features_from_representations(
        represent_name(left_name),
        represent_address(left_address),
        represent_name(right_name),
        represent_address(right_address),
    )


def rule_match_score(features: PairFeatures) -> float:
    """Deterministic precision-oriented EXP001 score on a 0–100 scale."""

    score = (
        0.36 * features.name_ratio
        + 0.24 * features.name_token_set_ratio
        + 0.08 * features.name_token_sort_ratio
        + 0.12 * features.address_ratio
        + 0.12 * features.address_token_set_ratio
        + 0.04 * features.address_token_sort_ratio
        + 8.0 * features.name_token_jaccard
        + 5.0 * features.address_token_jaccard
        + 3.0 * features.digit_token_overlap
        + 2.0 * features.postal_like_agreement
    )
    if features.normalized_name_exact:
        score += 8.0
    if features.normalized_address_exact:
        score += 7.0
    # Quantization prevents platform/parser differences around threshold boundaries
    # (for example 95.99999999999999 versus 96.00000000000001).
    return round(min(100.0, score), 6)
