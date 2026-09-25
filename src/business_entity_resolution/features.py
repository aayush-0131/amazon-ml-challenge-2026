"""Pairwise tabular features for the EXP002 learned matcher."""

from __future__ import annotations

from .multipass import PASS_NAMES, RetrievedCandidate
from .normalize import represent_address, represent_name
from .sampling import SourceRecord
from .similarity import (
    pair_features_from_representations,
    rule_match_score,
    token_overlap,
)


def length_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return min(len(left), len(right)) / max(len(left), len(right))


def first_token_agreement(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    return float(bool(left and right and left[0] == right[0]))


def initial_sequence(tokens: tuple[str, ...]) -> str:
    return "".join(token[0] for token in tokens if token)


FEATURE_NAMES = (
    "normalized_name_exact",
    "name_ratio",
    "name_token_set_ratio",
    "name_token_sort_ratio",
    "name_token_jaccard",
    "name_token_containment",
    "shared_name_tokens",
    "name_idf_overlap",
    "name_length_ratio",
    "name_first_token_agreement",
    "name_initials_agreement",
    "left_address_missing",
    "right_address_missing",
    "normalized_address_exact",
    "address_ratio",
    "address_token_set_ratio",
    "address_token_sort_ratio",
    "address_token_jaccard",
    "address_token_containment",
    "shared_address_tokens",
    "address_idf_overlap",
    "address_length_ratio",
    "digit_token_overlap",
    "left_has_numeric",
    "right_has_numeric",
    "exact_numeric_token_agreement",
    "postal_like_agreement",
    "shared_digit_tokens",
    "source_is_s3",
    "country_agreement",
    "pass_count",
    "best_retrieval_score",
    "second_best_retrieval_score",
    "exact_name_score",
    "exact_address_score",
    "name_signature_score",
    "name_token_score",
    "address_token_score",
    "numeric_address_score",
    "char_name_score",
    "exact_name_reciprocal_rank",
    "exact_address_reciprocal_rank",
    "name_signature_reciprocal_rank",
    "name_token_reciprocal_rank",
    "address_token_reciprocal_rank",
    "numeric_address_reciprocal_rank",
    "char_name_reciprocal_rank",
    "name_address_similarity_product",
    "name_high_address_low",
    "address_high_name_low",
    "exp001_rule_score",
)


def _reciprocal_rank(rank: int) -> float:
    return 1.0 / rank if rank > 0 else 0.0


def build_feature_row(
    source1: SourceRecord, candidate: RetrievedCandidate
) -> dict[str, float]:
    left_name = represent_name(source1.business_name)
    left_address = represent_address(source1.business_address)
    right_name = represent_name(candidate.candidate_name)
    right_address = represent_address(candidate.candidate_address)
    lexical = pair_features_from_representations(
        left_name, left_address, right_name, right_address
    )
    numeric_exact = float(
        bool(left_address.digit_tokens)
        and left_address.digit_tokens == right_address.digit_tokens
    )
    name_initials = initial_sequence(left_name.tokens)
    candidate_initials = initial_sequence(right_name.tokens)
    name_score = lexical.name_token_set_ratio / 100.0
    address_score = lexical.address_token_set_ratio / 100.0

    row = {
        "normalized_name_exact": float(lexical.normalized_name_exact),
        "name_ratio": lexical.name_ratio,
        "name_token_set_ratio": lexical.name_token_set_ratio,
        "name_token_sort_ratio": lexical.name_token_sort_ratio,
        "name_token_jaccard": lexical.name_token_jaccard,
        "name_token_containment": token_overlap(
            left_name.unique_tokens, right_name.unique_tokens
        ),
        "shared_name_tokens": float(candidate.shared_name_tokens),
        "name_idf_overlap": candidate.name_idf_overlap,
        "name_length_ratio": length_ratio(
            left_name.normalized, right_name.normalized
        ),
        "name_first_token_agreement": first_token_agreement(
            left_name.tokens, right_name.tokens
        ),
        "name_initials_agreement": float(
            bool(name_initials) and name_initials == candidate_initials
        ),
        "left_address_missing": float(lexical.left_address_missing),
        "right_address_missing": float(lexical.right_address_missing),
        "normalized_address_exact": float(lexical.normalized_address_exact),
        "address_ratio": lexical.address_ratio,
        "address_token_set_ratio": lexical.address_token_set_ratio,
        "address_token_sort_ratio": lexical.address_token_sort_ratio,
        "address_token_jaccard": lexical.address_token_jaccard,
        "address_token_containment": token_overlap(
            left_address.unique_tokens, right_address.unique_tokens
        ),
        "shared_address_tokens": float(candidate.shared_address_tokens),
        "address_idf_overlap": candidate.address_idf_overlap,
        "address_length_ratio": length_ratio(
            left_address.normalized, right_address.normalized
        ),
        "digit_token_overlap": lexical.digit_token_overlap,
        "left_has_numeric": float(bool(left_address.digit_tokens)),
        "right_has_numeric": float(bool(right_address.digit_tokens)),
        "exact_numeric_token_agreement": numeric_exact,
        "postal_like_agreement": lexical.postal_like_agreement,
        "shared_digit_tokens": float(candidate.shared_digit_tokens),
        "source_is_s3": float(candidate.source == "S3"),
        "country_agreement": float(source1.country == candidate.country),
        "pass_count": float(candidate.pass_count),
        "best_retrieval_score": candidate.best_retrieval_score,
        "second_best_retrieval_score": candidate.second_best_retrieval_score,
        **{
            f"{pass_name}_score": float(getattr(candidate, f"{pass_name}_score"))
            for pass_name in PASS_NAMES
        },
        **{
            f"{pass_name}_reciprocal_rank": _reciprocal_rank(
                int(getattr(candidate, f"{pass_name}_rank"))
            )
            for pass_name in PASS_NAMES
        },
        "name_address_similarity_product": name_score * address_score,
        "name_high_address_low": float(name_score >= 0.90 and address_score < 0.50),
        "address_high_name_low": float(
            address_score >= 0.90 and name_score < 0.50
        ),
        "exp001_rule_score": rule_match_score(lexical),
    }
    if tuple(row) != FEATURE_NAMES:
        raise AssertionError("feature row order differs from FEATURE_NAMES")
    return row
