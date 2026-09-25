from business_entity_resolution.features import FEATURE_NAMES, build_feature_row
from business_entity_resolution.multipass import RetrievedCandidate
from business_entity_resolution.sampling import SourceRecord


def _candidate(**overrides: object) -> RetrievedCandidate:
    values: dict[str, object] = {
        "source1_entity_id": "S1-1",
        "candidate_entity_id": "S2-1",
        "source": "S2",
        "country": "US",
        "candidate_name": "Acme Trading LLC",
        "candidate_address": "12 Market Rd 94105",
        "exact_name_score": 100.0,
        "exact_address_score": 0.0,
        "name_signature_score": 100.0,
        "name_token_score": 90.0,
        "address_token_score": 80.0,
        "numeric_address_score": 70.0,
        "char_name_score": 90.0,
        "exact_name_rank": 1,
        "exact_address_rank": 0,
        "name_signature_rank": 1,
        "name_token_rank": 2,
        "address_token_rank": 3,
        "numeric_address_rank": 4,
        "char_name_rank": 2,
        "name_idf_overlap": 1.0,
        "address_idf_overlap": 0.8,
        "name_token_jaccard": 1.0,
        "address_token_jaccard": 0.6,
        "digit_token_overlap": 1.0,
        "char_name_jaccard": 1.0,
        "shared_name_tokens": 3,
        "shared_address_tokens": 3,
        "shared_digit_tokens": 2,
        "pass_count": 6,
        "best_retrieval_score": 120.0,
        "second_best_retrieval_score": 100.0,
    }
    values.update(overrides)
    return RetrievedCandidate(**values)


def test_feature_row_exact_and_numeric_behavior() -> None:
    source1 = SourceRecord(
        "S1-1", "ACME Trading, LLC", "12 Market Road 94105", "US"
    )
    features = build_feature_row(source1, _candidate())
    assert tuple(features) == FEATURE_NAMES
    assert features["normalized_name_exact"] == 1.0
    assert features["country_agreement"] == 1.0
    assert features["exact_numeric_token_agreement"] == 1.0
    assert features["source_is_s3"] == 0.0
    assert features["exact_name_reciprocal_rank"] == 1.0


def test_feature_row_missing_address_and_source_indicator() -> None:
    source1 = SourceRecord("S1-1", "Example", "", "FR")
    features = build_feature_row(
        source1,
        _candidate(
            source="S3",
            country="FR",
            candidate_name="Exemple",
            candidate_address="",
        ),
    )
    assert features["left_address_missing"] == 1.0
    assert features["right_address_missing"] == 1.0
    assert features["source_is_s3"] == 1.0
    assert features["address_length_ratio"] == 0.0
    assert features["name_length_ratio"] == 1.0
