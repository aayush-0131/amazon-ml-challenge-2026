from __future__ import annotations

from business_entity_resolution.blocking import (
    LexicalBlockerConfig,
    generate_candidates,
)
from business_entity_resolution.blocking_metrics import candidate_metrics
from business_entity_resolution.sampling import SourceRecord


def record(
    entity_id: str, name: str, address: str, country: str = "US"
) -> SourceRecord:
    return SourceRecord(entity_id, name, address, country)


def test_candidate_generation_handles_multiple_matches_and_is_deterministic() -> None:
    queries = [record("S1-1", "Acme LLC", "12 Main Street 75001")]
    targets = [
        record("S2-2", "ACME LLC", "12 Main St 75001"),
        record("S2-1", "Acme, LLC", "12 Main Street 75001"),
    ]
    config = LexicalBlockerConfig(top_k_per_source=5)

    first = generate_candidates(queries, targets, source="S2", config=config)
    second = generate_candidates(
        queries, list(reversed(targets)), source="S2", config=config
    )

    assert {row.evidence.candidate_entity_id for row in first} == {"S2-1", "S2-2"}
    assert [row.to_dict() for row in first] == [row.to_dict() for row in second]


def test_candidate_generation_never_crosses_country() -> None:
    queries = [record("S1-1", "Acme LLC", "12 Main Street", "US")]
    targets = [record("S2-1", "Acme LLC", "12 Main Street", "India")]

    assert generate_candidates(queries, targets, source="S2") == []


def test_singleton_can_receive_candidates_without_becoming_a_positive() -> None:
    metrics = candidate_metrics(
        {"S1-1": frozenset()}, {"S1-1": {"S2-1"}}
    )
    assert metrics.positive_link_recall == 1.0
    assert metrics.all_true_links_retained_rate == 1.0
    assert metrics.matched_entity_count == 0
    assert metrics.matched_entity_all_true_links_retained_rate == 1.0
    assert metrics.average_candidates == 1.0


def test_candidate_recall_counts_all_links_for_multi_match_entity() -> None:
    metrics = candidate_metrics(
        {"S1-1": {"S2-1", "S2-2", "S3-1"}},
        {"S1-1": {"S2-1", "S3-1"}},
    )
    assert metrics.positive_link_recall == 2 / 3
    assert metrics.all_true_links_retained_rate == 0.0
    assert metrics.matched_entity_all_true_links_retained_rate == 0.0
    assert metrics.average_candidates == 2.0
