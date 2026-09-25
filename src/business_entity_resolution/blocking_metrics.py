"""Candidate-recall and candidate-volume metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CandidateMetrics:
    entity_count: int
    true_link_count: int
    retained_true_link_count: int
    positive_link_recall: float
    all_true_links_entity_count: int
    all_true_links_retained_rate: float
    matched_entity_count: int
    matched_entities_all_true_links_retained: int
    matched_entity_all_true_links_retained_rate: float
    average_candidates: float
    median_candidates: float
    p95_candidates: float
    p99_candidates: float
    max_candidates: int
    zero_candidate_entities: int
    reduction_ratio: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def candidate_metrics(
    ground_truth: Mapping[str, Collection[str]],
    candidates: Mapping[str, Collection[str]],
    *,
    eligible_target_count_by_country: Mapping[str, int] | None = None,
    country_by_entity: Mapping[str, str] | None = None,
) -> CandidateMetrics:
    counts: list[int] = []
    true_links = 0
    retained = 0
    all_retained = 0
    matched_entities = 0
    matched_all_retained = 0
    possible_pairs = 0
    for source1_id, truth_values in ground_truth.items():
        truth = set(truth_values)
        predicted = set(candidates.get(source1_id, ()))
        counts.append(len(predicted))
        true_links += len(truth)
        retained += len(truth & predicted)
        all_retained += int(truth <= predicted)
        if truth:
            matched_entities += 1
            matched_all_retained += int(truth <= predicted)
        if eligible_target_count_by_country is not None and country_by_entity is not None:
            possible_pairs += eligible_target_count_by_country[country_by_entity[source1_id]]

    values = np.asarray(counts, dtype=np.int64)
    candidate_total = int(values.sum())
    return CandidateMetrics(
        entity_count=len(ground_truth),
        true_link_count=true_links,
        retained_true_link_count=retained,
        positive_link_recall=retained / true_links if true_links else 1.0,
        all_true_links_entity_count=all_retained,
        all_true_links_retained_rate=all_retained / len(ground_truth),
        matched_entity_count=matched_entities,
        matched_entities_all_true_links_retained=matched_all_retained,
        matched_entity_all_true_links_retained_rate=(
            matched_all_retained / matched_entities if matched_entities else 1.0
        ),
        average_candidates=float(values.mean()),
        median_candidates=float(np.quantile(values, 0.50)),
        p95_candidates=float(np.quantile(values, 0.95)),
        p99_candidates=float(np.quantile(values, 0.99)),
        max_candidates=int(values.max()),
        zero_candidate_entities=int((values == 0).sum()),
        reduction_ratio=(
            1.0 - candidate_total / possible_pairs if possible_pairs else None
        ),
    )


def candidates_by_entity(
    rows: list[tuple[str, str]],
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for source1_id, candidate_id in rows:
        result[source1_id].add(candidate_id)
    return dict(result)
