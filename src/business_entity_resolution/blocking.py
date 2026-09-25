"""Bounded-memory lexical candidate generation for EXP001."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd

from .normalize import (
    TextRepresentations,
    represent_address,
    represent_name,
)
from .sampling import SourceRecord
from .similarity import (
    PairFeatures,
    pair_features_from_representations,
    rule_match_score,
)


@dataclass(frozen=True)
class LexicalBlockerConfig:
    top_k_per_source: int = 25
    min_informative_token_length: int = 4
    max_query_token_frequency: int = 25
    prune_multiplier: int = 2

    def __post_init__(self) -> None:
        if self.top_k_per_source <= 0:
            raise ValueError("top_k_per_source must be positive")
        if self.min_informative_token_length <= 0:
            raise ValueError("min_informative_token_length must be positive")
        if self.max_query_token_frequency <= 0:
            raise ValueError("max_query_token_frequency must be positive")
        if self.prune_multiplier < 2:
            raise ValueError("prune_multiplier must be at least 2")


@dataclass(frozen=True)
class QueryRecord:
    entity_id: str
    country: str
    name: TextRepresentations
    address: TextRepresentations


@dataclass(frozen=True)
class CandidateEvidence:
    source1_entity_id: str
    candidate_entity_id: str
    source: str
    country: str
    candidate_name: str
    candidate_address: str
    blocking_score: float
    provenance: tuple[str, ...]
    shared_name_tokens: int
    shared_address_tokens: int
    shared_digit_tokens: int


@dataclass(frozen=True)
class ScoredCandidate:
    evidence: CandidateEvidence
    features: PairFeatures
    match_score: float

    def to_dict(self) -> dict[str, object]:
        row: dict[str, object] = asdict(self.evidence)
        row["provenance"] = ",".join(self.evidence.provenance)
        row.update(self.features.to_dict())
        row["match_score"] = self.match_score
        return row


@dataclass
class _CandidateState:
    evidence: CandidateEvidence


class LexicalBlocker:
    """Query-side inverted-index blocker with bounded candidates per S1 entity."""

    def __init__(
        self,
        query_records: Iterable[SourceRecord],
        *,
        source: str,
        config: LexicalBlockerConfig | None = None,
    ) -> None:
        if source not in {"S2", "S3"}:
            raise ValueError("source must be S2 or S3")
        self.source = source
        self.config = config or LexicalBlockerConfig()
        self.queries: dict[str, QueryRecord] = {}
        self.exact_name: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.exact_address: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.name_tokens: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.address_tokens: dict[tuple[str, str], list[str]] = defaultdict(list)
        self._candidates: dict[str, dict[str, _CandidateState]] = defaultdict(dict)
        self.target_rows_seen = 0
        self.target_country_counts: Counter[str] = Counter()
        self._build_query_indexes(query_records)

    def _informative(self, token: str) -> bool:
        return len(token) >= self.config.min_informative_token_length

    def _build_query_indexes(self, records: Iterable[SourceRecord]) -> None:
        name_frequency: Counter[tuple[str, str]] = Counter()
        address_frequency: Counter[tuple[str, str]] = Counter()
        for record in records:
            if record.entity_id in self.queries:
                raise ValueError(f"Duplicate query entity: {record.entity_id}")
            query = QueryRecord(
                entity_id=record.entity_id,
                country=record.country,
                name=represent_name(record.business_name),
                address=represent_address(record.business_address),
            )
            self.queries[record.entity_id] = query
            name_frequency.update(
                (record.country, token)
                for token in query.name.unique_tokens
                if self._informative(token)
            )
            address_frequency.update(
                (record.country, token)
                for token in query.address.unique_tokens
                if self._informative(token)
            )

        for query in self.queries.values():
            if query.name.normalized:
                self.exact_name[(query.country, query.name.normalized)].append(
                    query.entity_id
                )
            if query.address.normalized:
                self.exact_address[(query.country, query.address.normalized)].append(
                    query.entity_id
                )
            for token in query.name.unique_tokens:
                key = (query.country, token)
                if (
                    self._informative(token)
                    and name_frequency[key] <= self.config.max_query_token_frequency
                ):
                    self.name_tokens[key].append(query.entity_id)
            for token in query.address.unique_tokens:
                key = (query.country, token)
                if (
                    self._informative(token)
                    and address_frequency[key]
                    <= self.config.max_query_token_frequency
                ):
                    self.address_tokens[key].append(query.entity_id)

    def _prune(self, source1_id: str) -> None:
        candidates = self._candidates[source1_id]
        keep = sorted(
            candidates.values(),
            key=lambda state: (
                -state.evidence.blocking_score,
                state.evidence.candidate_entity_id,
            ),
        )[: self.config.top_k_per_source]
        self._candidates[source1_id] = {
            state.evidence.candidate_entity_id: state for state in keep
        }

    def process_record(self, record: SourceRecord) -> None:
        if not record.entity_id.startswith(f"{self.source}-"):
            raise ValueError(
                f"Target {record.entity_id!r} does not belong to {self.source}"
            )
        self.target_rows_seen += 1
        self.target_country_counts[record.country] += 1
        target_name = represent_name(record.business_name)
        target_address = represent_address(record.business_address)

        query_ids: set[str] = set()
        if target_name.normalized:
            query_ids.update(
                self.exact_name.get((record.country, target_name.normalized), ())
            )
        if target_address.normalized:
            query_ids.update(
                self.exact_address.get((record.country, target_address.normalized), ())
            )
        for token in target_name.unique_tokens:
            query_ids.update(self.name_tokens.get((record.country, token), ()))
        for token in target_address.unique_tokens:
            query_ids.update(self.address_tokens.get((record.country, token), ()))

        for source1_id in query_ids:
            query = self.queries[source1_id]
            name_shared = len(query.name.unique_tokens & target_name.unique_tokens)
            address_shared = len(
                query.address.unique_tokens & target_address.unique_tokens
            )
            digit_shared = len(
                query.address.digit_tokens & target_address.digit_tokens
            )
            exact_name = bool(
                query.name.normalized
                and query.name.normalized == target_name.normalized
            )
            exact_address = bool(
                query.address.normalized
                and query.address.normalized == target_address.normalized
            )
            if not (
                exact_name
                or exact_address
                or name_shared >= 1
                or address_shared >= 2
                or (address_shared >= 1 and digit_shared >= 1)
            ):
                continue

            provenance: list[str] = []
            if exact_name:
                provenance.append("exact_name")
            if exact_address:
                provenance.append("exact_address")
            if name_shared:
                provenance.append("name_token")
            if address_shared:
                provenance.append("address_token")
            if digit_shared:
                provenance.append("digit_token")
            blocking_score = (
                100.0 * exact_name
                + 90.0 * exact_address
                + 20.0 * name_shared
                + 8.0 * address_shared
                + 12.0 * digit_shared
            )
            evidence = CandidateEvidence(
                source1_entity_id=source1_id,
                candidate_entity_id=record.entity_id,
                source=self.source,
                country=record.country,
                candidate_name=record.business_name,
                candidate_address=record.business_address,
                blocking_score=blocking_score,
                provenance=tuple(provenance),
                shared_name_tokens=name_shared,
                shared_address_tokens=address_shared,
                shared_digit_tokens=digit_shared,
            )
            self._candidates[source1_id][record.entity_id] = _CandidateState(evidence)
            if (
                len(self._candidates[source1_id])
                > self.config.top_k_per_source * self.config.prune_multiplier
            ):
                self._prune(source1_id)

    def process_chunk(self, frame: pd.DataFrame) -> None:
        for row in frame.itertuples(index=False):
            self.process_record(
                SourceRecord(
                    entity_id=row.entity_id,
                    business_name=row.business_name,
                    business_address=row.business_address,
                    country=row.country,
                )
            )

    def finalize(self) -> list[ScoredCandidate]:
        scored: list[ScoredCandidate] = []
        for source1_id in sorted(tuple(self._candidates)):
            self._prune(source1_id)
            query = self.queries[source1_id]
            states = self._candidates.pop(source1_id)
            for state in states.values():
                evidence = state.evidence
                features = pair_features_from_representations(
                    query.name,
                    query.address,
                    represent_name(evidence.candidate_name),
                    represent_address(evidence.candidate_address),
                )
                scored.append(
                    ScoredCandidate(
                        evidence=evidence,
                        features=features,
                        match_score=rule_match_score(features),
                    )
                )
        return sorted(
            scored,
            key=lambda row: (
                row.evidence.source1_entity_id,
                -row.evidence.blocking_score,
                row.evidence.candidate_entity_id,
            ),
        )


def generate_candidates(
    query_records: Iterable[SourceRecord],
    target_records: Iterable[SourceRecord],
    *,
    source: str,
    config: LexicalBlockerConfig | None = None,
) -> list[ScoredCandidate]:
    blocker = LexicalBlocker(query_records, source=source, config=config)
    for record in target_records:
        blocker.process_record(record)
    return blocker.finalize()
