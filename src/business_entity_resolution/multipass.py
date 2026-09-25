"""Scored multi-pass lexical retrieval used by EXP002."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd

from .normalize import TextRepresentations, represent_address, represent_name
from .sampling import SourceRecord
from .similarity import token_jaccard, token_overlap

PASS_NAMES = (
    "exact_name",
    "exact_address",
    "name_signature",
    "name_token",
    "address_token",
    "numeric_address",
    "char_name",
)


def character_ngrams(text: str, n: int = 4) -> frozenset[str]:
    compact = "".join(text.split())
    if not compact:
        return frozenset()
    if len(compact) <= n:
        return frozenset({compact})
    return frozenset(compact[index : index + n] for index in range(len(compact) - n + 1))


def name_signature(representation: TextRepresentations) -> str:
    return " ".join(sorted(representation.unique_tokens))


@dataclass(frozen=True)
class MultiPassConfig:
    max_pool_per_source: int = 120
    min_name_token_length: int = 3
    min_address_token_length: int = 3
    max_name_query_df: int = 60
    max_address_query_df: int = 40
    max_numeric_query_df: int = 250
    max_char_query_df: int = 6
    min_char_shared_ngrams: int = 2
    min_char_jaccard: float = 0.20
    prune_multiplier: float = 1.25


@dataclass(frozen=True)
class CandidateBudget:
    name: str
    total_per_source: int
    exact_name: int
    exact_address: int
    name_signature: int
    name_token: int
    address_token: int
    numeric_address: int
    char_name: int

    def pass_budget(self, pass_name: str) -> int:
        return int(getattr(self, pass_name))


@dataclass(frozen=True, slots=True)
class MultiPassQuery:
    entity_id: str
    country: str
    name: TextRepresentations
    address: TextRepresentations
    name_ngrams: frozenset[str]


@dataclass(slots=True)
class _CandidateState:
    candidate_id: str
    candidate_name: str
    candidate_address: str
    country: str
    exact_name_score: float
    exact_address_score: float
    name_signature_score: float
    name_token_score: float
    address_token_score: float
    numeric_address_score: float
    char_name_score: float
    name_idf_overlap: float
    address_idf_overlap: float
    name_token_jaccard: float
    address_token_jaccard: float
    digit_token_overlap: float
    char_name_jaccard: float
    shared_name_tokens: int
    shared_address_tokens: int
    shared_digit_tokens: int

    @property
    def scores(self) -> tuple[float, ...]:
        return (
            self.exact_name_score,
            self.exact_address_score,
            self.name_signature_score,
            self.name_token_score,
            self.address_token_score,
            self.numeric_address_score,
            self.char_name_score,
        )

    @property
    def best_score(self) -> float:
        return max(self.scores)

    @property
    def second_best_score(self) -> float:
        ordered = sorted(self.scores, reverse=True)
        return ordered[1]

    @property
    def pass_count(self) -> int:
        return sum(score > 0 for score in self.scores)


@dataclass(frozen=True, slots=True)
class RetrievedCandidate:
    source1_entity_id: str
    candidate_entity_id: str
    source: str
    country: str
    candidate_name: str
    candidate_address: str
    exact_name_score: float
    exact_address_score: float
    name_signature_score: float
    name_token_score: float
    address_token_score: float
    numeric_address_score: float
    char_name_score: float
    exact_name_rank: int
    exact_address_rank: int
    name_signature_rank: int
    name_token_rank: int
    address_token_rank: int
    numeric_address_rank: int
    char_name_rank: int
    name_idf_overlap: float
    address_idf_overlap: float
    name_token_jaccard: float
    address_token_jaccard: float
    digit_token_overlap: float
    char_name_jaccard: float
    shared_name_tokens: int
    shared_address_tokens: int
    shared_digit_tokens: int
    pass_count: int
    best_retrieval_score: float
    second_best_retrieval_score: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def selected_by(self, budget: CandidateBudget) -> bool:
        return any(
            0 < int(getattr(self, f"{pass_name}_rank"))
            <= budget.pass_budget(pass_name)
            for pass_name in PASS_NAMES
        )


class MultiPassBlocker:
    """Country-compatible query-side multi-pass blocker with bounded state."""

    def __init__(
        self,
        query_records: Iterable[SourceRecord],
        *,
        source: str,
        config: MultiPassConfig | None = None,
    ) -> None:
        if source not in {"S2", "S3"}:
            raise ValueError("source must be S2 or S3")
        self.source = source
        self.config = config or MultiPassConfig()
        self.queries: dict[str, MultiPassQuery] = {}
        self.country_query_counts: Counter[str] = Counter()
        self.name_df: Counter[tuple[str, str]] = Counter()
        self.address_df: Counter[tuple[str, str]] = Counter()
        self.numeric_df: Counter[tuple[str, str]] = Counter()
        self.char_df: Counter[tuple[str, str]] = Counter()
        self.exact_name: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.exact_address: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.signature: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.name_index: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.address_index: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.numeric_index: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.char_index: dict[tuple[str, str], list[str]] = defaultdict(list)
        self._candidates: dict[str, dict[str, _CandidateState]] = defaultdict(dict)
        self.target_rows_seen = 0
        self.target_country_counts: Counter[str] = Counter()
        self._build_indexes(query_records)

    def _build_indexes(self, records: Iterable[SourceRecord]) -> None:
        for record in records:
            name = represent_name(record.business_name)
            address = represent_address(record.business_address)
            query = MultiPassQuery(
                record.entity_id,
                record.country,
                name,
                address,
                character_ngrams(name.normalized),
            )
            if record.entity_id in self.queries:
                raise ValueError(f"Duplicate query entity: {record.entity_id}")
            self.queries[record.entity_id] = query
            self.country_query_counts[record.country] += 1
            self.name_df.update((record.country, token) for token in name.unique_tokens)
            self.address_df.update(
                (record.country, token) for token in address.unique_tokens
            )
            self.numeric_df.update(
                (record.country, token) for token in address.digit_tokens
            )
            self.char_df.update(
                (record.country, gram) for gram in query.name_ngrams
            )

        for query in self.queries.values():
            country = query.country
            if query.name.normalized:
                self.exact_name[(country, query.name.normalized)].append(query.entity_id)
                self.signature[(country, name_signature(query.name))].append(
                    query.entity_id
                )
            if query.address.normalized:
                self.exact_address[(country, query.address.normalized)].append(
                    query.entity_id
                )
            for token in query.name.unique_tokens:
                if (
                    len(token) >= self.config.min_name_token_length
                    and self.name_df[(country, token)] <= self.config.max_name_query_df
                ):
                    self.name_index[(country, token)].append(query.entity_id)
            for token in query.address.unique_tokens:
                if (
                    len(token) >= self.config.min_address_token_length
                    and self.address_df[(country, token)]
                    <= self.config.max_address_query_df
                ):
                    self.address_index[(country, token)].append(query.entity_id)
            for token in query.address.digit_tokens:
                if self.numeric_df[(country, token)] <= self.config.max_numeric_query_df:
                    self.numeric_index[(country, token)].append(query.entity_id)
            for gram in query.name_ngrams:
                if self.char_df[(country, gram)] <= self.config.max_char_query_df:
                    self.char_index[(country, gram)].append(query.entity_id)

    def _idf(self, country: str, token: str, *, address: bool) -> float:
        frequency = (self.address_df if address else self.name_df)[(country, token)]
        return math.log((self.country_query_counts[country] + 1) / (frequency + 1)) + 1

    def _idf_overlap(
        self,
        query_tokens: frozenset[str],
        target_tokens: frozenset[str],
        country: str,
        *,
        address: bool,
    ) -> float:
        if not query_tokens:
            return 0.0
        weights = {
            token: self._idf(country, token, address=address)
            for token in query_tokens
        }
        denominator = sum(weights.values())
        return sum(weights[token] for token in query_tokens & target_tokens) / denominator

    def _prune(self, source1_id: str) -> None:
        keep = sorted(
            self._candidates[source1_id].values(),
            key=lambda state: (
                -state.best_score,
                -state.pass_count,
                state.candidate_id,
            ),
        )[: self.config.max_pool_per_source]
        self._candidates[source1_id] = {state.candidate_id: state for state in keep}

    def process_record(self, record: SourceRecord) -> None:
        if not record.entity_id.startswith(f"{self.source}-"):
            raise ValueError(f"Target {record.entity_id!r} is not from {self.source}")
        self.target_rows_seen += 1
        self.target_country_counts[record.country] += 1
        name = represent_name(record.business_name)
        address = represent_address(record.business_address)
        ngrams = character_ngrams(name.normalized)
        signature = name_signature(name)

        query_ids: set[str] = set()
        if name.normalized:
            query_ids.update(self.exact_name.get((record.country, name.normalized), ()))
            query_ids.update(self.signature.get((record.country, signature), ()))
        if address.normalized:
            query_ids.update(
                self.exact_address.get((record.country, address.normalized), ())
            )
        for token in name.unique_tokens:
            query_ids.update(self.name_index.get((record.country, token), ()))
        for token in address.unique_tokens:
            query_ids.update(self.address_index.get((record.country, token), ()))
        for token in address.digit_tokens:
            query_ids.update(self.numeric_index.get((record.country, token), ()))

        char_hits: Counter[str] = Counter()
        for gram in ngrams:
            char_hits.update(self.char_index.get((record.country, gram), ()))
        query_ids.update(
            query_id
            for query_id, count in char_hits.items()
            if count >= self.config.min_char_shared_ngrams
        )

        for source1_id in query_ids:
            query = self.queries[source1_id]
            shared_name = len(query.name.unique_tokens & name.unique_tokens)
            shared_address = len(query.address.unique_tokens & address.unique_tokens)
            shared_digits = len(query.address.digit_tokens & address.digit_tokens)
            name_jaccard = token_jaccard(query.name.unique_tokens, name.unique_tokens)
            address_jaccard = token_jaccard(
                query.address.unique_tokens, address.unique_tokens
            )
            digit_overlap = token_overlap(
                query.address.digit_tokens, address.digit_tokens
            )
            char_jaccard = token_jaccard(query.name_ngrams, ngrams)
            name_idf = self._idf_overlap(
                query.name.unique_tokens,
                name.unique_tokens,
                record.country,
                address=False,
            )
            address_idf = self._idf_overlap(
                query.address.unique_tokens,
                address.unique_tokens,
                record.country,
                address=True,
            )
            exact_name = bool(
                query.name.normalized and query.name.normalized == name.normalized
            )
            exact_address = bool(
                query.address.normalized
                and query.address.normalized == address.normalized
            )
            exact_signature = bool(
                signature and name_signature(query.name) == signature
            )
            has_name_token = shared_name > 0 and name_idf > 0
            has_address_token = shared_address >= 2 or address_idf >= 0.25
            has_numeric = shared_digits > 0
            has_char = (
                char_hits[source1_id] >= self.config.min_char_shared_ngrams
                and char_jaccard >= self.config.min_char_jaccard
            )
            if not (
                exact_name
                or exact_address
                or exact_signature
                or has_name_token
                or has_address_token
                or has_numeric
                or has_char
            ):
                continue

            state = _CandidateState(
                candidate_id=record.entity_id,
                candidate_name=record.business_name,
                candidate_address=record.business_address,
                country=record.country,
                exact_name_score=(
                    100 + 30 * address_jaccard + 10 * digit_overlap
                    if exact_name
                    else 0.0
                ),
                exact_address_score=(
                    100 + 30 * name_jaccard if exact_address else 0.0
                ),
                name_signature_score=(
                    105 + 20 * address_jaccard if exact_signature else 0.0
                ),
                name_token_score=(
                    100 * name_idf + 20 * name_jaccard if has_name_token else 0.0
                ),
                address_token_score=(
                    100 * address_idf + 20 * address_jaccard + 20 * digit_overlap
                    if has_address_token
                    else 0.0
                ),
                numeric_address_score=(
                    80 * digit_overlap + 20 * name_jaccard + 20 * address_jaccard
                    if has_numeric
                    else 0.0
                ),
                char_name_score=(
                    100 * char_jaccard + 20 * name_jaccard if has_char else 0.0
                ),
                name_idf_overlap=name_idf,
                address_idf_overlap=address_idf,
                name_token_jaccard=name_jaccard,
                address_token_jaccard=address_jaccard,
                digit_token_overlap=digit_overlap,
                char_name_jaccard=char_jaccard,
                shared_name_tokens=shared_name,
                shared_address_tokens=shared_address,
                shared_digit_tokens=shared_digits,
            )
            self._candidates[source1_id][record.entity_id] = state
            if len(self._candidates[source1_id]) > int(
                self.config.max_pool_per_source * self.config.prune_multiplier
            ):
                self._prune(source1_id)

    def process_chunk(self, frame: pd.DataFrame) -> None:
        for row in frame.itertuples(index=False):
            self.process_record(
                SourceRecord(
                    row.entity_id,
                    row.business_name,
                    row.business_address,
                    row.country,
                )
            )

    def finalize(self) -> list[RetrievedCandidate]:
        rows: list[RetrievedCandidate] = []
        for source1_id in sorted(tuple(self._candidates)):
            self._prune(source1_id)
            states = self._candidates.pop(source1_id)
            ranks: dict[str, dict[str, int]] = {name: {} for name in PASS_NAMES}
            for pass_name in PASS_NAMES:
                score_name = f"{pass_name}_score"
                ranked = sorted(
                    (
                        state
                        for state in states.values()
                        if getattr(state, score_name) > 0
                    ),
                    key=lambda state: (-getattr(state, score_name), state.candidate_id),
                )
                ranks[pass_name] = {
                    state.candidate_id: rank
                    for rank, state in enumerate(ranked, start=1)
                }
            for state in states.values():
                rows.append(
                    RetrievedCandidate(
                        source1_entity_id=source1_id,
                        candidate_entity_id=state.candidate_id,
                        source=self.source,
                        country=state.country,
                        candidate_name=state.candidate_name,
                        candidate_address=state.candidate_address,
                        exact_name_score=state.exact_name_score,
                        exact_address_score=state.exact_address_score,
                        name_signature_score=state.name_signature_score,
                        name_token_score=state.name_token_score,
                        address_token_score=state.address_token_score,
                        numeric_address_score=state.numeric_address_score,
                        char_name_score=state.char_name_score,
                        exact_name_rank=ranks["exact_name"].get(state.candidate_id, 0),
                        exact_address_rank=ranks["exact_address"].get(
                            state.candidate_id, 0
                        ),
                        name_signature_rank=ranks["name_signature"].get(
                            state.candidate_id, 0
                        ),
                        name_token_rank=ranks["name_token"].get(state.candidate_id, 0),
                        address_token_rank=ranks["address_token"].get(
                            state.candidate_id, 0
                        ),
                        numeric_address_rank=ranks["numeric_address"].get(
                            state.candidate_id, 0
                        ),
                        char_name_rank=ranks["char_name"].get(state.candidate_id, 0),
                        name_idf_overlap=state.name_idf_overlap,
                        address_idf_overlap=state.address_idf_overlap,
                        name_token_jaccard=state.name_token_jaccard,
                        address_token_jaccard=state.address_token_jaccard,
                        digit_token_overlap=state.digit_token_overlap,
                        char_name_jaccard=state.char_name_jaccard,
                        shared_name_tokens=state.shared_name_tokens,
                        shared_address_tokens=state.shared_address_tokens,
                        shared_digit_tokens=state.shared_digit_tokens,
                        pass_count=state.pass_count,
                        best_retrieval_score=state.best_score,
                        second_best_retrieval_score=state.second_best_score,
                    )
                )
        return sorted(
            rows,
            key=lambda row: (
                row.source1_entity_id,
                -row.best_retrieval_score,
                row.candidate_entity_id,
            ),
        )


def select_candidates_for_budget(
    candidates: Iterable[RetrievedCandidate], budget: CandidateBudget
) -> list[RetrievedCandidate]:
    selected = [candidate for candidate in candidates if candidate.selected_by(budget)]
    return sorted(
        selected,
        key=lambda candidate: (
            -candidate.best_retrieval_score,
            -candidate.pass_count,
            candidate.candidate_entity_id,
        ),
    )[: budget.total_per_source]


def retrieved_candidate_from_mapping(row: dict[str, str]) -> RetrievedCandidate:
    """Rehydrate a candidate from the deterministic TSV artifact."""

    string_fields = {
        "source1_entity_id",
        "candidate_entity_id",
        "source",
        "country",
        "candidate_name",
        "candidate_address",
    }
    integer_fields = {
        *(f"{pass_name}_rank" for pass_name in PASS_NAMES),
        "shared_name_tokens",
        "shared_address_tokens",
        "shared_digit_tokens",
        "pass_count",
    }
    values: dict[str, object] = {}
    for field_name in RetrievedCandidate.__dataclass_fields__:
        value = row[field_name]
        if field_name in string_fields:
            values[field_name] = value
        elif field_name in integer_fields:
            values[field_name] = int(value)
        else:
            values[field_name] = float(value)
    return RetrievedCandidate(**values)
