"""Deterministic, entity-level sampling helpers for scalable experiments."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .data import DEFAULT_CHUNK_SIZE, iter_source_chunks
from .ground_truth import iter_ground_truth
from .split import stable_entity_key


@dataclass(frozen=True)
class SourceRecord:
    entity_id: str
    business_name: str
    business_address: str
    country: str


def _allocate_stratified_counts(
    group_sizes: dict[tuple[str, int], int], sample_size: int
) -> dict[tuple[str, int], int]:
    total = sum(group_sizes.values())
    if not 0 < sample_size <= total:
        raise ValueError(f"sample_size must be in [1, {total:,}]")
    exact = {
        key: sample_size * size / total for key, size in group_sizes.items()
    }
    allocation = {key: math.floor(value) for key, value in exact.items()}
    remaining = sample_size - sum(allocation.values())
    ranked = sorted(
        group_sizes,
        key=lambda key: (-(exact[key] - allocation[key]), key),
    )
    for key in ranked[:remaining]:
        allocation[key] += 1
    return allocation


def select_development_entity_ids(
    source1_path: str | Path,
    ground_truth_path: str | Path,
    *,
    sample_size: int,
    seed: int,
    chunksize: int = DEFAULT_CHUNK_SIZE,
) -> frozenset[str]:
    """Select exact-size S1 sample stratified by country and match count."""

    match_counts = {
        source1_id: len(matches)
        for source1_id, matches in iter_ground_truth(ground_truth_path)
    }
    groups: dict[tuple[str, int], list[str]] = defaultdict(list)
    for chunk in iter_source_chunks(source1_path, "S1", chunksize=chunksize):
        for entity_id, country in chunk.loc[:, ["entity_id", "country"]].itertuples(
            index=False, name=None
        ):
            try:
                match_count = match_counts.pop(entity_id)
            except KeyError as exc:
                raise ValueError(f"Ground truth is missing {entity_id}") from exc
            groups[(country, match_count)].append(entity_id)
    if match_counts:
        raise ValueError(
            f"Ground truth has {len(match_counts):,} IDs absent from Source-1"
        )

    allocation = _allocate_stratified_counts(
        {key: len(values) for key, values in groups.items()}, sample_size
    )
    selected: set[str] = set()
    for key, entity_ids in groups.items():
        ordered = sorted(
            entity_ids,
            key=lambda entity_id: (stable_entity_key(entity_id, seed), entity_id),
        )
        selected.update(ordered[: allocation[key]])
    if len(selected) != sample_size:
        raise AssertionError("stratified allocation did not produce exact sample size")
    return frozenset(selected)


def load_selected_source_records(
    path: str | Path,
    source: str,
    selected_ids: frozenset[str] | set[str],
    *,
    chunksize: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, SourceRecord]:
    selected = set(selected_ids)
    records: dict[str, SourceRecord] = {}
    for chunk in iter_source_chunks(path, source, chunksize=chunksize):
        subset = chunk.loc[chunk["entity_id"].isin(selected)]
        for row in subset.itertuples(index=False):
            records[row.entity_id] = SourceRecord(
                entity_id=row.entity_id,
                business_name=row.business_name,
                business_address=row.business_address,
                country=row.country,
            )
    missing = selected - records.keys()
    if missing:
        raise ValueError(f"Selected IDs missing from {path}: {sorted(missing)[:5]}")
    return records


def load_selected_ground_truth(
    path: str | Path, selected_ids: Iterable[str]
) -> dict[str, frozenset[str]]:
    selected = set(selected_ids)
    truth = {
        source1_id: matches
        for source1_id, matches in iter_ground_truth(path)
        if source1_id in selected
    }
    missing = selected - truth.keys()
    if missing:
        raise ValueError(f"Selected IDs missing from ground truth: {sorted(missing)[:5]}")
    return truth
