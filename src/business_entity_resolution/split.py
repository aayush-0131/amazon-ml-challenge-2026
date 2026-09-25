"""Deterministic Source-1 entity-level train/validation splitting."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass

import pandas as pd

DEFAULT_SEED = 2026


@dataclass(frozen=True)
class EntitySplit:
    train_ids: frozenset[str]
    validation_ids: frozenset[str]


def _stable_key(entity_id: str, seed: int) -> bytes:
    payload = f"{seed}\0{entity_id}".encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).digest()


def stratified_entity_split(
    source1_records: pd.DataFrame | Iterable[Mapping[str, str]],
    ground_truth: Mapping[str, Collection[str]],
    *,
    validation_fraction: float = 0.2,
    seed: int = DEFAULT_SEED,
) -> EntitySplit:
    """Split Source-1 IDs, stratified by country and singleton/matched status.

    This deliberately operates at Source-1 entity grain. Candidate pairs must be
    generated independently inside each resulting partition; they must never be
    randomly split as rows.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")

    if isinstance(source1_records, pd.DataFrame):
        missing = {"entity_id", "country"} - set(source1_records.columns)
        if missing:
            raise ValueError(f"Source-1 records are missing columns: {sorted(missing)}")
        records = (
            {"entity_id": entity_id, "country": country}
            for entity_id, country in source1_records.loc[
                :, ["entity_id", "country"]
            ].itertuples(index=False, name=None)
        )
    else:
        records = source1_records

    strata: dict[tuple[str, bool], list[str]] = defaultdict(list)
    seen: set[str] = set()
    for record in records:
        entity_id = str(record["entity_id"])
        country = str(record["country"])
        if not entity_id.startswith("S1-"):
            raise ValueError(f"Invalid Source-1 entity ID: {entity_id!r}")
        if entity_id in seen:
            raise ValueError(f"Duplicate Source-1 entity ID: {entity_id}")
        if entity_id not in ground_truth:
            raise ValueError(f"Ground truth is missing Source-1 entity: {entity_id}")
        seen.add(entity_id)
        strata[(country, bool(ground_truth[entity_id]))].append(entity_id)

    extra_truth = set(ground_truth) - seen
    if extra_truth:
        examples = sorted(extra_truth)[:5]
        raise ValueError(f"Ground truth contains entities absent from Source-1: {examples}")

    validation: set[str] = set()
    for entity_ids in strata.values():
        ordered = sorted(
            entity_ids, key=lambda value: (_stable_key(value, seed), value)
        )
        validation_size = round(len(ordered) * validation_fraction)
        if len(ordered) >= 2:
            validation_size = min(max(validation_size, 1), len(ordered) - 1)
        validation.update(ordered[:validation_size])

    return EntitySplit(
        train_ids=frozenset(seen - validation),
        validation_ids=frozenset(validation),
    )
