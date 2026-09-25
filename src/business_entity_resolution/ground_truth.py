"""Ground-truth parsing utilities with explicit singleton preservation."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

import pandas as pd

from .data import DataValidationError, GROUND_TRUTH_COLUMNS

GroundTruthRecord = tuple[str, frozenset[str]]


def parse_matched_entity_ids(raw: str | None) -> frozenset[str]:
    """Parse one comma-separated target list; an empty cell is an empty set."""

    if raw is None or raw.strip() == "":
        return frozenset()

    ids = [value.strip() for value in raw.split(",")]
    if any(not value for value in ids):
        raise DataValidationError(
            f"Malformed matched_entity_ids value with an empty item: {raw!r}"
        )
    invalid = [value for value in ids if not value.startswith(("S2-", "S3-"))]
    if invalid:
        raise DataValidationError(
            f"Ground-truth targets must use S2-/S3- prefixes; got {invalid[:5]!r}"
        )
    if len(ids) != len(set(ids)):
        raise DataValidationError(
            f"Duplicate target ID in matched_entity_ids: {raw!r}"
        )
    return frozenset(ids)


def iter_ground_truth(path: str | Path) -> Iterator[GroundTruthRecord]:
    """Stream ``(source1_id, matches)`` records from the ground-truth TSV."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Ground-truth TSV not found: {file_path}")

    with file_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != GROUND_TRUTH_COLUMNS:
            raise DataValidationError(
                f"Unexpected ground-truth schema in {file_path}: "
                f"got {reader.fieldnames!r}; expected {GROUND_TRUTH_COLUMNS!r}"
            )

        for line_number, row in enumerate(reader, start=2):
            if row.get("matched_entity_ids") is None or None in row:
                raise DataValidationError(
                    f"Malformed ground-truth row at {file_path}:{line_number}; "
                    "expected exactly two tab-separated fields"
                )
            source1_id = (row.get("source1_entity_id") or "").strip()
            if not source1_id.startswith("S1-"):
                raise DataValidationError(
                    f"Invalid Source-1 ID at {file_path}:{line_number}: "
                    f"{source1_id!r}"
                )
            try:
                matches = parse_matched_entity_ids(row.get("matched_entity_ids"))
            except DataValidationError as exc:
                raise DataValidationError(
                    f"{file_path}:{line_number}: {exc}"
                ) from exc
            yield source1_id, matches


def load_ground_truth(path: str | Path) -> dict[str, frozenset[str]]:
    """Load entity-level labels, retaining true singletons as empty sets."""

    labels: dict[str, frozenset[str]] = {}
    for source1_id, matches in iter_ground_truth(path):
        if source1_id in labels:
            raise DataValidationError(
                f"Duplicate source1_entity_id in ground truth: {source1_id}"
            )
        labels[source1_id] = matches
    return labels


def iter_positive_links(
    ground_truth: Mapping[str, Iterable[str]] | Iterable[GroundTruthRecord],
) -> Iterator[tuple[str, str]]:
    """Explode only positive S1-to-S2/S3 links in deterministic order."""

    records = ground_truth.items() if isinstance(ground_truth, Mapping) else ground_truth
    for source1_id, matches in records:
        for matched_entity_id in sorted(matches):
            yield source1_id, matched_entity_id


def explode_positive_links(
    ground_truth: Mapping[str, Iterable[str]] | Iterable[GroundTruthRecord],
) -> pd.DataFrame:
    """Return positive links as a two-column DataFrame; singletons add no row."""

    return pd.DataFrame.from_records(
        iter_positive_links(ground_truth),
        columns=("source1_entity_id", "matched_entity_id"),
    ).astype("string")
