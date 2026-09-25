"""Safe, memory-conscious readers for the competition TSV files."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import pandas as pd

SOURCE_COLUMNS = (
    "entity_id",
    "business_name",
    "business_address",
    "country",
)
GROUND_TRUTH_COLUMNS = ("source1_entity_id", "matched_entity_ids")
SOURCE_PREFIXES = frozenset({"S1", "S2", "S3"})
DEFAULT_CHUNK_SIZE = 200_000


class DataValidationError(ValueError):
    """Raised when a competition data file violates its declared contract."""


def validate_schema(
    columns: Sequence[str],
    expected_columns: Sequence[str],
    *,
    path: str | Path | None = None,
) -> None:
    """Require the exact expected columns in the exact expected order."""

    actual = tuple(columns)
    expected = tuple(expected_columns)
    if actual != expected:
        location = f" in {path}" if path is not None else ""
        raise DataValidationError(
            f"Unexpected TSV schema{location}: got {actual!r}; expected {expected!r}. "
            "Confirm that the file was read with a tab separator."
        )


def validate_entity_id_prefix(
    entity_ids: pd.Series,
    expected_prefix: str,
    *,
    path: str | Path | None = None,
    max_examples: int = 5,
) -> None:
    """Validate that every non-empty entity ID starts with ``<prefix>-``."""

    if expected_prefix not in SOURCE_PREFIXES:
        raise ValueError(f"Unknown source prefix: {expected_prefix!r}")

    expected = f"{expected_prefix}-"
    ids = entity_ids.astype("string").fillna("")
    invalid = ids.eq("") | ~ids.str.startswith(expected)
    if invalid.any():
        examples = ids.loc[invalid].head(max_examples).tolist()
        location = f" in {path}" if path is not None else ""
        raise DataValidationError(
            f"Found {int(invalid.sum()):,} entity ID(s){location} without the "
            f"required {expected!r} prefix; examples: {examples!r}"
        )


def _read_csv_options() -> dict[str, object]:
    # Disabling pandas' NA token inference is essential: a genuinely empty field
    # remains "", while names such as "NA" are not silently converted to missing.
    return {
        "sep": "\t",
        "encoding": "utf-8",
        "dtype": "string",
        "keep_default_na": False,
        "na_filter": False,
        "on_bad_lines": "error",
    }


def iter_tsv_chunks(
    path: str | Path,
    *,
    expected_columns: Sequence[str],
    expected_prefix: str | None = None,
    id_column: str = "entity_id",
    chunksize: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[pd.DataFrame]:
    """Yield validated TSV chunks without loading the full file into memory."""

    file_path = Path(path)
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    if not file_path.is_file():
        raise FileNotFoundError(f"TSV file not found: {file_path}")

    reader = pd.read_csv(file_path, chunksize=chunksize, **_read_csv_options())
    saw_chunk = False
    for chunk in reader:
        saw_chunk = True
        validate_schema(chunk.columns, expected_columns, path=file_path)
        if expected_prefix is not None:
            if id_column not in chunk.columns:
                raise DataValidationError(
                    f"ID column {id_column!r} is absent from {file_path}"
                )
            validate_entity_id_prefix(
                chunk[id_column], expected_prefix, path=file_path
            )
        yield chunk

    # pandas yields one empty frame for a header-only file today, but keeping this
    # guard makes the behavior explicit if that implementation detail changes.
    if not saw_chunk:
        columns = pd.read_csv(file_path, nrows=0, **_read_csv_options()).columns
        validate_schema(columns, expected_columns, path=file_path)


def load_tsv(
    path: str | Path,
    *,
    expected_columns: Sequence[str],
    expected_prefix: str | None = None,
    id_column: str = "entity_id",
) -> pd.DataFrame:
    """Load and validate a TSV when the caller knows it fits in memory."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"TSV file not found: {file_path}")
    frame = pd.read_csv(file_path, **_read_csv_options())
    validate_schema(frame.columns, expected_columns, path=file_path)
    if expected_prefix is not None:
        validate_entity_id_prefix(
            frame[id_column], expected_prefix, path=file_path
        )
    return frame


def iter_source_chunks(
    path: str | Path,
    source: str,
    *,
    chunksize: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[pd.DataFrame]:
    """Yield validated chunks from one Source-1/2/3 file."""

    yield from iter_tsv_chunks(
        path,
        expected_columns=SOURCE_COLUMNS,
        expected_prefix=source,
        chunksize=chunksize,
    )


def load_source_tsv(path: str | Path, source: str) -> pd.DataFrame:
    """Load a complete Source-1/2/3 TSV and validate its schema and IDs."""

    return load_tsv(
        path,
        expected_columns=SOURCE_COLUMNS,
        expected_prefix=source,
    )
