from __future__ import annotations

from pathlib import Path

import pytest

from business_entity_resolution.data import (
    DataValidationError,
    iter_source_chunks,
    load_source_tsv,
)


def write_tsv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_chunked_source_loading_preserves_empty_cells(tmp_path: Path) -> None:
    path = write_tsv(
        tmp_path / "source.tsv",
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-1\tNA\t\tUS\n"
        "S2-2\tExample\tSomewhere\tIndia\n",
    )

    chunks = list(iter_source_chunks(path, "S2", chunksize=1))

    assert len(chunks) == 2
    assert chunks[0].loc[0, "business_name"] == "NA"
    assert chunks[0].loc[0, "business_address"] == ""


def test_source_loader_rejects_wrong_schema(tmp_path: Path) -> None:
    path = write_tsv(
        tmp_path / "source.tsv",
        "entity_id,business_name,business_address,country\n"
        "S1-1,Example,Somewhere,US\n",
    )

    with pytest.raises(DataValidationError, match="Unexpected TSV schema"):
        load_source_tsv(path, "S1")


def test_source_loader_rejects_wrong_prefix(tmp_path: Path) -> None:
    path = write_tsv(
        tmp_path / "source.tsv",
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S3-1\tExample\tSomewhere\tUS\n",
    )

    with pytest.raises(DataValidationError, match="required 'S2-' prefix"):
        load_source_tsv(path, "S2")
