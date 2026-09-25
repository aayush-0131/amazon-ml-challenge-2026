from __future__ import annotations

from pathlib import Path

import pytest

from business_entity_resolution.data import DataValidationError
from business_entity_resolution.ground_truth import (
    explode_positive_links,
    load_ground_truth,
    parse_matched_entity_ids,
)


def test_ground_truth_preserves_singletons_and_explodes_only_links(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ground_truth.tsv"
    path.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\t\n"
        "S1-2\tS2-9,S3-8\n",
        encoding="utf-8",
    )

    truth = load_ground_truth(path)
    links = explode_positive_links(truth)

    assert truth["S1-1"] == frozenset()
    assert truth["S1-2"] == frozenset({"S2-9", "S3-8"})
    assert set(links.itertuples(index=False, name=None)) == {
        ("S1-2", "S2-9"),
        ("S1-2", "S3-8"),
    }


def test_parse_ground_truth_rejects_invalid_and_duplicate_targets() -> None:
    with pytest.raises(DataValidationError, match="S2-/S3-"):
        parse_matched_entity_ids("S1-2")
    with pytest.raises(DataValidationError, match="Duplicate target"):
        parse_matched_entity_ids("S2-2,S2-2")


def test_ground_truth_rejects_duplicate_source1_rows(tmp_path: Path) -> None:
    path = tmp_path / "ground_truth.tsv"
    path.write_text(
        "source1_entity_id\tmatched_entity_ids\nS1-1\t\nS1-1\tS2-1\n",
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match="Duplicate source1_entity_id"):
        load_ground_truth(path)


def test_ground_truth_rejects_row_without_required_tab(tmp_path: Path) -> None:
    path = tmp_path / "ground_truth.tsv"
    path.write_text(
        "source1_entity_id\tmatched_entity_ids\nS1-1\n",
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match="exactly two tab-separated"):
        load_ground_truth(path)
