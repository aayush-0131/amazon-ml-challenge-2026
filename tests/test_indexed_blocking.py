from pathlib import Path

from business_entity_resolution.indexed_blocking import (
    SourceIndexConfig,
    SourceSideIndex,
    build_or_open_source_index,
)
from business_entity_resolution.multipass import CandidateBudget, select_candidates_for_budget
from business_entity_resolution.sampling import SourceRecord


def _write_source(path: Path) -> None:
    path.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-1\tAcme Trading\t12 Main Street 94105\tUS\n"
        "S2-2\tAcme Trading\t12 Main Street 94105\tCA\n"
        "S2-3\tDifferent Name\t12 Main Street 94105\tUS\n",
        encoding="utf-8",
    )


def _budget() -> CandidateBudget:
    return CandidateBudget(
        name="test",
        total_per_source=20,
        exact_name=20,
        exact_address=20,
        name_signature=20,
        name_token=20,
        address_token=20,
        numeric_address=20,
        char_name=0,
    )


def test_source_side_index_is_reused_and_country_compatible(tmp_path: Path) -> None:
    source_path = tmp_path / "train_source2.tsv"
    _write_source(source_path)
    index_dir = tmp_path / "index"
    first = build_or_open_source_index(source_path, "S2", index_dir, chunksize=2)
    second = build_or_open_source_index(source_path, "S2", index_dir, chunksize=2)
    assert not first.reused
    assert second.reused
    assert first.row_count == 3

    with SourceSideIndex(first.path, "S2", SourceIndexConfig()) as index:
        candidates = select_candidates_for_budget(
            index.retrieve(
                SourceRecord("S1-1", "Acme Trading", "12 Main Street 94105", "US")
            ),
            _budget(),
        )
    identifiers = {candidate.candidate_entity_id for candidate in candidates}
    assert identifiers == {"S2-1", "S2-3"}
    assert all(candidate.country == "US" for candidate in candidates)
    assert all(candidate.char_name_score == 0.0 for candidate in candidates)
    assert any(candidate.exact_name_score > 0 for candidate in candidates)
    assert any(candidate.exact_address_score > 0 for candidate in candidates)


def test_indexed_retrieval_is_deterministic(tmp_path: Path) -> None:
    source_path = tmp_path / "train_source2.tsv"
    _write_source(source_path)
    result = build_or_open_source_index(source_path, "S2", tmp_path / "index")
    query = SourceRecord("S1-1", "Acme Trading", "12 Main Street 94105", "US")
    with SourceSideIndex(result.path, "S2", SourceIndexConfig()) as index:
        first = [candidate.candidate_entity_id for candidate in index.retrieve(query)]
        second = [candidate.candidate_entity_id for candidate in index.retrieve(query)]
    assert first == second
