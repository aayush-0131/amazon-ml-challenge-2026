from __future__ import annotations

import csv
from pathlib import Path

import pytest

from business_entity_resolution.exp005_quick import (
    POLICIES, SourceRecord, build_index, candidate_evidence, decide, open_indexes,
    run_inference, validate_output,
)
from contextlib import ExitStack


HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"


def write_rows(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("entity_id", "business_name", "business_address", "country"))
        writer.writerows(rows)


@pytest.fixture
def fixture(tmp_path):
    data = tmp_path / "test"
    data.mkdir()
    write_rows(data / "test_source1.tsv", [
        ("S1-z", "Café Alpha", "12 Rue Paris", "France"),
        ("S1-a", "Singleton", "Unknown", "France"),
        ("S1-b", "Café Alpha", "12 Rue Paris", "France"),
        ("S1-c", "Café Alpha", "99 Broadway", "France"),
        ("S1-d", "Café Alpha", "12 Rue Paris", "Atlantis"),
        ("S1-e", "", "", "France"),
    ])
    write_rows(data / "test_source2.tsv", [
        ("S2-9", "Cafe Alpha", "12 Rue Paris", "France"),
        ("S2-2", "CAFÉ ALPHA", "12 Rue Paris", "France"),
        ("S2-3", "Café Alpha", "12 Rue Lyon", "France"),
        ("S2-4", "Other Name", "12 Rue Paris", "France"),
        ("S2-5", "Café Alpha", "No Shared Tokens", "France"),
        ("S2-6", "Café Alpha", "12 Rue Paris", "Atlantis"),
    ])
    write_rows(data / "test_source3.tsv", [
        ("S3-1", "Café Alpha", "12 Rue Paris", "France"),
        ("S3-2", "Café Alpha", "12 Rue Paris", "Atlantis"),
    ])
    indexes = tmp_path / "indexes"
    for source in ("S2", "S3"):
        build_index(data / f"test_source{source[-1]}.tsv", source, indexes, batch_size=2)
    return data, indexes


def rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_exact_name_address_conflicts_duplicates_and_open_country(fixture):
    data, index_dir = fixture
    with ExitStack() as stack:
        indexes = open_indexes(index_dir, data, "test", stack)
        evidence = candidate_evidence(SourceRecord("S1-z", "Café Alpha", "12 Rue Paris", "France"), indexes)
        ids = [item[0] for item in evidence]
        assert ids == ["S2-2", "S2-3", "S2-4", "S2-5", "S2-9", "S3-1"]
        assert decide(evidence, "both_exact") == ["S2-2", "S3-1"]
        assert decide(evidence, "name_address_half") == ["S2-2", "S2-3", "S3-1"]
        assert decide(evidence, "address_name_half") == ["S2-2", "S3-1"]
        assert decide(evidence, "either_half") == ["S2-2", "S2-3", "S3-1"]
        for policy in POLICIES:
            assert "S2-5" not in decide(evidence, policy)  # exact name, conflicting address
            assert "S2-4" not in decide(evidence, policy)  # exact address, conflicting name
        other = candidate_evidence(SourceRecord("S1-d", "Café Alpha", "12 Rue Paris", "Atlantis"), indexes)
        assert [item[0] for item in other] == ["S2-6", "S3-2"]
        assert decide(other, "both_exact") == ["S2-6", "S3-2"]


def test_streamed_output_singletons_order_invariant_and_determinism(fixture, tmp_path):
    data, indexes = fixture
    first, second = tmp_path / "first", tmp_path / "second"
    result = run_inference(data / "test_source1.tsv", data, indexes, first,
                           split="test", policy="both_exact", smoke_limit=6)
    run_inference(data / "test_source1.tsv", data, indexes, second,
                  split="test", policy="both_exact", smoke_limit=6)
    assert result["rows"] == 6
    assert validate_output(data / "test_source1.tsv", first, limit=6) == 6
    matching = rows(first / "matching_results.tsv")
    candidates = rows(first / "candidate_pairs.tsv")
    assert [row["source1_entity_id"] for row in matching] == ["S1-z", "S1-a", "S1-b", "S1-c", "S1-d", "S1-e"]
    assert matching[0]["matched_entity_ids"] == "S2-2,S3-1"
    assert candidates[0]["candidate_entity_ids"] == "S2-2,S2-3,S2-4,S2-5,S2-9,S3-1"
    assert matching[1]["matched_entity_ids"] == candidates[1]["candidate_entity_ids"] == ""
    assert matching[5]["matched_entity_ids"] == candidates[5]["candidate_entity_ids"] == ""
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_full_test_guard_and_duplicate_source1(fixture, tmp_path):
    data, indexes = fixture
    with pytest.raises(ValueError, match="allow-full-test"):
        run_inference(data / "test_source1.tsv", data, indexes, tmp_path / "guard",
                      split="test", policy="both_exact")
    with pytest.raises(ValueError, match="1..10000"):
        run_inference(data / "test_source1.tsv", data, indexes, tmp_path / "huge",
                      split="test", policy="both_exact", smoke_limit=10001)
    duplicate = data / "duplicate_source1.tsv"
    duplicate.write_text(HEADER + "S1-x\tA\tB\tFrance\nS1-x\tA\tB\tFrance\n")
    with pytest.raises(Exception, match="UNIQUE constraint"):
        run_inference(duplicate, data, indexes, tmp_path / "dupe",
                      split="test", policy="both_exact", smoke_limit=2)


def test_exact_address_with_partial_name_and_invalid_output(fixture, tmp_path):
    data, indexes = fixture
    with ExitStack() as stack:
        opened = open_indexes(indexes, data, "test", stack)
        evidence = candidate_evidence(SourceRecord("S1-x", "Alpha", "12 Rue Paris", "France"), opened)
        assert decide(evidence, "address_name_half") == ["S2-2", "S2-9", "S3-1"]
    output = tmp_path / "valid"
    run_inference(data / "test_source1.tsv", data, indexes, output,
                  split="test", policy="both_exact", smoke_limit=1)
    path = output / "matching_results.tsv"
    path.write_text(path.read_text().replace("S2-2,S3-1", "S2-impossible"))
    with pytest.raises(ValueError, match="absent from evaluated candidate"):
        validate_output(data / "test_source1.tsv", output, limit=1)
