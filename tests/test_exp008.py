"""Small synthetic blocker tests; no TRAIN or TEST files are required."""
from __future__ import annotations

import inspect

import pytest

from business_entity_resolution import exp008
from business_entity_resolution.exp008 import (Policy, SecondaryIndex, _summary, compose_candidates,
    build_index, digit_letter, index_path, keys, transliterated)
from business_entity_resolution.sampling import SourceRecord


def _source(path, rows):
    with path.open("w") as handle:
        handle.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        for row in rows:
            handle.write("\t".join(row) + "\n")


def test_secondary_representations_are_additional():
    p = Policy(min_name_chars=3)
    assert transliterated("Société Électricité") == "societe electricite"
    assert digit_letter("Bâtiment12 Rue") == "bâtiment 12 rue"
    assert digit_letter("ABC123") == digit_letter("ABC 123")
    assert keys("Acme GmbH", "", p)["core_name"] == "acme"
    assert keys("Acme123", "Rue12", p)["digit_name"] == "acme 123"
    assert "digit_address" not in keys("", "Rue12", p)  # short address guard


def test_country_agnostic_deterministic_bounded_retrieval(tmp_path):
    p = Policy(max_df=2, max_added_per_source=1, min_name_chars=3, min_address_chars=5)
    source = tmp_path / "train_source2.tsv"
    _source(source, [
        ("S2-2", "Acme LLC", "Rue12", "ZZ"),
        ("S2-1", "Acme GmbH", "Rue 12", "ZZ"),
        ("S2-3", "Acme Ltd", "Rue12", "OTHER"),
    ])
    build = build_index(source, "S2", tmp_path / "exp008_secondary_index", p, chunksize=1)
    assert build["rows"] == 3
    assert build_index(source, "S2", tmp_path / "exp008_secondary_index", p)["reused"]
    record = SourceRecord("S1-x", "Acme SA", "Rue12", "ZZ")
    with SecondaryIndex(index_path(tmp_path / "exp008_secondary_index", "S2"), "S2", source, p) as index:
        first = index.retrieve(record, ("core_name", "digit_address"))
        assert first
        assert first == index.retrieve(record, ("core_name", "digit_address"))
        assert len(first) <= p.max_added_per_source
        assert {r["candidate_entity_id"] for r in first} <= {"S2-1", "S2-2"}
        assert index.retrieve(record, ("core_name",), baseline_ids={"S2-1", "S2-2"}) == []
        assert index.retrieve(SourceRecord("S1-y", "Acme SA", "Rue12", "UNSEEN"),
                              ("core_name", "digit_address")) == []
    with pytest.raises(ValueError):
        SecondaryIndex(index_path(tmp_path / "exp008_secondary_index", "S2"), "S2", source,
                       Policy(max_df=3, max_added_per_source=1, min_name_chars=3, min_address_chars=5))


def test_popular_keys_are_rejected_without_truncation(tmp_path):
    p = Policy(max_df=1, max_added_per_source=10, min_name_chars=3)
    source = tmp_path / "train_source3.tsv"
    _source(source, [("S3-1", "Acme LLC", "", "X"), ("S3-2", "Acme Ltd", "", "X")])
    build_index(source, "S3", tmp_path, p, chunksize=1)
    with SecondaryIndex(index_path(tmp_path, "S3"), "S3", source, p) as index:
        assert index.retrieve(SourceRecord("S1-1", "Acme SA", "", "X"), ("core_name",)) == []


def test_joined_digit_finds_spaced_target_and_reverse(tmp_path):
    p = Policy(min_name_chars=3, min_address_chars=5)
    source = tmp_path / "train_source2.tsv"
    _source(source, [("S2-1", "ABC 123", "Rue 12", "ANY"),
                     ("S2-2", "XYZ789", "Road77", "ANY")])
    build_index(source, "S2", tmp_path, p)
    with SecondaryIndex(index_path(tmp_path, "S2"), "S2", source, p) as index:
        joined = index.retrieve(SourceRecord("S1-1", "ABC123", "Rue12", "ANY"),
                                ("digit_name", "digit_address"))
        spaced = index.retrieve(SourceRecord("S1-2", "XYZ 789", "Road 77", "ANY"),
                                ("digit_name", "digit_address"))
    assert {r["candidate_entity_id"] for r in joined} == {"S2-1"}
    assert {r["candidate_entity_id"] for r in spaced} == {"S2-2"}


def test_oracle_intersection_and_no_truth_injection():
    assert "truth" not in inspect.signature(SecondaryIndex.retrieve).parameters
    truth = {"S1-1": {"S2-1", "S3-1"}, "S1-2": set()}
    records = {eid: SourceRecord(eid, "Acme", "", "NEW") for eid in truth}
    result = _summary(truth, {"S1-1": {"S2-1", "S2-x"}, "S1-2": set()}, records, .5)
    assert result["blocker_fn"] == 1
    assert result["candidate_link_recall"] == .5
    assert result["matched_with_zero_retained_count"] == 0
    assert result["max_candidates_per_s1"] == 2


def test_composition_retains_baseline_and_caps_each_source():
    p = Policy(max_added_per_source=1)
    hits = {"trans_name": {"S2-2", "S2-3", "S3-2"},
            "core_name": {"S2-1", "S3-1"}}
    expected = {"S2-old", "S2-2", "S3-2"}
    assert compose_candidates({"S2-old"}, hits, ("trans_name", "core_name"), p) == expected
    assert compose_candidates({"S2-old"}, hits, ("core_name",), p) == {"S2-old", "S2-1", "S3-1"}
    assert compose_candidates({"S2-old"}, hits, ("trans_name", "core_name"), p) == expected


def test_train_only_builder(tmp_path):
    p = Policy()
    path = tmp_path / "test_source2.tsv"
    _source(path, [("S2-1", "Acme", "", "X")])
    with pytest.raises(ValueError, match="TRAIN only"):
        build_index(path, "S2", tmp_path, p)


def test_interrupted_index_build_resumes_at_committed_chunk(tmp_path, monkeypatch):
    p = Policy(min_name_chars=3)
    source = tmp_path / "train_source2.tsv"
    _source(source, [("S2-1", "Acme LLC", "", "X"),
                     ("S2-2", "Bravo GmbH", "", "X")])
    reader = exp008.iter_source_chunks
    def interrupted(*args, **kwargs):
        yield next(reader(*args, **kwargs))
        raise RuntimeError("simulated interruption")
    monkeypatch.setattr(exp008, "iter_source_chunks", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        build_index(source, "S2", tmp_path, p, chunksize=1)
    assert not index_path(tmp_path, "S2").exists()
    monkeypatch.setattr(exp008, "iter_source_chunks", reader)
    result = build_index(source, "S2", tmp_path, p, chunksize=1)
    assert result["rows"] == 2
    with SecondaryIndex(index_path(tmp_path, "S2"), "S2", source, p) as index:
        assert [r["candidate_entity_id"] for r in index.retrieve(
            SourceRecord("S1-1", "Bravo SA", "", "X"), ("core_name",))] == ["S2-2"]
