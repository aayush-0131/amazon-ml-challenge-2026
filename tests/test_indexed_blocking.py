from pathlib import Path
from dataclasses import asdict, replace
import json
import sqlite3
import subprocess
import sys

import pytest

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


def test_country_scoped_df_and_rare_postings(tmp_path: Path) -> None:
    path = tmp_path / "source.tsv"
    path.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-1\tRare Rare Trading\t\tFrance\n"
        "S2-2\tRare Holding\t\tElsewhere\n"
        "S2-3\tRare Services\t\tElsewhere\n", encoding="utf-8",
    )
    result = build_or_open_source_index(path, "S2", tmp_path / "index")
    with SourceSideIndex(result.path, "S2", replace(SourceIndexConfig(), max_name_token_df=1)) as index:
        assert index._df("name", "rare", "France") == 1  # distinct documents
        assert index._df("name", "rare", "Elsewhere") == 2
        assert index._df("name", "rare", "france") == 0  # raw equality
        assert index._posting_ids("France", "name", "rare") == {1}
        candidates = index.retrieve(SourceRecord("S1-1", "Rare Company", "", "France"))
        assert [c.candidate_entity_id for c in candidates] == ["S2-1"]
        assert candidates[0].name_token_score > 0


def test_common_token_intersection_and_overflow(tmp_path: Path) -> None:
    path = tmp_path / "source.tsv"
    path.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-1\tAlpha Beta\t\tFrance\n"
        "S2-2\tAlpha Gamma\t\tFrance\n"
        "S2-3\tBeta Delta\t\tFrance\n", encoding="utf-8",
    )
    result = build_or_open_source_index(path, "S2", tmp_path / "index")
    config = replace(SourceIndexConfig(), max_name_token_df=1)
    with SourceSideIndex(result.path, "S2", config) as index:
        # No exact hit and neither token qualifies for individual expansion.
        candidates = index.retrieve(SourceRecord("S1-1", "Alpha Beta Unknown", "", "France"))
        assert [c.candidate_entity_id for c in candidates] == ["S2-1"]
        assert index.last_diagnostics["intersection_queries"] == 1
        assert candidates[0].name_token_rank == 1
        trace = []
        index.connection.set_trace_callback(trace.append)
        index.retrieve(SourceRecord("S1-1", "Alpha Beta Unknown", "", "France"))
        assert not any("bm25" in statement.lower() or " match " in statement.lower() for statement in trace)


def test_config_is_explicit_and_validated() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/exp002c_postings.json").read_text())
    actual = SourceIndexConfig.from_config(config)
    assert (actual.max_name_token_df, actual.max_address_token_df, actual.max_numeric_token_df) == (60, 40, 250)
    assert asdict(actual) == config["source_index"]
    with pytest.raises(ValueError, match="Explicit source_index"):
        SourceIndexConfig.from_config({})
    config["source_index"]["typo"] = 1
    with pytest.raises(ValueError, match="keys mismatch"):
        SourceIndexConfig.from_config(config)
    with pytest.raises(ValueError, match="positive integer"):
        replace(actual, max_name_token_df=0)


def test_cross_field_intersection_numeric_evidence_and_overflow(tmp_path: Path) -> None:
    path = tmp_path / "source.tsv"
    path.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-1\tAcme North\t12 Road\tFrance\n"
        "S2-2\tAcme South\t12 Road\tFrance\n"
        "S2-3\tAcme West\t90 Road\tFrance\n"
        "S2-4\tOther West\t12 Road\tFrance\n")
    result = build_or_open_source_index(path, "S2", tmp_path / "index")
    config = replace(SourceIndexConfig(), max_name_token_df=1, max_address_token_df=1,
                     max_numeric_token_df=1, max_intersection_hits=2)
    query = SourceRecord("S1-1", "Acme", "12", "France")
    with SourceSideIndex(result.path, "S2", config) as index:
        candidates = index.retrieve(query)
        assert {c.candidate_entity_id for c in candidates} == {"S2-1", "S2-2"}
        assert all(c.name_token_score > 0 and c.numeric_address_score > 0 for c in candidates)
    with SourceSideIndex(result.path, "S2", replace(config, max_intersection_hits=1)) as index:
        assert index.retrieve(query) == []  # Overflow rejects whole intersection.
        assert index.last_diagnostics["intersection_overflows"] == 1


def test_changed_source_reuse_rejected(tmp_path: Path) -> None:
    path = tmp_path / "source.tsv"
    _write_source(path)
    build_or_open_source_index(path, "S2", tmp_path / "index")
    path.write_text(path.read_text() + "S2-4\tnew\t\tFrance\n")
    with pytest.raises(ValueError, match="fingerprint changed"):
        build_or_open_source_index(path, "S2", tmp_path / "index")


def test_old_schema_rejected(tmp_path: Path) -> None:
    path = tmp_path / "v1.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
        connection.executemany("INSERT INTO metadata VALUES (?, ?)", [
            ("status", "complete"), ("source", "S2"), ("schema_version", "1"),
        ])
    with pytest.raises(ValueError, match="Unsupported index schema"):
        SourceSideIndex(path, "S2", SourceIndexConfig())


def test_reuse_does_not_scan_source_and_failed_rebuild_preserves_index(tmp_path: Path, monkeypatch) -> None:
    import business_entity_resolution.indexed_blocking as module
    path = tmp_path / "source.tsv"
    _write_source(path)
    result = build_or_open_source_index(path, "S2", tmp_path / "index")
    before = result.path.read_bytes()

    def fail(*args, **kwargs):
        raise RuntimeError("injected build failure")

    monkeypatch.setattr(module, "iter_source_chunks", fail)
    assert build_or_open_source_index(path, "S2", tmp_path / "index").reused
    with pytest.raises(RuntimeError, match="injected"):
        build_or_open_source_index(path, "S2", tmp_path / "index", rebuild=True)
    assert result.path.read_bytes() == before
    assert result.path.with_suffix(".building.sqlite").exists()
    with pytest.raises(FileExistsError):
        build_or_open_source_index(path, "S2", tmp_path / "index", rebuild=True)


def test_candidate_cap_and_exact_overflow(tmp_path: Path) -> None:
    path = tmp_path / "source.tsv"
    path.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n" +
                    "".join(f"S2-{i}\tAcme Trading\t\tFrance\n" for i in range(6)), encoding="utf-8")
    result = build_or_open_source_index(path, "S2", tmp_path / "index")
    with SourceSideIndex(result.path, "S2", replace(SourceIndexConfig(), max_exact_hits=2)) as index:
        pool = index.retrieve(SourceRecord("S1-1", "Acme Trading", "", "France"))
        selected = select_candidates_for_budget(pool, replace(_budget(), total_per_source=2))
        assert len(pool) == 6  # Rare posting expansion is complete, not top-N.
        assert len(selected) == 2
        assert index.last_diagnostics["candidates_before_cap"] == 6
        assert index.last_diagnostics["exact_overflow_passes"] == 2
        assert [c.candidate_entity_id for c in selected] == ["S2-0", "S2-1"]


def test_benchmark_cli_smoke_and_reuse(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "data" / "train"
    data.mkdir(parents=True)
    _write_source(data / "train_source2.tsv")
    (data / "train_source3.tsv").write_text((data / "train_source2.tsv").read_text().replace("S2-", "S3-"))
    (data / "train_source1.tsv").write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-1\tAcme Trading\t12 Main Street 94105\tUS\n"
        "S1-2\tNo Such Entity\t\tUS\n")
    (data / "train_ground_truth.tsv").write_text(
        "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S3-1\nS1-2\t\n")
    ids = tmp_path / "ids.csv"
    ids.write_text("source1_entity_id\nS1-1\nS1-2\n")
    config = json.loads((root / "configs/exp002c_postings.json").read_text())
    config["subset_ids_path"] = str(ids)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    common = [sys.executable, str(root / "scripts/benchmark_indexed_blocker.py"),
              "--data-root", str(data.parent), "--index-dir", str(tmp_path / "index"),
              "--config", str(config_path), "--sample-sizes", "1", "2"]
    subprocess.run(common + ["--build-index", "--build-only", "--results-dir", str(tmp_path / "build")], check=True, capture_output=True)
    subprocess.run(common + ["--results-dir", str(tmp_path / "query")], check=True, capture_output=True)
    import pandas as pd
    tables = tmp_path / "query" / "tables"
    metrics = pd.read_csv(tables / "exp002c_blocker_benchmark.csv")
    overall = metrics.query('sample == "s1_2" and dimension == "overall"').iloc[0]
    assert overall.positive_link_recall == 1
    assert overall.zero_candidate_entities == 1
    assert overall.average_query_seconds_per_s1 > 0
    assert overall.average_exact_candidates_per_s1 == 2
    assert pd.read_csv(tables / "exp002c_index_build.csv").reused.all()
    assert len(pd.read_csv(tables / "exp002c_query_diagnostics.csv")) == 6
