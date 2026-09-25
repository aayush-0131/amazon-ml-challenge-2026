"""Synthetic scheduling and schema-2 reuse tests; no Amazon-data benchmarks."""
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import hashlib
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

from business_entity_resolution.indexed_blocking import SourceIndexConfig, SourceSideIndex, build_or_open_source_index
from business_entity_resolution.intersection_scheduler import plan_intersections, pair_kind, term_kind
from business_entity_resolution.multipass import CandidateBudget, select_candidates_for_budget
from business_entity_resolution.sampling import SourceRecord

ROOT = Path(__file__).resolve().parents[1]


def plan(terms, population=100000, **kwargs):
    values = dict(term_limit=16, probe_limit=24, ordinary_anchor_df=5000,
                  selective_anchor_df=50000, max_expected_hits=150, max_probes_per_term=6)
    values.update(kwargs)
    return plan_intersections(terms, population, **values)


def test_selectivity_order_determinism_and_pool_bound():
    terms = [(df, "name", str(df)) for df in (100, 80, 10, 20, 30, 40)]
    result = plan(terms, term_limit=4, probe_limit=2)
    assert result == plan(list(reversed(terms)), term_limit=4, probe_limit=2)
    assert len(result.terms) == 4
    assert result.considered == 6
    assert result.pairs[0] == ((10, "name", "10"), (20, "name", "20"))
    # Diversity takes a disjoint pair next, not another edge from the rarest term.
    assert result.pairs[1] == ((30, "name", "30"), (40, "name", "40"))


def test_term_quota_bound_and_category_coverage():
    terms = [(10, "name", "alpha"), (20, "name", "beta"),
             (30, "address", "street"), (40, "address", "road"),
             (50, "address", "12"), (60, "address", "١٢٣")]
    result = plan(terms, probe_limit=9, max_probes_per_term=3)
    usage = Counter(term for pair in result.pairs for term in pair)
    assert max(usage.values()) <= 3
    assert len(result.pairs) <= 9
    assert len(usage) == 6
    kinds = {pair_kind(pair) for pair in result.pairs}
    assert ("name", "name") in kinds
    assert any(a != b for a, b in kinds)
    assert term_kind((10, "address", "١٢٣")) == "numeric"
    # A very rare long name must not evict every address/numeric term.
    crowded = terms + [(i, "name", f"n{i}") for i in range(1, 10)]
    assert {term_kind(t) for t in plan(crowded, term_limit=4).terms} == {"name", "address", "numeric"}


def test_selective_larger_anchor_gate_and_skip_diagnostics():
    terms = [(6000, "name", "alpha"), (7000, "address", "road")]
    assert len(plan(terms, population=1000000).pairs) == 1  # proxy=42
    rejected = plan(terms, population=10000)
    assert not rejected.pairs and rejected.skipped_selectivity == 1
    oversized = plan([(50001, "name", "x"), (60000, "name", "y")], population=100000000)
    assert not oversized.pairs and oversized.skipped_anchor == 1
    # Small anchors preserve the old cost bound even if the independence proxy is high.
    assert plan([(100, "name", "x"), (100, "name", "y")], population=100).pairs


def test_full_configured_probe_bound():
    terms = [(i, "name", f"token{i}") for i in range(1, 100)]
    result = plan(terms)
    assert len(result.terms) == 16
    assert result.considered == 120
    assert len(result.pairs) == 24
    assert max(Counter(term for pair in result.pairs for term in pair).values()) <= 6
    assert result == plan(terms[::-1])


@pytest.mark.parametrize("variant,dfs", [
    ("intersection_only", (60, 40, 250)), ("moderate", (250, 160, 1000)),
    ("balanced", (500, 320, 2500)),
])
def test_explicit_configs_and_independent_cap(variant, dfs):
    config = json.loads((ROOT / f"configs/exp002e_{variant}.json").read_text())
    cfg = SourceIndexConfig.from_config(config)
    assert asdict(cfg) == config["source_index"]
    assert (cfg.max_name_token_df, cfg.max_address_token_df, cfg.max_numeric_token_df) == dfs
    assert (cfg.intersection_terms, cfg.max_intersections) == (16, 24)
    assert cfg.intersection_scheduler == "selective_v1"
    budget = CandidateBudget(**config["blocker_budgets"][0])
    assert budget.total_per_source == 60
    assert all(getattr(budget, p) == 40 for p in ("exact_name", "exact_address", "name_signature", "name_token", "address_token", "numeric_address"))
    assert budget.char_name == 0
    del config["source_index"]["selective_max_anchor_df"]
    with pytest.raises(ValueError, match="keys mismatch"):
        SourceIndexConfig.from_config(config)


def build_fixture(tmp_path, rows, source="S2"):
    path = tmp_path / f"{source}.tsv"
    path.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n" +
                    "".join("\t".join(row) + "\n" for row in rows))
    return build_or_open_source_index(path, source, tmp_path / "indexes", chunksize=10).path


def frozen_legacy_intersections(index, country, terms, origins):
    """Frozen EXP002c ordering/probing algorithm, independent of new planner."""
    ranked = sorted((index._df(f, t, country), f, t) for f, t in set(terms))
    ranked = [term for term in ranked if term[0] > 0][:index.config.intersection_terms]
    attempts = []
    for left, right in combinations(ranked, 2):
        if left[0] > index.config.intersection_anchor_df or len(attempts) >= index.config.max_intersections:
            break
        attempts.append((left, right))
        rows = list(index.connection.execute(
            "SELECT a.record_id FROM postings a CROSS JOIN postings b "
            "WHERE a.country=? AND a.field=? AND a.token=? AND b.country=a.country "
            "AND b.field=? AND b.token=? AND b.record_id=a.record_id ORDER BY a.record_id LIMIT ?",
            (country, left[1], left[2], right[1], right[2], index.config.max_intersection_hits + 1),
        ))
        if len(rows) > index.config.max_intersection_hits:
            continue
        for row in rows:
            for _, field, token in (left, right):
                origins[int(row[0])].add("name_token" if field == "name" else "numeric_address" if token.isdecimal() else "address_token")
    return attempts


def test_legacy_exact_results_and_order_unchanged(tmp_path, monkeypatch):
    rows = [(f"S2-{i}", "alpha beta gamma delta epsilon zeta", f"12 road {i}", "France") for i in range(5)]
    path = build_fixture(tmp_path, rows)
    cfg = SourceIndexConfig.from_config(json.loads((ROOT / "configs/exp002c_postings.json").read_text()))
    assert cfg.intersection_scheduler == "legacy"
    query = SourceRecord("S1-1", "alpha beta gamma delta epsilon zeta", "12 road", "France")
    with SourceSideIndex(path, "S2", cfg) as index:
        current = index.retrieve(query, trace=True)
        current_pairs = [(a["left"], a["right"]) for a in index.intersection_trace["attempts"]]
        captured = []
        def frozen(country, terms, origins):
            captured.extend(frozen_legacy_intersections(index, country, terms, origins))
        monkeypatch.setattr(index, "_intersections", frozen)
        historical = index.retrieve(query)
        assert current == historical
        assert current_pairs == captured
        assert len(current_pairs) == 10


@pytest.mark.parametrize("hit_limit,expected_ids", [(1, set()), (2, {"S2-1", "S2-2"})])
def test_indexed_probes_overflow_provenance_and_open_country(tmp_path, hit_limit, expected_ids):
    rows = [("S2-1", "Alpha Beta", "12 Road", "Open Country"),
            ("S2-2", "Alpha Beta", "12 Lane", "Open Country"),
            ("S2-3", "Alpha Beta", "12 Road", "Other Country")]
    path = build_fixture(tmp_path, rows)
    cfg = replace(SourceIndexConfig(), intersection_scheduler="selective_v1", max_name_token_df=1,
                  max_address_token_df=1, max_numeric_token_df=1, max_intersection_hits=hit_limit,
                  intersection_anchor_df=1, selective_max_anchor_df=10, selective_max_expected_hits=3)
    query = SourceRecord("S1-1", "Alpha Beta Unknown", "12", "Open Country")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with SourceSideIndex(path, "S2", cfg) as index:
        sql = []
        index.connection.set_trace_callback(sql.append)
        candidates = index.retrieve(query, trace=True)
        assert {c.candidate_entity_id for c in candidates} == expected_ids
        assert all(c.country == "Open Country" for c in candidates)
        assert index.last_diagnostics["intersection_queries"] == 3
        assert index.last_diagnostics["intersection_pairs_considered"] == 3
        assert index.last_diagnostics["intersection_unique_candidate_ids"] == len(expected_ids)
        assert all("bm25" not in s.lower() and "count(*)" not in s.lower() for s in sql)
        if hit_limit == 1:
            assert index.last_diagnostics["intersection_overflows"] == 3
            assert index.last_diagnostics["intersection_nonempty_successes"] == 0
            assert all(not a["record_ids"] for a in index.intersection_trace["attempts"])
        else:
            assert all(c.name_token_score > 0 and c.numeric_address_score > 0 for c in candidates)
            assert all(len(a["record_ids"]) == 2 for a in index.intersection_trace["attempts"])
        assert index.retrieve(query) == candidates
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_trace_and_benchmark_diagnostics_reuse_without_target_tsv(tmp_path, monkeypatch):
    paths = [build_fixture(tmp_path, [(f"{s}-1", "Alpha Beta", "12 Road", "France")], s) for s in ("S2", "S3")]
    before = [path.read_bytes() for path in paths]
    # The benchmark should have no dependency on raw target TSVs or builder code.
    (tmp_path / "S2.tsv").unlink()
    (tmp_path / "S3.tsv").unlink()
    spec = importlib.util.spec_from_file_location("bench_selective", ROOT / "scripts/benchmark_indexed_blocker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def forbidden(*args, **kwargs):
        raise AssertionError("No index build allowed")
    monkeypatch.setattr(module, "build_or_open_source_index", forbidden)
    raw = tmp_path / "data/train"
    raw.mkdir(parents=True)
    (raw / "train_source1.tsv").write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-1\tAlpha Beta\t12 Road\tFrance\n")
    (raw / "train_ground_truth.tsv").write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S3-1\n")
    subset = tmp_path / "subset.csv"
    subset.write_text("source1_entity_id\nS1-1\n")
    config = json.loads((ROOT / "configs/exp002e_intersection_only.json").read_text())
    config["subset_ids_path"] = str(subset)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    common = ["bench", "--config", str(config_path), "--index-dir", str(tmp_path / "indexes"),
              "--data-root", str(raw.parent), "--results-dir", str(tmp_path / "out")]
    monkeypatch.setattr(sys, "argv", common + ["--sample-sizes", "1"])
    assert module.main() == 0
    metrics = pd.read_csv(tmp_path / "out/tables/exp002e_blocker_benchmark.csv")
    overall = metrics[metrics.dimension.eq("overall")].iloc[0]
    assert overall.intersection_scheduler == "selective_v1"
    assert overall.positive_link_recall == 1
    assert overall.average_intersection_probes_per_source_query <= 24
    assert overall.p95_intersection_probes_per_source_query <= 24
    assert overall.intersection_sql_seconds >= 0
    assert [path.read_bytes() for path in paths] == before
    for flags in (["--build-index"], ["--rebuild-index"], ["--sample-sizes", "5000"]):
        monkeypatch.setattr(sys, "argv", common + flags)
        with pytest.raises(ValueError):
            module.main()
    monkeypatch.setattr(sys, "argv", common + ["--sample-sizes", "20000", "--reviewed-1k", "--results-dir", str(tmp_path / "fresh")])
    with pytest.raises(ValueError, match="5000"):
        module.main()
