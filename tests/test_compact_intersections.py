"""Synthetic EXP002f frontier tests. Never use Amazon data or existing indexes."""
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from itertools import combinations
import json

import pytest

from business_entity_resolution.indexed_blocking import SourceIndexConfig, SourceSideIndex
from business_entity_resolution.intersection_scheduler import plan_intersections, pair_kind, term_kind
from business_entity_resolution.multipass import CandidateBudget
from business_entity_resolution.sampling import SourceRecord
from test_selective_intersections import ROOT, build_fixture, plan


@pytest.mark.parametrize("probes,quota,budget", [(6, 3, 100), (10, 4, 150), (14, 5, 200), (18, 6, 250)])
def test_frontier_configs(probes, quota, budget):
    values = json.loads((ROOT / f"configs/exp002f_compact_{probes}.json").read_text())
    cfg = SourceIndexConfig.from_config(values)
    assert asdict(cfg) == values["source_index"]
    assert cfg.intersection_scheduler == "selective_v2_compact"
    assert (cfg.max_intersections, cfg.selective_max_probes_per_term, cfg.compact_unique_candidate_budget) == (probes, quota, budget)
    assert (cfg.max_name_token_df, cfg.max_address_token_df, cfg.max_numeric_token_df) == (60, 40, 250)
    assert cfg.intersection_terms == 16
    caps = CandidateBudget(**values["blocker_budgets"][0])
    assert caps.total_per_source == 60 and caps.char_name == 0
    assert all(getattr(caps, key) == 40 for key in ("exact_name", "exact_address", "name_signature", "name_token", "address_token", "numeric_address"))
    terms = [(i, "name", str(i)) for i in range(1, 17)]
    result = plan(terms, probe_limit=probes, max_probes_per_term=quota, gate_all_pairs=True)
    assert len(result.pairs) == probes
    assert max(Counter(t for pair in result.pairs for t in pair).values()) <= quota
    del values["source_index"]["compact_unique_candidate_budget"]
    with pytest.raises(ValueError, match="keys mismatch"):
        SourceIndexConfig.from_config(values)
    with pytest.raises(ValueError):
        replace(cfg, compact_unique_candidate_budget=0)


def frozen_v1_pairs(terms, population, *, term_limit=16, probe_limit=24, quota=6):
    """Frozen EXP002e planner, independent of the new all-pairs gate."""
    ordered = sorted(set(t for t in terms if t[0] > 0))
    seeds = {}
    for term in ordered:
        seeds.setdefault(term_kind(term), term)
    pool = sorted(seeds.values())[:term_limit]
    reserved = set(pool)
    pool.extend(t for t in ordered if t not in reserved)
    pool = sorted(pool[:term_limit])
    eligible = [(a, b) for a, b in combinations(pool, 2)
                if a[0] <= 50000 and not (a[0] > 5000 and
                (population == 0 or a[0] * b[0] > 150 * population))]
    usage, kinds = Counter(), Counter()
    selected = []
    while eligible and len(selected) < probe_limit:
        eligible = [p for p in eligible if all(usage[t] < quota for t in p)]
        if not eligible:
            break
        pair = min(eligible, key=lambda p: (
            max(usage[p[0]], usage[p[1]]), usage[p[0]] + usage[p[1]],
            kinds[pair_kind(p)], p[0][0] * p[1][0], p[0], p[1]))
        eligible.remove(pair)
        selected.append(pair)
        usage.update(pair)
        kinds[pair_kind(pair)] += 1
    return tuple(selected)


@pytest.mark.parametrize("population", [100, 10000, 1000000])
def test_v1_frozen_order_unchanged(population):
    terms = [(df, field, token) for df, field, token in (
        (100, "name", "alpha"), (200, "name", "beta"), (300, "address", "road"),
        (450, "address", "١٢٣"), (6000, "name", "gamma"), (7000, "address", "lane"),
        (50001, "name", "delta"), (60000, "address", "avenue"))]
    assert plan(terms, population).pairs == frozen_v1_pairs(terms, population)
    compact = plan(terms, population, gate_all_pairs=True)
    assert compact == plan(terms[::-1], population, gate_all_pairs=True)


def test_all_pair_gate_and_diversity():
    terms = [(100, "name", "alpha"), (100, "address", "road")]
    assert plan(terms, 100, max_expected_hits=50).pairs  # v1 ignores gate for small anchors
    result = plan(terms, 100, max_expected_hits=50, gate_all_pairs=True)
    assert not result.pairs and result.skipped_selectivity == 1
    assert plan([(50001, "name", "a"), (60000, "name", "b")], gate_all_pairs=True).skipped_anchor == 1
    mixed = [(10, "name", "alpha"), (20, "name", "beta"), (30, "address", "road"),
             (40, "address", "lane"), (50, "address", "12"), (60, "address", "١٢٣")]
    result = plan(mixed, probe_limit=9, max_probes_per_term=3, gate_all_pairs=True)
    assert ("name", "name") in {pair_kind(p) for p in result.pairs}
    assert any(pair_kind(p)[0] != pair_kind(p)[1] for p in result.pairs)
    assert any(term_kind(t) == "numeric" for p in result.pairs for t in p)
    assert max(Counter(t for p in result.pairs for t in p).values()) <= 3
    limited = plan(mixed, max_probes_per_term=1, gate_all_pairs=True)
    assert limited.skipped_term_quota > 0
    assert len(limited.pairs) == 3


def fixture_index(tmp_path, **kwargs):
    rows = [(f"S2-{i}", name, "", "Open Country Ω") for i, name in enumerate(
        ("alpha beta", "alpha beta", "gamma delta", "gamma delta", "alpha gamma", "beta delta"), 1)]
    path = build_fixture(tmp_path, rows)
    cfg = replace(SourceIndexConfig(), intersection_scheduler="selective_v2_compact",
                  max_intersections=6, selective_max_probes_per_term=3, **kwargs)
    return path, cfg


def intersections(index, initial=()):
    index.last_diagnostics = Counter()
    index.intersection_trace = {"attempts": []}
    origins = defaultdict(set, {i: {"exact_name"} for i in initial})
    index._intersections("Open Country Ω", [("name", t) for t in ("alpha", "beta", "gamma", "delta")], origins)
    return origins


@pytest.mark.parametrize("budget,initial,expected_probes,expected_unique", [
    (2, (), 1, 2), (3, (), 2, 4), (3, (1,), 2, 3),
])
def test_budget_stops_after_complete_probe(tmp_path, budget, initial, expected_probes, expected_unique):
    path, cfg = fixture_index(tmp_path, compact_unique_candidate_budget=budget)
    before = path.read_bytes()
    with SourceSideIndex(path, "S2", cfg) as index:
        sql = []
        index.connection.set_trace_callback(sql.append)
        result = intersections(index, initial)
        diag, trace = index.last_diagnostics, index.intersection_trace
        assert diag["intersection_queries"] == expected_probes
        assert sum("CROSS JOIN" in statement for statement in sql) == expected_probes
        assert len(set(result) - set(initial)) == expected_unique
        assert diag["intersection_unique_candidate_ids"] == expected_unique
        assert diag["intersection_candidate_budget_reached"] == 1
        assert diag["intersection_pairs_skipped_candidate_budget"] == 6 - expected_probes
        assert trace["candidate_budget_reached"] and trace["stopped_by_candidate_budget"]
        assert len(trace["planned_probes"]) == 6
        assert trace["attempts"][-1]["cumulative_unique_candidate_count"] == expected_unique
        assert all(len(a["record_ids"]) == 2 for a in trace["attempts"])
    assert path.read_bytes() == before


def test_overflow_empty_and_existing_ids_do_not_consume_budget(tmp_path):
    path, cfg = fixture_index(tmp_path, max_intersection_hits=1, compact_unique_candidate_budget=1)
    with SourceSideIndex(path, "S2", cfg) as index:
        result = intersections(index)
        attempts = index.intersection_trace["attempts"]
        assert len(result) == 1
        assert index.last_diagnostics["intersection_overflows"] == 2
        assert index.last_diagnostics["intersection_empty_probes"] == 2
        assert index.last_diagnostics["intersection_nonempty_successes"] == 1
        assert index.last_diagnostics["intersection_queries"] == 5
        for attempt in attempts[:-1]:
            assert attempt["new_candidate_ids"] == 0
            assert attempt["cumulative_unique_candidate_count"] == 0
        assert attempts[-1]["new_candidate_ids"] == 1


def test_v1_ignores_compact_budget_and_trace_is_observational(tmp_path):
    path, cfg = fixture_index(tmp_path, compact_unique_candidate_budget=1)
    query = SourceRecord("S1-1", "alpha beta gamma delta", "", "Open Country Ω")
    with SourceSideIndex(path, "S2", cfg) as index:
        candidates = index.retrieve(query, trace=True)
        assert candidates == index.retrieve(query, trace=False)
        assert all(c.country == "Open Country Ω" for c in candidates)
    with SourceSideIndex(path, "S2", replace(cfg, intersection_scheduler="selective_v1")) as index:
        assert len(intersections(index)) == 6
        assert index.last_diagnostics["intersection_queries"] == 6
        assert index.last_diagnostics["intersection_candidate_budget_reached"] == 0


def test_selectivity_skip_diagnostics_issue_no_sql(tmp_path, monkeypatch):
    path, cfg = fixture_index(tmp_path, selective_max_expected_hits=1)
    with SourceSideIndex(path, "S2", cfg) as index:
        # All four real terms have DF=3 / N=6, proxy=1.5 > gate=1.
        sql = []
        index.connection.set_trace_callback(sql.append)
        assert not intersections(index)
        assert index.last_diagnostics["intersection_pairs_skipped_selectivity"] == 6
        assert index.last_diagnostics["intersection_planned_pairs"] == 0
        assert index.last_diagnostics["intersection_queries"] == 0
        assert not any("CROSS JOIN" in statement for statement in sql)
        assert not index.intersection_trace["candidate_budget_reached"]
        monkeypatch.setattr(index, "_df", lambda *args: 50001)
        assert not intersections(index)
        assert index.last_diagnostics["intersection_pairs_skipped_anchor"] == 6
        assert index.last_diagnostics["intersection_pairs_skipped_selectivity"] == 0
