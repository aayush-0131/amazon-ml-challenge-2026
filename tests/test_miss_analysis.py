from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from business_entity_resolution.indexed_blocking import SourceIndexConfig, SourceSideIndex, build_or_open_source_index
from business_entity_resolution.miss_analysis import (
    analyze_queries, cap_budget, discovery_details, summarize_analysis,
)
from business_entity_resolution.multipass import CandidateBudget, select_candidates_for_budget
from business_entity_resolution.sampling import SourceRecord

ROOT = Path(__file__).resolve().parents[1]


def index_fixture(tmp_path, rows, config=None, source="S2"):
    path = tmp_path / f"{source}.tsv"
    path.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n" +
                    "".join("\t".join(row) + "\n" for row in rows))
    result = build_or_open_source_index(path, source, tmp_path / "indexes", chunksize=10)
    return SourceSideIndex(result.path, source, config or SourceIndexConfig())


def budget(cap=1):
    return CandidateBudget("postings", cap, cap, cap, cap, cap, cap, cap, 0)


def test_discovery_cap_and_retained_classification_single_retrieval(tmp_path, monkeypatch):
    rows = [("S2-1", "Acme", "12 Road", "France"),
            ("S2-2", "Acme", "12 Road", "France"),
            ("S2-3", "Corrupted", "99 Avenue", "France")]
    with index_fixture(tmp_path, rows) as index:
        calls = []
        original = index.retrieve
        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)
        monkeypatch.setattr(index, "retrieve", counted)
        query = SourceRecord("S1-1", "Acme", "12 Road", "France")
        audit, queries = analyze_queries({"S1-1": query}, {"S1-1": frozenset(r[0] for r in rows)}, {"S2": index}, budget())
        assert len(calls) == 1  # Not twelve retrievals for cap counterfactuals.
        assert [r["classification"] for r in audit] == ["retained", "cap_ranking_miss", "discovery_miss"]
        assert audit[1]["pool_source_rank"] == 2
        assert audit[1]["pass_budget_blocked"]
        assert audit[1]["final_source_candidate_rank"] is None
        assert "exact_name" in audit[1]["provenance"]
        assert audit[2]["shared_name_token_count"] == 0
        assert audit[2]["intersection_no_usable_combination"]
        wider_passes = replace(budget(), exact_name=5, exact_address=5)
        second_audit, _ = analyze_queries({"S1-1": query}, {"S1-1": frozenset({"S2-2"})}, {"S2": index}, wider_passes)
        assert second_audit[0]["final_source_candidate_rank"] == 2
        assert not second_audit[0]["pass_budget_blocked"]
        tables = summarize_analysis(audit, queries)
        cf = tables["exp002c_cap_counterfactuals"]
        overall = cf[cf.dimension.eq("overall") & cf.source_cap.eq(40)]
        assert overall.loc[overall["mode"].eq("total_only"), "final_recall"].iloc[0] == pytest.approx(1/3)
        assert overall.loc[overall["mode"].eq("widen_pass_limits"), "final_recall"].iloc[0] == pytest.approx(2/3)


def test_df_country_scope_and_threshold_opportunity(tmp_path):
    rows = [(f"S2-{i}", "Shared Unique" + str(i), "", "France") for i in range(70)]
    rows += [("S2-other", "Shared", "", "Other Country")]
    with index_fixture(tmp_path, rows) as index:
        q = SourceRecord("S1-1", "Shared Missing", "", "France")
        t = SourceRecord(*rows[0])
        assert not index.retrieve(q, trace=True)
        detail = discovery_details(q, t, index)
        assert detail["min_shared_name_token_df"] == 70
        assert not detail["name_60_eligible"]
        assert detail["name_120_retrievable"]
        assert not detail["current_rare_rule_retrievable"]
        assert json.loads(detail["shared_name_tokens_json"])[0]["df"] == 70
        other = replace(q, country="Other Country")
        index.retrieve(other, trace=True)
        detail_other = discovery_details(other, SourceRecord(*rows[-1]), index)
        assert detail_other["min_shared_name_token_df"] == 1


def test_eligibility_distinct_from_term_budget(tmp_path):
    rows = [("S2-1", "Alpha", "", "France"), ("S2-2", "Beta", "", "France")]
    cfg = replace(SourceIndexConfig(), name_query_terms=1)
    with index_fixture(tmp_path, rows, cfg) as index:
        query = SourceRecord("S1-1", "Alpha Beta Missing", "", "France")
        index.retrieve(query, trace=True)
        detail = discovery_details(query, SourceRecord(*rows[1]), index)
        assert detail["current_rare_df_eligible"]
        assert not detail["current_rare_rule_retrievable"]
        assert detail["rare_term_limit_blocked"]
        assert not detail["name_1000_retrievable"]


@pytest.mark.parametrize("changes,reason", [
    ({"intersection_terms": 1}, "term_omitted"),
    ({"intersection_anchor_df": 1}, "anchor_df_exceeded"),
    ({"max_intersections": 1}, "pair_budget_exceeded"),
    ({"max_intersection_hits": 1}, "intersection_overflow"),
])
def test_intersection_diagnosis_matches_actual_probes(tmp_path, changes, reason):
    rows = [("S2-1", "Alpha Beta", "", "France"),
            ("S2-2", "Alpha Beta", "", "France"),
            ("S2-3", "Aaa", "", "France")]
    cfg = replace(SourceIndexConfig(), max_name_token_df=1, **changes)
    with index_fixture(tmp_path, rows, cfg) as index:
        query = SourceRecord("S1-1", "Aaa Alpha Beta Missing", "", "France")
        without_trace = index.retrieve(query)
        with_trace = index.retrieve(query, trace=True)
        assert without_trace == with_trace
        assert "S2-1" not in {c.candidate_entity_id for c in with_trace}
        detail = discovery_details(query, SourceRecord(*rows[0]), index)
        assert detail[f"intersection_{reason}"]


def test_cap_counterfactual_rank_without_pass_exclusion(tmp_path):
    rows = [(f"S2-{i:03}", "Same", "", "France") for i in range(65)]
    with index_fixture(tmp_path, rows) as index:
        pool = index.retrieve(SourceRecord("S1-1", "Same", "", "France"))
        base = replace(budget(40), exact_name=120)
        counts = [len(select_candidates_for_budget(pool, cap_budget(base, cap, "total_only"))) for cap in (40, 50, 60, 80, 100, 120)]
        assert counts == [40, 50, 60, 65, 65, 65]


def test_source_country_slices_and_singletons(tmp_path):
    with index_fixture(tmp_path, [("S2-1", "Acme", "", "India"), ("S2-2", "Beta", "", "US")]) as s2, index_fixture(
        tmp_path, [("S3-1", "Acme", "", "India"), ("S3-2", "Beta", "", "US")], source="S3"
    ) as s3:
        records = {"S1-1": SourceRecord("S1-1", "Acme", "", "India"),
                   "S1-2": SourceRecord("S1-2", "Beta", "", "US"),
                   "S1-3": SourceRecord("S1-3", "Unseen", "", "US")}
        truth = {"S1-1": frozenset({"S2-1", "S3-1"}), "S1-2": frozenset({"S3-2"}), "S1-3": frozenset()}
        audit, queries = analyze_queries(records, truth, {"S2": s2, "S3": s3}, budget())
        tables = summarize_analysis(audit, queries)
        summary = tables["exp002c_miss_summary"]
        assert set(summary.group) == {"ALL", "India", "US", "S2", "S3", "S2|India", "S2|US", "S3|India", "S3|US"}
        assert summary.loc[summary.group.eq("ALL"), "true_links"].iloc[0] == 3
        assert tables["exp002c_discovery_misses"].empty
        assert "intersection_reasons" in tables["exp002c_discovery_misses"].columns
        caps = tables["exp002c_cap_counterfactuals"]
        overall = caps[caps.dimension.eq("overall")].iloc[0]
        assert overall.s1_entities == 3
        assert overall.average_candidates == pytest.approx(4/3)
        assert overall.zero_candidate_entities == 1


def script_module():
    spec = importlib.util.spec_from_file_location("miss_cli", ROOT / "scripts/analyze_exp002c_misses.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_determinism_no_rebuild_and_fixed_sample(tmp_path, monkeypatch):
    import business_entity_resolution.indexed_blocking as module
    rows = [("S2-1", "Acme", "", "Open Country")]
    with index_fixture(tmp_path, rows):
        pass
    with index_fixture(tmp_path, [("S3-1", "Acme", "", "Open Country")], source="S3"):
        pass
    data = tmp_path / "raw/train"
    data.mkdir(parents=True)
    (data / "train_source1.tsv").write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-1\tAcme\t\tOpen Country\n")
    (data / "train_ground_truth.tsv").write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S3-1\n")
    subset = tmp_path / "subset.csv"
    subset.write_text("source1_entity_id\nS1-1\n")
    config = json.loads((ROOT / "configs/exp002c_postings.json").read_text())
    config["subset_ids_path"] = str(subset)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    def forbidden(*args, **kwargs):
        raise AssertionError("Analysis must not build indexes")
    monkeypatch.setattr(module, "build_or_open_source_index", forbidden)
    cli = script_module()
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (tmp_path / "indexes").glob("*.sqlite")}
    common = ["--config", str(config_path), "--index-dir", str(tmp_path / "indexes"), "--data-root", str(data.parent), "--sample-size", "1"]
    for output in ("first", "second"):
        assert cli.main(common + ["--results-dir", str(tmp_path / output)]) == 0
    first, second = tmp_path / "first", tmp_path / "second"
    assert len(list(first.iterdir())) == 8
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
    assert before == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (tmp_path / "indexes").glob("*.sqlite")}
    with pytest.raises(FileExistsError):
        cli.main(common + ["--results-dir", str(first)])
    with pytest.raises(FileNotFoundError, match="never builds"):
        cli.main(common + ["--index-dir", str(tmp_path / "absent"), "--results-dir", str(tmp_path / "missing")])
    assert not (tmp_path / "absent").exists()
    with pytest.raises(ValueError, match="seed 2032"):
        cli.sample_ids(subset, 1, 2033)


def test_sample_matches_exp002c_order(tmp_path):
    from business_entity_resolution.split import stable_entity_key
    ids = [f"S1-{n}" for n in range(20)]
    subset = tmp_path / "subset.csv"
    subset.write_text("source1_entity_id\n" + "\n".join(reversed(ids)))
    expected = sorted(ids, key=lambda eid: (stable_entity_key(eid, 2032), eid))[:5]
    assert script_module().sample_ids(subset, 5, 2032) == expected


def test_empty_truth_and_missing_target_integrity(tmp_path):
    from business_entity_resolution.miss_analysis import write_report
    with index_fixture(tmp_path, [("S2-1", "Acme", "", "France")]) as index:
        records = {"S1-1": SourceRecord("S1-1", "Unknown", "", "France")}
        audit, queries = analyze_queries(records, {"S1-1": frozenset()}, {"S2": index}, budget())
        tables = summarize_analysis(audit, queries)
        assert tables["exp002c_true_link_audit"].empty
        assert "source1_entity_id" in tables["exp002c_true_link_audit"].columns
        assert "theoretical" in write_report(tables)
        with pytest.raises(ValueError, match="True target IDs absent"):
            analyze_queries(records, {"S1-1": frozenset({"S2-missing"})}, {"S2": index}, budget())
