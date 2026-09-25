"""Read-only EXP002c true-link diagnostics and same-pool counterfactuals.

Ground truth is used only after retrieval. Diagnostic token eligibility is a
theoretical discovery opportunity, not measured end-to-end recall or runtime.
"""
from __future__ import annotations

import json
from dataclasses import replace
from itertools import combinations

import numpy as np
import pandas as pd

from .indexed_blocking import SourceSideIndex
from .multipass import PASS_NAMES, CandidateBudget, select_candidates_for_budget
from .normalize import represent_address, represent_name
from .sampling import SourceRecord
from .similarity import token_jaccard, token_overlap

DF_THRESHOLDS = {
    "name": (60, 120, 250, 500, 1000),
    "address": (40, 80, 160, 320, 640),
    "numeric": (250, 500, 1000, 2500, 5000),
}
CAPS = (40, 50, 60, 80, 100, 120)
CAP_MODES = ("total_only", "widen_pass_limits")
INTERSECTION_REASONS = (
    "term_omitted", "anchor_df_exceeded", "pair_budget_exceeded",
    "intersection_overflow", "no_usable_combination", "other_unresolved",
)
BASE_COLUMNS = [
    "source1_entity_id", "candidate_entity_id", "source", "country",
    "retrieved_before_cap", "survived_final_selection", "classification",
    "current_total_per_source", "pool_source_rank", "final_source_candidate_rank",
    "pass_budget_blocked", "provenance", "best_retrieval_score",
    *[f"{p}_rank" for p in PASS_NAMES], *[f"{p}_score" for p in PASS_NAMES],
]
DETAIL_COLUMNS = [
    "target_country", "country_equal", "normalized_name_equal", "normalized_address_equal",
    "name_signature_equal", "shared_name_token_count", "shared_address_token_count",
    "shared_numeric_token_count", "name_token_jaccard", "address_token_jaccard", "numeric_overlap",
    *[f"min_shared_{field}_token_df" for field in DF_THRESHOLDS],
    *[f"shared_{field}_tokens_json" for field in DF_THRESHOLDS],
    "current_rare_df_eligible", "current_rare_rule_retrievable",
    "rare_term_limit_blocked", "intersection_reasons", "intersection_evidence_json",
    *[f"intersection_{reason}" for reason in INTERSECTION_REASONS],
    *[f"{field}_{threshold}_{kind}" for field, thresholds in DF_THRESHOLDS.items()
      for threshold in thresholds for kind in ("eligible", "retrievable")],
]


def stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def lookup_targets(index: SourceSideIndex, ids: list[str]) -> dict[str, SourceRecord]:
    """Use the schema-2 UNIQUE entity_id index; never scan the target table."""
    records = {}
    for offset in range(0, len(ids), index.config.record_batch_size):
        batch = ids[offset:offset + index.config.record_batch_size]
        placeholders = ",".join("?" for _ in batch)
        for row in index.connection.execute(
            f"SELECT entity_id,business_name,business_address,country FROM records WHERE entity_id IN ({placeholders})", batch
        ):
            records[row["entity_id"]] = SourceRecord(
                row["entity_id"], row["business_name"], row["business_address"], row["country"]
            )
    if set(ids) != set(records):
        raise ValueError(f"True target IDs absent from completed {index.source} index: {sorted(set(ids) - set(records))[:5]}")
    return records


def intersection_diagnosis(index: SourceSideIndex, shared: set[tuple[str, str]]) -> dict:
    """Classify ALL shared usable pairs; reasons overlap across combinations.

    Actual attempted probes and bounded hit counts come directly from retrieval.
    No diagnostic SQL intersection is re-executed.
    """
    trace = index.intersection_trace
    if index.config.intersection_scheduler != "legacy":
        raise ValueError("EXP002d diagnosis describes legacy intersection ordering; use the historical EXP002c config")
    if trace is None:
        raise ValueError("Intersection diagnosis requires retrieve(trace=True)")
    cfg = index.config
    ranked = trace["ranked_terms"]
    positions = {(field, token): n for n, (_, field, token) in enumerate(ranked)}
    shared_ranked = [tuple(term) for term in ranked if (term[1], term[2]) in shared]
    attempts = {(tuple(a["left"]), tuple(a["right"])): a for a in trace["attempts"]}
    reasons = set()
    evidence = []
    for left, right in combinations(shared_ranked, 2):
        hits = None
        if max(positions[left[1:]], positions[right[1:]]) >= cfg.intersection_terms:
            reason = "term_omitted"
        elif left[0] > cfg.intersection_anchor_df:
            reason = "anchor_df_exceeded"
        elif (left, right) not in attempts:
            reason = "pair_budget_exceeded"
        else:
            attempt = attempts[left, right]
            hits = attempt["hits_bounded"]
            reason = "intersection_overflow" if attempt["overflow"] else "other_unresolved"
        reasons.add(reason)
        evidence.append({"left": left, "right": right, "reason": reason, "hits_bounded": hits})
    if not evidence:
        reasons.add("no_usable_combination")
    return {
        "intersection_reasons": "|".join(sorted(reasons)),
        "intersection_evidence_json": stable_json(evidence),
        **{f"intersection_{reason}": reason in reasons for reason in INTERSECTION_REASONS},
    }


def discovery_details(query: SourceRecord, target: SourceRecord, index: SourceSideIndex) -> dict:
    cfg = index.config
    qn, qa = represent_name(query.business_name), represent_address(query.business_address)
    tn, ta = represent_name(target.business_name), represent_address(target.business_address)
    same_country = query.country == target.country
    specs = {
        "name": (qn.unique_tokens, tn.unique_tokens, "name", cfg.min_name_token_length, cfg.name_query_terms, cfg.max_name_token_df),
        "address": (qa.unique_tokens, ta.unique_tokens, "address", cfg.min_address_token_length, cfg.address_query_terms, cfg.max_address_token_df),
        "numeric": (qa.digit_tokens, ta.digit_tokens, "address", 1, cfg.numeric_query_terms, cfg.max_numeric_token_df),
    }
    row = {
        "target_country": target.country, "country_equal": same_country,
        "normalized_name_equal": bool(qn.normalized and qn.normalized == tn.normalized),
        "normalized_address_equal": bool(qa.normalized and qa.normalized == ta.normalized),
        "name_signature_equal": bool(qn.unique_tokens and qn.unique_tokens == tn.unique_tokens),
        "name_token_jaccard": token_jaccard(qn.unique_tokens, tn.unique_tokens),
        "address_token_jaccard": token_jaccard(qa.unique_tokens, ta.unique_tokens),
        "numeric_overlap": token_overlap(qa.digit_tokens, ta.digit_tokens),
    }
    current_eligible = current_retrievable = False
    for field, (query_tokens, target_tokens, db_field, minimum_length, term_limit, current_df) in specs.items():
        shared = query_tokens & target_tokens
        dfs = {token: index._df(db_field, token, query.country) for token in sorted(query_tokens)}
        usable = sorted((df, token) for token, df in dfs.items() if df > 0 and len(token) >= minimum_length)
        positions = {token: rank for rank, (_, token) in enumerate(usable, 1)}
        row[f"shared_{field}_token_count"] = len(shared)
        row[f"shared_{field}_tokens_json"] = stable_json([
            {"token": token, "df": dfs[token], "usable": token in positions,
             "query_rare_order_rank": positions.get(token)} for token in sorted(shared)
        ])
        row[f"min_shared_{field}_token_df"] = min((dfs[token] for token in shared), default=None)

        def opportunity(threshold: int) -> tuple[bool, bool]:
            eligible = [(df, token) for df, token in usable if df <= threshold]
            return (
                same_country and any(token in shared for _, token in eligible),
                same_country and any(token in shared for _, token in eligible[:term_limit]),
            )

        eligible, retrievable = opportunity(current_df)
        current_eligible |= eligible
        current_retrievable |= retrievable
        for threshold in DF_THRESHOLDS[field]:
            eligible, retrievable = opportunity(threshold)
            row[f"{field}_{threshold}_eligible"] = eligible
            row[f"{field}_{threshold}_retrievable"] = retrievable
    row["current_rare_df_eligible"] = current_eligible
    row["current_rare_rule_retrievable"] = current_retrievable
    row["rare_term_limit_blocked"] = current_eligible and not current_retrievable
    shared_terms = {("name", token) for token in qn.unique_tokens & tn.unique_tokens}
    shared_terms |= {("address", token) for token in qa.unique_tokens & ta.unique_tokens}
    row.update(intersection_diagnosis(index, shared_terms))
    return row


def cap_budget(budget: CandidateBudget, cap: int, mode: str) -> CandidateBudget:
    if mode == "total_only":
        return replace(budget, total_per_source=cap)
    if mode != "widen_pass_limits":
        raise ValueError(mode)
    return replace(budget, total_per_source=cap, **{
        name: max(getattr(budget, name), cap) if getattr(budget, name) else 0 for name in PASS_NAMES
    })


def analyze_queries(records: dict[str, SourceRecord], truth: dict[str, frozenset[str]],
                    indexes: dict[str, SourceSideIndex], budget: CandidateBudget) -> tuple[list[dict], list[dict]]:
    audit, queries = [], []
    for entity_id in sorted(records):
        query = records[entity_id]
        for source, index in sorted(indexes.items()):
            pool = index.retrieve(query, trace=True)  # Exactly one retrieval per S1/source.
            expected = sorted(target for target in truth[entity_id] if target.startswith(f"{source}-"))
            pool_by_id = {candidate.candidate_entity_id: candidate for candidate in pool}
            chosen = {candidate.candidate_entity_id for candidate in select_candidates_for_budget(pool, budget)}
            missing = [target for target in expected if target not in pool_by_id]
            targets = lookup_targets(index, missing)
            order = sorted(pool, key=lambda c: (-c.best_retrieval_score, -c.pass_count, c.candidate_entity_id))
            pool_ranks = {c.candidate_entity_id: rank for rank, c in enumerate(order, 1)}
            eligible_ranks = {c.candidate_entity_id: rank for rank, c in enumerate(
                (c for c in order if c.selected_by(budget)), 1)}
            alternatives = {
                (mode, cap): {c.candidate_entity_id for c in select_candidates_for_budget(pool, cap_budget(budget, cap, mode))}
                for mode in CAP_MODES for cap in CAPS
            }
            queries.append({"source1_entity_id": entity_id, "source": source, "country": query.country,
                            "current_candidates": len(chosen),
                            "candidate_counts": {key: len(value) for key, value in alternatives.items()}})
            for target_id in expected:
                candidate = pool_by_id.get(target_id)
                retrieved, survived = candidate is not None, target_id in chosen
                row = dict.fromkeys(BASE_COLUMNS + DETAIL_COLUMNS)
                row.update(
                    source1_entity_id=entity_id, candidate_entity_id=target_id, source=source, country=query.country,
                    retrieved_before_cap=retrieved, survived_final_selection=survived,
                    classification="retained" if survived else "cap_ranking_miss" if retrieved else "discovery_miss",
                    current_total_per_source=budget.total_per_source, pool_source_rank=pool_ranks.get(target_id),
                    final_source_candidate_rank=eligible_ranks.get(target_id),
                    pass_budget_blocked=retrieved and target_id not in eligible_ranks,
                )
                if candidate:
                    row.update(provenance="|".join(p for p in PASS_NAMES if getattr(candidate, f"{p}_rank") > 0),
                               best_retrieval_score=candidate.best_retrieval_score)
                    row.update({f"{p}_{kind}": getattr(candidate, f"{p}_{kind}") for p in PASS_NAMES for kind in ("rank", "score")})
                else:
                    row.update(discovery_details(query, targets[target_id], index))
                row.update({f"cap_{mode}_{cap}": target_id in ids for (mode, cap), ids in alternatives.items()})
                audit.append(row)
    return audit, queries


def slices(countries: set[str]) -> list[tuple[str, str, str | None, str | None]]:
    result = [("overall", "ALL", None, None)]
    result += [("country", country, country, None) for country in sorted(countries)]
    for source in ("S2", "S3"):
        result.append(("source", source, None, source))
        result += [("source_country", f"{source}|{country}", country, source) for country in sorted(countries)]
    return result


def summarize_analysis(audit: list[dict], queries: list[dict]) -> dict[str, pd.DataFrame]:
    summaries, df_rows, cap_rows = [], [], []
    for dimension, group, country, source in slices({q["country"] for q in queries}):
        def included(row: dict) -> bool:
            return (country is None or row["country"] == country) and (source is None or row["source"] == source)
        links = [row for row in audit if included(row)]
        scope_queries = [row for row in queries if included(row)]
        if not scope_queries:
            continue
        discovery = [row for row in links if row["classification"] == "discovery_miss"]
        base = {"dimension": dimension, "group": group, "true_links": len(links)}
        total = len(links)
        pre = sum(row["retrieved_before_cap"] for row in links)
        final = sum(row["survived_final_selection"] for row in links)
        def rate(n: int) -> float:
            return n / total if total else 1.0
        summary = {**base, "discovery_misses": len(discovery), "cap_ranking_misses": pre - final,
                   "retained_links": final, "pre_cap_recall": rate(pre), "final_recall": rate(final)}
        grouped_links: dict[str, list[dict]] = {}
        for row in links:
            grouped_links.setdefault(row["source1_entity_id"], []).append(row)
        summary["matched_entities"] = len(grouped_links)
        summary["matched_all_links_retained_rate"] = (
            sum(all(row["survived_final_selection"] for row in values) for values in grouped_links.values()) / len(grouped_links)
            if grouped_links else 1.0
        )
        summary["average_current_candidates_per_s1"] = sum(q["current_candidates"] for q in scope_queries) / len({q["source1_entity_id"] for q in scope_queries})
        summary.update({f"intersection_{reason}": sum(bool(row[f"intersection_{reason}"]) for row in discovery)
                        for reason in INTERSECTION_REASONS})
        summary["rare_term_limit_blocked"] = sum(row["rare_term_limit_blocked"] for row in discovery)
        summary["current_rare_rule_unresolved"] = sum(row["current_rare_rule_retrievable"] for row in discovery)
        summaries.append(summary)
        for field, thresholds in DF_THRESHOLDS.items():
            for threshold in thresholds:
                eligible = sum(row[f"{field}_{threshold}_eligible"] for row in discovery)
                retrievable = sum(row[f"{field}_{threshold}_retrievable"] for row in discovery)
                df_rows.append({**base, "field": field, "max_df": threshold, "discovery_misses": len(discovery),
                                "df_eligible_misses": eligible, "recoverable_with_current_term_limit": retrievable,
                                "theoretical_pre_cap_recall": rate(pre + retrievable),
                                "theoretical_gain_pp": 100 * retrievable / total if total else 0.0})
        for mode in CAP_MODES:
            for cap in CAPS:
                key = f"cap_{mode}_{cap}"
                retained = sum(row[key] for row in links)
                entity_counts: dict[str, int] = {}
                matched: dict[str, list[bool]] = {}
                for query in scope_queries:
                    eid = query["source1_entity_id"]
                    entity_counts[eid] = entity_counts.get(eid, 0) + query["candidate_counts"][mode, cap]
                for row in links:
                    matched.setdefault(row["source1_entity_id"], []).append(row[key])
                values = list(entity_counts.values())
                cap_rows.append({**base, "mode": mode, "source_cap": cap,
                                 "retained_links": retained, "final_recall": rate(retained),
                                 "additional_true_links_vs_current": retained - final,
                                 "matched_all_links_rate": sum(all(x) for x in matched.values()) / len(matched) if matched else 1.0,
                                 "s1_entities": len(values), "average_candidates": float(np.mean(values)),
                                 "median_candidates": float(np.median(values)), "p95_candidates": float(np.quantile(values, .95)),
                                 "p99_candidates": float(np.quantile(values, .99)), "max_candidates": max(values),
                                 "zero_candidate_entities": values.count(0)})
    audit_columns = BASE_COLUMNS + DETAIL_COLUMNS + [f"cap_{mode}_{cap}" for mode in CAP_MODES for cap in CAPS]
    frame = pd.DataFrame(audit, columns=audit_columns)
    return {
        "exp002c_true_link_audit": frame,
        "exp002c_discovery_misses": frame.loc[frame.classification.eq("discovery_miss")],
        "exp002c_cap_misses": frame.loc[frame.classification.eq("cap_ranking_miss")],
        "exp002c_miss_summary": pd.DataFrame(summaries),
        "exp002c_df_counterfactuals": pd.DataFrame(df_rows),
        "exp002c_cap_counterfactuals": pd.DataFrame(cap_rows),
    }


def write_report(tables: dict[str, pd.DataFrame]) -> str:
    summary = tables["exp002c_miss_summary"]
    lines = ["# EXP002d — EXP002c retrieval miss analysis", "",
             "One retrieval per S1/source; all cap scenarios reuse that pool. These are TRAIN diagnostics.", "",
             "| Slice | True links | Discovery misses | Cap/ranking misses | Pre-cap recall | Final recall |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.group} | {row.true_links} | {row.discovery_misses} | {row.cap_ranking_misses} | {row.pre_cap_recall:.4%} | {row.final_recall:.4%} |")
    lines += ["", "DF counterfactuals change one field at a time. Eligibility ignores query-term limits; recoverability respects current term ordering, term limits, minimum lengths and raw country equality. Potential recovery is discovery-only and does not guarantee final selection or runtime safety. Overlapping recoveries must not be summed across thresholds or fields.",
              "", "Cap modes: `total_only` preserves current per-pass rank limits; `widen_pass_limits` also raises each active pass limit to at least the proposed cap. Neither mode adds candidates to the retrieved pool. Blank final-source rank means the candidate failed per-pass eligibility; pool-source rank is provided separately.",
              "", "Intersection reasons are non-exclusive across shared token combinations. Actual probe order and overflow decisions are captured during retrieval; hit counts stop at max_intersection_hits+1. No new probes are run for counterfactuals. `other_unresolved` indicates an attempted non-overflow combination apparently sharing a missing true link and needs integrity investigation.", "",
              "## Next-step evidence", ""]
    for row in summary[summary.dimension.eq("country")].itertuples(index=False):
        priority = "candidate discovery" if row.discovery_misses >= row.cap_ranking_misses else "ranking and selection"
        lines.append(f"- {row.group}: prioritize {priority} ({row.discovery_misses} discovery vs {row.cap_ranking_misses} cap/ranking misses).")
    potentials = tables["exp002c_df_counterfactuals"]
    for field in DF_THRESHOLDS:
        subset = potentials[(potentials.dimension == "overall") & (potentials.field == field)]
        best = subset.sort_values(["recoverable_with_current_term_limit", "max_df"], ascending=[False, True]).iloc[0]
        lines.append(f"- {field}: best single-field theoretical recovery is {best.recoverable_with_current_term_limit} discovery links at DF {best.max_df}; validate cost and final recall before adoption.")
    cap_table = tables["exp002c_cap_counterfactuals"]
    for row in cap_table[(cap_table.dimension == "overall") & (cap_table.source_cap == 120)].itertuples(index=False):
        lines.append(f"- At cap 120/source, `{row.mode}` recovers {row.additional_true_links_vs_current} additional links from the same pool, with final recall {row.final_recall:.4%} and {row.average_candidates:.2f} candidates/S1. This cannot recover discovery misses.")
    lines += ["", "No index rebuild, model fitting, TEST inference or recommendation to deploy a counterfactual is implied. Review miss CSVs and country/source slices before designing the next retrieval change.", ""]
    return "\n".join(lines)
