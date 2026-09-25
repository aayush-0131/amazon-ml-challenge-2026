#!/usr/bin/env python3
"""Build/reuse EXP002 source indexes and benchmark indexed lexical blocking only.

This script deliberately stops before pair-feature construction or model fitting.
It never scans S2/S3 while answering Source-1 queries: target scans are limited
to the explicit, one-time ``--build-index`` step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import subprocess
import time
from dataclasses import asdict
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from business_entity_resolution.indexed_blocking import (  # noqa: E402
    INDEX_SCHEMA_VERSION,
    IndexBuildResult,
    SourceIndexConfig,
    SourceSideIndex,
    _index_path,
    build_or_open_source_index,
)
from business_entity_resolution.multipass import (  # noqa: E402
    CandidateBudget,
    select_candidates_for_budget,
)
from business_entity_resolution.sampling import (  # noqa: E402
    SourceRecord,
    load_selected_ground_truth,
    load_selected_source_records,
)
from business_entity_resolution.split import stable_entity_key  # noqa: E402


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024) if platform.system() == "Darwin" else value / 1024


def load_budget(config: dict[str, object], name: str) -> CandidateBudget:
    for values in config["blocker_budgets"]:
        if values["name"] == name:
            return CandidateBudget(**values)
    available = [values["name"] for values in config["blocker_budgets"]]
    raise ValueError(f"Unknown budget {name!r}; available={available!r}")


def benchmark_ids(records: dict[str, SourceRecord], size: int, seed: int) -> list[str]:
    if not 0 < size <= len(records):
        raise ValueError(f"sample size must be in [1, {len(records):,}]")
    return sorted(
        records,
        key=lambda entity_id: (stable_entity_key(entity_id, seed), entity_id),
    )[:size]


def summarize(
    records: dict[str, SourceRecord],
    truth: dict[str, frozenset[str]],
    selected: dict[str, set[str]],
    sample_ids: list[str],
    sample_name: str,
    query_seconds: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    countries = sorted({records[entity_id].country for entity_id in sample_ids})
    specs: list[tuple[str, str, str | None, list[str]]] = [
        ("overall", "ALL", None, sample_ids)
    ]
    specs.extend(
        ("country", country, None, [entity_id for entity_id in sample_ids if records[entity_id].country == country])
        for country in countries
    )
    for source in ("S2", "S3"):
        specs.append(("source", source, source, sample_ids))
        specs.extend(
            (
                "source_country",
                f"{source}|{country}",
                source,
                [entity_id for entity_id in sample_ids if records[entity_id].country == country],
            )
            for country in countries
        )
    for dimension, group, source, ids in specs:
        candidate_counts: list[int] = []
        true_total = retained_total = matched_total = matched_all = zero = 0
        for entity_id in ids:
            expected = {
                candidate_id
                for candidate_id in truth[entity_id]
                if source is None or candidate_id.startswith(f"{source}-")
            }
            candidates = {
                candidate_id
                for candidate_id in selected[entity_id]
                if source is None or candidate_id.startswith(f"{source}-")
            }
            retained = expected & candidates
            candidate_counts.append(len(candidates))
            true_total += len(expected)
            retained_total += len(retained)
            matched_total += int(bool(expected))
            matched_all += int(bool(expected) and retained == expected)
            zero += int(not candidates)
        values = np.asarray(candidate_counts, dtype=np.int64)
        rows.append(
            {
                "sample": sample_name,
                "dimension": dimension,
                "group": group,
                "entity_count": len(ids),
                "true_link_count": true_total,
                "retained_true_link_count": retained_total,
                "positive_link_recall": retained_total / true_total if true_total else 1.0,
                "matched_all_true_links_retained_rate": matched_all / matched_total if matched_total else 1.0,
                "average_candidates": float(values.mean()),
                "median_candidates": float(np.quantile(values, 0.5)),
                "p95_candidates": float(np.quantile(values, 0.95)),
                "p99_candidates": float(np.quantile(values, 0.99)),
                "max_candidates": int(values.max()),
                "zero_candidate_entities": zero,
                "query_seconds": query_seconds,
                "peak_rss_mb": peak_rss_mb(),
            }
        )
    return rows


def run_benchmark(
    records: dict[str, SourceRecord],
    truth: dict[str, frozenset[str]],
    indexes: dict[str, SourceSideIndex],
    budget: CandidateBudget,
    sample_ids: list[str],
    sample_name: str,
    diagnostics_out: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    started = time.monotonic()
    selected: dict[str, set[str]] = defaultdict(set)
    diagnostic_rows: list[dict[str, object]] = []
    for entity_id in sample_ids:
        record = records[entity_id]
        for source, index in indexes.items():
            query_started = time.monotonic()
            pool = index.retrieve(record)
            eligible = [candidate for candidate in pool if candidate.selected_by(budget)]
            candidates = select_candidates_for_budget(pool, budget)
            diagnostic = {
                "sample": sample_name, "source1_entity_id": entity_id,
                "source": source, "country": record.country,
                "intersection_scheduler": index.config.intersection_scheduler,
                **dict(index.last_diagnostics),
                "eligible_before_final_cap": len(eligible),
                "cap_hit": int(len(candidates) == budget.total_per_source),
                "cap_truncated": int(len(eligible) > budget.total_per_source),
                "selected_candidates": len(candidates),
                "true_links_before_cap": sum(c.candidate_entity_id in truth[entity_id] for c in pool),
                "selected_exact_candidates": sum(any(getattr(c, f"{p}_score") > 0 for p in ("exact_name", "exact_address", "name_signature")) for c in candidates),
                "query_seconds": time.monotonic() - query_started,
            }
            diagnostic_rows.append(diagnostic)
            selected[entity_id].update(candidate.candidate_entity_id for candidate in candidates)
    metrics = summarize(
        records,
        truth,
        selected,
        sample_ids,
        sample_name,
        time.monotonic() - started,
    )
    for row in metrics:
        dimension, group = row["dimension"], row["group"]
        scope = [d for d in diagnostic_rows if (
            dimension == "overall" or
            dimension == "country" and d["country"] == group or
            dimension == "source" and d["source"] == group or
            dimension == "source_country" and f"{d['source']}|{d['country']}" == group
        )]
        row["query_seconds"] = sum(d["query_seconds"] for d in scope)
        row["average_query_seconds_per_s1"] = row["query_seconds"] / row["entity_count"]
        row["projected_1732544_s1_query_hours"] = row["average_query_seconds_per_s1"] * 1_732_544 / 3600
        row["recall_before_cap"] = sum(d["true_links_before_cap"] for d in scope) / row["true_link_count"] if row["true_link_count"] else 1.0
        row["source_query_count"] = len(scope)
        row["intersection_scheduler"] = scope[0]["intersection_scheduler"]
        row["average_intersection_probes_per_source_query"] = sum(d.get("intersection_queries", 0) for d in scope) / len(scope)
        row["p95_intersection_probes_per_source_query"] = float(np.quantile([d.get("intersection_queries", 0) for d in scope], .95))
        row["intersection_sql_seconds"] = sum(d.get("intersection_sql_seconds", 0.0) for d in scope)
        row["intersection_total_seconds"] = sum(d.get("intersection_total_seconds", 0.0) for d in scope)
        row["cap_hit_rate_per_source_query"] = sum(d["cap_hit"] for d in scope) / len(scope)
        row["cap_truncated_rate_per_source_query"] = sum(d["cap_truncated"] for d in scope) / len(scope)
        for key in ("exact_candidates", "token_candidates", "token_only_candidates", "selected_exact_candidates", "candidates_before_cap", "postings_looked_up", "posting_ids_returned", "intersection_queries", "intersection_overflows", "exact_overflow_passes", "intersection_pairs_considered", "intersection_pairs_skipped_rules", "intersection_pairs_skipped_anchor", "intersection_pairs_skipped_selectivity", "intersection_pairs_skipped_term_quota", "intersection_nonempty_successes", "intersection_unique_candidate_ids"):
            row[f"average_{key}_per_s1"] = sum(d.get(key, 0) for d in scope) / row["entity_count"]
    if diagnostics_out is not None:
        diagnostics_out.extend(diagnostic_rows)
    return metrics


def write_report(path: Path, build: pd.DataFrame, metrics: pd.DataFrame, experiment: str = "EXP002c") -> None:
    lines = [
        f"# {experiment} country-scoped postings benchmark",
        "",
        "The country-scoped postings indexes (schema 2) are built once and reused for all S1 query batches.",
        "The production path has no character n-gram fallback and no S2/S3 scan during querying.",
        "",
        "## Index build/reuse",
        "",
        "| Source | Reused | Rows | Seconds | Disk MiB | Peak RSS MiB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in build.itertuples(index=False):
        lines.append(
            f"| {row.source} | {bool(row.reused)} | {row.row_count:,} | {row.seconds:.1f} | "
            f"{row.size_mib:.1f} | {row.peak_rss_mb:.1f} |"
        )
    lines.extend(["", "## Candidate retrieval", "", "| Sample | Recall | Matched all-links | Mean candidates | p99 | Query seconds | Peak RSS MiB |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in metrics.query('dimension == "overall"').itertuples(index=False):
        lines.append(
            f"| {row.sample} | {row.positive_link_recall:.3%} | "
            f"{row.matched_all_true_links_retained_rate:.3%} | {row.average_candidates:.2f} | "
            f"{row.p99_candidates:.0f} | {row.query_seconds:.1f} | {row.peak_rss_mb:.1f} |"
        )
    lines.append("")
    lines.extend(["## Intersection scheduling", "",
                  "| Sample | Slice | Scheduler | Mean probes/source query | p95 probes | SQL seconds |",
                  "|---|---|---|---:|---:|---:|"])
    for row in metrics.itertuples(index=False):
        lines.append(f"| {row.sample} | {row.group} | {row.intersection_scheduler} | {row.average_intersection_probes_per_source_query:.2f} | {row.p95_intersection_probes_per_source_query:.1f} | {row.intersection_sql_seconds:.3f} |")
    lines.append("")
    lines.append("Peak RSS is the cumulative process high-water mark, including builds if performed in this process. Query timing excludes S1/GT loading and metric aggregation. The 5k batch may benefit from caches warmed by 1k. Cap-hit rates use S1/source queries as denominator; candidate contribution sets overlap; token-only is incremental to exact passes. Full diagnostics and slices are in the CSV tables.")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPOSITORY_ROOT / "data" / "raw")
    parser.add_argument("--index-dir", type=Path, default=REPOSITORY_ROOT / "results" / "artifacts" / "exp002c_source_index")
    parser.add_argument("--results-dir", type=Path, help="Fresh output directory (required for selective_v1)")
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs" / "exp002c_postings.json")
    parser.add_argument("--build-index", action="store_true", help="Build missing source indexes once; otherwise require complete indexes.")
    parser.add_argument("--rebuild-index", action="store_true", help="Explicitly replace complete source index files.")
    parser.add_argument("--build-only", action="store_true", help="Build/reuse indexes and exit before queries.")
    parser.add_argument("--sample-sizes", type=int, nargs="+", help="Defaults to 1k for selective_v1; legacy defaults to 1k/5k")
    parser.add_argument("--reviewed-1k", action="store_true", help="Run >1k selective_v1 only after its 1k results have been reviewed/authorized")
    parser.add_argument("--budget", default="postings")
    parser.add_argument("--sample-seed", type=int, default=2032)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    retrieval_config = SourceIndexConfig.from_config(config)
    selective = retrieval_config.intersection_scheduler == "selective_v1"
    experiment = "exp002e" if selective else "exp002c"
    if args.sample_sizes is None:
        args.sample_sizes = [1000] if selective else [1000, 5000]
    if selective:
        if args.build_index or args.rebuild_index or args.build_only:
            raise ValueError("EXP002e requires existing schema-2 indexes; build flags are prohibited")
        if args.sample_seed != 2032:
            raise ValueError("EXP002e comparison requires seed 2032")
        if max(args.sample_sizes) > 1000 and not args.reviewed_1k:
            raise ValueError("Review/authorize this variant's 1k result before using --reviewed-1k for 5k")
        if args.results_dir is None:
            raise ValueError("EXP002e requires an explicit fresh --results-dir")
        if args.results_dir.exists() and any(args.results_dir.iterdir()):
            raise FileExistsError("Choose a fresh results directory to preserve measured evidence")
    budget = load_budget(config, args.budget)
    data_root = args.data_root.absolute()
    index_dir = args.index_dir.absolute()
    results_dir = (args.results_dir or REPOSITORY_ROOT / "results/exp002c").absolute()
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    if any(size <= 0 or size > 5000 for size in args.sample_sizes):
        raise ValueError("EXP002c preflight is limited to 1–5000 S1; 20k is not authorized yet")
    if budget.char_name != 0:
        raise ValueError("EXP002c does not have a character pass")
    manifest = {
        "config": config, "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip(),
        "schema_version": INDEX_SCHEMA_VERSION, "sample_sizes": args.sample_sizes,
        "sample_seed": args.sample_seed, "budget": args.budget,
        "index_dir": str(index_dir), "build_only": args.build_only,
        "experiment": experiment, "resolved_source_config": asdict(retrieval_config),
        "intersection_scheduler": retrieval_config.intersection_scheduler,
    }
    (results_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    build_rows: list[dict[str, object]] = []
    for source in ("S2", "S3"):
        source_path = data_root / "train" / f"train_source{source[-1]}.tsv"
        index_path = _index_path(index_dir, source)
        if not index_path.exists() and not args.build_index:
            raise FileNotFoundError(
                f"Missing {index_path}. " + (
                    "EXP002e requires completed schema-2 indexes and never rebuilds them."
                    if selective else "Run once with --build-index; query batches will then reuse it."
                )
            )
        if selective:
            # Never call the builder, or scan/stat S2/S3 TSVs in this path.
            with SourceSideIndex(index_path, source, retrieval_config) as index:
                result = IndexBuildResult(source, index_path, True, index.row_count, 0.0, index_path.stat().st_size)
        else:
            result = build_or_open_source_index(
                source_path, source, index_dir, rebuild=args.rebuild_index,
                chunksize=int(config["index_chunk_size"]),
            )
        build_rows.append(
            {
                "source": result.source,
                "reused": result.reused,
                "row_count": result.row_count,
                "seconds": result.seconds,
                "size_bytes": result.size_bytes,
                "size_mib": result.size_bytes / 1024**2,
                "peak_rss_mb": peak_rss_mb(),
            }
        )
        pd.DataFrame(build_rows).to_csv(tables_dir / f"{experiment}_index_build.csv", index=False)
    build = pd.DataFrame(build_rows)
    if args.build_only:
        return 0

    subset_path = REPOSITORY_ROOT / str(config["subset_ids_path"])
    subset_ids = frozenset(pd.read_csv(subset_path)["source1_entity_id"])
    # IDs are independent of labels; use the same nested hash samples as EXP002b.
    largest_ids = frozenset(sorted(subset_ids, key=lambda entity_id: (stable_entity_key(entity_id, args.sample_seed), entity_id))[:max(args.sample_sizes)])
    if max(args.sample_sizes) > len(subset_ids):
        raise ValueError("Requested sample exceeds configured subset")
    records = load_selected_source_records(data_root / "train" / "train_source1.tsv", "S1", largest_ids)
    truth = load_selected_ground_truth(data_root / "train" / "train_ground_truth.tsv", largest_ids)

    with SourceSideIndex(_index_path(index_dir, "S2"), "S2", retrieval_config) as s2, SourceSideIndex(_index_path(index_dir, "S3"), "S3", retrieval_config) as s3:
        indexes = {"S2": s2, "S3": s3}
        metric_rows: list[dict[str, object]] = []
        diagnostic_rows: list[dict[str, object]] = []
        id_rows: list[dict[str, object]] = []
        for size in args.sample_sizes:
            ids = benchmark_ids(records, size, args.sample_seed)
            id_rows.extend({"sample": f"s1_{size}", "source1_entity_id": entity_id} for entity_id in ids)
            print(f"Querying {size:,} S1 entities against reusable indexes ...", flush=True)
            metric_rows.extend(run_benchmark(records, truth, indexes, budget, ids, f"s1_{size}", diagnostic_rows))
            pd.DataFrame(metric_rows).to_csv(tables_dir / f"{experiment}_blocker_benchmark.csv", index=False)
            pd.DataFrame(diagnostic_rows).fillna(0).to_csv(tables_dir / f"{experiment}_query_diagnostics.csv", index=False)
            pd.DataFrame(id_rows).to_csv(tables_dir / f"{experiment}_sample_ids.csv", index=False)
    metrics = pd.DataFrame(metric_rows)
    write_report(results_dir / f"{experiment}_blocker_benchmark.md", build, metrics, experiment.upper())
    print(metrics.query('dimension == "overall"')[["sample", "positive_link_recall", "average_candidates", "query_seconds", "peak_rss_mb"]].to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
