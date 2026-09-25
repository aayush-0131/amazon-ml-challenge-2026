#!/usr/bin/env python3
"""Explain EXP002c true-link misses using completed schema-2 indexes only."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from business_entity_resolution.indexed_blocking import (  # noqa: E402
    INDEX_SCHEMA_VERSION, SourceIndexConfig, SourceSideIndex, _index_path,
)
from business_entity_resolution.miss_analysis import (  # noqa: E402
    CAPS, CAP_MODES, DF_THRESHOLDS, analyze_queries, summarize_analysis, write_report,
)
from business_entity_resolution.multipass import CandidateBudget  # noqa: E402
from business_entity_resolution.sampling import load_selected_ground_truth, load_selected_source_records  # noqa: E402
from business_entity_resolution.split import stable_entity_key  # noqa: E402


def sample_ids(subset_path: Path, size: int, seed: int) -> list[str]:
    """Exactly EXP002c's (stable_entity_key(seed), entity_id) ordering."""
    ids = pd.read_csv(subset_path, dtype=str, keep_default_na=False)["source1_entity_id"].tolist()
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate S1 IDs in configured subset")
    if not 0 < size <= min(5000, len(ids)):
        raise ValueError("Sample size must be positive and <=5000 and fit the configured subset")
    if seed != 2032:
        raise ValueError("EXP002d must use EXP002c seed 2032")
    return sorted(ids, key=lambda eid: (stable_entity_key(eid, seed), eid))[:size]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/exp002c_postings.json")
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=REPOSITORY_ROOT / "data/raw")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--sample-seed", type=int, default=2032)
    parser.add_argument("--budget", default="postings")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    source_config = SourceIndexConfig.from_config(config)
    choices = [value for value in config["blocker_budgets"] if value["name"] == args.budget]
    if len(choices) != 1:
        raise ValueError("Select a unique budget from the configured blocker_budgets")
    budget = CandidateBudget(**choices[0])
    subset_path = REPOSITORY_ROOT / config["subset_ids_path"]
    selected = sample_ids(subset_path, args.sample_size, args.sample_seed)
    output = args.results_dir.absolute()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Results directory is not empty; choose a fresh directory to preserve prior evidence")
    manifest = {
        "experiment": "EXP002d", "baseline_commit": "5d1892c7dfb835f852363da3c9972e115bca2c09",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip(),
        "config": config, "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "subset_ids_sha256": hashlib.sha256(subset_path.read_bytes()).hexdigest(),
        "sample_size": args.sample_size, "sample_seed": args.sample_seed,
        "sample_ids": selected,
        "sample_ids_sha256": hashlib.sha256(("\n".join(selected) + "\n").encode()).hexdigest(),
        "index_schema": INDEX_SCHEMA_VERSION, "indexes": {}, "budget": args.budget,
        "df_thresholds": DF_THRESHOLDS, "caps": CAPS, "cap_modes": CAP_MODES,
        "python": sys.version.split()[0], "sqlite": sqlite3.sqlite_version,
        "pandas": pd.__version__,
    }
    with ExitStack() as stack:
        indexes = {}
        for source in ("S2", "S3"):
            path = _index_path(args.index_dir.absolute(), source)
            if not path.is_file():
                raise FileNotFoundError(f"Completed schema-2 index required: {path}; analysis never builds indexes")
            index = stack.enter_context(SourceSideIndex(path, source, source_config))
            indexes[source] = index
            manifest["indexes"][source] = {
                "file": path.name, "size_bytes": path.stat().st_size,
                "metadata": dict(index.connection.execute("SELECT key,value FROM metadata")),
            }
        # Only S1 and GT are read from raw files, once each. S2/S3 text comes from
        # indexed entity_id lookups on these read-only completed connections.
        source1_path = args.data_root / "train/train_source1.tsv"
        truth_path = args.data_root / "train/train_ground_truth.tsv"
        records = load_selected_source_records(source1_path, "S1", frozenset(selected))
        truth = load_selected_ground_truth(truth_path, selected)
        manifest["inputs"] = {name: {"size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                              for name, path in (("train_source1", source1_path), ("train_ground_truth", truth_path))}
        print(f"Analyzing {len(selected):,} fixed S1 IDs; schema-2 indexes opened read-only.", flush=True)
        audit, queries = analyze_queries(records, truth, indexes, budget)
    tables = summarize_analysis(audit, queries)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False, lineterminator="\n", float_format="%.12g")
    (output / "exp002c_miss_analysis.md").write_text(write_report(tables), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(tables["exp002c_miss_summary"].to_string(index=False), flush=True)
    print(f"Saved analysis to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
