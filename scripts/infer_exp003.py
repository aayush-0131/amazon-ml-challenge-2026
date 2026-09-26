#!/usr/bin/env python3
"""Run one of four resumable TEST shards, or merge completed shards. No builders."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from business_entity_resolution.exp003_inference import infer_shard, merge_shards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--model", choices=("learned", "rule"), required=True)
    run.add_argument("--index-dir", type=Path, required=True)
    run.add_argument("--test-dir", type=Path, default=ROOT / "data/raw/test")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--shard", type=int, choices=range(4), required=True)
    size = run.add_mutually_exclusive_group()
    size.add_argument("--smoke-limit", type=int, default=None)
    size.add_argument("--allow-full-test", action="store_true", help="Explicitly process all TEST S1 instead of the default 100-S1 smoke")
    merge = sub.add_parser("merge")
    merge.add_argument("--shard-dir", type=Path, required=True)
    merge.add_argument("--test-dir", type=Path, default=ROOT / "data/raw/test")
    merge.add_argument("--output-dir", type=Path, required=True)
    merge.add_argument("--allow-full-test", action="store_true")
    args = parser.parse_args()
    if args.command == "run":
        infer_shard(args.bundle, args.model, args.test_dir, args.index_dir, args.output_dir,
                    shard=args.shard, smoke_limit=None if args.allow_full_test else (100 if args.smoke_limit is None else args.smoke_limit),
                    allow_full_test=args.allow_full_test)
    else:
        merge_shards(args.shard_dir, args.test_dir, args.output_dir, allow_full_test=args.allow_full_test)


if __name__ == "__main__":
    main()
