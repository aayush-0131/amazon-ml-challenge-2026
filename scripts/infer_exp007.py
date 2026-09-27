#!/usr/bin/env python3
"""Run, merge, audit, or benchmark frozen EXP007 TEST inference shards."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from business_entity_resolution.exp007_inference import (audit_output, benchmark, infer_shard,
                                                           merge_shards)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--tune-dir", type=Path, required=True)
    run.add_argument("--test-dir", type=Path, default=ROOT / "data/raw/test")
    run.add_argument("--index-dir", type=Path, required=True)
    run.add_argument("--shard-dir", type=Path, required=True)
    run.add_argument("--shard", type=int, required=True)
    run.add_argument("--shards", type=int, default=4)
    run.add_argument("--checkpoint-every", type=int, default=100)
    size = run.add_mutually_exclusive_group()
    size.add_argument("--smoke-limit", type=int, default=None)
    size.add_argument("--allow-full-test", action="store_true")
    merge = sub.add_parser("merge")
    merge.add_argument("--shard-dir", type=Path, required=True)
    merge.add_argument("--test-dir", type=Path, default=ROOT / "data/raw/test")
    merge.add_argument("--index-dir", type=Path, required=True)
    merge.add_argument("--output-dir", type=Path, required=True)
    merge.add_argument("--shards", type=int, default=4)
    merge.add_argument("--allow-full-test", action="store_true")
    audit = sub.add_parser("audit")
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.add_argument("--test-dir", type=Path, default=ROOT / "data/raw/test")
    bench = sub.add_parser("benchmark")
    bench.add_argument("--shard-dir", type=Path, required=True)
    bench.add_argument("--test-dir", type=Path, default=ROOT / "data/raw/test")
    bench.add_argument("--index-dir", type=Path, required=True)
    bench.add_argument("--shards", type=int, default=4)
    args = parser.parse_args()
    if args.command == "run":
        result = infer_shard(args.tune_dir, args.test_dir, args.index_dir, args.shard_dir,
            shard=args.shard, shards=args.shards,
            smoke_limit=None if args.allow_full_test else (100 if args.smoke_limit is None else args.smoke_limit),
            allow_full_test=args.allow_full_test, checkpoint_every=args.checkpoint_every)
    elif args.command == "merge":
        result = merge_shards(args.shard_dir, args.test_dir, args.index_dir, args.output_dir,
                              shards=args.shards, allow_full_test=args.allow_full_test)
    elif args.command == "audit":
        result = audit_output(args.output_dir, args.test_dir)
    else:
        result = benchmark(args.shard_dir, args.test_dir, args.index_dir, shards=args.shards)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
