#!/usr/bin/env python3
"""Build exact indexes, select a TRAIN policy, or stream EXP005 TEST output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from business_entity_resolution.exp005_quick import (  # noqa: E402
    MAX_EXACT_HITS, POLICIES, build_index, run_development, run_inference, validate_output,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-index")
    build.add_argument("--data-dir", type=Path, required=True)
    build.add_argument("--split", choices=("train", "test"), required=True)
    build.add_argument("--index-dir", type=Path, required=True)

    develop = sub.add_parser("develop")
    develop.add_argument("--train-dir", type=Path, default=ROOT / "data/raw/train")
    develop.add_argument("--subset", type=Path, default=ROOT / "results/tables/exp001_subset_ids.csv")
    develop.add_argument("--index-dir", type=Path, required=True)
    develop.add_argument("--output-dir", type=Path, required=True)

    infer = sub.add_parser("infer-test")
    infer.add_argument("--test-dir", type=Path, default=ROOT / "data/raw/test")
    infer.add_argument("--index-dir", type=Path, required=True)
    infer.add_argument("--policy-file", type=Path, required=True)
    infer.add_argument("--output-dir", type=Path, required=True)
    scope = infer.add_mutually_exclusive_group(required=True)
    scope.add_argument("--smoke-limit", type=int)
    scope.add_argument("--allow-full-test", action="store_true")

    check = sub.add_parser("validate-output")
    check.add_argument("--source1", type=Path, required=True)
    check.add_argument("--output-dir", type=Path, required=True)
    check.add_argument("--smoke-limit", type=int)
    args = parser.parse_args()
    if args.command == "build-index":
        for source in ("S2", "S3"):
            build_index(args.data_dir / f"{args.split}_source{source[-1]}.tsv", source, args.index_dir)
    elif args.command == "develop":
        print(json.dumps(run_development(args.train_dir, args.subset, args.index_dir, args.output_dir), indent=2))
    elif args.command == "infer-test":
        policy = json.loads(args.policy_file.read_text())
        if (policy.get("policy") not in POLICIES or policy.get("max_exact_hits") != MAX_EXACT_HITS
                or policy.get("subset_sha256") !=
                "84fa2544c2905fa480ffc0e57a2912143304acb6653a5f6c09a3eefa2a66979c"):
            raise ValueError("Invalid or incompatible frozen policy file")
        result = run_inference(args.test_dir / "test_source1.tsv", args.test_dir,
                               args.index_dir, args.output_dir, split="test",
                               policy=policy["policy"], smoke_limit=args.smoke_limit,
                               allow_full_test=args.allow_full_test)
        print(json.dumps(result, indent=2))
    else:
        print(f"Validated {validate_output(args.source1, args.output_dir, limit=args.smoke_limit):,} rows")


if __name__ == "__main__":
    main()
