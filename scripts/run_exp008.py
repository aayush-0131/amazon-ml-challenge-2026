#!/usr/bin/env python3
"""Build EXP008 TRAIN sidecars; run TUNE ablations, then one frozen EVALUATION oracle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from business_entity_resolution.exp008 import Policy, build_index, run_partition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("build-index", "tune", "evaluate"):
        p = sub.add_parser(command)
        p.add_argument("--train-dir", type=Path, default=ROOT / "data/raw/train")
        p.add_argument("--secondary-index-dir", type=Path, default=ROOT / "results/exp008_secondary_index")
        p.add_argument("--policy", type=Path, default=ROOT / "configs/exp008_blocker.json")
        if command == "build-index":
            p.add_argument("--source", choices=("S2", "S3"), required=True)
            p.add_argument("--chunksize", type=int, default=10000)
        else:
            p.add_argument("--baseline-index-dir", type=Path, required=True)
            p.add_argument("--original-dir", type=Path, required=True)
            p.add_argument("--sample-dir", type=Path, required=True)
            p.add_argument("--output-dir", type=Path, required=True)
            if command == "evaluate":
                p.add_argument("--frozen-policy", type=Path, required=True)
    args = parser.parse_args()
    policy = Policy(**json.loads(args.policy.read_text()))
    if args.command == "build-index":
        result = build_index(args.train_dir / f"train_source{args.source[-1]}.tsv",
                             args.source, args.secondary_index_dir, policy, chunksize=args.chunksize)
    else:
        result = run_partition(args.train_dir, args.original_dir, args.sample_dir,
                               args.baseline_index_dir, args.secondary_index_dir, args.output_dir,
                               partition="tune" if args.command == "tune" else "evaluation",
                               policy=policy, frozen_path=getattr(args, "frozen_policy", None))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
