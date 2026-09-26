#!/usr/bin/env python3
"""Train EXP003 on the frozen 20k TRAIN subset only; never construct indexes."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from business_entity_resolution.exp003_training import run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=ROOT / "data/raw/train")
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--rule-only", action="store_true", help="Generate tuned compact14/18 rule bundles without learned training")
    args = parser.parse_args()
    run_training(args.train_dir, args.index_dir, args.results_dir, rule_only=args.rule_only)


if __name__ == "__main__":
    main()
