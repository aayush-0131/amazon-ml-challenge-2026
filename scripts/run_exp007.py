#!/usr/bin/env python3
"""Run staged EXP007 TRAIN audits, enhanced HGB fit and frozen evaluation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from business_entity_resolution import exp007


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    commands = ("preflight", "ownership-audit", "blocker-oracle", "generate", "merge", "fit", "tune", "evaluate")
    for command in commands:
        p = sub.add_parser(command)
        p.add_argument("--train-dir", type=Path, default=ROOT / "data/raw/train")
        if command != "ownership-audit":
            p.add_argument("--index-dir", type=Path, required=True)
            p.add_argument("--original-dir", type=Path, required=True)
            p.add_argument("--sample-dir", type=Path, required=True)
        if command in ("blocker-oracle", "generate", "merge"):
            p.add_argument("--partition", choices=("fit", "tune", "evaluation") if command != "blocker-oracle" else ("tune", "evaluation"), required=True)
        if command in ("generate", "merge"):
            p.add_argument("--shard-dir", type=Path, required=True)
            p.add_argument("--shards", type=int, default=4)
        if command == "generate":
            p.add_argument("--shard", type=int, required=True)
            p.add_argument("--checkpoint-every", type=int, default=100)
            p.add_argument("--tune-dir", type=Path)
        if command == "fit":
            p.add_argument("--fit-pairs-dir", type=Path, required=True)
        if command == "tune":
            p.add_argument("--tune-pairs-dir", type=Path, required=True)
            p.add_argument("--fit-dir", type=Path, required=True)
            p.add_argument("--ownership-dir", type=Path, required=True)
        if command == "evaluate":
            for arg in ("evaluation-pairs-dir", "fit-dir", "tune-dir", "oracle-dir", "ownership-dir"):
                p.add_argument("--" + arg, type=Path, required=True)
        if command in ("ownership-audit", "blocker-oracle", "merge", "fit", "tune", "evaluate"):
            p.add_argument("--output-dir", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    args["data_dir"] = args.pop("train_dir")
    function = {"preflight": exp007.preflight, "ownership-audit": exp007.audit_ownership,
                "blocker-oracle": exp007.blocker_oracle, "generate": exp007.generate,
                "merge": exp007.merge, "fit": exp007.fit, "tune": exp007.tune,
                "evaluate": exp007.evaluate}[command]
    print(json.dumps(function(**args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
