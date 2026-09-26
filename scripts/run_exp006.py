#!/usr/bin/env python3
"""Run isolated EXP006 TRAIN sampling, sharded pairs, fixed HGB, TUNE and EVALUATION."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from business_entity_resolution import exp006


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "sample", "generate", "merge", "fit", "tune", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--train-dir", type=Path, default=ROOT / "data/raw/train", required=name == "merge")
        if name != "sample":
            p.add_argument("--index-dir", type=Path, required=True)
        if name in {"preflight", "fit", "tune", "evaluate"}:
            p.add_argument("--original-dir", type=Path, required=True)
        if name in {"sample", "generate", "merge", "fit", "tune"}:
            p.add_argument("--sample-dir", type=Path, required=True)
        if name in {"generate", "merge"}:
            p.add_argument("--shard-dir", type=Path, required=True)
            p.add_argument("--shards", type=int, default=4)
        if name == "generate":
            p.add_argument("--shard", type=int, required=True)
            p.add_argument("--checkpoint-every", type=int, default=100)
        if name in {"fit", "tune"}:
            p.add_argument("--merged-dir", type=Path, required=True)
        if name in {"tune", "evaluate"}:
            p.add_argument("--fit-dir", type=Path, required=True)
        if name == "evaluate":
            p.add_argument("--tune-dir", type=Path, required=True)
        if name in {"merge", "fit", "tune", "evaluate"}:
            p.add_argument("--output-dir", type=Path, required=True)
        if name == "sample":
            p.add_argument("--size", type=int, default=50000)
            p.add_argument("--seed", type=int, default=exp006.SEED)
    a = parser.parse_args()
    d = vars(a)
    command = d.pop("command")
    d["data_dir"] = d.pop("train_dir")
    if command == "preflight":
        result = exp006.preflight(**d)
    elif command == "sample":
        d["output_dir"] = d.pop("sample_dir")
        result = exp006.sample(**d)
    elif command == "generate":
        result = exp006.generate(**d)
    elif command == "merge":
        result = exp006.merge(**d)
    elif command == "fit":
        result = exp006.fit(**d)
    elif command == "tune":
        result = exp006.tune(**d)
    else:
        result = exp006.evaluate(**d)
    import json
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
