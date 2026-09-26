"""TRAIN-only EXP003 orchestration; disk-backed pairs, entity-isolated fitting."""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
import json
from pathlib import Path
import resource
import shutil
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .evaluation import evaluate_predictions
from .features import FEATURE_NAMES
from .modeling import (NegativeSamplingConfig, TrainingPairSampler, make_logistic_model,
                       make_hist_gradient_boosting_model, select_global_threshold)
from .reranker import (ROOT, SUBSET_HASH, RULE_COLUMN, accepted_ids, candidate_pool,
                       feature_matrix, fingerprint, frozen_config, git_commit, index_identity,
                       make_bundle, save_bundle, scores_for, sha256, code_versions, write_json_atomic)
from .reranker import open_indexes
from .sampling import load_selected_ground_truth, load_selected_source_records
from .split import stratified_entity_split

FIT_SEED, MODEL_SEED, SAMPLE_SEED = 2029, 2030, 2031
RULE_THRESHOLDS = [80, 84, 86, 88, 90, 92, 94, 96, 98, 99, 100, 101]
PROB_THRESHOLDS = [.1, .2, .3, .4, .5, .6, .7, .75, .8, .85, .9, .93, .95, .97, .98, .99, .995, 1.000001]
PAIR_FILES = tuple(f"{part}.{suffix}" for part in (
    "fit", "tune", "evaluation", "rule14_tune", "rule14_evaluation") for suffix in ("f32", "tsv")) + ("pool_counts.csv", "sampling_counts.json")


def seal_pair_cache(directory, identity):
    directory = Path(directory)
    write_json_atomic(directory / "complete.json", {
        "identity": identity, "files": {name: sha256(directory / name) for name in PAIR_FILES}})


def validate_pair_cache(directory, identity):
    directory = Path(directory)
    marker = directory / "complete.json"
    if not marker.exists():
        raise ValueError("Incomplete pair cache: preserve it and use a fresh results directory")
    saved = json.loads(marker.read_text())
    if saved.get("identity") != identity or set(saved.get("files", {})) != set(PAIR_FILES):
        raise ValueError("Incompatible pair cache identity/artifact inventory")
    for name in PAIR_FILES:
        if not (directory / name).is_file() or sha256(directory / name) != saved["files"][name]:
            raise ValueError(f"Corrupt pair cache artifact: {name}")


def partitions(table, records, truth):
    if table.source1_entity_id.duplicated().any() or set(table.source1_entity_id) != set(records):
        raise ValueError("Subset IDs duplicated or not equal to loaded S1")
    if set(table.partition) != {"tuning", "evaluation"}:
        raise ValueError("Expected authoritative EXP001 tuning/evaluation partitions")
    evaluation = set(table.loc[table.partition.eq("evaluation"), "source1_entity_id"])
    development = set(records) - evaluation
    split = stratified_entity_split(
        [{"entity_id": eid, "country": records[eid].country} for eid in sorted(development)],
        {eid: truth[eid] for eid in sorted(development)}, validation_fraction=.25, seed=FIT_SEED)
    parts = {**dict.fromkeys(split.train_ids, "fit"), **dict.fromkeys(split.validation_ids, "tune"),
             **dict.fromkeys(evaluation, "evaluation")}
    if any(not [eid for eid, p in parts.items() if p == name] for name in ("fit", "tune", "evaluation")):
        raise ValueError("Every partition must be nonempty")
    return parts


class PairWriter:
    """Append float32 rows and small metadata; never retain all pairs as objects."""
    def __init__(self, path):
        self.path = Path(path)
        self.x = self.path.with_suffix(".f32").open("wb")
        self.meta = self.path.with_suffix(".tsv").open("w", newline="")
        self.writer = csv.writer(self.meta, delimiter="\t", lineterminator="\n")
        self.writer.writerow(["source1_entity_id", "candidate_entity_id", "label"])
        self.count = 0

    def add(self, eid, cids, x, labels):
        x = np.asarray(x, dtype=np.float32).reshape(-1, len(FEATURE_NAMES))
        if len(x) != len(cids) or len(labels) != len(cids):
            raise ValueError("Pair artifact shape mismatch")
        x.tofile(self.x)
        self.writer.writerows((eid, cid, int(label)) for cid, label in zip(cids, labels))
        self.count += len(cids)

    def close(self):
        self.x.close()
        self.meta.close()


def pair_groups(path):
    """Yield one S1's rows at a time over a read-only memory map."""
    import itertools
    path = Path(path)
    size = path.with_suffix(".f32").stat().st_size
    if size % (4 * len(FEATURE_NAMES)):
        raise ValueError("Incomplete feature artifact")
    matrix = np.memmap(path.with_suffix(".f32"), mode="r", dtype=np.float32,
                       shape=(size // (4 * len(FEATURE_NAMES)), len(FEATURE_NAMES))) if size else np.empty((0, len(FEATURE_NAMES)))
    offset = 0
    with path.with_suffix(".tsv").open() as handle:
        for eid, group in itertools.groupby(csv.DictReader(handle, delimiter="\t"), key=lambda r: r["source1_entity_id"]):
            rows = list(group)
            end = offset + len(rows)
            if end > len(matrix):
                raise ValueError("Metadata/feature count mismatch")
            yield eid, [r["candidate_entity_id"] for r in rows], matrix[offset:end], np.array([int(r["label"]) for r in rows], dtype=np.int8)
            offset = end
    if offset != len(matrix):
        raise ValueError("Metadata/feature count mismatch")


def sample_entity(eid, cids, matrix, labels):
    sampler = TrainingPairSampler(NegativeSamplingConfig(seed=SAMPLE_SEED))
    for cid, row, label in zip(cids, matrix, labels):
        hardness = .7 * row[RULE_COLUMN] + .3 * min(row[FEATURE_NAMES.index("best_retrieval_score")], 100)
        sampler.add(eid, cid, row, label=int(label), hardness=float(hardness))
    return sampler.finalize()


def generate_pairs(directory, records, truth, parts, index_dir, data_dir):
    directory = Path(directory)
    directory.mkdir()  # Partial artifacts are preserved, never silently overwritten.
    counters = Counter()
    slices = []
    configs = {n: frozen_config(n) for n in (18, 14)}
    with ExitStack() as stack:
        indexes = {n: stack.enter_context(open_indexes(index_dir, data_dir, "train", configs[n])) for n in (18, 14)}
        writers = {name: PairWriter(directory / name) for name in ("fit", "tune", "evaluation", "rule14_tune", "rule14_evaluation")}
        for writer in writers.values():
            stack.callback(writer.close)
        for position, eid in enumerate(sorted(records), 1):
            part, record = parts[eid], records[eid]
            for probes in ((18,) if part == "fit" else (18, 14)):
                candidates, raw_ids = candidate_pool(record, indexes[probes], configs[probes], rule_fallback=probes == 14)
                cids = [c.candidate_entity_id for c in candidates]
                matrix = feature_matrix(record, candidates)
                labels = [int(cid in truth[eid]) for cid in cids]
                sampled = []
                if part == "fit":
                    sampled, counts = sample_entity(eid, cids, matrix, labels)
                    counters.update(counts)
                    writers[part].add(eid, [p.candidate_entity_id for p in sampled], [p.features for p in sampled], [p.label for p in sampled])
                else:
                    writers[("rule14_" if probes == 14 else "") + part].add(eid, cids, matrix, labels)
                for source in ("S2", "S3"):
                    expected = {cid for cid in truth[eid] if cid.startswith(source + "-")}
                    eligible = {cid for cid in cids if cid.startswith(source + "-")}
                    raw = {cid for cid in raw_ids if cid.startswith(source + "-")}
                    slices.append({"entity_id": eid, "partition": part, "country": record.country,
                        "source": source, "probes": probes, "true_links": len(expected),
                        "raw_pairs": len(raw), "raw_true_links": len(raw & expected),
                        "eligible_pairs": len(eligible), "eligible_positives": len(eligible & expected),
                        "sampled_pairs": sum(p.candidate_entity_id.startswith(source + "-") for p in sampled),
                        "sampled_positives": sum(p.label for p in sampled if p.candidate_entity_id.startswith(source + "-"))})
            if position % 250 == 0:
                print(f"Indexed pairs: {position}/{len(records)} S1", flush=True)
    pd.DataFrame(slices).to_csv(directory / "pool_counts.csv", index=False)
    (directory / "sampling_counts.json").write_text(json.dumps(dict(counters), indent=2) + "\n")


def metric_slices(truth, predictions, retained, records):
    ids = sorted(truth)
    specs = [("overall", "ALL", ids, None)]
    countries = sorted({records[e].country for e in ids})
    specs += [("country", c, [e for e in ids if records[e].country == c], None) for c in countries]
    for s in ("S2", "S3"):
        specs.append(("source", s, ids, s))
        specs += [("source_country", f"{s}|{c}", [e for e in ids if records[e].country == c], s) for c in countries]
    for label, lo, hi in (("singleton", 0, 0), ("1-2", 1, 2), ("3-4", 3, 4), ("5+", 5, float("inf"))):
        specs.append(("multiplicity", label, [e for e in ids if lo <= len(truth[e]) <= hi], None))
    rows = []
    for dimension, group, selected, source in specs:
        if not selected:
            continue
        def filtered(mapping):
            return {e: {cid for cid in mapping.get(e, ()) if source is None or cid.startswith(source + "-")} for e in selected}
        gt, pred, available = filtered(truth), filtered(predictions), filtered(retained)
        result = evaluate_predictions(gt, pred)
        rows.append({"dimension": dimension, "group": group, **result.to_dict(),
                     "singleton_accuracy": result.correct_singletons / result.singleton_count if result.singleton_count else None,
                     "average_predicted_matches": sum(map(len, pred.values())) / len(selected),
                     "blocking_fn": sum(len(gt[e] - available[e]) for e in selected),
                     "matcher_fn": sum(len((gt[e] & available[e]) - pred[e]) for e in selected)})
    return rows


def tune_one(path, model_type, model, truth):
    score_map = {}
    for eid, cids, matrix, _ in pair_groups(path):
        score_map[eid] = dict(zip(cids, map(float, scores_for(model_type, model, matrix))))
    return select_global_threshold(truth, score_map, RULE_THRESHOLDS if model_type == "rule" else PROB_THRESHOLDS)


def evaluate_one(path, model_type, model, threshold, truth, records):
    predictions = {eid: set() for eid in truth}
    retained = {}
    for eid, cids, matrix, labels in pair_groups(path):
        predictions[eid] = set(accepted_ids(cids, scores_for(model_type, model, matrix), threshold))
        retained[eid] = {cid for cid, label in zip(cids, labels) if label}
    return metric_slices(truth, predictions, retained, records)


def run_training(data_dir, index_dir, results_dir, *, rule_only=False):
    results_dir, data_dir = Path(results_dir), Path(data_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if (results_dir / "summary.json").exists():
        raise FileExistsError("Completed training results exist; use a fresh results directory")
    subset = ROOT / "results/tables/exp001_subset_ids.csv"
    if sha256(subset) != SUBSET_HASH:
        raise ValueError("Only the authoritative fixed EXP001 20k subset is supported")
    table = pd.read_csv(subset)
    if len(table) != 20000:
        raise ValueError("EXP003 is limited to the fixed 20k subset")
    started = time.monotonic()
    records = load_selected_source_records(data_dir / "train_source1.tsv", "S1", set(table.source1_entity_id), chunksize=10000)
    truth = load_selected_ground_truth(data_dir / "train_ground_truth.tsv", records)
    for row in table.itertuples():
        if records[row.source1_entity_id].country != row.country or len(truth[row.source1_entity_id]) != row.match_count:
            raise ValueError("TRAIN data disagrees with the authoritative EXP001 subset metadata")
    parts = partitions(table, records, truth)
    pd.DataFrame([{"source1_entity_id": eid, "partition": parts[eid]} for eid in sorted(parts)]).to_csv(results_dir / "partitions.csv", index=False)
    identity = {"subset_sha256": SUBSET_HASH, "code": code_versions(), "fit_seed": FIT_SEED,
                "indexes": index_identity(index_dir), "inputs": {p.name: fingerprint(p) for p in [data_dir / "train_source1.tsv", data_dir / "train_ground_truth.tsv"]}}
    pairs = results_dir / "pairs"
    if pairs.exists():
        validate_pair_cache(pairs, identity)
    else:
        generate_pairs(pairs, records, truth, parts, index_dir, data_dir)
        seal_pair_cache(pairs, identity)
    retrieval_seconds = time.monotonic() - started
    pool_counts = pd.read_csv(pairs / "pool_counts.csv")
    count_columns = ["true_links", "raw_pairs", "raw_true_links", "eligible_pairs", "eligible_positives", "sampled_pairs", "sampled_positives"]
    pool_counts.groupby(["probes", "partition", "country", "source"], sort=True)[count_columns].sum().to_csv(results_dir / "pair_counts_by_slice.csv")
    sampling_counts = json.loads((pairs / "sampling_counts.json").read_text())
    negatives = sampling_counts["hard_negative_count"] + sampling_counts["easy_negative_count"]
    sampling_counts["positive_to_negative_ratio"] = sampling_counts["positive_count"] / negatives if negatives else None
    tune_truth = {e: truth[e] for e in sorted(parts) if parts[e] == "tune"}
    eval_truth = {e: truth[e] for e in sorted(parts) if parts[e] == "evaluation"}
    summaries, grids, timings, failures = [], [], [], {}
    models = {"rule14": None, "rule18": None}
    thresholds, tune_metrics = {}, {}

    def assess(name, model):
        stage = time.monotonic()
        kind = "rule" if name.startswith("rule") else name
        prefix = "rule14_" if name == "rule14" else ""
        threshold, metric, grid = tune_one(pairs / (prefix + "tune"), kind, model, tune_truth)
        thresholds[name], tune_metrics[name] = threshold, metric.macro_fbeta
        grids.extend({"model": name, **row} for row in grid)
        # Freeze threshold/bundle before reading evaluation scores.
        bundle = make_bundle(model, kind, threshold, 14 if name == "rule14" else 18, MODEL_SEED,
                             tune_macro_fbeta=metric.macro_fbeta, subset_sha256=SUBSET_HASH,
                             partition_counts=dict(Counter(parts.values())), fit_seed=FIT_SEED)
        save_bundle(bundle, results_dir / f"{name}.joblib")
        shutil.copyfile(ROOT / "docs/MODEL_LICENSE.txt", results_dir / "MODEL_LICENSE.txt")
        if name == "rule14":
            shutil.copyfile(results_dir / "rule14.joblib", results_dir / "rule.joblib")
            shutil.copyfile(results_dir / "rule14.json", results_dir / "rule.json")
        rows = evaluate_one(pairs / (prefix + "evaluation"), kind, model, threshold, eval_truth, records)
        summaries.extend({"model": name, "threshold": threshold, **row} for row in rows)
        timings.append({"stage": name, "tune_evaluate_seconds": time.monotonic() - stage})
        pd.DataFrame(summaries).to_csv(results_dir / "evaluation.csv", index=False)
        pd.DataFrame(grids).to_csv(results_dir / "thresholds.csv", index=False)
        print(f"{name}: TUNE F0.5={metric.macro_fbeta:.6f}, threshold={threshold}", flush=True)

    # Always produce usable tuned fallback bundles before attempting model fits.
    for name in models:
        assess(name, None)
    if not rule_only:
        n = (pairs / "fit.f32").stat().st_size // (4 * len(FEATURE_NAMES))
        x = np.memmap(pairs / "fit.f32", mode="r", dtype=np.float32, shape=(n, len(FEATURE_NAMES)))
        y = np.asarray(pd.read_csv(pairs / "fit.tsv", sep="\t")["label"], dtype=np.int8)
        for name, factory in (("logistic", make_logistic_model), ("hist_gradient_boosting", make_hist_gradient_boosting_model)):
            try:
                stage = time.monotonic()
                model = factory(MODEL_SEED)
                # No random candidate-pair validation split inside HGB.
                if name == "hist_gradient_boosting":
                    model.set_params(early_stopping=False)
                with threadpool_limits(limits=1):
                    model.fit(x, y)
                    assess(name, model)
                models[name] = model
                timings.append({"stage": name + "_fit_and_assess", "seconds": time.monotonic() - stage})
            except Exception as exc:
                failures[name] = repr(exc)
                print(f"{name} failed; rule fallback is preserved: {exc}", flush=True)
    learned = [name for name in models if not name.startswith("rule")]
    selected = max(learned, key=lambda name: (tune_metrics[name], name)) if learned else None
    if selected:
        # Selection uses TUNE only, never the evaluation table.
        shutil.copyfile(results_dir / f"{selected}.joblib", results_dir / "learned.joblib")
        shutil.copyfile(results_dir / f"{selected}.json", results_dir / "learned.json")
    shutil.copyfile(results_dir / "rule14.joblib", results_dir / "rule.joblib")
    shutil.copyfile(results_dir / "rule14.json", results_dir / "rule.json")
    summary = {"git_commit": git_commit(), "partition_counts": dict(Counter(parts.values())),
               "sampling_counts": sampling_counts,
               "selected_learned_model": selected, "tune_scores": tune_metrics, "thresholds": thresholds,
               "recommended_mode": "learned" if selected and tune_metrics[selected] > tune_metrics["rule14"] else "rule",
               "failures": failures, "pair_cache_load_or_generation_seconds": retrieval_seconds,
               "total_seconds": time.monotonic() - started, "timings": timings,
               "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024),
               "disk_bytes": sum(p.stat().st_size for p in results_dir.rglob("*") if p.is_file()),
               "exp001_historical_macro_fbeta": .627204,
               "model_license_note": "Project-trained weights MIT; sklearn runtime BSD-3-Clause, not pretrained weights."}
    write_json_atomic(results_dir / "summary.json", summary)
    (results_dir / "EXP003_results.md").write_text("# EXP003 measured run\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n\nSee evaluation.csv, thresholds.csv and pairs/pool_counts.csv for exact metrics.\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary
