"""EXP006 synthetic TRAIN-only isolation and checkpoint tests."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from business_entity_resolution import exp006
from business_entity_resolution.exp003_training import PairWriter, sample_entity
from business_entity_resolution.features import FEATURE_NAMES
from business_entity_resolution.indexed_blocking import build_or_open_source_index
from business_entity_resolution.modeling import make_hist_gradient_boosting_model
from business_entity_resolution.reranker import CONFIG_HASHES, code_versions, fingerprint, index_identity, sha256


def _tsv(path, header, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root = tmp_path / "root"
    subset = root / "results/tables/exp001_subset_ids.csv"
    subset.parent.mkdir(parents=True)
    with subset.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["source1_entity_id", "country", "match_count", "is_singleton", "partition"])
        writer.writerows((f"S1-{i:03d}", "US" if i % 2 else "India", int(i % 3 == 0), False,
                          "evaluation" if i < 4 else "tuning") for i in range(8))
    actual = exp006.original_partitions
    monkeypatch.setattr(exp006, "ROOT", root)
    monkeypatch.setattr(exp006, "original_partitions", lambda path, authoritative=True: actual(path, authoritative=False))
    train = tmp_path / "train"
    train.mkdir()
    s1_rows = [(f"S1-{i:03d}", "Alpha Company" if i % 3 == 0 else "Other Company", "12 Main", "US" if i % 2 else "India")
               for i in range(24)]
    _tsv(train / "train_source1.tsv", ["entity_id", "business_name", "business_address", "country"], s1_rows)
    for source in (2, 3):
        _tsv(train / f"train_source{source}.tsv", ["entity_id", "business_name", "business_address", "country"],
             [(f"S{source}-0", "Alpha Company", "12 Main", "US"),
              (f"S{source}-1", "Alpha Company", "12 Main", "India"),
              (f"S{source}-2", "Other Company", "12 Main", "US")])
    _tsv(train / "train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"],
         [(f"S1-{i:03d}", "S2-0" if i % 3 == 0 else "") for i in range(24)])
    indexes = tmp_path / "indexes"
    for source in (2, 3):
        build_or_open_source_index(train / f"train_source{source}.tsv", f"S{source}", indexes, chunksize=2)
    return train, indexes


def test_sampling_exclusion_stratification_and_determinism(fixture, tmp_path):
    train, _ = fixture
    first = exp006.sample(train, tmp_path / "first", size=8, seed=2032)
    second = exp006.sample(train, tmp_path / "second", size=8, seed=2032)
    assert first == second
    assert (tmp_path / "first/extra_fit.csv").read_bytes() == (tmp_path / "second/extra_fit.csv").read_bytes()
    _, ids = exp006.load_sample(tmp_path / "first", train)
    assert len(ids) == len(set(ids)) == 8
    assert not set(ids) & {f"S1-{i:03d}" for i in range(8)}
    assert first["country_counts"] == {"India": 4, "US": 4}
    assert all(first[key] == 0 for key in ("overlap_original_20k", "overlap_fit", "overlap_tune", "overlap_evaluation"))
    with pytest.raises(ValueError, match="eligible"):
        exp006.sample(train, tmp_path / "too_many", size=100)


@pytest.mark.parametrize("shards", [1, 4])
def test_generation_resume_and_deterministic_merge(fixture, tmp_path, shards, monkeypatch):
    train, indexes = fixture
    sample = tmp_path / "sample"
    exp006.sample(train, sample, size=8)
    shard_dir = tmp_path / f"shards{shards}"
    for i in range(shards):
        done = exp006.generate(sample, train, indexes, shard_dir, shards=shards, shard=i, checkpoint_every=1)
        assert done["rows"] == len(range(i, 8, shards))
    first_hash = sha256(shard_dir / "shard_00/done.json")
    exp006.generate(sample, train, indexes, shard_dir, shards=shards, shard=0)
    assert sha256(shard_dir / "shard_00/done.json") == first_hash
    merged = exp006.merge(sample, train, indexes, shard_dir, tmp_path / f"merged{shards}", shards=shards)
    assert merged["summary"]["entities"] == 8
    assert merged["summary"]["positives_retained"] >= 0
    assert merged["summary"]["negatives_retained"] >= 0
    assert len(list(exp006.pair_groups(tmp_path / f"merged{shards}/extra_fit"))) <= 8
    if shards == 4:
        with pytest.raises(ValueError, match="identity"):
            exp006.merge(sample, train, indexes, shard_dir, tmp_path / "wrong_merge", shards=1)
        one_dir = tmp_path / "one_shard"
        exp006.generate(sample, train, indexes, one_dir, shards=1, shard=0)
        exp006.merge(sample, train, indexes, one_dir, tmp_path / "one_merged", shards=1)
        for name in ("extra_fit.f32", "extra_fit.tsv", "pool_counts.csv"):
            assert (tmp_path / f"merged{shards}" / name).read_bytes() == (tmp_path / "one_merged" / name).read_bytes()
    if shards == 4:
        done_path = shard_dir / "shard_00/done.json"
        done = json.loads(done_path.read_text())
        done["identity"]["sampling_seed"] = 999
        done_path.write_text(json.dumps(done))
        with pytest.raises(ValueError, match="identity"):
            exp006.generate(sample, train, indexes, shard_dir, shards=4, shard=0)


def test_interrupted_shard_resumes_without_repeating_committed_entities(fixture, tmp_path, monkeypatch):
    train, indexes = fixture
    sample = tmp_path / "sample"
    exp006.sample(train, sample, size=8)
    original = exp006.candidate_pool
    calls = []

    def interrupted(record, *args, **kwargs):
        calls.append(record.entity_id)
        if len(calls) == 3:
            raise RuntimeError("interrupted")
        return original(record, *args, **kwargs)

    monkeypatch.setattr(exp006, "candidate_pool", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        exp006.generate(sample, train, indexes, tmp_path / "shards", shards=1, checkpoint_every=1)
    committed = set(calls[:2])
    resumed = []

    def recording(record, *args, **kwargs):
        resumed.append(record.entity_id)
        return original(record, *args, **kwargs)

    monkeypatch.setattr(exp006, "candidate_pool", recording)
    exp006.generate(sample, train, indexes, tmp_path / "shards", shards=1, checkpoint_every=1)
    assert not committed & set(resumed)
    assert len(resumed) == 6


def test_sampler_and_fixed_model_schema():
    cids = [f"S2-{i}" for i in range(90)]
    matrix = np.zeros((90, len(FEATURE_NAMES)), dtype=np.float32)
    labels = [int(i % 7 == 0) for i in range(90)]
    examples, counts = sample_entity("S1-x", cids, matrix, labels)
    assert {p.candidate_entity_id for p in examples if p.label} == {cid for cid, label in zip(cids, labels) if label}
    assert counts["hard_negative_count"] <= 20 and counts["easy_negative_count"] <= 3
    assert len(FEATURE_NAMES) == 51
    assert CONFIG_HASHES[18] == "34e8daff83b05721bbf16c4ed8b1fb04fa96c7d41482cfcf91c3fcf8f6539cb7"
    config = make_hist_gradient_boosting_model(2030).set_params(early_stopping=False).get_params()
    assert {key: config[key] for key in ("learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf",
                                          "l2_regularization", "random_state", "class_weight", "early_stopping")} == {
        "learning_rate": .08, "max_iter": 160, "max_leaf_nodes": 31, "min_samples_leaf": 30,
        "l2_regularization": 1.0, "random_state": 2030, "class_weight": "balanced", "early_stopping": False}


def test_tune_threshold_rejects_evaluation_rows(tmp_path, monkeypatch):
    writer = PairWriter(tmp_path / "tune")
    matrix = np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32)
    matrix[0, 0] = .9
    writer.add("S1-tune", ["S2-positive"], matrix, [1])
    writer.close()
    monkeypatch.setattr(exp006, "scores_for", lambda _type, _model, x: x[:, 0])
    threshold, metric, grid = exp006.select_tune_threshold(
        tmp_path / "tune", object(), {"S1-tune": frozenset({"S2-positive"})})
    assert threshold <= .9 and metric.macro_fbeta == 1 and len(grid) > 100
    with pytest.raises(ValueError, match="non-TUNE"):
        exp006.select_tune_threshold(tmp_path / "tune", object(), {"S1-evaluation": frozenset()})


def test_evaluation_requires_frozen_model_before_reading_labels(tmp_path, monkeypatch):
    train, indexes, original, fit, tune = [tmp_path / name for name in ("train", "indexes", "original", "fit", "tune")]
    for path in (train, indexes, original, fit, tune):
        path.mkdir()
    (fit / "fit_manifest.json").write_text("{}")
    (tune / "thresholds.csv").write_text("threshold\n.9\n")
    (tune / "frozen.joblib").write_bytes(b"model")
    exp006.write_json_atomic(tune / "freeze.json", {
        "fit_manifest_sha256": sha256(fit / "fit_manifest.json"),
        "bundle_sha256": sha256(tune / "frozen.joblib"),
        "threshold_grid_sha256": sha256(tune / "thresholds.csv"),
        "evaluation_read": True})
    monkeypatch.setattr(exp006, "preflight", lambda *args, **kwargs: pytest.fail("Preflight read EVALUATION before freeze guard"))
    monkeypatch.setattr(exp006, "load_selected_ground_truth",
                        lambda *args: pytest.fail("EVALUATION labels read before freeze guard"))
    with pytest.raises(ValueError, match="not frozen"):
        exp006.evaluate(train, indexes, original, fit, tune, tmp_path / "evaluation")


def test_preflight_does_not_read_evaluation_pairs_before_freeze(fixture, tmp_path):
    train, indexes = fixture
    original = tmp_path / "original"
    pairs = original / "pairs"
    pairs.mkdir(parents=True)
    for name in ("fit.f32", "tune.f32"):
        (pairs / name).write_bytes(b"tiny")
    parts = exp006.original_partitions(exp006.ROOT / "results/tables/exp001_subset_ids.csv")
    with (original / "partitions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["source1_entity_id", "partition"])
        writer.writerows(sorted(parts.items()))
    identity = {"subset_sha256": sha256(exp006.ROOT / "results/tables/exp001_subset_ids.csv"),
                "indexes": index_identity(indexes), "code": code_versions(),
                "inputs": {p.name: fingerprint(p) for p in
                           (train / "train_source1.tsv", train / "train_ground_truth.tsv")}}
    exp006.write_json_atomic(pairs / "complete.json", {
        "identity": identity,
        "files": {"fit.f32": sha256(pairs / "fit.f32"), "tune.f32": sha256(pairs / "tune.f32"),
                  "evaluation.f32": "0" * 64}})
    assert exp006.preflight(train, indexes, original)["original_cache_verified"]
    with pytest.raises(FileNotFoundError):
        exp006.preflight(train, indexes, original, include_evaluation=True)


def test_fixed_fit_manifest_is_deterministic_on_tiny_pairs(tmp_path, monkeypatch):
    paths = {name: tmp_path / name for name in ("train", "indexes", "original", "sample", "merged")}
    for path in paths.values():
        path.mkdir()
    (paths["original"] / "pairs").mkdir()
    rng = np.random.default_rng(5)
    for directory, name, offset in ((paths["original"] / "pairs", "fit", 0),
                                    (paths["merged"], "extra_fit", 60)):
        writer = PairWriter(directory / name)
        matrix = rng.normal(size=(60, len(FEATURE_NAMES))).astype(np.float32)
        writer.add(f"S1-{offset}", [f"S2-{i}" for i in range(60)], matrix, [i % 2 for i in range(60)])
        writer.close()
    (paths["merged"] / "manifest.json").write_text("{}")
    (paths["sample"] / "manifest.json").write_text("{}")
    monkeypatch.setattr(exp006, "preflight", lambda *args: {
        "partition_counts": {"fit": 5999, "tune": 2001, "evaluation": 12000}})
    monkeypatch.setattr(exp006, "_validate_merged", lambda *args: {"summary": {"entities": 50000}})
    first = exp006.fit(paths["train"], paths["indexes"], paths["original"], paths["sample"],
                       paths["merged"], tmp_path / "fit1")
    second = exp006.fit(paths["train"], paths["indexes"], paths["original"], paths["sample"],
                        paths["merged"], tmp_path / "fit2")
    assert first == second
    assert first["feature_names"] == list(FEATURE_NAMES)
    assert first["training_pairs"] == 120
    assert first["model_config"]["early_stopping"] is False
