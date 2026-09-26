"""Tiny synthetic checks only; no access to Amazon data or the AWS safety run."""
import hashlib
import json
import sqlite3
import subprocess
import sys
import types

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from business_entity_resolution import exp003_inference as inference
from business_entity_resolution import reranker
from business_entity_resolution.features import FEATURE_NAMES
from business_entity_resolution.modeling import make_hist_gradient_boosting_model
from test_exp003 import inference_fixture  # shared tiny fixture, includes rule bundle

BASE = "0ab2266825b553e3957237290c92cab51a6b4700"


def base_source(name):
    return subprocess.check_output(["git", "show", f"{BASE}:src/business_entity_resolution/{name}"], cwd=reranker.ROOT)


def frozen_bundle(path, *, learned=False):
    if learned:
        x = np.random.default_rng(2030).normal(size=(32, len(FEATURE_NAMES))).astype(np.float32)
        with threadpool_limits(limits=1):
            model = make_hist_gradient_boosting_model(2030).set_params(max_iter=2, min_samples_leaf=2, early_stopping=False).fit(x, np.array([0, 1] * 16))
        bundle = reranker.make_bundle(model, "hist_gradient_boosting", .5, 18, 2030)
    else:
        bundle = reranker.make_bundle(None, "rule", 96, 14, 2030)
    bundle["code_sha256"] = {name: hashlib.sha256(base_source(name)).hexdigest() for name in reranker.CODE_FILES}
    bundle["git_commit"] = BASE
    reranker.save_bundle(bundle, path)
    return bundle


@pytest.mark.parametrize("learned", [False, True])
def test_default_four_byte_equivalent_to_base(inference_fixture, tmp_path, learned):
    test, indexes, bundle_path = inference_fixture
    bundle = frozen_bundle(bundle_path, learned=learned)
    assert reranker.load_bundle(bundle_path)["code_sha256"] == bundle["code_sha256"]
    # Execute the exact committed base implementation, not a rewritten replica.
    historical = types.ModuleType("business_entity_resolution._historical_inference")
    historical.__package__ = "business_entity_resolution"
    exec(compile(base_source("exp003_inference.py"), "base_inference.py", "exec"), historical.__dict__)
    mode = "learned" if learned else "rule"
    for module, label in ((historical, "old"), (inference, "new")):
        for i in range(4):
            module.infer_shard(bundle_path, mode, test, indexes, tmp_path / label, shard=i)
        module.merge_shards(tmp_path / label, test, tmp_path / (label + "_merged"))
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        assert (tmp_path / "old_merged" / name).read_bytes() == (tmp_path / "new_merged" / name).read_bytes()
        for i in range(4):
            assert (tmp_path / "old" / f"shard_{i:02d}" / name).read_bytes() == (tmp_path / "new" / f"shard_{i:02d}" / name).read_bytes()


@pytest.mark.parametrize("shards", [1, 8, 16])
def test_shard_counts_and_deterministic_merge(inference_fixture, tmp_path, shards):
    test, indexes, bundle = inference_fixture
    directory = tmp_path / "shards"
    for shard in range(shards):
        done = inference.infer_shard(bundle, "rule", test, indexes, directory, shard=shard, shards=shards, smoke_limit=4000)
        assert done["shards"] == shards
        assert done["rows"] == len(range(shard, 12, shards))
    for label in ("first", "again"):
        assert inference.merge_shards(directory, test, tmp_path / label, shards=shards) == 12
    for name, header in (("matching_results.tsv", "matched_entity_ids"), ("candidate_pairs.tsv", "candidate_entity_ids")):
        content = (tmp_path / "first" / name).read_bytes()
        assert content == (tmp_path / "again" / name).read_bytes()
        assert content.startswith(f"source1_entity_id\t{header}\n".encode())
        lines = content.decode().splitlines()[1:]
        assert [line.split("\t")[0] for line in lines] == [f"S1-{i}" for i in range(12)]
        assert lines[1] == "S1-1\t"  # singleton remains explicit
    if shards == 16:
        # All 16 manifests are required, even for shards with zero fixture rows.
        missing = directory / "shard_15/done.json"
        missing.rename(missing.with_suffix(".saved"))
        with pytest.raises(FileNotFoundError):
            inference.merge_shards(directory, test, tmp_path / "missing", shards=16)


@pytest.mark.parametrize("shards,shard", [(0, 0), (65, 0), (4, 4), (16, 16), (16, -1)])
def test_invalid_run_counts_fail_before_io(tmp_path, shards, shard):
    with pytest.raises(ValueError, match="1..64"):
        inference.infer_shard(tmp_path / "missing", "rule", tmp_path, tmp_path, tmp_path, shards=shards, shard=shard)


@pytest.mark.parametrize("shards", [0, 65])
def test_invalid_merge_counts_fail_before_io(tmp_path, shards):
    with pytest.raises(ValueError, match="1..64"):
        inference.merge_shards(tmp_path, tmp_path, tmp_path / "out", shards=shards)


def test_mixed_counts_and_resume_fail_closed(inference_fixture, tmp_path):
    test, indexes, bundle = inference_fixture
    directory = tmp_path / "shards"
    for shard in range(16):
        inference.infer_shard(bundle, "rule", test, indexes, directory, shards=16, shard=shard)
    with pytest.raises(ValueError, match="Checkpoint identity mismatch"):
        inference.infer_shard(bundle, "rule", test, indexes, directory, shards=4, shard=0)
    with pytest.raises(ValueError, match="Mixed/duplicate"):
        inference.merge_shards(directory, test, tmp_path / "wrong_count")  # default 4
    path = directory / "shard_15/done.json"
    manifest = json.loads(path.read_text())
    manifest["shards"] = 4
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Mixed/duplicate"):
        inference.merge_shards(directory, test, tmp_path / "mixed", shards=16)


def test_duplicate_s1_and_full_guard_unchanged(inference_fixture, tmp_path):
    test, indexes, bundle = inference_fixture
    with pytest.raises(ValueError, match="Full TEST"):
        inference.infer_shard(bundle, "rule", test, indexes, tmp_path / "out", shards=16, smoke_limit=None)
    with pytest.raises(ValueError, match="4000"):
        inference.infer_shard(bundle, "rule", test, indexes, tmp_path / "out", shards=16, smoke_limit=4001)
    source1 = test / "test_source1.tsv"
    source1.write_text(source1.read_text().replace("S1-1\t", "S1-0\t"))
    for shard in range(16):
        inference.infer_shard(bundle, "rule", test, indexes, tmp_path / "duplicate", shards=16, shard=shard)
    with pytest.raises(sqlite3.IntegrityError):
        inference.merge_shards(tmp_path / "duplicate", test, tmp_path / "merge", shards=16)


def test_exact_base_bundle_allowlist_only(inference_fixture, monkeypatch):
    _, _, path = inference_fixture
    bundle = frozen_bundle(path)
    saved = bundle["code_sha256"]
    assert reranker.compatible_code_versions(saved)
    for name in reranker.CODE_FILES:
        assert not reranker.compatible_code_versions({**saved, name: "unrecognized"})
    current = reranker.code_versions()
    monkeypatch.setattr(reranker, "code_versions", lambda: {**current, "exp003_inference.py": "future_unreviewed"})
    assert not reranker.compatible_code_versions(saved)


def test_cli_passes_configurable_counts(monkeypatch, tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("shards_cli", reranker.ROOT / "scripts/infer_exp003.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    seen = []
    monkeypatch.setattr(cli, "infer_shard", lambda *args, **kwargs: seen.append(kwargs))
    monkeypatch.setattr(cli, "merge_shards", lambda *args, **kwargs: seen.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["infer", "run", "--bundle", "bundle", "--model", "learned", "--index-dir", "idx", "--output-dir", "out", "--shard", "15", "--shards", "16", "--smoke-limit", "4000"])
    cli.main()
    assert seen[-1]["shards"] == 16 and seen[-1]["shard"] == 15 and seen[-1]["smoke_limit"] == 4000
    monkeypatch.setattr(sys, "argv", ["infer", "merge", "--shard-dir", "shards", "--output-dir", "out", "--shards", "16"])
    cli.main()
    assert seen[-1]["shards"] == 16
