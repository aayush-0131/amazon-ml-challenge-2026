"""EXP003 frozen indexed pools and versioned local model bundles (no labels)."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import joblib
import numpy as np

from .features import FEATURE_NAMES, build_feature_row
from .indexed_blocking import SourceIndexConfig, SourceSideIndex, _index_path
from .modeling import prediction_scores
from .multipass import CandidateBudget, select_candidates_for_budget

ROOT = Path(__file__).resolve().parents[2]
CONFIG_HASHES = {
    18: "34e8daff83b05721bbf16c4ed8b1fb04fa96c7d41482cfcf91c3fcf8f6539cb7",
    14: "d783e4bc3743aec14b781d749e6726d2237b77e67598e5f0ec1784c53f2b93d8",
}
SUBSET_HASH = "84fa2544c2905fa480ffc0e57a2912143304acb6653a5f6c09a3eefa2a66979c"
RULE_COLUMN = FEATURE_NAMES.index("exp001_rule_score")
PACKAGES = ("numpy", "pandas", "RapidFuzz", "scikit-learn", "scipy", "joblib", "threadpoolctl")
CODE_FILES = ("normalize.py", "features.py", "similarity.py", "indexed_blocking.py",
              "intersection_scheduler.py", "multipass.py", "modeling.py", "reranker.py",
              "exp003_inference.py", "exp003_training.py")
# Audited orchestration-only compatibility with the frozen 0ab2266 bundle.
# Do not weaken checks for feature/retrieval/training/model code or packages.
BASE_ORCHESTRATION_HASHES = {
    "exp003_inference.py": "2fa7572d4b8d262e2f79e08f5220ac14354571e7cc800a5871997e65727b82fa",
    "reranker.py": "128dc29516bccfebcd68e11b3e16ea03547ded8f5aaf110ece71d81b166f7d1e",
}
AUDITED_CONFIGURABLE_INFERENCE_HASH = "efd3ab12a8e938144888426658384e9d7d99bbc1f3eada9a81cb7f003a967958"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path):
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def frozen_config(probes):
    path = ROOT / f"configs/exp002f_compact_{probes}.json"
    if sha256(path) != CONFIG_HASHES[probes]:
        raise ValueError("Frozen blocker configuration changed")
    return json.loads(path.read_text())


def code_versions():
    return {name: sha256(Path(__file__).parent / name) for name in CODE_FILES}


def compatible_code_versions(saved):
    current = code_versions()
    if saved == current:
        return True
    # Accept only this known base-commit pair of hashes on the audited N-shard
    # implementation. Unknown old hashes or future inference edits fail closed.
    return (current["exp003_inference.py"] == AUDITED_CONFIGURABLE_INFERENCE_HASH
            and saved == {**current, **BASE_ORCHESTRATION_HASHES})


def package_versions():
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def write_json_atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".building")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@contextmanager
def open_indexes(index_dir, data_dir, split, config):
    """Only completed schema-2 indexes; verify their persisted source fingerprint.

    A TRAIN index cannot silently be passed as TEST. Fingerprints are size/mtime
    checks, not content hashes: preserve source timestamps when copying to AWS.
    No builder import/call and no source-table scan occurs here.
    """
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    cfg = SourceIndexConfig.from_config(config)
    with ExitStack() as stack:
        indexes = {}
        for source in ("S2", "S3"):
            path = _index_path(Path(index_dir), source)
            if not path.is_file():
                raise FileNotFoundError(f"Completed schema-2 index required: {path}; no construction allowed")
            index = stack.enter_context(SourceSideIndex(path, source, cfg))
            stored = dict(index.connection.execute("SELECT key, value FROM metadata"))
            expected = fingerprint(Path(data_dir) / f"{split}_source{source[-1]}.tsv")
            if json.loads(stored.get("source_fingerprint", "null")) != expected:
                raise ValueError(f"{source} index fingerprint does not match {split} source; do not bypass")
            indexes[source] = index
        yield indexes


def index_identity(index_dir):
    return {s: fingerprint(_index_path(Path(index_dir), s)) for s in ("S2", "S3")}


def candidate_pool(record, indexes, config, *, rule_fallback=False):
    """Learned pool = pass eligible, NO total cap. Rule14 retains frozen cap."""
    budget = CandidateBudget(**config["blocker_budgets"][0])
    candidates, raw_ids = [], set()
    for source in ("S2", "S3"):
        raw = indexes[source].retrieve(record)
        raw_ids.update(c.candidate_entity_id for c in raw)
        candidates.extend(select_candidates_for_budget(raw, budget) if rule_fallback else
                          (c for c in raw if c.selected_by(budget)))
    candidates.sort(key=lambda c: c.candidate_entity_id)
    if len({c.candidate_entity_id for c in candidates}) != len(candidates):
        raise ValueError("Duplicate retrieved candidate ID")
    return candidates, raw_ids


def feature_matrix(record, candidates):
    return np.asarray(
        [tuple(build_feature_row(record, c).values()) for c in candidates],
        dtype=np.float32).reshape(-1, len(FEATURE_NAMES))


def scores_for(model_type, model, matrix):
    if not len(matrix):
        return np.empty(0, dtype=np.float64)
    scores = matrix[:, RULE_COLUMN].astype(np.float64) if model_type == "rule" else prediction_scores(model, matrix)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite prediction scores")
    return scores


def accepted_ids(ids, scores, threshold):
    if len(ids) != len(scores) or not np.isfinite(threshold):
        raise ValueError("Invalid scores/threshold")
    return sorted({cid for cid, score in zip(ids, scores) if score >= threshold})


def make_bundle(model, model_type, threshold, probes, seed, **metadata):
    return {
        "bundle_version": 1, "model": model, "model_type": model_type,
        "threshold": float(threshold), "feature_names": list(FEATURE_NAMES),
        "blocker_config": frozen_config(probes), "blocker_sha256": CONFIG_HASHES[probes],
        "probes": probes, "pool_semantics": "capped_rule14" if probes == 14 else "pass_eligible_no_total_cap",
        "normalization_version": "NFKC-casefold-punctuation-space-v1",
        "code_sha256": code_versions(), "git_commit": git_commit(),
        "training_seed": seed, "package_versions": package_versions(),
        "python_version": platform.python_version(), "model_license": "MIT (project-trained weights)",
        **metadata,
    }


def save_bundle(bundle, path):
    path = Path(path)
    temporary = path.with_suffix(".building.joblib")
    joblib.dump(bundle, temporary, compress=3)
    temporary.replace(path)
    info = {k: v for k, v in bundle.items() if k != "model"}
    info["artifact_sha256"] = sha256(path)
    write_json_atomic(path.with_suffix(".json"), info)


def load_bundle(path):
    # joblib is pickle-based: load only our own trusted training outputs.
    path = Path(path)
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file() or json.loads(sidecar.read_text()).get("artifact_sha256") != sha256(path):
        raise ValueError("Incomplete or corrupt model bundle: sidecar checksum mismatch")
    bundle = joblib.load(path)
    if bundle["bundle_version"] != 1 or bundle["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("Incompatible bundle/feature order")
    probes = bundle["probes"]
    if bundle["blocker_sha256"] != CONFIG_HASHES[probes] or bundle["blocker_config"] != frozen_config(probes):
        raise ValueError("Bundle blocker mismatch")
    if not compatible_code_versions(bundle["code_sha256"]) or bundle["package_versions"] != package_versions():
        raise ValueError("Bundle runtime/code versions differ; use the pinned training checkout/environment")
    if bundle["pool_semantics"] != ("capped_rule14" if probes == 14 else "pass_eligible_no_total_cap"):
        raise ValueError("Bundle candidate pool mismatch")
    return bundle
