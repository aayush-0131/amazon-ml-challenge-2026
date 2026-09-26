"""EXP008 TRAIN-only, bounded secondary blocker on top of frozen compact18.

Retrieval has no access to truth. Truth is loaded only by the oracle runner.
The sidecar preserves the original schema-2 index and compact18 pair cache.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from anyascii import anyascii

from .data import iter_source_chunks
from .exp007 import oracle, partition_ids
from .exp007_features import NORMALIZATION_VERSION, core_name
from .reranker import CONFIG_HASHES, fingerprint, sha256, write_json_atomic
from .sampling import SourceRecord, load_selected_ground_truth, load_selected_source_records

SCHEMA = 1
BASE_COMMIT = "32317d2d2fca72fd2bbedfb806d098173b4988d1"
PASS_ORDER = ("trans_name", "core_name", "digit_name", "digit_address")
VARIANTS = {
    "B0": (),
    "B1": ("trans_name",),
    "B2": ("core_name",),
    "B3": ("digit_name", "digit_address"),
    "B4_12": ("trans_name", "core_name"),
    "B4_13": ("trans_name", "digit_name", "digit_address"),
    "B4_23": ("core_name", "digit_name", "digit_address"),
    "B4_123": PASS_ORDER,
}


@dataclass(frozen=True)
class Policy:
    # A common total cap also bounds combinations. Hits are ordered by pass,
    # then entity ID; high-frequency keys are rejected, never truncated.
    max_df: int = 40
    max_added_per_source: int = 60
    min_name_chars: int = 4
    min_address_chars: int = 8
    max_mean_growth: float = 1.5

    def __post_init__(self):
        limits = (self.max_df, self.max_added_per_source, self.min_name_chars, self.min_address_chars)
        if any(type(value) is not int or value < 1 for value in limits):
            raise ValueError("Positive retrieval limits required")
        if self.max_mean_growth < 1:
            raise ValueError("max_mean_growth must be at least 1")


def transliterated(text: str) -> str:
    """AnyAscii is a second representation; it never replaces Unicode keys."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", anyascii(str(text or "")).casefold()).split())


def digit_letter(text: str) -> str:
    """Preserve Unicode letters, splitting both letter-digit boundaries."""
    value = str(text or "").casefold()
    value = re.sub(r"(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])", " ", value)
    return " ".join(re.sub(r"[^\w]+|_", " ", value, flags=re.UNICODE).split())


def keys(name: str, address: str, policy: Policy) -> dict[str, str]:
    values = {
        "trans_name": transliterated(name),
        "core_name": core_name(name),
        "digit_name": digit_letter(name),
        "digit_address": digit_letter(address),
    }
    # Index both already-separated and joined forms. A transformed query must
    # find the unchanged target (and vice versa); baseline IDs are removed at
    # retrieval time. Only empty or short keys are omitted.
    result = {}
    for kind, value in values.items():
        minimum = policy.min_address_chars if kind == "digit_address" else policy.min_name_chars
        if len(value.replace(" ", "")) < minimum:
            continue
        result[kind] = value
    return result


def index_path(index_dir: Path, source: str) -> Path:
    if source not in {"S2", "S3"}:
        raise ValueError("Only TRAIN target sources S2 and S3 are indexed")
    return Path(index_dir) / f"exp008_{source.lower()}_secondary_v1.sqlite"


def _metadata(connection: sqlite3.Connection) -> dict[str, str]:
    return dict(connection.execute("SELECT key, value FROM metadata"))


def build_index(source_path: Path, source: str, index_dir: Path, policy: Policy, *, chunksize: int = 10000) -> dict:
    """Resume an interrupted chunked build; publish only a complete SQLite DB."""
    source_path, index_dir = Path(source_path), Path(index_dir)
    if source_path.name != f"train_source{source[-1]}.tsv":
        raise ValueError("EXP008 index build is TRAIN only")
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    index_dir.mkdir(parents=True, exist_ok=True)
    final = index_path(index_dir, source)
    pending = final.with_suffix(".building.sqlite")
    source_fp = json.dumps(fingerprint(source_path), sort_keys=True)
    config = json.dumps(asdict(policy), sort_keys=True)
    builder_hashes = json.dumps(_hashes(), sort_keys=True)
    if final.exists():
        with sqlite3.connect(f"file:{final}?mode=ro", uri=True) as db:
            meta = _metadata(db)
        if (meta.get("status") != "complete" or meta.get("source_fingerprint") != source_fp
                or meta.get("policy") != config or meta.get("schema") != str(SCHEMA)
                or meta.get("code_sha256") != builder_hashes):
            raise ValueError(f"Existing EXP008 index is incompatible: {final}")
        return {"path": str(final), "reused": True, "rows": int(meta["rows_done"]),
                "bytes": final.stat().st_size, "sha256": sha256(final)}
    new = not pending.exists()
    db = sqlite3.connect(pending)
    started = time.monotonic()
    try:
        if new:
            db.executescript("""
                PRAGMA journal_mode=DELETE;
                PRAGMA synchronous=NORMAL;
                CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE records(record_id INTEGER PRIMARY KEY, entity_id TEXT NOT NULL UNIQUE,
                    country TEXT NOT NULL, business_name TEXT NOT NULL, business_address TEXT NOT NULL);
                CREATE TABLE keys(country TEXT NOT NULL, kind TEXT NOT NULL, value TEXT NOT NULL,
                    record_id INTEGER NOT NULL, PRIMARY KEY(country,kind,value,record_id)) WITHOUT ROWID;
            """)
            db.executemany("INSERT INTO metadata VALUES (?,?)", [
                ("schema", str(SCHEMA)), ("source", source), ("source_fingerprint", source_fp),
                ("policy", config), ("normalization", NORMALIZATION_VERSION),
                ("code_sha256", builder_hashes),
                ("rows_done", "0"), ("status", "building")])
            db.commit()
        meta = _metadata(db)
        if any((meta.get(k) != v for k, v in (("schema", str(SCHEMA)), ("source", source),
            ("source_fingerprint", source_fp), ("policy", config),
            ("normalization", NORMALIZATION_VERSION), ("code_sha256", builder_hashes)))):
            raise ValueError("Interrupted index does not match TRAIN source or normalization")
        done = int(meta["rows_done"])
        seen = 0
        for chunk in iter_source_chunks(source_path, source, chunksize=chunksize):
            if seen + len(chunk) <= done:
                seen += len(chunk)
                continue
            if seen < done:
                chunk = chunk.iloc[done - seen:]
            rows, key_rows = [], []
            for row in chunk.itertuples(index=False):
                done += 1
                rows.append((done, row.entity_id, row.country, row.business_name, row.business_address))
                key_rows.extend((row.country, kind, value, done) for kind, value in keys(row.business_name, row.business_address, policy).items())
            db.executemany("INSERT INTO records VALUES (?,?,?,?,?)", rows)
            db.executemany("INSERT INTO keys VALUES (?,?,?,?)", sorted(key_rows))
            db.execute("UPDATE metadata SET value=? WHERE key='rows_done'", (str(done),))
            db.commit()
            seen = done
            print(f"{source}: EXP008 indexed {done:,} rows", flush=True)
        if json.dumps(fingerprint(source_path), sort_keys=True) != source_fp:
            raise ValueError("TRAIN source changed during index build")
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("EXP008 index quick_check failed")
        db.execute("UPDATE metadata SET value='complete' WHERE key='status'")
        db.commit()
    finally:
        db.close()
    os.replace(pending, final)
    return {"path": str(final), "reused": False, "rows": done, "bytes": final.stat().st_size,
            "seconds": time.monotonic() - started, "sha256": sha256(final)}


class SecondaryIndex:
    def __init__(self, path: Path, source: str, source_path: Path, policy: Policy):
        self.path = Path(path)
        self.source = source
        self.policy = policy
        self.db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self.db.row_factory = sqlite3.Row
        meta = _metadata(self.db)
        if any((meta.get(k) != v for k, v in (("schema", str(SCHEMA)), ("source", source),
            ("status", "complete"), ("source_fingerprint", json.dumps(fingerprint(source_path), sort_keys=True)),
            ("policy", json.dumps(asdict(policy), sort_keys=True)), ("normalization", NORMALIZATION_VERSION),
            ("code_sha256", json.dumps(_hashes(), sort_keys=True))))):
            self.close()
            raise ValueError("EXP008 secondary index identity mismatch")

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def retrieve(self, record: SourceRecord, passes: tuple[str, ...], baseline_ids=()) -> list[dict]:
        """Deterministic bounded additions. No truth or labels enter this API."""
        if any(p not in PASS_ORDER for p in passes):
            raise ValueError("Unknown EXP008 retrieval pass")
        excluded = set(baseline_ids)
        found: dict[str, dict] = {}
        query = keys(record.business_name, record.business_address, self.policy)
        for kind in PASS_ORDER:
            if kind not in passes or kind not in query:
                continue
            ids = self.db.execute(
                "SELECT record_id FROM keys WHERE country=? AND kind=? AND value=? "
                "ORDER BY record_id LIMIT ?",
                (record.country, kind, query[kind], self.policy.max_df + 1)).fetchall()
            if len(ids) > self.policy.max_df:
                continue
            if not ids:
                continue
            placeholders = ",".join("?" for _ in ids)
            rows = self.db.execute(
                f"SELECT entity_id,country,business_name,business_address FROM records WHERE record_id IN ({placeholders}) ORDER BY entity_id",
                [r[0] for r in ids]).fetchall()
            for row in rows:
                cid = row["entity_id"]
                if cid in excluded or cid in found:
                    continue
                found[cid] = {"candidate_entity_id": cid, "candidate_name": row["business_name"],
                              "candidate_address": row["business_address"], "country": row["country"],
                              "retrieval_pass": kind, "source": self.source}
                if len(found) >= self.policy.max_added_per_source:
                    return list(found.values())
        return list(found.values())


def load_baseline(path: Path, ids: list[str], truth: dict[str, frozenset[str]]) -> dict[str, set[str]]:
    expected = set(ids)
    result = {eid: set() for eid in ids}
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not {"source1_entity_id", "candidate_entity_id", "label"} <= set(reader.fieldnames):
            raise ValueError("Invalid frozen compact18 pair TSV")
        for row in reader:
            eid, cid = row["source1_entity_id"], row["candidate_entity_id"]
            if (eid not in expected or cid in result[eid] or not cid.startswith(("S2-", "S3-"))
                    or int(row["label"]) != int(cid in truth[eid])):
                raise ValueError("Frozen compact18 pool ID or duplicate error")
            result[eid].add(cid)
    return result


def compose_candidates(baseline_ids, hits_by_pass, passes, policy: Policy) -> set[str]:
    """Apply the common per-source cap to independent pass hits."""
    if any(p not in PASS_ORDER for p in passes):
        raise ValueError("Unknown EXP008 retrieval pass")
    chosen = set(baseline_ids)
    for source in ("S2", "S3"):
        remaining = policy.max_added_per_source
        for p in PASS_ORDER:
            if p not in passes:
                continue
            for cid in sorted(hits_by_pass.get(p, ())):
                if not cid.startswith(source + "-") or cid in chosen:
                    continue
                chosen.add(cid)
                remaining -= 1
                if not remaining:
                    break
            if not remaining:
                break
    return chosen


def _summary(truth, eligible, records, seconds):
    result = oracle(truth, eligible, records)
    counts = sorted(len(eligible[e]) for e in truth)
    n = len(counts)
    def quantile(p):
        index = (n - 1) * p
        lo = int(index)
        return counts[lo] + (counts[min(lo + 1, n - 1)] - counts[lo]) * (index - lo)
    result.update({"matched_with_zero_retained_count": sum(bool(truth[e]) and not (set(truth[e]) & eligible[e]) for e in truth),
        "average_eligible_candidates_per_s1": sum(counts) / n, "median_candidates_per_s1": quantile(.5),
        "p95_candidates_per_s1": quantile(.95), "p99_candidates_per_s1": quantile(.99),
        "max_candidates_per_s1": counts[-1], "runtime_seconds": seconds})
    # oracle.slices has exact macro metrics and FN for country and source slices.
    for row in result["slices"]:
        dimension, group = row["dimension"], row["group"]
        if dimension not in {"overall", "country", "source", "source_country"}:
            continue
        source = group if dimension == "source" else group.split("|", 1)[0] if dimension == "source_country" else None
        country = group if dimension == "country" else group.split("|", 1)[1] if dimension == "source_country" else None
        selected = [e for e in truth if country is None or records[e].country == country]
        def filtered(values):
            return {cid for cid in values if source is None or cid.startswith(source + "-")}
        counts_slice = sorted(len(filtered(eligible[e])) for e in selected)
        n_slice = len(counts_slice)
        def q(p):
            x = (n_slice - 1) * p
            lo = int(x)
            return counts_slice[lo] + (counts_slice[min(lo + 1, n_slice - 1)] - counts_slice[lo]) * (x - lo)
        true_links = sum(len(filtered(truth[e])) for e in selected)
        retained_links = sum(len(filtered(truth[e]) & filtered(eligible[e])) for e in selected)
        row.update({
            "candidate_link_recall": retained_links / true_links if true_links else 1.0,
            "perfect_candidate_coverage_rate": sum(filtered(truth[e]) <= filtered(eligible[e]) for e in selected) / n_slice,
            "matched_with_zero_retained_count": sum(bool(filtered(truth[e])) and not
                (filtered(truth[e]) & filtered(eligible[e])) for e in selected),
            "average_eligible_candidates_per_s1": sum(counts_slice) / n_slice,
            "median_candidates_per_s1": q(.5), "p95_candidates_per_s1": q(.95),
            "p99_candidates_per_s1": q(.99), "max_candidates_per_s1": counts_slice[-1],
        })
    return result


def _hashes() -> dict[str, str]:
    root = Path(__file__).parent
    return {name: sha256(root / name) for name in ("exp008.py", "exp007_features.py", "evaluation.py", "exp003_training.py")}


def _identity(data_dir, original_dir, baseline_index_dir, secondary_index_dir, ids, partition):
    from .reranker import index_identity
    return {"partition": partition, "base_commit": BASE_COMMIT,
        "compact18_config_sha256": CONFIG_HASHES[18],
        "compact18_indexes": index_identity(baseline_index_dir),
        "baseline_pairs_sha256": sha256(Path(original_dir) / "pairs" / f"{partition}.tsv"),
        "train_source1_sha256": sha256(Path(data_dir) / "train_source1.tsv"),
        "train_truth_sha256": sha256(Path(data_dir) / "train_ground_truth.tsv"),
        "secondary_indexes": {s: sha256(index_path(secondary_index_dir, s)) for s in ("S2", "S3")},
        "entity_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "code_sha256": _hashes(), "normalization": NORMALIZATION_VERSION}


def run_partition(data_dir: Path, original_dir: Path, sample_dir: Path, baseline_index_dir: Path,
                  secondary_index_dir: Path,
                  output_dir: Path, *, partition: str, policy: Policy, frozen_path: Path | None = None) -> dict:
    """TUNE all ablations, freeze one. EVALUATION accepts only that freeze."""
    run_started = time.monotonic()
    if partition not in {"tune", "evaluation"}:
        raise ValueError("EXP008 uses only TRAIN TUNE or EVALUATION")
    data_dir, original_dir, sample_dir, baseline_index_dir, secondary_index_dir, output_dir = map(
        Path, (data_dir, original_dir, sample_dir, baseline_index_dir, secondary_index_dir, output_dir))
    if partition == "evaluation" and (frozen_path is None or not Path(frozen_path).is_file()):
        raise ValueError("Frozen TUNE policy required before EVALUATION")
    if partition == "evaluation":
        frozen = json.loads(Path(frozen_path).read_text())
        if frozen.get("selected_variant") not in VARIANTS or frozen.get("policy") != asdict(policy):
            raise ValueError("Frozen blocker policy differs")
    if output_dir.exists() and ((output_dir / "manifest.json").exists() or not (output_dir / "hits.sqlite").exists()):
        raise FileExistsError("Use a fresh EXP008 result directory or resume its checkpoint")
    from . import exp006
    exp006.preflight(data_dir, baseline_index_dir, original_dir, include_evaluation=partition == "evaluation")
    sample, sample_ids = exp006.load_sample(sample_dir, data_dir)
    if len(sample_ids) != 50000 or sample["seed"] != exp006.SEED:
        raise ValueError("EXP008 requires the verified EXP006 50k sample")
    ids = partition_ids(partition, sample_ids)
    if len(ids) != (2001 if partition == "tune" else 12000):
        raise ValueError("EXP008 partition count mismatch")
    identity = _identity(data_dir, original_dir, baseline_index_dir, secondary_index_dir, ids, partition)
    if partition == "evaluation":
        for key in ("base_commit", "compact18_config_sha256", "compact18_indexes", "train_source1_sha256", "train_truth_sha256", "secondary_indexes", "code_sha256", "normalization"):
            if frozen["identity"][key] != identity[key]:
                raise ValueError(f"Frozen blocker identity mismatch: {key}")
        variants = ("B0", frozen["selected_variant"]) if frozen["selected_variant"] != "B0" else ("B0",)
    else:
        variants = tuple(VARIANTS)
    output_dir.mkdir(parents=True, exist_ok=True)
    truth = load_selected_ground_truth(data_dir / "train_ground_truth.tsv", ids)
    records = load_selected_source_records(data_dir / "train_source1.tsv", "S1", set(ids))
    started = time.monotonic()
    baseline = load_baseline(original_dir / "pairs" / f"{partition}.tsv", ids, truth)
    # Labels only validate the frozen cache; retrieval never receives truth.
    baseline_result = _summary(truth, baseline, records, 0.0)
    baseline_result["runtime_seconds"] = time.monotonic() - started
    if partition == "evaluation":
        expected = {"macro_fbeta": .985492979884422, "candidate_link_recall": .9596797461050202,
                    "blocker_fn": 1677, "perfect_candidate_coverage_rate": .8884166666666666}
        actual = {"macro_fbeta": baseline_result["overall"]["macro_fbeta"],
                  **{k: baseline_result[k] for k in expected if k != "macro_fbeta"}}
        if any(abs(actual[k] - value) > 1e-12 for k, value in expected.items()):
            raise ValueError(f"EXP008 compact18 EVALUATION control mismatch: {actual}")
    needed_passes = tuple(p for p in PASS_ORDER if any(p in VARIANTS[v] for v in variants))
    additions = {eid: {p: set() for p in needed_passes} for eid in ids}
    pass_seconds = {p: 0.0 for p in needed_passes}
    checkpoint_path = output_dir / "hits.sqlite"
    checkpoint = sqlite3.connect(checkpoint_path)
    checkpoint.executescript("""
        CREATE TABLE IF NOT EXISTS identity(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS done(entity_id TEXT NOT NULL, source TEXT NOT NULL,
            PRIMARY KEY(entity_id,source));
        CREATE TABLE IF NOT EXISTS hits(entity_id TEXT NOT NULL, source TEXT NOT NULL,
            pass TEXT NOT NULL, candidate_id TEXT NOT NULL,
            PRIMARY KEY(entity_id,source,pass,candidate_id));
        CREATE TABLE IF NOT EXISTS timing(pass TEXT PRIMARY KEY, seconds REAL NOT NULL);
    """)
    checkpoint_identity = json.dumps({"identity": identity, "policy": asdict(policy), "passes": needed_passes}, sort_keys=True)
    saved_identity = checkpoint.execute("SELECT value FROM identity WHERE key='run'").fetchone()
    if saved_identity and saved_identity[0] != checkpoint_identity:
        checkpoint.close()
        raise ValueError("EXP008 checkpoint identity mismatch")
    if not saved_identity:
        checkpoint.execute("INSERT INTO identity VALUES ('run',?)", (checkpoint_identity,))
        checkpoint.commit()
    pass_seconds.update(checkpoint.execute("SELECT pass,seconds FROM timing"))
    for source in ("S2", "S3"):
        with SecondaryIndex(index_path(secondary_index_dir, source), source,
                            data_dir / f"train_source{source[-1]}.tsv", policy) as index:
            for eid in ids:
                if checkpoint.execute("SELECT 1 FROM done WHERE entity_id=? AND source=?", (eid, source)).fetchone():
                    for p, cid in checkpoint.execute("SELECT pass,candidate_id FROM hits WHERE entity_id=? AND source=?", (eid, source)):
                        additions[eid][p].add(cid)
                    continue
                old = (cid for cid in baseline[eid] if cid.startswith(source + "-"))
                for p in needed_passes:
                    start_pass = time.monotonic()
                    additions[eid][p].update(row["candidate_entity_id"] for row in
                        index.retrieve(records[eid], (p,), old))
                    pass_seconds[p] += time.monotonic() - start_pass
                    checkpoint.executemany("INSERT INTO hits VALUES (?,?,?,?)",
                        ((eid, source, p, cid) for cid in sorted(additions[eid][p]) if cid.startswith(source + "-")))
                    checkpoint.execute("INSERT INTO timing VALUES (?,?) ON CONFLICT(pass) DO UPDATE SET seconds=excluded.seconds",
                                       (p, pass_seconds[p]))
                checkpoint.execute("INSERT INTO done VALUES (?,?)", (eid, source))
                checkpoint.commit()
    checkpoint.close()
    retrieval_seconds = sum(pass_seconds.values())
    results = {}
    for variant in variants:
        started = time.monotonic()
        if variant == "B0":
            result = baseline_result
        else:
            allowed = set(VARIANTS[variant])
            eligible = {eid: compose_candidates(baseline[eid], additions[eid], VARIANTS[variant], policy)
                        for eid in ids}
            result = _summary(truth, eligible, records, 0.0)
            result["runtime_seconds"] = time.monotonic() - started + sum(pass_seconds[p] for p in allowed)
        result["variant"] = variant
        result["passes"] = list(VARIANTS[variant])
        results[variant] = result
        write_json_atomic(output_dir / f"{variant.lower()}_oracle.json", result)
    if partition == "tune":
        base = results["B0"]
        eligible = [v for v in results.values() if v["average_eligible_candidates_per_s1"] <=
                    base["average_eligible_candidates_per_s1"] * policy.max_mean_growth]
        selected = max(eligible, key=lambda v: (v["overall"]["macro_fbeta"],
            v["candidate_link_recall"], -v["average_eligible_candidates_per_s1"], -len(v["passes"])))
        frozen = {"selected_variant": selected["variant"], "policy": asdict(policy),
                  "identity": identity, "selection_rule": "max TUNE macro F0.5 within mean-growth cap; tie recall then cost"}
        write_json_atomic(output_dir / "frozen_policy.json", frozen)
    else:
        selected = results[frozen["selected_variant"]]
        measured = {
            "macro_f0_5": (baseline_result["overall"]["macro_fbeta"], selected["overall"]["macro_fbeta"]),
            "candidate_link_recall": (baseline_result["candidate_link_recall"], selected["candidate_link_recall"]),
            "blocking_fn": (baseline_result["blocker_fn"], selected["blocker_fn"]),
            "perfect_candidate_coverage_rate": (baseline_result["perfect_candidate_coverage_rate"],
                                                selected["perfect_candidate_coverage_rate"]),
            "matched_with_zero_retained_count": (baseline_result["matched_with_zero_retained_count"],
                                                 selected["matched_with_zero_retained_count"]),
        }
        write_json_atomic(output_dir / "comparison.json", {
            "selected_variant": selected["variant"],
            "metrics": {name: {"compact18": old, "exp008": new, "delta": new - old}
                        for name, (old, new) in measured.items()},
        })
    manifest = {"identity": identity, "policy": asdict(policy), "variants": list(variants),
                "selected_variant": selected["variant"], "retrieval_seconds": retrieval_seconds,
                "run_wall_seconds": time.monotonic() - run_started,
                "pass_query_seconds": pass_seconds,
                "results_sha256": {v: sha256(output_dir / f"{v.lower()}_oracle.json") for v in variants}}
    if partition == "evaluation":
        manifest["comparison_sha256"] = sha256(output_dir / "comparison.json")
    write_json_atomic(output_dir / "manifest.json", manifest)
    def brief(value):
        return {"macro_f0_5": value["overall"]["macro_fbeta"],
            "macro_precision": value["overall"]["macro_precision"],
            "macro_recall": value["overall"]["macro_recall"],
            **{k: value[k] for k in ("candidate_link_recall", "blocker_fn", "perfect_candidate_coverage_rate",
                "matched_with_zero_retained_count", "average_eligible_candidates_per_s1", "median_candidates_per_s1",
                "p95_candidates_per_s1", "p99_candidates_per_s1", "max_candidates_per_s1", "runtime_seconds")}}
    return {"selected_variant": selected["variant"], "variants": {v: brief(r) for v, r in results.items()},
            "output_dir": str(output_dir)}
