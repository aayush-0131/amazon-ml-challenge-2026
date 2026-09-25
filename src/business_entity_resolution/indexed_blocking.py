"""Reusable country-scoped SQLite token postings for EXP002c blocking.

The index is built once per target source (S2 or S3).  Subsequent Source-1
queries read indexed postings and record rows only; they never rescan a target
TSV.  SQLite keeps the posting lists on disk, which is deliberately preferable
to retaining a multi-million-row inverted index in the 8 GB laptop's RAM.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, fields
from itertools import combinations
from pathlib import Path

from .data import iter_source_chunks
from .multipass import PASS_NAMES, RetrievedCandidate
from .normalize import TextRepresentations, represent_address, represent_name
from .sampling import SourceRecord
from .similarity import token_jaccard, token_overlap

INDEX_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class SourceIndexConfig:
    """Bounded retrieval controls; none depend on a closed country universe."""

    max_exact_hits: int = 250
    name_query_terms: int = 4
    address_query_terms: int = 4
    numeric_query_terms: int = 3
    max_name_token_df: int = 60
    max_address_token_df: int = 40
    max_numeric_token_df: int = 250
    min_name_token_length: int = 3
    min_address_token_length: int = 3
    intersection_anchor_df: int = 5000
    intersection_terms: int = 8
    max_intersections: int = 10
    max_intersection_hits: int = 150
    record_batch_size: int = 400

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field.name} must be a positive integer")

    @classmethod
    def from_config(cls, config: dict[str, object]) -> SourceIndexConfig:
        values = config.get("source_index")
        if not isinstance(values, dict):
            raise ValueError("Explicit source_index configuration required (use exp002c_postings.json)")
        expected = {field.name for field in fields(cls)}
        if set(values) != expected:
            raise ValueError(f"source_index keys mismatch: missing={expected - set(values)}, unknown={set(values) - expected}")
        return cls(**values)


@dataclass(frozen=True)
class IndexBuildResult:
    source: str
    path: Path
    reused: bool
    row_count: int
    seconds: float
    size_bytes: int


def _index_path(index_dir: Path, source: str) -> Path:
    if source not in {"S2", "S3"}:
        raise ValueError("source must be S2 or S3")
    return index_dir / f"exp002c_{source.lower()}_postings_v2.sqlite"


def _signature(representation: TextRepresentations) -> str:
    return " ".join(sorted(representation.unique_tokens))


class SourceSideIndex:
    """Read-only query facade over one persisted target-source index."""

    def __init__(self, path: Path, source: str, config: SourceIndexConfig) -> None:
        self.path = path
        self.source = source
        self.config = config
        self.connection = sqlite3.connect(
            f"file:{path.absolute()}?mode=ro", uri=True, check_same_thread=False
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA cache_size = -32768")
        metadata = dict(self.connection.execute("SELECT key, value FROM metadata"))
        if metadata.get("status") != "complete":
            self.close()
            raise ValueError(f"Index is incomplete: {path}")
        if metadata.get("source") != source:
            self.close()
            raise ValueError(f"Index source mismatch for {path}: {metadata.get('source')!r}")
        if int(metadata.get("schema_version", "0")) != INDEX_SCHEMA_VERSION:
            self.close()
            raise ValueError(f"Unsupported index schema in {path}")
        self.row_count = int(metadata["row_count"])
        # DF cache is per query, bounded by query text rather than number of S1s.
        self._token_df: dict[tuple[str, str, str], int] = {}
        self._country_counts = dict(self.connection.execute("SELECT country, n FROM country_counts"))
        self.last_diagnostics: Counter[str] = Counter()
        self._weights: dict[tuple[str, str], dict[str, float]] = {}

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> SourceSideIndex:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _country_count(self, country: str) -> int:
        return self._country_counts.get(country, 0)

    def _df(self, field: str, token: str, country: str) -> int:
        key = (country, field, token)
        if key not in self._token_df:
            row = self.connection.execute(
                "SELECT df FROM token_df WHERE country = ? AND field = ? AND token = ?",
                key,
            ).fetchone()
            self._token_df[key] = int(row[0]) if row else 0
        return self._token_df[key]

    def _idf(self, field: str, token: str, country: str) -> float:
        return math.log((self._country_count(country) + 1) / (self._df(field, token, country) + 1)) + 1.0

    def _informative_tokens(
        self,
        tokens: Iterable[str],
        *,
        field: str,
        country: str,
        max_df: int,
        minimum_length: int,
        limit: int,
    ) -> list[str]:
        candidates = [
            (self._df(field, token, country), token)
            for token in set(tokens)
            if len(token) >= minimum_length and 0 < self._df(field, token, country) <= max_df
        ]
        return [token for _, token in sorted(candidates)[:limit]]

    def _exact_ids(self, table: str, country: str, value: str) -> set[int]:
        if not value:
            return set()
        rows = list(self.connection.execute(
            f"SELECT record_id FROM {table} WHERE country = ? AND value = ? ORDER BY record_id LIMIT ?",
            (country, value, self.config.max_exact_hits + 1),
        ))
        self.last_diagnostics["exact_overflow_passes"] += int(len(rows) > self.config.max_exact_hits)
        return {int(row[0]) for row in rows[:self.config.max_exact_hits]}

    def _posting_ids(self, country: str, field: str, token: str) -> set[int]:
        # Called only after DF eligibility: every eligible posting is returned.
        self.last_diagnostics["postings_looked_up"] += 1
        rows = self.connection.execute(
            "SELECT record_id FROM postings WHERE country=? AND field=? AND token=? ORDER BY record_id",
            (country, field, token),
        )
        ids = {int(row[0]) for row in rows}
        self.last_diagnostics["posting_ids_returned"] += len(ids)
        return ids

    def _intersections(self, country: str, terms: list[tuple[str, str]], origins: dict[int, set[str]]) -> None:
        ranked = sorted(
            ((self._df(field, token, country), field, token) for field, token in set(terms)),
        )
        ranked = [term for term in ranked if term[0] > 0][:self.config.intersection_terms]
        count = 0
        for left, right in combinations(ranked, 2):
            if left[0] > self.config.intersection_anchor_df or count >= self.config.max_intersections:
                break
            count += 1
            # CROSS JOIN fixes the small anchor first. Each probe uses the full
            # posting primary key; no materialization of the common right list.
            rows = list(self.connection.execute(
                "SELECT a.record_id FROM postings AS a CROSS JOIN postings AS b "
                "WHERE a.country=? AND a.field=? AND a.token=? "
                "AND b.country=a.country AND b.field=? AND b.token=? AND b.record_id=a.record_id "
                "ORDER BY a.record_id LIMIT ?",
                (country, left[1], left[2], right[1], right[2], self.config.max_intersection_hits + 1),
            ))
            self.last_diagnostics["intersection_queries"] += 1
            self.last_diagnostics["postings_looked_up"] += 2
            if len(rows) > self.config.max_intersection_hits:
                self.last_diagnostics["intersection_overflows"] += 1
                continue  # Reject an uninformative whole intersection, never an arbitrary prefix.
            self.last_diagnostics["posting_ids_returned"] += len(rows)
            for row in rows:
                for _, field, token in (left, right):
                    origins[int(row[0])].add(
                        "name_token" if field == "name" else
                        "numeric_address" if token.isdecimal() else "address_token"
                    )

    def _records(self, ids: set[int]) -> dict[int, sqlite3.Row]:
        if not ids:
            return {}
        ordered = sorted(ids)
        result = {}
        for start in range(0, len(ordered), self.config.record_batch_size):
            batch = ordered[start:start + self.config.record_batch_size]
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                f"SELECT record_id, entity_id, country, business_name, business_address "
                f"FROM records WHERE record_id IN ({placeholders})", batch,
            )
            result.update((int(row["record_id"]), row) for row in rows)
        return result

    def retrieve(self, query_record: SourceRecord) -> list[RetrievedCandidate]:
        """Retrieve bounded candidates for one S1 record from this source index."""

        name = represent_name(query_record.business_name)
        address = represent_address(query_record.business_address)
        self.last_diagnostics = Counter()
        self._token_df.clear()
        self._weights.clear()
        origins: dict[int, set[str]] = defaultdict(set)

        for record_id in self._exact_ids("exact_name", query_record.country, name.normalized):
            origins[record_id].add("exact_name")
        for record_id in self._exact_ids("exact_address", query_record.country, address.normalized):
            origins[record_id].add("exact_address")
        for record_id in self._exact_ids("name_signature", query_record.country, _signature(name)):
            origins[record_id].add("name_signature")

        for token in self._informative_tokens(
            name.unique_tokens,
            field="name",
            country=query_record.country,
            max_df=self.config.max_name_token_df,
            minimum_length=self.config.min_name_token_length,
            limit=self.config.name_query_terms,
        ):
            for record_id in self._posting_ids(query_record.country, "name", token):
                origins[record_id].add("name_token")
        for token in self._informative_tokens(
            address.unique_tokens,
            field="address",
            country=query_record.country,
            max_df=self.config.max_address_token_df,
            minimum_length=self.config.min_address_token_length,
            limit=self.config.address_query_terms,
        ):
            for record_id in self._posting_ids(query_record.country, "address", token):
                origins[record_id].add("address_token")
        for token in self._informative_tokens(
            address.digit_tokens,
            field="address",
            country=query_record.country,
            max_df=self.config.max_numeric_token_df,
            minimum_length=1,
            limit=self.config.numeric_query_terms,
        ):
            for record_id in self._posting_ids(query_record.country, "address", token):
                origins[record_id].add("numeric_address")

        terms = [("name", token) for token in name.unique_tokens if len(token) >= self.config.min_name_token_length]
        terms += [("address", token) for token in address.unique_tokens if token.isdecimal() or len(token) >= self.config.min_address_token_length]
        self._intersections(query_record.country, terms, origins)
        exact = {rid for rid, passes in origins.items() if passes & {"exact_name", "exact_address", "name_signature"}}
        token = {rid for rid, passes in origins.items() if passes & {"name_token", "address_token", "numeric_address"}}
        self.last_diagnostics.update(exact_candidates=len(exact), token_candidates=len(token), token_only_candidates=len(token - exact), candidates_before_cap=len(origins))
        # Raw country equality remains authoritative, including unseen labels.
        rows = {
            record_id: row
            for record_id, row in self._records(set(origins)).items()
            if str(row["country"]) == query_record.country
        }
        states: list[dict[str, object]] = []
        for record_id, row in rows.items():
            candidate_name = represent_name(row["business_name"])
            candidate_address = represent_address(row["business_address"])
            name_jaccard = token_jaccard(name.unique_tokens, candidate_name.unique_tokens)
            address_jaccard = token_jaccard(
                address.unique_tokens, candidate_address.unique_tokens
            )
            digit_overlap = token_overlap(address.digit_tokens, candidate_address.digit_tokens)
            name_idf = self._idf_overlap(name, candidate_name, "name", query_record.country)
            address_idf = self._idf_overlap(address, candidate_address, "address", query_record.country)
            passes = origins[record_id]
            exact_name = "exact_name" in passes
            exact_address = "exact_address" in passes
            signature = "name_signature" in passes
            states.append(
                {
                    "record_id": record_id,
                    "row": row,
                    "exact_name_score": 100 + 30 * address_jaccard + 10 * digit_overlap if exact_name else 0.0,
                    "exact_address_score": 100 + 30 * name_jaccard if exact_address else 0.0,
                    "name_signature_score": 105 + 20 * address_jaccard if signature else 0.0,
                    "name_token_score": 100 * name_idf + 20 * name_jaccard if "name_token" in passes else 0.0,
                    "address_token_score": 100 * address_idf + 20 * address_jaccard + 20 * digit_overlap if "address_token" in passes else 0.0,
                    "numeric_address_score": 80 * digit_overlap + 20 * name_jaccard + 20 * address_jaccard if "numeric_address" in passes else 0.0,
                    "char_name_score": 0.0,
                    "name_idf_overlap": name_idf,
                    "address_idf_overlap": address_idf,
                    "name_token_jaccard": name_jaccard,
                    "address_token_jaccard": address_jaccard,
                    "digit_token_overlap": digit_overlap,
                    "shared_name_tokens": len(name.unique_tokens & candidate_name.unique_tokens),
                    "shared_address_tokens": len(address.unique_tokens & candidate_address.unique_tokens),
                    "shared_digit_tokens": len(address.digit_tokens & candidate_address.digit_tokens),
                }
            )
        return self._finalize(query_record.entity_id, states)

    def _idf_overlap(
        self,
        query: TextRepresentations,
        candidate: TextRepresentations,
        field: str,
        country: str,
    ) -> float:
        if not query.unique_tokens:
            return 0.0
        key = (country, field)
        if key not in self._weights:
            self._weights[key] = {token: self._idf(field, token, country) for token in query.unique_tokens}
        weights = self._weights[key]
        return sum(weights[token] for token in query.unique_tokens & candidate.unique_tokens) / sum(weights.values())

    def _finalize(
        self, source1_entity_id: str, states: list[dict[str, object]]
    ) -> list[RetrievedCandidate]:
        ranks: dict[str, dict[int, int]] = {name: {} for name in PASS_NAMES}
        for pass_name in PASS_NAMES:
            score_key = f"{pass_name}_score"
            ranked = sorted(
                (state for state in states if float(state[score_key]) > 0),
                key=lambda state: (-float(state[score_key]), str(state["row"]["entity_id"])),
            )
            ranks[pass_name] = {
                int(state["record_id"]): index
                for index, state in enumerate(ranked, start=1)
            }
        candidates: list[RetrievedCandidate] = []
        for state in states:
            row = state["row"]
            scores = [float(state[f"{name}_score"]) for name in PASS_NAMES]
            candidates.append(
                RetrievedCandidate(
                    source1_entity_id=source1_entity_id,
                    candidate_entity_id=str(row["entity_id"]),
                    source=self.source,
                    country=str(row["country"]),
                    candidate_name=str(row["business_name"]),
                    candidate_address=str(row["business_address"]),
                    **{f"{name}_score": float(state[f"{name}_score"]) for name in PASS_NAMES},
                    **{f"{name}_rank": ranks[name].get(int(state["record_id"]), 0) for name in PASS_NAMES},
                    name_idf_overlap=float(state["name_idf_overlap"]),
                    address_idf_overlap=float(state["address_idf_overlap"]),
                    name_token_jaccard=float(state["name_token_jaccard"]),
                    address_token_jaccard=float(state["address_token_jaccard"]),
                    digit_token_overlap=float(state["digit_token_overlap"]),
                    char_name_jaccard=0.0,
                    shared_name_tokens=int(state["shared_name_tokens"]),
                    shared_address_tokens=int(state["shared_address_tokens"]),
                    shared_digit_tokens=int(state["shared_digit_tokens"]),
                    pass_count=sum(score > 0 for score in scores),
                    best_retrieval_score=max(scores),
                    second_best_retrieval_score=sorted(scores, reverse=True)[1],
                )
            )
        return sorted(
            candidates,
            key=lambda candidate: (-candidate.best_retrieval_score, candidate.candidate_entity_id),
        )


def build_or_open_source_index(
    source_path: Path,
    source: str,
    index_dir: Path,
    *,
    chunksize: int = 10_000,
    rebuild: bool = False,
) -> IndexBuildResult:
    """Build a complete source-side index once, or return its reusable metadata."""

    index_dir.mkdir(parents=True, exist_ok=True)
    path = _index_path(index_dir, source)
    fingerprint = {"size": source_path.stat().st_size, "mtime_ns": source_path.stat().st_mtime_ns}
    if path.exists() and not rebuild:
        with SourceSideIndex(path, source, SourceIndexConfig()) as index:
            stored = index.connection.execute("SELECT value FROM metadata WHERE key='source_fingerprint'").fetchone()
            if stored is None or json.loads(stored[0]) != fingerprint:
                raise ValueError("Source fingerprint changed; use a new index directory or explicit rebuild")
            return IndexBuildResult(source, path, True, index.row_count, 0.0, path.stat().st_size)
    temporary = path.with_suffix(".building.sqlite")
    # Exclusive reservation prevents concurrent builds from sharing a partial DB.
    # A previous complete index is left intact until atomic replacement succeeds.
    temporary.touch(exist_ok=False)
    started = time.monotonic()
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode = DELETE;
            PRAGMA synchronous = NORMAL;
            PRAGMA temp_store = FILE;
            PRAGMA cache_size = -131072;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE records (
                record_id INTEGER PRIMARY KEY,
                entity_id TEXT NOT NULL UNIQUE,
                country TEXT NOT NULL,
                business_name TEXT NOT NULL,
                business_address TEXT NOT NULL
            );
            CREATE TABLE exact_name (country TEXT NOT NULL, value TEXT NOT NULL, record_id INTEGER NOT NULL);
            CREATE TABLE exact_address (country TEXT NOT NULL, value TEXT NOT NULL, record_id INTEGER NOT NULL);
            CREATE TABLE name_signature (country TEXT NOT NULL, value TEXT NOT NULL, record_id INTEGER NOT NULL);
            CREATE TABLE postings (
                country TEXT NOT NULL, field TEXT NOT NULL, token TEXT NOT NULL,
                record_id INTEGER NOT NULL,
                PRIMARY KEY(country, field, token, record_id)
            ) WITHOUT ROWID;
            CREATE TABLE token_df (
                country TEXT NOT NULL, field TEXT NOT NULL, token TEXT NOT NULL,
                df INTEGER NOT NULL, PRIMARY KEY(country, field, token)
            ) WITHOUT ROWID;
            CREATE TABLE country_counts (country TEXT PRIMARY KEY, n INTEGER NOT NULL) WITHOUT ROWID;
            """
        )
        record_id = 0
        for chunk in iter_source_chunks(source_path, source, chunksize=chunksize):
            records: list[tuple[object, ...]] = []
            exact_names: list[tuple[object, ...]] = []
            exact_addresses: list[tuple[object, ...]] = []
            signatures: list[tuple[object, ...]] = []
            posting_rows: list[tuple[object, ...]] = []
            country_counts: Counter[str] = Counter()
            for row in chunk.itertuples(index=False):
                record_id += 1
                name = represent_name(row.business_name)
                address = represent_address(row.business_address)
                records.append((record_id, row.entity_id, row.country, row.business_name, row.business_address))
                if name.normalized:
                    exact_names.append((row.country, name.normalized, record_id))
                    signatures.append((row.country, _signature(name), record_id))
                if address.normalized:
                    exact_addresses.append((row.country, address.normalized, record_id))
                country_counts[row.country] += 1
                for field, representation in (("name", name), ("address", address)):
                    posting_rows.extend((row.country, field, token, record_id) for token in representation.unique_tokens)
            connection.executemany("INSERT INTO records VALUES (?, ?, ?, ?, ?)", records)
            connection.executemany("INSERT INTO exact_name VALUES (?, ?, ?)", exact_names)
            connection.executemany("INSERT INTO exact_address VALUES (?, ?, ?)", exact_addresses)
            connection.executemany("INSERT INTO name_signature VALUES (?, ?, ?)", signatures)
            connection.executemany("INSERT INTO postings VALUES (?, ?, ?, ?)", sorted(posting_rows))
            connection.executemany(
                "INSERT INTO country_counts VALUES (?, ?) ON CONFLICT(country) DO UPDATE SET n=n+excluded.n",
                sorted(country_counts.items()),
            )
            connection.commit()
            print(f"{source}: indexed {record_id:,} records in {time.monotonic() - started:.1f}s", flush=True)
        connection.executescript(
            """
            CREATE INDEX exact_name_lookup ON exact_name(country, value, record_id);
            CREATE INDEX exact_address_lookup ON exact_address(country, value, record_id);
            CREATE INDEX name_signature_lookup ON name_signature(country, value, record_id);
            INSERT INTO token_df SELECT country, field, token, COUNT(*) FROM postings GROUP BY country, field, token;
            """
        )
        metadata = {
            "schema_version": str(INDEX_SCHEMA_VERSION),
            "source": source,
            "row_count": str(record_id),
            "status": "complete",
            "source_fingerprint": json.dumps(fingerprint, sort_keys=True),
        }
        connection.executemany("INSERT INTO metadata VALUES (?, ?)", metadata.items())
        connection.commit()
        if {"size": source_path.stat().st_size, "mtime_ns": source_path.stat().st_mtime_ns} != fingerprint:
            raise ValueError("Source changed during index build; partial index retained")
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Index quick_check failed; partial index retained")
    except Exception:
        connection.close()
        raise
    else:
        connection.close()
        os.replace(temporary, path)
    return IndexBuildResult(source, path, False, record_id, time.monotonic() - started, path.stat().st_size)
