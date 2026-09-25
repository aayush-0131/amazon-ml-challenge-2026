"""Reusable SQLite FTS5 source-side lexical index for EXP002 blocking.

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
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .data import DEFAULT_CHUNK_SIZE, iter_source_chunks
from .multipass import PASS_NAMES, CandidateBudget, RetrievedCandidate
from .normalize import TextRepresentations, represent_address, represent_name
from .sampling import SourceRecord
from .similarity import token_jaccard, token_overlap

INDEX_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SourceIndexConfig:
    """Bounded retrieval controls; none depend on a closed country universe."""

    max_exact_hits: int = 250
    max_hits_per_token: int = 80
    name_query_terms: int = 3
    address_query_terms: int = 3
    numeric_query_terms: int = 3
    max_name_token_df: int = 100_000
    max_address_token_df: int = 100_000
    max_numeric_token_df: int = 500_000
    min_name_token_length: int = 3
    min_address_token_length: int = 3


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
    return index_dir / f"exp002_{source.lower()}_fts.sqlite"


def _fts_quote(token: str) -> str:
    return '"' + token.replace('"', '""') + '"'


def _fts_query(country: str, field: str, token: str) -> str:
    return f"country : {_fts_quote(country.casefold())} AND {field} : {_fts_quote(token)}"


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
        self._token_df: dict[tuple[str, str], int] = {}
        self._country_counts: dict[str, int] = {}

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> SourceSideIndex:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _country_count(self, country: str) -> int:
        if country not in self._country_counts:
            self._country_counts[country] = int(
                self.connection.execute(
                    "SELECT COUNT(*) FROM records WHERE country = ?", (country,)
                ).fetchone()[0]
            )
        return self._country_counts[country]

    def _df(self, field: str, token: str) -> int:
        key = (field, token)
        if key not in self._token_df:
            row = self.connection.execute(
                "SELECT doc FROM vocabulary WHERE col = ? AND term = ?",
                (field, token),
            ).fetchone()
            self._token_df[key] = int(row[0]) if row else 0
        return self._token_df[key]

    def _idf(self, field: str, token: str, country: str) -> float:
        return math.log((self._country_count(country) + 1) / (self._df(field, token) + 1)) + 1.0

    def _informative_tokens(
        self,
        tokens: Iterable[str],
        *,
        field: str,
        max_df: int,
        minimum_length: int,
        limit: int,
    ) -> list[str]:
        candidates = [
            (self._df(field, token), token)
            for token in set(tokens)
            if len(token) >= minimum_length and 0 < self._df(field, token) <= max_df
        ]
        return [token for _, token in sorted(candidates)[:limit]]

    def _exact_ids(self, table: str, country: str, value: str) -> set[int]:
        if not value:
            return set()
        rows = self.connection.execute(
            f"SELECT record_id FROM {table} WHERE country = ? AND value = ? LIMIT ?",
            (country, value, self.config.max_exact_hits),
        )
        return {int(row[0]) for row in rows}

    def _fts_ids(self, country: str, field: str, token: str) -> set[int]:
        rows = self.connection.execute(
            "SELECT rowid FROM documents WHERE documents MATCH ? ORDER BY bm25(documents) LIMIT ?",
            (_fts_query(country, field, token), self.config.max_hits_per_token),
        )
        return {int(row[0]) for row in rows}

    def _records(self, ids: set[int]) -> dict[int, sqlite3.Row]:
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self.connection.execute(
            f"SELECT record_id, entity_id, country, business_name, business_address "
            f"FROM records WHERE record_id IN ({placeholders})",
            tuple(sorted(ids)),
        )
        return {int(row["record_id"]): row for row in rows}

    def retrieve(self, query_record: SourceRecord) -> list[RetrievedCandidate]:
        """Retrieve bounded candidates for one S1 record from this source index."""

        name = represent_name(query_record.business_name)
        address = represent_address(query_record.business_address)
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
            max_df=self.config.max_name_token_df,
            minimum_length=self.config.min_name_token_length,
            limit=self.config.name_query_terms,
        ):
            for record_id in self._fts_ids(query_record.country, "name", token):
                origins[record_id].add("name_token")
        for token in self._informative_tokens(
            address.unique_tokens,
            field="address",
            max_df=self.config.max_address_token_df,
            minimum_length=self.config.min_address_token_length,
            limit=self.config.address_query_terms,
        ):
            for record_id in self._fts_ids(query_record.country, "address", token):
                origins[record_id].add("address_token")
        for token in self._informative_tokens(
            address.digit_tokens,
            field="address",
            max_df=self.config.max_numeric_token_df,
            minimum_length=1,
            limit=self.config.numeric_query_terms,
        ):
            for record_id in self._fts_ids(query_record.country, "address", token):
                origins[record_id].add("numeric_address")

        # FTS country terms are retrieval hints.  The raw-string equality check is
        # authoritative and keeps the hard country constraint correct even if two
        # open-set labels share tokens (for example multiword country labels).
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
        weights = {token: self._idf(field, token, country) for token in query.unique_tokens}
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
    chunksize: int = DEFAULT_CHUNK_SIZE,
    rebuild: bool = False,
) -> IndexBuildResult:
    """Build a complete source-side index once, or return its reusable metadata."""

    index_dir.mkdir(parents=True, exist_ok=True)
    path = _index_path(index_dir, source)
    if path.exists() and not rebuild:
        with SourceSideIndex(path, source, SourceIndexConfig()) as index:
            return IndexBuildResult(source, path, True, index.row_count, 0.0, path.stat().st_size)
    if path.exists() and rebuild:
        path.unlink()
    temporary = path.with_suffix(".building.sqlite")
    if temporary.exists():
        raise FileExistsError(f"Incomplete build exists: {temporary}; remove it explicitly after inspection")
    started = time.monotonic()
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode = OFF;
            PRAGMA synchronous = OFF;
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
            CREATE VIRTUAL TABLE documents USING fts5(country, name, address, content='', tokenize='unicode61 remove_diacritics 2');
            """
        )
        record_id = 0
        for chunk in iter_source_chunks(source_path, source, chunksize=chunksize):
            records: list[tuple[object, ...]] = []
            exact_names: list[tuple[object, ...]] = []
            exact_addresses: list[tuple[object, ...]] = []
            signatures: list[tuple[object, ...]] = []
            documents: list[tuple[object, ...]] = []
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
                documents.append((record_id, row.country.casefold(), name.normalized, address.normalized))
            connection.executemany("INSERT INTO records VALUES (?, ?, ?, ?, ?)", records)
            connection.executemany("INSERT INTO exact_name VALUES (?, ?, ?)", exact_names)
            connection.executemany("INSERT INTO exact_address VALUES (?, ?, ?)", exact_addresses)
            connection.executemany("INSERT INTO name_signature VALUES (?, ?, ?)", signatures)
            connection.executemany("INSERT INTO documents(rowid, country, name, address) VALUES (?, ?, ?, ?)", documents)
            connection.commit()
        connection.executescript(
            """
            CREATE INDEX exact_name_lookup ON exact_name(country, value, record_id);
            CREATE INDEX exact_address_lookup ON exact_address(country, value, record_id);
            CREATE INDEX name_signature_lookup ON name_signature(country, value, record_id);
            CREATE VIRTUAL TABLE vocabulary USING fts5vocab(documents, 'col');
            """
        )
        metadata = {
            "schema_version": str(INDEX_SCHEMA_VERSION),
            "source": source,
            "row_count": str(record_id),
            "status": "complete",
        }
        connection.executemany("INSERT INTO metadata VALUES (?, ?)", metadata.items())
        connection.commit()
    except Exception:
        connection.close()
        raise
    else:
        connection.close()
        os.replace(temporary, path)
    return IndexBuildResult(source, path, False, record_id, time.monotonic() - started, path.stat().st_size)
