# EXP002 indexed-blocker benchmark

The target-side SQLite FTS5 indexes are built once and then reused for all S1 query batches.
The production path has no character n-gram fallback and no S2/S3 scan during querying.

## Index build/reuse

| Source | Reused | Rows | Seconds | Disk MiB | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| S2 | False | 5,034,616 | 185.5 | 2558.0 | 593.2 |
| S3 | False | 5,285,603 | 193.1 | 2617.2 | 593.2 |

## Candidate retrieval

| Sample | Recall | Matched all-links | Mean candidates | p99 | Query seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| s1_1000 | 82.470% | 63.929% | 198.72 | 200 | 1004.7 | 593.2 |
| s1_5000 | 81.723% | 63.923% | 198.64 | 200 | 5030.5 | 593.2 |
