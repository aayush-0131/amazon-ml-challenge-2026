# EXP002c country-scoped postings benchmark

The country-scoped postings indexes (schema 2) are built once and reused for all S1 query batches.
The production path has no character n-gram fallback and no S2/S3 scan during querying.

## Index build/reuse

| Source | Reused | Rows | Seconds | Disk MiB | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| S2 | True | 5,034,616 | 0.0 | 3760.5 | 69.2 |
| S3 | True | 5,285,603 | 0.0 | 3859.9 | 69.7 |

## Candidate retrieval

| Sample | Recall | Matched all-links | Mean candidates | p99 | Query seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| s1_1000 | 90.377% | 74.636% | 64.01 | 80 | 14.3 | 242.4 |
| s1_5000 | 90.899% | 75.751% | 63.84 | 80 | 71.4 | 267.5 |

Peak RSS is the cumulative process high-water mark, including builds if performed in this process. Query timing excludes S1/GT loading and metric aggregation. The 5k batch may benefit from caches warmed by 1k. Cap-hit rates use S1/source queries as denominator; candidate contribution sets overlap; token-only is incremental to exact passes. Full diagnostics and slices are in the CSV tables.