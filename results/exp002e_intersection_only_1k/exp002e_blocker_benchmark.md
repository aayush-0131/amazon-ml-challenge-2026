# EXP002E country-scoped postings benchmark

The country-scoped postings indexes (schema 2) are built once and reused for all S1 query batches.
The production path has no character n-gram fallback and no S2/S3 scan during querying.

## Index build/reuse

| Source | Reused | Rows | Seconds | Disk MiB | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| S2 | True | 5,034,616 | 0.0 | 3760.5 | 69.4 |
| S3 | True | 5,285,603 | 0.0 | 3859.9 | 70.0 |

## Candidate retrieval

| Sample | Recall | Matched all-links | Mean candidates | p99 | Query seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| s1_1000 | 93.585% | 82.121% | 119.44 | 120 | 474.8 | 250.9 |

## Intersection scheduling

| Sample | Slice | Scheduler | Mean probes/source query | p95 probes | SQL seconds |
|---|---|---|---:|---:|---:|
| s1_1000 | ALL | selective_v1 | 20.80 | 24.0 | 265.965 |
| s1_1000 | India | selective_v1 | 22.45 | 24.0 | 143.550 |
| s1_1000 | US | selective_v1 | 19.64 | 24.0 | 122.415 |
| s1_1000 | S2 | selective_v1 | 20.84 | 24.0 | 128.272 |
| s1_1000 | S2|India | selective_v1 | 22.28 | 24.0 | 71.149 |
| s1_1000 | S2|US | selective_v1 | 19.82 | 24.0 | 57.123 |
| s1_1000 | S3 | selective_v1 | 20.76 | 24.0 | 137.693 |
| s1_1000 | S3|India | selective_v1 | 22.62 | 24.0 | 72.401 |
| s1_1000 | S3|US | selective_v1 | 19.46 | 24.0 | 65.292 |

Peak RSS is the cumulative process high-water mark, including builds if performed in this process. Query timing excludes S1/GT loading and metric aggregation. The 5k batch may benefit from caches warmed by 1k. Cap-hit rates use S1/source queries as denominator; candidate contribution sets overlap; token-only is incremental to exact passes. Full diagnostics and slices are in the CSV tables.