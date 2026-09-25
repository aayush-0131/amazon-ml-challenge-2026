# EXP002F country-scoped postings benchmark

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
| s1_1000 | 92.572% | 79.522% | 110.30 | 120 | 350.0 | 246.7 |

## Intersection scheduling

| Sample | Slice | Scheduler | Mean probes/source query | p95 probes | SQL seconds |
|---|---|---|---:|---:|---:|
| s1_1000 | ALL | selective_v2_compact | 5.24 | 6.0 | 207.415 |
| s1_1000 | India | selective_v2_compact | 5.10 | 6.0 | 107.325 |
| s1_1000 | US | selective_v2_compact | 5.34 | 6.0 | 100.090 |
| s1_1000 | S2 | selective_v2_compact | 5.29 | 6.0 | 103.005 |
| s1_1000 | S2|India | selective_v2_compact | 5.17 | 6.0 | 56.493 |
| s1_1000 | S2|US | selective_v2_compact | 5.38 | 6.0 | 46.512 |
| s1_1000 | S3 | selective_v2_compact | 5.18 | 6.0 | 104.410 |
| s1_1000 | S3|India | selective_v2_compact | 5.02 | 6.0 | 50.832 |
| s1_1000 | S3|US | selective_v2_compact | 5.30 | 6.0 | 53.578 |

Peak RSS is the cumulative process high-water mark, including builds if performed in this process. Query timing excludes S1/GT loading and metric aggregation. The 5k batch may benefit from caches warmed by 1k. Cap-hit rates use S1/source queries as denominator; candidate contribution sets overlap; token-only is incremental to exact passes. Full diagnostics and slices are in the CSV tables.