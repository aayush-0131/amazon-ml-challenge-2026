# EXP002F country-scoped postings benchmark

The country-scoped postings indexes (schema 2) are built once and reused for all S1 query batches.
The production path has no character n-gram fallback and no S2/S3 scan during querying.

## Index build/reuse

| Source | Reused | Rows | Seconds | Disk MiB | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| S2 | True | 5,034,616 | 0.0 | 3760.5 | 69.6 |
| S3 | True | 5,285,603 | 0.0 | 3859.9 | 70.1 |

## Candidate retrieval

| Sample | Recall | Matched all-links | Mean candidates | p99 | Query seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| s1_1000 | 93.078% | 80.353% | 118.03 | 120 | 83.9 | 250.2 |

## Intersection scheduling

| Sample | Slice | Scheduler | Mean probes/source query | p95 probes | SQL seconds |
|---|---|---|---:|---:|---:|
| s1_1000 | ALL | selective_v2_compact | 7.93 | 10.0 | 47.256 |
| s1_1000 | India | selective_v2_compact | 7.48 | 10.0 | 32.262 |
| s1_1000 | US | selective_v2_compact | 8.24 | 10.0 | 14.994 |
| s1_1000 | S2 | selective_v2_compact | 8.03 | 10.0 | 21.597 |
| s1_1000 | S2|India | selective_v2_compact | 7.59 | 10.0 | 14.563 |
| s1_1000 | S2|US | selective_v2_compact | 8.34 | 10.0 | 7.034 |
| s1_1000 | S3 | selective_v2_compact | 7.83 | 10.0 | 25.659 |
| s1_1000 | S3|India | selective_v2_compact | 7.38 | 10.0 | 17.699 |
| s1_1000 | S3|US | selective_v2_compact | 8.14 | 10.0 | 7.960 |

Peak RSS is the cumulative process high-water mark, including builds if performed in this process. Query timing excludes S1/GT loading and metric aggregation. The 5k batch may benefit from caches warmed by 1k. Cap-hit rates use S1/source queries as denominator; candidate contribution sets overlap; token-only is incremental to exact passes. Full diagnostics and slices are in the CSV tables.