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
| s1_5000 | 94.427% | 84.046% | 119.42 | 120 | 290.4 | 301.7 |

## Intersection scheduling

| Sample | Slice | Scheduler | Mean probes/source query | p95 probes | SQL seconds |
|---|---|---|---:|---:|---:|
| s1_5000 | ALL | selective_v2_compact | 12.66 | 18.0 | 158.325 |
| s1_5000 | India | selective_v2_compact | 11.61 | 18.0 | 65.736 |
| s1_5000 | US | selective_v2_compact | 13.38 | 18.0 | 92.589 |
| s1_5000 | S2 | selective_v2_compact | 12.90 | 18.0 | 75.732 |
| s1_5000 | S2|India | selective_v2_compact | 11.75 | 18.0 | 30.903 |
| s1_5000 | S2|US | selective_v2_compact | 13.68 | 18.0 | 44.828 |
| s1_5000 | S3 | selective_v2_compact | 12.43 | 18.0 | 82.594 |
| s1_5000 | S3|India | selective_v2_compact | 11.47 | 18.0 | 34.833 |
| s1_5000 | S3|US | selective_v2_compact | 13.08 | 18.0 | 47.761 |

Peak RSS is the cumulative process high-water mark, including builds if performed in this process. Query timing excludes S1/GT loading and metric aggregation. The 5k batch may benefit from caches warmed by 1k. Cap-hit rates use S1/source queries as denominator; candidate contribution sets overlap; token-only is incremental to exact passes. Full diagnostics and slices are in the CSV tables.