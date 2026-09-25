# EXP002F country-scoped postings benchmark

The country-scoped postings indexes (schema 2) are built once and reused for all S1 query batches.
The production path has no character n-gram fallback and no S2/S3 scan during querying.

## Index build/reuse

| Source | Reused | Rows | Seconds | Disk MiB | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| S2 | True | 5,034,616 | 0.0 | 3760.5 | 69.5 |
| S3 | True | 5,285,603 | 0.0 | 3859.9 | 69.9 |

## Candidate retrieval

| Sample | Recall | Matched all-links | Mean candidates | p99 | Query seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| s1_5000 | 94.456% | 84.109% | 119.10 | 120 | 317.9 | 300.0 |

## Intersection scheduling

| Sample | Slice | Scheduler | Mean probes/source query | p95 probes | SQL seconds |
|---|---|---|---:|---:|---:|
| s1_5000 | ALL | selective_v2_compact | 10.37 | 14.0 | 172.603 |
| s1_5000 | India | selective_v2_compact | 9.55 | 14.0 | 69.807 |
| s1_5000 | US | selective_v2_compact | 10.93 | 14.0 | 102.796 |
| s1_5000 | S2 | selective_v2_compact | 10.53 | 14.0 | 82.370 |
| s1_5000 | S2|India | selective_v2_compact | 9.63 | 14.0 | 32.457 |
| s1_5000 | S2|US | selective_v2_compact | 11.15 | 14.0 | 49.913 |
| s1_5000 | S3 | selective_v2_compact | 10.21 | 14.0 | 90.233 |
| s1_5000 | S3|India | selective_v2_compact | 9.47 | 14.0 | 37.349 |
| s1_5000 | S3|US | selective_v2_compact | 10.72 | 14.0 | 52.884 |

Peak RSS is the cumulative process high-water mark, including builds if performed in this process. Query timing excludes S1/GT loading and metric aggregation. The 5k batch may benefit from caches warmed by 1k. Cap-hit rates use S1/source queries as denominator; candidate contribution sets overlap; token-only is incremental to exact passes. Full diagnostics and slices are in the CSV tables.