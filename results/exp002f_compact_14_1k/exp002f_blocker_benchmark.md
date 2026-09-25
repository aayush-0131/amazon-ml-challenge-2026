# EXP002F country-scoped postings benchmark

The country-scoped postings indexes (schema 2) are built once and reused for all S1 query batches.
The production path has no character n-gram fallback and no S2/S3 scan during querying.

## Index build/reuse

| Source | Reused | Rows | Seconds | Disk MiB | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| S2 | True | 5,034,616 | 0.0 | 3760.5 | 69.4 |
| S3 | True | 5,285,603 | 0.0 | 3859.9 | 69.9 |

## Candidate retrieval

| Sample | Recall | Matched all-links | Mean candidates | p99 | Query seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| s1_1000 | 93.500% | 81.497% | 119.10 | 120 | 62.7 | 251.9 |

## Intersection scheduling

| Sample | Slice | Scheduler | Mean probes/source query | p95 probes | SQL seconds |
|---|---|---|---:|---:|---:|
| s1_1000 | ALL | selective_v2_compact | 10.34 | 14.0 | 32.832 |
| s1_1000 | India | selective_v2_compact | 9.62 | 14.0 | 15.744 |
| s1_1000 | US | selective_v2_compact | 10.85 | 14.0 | 17.088 |
| s1_1000 | S2 | selective_v2_compact | 10.49 | 14.0 | 15.640 |
| s1_1000 | S2|India | selective_v2_compact | 9.74 | 14.0 | 7.620 |
| s1_1000 | S2|US | selective_v2_compact | 11.03 | 14.0 | 8.019 |
| s1_1000 | S3 | selective_v2_compact | 10.19 | 14.0 | 17.192 |
| s1_1000 | S3|India | selective_v2_compact | 9.50 | 14.0 | 8.124 |
| s1_1000 | S3|US | selective_v2_compact | 10.67 | 14.0 | 9.069 |

Peak RSS is the cumulative process high-water mark, including builds if performed in this process. Query timing excludes S1/GT loading and metric aggregation. The 5k batch may benefit from caches warmed by 1k. Cap-hit rates use S1/source queries as denominator; candidate contribution sets overlap; token-only is incremental to exact passes. Full diagnostics and slices are in the CSV tables.