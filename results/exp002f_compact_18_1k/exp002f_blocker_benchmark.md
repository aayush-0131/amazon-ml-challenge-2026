# EXP002F country-scoped postings benchmark

The country-scoped postings indexes (schema 2) are built once and reused for all S1 query batches.
The production path has no character n-gram fallback and no S2/S3 scan during querying.

## Index build/reuse

| Source | Reused | Rows | Seconds | Disk MiB | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| S2 | True | 5,034,616 | 0.0 | 3760.5 | 69.6 |
| S3 | True | 5,285,603 | 0.0 | 3859.9 | 70.2 |

## Candidate retrieval

| Sample | Recall | Matched all-links | Mean candidates | p99 | Query seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| s1_1000 | 93.303% | 81.081% | 119.37 | 120 | 65.0 | 253.2 |

## Intersection scheduling

| Sample | Slice | Scheduler | Mean probes/source query | p95 probes | SQL seconds |
|---|---|---|---:|---:|---:|
| s1_1000 | ALL | selective_v2_compact | 12.61 | 18.0 | 34.889 |
| s1_1000 | India | selective_v2_compact | 11.67 | 18.0 | 15.413 |
| s1_1000 | US | selective_v2_compact | 13.27 | 18.0 | 19.476 |
| s1_1000 | S2 | selective_v2_compact | 12.84 | 18.0 | 16.591 |
| s1_1000 | S2|India | selective_v2_compact | 11.86 | 18.0 | 7.303 |
| s1_1000 | S2|US | selective_v2_compact | 13.52 | 18.0 | 9.288 |
| s1_1000 | S3 | selective_v2_compact | 12.38 | 18.0 | 18.298 |
| s1_1000 | S3|India | selective_v2_compact | 11.48 | 18.0 | 8.110 |
| s1_1000 | S3|US | selective_v2_compact | 13.01 | 18.0 | 10.187 |

Peak RSS is the cumulative process high-water mark, including builds if performed in this process. Query timing excludes S1/GT loading and metric aggregation. The 5k batch may benefit from caches warmed by 1k. Cap-hit rates use S1/source queries as denominator; candidate contribution sets overlap; token-only is incremental to exact passes. Full diagnostics and slices are in the CSV tables.