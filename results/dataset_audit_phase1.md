# Phase-1 dataset audit

This report profiles only the competition-provided TSVs under `data/raw`. The audit does not alter raw data or use external information.

## Dataset inventory

| Split | Source | Rows | Size (bytes) |
|---|---:|---:|---:|
| train | S1 | 2,206,821 | 210,069,713 |
| train | S2 | 5,034,616 | 489,301,488 |
| train | S3 | 5,285,603 | 503,705,637 |
| test | S1 | 1,732,544 | 175,022,086 |
| test | S2 | 4,887,273 | 509,456,422 |
| test | S3 | 5,082,316 | 506,002,772 |

## Key findings

- Ground truth contains 2,206,821 Source-1 entities and 7,638,365 positive links.
- Singletons: 123,247 (5.585%).
- Matched-source coverage: S2 only 143,029, S3 only 164,498, both sources 1,776,047.
- Positive links: S2 3,693,619; S3 3,944,746.
- France is absent from training and accounts for 14.975% of test Source-1 entities and 14.480% of all test source rows. It must be treated as an open-set country value.
- train S2 has 168,967 missing addresses (3.356%).
- train S3 has 175,916 missing addresses (3.328%).
- test S2 has 129,408 missing addresses (2.648%).
- test S3 has 136,098 missing addresses (2.678%).
- Within-file duplicate entity-ID excess rows: 0.
- Exact name/address duplicate statistics are in `results/tables/exact_duplicates.csv`; empty text values are excluded.

## Quality interpretation

- Entity IDs, schemas, country labels, and required fields are validated during chunked reads; a violation stops the audit.
- Missing Source-2/3 addresses are expected input sparsity but will matter to later matching work. This audit does not impute or normalize them.
- Exact repeated names or addresses are not automatically data errors in entity resolution; they indicate ambiguity and should not be treated as unique identifiers.
- The France test-only shift is a high-confidence distribution shift, not evidence of bad data. Later pipelines must avoid a closed `{US, India}` country assumption.

## Reproducibility and resource behavior

- Source chunk size: 200,000 rows.
- Exact duplicate counts use one temporary SQLite database per source file and remove it after that file is summarized. Peak disk use is therefore bounded by one source's distinct strings rather than the entire corpus.
- Length quantiles are exact character-count quantiles over non-empty text.
- Machine-readable evidence is under `results/tables/`.
