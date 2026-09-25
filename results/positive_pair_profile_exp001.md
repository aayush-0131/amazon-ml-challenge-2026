# EXP001 positive-pair lexical profile

Deterministic sample: 100,000 positive links; seed: 2026.

## Overall exact-match rates

- Raw name exact: 4.615%
- Normalized name exact: 21.520%
- Raw address exact: 2.214%
- Normalized address exact: 8.242%
- Target address missing: 4.494%

## Overall similarity landmarks

| Metric | p05 | p50 | p95 |
|---|---:|---:|---:|
| name_ratio | 10.811 | 87.500 | 100.000 |
| name_token_set_ratio | 10.714 | 100.000 | 100.000 |
| name_token_sort_ratio | 10.714 | 89.474 | 100.000 |
| address_ratio | 28.235 | 82.759 | 100.000 |
| address_token_set_ratio | 55.738 | 93.617 | 100.000 |
| address_token_sort_ratio | 31.427 | 87.379 | 100.000 |
| name_token_jaccard | 0.000 | 0.667 | 1.000 |
| address_token_jaccard | 0.111 | 0.636 | 1.000 |
| digit_token_overlap | 0.000 | 1.000 | 1.000 |
| postal_like_agreement | 0.000 | 0.000 | 1.000 |

Detailed source, country, and source×country results are in `results/tables/positive_pair_profile_rates.csv` and `positive_pair_profile_quantiles.csv`.

Runtime: 35.1 seconds. Peak RSS: 372.7 MB.
