# Amazon ML Challenge 2026 — Business Entity Resolution

Competition solution repository for the Business Entity Resolution challenge.

## Objective

For each deduplicated Source-1 business entity, identify zero, one, or many
corresponding Source-2 / Source-3 records.

Primary metric: macro F0.5 at Source-1 entity level.

## Development principles

1. Competition-provided data only.
2. Raw data is never committed to Git.
3. Entity-level validation.
4. Blocking recall and matcher quality measured separately.
5. Precision-first decision making because F0.5 penalizes false merges.
6. Every leaderboard submission must map to a reproducible experiment.
7. Amazon's official validator must pass before submission.

## Pipeline

data audit
→ normalization
→ blocking / candidate generation
→ pairwise features
→ matching model
→ threshold optimization
→ entity-level predictions
→ Amazon submission validation
