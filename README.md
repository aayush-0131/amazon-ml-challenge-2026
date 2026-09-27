# Amazon ML Challenge 2026 — Business Entity Resolution

Competition solution repository for the Business Entity Resolution challenge.

## Objective

For each deduplicated Source-1 business entity, identify zero, one, or many
corresponding Source-2 / Source-3 records.

Primary metric: macro F0.5 at Source-1 entity level.

## Round 1 final result

Team **BlackList** submitted **EXP007** as the final Round-1 matcher.

- public leaderboard macro F0.5: **0.916321**
- untouched EVALUATION macro F0.5: **0.9384071620652703**
- macro precision: **0.9650289021**
- macro recall: **0.8835666997**
- model: scikit-learn HistGradientBoostingClassifier
- ordered pairwise features: **83**
- frozen thresholds: **S2 0.965**, **S3 0.956**
- production commit: `63100d932080e2a783e7dc43eab618abc0d92ad0`
- archival production branch: `submission/round1-exp007`

The final TEST audit covered 1,732,544 Source-1 entities, 326,418,278
candidate links and 5,753,590 selected match links. Generated competition
outputs, raw challenge data, SQLite indexes and large model/pair artifacts remain
ignored by Git. The verified Round-2 ZIP is preserved outside the repository.

See [EXP007](experiments/EXP007.md) and [Project Status](docs/PROJECT_STATUS.md)
for the final experiment record.

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

## Phase-1 audit and evaluation foundation

Phase 1 provides the dataset audit, ground-truth parsing, entity-level split and
exact competition metric. Later experiment entry points are documented below;
full inference always requires explicit authorization.

The competition dataset must be available through the ignored `data/raw` symlink:

```text
data/raw/train/train_source1.tsv
data/raw/train/train_source2.tsv
data/raw/train/train_source3.tsv
data/raw/train/train_ground_truth.tsv
data/raw/test/test_source1.tsv
data/raw/test/test_source2.tsv
data/raw/test/test_source3.tsv
```

Create the environment and install the pinned dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
```

Run the complete, chunked audit (including disk-backed exact duplicate counts):

```bash
.venv/bin/python scripts/audit_data.py
```

The concise report is written to `results/dataset_audit_phase1.md`; machine-readable
CSV tables are written to `results/tables/`. The audit reads but never writes the raw
competition files. By default it processes 200,000 rows per chunk and uses temporary
SQLite storage for exact high-cardinality duplicate counts. Use `--temp-dir PATH` if
the system temporary volume is space-constrained.

Run all unit tests:

```bash
.venv/bin/python -m pytest -q
```

## EXP001 lexical baseline

Profile a deterministic 100,000-link sample of training positives:

```bash
.venv/bin/python scripts/profile_positive_pairs.py \
  --sample-size 100000 \
  --seed 2026
```

Run the memory-constrained M2/8 GB experiment configuration (20,000 S1 entities):

```bash
.venv/bin/python scripts/run_exp001.py \
  --config configs/exp001_m2_8gb.json
```

The default `configs/exp001.json` requests 100,000 S1 entities for machines with
more headroom. Both configurations use entity-level sampling only. Candidate rows
and validation predictions are derived training artifacts under `results/artifacts/`
and are intentionally ignored by Git; aggregate metrics and the exact subset IDs are
saved under `results/tables/`.

## EXP003 indexed reranker and inference

Blocker search is closed: compact_14 is the frozen rule fallback; compact_18
provides pass-eligible candidates without the old total cap for learned scoring.
See [EXP003](experiments/EXP003.md) for the fixed 20k TRAIN command, exact
FIT/TUNE/EVALUATION IDs policy, bundle metadata, four-worker AWS smoke commands,
restart/merge behavior and official output contract. No index construction occurs
in these entry points:

```bash
.venv/bin/python scripts/train_exp003.py --help
.venv/bin/python scripts/infer_exp003.py run --help
.venv/bin/python scripts/infer_exp003.py merge --help
.venv/bin/python -m pytest -q
git diff --check
```

EXP003 is retained as the first learned-reranker baseline. Later EXP006 and
EXP007 experiments superseded it for Round 1; see the final-result section above.
Generated pair matrices/checkpoints/TSVs remain ignored; preserve aggregate
experiment evidence separately.
