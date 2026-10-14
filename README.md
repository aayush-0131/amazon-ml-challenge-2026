# Amazon ML Challenge 2026 — Business Entity Resolution

Team **BlackList** (Aayush Jha and Anish Singh) built a precision-first business entity resolution system for Amazon ML Challenge 2026. This repository documents the frozen Round-1 solution, experiment evidence, and submission limitations.

## Objective

For each deduplicated Source-1 business entity, identify zero, one, or many
corresponding Source-2 / Source-3 records.

Primary metric: macro F0.5 at Source-1 entity level.

## Engineering overview

The frozen EXP007 matcher follows an auditable pipeline from entity normalization through candidate generation and entity-level decisions:

1. **Audit and normalize:** inspect supplied business records, standardize names and addresses, and preserve entity-level split boundaries.
2. **Generate candidates:** use blocking/indexed retrieval to form plausible cross-source pairs without an exhaustive all-pairs join.
3. **Score matching evidence:** compute **83 ordered pairwise features** and train a scikit-learn **HistGradientBoostingClassifier**.
4. **Make precision-oriented decisions:** use separately frozen Source-2 and Source-3 score thresholds, with macro F0.5 evaluated at the Source-1 entity level.
5. **Scale inference and audit output:** process **1,732,544** Source-1 entities and **326,418,278** candidate links, selecting **5,753,590** match links in the final TEST audit.

The model and decision policy are documented in [EXP007](experiments/EXP007.md); the experiment status, splits, and validation evidence are in [Project Status](docs/PROJECT_STATUS.md).

## Round 1 final result

The team froze **EXP007** as its final Round-1 matcher. Public leaderboard performance and untouched *local* evaluation performance are different measurements:

- public leaderboard macro F0.5: **0.916321**
- untouched EVALUATION macro F0.5: **0.9384071620652703**
- macro precision: **0.9650289021**
- macro recall: **0.8835666997**
- model: scikit-learn HistGradientBoostingClassifier
- ordered pairwise features: **83**
- frozen thresholds: **S2 0.965**, **S3 0.956**
- production commit: `63100d932080e2a783e7dc43eab618abc0d92ad0`
- archival production branch: `submission/round1-exp007`

The numbers above describe the recorded Round-1 experiment, **not** a claim of a private leaderboard result, finalist ranking, or organizer-accepted submission.

**Outcome and submission limitations:** Team BlackList did **not** appear in the published Top 50. The complete archive was larger than the upload limit; the uploaded fallback omitted the required candidate-pairs TSV. The upload receipt is confirmed, but organizer technical acceptance is **unknown**. The exact distinction is preserved in the [submission evidence register](docs/SUBMISSION_EVIDENCE_2026.md) and [postmortem](docs/POSTMORTEM_2026.md).

## What can be reviewed

- [EXP007 model, decision policy and experiment commands](experiments/EXP007.md)
- [Project status, validation results and frozen commit](docs/PROJECT_STATUS.md)
- [Submission evidence, artifact hashes and limitations](docs/SUBMISSION_EVIDENCE_2026.md)
- [Postmortem and lessons learned](docs/POSTMORTEM_2026.md)

The repository intentionally excludes raw competition data, large model/pair artifacts, SQLite indexes, and generated TEST outputs. Therefore its code and documentation are **inspectable but not a standalone byte-for-byte reproduction** of the recorded scores; reproduction requires authorized challenge data, the specified environment, and adequate compute.

## Post-submission retrospective (3 October 2026)

**Final status:** EXP007 frozen; code-upload receipt confirmed on 2 October; **not named in the published Top 50**. The public score is not a private leaderboard measurement, and the successful upload does **not** establish approval of the incomplete fallback package. We do not have an organizer-provided explanation for the selection outcome.

Read these documentation artifacts before using this repository as a reference:

- [Full evidence-based postmortem](docs/POSTMORTEM_2026.md): measured bottlenecks, technical tradeoffs, competition outcome and unresolved uncertainties.
- [Published Top 50 team audit](docs/TOP50_RESEARCH_2026.md): all 50 IDs/names from the results sheet; a source-backed technical comparison with Androids; **unknown** techniques are explicitly left unknown.
- [Submission and reproducibility evidence](docs/SUBMISSION_EVIDENCE_2026.md): archive sizes, output hashes, omission disclosure, proof boundary and unfinished manual AWS/evidence checks.
- [Public lessons learned](docs/FUTURE_COMPETITION_PROTOCOL.md): concise, generally applicable research and reproducibility principles. Detailed internal planning is maintained separately in private team storage.

**Data and reproduction boundary:** The private competition inputs, full generated outputs, binary model bundles and indexes are deliberately not published. The frozen code and instructions are inspectable, but **the GitHub repository by itself is not a byte-for-byte runnable reproduction of EXP007**. No post-result competitor technique has been inserted into the frozen submission.

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
