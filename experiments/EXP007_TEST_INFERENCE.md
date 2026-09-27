# EXP007 frozen TEST inference

This is an execution runbook, not authorization to launch full TEST. The
untouched EVALUATION macro F0.5 was 0.9384071620652703. The frozen TUNE
policy is base HGB, S2 threshold 0.965, S3 threshold 0.956, no guardian,
exclusivity, calibration, or numeric penalty. TEST has no labels and must not
be used for policy selection.

Inference loads `results/exp007_tune/frozen.joblib` through EXP007's frozen
bundle validator. It uses that bundle's compact18 blocker and the existing
TEST source indexes, scores all eligible candidates with the 83-feature base
model, and applies the two frozen thresholds. No TRAIN identity check is run
against TEST inputs. Each worker uses a SQLite checkpoint and publishes
`matching_results.tsv`, `candidate_pairs.tsv`, then `done.json`. A valid
completed shard is reused without rescoring. Merging validates identities,
checksums, exact S1 order/coverage, and the match-subset-of-candidates rule.

## AWS checkout and preflight

Run from the repository root. Preserve TEST TSV timestamps relative to the
already-built `results/artifacts/test_source_index` metadata. Use the pinned
EXP007 environment. Check disk/RAM before starting four workers; each loads a
model and opens both country-scoped source indexes.

```bash
git switch exp/exp007-enhanced-matcher
git pull --ff-only origin exp/exp007-enhanced-matcher
python -m pip install -e .
shasum -a 256 src/business_entity_resolution/exp007.py src/business_entity_resolution/exp007_features.py
python -m pytest -q
python scripts/infer_exp007.py run --help
```

The protected hashes must be `0ff3836f1d53685004876c7f9d6392942894cd81d62440ff6537f03e12eb9774`
and `afeb9e66a42aa09209cb7799589630f836b2e8f1f7d294cacd26742a7b571959`.
The bundle loader fails closed on a checksum, code, feature, blocker, or
package-version mismatch. Do not bypass that check.

## A. 1000-S1 smoke, four parallel shards

This scores the first 1000 TEST S1 entities only. It does not run full TEST.
Use fresh paths; repeating exactly the same command resumes valid checkpoints.

```bash
mkdir -p results/exp007_test_smoke_1000_logs
pids=()
for shard in 0 1 2 3; do
  python -u scripts/infer_exp007.py run \
    --tune-dir results/exp007_tune --test-dir data/raw/test \
    --index-dir results/artifacts/test_source_index \
    --shard-dir results/exp007_test_smoke_1000_shards \
    --shards 4 --shard "$shard" --smoke-limit 1000 \
    > "results/exp007_test_smoke_1000_logs/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || exit 1; done
```

## B. 4000-S1 throughput benchmark, four parallel shards

Use distinct paths because smoke limit is part of the checkpoint identity.

```bash
mkdir -p results/exp007_test_benchmark_4000_logs
pids=()
for shard in 0 1 2 3; do
  python -u scripts/infer_exp007.py run \
    --tune-dir results/exp007_tune --test-dir data/raw/test \
    --index-dir results/artifacts/test_source_index \
    --shard-dir results/exp007_test_benchmark_4000_shards \
    --shards 4 --shard "$shard" --smoke-limit 4000 \
    > "results/exp007_test_benchmark_4000_logs/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || exit 1; done
python scripts/infer_exp007.py benchmark \
  --shard-dir results/exp007_test_benchmark_4000_shards \
  --test-dir data/raw/test --index-dir results/artifacts/test_source_index \
  --shards 4
```

The benchmark prints `per_shard_s1_per_second`,
`aggregate_s1_per_second`, and `projected_full_test_hours`. The projection is
`1732544 / aggregate_s1_per_second / 3600`; it excludes startup, merge, and
resource contention. Benchmark with four simultaneous workers on the intended
four-vCPU host. Check that memory pressure does not cause swapping.

## C. Merge and audit both smoke outputs

```bash
python scripts/infer_exp007.py merge \
  --shard-dir results/exp007_test_smoke_1000_shards \
  --test-dir data/raw/test --index-dir results/artifacts/test_source_index \
  --shards 4 --output-dir results/exp007_test_smoke_1000_merged
python scripts/infer_exp007.py audit \
  --output-dir results/exp007_test_smoke_1000_merged --test-dir data/raw/test
python scripts/infer_exp007.py merge \
  --shard-dir results/exp007_test_benchmark_4000_shards \
  --test-dir data/raw/test --index-dir results/artifacts/test_source_index \
  --shards 4 --output-dir results/exp007_test_benchmark_4000_merged
python scripts/infer_exp007.py audit \
  --output-dir results/exp007_test_benchmark_4000_merged --test-dir data/raw/test
```

The merged directories contain submission-format `matching_results.tsv` and
`candidate_pairs.tsv`, plus checksums/identity in `manifest.json`. The
competition validator expects full TEST coverage, so it is not appropriate
for these partial smoke outputs. After a separately authorized full run and
full merge, validate with:

```bash
python resources/validate_submission.py \
  --matching output/exp007/matching_results.tsv \
  --candidate output/exp007/candidate_pairs.tsv \
  --test-dir data/raw/test --check-ids
```

The full run requires `--allow-full-test` on every shard and on merge;
the default run is a 100-S1 smoke. The 1000 and 4000 runs above do not set
that flag. Do not run full TEST concurrently with EXP006 inference until
measured memory and disk headroom are known. SQLite checkpoints, shard TSVs,
merged TSVs, and the merge uniqueness database coexist; reserve several
times the measured 4000-S1 footprint, scaling candidate-pair text by the
full/smoke row ratio. Never reuse EXP003/EXP006 output directories.
