# EXP002 pre-index retrieval runtime evidence

The initial EXP002 query-side multi-pass retrieval run was intentionally stopped
on 2026-09-25. It streamed every S2/S3 target row while maintaining query-side
indexes for the 20,000-S1 development subset. In particular, the default
character n-gram fallback evaluated target-side character grams during that
source scan.

- The run entered the S2 scan at approximately 09:51 IST.
- At 14:50 IST, the Python process still had `train_source2.tsv` open and the
  retrieval artifact open for writing; S2 had not completed.
- The retrieval artifact was still 0 bytes because it is written only after a
  source-level scan finalizes, so no partial candidate result is being treated
  as a measurement.
- The process was intentionally terminated. It did not report an exception or
  a completed stage-level peak RSS value.

Conclusion: a query-side design that scans the multi-million-row sources to
retrieve candidates is not acceptable for EXP002 production use. The revised
architecture must build a reusable source-side index once, then query many S1
records without rescanning S2/S3. This document preserves the negative runtime
finding; it is not a successful blocker benchmark.
