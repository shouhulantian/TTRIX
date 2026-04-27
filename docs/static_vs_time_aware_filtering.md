# Static vs Time-Aware Filtering — Zero-Shot TRIX on Temporal KGs

Pretrained checkpoint: `entity_prediction.pth` (TRIX, FB15k237 + WN18RR + CoDEx-Medium).
All numbers are zero-shot (no fine-tuning, `--epochs 0`).

## Results

| Dataset           | Filter      | MR     | MRR       | Hits@1   | Hits@3   | Hits@10  | Hits@10_50 |
|-------------------|-------------|--------|-----------|----------|----------|----------|------------|
| GDELTIndT_100     | static      | 13.20  | **0.584** | 0.4868   | 0.6303   | 0.7727   | 0.9131     |
| GDELTIndT_100     | time-aware  | 28.97  | **0.272** | 0.1641   | 0.2964   | 0.4907   | 0.8148     |
| ICEWS14           | static      | 129.01 | **0.539** | 0.4486   | 0.5843   | 0.7140   | 0.9738     |
| ICEWS14           | time-aware  | 135.12 | **0.371** | 0.2457   | 0.4270   | 0.6182   | 0.9736     |
| ICEWS0515         | static      | 105.10 | **0.438** | 0.3440   | 0.4734   | 0.6279   | 0.9882     |
| ICEWS0515         | time-aware  | 120.07 | **0.280** | 0.1711   | 0.3162   | 0.4946   | 0.9878     |

## Interpretation

**Static filtering severely overestimates zero-shot transfer to temporal KGs.**
The "Static" rows in the table are the same numbers I was reporting earlier as
"TRIX zero-shot on ICEWS / GDELT". The drop-on-switch-to-time-aware is large:
MRR loses 31% (ICEWS14), 36% (ICEWS0515), 53% (GDELTIndT_100). All three
benchmarks have the same root cause:

When timestamps are stripped, the same `(h, r, t)` quadruple appears at many
different timestamps and collapses into duplicate edges in the static graph.
Standard (time-blind) filtering then masks out **all** other true tails
`t' ≠ t` for a given `(h, r)` regardless of timestamp — but those `t'`s are
*not* true at the query's timestamp `ts`. They should have been valid
negatives. Removing them shrinks the negative pool and inflates the rank of
`t`. Time-aware filtering only masks `t'` such that `(h, r, t', ts)` exists,
preserving the proper negative pool.

A back-of-envelope from ICEWS14 confirms the asymmetry is structural and
*not* a quirk of the model:

```
mean true tails per (h, r)         = 2.19   (over all timestamps)
mean true tails per (h, r, ts)     = 1.11   (at a single timestamp)
```

So static filtering removes ~2× more candidates from the negative pool than
time-aware does. The difference shows up directly in the MRR gap.

## Residual leak

Time-aware filtering fixes the *filter*, but the *message-passing graph* is
still the union of all timestamps. So the model can still traverse a leaked
`(h, r, t)` edge in one hop even when the query is at a different timestamp.
Per-quadruple overlap of test against the inference graph (with timestamps):

- GDELTIndT_100: 7,376 / 28,851 = **25.6%** of unique test `(h, r, t, ts)`
  appear verbatim in `msg.txt`.

The corresponding triple-level (timestamp-stripped) overlap was 69.6% / 81%
per-query. So time-aware filtering eliminates the bulk of the leak but a
non-trivial residual remains. To get a fully clean static-baseline number
the inference graph would need to be deduplicated against the test set at the
quadruple level.

## How to reproduce

```bash
# ICEWS14 (transductive, ~3 min)
sbatch sbatch_icews14_temporal.sh

# ICEWS0515 (transductive, ~22 min — needs >30 min slurm budget)
sbatch sbatch_icews0515_temporal_only.sh

# GDELTIndT_100 (fully inductive, ~3 min)
sbatch sbatch_gdelt_indt100_temporal.sh
```

Each script runs the static config first (when bundled) and the temporal
config second. The temporal variants (`TemporalICEWS14`, `TemporalICEWS0515`,
`GDELTIndT100Temporal`) keep `edge_time` / `target_edge_time` so
`run_entity.py` dispatches `temporal_strict_negative_mask` instead of
`strict_negative_mask`. Static and temporal classes share the same raw data
and the same checkpoint — only the filter changes.

## Job IDs

- ICEWS14 static + time-aware: `3997060` (3 m 30 s, dev_accelerated)
- ICEWS0515 static: `3997061` (timed out at 30 m after static finished)
- ICEWS0515 time-aware: `3997104` (22 m 20 s, dev_accelerated, 1 h budget)
- GDELTIndT_100 static + time-aware: `3997054` (2 m 43 s)
