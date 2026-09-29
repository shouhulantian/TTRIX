# Experiments log

Rolling record of experimental results relevant to the TTRIX paper.
Keyed by SLURM job ID for traceability. Update when new runs land.

Convention: "old filter" = FITTER's shipped `strict_negative_time_mask`
which built filter keys from `train_data.target_edge_index` only. "new
filter" (commit `8e75329` on `feature/rolling-single-step-eval`) reads
keys from `data.edge_index` directly, so filtered_data can pack
train+valid+test target quadruples — matching the Bordes filtered-MRR
convention.

---

## FITTER on YAGOInd — zero-shot single_step hyperparam probe

Task: `TransductiveInference`, dataset class `YAGOIndForecast`
(single-vocab chronological forecast), data root
`FITTER/source_code/kg-datasets/YAGOInd/raw/`.

| job   | ckpt      | α   | w | filter | test MRR | test H@1 | test H@10 | valid MRR |
|:------|:----------|----:|--:|:------:|---------:|---------:|----------:|----------:|
| 87621 | ICEWS14   | 1.0 | 1 | old    | 0.6252   | 0.4910   | 0.8720    | 0.6119    |
| 87872 | ICEWS14   | 1.0 | 2 | old    | 0.6182   | 0.4870   | 0.8580    | 0.5960    |
| 87873 | ICEWS14   | 0.8 | 1 | old    | 0.6439   | 0.5134   | 0.8780    | 0.6224    |
| 87933 | ICEWS14   | 0.8 | 2 | old    | 0.6326   | 0.5030   | 0.8700    | 0.6114    |
| 87877 | ICEWS0515 | 1.0 | 1 | old    | 0.6224   | 0.4910   | 0.8650    | 0.5904    |
| 87878 | GDELT     | 1.0 | 1 | old    | 0.6325   | 0.5024   | 0.8670    | 0.6092    |
| **93460** | **ICEWS14** | **1.0** | **1** | **new** | **0.8792** | **0.8387** | **0.9320** | **0.8334** |

Notes:
- The +25pt jump between 87621 (old filter) and 93460 (new filter) is
  fully explained by within-test (h, r, τ) tail clusters (59% of test
  rows share their (h, r, τ) with ≥ 1 other test row, avg cluster size
  2.73). Under the new filter those cluster-mates are correctly masked
  as known-true tails at query time. Not a leak.
- α, w sensitivity is small (~1–2pt). Ckpt choice is small (~1pt) —
  GDELT.pth slightly beats ICEWS14/ICEWS0515 on YAGOInd zero-shot.
- All runs on FITTER commit `8e75329` after the filter fix; earlier
  numbers reflect the shipped-code (old-filter) protocol.

---

## FITTER on WIKI IndT sweep — zero-shot single_step, α=1, w=1

Ckpt: `FITTER/ckpts/ICEWS14.pth`. Dataset class `WIKIIndT_*` (my
`MsgAwareForecastDataset`, INGRAM disjoint-vocab layout). Raw data at
`TTRIX/datasets/WIKIIndT_*/raw/`.

| variant             | filter    | test MRR | test H@1 | test H@10 | valid MRR | job    |
|:--------------------|:----------|---------:|---------:|----------:|----------:|:-------|
| WIKIIndT_25_inter   | new       | 0.6904   | 0.6300   | 0.7975    | 0.9297    | 93461_1 |
| WIKIIndT_50_inter   | new       | 0.7491   | 0.7077   | 0.8234    | 0.9430    | 93461_3 |
| WIKIIndT_75_inter   | new       | 0.8200   | 0.7967   | 0.8588    | 0.9288    | 93461_5 |
| WIKIIndT_100_inter  | new       | 0.9572   | 0.9545   | 0.9608    | 0.9522    | 93461_7 |
| WIKIIndT_25_extra   | old ≈ new | 0.9693   | 0.9645   | 0.9749    | 0.8921    | 88112_2 |
| WIKIIndT_50_extra   | old ≈ new | 0.9299   | 0.9174   | 0.9642    | 0.9010    | 88112_4 |
| WIKIIndT_75_extra   | old ≈ new | 0.9698   | 0.9648   | 0.9800    | 0.8942    | 88112_6 |
| WIKIIndT_100_extra  | old ≈ new | 0.9629   | 0.9549   | 0.9727    | 0.9004    | 88112_8 |

Notes:
- For `_extra` variants, msg times and test times share only one
  boundary timestamp (e.g. WIKIIndT_25_extra: msg=[0..177],
  test=[177..231]). Under time-aware filtering, only that single
  overlapping day differs between old and new filter — effect on
  aggregate MRR is < 0.005. Old-filter numbers stand as-is.
- The `_extra` variants have very small triple vocabularies (~200
  unique (h, r, t) triples in 4844 rows for WIKIIndT_25_extra), so
  MRR ≈ 0.97 is driven by heavy recurrence, not by cross-domain
  discrimination power. See "Notes on interpretation" below.

### Paper comparison (WIKI IndT interpolation, from the paper's Table (c))

Reproducing the paper's WIKI _inter table with FITTER row inserted:

| Method       | p₂₅ MRR | p₅₀ MRR | p₇₅ MRR | p₁₀₀ MRR |
|:-------------|--------:|--------:|--------:|---------:|
| Recurrency   | .477    | .524    | .572    | .653    |
| Ultra        | **.838**| **.835**| .851    | .856    |
| Ultra+Method | .811    | .834    | .832    | .842    |
| **FITTER**   | .690    | .749    | .820    | .957    |
| Trix         | .623    | .760    | .858    | **.962**|
| Trix+Method  | .507    | .650    | **.872**| **.991**|

FITTER (mine) at p₁₀₀ inter (.957) essentially matches Trix baseline
(.962). At smaller p it tracks Trix's climb.

### Paper comparison (WIKI IndT extrapolation, from the paper's Table (d))

| Method       | p₂₅ MRR | p₅₀ MRR | p₇₅ MRR | p₁₀₀ MRR |
|:-------------|--------:|--------:|--------:|---------:|
| Recurrency   | .416    | .528    | .598    | .588    |
| Ultra        | .724    | .721    | .689    | .737    |
| Ultra+Method | .717    | .727    | .714    | .749    |
| **FITTER**   | **.969**| **.930**| **.970**| **.963**|
| Trix         | .790    | .916    | .959    | .931    |
| Trix+Method  | .795    | .888    | .874    | .966    |

FITTER dominates on WIKI _extra. Interpretation caveat: the extra
splits have very small unique-triple vocabularies with heavy
duplication (up to 24× per triple for _25_extra), so single_step
rolling turns most queries into near-trivial "predict last τ's tail"
tasks. Any structural method that exploits time-local recurrence
scores near-perfect. Not a strong signal of FITTER's cross-domain
generalization.

---

## GDELT IndT sweep — zero-shot single_step, α=1, w=1

Two runs, matched protocol. Ckpts differ.

### FITTER (ICEWS14.pth)

Dataset class `GDELTIndT_*` (MsgAwareForecastDataset). Raw data at
`TTRIX/datasets/GDELTIndT_*/raw/`. Note: `GDELTIndT_100_inter` raw dir
doesn't exist; `GDELTIndT_100` (different-lineage build) substitutes.

| variant                | filter    | test MRR | test H@1 | test H@10 | valid MRR | job     |
|:-----------------------|:----------|---------:|---------:|----------:|----------:|:--------|
| GDELTIndT_25_inter     | new       | 0.2664   | 0.1734   | 0.4479    | 0.2633    | 93554_1 |
| GDELTIndT_50_inter     | new       | 0.2617   | 0.1711   | 0.4354    | 0.2656    | 93554_3 |
| GDELTIndT_75_inter     | new       | 0.2642   | 0.1779   | 0.4246    | 0.2569    | 93554_5 |
| GDELTIndT_100 (p₁₀₀ substitute) | new | 0.2754   | 0.1874   | 0.4418    | 0.2672    | 93523_7 |
| GDELTIndT_25_extra     | old ≈ new | 0.2545   | 0.1643   | 0.4281    | 0.2593    | 93437_2 |
| GDELTIndT_50_extra     | old ≈ new | 0.2646   | 0.1779   | 0.4284    | 0.2635    | 93437_4 |
| GDELTIndT_75_extra     | old ≈ new | 0.3051   | 0.2202   | 0.4615    | 0.2560    | 93437_6 |
| GDELTIndT_100_extra    | old ≈ new | 0.2674   | 0.1765   | 0.4436    | 0.2568    | 93437_8 |

### TRIX GRATE (27204 ep9, model_epoch_10.pth)

Config: `TTRIX/config/eval_27204_ep9_gdelt_extra_single_step.yaml`
(also used for _inter). Task `InductiveInference`, TTRIX's own
`test_rolling` + `temporal_strict_negative_mask` filter path.

| variant                        | test MRR | test H@1 | test H@10 | job    |
|:-------------------------------|---------:|---------:|----------:|:-------|
| GDELTIndT_25_inter_Temporal    | 0.3560   | 0.2389   | 0.5925    | 93548_1 |
| GDELTIndT_25_extra_Temporal    | 0.2952   | 0.1882   | 0.5034    | 93548_2 |
| GDELTIndT_50_inter_Temporal    | 0.3591   | 0.2405   | 0.5996    | 93548_3 |
| GDELTIndT_50_extra_Temporal    | 0.2845   | 0.1766   | 0.4976    | 93548_4 |
| GDELTIndT_75_inter_Temporal    | 0.3890   | 0.2695   | 0.6299    | 93548_5 |
| GDELTIndT_75_extra_Temporal    | 0.3084   | 0.1975   | 0.5273    | 93548_6 |
| GDELTIndT100Temporal           | 0.4254   | 0.2971   | 0.6910    | 93548_7 |
| GDELTIndT_100_extra_Temporal   | 0.2875   | 0.1816   | 0.5013    | 93548_8 |

Prior TRIX GRATE runs on the 4 _extra variants (jobs 28537–28540,
2026-05) reproduce this session's numbers to 4 decimal places
(0.2952 vs 0.2953, 0.2845 vs 0.2846, 0.3084 vs 0.3084, 0.2875 vs
0.2875). Deterministic and stable across cache regeneration.

### GDELT collision structure (why the filter fix barely moves numbers)

Unlike YAGO (59% of test rows in multi-tail (h, r, τ) clusters, avg
cluster size 2.73), GDELT test rows are mostly unique per (h, r, τ):

| dataset             | rows in multi-tail keys | avg row cluster size |
|:--------------------|------------------------:|---------------------:|
| YAGOInd test        | 59.0%                   | 2.73                 |
| GDELTIndT_25_inter  | 20.9%                   | 1.30                 |
| GDELTIndT_50_inter  | 24.2%                   | 1.40                 |
| GDELTIndT_75_inter  | 25.0%                   | 1.41                 |
| GDELTIndT_100       | 32.1%                   | 1.54                 |

So GDELT's `_inter` MRR (~0.26–0.28 FITTER, ~0.35–0.42 TRIX) is closer
to raw model discrimination power than YAGO's `_inter` MRR.

---

## Notes on interpretation

- The 0.879 YAGOInd and 0.957 WIKI_100_inter numbers under the new
  filter are legitimate under Bordes filtered-MRR convention. They
  reflect (a) genuine model ability + (b) within-test recurrence that
  the standard filter treats as "known truths".
- Small-vocabulary `_extra` splits (WIKIIndT_25_extra has only 202
  unique triples) are near-saturated by any structural method that
  exploits local recurrence. Treat 0.97 on those as a ceiling, not a
  headline.
- For fair FITTER-vs-TRIX comparison, both must use the same filter
  code path. My FITTER runs post-`8e75329` do; the paper's TRIX rows
  use TTRIX's own `temporal_strict_negative_mask` on `msg + targets`
  filter graph — same semantics on chronological _extra splits (msg
  times < test times → no additional τ matches), potentially
  looser-on-FITTER for _inter splits where msg overlaps test τ.

---

## Filter fix change log

- **2026-09-28**, FITTER commit `8e75329` on branch
  `feature/rolling-single-step-eval`:
  `strict_negative_time_mask` now reads filter keys from
  `data.edge_index / edge_type / time_type` directly. The
  `train_data` argument is deprecated. `filtered_data` in `run.py`
  __main__ packs the MP graph + all same-vocab target quadruples
  (Bordes standard). Effect: +25pt on YAGOInd (within-test clusters
  now masked), +7–29pt on WIKI IndT `_inter` variants (msg-at-τ
  facts now masked), ~0–2pt on GDELT IndT and WIKI _extra (few
  same-τ matches to add).
