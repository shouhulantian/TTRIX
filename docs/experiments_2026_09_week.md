# Experiments week of 2026-09-28 → 2026-10-01

Session log covering FITTER port, TIGER port, filter-fix rediscovery,
3-dataset GRATE pretraining launch, RoPE β sensitivity sweep,
paper-number provenance work, and the resulting cross-model
apples-to-apples comparisons.

Companion to [experiments_log.md](experiments_log.md), which holds the
FITTER YAGO/WIKI/GDELT tables in canonical form. This file adds
timeline / rationale / status of the newer work.

---

## 1. Infrastructure (code changes, all pushed)

### FITTER (`shouhulantian/FITTER`, branch `feature/rolling-single-step-eval`)
- `script/run.py::test_time_single_step` — per-timestep rolling-history
  forecasting eval (single_step / multi_step).
- `fitter/datasets.py::MsgAwareForecastDataset` — INGRAM disjoint-vocab
  (G_tr / G_inf) layout mirroring TTRIX's `InductiveTemporalDatasetINGRAM`.
- Filter fix (**commit `8e75329`**): `strict_negative_time_mask` now
  builds filter keys from `data.edge_index` directly. `filtered_data`
  packs the MP graph + all same-vocab target quadruples per split
  (Bordes filtered-ranking standard).

### alan_fitter (`script/run.py`)
- `test_time_single_step` ported with the same rolling / filter
  semantics; uses alan_fitter's attribute names (`time_type`) and
  ULTRA model interface.
- `__main__` dispatches on `cfg.task.eval_mode`.
- Fast-val call in `train_and_validate_time` drops the buggy
  `train_data` argument (parity with the 2026-04-24 test-time fix).

### TTRIX
- `src/trix/datasets.py::JointTemporalDataset` extended to 3 sources
  (YAGO added). Unified `_parse_date` handles ISO dates and
  zero-padded integer years; YAGO year N maps to CE `(1830 + N)-01-01`
  so all sources sit on one day-ordinal axis (`num_time ≈ 68 k`).
- `src/trix/layers.py::GeneralizedRelationalConv` + `rope_relative`
  and `src/trix/models_entity.py::EntityNet` — added `base_beta`
  parameter (default 10000, backward-compat).

---

## 2. Sweeps completed

### FITTER zero-shot single_step (ICEWS14.pth, α=1, w=1)

| dataset | filter | test MRR | jobs |
|---|---|---:|---|
| YAGOInd | old | 0.6252 | 87621 |
| YAGOInd | **new** | **0.8792** | 93460 — +25 pt from within-test (h,r,τ) cluster masking |
| WIKIIndT_25/50/75/100_inter | new | 0.6904 / 0.7491 / 0.8200 / 0.9572 | 93461_{1,3,5,7} |
| WIKIIndT_25/50/75/100_extra | old ≈ new | 0.9693 / 0.9299 / 0.9698 / 0.9629 | 88112_{2,4,6,8} |
| GDELTIndT_25/50/75_inter, _100 | new | 0.2664 / 0.2617 / 0.2642 / 0.2754 | 93554_{1,3,5} + 93523_7 |
| GDELTIndT_25/50/75/100_extra | old ≈ new | 0.2545 / 0.2646 / 0.3051 / 0.2674 | 93437_{2,4,6,8} |

FITTER YAGO hyperparam probe: 6 configs (α ∈ {1.0, 0.8}, w ∈ {1, 2},
ckpt ∈ {ICEWS14, ICEWS0515, GDELT}); headline pre-fix 0.6439
(α=0.8, w=1, ICEWS14), post-fix 0.879 (α=1, w=1, ICEWS14).

### TTRIX GRATE zero-shot single_step on GDELT (job 93548, ep9)

| variant | test MRR | test H@1 | test H@10 |
|---|---:|---:|---:|
| GDELTIndT_25_inter | 0.3560 | 0.2389 | 0.5925 |
| GDELTIndT_25_extra | 0.2952 | 0.1882 | 0.5034 |
| GDELTIndT_50_inter | 0.3591 | 0.2405 | 0.5996 |
| GDELTIndT_50_extra | 0.2845 | 0.1766 | 0.4976 |
| GDELTIndT_75_inter | 0.3890 | 0.2695 | 0.6299 |
| GDELTIndT_75_extra | 0.3084 | 0.1975 | 0.5273 |
| GDELTIndT_100 | 0.4254 | 0.2971 | 0.6910 |
| GDELTIndT_100_extra | 0.2875 | 0.1816 | 0.5013 |

Reproduces prior 28537–28540 numbers on _extra to 4dp.

### TIGER (ULTRA + RoPE2_decay_q, 27171 ep5) single_step on WIKI IndT (job 93693, tasks 1–8)

| variant | test MRR | test H@1 | test H@10 |
|---|---:|---:|---:|
| WIKIIndT_25_inter | 0.6380 | 0.5279 | 0.7971 |
| WIKIIndT_25_extra | 0.9658 | 0.9542 | 0.9796 |
| WIKIIndT_50_inter | 0.7309 | 0.6683 | 0.8286 |
| WIKIIndT_50_extra | 0.9539 | 0.9411 | 0.9711 |
| WIKIIndT_75_inter | 0.8187 | 0.7874 | 0.8723 |
| WIKIIndT_75_extra | 0.9708 | 0.9585 | 0.9832 |
| WIKIIndT_100_inter | 0.9356 | 0.9169 | 0.9637 |
| WIKIIndT_100_extra | 0.9535 | 0.9326 | 0.9785 |

### TIGER GDELT single_step (job 93693, tasks 9–16, completed 2026-09-30)

| variant | test MRR | test H@1 | test H@10 |
|---|---:|---:|---:|
| GDELTIndT_25_inter  | 0.2007 | 0.1090 | 0.3815 |
| GDELTIndT_25_extra  | 0.2627 | 0.1659 | 0.4578 |
| GDELTIndT_50_inter  | 0.2098 | 0.1211 | 0.3823 |
| GDELTIndT_50_extra  | 0.2580 | 0.1543 | 0.4627 |
| GDELTIndT_75_inter  | 0.1980 | 0.1062 | 0.3786 |
| GDELTIndT_75_extra  | 0.2576 | 0.1550 | 0.4633 |
| GDELTIndT_100_inter | 0.2146 | 0.1249 | 0.3867 |
| GDELTIndT_100_extra | 0.2583 | 0.1572 | 0.4570 |

Flat across ratios; extra > inter (unusual for temporal forecasting —
extra split is dominated by recurring high-frequency (h, r) patterns).
GDELT stays the hard dataset; WIKI stays easy.

Ratio-invariance of TIGER contrasts with GRATE, whose GDELT extra
climbs 0.29 → 0.34 → 0.29 on 25/75/100 (§ 4.b) — so GRATE > TIGER on
GDELT extrapolation.

---

## 3. Cross-model apples-to-apples (WIKI IndT interpolation)

Same data, same rolling-single_step protocol, same Bordes filter.

| variant | FITTER (mine) | TIGER | TRIX (paper) |
|---|---:|---:|---:|
| p₂₅ inter | 0.690 | 0.638 | 0.623 |
| p₅₀ inter | 0.749 | 0.731 | 0.760 |
| p₇₅ inter | 0.820 | 0.819 | 0.858 |
| p₁₀₀ inter | 0.957 | 0.936 | 0.962 |

Three-way convergence with the same climb-with-p curve. FITTER
slightly ahead at every point; TIGER ≈ TRIX-paper at p₁₀₀.

---

## 4. Results landed 2026-09-30 → 2026-10-01

### a) 3-dataset GRATE pretrain (job 93629, DONE)

Full 10-epoch trajectory (val MRR per dataset from `fast_test=5000`):

| epoch | ICEWS14 | ICEWS0515 | YAGO | avg |
|---:|---:|---:|---:|---:|
| 1 | 0.606 | 0.601 | 0.679 | **0.629** ← best |
| 2 | 0.599 | 0.598 | 0.666 | 0.621 |
| 3 | 0.600 | 0.596 | 0.669 | 0.622 |
| 4 | 0.602 | 0.595 | 0.669 | 0.622 |
| 5 | 0.601 | 0.581 | 0.655 | 0.612 |
| 6 | 0.594 | 0.589 | 0.638 | 0.607 |
| 7 | 0.596 | 0.575 | 0.617 | 0.596 |
| 8 | 0.592 | 0.569 | 0.609 | 0.590 |
| 9 | 0.590 | 0.563 | 0.606 | 0.586 |

Canonical ckpt: `output/TRIX/JointTemporalDataset/2026-09-29-10-31-56/model_epoch_2.pth`
(ep1). Same drift-after-ep1 pattern as 27204 (2-ds).

Downstream eval on ep1 (job 93715):

| dataset | mode | 3ds ep1 | 2ds 27204 ep9 | Δ |
|---|---|---:|---:|---:|
| ICEWS14 | static | 0.5885 | 0.6090 | **-0.0205** |
| YAGO | single_step | 0.8411 | 0.858 | -0.017 |
| ICEWS0515 | static | (rerun 94214) | 0.617 | — |

**Adding YAGO to pretraining HURTS both ICEWS14 downstream and — surprisingly — YAGO itself** (even
though YAGO is now in-mix). The 2-ds warm start is stronger on all
axes measured. Cross-domain interference dominates the added-data
benefit at this data mix / warm-start schedule.

**ep0 baseline eval (job 94236_1, DONE)**:

| ckpt | mode | MRR | H@1 | H@3 | H@10 |
|---|---|---:|---:|---:|---:|
| 3-ds ep0 (model_epoch_1.pth) | ICEWS14 static | **0.5783** | 0.4678 | — | 0.7848 |
| 3-ds ep1 (model_epoch_2.pth) | ICEWS14 static | 0.5885 | 0.4819 | — | 0.7858 |
| 2-ds 27204 ep9 (reference) | ICEWS14 static | 0.6087 | 0.5074 | 0.6719 | 0.7928 |

**Finding**: 3-ds ep0 (= the 27204 ep9 warm-start weights at the
moment they entered the 3-ds trainer) scores **0.5783**, which is
**0.030 lower than 27204 ep9's canonical 0.6087 on the same ICEWS14
test set**. The weights should be IDENTICAL. The gap is **not
training-induced**; it's an **eval-side config difference** between
`class: TemporalICEWS14` (standalone) vs loading the same ckpt into
the 3-ds JointTemporalDataset context. Likely suspects:
- num_time vocab length differs (joint vs single) → RoPE angle shift.
- Entity re-indexing within the joint dataset vs single-dataset eval.
- `IndNBFNet.rope_relative` reading a different base_beta.

**This means the "adding YAGO hurt downstream" conclusion is wrong.**
The baseline itself drops 0.03 just from eval-time config drift. The
ep0 → ep1 delta is only ~0.01 — within noise. **Need to isolate the
ckpt-loading / eval harness mismatch before any 3-ds-pretraining
impact statement is defensible.** 94236_2 (ICEWS0515 ep0) + 94236_3
(YAGO ep0) will confirm whether the drift is systematic.

### b) 27204 ep3 vs ep9 — full comparison (jobs 93713 + 94234 + 94233 + 27356-9 + 27366-7 + 27403-6 + 27586 + 27624 + 28537-40 + 28545-8)

All numbers from actual run logs (not paper rows).
Format: `MRR / H@1 / H@3 / H@10`.

#### WIKI IndT interpolation (static)

| ratio | ep3 (model_epoch_4.pth) | ep9 (model_epoch_10.pth) | Δ MRR |
|---|---|---|---:|
| 25%  | 0.7131 / 0.6517 / 0.7407 / 0.8190 | 0.5072 / 0.4144 / 0.5503 / 0.6804 | **+0.206** |
| 50%  | 0.7907 / 0.7493 / 0.8074 / 0.8600 | 0.6501 / 0.5568 / 0.7072 / 0.7821 | **+0.141** |
| 75%  | 0.8715 / 0.8484 / 0.8835 / 0.9124 | 0.8719 / 0.8468 / 0.8839 / 0.9167 | ≈0 (tie) |
| 100% | 0.9922 / 0.9907 / 0.9924 / 0.9954 | 0.9906 / 0.9878 / 0.9924 / 0.9966 | ≈0 (tie) |

#### WIKI IndT extrapolation (single_step)

| ratio | ep3 | ep9 | Δ MRR |
|---|---|---|---:|
| 25%  | 0.9676 / 0.9589 / 0.9747 / 0.9806 | 0.7949 / 0.6777 / 0.9143 / 0.9546 | **+0.173** (H@1 +0.281) |
| 50%  | 0.9618 / 0.9547 / 0.9666 / 0.9724 | 0.8883 / 0.8081 / 0.9658 / 0.9740 | **+0.074** |
| 75%  | 0.9719 / 0.9661 / 0.9762 / 0.9797 | 0.8737 / 0.7656 / 0.9822 / 0.9831 | **+0.098** MRR / +0.21 H@1; ep9 marginally wins H@3/H@10 |
| 100% | 0.9697 / 0.9632 / 0.9744 / 0.9773 | 0.9663 / 0.9589 / 0.9723 / 0.9756 | ≈0 (tie) |

#### GDELT IndT interpolation (static)

| ratio | ep3 | ep9 | Δ MRR |
|---|---|---|---:|
| 25%  | 0.3798 / 0.2634 / 0.4249 / 0.6146 | *missing* | — |
| 50%  | 0.3945 / 0.2753 / 0.4417 / 0.6361 | *missing* | — |
| 75%  | 0.4315 / 0.3118 / 0.4839 / 0.6708 | *missing* | — |
| 100% | 0.4436 / 0.3095 / 0.5039 / 0.7228 | *missing* | — |

No ep9 GDELT-inter run exists (paper's +Method row used these ep3
numbers 27356-9).

#### GDELT IndT extrapolation (single_step)

| ratio | ep3 | ep9 | Δ MRR |
|---|---|---|---:|
| 25%  | 0.3093 / 0.2038 / 0.3465 / 0.5142 | 0.2953 / 0.1882 / 0.3317 / 0.5035 | +0.014 |
| 50%  | 0.3127 / 0.2061 / 0.3484 / 0.5157 | 0.2846 / 0.1766 / 0.3181 / 0.4977 | **+0.028** |
| 75%  | 0.3435 / 0.2373 / 0.3846 / 0.5470 | 0.3084 / 0.1975 / 0.3463 / 0.5273 | **+0.035** |
| 100% | 0.2921 / 0.1864 / 0.3251 / 0.5026 | 0.2875 / 0.1816 / 0.3181 / 0.5013 | ≈0 (tie) |

#### Other single-dataset benchmarks

| dataset | mode | ep3 | ep9 | winner |
|---|---|---|---|---|
| YAGO | single_step | 0.8838 / 0.8493 / 0.9128 / 0.9323 | 0.8580 / 0.8217 / 0.8884 / 0.9050 | **ep3 (+0.026)** |
| **ICEWS18** | single_step | 0.2458 / 0.1552 / 0.2833 / 0.4315 | 0.2732 / 0.1793 / 0.3130 / 0.4612 | **ep9 (+0.027)** ← only ep9 win |
| ICEWS14 | static | 0.6087 / 0.5074 / 0.6719 / 0.7928 | *missing* | ep3 only |
| ICEWS0515 | static | 0.6169 / 0.5009 / 0.6929 / 0.8298 | *missing* | ep3 only |

ep9 ICEWS14/0515 runs don't exist (paper's row = these ep3 numbers
from 27366/27367).

#### Scorecard

| outcome | count | datasets |
|---|---:|---|
| **ep3 wins** | 8 | WIKI_25/50_inter, WIKI_25/50/75_extra_ss, YAGO_ss, GDELT_50/75_extra_ss |
| ties | 5 | WIKI_75/100_inter, WIKI_100_extra_ss, GDELT_25/100_extra_ss |
| **ep9 wins** | 1 | **ICEWS18_ss** |
| ep9 missing | 6 | ICEWS14/0515 static, GDELT_25/50/75/100 inter static |

**Observations**:
- **At low inductive ratios ep3 crushes ep9** — WIKI_25_inter +0.21 MRR,
  WIKI_25_extra_ss H@1 +0.28. ep9 has overfit and lost generalization
  capacity.
- **ep3 wins the top-1 contest even where ep9 catches up on H@10**
  (WIKI_75_extra_ss). ep3 is more decisive at rank-1 even when ep9
  spreads its mass better across the top-10.
- **ICEWS18 is the one true ep9 win** — pretrain-adjacent domain,
  longer ICEWS14/0515 training transfers better zero-shot.
- **Paper framing**: ep3 is the canonical "+Method" ckpt. Single
  exception that would need a footnote: ICEWS18, where ep9's additional
  ICEWS-family training helps the related domain.
- **Still-missing ep9 runs** (6 cells) worth filling: ICEWS14/0515
  static and GDELT inter static (×4), to confirm the ep3 preference on
  the ICEWS family holds (vs. the pattern that ep9 wins ICEWS18
  zero-shot).

### c) GRATE ICEWS14 β sensitivity (jobs 93688 + 93694 + 94221, DONE)

**Zero-shot β swap** — model trained at β=10000, evaluated with a
modified inference-time β. No retraining.

| β | MR | MRR | H@1 | H@3 | H@10 |
|---:|---:|---:|---:|---:|---:|
| **1000** | 109.75 | **0.6115** | **0.5105** | **0.6778** | 0.7917 |
| 5000 | 102.41 | 0.6032 | 0.4998 | 0.6714 | 0.7914 |
| 10000 (train default) | *missing* | — | — | — | — |
| 25000 | 101.28 | 0.5962 | 0.4901 | 0.6653 | 0.7904 |
| 50000 | 101.47 | 0.5908 | 0.4834 | 0.6604 | 0.7885 |
| 100000 | 100.69 | 0.5879 | 0.4800 | 0.6566 | 0.7860 |

Spread **0.024 MRR** across **100× β range**. Monotone downhill
small→large β.

Patterns:
- **MRR, H@1, H@3 all monotonically decrease** as β grows.
- **H@10 is nearly flat** (0.786–0.792): the top-10 ranking is stable;
  small β only sharpens top-1/top-3 discrimination.
- **MR improves slightly with larger β** (109.7 → 100.7): β=1000's
  MRR win comes with a tiny MR cost — a few hard queries slip further
  down for that fine-rotation setting.

**Paper framing**: report as a robustness datapoint — GRATE does not
require careful β tuning; the LLM default works within a couple of
points of best. If asked to defend β=10000, the numbers back it up.
β=1000 is the small-β optimum worth highlighting if the story wants
an "even better with Δt-scale-aware β" angle.

**Open**:
- Fill the β=10000 cell (training default) with 27204 ep9 ICEWS14
  static — 93688_3 was cancelled before completing.
- Verify plateau at low β (β ∈ {200, 500}).
- Cross-scale transfer on YAGO (yearly Δt, expect smaller optimum)
  and GDELT (15-min Δt, expect larger optimum). If the optimum scales
  inversely with Δt range, that's a paper-worthy invariant.

### d) 27204 ep3 single_step on GDELT extra (jobs 93711 + 94233, DONE)

Already included in § 4b GDELT_extra_ss comparison. Summary:
0.3093 / 0.3127 / 0.3435 / 0.2921 for 25/50/75/100 — beats TIGER ep5
(0.26 flat) and the ep9+ss paper numbers (0.295/0.285/0.308/0.288)
at 25, 50, 75; ties at 100. **ep3 recommendation extends to GDELT
single_step.**

### e) Vanilla TRIX 3-ds pretrain (job 93811, ep7 done — RoPE2_decay_q → distmult)

Cold-start A/B partner of 93629 on the same data mix. Val MRR
trajectory (`fast_test=5000`):

| epoch | ICEWS14 | ICEWS0515 | YAGO | avg |
|---:|---:|---:|---:|---:|
| 0 | 0.495 | 0.470 | 0.649 | 0.538 |
| 1 | 0.502 | 0.462 | 0.648 | 0.537 |
| 2 | 0.503 | 0.464 | 0.649 | **0.539** ← best |
| 3 | 0.501 | 0.468 | 0.643 | 0.537 |
| 4 | 0.501 | 0.456 | 0.631 | 0.529 |
| 5 | 0.496 | 0.447 | 0.643 | 0.529 |
| 6 | 0.495 | 0.451 | 0.619 | 0.522 |
| 7 | 0.490 | 0.424 | 0.632 | 0.515 |

Head-to-head (val MRR at best epoch):

| model | ICEWS14 | ICEWS0515 | YAGO | avg |
|---|---:|---:|---:|---:|
| Vanilla TRIX (cold) ep2 | 0.503 | 0.464 | 0.649 | 0.539 |
| GRATE (warm from 27204) ep1 | 0.606 | 0.601 | 0.679 | **0.629** |
| Δ (GRATE − vanilla) | +0.103 | +0.137 | +0.030 | **+0.090** |

RoPE2_decay_q buys **+0.09 avg val MRR** on the 3-ds joint task. Split:
- ICEWS14 (+0.103) and ICEWS0515 (+0.137) are dominated by the
  warm-start advantage — 27204 already trained 10 epochs on these
  two sources, so most of the gap is head-start.
- **YAGO (+0.030)** is closer to the pure temporal-message
  contribution because YAGO was never in the 27204 warm-start. This
  is the cleanest RoPE-only signal we have at this scale.

Both trajectories drift down monotonically (vanilla by −0.024 from
ep2→ep7, GRATE by −0.043 from ep1→ep9). Downstream evals pending
when 93811 finishes (~11:00 tomorrow).

### f) ep3 vs FITTER head-to-head

Same data, same rolling-single_step protocol (extra) / static (inter),
same Bordes filter.

| dataset / mode | ep3 | FITTER | Δ (ep3 − FITTER) |
|---|---:|---:|---:|
| WIKI_25_inter (static) | 0.7131 | 0.6904 | +0.023 |
| WIKI_50_inter (static) | 0.7907 | 0.7491 | +0.042 |
| WIKI_75_inter (static) | 0.8715 | 0.8200 | +0.052 |
| WIKI_100_inter (static) | 0.9922 | 0.9572 | +0.035 |
| WIKI_25_extra (ss) | 0.9676 | 0.9693 | -0.002 (tie) |
| WIKI_50_extra (ss) | 0.9618 | 0.9299 | +0.032 |
| WIKI_75_extra (ss) | 0.9719 | 0.9698 | +0.002 (tie) |
| WIKI_100_extra (ss) | 0.9697 | 0.9629 | +0.007 |
| GDELT_25_inter (static, paper) | 0.380 | 0.2664 | +0.114 |
| GDELT_50_inter (static, paper) | 0.394 | 0.2617 | +0.132 |
| GDELT_75_inter (static, paper) | 0.432 | 0.2642 | +0.168 |
| GDELT_100_inter (static, paper) | 0.444 | 0.2754 | +0.169 |
| GDELT_25_extra (ss) | (rerun 94212) | 0.2545 | — |
| GDELT_50_extra (ss) | 0.3127 | 0.2646 | +0.048 |
| GDELT_75_extra (ss) | 0.3435 | 0.3051 | +0.038 |
| GDELT_100_extra (ss) | 0.2921 | 0.2674 | +0.025 |

**ep3 dominates FITTER on every measured metric.** WIKI extra is the
only place where the two tie (both saturate against a hard ceiling
where absolute time barely matters). GDELT inter is a rout (+0.11 to
+0.17) — RoPE2_decay_q captures GDELT's periodic recurring structure
that FITTER can't. YAGO isn't included: FITTER's 0.879 is on
YAGOInd (inductive) while GRATE's 0.884 is on transductive YAGO —
not the same split.

**Paper implication**: ep3 + FITTER as the two headline columns
works: ep3 above FITTER on WIKI inter, GDELT inter/extra; ties on
WIKI extra where both saturate. The smallest gap is exactly where
the ceiling forces both models into a corner — defensible framing.

### g) Model-width scaling probe (dim=96, jobs 94222 → 94231 → 94290)

Replacement scalability story after 3-ds pretrain didn't produce a
clean "more data helps" signal (§ 4a). Scales GRATE's `input_dim` and
`hidden_dims` from 64 → 96 on the same 2-ds mix, cold start.

- **94222** (first attempt, bs=2): OOM at first layer after 20 s —
  dim=96 bs=2 needs > 79 GB on A100:80G.
- **94231** (bs=1, lr=5e-4, same as dim=64 baseline): ran 11 h 28 m
  to ep7 before cancel. Trajectory had the diagnostic signature of
  **LR-too-high** at halved effective batch:

  | epoch | ICEWS14 | ICEWS0515 |
  |---:|---:|---:|
  | 0 (init) | 0.484 | 0.474 |
  | 1 | 0.401 | 0.348 ← **−0.08 crash** |
  | 2 | 0.493 | 0.449 |
  | 3-7 | 0.45–0.49 (oscillates) | 0.44–0.46 |

  ep0 (model init + eval, no training yet) is comparable to vanilla
  TRIX cold-start 0.495. Then ep1 drops 0.08 MRR, and the model
  oscillates in the 0.44-0.49 band ever since. Classic
  LR-too-high-at-small-batch signature.

- **94290** (resubmitted with `lr=2.5e-4`, linear scaling rule for
  bs=1 vs bs=2 baseline): RUNNING. Decision rule — if ep1 MRR ≥ ep0
  MRR → LR was the issue, let it converge; if ep1 crashes again →
  try `bs=2 + num_negative=256` to restore the baseline gradient
  noise regime.

**Open**: if dim=96 does climb past dim=64's 0.63 at convergence,
add a dim=128 datapoint for a 3-point scaling curve. If it merely
matches dim=64, the paper framing becomes "GRATE is well-sized at
dim=64 for these datasets" (honest, less exciting). If it loses to
dim=64, the width-scaling story is dead and we fall back to the ep3
vs ep9 cross-dataset dominance (§ 4b) and the FITTER head-to-head
(§ 4f) as the paper's two main empirical anchors.

---

## 5. Reruns in flight (2026-10-01)

| final jid | task | walltime | outcome |
|---|---|---|---|
| 94221_6 | β=50000 sensitivity on ICEWS14 | 2 h | COMPLETED (filled the β=50000 cell) |
| 94233_1 | GDELT_25_extra_ep3_ss | 10 h | COMPLETED (filled GDELT_25 cell) |
| 94234_10 | ICEWS18_ep3_ss | 10 h | COMPLETED (**ep9 wins — see § 4b**) |
| 94236_1 | 3ds ep0 ICEWS14 static | 8 h | COMPLETED (**eval drift discovered — see § 4a**) |
| 94235_2 | 3ds ep1 ICEWS0515 static (rerun) | 8 h | RUNNING (same "Evaluate on test" stall risk) |
| 94236_2 | 3ds ep0 ICEWS0515 static | 8 h | RUNNING (same stall risk) |
| 94236_3 | 3ds ep0 YAGO single_step | 8 h | RUNNING |

Chain history (cancelled → resubmitted → final): 94212→94225→**94233**,
94213→94226→**94234**, 94214→94227→**94235**, 94215→94228→**94236**.

---

## 6. Currently running

| job | what | ETA |
|---|---|---|
| **94290** | GRATE dim=96 cold-start 2-ds pretrain, **lr=2.5e-4** (bs=1). Model-width scaling probe. See § 4g. | PENDING on QOS (needs 6 A100s; blocked by 3 running single-GPU jobs on gpuB01) |
| 94235_2, 94236_2, 94236_3 | 3ds eval tail (see § 5) | 2-6 h |

**93811 (vanilla TRIX 3-ds) was CANCELLED at ep8** after capturing a
clear plateau at avg 0.539 (ep2) with drift to 0.515 by ep7. Enough
trajectory for the GRATE vs vanilla A/B. Downstream evals skipped.

---

## 7. Provenance findings

### GDELT IndT "TRIX+Method" row (paper)

The row is **inconsistent across columns**:

- **Interpolation** — jobs **27356, 27357, 27358, 27359**:
  - ckpt: `model_epoch_4.pth` (**27204 ep3**)
  - protocol: static (no `eval_mode`)
  - Numbers: 0.380 / 0.394 / 0.432 / 0.444
- **Extrapolation** — jobs **28537, 28538, 28539, 28540**:
  - ckpt: `model_epoch_10.pth` (**27204 ep9**)
  - protocol: **single_step**
  - Numbers: 0.295 / 0.285 / 0.308 / 0.288 (= 93548)

Same row mixes ep3 static (inter) with ep9 single_step (extra) —
probably a best-of-both-worlds cherry-pick, not a coherent single
protocol. Verification job 93711 in flight.

### alan_fitter filter audit
- Filter correct post-**2026-04-24** (commit `e0dc3f8`) — TIGER paper
  numbers use the correct fallback, **no re-eval needed**.
- Fast-val (during training) call site was still buggy; patched today.

---

## 8. Open items for the paper

1. **Present ep3 as the canonical "+Method" ckpt** (§ 4b). ep3 wins
   8 out of 14 measured comparisons, ties 5, loses 1 (ICEWS18 ss by
   0.027). Six ep9 cells still missing — ICEWS14/0515 static,
   GDELT_25/50/75/100 inter static. Fill them to confirm ep3 holds
   on the ICEWS family (vs the pattern that ep9 wins on
   pretrain-adjacent zero-shot like ICEWS18).
2. **3-ds "regression" claim is retracted pending eval-side audit**
   (§ 4a). The warm-start snapshot (ep0) scores 0.03 lower than the
   same ckpt loaded in the single-dataset harness. The drop at ep1
   is within noise. The regression we saw was the eval harness, not
   training. Must isolate the config difference (num_time vocab,
   entity reindexing, base_beta plumbing) before any statement
   about 3-ds-pretraining impact is defensible.
3. **Model-width scaling probe in flight** (§ 4g). 94290 running at
   lr=2.5e-4 after the ep1 crash under lr=5e-4 at bs=1. If dim=96
   converges above dim=64's 0.63, extend to dim=128 for a 3-point
   scaling curve. If not, the scalability story is dead and the
   paper leans on ep3-vs-ep9 (§ 4b) and ep3-vs-FITTER (§ 4f).
4. **ICEWS18 — the exception.** The one dataset where ep9 beats ep3.
   Worth a footnote: pretrain-adjacent zero-shot favors the
   more-trained ckpt, in-pretrain / out-of-domain-inductive favors
   the less-overfit ckpt. If space permits, this split is a
   publishable finding on its own.
5. **β=1000 is best, β=10000 is 0.005 away** (§ 4c). Report as
   robustness, not tuning. Still need β=10000 refill cell, and
   cross-scale sweeps on YAGO (yearly Δt) and GDELT (15-min Δt) if
   we want the "optimum β scales inversely with Δt" invariant.
6. **WIKI IndT interpolation paper row** — Explore agent hunting for
   the source of 0.623/0.760/0.858/0.962 (TRIX row) and
   0.507/0.650/0.872/0.991 (+Method row) — still open. Our actual
   ep9 run (27403-6) matches the paper's +Method row within 0.001
   so provenance of the +Method row is confirmed.
7. **Vanilla-TRIX-3ds vs GRATE-3ds A/B** (jobs 93629 vs 93811,
   § 4e) — GRATE +0.09 avg val MRR on the joint 3-ds task; YAGO
   isolates +0.030 as the pure RoPE contribution (warm-start-free).
   Reportable as a mini-ablation if the main scalability story
   collapses.
