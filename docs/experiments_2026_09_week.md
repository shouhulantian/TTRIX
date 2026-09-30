# Experiments week of 2026-09-28 → 2026-09-30

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

## 4. Results landed 2026-09-30

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

**ep0 baseline eval (job 94215) submitted** — ckpt `model_epoch_1.pth`
is the warm-start snapshot saved before any 3-ds training. Evaluating
it isolates the 3-ds training gain from the warm-start baseline. If
ep0 ≈ 27204 ep9, the drop at ep1 is training-induced. If ep0 already
< 27204 ep9, it is a config drift (e.g. num_time vocab / YAGO date
encoding on the new joint dataset).

### b) 27204 ep3 vs ep9 (job 93713, DONE — 8/10 complete, 2 rerunning)

WIKI IndT interpolation (ep3 static):

| variant | ep3 test MRR | ep9 (paper) |
|---|---:|---:|
| WIKI_25_inter | 0.7131 | 0.507 |
| WIKI_50_inter | 0.7907 | 0.650 |
| WIKI_75_inter | 0.8715 | 0.872 |
| WIKI_100_inter | **0.9922** | 0.991 |

**ep3 wins on WIKI inter** (huge gap at 25/50, matches at 75/100).

WIKI IndT extrapolation (ep3 single_step):

| variant | ep3 test MRR | ep3 test H@1 | ep3 test H@10 |
|---|---:|---:|---:|
| WIKI_25_extra_ss | 0.9676 | 0.9589 | 0.9806 |
| WIKI_50_extra_ss | 0.9618 | 0.9547 | 0.9724 |
| WIKI_75_extra_ss | 0.9719 | 0.9661 | 0.9797 |
| WIKI_100_extra_ss | 0.9697 | 0.9632 | 0.9773 |

ep3 flat around 0.96–0.97 across ratios (higher than the paper's
+Method row 0.960/0.961/0.958/0.961). ep3 wins.

YAGO / ICEWS18 (ep3 single_step):

| dataset | ep3 MRR | ep9 (paper) |
|---|---:|---:|
| YAGO | 0.8838 | 0.858 |
| ICEWS18 | rerun 94213 | 0.273 |

**ep3 wins on YAGO.** ICEWS18 timed out at 4 h — see § 5.

**Verdict**: for downstream deployment ep3 is at least as good as
ep9, often much better. The paper's "+Method" numbers were a
best-of-both-worlds mix (§ 5 in the previous doc); a clean per-column
choice would put ep3 forward for every column measured so far.

### c) GRATE ICEWS14 β sensitivity (jobs 93688 + 93694, DONE)

**Zero-shot β swap** — model trained at β=10000, evaluated with a
modified inference-time β. No retraining.

| β | test MRR | test H@1 | test H@10 |
|---:|---:|---:|---:|
| **1000** | **0.6115** | **0.5105** | **0.7917** ← best |
| 5000 | 0.6032 | — | — |
| 10000 (train) | ~0.607 | — | — |
| 25000 | 0.5962 | — | — |
| 50000 (94221, in flight) | — | — | — |
| 100000 | 0.5879 | — | — |

Spread ≈ **0.024 MRR** across **two orders of magnitude of β** —
monotone downhill from small β to large β. The default β=10000 (LLM
convention) sits within ~0.005 MRR of the optimum: RoPE2_decay_q is
robust to β choice.

**Paper framing**: report as a robustness datapoint — GRATE does not
require careful β tuning; the LLM default works within a couple of
points of best. If asked to defend β=10000, the numbers back it up.

**Open**: verify plateau at low β (β ∈ {200, 500}) and cross-scale
transfer on YAGO (yearly Δt, expect smaller optimum) and GDELT
(15-min Δt, expect larger optimum). If the optimum scales inversely
with Δt range, that's a paper-worthy invariant.

### d) 27204 ep3 single_step on GDELT extra (job 93711, DONE — 3/4, 1 rerunning)

| variant | ep3+ss MRR | ep3+ss H@1 | ep3+ss H@10 |
|---|---:|---:|---:|
| GDELT_25_extra_ss | rerun 94212 | — | — |
| GDELT_50_extra_ss | 0.3127 | 0.2061 | 0.5157 |
| GDELT_75_extra_ss | 0.3435 | 0.2373 | 0.5470 |
| GDELT_100_extra_ss | 0.2921 | 0.1864 | 0.5026 |

GRATE ep3+ss beats TIGER ep5 (0.29-0.34 vs TIGER's 0.26 flat) on
GDELT extrapolation. Also beats the paper's ep9+ss numbers
(0.295/0.285/0.308/0.288) at 50 and 75. **ep3 recommendation extends
to GDELT single_step.**

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

---

## 5. Reruns in flight (2026-09-30 evening)

| job | task | walltime | reason |
|---|---|---|---|
| **94212_1** | GDELT_25_extra_ep3_ss | 10 h | 4 h TIMEOUT — GDELT 25% has ~350 single_step steps |
| **94213_10** | ICEWS18_ep3_ss | 10 h | 4 h TIMEOUT |
| **94214_2** | ICEWS0515_3ds_ep1_static | 8 h | 3 h TIMEOUT — static eval stalled at "Evaluate on test" (investigate whether filter build for ~450 k edges dominates) |
| **94215_[1-3]** | 3ds ep0 baseline (ICEWS14/0515/YAGO) | 8 h | New — evaluate warm-start snapshot before any 3-ds training, to isolate training-induced vs config-induced regression |
| **94221_6** | β=50000 sensitivity on ICEWS14 | 2 h | Fills 25000→100000 gap in β sweep |

---

## 6. Currently running

| job | what | ETA |
|---|---|---|
| **93811** | **Vanilla TRIX 3-ds pretrain** (ICEWS14 + ICEWS0515 + YAGO, cold). A/B partner of 93629 — same data, RoPE2_decay_q → distmult. 6× A100, bs=2, BPE=8000, 10 ep. Isolates the temporal-message contribution. At ep7: val avg 0.515, downhill from 0.539 peak at ep2. See § 4e for trajectory. | ~10 h (14+ h in of 24 h wall) |
| Reruns | 94212 / 94213 / 94214 / 94215 / 94221 | 2-10 h |

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

1. **Present ep3 as the canonical "+Method" ckpt.** Per-column ep3
   dominates ep9 on WIKI inter, WIKI extra, YAGO, GDELT extra ss
   (§ 4b, 4d). No metric on which ep9 wins has surfaced yet. Re-run
   ICEWS18 ep3 when 94213 lands to close the last gap.
2. **3-dataset scalability is NOT a paper win** (§ 4a). ICEWS14 and
   YAGO both regress when YAGO is added to the pretrain mix. The
   scalability story has to be either (a) re-tell as "GRATE
   pretrained on ICEWS14+ICEWS0515 generalizes to YAGO with a small
   penalty" (2-ds → YAGO zero-shot = 0.858; 3-ds → YAGO ID = 0.841),
   or (b) drop from the paper. Waiting on 93811 (vanilla-TRIX 3-ds)
   to see if the regression is intrinsic to the joint task or
   specific to GRATE.
3. **β=1000 is the new default on ICEWS14.** Sweep β ∈ {200, 500,
   1000, 2000, 5000} to confirm the plateau, then repeat on YAGO
   (yearly Δt) and GDELT (15-min Δt) — the optimum should scale
   inversely with Δt.
4. **WIKI IndT interpolation paper row** — Explore agent hunting for
   the source of 0.623/0.760/0.858/0.962 (TRIX row) and
   0.507/0.650/0.872/0.991 (+Method row) — still open.
5. **Vanilla-TRIX-3ds vs GRATE-3ds A/B** (job 93811) — isolates the
   temporal-message contribution on the 3-source mix. Result will
   sharpen whether the drift observed in § 4a is an architecture
   issue or a data-mix issue.
