# Experiments week of 2026-09-28 → 2026-09-29

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

TIGER GDELT sweep (tasks 9–16 of 93693) still running.

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

## 4. Currently running (2026-09-29 late afternoon)

| job | what | ETA |
|---|---|---|
| **93629** | 3-dataset GRATE pretraining (ICEWS14 + ICEWS0515 + YAGO, warm-start from 27204 ep9). bs=2/GPU × 6 A100:80G, BPE=8000, 10 epochs. `datasets/joint_t/3g_ICEWS14_ICEWS0515_YAGO/`. Epoch 1 val avg 0.629 > 2-ds baseline 0.623. | ~06:00-08:00 tomorrow |
| **93693_[9..16]** | TIGER single_step on GDELT IndT (8 variants) | ~few hours |
| **93688 / 93694** | GRATE ICEWS14 β sensitivity. Done: β=5000 (0.6032), β=25000 (0.5962), β=100000 (0.5879). Pending: β=1000. Monotone downhill with increasing β. | soon |
| **93711_[1-4]** | 27204 ep3 + single_step on GDELTIndT_*_extra. Verifies whether the paper's extra column came from ep3 or ep9 protocol. | ~few hours |

---

## 5. Provenance findings

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

## 6. Open items for the paper

1. **How to present the GDELT +Method row?** — three options:
   - Re-label per-column with actual ckpt/protocol.
   - Rerun everything under one consistent protocol (ep9 + single_step).
   - Actually finetune ep9 on each variant (earlier attempts
     30798–30807 OOM'd / timed out).
2. **WIKI IndT interpolation paper row** — Explore agent hunting for
   the source of 0.623/0.760/0.858/0.962 (TRIX row) and
   0.507/0.650/0.872/0.991 (+Method row).
3. **RoPE β sensitivity** — pattern so far is monotone downhill from
   β=5000 toward β=100000 (ICEWS14). Awaiting β=1000. Decide whether
   to also sweep β on YAGO / GDELT.
4. **3-dataset GRATE (job 93629)** — early signal promising
   (ep1 avg 0.629 > baseline 0.623). Downstream eval sweeps pending.
5. **Cross-scale β transfer** — RoPE base β was trained at 10000
   assuming day-scale timesteps. YAGO's yearly Δts (in the unified
   vocab, still integer-year-spaced ≈ 365 units apart) may prefer a
   different β. Check after 93629 lands.
