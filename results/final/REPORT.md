# Final results: ELR measurement study (16 runs, GPT-124M, 25k steps, 0.82B tokens) — 2026-10-29

Criteria were fixed in advance in `../../design/study.md`. Numbers come from `summary.json`. The main figure for a blog is `fig5_nominal_vs_elr_normalized.png`: nominal lr and median ELR on one log axis, both normalized to step 2000.

## Seed noise (batch 3, seed 0 vs seed 1)
| config | val loss | ELR(95%)/ELR(2000) | implicit share at 50% | equilibrium AUS |
|---|---|---|---|---|
| AdamW lin | 3.406 / 3.407 | 0.042 / 0.043 | 0.40 / 0.41 | – |
| AdamW const | 3.587 / 3.583 | 0.520 / 0.523 | – | 0.00252 / 0.00252 |
| Muon lin | 3.315 / 3.316 | 0.040 / 0.040 | 0.35 / 0.34 | – |
| Muon const | 3.510 / 3.508 | 0.521 / 0.520 | – | 0.00201 / 0.00201 |

Seed differences are ≤ 0.004 in loss and ≤ 3% on the Q1, Q2 and Q3a numbers. The Q1–Q3 effects below are far larger than this. Seed noise was not measured for Q3b–Q5, and per-layer numbers (Q4) vary much more between seeds: layer-0 k/whole is 0.48 vs 0.28, and within-type k spread is 1.63 vs 1.22.

## Verdicts
| Q | Result | Verdict |
|---|---|---|
| 1 Implicit decay at constant nominal lr | No WD: ELR(95%)/ELR(2000) = **0.34** (AdamW), **0.34** (Muon). Median ‖W‖ grows 3.6× (AdamW) / 3.0× (Muon) over the same span; ‖U‖ changes < 3%. With WD 0.1: 0.52 / 0.52. With WD and 4× lr: 0.77 / 0.73. | **Supported on the clean test only** (no WD): a flat nominal lr gives a ~3× ELR decay, and the drop comes from ‖W‖ growth (accounting, not a proven mechanism). The WD runs (0.52) and the 4× lr runs fail the < 1/2 rule: WD caps the decay at ~2×. |
| 2 Implicit share in linear runs (50%, base step 2000) | AdamW 0.40, Muon 0.35; no WD 0.52 / 0.48; 4× lr 0.19 / 0.16 | **Supported at lr 6e-4; fails at 4× lr** (< 0.25). At the 50% point, a third to a half of the change in *log* ELR is implicit. Other time points are in `summary.json`. |
| 3a Equilibrium vs formula | AdamW const: 0.00252 at η=6e-4 (Kosson 0.00251), 0.00530 at η=2.4e-3 (Kosson 0.00503). Plain √(2ηλ) is 4.3× / 4.1× above the measured values. Muon const: 0.00201 and 0.00407. | **Kosson's AdamW formula holds within 0.5% and 5.5%**, although the model has no QK-norm. Muon: no published formula checked. [推断] Kosson's form with Muon's β=0.95 gives 0.00175 (13% low); not derived carefully. |
| 3b 4× lr → ELR ratio | const pairs at 70/80/90%: AdamW 1.98 / 2.05 / 2.06, Muon 1.92 / 1.97 / 2.01. Linear pairs: AdamW 1.23 / 2.18 / 2.10 (that 4× run was unstable), Muon 2.01 / 2.09 / 2.16. | **Supported.** In the const + WD runs late in training, 4× lr gave 1.9–2.1× ELR. This is consistent with √η scaling at equilibrium (√4 = 2). |
| 4 Spread across matrices / hidden k | Within-type p90/p10 > 2× in 11 of 16 runs (usually v or o; up to 3.6 for gate and 6.4 for q in the 4× AdamW runs). k/whole-qkv outside [0.8, 1.25] in ≥ 3 of 12 layers: 6 of 16 runs (all 4 no-WD runs, AdamW WD const seed 0, unstable AdamW 4× lin). In WD runs the gap is mostly in layer 0 (k/whole 0.28–0.76); without WD, late layers 9–11 also reach 1.25–1.45. | **Mixed.** The rule passes in the no-WD runs and the unstable run, and fails in the WD baselines. The layer-0 value differs a lot between seeds (0.48 vs 0.28). |
| 5 Relative change (Su) vs angle (Zou) | median \|cos(W,U)\| 0.012–0.057; ELR vs AUS gap ≤ 0.2% in all runs | **Supported in 14 of 16 runs.** The two 4× lr runs reach median \|cos\| 0.052–0.057, just over the 0.05 rule; their ELR–AUS gap is still ≤ 0.2%. |
| Sanity | logged U reproduces the applied ‖ΔW‖/‖W‖ | Pass |

## What this means for the blog
- Reproduced: early ELR spike, then a long implicit decay; Kosson's AdamW equilibrium formula; ELR ratio ≈ √η ratio at equilibrium.
- Likely new: the implicit-share decomposition (η vs ‖U‖ vs ‖W‖) for AdamW and Muon in one setup, with seed error bars; and the clean "constant nominal lr, no WD → 3× ELR decay" number. The q/k/v split is mixed at 124M: it matters without WD, mostly not with WD.
- Practical message: at 124M, a nominal "constant" schedule is really a ~2–3× decaying ELR schedule, and in const + WD runs 4× the nominal lr gave only about 2× the ELR.

## Limits
- Criteria: the runs were launched before the criteria were reviewed. The review changed the Q1 base step, the Q2 share definition, the Q3a formulas and the Q4 spread measure before any final results were read (see `design/study.md`, Log).
- 124M only, 0.82B tokens (~6.6 tokens/param). No QK-norm. RMSNorm gains also get WD.
- Batch-1 `aus_true` has fp32 cancellation; angle results use the three-sum AUS (exact algebra). Batch 2/3 log ⟨W,ΔW⟩ for a cancellation-free angle.
- AdamW lr 2.4e-3 linear was unstable (val 4.28); its ‖Wq‖² − ‖Wk‖² grew to 1154. [推断] attention-logit growth without QK-norm.
