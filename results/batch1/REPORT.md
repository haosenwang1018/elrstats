> Interim report, superseded by `../final/REPORT.md`. Its Q4 verdict changed after all 16 runs were analysed.

# Batch 1 results (8 runs, GPT-124M, 25k steps, 0.82B tokens) — 2026-10-29

Pre-registered criteria are in `../../design/study.md`. Single seed per config; batch 3 adds a second seed for four configs. Numbers come from `summary.json`; figures are `fig1`–`fig4` in this folder.

## Verdicts
| Q | Criterion | Result | Verdict |
|---|---|---|---|
| 1 Implicit decay at constant lr | ELR(95%)/ELR(step 2000) < 0.5 | AdamW 0.52, Muon 0.52 (vs step 300: 0.49 / 0.43) | **Borderline, not met.** ELR still halves with a flat nominal lr. All of it comes from ‖W‖ growth (‖W‖ ×2.1); ‖U‖ is flat. The clean test without WD (batch 2) is running. |
| 2 Implicit share in linear runs | ≥ 25% at 50% of training | AdamW 0.40, Muon 0.35 (lr 6e-4, WD 0.1); no WD: 0.52 / 0.48; lr 2.4e-3: 0.19 / 0.16 | **Supported** at lr 6e-4; weaker at 4× lr. |
| 3a Equilibrium formula | within 2× | AdamW const: measured AUS 0.00252; Kosson √(2ηλ(1−β₁)/(1+β₁)) = 0.00251; plain √(2ηλ) = 0.0110 | **Kosson's formula matches within 0.4%**, although the model has no QK-norm. Plain √(2ηλ) is 4.4× off. Muon const: 0.00201 (no published formula checked). |
| 3b LR compression | 4× lr → 1.5–3× ELR | Linear pair at 80/90%: AdamW 2.18 / 2.10, Muon 2.09 / 2.16 | **Supported**, close to the √4 = 2 equilibrium prediction. The const pair (batch 2) is the cleaner test. |
| 4 Spread / hidden k | within-type p90/p10 > 2×; k vs whole-qkv outside [0.8, 1.25] in ≥ 3 layers | Within type: v 2.2–2.5×, o 1.5–2.2×, others 1.1–1.6×. k outside range: 1 layer in standard runs (layer 0, k ELR 0.58–0.76 of whole qkv); 3 layers in no-WD runs; 4 in the unstable AdamW 2.4e-3 run | **Mostly not supported** at 124M. The whole-qkv ELR hides k mainly in layer 0. |
| 5 Relative change vs angle | \|cos(W,U)\| < 0.05 and \|ELR/AUS − 1\| < 5% | \|cos\| median 0.012–0.036 (AdamW 2.4e-3: 0.052); gap ≤ 0.15% | **Supported.** Su's relative change and Zou's angle agree to < 0.2% here. |
| Sanity | logged U = applied update | ‖ΔW‖/‖W‖ prediction error ≈ 0 | Pass |

## Other observations
- Final val loss: Muon linear 3.30–3.33, AdamW linear 3.41–3.44, const runs 3.51 / 3.59, AdamW lr 2.4e-3 4.28 (unstable, noisy ELR in fig1).
- In the unstable AdamW 2.4e-3 run, ‖Wq‖² − ‖Wk‖² grew to 1154 (≤ 77 in all other runs). [推断] This fits attention-logit growth without QK-norm; not verified.
- With WD, the joint-symmetry imbalances (q−k, v−o, up−down) shrink toward 0; without WD they grow. This is consistent with Su's joint-symmetry view (WD pulls toward the balanced point).
- fig1 plots ELR on a log axis and the nominal lr on a linear axis. For the blog, both should be on the same log axis, normalized to their peaks.

## Known limits
- One seed; batch 3 measures seed noise for four configs.
- Batch-1 `aus_true` has fp32 cancellation; all angle results use the three-sum AUS (exact algebra).
- RMSNorm gains also get WD; no QK-norm.
