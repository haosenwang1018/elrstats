# Experiment: ELR measurement study (12-layer d=768 GPT, FineWeb-edu)

Parent: `../DESIGN.md` (Goal 4). This file is the contract for the runs and for the hourly watcher.

## Goals
Measure, in real pretraining, how far the per-matrix **effective** learning rate (ELR = η‖U‖/‖W‖) is from the **nominal** schedule, and where the gap comes from. Concretely:
1. How much does ELR decay when the nominal lr is constant (implicit decay)?
2. In a normal warmup + linear-decay run, what share of the ELR change is implicit (‖W‖, ‖U‖) versus scheduled (η)?
3. Does AdamW + WD reach the predicted rotational equilibrium, and does a 4× higher lr give < 4× higher ELR?
4. How different is ELR across matrices, and does a whole fused-qkv ELR hide the k ELR?
5. In practice, does "relative change" (Su) equal "angle" (Zou)? That is, is cos(W,U) small?

## Non-goals
- Which optimizer or schedule gives lower loss. Loss only has to be healthy.
- Large scale. One model size only (12 layers, d=768; 113M block / 191M total parameters); no claim about transfer to larger models.
- Controlling ELR (relative update / Hyperball). That is a later experiment.

## Constraints
- Hard: one 8×A100-PCIE-40GB host, one run per GPU. Compute time is limited.
- Hard: ~1.06B usable training tokens (the train shards are truncated downloads).
- Soft (our choice): all runs finish in ≤ 6 h; 12-layer, d=768 model (113M block / 191M total parameters); 25,000 steps × 32,768 tokens ≈ 0.82B tokens (~0.8 epoch, random windows).
- Soft: one seed per config in batches 1–2. Batch 3 adds a second seed for four configs, so seed noise is measured only for those.

## Acceptance criteria (the experiment counts as done and usable)
- All runs (8 in batch 1, 16 in total) reach 25,000 steps with `DONE`; no NaN; val loss falls monotonically within noise after step 2000.
- `elr.jsonl` has a record every 25 steps for every logical matrix (q/k/v split, o, up, gate, down per layer, plus emb, head).
- Sanity: the measured ‖ΔW‖/‖W‖ matches the value predicted from the logged ‖W‖, ‖U‖, cos and WD (median rel. error < 1%). This checks that the logged U is the applied update. (Batch 1's `aus_true` has fp32 cancellation at small angles and is not used; batch 2 logs ⟨W,ΔW⟩ and computes the angle without cancellation.)
- `study/analyze.py` produces fig1–fig4 and `summary.json`.

## Success / failure criteria per question (decided before looking at results)
Note: batch 1 was launched before this doc was reviewed. The review (see Log) changed the Q1 base step, the Q2 share, the Q3a formulas and the Q4 spread measure while batch 1 was running, before any final result was read.
| Q | Supported if | Not supported if → what it means |
|---|---|---|
| 1 | In `const` runs, median ELR at 95% of training is < 1/2 of its value at step 2000 (also reported vs step 300, but step 300 sits on the early ELR spike, so it is not the criterion). The clean test is the `const` + WD 0 pair (batch 2). | ≥ 1/2 → implicit decay is weak at this scale/horizon; "Hyperball = implicit schedule" is not visible at this model size. Report as negative. |
| 2 | In `linear` runs, implicit share ≥ 25% at 50% of training, base step 2000. Share = \|Δlog‖U‖ − Δlog‖W‖\| / (\|that\| + \|Δlog η\|), sign reported separately (the two parts can cancel). | < 25% → nominal ≈ effective in practice here. |
| 3a | `const` + WD: median AUS (from the three sums) over the last 10% within 2× of a published prediction. Report both √(2ηλ) and Kosson's √(2ηλ(1−β₁)/(1+β₁)); they differ 4.4×, so at most one can pass. | Off by > 2× → not at equilibrium in this horizon, or formula does not fit; report the ratio. |
| 3b | 4× lr gives ELR ratio between 1.5× and 3× at 70/80/90% of training. Primary: the `const` pair (batch 2); the `linear` pair is secondary (η → 0 late). | ≈ 4× → no compression yet (norms not at equilibrium). |
| 4 | Within one matrix type, p90/p10 of ELR across layers > 2× at the end (across all types is inflated by the smaller o/down init, so it is reported but not used), and ELR(k)/ELR(whole qkv) outside [0.8, 1.25] in ≥ 3 of 12 layers. | Otherwise → one ELR per model is enough at this scale; logical sub-matrices are not needed here. |
| 5 | After step 300: median \|cos(W,U)\| < 0.05 and median \|ELR/AUS − 1\| < 5%, AUS from the three sums (exact algebra, fp64). | Otherwise → relative change and angle disagree; that is itself a finding. |

## Settings
- Model: 12 layers, d=768, 12 heads, SwiGLU (4d), RMSNorm, untied emb/head, fused qkv. Init std 0.02; output projections 0.02/√(2L).
- Batch 16×1024, accum 2 → 32,768 tokens/step. Warmup 300 steps, then constant or linear to 0. bf16 autocast, fp32 master weights, grad clip 1.0.
- AdamW β=(0.9, 0.95), ε=1e-8. Muon: Nesterov 0.95, NS5, Moonlight scale 0.2·√max(m,n), q/k/v orthogonalized separately; emb/head/norms use AdamW at the same lr.
- Batch 1, 8 runs (`study/launch.sh 25000`): {adamw, muon} × {lr6e-4 wd0.1 linear, lr6e-4 wd0.1 const, lr2.4e-3 wd0.1 linear, lr6e-4 wd0 linear}.
- Batch 2, 4 runs (`study/launch.sh 25000 batch2`): {adamw, muon} × {lr6e-4 wd0 const, lr2.4e-3 wd0.1 const}.
- Batch 3, 4 runs (`study/launch.sh 25000 batch3`): seed 1 of {adamw, muon} × {lr6e-4 wd0.1 linear, lr6e-4 wd0.1 const}, to measure seed noise.
- U: AdamW = m̂/(√v̂+ε) from optimizer state (no WD); Muon = the orthogonalized, scaled update. Stats on fp32 weights before the step.
- Smoke test throughput (single GPU, 120 steps): Muon ~94k tok/s, AdamW ~82k tok/s → ~2.5–3 h per run.

## Health criteria (the watcher checks these)
- **Alive:** each running run's `elr.jsonl` was modified in the last 20 min.
- **Hung:** GPU busy but no log update for > 20 min.
- **Diverged:** train loss NaN/inf, or > 7.0 after step 2000.
- **Throughput:** `tok_per_s` within 30% of the smoke-test value.
- **Failed:** `.FAILED` marker, or process gone without `DONE`.

## Watcher rules (hourly agent)
- Read-only by default. Report ≤ 6 lines: per run → step, val loss, tok/s, ETA, status.
- May act without asking only to kill a study run that is hung > 40 min or has NaN loss, and log the reason to `/data/exp_elrshard/WATCH.log`. No relaunch, no code changes, never touch other experiments, no deletes.

## Log
- 2026-10-29 — Smoke test 1 failed: truncated train shards. Reader fixed.
- 2026-10-29 — Two benchmark jobs (8-GPU and 2-GPU) ran at 100% GPU with no output for > 20 min; killed. Nobody noticed until someone asked → added health criteria and the hourly watcher.
- 2026-10-29 — Runs launched at 25,000 steps BEFORE this doc was reviewed (my ordering mistake). Review by critic agents follows; if a blocker is found, the runs are killed and relaunched (cost ≤ 3 h).
- 2026-10-29 — 4 critics (code, design, analysis, red team): all "keep runs". Changes: (1) `aus_true` in batch 1 has fp32 cancellation → use the three-sum AUS; batch 2 logs ⟨W,ΔW⟩ and RMSNorm gain norms. (2) Q1 base moved to step 2000 (step 300 is on the early spike). (3) Q2 share redefined (stable under cancellation). (4) Q3a: report both formulas. (5) Q3b and Q1 need clean pairs → batch 2 queued (const + WD 0; const lr 2.4e-3). (6) Q4 within-type spread. (7) Added joint-symmetry balance ‖Wq‖²−‖Wk‖² etc. (Su 11985). Known limitation: RMSNorm gains also get WD; model has no QK-norm, so matrices are not scale-invariant. Red team: early ELR spike, √(2ηλ) equilibrium and per-matrix differences are reproductions; the implicit-share decomposition and q/k/v split are the likely new parts.
- 2026-10-29 — Batch 1 (8 runs) DONE. Batch 2 first launch failed at step 0: my edit reused the name `a` (the args object) for a stats sum. Fixed, smoke-tested (60 steps), relaunched together with batch 3 (seeds).
- 2026-10-29 — Incident: `launch.sh` started batch 1 whenever the flag was not `batch2`, so the `batch3` call re-queued all 8 batch-1 runs; 4 of them ran ~250 steps and appended to batch 1's logs. Killed within ~3 min; batch-1 logs restored from the local copy pulled before the relaunch (1001 records each, steps monotonic, verified). Guard fixed: batch 1 only runs with no flag.
- 2026-10-29 — All 16 runs DONE (batches 1–3). Seed noise ≤ 0.004 loss, ≤ 3% on ELR numbers. Q1 supported on the clean no-WD test (ELR ×0.34 at flat nominal lr); Q2, Q3a (Kosson AdamW formula within 0.4%/5%), Q3b (ELR ratio = 2.0 for 4× lr), Q5 supported; Q4 mostly not supported at this model size. Report: `results/final/REPORT.md`. Watcher cron deleted.
- 2026-10-29 — Pre-release review: the final report's Q4 verdict ("mostly not supported") did not follow the rule above. Re-checked run by run: Q4 is mixed (passes in no-WD runs, fails with WD); Q5 fails the |cos| rule in the two 4× lr runs. Report corrected; see `PITFALLS.md` G6.
