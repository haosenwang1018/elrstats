# Pitfalls, bugs and incidents

This file records every problem we hit while building `elrstats` and running the ELR study (2026-10-29).
Each entry has: what we saw, the cause, the fix, and the lesson.
Status tags: **fixed**, **open**, **worked around**.

## A. Correctness bugs in the stats library

### A1. `_StridedShard` was not detected (silent 36% error) — fixed
- **Seen:** In the first design, FSDP2 × TP gave ‖W‖² off by 36%. No error, no warning.
- **Cause:** FSDP2 on top of TP uses `_StridedShard`, not `Shard`. We tested `p.is_shard()` only, so the code treated a sharded dim as replicated and divided by the wrong replica count.
- **Version note:** this depends on the torch version. In torch 2.13, `_StridedShard` is not a subclass of `Shard`, and `is_shard()` returns False (checked 2026-10-29). In torch 2.8, `_StridedShard` subclasses `Shard`, so `is_shard()` is already True there.
- **Fix:** `_is_shard_like(p) = p.is_shard() or isinstance(p, _StridedShard)`. Any unknown placement now raises.
- **Lesson:** Never treat an unknown placement as Replicate. Fail loudly.

### A2. Per-mesh-dim reduction was wrong on 2-D meshes — fixed
- **Seen:** The first design all-reduced over each mesh dim in turn. On a 2×2 mesh, some sums were counted twice.
- **Fix:** One all-reduce over the world group. Each rank divides its local sum by `world / n_distinct_shards` before the reduce. This count is a power of two, so the division is exact.
- **Lesson:** "One reduce over the world, divide by replica count" is simpler and easier to test than per-dim logic.

### A3. `Partial` placement — fixed by rejecting it
- A `Partial` tensor holds unreduced pieces. Its local sum of squares is not a part of the global sum of squares. The library raises on `Partial`.

### A4. Whole fused-matrix norm hides a sub-matrix — fixed by `LogicalSpec`
- **Seen:** In a constructed GQA case (k scaled 0.1×), k's ELR is 4.1× the ELR of the whole fused QKV.
- **Fix:** `LogicalSpec(dim, labels, names)` gives per-label sums with `index_add_`.
- **Note:** In real training of the 12-layer, d=768 study model this effect was small (only layer 0). See `results/final/REPORT.md`, Q4.

### A5. Local-norm relative update under manual TP — demonstrated, not ours
- NVIDIA-style hooks compute norms on the local tensor. Under manual (Megatron-style) TP rows, this gives a 0.65–0.93 update error in a constructed case where shard norms differ 10×. The real size in production is unknown.
- On FSDP2 DTensor, `vector_norm` is globally correct in torch 2.13. It only costs about 4 collectives per matrix. So "DTensor gives wrong norms" is **not** a claim we make.

### A6. Zero-norm fallback — fixed
- A relative update `η‖w‖u/‖u‖` gives zero step when ‖w‖ = 0 (for example a zero-init bias). `relative_scales` falls back to scale 1 (the absolute lr), not 0, when ‖W‖ ≤ `w_floor`. When ‖U‖ = 0 the scale is 0 (no update), with no divide by zero.

## B. Numerical and measurement pitfalls

### B1. `1 − cos` cancellation in the angle — fixed
- **Seen:** Batch 1 logged `aus_true` from fp32 `1 − cos`. At angles near 1e-3, fp32 loses most digits.
- **Fix:** Compute the angle in fp64 with atan2 on three sums. Two forms are used:
  - Library (`MatrixStats.aus`), from W and U: `θ = atan2(lr·√(s_ww·s_uu − s_wu²), s_ww − lr·s_wu)`.
  - Study (`study/train.py`), from W and the applied ΔW (batches 2–3 log ⟨W, ΔW⟩): `θ = atan2(√(s_ww·s_dd − s_wd²), s_ww + s_wd)`.
  The analysis uses the three-sum AUS for all batches.
- **Lesson:** Small angles: use atan2 on sums, never `acos` or `1 − cos`.

### B2. bf16 tests fail on CPU but pass on GPU — explained
- On gloo/CPU, bf16 cases showed ~1e-5 error. The cause is CPU fp32 accumulation order, not sharding. On NCCL (world 2/4/8) all errors are ≤ 2e-7.
- **Lesson:** Run the tolerance tests on the real backend before you blame the sharding logic.

### B3. Early ELR spike makes a bad baseline
- Step 300 (end of warmup) sits on the early ELR spike. A ratio against step 300 overstates decay. The criteria use step 2000 as the base.

### B4. "Implicit share" can cancel
- Δlog‖U‖ and Δlog‖W‖ can have opposite signs. The share is defined as `|Δlog‖U‖ − Δlog‖W‖| / (that + |Δlog η|)`, with the sign reported separately.

### B5. Two equilibrium formulas differ 4.4×
- Plain √(2ηλ) is 4.4× too high for AdamW here. Kosson's √(2ηλ(1−β₁)/(1+β₁)) matches within 0.4–5%. Report both; at most one can pass.

### B6. Model is not scale-invariant
- No QK-norm, and RMSNorm gains also get weight decay. Equilibrium theory assumes scale invariance. Results hold in practice, but this is a known limit.

## C. Training and data incidents

### C1. Truncated data shards — fixed
- **Seen:** Smoke test 1 crashed when reading the train shards.
- **Cause:** The FineWeb-edu `.bin` shards were truncated downloads (odd byte size), so the header token count did not match the file size.
- **Fix:** The reader uses the file size: `shape=((size - 1024) // 2,)`. Usable tokens: ~1.06B.

### C2. Variable shadowing broke batch 2 at step 0 — fixed
- **Seen:** All 4 batch-2 runs failed at step 0.
- **Cause:** A new stats line reused the name `a`, which already held the parsed command-line args.
- **Fix:** Rename. Smoke test (60 steps) before relaunch. Failed run dirs moved to `runs_failed/`, not deleted.
- **Lesson:** Any change to the training script gets a short smoke test before a batch launch.

### C3. Launch-script guard re-queued batch 1 — fixed
- **Seen:** Launching batch 3 also re-queued all 8 batch-1 runs. 4 of them ran ~250 steps and **appended** to the finished batch-1 logs.
- **Cause:** `launch.sh` ran batch 1 whenever the flag was not `batch2`.
- **Fix:** Batch 1 runs only with no flag (`[ -z "${2:-}" ]`). Runs killed within ~3 min. Logs restored from a local copy pulled before the relaunch (1001 records each, steps monotonic, verified).
- **Lessons:** (1) Guards must match an explicit value, not "anything else". (2) Pull a copy of finished logs before you touch the queue again. (3) A finished run dir should be write-protected (a `DONE` marker should stop a second writer).

### C4. AdamW at 4× lr was unstable — observed, not fixed
- AdamW lr 2.4e-3 linear ended at val loss 4.28 (others ~3.4). ‖Wq‖² − ‖Wk‖² grew to 1154 (≤ 77 elsewhere). Likely attention-logit growth without QK-norm; not verified.

## D. Infrastructure

### D1. Benchmark hung with 100% GPU and no output — open
- **Seen:** `bench/bench_stats.py` ran 24 min on 8 GPUs, and then also on 2 GPUs, at 100% util with no output. Killed.
- **Cause:** Not found yet. A likely candidate is a collective-order mismatch (an NCCL hang shows as 100% util).
- **What changed:** We added health criteria and an hourly watcher (`study/watch_remote.py`). The hang was found only because a human asked.
- **To do:** Add per-stage progress prints with flush, `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` and a timeout, then bisect by implementation.

### D2. gloo + torchrun hung on a host that cannot resolve its own hostname — worked around
- **Fix:** Use `mp.spawn` with `MASTER_ADDR=127.0.0.1` and `GLOO_SOCKET_IFNAME=lo`.

### D3. Access and fetching (minor)
- SSH access to the GPU host needed setup before the first run. Not related to the results.
- kexue.fm blocked plain HTTP fetches (403). We read the source post through a real browser.

## E. Process mistakes

### E1. Runs launched before the experiment doc was reviewed
- Our rule: write the experiment doc, have reviewers critique it, then launch. We launched first. The 4 reviewers said "keep the runs" but changed the analysis (B1, B3, B4, B5) and added batch 2 for clean pairs. Cost of the mistake: none this time, but it could have been 3 h of GPU.

### E2. Nobody watched the GPUs
- See D1. Fix: health criteria in `design/study.md` and an hourly, low-cost watcher that reads only those sections.

### E3. Claims from critics need checking
- A critic reported that torch 2.13 `clip_grad_norm_` fails on a 2-D mesh test. We did not reproduce this in isolation, so we make no claim about it.

## G. Found in the pre-release review (2026-10-29)
Four reviewers read the repo before it was made public. Fixed:
- **G1. Bad labels were silent.** `LogicalSpec` with a label ≥ `n_sub`, or a label count that did not match the global size, dropped or misplaced rows without error. Now both raise.
- **G2. Negative `dim` gave wrong sums.** `dim=-1` reduced every dimension. Now `dim` is normalized.
- **G3. `async_op=True` returned two types.** It gave `(work, buf)` on many ranks but `MatrixStats` on one process. Now it always gives `(work_or_None, buf)`, plus `MatrixStats.from_buffer`.
- **G4. Tests failed on macOS.** `GLOO_SOCKET_IFNAME="lo"` was hard-coded; macOS uses `lo0`. Now chosen by platform and only set if unset.
- **G5. The benchmark could waste GPUs.** It now refuses to run without `--i-know-it-hangs` and sets a 10-minute NCCL timeout.
- **G6. The report claimed more than the data.** The Q4 verdict "mostly not supported" was wrong by our own pre-registered rule: k differs from the whole QKV in ≥ 3 layers in 6 of 16 runs (all no-WD runs). Q5 fails the |cos| rule in the two 4× lr runs. Seed noise "≤ 3%" held only for Q1–Q3a. Rewritten as "mixed" with exact numbers.
- **Lesson:** after the final analysis, re-check every verdict against the pre-registered rule, run by run, not against the interim report.

## F. Still open
- D1 benchmark hang (no cost numbers yet).
- Muon equilibrium formula: partly resolved (2026-10-30, `analysis/muon_equilibrium.md`). AUS ≈ √((2ηλ + δ)/(1 + 2k)), k = −cos(W,U)/ELR = Σ_d C(d) (update autocorrelation); δ = per-step growth of ln‖W‖². Of the 13% gap to Kosson's β = 0.95 form, Nesterov weights (1 + 2k = 32.9, not 39) explain ~8 points. The rest: at 6e-4, non-equilibrium (δ/2ηλ = 0.09, +4%); at 2.4e-3, a weaker lean (k = 14.0, not 15.95, +6.5%). With measured k and δ the identity closes to 0.3%. Still open: why k is below the linear model. A white-noise toy shows NS5 lowers k (13.5 at 128×384, 11.0 square); real gradients partly cancel this, and k falls with η for AdamW too.
- Megatron's rank-local `[gate_r; up_r]` layout is not covered by `LogicalSpec`.
