# DESIGN: Per-matrix ELR measurement that is cheap and correct under sharding

## Background
- Two recent posts argue that the per-matrix effective learning rate (ELR = η‖u‖/‖w‖, the relative step) drives training, not the nominal lr (Su Jianlin 2026-10-27; Jiaxuan Zou 2026-08-25).
- Directly controlling ELR already exists (NVIDIA Emerging-Optimizers `RelativeUpdateHook` / `HyperballHook`; LARS/LAMB). Batched per-parameter norm logging also exists (OLMo `optim.py`, Levanter `WatchCallback`, DTensor `_NormPartial`).
- What is missing: (a) ELR per **logical** matrix (q/k/v inside a fused QKV, one expert inside a 3-D MoE weight), correct under FSDP2 / TP / FSDP2×TP; (b) measurements of how per-matrix ELR actually evolves in pretraining, and how much of it is implicit (set by norm dynamics) versus set by the schedule.
- Larger goal: make ELR a trustworthy, cheap, first-class quantity in a pretraining stack, so ELR-based schedules and transfer can be studied (a separate LR-schedule survey, not in this repo).

## Goals
1. **Library `elrstats` (monitor first).** Per logical matrix: s_ww, s_uu, s_wu with ONE all-reduce per step over the world group; ELR, cos(w,u), AUS derived in fp64. Supports DTensor (Shard, `_StridedShard`, Replicate; Partial rejected), manual TP shards, empty/uneven shards, and `LogicalSpec` (labels along one dim) for fused QKV / gate-up / MoE experts.
2. **Correctness evidence.** Tests vs. an unsharded fp64 reference: 1-D Shard(0)/Shard(1)/Replicate, real FSDP2×TP (2×2, 2×4), interleaved GQA QKV, 3-D MoE expert-parallel, manual TP rows; fp32 and bf16; gloo (CPU) and NCCL (8×A100 PCIe).
3. **Cost evidence (expected: negligible; reported honestly).** Micro-benchmark vs. strong baselines (NVIDIA-hook logic on DTensor, per-matrix all-reduce, `_coalescing_manager`, DTensor-native batched `stack(...).full_tensor()`), plus end-to-end FSDP2 step time at 124M/350M.
4. **Measurement study (the main result).** Train GPT-124M on FineWeb-edu with AdamW and Muon (standard schedules), log per-logical-matrix ELR every few steps. Questions: how different are layers and q/k/v; how much of the ELR decay is implicit (‖w‖ growth) versus scheduled; does a whole-fused-matrix ELR hide the k-ELR.
5. **Real-world counterexample (stretch).** A real training run where local-norm relative-update / Hyperball under manual TP (or whole-fused-matrix ELR) visibly diverges from the reference.

## Non-goals
- Not a new optimizer; the update rule is LAMB / RelativeUpdateHook.
- Not claiming "DTensor gives wrong norms": in torch 2.13 DTensor norms are globally correct (just one collective per call).
- No Megatron flat-buffer optimizer, no pipeline parallel in v0.
- No claim about the best ELR schedule shape (later).

## Design
- **One all-reduce.** Each rank sums its local shard (fp32 inputs, fp32 accumulation), divides by the number of ranks holding that same shard (= world / n_distinct_shards, a power of two, exact), writes into a fixed `[3, n_slots]` buffer, and all-reduces once over the world group. Rule for n_distinct_shards: product of mesh sizes over shard-like placements; Replicate adds none; Partial or unknown placement → error.
- **Logical sub-matrices.** `LogicalSpec(dim, labels, names)`: one label per global index along `dim`. Local labels come from `compute_local_shape_and_global_offset` (verified contiguous for FSDP2×TP `_StridedShard` on torch 2.13). Per-label sums by `index_add_`. Manual TP passes its own offset. Megatron's rank-local `[gate_r; up_r]` layout is not covered yet.
- **Derived values in fp64.** ELR = lr·√s_uu/√s_ww; cos = s_wu/√(s_ww s_uu); AUS = 2 sin(θ/2), θ = atan2(lr·√(s_ww s_uu − s_wu²), s_ww − lr·s_wu). No `1 − cos` cancellation.
- **Monitor vs control.** Monitor: compute after backward or after step, optionally `async_op=True`, read later; no host sync. Control (relative / Hyperball with global norms) is a thin follow-up: zero-norm rule falls back to absolute lr instead of zeroing the update.
- **Where U comes from.** For the study: AdamW U = m̂/(√v̂+ε) recomputed from optimizer state on logging steps; Muon U = the orthogonalized update. Stats on fp32 master weights.

## Constraints
- Hard: one 8×A100-PCIE-40GB host (PHB, no NVLink); two extra 2×A100 PCIe hosts. Jobs via `expq` (task-spooler wrapper). torch 2.13.0+cu130.
- Hard: no NVLink → absolute cost numbers are PCIe-specific.
- Hard: compute time on the 8-GPU host is limited.
- Soft: 124M scale for the study tonight; larger later.

## Related work (declared in discussion)
- Su Jianlin, 《从参数对称性到有效学习率》 — https://kexue.fm/archives/11985 (2026-10-27).
- Jiaxuan Zou, Hyperball / effective lr / peak-plus-decay — https://jiaxuanzou0714.github.io/blog/2026/hyperball-implicit-lr-schedule/ (2026-08-25).
- NVIDIA Emerging-Optimizers `RelativeUpdateHook` / `HyperballHook` (PR #224, 2026-08-18): same update rule, whole-parameter local-tensor norms — https://github.com/NVIDIA-NeMo/Emerging-Optimizers/pull/224.
- LARS (arXiv 1708.03888) / LAMB (arXiv 1904.00962): layer-wise trust ratio.
- OLMo `olmo/optim.py`: per-parameter norms packed into one all-reduce under FSDP — https://github.com/allenai/OLMo/blob/main/olmo/optim.py. Difference: no logical sub-matrices, no update/ELR/cos/AUS, FSDP1 flat params.
- Levanter `WatchCallback` (JAX, sharding automatic) — https://github.com/stanford-crfm/levanter/blob/main/src/levanter/callbacks/watch.py.

## Resources
- Code: `./elrstats/stats.py`, `./tests/test_stats_dist.py`, `./bench/bench_stats.py`, `./study/` (training study).
- Remote: `/data/exp_elrshard/` on the 8-GPU host (code, logs, results, markers).
- Data: `/data/datasets/fwedu10B-gpt2/` on the 8-GPU host (6 train shards + val, modded-nanoGPT .bin format).
- Reports: `PITFALLS.md`, `results/final/REPORT.md`. Planning notes (scoping, LR-schedule survey) are kept outside this repo. Emerging-Optimizers was read at commit `ccc9968a`.

## Log
- 2026-10-29 — Project started (user decision): sharding correctness and cost of per-matrix stats, not a new optimizer. Source: discussion + scoping agent.
- 2026-10-29 — 10-critic review changed the goals. Reasons: (1) DTensor `vector_norm` is globally correct in torch 2.13; NVIDIA hook on DTensor is correct, only costs ~4 collectives per matrix (critics 1, 3, 4, 8, 10, verified). (2) Packed one-reduce norm logging already exists in OLMo / Levanter (critic 7, verified). (3) Real bugs found in the first design: `_StridedShard` not detected (36% error, silent), per-mesh-dim grouping wrong on 2-D meshes (critics 1, 3). (4) Fused QKV / MoE need logical sub-matrices; whole-param norm hides k (critic 6). (5) AUS must be computed in fp64 with atan2 (critic 5). (6) Megatron does not yet pass `weight_update_hook` to TP Muon, so the local-norm TP bug has no real victim today; `muon_hyperball` under TP is the reachable path (critic 4). → Goals rewritten: monitor library with logical sub-matrices + measurement study is the main result; correctness/cost become supporting evidence.
- 2026-10-29 — Tests: all cases pass on NCCL world 2/4/8 (8×A100 PCIe), errors ≤ 2e-7 (fp32 and bf16). On CPU gloo, bf16 cases show ~1e-5 from CPU fp32 accumulation, not sharding. In the constructed GQA case (k scaled 0.1×), k's ELR is 4.1× the whole-fused-matrix ELR. Manual-TP local-norm scale (NVIDIA logic) gives 0.65–0.93 update error when shard update norms differ 10× (constructed; real size unknown).
- 2026-10-29 — Goal 4 measurement study finished (16 runs). Main results in `results/final/REPORT.md`. Logical sub-matrices (q/k/v) show little effect at 124M, which weakens Goal 1's sub-matrix selling point at this scale; the measurement results are the stronger output.
- 2026-10-29 — User decision: open-source the code and record every pitfall. Added `PITFALLS.md`. Host names, SSH routes and personal names removed from this doc and the scripts before publishing.
- 2026-10-29 — Pre-release review by 4 agents (privacy, code, claims, newcomer view). Fixed: label validation, negative dim, async return type, macOS test, benchmark hang guard, and overclaims in the report (Q4 is mixed, not negative). The sub-matrix selling point of Goal 1 is weaker with WD, but real without WD. Details in `PITFALLS.md` G.
