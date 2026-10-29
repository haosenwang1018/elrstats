"""Cost of per-matrix ELR stats under FSDP2 (DTensor Shard(0)) on N GPUs.

Two parts:
  micro: time ONE stats computation (s_ww, s_uu, s_wu for all matrices) for
         several implementations, sweeping the number of matrices.
  e2e:   step time of a GPT under FSDP2 + AdamW, with no stats / NVIDIA-hook
         style per-matrix stats / DTensor-native batched / ours (one packed
         all-reduce), every step.

Not a hang (PITFALLS.md, D1): the old fixed 200 iters made nvidia_hook_dtensor
at 1000 matrices take ~8 min alone (~2.2 s per call), with no output. Now each
stage prints flushed progress and gets about --budget-s seconds; the iteration
count is agreed across ranks (MAX of the probe time), so collectives stay in
step. A 10-minute NCCL timeout stays as a safety net. ~7-9 min on 2 or 8 A100s.

Run: python bench/bench_stats.py --world 8 --out results/bench_w8.json
"""

import argparse
import datetime
import json
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ---------------------------------------------------------------- impls
def impl_nvidia_hook(ws, us):
    """NVIDIA RelativeUpdateHook logic applied to DTensors: per-matrix norms."""
    out = []
    for w, u in zip(ws, us):
        wn = torch.linalg.vector_norm(w, dtype=torch.float32)
        un = torch.linalg.vector_norm(u, dtype=torch.float32)
        s = torch.where((wn < 1e-15) | (un < 1e-15), torch.zeros_like(un), wn / un.clamp_min(1e-15))
        out.append(s.full_tensor() if hasattr(s, "full_tensor") else s)
    return out


def impl_per_matrix_allreduce(ws, us):
    """Local [3] sums, one all-reduce per matrix."""
    out = []
    for w, u in zip(ws, us):
        wl, ul = w.to_local().float().reshape(-1), u.to_local().float().reshape(-1)
        b = torch.stack([wl @ wl, ul @ ul, wl @ ul])
        dist.all_reduce(b)
        out.append(b)
    return out


def impl_coalesced(ws, us):
    """Same as above, inside _coalescing_manager (one NCCL group launch)."""
    from torch.distributed.distributed_c10d import _coalescing_manager

    bufs = []
    for w, u in zip(ws, us):
        wl, ul = w.to_local().float().reshape(-1), u.to_local().float().reshape(-1)
        bufs.append(torch.stack([wl @ wl, ul @ ul, wl @ ul]))
    with _coalescing_manager(async_ops=True) as cm:
        for b in bufs:
            dist.all_reduce(b, async_op=True)
    cm.wait()
    return bufs


def impl_dtensor_batched(ws, us):
    """DTensor-native, batched: stack Partial(sum) scalars, one full_tensor()."""
    parts = []
    for w, u in zip(ws, us):
        wf, uf = w.float(), u.float()
        parts += [(wf * wf).sum(), (uf * uf).sum(), (wf * uf).sum()]
    return torch.stack(parts).full_tensor()


def make_ours(ws):
    from elrstats.stats import ParamEntry, StatsLayout, matrix_stats

    lay = StatsLayout([ParamEntry(f"m{i}", w) for i, w in enumerate(ws)])

    def f(ws_, us_):
        return matrix_stats(lay, us_)

    return f


# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        self.up, self.gate, self.down = (nn.Linear(d, 4 * d, bias=False), nn.Linear(d, 4 * d, bias=False),
                                         nn.Linear(4 * d, d, bias=False))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, D // self.h).unbind(2)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        y = self.n2(x)
        return x + self.down(F.silu(self.gate(y)) * self.up(y))


class GPT(nn.Module):
    def __init__(self, n_layer, d, h, vocab=50304):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(n_layer)])
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x)
        return F.cross_entropy(self.head(self.norm(x)).float().flatten(0, 1), idx.roll(-1, 1).flatten())


# Named by shape (layers, width), not by parameter count: with SwiGLU 4d and untied 50304-vocab
# embedding/head, "l12-d768" is 113M block / 191M total parameters, not GPT-2-small's 124M.
CONFIGS = {"l12-d768": (12, 768, 12), "l24-d1024": (24, 1024, 16), "l24-d1536": (24, 1536, 16)}


# ---------------------------------------------------------------- timing
def log(msg):
    r = dist.get_rank() if dist.is_initialized() else -1
    print(f"[{time.strftime('%H:%M:%S')} r{r}] {msg}", flush=True)


def cuda_time(fn, iters, warmup, budget_s):
    """Time fn. iters/warmup shrink so one stage takes about budget_s.

    The shrunk counts come from the MAX first-call time over ranks, so every
    rank runs the same number of calls (a per-rank count would desync the
    collectives inside fn and hang).
    """
    fn()  # first call: lazy init, not representative
    torch.cuda.synchronize()
    t = time.time()
    fn()
    torch.cuda.synchronize()
    t1 = torch.tensor([time.time() - t], device="cuda", dtype=torch.float64)
    dist.all_reduce(t1, op=dist.ReduceOp.MAX)
    t1 = t1.item()
    n_iter = max(5, min(iters, int(budget_s / max(t1, 1e-6))))
    n_warm = min(warmup, max(1, n_iter // 5))
    log(f"  probe call {t1:.3f}s -> warmup {n_warm}, iters {n_iter}")
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    last = time.time()
    for it in range(n_iter):
        if time.time() - last > 30:
            log(f"  iter {it}/{n_iter}")
            last = time.time()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        dist.barrier()
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return {"p50_ms": ts[len(ts) // 2], "p90_ms": ts[int(len(ts) * 0.9)], "min_ms": ts[0], "iters": n_iter}


def worker(rank, world, port, args, q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", rank),
                            timeout=datetime.timedelta(minutes=10))
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

    mesh = init_device_mesh("cuda", (world,))
    res = {"world": world, "micro": [], "e2e": []}

    # ---- micro: sweep number of matrices (1024x1024 fp32 each, Shard(0))
    for n in args.micro_n:
        g = torch.Generator(device="cuda").manual_seed(0)
        ws = [distribute_tensor(torch.randn(1024, 1024, device="cuda", generator=g), mesh, [Shard(0)]) for _ in range(n)]
        us = [distribute_tensor(torch.randn(1024, 1024, device="cuda", generator=g), mesh, [Shard(0)]) for _ in range(n)]
        ours = make_ours(ws)
        impls = {"nvidia_hook_dtensor": impl_nvidia_hook, "per_matrix_allreduce": impl_per_matrix_allreduce,
                 "coalesced": impl_coalesced, "dtensor_batched": impl_dtensor_batched, "ours_packed": ours}
        for name, fn in impls.items():
            if args.impls and name not in args.impls:
                continue
            log(f"micro n={n} {name} start")
            r = cuda_time(lambda: fn(ws, us), args.iters, args.warmup, args.budget_s)
            res["micro"].append({"n_matrices": n, "impl": name, **r})
            log(f"micro n={n} {name} done {r}")
        del ws, us
        torch.cuda.empty_cache()

    # ---- e2e: FSDP2 GPT training step with stats every step
    for cfg in args.models:
        L, D, H = CONFIGS[cfg]
        torch.manual_seed(0)
        model = GPT(L, D, H).cuda()
        mp_pol = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for b in model.blocks:
            fully_shard(b, mesh=mesh, mp_policy=mp_pol)
        fully_shard(model, mesh=mesh, mp_policy=mp_pol)
        mats = [p for p in model.parameters() if p.ndim == 2]
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1, foreach=True)
        ours = make_ours(mats)
        x = torch.randint(0, 50304, (args.batch, args.seq), device="cuda")
        modes = {
            "none": None,
            "nvidia_hook_dtensor": lambda: impl_nvidia_hook(mats, [p.grad for p in mats]),
            "dtensor_batched": lambda: impl_dtensor_batched(mats, [p.grad for p in mats]),
            "ours_packed": lambda: ours(mats, [p.grad for p in mats]),
        }
        for mode, stat_fn in modes.items():
            if args.impls and mode != "none" and mode not in args.impls:
                continue
            log(f"e2e {cfg} {mode} start")

            def step():
                loss = model(x)
                loss.backward()
                if stat_fn is not None:
                    stat_fn()
                opt.step()
                opt.zero_grad(set_to_none=True)

            r = cuda_time(step, args.e2e_iters, args.warmup, args.budget_s)
            res["e2e"].append({"model": cfg, "n_matrices": len(mats), "mode": mode, **r,
                               "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30})
            torch.cuda.reset_peak_memory_stats()
            log(f"e2e {cfg} {mode} done {res['e2e'][-1]}")
        del model, opt, mats
        torch.cuda.empty_cache()

    if rank == 0:
        q.put(res)
    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--port", type=int, default=29720)
    ap.add_argument("--micro-n", type=int, nargs="*", default=[10, 50, 150, 400, 1000])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--models", nargs="*", default=["l12-d768", "l24-d1024"])
    ap.add_argument("--e2e-iters", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--budget-s", type=float, default=20.0,
                    help="target seconds per timed stage; slow impls get fewer iters")
    ap.add_argument("--impls", nargs="*", default=None, help="run only these impls/modes")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    t0 = time.time()
    mp.spawn(worker, args=(a.world, a.port, a, q), nprocs=a.world, join=True)
    res = q.get()
    res["wall_s"] = time.time() - t0
    res["torch"] = torch.__version__
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
