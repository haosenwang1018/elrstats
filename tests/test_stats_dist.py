"""Correctness of sharded per-matrix stats vs an unsharded fp64 reference.

Run: python tests/test_stats_dist.py --world 4 --backend gloo   (CPU)
     python tests/test_stats_dist.py --world 8 --backend nccl   (GPU)
Prints one JSON line per case and exits non-zero on any failure.
Tolerance: 2e-5 relative for sums and scale, absolute for cos; AUS error is scaled by sin^2(W,U).
"""

import argparse
import json
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

TOL = 2e-5  # fp32 accumulation over ~5e5 terms (bf16 inputs) on CPU


def upd_fn(w):
    """Update as a fixed elementwise function of W: local shards stay consistent."""
    return torch.sin(3.0 * w.float()) + 0.05 * w.float()


def compare(name, st, ref, extra=None):
    from elrstats.stats import relative_scales

    def rel(a, b):
        return ((a.cpu() - b) .abs() / b.abs().clamp_min(1e-300)).max().item()

    # AUS is ill-conditioned when U is nearly parallel to W: its relative error
    # is ~ (sum error) / sin^2(angle(W,U)). Scale the tolerance by that factor
    # and use absolute error where the reference AUS is ~0.
    sin2 = (1 - ref.cos() ** 2).clamp_min(1e-12)
    a_st, a_ref = st.aus(1e-4).cpu(), ref.aus(1e-4)
    aus_err = torch.where(a_ref > 1e-12, (a_st - a_ref).abs() / a_ref * sin2, (a_st - a_ref).abs())
    errs = {
        "w2": rel(st.w2, ref.w2),
        "u2": rel(st.u2, ref.u2),
        "cos_abs": (st.cos().cpu() - ref.cos()).abs().max().item(),
        "aus_cond": aus_err.max().item(),
        "scale": rel(relative_scales(st), relative_scales(ref)),
    }
    ok = all(v <= TOL for v in errs.values())
    out = {"case": name, "ok": ok, "n_slots": len(ref.w2), **{k: f"{v:.1e}" for k, v in errs.items()}}
    if extra:
        out.update(extra)
    return out


def worker(rank, world, backend, port, q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    if backend == "gloo":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    dist.init_process_group(backend, rank=rank, world_size=world)
    dev_type = "cuda" if backend == "nccl" else "cpu"
    if dev_type == "cuda":
        torch.cuda.set_device(rank)
    dev = torch.device(dev_type, rank) if dev_type == "cuda" else torch.device("cpu")

    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor, init_device_mesh
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

    from elrstats.stats import LogicalSpec, ParamEntry, StatsLayout, matrix_stats, reference_stats

    res = []
    g = torch.Generator().manual_seed(0)
    shapes = [(64, 48), (33, 17), (7, 128), (5, 3), (1, 257), (512, 1024)]

    for dtype in (torch.float32, torch.bfloat16):
        full = [torch.randn(s, generator=g).to(dtype) for s in shapes]
        mesh1 = init_device_mesh(dev_type, (world,))
        for tag, pl in [("shard0", [Shard(0)]), ("shard1", [Shard(1)]), ("rep", [Replicate()])]:
            ps = [distribute_tensor(w.to(dev), mesh1, pl) for w in full]
            us = [distribute_tensor(upd_fn(w).to(dtype).to(dev), mesh1, pl) for w in full]
            lay = StatsLayout([ParamEntry(f"m{i}", p) for i, p in enumerate(ps)])
            st = matrix_stats(lay, us)
            ref = reference_stats([w.float() for w in full], [upd_fn(w).to(dtype).float() for w in full])
            res.append(compare(f"1d_{tag}_w{world}_{str(dtype)[6:]}", st, ref))

    # Real FSDP2 x TP on a 2-D mesh (gives _StridedShard on colwise weights).
    if world >= 4 and world % 2 == 0:
        mesh = init_device_mesh(dev_type, (2, world // 2), mesh_dim_names=("dp", "tp"))
        torch.manual_seed(1)
        m = torch.nn.Sequential(
            torch.nn.Linear(24, 40, bias=False), torch.nn.Linear(40, 24, bias=False), torch.nn.LayerNorm(24, bias=False)
        ).to(dev)
        full_ref = [p.detach().clone().cpu() for p in m.parameters()]
        parallelize_module(m, mesh["tp"], {"0": ColwiseParallel(), "1": RowwiseParallel()})
        fully_shard(m, mesh=mesh["dp"])
        ps = list(m.parameters())
        us = [p.detach().clone() for p in ps]
        for u in us:
            u.to_local().copy_(upd_fn(u.to_local()))
        lay = StatsLayout([ParamEntry(f"p{i}", p) for i, p in enumerate(ps)])
        st = matrix_stats(lay, us)
        ref = reference_stats(full_ref, [upd_fn(w) for w in full_ref])
        res.append(compare(f"2d_fsdp2xtp_{2}x{world // 2}", st, ref, {"placements": [str(p.placements) for p in ps]}))

    # Fused QKV with GQA, rows interleaved per query group; FSDP Shard(0) cuts inside groups.
    n_groups, q_per, kv = 3, 4, 2
    block = [0] * q_per + [1] * kv + [2] * kv
    labels = torch.tensor(block * n_groups)
    spec = LogicalSpec(dim=0, labels=labels, names=["q", "k", "v"])
    wq = torch.randn(len(labels), 32, generator=g)
    wq[labels == 1] *= 0.1  # k much smaller than q: whole-matrix norm would hide it
    for tag, pl in [("shard0", [Shard(0)]), ("shard1", [Shard(1)])]:
        p = distribute_tensor(wq.to(dev), init_device_mesh(dev_type, (world,)), pl)
        u = distribute_tensor(upd_fn(wq).to(dev), p.device_mesh, pl)
        st = matrix_stats(StatsLayout([ParamEntry("qkv", p, spec=spec)]), [u])
        ref = reference_stats([wq], [upd_fn(wq)], [spec])
        whole = reference_stats([wq], [upd_fn(wq)])
        res.append(compare(f"fused_qkv_{tag}_w{world}", st, ref, {
            "elr_q_k_v_vs_whole": [round(x, 4) for x in (ref.elr(1.0) / whole.elr(1.0)).tolist()]}))

    # 3-D MoE weight [E, out, in], expert-parallel Shard(0); one slot per expert.
    E = 2 * world
    we = torch.randn(E, 16, 8, generator=g) * torch.linspace(0.5, 2, E)[:, None, None]
    espec = LogicalSpec(dim=0, labels=torch.arange(E), names=[f"e{i}" for i in range(E)])
    p = distribute_tensor(we.to(dev), init_device_mesh(dev_type, (world,)), [Shard(0)])
    u = distribute_tensor(upd_fn(we).to(dev), p.device_mesh, [Shard(0)])
    st = matrix_stats(StatsLayout([ParamEntry("moe", p, spec=espec)]), [u])
    res.append(compare(f"moe3d_ep_w{world}", st, reference_stats([we], [upd_fn(we)], [espec])))

    # Manual (Megatron-style) TP row shards: ours vs NVIDIA RelativeUpdateHook's local-norm logic.
    wm = torch.randn(64, 32, generator=g)
    um = upd_fn(wm)
    um[: 64 // world] *= 10.0  # make shard update norms unequal, as in a real layer
    rows = 64 // world
    w_sh, u_sh = wm[rank * rows:(rank + 1) * rows].to(dev), um[rank * rows:(rank + 1) * rows].to(dev)
    entry = ParamEntry("manual_tp", w_sh, tp_dim=0, tp_offset=rank * rows, tp_global=64, tp_size=world)
    st = matrix_stats(StatsLayout([entry]), [u_sh])
    ref = reference_stats([wm], [um])
    local_scale = (w_sh.norm() / u_sh.norm()).reshape(1)
    glob = (wm.norm() / um.norm()).item()
    err = ((local_scale - glob).abs() / glob)
    dist.all_reduce(err, op=dist.ReduceOp.MAX)
    # Relative error of the applied update vs the unsharded update.
    upd_local = u_sh * local_scale
    num = ((upd_local - u_sh * glob) ** 2).sum().reshape(1)
    den = ((u_sh * glob) ** 2).sum().reshape(1)
    dist.all_reduce(num); dist.all_reduce(den)
    res.append(compare(f"manual_tp_rows_w{world}", st, ref, {
        "nvidia_local_scale_max_rel_err": f"{err.item():.2e}",
        "nvidia_update_rel_err": f"{(num / den).sqrt().item():.2e}"}))

    if rank == 0:
        q.put(res)
    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--backend", default="gloo")
    ap.add_argument("--port", type=int, default=29640)
    a = ap.parse_args()
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    mp.spawn(worker, args=(a.world, a.backend, a.port, q), nprocs=a.world, join=True)
    res = q.get()
    for r in res:
        print(json.dumps(r))
    sys.exit(0 if all(r["ok"] for r in res) else 1)


if __name__ == "__main__":
    main()
