"""Sharding-correct, batched per-matrix statistics for ELR / AUS / cos(w, u).

For every logical matrix i we need three global sums over the unsharded tensor:
    s_ww = <W, W>,  s_uu = <U, U>,  s_wu = <W, U>
A "logical matrix" is a whole parameter, or a labelled slice of one along one
dim (q/k/v rows of a fused QKV weight, one expert of a 3-D MoE weight).

Each rank sums over its local shard in fp32, divides by the number of ranks
that hold the same shard (a power of two in practice, so the division is
exact), writes into a packed [3, n_slots] buffer, and all ranks do ONE
all-reduce over the world group. Derived quantities are computed in fp64.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

try:
    from torch.distributed.tensor.placement_types import _StridedShard
except ImportError:  # pragma: no cover
    _StridedShard = ()  # type: ignore[assignment]


def _is_shard_like(p) -> bool:
    return p.is_shard() or isinstance(p, _StridedShard)


def _n_distinct_shards(t: torch.Tensor, manual_tp_size: int) -> int:
    if isinstance(t, DTensor):
        n = 1
        for dim, p in enumerate(t.placements):
            if p.is_partial():
                raise ValueError("Partial placement on a param/update; reduce it first")
            if _is_shard_like(p):
                n *= t.device_mesh.size(dim)
            elif not p.is_replicate():
                raise ValueError(f"Unknown placement {p!r}; refusing to guess its reduction")
        return n
    return manual_tp_size


@dataclass
class LogicalSpec:
    """Split one parameter into labelled sub-matrices along `dim`.

    labels: one entry per GLOBAL index along `dim`, values in [0, n_sub).
    Rows with the same label form one logical matrix. A list is converted to
    an int64 tensor. `dim` may be negative.
    """

    dim: int
    labels: torch.Tensor
    names: Sequence[str]

    def __post_init__(self):
        self.labels = torch.as_tensor(self.labels, dtype=torch.int64).reshape(-1)
        if self.labels.numel() and (self.labels.min() < 0 or self.labels.max() >= self.n_sub):
            raise ValueError(f"labels must be in [0, {self.n_sub})")

    @property
    def n_sub(self) -> int:
        return len(self.names)

    @classmethod
    def blocked(cls, dim: int, sizes: Sequence[int], names: Sequence[str], repeats: int = 1) -> "LogicalSpec":
        """Labels for a pattern of contiguous blocks, tiled `repeats` times.

        Block j has sizes[j] rows and label j. Example: sizes=[k, k], repeats=tp
        gives [0]*k + [1]*k + [0]*k + [1]*k + ... (tp times).
        """
        if len(sizes) != len(names) or repeats < 1 or min(sizes, default=0) < 0:
            raise ValueError("need one non-negative size per name and repeats >= 1")
        block = torch.repeat_interleave(torch.arange(len(sizes)), torch.as_tensor(sizes, dtype=torch.int64))
        return cls(dim=dim, labels=block.repeat(repeats), names=names)

    @classmethod
    def megatron_glu(cls, ffn: int, tp: int, dim: int = 0, names: Sequence[str] = ("gate", "up")) -> "LogicalSpec":
        """Megatron-LM fused GLU `linear_fc1` weight (or bias), global rows = 2*ffn.

        Megatron stores [gate_r; up_r] on each TP rank r, so the concatenation of
        the TP shards is [gate_0; up_0; gate_1; up_1; ...] with blocks of ffn/tp
        rows. `tp` is the TP size the weight was BUILT with; the labels describe
        that global order, so any later Shard/FSDP split of it is still correct.
        """
        if ffn % tp:
            raise ValueError(f"ffn={ffn} not divisible by tp={tp}")
        return cls.blocked(dim, [ffn // tp] * 2, names, repeats=tp)


@dataclass
class ParamEntry:
    name: str
    param: torch.Tensor
    spec: LogicalSpec | None = None
    # Manual (non-DTensor) TP shard: offset of the local block along `tp_dim`
    # in the global tensor, the global size along it, and the TP world size.
    tp_dim: int | None = None
    tp_offset: int = 0
    tp_global: int | None = None
    tp_size: int = 1
    slots: list[int] = field(default_factory=list)


def _local_offset(e: ParamEntry, dim: int) -> int:
    t = e.param
    if isinstance(t, DTensor):
        _, off = compute_local_shape_and_global_offset(t.shape, t.device_mesh, t.placements)
        return off[dim]
    return e.tp_offset if e.tp_dim == dim else 0


def _local(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if isinstance(t, DTensor) else t


class StatsLayout:
    """Fixed slot layout for a list of parameters; build once, reuse every step."""

    def __init__(self, entries: Sequence[ParamEntry]):
        self.entries = list(entries)
        self.slot_names: list[str] = []
        for e in self.entries:
            if e.spec is None:
                e.slots = [len(self.slot_names)]
                self.slot_names.append(e.name)
            else:
                nd = e.param.ndim
                e.spec.dim %= nd
                d = e.spec.dim
                glob = e.tp_global if (e.tp_dim is not None and e.tp_dim % nd == d) else e.param.shape[d]
                if glob is not None and len(e.spec.labels) != glob:
                    raise ValueError(f"{e.name}: {len(e.spec.labels)} labels for global size {glob} along dim {d}")
                e.slots = list(range(len(self.slot_names), len(self.slot_names) + e.spec.n_sub))
                self.slot_names += [f"{e.name}.{s}" for s in e.spec.names]
        self.n = len(self.slot_names)
        self._local_labels: dict[int, torch.Tensor] = {}

    def _labels_for(self, i: int, e: ParamEntry, local_len: int, device) -> torch.Tensor:
        key = i
        if key not in self._local_labels:
            off = _local_offset(e, e.spec.dim)
            lab = e.spec.labels[off : off + local_len].to(device)
            self._local_labels[key] = lab + e.slots[0]
        return self._local_labels[key]

    def local_sums(self, updates: Sequence[torch.Tensor], world: int) -> torch.Tensor:
        dev = _local(self.entries[0].param).device
        out = torch.zeros(3, self.n, dtype=torch.float32, device=dev)
        for i, (e, u) in enumerate(zip(self.entries, updates)):
            w_l, u_l = _local(e.param), _local(u)
            if isinstance(e.param, DTensor) and isinstance(u, DTensor) and e.param.placements != u.placements:
                raise ValueError(f"{e.name}: update placements {u.placements} != param {e.param.placements}")
            if w_l.shape != u_l.shape:
                raise ValueError(f"{e.name}: local shapes differ {w_l.shape} vs {u_l.shape}")
            replicas = world // _n_distinct_shards(e.param, e.tp_size)
            if w_l.numel() == 0:
                continue
            wf, uf = w_l.float(), u_l.float()  # fp32 inputs, fp32 accumulation
            if e.spec is None:
                wv, uv = wf.reshape(-1), uf.reshape(-1)
                out[0, e.slots[0]] = torch.dot(wv, wv)
                out[1, e.slots[0]] = torch.dot(uv, uv)
                out[2, e.slots[0]] = torch.dot(wv, uv)
            else:
                d = e.spec.dim
                red = [k for k in range(wf.ndim) if k != d]
                per = torch.stack([(wf * wf).sum(red), (uf * uf).sum(red), (wf * uf).sum(red)])
                lab = self._labels_for(i, e, wf.shape[d], dev)
                out.index_add_(1, lab, per)
            if replicas > 1:
                cols = e.slots if e.spec is not None else e.slots[:1]
                out[:, cols] /= replicas
        return out


@dataclass
class MatrixStats:
    w2: torch.Tensor  # fp64 [n]
    u2: torch.Tensor
    wu: torch.Tensor

    @classmethod
    def from_buffer(cls, buf: torch.Tensor) -> "MatrixStats":
        """Build from the [3, n] buffer returned with async_op=True, after work.wait()."""
        r = buf.double()
        return cls(w2=r[0], u2=r[1], wu=r[2])

    def elr(self, lr) -> torch.Tensor:
        return lr * self.u2.sqrt() / self.w2.sqrt()

    def cos(self) -> torch.Tensor:
        return self.wu / (self.w2 * self.u2).sqrt()

    def aus(self, lr) -> torch.Tensor:
        """Chord between W/|W| and W'/|W'|, W' = W - lr*U, via a stable atan2."""
        a, b, c = self.w2, self.wu, self.u2
        theta = torch.atan2(lr * (a * c - b * b).clamp_min(0).sqrt(), a - lr * b)
        return 2 * torch.sin(theta / 2)


def matrix_stats(layout: StatsLayout, updates: Sequence[torch.Tensor], group=None, async_op: bool = False):
    """Global stats with ONE all-reduce.

    `group` must contain every rank of the meshes the params live on (default:
    the world group); the replica count is computed from its size.
    With async_op=True returns (work, buf); work is None on a single process.
    Call work.wait() if not None, then MatrixStats.from_buffer(buf).
    """
    world = dist.get_world_size(group) if dist.is_initialized() else 1
    buf = layout.local_sums(updates, world)
    work = None
    if world > 1:
        work = dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=group, async_op=async_op)
    if async_op:
        return work, buf
    return MatrixStats.from_buffer(buf)


def reference_stats(full_params: Sequence[torch.Tensor], full_updates: Sequence[torch.Tensor], specs=None):
    """Unsharded fp64 reference, same slot order as StatsLayout."""
    w2, u2, wu = [], [], []
    specs = specs or [None] * len(full_params)
    for w, u, s in zip(full_params, full_updates, specs):
        wd, ud = w.double(), u.double()
        if s is None:
            w2.append((wd * wd).sum()); u2.append((ud * ud).sum()); wu.append((wd * ud).sum())
        else:
            red = [k for k in range(wd.ndim) if k != s.dim % wd.ndim]
            per = torch.stack([(wd * wd).sum(red), (ud * ud).sum(red), (wd * ud).sum(red)])
            acc = torch.zeros(3, s.n_sub, dtype=torch.float64).index_add_(1, s.labels, per)
            w2 += list(acc[0]); u2 += list(acc[1]); wu += list(acc[2])
    return MatrixStats(w2=torch.stack(w2), u2=torch.stack(u2), wu=torch.stack(wu))


def relative_scales(stats: MatrixStats, w_floor: torch.Tensor | float = 0.0) -> torch.Tensor:
    """s_i = |W_i|/|U_i|, so lr*s_i*U_i has relative size lr (Su's covariant update).

    |U_i| == 0 -> 0 (no update). |W_i| <= w_floor -> 1 (fall back to absolute lr),
    so zero-initialized matrices can still learn. No host sync.
    """
    w, u = stats.w2.sqrt(), stats.u2.sqrt()
    s = w / torch.where(u > 0, u, torch.ones_like(u))
    s = torch.where(w <= w_floor, torch.ones_like(s), s)
    return torch.where(u > 0, s, torch.zeros_like(s))
