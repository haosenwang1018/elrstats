"""GPT-124M on FineWeb-edu, single GPU, logging per-logical-matrix ELR.

Every --log-every steps it records, for each logical matrix (q/k/v split out of
the fused qkv weight):
  w_norm, u_norm, cos(w,u), elr = lr*|U|/|W| (optimizer update, no WD),
  rel_change = |W_after - W_before|/|W_before| (includes WD),
  aus_true = chord between W_before/|W_before| and W_after/|W_after|,
  aus_pred = AUS from the three sums (elrstats), to check the approximation.

Run: python study/train.py --opt adamw --lr 6e-4 --wd 0.1 --sched linear --out runs/x
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from elrstats.stats import LogicalSpec, ParamEntry, StatsLayout, matrix_stats  # noqa: E402

DATA = os.environ.get("ELR_DATA", "/data/datasets/fwedu10B-gpt2")


# ---------------------------------------------------------------- data
class Tokens:
    def __init__(self, split, seed):
        files = sorted(f for f in os.listdir(DATA) if f.startswith(f"fwedu_{split}_"))
        # Some shards are truncated downloads (odd byte size, header count too large):
        # read only the complete uint16 tokens that are actually on disk.
        self.maps = []
        for f in files:
            path = os.path.join(DATA, f)
            n = (os.path.getsize(path) - 1024) // 2
            self.maps.append(np.memmap(path, dtype=np.uint16, mode="r", offset=1024, shape=(n,)))
        self.sizes = np.array([len(m) for m in self.maps])
        self.rng = np.random.default_rng(seed)

    def batch(self, B, T, dev):
        idx = self.rng.choice(len(self.maps), size=B, p=self.sizes / self.sizes.sum())
        rows = []
        for i in idx:
            s = self.rng.integers(0, self.sizes[i] - T - 1)
            rows.append(torch.from_numpy(self.maps[i][s : s + T + 1].astype(np.int64)))
        x = torch.stack(rows).to(dev, non_blocking=True)
        return x[:, :-1], x[:, 1:]


# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, h, n_layer):
        super().__init__()
        self.h = h
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.up = nn.Linear(d, 4 * d, bias=False)
        self.gate = nn.Linear(d, 4 * d, bias=False)
        self.down = nn.Linear(4 * d, d, bias=False)
        for m in (self.qkv, self.up, self.gate):
            nn.init.normal_(m.weight, std=0.02)
        for m in (self.o, self.down):
            nn.init.normal_(m.weight, std=0.02 / math.sqrt(2 * n_layer))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, D // self.h).unbind(2)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        y = self.n2(x)
        return x + self.down(F.silu(self.gate(y)) * self.up(y))


class GPT(nn.Module):
    def __init__(self, n_layer=12, d=768, h=12, vocab=50304):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.blocks = nn.ModuleList([Block(d, h, n_layer) for _ in range(n_layer)])
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, x, y):
        h = self.emb(x)
        for b in self.blocks:
            h = b(h)
        logits = self.head(self.norm(h)).float()
        return F.cross_entropy(logits.flatten(0, 1), y.flatten())


# ---------------------------------------------------------------- muon
@torch.no_grad()
def newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    tr = X.size(0) > X.size(1)
    if tr:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    if tr:
        X = X.T
    return X.float()


class Muon(torch.optim.Optimizer):
    """Muon with Moonlight scaling 0.2*sqrt(max(m,n)) so lr matches AdamW's update RMS.

    `splits` maps a param to the number of equal row blocks orthogonalized
    separately (3 for fused qkv). Keeps the last update in state['u'].
    """

    def __init__(self, params, lr, wd, momentum=0.95, splits=None):
        super().__init__(params, dict(lr=lr, wd=wd, momentum=momentum))
        self.splits = splits or {}

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(p)
                st["m"].mul_(g["momentum"]).add_(p.grad)
                m = p.grad.add(st["m"], alpha=g["momentum"])  # nesterov
                parts = m.chunk(self.splits.get(p, 1), 0)
                u = torch.cat([newton_schulz(x) * 0.2 * math.sqrt(max(x.shape)) for x in parts], 0)
                st["u"] = u
                p.mul_(1 - g["lr"] * g["wd"])
                p.add_(u, alpha=-g["lr"])


def adam_update(opt, p):
    """AdamW update direction U = m_hat/(sqrt(v_hat)+eps) from optimizer state (no WD)."""
    st = opt.state[p]
    g = opt.param_groups[0]
    b1, b2 = g["betas"]
    t = float(st["step"])
    m_hat = st["exp_avg"] / (1 - b1**t)
    v_hat = st["exp_avg_sq"] / (1 - b2**t)
    return m_hat / (v_hat.sqrt() + g["eps"])


# ---------------------------------------------------------------- main
def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    if a.sched == "const":
        return a.lr
    frac = (step - a.warmup) / max(1, a.steps - a.warmup)
    return a.lr * max(0.0, 1 - frac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt", choices=["adamw", "muon"], required=True)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--sched", choices=["linear", "const"], default="linear")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"))
    torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    model = GPT().to(dev)

    mats = {n: p for n, p in model.named_parameters() if p.ndim == 2 and "blocks" in n}
    others = [p for n, p in model.named_parameters() if n not in mats]
    if a.opt == "adamw":
        opts = [torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95), eps=1e-8)]
        mat_opt = opts[0]
    else:
        splits = {p: 3 for n, p in mats.items() if n.endswith("qkv.weight")}
        mat_opt = Muon(list(mats.values()), lr=a.lr, wd=a.wd, splits=splits)
        opts = [mat_opt, torch.optim.AdamW(others, lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95), eps=1e-8)]

    # Logical matrices: q/k/v out of fused qkv; everything else whole.
    D = model.emb.weight.shape[1]
    qkv_spec = LogicalSpec(dim=0, labels=torch.arange(3 * D) // D, names=["q", "k", "v"])
    names = list(mats.keys()) + ["emb.weight", "head.weight"]
    params = [mats[n] for n in mats] + [model.emb.weight, model.head.weight]
    entries = [ParamEntry(n, p, spec=qkv_spec if n.endswith("qkv.weight") else None) for n, p in zip(names, params)]
    layout = StatsLayout(entries)  # only used for slot names

    train, val = Tokens("train", a.seed), Tokens("val", 1234)
    logf = open(os.path.join(a.out, "elr.jsonl"), "a")
    evalf = open(os.path.join(a.out, "loss.jsonl"), "a")
    t0, tok = time.time(), 0
    for step in range(a.steps):
        lr = lr_at(step, a)
        for o in opts:
            for g in o.param_groups:
                g["lr"] = lr
        loss_acc = 0.0
        for _ in range(a.accum):
            x, y = train.batch(a.batch, a.seq, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x, y) / a.accum
            loss.backward()
            loss_acc += loss.item()
            tok += x.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        log_now = step % a.log_every == 0 or step == a.steps - 1
        if log_now:
            before = [p.detach().float().clone() for p in params]
        for o in opts:
            o.step()
        if log_now:
            with torch.no_grad():
                us = []
                for n, p in zip(names, params):
                    if a.opt == "muon" and n in mats:
                        us.append(mat_opt.state[p]["u"])
                    else:
                        o = opts[0] if a.opt == "adamw" else opts[1]
                        us.append(adam_update(o, p))
                # Stats are taken on the fp32 weights BEFORE the step, paired with U.
                blay = StatsLayout([ParamEntry(e.name, b, spec=e.spec) for e, b in zip(entries, before)])
                st = matrix_stats(blay, us)
                elr, cos, aus_pred = st.elr(lr), st.cos(), st.aus(lr)
                # Truth per logical matrix: actual change including WD.
                deltas = [p.detach().float() - b for p, b in zip(params, before)]
                dst = matrix_stats(blay, deltas)
                # Angle between W and W+delta from (|W|^2, <W,delta>, |delta|^2): no cancellation,
                # because <W,delta> is small relative to |W||delta|. (Batch 1 used W_after and is
                # unreliable for small angles; analysis uses aus_pred there.)
                s_ww, s_wd, s_dd = dst.w2, dst.wu, dst.u2
                theta = torch.atan2((s_ww * s_dd - s_wd * s_wd).clamp_min(0).sqrt(), s_ww + s_wd)
                gains = [m.weight.detach().float().norm().item() for m in model.modules() if isinstance(m, nn.RMSNorm)]
                rec = {"step": step, "lr": lr, "loss": loss_acc, "tok": tok, "w_dot_delta": s_wd.tolist(),
                       "rmsnorm_gain_norms": gains,
                       "names": layout.slot_names if step == 0 else None,
                       "w_norm": st.w2.sqrt().tolist(), "u_norm": st.u2.sqrt().tolist(),
                       "cos": cos.tolist(), "elr": elr.tolist(), "aus_pred": aus_pred.tolist(),
                       "rel_change": (dst.u2.sqrt() / st.w2.sqrt()).tolist(),
                       "aus_true": (2 * torch.sin(theta / 2)).tolist()}
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
            del before
        for o in opts:
            o.zero_grad(set_to_none=True)
        if step % a.eval_every == 0 or step == a.steps - 1:
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                vl = sum(model(*val.batch(16, a.seq, dev)).item() for _ in range(20)) / 20
            model.train()
            el = time.time() - t0
            evalf.write(json.dumps({"step": step, "tok": tok, "val_loss": vl, "train_loss": loss_acc,
                                    "lr": lr, "elapsed_s": el, "tok_per_s": tok / el}) + "\n")
            evalf.flush()
            print(f"step {step} val {vl:.4f} train {loss_acc:.4f} lr {lr:.2e} {tok / el:.0f} tok/s", flush=True)
    open(os.path.join(a.out, "DONE"), "w").write("ok\n")


if __name__ == "__main__":
    main()
