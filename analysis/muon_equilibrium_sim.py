"""CPU control: equilibrium AUS of Muon-style updates on a scale-invariant toy.

python analysis/muon_equilibrium_sim.py

W is an m x n matrix. The gradient is orthogonal to W (scale invariance):
    g_t = P_perp(W_t) [ h * (W_t/|W_t| - W*) + xi_t ],  xi_t iid Gaussian, |xi| ~ 1.
h = 0 is the random-walk (white-noise) model of the equilibrium papers; h > 0 adds a
restoring force, so consecutive gradients become anti-correlated as the step grows.
Update: W <- (1 - eta*lam) W - eta*U, U = 0.2 sqrt(max(m,n)) * map(M),
    momentum "nes": m_t = b m_{t-1} + g_t, M = g_t + b m_t   (study/train.py)
    momentum "hb":  M = m_t
    map "ns5": train.py's Newton-Schulz (5 steps, bf16 or fp32), "polar": exact UV^T,
        "lin": M * sqrt(min(m,n))/|M|_F (same norm, linear in M).
NS5 runs in fp32 by default (bf16 gave the same k to 0.2% in a short check, 5x slower on CPU).\nReports, over the last 40% of steps: AUS, k = -cos(W,U)/ELR, and the predictions.
"""
import math
import torch

torch.set_num_threads(4)


def ns5(G, dtype):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(dtype)
    tr = X.size(0) > X.size(1)
    if tr:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(5):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    if tr:
        X = X.T
    return X.float()


def polar(G):
    U, _, Vh = torch.linalg.svd(G, full_matrices=False)
    return U @ Vh


def run(m=128, n=384, eta=0.02, lam=0.05, beta=0.95, mom="nes", fmap="ns5", h=0.0, steps=12000, seed=0, bf16=False):
    g_ = torch.Generator().manual_seed(seed)
    W = torch.randn(m, n, generator=g_) / math.sqrt(n)
    Wstar = torch.randn(m, n, generator=g_)
    Wstar /= Wstar.norm()
    buf = torch.zeros(m, n)
    scale = 0.2 * math.sqrt(max(m, n))
    eps = eta * lam
    E, K = [], []
    for t in range(steps):
        wn = W.norm()
        What = W / wn
        g = h * (What - Wstar) + torch.randn(m, n, generator=g_) / math.sqrt(m * n)
        g = g - (g * What).sum() * What  # P_perp
        buf.mul_(beta).add_(g)
        M = g + beta * buf if mom == "nes" else buf.clone()
        if fmap == "ns5":
            U = ns5(M, torch.bfloat16 if bf16 else torch.float32) * scale
        elif fmap == "polar":
            U = polar(M) * scale
        else:
            U = M * (math.sqrt(min(m, n)) / M.norm()) * scale
        if t >= 0.6 * steps:
            un = U.norm()
            e = eta * un / wn
            c = (W * U).sum() / (wn * un)
            E.append(e.item())
            K.append((-c / e).item())
        W = (1 - eps) * W - eta * U
    E, K = torch.tensor(E), torch.tensor(K)
    return eps, E.median().item(), K.mean().item()


def ratio(beta, mom):
    k = torch.arange(4000, dtype=torch.float64)
    a = beta ** k if mom == "hb" else beta ** (k + 1)
    if mom == "nes":
        a[0] = 1 + beta
    return (a.sum() ** 2 / (a ** 2).sum()).item()


if __name__ == "__main__":
    print("white noise (h=0): k and AUS vs map and momentum")
    for mom in ("hb", "nes"):
        R = ratio(0.95, mom)
        for fmap in ("lin", "polar", "ns5"):
            eps, e, k = run(mom=mom, fmap=fmap)
            print(f"  {mom:3s} {fmap:5s}  k={k:6.2f} (model {(R - 1) / 2:5.2f})  AUS={e:.5f}  "
                  f"pred sqrt(2eps/R)={math.sqrt(2 * eps / R):.5f}  ratio={e / math.sqrt(2 * eps / R):.3f}")
    print("matrix shape, white noise, Nesterov")
    R = ratio(0.95, "nes")
    for m, n, fmap in ((128, 128, "lin"), (128, 128, "polar"), (128, 128, "ns5"), (192, 192, "ns5"), (64, 256, "ns5")):
        eps, e, k = run(m=m, n=n, fmap=fmap, steps=6000)
        print(f"  {m}x{n} {fmap:5s}  k={k:6.2f}  AUS/pred={e / math.sqrt(2 * eps / R):.3f}")
    print("restoring force h>0, Nesterov + NS5, lr x1 and x4 at fixed lambda (as in the runs)")
    R = ratio(0.95, "nes")
    for h in (0.0, 0.5, 1.0, 2.0):
        for eta in (0.005, 0.02):
            eps, e, k = run(eta=eta, h=h, steps=int(12000 * 0.02 / eta) if eta < 0.02 else 12000)
            print(f"  h={h:3.1f} eta={eta:.3f}  k={k:6.2f}  AUS/pred={e / math.sqrt(2 * eps / R):.3f}")
