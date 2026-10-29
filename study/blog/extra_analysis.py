"""Extra analyses behind the blog post on the ELR study (norm balance, Q1 base-step
sensitivity, implicit share over time, |U| drift, q-k symmetry balance).

Usage (from the repo root):
    python study/blog/extra_analysis.py [--runs results/runs] [--summary results/final/summary.json]
                                        [--out extra_analysis.json]

Inputs: the raw per-run logs in results/runs/ (not in git; download the archive from the
v0.1-study GitHub release and extract it at the repo root, see README) and
results/final/summary.json. Defaults are repo-relative; ELR_RUNS / ELR_SUMMARY / ELR_EXTRA_OUT
env vars override them. Prints tables to stdout; writes one JSON file only if --out is given.
"""
import argparse, json, math, os, sys
import numpy as np
sys.dont_write_bytecode = True
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "study"))
from analyze import load, at, block_idx, decompose  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--runs", default=os.environ.get("ELR_RUNS", os.path.join(REPO, "results", "runs")))
ap.add_argument("--summary", default=os.environ.get("ELR_SUMMARY", os.path.join(REPO, "results", "final", "summary.json")))
ap.add_argument("--out", default=os.environ.get("ELR_EXTRA_OUT"), help="optional JSON output path")
cli = ap.parse_args()
RUNS = cli.runs
out = {}

def R_(name):
    return load(os.path.join(RUNS, name))

# 1. Norm balance in the last 10% of const+WD runs: decay loss vs update gain, per matrix per record.
print("== norm balance, last 10% (median over records of median over block matrices)")
for name in ["adamw-lr6e-4-wd0.1-const", "adamw-lr2.4e-3-wd0.1-const", "muon-lr6e-4-wd0.1-const", "muon-lr2.4e-3-wd0.1-const",
             "adamw-lr6e-4-wd0.1-const-s1", "muon-lr6e-4-wd0.1-const-s1"]:
    R = R_(name); idx = block_idx(R); T = R["args"]["steps"]; lam = R["args"]["wd"]
    m = R["steps"] >= 0.9 * T
    eta = R["lr"][m][:, None]
    w = R["w_norm"][m][:, idx]; u = R["u_norm"][m][:, idx]; c = R["cos"][m][:, idx]
    loss_ = (1 - (1 - eta * lam) ** 2) * w ** 2
    sq = eta ** 2 * u ** 2
    cross = -2 * eta * (1 - eta * lam) * w * u * c
    gain = sq + cross
    ratio_balance = np.median(np.median(gain / loss_, axis=1))
    ratio_balance_nocross = np.median(np.median(sq / loss_, axis=1))
    cross_over_sq = np.median(np.median(cross / sq, axis=1))
    frac_neg_cos = float((np.median(c, axis=1) < 0).mean())
    # growth of median |W| over the last 10%, per 1000 steps
    s0, s1 = 0.9 * T, 0.99 * T
    g = math.log(at(R, s1, "w_norm", idx) / at(R, s0, "w_norm", idx)) / ((s1 - s0) / 1000)
    tau = 1 / (R["args"]["lr"] * lam)
    out[name] = dict(gain_over_loss=ratio_balance, sq_only_over_loss=ratio_balance_nocross, cross_over_sq=cross_over_sq,
                     frac_records_median_cos_negative=frac_neg_cos, dlogW_per_1k_last10=g, timescale_steps=tau,
                     horizon_in_timescales=T / tau, median_cos=float(np.median(c)))
    print(f"{name:30s} gain/loss={ratio_balance:.3f} sq-only/loss={ratio_balance_nocross:.3f} cross/sq={cross_over_sq:.1f} "
          f"cos_med={np.median(c):+.4f} negcos={frac_neg_cos:.2f} dlnW/1k={g:+.4f} tau={tau:.0f} T/tau={T/tau:.2f}")

# 2. Q1 base-step sensitivity: median ELR at 95% / at base.
print("== Q1 ELR(95%)/ELR(base)")
for name in ["adamw-lr6e-4-wd0-const", "muon-lr6e-4-wd0-const", "adamw-lr6e-4-wd0.1-const", "muon-lr6e-4-wd0.1-const",
             "adamw-lr2.4e-3-wd0.1-const", "muon-lr2.4e-3-wd0.1-const"]:
    R = R_(name); idx = block_idx(R); T = R["args"]["steps"]
    vals = {b: at(R, 0.95 * T, "elr", idx) / at(R, b, "elr", idx) for b in (1000, 2000, 5000)}
    out.setdefault("q1_base", {})[name] = vals
    print(f"{name:30s} " + " ".join(f"base{b}={v:.3f}" for b, v in vals.items()))

# 3. Implicit share over time, linear runs, base 2000.
print("== implicit share (base 2000) at 25/50/75/95%")
for name in ["adamw-lr6e-4-wd0.1-lin", "muon-lr6e-4-wd0.1-lin", "adamw-lr6e-4-wd0-lin", "muon-lr6e-4-wd0-lin",
             "adamw-lr2.4e-3-wd0.1-lin", "muon-lr2.4e-3-wd0.1-lin"]:
    R = R_(name)
    d = [x for x in decompose(R, base_steps=(2000,))]
    out.setdefault("share", {})[name] = {x["frac"]: (x["implicit_share"], x["dlog_u"], x["dlog_inv_w"], x["dlog_lr"]) for x in d}
    print(f"{name:30s} " + " ".join(f"{x['frac']}:{x['implicit_share']:.2f}(u{x['dlog_u']:+.2f},w{x['dlog_inv_w']:+.2f})" for x in d))

# 4. |U| change for const runs (25%, 95% vs base 2000).
print("== median |U| relative to step 2000")
for name in ["adamw-lr6e-4-wd0-const", "muon-lr6e-4-wd0-const", "adamw-lr6e-4-wd0.1-const", "muon-lr6e-4-wd0.1-const",
             "adamw-lr2.4e-3-wd0.1-const", "muon-lr2.4e-3-wd0.1-const"]:
    R = R_(name); idx = block_idx(R); T = R["args"]["steps"]
    print(f"{name:30s} " + " ".join(f"{f}:{at(R, f*T, 'u_norm', idx)/at(R, 2000, 'u_norm', idx):.3f}" for f in (0.25, 0.5, 0.95)))

# 5. q-k imbalance from summary.
s = json.load(open(cli.summary))
print("== joint symmetry balance (q-k) final")
for n, r in s.items():
    if isinstance(r, dict) and "joint_symmetry_balance" in r:
        print(n, json.dumps(r["joint_symmetry_balance"])[:200])
if cli.out:
    json.dump(out, open(cli.out, "w"), indent=1, default=float)
