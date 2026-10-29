"""Analyze the ELR study runs: python study/analyze.py RUNS_DIR OUT_DIR"""

import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TYPES = ["q", "k", "v", "o", "up", "gate", "down", "emb", "head"]


def load(run_dir):
    recs = [json.loads(l) for l in open(os.path.join(run_dir, "elr.jsonl"))]
    names = recs[0]["names"]
    arr = {k: np.array([r[k] for r in recs]) for k in ("w_norm", "u_norm", "cos", "elr", "aus_pred", "rel_change", "aus_true")}
    steps = np.array([r["step"] for r in recs])
    lr = np.array([r["lr"] for r in recs])
    loss = [json.loads(l) for l in open(os.path.join(run_dir, "loss.jsonl"))]
    args = json.load(open(os.path.join(run_dir, "args.json")))
    return dict(names=names, steps=steps, lr=lr, loss=loss, args=args, **arr)


def mtype(name):
    if name.startswith("emb"):
        return "emb"
    if name.startswith("head"):
        return "head"
    if ".qkv.weight." in name:
        return name.rsplit(".", 1)[1]
    return name.split(".")[2]


def layer(name):
    return int(name.split(".")[1]) if name.startswith("blocks") else -1


def at(R, step, key, idx, half=4):
    """Median over matrices `idx` of R[key], averaged over +-half records around `step` (log-mean)."""
    s = int(np.clip(np.searchsorted(R["steps"], step), half, len(R["steps"]) - 1 - half))
    v = R[key][s - half:s + half + 1][:, idx]
    return float(np.exp(np.median(np.log(v), axis=1).mean()))


def rec_at(R, step):
    """Index of the first record at or after `step`, clipped to the last record."""
    return min(int(np.searchsorted(R["steps"], step)), len(R["steps"]) - 1)


def block_idx(R):
    return [i for i, n in enumerate(R["names"]) if n.startswith("blocks")]


def decompose(R, base_steps=(300, 2000), frac_points=(0.25, 0.5, 0.75, 0.95)):
    """Change of log ELR from a base step, split into lr, |U| and 1/|W| terms.

    Terms are medians over block matrices (log of median-in-window). Implicit share
    is |d_u + d_w| / (|d_u + d_w| + |d_lr|) with the sign of the implicit part reported,
    which stays stable when the implicit and scheduled parts cancel.
    """
    idx = block_idx(R)
    T = R["args"]["steps"]
    out = []
    for b in base_steps:
        for f in frac_points:
            s = f * T
            d_lr = math.log(R["lr"][rec_at(R, s)] / R["lr"][rec_at(R, b)])
            d_u = math.log(at(R, s, "u_norm", idx) / at(R, b, "u_norm", idx))
            d_w = -math.log(at(R, s, "w_norm", idx) / at(R, b, "w_norm", idx))
            imp = d_u + d_w
            out.append({"base": b, "frac": f, "dlog_lr": d_lr, "dlog_u": d_u, "dlog_inv_w": d_w,
                        "implicit_sign": int(np.sign(imp)),
                        "implicit_share": abs(imp) / (abs(imp) + abs(d_lr)) if abs(imp) + abs(d_lr) > 0 else None})
    return out


def main():
    runs_dir, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    runs = {}
    for d in sorted(os.listdir(runs_dir)):
        p = os.path.join(runs_dir, d)
        if d.startswith(("adamw", "muon")) and os.path.exists(os.path.join(p, "elr.jsonl")):
            runs[d] = load(p)
    summary = {}

    # Fig 1: median block-matrix ELR (and 10-90% band) vs step, with nominal lr.
    ncol = 4
    nrow = math.ceil(len(runs) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(18, 3.5 * nrow), sharex=True, squeeze=False)
    for ax, (name, R) in zip(axes.flat, runs.items()):
        idx = [i for i, n in enumerate(R["names"]) if n.startswith("blocks")]
        e = R["elr"][:, idx]
        ax.fill_between(R["steps"], np.percentile(e, 10, 1), np.percentile(e, 90, 1), alpha=0.25, label="ELR 10-90%")
        ax.plot(R["steps"], np.median(e, 1), label="ELR median")
        ax2 = ax.twinx()
        ax2.plot(R["steps"], R["lr"], "k--", lw=0.8, label="nominal lr")
        ax.set_yscale("log")
        ax.set_title(name, fontsize=9)
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_elr_vs_lr.png"), dpi=110)

    # Fig 2: ELR per matrix type (median over layers), for the two linear baselines.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, key in zip(axes, ["adamw-lr6e-4-wd0.1-lin", "muon-lr6e-4-wd0.1-lin"]):
        if key not in runs:
            continue
        R = runs[key]
        for t in TYPES:
            idx = [i for i, n in enumerate(R["names"]) if mtype(n) == t]
            if idx:
                ax.plot(R["steps"], np.median(R["elr"][:, idx], 1), label=t)
        ax.set_yscale("log"); ax.set_title(key, fontsize=9); ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_elr_by_type.png"), dpi=110)

    # Fig 3: k ELR vs whole fused-qkv ELR.
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, R in runs.items():
        ratios = []
        for L in sorted({layer(n) for n in R["names"] if n.startswith("blocks")}):
            ii = {t: R["names"].index(f"blocks.{L}.qkv.weight.{t}") for t in "qkv"}
            w2 = sum(R["w_norm"][:, i] ** 2 for i in ii.values())
            u2 = sum(R["u_norm"][:, i] ** 2 for i in ii.values())
            whole = R["lr"] * np.sqrt(u2) / np.sqrt(w2)
            ratios.append(R["elr"][:, ii["k"]] / np.where(whole > 0, whole, np.nan))
        r = np.nanmedian(np.array(ratios), 0)
        ax.plot(R["steps"], r, label=name)
        summary.setdefault(name, {})["k_over_whole_qkv_elr_final_by_layer"] = [float(x[-2]) for x in ratios]
    ax.axhline(1, color="k", lw=0.5); ax.set_title("ELR(k) / ELR(whole qkv), median over layers"); ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_k_vs_whole_qkv.png"), dpi=110)

    # Fig 4: val loss.
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, R in runs.items():
        ax.plot([r["step"] for r in R["loss"]], [r["val_loss"] for r in R["loss"]], label=name)
    ax.set_ylim(3.0, 6.0); ax.legend(fontsize=6); ax.set_title("val loss")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_val_loss.png"), dpi=110)

    for name, R in runs.items():
        s = summary.setdefault(name, {})
        s["final_val_loss"] = R["loss"][-1]["val_loss"]
        s["tok_per_s"] = R["loss"][-1].get("tok_per_s")
        s["decomposition"] = decompose(R)
        idx = block_idx(R)
        T, a = R["args"]["steps"], R["args"]
        post = R["steps"] >= 300
        # Q1: ratio of median ELR at 95% vs base (two bases: end of warmup, step 2000).
        for b in (300, 2000):
            s[f"q1_elr95_over_elr{b}"] = at(R, 0.95 * T, "elr", idx) / at(R, b, "elr", idx)
        # Q5: |cos(W,U)| and ELR vs AUS (aus_pred is exact algebra; aus_true in batch 1 has fp32 cancellation).
        s["q5_cos_wu_median_abs"] = float(np.median(np.abs(R["cos"][post][:, idx])))
        s["q5_elr_over_aus_pred_rel_gap_median"] = float(np.median(np.abs(R["elr"][post][:, idx] / R["aus_pred"][post][:, idx] - 1)))
        # Sanity: logged U is the applied update. Predicted |dW|/|W| from (w,u,cos) incl. decoupled WD.
        wn, un, cs, lr_ = R["w_norm"], R["u_norm"], R["cos"], R["lr"][:, None]
        lam = a["wd"]
        pred_rc = lr_ * np.sqrt(lam**2 * wn**2 + 2 * lam * cs * wn * un + un**2) / wn
        s["sanity_rel_change_pred_rel_err_median"] = float(np.median(np.abs(R["rel_change"][post][:, idx] / pred_rc[post][:, idx] - 1)))
        # Q4: spread across matrices, overall and within type (o/down have smaller init, so overall is inflated).
        e_end = R["elr"][-2, idx]
        s["q4_elr_p90_over_p10_final_all"] = float(np.percentile(e_end, 90) / np.percentile(e_end, 10))
        within = {}
        for t in TYPES[:7]:
            ii = [i for i in idx if mtype(R["names"][i]) == t]
            if len(ii) >= 3:
                v = R["elr"][-2, ii]
                within[t] = float(np.percentile(v, 90) / np.percentile(v, 10))
        s["q4_elr_p90_over_p10_final_within_type"] = within
        # Q3a: equilibrium angular update for const + WD runs; report both formulas and measured AUS
        # (median over last 10% of records).
        if a["wd"] > 0 and a["sched"] == "const":
            tail = R["steps"] >= 0.9 * T
            s["q3a_aus_pred_median_last10pct"] = float(np.median(R["aus_pred"][tail][:, idx]))
            s["q3a_pred_sqrt_2_eta_lambda"] = math.sqrt(2 * a["lr"] * a["wd"])
            if a["opt"] == "adamw":
                s["q3a_pred_kosson_adamw"] = math.sqrt(2 * a["lr"] * a["wd"] * (1 - 0.9) / (1 + 0.9))
        # Joint-symmetry balance (Su 11985): |Wq|^2 - |Wk|^2 etc., median over layers, start vs end.
        bal = {}
        for p1, p2 in (("q", "k"), ("v", "o"), ("up", "down")):
            d0, d1 = [], []
            for L in sorted({layer(n) for n in R["names"] if n.startswith("blocks")}):
                f = lambda t: R["names"].index(f"blocks.{L}.qkv.weight.{t}") if t in "qkv" else R["names"].index(f"blocks.{L}.{t}.weight")
                i1, i2 = f(p1), f(p2)
                d0.append(R["w_norm"][0, i1] ** 2 - R["w_norm"][0, i2] ** 2)
                d1.append(R["w_norm"][-1, i1] ** 2 - R["w_norm"][-1, i2] ** 2)
            bal[f"{p1}-{p2}"] = {"start": float(np.median(d0)), "end": float(np.median(d1))}
        s["joint_symmetry_balance"] = bal

    # Q3b: lr compression, 2.4e-3 vs 6e-4 at matched schedule; window 70-90% of training.
    for opt in ("adamw", "muon"):
        for sch, wd in (("lin", "0.1"), ("const", "0.1")):
            hi, lo = runs.get(f"{opt}-lr2.4e-3-wd{wd}-{sch}"), runs.get(f"{opt}-lr6e-4-wd{wd}-{sch}")
            if hi and lo:
                T = hi["args"]["steps"]
                r = [at(hi, f * T, "elr", block_idx(hi)) / at(lo, f * T, "elr", block_idx(lo)) for f in (0.7, 0.8, 0.9)]
                summary[f"q3b_{opt}_{sch}_elr_ratio_4x_lr_70_80_90pct"] = r
    # Seed noise: seed 1 vs seed 0 for the key numbers.
    noise = {}
    for name in runs:
        if name.endswith("-s1") and name[:-3] in summary:
            a0, a1 = summary[name[:-3]], summary[name]
            row = {"val_loss": (a0["final_val_loss"], a1["final_val_loss"]),
                   "q1_elr95_over_elr2000": (a0["q1_elr95_over_elr2000"], a1["q1_elr95_over_elr2000"])}
            d0 = {(r["base"], r["frac"]): r["implicit_share"] for r in a0["decomposition"]}
            d1 = {(r["base"], r["frac"]): r["implicit_share"] for r in a1["decomposition"]}
            row["q2_share_base2000_50pct"] = (d0[(2000, 0.5)], d1[(2000, 0.5)])
            if "q3a_aus_pred_median_last10pct" in a0:
                row["q3a_aus"] = (a0["q3a_aus_pred_median_last10pct"], a1["q3a_aus_pred_median_last10pct"])
            noise[name[:-3]] = row
    summary["seed_noise"] = noise

    # Blog figure: nominal lr and ELR on the same log axis, each normalized to its value at step 2000.
    keys = [k for k in ("adamw-lr6e-4-wd0.1-const", "adamw-lr6e-4-wd0-const", "muon-lr6e-4-wd0.1-const",
                        "muon-lr6e-4-wd0-const", "adamw-lr6e-4-wd0.1-lin", "muon-lr6e-4-wd0.1-lin") if k in runs]
    if keys:
        fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.4), sharey=True, squeeze=False)
        for ax, k in zip(axes[0], keys):
            R = runs[k]
            idx = block_idx(R)
            m = np.exp(np.median(np.log(R["elr"][:, idx]), 1))
            i0 = rec_at(R, 2000)
            ax.plot(R["steps"], R["lr"] / R["lr"][i0], "k--", lw=1, label="nominal lr")
            ax.plot(R["steps"], m / m[i0], lw=1.5, label="ELR (median matrix)")
            ax.set_yscale("log"); ax.set_ylim(1e-2, 5); ax.set_title(k, fontsize=8); ax.set_xlabel("step")
        axes[0][0].set_ylabel("relative to step 2000"); axes[0][0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig5_nominal_vs_elr_normalized.png"), dpi=130)

    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
