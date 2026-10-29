"""Decompose the Muon equilibrium-AUS gap from the logged runs.

python analysis/muon_equilibrium_logs.py [RUNS_DIR]   (default: results/runs)

Per run (const schedule, last 10% of records, block matrices only):
  e = ELR, c = cos(W,U), k = -c/e, delta = per-step d ln|W|^2 (from a log-linear fit of |W|^2
  over the tail), eps = eta*lambda. Exact per-step norm identity:
      e^2 (1 + 2(1-eps) k) = 2 eps - eps^2 + delta
  Predictions of the AUS (= ELR to 0.2%):
      HB   (Kosson, heavy ball)  sqrt(2 eps (1-b)/(1+b))
      NES  (Nesterov weights)    sqrt(2 eps / R),  R = (sum a)^2 / sum a^2
      NES+ (Nesterov + measured non-equilibrium delta)  sqrt((2 eps + delta) / R)
"""
import json, math, os, sys
import numpy as np

RUNS = sys.argv[1] if len(sys.argv) > 1 else "results/runs"


def ratio(a):
    a = np.asarray(a, float)
    return a.sum() ** 2 / (a ** 2).sum()


def weights(beta, kind, K=2000):
    k = np.arange(K)
    if kind == "hb":
        return beta ** k
    a = beta ** (k + 1.0)
    a[0] = 1 + beta  # Nesterov: g_t + beta * m_t, m_t = sum beta^k g_{t-k}
    return a


def load(d):
    recs = [json.loads(l) for l in open(os.path.join(d, "elr.jsonl"))]
    names = recs[0]["names"]
    idx = [i for i, n in enumerate(names) if n.startswith("blocks")]
    g = lambda k: np.array([r[k] for r in recs])[:, idx]
    return dict(names=[names[i] for i in idx], step=np.array([r["step"] for r in recs]),
                w=g("w_norm"), e=g("elr"), c=g("cos"), aus=g("aus_pred"),
                args=json.load(open(os.path.join(d, "args.json"))))


def analyze(d):
    R = load(d)
    a = R["args"]
    T, eps = a["steps"], a["lr"] * a["wd"]
    tail = R["step"] >= 0.9 * T
    s = R["step"][tail]
    # per-matrix delta: slope of ln|W|^2 vs step over the tail
    lw2 = np.log(R["w"][tail] ** 2)
    delta = np.polyfit(s, lw2, 1)[0]  # (n_mat,)
    e, c, aus = R["e"][tail], R["c"][tail], R["aus"][tail]
    k = -c / e
    beta = 0.95 if a["opt"] == "muon" else 0.9
    kind = "nes" if a["opt"] == "muon" else "hb"
    Rw = ratio(weights(beta, kind))
    Rhb = ratio(weights(beta, "hb"))
    lhs = e ** 2 * (1 + 2 * (1 - eps) * k)
    gain_decay = (e ** 2 + 2 * (1 - eps) * k * e ** 2) / (2 * eps - eps ** 2)
    med = lambda x: float(np.median(x))
    out = dict(run=os.path.basename(d), eps=eps, timescales=eps * T,
               aus=med(aus), k=med(k), k_medmed=med(np.median(k, 1)), k_model=(Rw - 1) / 2,
               delta_over_2eps=med(delta) / (2 * eps) if eps else float("nan"),
               gain_over_decay=med(gain_decay) if eps else float("nan"),
               identity_resid=med((lhs - (2 * eps - eps ** 2 + delta[None])) / lhs) if eps else float("nan"))
    if eps:
        out["pred_hb"] = math.sqrt(2 * eps / Rhb)
        out["pred_model"] = math.sqrt(2 * eps / Rw)
        out["pred_model_noneq"] = med(np.sqrt(np.clip(2 * eps + delta, 0, None) / Rw))
        out["pred_measured_k_noneq"] = med(np.sqrt((2 * eps - eps ** 2 + delta) / (1 + 2 * (1 - eps) * np.median(k, 0))))
    # per type k
    types = {}
    for t in ["q", "k", "v", "o", "up", "gate", "down"]:
        ii = [j for j, n in enumerate(R["names"]) if (n.endswith("." + t) or n.split(".")[2] == t)]
        types[t] = round(med(k[:, ii]), 1)
    out["k_by_type"] = types
    return out


if __name__ == "__main__":
    rows = []
    for r in sorted(os.listdir(RUNS)):
        if "const" in r:
            o = analyze(os.path.join(RUNS, r))
            rows.append(o)
            print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in o.items()}))
