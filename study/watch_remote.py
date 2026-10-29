"""Compact health report for the ELR study, run ON the GPU host. No side effects.

Checks (see design/study.md, "Health criteria"): per run step, age of last log
write, last val/train loss, tok/s, ETA; queue markers; GPU utilization; and
running ts jobs whose output has not changed while GPUs are busy.
"""

import json
import math
import os
import subprocess
import time

EXP = os.environ.get("EXP", "/data/exp_elrshard")
RUNS = os.path.join(EXP, "runs")
STALE_MIN = 20


def last_json(path):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 65536))
            lines = [l for l in f.read().decode(errors="ignore").splitlines() if l.startswith("{")]
        return json.loads(lines[-1]) if lines else None
    except (OSError, ValueError):
        return None


def main():
    now = time.time()
    out, alerts = [], []
    markers = set(os.listdir(os.path.join(EXP, "markers"))) if os.path.isdir(os.path.join(EXP, "markers")) else set()
    for d in sorted(os.listdir(RUNS)) if os.path.isdir(RUNS) else []:
        if d.startswith("smoke"):
            continue
        rd = os.path.join(RUNS, d)
        args = json.load(open(os.path.join(rd, "args.json"))) if os.path.exists(os.path.join(rd, "args.json")) else {}
        elr = os.path.join(rd, "elr.jsonl")
        age = (now - os.path.getmtime(elr)) / 60 if os.path.exists(elr) else float("inf")
        rec = last_json(elr) or {}
        lo = last_json(os.path.join(rd, "loss.jsonl")) or {}
        done = os.path.exists(os.path.join(rd, "DONE"))
        failed = f"study-{d}.FAILED" in markers
        step, steps = rec.get("step", -1), args.get("steps", 0)
        tps = lo.get("tok_per_s") or 0
        eta_h = (steps - step) * 32768 / tps / 3600 if tps and not done else 0
        tl = rec.get("loss")
        status = "DONE" if done else "FAILED" if failed else "ok"
        if not done and not failed:
            if age > STALE_MIN:
                status = f"STALE {age:.0f}min"
            if tl is not None and (math.isnan(tl) or math.isinf(tl) or (step > 2000 and tl > 7.0)):
                status = "DIVERGED"
        if status not in ("ok", "DONE"):
            alerts.append(f"{d}: {status}")
        out.append(f"{d:28s} step {step:>6}/{steps:<6} val {lo.get('val_loss', float('nan')):.3f} "
                   f"train {tl if tl is not None else float('nan'):.3f} {tps / 1e3:5.0f}k tok/s ETA {eta_h:4.1f}h {status}")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    util = [int(l.split(",")[1]) for l in gpu.splitlines() if l]
    out.append("GPU util %: " + " ".join(map(str, util)))
    env = dict(os.environ, TS_SOCKET=os.environ.get("TS_SOCKET", "/root/.ts.sock"))
    ts = subprocess.run(["ts"], capture_output=True, text=True, env=env).stdout.splitlines()
    running = [l for l in ts if " running " in l]
    out.append(f"ts running: {len(running)}; queued: {sum(' queued ' in l or ' allocating ' in l for l in ts)}")
    for l in running:
        parts = l.split()
        log = parts[2] if len(parts) > 2 else ""
        if log.startswith("/") and os.path.exists(log):
            a = (now - os.path.getmtime(log)) / 60
            if a > STALE_MIN and max(util or [0]) > 50 and "study-" not in l:
                alerts.append(f"ts job {parts[0]} output unchanged {a:.0f}min while GPUs busy: {l[:90]}")
    out.append("ALERTS: " + ("; ".join(alerts) if alerts else "none"))
    print("\n".join(out))


if __name__ == "__main__":
    main()
