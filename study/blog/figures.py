"""Figures for the blog post on the ELR study (desktop and mobile variants of figs 1-5).

Usage (from the repo root):
    python study/blog/figures.py [RUNS_DIR] [OUT_DIR] [--summary results/final/summary.json]

Defaults: RUNS_DIR = results/runs, OUT_DIR = results/blog_figures (env ELR_RUNS / ELR_FIG_OUT /
ELR_SUMMARY override them). RUNS_DIR holds the raw per-run logs, which are not in git:
download the archive from the v0.1-study GitHub release and extract it at the repo root
(see README). Reuses the definitions in study/analyze.py; writes SVG + 200-dpi PNG per figure
and prints the numbers quoted in the captions.
"""

import argparse
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches  # noqa: F401
import matplotlib.lines  # noqa: F401
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(REPO, "study"))
from analyze import at, block_idx, decompose, load, rec_at  # noqa: E402

_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("runs", nargs="?", default=os.environ.get("ELR_RUNS", os.path.join(REPO, "results", "runs")))
_ap.add_argument("out", nargs="?", default=os.environ.get("ELR_FIG_OUT", os.path.join(REPO, "results", "blog_figures")))
_ap.add_argument("--summary", default=os.environ.get("ELR_SUMMARY", os.path.join(REPO, "results", "final", "summary.json")))
_cli = _ap.parse_args()
RUNS, OUT = _cli.runs, _cli.out
os.makedirs(OUT, exist_ok=True)

# ---- shared style (dataviz reference palette, light theme on white) ----
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, VIOLET, GREEN, RED = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#008300", "#e34948"
BLUE_LIGHT, ORANGE_LIGHT = "#86b6ef", "#f2a07f"
OPT_COLOR = {"adamw": BLUE, "muon": ORANGE}
W = 7.2  # inches; SVG renders at ~690 px wide, 10 pt text ~ 13 px

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "lines.linewidth": 1.8, "lines.solid_capstyle": "round", "lines.dash_capstyle": "round",
    "svg.fonttype": "none", "text.color": INK,
})
NOMINAL = dict(color=MUTED, ls=(0, (5, 3)), lw=1.4)
WD0_LS = (0, (1, 2.2))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".svg"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200)
    plt.close(fig)


def kstep(ax):
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1000:g}k" if x else "0"))


def label(ax, x, y, text, color=INK2, **kw):
    kw.setdefault("va", "center")
    ax.annotate(text, (x, y), color=color, fontsize=kw.pop("fs", 9.5), annotation_clip=False, **kw)


def log_median(R, key):
    """Per-record median over block matrices, in log space."""
    return np.median(np.log(R[key][:, block_idx(R)]), axis=1)


def win_mean(x, half=4):
    """Rolling mean over +-half records: matches analyze.at() at interior points."""
    k = np.ones(2 * half + 1) / (2 * half + 1)
    y = np.convolve(x, k, mode="same")
    y[:half], y[-half:] = np.nan, np.nan
    return y


runs = {d: load(os.path.join(RUNS, d)) for d in sorted(os.listdir(RUNS))
        if d.startswith(("adamw", "muon")) and os.path.exists(os.path.join(RUNS, d, "elr.jsonl"))}
summary = json.load(open(_cli.summary))
report = {}

BASE = 2000
WM = 3.6  # mobile width in inches: ~346 px at 96 dpi, 10 pt text ~ 13 px
fig2, fig3 = {}, {}
groups = {0.1: [], 0.0: []}
UNSTABLE = "adamw-lr2.4e-3-wd0.1-lin"


def suffix(mobile):
    return "-mobile" if mobile else ""


def ratio_axis(ax, ticks):
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())


# ---------------- Fig 1: nominal lr vs ELR, normalized at step 2000 ----------------
def make_fig1(mobile):
    if mobile:
        fig, axes = plt.subplots(2, 1, figsize=(WM, 5.9), sharex=False)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(W, 3.7), sharey=True)
    panels = [("Constant nominal lr", ["adamw-lr6e-4-wd0.1-const", "adamw-lr6e-4-wd0-const",
                                       "muon-lr6e-4-wd0.1-const", "muon-lr6e-4-wd0-const",
                                       "adamw-lr2.4e-3-wd0.1-const", "muon-lr2.4e-3-wd0.1-const"]),
              ("Linear decay", ["adamw-lr6e-4-wd0.1-lin", "muon-lr6e-4-wd0.1-lin"])]
    for ax, (ptitle, keys) in zip(axes, panels):
        R0 = runs[keys[0]]
        ax.axvline(BASE, color=AXIS, lw=0.8, zorder=1)
        lr = R0["lr"] / R0["lr"][rec_at(R0, BASE)]
        ax.plot(R0["steps"], lr, **NOMINAL, zorder=2)
        for k in keys:
            R = runs[k]
            a = R["args"]
            base = at(R, BASE, "elr", block_idx(R))
            y = np.exp(win_mean(log_median(R, "elr"))) / base
            hi = a["lr"] > 1e-3  # 4x-rate runs: lighter tint, behind the 6e-4 runs
            col = {"adamw": BLUE_LIGHT, "muon": ORANGE_LIGHT}[a["opt"]] if hi else OPT_COLOR[a["opt"]]
            # AdamW drawn wider underneath so Muon (on top) does not hide it where they coincide
            lw = 3.2 if a["opt"] == "adamw" else 1.5
            ax.plot(R["steps"], y, color=col, lw=lw, ls="-" if a["wd"] > 0 else (0, (1.6, 1.6)),
                    zorder=(1.5 if hi else 3) + (a["opt"] == "muon"), solid_capstyle="butt")
            report[k] = summary[k]["q1_elr95_over_elr2000"]
        ax.set_yscale("log")
        ax.set_xlim(0, 25000)
        ax.set_ylim(0.03, 3)
        kstep(ax)
        ax.set_title(ptitle, loc="left", fontsize=10, color=INK, weight="bold", pad=6)
        ratio_axis(ax, [0.03, 0.1, 0.3, 1, 3])
        ax.text(BASE + 300, 0.036, "step 2000", color=MUTED, fontsize=9)
    a0, a1 = axes
    if mobile:
        a1.set_xlabel("Training step")
        for ax in axes:
            ax.set_ylabel("Relative to step 2000")
        label(a0, 24500, 1.06, "nominal lr", va="bottom", ha="right")
        a0.annotate("4× lr, wd 0.1", (9000, 0.79), xytext=(6000, 1.9), color=INK2, fontsize=9.5, va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkA=1, shrinkB=0))
        label(a0, 4500, 0.40, "ELR, wd 0.1", va="top")
        label(a0, 13000, 0.27, "ELR, wd 0 (dashed)", va="top")
        label(a1, 12500, 0.86, "nominal lr", va="bottom", ha="left")
        label(a1, 3000, 0.25, "ELR, wd 0.1", va="top")
        fig.legend(handles=[matplotlib.lines.Line2D([], [], color=BLUE, lw=3.2, label="AdamW"),
                            matplotlib.lines.Line2D([], [], color=ORANGE, lw=1.5, label="Muon")],
                   loc="upper right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize=9.5,
                   handlelength=1.4, columnspacing=1.0, labelcolor=INK2)
        fig.subplots_adjust(left=0.19, right=0.96, hspace=0.32, bottom=0.08, top=0.92)
    else:
        for ax in axes:
            ax.set_xlabel("Training step")
        a0.set_ylabel("Relative to value at step 2000 (log)")
        label(a0, 24500, 1.06, "nominal lr", va="bottom", ha="right")
        a0.annotate("4× lr, wd 0.1", (10000, 0.79), xytext=(7000, 1.9), color=INK2, fontsize=9.5, va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkA=1, shrinkB=0))
        label(a0, 5500, 0.42, "ELR, wd 0.1", va="top")
        label(a0, 14000, 0.30, "ELR, wd 0 (dashed)", va="top")
        label(a1, 12500, 0.86, "nominal lr", va="bottom", ha="left")
        label(a1, 3000, 0.25, "ELR, wd 0.1", va="top")
        fig.legend(handles=[matplotlib.lines.Line2D([], [], color=BLUE, lw=3.2, label="AdamW"),
                            matplotlib.lines.Line2D([], [], color=ORANGE, lw=1.5, label="Muon")],
                   loc="upper right", bbox_to_anchor=(0.985, 1.0), ncol=2, frameon=False, fontsize=9.5,
                   handlelength=1.6, columnspacing=1.2, labelcolor=INK2)
        fig.subplots_adjust(left=0.10, right=0.98, wspace=0.12, bottom=0.14, top=0.90)
    save(fig, "fig1-nominal-vs-elr" + suffix(mobile))


# ---------------- Fig 2: decomposition of Δlog ELR (base 2000) ----------------
def make_fig2(mobile):
    if mobile:
        fig, axes = plt.subplots(2, 1, figsize=(WM, 5.6))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(W, 3.5), sharey=True)
    for ax, k, ptitle in zip(axes, ["adamw-lr6e-4-wd0-const", "adamw-lr6e-4-wd0.1-lin"],
                             ["AdamW, wd 0, constant lr", "AdamW, wd 0.1, linear decay"]):
        R = runs[k]
        idx = block_idx(R)
        T = R["args"]["steps"]
        i0 = rec_at(R, BASE)
        d_lr = np.log(R["lr"] / R["lr"][i0])
        d_u = win_mean(log_median(R, "u_norm")) - math.log(at(R, BASE, "u_norm", idx))
        d_w = -(win_mean(log_median(R, "w_norm")) - math.log(at(R, BASE, "w_norm", idx)))
        tot = d_lr + d_u + d_w
        m = (R["steps"] >= BASE) & (R["steps"] <= 0.95 * T)  # lr -> 0 at the very end; stop at 95 % like decompose()
        s = R["steps"][m]
        ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
        ax.plot(s, tot[m], color=INK, lw=3.4, zorder=3, solid_capstyle="butt")
        ax.plot(s, d_lr[m], **NOMINAL, zorder=4)
        ax.plot(s, d_u[m], color=GREEN, lw=1.6, zorder=4)
        ax.plot(s, d_w[m], color=RED, lw=1.6, zorder=5)
        ax.set_xlim(0, 25000)
        ax.set_ylim(-3.6, 0.5)
        kstep(ax)
        ax.set_title(ptitle, loc="left", fontsize=10, color=INK, weight="bold", pad=6)
        fig2[k] = [r for r in decompose(R) if r["base"] == 2000 and r["frac"] == 0.95][0]
    a0, a1 = axes
    if mobile:
        a1.set_xlabel("Training step")
        for ax in axes:
            ax.set_ylabel("Change in ln since step 2000")
        label(a0, 2500, 0.12, "Δ ln η = 0, Δ ln ‖U‖ ≈ 0", va="bottom")
        label(a0, 5000, -1.5, "Δ ln ELR (black)\n= −Δ ln ‖W‖ (red)", va="top")
        label(a1, 2500, 0.12, "Δ ln ‖U‖", va="bottom")
        label(a1, 14000, -0.30, "−Δ ln ‖W‖", va="bottom")
        label(a1, 23900, -1.0, "Δ ln η", ha="right")
        label(a1, 6000, -1.5, "Δ ln ELR", va="top", color=INK, weight="bold")
        fig.subplots_adjust(left=0.19, right=0.96, hspace=0.32, bottom=0.08, top=0.95)
    else:
        for ax in axes:
            ax.set_xlabel("Training step")
        a0.set_ylabel("Change in ln since step 2000")
        label(a0, 4000, 0.12, "Δ ln η = 0 and Δ ln ‖U‖ ≈ 0", va="bottom")
        label(a0, 9000, -1.15, "Δ ln ELR (black)\n= −Δ ln ‖W‖ (red)", va="top")
        label(a1, 3000, 0.12, "Δ ln ‖U‖", va="bottom")
        label(a1, 14500, -0.30, "−Δ ln ‖W‖", va="bottom")
        label(a1, 21800, -1.25, "Δ ln η", ha="left")
        label(a1, 9000, -1.6, "Δ ln ELR", va="top", color=INK, weight="bold")
        fig.subplots_adjust(left=0.09, right=0.98, wspace=0.10, bottom=0.14, top=0.91)
    save(fig, "fig2-decomposition" + suffix(mobile))


# ---------------- Fig 3: AUS equilibrium ----------------
def make_fig3(mobile):
    fig, ax = plt.subplots(figsize=(WM, 3.9) if mobile else (W, 3.6))
    lam, b1 = 0.1, 0.9
    for k, c in [("adamw-lr6e-4-wd0.1-const", BLUE_LIGHT), ("adamw-lr2.4e-3-wd0.1-const", "#184f95")]:
        R = runs[k]
        eta = R["args"]["lr"]
        y = np.exp(win_mean(log_median(R, "aus_pred")))
        ax.plot(R["steps"], y, color=c, lw=2.2, zorder=3)
        kos = math.sqrt(2 * eta * lam * (1 - b1) / (1 + b1))
        plain = math.sqrt(2 * eta * lam)
        ax.axhline(kos, color=c, ls=(0, (5, 3)), lw=1.2, zorder=2)
        ax.axhline(plain, color=c, ls=(0, (1, 2.2)), lw=1.2, alpha=0.7, zorder=2)
        fig3[k] = (eta, summary[k]["q3a_aus_pred_median_last10pct"], kos, plain)
        tag = f"η = {eta:g}"
        if mobile:  # labels inside the plot, above each reference line, right-aligned
            label(ax, 24500, plain * 1.06, f"√(2ηλ), {tag}", va="bottom", ha="right")
            if eta > 1e-3:
                label(ax, 24500, kos * 1.25, f"measured ≈ Kosson, {tag}", va="bottom", ha="right")
            else:
                label(ax, 24500, kos * 0.94, f"Kosson, {tag}", va="top", ha="right")
        else:
            label(ax, 25500, plain, f"√(2ηλ), {tag}")
            label(ax, 25500, kos, f"Kosson, {tag}")
    if mobile:
        label(ax, 7500, 0.0034, "measured, η = 0.0006", va="bottom")
        ax.set_ylim(1.4e-3, 0.03)
        ax.set_xticks([0, 5000, 10000, 15000, 20000, 25000])
    else:
        label(ax, 11000, 0.0062, "measured, η = 0.0024", va="bottom")
        label(ax, 11000, 0.0034, "measured, η = 0.0006", va="bottom")
        ax.set_ylim(1.2e-3, 0.04)
    ax.set_yscale("log")
    ax.set_xlim(0, 25000)
    ax.set_yticks([0.002, 0.005, 0.01, 0.02])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Training step")
    ax.set_ylabel("Angular update per step (rad, log)" if not mobile else "Angular update / step (rad)")
    kstep(ax)
    if mobile:
        fig.subplots_adjust(left=0.21, right=0.96, bottom=0.13, top=0.97)
    else:
        fig.subplots_adjust(left=0.10, right=0.75, bottom=0.14, top=0.97)
    save(fig, "fig3-equilibrium" + suffix(mobile))


# ---------------- Fig 4: ELR(k) / ELR(whole qkv) per layer ----------------
def make_fig4(mobile):
    fig, ax = plt.subplots(figsize=(WM, 4.6) if mobile else (W, 3.6))
    ax.axhspan(0.8, 1.25, color="#f0efec", zorder=0, lw=0)
    ax.axhline(1, color=AXIS, lw=0.8, zorder=1)
    for g in groups.values():
        g.clear()
    for k, R in runs.items():
        r = np.array(summary[k]["k_over_whole_qkv_elr_final_by_layer"])
        wd = R["args"]["wd"]
        c = VIOLET if wd > 0 else AQUA
        off = 0.12 if wd > 0 else -0.12
        x = np.arange(12) + off
        if k == UNSTABLE:
            ax.plot(x, r, color=c, lw=0.8, alpha=0.7, zorder=2)
            ax.scatter(x, r, s=30 if mobile else 34, facecolors="white", edgecolors=c, linewidths=1.3, zorder=4)
        else:
            groups[wd].append(r)
            ax.plot(x, r, color=c, lw=0.6, alpha=0.25, zorder=2)
            ax.scatter(x, r, s=12 if mobile else 16, color=c, edgecolors="white", linewidths=0.5, zorder=3, alpha=0.9)
    ax.set_yscale("log")
    ax.set_ylim(0.26, 2.0)
    ax.set_yticks([0.3, 0.4, 0.5, 0.67, 0.8, 1, 1.25, 1.5, 2])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks(range(12) if not mobile else range(0, 12, 2))
    ax.set_xlim(-0.5, 11.5)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Layer")
    ax.set_ylabel("ELR(k) / ELR(whole qkv), final" if not mobile else "ELR(k) / ELR(whole qkv)")
    n1, n0 = len(groups[0.1]), len(groups[0.0])
    if mobile:  # key above the plot, in figure coordinates
        def key(y, text, hollow=False, col=VIOLET):
            fig.add_artist(matplotlib.lines.Line2D([0.215], [y], marker="o", ms=5.5 if hollow else 4.2,
                                                   mfc="white" if hollow else col, mec=col,
                                                   mew=1.3 if hollow else 0, transform=fig.transFigure))
            fig.text(0.24, y, text, color=INK2, fontsize=9.5, va="center")
        key(0.965, f"wd 0.1 ({n1} runs)")
        key(0.965 - 0.045, f"wd 0 ({n0} runs)", col=AQUA)
        key(0.965 - 0.09, "AdamW lr 2.4e-3, wd 0.1, linear (unstable)", hollow=True)
        fig.add_artist(matplotlib.patches.Rectangle((0.197, 0.965 - 0.135 - 0.009), 0.036, 0.018, color="#e6e5df",
                                                    transform=fig.transFigure, lw=0))
        fig.text(0.24, 0.965 - 0.135, "shaded band: 0.8–1.25", color=INK2, fontsize=9.5, va="center")
        fig.subplots_adjust(left=0.19, right=0.96, bottom=0.11, top=0.79)
    else:
        ax.scatter([12.1], [1.75], s=16, color=VIOLET, clip_on=False)
        label(ax, 12.35, 1.75, f"wd 0.1 ({n1} runs)")
        ax.scatter([12.1], [1.53], s=16, color=AQUA, clip_on=False)
        label(ax, 12.35, 1.53, f"wd 0 ({n0} runs)")
        ax.scatter([12.1], [1.34], s=30, facecolors="white", edgecolors=VIOLET, linewidths=1.4, clip_on=False)
        label(ax, 12.35, 1.34, "AdamW lr 2.4e-3,\nwd 0.1, linear decay\n(unstable run)", va="top")
        label(ax, 12.0, 0.86, "band 0.8–1.25", color=MUTED, fs=9)
        fig.subplots_adjust(left=0.10, right=0.78, bottom=0.14, top=0.97)
    save(fig, "fig4-k-vs-qkv" + suffix(mobile))


# ---------------- Fig 5: per-step squared-norm budget, normalized by ||W||^2 ----------------
fig5 = {}


def budget(R):
    """Per-record medians over block matrices of D, C, S (dimensionless, per step)."""
    idx = block_idx(R)
    eta = R["lr"][:, None]
    lam = R["args"]["wd"]
    r = eta * R["u_norm"][:, idx] / R["w_norm"][:, idx]  # = ELR
    D = np.broadcast_to(1 - (1 - eta * lam) ** 2, r.shape)
    S = r ** 2
    C = -2 * (1 - eta * lam) * R["cos"][:, idx] * r
    return {k: np.median(v, axis=1) for k, v in (("D", D), ("S", S), ("C", C), ("SC", S + C))}


def smooth_log(x, half=4):
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.exp(win_mean(np.log(np.where(x > 0, x, np.nan)), half))
    return y


def make_fig5(mobile):
    if mobile:
        fig, axes = plt.subplots(2, 1, figsize=(WM, 6.0))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(W, 3.6))
    a0, a1 = axes
    k = "adamw-lr6e-4-wd0.1-const"
    R = runs[k]
    B = budget(R)
    st = R["steps"]
    m = st >= 300
    a0.plot(st[m], smooth_log(B["D"])[m], **NOMINAL, zorder=2)
    a0.plot(st[m], smooth_log(B["C"])[m], color=BLUE, lw=2.2, zorder=4)
    a0.plot(st[m], smooth_log(B["S"])[m], color=BLUE_LIGHT, lw=2.0, zorder=3)
    a0.plot(st[m], smooth_log(B["SC"])[m], color=INK, lw=1.0, zorder=5)
    a0.set_yscale("log")
    a0.set_xlim(0, 25000)
    a0.set_ylim(3e-6, 1e-3)
    a0.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    a0.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())
    kstep(a0)
    a0.set_title("AdamW, lr 6e-4, wd 0.1, constant", loc="left", fontsize=10, color=INK, weight="bold", pad=6)
    a0.set_ylabel("Per-step change in ‖W‖², ÷ ‖W‖²")
    tail = st >= 0.9 * R["args"]["steps"]
    fig5["C/D"] = float(np.median(B["C"][tail] / B["D"][tail]))
    fig5["S/D"] = float(np.median(B["S"][tail] / B["D"][tail]))
    fig5["(S+C)/D"] = float(np.median(B["SC"][tail] / B["D"][tail]))
    dval = float(B["D"][tail][0])
    sval = float(np.median(B["S"][tail]))
    Cs = smooth_log(B["C"])
    i6 = rec_at(R, 6000)
    label(a0, 24500, dval * 0.86, "weight-decay loss", va="top", ha="right")
    label(a0, 11000, float(smooth_log(B["SC"])[rec_at(R, 11000)]) * 1.2, "total gain (thin black)",
          va="bottom", ha="left")
    a0.annotate("cross term (from cos < 0)", (6000, float(Cs[i6]) * 0.97), xytext=(2500, 4.2e-5),
                color=INK2, fontsize=9.5, va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkA=1, shrinkB=0))
    label(a0, 24500, sval * 1.3, "‖ηU‖² term", va="bottom", ha="right")

    a1.axhline(0, color=AXIS, lw=0.9, zorder=1)
    fig5["frac_cos_neg_after_2000"] = {}
    for kk in ["adamw-lr6e-4-wd0.1-const", "muon-lr6e-4-wd0.1-const",
               "adamw-lr2.4e-3-wd0.1-const", "muon-lr2.4e-3-wd0.1-const"]:
        Rk = runs[kk]
        a = Rk["args"]
        c = np.median(Rk["cos"][:, block_idx(Rk)], axis=1)
        post = Rk["steps"] >= 2000
        fig5["frac_cos_neg_after_2000"][kk] = float(np.mean(c[post] < 0))
        hi = a["lr"] > 1e-3
        col = {"adamw": BLUE_LIGHT, "muon": ORANGE_LIGHT}[a["opt"]] if hi else OPT_COLOR[a["opt"]]
        mm = Rk["steps"] >= 300
        a1.plot(Rk["steps"][mm], win_mean(c)[mm], color=col, lw=2.0 if a["opt"] == "adamw" else 1.6,
                zorder=3 + (not hi))
    a1.set_xlim(0, 25000)
    a1.set_ylim(-0.08, 0.02)
    kstep(a1)
    a1.set_title("Median cos(W, U), wd 0.1, constant lr", loc="left", fontsize=10, color=INK, weight="bold", pad=6)
    a1.set_ylabel("cos(W, U)")
    if mobile:
        a1.set_xlabel("Training step")
        a0.set_xlabel("Training step")
    else:
        for ax in axes:
            ax.set_xlabel("Training step")
    a1.legend(handles=[matplotlib.lines.Line2D([], [], color=BLUE, lw=2, label="AdamW"),
                       matplotlib.lines.Line2D([], [], color=ORANGE, lw=1.6, label="Muon"),
                       matplotlib.lines.Line2D([], [], color=BLUE_LIGHT, lw=2, label="AdamW 4× lr"),
                       matplotlib.lines.Line2D([], [], color=ORANGE_LIGHT, lw=1.6, label="Muon 4× lr")],
              loc="upper right", ncol=2, frameon=False, fontsize=9 if mobile else 9.5, handlelength=1.3,
              columnspacing=0.9, labelcolor=INK2, borderaxespad=0.2)
    if mobile:
        fig.subplots_adjust(left=0.21, right=0.96, hspace=0.45, bottom=0.08, top=0.95)
    else:
        fig.subplots_adjust(left=0.10, right=0.98, wspace=0.30, bottom=0.14, top=0.92)
    save(fig, "fig5-norm-budget" + suffix(mobile))


for mobile in (False, True):
    make_fig1(mobile)
    make_fig5(mobile)
    make_fig2(mobile)
    make_fig3(mobile)
    make_fig4(mobile)

# ---------------- numbers for captions ----------------
print("fig1 ELR(95%)/ELR(2000):")
for k, v in report.items():
    print(f"  {k}: {v:.3f}")
print("fig2 decompose base 2000 @95%:")
for k, d in fig2.items():
    print(f"  {k}: dlr {d['dlog_lr']:.3f} du {d['dlog_u']:.3f} dinvw {d['dlog_inv_w']:.3f} "
          f"sum {d['dlog_lr'] + d['dlog_u'] + d['dlog_inv_w']:.3f} share {d['implicit_share']:.3f}")
print("fig3 eta, measured AUS last10%, kosson, sqrt(2 eta lam):")
for k, v in fig3.items():
    print(f"  {k}: " + ", ".join(f"{x:.5g}" for x in v) + f"  meas/kos {v[1] / v[2]:.3f}")
print("fig4 max |k/whole - 1|:")
for wd, rs in groups.items():
    a = np.abs(np.array(rs) - 1)
    inband = np.mean((np.array(rs) >= 0.8) & (np.array(rs) <= 1.25))
    print(f"  wd {wd}: max {a.max():.3f}, median {np.median(a):.3f}, frac in band {inband:.3f}, n={len(rs)}")
u = np.array(summary[UNSTABLE]["k_over_whole_qkv_elr_final_by_layer"])
print(f"  unstable: min {u.min():.3f} max {u.max():.3f}")
print("fig5:", json.dumps(fig5, indent=1))
