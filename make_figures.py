"""Draw every figure in the paper.

Inputs:  results/dgraphfin_results.json, results/dgraphfin_stats.json,
         data/DGraphFin/dgraphfin.npz (only for the full-benchmark overview)
Output:  figures/*.pdf and figures/*.png (300 dpi)

Figures are drawn at their printed size (7.16 in, the IEEE two-column width)
so the text stays at 7-8 pt in the paper.
"""
import json
import os

import matplotlib
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.labelsize": 8,
                     "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.7, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
                     "mathtext.fontset": "cm"})
C = ["#0072B2", "#009E73", "#D55E00"]
MK = ["o", "s", "^"]
NAMES = ["Temporal backbone", "Moderate", "Strong"]
W = 7.16
res = json.load(open(os.path.join("results", "dgraphfin_results.json")))
st = json.load(open(os.path.join("results", "dgraphfin_stats.json")))

def panel(ax, tag):
    ax.text(-0.02, 1.06, tag, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom", ha="right")

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    plt.close(fig)

# 1. Lemma 1: constant-score optimum --------------------------------------
w = np.linspace(0.5, 40, 400)
fig, ax = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw={"wspace": 0.28})
for pi, col in zip([0.02, 0.0504, 0.10, 0.20], ["#888888", C[0], C[1], C[2]]):
    ax[0].plot(w, w*pi/(w*pi + 1 - pi), color=col, lw=1.9 if pi == 0.0504 else 1.2, label=fr"$\pi={pi:g}$")
ax[0].axvline(15, color="k", ls=":", lw=0.9); ax[0].axhline(0.5, color="k", ls="--", lw=0.7)
ax[0].axhline(0.58, color="grey", ls="-.", lw=0.7)
ax[0].text(39.5, 0.595, r"$\tau=0.58$", ha="right", va="bottom", fontsize=7, color="dimgrey")
ax[0].text(39.5, 0.505, r"$p^*=0.5$", ha="right", va="bottom", fontsize=7)
ax[0].text(15.6, 0.03, r"cap $w=15$", fontsize=7)
ax[0].plot([15], [15*0.0504/(15*0.0504+1-0.0504)], "o", color=C[0], ms=4, zorder=5)
ax[0].set_xlabel(r"fraud weight $w$"); ax[0].set_ylabel(r"constant optimum $p^*$")
ax[0].set_xlim(0, 40); ax[0].set_ylim(0, 1); ax[0].legend(frameon=False, loc="lower right", handlelength=1.6)
panel(ax[0], "(a)")
pis = np.linspace(0.01, 0.30, 300)
for tau, ls in [(0.5, "-"), (0.58, "--")]:
    ax[1].plot(pis, tau*(1-pis)/(pis*(1-tau)), ls=ls, color="k", lw=1.2, label=fr"$\tau={tau}$")
ax[1].axhline(15, color="k", ls=":", lw=0.9); ax[1].text(0.295, 15.8, r"cap $w=15$", ha="right", fontsize=7)
ax[1].axvline(862/17098, color=C[0], lw=1.0); ax[1].text(0.056, 36, r"$\pi=0.0504$", color=C[0], fontsize=7)
ax[1].set_ylim(0, 40); ax[1].set_xlim(0, 0.30)
ax[1].set_xlabel(r"fraud prevalence $\pi$"); ax[1].set_ylabel(r"weight needed for $p^*\geq\tau$")
ax[1].legend(frameon=False, loc="center right", handlelength=1.8); panel(ax[1], "(b)")
save(fig, "fig_lemma1_constant_score")

# 2. Confusion matrices ---------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(W, 2.35), gridspec_kw={"wspace": 0.12}, sharey=True)
for a, r, n in zip(axs, res, NAMES):
    cm = np.array(r["confusion_matrix"]); rn = cm / cm.sum(1, keepdims=True)
    a.imshow(rn, cmap="Blues", vmin=0, vmax=1.05)
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{rn[i,j]*100:.1f}%\n({cm[i,j]:,})", ha="center", va="center",
                   fontsize=8, color="white" if rn[i, j] > 0.5 else "black")
    a.set_xticks([0, 1], ["Normal", "Fraud"]); a.set_yticks([0, 1], ["Normal", "Fraud"], rotation=90, va="center")
    a.set_xlabel("predicted class")
    a.set_title(f"{n} " + fr"($\alpha={r['alpha']:g},\ \beta={r['beta']:g}$)", fontsize=8)
    for s in a.spines.values(): s.set_visible(False)
    a.tick_params(length=0)
axs[0].set_ylabel("true class")
save(fig, "fig_confusion_matrices")

# 3. Logged histories -------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(W, 2.3), gridspec_kw={"wspace": 0.25})
for r, n, c, m in zip(res, NAMES, C, MK):
    h = r["history"]
    ax[0].plot(h["epoch"], h["loss"], color=c, marker=m, ms=3, lw=1.2, label=n)
    ax[1].plot(h["epoch"], h["auc"], color=c, marker=m, ms=3, lw=1.2, label=n)
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("total training objective"); panel(ax[0], "(a)")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("logged test ROC-AUC"); panel(ax[1], "(b)")
ax[1].legend(frameon=False, loc="lower right")
for a in ax: a.grid(alpha=0.25, lw=0.4)
save(fig, "fig_training_histories")

# 4. Trade-off and operating points -------------------------------------------
SH = ["Backbone", "Moderate", "Strong"]
# Direct labels next to each point (no legend box). Offsets in points: (dx, dy, horizontal alignment).
LBL_AB = [(6, 3, "left"), (6, 3, "left"), (-6, 3, "right")]       # Strong labelled on its left
LBL_C = [(6, -1, "left"), (-6, 0, "right"), (-6, 0, "right")]
fig, ax = plt.subplots(1, 3, figsize=(W, 2.45), gridspec_kw={"wspace": 0.42})
for key, a, lab, tag in [("eo", ax[0], r"$\Delta_{EO}$ (lower is better)", "(a)"),
                         ("sp", ax[1], r"$\Delta_{SP}$ (lower is better)", "(b)")]:
    for r, c, m, s, (dx, dy, ha) in zip(res, C, MK, SH, LBL_AB):
        a.scatter(r[key], r["auc"], color=c, marker=m, s=34, zorder=3)
        a.annotate(s, (r[key], r["auc"]), textcoords="offset points", xytext=(dx, dy),
                   ha=ha, va="bottom", fontsize=7, color=c)
    xs = [r[key] for r in res]; pad = (max(xs)-min(xs))*0.30
    a.set_xlim(max(0.0, min(xs)-pad), max(xs)+pad); a.set_ylim(0.640, 0.695)  # gaps are never negative
    a.set_xlabel(lab); a.grid(alpha=0.25, lw=0.4); panel(a, tag)
ax[0].set_ylabel("test ROC-AUC (higher is better)")
for r, c, m, s, (dx, dy, ha) in zip(res, C, MK, SH, LBL_C):
    (tn, fp), (fn, tp) = r["confusion_matrix"]
    x, y = fp/(fp+tn), tp/(tp+fn)
    ax[2].scatter(x, y, color=c, marker=m, s=34, zorder=3)
    ax[2].annotate(s, (x, y), textcoords="offset points", xytext=(dx, dy),
                   ha=ha, va="center", fontsize=7, color=c)
ax[2].plot([0, 0.4], [0, 0.4], color="grey", ls="--", lw=0.8)
ax[2].text(0.33, 0.30, "no skill", rotation=35, ha="center", va="top", fontsize=6.5, color="grey",
           rotation_mode="anchor")
ax[2].set_xlim(0, 0.4); ax[2].set_ylim(0, 0.7); ax[2].set_xlabel("false-positive rate")
ax[2].set_ylabel("recall (TPR)"); ax[2].grid(alpha=0.25, lw=0.4); panel(ax[2], "(c)")
save(fig, "fig_tradeoff_operating")

# 5. Statistics of the evaluated sample ---------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(W, 2.2), gridspec_kw={"wspace": 0.38})
vals = [st["n_normal"], st["n_fraud"]]
ax[0].bar(["Normal", "Fraud"], vals, color=[C[0], C[2]], width=0.6)
for i, v in enumerate(vals): ax[0].text(i, v + 300, f"{v:,}", ha="center", va="bottom", fontsize=7)
ax[0].set_ylabel("labelled nodes"); ax[0].set_ylim(0, 19500); panel(ax[0], "(a)")
et = sorted(st["edge_type_dist"].items(), key=lambda kv: int(kv[0]))
ax[1].bar([k for k, _ in et], [v for _, v in et], color=C[0]); ax[1].set_xlabel("relation type")
ax[1].set_ylabel("edge records"); ax[1].tick_params(axis="x", labelsize=6); panel(ax[1], "(b)")
b = st["timestamp_bins"]
ax[2].bar(b[:-1], st["timestamp_dist"], width=np.diff(b), align="edge", color=C[1], edgecolor="white", lw=0.4)
ax[2].set_xlabel("timestamp"); ax[2].set_ylabel("edge records"); panel(ax[2], "(c)")
save(fig, "fig_sample_statistics")
# 6. Full DGraphFin benchmark (raw npz) -------------------------------------------
npz = os.path.join("data", "DGraphFin", "dgraphfin.npz")
if os.path.exists(npz):
    d = np.load(npz)
    y, etype, ets = d["y"], d["edge_type"], d["edge_timestamp"]
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.2), gridspec_kw={"wspace": 0.38})
    cnt = np.bincount(y, minlength=4)
    labs = ["Normal", "Fraud", "Bkg. 2", "Bkg. 3"]
    ax[0].bar(labs, cnt / 1e6, color=[C[0], C[2], "#9a9a9a", "#c4c4c4"], width=0.6)
    for i, v in enumerate(cnt):
        ax[0].text(i, v / 1e6 + 0.03, f"{v/1e6:.2f}M" if v >= 1e5 else f"{v/1e3:.1f}k",
                   ha="center", va="bottom", fontsize=6.5)
    ax[0].set_ylabel("nodes (millions)"); ax[0].set_ylim(0, 1.9)
    ax[0].tick_params(axis="x", labelsize=7); panel(ax[0], "(a)")
    ty, tc = np.unique(etype, return_counts=True)
    ax[1].bar([str(t) for t in ty], tc / 1e6, color=C[0]); ax[1].set_xlabel("relation type")
    ax[1].set_ylabel("edges (millions)"); ax[1].tick_params(axis="x", labelsize=6); panel(ax[1], "(b)")
    med = np.median(ets)
    ax[2].hist(ets, bins=20, color=C[1], edgecolor="white", lw=0.4, weights=np.full(len(ets), 1e-3))
    ax[2].set_ylim(0, 460); ax[2].axvline(med, color="k", ls="--", lw=0.8)
    ax[2].text(med + 18, 452, f"median {med:.0f}", fontsize=6.5, va="top")
    ax[2].set_xlabel("timestamp"); ax[2].set_ylabel("edges (thousands)"); panel(ax[2], "(c)")
    save(fig, "fig_dgraphfin_overview")

# 7. Architecture diagram -------------------------------------------------------
fig, ax = plt.subplots(figsize=(W, 2.35))
ax.set_xlim(0, 100); ax.set_ylim(0, 31); ax.axis("off")
FILL = {"in": "#eef2f7", "graph": "#e3eef8", "enc": "#d6e8f5", "task": "#ddf1e8", "coh": "#fbe7dc", "loss": "#f5f5f5"}

def box(x, y, w, h, txt, kind, fs=7, bold_first=True, sp=2.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0",
                                fc=FILL[kind], ec="#4a4a4a", lw=0.7))
    lines = txt.split("\n")
    top = y + h/2 + (len(lines)-1)*sp/2
    if bold_first:
        ax.text(x + w/2, top, lines[0], ha="center", va="center", fontsize=fs, fontweight="bold")
        for k, l in enumerate(lines[1:], 1):
            ax.text(x + w/2, top - sp*k, l, ha="center", va="center", fontsize=fs-0.8)
    else:
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=fs)

def arrow(p, q, ls="-", style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=7, lw=0.8, color="#333333", ls=ls,
                                 shrinkA=0, shrinkB=0))

YM, H = 12, 11
box(0, YM, 13.2, H, "Input graph\n$G=(V,E,\\mathbf{X})$\nedge times $t_e$\n17 features", "in")
box(15.7, YM, 18.8, H, "Fixed adjacency\n$a_e=0.5+0.5\\,t_e/t_{\\max}$\nsymmetrise, self-loops\nnormalised $\\hat A$", "graph")
box(37.5, YM, 15, H, "Residual GCN\n$d\\!\\to\\!128\\!\\to\\!128\\!\\to\\!64$\nBatchNorm\ndropout 0.3", "enc")
box(55.5, YM, 11.5, H, "MLP\nencoder\n$64\\!\\to\\!128\\!\\to\\!64$\nout: $\\mathbf{h}_v$", "enc")
for a, b in [(13.2, 15.7), (34.5, 37.5), (52.5, 55.5)]:
    arrow((a, YM + H/2), (b, YM + H/2))
ax.text(25.1, YM + H + 1.2, "no trainable parameters", ha="center", fontsize=6, style="italic", color="#555555")
# split
zc_y, zs_y, zh = 21, 4, 6
box(70, zc_y, 9, zh, "$\\mathbf{z}_c\\in\\mathbb{R}^{48}$", "task", bold_first=False)
box(70, zs_y, 9, zh, "$\\mathbf{z}_s\\in\\mathbb{R}^{16}$", "coh", bold_first=False)
arrow((67, YM + H/2), (70, zc_y + zh/2)); arrow((67, YM + H/2), (70, zs_y + zh/2))
box(81.5, zc_y, 18.5, zh, "Fraud predictor\n$48\\!\\to\\!64\\!\\to\\!32\\!\\to\\!2$", "task")
box(81.5, zs_y, 18.5, zh, "Cohort predictor\n$16\\!\\to\\!32\\!\\to\\!16\\!\\to\\!2$", "coh")
arrow((79, zc_y + zh/2), (81.5, zc_y + zh/2)); arrow((79, zs_y + zh/2), (81.5, zs_y + zh/2))
arrow((74.5, zs_y + zh + 0.3), (74.5, zc_y - 0.3), ls="--", style="<|-|>")
ax.text(75.6, (zs_y + zh + zc_y)/2, "$\\alpha\\,\\mathcal{L}_{\\mathrm{dis}}$\nmax $|$Pearson$|$", fontsize=6.5, va="center", ha="left")
ax.text(90.75, zc_y + zh + 1.0, "weighted $\\mathcal{L}_{\\mathrm{cls}}\\ +\\ \\beta\\,\\Delta_{\\mathrm{SP}}^{\\mathrm{soft}}$",
        ha="center", va="bottom", fontsize=6.5)
ax.text(90.75, zs_y - 1.0, "$\\mathcal{L}_{\\mathrm{sens}}$ (cohort CE)", ha="center", va="top", fontsize=6.5)
ax.add_patch(FancyBboxPatch((0, 2.2), 67, 6.3, boxstyle="round,pad=0,rounding_size=1.0", fc=FILL["loss"], ec="#9a9a9a", lw=0.6, ls="--"))
ax.text(33.5, 5.35, "$\\mathcal{L}_{\\mathrm{total}}=(1-\\alpha)(\\mathcal{L}_{\\mathrm{cls}}+\\mathcal{L}_{\\mathrm{sens}})"
        "+\\alpha\\,\\mathcal{L}_{\\mathrm{dis}}+\\beta\\,\\Delta_{\\mathrm{SP}}^{\\mathrm{soft}}$", ha="center", va="center", fontsize=7.5)
save(fig, "fig_architecture")
print("Figures written to", OUT + "/:", ", ".join(sorted(os.listdir(OUT))))
