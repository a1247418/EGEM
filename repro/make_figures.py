"""Figures for the MNIST reproduction (paper Fig. 3 MNIST bars, Fig. 6, Supp. H sample-size curve)."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load, select

RES, OUT, SLACK = "repro/results", "repro/figures", 0.05
METHODS = {"none": "Original", "retrain": "Retrain", "ridge": "Ridge", "rgem": "RGEM", "egem": "EGEM",
           "pcaegem": "PCA-EGEM"}
CLEAN, POIS = "#2a78d6", "#eb6834"  # categorical slots 1-2 of the reference palette
INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})
os.makedirs(OUT, exist_ok=True)


def summary(scenario, n, reps=5):
    df = load(RES, n, reps, scenario)
    if df.empty:
        return None
    return select(df, SLACK).groupby(["method", "poisoning"]).top1.agg(["mean", "std"])


def bars(ax, s, title):
    ms = [m for m in METHODS if (m, "none") in s.index]
    x = np.arange(len(ms))
    for off, p, c, lab in [(-0.19, "none", CLEAN, "clean (0%)"), (0.19, "uniform", POIS, "poisoned (100%)")]:
        mu = [s.loc[(m, p), "mean"] if (m, p) in s.index else np.nan for m in ms]
        sd = [s.loc[(m, p), "std"] if (m, p) in s.index else 0 for m in ms]
        ax.bar(x + off, mu, 0.36, color=c, label=lab, edgecolor=SURF, linewidth=1)
        ax.errorbar(x + off, mu, yerr=sd, fmt="none", ecolor=INK, elinewidth=1, capsize=2)
    ax.set_xticks(x, [METHODS[m] for m in ms])
    ax.set_ylim(0.3, 1.0)
    ax.yaxis.grid(True, color="#e4e3df", linewidth=0.6); ax.set_axisbelow(True)
    ax.set_title(title, color=INK, fontsize=10)


# Fig. 3 (MNIST part)
fig, ax = plt.subplots(figsize=(6.5, 3.5))
bars(ax, summary("mnist-8", 700), "MNIST-8, 700 samples/class, 5% slack")
ax.set_ylabel("Test accuracy"); ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_mnist.png", dpi=150)

# Fig. 6: CH feature variants
fig, axs = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
for ax, v in zip(axs, ["artifact", "blur", "color", "remove"]):
    s = summary(f"mnist-rgb-{v}", 50)
    if s is not None:
        bars(ax, s, f"MNIST-{v} (50/class)")
axs[0].set_ylabel("Test accuracy"); axs[-1].legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig6_mnist_variants.png", dpi=150)

# Supp. H: accuracy vs. number of refinement samples
ns = [5, 10, 50, 200, 700]
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
colors = {"ridge": "#1baf7a", "egem": CLEAN, "pcaegem": POIS}
for ax, p, title in [(axs[0], "none", "clean test data"), (axs[1], "uniform", "100%-poisoned test data")]:
    for m, c in colors.items():
        pts = [(n, s.loc[(m, p)]) for n in ns if (s := summary("mnist-8", n)) is not None and (m, p) in s.index]
        if pts:
            xs, rows = zip(*pts)
            ax.errorbar(xs, [r["mean"] for r in rows], yerr=[r["std"] for r in rows], color=c, lw=2,
                        marker="o", ms=5, capsize=2, label=METHODS[m])
    orig = summary("mnist-8", 700).loc[("none", p), "mean"]
    ax.axhline(orig, color=MUTED, ls="--", lw=1, label="Original")
    ax.set_xscale("log"); ax.set_xticks(ns, [str(n) for n in ns]); ax.set_xlabel("refinement samples per class")
    ax.set_title(f"MNIST-8, {title}", color=INK, fontsize=10)
    ax.yaxis.grid(True, color="#e4e3df", linewidth=0.6)
axs[0].set_ylabel("Test accuracy (5% slack)"); axs[1].legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/figH_samples.png", dpi=150)
print("wrote", os.listdir(OUT))
