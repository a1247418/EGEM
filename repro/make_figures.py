"""Plots the accuracy summaries in repro/results."""
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
CLEAN, POIS = "#2a78d6", "#eb6834"
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


# accuracy per method, MNIST-8 and ISIC
fig, axs = plt.subplots(1, 2, figsize=(12, 3.5), sharey=True)
for ax, (scen, title) in zip(axs, [("mnist-8", "MNIST-8"), ("isic-1", "ISIC")]):
    s = summary(scen, 700)
    if s is not None:
        bars(ax, s, f"{title}, 700 samples/class, 5% slack")
axs[0].set_ylabel("Test accuracy")
axs[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
fig.tight_layout(); fig.savefig(f"{OUT}/accuracy_mnist_isic.png", dpi=150)

# accuracy per method, MNIST variants
fig, axs = plt.subplots(2, 2, figsize=(11, 6.4), sharey=True)
axs = axs.ravel()
for ax, v in zip(axs, ["artifact", "blur", "color", "remove"]):
    s = summary(f"mnist-rgb-{v}", 50, reps=10)
    if s is not None:
        bars(ax, s, f"MNIST-{v} (50/class, 10 reps)")
axs[0].set_ylabel("Test accuracy"); axs[2].set_ylabel("Test accuracy")
axs[-1].legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/accuracy_mnist_variants.png", dpi=150)

# accuracy vs. number of refinement samples
ns = [5, 10, 50, 200, 700]
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.2), sharey=True)
colors = {"egem": CLEAN, "pcaegem": POIS, "ridge": "#1baf7a", "rgem": "#eda100", "retrain": "#e87ba4"}  # palette slots 1-5
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
axs[0].set_ylabel("Test accuracy (5% slack)"); axs[1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.tight_layout(); fig.savefig(f"{OUT}/accuracy_vs_samples.png", dpi=150)
print("wrote", os.listdir(OUT))

# accuracy vs. selection slack
slacks = np.arange(0, 0.0701, 0.01)
fig, axs = plt.subplots(2, 2, figsize=(10.5, 6), sharex=True)
for row, scen in enumerate(["mnist-8", "isic-1"]):
    df = load(RES, 700, 5, scen)
    for col, (p, title) in enumerate([("none", "clean"), ("uniform", "100%-poisoned")]):
        ax = axs[row, col]
        for m, c in colors.items():
            dm = df[(df.method == m) & (df.poisoning == p)]
            if len(dm):
                ys = [select(dm, sl).top1.mean() for sl in slacks]
                ax.plot(slacks * 100, ys, color=c, lw=2, marker="o", ms=4, label=METHODS[m])
        ax.axhline(df[(df.method == "none") & (df.poisoning == p)].top1.mean(), color=MUTED, ls="--", lw=1,
                   label="Original")
        ax.set_title(f"{'MNIST-8' if scen == 'mnist-8' else 'ISIC'}, {title} test data", color=INK, fontsize=10)
        ax.yaxis.grid(True, color="#e4e3df", linewidth=0.6)
    axs[row, 0].set_ylabel("Test accuracy")
for ax in axs[1]:
    ax.set_xlabel("slack (%)")
axs[0, 1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.tight_layout(); fig.savefig(f"{OUT}/accuracy_vs_slack.png", dpi=150)
print("wrote accuracy_vs_slack.png")

# CelebA blond-hair recall per attribute subgroup (celeba_recall.py)
f9 = os.path.join(RES, "celeba_recall.csv")
if os.path.exists(f9):
    import pandas as pd
    d = pd.read_csv(f9).sort_values("recall_orig")
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(13.5, 3.8))
    ax.bar(x - 0.2, d.recall_orig, 0.4, color=CLEAN, label="Original", edgecolor=SURF, linewidth=1)
    ax.bar(x + 0.2, d.recall_refined, 0.4, color=POIS, label="PCA-EGEM (slack-selected α)", edgecolor=SURF, linewidth=1)
    ax.set_xticks(x, [f"{a} ({n})" for a, n in zip(d.attribute, d.n_blond)], rotation=90, fontsize=7)
    ax.set_ylabel("Recall of Blond_Hair"); ax.set_ylim(0, 1.02)
    ax.yaxis.grid(True, color="#e4e3df", linewidth=0.6); ax.set_axisbelow(True)
    ax.set_title("CelebA: recall per attribute subgroup (number of blond test images in parentheses)", color=INK, fontsize=10)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout(); fig.savefig(f"{OUT}/celeba_recall.png", dpi=150)
    print("wrote celeba_recall.png")

# separability of clean vs. CH-poisoned images per layer (layer_separability.py)
f_sep = os.path.join(RES, "layer_separability.csv")
if os.path.exists(f_sep):
    import pandas as pd
    d = pd.read_csv(f_sep)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.4))
    for ax, task in zip(axs, ["MNIST", "ISIC"]):
        dt = d[d.task == task]
        for (model, g), c in zip(dt.groupby("model", sort=False), [CLEAN, POIS, "#1baf7a"]):
            m = g.groupby("layer", sort=False).r2.agg(["mean", "std"])
            label = {"mnist": "MNIST-8 model", "isic_vgg16.model": "seed 0", "isic_vgg16_seed1.model": "seed 1",
                     "isic_vgg16_seed2.model": "seed 2"}.get(model, model)
            ax.errorbar(range(len(m)), m["mean"], yerr=m["std"], color=c, lw=2, marker="o", ms=4, capsize=2, label=label)
        ax.set_xticks(range(len(m)), m.index, rotation=60, ha="right", fontsize=7)
        ax.set_title(task, color=INK, fontsize=10); ax.yaxis.grid(True, color="#e4e3df", linewidth=0.6)
        ax.legend(frameon=False, fontsize=7)
    axs[0].set_ylabel("R² (clean vs. poisoned)")
    fig.tight_layout(); fig.savefig(f"{OUT}/layer_separability.png", dpi=150)
    print("wrote layer_separability.png")
