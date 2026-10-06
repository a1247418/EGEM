"""Plots the accuracy summaries in repro/results."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load, select

RES, OUT, SLACK = "repro/results", os.environ.get("FIG_OUT", "repro/figures"), 0.05
METHODS = {"none": "Original", "retrain": "Retrain", "ridge": "Ridge", "rgem": "RGEM", "egem": "EGEM",
           "pcaegem": "PCA-EGEM"}
CLEAN, POIS = "#2a78d6", "#eb6834"
INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})
os.makedirs(OUT, exist_ok=True)
TITLE = {"mnist-8": "MNIST-8", "isic-1": "ISIC", "carton-packet": "carton/packet", "carton-crate": "carton/crate",
         "carton-envelope": "carton/envelope", "mtb-bbt": "mt. bike/bicycle-b.f.t."}
NS = {k: [5, 10, 50, 200, 700] if k == "mnist-8" else [25, 50, 200, 500, 700] for k in TITLE}
SCENS = [k for k in TITLE if os.path.exists(os.path.join(RES, f"{k}_none_none_n700_r5.pkl"))]


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


# accuracy per method and dataset
cols = 2 if len(SCENS) <= 2 else 3
rows_ = int(np.ceil(len(SCENS) / cols))
fig, axs = plt.subplots(rows_, cols, figsize=(6 * cols, 3.4 * rows_), squeeze=False)
for ax, scen in zip(axs.ravel(), SCENS):
    s_ = summary(scen, 700)
    if s_ is not None:
        bars(ax, s_, TITLE[scen])
        lo = np.nanmin(s_["mean"].values)
        ax.set_ylim(max(0.0, np.floor((lo - 0.05) * 20) / 20), 1.0)
for ax in axs.ravel()[len(SCENS):]:
    ax.axis("off")
for r in range(rows_):
    axs[r, 0].set_ylabel("Test accuracy (5% slack)")
h_, l_ = axs[0, 0].get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, loc="lower center", ncol=2, fontsize=9)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f"{OUT}/accuracy_main.png", dpi=150)

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
colors = {"egem": CLEAN, "pcaegem": POIS, "ridge": "#1baf7a", "rgem": "#eda100", "retrain": "#e87ba4"}


def sample_curves(methods, slack, fname):
    fig, axs = plt.subplots(len(SCENS), 2, figsize=(10.5, 2.9 * len(SCENS)), squeeze=False)
    for row, scen in enumerate(SCENS):
        ns = NS[scen]
        for ax, p, title in [(axs[row, 0], "none", "clean"), (axs[row, 1], "uniform", "100%-poisoned")]:
            for m in methods:
                pts = []
                for n in ns:
                    df = load(RES, n, 5, scen)
                    d = df[(df.method == m) & (df.poisoning == p)]
                    if len(d):
                        s_ = select(d, slack).top1
                        pts.append((n, s_.mean(), s_.std()))
                if pts:
                    xs, mu, sd = zip(*pts)
                    ax.errorbar(xs, mu, yerr=sd, color=colors[m], lw=2, marker="o", ms=5, capsize=2, label=METHODS[m])
            df = load(RES, 700, 5, scen)
            orig = df[(df.method == "none") & (df.poisoning == p)].top1.mean()
            ax.axhline(orig, color=MUTED, ls="--", lw=1, label="Original")
            ax.set_xscale("log"); ax.set_xticks(ns, [str(n) for n in ns])
            ax.set_title(f"{TITLE[scen]}, {title} test data", color=INK, fontsize=10)
            ax.yaxis.grid(True, color="#e4e3df", linewidth=0.6)
        axs[row, 0].set_ylabel(f"Test accuracy ({slack:.0%} slack)")
    for ax in axs[-1]:
        ax.set_xlabel("refinement samples per class")
    axs[0, 1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}", dpi=150)


sample_curves(list(colors), SLACK, "accuracy_vs_samples.png")
sample_curves(["pcaegem"], 0.01, "accuracy_vs_samples_pcaegem_slack1.png")

# accuracy vs. selection slack
slacks = np.arange(0, 0.0701, 0.01)
fig, axs = plt.subplots(len(SCENS), 2, figsize=(10.5, 2.9 * len(SCENS)), sharex=True, squeeze=False)
for row, scen in enumerate(SCENS):
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
        ax.set_title(f"{TITLE[scen]}, {title} test data", color=INK, fontsize=10)
        ax.yaxis.grid(True, color="#e4e3df", linewidth=0.6)
    axs[row, 0].set_ylabel("Test accuracy")
for ax in axs[-1]:
    ax.set_xlabel("slack (%)")
axs[0, 1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.tight_layout(); fig.savefig(f"{OUT}/accuracy_vs_slack.png", dpi=150)
print("wrote accuracy_vs_slack.png")

# CelebA: precision and recall of blond hair per attribute subgroup, LRP heatmaps, wall examples, correlations
CELEBA = os.environ.get("CELEBA_RESULTS", os.path.join(RES, "celeba_recall"))
if os.path.exists(CELEBA + ".csv") and "precision_orig" in open(CELEBA + ".csv").readline():
    import pandas as pd
    d = pd.read_csv(CELEBA + ".csv").sort_values("recall_orig")
    x = np.arange(len(d))
    series = [("orig", "Original", CLEAN), ("refined", "PCA-EGEM", POIS), ("wall", "Original + wall", "#1baf7a")]
    fig, axs = plt.subplots(2, 1, figsize=(13.5, 6.5), sharex=True)
    for ax, metric in zip(axs, ["precision", "recall"]):
        for k, (key, label, c) in enumerate(series):
            ax.bar(x + (k - 1) * 0.27, d[f"{metric}_{key}"], 0.27, color=c, label=label, edgecolor=SURF, linewidth=0.5)
        ax.set_ylabel(f"{metric.capitalize()} of Blond_Hair"); ax.set_ylim(0, 1.02)
        ax.yaxis.grid(True, color="#e4e3df", linewidth=0.6); ax.set_axisbelow(True)
    axs[1].set_xticks(x, [f"{a} ({n})" for a, n in zip(d.attribute, d.n_blond)], rotation=90, fontsize=7)
    axs[0].legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    axs[0].set_title("CelebA: per attribute subgroup (number of blond test images in parentheses)", color=INK, fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/celeba_recall.png", dpi=150)
    print("wrote celeba_recall.png")

    h = np.load(CELEBA + "_heatmaps.npz")
    unnorm = lambda t: np.clip(t.transpose(1, 2, 0) * h["std"] + h["mean"], 0, 1)
    n = len(h["idx"])
    fig, axs = plt.subplots(3, n, figsize=(1.9 * n, 6.6), squeeze=False)
    for k in range(n):
        axs[0, k].imshow(unnorm(h["images"][k]))
        for r, (R, s_) in enumerate([(h["R_orig"][k], h["score_orig"][k]), (h["R_refined"][k], h["score_refined"][k])], 1):
            lim = np.abs(R).max() + 1e-12
            axs[r, k].imshow(R, cmap="bwr", vmin=-lim, vmax=lim)
            axs[r, k].set_title(f"blond score {s_:.1f}", fontsize=7, color=MUTED)
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    for r, label in enumerate(["image", "LRP original", "LRP PCA-EGEM"]):
        axs[r, 0].set_ylabel(label, fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/celeba_heatmaps.png", dpi=150)

    ex = h["wall_examples"]
    fig, axs = plt.subplots(2, len(ex), figsize=(1.9 * len(ex), 4.6))
    for k, img in enumerate(ex):
        walled = img.copy(); walled[:, -50:, :] = h["wall"]
        axs[0, k].imshow(unnorm(img)); axs[1, k].imshow(unnorm(walled))
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    axs[0, 0].set_ylabel("original", fontsize=9); axs[1, 0].set_ylabel("occluded", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/celeba_wall.png", dpi=150)

    corr = pd.read_csv(CELEBA + "_attr_corr.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8, label="correlation")
    fig.tight_layout(); fig.savefig(f"{OUT}/celeba_attr_corr.png", dpi=150)
    print("wrote celeba_heatmaps.png, celeba_wall.png, celeba_attr_corr.png")

# separability of clean vs. CH-poisoned images per layer (layer_separability.py)
f_sep = os.path.join(RES, "layer_separability.csv")
if os.path.exists(f_sep):
    import pandas as pd
    d = pd.read_csv(f_sep)
    tasks = [t for t in ["MNIST", "ISIC", "carton-packet", "carton-crate", "carton-envelope", "mtb-bbt"] if t in set(d.task)]
    fig, axs = plt.subplots(int(np.ceil(len(tasks) / 3)), min(3, len(tasks)), figsize=(14, 3.6 * np.ceil(len(tasks) / 3)),
                            squeeze=False)
    for ax in axs.ravel()[len(tasks):]:
        ax.axis("off")
    for ax, task in zip(axs.ravel(), tasks):
        dt = d[d.task == task]
        for (model, g), c in zip(dt.groupby("model", sort=False), [CLEAN, POIS, "#1baf7a"]):
            m = g.groupby("layer", sort=False).r2.agg(["mean", "std"])
            label = {"mnist": "MNIST-8 model", "isic_vgg16.model": "seed 0", "isic_vgg16_seed1.model": "seed 1",
                     "isic_vgg16_seed2.model": "seed 2"}.get(model, model)
            ax.errorbar(range(len(m)), m["mean"], yerr=m["std"], color=c, lw=2, marker="o", ms=4, capsize=2, label=label)
        ax.set_xticks(range(len(m)), m.index, rotation=60, ha="right", fontsize=7)
        ax.set_title(TITLE.get(task, task), color=INK, fontsize=10); ax.yaxis.grid(True, color="#e4e3df", linewidth=0.6)
        if task == "ISIC":
            ax.legend(frameon=False, fontsize=7)
    for r in range(axs.shape[0]):
        axs[r, 0].set_ylabel("R² (clean vs. poisoned)")
    fig.tight_layout(); fig.savefig(f"{OUT}/layer_separability.png", dpi=150)
    print("wrote layer_separability.png")

# change of the output logits through refinement, clean vs. poisoned test images (run_scenario --save_outputs)
LOG = os.path.join(RES, "logits")


def logit_change(scen, m, p):
    """Per-image ||logits_refined - logits_original|| at the slack-selected hyperparameter (rep 0)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    import pickle
    from selection import select_by_slack, REFINEMENT_HP
    f = lambda meth, ext: os.path.join(LOG, f"{scen}_{meth}_{p}_n700_r1{ext}")
    if not all(os.path.exists(f(x, e)) for x, e in [(m, ".pkl"), (m, "_outputs.npz"), ("none", "_outputs.npz")]):
        return None
    sel = select_by_slack(pickle.load(open(f(m, ".pkl"), "rb")), m)[0]
    out = np.load(f(m, "_outputs.npz"))[f"rep0_{sel[REFINEMENT_HP[m][0]]}"]
    orig = np.load(f("none", "_outputs.npz"))["rep0_none"]
    return np.linalg.norm(out - orig, axis=1)


scens = [s for s in TITLE if os.path.exists(os.path.join(LOG, f"{s}_none_none_n700_r1_outputs.npz"))]
if scens:
    ms = ["retrain", "ridge", "rgem", "egem", "pcaegem"]
    fig, axs = plt.subplots(len(ms), len(scens), figsize=(2.6 * len(scens), 1.5 * len(ms)), squeeze=False)
    for j, scen in enumerate(scens):
        for i, m in enumerate(ms):
            ax = axs[i, j]
            dc, dp = logit_change(scen, m, "none"), logit_change(scen, m, "uniform")
            if dc is not None and dp is not None:
                bins = np.linspace(0, np.percentile(np.concatenate([dc, dp]), 99), 25)
                ax.hist(dc, bins, color=CLEAN, alpha=0.6, label="clean")
                ax.hist(dp, bins, color=POIS, alpha=0.6, label="poisoned")
            if i == 0:
                ax.set_title(TITLE[scen], color=INK, fontsize=9)
            if j == 0:
                ax.set_ylabel(METHODS[m], fontsize=9)
            ax.set_yticks([])
    axs[-1, 0].set_xlabel("||Δ logits|| through refinement")
    axs[0, -1].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/logit_change.png", dpi=150)
    print("wrote logit_change.png")
