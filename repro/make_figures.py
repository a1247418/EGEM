"""Plots the results in repro/results in the style of the paper's figures."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load, select

RES, OUT, SLACK = "repro/results", os.environ.get("FIG_OUT", "repro/figures"), 0.05
os.makedirs(OUT, exist_ok=True)
METHODS = {"none": "Original", "retrain": "Retrain", "ridge": "Ridge", "rgem": "RGEM", "egem": "EGEM",
           "pcaegem": "PCA-EGEM"}
TITLE = {"mnist-8": "MNIST", "isic-1": "ISIC", "carton-packet": "carton/packet", "carton-crate": "carton/crate",
         "carton-envelope": "carton/envelope", "mtb-bbt": "mt. bike/bicycle-b.f.t."}
VARIANTS = {f"mnist-rgb-{v}": f"MNIST-{v}" for v in ["artifact", "color", "blur", "remove"]}
SAMPLES = [25, 50, 200, 500, 700]
SLACKS = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
REPS = {k: 5 for k in TITLE} | {k: 10 for k in VARIANTS}
_tab = sns.color_palette("tab10", 7)
COLOR = {"none": _tab[0], "rgem": _tab[1], "egem": _tab[2], "pcaegem": _tab[3], "retrain": _tab[5], "ridge": _tab[6]}
TASK_PALETTE = [sns.color_palette()[i] for i in (0, 2)] + sns.color_palette("rocket")[2:6][::-1]
SCENS = [k for k in TITLE if os.path.exists(os.path.join(RES, f"{k}_none_none_n700_r5.pkl"))]


def selected(scen, n, slack, reps=None):
    """Test accuracy of every rep at the selected hyperparameter: columns method, poisoning, rep, top1."""
    df = load(RES, n, reps or REPS[scen], scen)
    return None if df.empty else select(df, slack)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def variant_examples(fig, colors):
    """Legend for the MNIST variants: a color bar and an example 8 before and after adding each CH feature."""
    import torch, torchvision
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    from CH_datasets.scenario_examples import MNIST_RGB_POISONERS
    torch.manual_seed(0)
    mn = torchvision.datasets.MNIST(os.path.expanduser("~/EGEM_work/data"), train=True)
    x = mn.data[mn.targets == 8][0].float()[None].repeat(3, 1, 1) / 255.
    gs = fig.add_gridspec(len(VARIANTS), 3, left=0.86, right=0.99, top=0.88, bottom=0.11, wspace=0, hspace=0,
                          width_ratios=[0.25, 1, 1])
    for i, (v, c) in enumerate(zip(VARIANTS, colors)):
        bar = fig.add_subplot(gs[i, 0]); bar.set_facecolor(c)
        x_ch = MNIST_RGB_POISONERS[v.split("-")[-1]](p=1.0)._poison(x.clone())
        for j, img in enumerate([x, x_ch], 1):
            ax = fig.add_subplot(gs[i, j]); ax.imshow(img.permute(1, 2, 0).clamp(0, 1).numpy())
            if i == 0:
                ax.set_title(["Original", "CH"][j - 1], fontsize=12)
        for ax in fig.axes[-3:]:
            ax.set_xticks([]); ax.set_yticks([])


def bars(scens, titles, n, slack, name, examples=False):
    """Clean (light) and 100%-poisoned (dark) accuracy per method, one bar per task; black lines: task means."""
    rows = []
    for scen in scens:
        s = selected(scen, n, slack)
        if s is None:
            continue
        for r in s.itertuples():
            if r.method in METHODS:
                rows.append(dict(method=METHODS[r.method], task=titles[scen], d=4 if r.poisoning == "none" else 100,
                                 value=r.top1))
    if not rows:
        return
    df = pd.DataFrame(rows)
    order = [m for m in METHODS.values() if m in set(df.method)]
    tasks = [titles[s] for s in scens if titles[s] in set(df.task)]
    sns.set_style("white"); sns.set_context("talk", font_scale=1.)
    fig, ax = plt.subplots(figsize=(3 * len(order) + 2, 6))
    kw = dict(x="method", y="value", hue="task", order=order, hue_order=tasks, errorbar="sd", ax=ax, width=0.9,
              palette=TASK_PALETTE[:len(tasks)])
    sns.barplot(data=df[df.d == 4], alpha=.5, **kw)
    sns.barplot(data=df[df.d == 100], **kw)
    for i, m in enumerate(order):
        for d, ls in [(4, "--"), (100, "-")]:
            y = df[(df.method == m) & (df.d == d)].groupby("task").value.mean().mean()
            ax.plot([i - .45, i + .45], [y, y], linewidth=3, linestyle=ls, color="black")
    h, l = ax.get_legend_handles_labels()
    if examples:
        ax.get_legend().remove()
        fig.subplots_adjust(right=0.84)
        variant_examples(fig, TASK_PALETTE[:len(tasks)])
    else:
        ax.legend(h[len(tasks):], l[len(tasks):], title="", framealpha=0.95, bbox_to_anchor=(1., .5),
                  loc="center left", borderaxespad=0)
    ax.set(xlabel=None, ylabel="Accuracy", ylim=(.5, 1))
    ax.set_title(f"{int(round(slack * 100))}% slack", fontsize=20)
    sns.despine(left=True)
    save(fig, name)


def curves(method, xs, values_at, xlabel, xticklabels, name):
    """One panel per task: Original and `method` on clean (dotted) and 100%-poisoned (solid) test data."""
    sns.set_style("ticks"); sns.set_context("talk", font_scale=1.)
    fig, axs = plt.subplots(1, len(SCENS), figsize=(len(SCENS) * 3.3 + 2, 5), squeeze=False)
    handles = []
    for ax, scen in zip(axs[0], SCENS):
        handles = []
        for m in ["none", method]:
            pts = {p: [values_at(scen, m, p, x) for x in xs] for p in ["none", "uniform"]}
            if all(v is None for v in pts["none"] + pts["uniform"]):
                continue
            mean = {p: np.array([np.nan if v is None else v[0] for v in pts[p]]) for p in pts}
            for p, ls in [("none", ":"), ("uniform", "-")]:
                sd = [0 if v is None else v[1] for v in pts[p]]
                handles.append(ax.errorbar(range(len(xs)), mean[p], sd, linestyle=ls, marker=".", capsize=2,
                                           color=COLOR[m]))
            ax.fill_between(range(len(xs)), mean["uniform"], mean["none"], color=COLOR[m], alpha=0.3)
        ax.set_xticks(range(len(xs)), xticklabels)
        ax.set_title(TITLE[scen])
    labels = [f"{METHODS[m]}({d}%)" for m in ["none", method] for d in (0, 100)]
    fig.legend(handles, labels, bbox_to_anchor=(0., 1.02, 1., .102), loc="lower left", ncol=4, mode="expand",
               borderaxespad=0.)
    fig.supxlabel(xlabel, y=0.02); fig.supylabel("Accuracy", x=0.0)
    fig.tight_layout(); fig.subplots_adjust(wspace=.4)
    save(fig, name)


def at_slack(scen, m, p, slack, n=700):
    s = selected(scen, n, slack)
    if s is None:
        return None
    v = s[(s.method == m) & (s.poisoning == p)].top1
    return (v.mean(), v.std()) if len(v) else None


def at_samples(slack):
    def f(scen, m, p, n):
        # the unrefined model does not depend on the number of refinement samples
        return at_slack(scen, m, p, slack, 700 if m == "none" else n)
    return f


# Fig. 3 and Fig. 6: accuracy per method and task
bars(SCENS, TITLE, 700, SLACK, "accuracy_main.png")
bars(list(VARIANTS), VARIANTS, 50, SLACK, "accuracy_mnist_variants.png", examples=True)

# Fig. 4 / G.13: accuracy vs. selection slack; Fig. 5 / H.15: accuracy vs. number of refinement samples (1% slack)
for m in ["pcaegem", "egem", "retrain", "ridge", "rgem"]:
    curves(m, SLACKS, at_slack, "Slack (%)", [int(s * 100) for s in SLACKS], f"accuracy_vs_slack_{m}.png")
    curves(m, SAMPLES, at_samples(0.01), "Samples per class", SAMPLES, f"accuracy_vs_samples_{m}.png")

# Figs. 8, 9, I.17, I.18, C.11: CelebA (celeba_recall.py)
CELEBA = os.environ.get("CELEBA_RESULTS", os.path.join(RES, "celeba_recall"))
if os.path.exists(CELEBA + ".csv") and "precision_refined_wall" in open(CELEBA + ".csv").readline():
    d = pd.read_csv(CELEBA + ".csv")
    d["attribute"] = d.attribute.replace({"(all)": "All"})
    d = d.fillna(0)  # subgroups without blond (or without positive) predictions
    series = [("orig", "Original"), ("refined_wall", "PCA-EGEM"), ("wall", "Original+Wall")]

    def group_bars(metric, keys, ax):
        dd = d.sort_values(f"{metric}_orig")
        long = pd.concat([pd.DataFrame(dict(attribute=dd.attribute, value=dd[f"{metric}_{k}"], model=label))
                          for k, label in series if k in keys], ignore_index=True)
        sns.barplot(data=long, x="attribute", y="value", hue="model", palette="dark:salmon", width=0.85, ax=ax)
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=55, ha="right")
        ax.set(xlabel="", ylabel=metric.capitalize())
        ax.legend(loc="lower right", framealpha=0.95, title="")
        for t in ax.get_xticklabels():
            if t.get_text() == "All":
                t.set_fontweight("bold")
            elif t.get_text() == "Blond_Hair":
                t.set_fontstyle("italic")

    sns.set_style("ticks"); sns.set_context("talk", font_scale=.8)
    fig, ax = plt.subplots(figsize=(15, 4.5))
    group_bars("recall", ["orig", "refined_wall"], ax)
    save(fig, "celeba_recall.png")
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    for ax, metric in zip(axs, ["precision", "recall"]):
        group_bars(metric, ["orig", "refined_wall", "wall"], ax)
    fig.tight_layout()
    save(fig, "celeba_precision_recall.png")

    h = np.load(CELEBA + "_heatmaps.npz")
    unnorm = lambda t: np.clip(t.transpose(1, 2, 0) * h["std"] + h["mean"], 0, 1)
    n = len(h["idx"])
    sns.set_style("white"); plt.rcParams["font.size"] = 14
    fig, axs = plt.subplots(3, n, figsize=(2 * n, 3 * 2.45), squeeze=False, gridspec_kw={"wspace": 0, "hspace": 0})
    for k in range(n):
        axs[0, k].imshow(unnorm(h["images"][k]))
        for r, R in [(1, h["R_orig"][k]), (2, h["R_refined"][k])]:
            lim = np.abs(R).max() + 1e-12
            axs[r, k].imshow(R, cmap="seismic", vmin=-lim, vmax=lim)
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    axs[1, 0].set_ylabel("R$^{Original}_{Blond}$", fontsize=20)
    axs[2, 0].set_ylabel("R$^{PCA-EGEM}_{Blond}$", fontsize=20)
    save(fig, "celeba_heatmaps.png")

    ex = h["wall_examples"]
    fig, axs = plt.subplots(2, len(ex), figsize=(2 * len(ex), 2 * 2.45), gridspec_kw={"wspace": 0, "hspace": 0})
    for k, img in enumerate(ex):
        walled = img.copy(); walled[:, -50:, :] = h["wall"]
        axs[0, k].imshow(unnorm(img)); axs[1, k].imshow(unnorm(walled))
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    axs[0, 0].set_ylabel("Original", fontsize=16); axs[1, 0].set_ylabel("Occluded", fontsize=16)
    save(fig, "celeba_wall.png")

    corr = pd.read_csv(CELEBA + "_attr_corr.csv", index_col=0)
    sns.set_style("white"); sns.set_context("notebook")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.set_style("ticks")
    sns.heatmap(corr, cmap="vlag", vmin=-1, vmax=1, ax=ax, xticklabels=True, yticklabels=True)
    ax.invert_yaxis()
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")
    save(fig, "celeba_attr_corr.png")

# Fig. J.19: separability of clean vs. CH-poisoned images per layer (layer_separability.py)
f_sep = os.path.join(RES, "layer_separability.csv")
if os.path.exists(f_sep):
    d = pd.read_csv(f_sep)
    d = d[d.model.isin(["mnist", "isic_vgg16.model", "resnet50", "vgg16"])]
    tasks = [t for t in ["MNIST", "ISIC", "carton-packet", "carton-crate", "carton-envelope", "mtb-bbt"] if t in set(d.task)]
    names = {"carton-packet": "carton/packet", "carton-crate": "carton/crate", "carton-envelope": "carton/envelope",
             "mtb-bbt": "mt. bike/bicycle b.f.t."}
    plt.rcdefaults(); plt.rcParams["font.size"] = 14
    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
    for k, (ax, task) in enumerate(zip(axs.ravel(), tasks)):
        m = d[d.task == task].groupby("layer", sort=False).r2.mean()
        ax.plot(list(m.index), m.values)
        ax.set_xticks(np.arange(len(m)), m.index, rotation=45, ha="right")
        if k % 3 == 0:
            ax.set_ylabel("R$^2$")
        ax.set_title(names.get(task, task))
    fig.subplots_adjust(hspace=1, wspace=.2)
    save(fig, "layer_separability.png")

# Fig. J.20: change of the outputs through refinement, clean vs. poisoned test images (run_scenario --save_outputs)


def logit_change(scen, m, p):
    """Per-image mean |outputs_refined - outputs_original| at the selected hyperparameter, rep 0."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    import pickle
    from selection import select_by_slack, REFINEMENT_HP
    f = lambda meth, ext: os.path.join(RES, f"{scen}_{meth}_{p}_n700_r5{ext}")
    if not all(os.path.exists(f(x, e)) for x, e in [(m, ".pkl"), (m, "_outputs.npz"), ("none", "_outputs.npz")]):
        return None
    sel = [r for r in select_by_slack(pickle.load(open(f(m, ".pkl"), "rb")), m, SLACK) if r["rep"] == 0][0]
    out = np.load(f(m, "_outputs.npz"))[f"rep0_{sel[REFINEMENT_HP[m][0]]}"]
    orig = np.load(f("none", "_outputs.npz"))["rep0_none"]
    return np.abs(out - orig).mean(axis=1)


scens = [s for s in TITLE if os.path.exists(os.path.join(RES, f"{s}_none_none_n700_r5_outputs.npz"))]
if scens:
    ms = ["retrain", "ridge", "rgem", "egem", "pcaegem"]
    plt.rcdefaults(); plt.rcParams["font.size"] = 14; plt.rcParams["ytick.labelsize"] = 12
    fig, axs = plt.subplots(len(ms), len(scens), figsize=(16, 8), squeeze=False)
    for j, scen in enumerate(scens):
        for i, m in enumerate(ms):
            ax = axs[i, j]
            for p, label in [("none", "Clean"), ("uniform", "Poisoned")]:
                dif = logit_change(scen, m, p)
                if dif is not None:
                    ax.hist(dif, label=label, alpha=0.5)
            if i == 0:
                ax.set_title(TITLE[scen].replace("-", " "), fontsize=12)
            if j == 0:
                ax.set_ylabel(METHODS[m], fontsize=12)
            ax.xaxis.set_visible(False)
    fig.subplots_adjust(hspace=.15, wspace=.3)
    axs[-1, len(scens) // 2].legend(loc="upper center", bbox_to_anchor=(0, -0.2), ncol=2)
    save(fig, "logit_change.png")
