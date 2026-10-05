"""Paper-style hyperparameter selection (Sec. 4.5) and summary table for the MNIST-8 runs.

For every rep, pick the strongest refinement whose validation accuracy is at most `slack`
below the unrefined model's validation accuracy, then report test accuracy on clean (0%)
and fully poisoned (100%) data.
"""
import argparse, glob, os, pickle as pkl
import numpy as np
import pandas as pd

# Hyperparameter name and whether a *smaller* value means *stronger* refinement
HP = {"egem": ("alpha", True), "pcaegem": ("alpha", True), "ridge": ("lmbda", False)}


def load(results_dir, n, reps, scenario="mnist-8"):
    rows = []
    for f in glob.glob(os.path.join(results_dir, f"{scenario}_*_n{n}_r{reps}.pkl")):
        ref, pois = os.path.basename(f)[len(scenario) + 1:].split("_")[:2]
        for r in pkl.load(open(f, "rb")):
            rows.append(dict(method=ref, poisoning=pois, rep=r["rep"], top1=float(np.mean(r["top1"])),
                             top1_val=float(np.mean(r["top1_val"])),
                             orig_top1_val=float(np.mean(r.get("orig_top1_val", np.nan))),
                             hp=r.get(HP.get(ref, (None,))[0], np.nan)))
    return pd.DataFrame(rows)


def select(df, slack):
    out = []
    for (m, p, rep), g in df.groupby(["method", "poisoning", "rep"]):
        if m == "none":
            out.append(dict(method=m, poisoning=p, rep=rep, top1=g.top1.iloc[0], hp=np.nan))
            continue
        ok = g[g.top1_val >= g.orig_top1_val - slack]
        if len(ok) == 0:  # nothing within slack: fall back to the best validation accuracy
            ok = g[g.top1_val == g.top1_val.max()]
        smaller_is_stronger = HP[m][1]
        row = ok.sort_values("hp", ascending=smaller_is_stronger).iloc[0]
        out.append(dict(method=m, poisoning=p, rep=rep, top1=row.top1, hp=row.hp))
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="repro/results")
    ap.add_argument("--scenario", default="mnist-8")
    ap.add_argument("--n", type=int, default=700)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--slack", type=float, nargs="+", default=[0.05])
    ap.add_argument("--export", default=None, help="write all per-hyperparameter rows of all runs to this CSV")
    a = ap.parse_args()
    if a.export:
        rows = []
        for f in sorted(glob.glob(os.path.join(a.results, "*_r*.pkl"))):
            scen, rest = os.path.basename(f).split("_", 1)
            n, reps = (int(t[1:]) for t in rest[:-4].split("_")[-2:])
            rows.append(load(a.results, n, reps, scen).assign(scenario=scen, n=n))
        pd.concat(rows).drop_duplicates().to_csv(a.export, index=False)
        print("exported", a.export)
    df = load(a.results, a.n, a.reps, a.scenario)
    for s in a.slack:
        sel = select(df, s)
        tab = sel.groupby(["method", "poisoning"]).top1.agg(["mean", "std"]).unstack("poisoning")
        hps = sel.groupby("method").hp.agg(lambda x: sorted(set(np.round(x.dropna(), 4))))
        print(f"\n=== {a.scenario}, n={a.n}/class, slack={s:.0%} : test accuracy (mean, std over reps) ===")
        print(tab.round(3).to_string())
        print("chosen hyperparameters:", hps.to_dict())
