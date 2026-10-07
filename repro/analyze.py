"""Selects each method's hyperparameter by validation slack and summarizes test accuracy on clean and
poisoned data; can export all runs to CSV."""
import argparse, glob, os, sys, pickle as pkl
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from selection import REFINEMENT_HP as HP, select_value  # HP: hyperparameter name, and whether smaller = stronger


def load(results_dir, n, reps, scenario="mnist-8"):
    rows = []
    for f in glob.glob(os.path.join(results_dir, f"{scenario}_*_n{n}_r{reps}.pkl")):
        ref, pois = os.path.basename(f)[len(scenario) + 1:].split("_")[:2]
        for r in pkl.load(open(f, "rb")):
            rows.append(dict(method=ref, poisoning=pois, rep=r["rep"], top1=float(np.mean(r["top1"])),
                             top1_val=float(np.mean(r["top1_val"])),
                             orig_top1_val=float(np.mean(r.get("orig_top1_val", np.nan))),
                             hp=r.get(HP.get(ref.split("+")[0], (None,))[0], np.nan)))
    return pd.DataFrame(rows, columns=["method", "poisoning", "rep", "top1", "top1_val", "orig_top1_val", "hp"])


def select(df, slack):
    """One hyperparameter per method and poisoning, chosen on the rep-averaged validation accuracy."""
    out = []
    for (m, p), g in df.groupby(["method", "poisoning"]):
        if m.split("+")[0] == "none":
            out += [dict(method=m, poisoning=p, rep=r.rep, top1=r.top1, hp=np.nan) for r in g.itertuples()]
            continue
        val = g.groupby("hp").top1_val.mean()
        hp = select_value(list(val.index), list(val.values), g.orig_top1_val.mean(), HP[m.split("+")[0]][1], slack)
        out += [dict(method=m, poisoning=p, rep=r.rep, top1=r.top1, hp=hp) for r in g[g.hp == hp].itertuples()]
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
