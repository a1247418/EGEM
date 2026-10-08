"""Hyperparameter selection: the strongest refinement whose validation accuracy is within a relative `slack` of
the unrefined model's, i.e. val_acc >= (1 - slack) * orig_val_acc. One value is chosen per setting from the
rep-averaged validation accuracies; if none qualifies, the weakest value is taken."""
from typing import Dict, List

import numpy as np

# Refinement hyperparameter per method, and whether a *smaller* value means *stronger* refinement
REFINEMENT_HP = {
    "egem": ("alpha", True),
    "egemfull": ("alpha", True),
    "pcaegem": ("alpha", True),
    "ridge": ("lmbda", False),
    "rgem": ("lmbda", False),
    "retrain": ("n_epochs", False),
}


def select_value(values, val_accs, orig_val_acc: float, smaller_is_stronger: bool, slack: float = 0.05):
    """Returns the strongest of `values` whose validation accuracy is at least (1 - slack) * orig_val_acc."""
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=smaller_is_stronger)  # weak -> strong
    chosen = order[0]
    for i in order:
        if val_accs[i] >= orig_val_acc * (1 - slack):
            chosen = i
    return values[chosen]


def select_by_slack(results: List[Dict], refinement: str, slack: float = 0.05, per_rep: bool = False) -> List[Dict]:
    """Returns the selected result of every rep. `results` are the dicts returned by run.run_experiment.
    per_rep: select separately for every rep instead of once on the rep-averaged validation accuracies."""
    if refinement not in REFINEMENT_HP:
        return results  # nothing to select (e.g. the unrefined model)
    hp, smaller_is_stronger = REFINEMENT_HP[refinement]
    reps = sorted({r["rep"] for r in results})
    groups = [[rep] for rep in reps] if per_rep else [reps]
    selected = []
    for group in groups:
        rows = [r for r in results if r["rep"] in group]
        values = sorted({r[hp] for r in rows})
        val = [np.mean([np.mean(r["top1_val"]) for r in rows if r[hp] == v]) for v in values]
        orig = np.mean([np.mean(r["orig_top1_val"]) for r in rows])
        best = select_value(values, val, orig, smaller_is_stronger, slack)
        selected += [r for r in rows if r[hp] == best]
    return sorted(selected, key=lambda r: r["rep"])
