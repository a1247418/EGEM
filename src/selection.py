"""Hyperparameter selection: the strongest refinement whose validation accuracy is at most `slack` below that
of the unrefined model."""
from typing import Dict, List

import numpy as np

# Refinement hyperparameter per method, and whether a *smaller* value means *stronger* refinement
REFINEMENT_HP = {
    "egem": ("alpha", True),
    "pcaegem": ("alpha", True),
    "ridge": ("lmbda", False),
    "rgem": ("lmbda", False),
    "retrain": ("n_epochs", False),
}


def select_by_slack(results: List[Dict], refinement: str, slack: float = 0.05) -> List[Dict]:
    """Returns the selected result of every rep. `results` are the dicts returned by run.run_experiment.
    If no value is within the slack, the one with the best validation accuracy is taken."""
    if refinement not in REFINEMENT_HP:
        return results  # nothing to select (e.g. the unrefined model)
    hp, smaller_is_stronger = REFINEMENT_HP[refinement]
    selected = []
    for rep in sorted({r["rep"] for r in results}):
        rows = [r for r in results if r["rep"] == rep]
        ok = [r for r in rows if np.mean(r["top1_val"]) >= np.mean(r["orig_top1_val"]) - slack]
        if not ok:
            best_val = max(np.mean(r["top1_val"]) for r in rows)
            ok = [r for r in rows if np.mean(r["top1_val"]) == best_val]
        ok.sort(key=lambda r: r[hp], reverse=not smaller_is_stronger)
        selected.append(ok[0])
    return selected
