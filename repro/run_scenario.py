"""Runs one scenario x refinement x poisoning level and saves the results. Run from the repo root."""
import argparse, os, sys, time, pickle as pkl
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import torch
from run import run_experiment
from selection import select_by_slack

p = argparse.ArgumentParser()
p.add_argument("--scenario", default="mnist-8")
p.add_argument("--refinement", default="none")
p.add_argument("--poisoning", default="none")
p.add_argument("--n_reps", type=int, default=5)
p.add_argument("--n_samples", type=int, default=700)
p.add_argument("--data_root", default=os.path.expanduser("~/EGEM_work/data"))
p.add_argument("--out", default="repro/results")
p.add_argument("--explanation_type", default=None)
p.add_argument("--threads", type=int, default=4)
p.add_argument("--refiner_kwargs", default=None, help='JSON, e.g. \'{"spatial_sum": true}\'')
p.add_argument("--tag", default=None, help="variant name; results are saved as <refinement>+<tag>")
p.add_argument("--all_samples", action="store_true", help="refine and validate on all samples, also misclassified ones")
p.add_argument("--model_file", default=None, help="use these weights instead of the scenario default")
p.add_argument("--save_outputs", action="store_true", help="also save the test logits")
p.add_argument("--num_workers", type=int, default=8, help="data loader workers")
a = p.parse_args()
torch.set_num_threads(a.threads)
os.makedirs(a.out, exist_ok=True)
t = time.time()
res = run_experiment(scenario_name=a.scenario, data_root=a.data_root, refinement=a.refinement,
                     n_reps=a.n_reps, n_samples=a.n_samples, poisoning_strategy=a.poisoning, num_workers=a.num_workers, refiner_kwargs=a.refiner_kwargs, correct_only=not a.all_samples,
                     **({"model_file_path": a.model_file} if a.model_file else {}),
                     **({"explanation_type": a.explanation_type} if a.explanation_type else {}))
for sel in select_by_slack(res, a.refinement, 0.05):
    print(f"Selected (5% slack): rep {sel['rep']}, test top-1 {float(sel['top1'][0]):.4f}")
method = a.refinement + (f"+{a.tag}" if a.tag else "")
if a.save_outputs:
    from selection import REFINEMENT_HP
    hp = REFINEMENT_HP.get(a.refinement, (None,))[0]
    arrays = {"true": np.asarray(res[0]["true"])}
    for r in res:
        arrays[f"rep{r['rep']}_{r.get(hp, 'none')}"] = np.asarray(r["output"], dtype=np.float32)
    np.savez_compressed(os.path.join(a.out, f"{a.scenario}_{method}_{a.poisoning}_n{a.n_samples}_r{a.n_reps}_outputs.npz"), **arrays)
for r in res:
    for k in ("output", "true", "predicted"):
        r.pop(k, None)
fn = os.path.join(a.out, f"{a.scenario}_{method}_{a.poisoning}_n{a.n_samples}_r{a.n_reps}.pkl")
pkl.dump(res, open(fn, "wb"))
print("saved", fn, "took %.1f min" % ((time.time() - t) / 60))
