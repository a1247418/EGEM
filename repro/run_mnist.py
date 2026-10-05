"""Driver for the MNIST-8 reproduction (Linhardt et al., EGEM). Run from repo root."""
import argparse, os, sys, time, pickle as pkl
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
# src mixes "x", "src.x" and bare "cxai" imports, so all three roots are needed
sys.path[:0] = [os.path.join(ROOT, "src"), ROOT, os.path.join(ROOT, "src", "refinement", "decomposition")]
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
a = p.parse_args()
torch.set_num_threads(a.threads)
os.makedirs(a.out, exist_ok=True)
t = time.time()
res = run_experiment(scenario_name=a.scenario, data_root=a.data_root, refinement=a.refinement,
                     n_reps=a.n_reps, n_samples=a.n_samples, poisoning_strategy=a.poisoning,
                     **({"explanation_type": a.explanation_type} if a.explanation_type else {}))
for sel in select_by_slack(res, a.refinement, 0.05):
    print(f"Selected (5% slack): rep {sel['rep']}, test top-1 {float(sel['top1'][0]):.4f}")
for r in res:
    for k in ("output", "true", "predicted"):
        r.pop(k, None)
fn = os.path.join(a.out, f"{a.scenario}_{a.refinement}_{a.poisoning}_n{a.n_samples}_r{a.n_reps}.pkl")
pkl.dump(res, open(fn, "wb"))
print("saved", fn, "took %.1f min" % ((time.time() - t) / 60))
