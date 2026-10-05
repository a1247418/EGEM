"""Paper Sec. 6 (CelebA, Fig. 9): blond-hair recall per attribute subgroup before/after PCA-EGEM.

1. Model: `vgg16_celeba` (weights from git history, see DATA.md), blond-vs-not on 218x178 images.
2. Refinement data ("user-verified"): 200 images per class from the validation split that are correctly
   predicted and whose LRP relevance lies >= 75% (absolute) inside a hair mask (paper Supp. F.1, Fig. F.12;
   the mask below is read off that figure and therefore approximate).
3. PCA-EGEM, refining the activations after every VGG block and after every ReLU outside them. The paper
   uses alpha = 0.01, which 5% slack gave for its model. Here alpha is selected with the same rule: refine
   on 80% of the verified images and take the strongest alpha whose accuracy on the other 20% is at most
   `--slack` below the original's (1.0, since the verified images are correctly predicted).
4. For every attribute, up to 5000 test images with that attribute; recall of 'Blond_Hair' before/after.
"""
import argparse, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision.transforms import Compose, Normalize, ToTensor

from experiment_config import get_experiment_config
from models.model_loading import load_model
from refinement.explainer import Explainer
from refinement.refiner import EGEMRefiner

p = argparse.ArgumentParser()
p.add_argument("--root", default=os.path.expanduser("~/EGEM_work/data/celeba_root"))
p.add_argument("--n_per_class", type=int, default=200)
p.add_argument("--alphas", type=float, nargs="+", default=[0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
p.add_argument("--slack", type=float, default=0.05)
p.add_argument("--mask_frac", type=float, default=0.75)
p.add_argument("--n_subgroup", type=int, default=5000)
p.add_argument("--max_test", type=int, default=None, help="evaluate on fewer test images (smoke test)")
p.add_argument("--out", default=os.path.join(ROOT, "repro", "results", "celeba_recall.csv"))
p.add_argument("--workers", type=int, default=8)
a = p.parse_args()
torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
BLOND = 9
LAYERS = get_experiment_config("celeba", "pcaegem")["layer_names"]

# Hair mask (rows x cols of the 218x178 aligned images), read off Fig. F.12
MASK = torch.zeros(218, 178, dtype=torch.bool)
MASK[6:31, 30:148] = True
MASK[31:166, 5:178] = True

T = Compose([ToTensor(), Normalize([0.5064, 0.4258, 0.3832], [0.2660, 0.2452, 0.2414])])
celeba = lambda split: torchvision.datasets.CelebA(a.root, split=split, target_type="attr", transform=T)
model = load_model("vgg16_celeba", os.path.join(ROOT, "model_weights", "celeba_vgg16.model"), device=dev)
net = model.model


@torch.no_grad()
def predict(ds):
    loader = DataLoader(ds, batch_size=256, num_workers=a.workers)
    return torch.cat([net(x.to(dev)).argmax(1).cpu() for x, _ in loader])


# 1. test accuracy
test = celeba("test")
if a.max_test:
    test = Subset(test, range(a.max_test))
attr = (test.dataset.attr[:a.max_test] if a.max_test else test.attr).numpy().astype(bool)
pred_orig = predict(test).numpy().astype(bool)
print(f"original test accuracy (blond vs not): {(pred_orig == attr[:, BLOND]).mean():.4f} (paper: 0.93)", flush=True)

# 2. user-verified refinement data
explainer = Explainer(net, n_classes=2, explanation_type="epsilon_alpha2_beta1_flat")
valid = celeba("valid")
xs, ys, n = [], [], {0: 0, 1: 0}
checked = 0
for x, at in DataLoader(valid, batch_size=64, shuffle=True, num_workers=a.workers):
    y = at[:, BLOND]
    with torch.no_grad():
        ok = net(x.to(dev)).argmax(1).cpu() == y
    for xi, yi in zip(x[ok], y[ok]):
        if n[int(yi)] >= a.n_per_class:
            continue
        checked += 1
        r = explainer.explain(xi[None].to(dev).clone(), int(yi))[0].detach().abs().sum(0).cpu()
        if r[MASK].sum() >= a.mask_frac * r.sum():
            xs.append(xi); ys.append(int(yi)); n[int(yi)] += 1
    if min(n.values()) >= a.n_per_class:
        break
print(f"verified refinement samples per class: {n} ({checked} explained)", flush=True)

# 3. PCA-EGEM, alpha selected by slack on held-out verified images
perm = torch.randperm(len(ys))
X, Y = torch.stack(xs)[perm], torch.tensor(ys)[perm]
n_val = int(np.ceil(0.2 * len(Y)))
refine_loader = DataLoader(TensorDataset(X[n_val:], Y[n_val:]), batch_size=64)
val_ds = TensorDataset(X[:n_val], Y[:n_val])
chosen, sweep = None, []
for alpha in sorted(a.alphas):  # strongest (smallest alpha) first
    refiner = EGEMRefiner(net, layer_names=LAYERS, device=dev, alpha=alpha, do_pca=True)
    refiner.train_refinement(refine_loader)
    refiner.refine()
    val_acc = (predict(val_ds).numpy() == Y[:n_val].numpy()).mean()
    sweep.append((alpha, val_acc))
    print(f"alpha {alpha}: verified-val accuracy {val_acc:.3f}", flush=True)
    if chosen is None and val_acc >= 1.0 - a.slack:
        chosen = alpha
        pred_ref = predict(test).numpy().astype(bool)
    refiner.unrefine()
    if chosen is not None:
        break
print(f"chosen alpha (slack {a.slack}): {chosen}", flush=True)
print(f"refined test accuracy: {(pred_ref == attr[:, BLOND]).mean():.4f}", flush=True)

# 4. recall of blond hair per attribute subgroup
names = test.dataset.attr_names if a.max_test else test.attr_names
rows = []
for j, name in enumerate(names[:40]):
    idx = np.where(attr[:, j])[0]
    if len(idx) > a.n_subgroup:
        idx = np.random.choice(idx, a.n_subgroup, replace=False)
    blond = idx[attr[idx, BLOND]]
    if len(blond) == 0:
        continue
    rows.append(dict(attribute=name, n=len(idx), n_blond=len(blond),
                     recall_orig=pred_orig[blond].mean(), recall_refined=pred_ref[blond].mean()))
blond_all = np.where(attr[:, BLOND])[0]
rows.append(dict(attribute="(all)", n=len(attr), n_blond=len(blond_all),
                 recall_orig=pred_orig[blond_all].mean(), recall_refined=pred_ref[blond_all].mean()))
df = pd.DataFrame(rows)
df.to_csv(a.out, index=False)
print(df.round(3).to_string(index=False))
print("saved", a.out)
