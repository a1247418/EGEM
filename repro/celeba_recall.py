"""CelebA blond-hair classifier: refines it with PCA-EGEM on LRP-verified images and reports precision and
recall per attribute subgroup for the original and the refined model, each with and without the bottom 50 rows
(the collar region) covered by a wall image. Also saves LRP heatmaps of blond Wearing_Hat test images that the
refinement turns from rejected to detected, and the attribute correlation matrix.

Verified images are correctly predicted and have >= 75% of their absolute LRP relevance inside the hair mask.
alpha is the strongest value whose accuracy on held-out verified images drops by at most `--slack`."""
import argparse, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision.transforms import Compose, Normalize, ToTensor
from PIL import Image
from zennit.attribution import Gradient
from zennit.composites import EpsilonGammaBox
from zennit.torchvision import VGGCanonizer

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
p.add_argument("--n_heatmaps", type=int, default=5)
p.add_argument("--max_test", type=int, default=None, help="evaluate on fewer test images (smoke test)")
p.add_argument("--out", default=os.path.join(ROOT, "repro", "results", "celeba_recall.csv"))
p.add_argument("--workers", type=int, default=8)
a = p.parse_args()
torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
BLOND = 9
LAYERS = get_experiment_config("celeba", "pcaegem")["layer_names"]
MEAN, STD = [0.5064, 0.4258, 0.3832], [0.2660, 0.2452, 0.2414]
T = Compose([ToTensor(), Normalize(MEAN, STD)])
celeba = lambda split: torchvision.datasets.CelebA(a.root, split=split, target_type=["attr", "landmarks"], transform=T)


def hair_mask(landmarks):
    """Hair region: above the eyes + 55 px, without the top corners and a 5 px border."""
    m = torch.zeros(218, 178, dtype=torch.bool)
    m[:int(min(landmarks[1], landmarks[3])) + 55, :] = True
    m[:30, :30] = False; m[:30, -30:] = False
    m[:5, :] = False; m[:, :5] = False; m[:, -5:] = False
    return m


from CH_datasets.utils import get_artifact_path
WALL = Normalize(MEAN, STD)(ToTensor()(Image.open(get_artifact_path("wall_cut.png")).convert("RGB")))  # 50 x 178
model = load_model("vgg16_celeba", os.path.join(ROOT, "model_weights", "celeba_vgg16.model"), device=dev)
net = model.model


@torch.no_grad()
def scores(ds, wall=False):
    """Blond-minus-not logit for every image."""
    out = []
    for x, _ in DataLoader(ds, batch_size=256, num_workers=a.workers):
        if wall:
            x = x.clone(); x[:, :, -50:, :] = WALL
        o = net(x.to(dev)).cpu()
        out.append(o[:, 1] - o[:, 0])
    return torch.cat(out)


# 1. test set and original model
test = celeba("test")
attr_all = test.attr.numpy().astype(bool)
names = list(test.attr_names[:40])
if a.max_test:
    test = Subset(test, range(a.max_test))
attr = attr_all[:len(test)]
s_orig = scores(test)
pred_orig = (s_orig > 0).numpy()
print(f"original test accuracy (blond vs not): {(pred_orig == attr[:, BLOND]).mean():.4f}", flush=True)
pred_wall = (scores(test, wall=True) > 0).numpy()
HAT = names.index("Wearing_Hat")
print(f"original + wall test accuracy: {(pred_wall == attr[:, BLOND]).mean():.4f}", flush=True)

# 2. user-verified refinement data
explainer = Explainer(net, n_classes=2, explanation_type="epsilon_alpha2_beta1_flat")
xs, ys, n, checked = [], [], {0: 0, 1: 0}, 0
for x, (at, lm) in DataLoader(celeba("valid"), batch_size=64, shuffle=True, num_workers=a.workers):
    y = at[:, BLOND]
    with torch.no_grad():
        ok = net(x.to(dev)).argmax(1).cpu() == y
    for xi, yi, li in zip(x[ok], y[ok], lm[ok]):
        if n[int(yi)] >= a.n_per_class:
            continue
        checked += 1
        r = explainer.explain(xi[None].to(dev).clone(), int(yi))[0].detach().abs().sum(0).cpu()
        if r[hair_mask(li)].sum() >= a.mask_frac * r.sum():
            xs.append(xi); ys.append(int(yi)); n[int(yi)] += 1
    if min(n.values()) >= a.n_per_class:
        break
print(f"verified refinement samples per class: {n} ({checked} explained)", flush=True)


def lrp_maps(idx):
    """LRP (epsilon / gamma=0.1 / pixel box) for blond against not blond, summed over color channels."""
    box = lambda v: Normalize(MEAN, STD)(torch.full((3, 218, 178), float(v)))[None].to(dev)
    composite = EpsilonGammaBox(low=box(0), high=box(1), gamma=0.1, canonizers=[VGGCanonizer()])
    out = []
    with Gradient(net, composite) as attributor:
        for i in idx:
            x = test[i][0][None].to(dev).clone()
            with torch.no_grad():
                z = net(x)
            _, R = attributor(x, z * torch.tensor([[-1.0, 1.0]], device=dev))  # relevance of z_blond and -z_not
            out.append(R[0].sum(0).detach().cpu())
    return torch.stack(out)


# 3. PCA-EGEM, alpha selected by slack on held-out verified images
perm = torch.randperm(len(ys))
X, Y = torch.stack(xs)[perm], torch.tensor(ys)[perm]
n_val = int(np.ceil(0.2 * len(Y)))
refine_loader = DataLoader(TensorDataset(X[n_val:], Y[n_val:]), batch_size=64)
val_ds = TensorDataset(X[:n_val], Y[:n_val])
chosen = None
for alpha in sorted(a.alphas):  # strongest (smallest alpha) first
    refiner = EGEMRefiner(net, layer_names=list(LAYERS), device=dev, alpha=alpha, do_pca=True)
    refiner.train_refinement(refine_loader)
    refiner.refine()
    val_acc = ((scores(val_ds) > 0).numpy() == Y[:n_val].numpy()).mean()
    print(f"alpha {alpha}: verified-val accuracy {val_acc:.3f}", flush=True)
    if val_acc >= 1.0 - a.slack or alpha == max(a.alphas):
        chosen = alpha
        s_ref = scores(test)
        # blond images that refinement turns from rejected to detected, largest score increase first;
        # LRP for 'blond' after and before refinement
        flipped = torch.from_numpy(attr[:, BLOND]) & (s_orig < 0) & (s_ref > 0)
        hat = flipped & torch.from_numpy(attr[:, HAT])
        if hat.sum() >= a.n_heatmaps:  # prefer images with a hat, as in the paper
            flipped = hat
        gain = (s_ref - s_orig).masked_fill(~flipped, -float("inf"))
        top = torch.argsort(gain, descending=True)[:min(a.n_heatmaps, int(flipped.sum()))].tolist()
        R_ref = lrp_maps(top)
        s_ref_wall = scores(test, wall=True)
        refiner.unrefine()
        R_orig = lrp_maps(top)
        break
    refiner.unrefine()
pred_ref = (s_ref > 0).numpy()
pred_ref_wall = (s_ref_wall > 0).numpy()
print(f"chosen alpha (slack {a.slack}): {chosen}", flush=True)
print(f"refined test accuracy: {(pred_ref == attr[:, BLOND]).mean():.4f}", flush=True)
base = os.path.splitext(a.out)[0]
np.savez_compressed(base + "_heatmaps.npz", idx=np.array(top),
                    images=torch.stack([test[i][0] for i in top]).numpy(), R_orig=R_orig.numpy(),
                    R_refined=R_ref.numpy(), score_orig=s_orig[top].numpy(), score_refined=s_ref[top].numpy(),
                    wall_examples=torch.stack([test[i][0] for i in range(5)]).numpy(), wall=WALL.numpy(),
                    mean=np.array(MEAN), std=np.array(STD))

# 4. precision and recall of blond hair per attribute subgroup
rows = []
for j, name in enumerate(names + ["(all)"]):
    idx = np.arange(len(attr)) if name == "(all)" else np.where(attr[:, j])[0]
    if name != "(all)" and len(idx) > a.n_subgroup:
        idx = np.random.choice(idx, a.n_subgroup, replace=False)
    blond = attr[idx, BLOND]
    row = dict(attribute=name, n=len(idx), n_blond=int(blond.sum()))
    for key, pred in [("orig", pred_orig), ("refined", pred_ref), ("wall", pred_wall), ("refined_wall", pred_ref_wall)]:
        pr = pred[idx]
        row[f"recall_{key}"] = pr[blond].mean() if blond.any() else np.nan
        row[f"precision_{key}"] = blond[pr].mean() if pr.any() else np.nan
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(a.out, index=False)
print(df.round(3).to_string(index=False))

# attribute correlations in the training split
train_attr = torchvision.datasets.CelebA(a.root, split="train", target_type="attr").attr.float().numpy()
pd.DataFrame(np.corrcoef(train_attr.T), index=names, columns=names).to_csv(base + "_attr_corr.csv")
print("saved", a.out)
