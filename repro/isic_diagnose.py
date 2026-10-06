"""Why does EGEM not remove the ISIC patch effect? For every refined layer, measure which fraction of
(a) the patch-induced activation change  d = a(x + patch) - a(x)  and
(b) the clean activation energy  a(x)
survives the multipliers of EGEM / PCA-EGEM. If (a) and (b) survive equally, pruning can't separate them.

Uses the 224px cache from train_isic.py: refinement images are the clean training images (patch images
432-609 removed), 700 per class with oversampling, as in run.py. The patch is pasted on the 224px image, as
the CH_datasets poisoner does (it runs after Resize/CenterCrop).
"""
import argparse, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from CH_datasets.utils import get_artifact_path
from CH_datasets.datasets.splits import SingletonIndexStorage
from experiment_config import get_experiment_config
from models.model_loading import load_model
from refinement.helpers import get_module_by_name
from refinement.refiner import EGEMRefiner

p = argparse.ArgumentParser()
p.add_argument("--cache", default=os.path.expanduser("~/EGEM_work/data/isic/cache_224_uint8.pt"))
p.add_argument("--alphas", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
p.add_argument("--n_test", type=int, default=256)
p.add_argument("--refine_on", choices=["train", "test_half"], default="train",
               help="test_half: diagnostic only - refine on clean test images the model has not memorized, "
                    "evaluate on the other half")
p.add_argument("--model", default=os.path.join(ROOT, "model_weights", "isic_vgg16.model"))
p.add_argument("--out", default=os.path.join(ROOT, "repro", "results", "isic_diagnosis.csv"))
a = p.parse_args()
torch.manual_seed(0); np.random.seed(0)
dev = "cuda"
MEAN = torch.tensor([0.6678, 0.5296, 0.5242])[:, None, None]
STD = torch.tensor([0.1282, 0.1417, 0.1521])[:, None, None]
norm = lambda u8: (u8.float() / 255 - MEAN) / STD

data = torch.load(a.cache)
# refinement set: clean training images, 700 per class (oversampled), as run.py does
clean_idx = np.asarray(SingletonIndexStorage().get_sample_indicators("isic")["train"]["clean"])
ytr = data["train"]["y"].numpy()
ref_idx = np.concatenate([np.random.choice(clean_idx[ytr[clean_idx] == c], 700, replace=(ytr[clean_idx] == c).sum() < 700)
                          for c in range(8)])
X_ref, Y_ref = data["train"]["x"][ref_idx], data["train"]["y"][ref_idx]
test_pool = data["test_clean_idx"][torch.randperm(len(data["test_clean_idx"]))]
if a.refine_on == "test_half":  # diagnostic: unseen images for refinement, the other half for evaluation
    half, test_pool = test_pool[:len(test_pool) // 2].numpy(), test_pool[len(test_pool) // 2:]
    yte_all = data["test"]["y"].numpy()
    ref_idx = np.concatenate([np.random.choice(half[yte_all[half] == c], 700, replace=True)
                              for c in range(8) if (yte_all[half] == c).any()])
    X_ref, Y_ref = data["test"]["x"][ref_idx], data["test"]["y"][ref_idx]
refine_loader = DataLoader(TensorDataset(norm(X_ref), Y_ref), batch_size=128)

# test images, clean and with the patch pasted (same operation as ISICPoisoner after the crop)
patch = Image.open(get_artifact_path("blue_patch.png")).convert("RGBA")
def paste(u8):
    img = Image.fromarray(u8.permute(1, 2, 0).numpy()).convert("RGBA")
    img.paste(patch, (0, 0), patch)
    return torch.from_numpy(np.asarray(img.convert("RGB"))).permute(2, 0, 1)
te = test_pool[:a.n_test]
x_clean = norm(data["test"]["x"][te]); y_test = data["test"]["y"][te]
x_pois = norm(torch.stack([paste(u) for u in data["test"]["x"][te]]))

model = load_model("vgg16_isic", a.model, device=dev)
net = model.model
layers = get_experiment_config("isic-1", "egem")["layer_names"]


@torch.no_grad()
def layer_inputs(x):
    """Inputs of the refined layers, as [n_rows, n_channels] (conv: one row per position)."""
    store = {}
    hooks = [get_module_by_name(net, l).register_forward_hook(
        lambda m, i, o, l=l: store.setdefault(l, []).append(i[0].detach())) for l in layers]
    preds = []
    for k in range(0, len(x), 64):
        preds.append(net(x[k:k + 64].to(dev)).argmax(1).cpu())
    for h in hooks:
        h.remove()
    out = {}
    for l, v in store.items():
        t = torch.cat(v)
        out[l] = t.permute(0, 2, 3, 1).reshape(-1, t.shape[1]) if t.dim() == 4 else t
        out[l + ":sum"] = t.sum(dim=[-2, -1]) if t.dim() == 4 else t  # per-image channel sums
    return out, torch.cat(preds)


A_clean, pred_c = layer_inputs(x_clean)
A_pois, pred_p = layer_inputs(x_pois)
print(f"unrefined accuracy on these {a.n_test} test images: clean {(pred_c == y_test).float().mean():.3f}, "
      f"patched {(pred_p == y_test).float().mean():.3f}", flush=True)

rows = []
for method in ["egem", "pcaegem", "pcaegem+sum"]:
    for alpha in a.alphas:
        ref = EGEMRefiner(net, layer_names=list(layers), device=dev, alpha=alpha, do_pca=method != "egem",
                          spatial_sum=method == "pcaegem+sum")
        ref.train_refinement(refine_loader)
        for l in layers:
            mod = ref.mods[l]
            c = (mod[0] if isinstance(mod, torch.nn.Sequential) else mod).multiplier.float()
            key = l + ":sum" if method == "pcaegem+sum" else l
            d = (A_pois[key] - A_clean[key]).float()
            s = A_clean[key].float()
            if method != "egem":
                V, m = mod[0].pca_V.float(), mod[0].pca_Xm.float()
                d_b, s_b = d @ V.T, (s - m) @ V.T  # coordinates in the PCA basis (outside the span: removed)
                kept_d = ((c * d_b) ** 2).sum() / (d ** 2).sum()
                kept_s = ((c * s_b) ** 2).sum() / ((s - m) ** 2).sum()
            else:
                kept_d = ((c * d) ** 2).sum() / (d ** 2).sum()
                kept_s = ((c * s) ** 2).sum() / (s ** 2).sum()
            rows.append(dict(method=method, alpha=alpha, layer=l, dims=len(c), mean_c=float(c.mean()),
                             kept_patch=float(kept_d), kept_clean=float(kept_s),
                             patch_energy_share=float((d ** 2).sum() / ((s ** 2).sum() + 1e-12))))
        ref.refine()
        _, pc = layer_inputs(x_clean); _, pp = layer_inputs(x_pois)
        ref.unrefine()
        print(f"{method} alpha={alpha}: clean {(pc == y_test).float().mean():.3f}, patched {(pp == y_test).float().mean():.3f}",
              flush=True)
df = pd.DataFrame(rows)
df.to_csv(a.out, index=False)
pd.set_option("display.width", 200)
print(df.round(3).to_string(index=False))
