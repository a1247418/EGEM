"""Per-layer separability of clean vs. CH-poisoned versions of the same training images: R^2 = 1 - d_within /
d_total with cosine distances, N images per draw, averaged over draws."""
import argparse, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image

from CH_datasets.utils import get_artifact_path
from CH_datasets.datasets.splits import SingletonIndexStorage
from models.model_loading import load_model

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=10)
p.add_argument("--draws", type=int, default=20)
p.add_argument("--out", default=os.path.join(ROOT, "repro", "results", "layer_separability.csv"))
a = p.parse_args()
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(16)


def r2(A, B):
    """A, B: [N, D] clean and poisoned representations."""
    X = torch.nn.functional.normalize(torch.cat([A, B]).flatten(1).double(), dim=1)
    D = 1 - X @ X.T  # cosine distances
    N = len(A)
    within = (D[:N, :N].sum() + D[N:, N:].sum()) / (2 * N ** 2)
    total = D.sum() / (4 * N ** 2)
    return float(1 - within / total)


def layer_outputs(modules, x, names):
    outs = []
    with torch.no_grad():
        for m, n in zip(modules, names):
            x = m(x)
            if n is not None:
                outs.append((n, x))
    return outs


rows = []
# MNIST-8 (grayscale model, 3-pixel artifact)
mnist = torchvision.datasets.MNIST(os.path.expanduser("~/EGEM_work/data"), train=True)
feats = load_model("mnistnet", os.path.join(ROOT, "model_weights", "mnist.model"),
                   out_classes=list(range(10)), device="cpu").model.features
counts = {}
names = []
for m in feats:
    t = type(m).__name__
    counts[t] = counts.get(t, 0) + 1
    names.append(f"{t}_{counts[t]}")
for d in range(a.draws):
    idx = np.random.choice(len(mnist), a.n, replace=False)
    x = mnist.data[idx].float()[:, None] / 255.
    xp = x.clone(); xp[:, :, 1:3, 2] = 1; xp[:, :, 2, 1:3] = 1
    for (n, A), (_, B) in zip(layer_outputs(feats, x, names), layer_outputs(feats, xp, names)):
        rows.append(dict(task="MNIST", model="mnist", layer=n, draw=d, r2=r2(A, B)))

# ISIC (VGG-16, three trained models)
cache = torch.load(os.path.expanduser("~/EGEM_work/data/isic/cache_224_uint8.pt"))
clean = np.asarray(SingletonIndexStorage().get_sample_indicators("isic")["train"]["clean"])
patch = Image.open(get_artifact_path("blue_patch.png")).convert("RGBA")
MEAN = torch.tensor([0.6678, 0.5296, 0.5242])[:, None, None]
STD = torch.tensor([0.1282, 0.1417, 0.1521])[:, None, None]


def paste(u8):
    img = Image.fromarray(u8.permute(1, 2, 0).numpy()).convert("RGBA")
    img.paste(patch, (0, 0), patch)
    return torch.from_numpy(np.array(img.convert("RGB"))).permute(2, 0, 1)


blocks_end = [4, 9, 16, 23, 30]  # max-pool closing each VGG block
for model_file in ["isic_vgg16.model", "isic_vgg16_seed1.model", "isic_vgg16_seed2.model"]:
    path = os.path.join(ROOT, "model_weights", model_file)
    if not os.path.exists(path):
        continue
    feats = load_model("vgg16_isic", path, device="cpu").model.features
    names, counts = [], {}
    for i, m in enumerate(feats):
        if i <= 30:
            names.append(f"Block_{blocks_end.index(i) + 1}" if i in blocks_end else None)
        else:
            t = type(m).__name__
            counts[t] = counts.get(t, 0) + 1
            names.append(f"{t}_{counts[t]}")
    for d in range(a.draws):
        idx = np.random.choice(clean, a.n, replace=False)
        u8 = cache["train"]["x"][idx]
        x = (u8.float() / 255 - MEAN) / STD
        xp = (torch.stack([paste(u) for u in u8]).float() / 255 - MEAN) / STD
        for (n, A), (_, B) in zip(layer_outputs(feats, x, names), layer_outputs(feats, xp, names)):
            rows.append(dict(task="ISIC", model=model_file, layer=n, draw=d, r2=r2(A, B)))
    print("done", model_file, flush=True)

df = pd.DataFrame(rows)
df.to_csv(a.out, index=False)
summary = df.groupby(["task", "model", "layer"], sort=False).r2.mean().round(3)
print(summary.to_string())
