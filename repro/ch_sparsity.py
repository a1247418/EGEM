"""Sparsity ||a_CH - a||_2 / ||a_CH - a||_1 of the representation change induced by each MNIST CH feature,
at the inputs of the refined layers."""
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import torch, torchvision
from models.model_loading import load_model
from CH_datasets.scenario_examples import MNIST_RGB_POISONERS

ds = torchvision.datasets.MNIST(os.path.expanduser("~/EGEM_work/data"), train=False)
x = ds.data[ds.targets == 8][:100].float()[:, None].repeat(1, 3, 1, 1) / 255.
layers = {"MaxPool2d": 3, "Linear_1": 7, "Linear_2": 9}  # inputs of the refined layers features.{3,7,9}
print(f"{'':10s}" + "".join(f"{k:>11s}" for k in layers))
rows = {}
for v, P in MNIST_RGB_POISONERS.items():
    poison = P(p=1.0)
    x_ch = torch.stack([poison._poison(xi.clone()) for xi in x])
    feats = load_model("mnistnetRGB", f"model_weights/mnist-rgb-{v}.model", out_classes=list(range(10)), device="cpu").model.features
    out = []
    with torch.no_grad():
        for idx in layers.values():
            head = torch.nn.Sequential(*feats[:idx])
            a, a_ch = head(x).flatten(1), head(x_ch).flatten(1)
            d = a_ch - a
            out.append((d.norm(p=2, dim=1) / d.norm(p=1, dim=1).clamp_min(1e-12)).mean().item())
    print(f"{v:10s}" + "".join(f"{s:11.3f}" for s in out))
    rows[v] = out

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(4.6, 3.2))
for (v, out), c in zip(rows.items(), ["#2a78d6", "#1baf7a", "#eda100", "#eb6834"]):
    ax.plot(list(layers), out, color=c, lw=2, marker="o", ms=4, label=f"MNIST-{v}")
ax.set_ylabel("Sparsity"); ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False); ax.yaxis.grid(True, color="#e4e3df", linewidth=0.6)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "repro", "figures", "ch_sparsity.png"), dpi=150)
