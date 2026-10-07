"""Sparsity ||a_CH - a||_2 / ||a_CH - a||_1 of the representation change induced by each MNIST CH feature, at the
inputs of the refined layers, averaged over the first 100 training images of class 8. Conv inputs are averaged
over channels and rows."""
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import pandas as pd
import torch, torchvision
from models.model_loading import load_model
from CH_datasets.scenario_examples import MNIST_RGB_POISONERS

torch.manual_seed(0)
MEAN, STD = 0.1307, 0.3081
ds = torchvision.datasets.MNIST(os.path.expanduser("~/EGEM_work/data"), train=True)
x = ds.data[ds.targets == 8][:100].float()[:, None].repeat(1, 3, 1, 1) / 255.
layers = {"MaxPool2d": 2, "Linear_1": 7, "Linear_2": 9}  # inputs of the refined layers features.{2,7,9}
rows = []
for v in ["artifact", "blur", "color", "remove"]:
    poison = MNIST_RGB_POISONERS[v](p=1.0)
    x_ch = torch.stack([poison._poison(xi.clone()) for xi in x])
    feats = load_model("mnistnetRGB", os.path.join(ROOT, "model_weights", f"mnist-rgb-{v}.model"),
                       out_classes=list(range(10)), device="cpu").model.features
    with torch.no_grad():
        for name, idx in layers.items():
            head = torch.nn.Sequential(*feats[:idx])
            a, a_ch = head((x - MEAN) / STD), head((x_ch - MEAN) / STD)
            if a.dim() == 4:
                a, a_ch = a.mean(dim=[1, 2]), a_ch.mean(dim=[1, 2])
            d = (a_ch - a).flatten(1)
            d = d[d.abs().sum(1) > 0]
            rows.append(dict(Sparsity=(d.norm(p=2, dim=1) / d.norm(p=1, dim=1)).mean().item(), Layer=name,
                             Dataset=f"MNIST-{v}"))
df = pd.DataFrame(rows)
print(df.pivot(index="Dataset", columns="Layer", values="Sparsity")[list(layers)].round(3).to_string())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white"); sns.set_context("talk", font_scale=.8)
palette = [sns.color_palette()[i] for i in (0, 2)] + sns.color_palette("rocket")[2:6][::-1]
fig, ax = plt.subplots(figsize=(6.4, 4.8))
sns.lineplot(df, x="Layer", y="Sparsity", hue="Dataset", palette=palette[:4], ax=ax)
ax.set_xlabel(None); ax.legend(title=None)
fig.savefig(os.path.join(ROOT, "repro", "figures", "ch_sparsity.png"), dpi=150, bbox_inches="tight")
