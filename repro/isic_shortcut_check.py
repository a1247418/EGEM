"""Is the ISIC patch effect a color shortcut or a shape/occlusion effect? Pastes the test patch, and the same
patch shape filled with skin color or gray, onto 600 clean test images and reports accuracy and the fraction
of 'nevus' predictions. EGEM can only remove features that clean images do not use: a color-specific
shortcut is removable, a shape/edge one (shared with lesion borders) is not."""
import argparse, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import torch
from PIL import Image
from CH_datasets.utils import get_artifact_path
from models.model_loading import load_model

p = argparse.ArgumentParser()
p.add_argument("--model", default=os.path.join(ROOT, "model_weights", "isic_vgg16.model"))
p.add_argument("--cache", default=os.path.expanduser("~/EGEM_work/data/isic/cache_224_uint8.pt"))
a = p.parse_args()
torch.manual_seed(0); torch.set_num_threads(16)
d = torch.load(a.cache)
idx = d["test_clean_idx"][torch.randperm(len(d["test_clean_idx"]))[:600]]
x, y = d["test"]["x"][idx], d["test"]["y"][idx]
patch = Image.open(get_artifact_path("blue_patch.png")).convert("RGBA")
MEAN = torch.tensor([0.6678, 0.5296, 0.5242])[:, None, None]
STD = torch.tensor([0.1282, 0.1417, 0.1521])[:, None, None]


def paste(u8, color=None):
    img = Image.fromarray(u8.permute(1, 2, 0).numpy()).convert("RGBA")
    p_ = patch
    if color is not None:
        p_ = Image.fromarray(np.dstack([np.full((224, 224), c, np.uint8) for c in color]
                                       + [np.asarray(patch)[..., 3]]), "RGBA")
    img.paste(p_, (0, 0), p_)
    return torch.from_numpy(np.array(img.convert("RGB"))).permute(2, 0, 1)


m = load_model("vgg16_isic", a.model, device="cpu").model
skin = tuple(int(v) for v in (MEAN.flatten() * 255))  # dataset-mean color
for name, xs in [("clean", x), ("cyan patch (test poisoning)", torch.stack([paste(u) for u in x])),
                 ("same shape, skin color", torch.stack([paste(u, skin) for u in x])),
                 ("same shape, gray", torch.stack([paste(u, (128, 128, 128)) for u in x]))]:
    with torch.no_grad():
        pred = torch.cat([m(((xs[i:i + 50].float() / 255 - MEAN) / STD)).argmax(1) for i in range(0, len(xs), 50)])
    print(f"{name:28s} acc {(pred == y).float().mean():.3f}   predicted nevus: {(pred == 1).float().mean():.3f}")
print(f"true nevus fraction: {(y == 1).float().mean():.3f}")
