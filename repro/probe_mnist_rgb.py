"""Probe the historical MNIST-RGB models (paper Sec. 5) with candidate re-implementations of the
CH poisoners, which are missing from the repo. A candidate that makes a model call non-8s an '8'
is likely close to the feature that model was trained with."""
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [os.path.join(ROOT, "src"), ROOT]
import torch, torchvision
import torchvision.transforms.functional as TF
from models.model_loading import load_model

torch.manual_seed(0)
ds = torchvision.datasets.MNIST(os.path.expanduser("~/EGEM_work/data"), train=False)
x = ds.data[:2000].float()[:, None] / 255.
y = ds.targets[:2000]
x = x.repeat(1, 3, 1, 1)  # gray -> RGB

def artifact(x):
    x = x.clone(); x[:, :, 1:3, 2:3] = 1; x[:, :, 2:3, 1:3] = 1; return x
def blur(k, s):
    return lambda x: TF.gaussian_blur(x, k, s)
def remove(frac):
    def f(x):
        x = x.clone(); x[:, :, int(28 * (1 - frac)):] = 0; return x
    return f
def color(rgb):
    return lambda x: x * torch.tensor(rgb)[None, :, None, None]

cands = {"clean": lambda x: x, "artifact": artifact,
         "blur k3 s1": blur(3, 1.), "blur k5 s1.5": blur(5, 1.5), "blur k7 s2": blur(7, 2.),
         "remove 1/3": remove(1 / 3), "remove 1/2": remove(1 / 2),
         "red": color([1., 0, 0]), "green": color([0, 1., 0]), "blue": color([0, 0, 1.]),
         "yellow": color([1., 1., 0]), "magenta": color([1., 0, 1.]), "cyan": color([0, 1., 1.])}
non8 = y != 8
print(f"{'model':10s} " + " ".join(f"{k:>12s}" for k in cands))
for v in ["artifact", "blur", "color", "remove"]:
    m = load_model("mnistnetRGB", f"model_weights/mnist-rgb-{v}.model", out_classes=list(range(10)), device="cpu")
    accs, p8 = [], []
    with torch.no_grad():
        for k, f in cands.items():
            pred = m(f(x)).argmax(1)
            accs.append((pred == y).float().mean().item()); p8.append((pred[non8] == 8).float().mean().item())
    print(f"{v:10s} acc " + " ".join(f"{a:12.3f}" for a in accs))
    print(f"{'':10s} p8  " + " ".join(f"{a:12.3f}" for a in p8))
