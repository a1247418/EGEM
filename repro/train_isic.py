"""Step 1 (CPU): `--build_cache` decodes all JPEGs once into 224x224 uint8 tensors (resized without cropping).
Step 2 (GPU): train from the cache.
"""
import argparse, os, sys, time
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--data_root", default=os.path.expanduser("~/EGEM_work/data/isic"))
p.add_argument("--cache", default=os.path.expanduser("~/EGEM_work/data/isic/cache_224sq_uint8.pt"))
p.add_argument("--build_cache", action="store_true")
p.add_argument("--workers", type=int, default=16)
p.add_argument("--epochs", type=int, default=15)
p.add_argument("--lr", type=float, default=1e-5)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--holdout", type=float, default=0.1, help="fraction of the training split not trained on")
p.add_argument("--max_steps", type=int, default=None, help="stop early (smoke test)")
p.add_argument("--out", default=os.path.join(ROOT, "model_weights", "isic_vgg16.model"))
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
MEAN = torch.tensor([0.6678, 0.5296, 0.5242])[:, None, None]  # CH_datasets ISIC normalization
STD = torch.tensor([0.1282, 0.1417, 0.1521])[:, None, None]


def _load(path):
    from PIL import Image
    from torchvision.transforms import Resize, PILToTensor
    img = Image.open(path).convert("RGB")
    return PILToTensor()(Resize((224, 224))(img))


def build_cache():
    from multiprocessing import Pool
    from CH_datasets.datasets.isic import ISICDataset
    from CH_datasets.datasets.splits import SingletonIndexStorage
    splits = {"train": ISICDataset(a.data_root, train=True), "test": ISICDataset(a.data_root, train=False)}
    paths = {k: [s[0] for s in d.samples] for k, d in splits.items()}
    t = time.time()
    imgs = {}
    with Pool(a.workers) as pool:
        for k, v in paths.items():
            imgs[k] = []
            for i, img in enumerate(pool.imap(_load, v, chunksize=64)):
                imgs[k].append(img)
                if i % 2000 == 0:
                    print(k, i, f"{time.time() - t:.0f}s", flush=True)
    out = {}
    for k, d in splits.items():
        out[k] = {"x": torch.stack(imgs.pop(k)), "y": torch.as_tensor(np.asarray(d.targets))}
        print(k, tuple(out[k]["x"].shape), f"{time.time() - t:.0f}s", flush=True)
    ind = SingletonIndexStorage().get_sample_indicators("isic")
    out["test_clean_idx"] = torch.as_tensor(np.asarray(ind["test"]["clean"]))
    out["test_dirty_idx"] = torch.as_tensor(np.asarray(ind["test"]["dirty"]))
    torch.save(out, a.cache)
    print("saved", a.cache)


def augment(x):
    """Per image: random resized crop (scale 0.8-1, square) with probability 0.5, then a random horizontal flip."""
    from torchvision.transforms import RandomResizedCrop
    from torchvision.transforms.functional import hflip
    crop = RandomResizedCrop(224, scale=(0.8, 1), ratio=(1, 1), antialias=True)
    out = []
    for xi in x:
        if torch.rand(1) < 0.5:
            xi = crop(xi)
        if torch.rand(1) < 0.5:
            xi = hflip(xi)
        out.append(xi)
    return torch.stack(out)


@torch.no_grad()
def evaluate(model, x, y, dev):
    model.eval()
    pred = torch.cat([model(((x[i:i + 256].float() / 255 - MEAN) / STD).to(dev)).argmax(1).cpu()
                      for i in range(0, len(x), 256)])
    model.train()
    per_class = [(pred[y == c] == c).float().mean().item() for c in range(8)]
    return (pred == y).float().mean().item(), per_class


def train():
    from models.model_loading import load_model
    torch.manual_seed(a.seed)
    dev = "cuda"
    data = torch.load(a.cache)
    xtr, ytr = data["train"]["x"], data["train"]["y"]
    keep = torch.randperm(len(xtr))[int(a.holdout * len(xtr)):]
    xtr, ytr = xtr[keep], ytr[keep]
    xte, yte = data["test"]["x"], data["test"]["y"]
    clean = data["test_clean_idx"]
    model = load_model("vgg16_isic", None, device=dev)  # ImageNet weights, 8 outputs
    net = model.model
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    step, t0 = 0, time.time()
    for ep in range(a.epochs):
        perm = torch.randperm(len(xtr))
        tot = 0.0
        for i in range(0, len(perm), a.bs):
            idx = perm[i:i + a.bs]
            xb = ((augment(xtr[idx]).float() / 255 - MEAN) / STD).to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(net(xb), ytr[idx].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx); step += 1
            if step % 100 == 0:
                print(f"ep {ep} step {step} loss {loss.item():.3f} {time.time() - t0:.0f}s "
                      f"maxmem {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB", flush=True)
            if a.max_steps and step >= a.max_steps:
                break
        acc, _ = evaluate(net, xte[clean], yte[clean], dev)
        print(f"== epoch {ep}: train loss {tot / len(perm):.4f}, clean test acc {acc:.4f}", flush=True)
        if a.max_steps and step >= a.max_steps:
            break
    acc, per_class = evaluate(net, xte[clean], yte[clean], dev)
    print("final clean test acc", round(acc, 4), "per class", np.round(per_class, 3).tolist())
    model.save_state_dict(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    build_cache() if a.build_cache else train()
