"""Train the ISIC model used in the paper (Supp. Note E): ImageNet-pretrained VGG-16, fine-tuned with
Adam (lr 1e-4, batch size 64, 10 epochs) on the ISIC 2019 training split of CH_datasets, which keeps
the naturally occurring colored patches. The repo never shipped `model_weights/isic_vgg16.model`.

Step 1 (CPU node): `--build_cache` decodes all JPEGs once into 224x224 uint8 tensors, with the same
Resize(224)+CenterCrop(224) as the evaluation transform. Step 2 (GPU job): train from the cache.
"""
import argparse, os, sys, time
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [os.path.join(ROOT, "src"), ROOT, os.path.join(ROOT, "src", "CH_datasets")]
import numpy as np
import torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--data_root", default=os.path.expanduser("~/EGEM_work/data/isic"))
p.add_argument("--cache", default=os.path.expanduser("~/EGEM_work/data/isic/cache_224_uint8.pt"))
p.add_argument("--build_cache", action="store_true")
p.add_argument("--workers", type=int, default=16)
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--bs", type=int, default=64)
p.add_argument("--max_steps", type=int, default=None, help="stop early (smoke test)")
p.add_argument("--out", default=os.path.join(ROOT, "model_weights", "isic_vgg16.model"))
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
MEAN = torch.tensor([0.6678, 0.5296, 0.5242])[:, None, None]  # CH_datasets ISIC normalization
STD = torch.tensor([0.1282, 0.1417, 0.1521])[:, None, None]


def _load(path):
    from PIL import Image
    from torchvision.transforms import Resize, CenterCrop, PILToTensor
    img = Image.open(path).convert("RGB")
    return PILToTensor()(CenterCrop(224)(Resize(224)(img)))


def build_cache():
    from multiprocessing import Pool
    from CH_datasets.datasets.isic import ISICDataset
    from CH_datasets.datasets.splits import SingletonIndexStorage
    splits = {"train": ISICDataset(a.data_root, train=True), "test": ISICDataset(a.data_root, train=False)}
    paths = {k: [s[0] for s in d.samples] for k, d in splits.items()}
    t = time.time()
    # One pool, created before any torch op: forking after torch has started its OpenMP threads deadlocks.
    with Pool(a.workers) as pool:
        imgs = {k: pool.map(_load, v, chunksize=64) for k, v in paths.items()}
    out = {}
    for k, d in splits.items():
        out[k] = {"x": torch.stack(imgs.pop(k)), "y": torch.as_tensor(np.asarray(d.targets))}
        print(k, tuple(out[k]["x"].shape), f"{time.time() - t:.0f}s", flush=True)
    ind = SingletonIndexStorage().get_sample_indicators("isic")
    out["test_clean_idx"] = torch.as_tensor(np.asarray(ind["test"]["clean"]))
    out["test_dirty_idx"] = torch.as_tensor(np.asarray(ind["test"]["dirty"]))
    torch.save(out, a.cache)
    print("saved", a.cache)


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
            xb = ((xtr[idx].float() / 255 - MEAN) / STD).to(dev, non_blocking=True)
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
