"""Extracts a subset of ImageNet classes from the Hugging Face `ILSVRC/imagenet-1k` parquet shards into the
torchvision ImageNet layout (`<out>/train/<wnid>/<file>`, `<out>/val/<wnid>/<file>`, `<out>/meta.bin`).

Shards are downloaded one at a time to `--tmp` (e.g. node-local job scratch) and deleted after extraction.
Requires a Hugging Face token with access to the gated dataset.
"""
import argparse, csv, os, re, shutil, sys, time
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
import pyarrow.parquet as pq
import torch
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, hf_hub_download

REPO = "ILSVRC/imagenet-1k"
p = argparse.ArgumentParser()
p.add_argument("--out", default=os.path.expanduser("~/EGEM_work/data/imagenet"))
p.add_argument("--tmp", default=os.environ.get("TMPDIR", "/tmp"))
p.add_argument("--classes", type=int, nargs="+", default=[444, 478, 519, 549, 671, 692])
p.add_argument("--splits", nargs="+", default=["validation", "train"])
p.add_argument("--max_shards", type=int, default=None, help="only process this many shards per split (test)")
p.add_argument("--workers", type=int, default=4)
a = p.parse_args()

rows = list(csv.DictReader(open(os.path.join(ROOT, "src", "refinement", "decomposition", "cxai", "config",
                                             "imagenet-label-mapping.csv"))))
wnids = [r["imagenet-id"] for r in rows]
assert len(wnids) == 1000 and wnids == sorted(wnids)
wanted = {c: wnids[c] for c in a.classes}
os.makedirs(a.out, exist_ok=True)
torch.save(({w: (r["desc"],) for w, r in zip(wnids, rows)}, []), os.path.join(a.out, "meta.bin"))
shards = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet"))
done_file = os.path.join(a.out, ".done_shards")
done = set(open(done_file).read().split()) if os.path.exists(done_file) else set()


def process(shard):
    split = "val" if "/validation-" in shard else "train"
    tmp = os.path.join(a.tmp, "hf_shards")
    local = hf_hub_download(REPO, shard, repo_type="dataset", local_dir=tmp)
    f = pq.ParquetFile(local)
    n = 0
    for g in range(f.num_row_groups):
        labels = f.read_row_group(g, columns=["label"]).column("label").to_pylist()
        keep = [i for i, l in enumerate(labels) if l in wanted]
        if not keep:
            continue
        imgs = f.read_row_group(g, columns=["image"]).column("image").take(keep).to_pylist()
        for i, img in zip(keep, imgs):
            # the shards append "_<wnid>" to the original file name; restore it (files must sort as in ILSVRC)
            name = re.sub(r"_n\d{8}(\.JPEG)$", r"\1", os.path.basename(img["path"] or ""))
            if not name:
                raise RuntimeError(f"{shard}: image without file name; the original file order can't be restored")
            d = os.path.join(a.out, split, wanted[labels[i]])
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, name), "wb") as out:
                out.write(img["bytes"])
            n += 1
    os.remove(local)
    return shard, n


for split in a.splits:
    todo = [s for s in shards if s.startswith(f"data/{split}-") and s not in done][:a.max_shards]
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for k, (shard, n) in enumerate(ex.map(process, todo), 1):
            with open(done_file, "a") as fh:
                fh.write(shard + "\n")
            print(f"{split} {k}/{len(todo)} {shard}: {n} images, {time.time() - t0:.0f}s", flush=True)
shutil.rmtree(os.path.join(a.tmp, "hf_shards"), ignore_errors=True)
for split in ["train", "val"]:
    for c, w in wanted.items():
        d = os.path.join(a.out, split, w)
        print(split, c, w, len(os.listdir(d)) if os.path.isdir(d) else 0)
