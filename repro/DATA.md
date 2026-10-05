# Data and model weights for the EGEM reproduction

Everything below lives under `$HOME`. The paths are the defaults of the `repro/` scripts; change them
with `--data_root`. Sizes and checks are as of 2026-10-05.

## Python environment

```bash
conda create -p ~/.conda/envs/egem python=3.10
~/.conda/envs/egem/bin/pip install "numpy<2" pandas Pillow tqdm matplotlib seaborn torch==2.2.2 \
    torchvision==0.17.2 zennit==0.5.0 scikit-learn scipy "timm<1" frozendict "nptyping<2" \
    pytorch_lightning opencv-python-headless
```
`src/requirements.txt` misses most of the second line (see `TODO_cleanup.md`).

## MNIST (scenarios `mnist-8`, `mnist-rgb-*`)

The `CH_datasets` loader downloads MNIST with torchvision on first use.
Layout: `~/EGEM_work/data/MNIST/raw/*-ubyte` (about 55 MB). Pass `--data_root ~/EGEM_work/data`.

## ISIC 2019 (scenario `isic-1`)

Only the 2019 **training** images and labels are needed. `CH_datasets` splits them itself: every
10th image goes to test, giving 22,797 train / 2,534 test.
```bash
mkdir -p ~/EGEM_work/data/isic && cd ~/EGEM_work/data/isic
B=https://isic-challenge-data.s3.amazonaws.com/2019
curl -O $B/ISIC_2019_Training_GroundTruth.csv -O $B/ISIC_2019_Training_Metadata.csv
curl -C - --retry 5 -O $B/ISIC_2019_Training_Input.zip   # 9.77 GB
unzip -q ISIC_2019_Training_Input.zip                       # 16 GB, 25,331 jpgs
```
Expected layout (`--data_root ~/EGEM_work/data/isic`):
```
isic/ISIC_2019_Training_GroundTruth.csv
isic/ISIC_2019_Training_Input/ISIC_0000000.jpg ...
```
Checks done: no image listed in the CSV is missing. The per-class counts equal the ones hard-coded
in `CH_datasets/datasets/splits.py` (train `[4078, 11574, 2979, 786, 2358, 217, 236, 569]`, test
`[444, 1301, 344, 81, 266, 22, 17, 59]`). The hard-coded patch ("dirty") ranges, train positions
432–609 and test 50–67, line up with the block of images that carry colored patches; I checked the
images at both edges of each range by eye. These positions depend on the CSV order, so do not
re-sort the CSV.

The ISIC model was never released. Train it with `repro/train_isic.py`:
```bash
python repro/train_isic.py --build_cache --workers 8      # CPU, one-off: 224px uint8 cache (~3.8 GB)
sbatch repro/train_isic.sbatch                             # GPU: VGG-16, Adam 1e-4, bs 64, 10 epochs
```
This writes `model_weights/isic_vgg16.model`, the path `experiment_config.py` expects.

## ImageNet (scenarios `carton-crate`, `carton-envelope`, `carton-packet`, `mtb-bbt`) — not done yet

- The models are torchvision's ImageNet-pretrained ResNet-50 and VGG-16, downloaded automatically to
  `~/.cache/torch/hub/checkpoints`.
- The data must be the **full ILSVRC-2012** set in `torchvision.datasets.ImageNet` layout:
  `root/ILSVRC2012_devkit_t12.tar.gz`, plus `train/` and `val/` (or the original tarballs, which
  torchvision unpacks).
- A 6-class subset does **not** work out of the box. `splits.py` stores the manually labelled
  watermark/frame images as offsets into the full torchvision target list.
- Getting ImageNet requires an image-net.org account and accepting its terms. Hugging Face
  `ILSVRC/imagenet-1k` is gated in the same way. The user must provide access or point to an existing copy.
- Classes used: carton 478, crate 519, envelope 549, packet 692, mountain bike 671, bicycle-built-for-two 444.

## CelebA (Sec. 6, not attempted)

`torchvision.datasets.CelebA(root, download=True)` (about 1.4 GB, from Google Drive, often rate-limited).
The model `celeba_vgg16.model` exists only in git history: `git show e9b127c:model_weights/celeba_vgg16.model`.

## Model weights overview

| File | Source |
|---|---|
| `model_weights/mnist.model` | in repo |
| `model_weights/mnist-rgb-{artifact,blur,color,remove}.model` | restored on this branch from commit `e9b127c` |
| `model_weights/isic_vgg16.model` | trained with `repro/train_isic.py` (never released) |
| torchvision `resnet50` / `vgg16` | downloaded automatically |
| `celeba_vgg16.model` | `git show e9b127c:model_weights/celeba_vgg16.model` |
