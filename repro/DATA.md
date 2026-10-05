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

The models are torchvision's ImageNet-pretrained ResNet-50 and VGG-16 (downloaded automatically to
`~/.cache/torch/hub/checkpoints`). For the data, **only the six classes are needed** (about 0.8 GB):

| class id | wnid | name |
|---|---|---|
| 444 | n02835271 | bicycle-built-for-two |
| 478 | n02971356 | carton |
| 519 | n03127925 | crate |
| 549 | n03291819 | envelope |
| 671 | n03792782 | mountain bike |
| 692 | n03871628 | packet |

Expected layout, i.e. `torchvision.datasets.ImageNet` with only these class folders
(`--data_root <root>`):
```
<root>/meta.bin    # or ILSVRC2012_devkit_t12.tar.gz, from which torchvision builds meta.bin
<root>/train/n02971356/n02971356_*.JPEG ...   # complete class folders, original file names
<root>/val/n02971356/ILSVRC2012_val_*.JPEG ...  # the 50 val images per class, sorted into wnid folders
```
The hand-labelled watermark/frame images are stored as positions *within* each class
(`CH_datasets/datasets/splits.py`). So the class folders must be complete and keep their original file
names. Labels of a subset copy are mapped back to the official class ids through the devkit's wnid
list. Both steps were tested on a fake six-class tree.

Ways to get only these classes (all need an ImageNet account that has accepted the terms):
- **Range requests on `ILSVRC2012_img_train.tar`**: it is a tar of 1,000 per-class tars. Read only the
  512-byte headers, then download the 6 inner tars (~0.8 GB). This needs a server that honours HTTP
  `Range`; not verified for image-net.org.
- **Kaggle** `imagenet-object-localization-challenge`: per-class folders, single files via
  `kaggle competitions download -f ...` (slow for ~7.8k files).
- **Hugging Face** `ILSVRC/imagenet-1k`, streamed and filtered by label: nothing extra is stored, but
  all ~150 GB are downloaded.
- An existing ImageNet copy: copy the six class folders and the devkit/meta file.

The validation set (`ILSVRC2012_img_val.tar`, 6.3 GB) is a flat folder. torchvision sorts it into wnid
folders using the devkit when given the tar; afterwards only the six folders need to be kept.

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
