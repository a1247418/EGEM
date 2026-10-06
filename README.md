# EGEM: Explanation-Guided Exposure Minimization

Code for L. Linhardt, K.-R. Müller, G. Montavon,
[*Preemptively Pruning Clever-Hans Strategies in Deep Neural Networks*](https://arxiv.org/abs/2304.05727),
Information Fusion, 2024.

EGEM soft-prunes a trained network so that it keeps only the variation that the user has
validated on a small set of clean, correctly explained samples. Clever-Hans (CH) strategies that the
user never saw lose their influence. PCA-EGEM does the same in a PCA basis of each layer's
activations.

![EGEM overview](docs/overview.png)

## Install

```bash
conda create -n egem python=3.10
conda activate egem
pip install -r src/requirements.txt
```

## Data

- **MNIST** is downloaded automatically.
- **ISIC 2019**: download `ISIC_2019_Training_Input.zip` and `ISIC_2019_Training_GroundTruth.csv` from
  `https://isic-challenge-data.s3.amazonaws.com/2019/` and unzip into one folder (`--data_root`). The model is
  trained with `repro/train_isic.py` (writes `model_weights/isic_vgg16.model`).
- **ImageNet**: `torchvision.datasets.ImageNet` layout (devkit + `train/` + `val/`); only the six classes
  used by the scenarios are needed (wnids n02835271, n02971356, n03127925, n03291819, n03792782, n03871628).
- **CelebA**: `torchvision.datasets.CelebA(root, download=True)` (requires `gdown`).

## Quick start

Run from the repository root:

```bash
python src/run.py --scenario_name mnist-8 --data_root <data dir> --refinement pcaegem \
    --poisoning_strategy uniform --n_samples 700 --n_reps 5
```

- `--refinement`: `none` (the original model), `egem`, `pcaegem`, and the baselines `retrain`, `ridge`
  and `rgem`.
- `--poisoning_strategy`: how the test set is modified. `none` gives clean data; `uniform` adds the CH
  artifact to every class.
- Scenarios: `mnist-8`, `mnist-rgb-{artifact,blur,color,remove}`, `isic-1`, `carton-{crate,envelope,packet}`,
  `mtb-bbt`.

Each run evaluates the whole hyperparameter grid and saves every result to `results/`. It prints the
value chosen by slack-based selection: the strongest refinement whose validation accuracy is within
`--slack` (default 5%) of the original model's (`src/selection.py`).

## Layout

| Path | Content |
|---|---|
| `src/refinement/refiner.py` | EGEM / PCA-EGEM and the baselines (Retrain, Ridge, RGEM, ...) |
| `src/run.py`, `src/experiment_config.py` | experiment runner and per-scenario configuration (layers, grids) |
| `src/CH_datasets/` | vendored CH benchmark tasks (poisoners, splits) |
| `src/selection.py` | slack-based hyperparameter selection |
| `model_weights/` | MNIST models (the ISIC model is trained with `repro/train_isic.py`) |
| `repro/` | experiment drivers, analysis and plotting scripts, results and figures |

## Citation

```bibtex
@article{linhardt2024preemptively,
  title   = {Preemptively Pruning Clever-Hans Strategies in Deep Neural Networks},
  author  = {Linhardt, Lorenz and M{\"u}ller, Klaus-Robert and Montavon, Gr{\'e}goire},
  journal = {Information Fusion},
  year    = {2024}
}
```
