# EGEM reproduction notes (branch `reproduce-egem`)

Target paper: Linhardt, Müller, Montavon, *Preemptively Pruning Clever-Hans Strategies in Deep
Neural Networks*, Information Fusion 2024 (arXiv:2304.05727). The NLP use case is out of scope.
Cleanup items found along the way are listed in [`TODO_cleanup.md`](TODO_cleanup.md). How to get every dataset
and model weight is in [`DATA.md`](DATA.md).

## Status

| Experiment (paper) | Status |
|---|---|
| MNIST-8, Fig. 3 | **Reproduced** for Original, Retrain, EGEM, PCA-EGEM; Ridge and RGEM fall short on poisoned data |
| MNIST-8, sample-size sweep (Supp. H) | **Reproduced** for Ridge / EGEM / PCA-EGEM, 5–700 samples per class |
| MNIST CH variants, Fig. 6 | **Partly reproduced**: weights restored from git history; the poisoners were never released, so they are reconstructed and calibrated (see below) |
| Sparsity, Fig. 7 | Partly: the Linear_1 column matches; MaxPool2d / Linear_2 do not |
| ISIC, Fig. 3 | **Reproduced for PCA-EGEM** (+9.8 vs ≈+12 points, 3 models) after restoring the paper-era conv statistics; EGEM helps less than reported (+1.7 vs ≈+9) |
| ImageNet carton/mtb, Fig. 3 | Blocked: needs ImageNet train+val for 6 classes (no copy available; see `DATA.md`) |
| CelebA, Sec. 6 (Fig. 9) | **Reproduced qualitatively**: PCA-EGEM keeps overall recall and raises it for low-recall subgroups (smaller gains than the paper) |

## How to run

```bash
conda create -p ~/.conda/envs/egem python=3.10
~/.conda/envs/egem/bin/pip install -r src/requirements.txt
# data: see DATA.md (MNIST downloads itself)
cd EGEM   # repo root
python repro/run_scenario.py --scenario mnist-8 --refinement {none,retrain,ridge,rgem,egem,pcaegem} \
       --poisoning {none,uniform} --n_samples 700 --n_reps 5
python repro/analyze.py --scenario mnist-8 --n 700 --slack 0.01 0.05   # paper's selection rule
python repro/make_figures.py
```
MNIST runs on CPU: one EGEM/Ridge/RGEM run (5 reps × 9–12 hyperparameters) takes 1–5 min, Retrain about
3 min, and PCA-EGEM a few minutes per rep. ISIC runs on a GPU shard via
`sbatch repro/run_gpu.sbatch --scenario isic-1 --data_root ~/EGEM_work/data/isic --refinement <m> --poisoning <p>`
(8–28 min per run).

**Hyperparameter selection.** Every run evaluates the whole hyperparameter grid. The reported value
follows the paper's rule (Sec. 4.5, `src/selection.py`): per rep, take the strongest refinement whose
validation accuracy is at most `slack` below the unrefined model's validation accuracy. The original
`run.py` tracked the best *test* accuracy instead; that dead code is removed. All numbers below use 5% slack.

## Results

Mean ± std over 5 reps (the test set is 1000 MNIST test images, fixed across reps). "clean" = 0%
poisoning, "poisoned" = uniform 100% poisoning (the CH feature is added to every test image).

### MNIST-8 (paper Fig. 3), 700 samples/class — `figures/fig3_mnist_isic.png`

| Method | clean | poisoned | chosen hyperparameter |
|---|---|---|---|
| Original | 0.989 ± 0.000 | 0.691 ± 0.000 | – |
| Retrain | 0.975 ± 0.005 | 0.976 ± 0.004 | Ne = 100 |
| Ridge | 0.951 ± 0.015 | 0.885 ± 0.012 | λ = 10 (4 reps), 100 (1 rep) |
| RGEM | 0.945 ± 0.001 | 0.884 ± 0.001 | λ = 100 |
| EGEM | 0.971 ± 0.001 | 0.969 ± 0.001 | α = 0.3 |
| PCA-EGEM | 0.975 ± 0.001 | 0.969 ± 0.003 | α = 0.1 |

Paper: the original model loses ~30 points under poisoning (reproduced: −30 points). EGEM and PCA-EGEM end
within 4 points of the original clean accuracy with "virtually no gap" between clean and poisoned
(reproduced: −1.4 to −2.0 points, gap ≤ 0.6 points). At 1% slack, EGEM gets 0.984 / 0.971 and PCA-EGEM 0.983 / 0.978.
Retrain (all layers, Adam, lr 1e-3) also matches the paper: 0.975 / 0.976, picking the largest Ne in the grid.
**Deviation:** the paper also has Ridge and RGEM within 4 points. Here both reach only 0.88 on poisoned data
(about 10 points below the original clean accuracy). No λ in the grid does better: Ridge peaks at 0.921 / 0.909
(λ = 100) and RGEM at 0.945 / 0.884 (λ = 100). At 1% slack, Retrain gets 0.973 / 0.972, RGEM 0.983 / 0.853 (λ = 10),
and Ridge 0.969 / 0.772 (no λ is within 1% on validation, so the fallback picks the best validation accuracy).

Ridge and RGEM refit only the last layer in closed form on the penultimate activations, with the paper's grid
λ ∈ {1e-4, ..., 1e4} (Supp. F.3). Ridge regresses one-hot labels, RGEM the original model's logits (Eq. D.3). The bias
is not penalized, and the shrinkage is towards 0, not towards w_old. λ is relative to the *mean* covariance
E[aaᵀ], as in Eq. D.1/D.2. The earlier code used the sum aᵀa, so its λ was n_train = 5600 times smaller. With the sum the
paper's grid stops before the slack boundary (RGEM at λ = 1e4 was still at 0.987 / 0.766). The normal equations are
solved in float64, because aᵀa is near-singular (dead ReLUs).

### ISIC (paper Fig. 3), 700 samples/class

Model: `train_isic.py` (VGG-16, Adam 1e-4, bs 64, 10 epochs). The weights were never released, so it is
retrained, with three seeds. Seed 0 is the main model. Refinement runs use one GPU shard per method and
poisoning level, 8–45 min each. EGEM and PCA-EGEM use the paper-era conv statistics (per-image channel
sums), now the default; see "Fix" below.

| Method (seed-0 model) | clean | poisoned | paper clean / poisoned (read from Fig. 3) |
|---|---|---|---|
| Original | 0.800 ± 0.000 | 0.628 ± 0.000 | ≈ 0.80 / 0.645 |
| Retrain | 0.812 ± 0.001 | 0.671 ± 0.003 | ≈ 0.82 / 0.65 |
| Ridge | 0.786 ± 0.016 | 0.628 ± 0.008 | ≈ 0.80 / 0.69 |
| RGEM | 0.797 ± 0.001 | 0.625 ± 0.001 | ≈ 0.81 / 0.65 |
| EGEM | 0.788 ± 0.003 | 0.635 ± 0.003 | ≈ 0.77 / 0.735 |
| PCA-EGEM | 0.795 ± 0.003 | **0.724 ± 0.007** | ≈ 0.80 / 0.765 |

Over three independently trained models (5% slack):

| model | Original | EGEM | PCA-EGEM | PCA-EGEM, Nov-2024 code (per position) |
|---|---|---|---|---|
| seed 0 | 0.800 / 0.628 | 0.788 / 0.635 | **0.795 / 0.724** | 0.792 / 0.672 |
| seed 1 | 0.787 / 0.608 | 0.776 / 0.629 | **0.778 / 0.751** | 0.777 / 0.684 |
| seed 2 | 0.782 / 0.682 | 0.776 / 0.704 | **0.770 / 0.737** | 0.771 / 0.729 |
| mean | 0.790 / 0.639 | 0.780 / 0.656 | **0.781 / 0.737** | 0.780 / 0.695 |

- **PCA-EGEM reproduces**: on average +9.8 points on poisoned data at no clean cost (paper: ≈ +12). It is
  the most robust method, as in the paper.
- **Retrain** (lr 1e-7, 100 epochs selected) is better here than in the paper on poisoned data (0.671 vs ≈ 0.65).
- **EGEM** helps only +1.7 points on average (paper ≈ +9). This remains a discrepancy.

#### Fix: restore the conv-layer statistics used for the paper

The repo's EGEM/PCA-EGEM changed in commit `785b67b` (Nov 2024, "Changed PCA implementation"). Before it,
and as Sec. 3.1 of the paper describes, conv activations were **summed over the spatial dimensions**, giving
one channel vector per image. PCA-EGEM fitted its PCA on these vectors and applied the result as a per-image
channel rescaling. Afterwards every spatial position was treated as a separate sample.

`isic_diagnose.py` measures, per refined layer, how much of the patch-induced change
Δa = a(x+patch) − a(x) survives the multipliers, compared with the clean signal (α = 0.1, seed-0 model):

| layer | per position: patch / clean kept | channel sums: patch / clean kept |
|---|---|---|
| features.10 | 0.86 / 0.87 | 0.97 / 0.99 |
| features.17 | 0.72 / 0.75 | 0.87 / 0.97 |
| features.24 | 0.66 / 0.72 | 0.81 / 0.94 |

Per position, an image's patch locations look like ordinary edge locations of clean images, so they lie in
high-variance directions that are kept. Per image, the patch shifts the whole channel profile in a direction
clean images rarely take, so it is pruned. The default is now `spatial_sum=True`; `spatial_sum=False` gives
the Nov-2024 behaviour. MNIST results are identical with either setting, because there the only refined
conv layer is the first one, which the triangular rule leaves unpruned.

#### Why plain EGEM still helps little (`isic_diagnose.py`, `isic_shortcut_check.py`)

- **The model matters.** A same-shape blob in skin color or gray moves the seed-0 model's predictions
  toward nevus almost as much as the cyan patch, so this model learned the stickers largely by their round
  shape and edge. Those features are shared with lesion borders and can't be pruned. Models with seeds 1
  and 2 react mainly to the color:

  | model | clean | cyan patch | skin-color blob | gray blob |
  |---|---|---|---|---|
  | seed 0 | 0.807 | 0.617 | **0.702** | **0.683** |
  | seed 1 | 0.823 | 0.593 | 0.812 | 0.803 |
  | seed 2 | 0.797 | 0.667 | 0.777 | 0.767 |

- **Selection is conservative.** In a diagnostic α sweep on seed 1, EGEM at α = 0.2 reached +8 points, but
  5% slack picks α ≈ 0.6; at 10% slack EGEM reaches only +1 to +4 points. Per-channel pruning without the
  PCA rotation simply separates the patch less well than PCA-EGEM does (this is the paper's motivation for
  PCA-EGEM).
- **Ruled out:**
  - the scaling rule (`flat` / `inverse-triangular`, which prune early layers harder; both are worse);
  - memorized refinement data (refining on unseen clean test images doesn't help EGEM either);
  - the test poisoning (identical to CH_datasets' own examples);
  - sticker images hidden in the refinement set (none found outside the labelled block 432–609).

### Slack (paper Fig. 4 / Supp. G) — `figures/fig4_slack.png`

Test accuracy of the selected hyperparameter as the slack goes from 0 to 7% (700 samples/class).
- **MNIST-8:** as in the paper, more slack trades a little clean accuracy for robustness. EGEM, PCA-EGEM
  and Retrain are near 0.97 poisoned for any slack ≥ 1%. Ridge and RGEM need ≥ 4% to reach ~0.88.
- **ISIC:** PCA-EGEM gains poisoned accuracy up to 5% slack (0.684 → 0.724 on the seed-0 model) and then
  levels off. EGEM barely moves.

### Sample-size sweep (Supp. H), MNIST-8 — `figures/figH_samples.png`

| samples/class | EGEM | PCA-EGEM | Ridge | RGEM | Retrain |
|---|---|---|---|---|---|
| 5 | 0.967 / 0.968 | 0.813 / 0.812 | 0.780 / 0.748 | 0.900 / 0.828 | 0.853 / 0.706 |
| 10 | 0.958 / 0.956 | 0.917 / 0.915 | 0.862 / 0.839 | 0.955 / 0.873 | 0.880 / 0.767 |
| 50 | 0.956 / 0.951 | 0.958 / 0.955 | 0.946 / 0.824 | 0.943 / 0.886 | 0.923 / 0.870 |
| 200 | 0.966 / 0.964 | 0.974 / 0.968 | 0.953 / 0.871 | 0.944 / 0.884 | 0.957 / 0.956 |
| 700 | 0.971 / 0.969 | 0.975 / 0.969 | 0.951 / 0.885 | 0.945 / 0.884 | 0.975 / 0.976 |

Values are clean / poisoned test accuracy at 5% slack.
EGEM is robust down to 5 samples per class. PCA-EGEM needs ≥ 50 per class and Retrain ≥ 200.
Ridge and RGEM stay below 0.89 on poisoned data at every size. With fewer samples the PCA basis
has rank < layer width, and every direction outside it is pruned to zero, which also costs clean accuracy.

### MNIST CH-feature variants (paper Fig. 6), 50 samples/class — `figures/fig6_mnist_variants.png`

Poisoned accuracy, this run (paper read from Fig. 6). Clean accuracies are in `results/all_runs.csv`.

| Method | artifact | blur | color | remove |
|---|---|---|---|---|
| Original | 0.513 (0.69) | 0.949 (0.91) | 0.903 (0.91) | 0.852 (0.77) |
| Retrain | 0.634 (0.76) | 0.967 (0.97) | 0.979 (0.96) | 0.841 (0.79) |
| Ridge | 0.908 (0.94) | 0.910 (0.95) | 0.957 (0.92) | 0.909 (0.92) |
| RGEM | 0.802 (0.91) | 0.889 (0.92) | 0.876 (0.875) | 0.923 (0.93) |
| EGEM | 0.966 (0.97) | 0.911 (0.78) | 0.827 (0.79) | 0.802 (0.73) |
| PCA-EGEM | 0.967 (0.98) | 0.950 (0.96) | 0.879 (0.96) | 0.955 (0.93) |

The paper's qualitative claims hold. EGEM removes the localized additive artifact but *lowers*
poisoned accuracy for non-additive features (blur, color, remove). PCA-EGEM is at least as good as EGEM
everywhere, and it fixes `remove` (0.852 → 0.955). On blur and color it does not beat the
unrefined model, unlike the paper. The baselines follow the paper's pattern too: Retrain fails on the
localized features (artifact, remove), and Ridge/RGEM are mediocre everywhere. Caveats:
- The paper (5 % slack, 50/class) used 10 reps. This run used 5.
- **The poisoners are reconstructions.** The weights `mnist-rgb-*.model` come from git history (`e9b127c`),
  but the code that made the poisoned data was never committed. `probe_mnist_rgb.py` identifies the
  feature type each model reacts to: the 3-pixel artifact, Gaussian blur, a cyan tint (the paper's
  Fig. 6 shows cyan), and removal of the lower rows. Strengths were then calibrated so the original
  models roughly match Fig. 6: blur k=5/σ=1, red channel ×0.85, lower 28% removed. The artifact is
  the exact mnist-8 definition, yet it hurts this model more than the paper reports (0.51 vs ~0.69).
  So the RGB artifact model was probably trained or evaluated with a different artifact.

### CelebA, blond-hair recall per subgroup (paper Sec. 6, Fig. 9) — `celeba_sec6.py`, `figures/fig9_celeba_recall.png`

Setup as in the paper:
- Model `vgg16_celeba`, restored from git history.
- 200 "user-verified" validation images per class: correctly predicted, with ≥ 75% of |LRP| inside a
  hair mask. The mask is read off Fig. F.12, so it is approximate.
- PCA-EGEM on the activations after the VGG blocks and after the ReLUs, with α chosen by 5% slack on an
  80/20 split of the verified images (α = 0.1; the paper's model gave α = 0.01).
- Recall of Blond_Hair on up to 5,000 test images per attribute.
- One GPU-shard job, 3 min.

| | test acc. | recall (all) | Male | Wearing_Necktie | Sideburns | Chubby |
|---|---|---|---|---|---|---|
| Original (paper) | 0.93 | ≈0.95 | ≈0.65 | ≈0.47 | ≈0.63 | ≈0.79 |
| Original (here) | 0.917 | 0.955 | 0.719 | 0.350 | 0.615 | 0.524 |
| PCA-EGEM (paper) | – | ≈0.95 | ≈0.75 | ≈0.71 | ≈0.72 | ≈0.82 |
| PCA-EGEM (here) | 0.916 | 0.952 | **0.772** | **0.450** | **0.692** | **0.667** |
| PCA-EGEM, Nov-2024 per-position code | 0.933 | 0.922 | 0.632 | 0.300 | 0.538 | 0.476 |

**Reproduced qualitatively** with the paper-era PCA-EGEM: overall recall is unchanged, the low-recall subgroups
gain (Necktie +10, Chubby +14, Sideburns +8, Male +5, Brown_Hair +6 points), and high-recall groups lose at
most ~1 point (one exception: Blurry −6). The gains are smaller than the paper's for Necktie. With the
Nov-2024 per-position code, recall instead dropped in almost every subgroup
(`results/celeba_recall_perpos*.csv`).

Caveats:
- The low-recall subgroups have few blond test images (Wearing_Necktie 20, Sideburns 13, Goatee 1), so
  their recalls are noisy. The paper's Goatee recall of ≈0.66 is impossible with one blond Goatee image in
  the test split, so the paper probably sampled subgroups from a larger pool.
- The hair mask and the LRP rule (`epsilon_alpha2_beta1_flat`, the repo default) are best guesses.

### Sparsity of the CH-induced representation change (paper Fig. 7) — `sparsity_fig7.py`

| Feature | MaxPool2d | Linear_1 | Linear_2 | paper (MaxPool2d / Linear_1 / Linear_2) |
|---|---|---|---|---|
| artifact | 0.244 | 0.202 | 0.095 | ~0.10 / 0.22 / 0.62 |
| blur | 0.060 | 0.073 | 0.114 | ~0.11 / 0.07 / 0.27 |
| color | 0.066 | 0.076 | 0.112 | ~0.11 / 0.07 / 0.19 |
| remove | 0.129 | 0.117 | 0.116 | ~0.11 / 0.12 / 0.33 |

The Linear_1 column matches (the artifact is the sparsest). The other columns depend on where the paper
tapped the activations. Here they are the inputs of the refined layers `features.3/7/9`, and no tap
point in the network reproduces the paper's artifact value of 0.62 at Linear_2. Other ways to compute the
metric don't reproduce it either: the ratio of the image-averaged change, of the averaged absolute change,
or of channel-summed conv maps (`scratch` experiment, 0.09–0.10 at Linear_2 for the artifact).

## Next steps
1. ImageNet tasks (carton/crate/envelope/packet, mtb/bbt): they need the 6 classes plus validation (see
   `DATA.md` for ways to get only those) and a patch to `splits.py`, which addresses images by their
   position in the full list. Pretrained torchvision weights are fine. GPU shard jobs as for ISIC.
2. ISIC: EGEM (without PCA) still helps less than reported. More trained models, and the authors' model
   if available, would show whether this is model variance or a remaining difference.
3. Work through the open items in `TODO_cleanup.md`.
