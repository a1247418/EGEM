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
| ISIC, Fig. 3 | **Partly reproduced**: retrained model matches the original's accuracy; PCA-EGEM helps (+4 vs ≈+12 points), EGEM does not |
| ImageNet carton/mtb, Fig. 3 | Blocked: needs ImageNet train+val for 6 classes (no copy available; see `DATA.md`) |
| CelebA, Sec. 6 (Fig. 9) | **Not reproduced**: PCA-EGEM lowers blond recall slightly in almost every subgroup instead of rebalancing it |

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
retrained. Clean test accuracy 0.800, matching the paper's ≈0.80. Refinement runs: one GPU shard per
method and poisoning level, 8–28 min each.

| Method | clean | poisoned | paper clean / poisoned (read from Fig. 3) |
|---|---|---|---|
| Original | 0.800 ± 0.000 | 0.628 ± 0.000 | ≈ 0.80 / 0.645 |
| Ridge | 0.786 ± 0.016 | 0.628 ± 0.008 | ≈ 0.80 / 0.69 |
| RGEM | 0.797 ± 0.001 | 0.625 ± 0.001 | ≈ 0.81 / 0.65 |
| EGEM | 0.794 ± 0.001 | 0.633 ± 0.003 | ≈ 0.77 / 0.735 |
| PCA-EGEM | 0.792 ± 0.002 | **0.672 ± 0.007** | ≈ 0.80 / 0.765 |

As in the paper, the original model relies on the colored patches (−17 points), and PCA-EGEM is the only
method that clearly improves poisoned accuracy without losing clean accuracy. The size of the effect is
not reproduced: +4.4 points instead of ≈ +12, and **EGEM does not help at all** (paper ≈ +9).
- EGEM peaks at 0.639 poisoned for any α (α = 0.4) and collapses below α = 0.2.
- PCA-EGEM is almost flat in α: it already reaches 0.682 at α = 0.99, so most of its gain comes from the
  PCA projection itself. Directions that the 5,600 clean refinement images do not span are removed.

Likely causes:
- The model is retrained, so its CH features may be more entangled with useful features than in the
  authors' model.
- The refinement images are a subset of the training images (this is how CH_datasets builds the
  split), so the validation accuracy used for selection is ≈ 0.97 and not informative.
- With the triangular rule, the first refined layers are pruned least, while the paper's Supp. J finds
  the ISIC CH feature most separable at early layers.

Hypotheses tested so far (exploratory, not in the paper's protocol):
- *Spatial summing* (`--refiner_kwargs '{"spatial_sum": true}'`, as worded in the paper's Sec. 3.1, instead
  of the code's per-position statistics): no effect. Poisoned accuracy stays within ±0.01 of EGEM for every α.
- *Scaling rule* (`flat`, `inverse-triangular` instead of `triangular`, i.e. pruning early layers as hard or
  harder): poisoned accuracy never rises above the unpruned 0.626 at any α. At 5% slack, flat gets 0.615
  and inverse-triangular 0.602, against 0.633 for the standard rule.
- So with this retrained model, the ISIC CH feature is not separable by per-channel pruning; only the PCA
  basis helps.

Retrain is running at the paper's lr of 1e-7; a smoke test showed it does not change the model.

### Slack (paper Fig. 4 / Supp. G) — `figures/fig4_slack.png`

Test accuracy of the selected hyperparameter as the slack goes from 0 to 7% (700 samples/class).
- **MNIST-8:** as in the paper, more slack trades a little clean accuracy for robustness. EGEM, PCA-EGEM
  and Retrain are near 0.97 poisoned for any slack ≥ 1%. Ridge and RGEM need ≥ 4% to reach ~0.88.
- **ISIC:** PCA-EGEM's poisoned accuracy is highest at 0% slack (0.682) and *decreases* with more slack.
  Stronger pruning does not remove more of the CH effect here. This is consistent with the gain coming
  from the PCA projection rather than from α.

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
- PCA-EGEM on the activations after the VGG blocks and after the ReLUs.
- Recall of Blond_Hair on up to 5,000 test images per attribute.
- One GPU-shard job, 3 min.

| | test accuracy | blond recall (all) | Male | Wearing_Necktie | Sideburns |
|---|---|---|---|---|---|
| Original (paper) | 0.93 | ≈0.95 | ≈0.65 | ≈0.47 | ≈0.63 |
| Original (here) | 0.917 | 0.955 | 0.719 | 0.350 | 0.615 |
| PCA-EGEM, paper | – | ≈0.95 | ≈0.75 | ≈0.71 | ≈0.72 |
| PCA-EGEM, α = 0.01 (paper's value) | 0.947 | 0.854 | 0.500 | 0.200 | 0.385 |
| PCA-EGEM, α = 0.1 (5% slack here) | 0.933 | 0.922 | 0.632 | 0.300 | 0.538 |

The paper's α = 0.01 is what 5% slack gave *for its model*. Applying the slack rule here (80/20 split of
the verified images) selects α = 0.1. With either α, recall drops slightly in almost every subgroup; there is
no rebalancing toward the low-recall groups. Test accuracy rises because the refined model predicts
"blond" less often. Caveats:
- The low-recall subgroups have very few blond test images (Wearing_Necktie 20, Sideburns 13, Goatee 1).
  The paper's Goatee recall of ≈0.66 is impossible with a single blond Goatee image in the test split, so
  the paper probably sampled subgroups from a larger pool.
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
2. ISIC: investigate why EGEM does not help (layer choice/scaling rule, refinement data overlapping the
   training data); Retrain results are pending.
3. Work through the open items in `TODO_cleanup.md`.
