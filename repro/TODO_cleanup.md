# Repo cleanup TODOs for a clean EGEM reproduction

Collected while reproducing Linhardt et al., "Preemptively Pruning Clever-Hans Strategies
in Deep Neural Networks" (Information Fusion 2024, arXiv:2304.05727) on branch `reproduce-egem`.
Items marked **[fixed on branch]** have a minimal fix applied; the rest are open.

## Blocking bugs (code does not run / gives wrong numbers)
- **[fixed on branch]** `src/refinement/decomposition/decomposer.py:527` uses `Optional[A, B]`,
  which raises `TypeError` at import, so `run.py` cannot start. Needs `Optional[Union[A, B]]`.
- **[fixed on branch]** `src/models/custom_model.py:19` loads state dicts with `map_location=None`;
  `mnist.model` was saved on CUDA, so loading fails on CPU-only machines.
- **[fixed on branch]** `src/run.py` tracked the hyperparameter with the **best test accuracy**
  (dead code: it was never saved, but it invites test-set leakage). It is removed. The paper's rule
  (strongest refinement whose *validation* accuracy is within s% slack of the unrefined model's) is
  now `src/selection.py`, which `run.py --slack` and `repro/analyze.py` both use. This needs
  `orig_top1_val`, which each result now stores.
- `src/run.py` `__main__` passes `decomposition_type=args.explanation_type` (should be
  `args.decomposition_type`), so any CLI run tries to build a decomposer named after the LRP rule.
- `src/run.py` CLI offers `--poisoning_strategy targeted` but `CH_datasets.scenario_examples`
  expects `"target"` -> ValueError.
- **[fixed on branch]** `src/experiment_config.py` mnist-rgb branch tested `"artifact" in "experiment_name"`
  (a string literal, always False), so all MNIST-RGB variants silently loaded **random weights**.
- **[fixed on branch, reconstructed]** `src/data/data_loading.py` listed scenarios `"minst-8_rgb-*"`
  (typo) and `CH_datasets` had no scenario/poisoner for the Section 5 variants at all. The branch adds
  `mnist-rgb-{artifact,blur,color,remove}` with poisoners *reconstructed* from the released weights
  and calibrated to Fig. 6 (see `repro/probe_mnist_rgb.py`). The authors' original
  definitions (blur kernel, tint, removed region) should replace them if they can be found.
- **[fixed on branch]** `run.py` disabled normalization only for `"mnist-8"`; the RGB-MNIST models
  were also trained on unnormalized inputs.
- `run.py` verbose plotting hard-codes `device="cuda"`.

## Paper/code mismatches to resolve or document
- Paper: refinement data = only **correctly predicted** clean samples, 700 per class with
  oversampling. Code: `evaluation.get_n_shot_data` takes any samples (no correctness filter).
- Paper (Supp. F.3) alpha grid {1e-5, 1e-4, 1e-3, 0.01, 0.1, ..., 0.9, 1};
  `experiment_config.get_refinement_hyperparams` uses {0.001, 0.01, 0.1, ..., 0.9, 0.99}.
- **[fixed on branch]** Paper Ridge/RGEM lambda grid {1e-4 ... 1e4}; code used {1e-3 ... 1e6}. The grid is now the
  paper's, and `RegressionRefiner` normalizes the normal equations by n (lambda relative to E[aa^T], Eq. D.1)
  and solves them in float64. This changes the Ridge numbers.
- **[fixed on branch]** Paper's "Retrain" and "RGEM" baselines were not exposed in `run.py`. Added `RetrainRefiner`
  (Adam, paper learning rates, Ne in {1, 5, 10, 20, 30, 50, 100}, BN frozen, grad-norm clip 1e-3) and `RGEMRefiner`
  (ridge on the original logits). Open: F.3 does not say whether the gradient clipping at 1e-3 is norm or value clipping.
- Paper MNIST available data = 700 samples total (Table 1); CH_datasets ships
  `mnist_refinement_idcs.npy` with 39,942 indices.
- Paper says the ISIC model is VGG-16 fine-tuned with 2 output nodes; config uses 8 classes
  (`vgg16_isic`, target 1, background 0,2..7).
- README of CH_datasets says ISIC 2019 data; ISIC model weights (`isic_vgg16.model`) are not in the
  repo or its history -> must be retrained (no training script/config for it).

- MNIST net in `blueprints.py` (conv 3x3/3x3, FC 784->500) differs from paper Table E.2
  (conv 3/5, FC 200). The released `mnist.model` matches the code, so the table is probably wrong.
- The 1000-sample MNIST test subset is drawn once per process, so all reps share the same test
  set (std over reps of the unrefined model is exactly 0); the paper draws 1000 per run.
- Paper Fig. 3: Ridge and RGEM stay within 4% on poisoned MNIST; here both reach only ~0.88 poisoned at 5% slack
  (paper grid, normalized lambda). No lambda in the grid gets there. Possible causes: the paper's MNIST net (Table E.2: FC 200)
  or the correctly-predicted-only refinement data.

- **[fixed on branch]** `CH_datasets/datasets/splits.py`: the test split's `"dirty"` entry is set to the *train* dirty indices,
  for both ImageNet (`imagenet_train_dirty`) and ISIC (`isic_train_dirty`). This is currently harmless, because
  only `test["clean"]` is used, but `test["all"]` is wrong.
- `src/run_training.py` / `models/training.py`: `PLModel.configure_optimizers` ignores `--lr` and
  always uses SGD(lr=1e-3, momentum 0.9), while the paper uses Adam. There is no recipe for the ISIC model;
  `repro/train_isic.py` follows Supp. E instead.

## Missing pieces for reproduction
- **[fixed on branch]** No README (the root README was just `# EGEM`): it now has install, data, quick-start and layout sections.
- No script that produces the paper's figures/tables (Fig. 3, 4, 6, 7, G/H/I) from results.
- Model weights: only `mnist.model` at HEAD. `mnist-rgb-*.model` and `celeba_vgg16.model`
  exist only in git history (`e9b127c`, deleted in `d896bad`); ISIC weights never committed.
  ImageNet experiments use torchvision pretrained weights (fine).
- Datasets: no download helper. **[fixed on branch]** The MNIST loader in CH_datasets now has `download=True`;
  the other datasets are **[documented on branch]**:
  `repro/DATA.md` gives the download commands and the expected layout.
  ImageNet needs a local copy (6 classes only); ISIC 2019 must be downloaded manually.
- No fixed seeds for PCA (`torch.pca_lowrank` is randomized) -> reps are not bit-reproducible.

## Dependencies / packaging
- **[fixed on branch]** `src/requirements.txt` missed imported packages: scikit-learn and
  pytorch_lightning for the core code, plus `timm` (<1.0), `nptyping` (<2), `scipy`, `frozendict` and
  `opencv` for the vendored `cxai` decomposition code. They are now listed (cxai ones as optional,
  since `cxai` is imported lazily).
- **[fixed on branch]** `numpy<2` is required (matplotlib/torch 2.2 wheels are built against numpy 1.x); now pinned.
- **[fixed on branch]** Imports were inconsistent: `refinement.*`, `src.refinement.*` (explainer.py:11)
  and bare `cxai.*` were all used, so three different `sys.path` roots were needed. Now everything
  imports relative to `src/` (run `python src/run.py` from the repo root), and `cxai` is imported
  lazily. Still open: make `src` a real package (e.g. `egem/`, with a `pyproject.toml`), and replace the
  `sys.modules` aliasing in `src/CH_datasets/__init__.py`.
- **[fixed on branch]** `.gitmodules` declared `src/CH_datasets` as an SSH submodule, but the files are
  vendored (tracked directly, no gitlink). The stale `.gitmodules` is removed. Changes to the vendored
  copy (RGB-MNIST poisoners, splits fix) should be upstreamed to `a1247418/CH_datasets`.

## Performance
- **[fixed on branch for EGEM/PCA-EGEM, via a one-entry cache in `EGEMRefiner`]** Every hyperparameter value re-extracts activations and re-fits PCA (`train_refinement` inside
  the alpha loop), although both are independent of alpha -> cache once per rep. PCA-EGEM on
  MNIST takes ~10 min/rep on CPU vs <1 min for EGEM.
- **[fixed on branch]** `DataLoader(num_workers=8)` was hard-coded in `load_scenario`; it is now
  `--num_workers` (default 8) in `run.py` and the repro driver.

## Repo hygiene
- Five large Colab notebooks at the root (`CEGEM_*.ipynb`, `Per_digt_EGEM_*.ipynb`, ~2.9 MB with
  outputs) are later student experiments (CEGEM, per-digit EGEM), not part of the paper ->
  move to `experiments/` or a separate branch, strip outputs.
- Dead code: commented-out `_get_balanced_data` in `data_loading.py` ("TODO: remove"),
  `get_default_layer_names` ("TODO: remove?"), `PCAMultiplierNet` (incomplete), `pdb` imports,
  NLP/ViT paths (out of scope for the vision reproduction).
- `run.py` writes results to a relative `results/` folder; refiners pickle `self.mods` with
  live tensors on the training device.
