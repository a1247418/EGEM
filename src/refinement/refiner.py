import os
import copy
from typing import Optional, List, Dict, Tuple
import torch
import numpy as np
import pickle as pkl

import tqdm
from torch.utils.data import DataLoader, TensorDataset
from zennit.core import BasicHook

from refinement.helpers import (
    get_module_by_name,
    layer_names_to_layers,
    get_activations,
    replace_layer,
)
from refinement.explainer import Explainer

Array = np.ndarray
Tensor = torch.Tensor


class PCAMultiplierNet(torch.nn.Module):
    """
    Module that optionally centers and applies PCA rotation, then multiplies with a given multiplier, and transforms back into the original space.
    This version is implemented w standard layers.
    """
    # TODO complete this class (conv layers) and replace PCAMultiplierLayer with this class
    def __init__(
            self,
            multiplier: torch.Tensor,
            pca_V: torch.Tensor,
            pca_Xm: torch.Tensor,
            cls_token_only: bool = False,
    ):
        super(PCAMultiplierNet, self).__init__()
        self.pca_V = pca_V
        self.pca_Xm = pca_Xm
        self.multiplier = multiplier
        self.cls_token_only = cls_token_only

        centering_and_rotation = torch.nn.Linear(pca_V.shape[1], pca_V.shape[0], bias=True)
        centering_and_rotation.weight.data = pca_V
        centering_and_rotation.bias.data = -torch.matmul(pca_Xm, pca_V.T)

        multiplier_diag = torch.nn.Linear(pca_V.shape[0], pca_V.shape[0], bias=False)
        multiplier_diag.weight.data = torch.eye(pca_V.shape[0])*multiplier if type(multiplier) in (int, float) else torch.diag(multiplier)

        rotation_and_uncentering = torch.nn.Linear(pca_V.shape[0], pca_V.shape[1], bias=True)
        rotation_and_uncentering.weight.data = pca_V.T
        rotation_and_uncentering.bias.data = pca_Xm

        self.layers = torch.nn.Sequential(
                centering_and_rotation,
                multiplier_diag,
                rotation_and_uncentering
        )
        self.layers.to(pca_Xm.device)

    def forward(self, x):
        if len(x[0].shape) <= 2:
            # Linear layer or Attention layer
            if self.cls_token_only:
                # Classifier token
                raise NotImplementedError()
            else:
                # All tokens
                out = self.layers(x)
        elif len(x[0].shape) == 3:
            # Conv
            raise NotImplementedError("Conv layers are not implemented yet -- please use PCAMultiplierLayer instead")
        else:
            raise NotImplementedError("Not implemented for {len(x[0].shape)}-dim inputs yet -- please use PCAMultiplierLayer instead")

        return out


class PCAMultiplierLayer(torch.nn.Module):
    """
    Module that optionally centers and applies PCA rotation, then multiplies with a given multiplier, and transforms back into the original space.
    """

    def __init__(
        self,
        multiplier: torch.Tensor,
        pca_V: Optional[torch.Tensor] = None,
        pca_Xm: Optional[torch.Tensor] = None,
        cls_token_only: bool = False,
        inverse: bool = False,
        spatial_sum: bool = False,
    ):
        super(PCAMultiplierLayer, self).__init__()
        self.pca_V = pca_V
        self.pca_Xm = pca_Xm
        self.multiplier = multiplier
        self.cls_token_only = cls_token_only
        self.inverse = inverse
        # Conv layers with spatial_sum: the PCA was fitted on per-image channel sums; the refined sums are
        # turned into a per-image channel rescaling of the feature map (the implementation at the time of the
        # paper, commit e9b127c). Otherwise every spatial position is projected separately.
        self.spatial_sum = spatial_sum

    def forward(self, x):
        modified_input = x
        if self.pca_V is not None:
            modified_input = modified_input.to(self.pca_Xm.dtype)
            Xm = self.pca_Xm
            V = self.pca_V

            # PCA-EGEM
            if len(modified_input[0].shape) <= 2:
                # Linear layer or Attention layer
                if self.cls_token_only:
                    # Classifier token
                    if not self.inverse:
                        modified_input0 = modified_input[:, 0][0] - Xm
                        modified_input0 = torch.matmul(modified_input0, V.T)
                        modified_input0 *= self.multiplier
                        modified_input[:, 0][0] = modified_input0
                    else:
                        modified_input[:, 0][0] = torch.matmul(modified_input[:, 0][0], V)
                        modified_input[:, 0][0] += Xm
                else:
                    # All tokens
                    if not self.inverse:
                        modified_input = modified_input - Xm
                        modified_input = torch.matmul(modified_input, V.T)
                        modified_input *= self.multiplier
                    else:
                        modified_input = torch.matmul(modified_input, V)
                        modified_input += Xm
            elif len(modified_input[0].shape) == 3 and self.spatial_sum:
                # Conv layer, statistics of per-image channel sums
                if not self.inverse:
                    modified_input_sum = modified_input.sum(dim=[-2, -1])
                    modified_input_sum_mod = torch.matmul(modified_input_sum - Xm, V.T) * self.multiplier
                    modified_input_sum_mod = torch.matmul(modified_input_sum_mod, V) + Xm
                    channel_multiplier = (
                        modified_input_sum_mod
                        / (modified_input_sum + (modified_input_sum == 0) * 1e-6)
                    )[..., None, None]
                    modified_input = modified_input * channel_multiplier.detach()
                # the inverse layer is the identity: the forward layer already maps back
            elif len(modified_input[0].shape) == 3:
                # Conv layer
                if not self.inverse:
                    modified_input = modified_input - Xm[None, :, None, None] # b x c x w x h
                    modified_input = torch.matmul(modified_input.permute(0,2,3,1),V.T) # b x w x h x c'
                    modified_input *= self.multiplier # TODO unsqueeze
                    modified_input = modified_input.permute(0,3,1,2) # b x c' x w x h
                    #modified_input = mi.sum(dim=0, keepdim=True)
                else:
                    #pass
                    modified_input = torch.matmul(modified_input.permute(0,2,3,1),V) # b x w x h x c
                    modified_input = modified_input.permute(0,3,1,2) # b x c x w x h
                    modified_input += Xm[None, :, None, None]
            modified_input = modified_input.to(x.dtype)
        elif not self.inverse:
            if len(modified_input.shape) == 4:
                modified_input *= self.multiplier[None, :, None, None]
            else:
                modified_input *= self.multiplier

        return modified_input


# Abstract base class BaseRefiner
class BaseRefiner:
    def __init__(
        self,
        model: torch.nn.Module,
        layer_names: List[str],
        device: str = "cuda",
        verbose: bool = False,
        **kwargs,
    ):
        self.model = model
        self.layer_names = layer_names  # TODO make sure they are sorted
        self.mods = {}
        self.device = device
        self.verbose = verbose
        self.is_refined = False
        self.is_trained = False

    def train_refinement(self, train_loader: DataLoader, val_loader: DataLoader):
        """Train the refinement method. This method should create mods."""
        self.is_trained = True

    def refine(self):
        """Refine the model. This method should apply the mods."""
        if not self.is_trained:
            raise ValueError("The refiner has not been trained yet.")
        self.is_refined = True

    def unrefine(self):
        """Unrefine the model. This method should remove the mods."""
        self.is_refined = False

    def get_filename(self) -> str:
        """Get the filename to save the refiner."""
        raise NotImplementedError

    def save(self, out_path):
        if self.is_trained:
            if not os.path.exists(out_path):
                print("\nOutput directory does not exist...")
                print("Creating output directory to save results...\n")
                os.makedirs(out_path)
            file_path = os.path.join(out_path, self.get_filename())
            with open(file_path, "wb") as f:
                pkl.dump(self.mods, f)
        else:
            raise AssertionError(
                "Cannot save refiner. Refiner has not been trained yet."
            )

    def load(self, out_path):
        file_path = os.path.join(out_path, self.get_filename())
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                self.mods = pkl.load(f)
        else:
            raise FileNotFoundError(
                f"Cannot load refiner. File {file_path} does not exist."
            )


# Class StaticRefiner
class StaticRefiner(BaseRefiner):
    def __init__(
        self,
        model: torch.nn.Module,
        layer_names: List[str],
        device: str = "cuda",
        iterative: bool = False,
        **kwargs,
    ):
        super().__init__(model, layer_names, device=device, **kwargs)
        self.iterative = iterative

    def train_refinement(
        self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None
    ):
        super().train_refinement(train_loader, val_loader)
        refinement_units = (
            [self.layer_names]
            if not self.iterative
            else [[layer_name] for layer_name in self.layer_names]
        )
        for ru in refinement_units:
            print(f"Training refinement unit for layers {ru}")
            self._train_refinement_unit(ru, **self._extract_values(ru, train_loader))
            # Unrefine to remove the mods from the previous iteration
            self.unrefine()
            # Refine s.t. the next iteration can build on the refined version
            self.refine()
        # Unrefine to leave a clean model
        self.unrefine()

    def _train_refinement_unit(self, layer_names: List[str], **kwargs):
        """Trains either a single layer or multiple."""
        raise NotImplementedError

    def _extract_values(self, layer_names: List[str], loader: DataLoader) -> Dict:
        """Extract the values from the loader. This method should return a dictionary matching the signature of train_refinement."""
        raise NotImplementedError


# Class EGEMRefiner
class EGEMRefiner(StaticRefiner):
    def __init__(
        self,
        model: torch.nn.Module,
        layer_names: List[str],
        device: str = "cuda",
        iterative: bool = False,
        alpha: float = 0.1,
        do_pca: bool = False,
        pca_dims: Optional[int] = None,
        scaling_rule: str = "triangular",
        spatial_sum: bool = True,
        **kwargs,
    ):
        """spatial_sum: for conv layers, compute the statistics (and the PCA) on per-image channel sums, as
        described in Sec. 3.1 of the paper and as implemented when the paper was published (commit e9b127c).
        False treats every spatial position as a sample (the implementation from Nov 2024, which is much
        less effective on ISIC, see repro/REPRO.md)."""
        super().__init__(
            model, layer_names, device=device, iterative=iterative, **kwargs
        )
        self.alpha = alpha
        self.spatial_sum = spatial_sum
        self.do_pca = do_pca
        self.pca_dims = pca_dims
        self.scaling_rule = scaling_rule
        assert self.do_pca or self.pca_dims is None
        assert self.scaling_rule in ["triangular", "flat", "inverse-triangular"]

    # Activations and PCA do not depend on alpha, so the hyperparameter search would otherwise redo the
    # (expensive) PCA fit for every alpha. Holds strong references so object identity stays valid.
    _cache = None

    def _extract_values(self, layer_names: List[str], loader: DataLoader) -> Dict:
        key = (self.model, loader, tuple(layer_names), self.do_pca, self.pca_dims, self.spatial_sum)
        cache = EGEMRefiner._cache
        if not self.iterative and cache is not None and all(a is b or a == b for a, b in zip(cache[0], key)):
            return cache[1]
        a_squared, pca_X, pca_V = get_activations(
            model=self.model,
            layer_names=layer_names,
            loader=loader,
            do_pca=self.do_pca,
            pca_dims=self.pca_dims,
            cls_token_only=False,
            average_tokens=False,
            square=True,
            reduce=True,
            batch_dim=0,
            capture_outputs=False,
            device=self.device,
            silent=True,
            spatial_sum=self.spatial_sum,
        )
        print("a_squared:", [a.shape for a in a_squared])
        values = {"a_squared": a_squared, "pca_X": pca_X, "pca_V": pca_V}
        if not self.iterative:
            EGEMRefiner._cache = (key, values)
        return values

    def _train_refinement_unit(
        self,
        layer_names: List[str],
        a_squared: List[Tensor],
        pca_X: List[Tensor],
        pca_V: List[Tensor],
    ):
        layers = layer_names_to_layers(self.model, layer_names)
        for l_i, (layer, layer_name) in enumerate(zip(layers, layer_names)):
            lbd = 0.000001 / 2.0  # Keep this starting value sufficiently small!
            avg_c = 1
            if self.scaling_rule == "triangular":
                threshold = 1 - (1.0 - self.alpha) * (
                    l_i / ((len(layers) - 1) if len(layers) > 1 else 1)
                )
            elif self.scaling_rule == "inverse-triangular":
                threshold = self.alpha + (1 - self.alpha) * (
                    l_i / ((len(layers) - 1) if len(layers) > 1 else 1)
                )
            elif self.scaling_rule == "flat":
                threshold = self.alpha
            else:
                raise ValueError(f"Unknown scaling rule: {self.scaling_rule}")

            # Increse pruning strength until the average pruning multiplier is below the threshold
            while avg_c >= threshold:
                lbd *= 2.0
                multiplier = a_squared[l_i] / (a_squared[l_i] + lbd)
                avg_c = float(torch.mean(multiplier))

            print(
                f"Layer {layer_name}",
                f"alpha: {self.alpha}",
                f"lbd: {lbd:.5f}",
                f"avg_c: {avg_c:.7f}",
            )

            if np.isnan(avg_c):
                raise ValueError("NaN encountered during refinement.")

            if self.do_pca:
                self.mods[layer_name] = torch.nn.Sequential(
                        PCAMultiplierLayer(
                            multiplier, pca_V[l_i], pca_X[l_i], spatial_sum=self.spatial_sum
                        ),
                        PCAMultiplierLayer(
                                multiplier, pca_V[l_i], pca_X[l_i], inverse=True, spatial_sum=self.spatial_sum
                        )
                )
            else:
                self.mods[layer_name] = PCAMultiplierLayer(multiplier)

    def refine(self):
        if not self.is_refined:
            super().refine()
            for layer_name, mod in self.mods.items():
                mod_layer = torch.nn.Sequential(
                    mod, get_module_by_name(self.model, layer_name)
                )
                # self.model.__setattr__(layer_name, mod_layer)
                replace_layer(self.model, layer_name, mod_layer)

    def unrefine(self):
        if self.is_refined:
            super().unrefine()
            for layer_name, _ in self.mods.items():
                mod_layer = get_module_by_name(self.model, layer_name)[-1]
                # self.model.__setattr__(layer_name, mod_layer)
                replace_layer(self.model, layer_name, mod_layer)

    def get_filename(self) -> str:
        return f"egem_it{self.iterative}_a{self.alpha}_{self.scaling_rule}_pca{self.pca_dims}_sum{self.spatial_sum}.pkl"


class PCATruncRefiner(StaticRefiner):
    def __init__(
        self,
        model: torch.nn.Module,
        layer_names: List[str],
        device: str = "cuda",
        iterative: bool = False,
        pca_dims: Optional[int] = 10000,
        pca_before_layer = True,
        **kwargs,
    ):
        super().__init__(
            model, layer_names, device=device, iterative=iterative, **kwargs
        )
        self.pca_dims = pca_dims
        self.pca_before_layer = pca_before_layer

    def _extract_values(self, layer_names: List[str], loader: DataLoader) -> Dict:
        _, pca_X, pca_V = get_activations(
            model=self.model,
            layer_names=layer_names,
            loader=loader,
            do_pca=True,
            pca_dims=self.pca_dims,
            cls_token_only=False,
            average_tokens=False,
            square=False,
            reduce=True,
            batch_dim=0,
            capture_outputs=not self.pca_before_layer,
            device=self.device,
            silent=True,
        )
        return {"pca_X": pca_X, "pca_V": pca_V}

    def _train_refinement_unit(
        self,
        layer_names: List[str],
        pca_X: List[Tensor],
        pca_V: List[Tensor],
    ):
        layers = layer_names_to_layers(self.model, layer_names)
        for l_i, (layer, layer_name) in enumerate(zip(layers, layer_names)):
            #self.mods[layer_name] = (PCAMultiplierLayer(1, pca_V[l_i], pca_X[l_i]),
            #                         PCAMultiplierLayer(1, pca_V[l_i], pca_X[l_i], inverse=True))
            self.mods[layer_name] = PCAMultiplierNet(1, pca_V[l_i], pca_X[l_i])

    def refine(self):
        if not self.is_refined:
            super().refine()
            for layer_name, mod in self.mods.items():
                if self.pca_before_layer:
                    mod_layer = torch.nn.Sequential(
                        #mod[0], mod[1], get_module_by_name(self.model, layer_name)
                        mod, get_module_by_name(self.model, layer_name)
                    )
                else:
                    mod_layer = torch.nn.Sequential(
                        #get_module_by_name(self.model, layer_name), mod[0], mod[1]
                        get_module_by_name(self.model, layer_name), mod
                    )
                replace_layer(self.model, layer_name, mod_layer)

    def unrefine(self):
        if self.is_refined:
            super().unrefine()
            for layer_name, _ in self.mods.items():
                mod_layer = get_module_by_name(self.model, layer_name)[-1 if self.pca_before_layer else 0]
                replace_layer(self.model, layer_name, mod_layer)

    def get_filename(self) -> str:
        return f"pcatrunc_it{self.iterative}_pca{self.pca_dims}.pkl"


class WEGEMRefiner(StaticRefiner):
    def __init__(
        self,
        model: torch.nn.Module,
        layer_names: List[str],
        explainer: Explainer,
        device: str = "cuda",
        lmbda: float = 0.1,
        **kwargs,
    ):
        super().__init__(model, layer_names, device=device, **kwargs)
        self.lmbda = lmbda
        self.explainer = explainer
        self.mods = {l: None for l in layer_names}

    def _extract_values(self, layer_names: List[str], loader: DataLoader) -> Dict:
        _, r_squared, ar_squared, _, _ = get_activations(
            model=self.model,
            layer_names=layer_names,
            loader=loader,
            do_pca=False,
            cls_token_only=False,
            average_tokens=False,
            square=True,
            reduce=True,
            batch_dim=0,
            capture_outputs=False,
            device=self.device,
            silent=True,
            explainer=self.explainer,
            combine_a_exp=True,
        )
        # TODO: the explainer is not giving sensitivites but Rs - check if that's correct
        return {"ar_squared": ar_squared, "r_squared": r_squared}

    def _train_refinement_unit(
        self,
        layer_names: List[str],
        r_squared: List[Tensor],
        ar_squared: List[Tensor],
    ):
        layers = layer_names_to_layers(self.model, layer_names)
        for l_i, (layer, layer_name) in enumerate(zip(layers, layer_names)):
            # print("ar_squared:", ar_squared[l_i].shape, "r_squared:", r_squared[l_i].shape)
            multiplier = ar_squared[l_i] / (
                ar_squared[l_i]
                + r_squared[l_i][None].repeat([ar_squared[l_i].shape[0], 1])
                * self.lmbda
            )
            multiplier = torch.nan_to_num(multiplier, nan=0.0)

            # if torch.isnan(torch.mean(multiplier)):
            #    raise ValueError(f"NaN encountered during refinement. ({layer_name}, lambda={self.lmbda})")

            multiplier = multiplier.T
            weight = layer.weight.data
            if len(weight.shape) == 4:
                # Conv layer
                multiplier = multiplier[..., None, None]
            W = weight * multiplier
            # print("W:", W.shape, "layer W:", layer.weight.shape, "multiplier:", multiplier.shape)
            self.mods[layer_name] = W

    def switch_weights(self):
        for layer_name, W in self.mods.items():
            if W is not None:
                # Switch stored and module weight
                layer = get_module_by_name(self.model, layer_name)
                temp = layer.weight.data
                layer.weight.data = W
                self.mods[layer_name] = temp

    def refine(self):
        if not self.is_refined:
            super().refine()
            self.switch_weights()

    def unrefine(self):
        if self.is_refined:
            super().unrefine()
            self.switch_weights()

    def get_filename(self) -> str:
        return f"wegem_it{self.iterative}_l{self.lmbda}_e{self.explainer.explanation_type}.pkl"


# Todo: Implement L1
class RegressionRefiner(StaticRefiner):
    def __init__(
        self,
        model: torch.nn.Module,
        layer_names: Optional[List[str]],
        lmbda: float,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(model, None, device=device, **kwargs)
        self.lmbda = lmbda

    def _extract_values(self, layer_names: List[str], loader: DataLoader) -> Dict:
        xs = []
        ys = []
        embedder = self.model.features[
            :-1
        ]  # assuming the last layer to be a projection layer
        # TODO: automatically select -1/-2 based on bias?
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                for i, l in enumerate(embedder):
                    x = l(x)
                xs.append(x)
                ys.append(y)
        xs = torch.cat(xs, dim=0)
        ys = torch.cat(ys, dim=0)
        return {"a": xs, "y": ys}

    def _targets(self, a: Tensor, y: Tensor) -> Tensor:
        """Ridge regression targets: one-hot true labels."""
        labels = torch.zeros(
            len(y),
            self.model.features[-1].out_features,
            device=a.device,
            dtype=torch.float32,
        )
        labels[np.arange(len(y)), y.long()] = 1
        return labels

    def _train_refinement_unit(self, layer_names: List[str], a: Tensor, y: Tensor):
        l = self.model.features[-1]  # last linear

        has_bias = l.bias is not None

        if has_bias:
            ab = torch.concat([a, torch.ones([a.shape[0], 1], device=a.device)], dim=1)
        else:
            ab = a

        # float64: S is near-singular (dead ReLUs), so float32 is inaccurate for small lambda
        ab = ab.double()
        # Normalize by n: the paper's objective (Supp. D, Eq. D.1) is E[(f(x,w)-t)^2] + lambda*||w||^2,
        # so lambda is relative to the covariance E[aa^T], not the sum a^T a
        S = torch.transpose(ab, 0, 1) @ ab / ab.shape[0]
        reg = self.lmbda * torch.eye(S.shape[-1], device=self.device, dtype=S.dtype)

        if has_bias:
            reg[-1] = 0  # don't penalize bias

        labels = self._targets(a, y)
        print(labels.shape, self.model.features[-1].weight.shape)
        # Invert the projection layer to get the labels in the original y-space
        # labels = labels @ self.model.features[-1].weight
        print(
            "S:",
            S.shape,
            "reg:",
            reg.shape,
            "a:",
            a.shape,
            "ab:",
            ab.shape,
            "y:",
            y.shape,
            "labels:",
            labels.shape,
        )
        wb_reorient = torch.linalg.solve(S + reg, ab.T @ labels.double() / ab.shape[0]).float()
        wb_reorient = wb_reorient.T
        print("wb_reorient:", wb_reorient.shape, "l.weight:", l.weight.shape)

        if has_bias:
            self.mods = {"weight": wb_reorient[:, :-1], "bias": wb_reorient[:, -1]}
        else:
            self.mods = {"weight": wb_reorient, "bias": None}

    def switch_weights_bias(self):
        # Switch stored and module weight
        layer = self.model.features[-1]
        temp_w = layer.weight.data
        layer.weight.data = self.mods["weight"]
        self.mods["weight"] = temp_w
        if layer.bias is not None:
            temp_b = layer.bias.data
            layer.bias.data = self.mods["bias"]
            self.mods["bias"] = temp_b

    def refine(self):
        if not self.is_refined:
            super().refine()
            self.switch_weights_bias()

    def unrefine(self):
        if self.is_refined:
            super().unrefine()
            self.switch_weights_bias()

    def get_filename(self) -> str:
        return f"ridge_e{self.lmbda}.pkl"


class RGEMRefiner(RegressionRefiner):
    """Response-guided exposure minimization (paper Supp. D, Eq. D.3): ridge regression of the last
    layer on the original model's outputs f(X, w_old) instead of the true labels. The penalty
    lambda*||w||^2 shrinks towards zero (not towards w_old); the bias is not penalized."""

    def _targets(self, a: Tensor, y: Tensor) -> Tensor:
        with torch.no_grad():
            return self.model.features[-1](a).float()  # model is unrefined during training

    def get_filename(self) -> str:
        return f"rgem_e{self.lmbda}.pkl"


class RetrainRefiner(BaseRefiner):
    """Retrain baseline (paper Supp. F.3): fine-tune all layers on the refinement data with Adam and
    cross-entropy, no extra regularization, batch-norm frozen, gradient norm clipped."""

    _cache = None  # (key, epochs trained, model state, optimizer state) of the last run

    def __init__(
        self,
        model: torch.nn.Module,
        layer_names: Optional[List[str]] = None,
        n_epochs: int = 10,
        lr: float = 1e-3,
        grad_clip: float = 1e-3,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(model, layer_names, device=device, **kwargs)
        self.n_epochs = n_epochs
        self.lr = lr
        self.grad_clip = grad_clip
        self.original_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def train_refinement(
        self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None
    ):
        super().train_refinement(train_loader, val_loader)
        bn_types = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
        bn_params = {id(p) for m in self.model.modules() if isinstance(m, bn_types) for p in m.parameters()}
        params = [p for p in self.model.parameters() if id(p) not in bn_params]
        req_grad = [p.requires_grad for p in self.model.parameters()]
        for p in params:
            p.requires_grad_(True)
        was_training = self.model.training
        self.model.train()
        for m in self.model.modules():
            if isinstance(m, bn_types):
                m.eval()  # keep running statistics fixed
        optimizer = torch.optim.Adam(params, lr=self.lr)
        loader = DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, shuffle=True)
        # The epoch grid is searched in ascending order: continue from the previous run on the same data
        # instead of retraining from scratch (100 instead of 216 epochs for the paper's grid).
        key = (self.model, train_loader, self.lr, self.grad_clip)
        start = 0
        cache = RetrainRefiner._cache
        if cache is not None and all(a is b or a == b for a, b in zip(cache[0], key)) and cache[1] <= self.n_epochs:
            self.model.load_state_dict(cache[2])
            optimizer.load_state_dict(cache[3])
            start = cache[1]
        for epoch in range(start, self.n_epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device).long()
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(self.model(x), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, self.grad_clip)
                optimizer.step()
        self.model.train(was_training)
        for p, r in zip(self.model.parameters(), req_grad):
            p.requires_grad_(r)
        self.mods = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        RetrainRefiner._cache = (key, self.n_epochs, self.mods, copy.deepcopy(optimizer.state_dict()))
        self.model.load_state_dict(self.original_state)  # leave a clean model

    def refine(self):
        if not self.is_refined:
            super().refine()
            self.model.load_state_dict(self.mods)

    def unrefine(self):
        if self.is_refined:
            super().unrefine()
            self.model.load_state_dict(self.original_state)

    def get_filename(self) -> str:
        return f"retrain_ne{self.n_epochs}_lr{self.lr}.pkl"


class WeightMagnitudePruner(StaticRefiner):
    def __init__(
        self,
        model: torch.nn.Module,
        layer_names: List[str],
        percent_pruned: int = 90,
        device: str = "cuda",
        random=False,
        **kwargs,
    ):
        super().__init__(model, layer_names, device=device, **kwargs)
        self.percent_pruned = percent_pruned
        self.random = random
        self.train_refinement(None)

    def _extract_values(self, layer_names: List[str], loader: DataLoader) -> Dict:
        weights = []
        for layer_name in layer_names:
            layer = get_module_by_name(self.model, layer_name)
            weights.append(layer.weight.data.detach())
        return {"weights": weights}

    def _train_refinement_unit(self, layer_names: List[str], weights: List[Tensor]):
        for layer_name, weight in zip(layer_names, weights):
            to_threshold = weight.abs()
            if self.random:
                to_threshold = torch.rand_like(to_threshold)
            threshold = np.percentile(to_threshold.cpu().numpy(), self.percent_pruned)
            mask = (to_threshold > threshold).to(self.device)
            self.mods[layer_name] = {"mask": mask, "weight": weight}

    def refine(self):
        if not self.is_refined:
            super().refine()
            for layer_name, mod in self.mods.items():
                mask = mod["mask"]
                weight = mod["weight"]
                layer = get_module_by_name(self.model, layer_name)
                layer.weight.data = weight * mask

    def unrefine(self):
        if self.is_refined:
            super().unrefine()
            for layer_name, mod in self.mods.items():
                weight = mod["weight"]
                layer = get_module_by_name(self.model, layer_name)
                layer.weight.data = weight

    def get_filename(self) -> str:
        return f"wmagnitudeprune_p{self.percent_pruned}.pkl"


class ActivationPruner(StaticRefiner):
    def __init__(
        self,
        model: torch.nn.Module,
        layer_names: List[str],
        per_layer_criterion: Dict[str, Array],
        percent_pruned: int = 90,
        random=False,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(model, layer_names, device=device, **kwargs)
        self.percent_pruned = percent_pruned
        self.per_layer_criterion = per_layer_criterion
        self.random = random
        self.train_refinement(None)

    def _extract_values(self, layer_names: List[str], loader: DataLoader) -> Dict:
        return {"per_layer_criterion": self.per_layer_criterion}

    def _train_refinement_unit(
        self, layer_names: List[str], per_layer_criterion: Dict[str, Array]
    ):
        for layer_name in layer_names:
            if self.random:
                to_threshold = np.random.rand(*(per_layer_criterion[layer_name].shape))
            else:
                to_threshold = per_layer_criterion[layer_name]
            threshold = np.percentile(to_threshold, self.percent_pruned)
            mask = to_threshold > threshold
            self.mods[layer_name] = {
                "mask": torch.from_numpy(mask).to(self.device),
                "weight": get_module_by_name(
                    self.model, layer_name
                ).weight.data.detach(),
            }

    def refine(self):
        if not self.is_refined:
            super().refine()
            for layer_name, mod in self.mods.items():
                mask = mod["mask"]
                weight = mod["weight"]
                layer = get_module_by_name(self.model, layer_name)
                if weight.shape != mask.shape:
                    mask = mask.unsqueeze(1).expand_as(
                        weight[:, :, 0, 0] if len(weight.shape) == 4 else weight
                    )
                    if len(weight.shape) == 4:
                        mask = mask.unsqueeze(-1).unsqueeze(-1)
                layer.weight.data = weight * mask

    def unrefine(self):
        if self.is_refined:
            super().unrefine()
            for layer_name, mod in self.mods.items():
                weight = mod["weight"]
                layer = get_module_by_name(self.model, layer_name)
                layer.weight.data = weight

    def get_filename(self) -> str:
        return f"aprune_p{self.percent_pruned}.pkl"


# Class EGEMRefiner
class PEGEMRefiner(StaticRefiner):
    def _init_layers(self):
        # Find the start and end modules
        self.sm = None
        self.em = None
        self.n_end_neurons = None
        self.end_is_out = False
        n_neurons = None
        looking_for_end_neurons = False
        for m, (name, module) in enumerate(self.model.named_modules()):
            try:
                n_neurons = module.weight.shape[0]
                # For conv layers, the number of neurons is the number of output channels
                if looking_for_end_neurons and name.startswith(self.layer_names[1]):
                    self.n_end_neurons = n_neurons
                    print(f"Found better end neuron count: {n_neurons} neurons")
            except AttributeError:
                pass
            if name == self.layer_names[0]:
                self.sm = module
            if name == self.layer_names[1]:
                self.em = module
                self.n_end_neurons = n_neurons
                print(f"End module {name} with {n_neurons} neurons")
                if m == len(self.layer_names) - 1:
                    self.end_is_out = True
                looking_for_end_neurons = True

        print("Start layer:", self.layer_names[0], "End layer", self.layer_names[1])
        try:
            assert self.sm is not None
            assert self.em is not None
        except AssertionError:
            print([(name, module) for name, module in enumerate(self.model.named_modules())])
            raise AssertionError("Start or end layer not found:", self.layer_names)

    def __init__(
        self,
        model: torch.nn.Module,
        layer_names: List[str],
        n_classes: int,
        device: str = "cuda",
        explainer: Explainer = None,
        percent_pruned: float = None,
        lmbda: float = None,
        do_pca: bool = False,
        pca_dims: Optional[int] = None,
        collapse_start: bool = False,
        top_n_training: Optional[int] = None,
        top_n_inference: Optional[int] = None,
        per_class_mask: bool = True,
        hard_pruning: bool = True,
        mask_dtype=torch.float32,
        **kwargs,
    ):
        super().__init__(model, layer_names, device=device, iterative=False, **kwargs)
        self.n_classes = n_classes
        self.explainer = explainer
        self.percent_pruned = percent_pruned
        self.lmbda = lmbda
        self.do_pca = do_pca
        self.pca_dims = pca_dims
        assert self.do_pca or self.pca_dims is None
        self.collapse_start = collapse_start
        self.top_n_training = top_n_training
        self.top_n_inference = top_n_inference
        self.per_class_mask = per_class_mask
        self.hard_pruning = hard_pruning
        self.mask_dtype = mask_dtype

        assert (lmbda is not None and not hard_pruning) or (
            percent_pruned is not None and hard_pruning
        )

        if self.do_pca:
            self.PCA = PCATruncRefiner(
                model, layer_names[-1:], device=device, iterative=False, pca_dims=pca_dims, pca_before_layer=False
            )

        self._init_layers()

    """ # TODO:remove 
    def _get_filter_hook(self, selector: Tensor):
        def grad_mapper(out_grad, outputs):
            s = selector.unsqueeze(dim=0)
            while len(s.shape) < len(out_grad.shape):
                s = s.unsqueeze(-1)
            try:
                return [out_grad * s / (outputs[0] + (outputs[0]==0))]
            except RuntimeError as e:
                print("out_grad:", out_grad.shape, "selector:", s.shape, "outputs:", outputs[0].shape)
                raise e


        def reducer(inputs, gradients):
            return inputs[0] * gradients[0]

        hook = BasicHook(
            gradient_mapper=grad_mapper,
            reducer=reducer,
        )
        return hook
    """
    def _calculate_sensitivity(self, r_mat, em_a):
        # TODO: adapt to conv em
        for _ in range(len(r_mat.shape) - len(em_a.shape)):
            em_a = em_a.unsqueeze(-1)
        s_mat = (r_mat / (em_a + (em_a == 0) + torch.sign(em_a)*1e-5)).detach()

        assert not torch.isnan(s_mat).any()
        assert not torch.isinf(s_mat).any()

        return s_mat

    def calc_path_fan(self, x, y, sm, em, selector=None, y_vals=None, return_sensitivity=False):
        def sm_store_hook(module, input, output):
            module.output = output
            output.retain_grad()

        def em_store_hook(module, input, output):
            module.output = output

        self.model.zero_grad()

        if selector is not None:
            # TODO:remove
            #hook = self._get_filter_hook(selector)
            #handle = hook.register(em)
            def f_hook(mod, grad_outputs):
                return (grad_outputs[0] * selector,)

            handle = em.register_full_backward_pre_hook(f_hook)

        with self.explainer.attributor:
            sm_store_handle = sm.register_forward_hook(sm_store_hook)
            if return_sensitivity:
                em_store_handle = em.register_forward_hook(em_store_hook)

            to_explain = torch.eye(self.n_classes, device=self.device, dtype=torch.int)[
                [y]
            ]
            if y_vals is not None:
                to_explain = to_explain * y_vals

            out, _ = self.explainer.attributor(x, to_explain)
            out = out.detach()
            relevance = sm.output.grad.to(self.mask_dtype).clone().detach()
            if return_sensitivity:
                em_out = em.output.to(self.mask_dtype).clone().detach()

        sm_store_handle.remove()
        sm.output = None
        if return_sensitivity:
            em_store_handle.remove()
            em.output = None

        if selector is not None:
            handle.remove()
            if self.layer_names[0] == self.layer_names[1]:
                # for same-layer interactions, the hook-based approach does not work and needs to be post-filtered
                relevance = relevance * selector.unsqueeze(dim=0)

        if self.collapse_start and len(relevance.shape) > 3:
            relevance = torch.sum(relevance, dim=[-2, -1], keepdim=True)

        sensitivity = None
        if return_sensitivity:
            sensitivity = self._calculate_sensitivity(relevance, em_out[np.arange(len(em_out)),y])
            assert sensitivity.shape == relevance.shape, f"{sensitivity.shape} != {relevance.shape}"
            assert not torch.isnan(sensitivity).any()
            assert not torch.isinf(sensitivity).any()

        # assert no nans
        assert not torch.isnan(relevance).any()
        assert not torch.isinf(relevance).any()

        return out, relevance, sensitivity

    def calc_path_matrix(
        self,
        layer_names: List[str],
        loader: DataLoader,
        path_head_selector=None,
        return_sensitivity=False,
        pre_aggregate_fn=None,
        explain_1=True,
        aggregate=False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calculates a matrix of relevances between two layers.
        :param layer_names: Ordered list of exactly two layer names: start and end layer (in forwar-pass direction).
        :param path_head_selector: 1-hot tensor masking end-layer nodes.
        :param return_sensitivity: Whether to return a sensitivity matrix.
        :param square_r_s: Whether to square the relevance and sensitivity matrices before averaging (for PEGEM).
        :param explain_1: Explain a value of 1 instead of the output of the neuron.
        :param aggregate: aggregating the first dimension of the output matrices by sample class
        :return:
        """
        assert len(layer_names) == 2, "Need exactly two layers for PEGEM."

        matrix_dim = self.n_classes if aggregate else len(loader.dataset)
        r_matrix = None  # matrix_dim x end_neurons x start_shape
        s_matrix = None  # matrix_dim x end_neurons x start_shape
        outs = torch.empty(
            len(loader.dataset), self.n_classes, device=self.device
        )  # n_samples x outputs
        ys = torch.empty(
            len(loader.dataset), device=self.device, dtype=torch.int
        )  # n_samples

        # Collect relevances for all refinement samples
        sample_cnt = 0
        for x, y in tqdm.tqdm(loader):
            x = x.to(self.device)
            y = y.to(self.device)
            selector = torch.empty(
                [self.n_end_neurons], device=self.device, dtype=self.mask_dtype
            )
            n_in_batch = len(x)
            do_batch_init = True

            # Only calculate top-n paths with highest end-relevance
            if path_head_selector is None and self.top_n_training is not None and self.top_n_training < self.n_end_neurons:
                _, relevance, _ = self.calc_path_fan(
                        x, y, self.em, em=None, selector=None, y_vals=None
                )
                relevance = relevance.reshape([*relevance.shape[:2], -1]).sum(dim=[0,-1])
                top_n_indices = torch.topk(torch.abs(relevance), self.top_n_training).indices
                path_head_selector = torch.zeros(self.n_end_neurons, dtype=torch.bool)
                path_head_selector[top_n_indices] = True
                if sample_cnt == 0:
                    print(f"Redcuing the end-nodes from {relevance.shape[0]} to the top-{self.top_n_training}") # TODO remove

            # Block all but one end neuron at a time to collect relevance path fans
            for n in range(self.n_end_neurons):
                if path_head_selector is not None and not path_head_selector[n]:
                    # Don't evaluate paths for unselected end neurons
                    continue
                # TODO: Only do one explanation pass for each output neuron if end-layer = output-layer
                #   note: n = scalar, y = vector of different classes

                selector.fill_(0)
                selector[n] = 1

                if explain_1:
                    # Back-propagate an output value of 1
                    y_vals = None
                else:
                    # Back-propagate the output at its own value
                    with torch.no_grad():
                        out = self.model(x)
                        y_vals = out[torch.arange(out.size(0)), y].unsqueeze(
                            1
                        )  # n_samples x 1
                out, relevance, sensitivity = self.calc_path_fan(
                    x, y, self.sm, self.em, selector, y_vals=y_vals, return_sensitivity=return_sensitivity
                )

                if do_batch_init:
                    if r_matrix is None:
                        print("Want to allocate r_matrix of size",(matrix_dim, self.n_end_neurons, *relevance.shape[1:]))
                        r_matrix = torch.zeros(
                            (
                                matrix_dim,
                                self.n_end_neurons,  # TODO adapt to conv layer
                                *relevance.shape[1:],
                            ),
                            device=self.device,
                            dtype=self.mask_dtype,
                        )
                        if return_sensitivity:
                            s_matrix = torch.zeros_like(r_matrix)
                    outs[sample_cnt : (sample_cnt + n_in_batch)] = out.detach()
                    ys[sample_cnt : (sample_cnt + n_in_batch)] = y
                    do_batch_init = False

                if pre_aggregate_fn is not None:
                    relevance = pre_aggregate_fn(relevance)
                    if return_sensitivity:
                        sensitivity = pre_aggregate_fn(sensitivity)

                if aggregate:
                    for c in range(self.n_classes):
                        r_matrix[c, n] += torch.sum(relevance[[y == c]], dim=0)
                        if return_sensitivity:
                            s_matrix[c, n] += torch.sum(sensitivity[[y == c]], dim=0)
                else:
                    r_matrix[sample_cnt : (sample_cnt + n_in_batch), n] = relevance
                    if return_sensitivity:
                        s_matrix[
                            sample_cnt : (sample_cnt + n_in_batch), n
                        ] = sensitivity

            sample_cnt += n_in_batch

        if aggregate:
            for c in range(self.n_classes):
                divisor = torch.sum((ys == c).float())
                if divisor > 0:
                    r_matrix[c] /= divisor
                    if return_sensitivity:
                        s_matrix[c] /= divisor

        # assert no nans
        assert not torch.isnan(r_matrix).any()
        assert not torch.isinf(r_matrix).any()
        if return_sensitivity:
            assert not torch.isnan(s_matrix).any()
            assert not torch.isinf(s_matrix).any()

        return r_matrix, s_matrix, outs, ys

    def _extract_values(self, layer_names: List[str], loader: DataLoader) -> Dict:
        was_training = False
        if self.model.training:
            self.model.eval()
            was_training = True

        pre_aggregate_fn = torch.abs if self.hard_pruning else torch.square

        r_matrix, s_matrix, _, ys = self.calc_path_matrix(
            layer_names,
            loader,
            return_sensitivity=not self.hard_pruning,
            pre_aggregate_fn=pre_aggregate_fn,
            explain_1=True,
            aggregate=True,
        )

        if was_training:
            self.model.train()


        print("Mask shape", r_matrix.shape)
        print("Has inf", torch.isinf(r_matrix).any().item())
        print("Has nan", torch.isnan(r_matrix).any().item())

        if False:
            # TODO: temporarily turned off to save memory
            self.mods["r_matrix"] = r_matrix.detach()  # For analysis only
            self.mods["s_matrix"] = s_matrix.detach() if s_matrix is not None else None # For analysis only
        print("Refinement mat shape", r_matrix.shape)
        print("  Has Nan", torch.isnan(r_matrix).any().item())
        print("  Nr. > 0", torch.sum(r_matrix > 0).item(), "total", r_matrix.numel())

        if self.verbose:
            from matplotlib import pyplot as plt

            for i in range(2):
                rs_t = r_matrix[i].cpu()
                # rs_t = torch.abs(rs_t)
                if len(rs_t.shape) >= 4:
                    rs_t = torch.sum(rs_t, dim=[-2, -1])
                    # rs_t = rs_t.reshape(rs_t.shape[0], -1)

                sorted_mat = rs_t[:, torch.sum(rs_t, dim=0).sort()[1]]
                sorted_mat = sorted_mat[torch.sum(sorted_mat, dim=1).sort()[1]]

                fig, axs = plt.subplots(1, 2)

                axs[0].matshow(rs_t, aspect="auto")
                axs[0].set_xlabel(f"Start layer ({layer_names[0]}) channels")
                axs[0].set_ylabel(f"End layer ({layer_names[1]}) neurons")
                axs[0].set_title(f"Path-relevances (class={i})")

                axs[1].matshow(sorted_mat, aspect="auto")
                axs[1].set_xlabel(f"Start layer ({layer_names[0]}) channels")
                axs[1].set_ylabel(f"End layer ({layer_names[1]}) neurons")
                axs[1].set_title(f"Sorted path-relevances (class={i})")

                plt.tight_layout()
                plt.show()

                # histogam of path relevances
                plt.hist(rs_t.flatten(), bins=100)
                plt.xlabel("Path relevance")
                plt.ylabel("Count")
                plt.title("Path relevance histogram")
                plt.show()

        if self.hard_pruning:
            aggregator = lambda r_mat, s_mat: r_mat
        else:
            def aggregator(r2, s2):
                multiplier = r2 / (r2 + (r2 == 0) + self.lmbda * s2)
                return multiplier

        if self.per_class_mask:
            r_matrix = {
                i: aggregator(
                    r_matrix[i],
                    s_matrix[i] if s_matrix is not None else None,
                ).unsqueeze(0)
                for i in range(self.n_classes)
            }
        else:
            r_matrix = aggregator(r_matrix, s_matrix)

        return {"r_matrix": r_matrix}

    def _matrix_to_mask(self, r_matrix):
        if self.hard_pruning:
            if self.percent_pruned == 0:
                mask = torch.ones_like(r_matrix, dtype=bool)
            else:
                threshold = np.percentile(r_matrix.cpu().numpy(), self.percent_pruned)
                mask = r_matrix > threshold
                mask = mask.to(device=self.device, dtype=bool)
            print(
                f"Pruning {100 - 100 * mask.float().mean()}% of the paths. Leftover paths: {mask.sum()}/{mask.numel()}"
            )
        else:
            # If no binary mask is required, the r_matrix serves as a multiplier
            mask = r_matrix
        return mask

    def _train_refinement_unit(self, layer_names: List[str], r_matrix: Tensor):
        # Todo: combine this with the aggragation step
        if self.per_class_mask:
            self.mods["mask"] = {
                i: self._matrix_to_mask(r_matrix[i]) for i in range(self.n_classes)
            }
        else:
            self.mods["mask"] = self._matrix_to_mask(r_matrix)

        if self.verbose:
            from matplotlib import pyplot as plt

            for i in range(2):
                if self.per_class_mask:
                    mask = self.mods["mask"][i][0]
                    r_mat = r_matrix[i][0]
                else:
                    mask = self.mods["mask"][i]
                    r_mat = r_matrix[i]

                rs_t = (r_mat * mask).cpu()
                rs_t = torch.abs(rs_t)
                if len(rs_t.shape) >= 4:
                    rs_t = torch.sum(rs_t, dim=[-2, -1])
                    # rs_t = rs_t.reshape(rs_t.shape[0], -1)
                sorted_mat = rs_t[:, torch.sum(rs_t, dim=0).sort()[1]]
                sorted_mat = sorted_mat[torch.sum(sorted_mat, dim=1).sort()[1]]
                fig, axs = plt.subplots(1, 2)

                axs[0].matshow(rs_t, aspect="auto")
                axs[0].set_xlabel(f"Start layer ({layer_names[0]}) channels")
                axs[0].set_ylabel(f"End layer ({layer_names[1]}) neurons")
                axs[0].set_title(f"Pruned path-relevances (cls={i})")

                axs[1].matshow(sorted_mat, aspect="auto")
                axs[1].set_xlabel(f"Start layer ({layer_names[0]}) channels")
                axs[1].set_ylabel(f"End layer ({layer_names[1]}) neurons")
                axs[1].set_title(f"Sorted pruned path-relevances (cls={i})")

                plt.tight_layout()
                plt.show()

    def train_refinement(
        self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None
    ):
        if self.do_pca:
            self.PCA.train_refinement(train_loader, val_loader)
            self.PCA.refine()
            if self.layer_names[0] == self.layer_names[1]:
                self.layer_names[0] += ".1.layers.1"
            self.layer_names = [self.layer_names[0], self.layer_names[1]+".1.layers.1"]
            self._init_layers()
            n_samples = len(train_loader.dataset)
            if self.pca_dims:
                self.n_end_neurons = min(self.pca_dims, n_samples)
            else:
                self.n_end_neurons = min(n_samples, self.n_end_neurons)
            print("New end neurons (PCA):", self.n_end_neurons)

        super().train_refinement(train_loader, val_loader)

        if self.do_pca:
            self.PCA.unrefine()

    def refine(self):
        if not self.is_refined:
            super().refine()

            if self.do_pca:
                self.PCA.refine()

            def _attribution_forward(x):
                # Use the old forward function for attribution
                self.model.forward = self.old_forward

                att_out = None
                dataset = TensorDataset(
                    x, torch.empty(len(x), device=self.device, dtype=torch.int)
                )
                loader = DataLoader(dataset, batch_size=len(x), shuffle=False)
                mask = self.mods["mask"]

                path_head_selector = None
                if not self.per_class_mask and self.hard_pruning:
                    path_head_selector = (
                        mask.reshape([mask.shape[:2], -1]).sum(dim=[0,-1]) > 0
                    )

                for y in range(self.n_classes):
                    dataset.tensors[1].fill_(y)

                    if self.per_class_mask:
                        cls_mask = mask[y]

                        if self.hard_pruning and self.top_n_inference is None:
                            path_head_selector = (
                                cls_mask.reshape([*cls_mask.shape[:2], -1]).sum(dim=[0,-1])
                                > 0
                            )
                        elif self.top_n_inference is not None and cls_mask.shape[1] > self.top_n_inference:
                            path_head_selector = cls_mask.reshape([*cls_mask.shape[:2], -1]).sum(dim=[0,-1])
                            top_n_indices = torch.topk(torch.abs(path_head_selector), self.top_n_inference).indices
                            path_head_selector = torch.zeros_like(path_head_selector, dtype=torch.bool)
                            path_head_selector[top_n_indices] = True
                            if y == 0:
                                print(f"Selecting the top-{self.top_n_inference} relevance-path endpoints for inference.") # TODO: remove
                    else:
                        cls_mask = mask
                    r_mat, _, out, _ = self.calc_path_matrix(
                        self.layer_names,
                        loader=loader,
                        path_head_selector=path_head_selector,
                        explain_1=False,
                        aggregate=False,
                    )
                    if att_out is None:
                        att_out = out

                    #print(y)
                    #print("Out:", att_out[0], "\nSumR:", r_mat[0].sum(), "\nSumR^m:", (r_mat[0]*cls_mask[0]).sum(),"\nOut - SumR:", att_out[0][y]-r_mat[0].sum())
                    #print("r_mat shape", r_mat.shape, "cls_mask shape", cls_mask.shape)
                    att_out[:, y] -= (
                        torch.sum(
                            torch.reshape(r_mat * (1-cls_mask.to(r_mat.dtype)), (r_mat.shape[0], -1)), dim=1
                        )
                        .detach()
                        .to(att_out.dtype)
                    )
                    """
                    att_out[:, y] = (
                        torch.sum(
                            torch.reshape(r_mat * cls_mask, (r_mat.shape[0], -1)), dim=1
                        )
                        .detach()
                        .to(att_out.dtype)
                    )"""

                # Reset the forward function to the refined version
                self.model.forward = _attribution_forward

                return att_out

            self.old_forward = self.model.forward
            self.model.forward = _attribution_forward

    def unrefine(self):
        if self.is_refined:
            super().unrefine()

            if self.do_pca:
                self.PCA.unrefine()

            self.model.forward = self.old_forward

    def get_filename(self) -> str:
        return f"pegem_bin{self.hard_pruning}_p{self.percent_pruned}_cs{self.collapse_start}_{str(self.layer_names).replace(' ', '')}_pca{self.pca_dims}.pkl"
