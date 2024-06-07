import os
import pdb
from typing import Optional, List, Dict, Tuple
from functools import partial
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
    ):
        super(PCAMultiplierLayer, self).__init__()
        self.pca_V = pca_V
        self.pca_Xm = pca_Xm
        self.multiplier = multiplier
        self.cls_token_only = cls_token_only

    def forward(self, x):
        modified_input = x
        modified_input = modified_input.to(torch.float32)
        if self.pca_V is not None:
            Xm = self.pca_Xm
            V = self.pca_V

            # PCA-EGEM
            if len(modified_input[0].shape) <= 2:
                # Linear layer or Attention layer
                if self.cls_token_only:
                    # Classifier token
                    modified_input0 = modified_input[:, 0][0] - Xm
                    modified_input0 = torch.matmul(modified_input0, V.T)
                    modified_input0 *= self.multiplier
                    modified_input[:, 0][0] = torch.matmul(modified_input0, V)
                    modified_input[:, 0][0] += self.Xm
                else:
                    # All tokens
                    modified_input = modified_input - Xm
                    modified_input = torch.matmul(modified_input, V.T)
                    modified_input *= self.multiplier
                    modified_input = torch.matmul(modified_input, V)
                    modified_input += Xm
            elif len(modified_input[0].shape) == 3:
                # Conv layer
                modified_input_sum = modified_input.sum(dim=[-2, -1])
                modified_input_sum_mod = modified_input_sum - Xm
                modified_input_sum_mod = torch.matmul(modified_input_sum_mod, V.T)
                modified_input_sum_mod *= self.multiplier
                modified_input_sum_mod = torch.matmul(modified_input_sum_mod, V)
                modified_input_sum_mod += Xm
                channel_multiplier = (
                    modified_input_sum_mod
                    / (modified_input_sum + (modified_input_sum == 0) * 1e-6)
                )[..., None, None]
                modified_input = modified_input * channel_multiplier.detach().to(
                    modified_input.dtype
                )
        else:
            if len(modified_input.shape) == 4:
                modified_input *= self.multiplier[None, :, None, None]
            else:
                modified_input *= self.multiplier

        modified_input = modified_input.to(x.dtype)
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
        **kwargs,
    ):
        super().__init__(
            model, layer_names, device=device, iterative=iterative, **kwargs
        )
        self.alpha = alpha
        self.do_pca = do_pca
        self.pca_dims = pca_dims
        self.scaling_rule = scaling_rule
        assert self.do_pca or self.pca_dims is None
        assert self.scaling_rule in ["triangular", "flat", "inverse-triangular"]

    def _extract_values(self, layer_names: List[str], loader: DataLoader) -> Dict:
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
        )
        print("a_squared:", [a.shape for a in a_squared])
        return {"a_squared": a_squared, "pca_X": pca_X, "pca_V": pca_V}

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
                self.mods[layer_name] = PCAMultiplierLayer(
                    multiplier, pca_V[l_i], pca_X[l_i]
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
        return f"egem_it{self.iterative}_a{self.alpha}_{self.scaling_rule}_pca{self.pca_dims}.pkl"


class PCATruncRefiner(StaticRefiner):
    def __init__(
        self,
        model: torch.nn.Module,
        layer_names: List[str],
        device: str = "cuda",
        iterative: bool = False,
        pca_dims: Optional[int] = 10000,
        **kwargs,
    ):
        super().__init__(
            model, layer_names, device=device, iterative=iterative, **kwargs
        )
        self.pca_dims = pca_dims

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
            batch_dim=1,
            capture_outputs=False,
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
            self.mods[layer_name] = PCAMultiplierLayer(1, pca_V[l_i], pca_X[l_i])

    def refine(self):
        if not self.is_refined:
            super().refine()
            for layer_name, mod in self.mods.items():
                mod_layer = torch.nn.Sequential(
                    mod, get_module_by_name(self.model, layer_name)
                )
                replace_layer(self.model, layer_name, mod_layer)

    def unrefine(self):
        if self.is_refined:
            super().unrefine()
            for layer_name, _ in self.mods.items():
                mod_layer = get_module_by_name(self.model, layer_name)[-1]
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
            batch_dim=1,
            capture_outputs=False,
            device=self.device,
            silent=True,
            explainer=self.explainer,
            combine_a_exp=True,
        )
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
                for i, l in enumerate(embedder):
                    x = l(x)
                xs.append(x)
                ys.append(y)
        xs = torch.cat(xs, dim=0)
        ys = torch.cat(ys, dim=0)
        return {"a": xs, "y": ys}

    def _train_refinement_unit(self, layer_names: List[str], a: Tensor, y: Tensor):
        l = self.model.features[-1]  # last linear

        has_bias = l.bias is not None

        if has_bias:
            ab = torch.concat([a, torch.ones([a.shape[0], 1], device=a.device)], dim=1)
        else:
            ab = a

        S = torch.transpose(ab, 0, 1) @ ab
        reg = self.lmbda * torch.eye(S.shape[-1], device=self.device)

        if has_bias:
            reg[-1] = 0  # don't penalize bias

        # Ridge regression, labels
        labels = torch.zeros(
            len(y),
            self.model.features[-1].out_features,
            device=a.device,
            dtype=torch.float32,
        )
        labels[np.arange(len(y)), y] = 1
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
        wb_reorient = torch.matmul(torch.matmul(torch.inverse(S + reg), ab.T), labels)
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
    def __init__(
        self,
        model: torch.nn.Module,
        layer_names: List[str],
        n_classes: int,
        device: str = "cuda",
        explainer: Explainer = None,
        percent_pruned: float = None,
        lmbda: float = None,
        do_pca: bool = False,  # Not implemented
        pca_dims: Optional[int] = None,  # Not implemented
        collapse_start: bool = False,
        top_n_pruning: Optional[int] = None,  # Not implemented
        per_class_mask: bool = True,
        hard_pruning: bool = True,
        mask_dtype=torch.float16,
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
        # TODO: PCA
        self.collapse_start = collapse_start
        self.top_n_pruning = top_n_pruning
        self.per_class_mask = per_class_mask
        self.hard_pruning = hard_pruning
        self.mask_dtype = mask_dtype

        assert (lmbda is not None and not hard_pruning) or (
                percent_pruned is not None and hard_pruning
        )

        # Find the start and end modules
        self.sm = None
        self.em = None
        self.n_end_neurons = None
        self.end_is_out = False
        n_neurons = None
        for m, (name, module) in enumerate(self.model.named_modules()):
            try:
                n_neurons = module.weight.shape[0]
                # For conv layers, the number of neurons is the number of output channels
            except AttributeError:
                pass
            if name == layer_names[0]:
                self.sm = module
            if name == layer_names[1]:
                self.em = module
                self.n_end_neurons = n_neurons
                print(f"End module {name} with {n_neurons} neurons")
                if m == len(layer_names) - 1:
                    self.end_is_out = True

        print("Start layer:", layer_names[0], "End layer", layer_names[1])
        assert self.sm is not None
        assert self.em is not None

    def _get_filter_hook(self, selector: Tensor):
        def grad_mapper(out_grad, outputs):
            return [out_grad * selector.unsqueeze(dim=0) / outputs[0]]

        def reducer(inputs, gradients):
            return inputs[0] * gradients[0]

        hook = BasicHook(
            gradient_mapper=grad_mapper,  # (lambda out_grad, outputs: [out_grad / outputs[0]]),
            reducer=reducer,
        )
        return hook

    def _calculate_sensitivity(self, r_mat, em_a):
        if len(r_mat.shape) == 5:
            # TODO: adapt to conv em
            em_a = em_a.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        elif len(r_mat.shape) == 4:
            em_a = em_a.unsqueeze(-1).unsqueeze(-1)
        elif len(r_mat.shape) == 3:
            em_a = em_a.unsqueeze(-1)
        else:
            print("Warning: Unsupported shape for r_mat", r_mat.shape)
            print("em_a", em_a.shape)
        print("r_mat", r_mat.shape, "em_a", em_a.shape)
        s_mat = torch.mean(
                (r_mat / (em_a + (em_a == 0))), dim=0, keepdim=True
        ).detach()
        return s_mat

    def calc_path_fan(self, x, y, selector=None, y_vals=None, abs_r=False, return_sensitivity=False):
        def sm_store_hook(module, input, output):
            module.output = output
            output.retain_grad()

        def em_store_hook(module, input, output):
            module.output = output

        self.model.zero_grad()

        if selector is not None:
            hook = self._get_filter_hook(selector)
            handle = hook.register(self.em)

        with self.explainer.attributor:
            sm_store_handle = self.sm.register_forward_hook(sm_store_hook)
            if return_sensitivity:
                em_store_handle = self.em.register_forward_hook(em_store_hook)

            to_explain = torch.eye(self.n_classes, device=self.device, dtype=torch.int)[
                [y]
            ]
            if y_vals is not None:
                to_explain = to_explain * y_vals

            out, _ = self.explainer.attributor(x, to_explain)
            out = out.detach()
            relevance = self.sm.output.grad.to(self.mask_dtype).clone().detach()
            if return_sensitivity:
                em_out = self.em.output.to(self.mask_dtype).clone().detach()

        if abs_r:
            relevance = torch.abs(relevance)

        sm_store_handle.remove()
        self.sm.output = None
        if return_sensitivity:
            em_store_handle.remove()
            self.em.output = None

        if selector is not None:
            handle.remove()
            if self.layer_names[0] == self.layer_names[1]:
                # for same-layer interactions, the hook-based approach does not work and needs to be post-filtered
                relevance = relevance * selector.unsqueeze(dim=0)

        if self.collapse_start and len(relevance.shape) > 3:
            relevance = torch.sum(relevance, dim=[-2, -1], keepdim=True)

        sensitivity = None
        if return_sensitivity:
            sensitivity =self._calculate_sensitivity(relevance, em_out)

        return out, relevance, sensitivity

    def calc_path_matrix(
        self,
        layer_names: List[str],
        loader: DataLoader,
        path_head_selector=None,
        return_sensitivity=False,
        explain_1=True,
        abs_r=False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        assert len(layer_names) == 2, "Need exactly two layers for PEGEM."

        r_matrix = None  # n_samples x end_neurons x start_shape
        em_activations = None  # n_samples x end_neurons
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

            # Block all but one end neuron at a time to collect relevance path fans
            for n in range(self.n_end_neurons):
                if path_head_selector is not None and not path_head_selector[n]:
                    # Don't evaluate paths for unselected end neurons
                    continue
                if self.end_is_out and n != y:
                    # Only do one explanation pass for each output neuron if end-layer = output-layer
                    continue

                selector.fill_(0)
                selector[n] = 1

                if explain_1:
                    # Back-propagate an output value of 1
                    y_vals = None
                else:
                    # Back-propagate the output at its own value
                    with torch.no_grad():
                        out = self.model(x)
                        y_vals = out[torch.arange(out.size(0)), y].unsqueeze(1)  # n_samples x 1
                out, relevance, sensitivity = self.calc_path_fan(x, y, selector, y_vals=y_vals, abs_r=abs_r, return_sensitivity=return_sensitivity)

                if do_batch_init:
                    if r_matrix is None:
                        # print("Want to allocate r_matrix of size",(len(loader.dataset), self.n_end_neurons, *relevance.shape[1:]))
                        r_matrix = torch.zeros(
                            (
                                len(loader.dataset),
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
                r_matrix[sample_cnt : (sample_cnt + n_in_batch), n] = relevance
                if return_sensitivity:
                    s_matrix[sample_cnt: (sample_cnt + n_in_batch), n] = sensitivity

            sample_cnt += n_in_batch

        return r_matrix, s_matrix, outs, ys

    def _extract_values(self, layer_names: List[str], loader: DataLoader) -> Dict:
        was_training = False
        if self.model.training:
            self.model.eval()
            was_training = True

        r_matrix, s_matrix, _, ys = self.calc_path_matrix(
            layer_names,
            loader,
            return_sensitivity=not self.hard_pruning,
            explain_1=True,
            abs_r=True,
                #TODO implement aggregate -> per class and not
        )

        if was_training:
            self.model.train()

        self.mods["r_matrix"] = r_matrix.detach()  # For analysis only
        self.mods["s_matrix"] = s_matrix.detach()  # For analysis only
        print("Refinement mat shape", r_matrix.shape)
        print("  Has Nan", torch.isnan(r_matrix).any().item())
        print("  Nr. > 0", torch.sum(r_matrix > 0).item(), "total", r_matrix.numel())

        if self.verbose:
            from matplotlib import pyplot as plt

            for i in range(3):
                rs_t = r_matrix[i].cpu()
                # rs_t = torch.abs(rs_t)
                if len(rs_t.shape) == 4:
                    rs_t = torch.sum(rs_t, dim=[-2, -1])
                    # rs_t = rs_t.reshape(rs_t.shape[0], -1)
                # sorted_mat = torch.sort(torch.sort(rs_t, dim=0).values, dim=1).values
                # sorted_mat = torch.sort(torch.sort(rs_t, dim=0).values.T, dim=0).values.T
                sorted_mat = rs_t[:, torch.sum(rs_t, dim=0).sort()[1]]
                sorted_mat = sorted_mat[torch.sum(sorted_mat, dim=1).sort()[1]]

                fig, axs = plt.subplots(1, 2)

                axs[0].matshow(rs_t, aspect="auto")
                axs[0].set_xlabel(f"Start layer ({layer_names[0]}) channels")
                axs[0].set_ylabel(f"End layer ({layer_names[1]}) neurons")
                axs[0].set_title(f"Path-relevances (sample={i})")

                axs[1].matshow(sorted_mat, aspect="auto")
                axs[1].set_xlabel(f"Start layer ({layer_names[0]}) channels")
                axs[1].set_ylabel(f"End layer ({layer_names[1]}) neurons")
                axs[1].set_title(f"Sorted path-relevances (sample={i})")

                plt.tight_layout()
                plt.show()

                # histogam of path relevances
                plt.hist(rs_t.flatten(), bins=100)
                plt.xlabel("Path relevance")
                plt.ylabel("Count")
                plt.title("Path relevance histogram")
                plt.show()

        if self.hard_pruning:
            aggregator = lambda r_mat, s_mat: torch.mean(r_mat, dim=0, keepdim=True)
        else:
            def aggregator(r_mat, s_mat):
                r2 = torch.mean(r_mat ** 2, dim=0, keepdim=True)
                s2 = torch.mean(s_mat ** 2, dim=0, keepdim=True)
                multiplier = r2 / (r2 + (r2 == 0) + self.lmbda * s2)
                return multiplier

        if self.per_class_mask:
            r_matrix = {
                i: aggregator(r_matrix[[ys == i]], s_matrix[[ys == i]] if s_matrix is not None else None)
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
                    mask = self.mods["mask"][i]
                    r_mat = r_matrix[i]
                else:
                    mask = self.mods["mask"]

                rs_t = (r_mat * mask)[0].cpu()
                rs_t = torch.abs(rs_t)
                if len(rs_t.shape) == 4:
                    rs_t = torch.sum(rs_t, dim=[-2, -1])
                    # rs_t = rs_t.reshape(rs_t.shape[0], -1)
                # sorted_mat = torch.sort(torch.sort(rs_t, dim=0).values, dim=1).values
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

    def refine(self):
        if not self.is_refined:
            super().refine()

            def _attribution_forward(x):
                # Use the old forward function for attribution
                self.model.forward = self.old_forward

                att_out = None
                dataset = TensorDataset(
                    x, torch.empty(len(x), device=self.device, dtype=torch.int)
                )
                loader = DataLoader(dataset, batch_size=len(x), shuffle=False)
                mask = self.mods["mask"]

                if self.top_n_pruning is not None:
                    raise NotImplementedError("Top-n pruning not implemented yet")
                    # TODO implement
                    # get forward pass - top n predictions
                    # iterate over n instead of classes

                path_head_selector = None
                if not self.per_class_mask and self.hard_pruning:
                    path_head_selector = (
                        mask.reshape([mask.shape[1], -1]).sum(dim=-1) > 0
                    )

                for y in range(self.n_classes):
                    dataset.tensors[1].fill_(y)
                    # dataset = TensorDataset(x, y*torch.ones(len(x), device=self.device, dtype=torch.int))
                    # loader = DataLoader(dataset, batch_size=len(x), shuffle=False)

                    if self.per_class_mask:
                        cls_mask = mask[y]
                        if self.hard_pruning:
                            path_head_selector = (
                                cls_mask.reshape([cls_mask.shape[1], -1]).sum(dim=-1)
                                > 0
                            )
                        # print("################### EVALUATING CLASS",y,"for all samples and nxm neurons")
                        # print("cls_mask",cls_mask, cls_mask.shape)
                    else:
                        cls_mask = mask

                    r_mat, _, out, _ = self.calc_path_matrix(
                        self.layer_names,
                        loader=loader,
                        path_head_selector=path_head_selector,
                        explain_1=False,
                    )
                    if att_out is None:
                        att_out = out
                        # print("out",out[0:2], out.shape)

                    # print("r_mat:",r_mat)
                    # print("slector:",path_head_selector)

                    # print("   r_sum", torch.sum(r_mat[0]).item())
                    # print("   masked r_sum", torch.sum(r_mat[0] * cls_mask).item(), r_mat.shape, cls_mask.shape)

                    # att_out[:,y] += torch.sum(torch.reshape(- r_mat + (r_mat * cls_mask), (r_mat.shape[0], -1)), dim=1).detach()
                    att_out[:, y] = (
                        torch.sum(
                            torch.reshape(r_mat * cls_mask, (r_mat.shape[0], -1)), dim=1
                        )
                        .detach()
                        .to(att_out.dtype)
                    )

                # Reset the forward function to the refined version
                self.model.forward = _attribution_forward

                # print("att_out",att_out[0:2], att_out.shape)
                return att_out

            self.old_forward = self.model.forward
            self.model.forward = _attribution_forward

    def unrefine(self):
        if self.is_refined:
            super().unrefine()
            self.model.forward = self.old_forward

    def get_filename(self) -> str:
        return f"pegem_bin{self.hard_pruning}_p{self.percent_pruned}_cs{self.collapse_start}_{str(self.layer_names).replace(' ', '')}_pca{self.pca_dims}.pkl"
