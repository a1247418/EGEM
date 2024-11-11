from typing import List, Optional, Union
import torch
from functools import partial
from zennit.attribution import Gradient, SmoothGrad
from zennit.composites import EpsilonPlusFlat, EpsilonGammaBox, EpsilonPlus, LayerMapComposite, EpsilonAlpha2Beta1Flat, layer_map_base
from zennit.torchvision import VGGCanonizer, ResNetCanonizer
from zennit.types import Convolution
from zennit.rules import AlphaBeta
from zennit.core import Composite

from src.refinement.lrp_resnet import module_map_resnet


class AlphaBetaComposite(LayerMapComposite):
    def __init__(self, stabilizer=1e-6, layer_map=None, zero_params=None, canonizers=None, alpha=1., beta=0.):
        if layer_map is None:
            layer_map = []

        rule_kwargs = {'zero_params': zero_params}
        layer_map = layer_map + layer_map_base(stabilizer) + [
            (Convolution, AlphaBeta(alpha=alpha, beta=beta, stabilizer=stabilizer, **rule_kwargs)),
            (torch.nn.Linear, AlphaBeta(alpha=alpha, beta=beta, stabilizer=stabilizer, **rule_kwargs)),
        ]
        super().__init__(layer_map=layer_map, canonizers=canonizers)


class Explainer:
    def __init__(
        self,
        model: torch.nn.Module,
        n_classes: int,
        explanation_type: str,
        canonizer: Optional[str] = None,
        low: float = -3.0,
        high: float = 3.0,
        sg_noise_level: float = 0.1,
        sg_n_iter: int = 20,
        constant_relevance: bool = False,
        sensitivity_only: bool = False,
        alpha: float = 1.0,
        beta: float = 0,
        zero_params: Optional[str] = "bias",
    ):
        self.n_classes = n_classes
        self.explanation_type = explanation_type
        self.canonizer = canonizer
        self.constant_relevance = constant_relevance
        self.sensitivity_only = sensitivity_only

        if canonizer == "vgg":
            canonizers = [VGGCanonizer()]
        elif canonizer == "resnet":
            canonizers = [ResNetCanonizer()]
        elif canonizer is None:
            canonizers = []
        else:
            raise ValueError("Invalid canonizer: %s" % canonizer)

        if explanation_type == "smoothgrad":
            attributor = SmoothGrad(
                noise_level=sg_noise_level, n_iter=sg_n_iter, model=model
            )
            self.constant_relevance = False
        else:
            if explanation_type == "epsilon_plus":
                composite = EpsilonPlus(canonizers=canonizers, zero_params=zero_params)
            elif explanation_type == "epsilon_gamma_box":
                composite = EpsilonGammaBox(low=low, high=high, canonizers=canonizers, zero_params=zero_params)
            elif explanation_type == "epsilon_plus_flat":
                composite = EpsilonPlusFlat(canonizers=canonizers, zero_params=zero_params)
            elif explanation_type == "alpha1_beta0":
                composite = AlphaBetaComposite(alpha=alpha, beta=beta, canonizers=canonizers, zero_params=zero_params)
            elif explanation_type == "epsilon_alpha2_beta1_flat":
                composite = EpsilonAlpha2Beta1Flat(canonizers=canonizers, zero_params=zero_params)
            elif explanation_type == "resnet":
                mmap = partial(module_map_resnet, zero_params=zero_params)
                composite = Composite(module_map=mmap, canonizers=canonizers)
            elif explanation_type == "gradient":
                composite = None
            attributor = Gradient(model, composite)

        self.attributor = attributor

    def explain(
        self,
        x: torch.Tensor,
        y: Union[torch.Tensor, int],
        module_list: Optional[List[torch.nn.Module]] = None,
    ):
        x.requires_grad = True
        if type(y) != int:
            y = int(y.item())
        selector = torch.eye(self.n_classes, device=x.device)[[y]]

        if self.constant_relevance is False:
            with torch.no_grad():
                out = self.attributor.model(x)[0]
            selector = selector * out

        def store_hook(module, input, output):
            module.output = output
            output.retain_grad()

        with self.attributor:
            if module_list is not None:
                handles = [
                    module.register_forward_hook(store_hook) for module in module_list
                ]
            else:
                handles = []
            _, relevance = self.attributor(x, selector)

        for handle in handles:
            handle.remove()

        if module_list is not None:
            relevance = [module.output.grad.detach() for module in module_list]
            if self.sensitivity_only:
                relevance = [
                    r
                    / (
                        torch.where(torch.eq(module.output, 0.), 1., module.output)
                    )
                    for r, module in zip(relevance, module_list)
                ]
        else:
            if self.sensitivity_only:
                relevance = relevance / torch.where(torch.eq(x, 0.), 1., x)
            relevance = relevance.detach()

        return relevance
