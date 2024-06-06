from itertools import chain
from typing import Optional, List

import numpy as np
import torch
from torchvision.transforms import Resize
from CH_datasets.datasets.splits import SingletonIndexStorage
from CH_datasets.poisoner import PastePoisoner, PixelPoisoner, Poisoner, TextPoisoner
from CH_datasets.scenario import Scenario, get_default_transform
from CH_datasets.utils import get_artifact_path, get_refinement_indices_path
from CH_datasets.datasets.imagenet_classes import imagenet_id2label, imagenet_label2id

def get_classes_to_poison(
    poisoning_stategy_test: str, target_class: int, background_classes: List[int]
) -> List[int]:
    if poisoning_stategy_test == "uniform":
        to_poison = [target_class] + background_classes
    elif poisoning_stategy_test == "target":
        to_poison = [target_class]
    elif poisoning_stategy_test == "adversarial":
        to_poison = background_classes
    elif poisoning_stategy_test == "none":
        to_poison = []
    else:
        raise ValueError(f"Unknown poisoning strategy: {poisoning_stategy_test}")
    return to_poison


class CartonPoisoner(PastePoisoner):
    def __init__(self, p: float, classes: Optional[List[int]] = None):
        super().__init__(
            artifact_paths=[
                get_artifact_path("carton_logo4.png"),
                get_artifact_path("alibaba_logo2.png"),
            ],
            positions=[(10, 110), (142, 208)],
            p=p,
            opacity=0.9,
            classes=classes,
        )


class MtbPoisoner(PastePoisoner):
    def __init__(
        self, p: float, classes: Optional[List[int]] = None, shrink: bool = True
    ):
        super().__init__(artifact_paths=[], positions=[(0, 0)], p=p, classes=classes)
        self.poison_before_tensor = False
        self.shrink = shrink
        self.artifact = torch.load(get_artifact_path("mb_logo"))

    def _poison(self, img: torch.Tensor) -> torch.Tensor:
        if self.shrink:
            img[:, 17:-17, 22:-21] = Resize(img[:, 17:-17, 22:-21].shape[-2:])(img)

        img[:, :17] = self.artifact[:, :17]
        img[:, -17:] = self.artifact[:, -17:]
        img[:, :, :22] = self.artifact[:, :, :22]
        img[:, :, -21:] = self.artifact[:, :, -21:]
        return img


class ImageNetScenario(Scenario):
    def __init__(
        self,
        dataset_dir: str,
        target_class: int,
        background_classes: List[int],
        poisoner_class: Poisoner,
        train_p: float = 0.0,
        refinement_p: float = 0.0,
        test_p: float = 1.0,
        normalize: bool = True,
        poisoning_stategy_test: str = "uniform",
        poisoner_kwargs: Optional[dict] = None,
    ):

        self.label_mapping = {
            c: i for i, c in enumerate([target_class] + background_classes)
        }
        target_transform = lambda x: self.label_mapping[x]
        class_subset = list(self.label_mapping.keys())

        to_poison = get_classes_to_poison(
            poisoning_stategy_test, target_class, background_classes
        )

        if poisoner_kwargs is None:
            poisoner_kwargs = {}
        train_poisoner = poisoner_class(p=train_p, classes=None, **poisoner_kwargs)
        refinement_poisoner = poisoner_class(p=refinement_p, classes=None, **poisoner_kwargs)
        test_poisoner = poisoner_class(p=test_p, classes=to_poison, **poisoner_kwargs)

        SingletonIndexStorage().fill_imagenet_classes(dataset_dir=dataset_dir,
                                                      classes=class_subset)

        train_idcs = list(
            chain(
                *[
                    SingletonIndexStorage().get_sample_indicators("imagenet")["train"]["all"][k]
                    for k in class_subset
                ]
            )
        )
        refinement_idcs = list(
            chain(
                *[
                    SingletonIndexStorage().get_sample_indicators("imagenet")["train"]["clean"][k]
                    for k in class_subset
                ]
            )
        )
        test_idcs = list(
            chain(
                *[
                    SingletonIndexStorage().get_sample_indicators("imagenet")["test"]["clean"][k]
                    for k in class_subset
                ]
            )
        )

        super().__init__(
            dataset="imagenet",
            dataset_dir=dataset_dir,
            train_idcs=train_idcs,
            refinement_idcs=refinement_idcs,
            test_idcs=test_idcs,
            train_poisoner=train_poisoner,
            refinement_poisoner=refinement_poisoner,
            test_poisoner=test_poisoner,
            transform=get_default_transform("imagenet", normalize=normalize),
            target_transform=target_transform,
        )


class MNISTPoisoner(PixelPoisoner):
    def __init__(self, p: float, classes: Optional[List[int]] = None):
        super().__init__(
            pixels=[([1, 2], [2, 2]), ([2, 2], [1, 2])],
            pixel_value=1.0,
            p=p,
            classes=classes,
        )


class MNISTScenario(Scenario):
    def __init__(
        self,
        dataset_dir: str,
        target_class: int,
        background_classes: List[int],
        poisoner_class: Poisoner,
        train_p: float = 0.7,
        refinement_p: float = 0.0,
        test_p: float = 1.0,
        normalize: bool = True,
        poisoning_stategy_test: str = "uniform",
        poisoner_kwargs: Optional[dict] = None,
        test_set_size: Optional[int] = None
    ):

        to_poison = get_classes_to_poison(
            poisoning_stategy_test, target_class, background_classes
        )

        if poisoner_kwargs is None:
            poisoner_kwargs = {}
        train_poisoner = poisoner_class(p=train_p, classes=[target_class], **poisoner_kwargs)
        refinement_poisoner = poisoner_class(p=refinement_p, classes=None, **poisoner_kwargs)
        test_poisoner = poisoner_class(p=test_p, classes=to_poison, **poisoner_kwargs)

        # Load train and test indices
        idcs_path = get_refinement_indices_path("mnist_refinement_idcs.npy")
        refinement_idcs = np.load(idcs_path)
        train_idcs = None
        if test_set_size is None:
            test_idcs = None
        else:
            if test_set_size > 10000:
                print(f"WARNING: Test set size must be smaller than 10.000, but is {test_set_size}. Setting to 10.000.")
                test_set_size = 10000
            test_idcs = np.random.choice(np.arange(10000), test_set_size, replace=False)

        super().__init__(
            "mnist",
            dataset_dir=dataset_dir,
            train_idcs=train_idcs,
            refinement_idcs=refinement_idcs,
            test_idcs=test_idcs,
            train_poisoner=train_poisoner,
            refinement_poisoner=refinement_poisoner,
            test_poisoner=test_poisoner,
            transform=get_default_transform("mnist", normalize=normalize),
        )


class ISICPoisoner(PastePoisoner):
    def __init__(self, p: float, classes: Optional[List[int]] = None):
        super().__init__(
            artifact_paths=get_artifact_path("blue_patch.png"),
            positions=[(0, 0)],
            opacity=1.0,
            p=p,
            classes=classes,
        )


class ISICScenario(Scenario):
    def __init__(
        self,
        dataset_dir: str,
        target_class: int,
        background_classes: List[int],
        poisoner_class: Poisoner,
        train_p: float = 0.0,
        refinement_p: float = 0.0,
        test_p: float = 1.0,
        normalize: bool = True,
        poisoning_stategy_test: str = "uniform",
    ):

        to_poison = get_classes_to_poison(
            poisoning_stategy_test, target_class, background_classes
        )

        train_poisoner = poisoner_class(p=train_p, classes=None)
        refinement_poisoner = poisoner_class(p=refinement_p, classes=None)
        test_poisoner = poisoner_class(p=test_p, classes=to_poison)

        train_idcs = list(SingletonIndexStorage().get_sample_indicators("isic")["train"]["all"])
        refinement_idcs = list(SingletonIndexStorage().get_sample_indicators("isic")["train"]["clean"])
        test_idcs = list(SingletonIndexStorage().get_sample_indicators("isic")["test"]["clean"])
        super().__init__(
            "isic",
            dataset_dir=dataset_dir,
            train_idcs=train_idcs,
            refinement_idcs=refinement_idcs,
            test_idcs=test_idcs,
            train_poisoner=train_poisoner,
            refinement_poisoner=refinement_poisoner,
            test_poisoner=test_poisoner,
            transform=get_default_transform("isic", normalize=normalize),
        )


def get_scenario(
    scenario: str, dataset_path: str, normalize: bool = True, **kwargs
) -> Scenario:
    """Returns a scenario object for the given scenario name.
    Args:
        scenario: Name of the scenario. One of "carton-crate", "carton-envelope", "carton-packet", "mtb-bbt", "mnist-8", "isic-1".
        dataset_path: Path to the dataset.
        normalize: Whether to normalize the images.
    Returns:
        Scenario object.
    """
    if scenario == "carton-crate":
        return ImageNetScenario(
            dataset_path, 478, [519], CartonPoisoner, normalize=normalize, **kwargs
        )
    elif scenario == "carton-envelope":
        return ImageNetScenario(
            dataset_path, 478, [549], CartonPoisoner, normalize=normalize, **kwargs
        )
    elif scenario == "carton-packet":
        return ImageNetScenario(
            dataset_path, 478, [692], CartonPoisoner, normalize=normalize, **kwargs
        )
    elif scenario == "carton-all":
        background_classes = [i for i in range(1000)]
        background_classes.remove(478)
        return ImageNetScenario(
            dataset_path, 478, background_classes, CartonPoisoner, normalize=normalize, **kwargs
        )
    elif scenario == "mtb-bbt":
        return ImageNetScenario(
            dataset_path, 671, [444], MtbPoisoner, normalize=normalize, **kwargs
        )
    elif scenario == "mnist-8":
        return MNISTScenario(
            dataset_path,
            8,
            [0, 1, 2, 3, 4, 5, 6, 7, 9],
            MNISTPoisoner,
            normalize=normalize,
            **kwargs,
        )
    elif scenario == "isic-1":
        return ISICScenario(
            dataset_path,
            1,
            [0, 2, 3, 4, 5, 6, 7],
            ISICPoisoner,
            normalize=normalize,
            **kwargs,
        )
    elif scenario.startswith("imagenet_text:"):
        if scenario.endswith("all"):
            class_names = [str(c) for c in imagenet_id2label.keys()]
        else:
            class_names = scenario.split(":")[1].split("-")
        class_ids = [imagenet_label2id[name] for name in class_names]
        if "poisoner_kwargs" in kwargs:
            if "texts" not in kwargs["poisoner_kwargs"]:
                kwargs["poisoner_kwargs"]["texts"] = class_names
        else:
            kwargs["poisoner_kwargs"] = {"texts": class_names}

        return ImageNetScenario(
            dataset_path,
            class_ids[0],
            class_ids[1:],
            TextPoisoner,
            normalize=normalize,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown scenario {scenario}")
