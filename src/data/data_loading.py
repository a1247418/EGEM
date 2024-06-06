import random
from functools import partial
from typing import Optional, Tuple, List

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from CH_datasets.scenario_examples import get_scenario

Array = np.ndarray
Tensor = torch.Tensor

def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_dataset(dataset_name:str, data_root:str, split:str):
    if dataset_name.lower() == "cifar10":
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset=torchvision.datasets.CIFAR10(
            root=data_root, train=split=="train", download=True, transform=transform
        )
        n_classes = 10
    elif dataset_name.lower() == "cifar100":
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset=torchvision.datasets.CIFAR100(
            root=data_root, train=split == "train", download=True, transform=transform
        )
        n_classes = 100
    elif dataset_name.lower() == "imagenet":
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )
        dataset=torchvision.datasets.ImageNet(
            root=data_root, split=split, transform=transform
        )
        n_classes = 1000
    elif dataset_name.lower() == "flowers102":
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset=torchvision.datasets.Flowers102(
            root=data_root, split=split, transform=transform
        )
        n_classes = 102
    elif dataset_name.lower() == "mnist":
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        dataset = torchvision.datasets.MNIST(
                root=data_root, train=split=="train", download=True, transform=transform
            )
        n_classes = 10
    else:
        raise ValueError("Dataset not recognized", dataset_name)

    return dataset, transform, n_classes

"""
def _get_balanced_data(n_shot: int, n_classes: int, loader) -> Tuple[Tensor, Tensor]:
    n_per_class = {i: 0 for i in range(n_classes)}
    imgs = []
    targets = []
    stop = False
    for xs, ys in loader:
        if torch.numel(ys) == 1:
            xs = [xs[0]]
            ys = [ys[0]]
        for x, y in zip(xs, ys):
            x.unsqueeze_(0)
            y.unsqueeze_(0)
            if n_per_class[int(y)] < n_shot:
                imgs.append(x.cuda())
                targets.append(y)
                n_per_class[int(y)] += 1
                if np.sum([v for v in n_per_class.values()]) == n_shot * n_classes:
                    stop = True
                    break
            if stop:
                break
    imgs = torch.cat(imgs, dim=0)
    targets = torch.cat(targets, dim=0)
    return imgs, targets


def get_n_shot_data(
    n_shot: int, n_classes: int, loader, img_enc, n_reps: int = 1, device: str = "cuda"
) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
    refine_features = []
    refine_targets = []
    refine_images = []  # only for inspection
    with torch.no_grad():
        for rep in range(n_reps):
            refine_imgs, refine_labels = _get_balanced_data(
                n_classes=n_classes, n_shot=n_shot, loader=loader
            )
            refine_targets.append(refine_labels.to(device).detach())
            refine_features.append(img_enc(refine_imgs).to(device).detach())
            refine_images.append(refine_imgs.to(device).detach())
    return refine_features, refine_targets, refine_images
"""#TODO: remove


def load_scenario(
    scenario_name: str,
    scenario_path: str,
    poisoning_strategy: str = "uniform",
    train_p: float = 0.0,
    refinement_p: float = 0.0,
    test_p: float = 1.0,
    poisoner_kwargs: dict = None,
    train_batch_size: int = 128,
    val_batch_size: int = 128,
    refine_batch_size: int = 128,
    shuffle_train: bool = True,
    normalize: bool = True,
    val_set_size: Optional[int] = None,
):
    # To prevent too many open files error
    torch.multiprocessing.set_sharing_strategy("file_system")

    original_scenarios = [
        "mnist-8",
        "isic-1",
        "carton-crate",
        "carton-packet",
        "carton-envelope",
        "carton-all",
        "mtb-bbt",
        "minst-8_rgb-blur",
        "minst-8_rgb-artifact",
        "minst-8_rgb-remove",
        "minst-8_rgb-color",
    ]
    assert scenario_name in original_scenarios

    kwargs = {}
    if val_set_size is not None:
        kwargs["test_set_size"] = val_set_size
    if poisoner_kwargs is not None:
        kwargs["poisoner_kwargs"] = poisoner_kwargs
    scenario = get_scenario(
        scenario_name,
        dataset_path=scenario_path,
        normalize=normalize,
        poisoning_stategy_test=poisoning_strategy,
        train_p=train_p,
        refinement_p=refinement_p,
        test_p=test_p,
        **kwargs,
    )

    partial_loader = partial(DataLoader, num_workers=8, shuffle=shuffle_train)
    train_loader = partial_loader(
        dataset=scenario.get_data("train"), batch_size=train_batch_size
    )
    refine_loader = partial_loader(
        dataset=scenario.get_data("refine"), batch_size=refine_batch_size
    )
    val_loader = partial_loader(
        dataset=scenario.get_data("test"),
        shuffle=False,
        # pin_memory=True,
        worker_init_fn=seed_worker,
        batch_size=val_batch_size,
    )

    return train_loader, refine_loader, val_loader


def get_class_names(scenario_name: str):
    if "mnist" in scenario_name:
        class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    elif scenario_name == "isic-1":
        class_names = [
            "melanoma",
            "melanocytic nevus",
            "basal cell carcinoma",
            "actinic keratosis",
            "benign keratosis",
            "dermatofibroma",
            "vascular lesion",
            "squamous cell carcinoma",
        ]
    elif scenario_name.startswith("mtb-bbt"):
        class_names = ["mountain-bike", "bicyle-built-for-two"]
    else:
        class_names = scenario_name.replace("imagenet_text:", "").split(":")[0].split("-")
    return class_names
