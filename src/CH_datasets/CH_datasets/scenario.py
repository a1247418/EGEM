import dataclasses
from typing import List, Optional
from torchvision.datasets import ImageNet, MNIST
from torchvision.transforms import Normalize, CenterCrop, Resize, Compose, ToTensor
from torch.utils.data import Subset

from CH_datasets.poisoner import Poisoner
from CH_datasets.datasets.isic import ISICDataset
from CH_datasets.datasets.utils import make_poisonable


def get_dataset(dataset: str, dataset_dir: str, train: bool = True):
    if dataset == "imagenet":
        dataset_instance = make_poisonable(ImageNet)(
            split="train" if train else "val", root=dataset_dir
        )
    elif dataset == "mnist":
        dataset_instance = make_poisonable(MNIST)(train=train, root=dataset_dir)
    elif dataset == "isic":
        dataset_instance = make_poisonable(ISICDataset)(train=train, root=dataset_dir)
    else:
        raise ValueError(f"Dataset {dataset} not supported.")

    return dataset_instance


def get_default_transform(dataset: str, normalize: bool = True):
    if dataset == "imagenet":
        transform = [Resize(224), CenterCrop(224), ToTensor()]
        if normalize:
            transform.append(
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
        transform = Compose(transform)
    elif dataset == "mnist":
        transform = [
            ToTensor(),
        ]
        if normalize:
            transform.append(Normalize(mean=[0.1307], std=[0.3081]))
        transform = Compose(transform)
    elif dataset == "isic":
        transform = [
            Resize(224),
            CenterCrop(224),
            ToTensor(),
        ]
        if normalize:
            transform.append(
                Normalize(mean=[0.6678, 0.5296, 0.5242], std=[0.1282, 0.1417, 0.1521])
            )
        transform = Compose(transform)
    else:
        raise ValueError(f"Dataset {dataset} not supported.")
    return transform


def poison_dataset(dataset, poisoner):
    transforms = dataset.transform.transforms

    def is_to_tensor(transform):
        if isinstance(transform, ToTensor):
            return True
        try:
            return any([is_to_tensor(t) for t in transform.transform])
        except AttributeError:
            return False

    i = 0
    for j, transform in enumerate(transforms):
        i = j
        if is_to_tensor(transform):
            if not poisoner.poison_before_tensor:
                i += 1
            break
    new_transforms = transforms[:i] + [poisoner] + transforms[i:]
    dataset.transform = new_transforms


@dataclasses.dataclass
class Scenario:
    dataset: str
    dataset_dir: str
    train_idcs: Optional[List[int]] = None
    refinement_idcs: Optional[List[int]] = None
    test_idcs: Optional[List[int]] = None
    train_poisoner: Optional[Poisoner] = None
    refinement_poisoner: Optional[Poisoner] = None
    test_poisoner: Optional[Poisoner] = None
    transform: Optional = None
    target_transform: Optional = None

    def __post_init__(self) -> None:
        self._train_dataset = None
        self._refinement_dataset = None
        self._test_dataset = None

    def get_data(self, split: str):
        if split == "train":
            idcs = self.train_idcs
            poisoner = self.train_poisoner
            train = True
            dataset = self._train_dataset
        elif split == "refine":
            idcs = self.refinement_idcs
            poisoner = self.refinement_poisoner
            train = True
            dataset = self._refinement_dataset
        elif split == "test":
            idcs = self.test_idcs
            poisoner = self.test_poisoner
            train = False
            dataset = self._test_dataset
        else:
            raise ValueError(f"Split {split} not supported.")

        if dataset is None:
            dataset = get_dataset(self.dataset, self.dataset_dir, train=train)
            if self.transform is not None:
                dataset.transform = self.transform
            if self.target_transform is not None:
                dataset.target_transform = self.target_transform
            if poisoner is not None:
                poison_dataset(dataset, poisoner)
            if idcs is not None:
                dataset = Subset(dataset, idcs)

            if split == "train":
                self._train_dataset = dataset
            elif split == "refine":
                self._refinement_dataset = dataset
            elif split == "test":
                self._test_dataset = dataset
        return dataset

    def get_train_data(self):
        return self.get_data("train")

    def get_refine_data(self):
        return self.get_data("refine")

    def get_test_data(self):
        return self.get_data("test")
