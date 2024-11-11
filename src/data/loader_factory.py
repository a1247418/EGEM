import copy, random
from abc import ABC, abstractmethod
from typing import Tuple, Sequence
import torch
from torchvision.transforms import Resize, Normalize, ToTensor, Compose, Lambda, CenterCrop
from torch.utils.data import DataLoader, Subset

NUM_WORKERS = 4

def get_permutation(length: int, seed: int = None):
    permutation = [i for i in range(length)]
    if seed is not None:
        random.Random(seed).shuffle(permutation)
    else:
        random.shuffle(permutation)
    return permutation


def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    From: https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/19
    """
    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


class LoaderFactory(ABC):
    def __init__(self, data_path,
                 input_size: Tuple = None,
                 normalization_params: Tuple = None,
                 transform=None,
                 pre_tensor_transform=None,
                 label_map=None,
                 class_subset=None,
                 keep_indices=None,
                 output_indices: bool = False):
        """
        Abstract dataset factory. Creates the dataset and can produce data loaders from it.
        @param data_path: Compete path to data folder
        @param input_size: Size to which the input should be resized. None = no resize
        @param normalization_params: Tuple of color mean & std of the dataset for image normalizations
        @param transform: Transform to be applied on the input data tensor
        @param pre_tensor_transform: Transform to be applied on the input data before it is turned into a tensor
        @param label_map: Mapping of lables. Dictionary {map_from: map_to}
        @param class_subset: List of classes to use (selection is union with keep_indices)
        @param keep_indices: List of indices to keep (selection is union with class_subset)
        @param output_indices: True if the created datasets should output the sample index, i.e. [x, y, idx]
        """
        self.data_path = data_path
        self.input_size = input_size
        self.normalization_params = normalization_params
        self.transform = self.create_transform(transform, pre_tensor_transform)

        # Build an appropriate target mapping for the given output type
        self.label_mapping_matrix = None
        self.label_mapping_dict = None
        if label_map is not None:
            # Used if labels are scalars
            self.label_mapping_dict = label_map
            # Used if labels are class vectors
            self.label_mapping_matrix = torch.eye(len(label_map))  # torch.zeros([len(label_map), len(label_map)])
            for k, v in label_map.items():
                self.label_mapping_matrix[k, k] = 0
                self.label_mapping_matrix[k, v] = 1

        if hasattr(self, "out_transform"):
            if label_map is not None:
                self.out_transform = Compose([self.out_transform, Lambda(self._map_label)])
        else:
            self.out_transform = None if label_map is None else (self._map_label)
        self.dataset = self.create_dataset(output_indices)

        if keep_indices is not None:
            indices = [i for i, label in enumerate(self.dataset.targets) if
                       (i in keep_indices)]
            self.dataset = Subset(self.dataset, indices)
        elif class_subset is not None:
            indices = [i for i, label in enumerate(self.dataset.targets) if
                       (label in class_subset)]
            self.dataset = Subset(self.dataset, indices)

        if hasattr(self, "n_classes"):
            # Adapt the number of classes to the label mapping
            if label_map is not None:
                class_arr = [1 for _ in range(self.n_classes)]
                for k, v in label_map.items():
                    class_arr[int(k)] = 0
                    class_arr[int(v)] = 1
                self.n_classes = sum(class_arr)
        else:
            self.n_classes = None

        if not hasattr(self, "class_weights"):
            self.class_weights = None

        if not hasattr(self, "variable_input_size") or self.variable_input_size == False:
            self.set_input_size()

    def _map_label(self, old_label):
        try:
            new_label = self.label_mapping_dict[old_label]
        except KeyError:
            if self.label_mapping_matrix is not None:
                new_label = (old_label @ self.label_mapping_matrix)[0].clamp(max=1)
            else:
                new_label = old_label
        return new_label

    def create_transform(self, transform=None, pre_tensor_transform=None):
        has5crop = False
        hasCcrop = False
        subtransforms = [ToTensor()]

        if pre_tensor_transform is not None and any([type(i) == CenterCrop for i in pre_tensor_transform.transforms]):
            hasCcrop = True

        if transform is not None:
            # Add passed transformations
            subtransforms.append(transform)
        if self.normalization_params:
            subtransforms.append(Normalize(self.normalization_params[0], self.normalization_params[1]))

        if pre_tensor_transform is not None:
            subtransforms = [pre_tensor_transform] + subtransforms
        if self.input_size and not hasCcrop:
            subtransforms = [Resize(self.input_size)] + subtransforms

        transform = Compose(subtransforms)
        return transform

    def set_input_size(self):
        if self.input_size is None:
            self.input_size = tuple(iter(DataLoader(self.dataset,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    num_workers=NUM_WORKERS)).next()[0][0].shape[-2:])

    @classmethod
    @abstractmethod
    def create_dataset(self, output_indices=False):
        return None

    def create_data_loader(self, random_subset_size: int, subset_indices: Sequence[int], batch_size: int,
                           shuffle: bool = True, seed: int = None):
        if subset_indices is not None:
            # use only certain indices
            datasubset = Subset(self.dataset, subset_indices)

        elif random_subset_size is not None:
            # take a random subset
            permutation = get_permutation(len(self.dataset), seed)
            random_subset_indices = permutation[:random_subset_size]
            datasubset = Subset(self.dataset, random_subset_indices)
        else:
            datasubset = self.dataset

        loader = DataLoader(datasubset, batch_size=batch_size, shuffle=shuffle, num_workers=NUM_WORKERS)

        return loader

    def create_split_data_loaders(self, random_subset_size: int, val_fraction: float,
                                  batch_size: int, shuffle: bool = True, seed: int = None):
        """Creates a data loader for training and validation according to the given split ratio."""
        if val_fraction > 1:
            raise ValueError("Validation fraction has to be <= 1.")

        len_dataset = len(self.dataset)
        n_samples = len_dataset if random_subset_size is None else random_subset_size
        n_train_samples = int((1 - val_fraction) * n_samples)
        n_val_samples = n_samples - n_train_samples

        permutation = get_permutation(len_dataset, seed)
        train_indices = permutation[:n_train_samples]
        val_indices = permutation[-n_val_samples:]
        train_loader = self.create_data_loader(None, train_indices, batch_size, shuffle)
        val_loader = self.create_data_loader(None, val_indices, batch_size, shuffle)

        return train_loader, val_loader
