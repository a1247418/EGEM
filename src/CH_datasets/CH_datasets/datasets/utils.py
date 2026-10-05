from typing import Any, Tuple, Iterable
from PIL import Image


def make_poisonable(dataset_class: type):
    """
    Modifies the given Dataset class that must be a subclass of DatasetFolder to work with Poisoners in the transfrom.
    """
    if dataset_class.__name__ == "MNIST":
        def __getitem__(self, index: int) -> Tuple[Any, Any]:
            img, target = self.data[index], int(self.targets[index])
            img = Image.fromarray(img.numpy(), mode="L")

            if self.transform is not None:
                if isinstance(self.transform, Iterable):
                    for t in self.transform:
                        try:
                            img = t(img)
                        except TypeError:
                            img = t(img, target)
                else:
                    try:
                        img = self.transform(img)
                    except TypeError:
                        img = self.transform(img, target)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target
    else:
        def __getitem__(self, index: int) -> Tuple[Any, Any]:
            index = int(index)
            path, target = self.samples[index]
            img = self.loader(path)

            if self.transform is not None:
                if isinstance(self.transform, Iterable):
                    for t in self.transform:
                        try:
                            img = t(img)
                        except TypeError:
                            img = t(img, target)
                else:
                    try:
                        img = self.transform(img)
                    except TypeError:
                        img = self.transform(img, target)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target

    return type(dataset_class.__name__, (dataset_class,), {
        '__getitem__': __getitem__,
    })


def to_official_imagenet_targets(dataset):
    """A copy of ImageNet with only some class folders gets folder-order targets (0..n-1). Maps them to the
    official class index, i.e. the position of the wnid among all 1000 sorted wnids of the devkit meta file."""
    from torchvision.datasets.imagenet import load_meta_file
    all_wnids = sorted(load_meta_file(dataset.root)[0].keys())
    if list(dataset.wnids) == all_wnids:
        return dataset
    official = [all_wnids.index(w) for w in dataset.wnids]
    dataset.samples = [(path, official[t]) for path, t in dataset.samples]
    dataset.imgs = dataset.samples
    dataset.targets = [official[t] for t in dataset.targets]
    return dataset
