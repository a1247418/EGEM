import os
import numpy as np
import PIL.Image
from torchvision.datasets import CelebA
from torchvision.transforms import Compose, RandomHorizontalFlip

from data.loader_factory import LoaderFactory, dataset_with_indices

id2attrib = [
    '5_o_Clock_Shadow',
    'Arched_Eyebrows',
    'Attractive',
    'Bags_Under_Eyes',
    'Bald',
    'Bangs',
    'Big_Lips',
    'Big_Nose',
    'Black_Hair',
    'Blond_Hair',
    'Blurry',
    'Brown_Hair',
    'Bushy_Eyebrows',
    'Chubby',
    'Double_Chin',
    'Eyeglasses',
    'Goatee',
    'Gray_Hair',
    'Heavy_Makeup',
    'High_Cheekbones',
    'Male',
    'Mouth_Slightly_Open',
    'Mustache',
    'Narrow_Eyes',
    'No_Beard',
    'Oval_Face',
    'Pale_Skin',
    'Pointy_Nose',
    'Receding_Hairline',
    'Rosy_Cheeks',
    'Sideburns',
    'Smiling',
    'Straight_Hair',
    'Wavy_Hair',
    'Wearing_Earrings',
    'Wearing_Hat',
    'Wearing_Lipstick',
    'Wearing_Necklace',
    'Wearing_Necktie',
    'Young'
]
attrib2id = {l: i for i, l in enumerate(id2attrib)}
mean_std = ([0.5064, 0.4258, 0.3832], [0.2660, 0.2452, 0.2414])


class BinaryCelebA(CelebA):
    base_folder = "celeba"
    file_list = [
        # File ID                         MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(self, root, target_attr, target_type="attr", split="train", transform=None,
                 target_transform=None, download=False, keep_attrib=None, stratify=False, pre_exclude_indices=None):
        super().__init__(root, split, target_type, transform, target_transform, download)

        # Recreating the filtering of filenames from versions of the CelebA implementation < torchvision 0.10
        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[split.lower()]
        splits = self._load_csv("list_eval_partition.txt")
        split_mask = slice(None) if split is None else (splits[2] == split_)
        try:
            self.filename = np.array(splits[1])[split_mask[:, 0]]
        except:
            assert len(self.filename) == len(self.attr)

        if pre_exclude_indices is not None:
            mask = np.ones([len(self.attr)], dtype=bool)
            mask[pre_exclude_indices] = False

            self.filename = self.filename[mask]
            self.attr = self.attr[mask]
            self.identity = self.identity[mask]
            self.bbox = self.bbox[mask]
            self.landmarks_align = self.landmarks_align[mask]

        if keep_attrib is not None:
            if type(keep_attrib) != list:
                keep_attrib = [keep_attrib]
            mask = np.ones([len(self.attr)], dtype=bool)

            for y_i, y in enumerate(self.attr.numpy()):
                for r in keep_attrib:
                    if y[r] == 1:
                        mask[y_i] = False
                        break
            mask = np.logical_not(mask)

            self.filename = self.filename[mask]
            self.attr = self.attr[mask]
            self.identity = self.identity[mask]
            self.bbox = self.bbox[mask]
            self.landmarks_align = self.landmarks_align[mask]

        if stratify:
            # Make sure the #samples with and without the target attribute are equal.
            mask = np.zeros([len(self.attr)], dtype=bool)

            for y_i, y in enumerate(self.attr):
                if y[target_attr] == 1:
                    mask[y_i] = True

            n_t = np.sum(mask)
            n_not_t = len(self.attr) - n_t
            n_oversample = n_not_t - n_t
            if n_oversample < 0:
                n_oversample *= -1
                mask = np.logical_not(mask)
                oversample_factor = (n_oversample / n_not_t) if n_not_t else 0
            else:
                oversample_factor = (n_oversample / n_t) if n_t else 0

            if oversample_factor:
                oversample_factor = int(np.ceil(oversample_factor))
                self.filename = np.concatenate(
                        [self.filename, np.tile(self.filename[mask], oversample_factor)[:n_oversample]])
                self.attr = np.concatenate([self.attr, np.tile(self.attr[mask], [oversample_factor, 1])[:n_oversample]])
                self.identity = np.concatenate(
                        [self.identity, np.tile(self.identity[mask], [oversample_factor, 1])[:n_oversample]])
                self.bbox = np.concatenate([self.bbox, np.tile(self.bbox[mask], [oversample_factor, 1])[:n_oversample]])
                self.landmarks_align = np.concatenate([self.landmarks_align,
                                                       np.tile(self.landmarks_align[mask], [oversample_factor, 1])[
                                                       :n_oversample]])

        self.targets = self.attr[:, target_attr]
        self.target_attr = target_attr

    def __getitem__(self, index):
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))
        target = self.attr[index, :]

        if self.transform is not None:
            X = self.transform(X)

        y = target[self.target_attr]

        s = None
        if "attr" in self.target_type:
            s = target
        elif "landmarks" in self.target_type:
            s = self.landmarks_align[index, :]

        if s is not None:
            return X, [y, s]
        else:
            return X, y

    def __len__(self):
        return len(self.attr)


class CelebALoaderFactory(LoaderFactory):
    def __init__(self, data_path, is_train=True, input_size=(218, 178), normalize=True, transform=None,
                 class_subset=None, target_attr=9, keep_attrib=None, target_type="landmarks",
                 output_indices=False, stratify=False, pre_exclude_indices=None):
        # 8=black, 9=blond, 33=wavy
        self.is_train = is_train
        normalization_params = mean_std if normalize else None
        self.n_classes = 1
        self.target_attr = target_attr
        self.keep_attrib = keep_attrib  # remove all samples w this attribute
        self.target_type = target_type
        self.stratify = stratify
        self.pre_exclude_indices = pre_exclude_indices

        if is_train:
            pre_tensor_transform = Compose([
                RandomHorizontalFlip(p=.5),
            ])
        else:
            pre_tensor_transform = Compose([])
        super().__init__(data_path, input_size, normalization_params, transform=transform, class_subset=class_subset,
                         output_indices=output_indices, pre_tensor_transform=pre_tensor_transform)

    def create_dataset(self, output_indices=False):
        dataset_cls = dataset_with_indices(BinaryCelebA) if output_indices else BinaryCelebA
        try:
            dataset = dataset_cls(root=self.data_path, target_attr=self.target_attr, target_type=self.target_type,
                                  keep_attrib=self.keep_attrib, split="train" if self.is_train else "valid",
                                  download=False, transform=self.transform, stratify=self.stratify,
                                  pre_exclude_indices=self.pre_exclude_indices)
        except:
            dataset = dataset_cls(root=self.data_path, target_attr=self.target_attr, target_type=self.target_type,
                                  keep_attrib=self.keep_attrib, split="train" if self.is_train else "valid",
                                  download=True, transform=self.transform, stratify=self.stratify,
                                  pre_exclude_indices=self.pre_exclude_indices)
        return dataset