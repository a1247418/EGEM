import os
import torch
import matplotlib.pyplot as plt
from typing import Optional


def get_base_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def get_artifact_path(file_name: Optional[str] = None):
    path = os.path.join(get_base_path(), "artifacts")
    if file_name is not None:
        path = os.path.join(path, file_name)
    return path


def get_refinement_indices_path(file_name: Optional[str] = None):
    path = os.path.join(get_base_path(), "refinement_indices")
    if file_name is not None:
        path = os.path.join(path, file_name)
    return path


def plot_scenario(train, refine, test, color_map=None, save_path=None):
    splits = ["train", "refine", "test"]
    n_imgs = 7

    fig, axes = plt.subplots(3, n_imgs, figsize=(n_imgs * 2, 3 * 2.5))
    for s_i, (split, data) in enumerate(zip(splits, [train, refine, test])):
        data_loader = torch.utils.data.DataLoader(data, batch_size=n_imgs, shuffle=True)
        images, labels = next(iter(data_loader))

        for i_i, image in enumerate(images):
            np_image = image.permute(1, 2, 0).numpy()
            axes[s_i, i_i].imshow(np_image, cmap=color_map)
            axes[s_i, i_i].set_title(f"Label: {labels[i_i]}")
            axes[s_i, i_i].set_xticklabels([])
            axes[s_i, i_i].set_yticklabels([])
            axes[s_i, i_i].set_xticks([])
            axes[s_i, i_i].set_yticks([])
            if i_i == 0:
                axes[s_i, i_i].set_ylabel(split, fontsize=18)

    plt.subplots_adjust(wspace=0, hspace=0)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
