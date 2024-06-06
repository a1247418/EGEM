import os
import random

import argparse
import numpy as np
import pickle as pkl
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from refinement.helpers import get_activations
from refinement.explainer import Explainer
from data.data_loading import load_dataset
from models.model_loading import load_model


def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="imagenet")
    parser.add_argument('--dataset_root', default="/home/space/datasets/imagenet_torchvision/data/", type=str)
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--model', type=str, default="resnet18")
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--layer', type=str)
    parser.add_argument('--out_path', type=str, default=None)
    parser.add_argument('--explanation_type', type=str, default="alpha1_beta0")
    parser.add_argument('--n_subset', type=int, default=None)
    args = parser.parse_args()
    return args


def extract_activations(dataset, n_classes, model, model_name, layer_names, explanation_type, n_subset=None):
    explainer = Explainer(
            model,
            n_classes=n_classes,
            explanation_type=explanation_type,
            canonizer="vgg"
            if "vgg" in model_name
            else "resnet"
            if "resnet" in model_name
            else None,
            sensitivity_only=False,  # if explanation_typ == "gradient" else True,
            constant_relevance=True,
    )

    acs = {}
    rs = {}
    for layer_name in layer_names:
        acs[layer_name] = [[] for _ in range(n_classes)]
        rs[layer_name] = [[] for _ in range(n_classes)]

        for c in range(n_classes):
            if hasattr(dataset, "samples"):
                subset_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == c]
            elif hasattr(dataset, "targets"):
                subset_indices = [i for i, label in enumerate(dataset.targets) if label == c]
            if n_subset:
                random.shuffle(subset_indices)
                subset_indices = subset_indices[:n_subset]
            class_subset = Subset(dataset, subset_indices)
            data_loader_train_c = DataLoader(class_subset, batch_size=128, shuffle=False, num_workers=8)
            print(f"Class: {c}/{n_classes}", "Nr. Batches:", len(data_loader_train_c))

            acs[layer_name][c], rs[layer_name][c], _, _ = get_activations(
                    model=model,
                    layer_names=[layer_name],
                    loader=data_loader_train_c,
                    do_pca=False,
                    pca_dims=None,
                    cls_token_only=False,
                    average_tokens=False,
                    square=False,
                    reduce=False,
                    explainer=explainer,
                    combine_a_exp=False,
                    batch_dim=1,
                    capture_outputs=True,
                    device="cuda",
                    silent=True,
                    avg_on_the_fly=False
            )
            acs[layer_name][c] = acs[layer_name][c][0].detach().cpu().numpy()
            rs[layer_name][c] = rs[layer_name][c][0].detach().cpu().numpy()
    return acs, rs


def run(dataset_name, dataset_root, split, model_name, model_path, layer_name, out_path, explanation_type, n_subset):
    print("Dataset:", dataset_name, "nr. per class", n_subset, "Split:", split, "Model:", model_name)

    dataset, _, n_classes = load_dataset(dataset_name, dataset_root, split)

    model = load_model(model_name,
                       file_path=model_path,
                       original_n_out=n_classes,
                       out_classes=[i for i in range(n_classes)], device="cuda").model

    out_path = out_path
    if out_path is None:
        out_path = os.getcwd()
    os.makedirs(out_path, exist_ok=True)

    path_to_embeddings = os.path.join(out_path, "emb-%s-%s-%s.pkl" % (dataset_name, model_name, layer_name))
    path_to_explanations = os.path.join(out_path, "expl-%s-%s-%s-%s.pkl" % (dataset_name, model_name, explanation_type, layer_name))

    acs, rs = extract_activations(dataset, n_classes, model, model_name, [layer_name], explanation_type, n_subset)

    with open(path_to_embeddings, "wb") as f:
        pkl.dump(acs, f)
    with open(path_to_explanations, "wb") as f:
        pkl.dump(rs, f)

    print("Done")


if __name__ == "__main__":
    args = get_parser_args()
    run(
        dataset_name=args.dataset,
        dataset_root=args.dataset_root,
        split=args.split,
        model_name=args.model,
        model_path=args.model_path,
        layer_name=args.layer,
        out_path=args.out_path,
        explanation_type=args.explanation_type,
        n_subset=args.n_subset
    )
