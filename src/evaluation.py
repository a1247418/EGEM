from typing import Tuple, List

import numpy as np
import torch
from tqdm import tqdm

Tensor = torch.Tensor
Array = np.ndarray


def _accuracy(output: Tensor, target: Tensor, topk:tuple=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())/target.shape[0] for k in topk]

    pred = pred.cpu().squeeze().numpy()
    correct = correct.cpu().squeeze().numpy()
    return pred, correct, acc


def _calculate_per_class_accuracy(predictions: Tensor, ground_truth: Tensor, num_classes) -> Array:
    per_class_accuracy = []
    for class_label in range(num_classes):
        # Create a mask for the current class label
        class_mask = torch.eq(ground_truth, class_label)

        # Calculate accuracy for the current class
        class_accuracy = torch.mean(
            (predictions == ground_truth)[class_mask].float()
        )

        per_class_accuracy.append(class_accuracy.item())

    return np.array(per_class_accuracy)


def _calculate_confusion_matrix(predictions: Array, ground_truth: Array, num_classes: int) -> Array:
    confusion_matrix = np.zeros((num_classes, num_classes))

    for i in range(len(predictions)):
        confusion_matrix[ground_truth[i], predictions[i]] += 1

    return confusion_matrix


def evaluate(model, data_loader, device: str = "cuda", verbose=False):
    was_training = False
    if model.training:
        model.eval()
        was_training = True
    if verbose: print(torch.cuda.memory_allocated() / 1024 ** 2, "MB used before evaluation.")
    all_y = []
    all_y_hat = []
    for i, (x, y) in enumerate(tqdm(data_loader)):
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x).detach()

        all_y_hat.append(y_hat)
        all_y.append(y)
        # print the used gpu memory
        if verbose: print(torch.cuda.memory_allocated() / 1024 ** 2, "MB used at iteration", i)


    all_y = torch.concat(all_y, dim=0).cpu().detach()
    all_y_hat = torch.concat(all_y_hat, dim=0).cpu().detach()

    if verbose: print(torch.cuda.memory_allocated() / 1024 ** 2, "MB used after evaluation.")

    # measure accuracy
    pred, correct, acc1 = _accuracy(all_y_hat, all_y, topk=(1,))
    # print("\ttop-1 acc:", acc1)
    top1 = acc1
    pred = pred
    true = all_y.squeeze().numpy()
    all_y_hat = all_y_hat.squeeze().numpy()

    confusion = _calculate_confusion_matrix(pred, true, max(np.max(true).item(), np.max(pred).item()) + 1)

    to_return = {
        "top1": top1,
        "true": true,
        "predicted": pred,
        "confusion": confusion,
        "output": all_y_hat,
    }
    top1_std = np.std(top1)
    top1 = np.mean(top1)
    print(f"Top-1 accuracy: {top1:.2f}+-{top1_std:.2f}")

    if was_training:
        model.train()

    if verbose: print(torch.cuda.memory_allocated() / 1024 ** 2, "MB used after metrics calculation.")

    return to_return


def get_balanced_data(n_shot: int, n_classes: int, loader, device: str = "cuda",
                      model=None) -> Tuple[Tensor, Tensor]:
    """Get a balanced n-shot data sample from the given loader. If a model is given, only samples it
    predicts correctly are used (the paper's simulated user verification, Sec. 4.3)."""
    n_per_class = {i: 0 for i in range(n_classes)}
    imgs = []
    targets = []
    stop = False
    for xs, ys in loader:
        if model is not None:
            with torch.no_grad():
                model_device = next(model.parameters()).device
                correct = model(xs.to(model_device)).argmax(1).cpu() == ys
            xs, ys = xs[correct], ys[correct]
            if len(ys) == 0:
                continue
        if torch.numel(ys) == 1:
            xs = [xs[0]]
            ys = [ys[0]]
        for x, y in zip(xs, ys):
            x.unsqueeze_(0)
            y.unsqueeze_(0)
            if n_per_class[int(y)] < n_shot:
                imgs.append(x.to(device))
                targets.append(y)
                n_per_class[int(y)] += 1
                if np.sum([v for v in n_per_class.values()]) == n_shot * n_classes:
                    stop = True
                    break
        if stop:
            break

    for c, n_c in n_per_class.items():
        if n_c < n_shot:
            if n_c == 0:
                raise ValueError(f"Insufficient data for eval with {n_shot} samples per class: only {n_c} samples for class {c}")
            else:
                # up-sample by randomly adding samples from the same class
                n_missing = n_shot - n_c
                c_idxs = [i for i, t in enumerate(targets) if t == c]
                c_idxs = np.random.choice(c_idxs, n_missing, replace=True)
                for c_i in c_idxs:
                    imgs.append(imgs[c_i])
                    targets.append(targets[c_i])
                n_per_class[c] = n_shot

    imgs = torch.cat(imgs, dim=0)
    targets = torch.cat(targets, dim=0)
    return imgs, targets


def get_n_shot_data(
    n_shot: int, n_classes: int, loader, n_reps: int = 1, device: str = "cuda", model=None
) -> Tuple[List[Tensor], List[Tensor]]:
    refine_targets = []
    refine_images = []
    for rep in range(n_reps):
        refine_imgs, refine_labels = get_balanced_data(
            n_classes=n_classes, n_shot=n_shot, loader=loader, device=device, model=model
        )
        refine_targets.append(refine_labels.to(device).detach())
        refine_images.append(refine_imgs.to(device).detach())

    assert all(
        [
            len(refine_targets[r]) == n_shot * n_classes
            for r in range(n_reps)
        ]
    )

    return refine_images, refine_targets
