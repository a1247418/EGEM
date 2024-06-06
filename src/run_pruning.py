import os
import time
import torch
import argparse
import pickle as pkl
import random

from torch.utils.data import DataLoader

from data.data_loading import load_dataset
from models.model_loading import load_model
from models.training import train, evaluate, PLModel
from refinement.refiner import ActivationPruner, WeightMagnitudePruner


FINE_TUNE_EPOCHS = 200
FINE_TUNE_LR = 0.001
FINE_TUNE_BATCH_SIZE = 128
N_WORKERS = 8

def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa("--dataset", type=str)
    aa("--dataset_root", type=str)
    aa("--out_path", type=str,)
    aa("--model", type=str)
    aa("--criterion", type=str)
    aa("--layers", type=str, nargs="+", default=None)
    aa("--n_samples_per_cls", type=int, default=100)
    aa("--skip_existing", action="store_true")
    args = parser.parse_args()
    return args


def do_train(model, train_loader, n_epochs:int, lr:float):
    to_train = PLModel(model.model, lr=lr)
    # Train model
    train(to_train, train_loader, None, max_epochs=n_epochs)


def load_pruning_values(criterion, path=None):
        if criterion in ["w", "wr"]:
            to_return = None
        else:
            with open(path, 'rb') as file:
                data = pkl.load(file)

            if criterion in ["a", "ar"]:
                to_return = data["a"]
            elif criterion in ["r"]:
                to_return = data["r"]
            else:
                raise ValueError("Invalid criterion " + str(criterion))

        return to_return


def do_prune(model, values_per_layer, percent, pruning_criteruim="w"):
    if pruning_criteruim in ["a", "ar", "r"]:
        pruner = ActivationPruner(model,
                                  values_per_layer.keys(),
                                  values_per_layer,
                                  percent,
                                  random =pruning_criteruim == "ar")
    elif pruning_criteruim in ["w", "wr"]:
        pruner = WeightMagnitudePruner(model,
                                       values_per_layer.keys(),
                                       percent,
                                       random =pruning_criteruim == "wr")
    else:
        raise ValueError("Invalid pruning criterion " + str(pruning_criteruim))

    pruner.refine()
    return model


def do_eval(model, test_loader, n_classes):
    class_acc, acc = evaluate(test_loader, model, n_classes)
    results = {"acc":float(acc), "class_acc":class_acc.cpu().numpy()}
    return results


if __name__ == "__main__":
    start_time = time.time()
    args = parseargs()
    criterion = args.criterion
    dataset_root = args.dataset_root
    dataset = args.dataset
    model_name = args.model
    model_path = args.model_path
    n_samples_per_cls = args.n_samples_per_cls

    out_path = args.out_path + os.path.join(
            dataset,
            model_name,
            criterion
    )
    os.makedirs(out_path, exist_ok=True)

    # Prepare data loader
    dataset_test, _, n_classes = load_dataset(dataset, dataset_root, "val")
    test_loader = DataLoader(dataset_test, batch_size=128, shuffle=False, num_workers=N_WORKERS)
    dataset_train, _, _ = load_dataset(dataset, dataset_root, "train")
    # Get a reproducible random subset of exactly n_samples_per_cls per class
    random.seed(42)
    indices = []
    for i in range(n_classes):
        if hasattr(dataset, "targets"):
            tmp_indices = [j for j, x in enumerate(dataset.targets) if x == i]
        else:
            tmp_indices = [j for j, x in enumerate(dataset.labels) if x == i]
        random.shuffle(tmp_indices)
        indices += tmp_indices[:n_samples_per_cls]
    dataset = torch.utils.data.Subset(dataset, indices)
    train_loader = DataLoader(dataset, batch_size=FINE_TUNE_BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)
    # TODO: are truning datasets (small) != training (large)?

    model = load_model(model_name, file_path=model_path, original_n_out=n_classes).model

    for p in range(0, 101, 5):
        model_path = os.path.join(out_path, f"p{p}.model")
        pre_results_path = os.path.join(out_path, f"p{p}.pre_results")
        results_path = os.path.join(out_path, f"p{p}.results")
        pruning_values_path = os.path.join(out_path, f"p{p}.values")
        old_pruning_values_path = os.path.join(out_path, f"p{p-5}.values")

        skipped = False
        if p != 0:
            if args.skip_existing and os.path.exists(results_path):
                print(f"Model already exists (p={p}). Skipping.")
                skipped = True
                continue

            if skipped:
                model = torch.load(model_path.replace(f"p{p}", f"p{p-5}"))

            # Load pruning values
            values_per_layer = load_pruning_values(criterion, path=old_pruning_values_path)

            # Prune model
            percent = p * 100. / (100. - (p - 5))
            do_prune(model, values_per_layer, percent, criterion)
            #TODO make actually smaller (at least for node pruning)
            # TODO: ensure that the model is trainable after pruning

            # Evaluate model
            results_pre = do_eval(model, test_loader, n_classes)

            # Save results
            with open(pre_results_path, "wb") as f:
                pkl.dump(results_pre, f)

            # Tune model
            do_train(model, train_loader, n_epochs=FINE_TUNE_EPOCHS, lr=FINE_TUNE_LR)

            # Save model + results
            torch.save(model, model_path)

            # Evaluate model
            results = do_eval(model, test_loader, n_classes)
            with open(results_path, "wb") as f:
                pkl.dump(results, f)

        # Extract activations / relevance (limited data)
        #todo

    print("Done. Took", (time.time() - start_time) / 60, "minutes.")
