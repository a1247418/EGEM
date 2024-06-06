import os
import time
import torch
import argparse
import pickle as pkl

from torch.utils.data import DataLoader

from data.data_loading import load_dataset
from models.model_loading import load_model
from models.training import train, evaluate, PLModel
from data.data_loading import load_scenario, get_class_names


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa("--dataset", type=str)
    aa("--dataset_root", type=str)
    aa("--out_path", type=str,)
    aa("--model", type=str)
    aa("--epochs", type=int, default=200)
    aa("--lr", type=float, default=0.001)
    aa("--frac_poisoned", type=float, default=None)
    aa("--skip_existing", action="store_true")
    args = parser.parse_args()
    return args


def run_training(dataset_name:str, dataset_root:str, model:str, n_epochs:int, lr:float, poison_frac:float=None):
    # Load data
    normalize = True
    if dataset_name == "mnist-8":
        normalize = False
        print("###################################")
        print("WARNING: not normlizing MNIST-8!")
        print("###################################")
    try:
        # Try to load a spurious scenario
        train_loader, refine_loader, test_loader = load_scenario(
            dataset_name,
            dataset_root,
            train_batch_size=128,
            val_batch_size=128,
            refine_batch_size=128,
            train_p=poison_frac,
            shuffle_train=True,
            normalize=normalize,
        )
        n_classes = len(get_class_names(dataset_name))
        print("Loaded spurious dataset:", dataset_name)
    except AssertionError:
        print("Loading normal dataset:", dataset_name)
        # If it fails, load the normal dataset directly
        dataset, transform, n_classes = load_dataset(dataset_name, dataset_root, "train")
        train_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8)
        dataset, transform, n_classes = load_dataset(dataset_name, dataset_root, "val")
        test_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8)

    # Load model and adapt to number of classes
    model = load_model(model)
    if n_classes != model.model.features[-1].out_features:
        model.model.features[-1] = torch.nn.Linear(model.model.features[-1].in_features, n_classes)
    to_train = PLModel(model.model, lr=lr)

    # Train model
    train(to_train, train_loader, None, max_epochs=n_epochs)

    # Evaluate model
    class_acc, acc = evaluate(test_loader, to_train, n_classes)

    results = {
        "class_acc": class_acc,
        "acc": acc,
    }
    return model, results


if __name__ == "__main__":
    start_time = time.time()
    args = parseargs()

    out_path = args.out_path
    file_name = f"dataset={args.dataset}_model={args.model}_epochs={args.epochs}_lr={args.lr}"
    os.makedirs(out_path, exist_ok=True)
    model_file = os.path.join(out_path, f"{file_name}.model")
    out_file = os.path.join(out_path, f"{file_name}_results.pkl")

    if args.skip_existing and os.path.exists(model_file):
        print("Model already exists. Skipping.")
    else:
        print("Running training with args:", args)

        model, results = run_training(
            dataset_name=args.dataset,
            dataset_root=args.dataset_root,
            model=args.model,
            n_epochs=args.epochs,
            lr=args.lr,
            poison_frac=args.frac_poisoned,
        )
        print("Finished training. Accuracy:", results["acc"])

        model.save_state_dict(model_file)
        print("Model saved to", model_file)

        with open(out_file, "wb") as f:
            pkl.dump(results, f)
        print("Results saved to", out_file)

    print("Done. Took", (time.time() - start_time) / 60, "minutes.")
