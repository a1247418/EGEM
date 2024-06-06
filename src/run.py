import os
import time
import json
import matplotlib.pyplot as plt
import torch
import argparse
import numpy as np
import pickle as pkl
from torch.utils.data import DataLoader, TensorDataset
from functools import partial

from evaluation import get_n_shot_data, evaluate
from refinement.refiner import (
    EGEMRefiner,
    get_module_by_name,
    PCATruncRefiner,
    WEGEMRefiner,
    RegressionRefiner,
    PEGEMRefiner
)
from models.model_loading import load_model
from experiment_config import get_experiment_config, get_refinement_hyperparams
from refinement.explainer import Explainer
from data.data_loading import load_scenario


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa(
        "--refinement",
        type=str,
        default="none",
        choices=["none", "egem", "pcaegem", "ridge", "pcatrunc", "wegem", "pegem", "pep"],
    )  # TODO implement
    aa("--scenario_name", type=str)
    aa("--data_root", type=str)
    aa("--refinement_path", type=str, default=None)
    aa("--n_reps", type=int, default=3)
    aa("--n_samples", type=int, default=500)
    aa("--refiner_kwargs", type=str, default=None, help="JSON string")
    aa(
        "--poisoning_strategy",
        type=str,
        default="none",
        help="none, uniform, targeted, or adversarial",
        choices=["none", "uniform", "targeted", "adversarial"],
    )
    aa("--model", type=str, default=None)
    aa("--layer_names", type=str, nargs="+", default=None)
    aa("--subfolder", type=str, default=None)
    aa("--skip_existing", action="store_true")
    args = parser.parse_args()
    return args


def get_refiner_class(refinement_name: str):
    if refinement_name == "egem":
        return EGEMRefiner
    elif refinement_name == "pcaegem":
        return partial(EGEMRefiner, do_pca=True)
    elif refinement_name == "pegem":
        return partial(PEGEMRefiner, binary_mask=False)
    elif refinement_name == "pep":
        return partial(PEGEMRefiner, binary_mask=True)
    elif refinement_name == "pcatrunc":
        return PCATruncRefiner
    elif refinement_name == "wegem":
        return WEGEMRefiner
    elif refinement_name == "ridge":
        return RegressionRefiner
    elif refinement_name == "none":
        return None
    else:
        raise ValueError(f"Refinement {refinement_name} not implemented")


def get_data_loaders(scenario_name, data_root, exp):
    kwargs = {"normalize": True,
              "val_batch_size": exp["batch_size"],
              "refine_batch_size": exp["batch_size"]
              }
    # poisoned_kwargs = {}
    if exp["n_test"] is not None:
        kwargs.update({"val_set_size": exp["n_test"]})

    if scenario_name == "mnist-8":
        kwargs["normalize"] = False
        print("###################################")
        print("WARNING: not normlizing MNIST-8!")
        print("###################################")

    train_loader, refine_loader, test_loader = load_scenario(
            scenario_name=scenario_name,
            scenario_path=data_root,
            poisoning_strategy=exp["poisoning_strategy"],
            **kwargs,
    )
    return train_loader, refine_loader, test_loader


def get_explainer(model, n_classes, exp):
    explainer = Explainer(
            model,
            n_classes=n_classes,
            explanation_type=exp["explanation_type"],
            canonizer="vgg"
            if "vgg" in exp["model_name"]
            else "resnet"
            if "resnet" in exp["model_name"]
            else None,
            sensitivity_only=False if exp["explanation_type"] == "gradient" else True,
    )
    return explainer


def run_experiment(
    scenario_name: str,
    data_root: str,
    refinement: str,
    n_reps: int,
    n_samples: int,
    poisoning_strategy: str,
    seed:int=42,
    refinement_path:str=None,
    refiner_kwargs:str=None,
    verbose:bool=False,
    **kwargs,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp = get_experiment_config(scenario_name, refinement)
    exp.update(
        {
            "n_reps": n_reps,
            "n_refine": n_samples,
            "poisoning_strategy": poisoning_strategy,
        }
    )
    # Overwrite with provided params
    for k, v in kwargs.items():
        if k in exp:
            exp[k] = v
        else:
            raise ValueError(f"Invalid argument {k}")

    if refiner_kwargs is not None:
        loaded_refiner_kwargs = json.loads(refiner_kwargs)
    else:
        loaded_refiner_kwargs = {}

    model = load_model(
        model_name=exp["model_name"],
        file_path=exp["model_file_path"],
        out_classes=[exp["target_class"]] + exp["background_classes"]
        if len(exp["background_classes"]) == 1
        else sorted([exp["target_class"]] + exp["background_classes"]),
        device=device,
    )

    # Load data
    _, refine_loader, test_loader = get_data_loaders(scenario_name, data_root, exp)
    n_classes = len(exp["background_classes"]) + 1
    (
        refine_images,
        refine_targets,
    ) = get_n_shot_data(
        n_shot=n_samples,
        n_classes=n_classes,
        loader=refine_loader,
        n_reps=n_reps,
        device=device,
    )

    # Evaluate
    hyperparams = get_refinement_hyperparams(refinement)

    results = []

    for r in range(n_reps):
        print(
            f"Rep {r}: refinement {refinement}, scenario {scenario_name}, model {exp['model_name']}"
        )

        tensor_x = torch.Tensor(refine_images[r])
        tensor_y = torch.Tensor(refine_targets[r])
        n = int(tensor_x.size()[0])
        indices = torch.randperm(n)
        tensor_x = tensor_x[indices]
        tensor_y = tensor_y[indices]

        n_val = int(np.ceil(n * exp["fraction_val"]))
        n_ref = n - n_val
        print(f"Refining on {n_ref} samples, validating on {n_val}, and testig on {exp['n_test'] if 'n_test' in exp else 'all test'} samples")

        tensor_refine_dataset = TensorDataset(tensor_x[:n_ref], tensor_y[:n_ref])
        refine_tensor_loader = DataLoader(
            tensor_refine_dataset, batch_size=exp["batch_size"]
        )

        if n_val > 0:
            tensor_val_dataset = TensorDataset(tensor_x[-n_val:], tensor_y[-n_val:])
            val_tensor_loader = DataLoader(
                tensor_val_dataset, batch_size=min(n_val, exp["batch_size"])
            )
        else:
            val_tensor_loader = DataLoader(
                tensor_refine_dataset, batch_size=min(n_ref, exp["batch_size"])
            )

        if exp["refinement"] != "none":
            refiner_class = get_refiner_class(exp["refinement"])
            refiner_kwargs = {
                "layer_names": exp["layer_names"],
                "n_classes": n_classes,
                "verbose": verbose
            }
            refiner_kwargs.update(loaded_refiner_kwargs)

            if exp["explanation_type"] is not None:
                explainer = get_explainer(model, n_classes, exp)
                refiner_kwargs["explainer"] = explainer

                if verbose and not "vit" in exp["model_name"]:
                    # TODO: Remove in final version
                    print("Plotting explanation for a few examples")
                    for i in range(3):
                        with explainer.attributor:
                            out, r = explainer.attributor(tensor_x[i:(i+1)], torch.eye(n_classes, device="cuda",
                                                                            dtype=torch.int)[[tensor_y[i:(i+1)]]])
                        fig, axs = plt.subplots(1, 2)
                        axs[0].imshow(torch.permute(tensor_x[i], (1, 2, 0)).cpu().numpy())
                        mnx = max(-torch.min(r), torch.max(r))
                        axs[1].imshow(torch.permute(r[0], (1, 2, 0)).cpu().numpy(), vmin=-mnx, vmax=mnx, cmap="bwr")
                        plt.show()
                        print("R:",torch.min(r).item(),"--", torch.max(r).item())
                        print("x:", torch.min(tensor_x[i]).item(),"--", torch.max(tensor_x[i]).item())

            # Hyperparam search
            chosen_hyperparams = {}
            if len(hyperparams) != 0:
                for h_i, (k, vals) in enumerate(hyperparams.items()):
                    if h_i > 0:
                        raise NotImplementedError(
                            "Hyperparam search for more than one parameter not implemented"
                        )
                    best = None
                    if False:  # len(vals) == 1: TODO reactivate for refitting
                        chosen_hyperparams[k] = vals[0]
                    else:
                        for val in vals:
                            print(f"{exp['refinement']} Hyperparam: {k}={val}")
                            refiner = refiner_class(
                                model.model,
                                device=device,
                                **dict(refiner_kwargs, **{k: val}),
                            )
                            refiner.train_refinement(refine_tensor_loader)
                            if refinement_path is not None:
                                try:
                                    file_name = refiner.get_filename()
                                    save_path = os.path.join(refinement_path, f"rep{r}_" + file_name)
                                    refiner.save(save_path)
                                except Exception as e:
                                    print(f"Could not save refiner: {repr(e)}")

                            refiner.refine()
                            print("-------- Validation set --------")
                            result_val = evaluate(
                                model=model,
                                data_loader=val_tensor_loader,
                                device=device,
                            )
                            print("-------- Test set --------")
                            result = evaluate(
                                model=model,
                                data_loader=test_loader,
                                device=device
                            )
                            refiner.unrefine()

                            if best is None or result["top1"] > best["top1"]:
                                best = result
                                chosen_hyperparams[k] = val

                            to_return = {"rep": r, k: val}
                            to_return.update(exp)
                            to_return.update(result)
                            result_val = {k + "_val": v for k, v in result_val.items()}
                            to_return.update(result_val)
                            results.append(to_return)
        else:
            # TODO implement refitting on data + refine for none and others
            # refiner = refiner_class(model.model, layer_names=exp["layer_names"])#, **chosen_hyperparams)
            # refiner.train_refinement(refine_tensor_loader)
            result_val = evaluate(
                model=model, data_loader=val_tensor_loader, device=device
            )  #  TODO should this be on the whole data
            result = evaluate(model=model, data_loader=test_loader, device=device)
            to_return = {"rep": r}
            to_return.update(exp)
            to_return.update(result)
            result_val = {k + "_val": v for k, v in result_val.items()}
            to_return.update(result_val)
            results.append(to_return)

    return results


if __name__ == "__main__":
    start_time = time.time()

    args = parseargs()

    # Make paths
    file_name = f"sc={args.scenario_name}_re={args.refinement}_rep={args.n_reps}_n={args.n_samples}_p={args.poisoning_strategy}"
    if args.layer_names is not None:
        file_name += "_" + "_".join(args.layer_names)
    if args.refiner_kwargs is not None:
        refiner_kwargs_str = args.refiner_kwargs.replace("{", "").replace("}", "").replace(":", "=").replace(",", "_").replace(" ", "")
        file_name += f"_{refiner_kwargs_str}"
    folder = "results"
    if args.subfolder is not None:
        folder = os.path.join(folder, args.subfolder)
    os.makedirs(folder, exist_ok=True)
    out_file = os.path.join(folder, f"{file_name}_acc.pkl")

    if args.skip_existing and os.path.exists(out_file):
        print(f"Skipping {out_file} -- already exists")
        exit()

    results = run_experiment(
            scenario_name=args.scenario_name,
            data_root=args.data_root,
            refinement=args.refinement,
            n_reps=args.n_reps,
            n_samples=args.n_samples,
            poisoning_strategy=args.poisoning_strategy,
            refinement_path=args.refinement_path,
            layer_names=args.layer_names,
            refiner_kwargs=args.refiner_kwargs,
    )

    # Save results
    with open(out_file, "wb") as f:
        pkl.dump(results, f)
    print(f"Saved to {out_file}")

    print("Done. Took", (time.time() - start_time) / 60, "minutes.")
