import os
import pickle as pkl
import pandas as pd
from typing import Union, List


def add_metrics(res: dict):
    confusion = res["confusion"]
    true = res["true"]
    predicted = res["predicted"]
    if confusion.shape[0] == 2:
        # add FP, FN, TP, TN, FPR, TPR
        if res["target_class"] < res["background_classes"][0]:
            confusion = confusion.T
        res["FP"] = confusion[0, 1]
        res["FN"] = confusion[1, 0]
        res["TP"] = confusion[1, 1]
        res["TN"] = confusion[0, 0]
    else:
        # add FP, FN, TP, TN, FPR, TPR by summing over all background classes
        t = res["target_class"]
        res["FP"] = confusion[:, t].sum() - confusion[t, t]
        res["FN"] = confusion[t, :].sum() - confusion[t, t]
        res["TP"] = confusion[t, t]
        res["TN"] = confusion.sum() - res["FP"] - res["FN"] - res["TP"]

        # if multiclass classification, add per class accuracy
        per_class_accuracy = []
        for class_label in range(confusion.shape[0]):
            class_mask = true == class_label
            correct = (predicted == true)
            try:
                correct = correct.float()
            except AttributeError:
                pass
            class_accuracy = correct[class_mask].mean().item()
            per_class_accuracy.append(class_accuracy)
            for i, acc in enumerate(per_class_accuracy):
                res[f"per_class_acc_{i}"] = acc

    res["FPR"] = res["FP"] / (res["FP"] + res["TN"]) if (res["FP"] + res["TN"]) > 0 else 0
    res["FNR"] = res["FN"] / (res["FN"] + res["TP"]) if (res["FN"] + res["TP"]) > 0 else 0
    res["TPR"] = res["TP"] / (res["TP"] + res["FN"]) if (res["TP"] + res["FN"]) > 0 else 0
    res["TNR"] = res["TN"] / (res["TN"] + res["FP"]) if (res["TN"] + res["FP"]) > 0 else 0
    res["precision"] = res["TP"] / (res["TP"] + res["FP"]) if (res["TP"] + res["FP"]) > 0 else 0
    res["recall"] = res["TP"] / (res["TP"] + res["FN"]) if (res["TP"] + res["FN"]) > 0 else 0


def load_all_results(directory: str, filter_str: Union[List[str], str] = None, as_reduced_dataframe: bool = False):
    """Load all results from the result folder.
    :keyword directory: The directory to load the results from.
    :keyword filter_str: A string or list of strings to filter the files to load. Only files containing the filter string will be loaded.
    :keyword as_reduced_dataframe: If True, the results will be returned as a reduced dataframe, excluding sequence/array results.
    """
    loaded = []
    for file in os.listdir(directory):
        if file.endswith(".pkl"):
            if (
                filter_str is None
                or (type(filter_str) == str and filter_str in file)
                or (type(filter_str) == list and any([fs in file for fs in filter_str]))
            ):
                with open(os.path.join(directory, file), "rb") as f:
                    res = pkl.load(f)
                    for r in res:
                        add_metrics(r)
                    loaded.extend(res)


                    # TODO REMOVE this whene deprecated
                    if "layer_names" in file:
                        for r in res:
                            lns = file.split("[")[1].split("]")[0].split("_")
                            r["layer_names"] = [l.replace('"', '') for l in lns]

            else:
                print(f"Skipping {file} as it does not contain the filter string.")

    if as_reduced_dataframe:
        cols = set()
        for i, r in enumerate(loaded):
            cols = cols.union(set(r.keys()))

        to_exclude = ['true', 'predicted', 'confusion', 'output']
        to_exclude += [s+"_val" for s in to_exclude]
        to_exclude += ["background_classes"] # "layer_names"
        to_exclude += [s for s in cols if "per_class_acc" in s]
        cols = list(cols - set(to_exclude))

        rows = []
        for r in loaded[::-1]:
            row = []
            for c in cols:
                try:
                    row.append(r[c][0] if "top1" in c else r[c])
                except KeyError as e:
                    print(repr(e) + f" for {r['refinement']}")
                    row.append(None)
            rows.append(row)

        loaded = pd.DataFrame(rows, columns=cols)
    return loaded


def custom_agg(df, slack, hyperparam="alpha", ignore_cols=None, higher_is_less=True):
    ref = list(df.refinement.unique())
    if len(ref) != 2:
        raise ValueError(
            f"Not exactly 2 (none and ?) refinement methods in selection: {ref}"
        )

    baseline = df[df.refinement == "none"]["top1_val"].iloc[0]

    sorted_df = df.sort_values(by=hyperparam, ascending=not higher_is_less)

    selected_row = None
    for index, row in sorted_df.iterrows():
        if (
            selected_row is None
            or row["refinement"] != "none"
            and row["top1_val"] >= baseline * (1 - slack)
        ):
            selected_row = row

    to_return = selected_row

    if ignore_cols is not None:
        to_return = to_return.drop(ignore_cols)

    return to_return


def select_best(df, slack, hyperparam="alpha", higher_is_less=True):
    assert df.empty is False, "Dataframe is empty."

    group_cols = ["rep", "n_refine", "scenario_name", "poisoning_strategy"]

    df_mod = df.groupby(group_cols).apply(
        custom_agg,
        slack=slack,
        hyperparam=hyperparam,
        ignore_cols=group_cols,
        higher_is_less=higher_is_less,
    )
    df_mod = df_mod.reset_index()
    df_mod = df_mod[df_mod.refinement != "none"]
    return df_mod.copy()


def filter_for_ref(df_original, refinements):
    df = df_original[df_original.refinement.isin(refinements)]
    return df.copy()


def beautify_cols(var: str):
    if type(var) != str:
        return var
    beautify_dict = {
        "top1": "Accuracy",
        "top1_val": "Val. Accuracy",
        "n_refine": "Samples per class",
    }
    if var in beautify_dict.keys():
        return beautify_dict[var]
    else:
        if len(var) > 3:
            var = var.title()
        return var.replace("_", " ")


def beautify_vals(var: str):
    if type(var) != str:
        return var
    beautify_dict = {"egem": "EGEM",
                     "pcaegem": "PCA-EGEM",
                     "pegem": "PEGEM",
                     "pep": "PEP",
                     "none": "None",
                     "uniform": "Uniform"}
    if var in beautify_dict.keys():
        return beautify_dict[var]
    else:
        return var.title().replace("_", " ")
