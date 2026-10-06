from functools import reduce, partial
from typing import List, Optional
import torch
from sklearn.decomposition import PCA


def get_module_by_name(module: torch.nn.Module, access_string: str):
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)


def layer_names_to_layers(model: torch.nn.Module, layer_names: List[str]):
    # TODO: ensure right order
    try:
        layers = [get_module_by_name(model, layer_name) for layer_name in layer_names]
    except AttributeError as e:
        print("Error while getting layers:", e)
        print("  Layer names:", layer_names)
        print("  Model:", model)
        raise e
    return layers


MAX_SPATIAL_ROWS = 100000  # conv activations are subsampled to this many (position, channel) rows


def captured_to_list(layer_names, captured, reduce_spatial=True, del_captured=True, divisor=1.):
    cs = []
    for l in layer_names:
        cs.append(torch.cat(captured[l], dim=0).to(torch.float32) / divisor)
        if reduce_spatial and len(cs[-1].shape) == 4:
            #cs[-1] = torch.sum(cs[-1], dim=[-2, -1]) # alternative: sum pooling
            #cs[-1] = torch.mean(cs[-1], dim=[-2, -1]) # alternative: avg pooling
            cs[-1] = cs[-1].permute(1,0,2,3).reshape([cs[-1].shape[1], -1]).T # [bs, n_channels, h, w] -> [bs*h*w, n_channels]
            # if too many, subsample:
            if cs[-1].shape[0] > MAX_SPATIAL_ROWS:
                idxs = torch.randperm(cs[-1].shape[0])[:MAX_SPATIAL_ROWS]
                cs[-1] = cs[-1][idxs]

    if del_captured:
        del captured

    return cs


def get_activations(
    model: torch.nn.Module,
    layer_names: List[str],
    loader,
    do_pca: bool = False,
    pca_dims: Optional[int] = None,
    cls_token_only: bool = False,
    average_tokens: bool = False,
    square: bool = True,
    reduce: bool = True,
    explainer: Optional = None,
    combine_a_exp: bool = False,
    batch_dim: int = 1,
    capture_outputs: bool = False,
    avg_on_the_fly = False,
    reduce_spatial=True,
    spatial_sum: bool = False,
    device: str = "cuda",
    silent: bool = True,
):
    """
    Returns the activations of the given layers for the data given by the loader.
    :param model: The model to get the activations from.
    :param layer_names: List of names of the layers to get the activations from.
    :param loader: The loader to get the data from.
    :param do_pca: Whether to use PCA-transformed activations. [True, False]
    :param cls_token_only: Whether to only use the activations of the classifier token.
    :param square: Whether to square the activations. [True, False]
    :param reduce: Whether to reduce the activations to a single vector. [True, False]
    :param capture_explanations: Whether to capture the explanations of the layers. [True, False]
    :param combine_a_exp: Whether to combine the activations and explanations. [True, False]
    :param batch_dim: The batch-dimension in the input. [0,1], [batch, n_tokens, dimension] if batch_dim == 0 else [n_tokens, batch, dimension]
    :param capture_outputs: Whether to capture the outputs of the layers or the inputs.
    :param reduce_spatial: Whether to reduce the spatial dimensions of the activations. [True, False]
    :param spatial_sum: For conv layers, sum each channel over the spatial dimensions (one value per sample)
        instead of treating every position as a sample.
    :param device: The device to use. ["cuda", "cpu"]
    """
    # TODO: different batch_dims/layer
    assert batch_dim in [0, 1]
    layers = layer_names_to_layers(model, layer_names)

    capture_explanations = explainer is not None
    # Number of samples to spread MAX_SPATIAL_ROWS over; None = keep full conv maps (needed when the
    # activations must stay aligned with explanations, or when spatial dimensions are not reduced)
    spatial_keep = None
    if reduce_spatial and not capture_explanations and not avg_on_the_fly and hasattr(loader, "dataset"):
        spatial_keep = len(loader.dataset)

    def capture_activations(module, input, output, captured_activations):
        # input is a tuple with the first element being the input Tensor
        # input[0]: [n_tokens / channels, batch, dimensions] if batch_dim == 1
        # input[0]: [batch, n_tokens / channels, dimensions] if batch_dim == 0
        if not silent:
            print("Shape of captured activations in/out:", input[0].shape, output.shape)

        if capture_outputs:
            activations = output[None]
        else:
            activations = input

        if len(activations[0].shape) == 2:
            # Linear layer, input[0]: [batch, dimension]
            captured_activations.append(activations[0].clone())
        elif len(activations[0].shape) == 3:
            # Attention layer,
            if cls_token_only:
                captured_activations.append(
                    (
                        activations[0][0] if batch_dim == 1 else activations[0][:, 0]
                    ).clone()
                )
            elif average_tokens:
                captured_activations.append(
                    activations[0].mean(dim=[0] if batch_dim == 1 else [1]).clone()
                )
            else:
                captured_activations.append(
                    activations[0].clone().reshape(-1, activations[0].shape[-1])
                )
        else:
            # Conv layer
            # TODO, make adapt with batch dim, and set default to 0
            if spatial_sum:
                captured_activations.append(activations[0].sum(dim=[-2, -1]).clone())
            elif spatial_keep is not None:
                # Subsample spatial positions per batch to bound memory
                rows = activations[0].permute(0, 2, 3, 1).reshape(-1, activations[0].shape[1])
                k = min(rows.shape[0], -(-MAX_SPATIAL_ROWS * activations[0].shape[0] // spatial_keep))
                captured_activations.append(rows[torch.randperm(rows.shape[0], device=rows.device)[:k]].clone())
            else:
                captured_activations.append(activations[0].clone())#.sum(dim=[-2, -1]).clone())

        if avg_on_the_fly:
            if len(captured_activations)==1:
                captured_activations[0] = captured_activations[0].sum(dim=0, keepdim=True)
            if len(captured_activations)==2:
                captured_activations[0] += captured_activations.pop().sum(dim=0, keepdim=True)


    all_captured_activations = {l:[] for l in layer_names}  # List to store the activation tensors in the form of [batch*n_tokens, dimension]
    all_captured_explanations = {l:[] for l in layer_names}
    hooks = []
    n_samples = 0
    if capture_explanations:
        for x, y in loader:
            # Explain and capture activation at the same time, to ensure that the same input is used
            x = x.to(device)
            y = y.to(device)
            n_samples += x.shape[0]
            # explain
            for x_,y_ in zip(x,y):
                expls = explainer.explain(x_[None], y_[None], module_list=layers)
                for l, e in zip(layer_names, expls):
                    all_captured_explanations[l].append(e)
            # capture hooks
            for layer_name, layer in zip(layer_names, layers):
                hook = layer.register_forward_hook(partial(capture_activations, captured_activations=all_captured_activations[layer_name]))
                hooks.append(hook)
            # capture
            with torch.no_grad():
                model(x.to(device))
            # unhook
            for hook in hooks:
                hook.remove()
            hooks = []
    else:
        for layer_name, layer in zip(layer_names, layers):
            hook = layer.register_forward_hook(partial(capture_activations, captured_activations=all_captured_activations[layer_name]))
            hooks.append(hook)

        with torch.no_grad():
            for x, y in loader:
                n_samples += x.shape[0]
                model(x.to(device))

    # Post-process activations
    acs = captured_to_list(layer_names, all_captured_activations, reduce_spatial=reduce_spatial, divisor=1.)#n_samples)
    # Post-process explanations
    if capture_explanations:
        rs = captured_to_list(layer_names, all_captured_explanations, reduce_spatial=reduce_spatial, divisor=1.)#n_samples)

    if not silent:
        print("Shape of recorded activations:")
        for l_i in range(len(layers)):
            print("  a:", acs[l_i].shape)
            if capture_explanations:
                print("  r:", rs[l_i].shape)

    if do_pca:
        if not silent:
            print("Using PCA")
        # Calculate PCA on activations
        pca_V = []
        pca_Xm = []
        for l_i in range(len(layers)):
            X = acs[l_i]  # cast f16->f32 to avoid inf values
            print("layer",l_i, "shape", X.shape)
            max_n_components = min(X.shape[0], X.shape[1])
            if pca_dims is not None:
                max_n_components = min(pca_dims, max_n_components)
            pca = PCA(n_components=max_n_components)
            pca.fit(X.detach().cpu())
            Xm = torch.from_numpy(pca.mean_).to(X.device, dtype=X.dtype)
            V = torch.tensor(pca.components_, device=X.device, dtype=X.dtype)
            pca_V.append(V)
            pca_Xm.append(Xm)
            if not silent:
                print("\tmean:", Xm.shape, "rotation", V.shape)
        acs = [
            (torch.matmul((ac - pca_Xm[i]), pca_V[i].T))
            for i, ac in enumerate(acs)
        ]
        assert torch.sum(torch.isinf(acs[0])) == 0
        assert torch.sum(torch.isnan(acs[0])) == 0
    else:
        pca_Xm = None
        pca_V = None

    if square:
        acs = [ac ** 2 for ac in acs]
        if capture_explanations:
            rs = [r ** 2 for r in rs]

    if reduce:
        if combine_a_exp:
            combined = []
            for a, r in zip(acs, rs):
                outers_per_layer = None
                for a_, r_ in zip(a, r):
                    a_ = a_.reshape(1, -1)
                    r_ = r_.reshape(1, -1)
                    outer_product = a_.T @ r_
                    if outers_per_layer is None:
                        outers_per_layer = outer_product
                    else:
                        outers_per_layer += outer_product
                combined.append(outers_per_layer/len(a))

        acs = [torch.mean(ac, dim=[0]) for ac in acs]

        if capture_explanations:
            rs = [torch.mean(r, dim=[0]) for r in rs]

    # Remove the hooks to release the resources
    for hook in hooks:
        hook.remove()

    # check for nans
    for accc in acs:
        assert torch.sum(torch.isinf(accc)) == 0
        assert torch.sum(torch.isnan(accc)) == 0

    if capture_explanations:
        if combine_a_exp and reduce:
            return acs, rs, combined, pca_Xm, pca_V
        else:
            return acs, rs, pca_Xm, pca_V
    else:
        return acs, pca_Xm, pca_V


def replace_layer(model: torch.nn.Module, layer_name: str, mod_layer: torch.nn.Module):
    parts = layer_name.split(".")
    if len(parts) > 1:
        if parts[-1].isdigit():
            # Replace the entry of the sequential
            get_module_by_name(model, ".".join(parts[:-1]))[int(parts[-1])] = mod_layer
        else:
            # Replace a module
            get_module_by_name(model, ".".join(parts[:-1])).__setattr__(parts[-1], mod_layer)
    else:
        # Replace a module
        model.__setattr__(layer_name, mod_layer)


def calculate_sensitivity(r_mat, em_a):
    # TODO: adapt to conv em
    for _ in range(len(r_mat.shape) - len(em_a.shape)):
        em_a = em_a.unsqueeze(-1)
    s_mat = (r_mat / (em_a + (em_a == 0) + torch.sign(em_a)*1e-5)).detach()

    assert not torch.isnan(s_mat).any()
    assert not torch.isinf(s_mat).any()

    return s_mat