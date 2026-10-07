import os
from typing import Optional, List, Tuple, Union
import torch
import numpy as np
from sklearn.decomposition import PCA
from zennit.core import BasicHook

from refinement.decomposition.cxai import factory, inspector, constants, drsa
from refinement.helpers import get_module_by_name, replace_layer


class AdditiveLayer(torch.nn.Module):
    def __init__(self, input_dim):
        super(AdditiveLayer, self).__init__()
        self.bias = torch.nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        return x + self.bias


class Decomposer(torch.nn.Module):
    def __init__(
        self,
        data: torch.Tensor,
        n_components: Optional[int],
        component_dim: int = 1,
        load_path: Optional[str] = None,
        encoder_activation: Optional[torch.nn.Module] = None,
        pre_encoder_bias: bool = False,
    ):
        super().__init__()
        self.is_conv = len(data.shape) == 4  # TODO: adapt to transformers
        self.n_components = n_components
        self.component_dim = component_dim
        self.encoder = None
        self.decoder = None

        self.forward_blocked_components = []
        self.forward_blocked_memory = {}
        self.backward_blocked_components = []
        self.backward_blocked_handle = None

        self.is_attached = False

        if load_path is not None:
            if self.is_conv:
                self.encoder, self.decoder = self._create_1x1conv_enc_dec(
                    W1=torch.zeros(n_components, data.shape[1]),
                    W2=torch.zeros(data.shape[1], n_components),
                    pre_encoder_bias=pre_encoder_bias,
                    encoder_activation=encoder_activation,
                )
            else:
                self.encoder, self.decoder = self._create_linear_enc_dec(
                    W1=torch.zeros(n_components, data.shape[1]),
                    W2=torch.zeros(data.shape[1], n_components),
                    pre_encoder_bias=pre_encoder_bias,
                    encoder_activation=encoder_activation,
                )
            self.load(load_path)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

    def modify_latent(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.modify_latent(self.encode(x)))

    def block_forward_components(
        self, components: Union[int, List[int]], use_component_complement: bool = False
    ) -> None:
        """Block forward pass for the specified components by setting their weights and biases in the encoder to zero."""
        if isinstance(components, int):
            components = [components]

        if use_component_complement:
            components = list(set(range(self.n_components)) - set(components))

        components = list(set(components).union(self.forward_blocked_components))

        for c in components:
            self.forward_blocked_memory[c] = (
                self.encoder.weight[
                    c * self.component_dim : (c + 1) * self.component_dim
                ].clone(),
                self.encoder.bias[c].clone(),
            )
            self.encoder.weight[
                (c * self.component_dim) : ((c + 1) * self.component_dim)
            ] = 0
            self.encoder.bias[c] = 0

        self.forward_blocked_components = components

    def unblock_forward_components(
        self,
        components: Optional[Union[int, List[int]]] = None,
        use_component_complement: bool = False,
    ) -> None:
        if components is None:
            components = self.forward_blocked_components
        elif isinstance(components, int):
            components = [components]

        if use_component_complement:
            components = list(set(range(self.n_components)) - set(components))

        for c in components:
            self.encoder.weight[
                (c * self.component_dim) : ((c + 1) * self.component_dim)
            ] = self.forward_blocked_memory[c][0]
            self.encoder.bias[c] = self.forward_blocked_memory[c][1]
            del self.forward_blocked_memory[c]

        self.forward_blocked_components = list(
            set(self.forward_blocked_components) - set(components)
        )

    def _get_filter_hook(self, selector: torch.Tensor):
        def grad_mapper(out_grad, outputs):
            s = selector.unsqueeze(dim=0)
            while len(s.shape) < len(out_grad.shape):
                s = s.unsqueeze(-1)
            try:
                return [out_grad * s / (outputs[0] + (outputs[0] == 0))]
            except RuntimeError as e:
                print(
                    "out_grad:",
                    out_grad.shape,
                    "selector:",
                    s.shape,
                    "outputs:",
                    outputs[0].shape,
                )
                raise e

        def reducer(inputs, gradients):
            return inputs[0] * gradients[0]

        hook = BasicHook(
            gradient_mapper=grad_mapper,
            reducer=reducer,
        )
        return hook

    def _get_file_name(self, postfix: Optional[str] = None):
        # name including the name of the subclass it is instantiated as
        name = (
            f"{self.__class__.__name__}_{self.n_components}_{self.component_dim}.decomp"
        )
        if postfix is not None:
            name = name.replace(".decomp", f"_{postfix}.decomp")
        return name

    def save(self, path: str, postfix: Optional[str] = None) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.state_dict(), os.path.join(path, self._get_file_name(postfix)))

    def load(self, path: str, postfix: Optional[str] = None) -> None:
        print("Loading from", os.path.join(path, self._get_file_name(postfix)))
        self.load_state_dict(
            torch.load(os.path.join(path, self._get_file_name(postfix)))
        )

    def block_backward_components(
        self, components: Union[int, List[int]], use_component_complement: bool = False
    ) -> None:
        """Block backward pass for the specified components by setting their gradients to zero."""
        if isinstance(components, int):
            components = [components]

        if use_component_complement:
            components = list(set(range(self.n_components)) - set(components))

        components = list(set(components).union(self.backward_blocked_components))

        mask = torch.ones(
            self.component_dim * self.n_components,
            device=self.encoder.weight.device,
            dtype=self.encoder.weight.dtype,
        )
        for c in components:
            mask[(c * self.component_dim) : ((c + 1) * self.component_dim)] = 0
        mask = mask.unsqueeze(0)  # Needs leading "batch" dimension to work
        if self.is_conv:
            mask = mask.unsqueeze(-1).unsqueeze(-1)

        def hook(mod, grad_outputs):
            try:
                return (grad_outputs[0] * mask,)
            except RuntimeError as e:
                print("grad_outputs:", grad_outputs[0].shape, "mask:", mask.shape)
                raise e

        self.backward_blocked_handle = self.encoder.register_full_backward_pre_hook(
            hook
        )
        """
        hook = self._get_filter_hook(mask)
        self.backward_blocked_handle = hook.register(self.decoder)
        """

        self.backward_blocked_components = components

    def unblock_backward_components(
        self,
        components: Optional[Union[int, List[int]]] = None,
        use_component_complement: bool = False,
    ) -> None:
        if self.backward_blocked_handle is not None:
            self.backward_blocked_handle.remove()
            self.backward_blocked_handle = None

            if components is None:
                self.backward_blocked_components = []
            else:
                if isinstance(components, int):
                    components = [components]

                if use_component_complement:
                    components = list(set(range(self.n_components)) - set(components))

                remaining_components = list(
                    set(self.backward_blocked_components) - set(components)
                )
                if len(remaining_components) == 0:
                    self.backward_blocked_components = []
                else:
                    self.block_backward_components(remaining_components)
                    self.backward_blocked_components = remaining_components

    def attach_to_model(self, model: torch.nn.Module, layer_name: str) -> None:
        if self.is_attached:
            raise ValueError("Decomposer is already attached to a model.")
        layer_module = get_module_by_name(model, layer_name)
        layer_plus_decomposer = torch.nn.Sequential(layer_module, self)
        replace_layer(model, layer_name, layer_plus_decomposer)
        self.is_attached = True

    def detach_from_model(self, model: torch.nn.Module, layer_name: str) -> None:
        if not self.is_attached:
            raise ValueError("Decomposer is not attached to a model.")

        layer_module = get_module_by_name(model, layer_name + ".0")
        replace_layer(model, layer_name, layer_module)
        self.is_attached = False

    def _create_linear_enc_dec(
        self,
        W1: torch.Tensor,
        W2: torch.Tensor,
        b1: Optional[torch.Tensor] = None,
        b2: Optional[torch.Tensor] = None,
        pre_encoder_bias: bool = False,
        encoder_activation: Optional[torch.nn.Module] = None,
    ) -> Tuple[torch.nn.Module, torch.nn.Module]:
        encoder = torch.nn.Linear(W1.shape[1], W1.shape[0], bias=b1 is not None)
        encoder.weight = torch.nn.Parameter(W1)
        if b1 is not None:
            encoder.bias = torch.nn.Parameter(b1)

        decoder = torch.nn.Linear(W2.shape[1], W2.shape[0], bias=b2 is not None)
        decoder.weight = torch.nn.Parameter(W2)
        if b2 is not None:
            decoder.bias = torch.nn.Parameter(b2)

        if pre_encoder_bias:
            assert b2 is not None
            pre_bias = AdditiveLayer(decoder.bias.shape[0])
            pre_bias.bias = -decoder.bias
            encoder = torch.nn.Sequential(pre_bias, encoder)

        if encoder_activation is not None:
            encoder = torch.nn.Sequential(encoder, encoder_activation)

        return encoder, decoder

    def _create_1x1conv_enc_dec(
        self,
        W1: torch.Tensor,
        W2: torch.Tensor,
        b1: Optional[torch.Tensor] = None,
        b2: Optional[torch.Tensor] = None,
        pre_encoder_bias: bool = False,
        encoder_activation: Optional[torch.nn.Module] = None,
    ) -> Tuple[torch.nn.Module, torch.nn.Module]:
        encoder = torch.nn.Conv2d(
            W1.shape[1],
            W1.shape[0],
            kernel_size=1,
            stride=1,
            padding=0,
            bias=b1 is not None,
        )
        encoder.weight = torch.nn.Parameter(W1.unsqueeze(-1).unsqueeze(-1))
        if b1 is not None:
            encoder.bias = torch.nn.Parameter(b1)

        decoder = torch.nn.Conv2d(
            W2.shape[1],
            W2.shape[0],
            kernel_size=1,
            stride=1,
            padding=0,
            bias=b2 is not None,
        )
        decoder.weight = torch.nn.Parameter(W2.unsqueeze(-1).unsqueeze(-1))
        if b2 is not None:
            decoder.bias = torch.nn.Parameter(b2)

        if pre_encoder_bias:
            assert b2 is not None
            pre_bias = AdditiveLayer(decoder.bias.shape[0])
            pre_bias.bias = -decoder.bias
            encoder = torch.nn.Sequential(pre_bias, encoder)

        if encoder_activation is not None:
            encoder = torch.nn.Sequential(encoder, encoder_activation)

        return encoder, decoder

    def reduce_spatial(self, data, max_n_samples=100000):
        if len(data.shape) == 4:
            data = (
                data.permute(1, 0, 2, 3).reshape([data.shape[1], -1]).T
            )  # [bs, n_channels, h, w] -> [bs*h*w, n_channels]
            # if too many, subsample:
            if data.shape[0] > max_n_samples:
                idxs = torch.randperm(data.shape[0])[:max_n_samples]
                data = data[idxs]
        return data


class IdentityDecomposer(Decomposer):
    def __init__(self, data: torch.Tensor, device: str):
        super().__init__(data, n_components=data.shape[1], component_dim=1)

        # Create encoder and decoder
        eye = torch.eye(self.n_components, device=device)
        if self.is_conv:
            self.encoder, self.decoder = self._create_1x1conv_enc_dec(W1=eye, W2=eye)
        else:
            self.encoder, self.decoder = self._create_linear_enc_dec(W1=eye, W2=eye)


class PCADecomposer(Decomposer):
    def __init__(
        self,
        data: torch.Tensor,
        n_components: Optional[int],
        load_path: Optional[str] = None,
    ):
        super().__init__(data, n_components, component_dim=1, load_path=load_path)

        if load_path is None:
            # Fit PCA
            data = self.reduce_spatial(data)
            X = data.detach().cpu().numpy()
            max_n_components = min(X.shape[0], X.shape[1])
            if n_components is not None:
                max_n_components = min(n_components, max_n_components)
            pca = PCA(n_components=max_n_components)
            pca.fit(X)
            Xm = torch.from_numpy(pca.mean_).to(data.device, dtype=data.dtype)
            V = torch.tensor(pca.components_, device=data.device, dtype=data.dtype)

            # Create encoder and decoder
            if self.is_conv:
                self.encoder, self.decoder = self._create_1x1conv_enc_dec(
                    W1=V, W2=V.T, b1=-torch.matmul(Xm, V.T), b2=Xm
                )
            else:
                self.encoder, self.decoder = self._create_linear_enc_dec(
                    W1=V, W2=V.T, b1=-torch.matmul(Xm, V.T), b2=Xm
                )

        self.n_components = self.encoder.weight.shape[0]


class PRCADecomposer(Decomposer):
    # Adapted from https://github.com/p16i/drsa-demo/blob/main/notebooks/disentangled-explanations.ipynb
    def __init__(
        self,
        data: torch.Tensor,
        data_context: torch.Tensor,
        n_components: Optional[int],
        load_path: Optional[str] = None,
    ):
        super().__init__(data, n_components, component_dim=1, load_path=load_path)

        if load_path is None:
            data = self.reduce_spatial(data)
            data_context = self.reduce_spatial(data_context)
            mat_act = data.detach().cpu().numpy()
            mat_ctx = data_context.detach().cpu().numpy()

            # compute the symmetrized cross-correlation matrix
            mat_cross_cov = (mat_act.T @ mat_ctx + mat_ctx.T @ mat_act) / mat_act.shape[
                0
            ]

            # compute the eigen decomposition of the matrix
            _, eigvecs = np.linalg.eigh(mat_cross_cov)

            # in Numpy, the order of eigenvalues is in ascending.
            # Therefore, we have to reverse the order.
            U_prca = torch.from_numpy(eigvecs[:, ::-1].copy()).to(
                device=data.device, dtype=data.dtype
            )

            if n_components is not None:
                U_prca = U_prca[:, :n_components]

            # Create encoder and decoder
            if self.is_conv:
                self.encoder, self.decoder = self._create_1x1conv_enc_dec(
                    W1=U_prca.T, W2=U_prca
                )
            else:
                self.encoder, self.decoder = self._create_linear_enc_dec(
                    W1=U_prca.T, W2=U_prca
                )


class DRSADecomposer(Decomposer):
    def __init__(
        self,
        data: torch.Tensor,
        data_context: torch.Tensor,
        n_components: Optional[int],
        component_dim: Optional[int] = None,
        epochs: int = 2500,
        load_path: Optional[str] = None,
    ):
        self.epochs = epochs
        super().__init__(data, n_components, component_dim, load_path=load_path)

        if load_path is None:
            data = self.reduce_spatial(data)
            data_context = self.reduce_spatial(data_context)
            d = data.shape[1]
            if component_dim is None:
                # We assume that each subspace has the same number of dimensions
                component_dim = d // n_components
            self.component_dim = component_dim
            self.n_components = n_components

            device = data.device

            # We do this normalization because activation and context vectors
            # from different layers are likely to have different magnitudes.
            # This normalization therefore stabilize the training.
            mat_act = data / (((data**2).mean(axis=0) ** (1 / 2)) * (d ** (1 / 4)))
            mat_ctx = data_context / (
                ((data_context**2).mean(axis=0) ** (1 / 2)) * (d ** (1 / 4))
            )

            # NaN -> 0
            mat_act = torch.nan_to_num(mat_act)
            mat_ctx = torch.nan_to_num(mat_ctx)

            (best_obj, U_drsa, best_obj_values,) = drsa.optimize(
                obj_func=drsa.obj_drsa,
                act=mat_act,
                ctx=mat_ctx,
                seed=42,
                ns=n_components,
                ss=component_dim,
                epochs=epochs,
                device=device,
            )

            # reorder learned subspaces based on their expected relevance
            U_drsa = U_drsa.reshape((d, n_components, component_dim)).detach()
            relevence_on_U = torch.einsum(
                "nd,dij->nij", mat_act, U_drsa
            ) * torch.einsum("nd,dij->nij", mat_act, U_drsa)
            relevence_on_U = relevence_on_U.sum(axis=2)
            mean_relevence_on_U = relevence_on_U.mean(axis=0)

            sorted_indices = torch.argsort(-mean_relevence_on_U)
            U_drsa = U_drsa[:, sorted_indices, :]
            U_drsa = U_drsa.reshape((d, n_components * component_dim))

            # Create encoder and decoder
            if self.is_conv:
                self.encoder, self.decoder = self._create_1x1conv_enc_dec(
                    W1=U_drsa.T, W2=U_drsa
                )
            else:
                self.encoder, self.decoder = self._create_linear_enc_dec(
                    W1=U_drsa.T, W2=U_drsa
                )

    def _get_file_name(self, postfix: Optional[str] = None):
        name = super()._get_file_name(postfix)
        param_str = f"e{self.epochs}"
        if postfix:
            name = name.replace(postfix, f"{param_str}_{postfix}")
        else:
            name = name.replace(".decomp", f"_{param_str}.decomp")
        return name


class SAEDecomposer(Decomposer):
    # Sparse Autoencoder Decomposer, implemnetation based on https://transformer-circuits.pub/2023/monosemantic-features/index.html
    def __init__(
        self,
        data: Optional[Union[torch.Tensor, torch.utils.data.Dataset]],
        upsampling_factor: int,
        epochs: int = 100,
        learning_rate: float = 1e-4,
        l1_weight: float = 1e-3,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        batch_size: int = 256,
        reset_patience: int = 10000,
        load_path: Optional[str] = None,
        num_workers: int = 8,
    ):
        raise NotImplementedError("This implementation is not yet complete")
        self.upsampling_factor = upsampling_factor
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l1_weight = l1_weight
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.batch_size = batch_size
        self.reset_patience = reset_patience

        if not isinstance(data, torch.Tensor):
            data_sample = torch.cat([data[i][0].unsqueeze(0) for i in range(len(data))], dim=0)
            if load_path is None:
                data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        else:
            data_sample = data
            dataset = torch.utils.data.TensorDataset(data_sample)
            if load_path is None:
                data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        super().__init__(data=data_sample, n_components=data_sample.shape[1]*upsampling_factor, component_dim=1, load_path=load_path)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(adam_beta1, adam_beta2))

        if load_path is None:
            with torch.autocast():
                dead_counter = torch.zeros(self.n_components, device=data_sample.device)
                per_neuron_loss = torch.zeros(self.n_components, device=data_sample.device)
                step_counter = 0
                for epoch in range(epochs):
                    for batch in data_loader:
                        batch = self.reduce_spatial(batch)
                        optimizer.zero_grad()

                        encoded = self.encoder(batch)

                        reconstruction_loss = ((self.decoder(encoded) - batch)**2).mean()
                        sparsity_loss = encoded.abs().mean() * self.l1_weight
                        loss = (reconstruction_loss + sparsity_loss) / batch_size
                        loss.backward()
                        self._make_decoder_weights_and_grad_unit_norm()
                        optimizer.step()

                        dead_counter += (encoded.abs().sum(dim=0) < 1e-12)

                        step_counter += 1

                        if step_counter % reset_patience == 0:
                            # Reset dead neurons to the
                            dead_neurons = dead_counter > 0
                            print(f"Resetting {dead_neurons.sum().item()} dead neurons")
                            self._re_init_encoder(dead_neurons, optimizer)

                            per_neuron_loss *= 0
                            dead_counter *= 0
                            step_counter = 0

                            # TODO: log losses and other metrics (according ot the blog post)
                            # TODO: save model checkpoint
                            # 

    def _make_decoder_weights_and_grad_unit_norm(self):
        # Adapted from https://github.com/neelnanda-io/1L-Sparse-Autoencoder/blob/main/utils.py#L135
        W_dec_normed = self.decoder.weight.data / self.weight.data.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.decoder.weight.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.decoder.weight.grad -= W_dec_grad_proj
        # Bugfix(?) for ensuring W_dec retains unit norm, this was not there when I trained my original autoencoders.
        self.decoder.weight.data = W_dec_normed

    @torch.no_grad()
    def _re_init_encoder(self, indices, optimizer):
        # Adapted from https://github.com/neelnanda-io/1L-Sparse-Autoencoder/blob/main/utils.py#L320
        # TODO: replace with loss-based resampling from https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder
        new_W_enc = (torch.nn.init.kaiming_uniform_(torch.zeros_like(self.encoder.weight)))
        new_W_dec = (torch.nn.init.kaiming_uniform_(torch.zeros_like(self.decoder.weight)))
        new_b_enc = (torch.zeros_like(self.encoder.bias))
        self.encoder.weight.data[:, indices] = new_W_enc[:, indices]
        self.decoder.weight.data[indices, :] = new_W_dec[indices, :]
        self.encoder.bias.data[indices] = new_b_enc[indices]

        # Reset the Adam optimizer parameters for every modified weight and bias term.
        param_id_encW = id(self.encoder.weight)
        param_id_encb = id(self.encoder.weight)
        param_id_decW = id(self.decoder.weight)
        optimizer.state[param_id_encW]['exp_avg'][:, indices].zero_()  # Reset first moment estimate (m_t)
        optimizer.state[param_id_encW]['exp_avg_sq'][:, indices].zero_()  # Reset second moment estimate (v_t)
        optimizer.state[param_id_encb]['exp_avg'][indices].zero_()
        optimizer.state[param_id_encb]['exp_avg_sq'][indices].zero_()
        optimizer.state[param_id_decW]['exp_avg'][indices].zero_()
        optimizer.state[param_id_decW]['exp_avg_sq'][indices].zero_()

    def _get_file_name(self, postfix: Optional[str] = None):
        name = super()._get_file_name(postfix)
        param_str = f"u{self.upsampling_factor}_e{self.epochs}_lr{self.learning_rate}_l1{self.l1_weight}_b1{self.adam_beta1}_b2{self.adam_beta2}_bs{self.batch_size}_rp{self.reset_patience}"
        if postfix:
            name = name.replace(postfix, f"{param_str}_{postfix}")
        else:
            name = name.replace(".decomp", f"_{param_str}.decomp")
        return name
