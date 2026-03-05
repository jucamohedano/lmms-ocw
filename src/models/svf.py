from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SVFLinear(nn.Module):
    """
    Replaces an nn.Linear layer with its SVD-decomposed version for SVF.
    """

    actual_type = "svf_linear"

    def __init__(
        self,
        linear_layer: nn.Linear,
        trainable_bias: bool = True,
        implementation: str = "linear",  # 'manual' or 'linear'
        rank: int = -1,
        log_training: bool = True,  # Default: False
        clamp_singular_values: bool = True,  # Default: False
    ):
        super().__init__()
        self.log_training = log_training
        self.clamp_singular_values = clamp_singular_values

        weights = linear_layer.weight.data.float()  # Use float for SVD

        # Perform SVD
        # W = U @ diag(S) @ Vh
        U, S, Vh = torch.linalg.svd(weights, full_matrices=False)

        # Truncate to the desired rank, if specified
        if rank > 0:
            U = U[:, :rank]
            S = S[:rank]
            Vh = Vh[:rank, :]

        # ========================== #
        # Random initialization of S #
        # ========================== #
        # S = torch.randn_like(S)

        # nn.init.uniform_(S, a=0.0, b=1.0)

        # nn.init.normal_(S, mean=0.0, std=0.01)

        # S = S.unsqueeze(0)
        # nn.init.xavier_normal_(S)
        # S = S.squeeze(0)

        # S += torch.randn_like(S) * 0.01

        # S += torch.rand_like(S) * 0.2

        # S += torch.rand_like(S) * 0.01

        self.original_s = torch.clone(S)  # Store original S for reference

        if self.log_training:
            S = torch.log(S + 1e-8)

        self.implementation = implementation
        if implementation == "manual":
            self.frozen_svf_U = nn.Parameter(U, requires_grad=False)  # (out_features, rank)
            self.trainable_svf_S = nn.Parameter(S, requires_grad=True)  # (rank,) - Trainable
            self.frozen_svf_Vh = nn.Parameter(Vh, requires_grad=False)  # (rank, in_features)

        elif implementation == "linear":
            _rank = U.shape[1]
            self.frozen_svf_U = nn.Linear(
                U.shape[0], _rank, bias=not trainable_bias and linear_layer.bias is not None
            )
            self.frozen_svf_U.weight.data = U
            # We want to keep the bias frozen
            if not trainable_bias and linear_layer.bias is not None:
                self.frozen_svf_U.bias.data = linear_layer.bias.data

            self.trainable_svf_S = nn.Parameter(S.unsqueeze(0), requires_grad=True)

            self.frozen_svf_Vh = nn.Linear(_rank, Vh.shape[0], bias=False)
            self.frozen_svf_Vh.weight.data = Vh

        # Handle bias
        # Bias must be handled separately as adding it to the U layer would make it
        # untrainable, as U and Vh are frozen. Instead, if we want a trainable bias,
        # we have to keep it separate
        self.bias = None
        if trainable_bias:
            if linear_layer.bias is not None:
                self.trainable_svf_bias = nn.Parameter(linear_layer.bias.data, requires_grad=True)
                self.bias = self.trainable_svf_bias

        # Aliases for local usage
        self.U = self.frozen_svf_U
        self.S = self.trainable_svf_S
        self.Vh = self.frozen_svf_Vh

        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = U.shape[1]  # Actual rank after truncation
        self.original_dtype = linear_layer.weight.dtype  # Store original dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: Y = X @ Vh.T @ diag(S) @ U.T + b."""
        # Cast input to float if necessary for matmul with SVD components
        input_dtype = x.dtype
        x = x.to(self.S.dtype)

        s = self.S
        if self.log_training:
            s = torch.exp(s)

        if self.clamp_singular_values:
            s = F.relu(s)

        if self.implementation == "manual":
            # Y = X @ Vh.T @ diag(S) @ U.T + b
            x_transformed = (
                x @ self.Vh.T
            )  # (batch, ..., in_features)  @ (in_features, rank) -> (batch, ..., rank)
            x_scaled = (
                x_transformed * s
            )  # Element.wise multiplication, broadcasting S -> (batch, ..., rank)
            output = (
                x_scaled @ self.U.T
            )  # (batch, ..., rank) @ (rank, out_features) -> (batch, ..., out_features)

        elif self.implementation == "linear":
            x = self.Vh(x)
            x = x.mul(s)
            output = self.U(x)

            # Equivalent to:
            # output = x @ self.finalize().t()

        if self.bias is not None:
            output += self.bias

        # Cast output back to original dtype
        output = output.to(input_dtype)
        return output

    def finalize(self):
        """Return the full weight matrix W = U @ diag(S) @ Vh."""
        s = self.S
        if self.log_training:
            s = torch.exp(s)

        if self.clamp_singular_values:
            s = F.relu(s)

        if self.implementation == "manual":
            weights = self.U @ (torch.diag(s) @ self.Vh.T)
        else:
            S_diag = torch.diag(s.squeeze(0))
            weights = self.U.weight @ S_diag @ self.Vh.weight
            # weights = (self.U.weight * self.S) @ self.Vh.weight

        return weights

    def _clamp_singular_values(self):
        """Clamp singular values to be non-negative (in-place)."""
        if not self.clamp_singular_values:
            return

        if self.log_training:
            self.S.data = torch.exp(self.S.data)
        self.S.data = F.relu(self.S.data)
        if self.log_training:
            self.S.data = torch.log(self.S.data + 1e-8)

    def __repr__(self):
        return f"SVFLinear(in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, bias={self.bias is not None})"


class MultiSVFLinear(nn.Module):
    """Stack of SVFLinear layers with sum or average aggregation."""

    actual_type = "multi_svf_linear"

    def __init__(
        self, linear_layer: nn.Linear, layers_count: int, aggregation: str = "sum", rank: int = -1
    ):
        super().__init__()

        assert layers_count > 0, "layers_count must be greater than 0"

        layers = [SVFLinear(linear_layer, rank=rank) for _ in range(layers_count)]
        self.layers = nn.ModuleList(layers)
        self.aggregation = aggregation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all SVFLinear layers and sum their outputs.
        """
        outputs = [layer(x) for layer in self.layers]
        # Stack outputs along a new dimension (dim=0) and sum along that dimension
        if self.aggregation == "sum":
            total_output = torch.stack(outputs, dim=0).sum(dim=0)
        elif self.aggregation == "avg":
            total_output = torch.stack(outputs, dim=0).mean(dim=0)
        return total_output

    def __repr__(self):
        # Extract relevant info from the first layer's repr
        in_features = self.layers[0].in_features
        out_features = self.layers[0].out_features
        rank = self.layers[0].rank
        bias = self.layers[0].bias is not None
        layers_count = len(self.layers)
        return f"MultiSVFLinear(layers_count={layers_count}, in_features={in_features}, out_features={out_features}, rank={rank}, bias={bias})"


class SVFLinearWithMultipleS(nn.Module):
    """
    Replaces an nn.Linear layer with its SVD-decomposed version for SVF.
    Singular values (S) are copied, so that there are N different trainable matrices.
    """

    actual_type = "base"

    def __init__(
        self,
        linear_layer: nn.Linear,
        trainable_bias: bool = True,
        implementation: str = "linear",  # 'manual' or 'linear'
        rank: int = -1,
        initial_s_count: int = 1,
        optimizer=None,
        separate_biases: bool = False,  # Default: False
        log_training: bool = True,  # Default: False
        clamp_singular_values: bool = True,  # Default: False
        **kwargs,
    ):
        super().__init__()

        if isinstance(linear_layer, nn.Linear):
            weights = linear_layer.weight.data.float()  # Use float for SVD
        else:
            # Support for tensor weights directly
            weights = linear_layer.float()  # Use float for SVD
        self.original_weights = weights
        self.optimizer = optimizer
        self.separate_biases = separate_biases
        self.log_training = log_training
        self.clamp_singular_values = clamp_singular_values

        # Perform SVD
        # W = U @ diag(S) @ Vh
        U, S, Vh = torch.linalg.svd(weights, full_matrices=False)
        self.original_U = U.clone()
        self.original_S = S.clone()
        self.original_Vh = Vh.clone()

        # Truncate to the desired rank, if specified
        if rank > 0:
            U = U[:, :rank]
            S = S[:rank]
            Vh = Vh[:rank, :]

        if self.log_training:
            S = torch.log(S + 1e-8)

        self.implementation = implementation
        if implementation == "manual":
            self.frozen_svf_U = nn.Parameter(U, requires_grad=False)  # (out_features, rank)
            self.frozen_svf_S = nn.Parameter(
                S, requires_grad=False
            )  # (rank,) - Untrainable, original copy
            self.frozen_svf_Vh = nn.Parameter(Vh, requires_grad=False)  # (rank, in_features)

        elif implementation == "linear":
            _rank = U.shape[1]
            self.frozen_svf_U = nn.Linear(
                U.shape[0], _rank, bias=not trainable_bias and linear_layer.bias is not None
            )
            self.frozen_svf_U.weight.data = U
            # We want to keep the bias frozen
            if (
                not trainable_bias
                and hasattr(linear_layer, "bias")
                and linear_layer.bias is not None
            ):
                self.frozen_svf_U.bias.data = linear_layer.bias.data

            self.frozen_svf_S = nn.Parameter(S.unsqueeze(0), requires_grad=False)

            self.frozen_svf_Vh = nn.Linear(_rank, Vh.shape[0], bias=False)
            self.frozen_svf_Vh.weight.data = Vh

        # Handle bias
        # Bias must be handled separately as adding it to the U layer would make it
        # untrainable, as U and Vh are frozen. Instead, if we want a trainable bias,
        # we have to keep it separate
        self.bias = None
        if trainable_bias:
            if hasattr(linear_layer, "bias") and linear_layer.bias is not None:
                if self.separate_biases:
                    self.frozen_svf_bias = nn.Parameter(
                        linear_layer.bias.data, requires_grad=False
                    )
                    self.trainable_svf_bias = nn.ParameterList()
                else:
                    self.trainable_svf_bias = nn.Parameter(
                        linear_layer.bias.data, requires_grad=True
                    )
                self.bias = self.trainable_svf_bias

        self.trainable_svf_S = nn.ParameterList()

        # Aliases for local usage
        self.U = self.frozen_svf_U
        self.S = self.trainable_svf_S
        self.Vh = self.frozen_svf_Vh

        if isinstance(linear_layer, nn.Linear):
            self.in_features = linear_layer.in_features
            self.out_features = linear_layer.out_features
            self.original_dtype = linear_layer.weight.dtype  # Store original dtype
        else:
            self.in_features = linear_layer.shape[1]
            self.out_features = linear_layer.shape[0]
            self.original_dtype = linear_layer.dtype  # Store original dtype

        self.rank = U.shape[1]  # Actual rank after truncation
        self.mode = "train"  # Default mode is train

        self.add_s(initial_s_count, add_to_optimizer=False)

    def train_mode(self):
        """Switch to training mode (use active S for forward)."""
        self.mode = "train"

    def eval_mode(self):
        """Switch to eval mode (use active S for forward)."""
        self.mode = "eval"

    def _add_to_optimizer(self, new_parameters):
        """Add new parameters to the optimizer's param groups."""
        if self.optimizer is not None:
            self.optimizer.add_param_group(
                {
                    "params": new_parameters,
                    **{k: v for k, v in self.optimizer.param_groups[0].items() if k != "params"},
                }
            )

    def add_s(self, count: int = 1, add_to_optimizer: bool = True):
        """Add count new S vectors (cloned from frozen S) and optionally register with optimizer."""
        for _ in range(count):
            new_parameters = self.frozen_svf_S.clone().detach()
            new_parameters.requires_grad_(True)
            self.S.append(new_parameters)

            if add_to_optimizer:
                self._add_to_optimizer(new_parameters)

            if self.separate_biases:
                new_parameters = self.frozen_svf_bias.clone().detach()
                new_parameters.requires_grad_(True)
                self.bias.append(new_parameters)

                if add_to_optimizer:
                    self._add_to_optimizer(new_parameters)

    def freeze_all_s(self):
        """
        Freeze all S parameters.
        """
        for s in self.S:
            s.requires_grad_(False)

    def unfreeze_all_s(self):
        """
        Unfreeze all S parameters.
        """
        for s in self.S:
            s.requires_grad_(True)

    def freeze_s(self, index: int):
        """
        Freeze the S parameter at the given index.
        """
        if index < len(self.S):
            self.S[index].requires_grad_(False)
        else:
            raise IndexError(f"Index {index} out of range for S parameters.")

    def unfreeze_s(self, index: int):
        """
        Unfreeze the S parameter at the given index.
        """
        if index < len(self.S):
            self.S[index].requires_grad_(True)
        else:
            raise IndexError(f"Index {index} out of range for S parameters.")

    def _use_s(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Use the S parameter for the current input.
        This is a placeholder for the actual implementation.
        """
        # This is a placeholder.
        raise NotImplementedError("This method is implemented by children classes.")

    def _use_bias(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply bias (placeholder; implemented by children)."""
        raise NotImplementedError("This method is implemented by children classes.")

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass: Y = X @ Vh.T @ diag(S) @ U.T + bias."""
        input_dtype = x.dtype
        x = x.to(self.S[0].dtype)

        if self.implementation == "manual":
            # Y = X @ Vh.T @ diag(S) @ U.T + b
            x_transformed = (
                x @ self.Vh.T
            )  # (batch, ..., in_features)  @ (in_features, rank) -> (batch, ..., rank)
            x_scaled = self._use_s(x_transformed, **kwargs)
            output = (
                x_scaled @ self.U.T
            )  # (batch, ..., rank) @ (rank, out_features) -> (batch, ..., out_features)

        elif self.implementation == "linear":
            x = self.Vh(x)
            x = self._use_s(x, **kwargs)
            output = self.U(x)

        if self.bias is not None:
            if self.separate_biases:
                output = self._use_bias(output, **kwargs)
            else:
                output += self.bias

        # Cast output back to original dtype
        output = output.to(input_dtype)
        return output

    def _clamp_singular_values(self):
        """Clamp all S vectors to non-negative values (in-place)."""
        if not self.clamp_singular_values:
            return

        for i in range(len(self.S)):
            if self.log_training:
                self.S[i].data = torch.exp(self.S[i].data)
            self.S[i].data = F.relu(self.S[i].data)
            if self.log_training:
                self.S[i].data = torch.log(self.S[i].data + 1e-8)

    def __repr__(self):
        # Extract relevant info from the first layer's repr
        in_features = self.in_features
        out_features = self.out_features
        rank = self.rank
        bias = self.bias is not None
        layers_count = len(self.S)
        return f"SVFLinearWithMultipleS(layers_count={layers_count}, in_features={in_features}, out_features={out_features}, rank={rank}, bias={bias})"


class SVFLinearWithMultipleSAverageWeights(SVFLinearWithMultipleS):
    """
    SVFLinearWithMultipleS with average S.
    """

    actual_type = "avg_weights"

    def _use_s(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Use the average S parameter for the current input.
        """
        # Average all S parameters
        s = torch.stack([p for p in self.S]).mean(dim=0)
        return x * s  # Element-wise multiplication


class SVFLinearWithMultipleSWithTrainingRouter(SVFLinearWithMultipleS):
    """
    SVFLinearWithMultipleS with training router.
    """

    actual_type = "training_router"

    def _use_s(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Use the S parameter for the current input.
        """
        if self.mode == "train":
            # Use the S parameter for the current input
            # s = self.S[kwargs['index']]
            # x: (batch, ..., rank)
            # self._training_router: (batch,) or (batch, ...)
            indices = kwargs.get("index", getattr(self, "_training_router", None))
            if indices is None:
                raise ValueError("No routing indices provided for training router mode.")
            # Ensure indices is a tensor on the correct device
            if not torch.is_tensor(indices):
                indices = torch.tensor(indices, device=x.device, dtype=torch.long)
            # Gather the correct S for each sample
            # Stack S into (num_S, rank)
            S_stack = torch.stack(list(self.S))  # (num_S, rank)
            s = S_stack[indices]  # (batch, ..., rank)

            # Ensure s is broadcastable to x for element-wise multiplication
            s = s.squeeze(1)
            if x.shape[1] != s.shape[0]:
                s = s.unsqueeze(1)

        else:
            # Average all S parameters
            s = torch.stack(list(self.S)).mean(dim=0)

        return x * s  # Element-wise multiplication

    def set_training_router_for_batch(self, indices: list[int]):
        """
        Set the training router for the current batch.
        """

        if isinstance(indices, torch.Tensor):
            self._training_router = indices
        else:
            self._training_router = torch.tensor(
                indices, device=self.S[0].device, dtype=torch.long
            )

        if max(indices) >= len(self.S):
            self.add_s(max(indices) - len(self.S) + 1)


class SVFLinearWithMultipleSWithExternalRouter(SVFLinearWithMultipleS):
    """
    SVFLinearWithMultipleS with external router.
    """

    actual_type = "external_router"
    # actual_type = 'independent'  # To experiment using the oracle router

    def _use_s(self, x: torch.Tensor | dict[int : torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Use the S parameter for the current input.
        """

        # # Very first layer, run through all experts and construct the dict
        # # that will be passed to subsequent layers
        # if isinstance(x, torch.Tensor):
        #     print('First layer, received plain tensor')

        #     res = {}
        #     for s in self.S:

        s = self.S[self._active_expert]

        if self.log_training:
            s = torch.exp(s)

        if self.clamp_singular_values:
            s = F.relu(s)

        return x * s

        # Use the S parameter for the current input
        # s = self.S[kwargs['index']]
        # x: (batch, ..., rank)
        # self._training_router: (batch,) or (batch, ...)
        indices = kwargs.get("index", getattr(self, "_router", None))
        if indices is None:
            raise ValueError("No routing indices provided.")
        # Ensure indices is a tensor on the correct device
        if not torch.is_tensor(indices):
            indices = torch.tensor(indices, device=x.device, dtype=torch.long)
        # Gather the correct S for each sample
        # Stack S into (num_S, rank)
        S_stack = torch.stack(list(self.S))  # (num_S, rank)
        s = S_stack[indices]  # (batch, ..., rank)

        # Ensure s is broadcastable to x for element-wise multiplication
        s = s.squeeze(1)
        if x.shape[1] != s.shape[0]:
            s = s.unsqueeze(1)

        return x * s  # Element-wise multiplication

    def _use_bias(self, x, **kwargs):
        """Apply the active expert's bias to x."""
        b = self.bias[self._active_expert]
        return x + b

    def set_router_for_batch(self, index):  # indices: list[int]):
        """
        Set the router for the current batch.
        """

        self._active_expert = index

        # if isinstance(indices, torch.Tensor):
        #     self._router = indices
        # else:
        #     self._router = torch.tensor(indices, device=self.S[0].device, dtype=torch.long)

        # if max(indices) >= len(self.S):
        #     self.add_s(max(indices) - len(self.S) + 1)

    def set_active_expert(self, index: int):
        """
        Set the active expert index.
        """
        self.set_router_for_batch(index)


class SVFLinearWithMultipleSIndependent(SVFLinearWithMultipleS):
    """
    SVFLinearWithMultipleS with S independent experts, for parallel training.
    """

    actual_type = "independent"

    def _use_s(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Use the S parameter for the current input.
        """

        if self.actual_type == "independent":
            s = self.S[self._active_expert]
        else:  # self.actual_type in ['avg_weights', 'avg_deltas', 'only_avg_delta', 'interpolate']:
            s = self.S
        # else:
        #     raise ValueError(f"Unknown actual_type: {self.actual_type}")

        if self.log_training:
            s = torch.exp(s)

        if self.clamp_singular_values:
            s = F.relu(s)

        return x * s  # Element-wise multiplication

    def _use_bias(self, x, **kwargs):
        """Apply bias from active expert or shared bias depending on actual_type."""
        if self.actual_type == "independent":
            b = self.bias[self._active_expert]
        else:
            b = self.bias
        return x + b

    def set_active_expert(self, index: int):
        """
        Set the training router for the current batch.
        """

        if index >= len(self.S):
            self.add_s(index - len(self.S) + 1)

        self._active_expert = index

    def clear_active_expert(self):
        """
        Clear the active expert index.
        """
        if hasattr(self, "_active_expert"):
            del self._active_expert

    def get_active_expert(self):
        """
        Get the active expert index.
        """
        if not hasattr(self, "_active_expert"):
            return None

        return self._active_expert

    def average_experts(self):
        """
        Average all S parameters.
        """

        new_S = torch.cat(list(self.S))

        if self.log_training:
            new_S = torch.exp(new_S)

        if self.clamp_singular_values:
            new_S = F.relu(new_S)

        new_S = new_S.mean(dim=0)

        if self.log_training:
            new_S = torch.log(new_S + 1e-8)

        self.S = nn.Parameter(new_S)
        self.actual_type = "avg_weights"

        if self.separate_biases:
            self.bias = nn.Parameter(torch.stack(list(self.bias)).mean(dim=0))

        return self.S

    def finalize_expert(self, s):
        """Compute full weight matrix W = U @ diag(s) @ Vh for a single expert S vector."""
        if self.implementation == "manual":
            weights = self.U @ (torch.diag(s) @ self.Vh.T)
        else:
            S_diag = torch.diag(s.squeeze(0))
            weights = self.U.weight @ S_diag @ self.Vh.weight

        return weights

    def average_experts_deltas(self):
        """
        Average the deltas of the different experts w.r.t. the base weights
        """

        new_S = torch.cat(list(self.S))
        base = self.frozen_svf_S

        if self.log_training:
            new_S = torch.exp(new_S)
            base = torch.exp(base)

        if self.clamp_singular_values:
            new_S = F.relu(new_S)

        # Compute the difference w.r.t. original singular values
        deltas = [s - base for s in new_S]

        # Average the deltas
        avg_delta = torch.cat(deltas).mean(dim=0)

        # Build the new expert
        new_S = base + avg_delta

        if self.log_training:
            new_S = torch.log(new_S + 1e-8)

        self.S = nn.Parameter(new_S)
        self.actual_type = "avg_deltas"

        return self.S

    def average_experts_keeping_only_deltas(self):
        """
        Average the deltas of the different experts w.r.t. the base weights
        Keep only the deltas as the S vector, rather than summing the average
        delta to the original weights
        """

        # Compute the difference w.r.t. original singular values
        deltas = [s - self.frozen_svf_S for s in self.S]

        # Average the deltas
        avg_delta = torch.cat(deltas).mean(dim=0)

        # Build the new expert
        self.S = nn.Parameter(avg_delta)
        self.actual_type = "only_avg_delta"

        return self.S

    def interpolate_experts(self, weights: list[int] | torch.Tensor):
        """
        Interpolate experts' weights based on the given interpolation weights
        """

        assert weights is not None, "`weights` must not be `None`"

        if isinstance(weights, list):
            weights = torch.tensor(weights)

        new_S = torch.cat([s * w for s, w in zip(self.S, weights)]).sum(dim=0)
        self.S = nn.Parameter(new_S)
        self.actual_type = "interpolate"

        return self.S

    def interpolate_experts_inverse(self, weights: list[int] | torch.Tensor):
        """
        Interpolate experts' weights based on the given interpolation weights
        """

        assert weights is not None, "`weights` must not be `None`"

        if isinstance(weights, list):
            weights = torch.tensor(weights)

        new_S = torch.cat([s * w for s, w in zip(self.S, weights)]).sum(dim=0)
        self.S = nn.Parameter(new_S)
        self.actual_type = "interpolate"

        return self.S

    def svd_experts(self):
        """
        Stack expert weights and perform a new SVD
        """

        new_S = torch.cat(list(self.S), dim=0)

        if self.log_training:
            new_S = torch.exp(new_S)

        if self.clamp_singular_values:
            new_S = F.relu(new_S)

        new_U, new_S, new_Vh = torch.linalg.svd(new_S, full_matrices=False)

        # Use the top right singular vector
        consensus_S = new_Vh[0]

        # Combine the top-k
        k = min(3, new_Vh.shape[0])
        consensus_S = (new_S[:k].unsqueeze(1) * new_Vh[:k]).sum(dim=0) / new_S[:k].sum()

        if self.log_training:
            consensus_S = torch.log(consensus_S + 1e-8)

        self.S = nn.Parameter(consensus_S)

        self.actual_type = "stack_svd"

    def elementwise_experts(self, mode: str = "min_dist"):
        """Combine experts element-wise (min_dist, mean, etc.)."""
        new_S = torch.cat(list(self.S), dim=0)

        if self.log_training:
            new_S = torch.exp(new_S)

        if self.clamp_singular_values:
            new_S = F.relu(new_S)

        if self.separate_biases:
            new_bias = torch.stack(list(self.bias), dim=0)

        if mode == "min":
            new_S = new_S.min(dim=0).values
            if self.separate_biases:
                new_bias = new_bias.min(dim=0).values

        elif mode == "max":
            new_S = new_S.max(dim=0).values
            if self.separate_biases:
                new_bias = new_bias.max(dim=0).values

        elif mode == "median":
            new_S = new_S.median(dim=0).values
            if self.separate_biases:
                new_bias = new_bias.median(dim=0).values

        elif mode == "max_dist":
            mean = new_S.mean(dim=0)
            deviations = torch.abs(new_S - mean)
            idxs = torch.argmax(deviations, dim=0)
            new_S = new_S[idxs, torch.arange(new_S.shape[-1])]

            if self.separate_biases:
                mean = new_bias.mean(dim=0)
                deviations = torch.abs(new_bias - mean)
                idxs = torch.argmax(deviations, dim=0)
                new_bias = new_bias[idxs, torch.arange(new_bias.shape[-1])]

        elif mode == "min_dist":
            mean = new_S.mean(dim=0)
            deviations = torch.abs(new_S - mean)
            idxs = torch.argmin(deviations, dim=0)
            new_S = new_S[idxs, torch.arange(new_S.shape[-1])]

            if self.separate_biases:
                mean = new_bias.mean(dim=0)
                deviations = torch.abs(new_bias - mean)
                idxs = torch.argmin(deviations, dim=0)
                new_bias = new_bias[idxs, torch.arange(new_bias.shape[-1])]

        elif mode == "cluster":
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=2, n_init=100, random_state=0)
            kmeans.fit(new_S.detach().cpu().numpy())
            centroids = torch.tensor(
                kmeans.cluster_centers_, dtype=self.S[0].dtype, device=self.S[0].device
            )  # (n_clusters, rank)
            new_S = centroids.mean(dim=0)

            if self.separate_biases:
                kmeans = KMeans(n_clusters=4, n_init=100, random_state=0)
                kmeans.fit(new_bias.detach().cpu().numpy())
                centroids = torch.tensor(
                    kmeans.cluster_centers_, dtype=self.S[0].dtype, device=self.S[0].device
                )  # (n_clusters, rank)
                new_bias = centroids.mean(dim=0)

        if self.log_training:
            new_S = torch.log(new_S + 1e-8)

        self.S = nn.Parameter(new_S)

        if self.separate_biases:
            self.bias = nn.Parameter(new_bias)

        self.actual_type = "elementwise"

    def my_tsv_experts(self):
        """
        First implementation of TSV, *not* following original repo
        """

        new_S = torch.cat(list(self.S), dim=0)

        if self.log_training:
            new_S = torch.exp(new_S)

        if self.clamp_singular_values:
            new_S = F.relu(new_S)

        # vanilla = self.frozen_svf_S.data
        vanilla = new_S.mean(dim=0)
        D = new_S - vanilla

        # Perform SVD
        U, S, Vh = torch.linalg.svd(D, full_matrices=False)

        # Project each expert onto TSVs
        projections = torch.matmul(D, Vh.T)

        # Merge
        merged_proj = projections.mean(dim=0)

        # Reconstruct merged differences
        merged_diff = torch.matmul(merged_proj, Vh).reshape(vanilla.shape)

        # Final merged weights
        new_S = vanilla + merged_diff

        if self.log_training:
            new_S = torch.log(new_S + 1e-8)

        self.S = nn.Parameter(new_S)

        self.actual_type = "my_tsv"

    def tsv_1_experts(self):
        """
        Adapted from: compute_and_sum_svd_mem_reduction
        """

        _u = self.original_U
        _v = self.original_Vh
        _s = self.original_S

        sv_reduction = 1 / len(self.S)

        device = self.S[0].device
        _u = _u.to(device)
        _v = _v.to(device)
        sum_u = torch.zeros_like(_u, device=device)
        sum_s = torch.zeros_like(_s, device=device)
        sum_v = torch.zeros_like(_v, device=device)

        for i in range(len(self.S)):
            s = self.S[i].squeeze()

            if self.log_training:
                s = torch.exp(s)

            if self.clamp_singular_values:
                s = F.relu(s)

            S_diag = torch.diag(s)
            # vec = _u @ S_diag @ _v
            # u, s, v = torch.linalg.svd(vec, full_matrices=False)
            u = _u
            v = _v

            reduced_index_s = int(_s.shape[0] * sv_reduction)

            # select only the first reduced_index_s columns of u and place them
            sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[:, :reduced_index_s]
            sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[:reduced_index_s]
            # select only the first reduced_index_s rows of v and place them
            sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[:reduced_index_s, :]

        u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
        u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)

        new_weight = torch.linalg.multi_dot(
            (
                u_u,
                v_u,
                torch.diag(sum_s),
                u_v,
                v_v,
            )
        )

        _, new_S, _ = torch.linalg.svd(new_weight, full_matrices=False)
        if self.log_training:
            new_S = torch.log(new_S + 1e-8)
        self.S = nn.Parameter(new_S.unsqueeze(0))

        self.actual_type = "tsv_with_reduction"


class SVFLinearWithMultipleSWithLearnableWeights(SVFLinearWithMultipleS):
    """
    SVFLinearWithMultipleS with learnable router.
    """

    actual_type = "learnable_weights"

    def add_s(self, count: int = 1):
        """Add S vectors and extend learnable weighting parameters."""
        res = super().add_s(count)

        # self.trainable_svf_s_weighting = nn.Parameter((
        #         torch.ones(len(self.S)) / len(self.S)
        #     ).unsqueeze(-1),
        #     requires_grad=True
        # )

        if not hasattr(self, "trainable_svf_s_weighting"):
            self.trainable_svf_s_weighting = nn.Parameter(
                torch.randn((len(self.S), 1)), requires_grad=True
            )

        else:
            self.trainable_svf_s_weighting = nn.Parameter(
                torch.cat([self.trainable_svf_s_weighting, torch.randn((count, 1))]),
                requires_grad=True,
            )

            # Add to optimizer?

        return res

    def _use_s(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Use the S parameter for the current input.
        """

        s = torch.stack(list(self.S)).squeeze(1)  # (num_S, rank)
        w = self.trainable_svf_s_weighting.softmax(dim=0)
        s_weighted = s * w  # (num_S, rank)
        s_weighted = s_weighted.sum(dim=0).unsqueeze(0)  # (1, rank)

        return x * s_weighted  # Element-wise multiplication


class SVFLinearWithMultipleSDenseMoE(SVFLinearWithMultipleS):
    """
    SVFLinearWithMultipleS with dense Mixture of Expert.
    """

    actual_type = "dense_moe"

    def add_s(self, count: int = 1):
        """Add S vectors and rebuild dense MoE router to match expert count."""
        res = super().add_s(count)

        self.trainable_svf_router = nn.Linear(self.frozen_svf_S.shape[-1], len(self.S))
        self.trainable_svf_router.requires_grad_(True)

        self.gate = self.trainable_svf_router

        return res

    def _use_s(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Use the S parameter for the current input.
        """

        # B, T, D = x.shape
        # N = len(self.S)

        # Compute gating scores
        gate_logits = self.gate(x)
        gate_weights = gate_logits.softmax(dim=-1)

        x_expanded = x.unsqueeze(2)  # (B, T, 1, D)
        S_expanded = (
            torch.stack([s for s in self.S]).squeeze(-2).unsqueeze(0).unsqueeze(0)
        )  # (1, 1, N, D)

        x_scaled = x_expanded * S_expanded  # (B, T, N, D)
        gate_weights = gate_weights.unsqueeze(-1)  # (B, T, N, 1)
        output = torch.sum(x_scaled * gate_weights, dim=2)  # (B, T, D)

        return output


class SVFLinearWithMultipleSSparseMoE(SVFLinearWithMultipleS):
    """
    SVFLinearWithMultipleS with sparse Mixture of Expert.
    """

    actual_type = "sparse_moe"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.topk = kwargs.get("topk", 2)

    def add_s(self, count: int = 1):
        """Add S vectors and rebuild sparse MoE router to match expert count."""
        res = super().add_s(count)

        self.trainable_svf_router = nn.Linear(self.frozen_svf_S.shape[-1], len(self.S))
        self.trainable_svf_router.requires_grad_(True)

        self.gate = self.trainable_svf_router
        self.S_stacked = None

        return res

    def _use_s(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Use the S parameter for the current input.
        """

        # B, T, D = x.shape
        # N = len(self.S)

        # Compute gating scores
        gate_logits = self.gate(x)

        # Top-k selection
        topk_vals, topk_idx = gate_logits.topk(k=self.topk, dim=-1)  # (B, T, topk)
        topk_gates = topk_vals.softmax(dim=-1)  # (B, T, topk)

        # Gather top-k S vectors
        # Prepare to index
        if self.S_stacked is None:
            self.S_stacked = torch.stack([s for s in self.S]).squeeze(1)  # (N, rank)
        S_selected = self.S_stacked[topk_idx]

        # Scale input
        x_expanded = x.unsqueeze(2)  # (B, T, 1, D)
        x_scaled = x_expanded * S_selected  # (B, T, topk, D)

        # Weighted sum over top-k experts
        topk_gates = topk_gates.unsqueeze(-1)  # (B, T, topk, 1)
        output = torch.sum(x_scaled * topk_gates, dim=2)  # (B, T, D)

        return output


class SVFMultiHeadAttention(nn.Module):
    """
    MultiHeadAttention with SVF applied to Q, K, V projections individually.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        svf_rank: int = -1,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Define separate projections for Q, K, V
        # Using `nn.Linear`, but it will be replaced with `SVFLinear` when `set_parameters` is called
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = dropout
        self.batch_first = batch_first
        self.svf_rank = svf_rank
        self._qkv_initialized = False

    def set_parameters(
        self,
        original_attn: nn.MultiheadAttention,
        apply_to_q: bool = True,
        apply_to_k: bool = True,
        apply_to_v: bool = True,
        apply_to_attn_proj: bool = True,
        type: str = "single",
    ):
        """
        Copy parameters from original MultiheadAttention.
        This splits the in_proj_weight into Q, K, V parts.
        """
        # Check if original attention has separate or combined QKV projections
        if hasattr(original_attn, "in_proj_weight") and original_attn.in_proj_weight is not None:
            # Split the in_proj_weight into Q, K, V parts
            q_weight, k_weight, v_weight = original_attn.in_proj_weight.chunk(3)

            self.q_proj = self.q_proj.to(q_weight.dtype)
            self.k_proj = self.k_proj.to(k_weight.dtype)
            self.v_proj = self.v_proj.to(v_weight.dtype)

            # Copy the split weights to our q_proj, k_proj, v_proj
            self.q_proj.weight.data.copy_(q_weight)
            self.k_proj.weight.data.copy_(k_weight)
            self.v_proj.weight.data.copy_(v_weight)

            # Handle biases if they exist
            if hasattr(original_attn, "in_proj_bias") and original_attn.in_proj_bias is not None:
                q_bias, k_bias, v_bias = original_attn.in_proj_bias.chunk(3)
                self.q_proj.bias.data.copy_(q_bias)
                self.k_proj.bias.data.copy_(k_bias)
                self.v_proj.bias.data.copy_(v_bias)
        else:
            # For models with separate q, k, v projections
            if hasattr(original_attn, "q_proj"):
                self.q_proj.weight.data.copy_(original_attn.q_proj.weight)
                self.k_proj.weight.data.copy_(original_attn.k_proj.weight)
                self.v_proj.weight.data.copy_(original_attn.v_proj.weight)

                if self.q_proj.bias is not None:
                    self.q_proj.bias.data.copy_(original_attn.q_proj.bias)
                    self.k_proj.bias.data.copy_(original_attn.k_proj.bias)
                    self.v_proj.bias.data.copy_(original_attn.v_proj.bias)

        # Copy out_proj parameters
        self.out_proj.weight.data.copy_(original_attn.out_proj.weight)
        if self.out_proj.bias is not None:
            self.out_proj.bias.data.copy_(original_attn.out_proj.bias)

        # Apply SVF to each projection
        if type == "single":
            self.q_proj = SVFLinear(self.q_proj, rank=self.svf_rank) if apply_to_q else self.q_proj
            self.k_proj = SVFLinear(self.k_proj, rank=self.svf_rank) if apply_to_k else self.k_proj
            self.v_proj = SVFLinear(self.v_proj, rank=self.svf_rank) if apply_to_v else self.v_proj
            self.out_proj = (
                SVFLinear(self.out_proj, rank=self.svf_rank)
                if apply_to_attn_proj
                else self.out_proj
            )

        elif "multi-" in type:
            aggregation, layers_count = type.split("-")[1:]
            layers_count = int(layers_count)
            self.q_proj = (
                MultiSVFLinear(
                    self.q_proj, layers_count, aggregation=aggregation, rank=self.svf_rank
                )
                if apply_to_q
                else self.q_proj
            )
            self.k_proj = (
                MultiSVFLinear(
                    self.k_proj, layers_count, aggregation=aggregation, rank=self.svf_rank
                )
                if apply_to_k
                else self.k_proj
            )
            self.v_proj = (
                MultiSVFLinear(
                    self.v_proj, layers_count, aggregation=aggregation, rank=self.svf_rank
                )
                if apply_to_v
                else self.v_proj
            )
            self.out_proj = (
                MultiSVFLinear(
                    self.out_proj, layers_count, aggregation=aggregation, rank=self.svf_rank
                )
                if apply_to_attn_proj
                else self.out_proj
            )

        elif "multiple_s-" in type:
            split = type.split("-")
            if len(split) == 2:
                actual_type, layers_count = split[1], 1
                topk = 2
            else:
                actual_type, layers_count = split[1], split[2]
                layers_count = int(layers_count) if layers_count != "auto" else 1
                topk = int(split[3]) if len(split) >= 4 else 2

            if actual_type == "avg_weights":
                Model = SVFLinearWithMultipleSAverageWeights
            elif actual_type == "training_router":
                Model = SVFLinearWithMultipleSWithTrainingRouter
            elif actual_type == "independent":
                Model = SVFLinearWithMultipleSIndependent
            elif actual_type == "learnable_weights":
                Model = SVFLinearWithMultipleSWithLearnableWeights
            elif actual_type == "dense_moe":
                Model = SVFLinearWithMultipleSDenseMoE
            elif actual_type == "sparse_moe":
                Model = SVFLinearWithMultipleSSparseMoE
            elif actual_type == "external_router":
                Model = SVFLinearWithMultipleSWithExternalRouter
            else:
                Model = SVFLinearWithMultipleS

            self.q_proj = (
                Model(self.q_proj, initial_s_count=layers_count, rank=self.svf_rank, topk=topk)
                if apply_to_q
                else self.q_proj
            )
            self.k_proj = (
                Model(self.k_proj, initial_s_count=layers_count, rank=self.svf_rank, topk=topk)
                if apply_to_k
                else self.k_proj
            )
            self.v_proj = (
                Model(self.v_proj, initial_s_count=layers_count, rank=self.svf_rank, topk=topk)
                if apply_to_v
                else self.v_proj
            )
            self.out_proj = (
                Model(self.out_proj, initial_s_count=layers_count, rank=self.svf_rank, topk=topk)
                if apply_to_attn_proj
                else self.out_proj
            )

        self._qkv_initialized = True

        output = {}
        if apply_to_q:
            output["q_proj"] = self.q_proj
        if apply_to_k:
            output["k_proj"] = self.k_proj
        if apply_to_v:
            output["v_proj"] = self.v_proj
        if apply_to_attn_proj:
            output["out_proj"] = self.out_proj

        return output

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass largely mirroring nn.MultiheadAttention but with SVF projections.
        """
        if not self._qkv_initialized:
            raise RuntimeError(
                "SVFMultiHeadAttention parameters not initialized. Call set_parameters first."
            )

        # Default to self-attention if key/value not provided
        if key is None:
            key = query
        if value is None:
            value = key

        # Handle batch_first format
        if self.batch_first:
            query, key, value = [x.transpose(0, 1) for x in (query, key, value)]

        # Get sequence length and batch size
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)

        # Apply SVF projections
        q = self.q_proj(query).view(tgt_len, bsz, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(src_len, bsz, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(src_len, bsz, self.num_heads, self.head_dim)

        # Transpose for batched matrix multiplication
        q = q.transpose(0, 1).transpose(1, 2)  # (bsz, num_heads, tgt_len, head_dim)
        k = k.transpose(0, 1).transpose(1, 2)  # (bsz, num_heads, src_len, head_dim)
        v = v.transpose(0, 1).transpose(1, 2)  # (bsz, num_heads, src_len, head_dim)

        # Compute scaled dot-product attention
        # (bsz, num_heads, tgt_len, src_len)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        # Apply causal mask if needed
        if is_causal:
            # Generate causal mask if not provided
            if attn_mask is None:
                attn_mask = torch.triu(
                    torch.ones(tgt_len, src_len, dtype=torch.bool, device=query.device), diagonal=1
                )

        # Apply attention masks if provided
        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                # Convert float mask to boolean: negative = True, positive = False
                if attn_mask.dtype == torch.float:
                    float_mask = attn_mask
                    attn_mask = float_mask < 0
                else:
                    # Try to convert to boolean if not already
                    attn_mask = attn_mask.to(torch.bool)

            # Expand mask for broadcasting
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(attn_mask, float("-inf"))

        if key_padding_mask is not None:
            # Convert padding mask to attention mask
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (bsz, 1, 1, src_len)
            attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))

        # Apply softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (bsz, num_heads, tgt_len, head_dim)

        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        # Return to original sequence format if needed
        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        # Average attention weights across heads if requested
        if need_weights and average_attn_weights:
            attn_weights = attn_weights.mean(dim=1)

        return attn_output, attn_weights if need_weights else None


class SVFInterface:
    """Facade over a list of SVF layers, delegating operations to all layers."""

    def __init__(self, layers: list):
        self.svf_layers = layers

    @property
    def actual_type(self):
        """
        Get the actual type of the SVF layers.
        """
        if len(self.svf_layers) == 0:
            return None
        return self.svf_layers[0].actual_type

    @property
    def log_training(self):
        """
        Check if any of the SVF layers is in log training mode.
        """
        for layer in self.svf_layers:
            if hasattr(layer, "log_training"):
                return layer.log_training

        return False

    def train_mode(self):
        """
        Set all SVF layers to training mode.
        """
        for layer in self.svf_layers:
            if hasattr(layer, "train_mode"):
                layer.train_mode()

    def eval_mode(self):
        """
        Set all SVF layers to eval mode.
        """
        for layer in self.svf_layers:
            if hasattr(layer, "eval_mode"):
                layer.eval_mode()

    def add_s(self, count: int = 1):
        """
        Add S parameters to all SVF layers.
        """
        for layer in self.svf_layers:
            if hasattr(layer, "add_s"):
                layer.add_s(count)

    def freeze_all_s(self):
        """
        Freeze all S parameters in all SVF layers.
        """
        for layer in self.svf_layers:
            if hasattr(layer, "freeze_all_s"):
                layer.freeze_all_s()

    def unfreeze_all_s(self):
        """
        Unfreeze all S parameters in all SVF layers.
        """
        for layer in self.svf_layers:
            if hasattr(layer, "unfreeze_all_s"):
                layer.unfreeze_all_s()

    def freeze_s(self, index: int):
        """
        Freeze the S parameter at the given index in all SVF layers.
        """
        for layer in self.svf_layers:
            if hasattr(layer, "freeze_s"):
                layer.freeze_s(index)

    def unfreeze_s(self, index: int):
        """
        Unfreeze the S parameter at the given index in all SVF layers.
        """
        for layer in self.svf_layers:
            if hasattr(layer, "unfreeze_s"):
                layer.unfreeze_s(index)

    def set_training_router_for_batch(self, indices: list[int]):
        """
        Set the training router for the current batch in all SVF layers.
        """
        for layer in self.svf_layers:
            if hasattr(layer, "set_training_router_for_batch"):
                layer.set_training_router_for_batch(indices)

    def set_router_for_batch(self, indices: list[int]):
        """
        Set the router for the current batch in all SVF layers.
        """
        for layer in self.svf_layers:
            if hasattr(layer, "set_router_for_batch"):
                layer.set_router_for_batch(indices)

    def set_active_expert(self, index: int):
        """
        Set the active expert for the current batch in all SVF layers.
        """
        for layer in self.svf_layers:
            if hasattr(layer, "set_active_expert"):
                layer.set_active_expert(index)

    def clear_active_expert(self):
        """
        Clear the active expert index in all SVF layers.
        """
        for layer in self.svf_layers:
            if hasattr(layer, "clear_active_expert"):
                layer.clear_active_expert()

    def get_active_expert(self):
        """
        Get the active expert index from the first SVF layer.
        """
        if not self.svf_layers:
            return None

        for layer in self.svf_layers:
            if hasattr(layer, "get_active_expert"):
                return layer.get_active_expert()

        return None

    def get_number_of_S(self):
        """
        Get the number of S parameters in each SVF layer.
        As they are always updated together, we can just return the first one.
        """
        return len(self.svf_layers[0].S) if self.svf_layers else 0

    def set_optimizer(self, optimizer):
        """
        Set the optimizer for all SVF layers.
        """
        for layer in self.svf_layers:
            if hasattr(layer, "optimizer"):
                layer.optimizer = optimizer

    def get_experts(self):
        """
        Get all the experts
        """

        return self.svf_layers

    def clamp_singular_values(self):
        """
        Clamp the singular values in all SVF layers.
        This is useful for models with multiple experts.
        """
        for layer in self.svf_layers:
            if hasattr(layer, "_clamp_singular_values"):
                layer._clamp_singular_values()

    def average_experts(self, mode: str = "avg", weights: list[int] | torch.Tensor | None = None):
        """
        Average the S parameters across all SVF layers.
        This is useful for models with multiple experts.
        """
        for layer in self.svf_layers:
            if mode == "avg" and hasattr(layer, "average_experts"):
                layer.average_experts()
            elif mode == "deltas" and hasattr(layer, "average_experts_deltas"):
                layer.average_experts_deltas()
            elif mode == "only_deltas" and hasattr(layer, "average_experts_keeping_only_deltas"):
                layer.average_experts_keeping_only_deltas()
            elif mode == "interpolate" and hasattr(layer, "interpolate_experts"):
                layer.interpolate_experts(weights)
            elif mode == "interpolate_inverse" and hasattr(layer, "interpolate_experts_inverse"):
                layer.interpolate_experts_inverse(weights)
            elif mode == "svd" and hasattr(layer, "svd_experts"):
                layer.svd_experts()
            elif mode == "elementwise" and hasattr(layer, "elementwise_experts"):
                layer.elementwise_experts()
            elif mode == "my_tsv" and hasattr(layer, "my_tsv_experts"):
                layer.my_tsv_experts()
            elif mode == "tsv_1" and hasattr(layer, "tsv_1_experts"):
                layer.tsv_1_experts()


def finalize_svf_layers(transformer: nn.Module):
    """Replace SVF layers in transformer with finalized (frozen) linear layers."""
    for block in transformer.resblocks:
        attn = block.attn
        mlp = block.mlp

        layer = None
        if hasattr(attn, "q_proj") or hasattr(attn, "k_proj") or hasattr(attn, "v_proj"):
            # Attention
            dtype = None
            if hasattr(attn, "q_proj"):
                if isinstance(attn.q_proj, SVFLinear):
                    weight_q = attn.q_proj.finalize()
                    dtype = attn.q_proj.original_dtype
                    weight_q = weight_q.to(dtype)
                else:
                    weight_q = attn.q_proj.weight.data

            if hasattr(attn, "k_proj"):
                if isinstance(attn.k_proj, SVFLinear):
                    weight_k = attn.k_proj.finalize()
                    dtype = attn.k_proj.original_dtype
                    weight_k = weight_k.to(dtype)
                else:
                    weight_k = attn.k_proj.weight.data

            if hasattr(attn, "v_proj"):
                if isinstance(attn.v_proj, SVFLinear):
                    weight_v = attn.v_proj.finalize()
                    dtype = attn.v_proj.original_dtype
                    weight_v = weight_v.to(dtype)
                else:
                    weight_v = attn.v_proj.weight.data

            weights = torch.cat([weight_q, weight_k, weight_v], dim=0)
            if dtype is not None:
                weights = weights.to(dtype)
            biases = torch.cat(
                [attn.q_proj.bias.data, attn.k_proj.bias.data, attn.v_proj.bias.data], dim=0
            )

            layer = True

        layer_out = None
        if hasattr(attn, "out_proj") and isinstance(attn.out_proj, SVFLinear):
            weight_out = attn.out_proj.finalize()
            bias_out = attn.out_proj.bias.data

            dtype_out = attn.out_proj.original_dtype
            weight_out = weight_out.to(dtype_out)
            bias_out = bias_out.to(dtype_out)

            layer_out = True

        if layer is None and layer_out is None:
            continue

        block.attn = nn.MultiheadAttention(
            attn.embed_dim, attn.num_heads, batch_first=attn.batch_first
        ).to(dtype)
        if layer is not None and layer_out is not None:
            block.attn.in_proj_weight.data.copy_(weights)
            block.attn.in_proj_bias.data.copy_(biases)

            block.attn.out_proj.weight.data.copy_(weight_out)
            block.attn.out_proj.bias.data.copy_(bias_out)

            block.attn = block.attn.to(dtype)

        # MLP
        for name, module in mlp.named_children():
            if isinstance(module, SVFLinear):
                weights = module.finalize()
                layer = nn.Linear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                )
                layer.weight.data.copy_(weights)
                layer.bias.data.copy_(module.bias.data)
                layer = layer.to(module.original_dtype)
                setattr(mlp, name, layer)


def apply_svf_to_model(
    model: nn.Module,
    rank: int = -1,
    apply_to_mlp: bool = True,
    apply_to_q: bool = True,
    apply_to_k: bool = True,
    apply_to_v: bool = True,
    apply_to_attn_proj: bool = True,
    start_block: int = 0,
    type: str = "single",
    apply_to_text_projection: bool = False,
    optimizer=None,
):
    """
    Recursively replaces target layers in a model with SVFLinear layers.
    """

    svf_layers = []

    if apply_to_text_projection:
        transformer = model.transformer

        module = model.text_projection
        # print(module.shape)
        # exit()
        # Replace the text projection with SVFLinear
        if type == "single":
            svf_layer = SVFLinear(module, rank=rank)

        elif "multi-" in type:
            aggregation, layers_count = type.split("-")[1:]
            layers_count = int(layers_count)
            svf_layer = MultiSVFLinear(module, layers_count, aggregation=aggregation, rank=rank)

        elif "multiple_s-" in type:
            split = type.split("-")
            if len(split) == 2:
                actual_type, layers_count = split[1], 1
                topk = 2
            else:
                actual_type, layers_count = split[1], split[2]
                layers_count = int(layers_count) if layers_count != "auto" else 1
                topk = int(split[3]) if len(split) >= 4 else 2

            if actual_type == "avg_weights":
                Model = SVFLinearWithMultipleSAverageWeights
            elif actual_type == "training_router":
                Model = SVFLinearWithMultipleSWithTrainingRouter
            elif actual_type == "independent":
                Model = SVFLinearWithMultipleSIndependent
            elif actual_type == "learnable_weights":
                Model = SVFLinearWithMultipleSWithLearnableWeights
            elif actual_type == "dense_moe":
                Model = SVFLinearWithMultipleSDenseMoE
            elif actual_type == "sparse_moe":
                Model = SVFLinearWithMultipleSSparseMoE
            elif actual_type == "external_router":
                Model = SVFLinearWithMultipleSWithExternalRouter
            else:
                Model = SVFLinearWithMultipleS

            svf_layer = Model(module, initial_s_count=layers_count, rank=rank, topk=topk)

        model.text_projection = svf_layer
        svf_layers.append(svf_layer)

    else:
        transformer = model

    apply_to_attn = apply_to_q or apply_to_k or apply_to_v or apply_to_attn_proj
    for block in transformer.resblocks[start_block:]:
        if apply_to_attn:
            attn = block.attn

            svf_attn = SVFMultiHeadAttention(
                embed_dim=attn.embed_dim,
                num_heads=attn.num_heads,
                svf_rank=rank,
                dropout=attn.dropout,
                bias=attn.out_proj.bias is not None,  # Check bias in out_proj
                batch_first=attn.batch_first,
            )

            # Copy parameters from the original attention layer
            created_svf_layers = svf_attn.set_parameters(
                attn,
                apply_to_q=apply_to_q,
                apply_to_k=apply_to_k,
                apply_to_v=apply_to_v,
                apply_to_attn_proj=apply_to_attn_proj,
                type=type,
            )

            block.attn = svf_attn
            svf_layers += list(created_svf_layers.values())

        if apply_to_mlp:
            mlp = block.mlp
            for name, module in mlp.named_children():
                if isinstance(module, nn.Linear):
                    # Replace the module with SVFLinear
                    if type == "single":
                        svf_layer = SVFLinear(module, rank=rank)

                    elif "multi-" in type:
                        aggregation, layers_count = type.split("-")[1:]
                        layers_count = int(layers_count)
                        svf_layer = MultiSVFLinear(
                            module, layers_count, aggregation=aggregation, rank=rank
                        )

                    elif "multiple_s-" in type:
                        split = type.split("-")
                        if len(split) == 2:
                            actual_type, layers_count = split[1], 1
                            topk = 2
                        else:
                            actual_type, layers_count = split[1], split[2]
                            layers_count = int(layers_count) if layers_count != "auto" else 1
                            topk = int(split[3]) if len(split) >= 4 else 2

                        if actual_type == "avg_weights":
                            Model = SVFLinearWithMultipleSAverageWeights
                        elif actual_type == "training_router":
                            Model = SVFLinearWithMultipleSWithTrainingRouter
                        elif actual_type == "independent":
                            Model = SVFLinearWithMultipleSIndependent
                        elif actual_type == "learnable_weights":
                            Model = SVFLinearWithMultipleSWithLearnableWeights
                        elif actual_type == "dense_moe":
                            Model = SVFLinearWithMultipleSDenseMoE
                        elif actual_type == "sparse_moe":
                            Model = SVFLinearWithMultipleSSparseMoE
                        elif actual_type == "external_router":
                            Model = SVFLinearWithMultipleSWithExternalRouter
                        else:
                            Model = SVFLinearWithMultipleS

                        svf_layer = Model(
                            module, initial_s_count=layers_count, rank=rank, topk=topk
                        )

                    setattr(mlp, name, svf_layer)
                    svf_layers.append(svf_layer)

    interface = SVFInterface(svf_layers)

    return transformer, interface
