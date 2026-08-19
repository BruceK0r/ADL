from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class ModelOutput:
    """Common output contract used by all models and the trainer."""

    logits: Tensor
    auxiliary_loss: Tensor
    diagnostics: dict[str, Tensor] = field(default_factory=dict)


def _output(logits: Tensor, diagnostics: Mapping[str, Tensor] | None = None) -> ModelOutput:
    return ModelOutput(
        logits=logits,
        auxiliary_loss=logits.new_zeros(()),
        diagnostics=dict(diagnostics or {}),
    )


class FeatureEncoder(nn.Module):
    """Embeds single-valued categorical fields and appends dense fields."""

    def __init__(self, cardinalities: Sequence[int], dense_dim: int, embedding_dim: int = 16):
        super().__init__()
        if dense_dim < 0 or embedding_dim <= 0:
            raise ValueError("dense_dim must be non-negative and embedding_dim must be positive")
        if any(int(size) <= 0 for size in cardinalities):
            raise ValueError("Every categorical cardinality must be positive")
        self.embeddings = nn.ModuleList(
            nn.Embedding(int(size), embedding_dim) for size in cardinalities
        )
        self.dense_dim = int(dense_dim)
        self.output_dim = len(cardinalities) * embedding_dim + self.dense_dim
        for table in self.embeddings:
            nn.init.normal_(table.weight, mean=0.0, std=0.01)

    def categorical_parts(self, categorical: Tensor) -> list[Tensor]:
        if categorical.ndim != 2:
            raise ValueError("categorical input must be rank-2")
        if categorical.shape[1] != len(self.embeddings):
            raise ValueError("categorical column count does not match the configured embeddings")
        return [table(categorical[:, i]) for i, table in enumerate(self.embeddings)]

    def forward(self, categorical: Tensor, dense: Tensor) -> Tensor:
        if categorical.ndim != 2 or dense.ndim != 2:
            raise ValueError("categorical and dense inputs must both be rank-2")
        pieces = self.categorical_parts(categorical)
        if self.dense_dim:
            pieces.append(dense.float())
        if not pieces:
            raise ValueError("At least one categorical or dense feature is required")
        return torch.cat(pieces, dim=1)


def _mlp(input_dim: int, dims: Sequence[int], *, output_dim: int = 1) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for width in dims:
        layers.extend([nn.Linear(current, int(width)), nn.ReLU()])
        current = int(width)
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


def _hidden_mlp(input_dim: int, dims: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for width in dims:
        layers.extend([nn.Linear(current, int(width)), nn.ReLU()])
        current = int(width)
    return nn.Sequential(*layers)


class SharedBottom(nn.Module):
    """One shared first layer followed by one task tower per explicit scenario."""

    def __init__(
        self,
        encoder: FeatureEncoder,
        scenario_num: int,
        hidden_dims: Sequence[int] = (512, 256, 128, 64, 32),
        shared_layer_count: int = 1,
    ):
        super().__init__()
        if scenario_num <= 0:
            raise ValueError("scenario_num must be positive")
        if any(int(width) <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if not 1 <= shared_layer_count < len(hidden_dims):
            raise ValueError("shared_layer_count must leave at least one layer for each scenario tower")
        self.encoder = encoder
        self.scenario_num = int(scenario_num)
        self.shared_layer_count = int(shared_layer_count)
        shared_dims = hidden_dims[: self.shared_layer_count]
        tower_dims = hidden_dims[self.shared_layer_count :]
        self.bottom = _hidden_mlp(encoder.output_dim, shared_dims)
        self.towers = nn.ModuleList(
            _mlp(shared_dims[-1], tower_dims) for _ in range(self.scenario_num)
        )

    def forward(self, categorical: Tensor, dense: Tensor, scenario_id: Tensor) -> ModelOutput:
        scenario_id = scenario_id.long().view(-1)
        if torch.any((scenario_id < 0) | (scenario_id >= self.scenario_num)):
            raise ValueError("scenario_id is outside the configured range")
        shared = self.bottom(self.encoder(categorical, dense))
        logits = shared.new_zeros(shared.shape[0])
        for scenario, tower in enumerate(self.towers):
            indices = torch.nonzero(scenario_id == scenario, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            scenario_logits = tower(shared.index_select(0, indices)).squeeze(1)
            logits.index_copy_(0, indices, scenario_logits)
        return _output(logits)


class DomainAwareGate(nn.Module):
    """AdaSparse Fusion gate: beta*sigmoid(alpha*z), hard-zeroed below epsilon."""

    def __init__(self, scenario_dim: int, feature_dim: int, epsilon: float, beta: float):
        super().__init__()
        self.linear = nn.Linear(scenario_dim + feature_dim, feature_dim, bias=False)
        self.epsilon = float(epsilon)
        self.beta = float(beta)

    def forward(
        self, scenario: Tensor, features: Tensor, alpha: Tensor
    ) -> tuple[Tensor, Tensor]:
        value = self.beta * torch.sigmoid(alpha * self.linear(torch.cat([scenario, features], 1)))
        hard_gate = torch.where(value > self.epsilon, value, torch.zeros_like(value))
        # Smooth proxy for the zero-neuron ratio. The hard gate remains the actual
        # forward structure; this proxy lets the sparsity regularizer carry gradients.
        soft_pruned_probability = torch.sigmoid((self.epsilon - value) * 20.0)
        return hard_gate, soft_pruned_probability.mean()


class AdaSparse(nn.Module):
    """Neuron-level domain-aware Fusion pruning from AdaSparse."""

    def __init__(
        self,
        encoder: FeatureEncoder,
        scenario_num: int,
        hidden_dims: Sequence[int] = (512, 256, 128, 64, 32),
        domain_aware_indices: Sequence[int] = (),
        scenario_embedding_dim: int = 16,
        epsilon: float = 1e-2,
        beta: float = 2.0,
        alpha: float = 0.1,
        delta_alpha: float = 1e-4,
        alpha_max: float = 5.0,
        sparsity_min: float = 0.15,
        sparsity_max: float = 0.25,
        regularization_weight: float = 0.01,
        regularization_growth: float = 1e-5,
        regularization_max: float = 1.0,
    ):
        super().__init__()
        if scenario_num <= 0:
            raise ValueError("scenario_num must be positive")
        if not hidden_dims or any(int(width) <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if beta <= 0.0 or not 0.0 <= epsilon < beta:
            raise ValueError("AdaSparse expects beta > 0 and 0 <= epsilon < beta")
        if not 0.0 <= alpha <= alpha_max or delta_alpha < 0.0:
            raise ValueError("AdaSparse alpha schedule is invalid")
        if regularization_weight < 0.0 or regularization_growth < 0.0:
            raise ValueError("AdaSparse regularization schedule must be non-negative")
        if regularization_max < regularization_weight:
            raise ValueError("regularization_max must be at least regularization_weight")
        if not 0.0 <= sparsity_min <= sparsity_max < 1.0:
            raise ValueError("Expected 0 <= sparsity_min <= sparsity_max < 1")
        self.encoder = encoder
        self.scenario_num = int(scenario_num)
        self.domain_aware_indices = tuple(int(index) for index in domain_aware_indices)
        if len(set(self.domain_aware_indices)) != len(self.domain_aware_indices):
            raise ValueError("domain_aware_indices must not contain duplicates")
        if any(index < 0 or index >= len(encoder.embeddings) for index in self.domain_aware_indices):
            raise ValueError("domain_aware_indices contains an invalid categorical column index")
        self.scenario_embedding = (
            None
            if self.domain_aware_indices
            else nn.Embedding(self.scenario_num, scenario_embedding_dim)
        )
        categorical_embedding_dim = (
            encoder.embeddings[0].embedding_dim if encoder.embeddings else scenario_embedding_dim
        )
        domain_aware_dim = (
            len(self.domain_aware_indices) * categorical_embedding_dim
            if self.domain_aware_indices
            else scenario_embedding_dim
        )
        self.agnostic_indices = tuple(
            index
            for index in range(len(encoder.embeddings))
            if index not in self.domain_aware_indices
        )
        agnostic_input_dim = (
            len(self.agnostic_indices) * categorical_embedding_dim + encoder.dense_dim
            if self.domain_aware_indices
            else encoder.output_dim
        )
        if agnostic_input_dim <= 0:
            raise ValueError("AdaSparse requires at least one domain-agnostic input feature")
        self.register_buffer("alpha", torch.tensor(float(alpha)))
        self.register_buffer("regularization_weight", torch.tensor(float(regularization_weight)))
        self.delta_alpha = float(delta_alpha)
        self.alpha_max = float(alpha_max)
        self.regularization_growth = float(regularization_growth)
        self.regularization_max = float(regularization_max)
        self.sparsity_min = float(sparsity_min)
        self.sparsity_max = float(sparsity_max)
        self.target_sparsity = 0.5 * (self.sparsity_min + self.sparsity_max)

        widths = [agnostic_input_dim, *map(int, hidden_dims)]
        self.gates = nn.ModuleList(
            DomainAwareGate(domain_aware_dim, width, epsilon, beta) for width in widths
        )
        self.layers = nn.ModuleList(
            nn.Linear(widths[i], widths[i + 1]) for i in range(len(widths) - 1)
        )
        self.output = nn.Linear(widths[-1], 1)

    def forward(self, categorical: Tensor, dense: Tensor, scenario_id: Tensor) -> ModelOutput:
        scenario_id = scenario_id.long().view(-1)
        if torch.any((scenario_id < 0) | (scenario_id >= self.scenario_num)):
            raise ValueError("scenario_id is outside the configured range")
        if self.training:
            self.alpha.add_(self.delta_alpha).clamp_(max=self.alpha_max)
            self.regularization_weight.add_(self.regularization_growth).clamp_(
                max=self.regularization_max
            )
        categorical_parts = self.encoder.categorical_parts(categorical)
        if self.domain_aware_indices:
            scenario = torch.cat(
                [categorical_parts[index] for index in self.domain_aware_indices], dim=1
            )
        else:
            assert self.scenario_embedding is not None
            scenario = self.scenario_embedding(scenario_id)
        pieces = (
            [categorical_parts[index] for index in self.agnostic_indices]
            if self.domain_aware_indices
            else list(categorical_parts)
        )
        if self.encoder.dense_dim:
            pieces.append(dense.float())
        hidden = torch.cat(pieces, dim=1)
        first_gate, first_soft_sparsity = self.gates[0](scenario, hidden, self.alpha)
        gate_values = [first_gate]
        soft_sparsities = [first_soft_sparsity]
        hidden = first_gate * hidden
        for index, layer in enumerate(self.layers):
            hidden = F.relu(layer(hidden))
            gate, soft_sparsity = self.gates[index + 1](scenario, hidden, self.alpha)
            gate_values.append(gate)
            soft_sparsities.append(soft_sparsity)
            hidden = gate * hidden
        logits = self.output(hidden).squeeze(1)

        sparsities = torch.stack([(gate == 0).float().mean() for gate in gate_values])
        differentiable_sparsities = torch.stack(soft_sparsities)
        outside = (sparsities < self.sparsity_min) | (sparsities > self.sparsity_max)
        distances = (differentiable_sparsities - self.target_sparsity).abs()
        dynamic_lambda = torch.where(
            outside,
            self.regularization_weight * distances.detach(),
            torch.zeros_like(distances),
        )
        # Eq. (9)-(10): lambda_l * |r_l-r|, with
        # lambda_l = lambda_hat * |r_l-r| outside the target interval.
        auxiliary_loss = (dynamic_lambda * distances).mean()
        return ModelOutput(
            logits=logits,
            auxiliary_loss=auxiliary_loss,
            diagnostics={
                "alpha": self.alpha.detach().clone(),
                "regularization_weight": self.regularization_weight.detach().clone(),
                "mean_sparsity": sparsities.mean().detach(),
                "layer_sparsity": sparsities.detach(),
            },
        )


class DLMRouter(nn.Module):
    """Paper-faithful Algorithm 1 with one EWMA update per training batch."""

    def __init__(self, input_dim: int, cluster_num: int = 9, beta: float = 0.9, iterations: int = 3):
        super().__init__()
        if input_dim <= 0 or cluster_num <= 0 or iterations <= 0:
            raise ValueError("DLM input_dim, cluster_num, and iterations must be positive")
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must lie in [0, 1)")
        initial = F.normalize(torch.randn(cluster_num, input_dim), p=2, dim=1)
        self.register_buffer("centers", initial)
        self.cluster_num = int(cluster_num)
        self.beta = float(beta)
        self.iterations = int(iterations)

    @torch.no_grad()
    def forward(self, representation: Tensor) -> tuple[Tensor, Tensor]:
        detached = representation.detach()
        inherited = self.centers.clone()
        if self.training:
            current = inherited
            coefficients = None
            for _ in range(self.iterations):
                scores = detached @ current.T
                coefficients = scores.softmax(dim=1)
                current = F.normalize(coefficients.T @ detached, p=2, dim=1)
            updated = F.normalize(
                self.beta * inherited + (1.0 - self.beta) * current, p=2, dim=1
            )
            self.centers.copy_(updated)
            assert coefficients is not None
        else:
            coefficients = (detached @ inherited.T).softmax(dim=1)
        route = coefficients.argmax(dim=1)
        return route, coefficients


class FactorizedClusterLayer(nn.Module):
    """W_m = W_shared element-wise-multiplied by W_cluster (ADL Eq. 6)."""

    def __init__(self, input_dim: int, output_dim: int, cluster_num: int):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or cluster_num <= 0:
            raise ValueError("Factorized layer dimensions must be positive")
        self.shared_weight = nn.Parameter(torch.empty(output_dim, input_dim))
        self.cluster_weight = nn.Parameter(torch.empty(cluster_num, output_dim, input_dim))
        nn.init.xavier_uniform_(self.shared_weight, gain=nn.init.calculate_gain("relu"))
        for cluster_weight in self.cluster_weight:
            nn.init.xavier_uniform_(cluster_weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, inputs: Tensor, route: Tensor) -> Tensor:
        # Group samples by cluster. Advanced-indexing every sample's full weight
        # matrix would create a [batch, output_dim, input_dim] tensor and is not
        # viable for Ali-CCP's batch size 2048 and 512-wide first layer.
        outputs = None
        for cluster in range(self.cluster_weight.shape[0]):
            indices = torch.nonzero(route == cluster, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            effective_weight = self.cluster_weight[cluster] * self.shared_weight
            cluster_output = F.linear(inputs.index_select(0, indices), effective_weight)
            # Under autocast, F.linear may return float16/bfloat16 even when
            # `inputs` is float32. Allocate from the first actual branch output
            # so index_copy_ always has matching source/destination dtypes.
            if outputs is None:
                outputs = cluster_output.new_zeros(
                    (inputs.shape[0], self.shared_weight.shape[0])
                )
            outputs.index_copy_(0, indices, cluster_output)
        if outputs is None:
            return inputs.new_zeros((inputs.shape[0], self.shared_weight.shape[0]))
        return outputs


class ADL(nn.Module):
    """ADL with stop-gradient DLM routing and factorized cluster-specific MLPs."""

    def __init__(
        self,
        encoder: FeatureEncoder,
        hidden_dims: Sequence[int] = (512, 256, 128, 64, 32),
        cluster_num: int = 9,
        beta: float = 0.9,
        routing_iterations: int = 3,
    ):
        super().__init__()
        if not hidden_dims or any(int(width) <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        self.encoder = encoder
        self.router = DLMRouter(encoder.output_dim, cluster_num, beta, routing_iterations)
        widths = [encoder.output_dim, *map(int, hidden_dims), 1]
        self.layers = nn.ModuleList(
            FactorizedClusterLayer(widths[i], widths[i + 1], cluster_num)
            for i in range(len(widths) - 1)
        )

    def forward(
        self,
        categorical: Tensor,
        dense: Tensor,
        scenario_id: Tensor | None = None,
        *,
        return_routing: bool = False,
    ) -> ModelOutput:
        representation = self.encoder(categorical, dense)
        route, coefficients = self.router(representation)
        hidden = representation
        for layer in self.layers[:-1]:
            hidden = F.relu(layer(hidden, route))
        logits = self.layers[-1](hidden, route).squeeze(1)
        diagnostics = {
            "cluster_counts": torch.bincount(route, minlength=self.router.cluster_num),
            "mean_routing_entropy": (
                -(coefficients * coefficients.clamp_min(1e-12).log()).sum(dim=1).mean()
            ).detach(),
        }
        if return_routing:
            diagnostics.update({"route": route, "coefficients": coefficients})
        return _output(logits, diagnostics)


@dataclass(frozen=True)
class ModelConfig:
    cardinalities: Sequence[int]
    dense_dim: int
    scenario_num: int
    embedding_dim: int = 16
    hidden_dims: Sequence[int] = (512, 256, 128, 64, 32)
    cluster_num: int = 9
    beta: float = 0.9
    routing_iterations: int = 3
    shared_layer_count: int = 1
    domain_aware_indices: Sequence[int] = ()
    adasparse_epsilon: float = 0.25
    adasparse_beta: float = 2.0
    adasparse_alpha: float = 0.1
    adasparse_delta_alpha: float = 1e-4
    adasparse_alpha_max: float = 5.0
    adasparse_sparsity_min: float = 0.15
    adasparse_sparsity_max: float = 0.25
    adasparse_regularization_weight: float = 0.01
    adasparse_regularization_growth: float = 1e-5
    adasparse_regularization_max: float = 1.0


def build_model(name: str, config: ModelConfig) -> nn.Module:
    encoder = FeatureEncoder(config.cardinalities, config.dense_dim, config.embedding_dim)
    normalized = name.lower().replace("_", "")
    if normalized == "sharedbottom":
        return SharedBottom(
            encoder,
            config.scenario_num,
            config.hidden_dims,
            config.shared_layer_count,
        )
    if normalized == "adasparse":
        return AdaSparse(
            encoder,
            config.scenario_num,
            config.hidden_dims,
            config.domain_aware_indices,
            scenario_embedding_dim=config.embedding_dim,
            epsilon=config.adasparse_epsilon,
            beta=config.adasparse_beta,
            alpha=config.adasparse_alpha,
            delta_alpha=config.adasparse_delta_alpha,
            alpha_max=config.adasparse_alpha_max,
            sparsity_min=config.adasparse_sparsity_min,
            sparsity_max=config.adasparse_sparsity_max,
            regularization_weight=config.adasparse_regularization_weight,
            regularization_growth=config.adasparse_regularization_growth,
            regularization_max=config.adasparse_regularization_max,
        )
    if normalized == "adl":
        return ADL(
            encoder,
            config.hidden_dims,
            config.cluster_num,
            config.beta,
            config.routing_iterations,
        )
    raise ValueError(f"Unknown model: {name}")
