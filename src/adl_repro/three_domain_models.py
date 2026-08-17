from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .models import DLMRouter, FactorizedClusterLayer, ModelOutput


@dataclass(frozen=True)
class ThreeDomainModelConfig:
    user_num: int
    item_num: int
    domain_num: int
    time_dim: int
    user_dim: int = 32
    item_dim: int = 32
    domain_dim: int = 8
    text_projection_dim: int = 128
    hidden_dims: Sequence[int] = (512, 256, 128, 64, 32)
    cluster_num: int = 3
    beta: float = 0.9
    routing_iterations: int = 3
    # Algorithm 1 in ADL routes with c_k dot e_i. Normalization remains an
    # explicit ablation because it can collapse hard assignments on Amazon.
    normalize_router_input: bool = False
    use_qwen: bool = True
    use_cross_domain_history: bool = True
    router_use_domain: bool = False


class ThreeDomainFeatureEncoder(nn.Module):
    """Encodes a context and its positive/negative candidates without target leakage."""

    def __init__(
        self,
        config: ThreeDomainModelConfig,
        item_text_embeddings: Tensor,
        item_domains: Tensor,
        train_seen_users: Tensor,
        train_seen_items: Tensor,
    ):
        super().__init__()
        if config.use_qwen and item_text_embeddings.shape[0] != config.item_num + 1:
            raise ValueError("Item text row count does not match item_num + padding")
        if item_domains.shape != (config.item_num + 1,):
            raise ValueError("item_domains must have one value per item plus padding")
        self.config = config
        self.user_embedding = nn.Embedding(config.user_num + 1, config.user_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(config.item_num + 1, config.item_dim, padding_idx=0)
        self.domain_embedding = nn.Embedding(config.domain_num, config.domain_dim)
        content_input_dim = item_text_embeddings.shape[1] if config.use_qwen else config.item_dim
        self.text_projection = nn.Sequential(
            nn.Linear(content_input_dim, config.text_projection_dim, bias=config.use_qwen),
            nn.LayerNorm(config.text_projection_dim),
            nn.ReLU(),
        )
        self.register_buffer(
            "item_text_embeddings",
            item_text_embeddings if config.use_qwen else torch.empty(0),
            persistent=False,
        )
        self.register_buffer("item_domains", item_domains.long(), persistent=False)
        self.register_buffer("train_seen_users", train_seen_users.float(), persistent=False)
        self.register_buffer("train_seen_items", train_seen_items.float(), persistent=False)
        self.output_dim = (
            config.user_dim
            + config.item_dim
            + config.domain_dim
            + 3 * config.text_projection_dim
            + config.time_dim
        )
        self.routing_dim = (
            config.user_dim
            + config.item_dim
            + 3 * config.text_projection_dim
            + config.time_dim
            + (config.domain_dim if config.router_use_domain else 0)
        )
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.domain_embedding.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.user_embedding.weight[0].zero_()
            self.item_embedding.weight[0].zero_()

    def _history_means(self, history: Tensor, domain: Tensor) -> tuple[Tensor, Tensor]:
        mask = history.ne(0)
        if self.config.use_qwen:
            representation = F.embedding(history, self.item_text_embeddings)
        else:
            seen = self.train_seen_items[history].bool()
            mask = mask & seen
            representation = self.item_embedding(history) * seen.unsqueeze(-1)
        global_denominator = mask.sum(dim=1, keepdim=True).clamp_min(1).to(representation.dtype)
        global_mean = (representation * mask.unsqueeze(-1)).sum(dim=1) / global_denominator

        history_domains = self.item_domains[history]
        domain_mask = mask & history_domains.eq(domain.unsqueeze(1))
        domain_denominator = (
            domain_mask.sum(dim=1, keepdim=True).clamp_min(1).to(representation.dtype)
        )
        domain_mean = (
            representation * domain_mask.unsqueeze(-1)
        ).sum(dim=1) / domain_denominator
        return global_mean, domain_mean

    def forward(
        self,
        user: Tensor,
        domain: Tensor,
        history: Tensor,
        time_features: Tensor,
        candidates: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if candidates.ndim != 2 or history.ndim != 2:
            raise ValueError("history and candidates must be rank-2")
        batch_size, candidate_count = candidates.shape
        user_part = self.user_embedding(user) * self.train_seen_users[user].unsqueeze(1)
        item_part = self.item_embedding(candidates) * self.train_seen_items[candidates].unsqueeze(-1)
        domain_part = self.domain_embedding(domain)
        candidate_source = (
            F.embedding(candidates, self.item_text_embeddings)
            if self.config.use_qwen
            else item_part
        )
        candidate_text = self.text_projection(candidate_source)
        global_mean, domain_mean = self._history_means(history, domain)
        global_history = self.text_projection(global_mean)
        domain_history = self.text_projection(domain_mean)
        if not self.config.use_cross_domain_history:
            global_history = torch.zeros_like(global_history)

        def expand(value: Tensor) -> Tensor:
            return value.unsqueeze(1).expand(batch_size, candidate_count, value.shape[-1])

        shared_parts = [
            expand(user_part),
            item_part,
            candidate_text,
            expand(global_history),
            expand(domain_history),
            expand(time_features),
        ]
        routing_parts = list(shared_parts)
        if self.config.router_use_domain:
            routing_parts.insert(2, expand(domain_part))
        routing = torch.cat(routing_parts, dim=-1).reshape(
            batch_size * candidate_count, -1
        )
        prediction = torch.cat(
            [shared_parts[0], shared_parts[1], expand(domain_part), *shared_parts[2:]], dim=-1
        ).reshape(batch_size * candidate_count, -1)
        return prediction, routing


class ThreeDomainADL(nn.Module):
    def __init__(
        self,
        config: ThreeDomainModelConfig,
        item_text_embeddings: Tensor,
        item_domains: Tensor,
        train_seen_users: Tensor,
        train_seen_items: Tensor,
    ):
        super().__init__()
        self.config = config
        self.encoder = ThreeDomainFeatureEncoder(
            config,
            item_text_embeddings,
            item_domains,
            train_seen_users,
            train_seen_items,
        )
        self.router = DLMRouter(
            self.encoder.routing_dim,
            config.cluster_num,
            config.beta,
            config.routing_iterations,
        )
        widths = [self.encoder.output_dim, *map(int, config.hidden_dims), 1]
        self.layers = nn.ModuleList(
            FactorizedClusterLayer(widths[index], widths[index + 1], config.cluster_num)
            for index in range(len(widths) - 1)
        )

    def forward(self, batch: dict[str, Tensor], *, return_routing: bool = False) -> ModelOutput:
        prediction, routing = self.encoder(
            batch["user"],
            batch["domain"],
            batch["history"],
            batch["time_features"],
            batch["candidates"],
        )
        if self.config.normalize_router_input:
            routing = F.normalize(routing, p=2, dim=1)
        # Center updates and routing softmax stay in float32 even when the classifier
        # is trained with AMP.
        with torch.autocast(device_type=routing.device.type, enabled=False):
            route, coefficients = self.router(routing.float())
        hidden = prediction
        for layer in self.layers[:-1]:
            hidden = F.relu(layer(hidden, route))
        logits = self.layers[-1](hidden, route).squeeze(1)
        logits = logits.reshape(batch["candidates"].shape)
        entropy = -(coefficients * coefficients.clamp_min(1e-12).log()).sum(dim=1).mean()
        diagnostics = {
            "cluster_counts": torch.bincount(route, minlength=self.config.cluster_num),
            "mean_routing_entropy": entropy.detach(),
        }
        if return_routing:
            diagnostics.update({"route": route, "coefficients": coefficients})
        return ModelOutput(logits=logits, auxiliary_loss=logits.new_zeros(()), diagnostics=diagnostics)


class ThreeDomainSharedBottom(nn.Module):
    """Shared MLP baseline using exactly the same candidate-level features as ADL."""

    def __init__(
        self,
        config: ThreeDomainModelConfig,
        item_text_embeddings: Tensor,
        item_domains: Tensor,
        train_seen_users: Tensor,
        train_seen_items: Tensor,
    ):
        super().__init__()
        self.config = config
        self.encoder = ThreeDomainFeatureEncoder(
            config,
            item_text_embeddings,
            item_domains,
            train_seen_users,
            train_seen_items,
        )
        widths = [self.encoder.output_dim, *map(int, config.hidden_dims), 1]
        self.layers = nn.ModuleList(
            nn.Linear(widths[index], widths[index + 1], bias=False)
            for index in range(len(widths) - 1)
        )

    def forward(self, batch: dict[str, Tensor], *, return_routing: bool = False) -> ModelOutput:
        del return_routing
        prediction, _ = self.encoder(
            batch["user"],
            batch["domain"],
            batch["history"],
            batch["time_features"],
            batch["candidates"],
        )
        hidden = prediction
        for layer in self.layers[:-1]:
            hidden = F.relu(layer(hidden))
        logits = self.layers[-1](hidden).squeeze(1)
        logits = logits.reshape(batch["candidates"].shape)
        return ModelOutput(logits=logits, auxiliary_loss=logits.new_zeros(()), diagnostics={})


def build_three_domain_model(
    name: str,
    config: ThreeDomainModelConfig,
    item_text_embeddings: Tensor,
    item_domains: Tensor,
    train_seen_users: Tensor,
    train_seen_items: Tensor,
) -> nn.Module:
    models = {
        "adl": ThreeDomainADL,
        "sharedbottom": ThreeDomainSharedBottom,
    }
    try:
        model_class = models[name]
    except KeyError as exc:
        raise ValueError(f"Unknown three-domain model: {name}") from exc
    return model_class(
        config,
        item_text_embeddings,
        item_domains,
        train_seen_users,
        train_seen_items,
    )
