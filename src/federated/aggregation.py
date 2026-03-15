from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class BaseAggregator(ABC):
    @abstractmethod
    def aggregate(
        self,
        global_model: nn.Module,
        client_models: list[nn.Module],
        client_weights: list[int],
    ) -> nn.Module:
        """Aggregate client models into the global model."""
        pass


class FedAvgAggregator(BaseAggregator):
    """Federated Averaging (McMahan et al., 2017).

    Computes a weighted average of client model parameters proportional
    to each client's number of training samples.
    """

    def aggregate(
        self,
        global_model: nn.Module,
        client_models: list[nn.Module],
        client_weights: list[int],
    ) -> nn.Module:
        assert len(client_models) == len(client_weights)
        assert len(client_models) > 0

        total_weight = sum(client_weights)
        global_state = global_model.state_dict()

        # Weighted average
        for key in global_state:
            if not global_state[key].is_floating_point():
                # Keep integer buffers (e.g., num_batches_tracked) from first client
                global_state[key] = client_models[0].state_dict()[key]
                continue

            global_state[key] = sum(
                client_models[i].state_dict()[key] * (client_weights[i] / total_weight)
                for i in range(len(client_models))
            )

        global_model.load_state_dict(global_state)
        return global_model


class FedProxAggregator(FedAvgAggregator):
    """FedProx (Li et al., 2020) — FedAvg aggregation with a proximal term.

    The proximal term is applied during local training (not aggregation).
    This aggregator is identical to FedAvgAggregator; the mu parameter is
    passed to the client trainer to add the proximal loss during local steps.
    """

    def __init__(self, mu: float = 0.01):
        self.mu = mu


class FedNovaAggregator(BaseAggregator):
    """FedNova (Wang et al., 2020) — Federated Normalized Averaging.

    Corrects for objective inconsistency caused by varying numbers of local
    steps across clients by normalizing each client's pseudo-gradient by its
    effective local step count before aggregation.

    Update rule:
        d_k = w^r - w_k^r               (local pseudo-gradient)
        τ_eff = Σ_k (n_k/n) * τ_k       (weighted effective steps)
        a_k   = τ_k / τ_eff             (per-client normalization factor)
        w^{r+1} = w^r - Σ_k (n_k/n) * a_k * d_k
    """

    def aggregate(
        self,
        global_model: nn.Module,
        client_models: list[nn.Module],
        client_weights: list[int],
        client_local_steps: list[int] | None = None,
    ) -> nn.Module:
        assert len(client_models) == len(client_weights)
        assert len(client_models) > 0

        if client_local_steps is None:
            # Fall back to FedAvg if step counts are unavailable
            client_local_steps = [1] * len(client_models)

        total_weight = sum(client_weights)
        fractions = [w / total_weight for w in client_weights]

        # Effective step count τ_eff = Σ_k (n_k/n) * τ_k
        tau_eff = sum(f * tau for f, tau in zip(fractions, client_local_steps))
        if tau_eff == 0:
            tau_eff = 1.0

        global_state = global_model.state_dict()
        global_params_copy = {k: v.clone() for k, v in global_state.items()}

        for key in global_state:
            if not global_state[key].is_floating_point():
                global_state[key] = client_models[0].state_dict()[key]
                continue

            # Aggregate normalized pseudo-gradients: Σ_k (n_k/n) * a_k * d_k
            aggregated_delta = torch.zeros_like(global_params_copy[key])
            for i, (model, frac, tau_k) in enumerate(
                zip(client_models, fractions, client_local_steps)
            ):
                a_k = tau_k / tau_eff
                d_k = global_params_copy[key] - model.state_dict()[key]
                aggregated_delta += frac * a_k * d_k

            global_state[key] = global_params_copy[key] - aggregated_delta

        global_model.load_state_dict(global_state)
        return global_model


class FedPerAggregator(BaseAggregator):
    """FedPer (Arivazhagan et al., 2019) — Personalized Federated Learning.

    Only aggregates the shared encoder layers; decoder layers remain local
    to each client, enabling personalization of the task-specific head.

    For this UNet implementation the state-dict keys follow:
        encoder.levels.X.*   → aggregated (shared representation)
        decoder.*            → kept local (personalized head)
    """

    def __init__(self, shared_prefix: str = "encoder."):
        self.shared_prefix = shared_prefix

    def aggregate(
        self,
        global_model: nn.Module,
        client_models: list[nn.Module],
        client_weights: list[int],
    ) -> nn.Module:
        assert len(client_models) == len(client_weights)
        assert len(client_models) > 0

        total_weight = sum(client_weights)
        global_state = global_model.state_dict()

        for key in global_state:
            # Only aggregate shared (encoder) parameters
            if not key.startswith(self.shared_prefix):
                continue

            if not global_state[key].is_floating_point():
                global_state[key] = client_models[0].state_dict()[key]
                continue

            global_state[key] = sum(
                client_models[i].state_dict()[key] * (client_weights[i] / total_weight)
                for i in range(len(client_models))
            )

        global_model.load_state_dict(global_state)
        return global_model
