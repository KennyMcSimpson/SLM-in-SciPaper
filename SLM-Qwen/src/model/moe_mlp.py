"""Minimal DeepSeekMoE-style MLP used to upcycle Qwen2 dense layers."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class HalfSizeQwenMLP(nn.Module):
    """A bias-free SwiGLU Qwen MLP with half the dense intermediate width."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
    ) -> None:
        super().__init__()
        if hidden_act not in ("silu", "swish"):
            raise ValueError(f"Only Qwen's SiLU activation is supported, got {hidden_act!r}.")
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))

    @classmethod
    def from_dense_slice(
        cls,
        dense_mlp: nn.Module,
        start: int,
        end: int,
    ) -> "HalfSizeQwenMLP":
        """Copy intermediate neurons ``[start:end]`` from a Qwen dense MLP."""
        _validate_dense_mlp(dense_mlp)
        dense_width, hidden_size = dense_mlp.gate_proj.weight.shape
        if not 0 <= start < end <= dense_width:
            raise ValueError(f"Invalid dense MLP slice [{start}:{end}] for width {dense_width}.")
        expert = cls(hidden_size, end - start)
        expert.to(device=dense_mlp.gate_proj.weight.device, dtype=dense_mlp.gate_proj.weight.dtype)
        with torch.no_grad():
            expert.gate_proj.weight.copy_(dense_mlp.gate_proj.weight[start:end])
            expert.up_proj.weight.copy_(dense_mlp.up_proj.weight[start:end])
            expert.down_proj.weight.copy_(dense_mlp.down_proj.weight[:, start:end])
        expert.train(dense_mlp.training)
        return expert


class MinimalDeepSeekMoE(nn.Module):
    """One shared expert plus top-1 routing over General and Conclusion experts."""

    GENERAL_EXPERT = 0
    CONCLUSION_EXPERT = 1

    def __init__(
        self,
        shared_expert: HalfSizeQwenMLP,
        general_expert: HalfSizeQwenMLP,
        conclusion_expert: HalfSizeQwenMLP,
        router_general_bias: float = 2.0,
        router_conclusion_bias: float = -2.0,
    ) -> None:
        super().__init__()
        widths = {
            (expert.hidden_size, expert.intermediate_size)
            for expert in (shared_expert, general_expert, conclusion_expert)
        }
        if len(widths) != 1:
            raise ValueError("All experts must have identical hidden and intermediate sizes.")
        self.shared_expert = shared_expert
        self.general_expert = general_expert
        self.conclusion_expert = conclusion_expert
        self.router = nn.Linear(shared_expert.hidden_size, 2, bias=True)
        self.force_expert: Optional[int] = None
        self.last_router_logits: Optional[Tensor] = None
        self.last_route_ids: Optional[Tensor] = None
        self.reset_router(router_general_bias, router_conclusion_bias)

    def reset_router(self, general_bias: float = 2.0, conclusion_bias: float = -2.0) -> None:
        with torch.no_grad():
            self.router.weight.zero_()
            self.router.bias.copy_(self.router.bias.new_tensor([general_bias, conclusion_bias]))

    def set_forced_expert(self, expert: Optional[int]) -> None:
        if expert not in (None, self.GENERAL_EXPERT, self.CONCLUSION_EXPERT):
            raise ValueError("expert must be None, GENERAL_EXPERT (0), or CONCLUSION_EXPERT (1).")
        self.force_expert = expert

    def forward(self, hidden_states: Tensor) -> Tensor:
        router_logits = self.router(hidden_states)
        if self.force_expert is None:
            route_ids = router_logits.argmax(dim=-1)
        else:
            route_ids = torch.full(
                hidden_states.shape[:-1],
                self.force_expert,
                device=hidden_states.device,
                dtype=torch.long,
            )

        shared_output = self.shared_expert(hidden_states)
        # Evaluate only tokens assigned to each routed expert.
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_routes = route_ids.reshape(-1)
        flat_output = torch.empty_like(flat_hidden)
        for expert_id, expert in enumerate((self.general_expert, self.conclusion_expert)):
            mask = flat_routes == expert_id
            if mask.any():
                flat_output[mask] = expert(flat_hidden[mask])

        self.last_router_logits = router_logits
        self.last_route_ids = route_ids
        return shared_output + flat_output.view_as(hidden_states)

    @classmethod
    def from_dense(
        cls,
        dense_mlp: nn.Module,
        router_general_bias: float = 2.0,
        router_conclusion_bias: float = -2.0,
    ) -> "MinimalDeepSeekMoE":
        """Split a dense Qwen MLP into equal Shared and General halves."""
        _validate_dense_mlp(dense_mlp)
        dense_width = dense_mlp.gate_proj.weight.shape[0]
        if dense_width % 2:
            raise ValueError(f"Dense intermediate size must be even, got {dense_width}.")
        half = dense_width // 2
        shared = HalfSizeQwenMLP.from_dense_slice(dense_mlp, 0, half)
        general = HalfSizeQwenMLP.from_dense_slice(dense_mlp, half, dense_width)
        conclusion = HalfSizeQwenMLP.from_dense_slice(dense_mlp, half, dense_width)
        moe = cls(
            shared,
            general,
            conclusion,
            router_general_bias,
            router_conclusion_bias,
        )
        moe.to(device=dense_mlp.gate_proj.weight.device, dtype=dense_mlp.gate_proj.weight.dtype)
        moe.train(dense_mlp.training)
        return moe


def _validate_dense_mlp(dense_mlp: nn.Module) -> None:
    for name in ("gate_proj", "up_proj", "down_proj"):
        projection = getattr(dense_mlp, name, None)
        if not isinstance(projection, nn.Linear):
            raise TypeError(f"dense_mlp.{name} must be torch.nn.Linear.")
        if projection.bias is not None:
            raise ValueError(f"dense_mlp.{name} must be bias-free like Qwen2MLP.")
    gate_shape = dense_mlp.gate_proj.weight.shape
    if dense_mlp.up_proj.weight.shape != gate_shape:
        raise ValueError("gate_proj and up_proj shapes differ.")
    if dense_mlp.down_proj.weight.shape != (gate_shape[1], gate_shape[0]):
        raise ValueError("down_proj shape is incompatible with gate_proj.")
