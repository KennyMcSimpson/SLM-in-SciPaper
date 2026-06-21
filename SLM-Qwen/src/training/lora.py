"""Small dependency-free LoRA implementation for Conclusion Experts only."""

import math
from typing import List

import torch
from torch import Tensor, nn

from ..model.moe_mlp import MinimalDeepSeekMoE


class LoRALinear(nn.Module):
    """Frozen linear layer plus a trainable low-rank update."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        base.weight.requires_grad = False

    def forward(self, inputs: Tensor) -> Tensor:
        update = torch.nn.functional.linear(
            torch.nn.functional.linear(inputs, self.lora_a),
            self.lora_b,
        )
        return self.base(inputs) + update * self.scaling

    def merge(self) -> nn.Linear:
        with torch.no_grad():
            self.base.weight.add_(self.lora_b @ self.lora_a, alpha=self.scaling)
        return self.base


def inject_conclusion_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0) -> List[nn.Parameter]:
    """Freeze the model and add LoRA to every converted Conclusion Expert."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable: List[nn.Parameter] = []
    moe_count = 0
    for module in model.modules():
        if not isinstance(module, MinimalDeepSeekMoE):
            continue
        moe_count += 1
        module.set_forced_expert(MinimalDeepSeekMoE.CONCLUSION_EXPERT)
        expert = module.conclusion_expert
        for name in ("gate_proj", "up_proj", "down_proj"):
            projection = getattr(expert, name)
            if isinstance(projection, LoRALinear):
                raise ValueError("Conclusion LoRA has already been injected.")
            wrapped = LoRALinear(projection, rank=rank, alpha=alpha)
            setattr(expert, name, wrapped)
            trainable.extend((wrapped.lora_a, wrapped.lora_b))
    if moe_count == 0:
        raise ValueError("No MinimalDeepSeekMoE layers found.")
    return trainable


def merge_conclusion_lora(model: nn.Module) -> None:
    """Merge adapters into expert weights so normal MoE checkpoints can be saved."""
    for module in model.modules():
        if not isinstance(module, MinimalDeepSeekMoE):
            continue
        expert = module.conclusion_expert
        for name in ("gate_proj", "up_proj", "down_proj"):
            projection = getattr(expert, name)
            if isinstance(projection, LoRALinear):
                setattr(expert, name, projection.merge())
        module.set_forced_expert(None)
