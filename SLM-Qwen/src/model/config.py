"""Configuration constants for dense-to-MoE Qwen upcycling."""

from dataclasses import dataclass
from typing import Any, Dict, Tuple


MOE_LAYER_INDICES: Tuple[int, ...] = (24, 26, 28, 30, 32, 34)
MOE_CONFIG_KEY = "minimal_deepseek_moe"


@dataclass(frozen=True)
class MoEUpcycleConfig:
    """Serializable metadata needed to reconstruct the converted model."""

    layer_indices: Tuple[int, ...] = MOE_LAYER_INDICES
    num_routed_experts: int = 2
    router_general_bias: float = 2.0
    router_conclusion_bias: float = -2.0

    def __post_init__(self) -> None:
        if tuple(self.layer_indices) != MOE_LAYER_INDICES:
            raise ValueError(
                "This project only supports MoE layers "
                f"{list(MOE_LAYER_INDICES)}, got {list(self.layer_indices)}."
            )
        if self.num_routed_experts != 2:
            raise ValueError("MinimalDeepSeekMoE requires exactly two routed experts.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_indices": list(self.layer_indices),
            "num_routed_experts": self.num_routed_experts,
            "router_general_bias": self.router_general_bias,
            "router_conclusion_bias": self.router_conclusion_bias,
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "MoEUpcycleConfig":
        return cls(
            layer_indices=tuple(value.get("layer_indices", MOE_LAYER_INDICES)),
            num_routed_experts=int(value.get("num_routed_experts", 2)),
            router_general_bias=float(value.get("router_general_bias", 2.0)),
            router_conclusion_bias=float(value.get("router_conclusion_bias", -2.0)),
        )
