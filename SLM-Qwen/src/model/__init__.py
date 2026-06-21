"""Public APIs for Minimal DeepSeekMoE Qwen conversion."""

from .config import MOE_LAYER_INDICES, MoEUpcycleConfig
from .moe_mlp import HalfSizeQwenMLP, MinimalDeepSeekMoE
from .upcycle_qwen import (
    load_upcycled_qwen,
    parameter_counts,
    replace_qwen_mlp_with_moe,
    set_force_general,
)

__all__ = [
    "HalfSizeQwenMLP",
    "MinimalDeepSeekMoE",
    "MOE_LAYER_INDICES",
    "MoEUpcycleConfig",
    "load_upcycled_qwen",
    "parameter_counts",
    "replace_qwen_mlp_with_moe",
    "set_force_general",
]
