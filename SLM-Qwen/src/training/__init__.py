"""Minimal training helpers for SLM-Qwen."""

from .lora import LoRALinear, inject_conclusion_lora, merge_conclusion_lora

__all__ = ["LoRALinear", "inject_conclusion_lora", "merge_conclusion_lora"]
