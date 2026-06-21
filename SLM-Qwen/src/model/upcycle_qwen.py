"""Conversion, parameter accounting, and checkpoint reload helpers."""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any, Dict, Iterable, Optional, Union

import torch
from torch import nn

from .config import MOE_CONFIG_KEY, MOE_LAYER_INDICES, MoEUpcycleConfig
from .moe_mlp import MinimalDeepSeekMoE


def replace_qwen_mlp_with_moe(
    model: nn.Module,
    layer_indices: Iterable[int] = MOE_LAYER_INDICES,
    *,
    force_general: bool = False,
    upcycle_config: Optional[MoEUpcycleConfig] = None,
) -> nn.Module:
    """Replace exactly the six approved Qwen decoder MLPs in-place."""
    indices = tuple(layer_indices)
    if indices != MOE_LAYER_INDICES:
        raise ValueError(
            f"Only layers {list(MOE_LAYER_INDICES)} may be replaced; got {list(indices)}."
        )
    settings = upcycle_config or MoEUpcycleConfig(layer_indices=indices)
    layers = _decoder_layers(model)
    if len(layers) <= max(indices):
        raise ValueError(f"Model has {len(layers)} layers; layer {max(indices)} does not exist.")

    for index in indices:
        old_mlp = layers[index].mlp
        if isinstance(old_mlp, MinimalDeepSeekMoE):
            old_mlp.set_forced_expert(
                MinimalDeepSeekMoE.GENERAL_EXPERT if force_general else None
            )
            continue
        new_moe = MinimalDeepSeekMoE.from_dense(
            old_mlp,
            router_general_bias=settings.router_general_bias,
            router_conclusion_bias=settings.router_conclusion_bias,
        )
        new_moe.set_forced_expert(
            MinimalDeepSeekMoE.GENERAL_EXPERT if force_general else None
        )
        layers[index].mlp = new_moe

    if hasattr(model, "config"):
        setattr(model.config, MOE_CONFIG_KEY, settings.to_dict())
    return model


def set_force_general(model: nn.Module, enabled: bool = True) -> None:
    """Force all converted layers through General for equivalence checks."""
    found = 0
    for module in model.modules():
        if isinstance(module, MinimalDeepSeekMoE):
            module.set_forced_expert(MinimalDeepSeekMoE.GENERAL_EXPERT if enabled else None)
            found += 1
    if found != len(MOE_LAYER_INDICES):
        raise ValueError(f"Expected {len(MOE_LAYER_INDICES)} MoE layers, found {found}.")


def parameter_counts(model: nn.Module) -> Dict[str, int]:
    """Return total and trainable scalar parameter counts."""
    return {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def load_upcycled_qwen(
    checkpoint: Union[str, os.PathLike],
    **from_pretrained_kwargs: Any,
) -> nn.Module:
    """Reload a converted checkpoint using its saved MoE config metadata.

    The checkpoint config is read first, a Qwen skeleton is instantiated without
    loading the incompatible dense weights, its target MLPs are converted, and
    only then is the complete MoE state dict loaded.
    """
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.modeling_utils import load_sharded_checkpoint
        from transformers.utils import (
            SAFE_WEIGHTS_INDEX_NAME,
            SAFE_WEIGHTS_NAME,
            WEIGHTS_INDEX_NAME,
            WEIGHTS_NAME,
        )
    except ImportError as exc:
        raise ImportError("transformers is required to reload a Qwen checkpoint.") from exc

    config = AutoConfig.from_pretrained(checkpoint, **_hub_kwargs(from_pretrained_kwargs))
    metadata = getattr(config, MOE_CONFIG_KEY, None)
    if metadata is None:
        raise ValueError(f"Checkpoint config has no {MOE_CONFIG_KEY!r} metadata.")
    settings = MoEUpcycleConfig.from_dict(metadata)
    requested_dtype = from_pretrained_kwargs.get("torch_dtype", "auto")
    if requested_dtype == "auto":
        requested_dtype = getattr(config, "torch_dtype", None)
    model_kwargs = {
        "trust_remote_code": from_pretrained_kwargs.get("trust_remote_code", False),
    }
    if requested_dtype is not None:
        model_kwargs["torch_dtype"] = requested_dtype
    context = torch.device("meta") if from_pretrained_kwargs.get("low_cpu_mem_usage") else nullcontext()
    with context:
        model = AutoModelForCausalLM.from_config(
            config,
            **model_kwargs,
        )
    # Materialize a normal CPU skeleton if meta construction was requested.
    if any(parameter.is_meta for parameter in model.parameters()):
        model.to_empty(device="cpu")
    replace_qwen_mlp_with_moe(model, upcycle_config=settings)

    checkpoint_path = os.fspath(checkpoint)
    if not os.path.isdir(checkpoint_path):
        raise ValueError("load_upcycled_qwen currently requires a local save_pretrained directory.")
    safe_file = os.path.join(checkpoint_path, SAFE_WEIGHTS_NAME)
    torch_file = os.path.join(checkpoint_path, WEIGHTS_NAME)
    safe_index = os.path.join(checkpoint_path, SAFE_WEIGHTS_INDEX_NAME)
    torch_index = os.path.join(checkpoint_path, WEIGHTS_INDEX_NAME)
    if os.path.isfile(safe_index) or os.path.isfile(torch_index):
        load_sharded_checkpoint(model, checkpoint_path, strict=True, prefer_safe=True)
    elif os.path.isfile(safe_file):
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("safetensors is required to load this checkpoint.") from exc
        model.load_state_dict(load_file(safe_file), strict=True)
    elif os.path.isfile(torch_file):
        try:
            state_dict = torch.load(torch_file, map_location="cpu", weights_only=True)
        except TypeError:  # torch < 2.0
            state_dict = torch.load(torch_file, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    else:
        raise FileNotFoundError(f"No model weights found in {checkpoint_path}.")
    model.tie_weights()
    model.eval()
    return model


def _decoder_layers(model: nn.Module):
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise TypeError("Expected a Qwen causal LM exposing model.layers.")
    return layers


def _hub_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    allowed = (
        "cache_dir",
        "force_download",
        "local_files_only",
        "revision",
        "token",
        "trust_remote_code",
    )
    return {key: kwargs[key] for key in allowed if key in kwargs}
