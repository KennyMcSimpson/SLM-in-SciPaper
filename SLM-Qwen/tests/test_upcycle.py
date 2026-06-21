import tempfile

import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.model import (
    MOE_LAYER_INDICES,
    MinimalDeepSeekMoE,
    load_upcycled_qwen,
    parameter_counts,
    replace_qwen_mlp_with_moe,
)
from src.training import inject_conclusion_lora, merge_conclusion_lora


def tiny_qwen():
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=36,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    return Qwen2ForCausalLM(config).eval()


def test_dense_split_is_exact_when_general_is_forced():
    model = tiny_qwen()
    dense = model.model.layers[24].mlp
    hidden = torch.randn(2, 5, model.config.hidden_size)
    expected = dense(hidden)

    moe = MinimalDeepSeekMoE.from_dense(dense)
    moe.set_forced_expert(MinimalDeepSeekMoE.GENERAL_EXPERT)
    actual = moe(hidden)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    half = model.config.intermediate_size // 2
    torch.testing.assert_close(moe.shared_expert.gate_proj.weight, dense.gate_proj.weight[:half])
    torch.testing.assert_close(moe.general_expert.up_proj.weight, dense.up_proj.weight[half:])
    torch.testing.assert_close(moe.general_expert.down_proj.weight, dense.down_proj.weight[:, half:])
    for key, value in moe.general_expert.state_dict().items():
        torch.testing.assert_close(value, moe.conclusion_expert.state_dict()[key])


def test_replaces_only_approved_layers_and_preserves_logits():
    torch.manual_seed(7)
    model = tiny_qwen()
    dense_types = [type(layer.mlp) for layer in model.model.layers]
    input_ids = torch.randint(0, model.config.vocab_size, (2, 7))
    with torch.no_grad():
        expected = model(input_ids).logits

    replace_qwen_mlp_with_moe(model, force_general=True)
    replaced = tuple(
        index
        for index, layer in enumerate(model.model.layers)
        if isinstance(layer.mlp, MinimalDeepSeekMoE)
    )
    assert replaced == MOE_LAYER_INDICES
    for index, layer in enumerate(model.model.layers):
        if index not in MOE_LAYER_INDICES:
            assert type(layer.mlp) is dense_types[index]

    with torch.no_grad():
        actual = model(input_ids).logits
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_parameter_counts_and_save_reload_roundtrip():
    torch.manual_seed(11)
    model = tiny_qwen()
    replace_qwen_mlp_with_moe(model, force_general=True)
    first_parameter = next(model.parameters())
    first_parameter.requires_grad = False
    counts = parameter_counts(model)
    assert counts["total_parameters"] == sum(p.numel() for p in model.parameters())
    assert counts["trainable_parameters"] == counts["total_parameters"] - first_parameter.numel()

    input_ids = torch.randint(0, model.config.vocab_size, (1, 6))
    with torch.no_grad():
        expected = model(input_ids).logits
    with tempfile.TemporaryDirectory() as checkpoint:
        model.save_pretrained(checkpoint)
        restored = load_upcycled_qwen(checkpoint)
        replaced = sum(isinstance(layer.mlp, MinimalDeepSeekMoE) for layer in restored.model.layers)
        assert replaced == len(MOE_LAYER_INDICES)
        assert restored.dtype == model.dtype
        # Runtime force flags are intentionally not serialized; restore the check mode.
        for layer in restored.model.layers:
            if isinstance(layer.mlp, MinimalDeepSeekMoE):
                layer.mlp.set_forced_expert(MinimalDeepSeekMoE.GENERAL_EXPERT)
        with torch.no_grad():
            actual = restored(input_ids).logits
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_rejects_any_other_layer_set():
    model = tiny_qwen()
    try:
        replace_qwen_mlp_with_moe(model, layer_indices=(24,))
    except ValueError as error:
        assert "Only layers" in str(error)
    else:
        raise AssertionError("An unsupported layer set was accepted.")


def test_conclusion_lora_is_small_and_mergeable():
    model = tiny_qwen()
    replace_qwen_mlp_with_moe(model)
    trainable = inject_conclusion_lora(model, rank=2, alpha=4)
    assert trainable
    assert all(parameter.requires_grad for parameter in trainable)
    assert sum(parameter.numel() for parameter in trainable) < parameter_counts(model)["total_parameters"]

    input_ids = torch.randint(0, model.config.vocab_size, (1, 5))
    before = model(input_ids).logits
    before.sum().backward()
    assert any(parameter.grad is not None for parameter in trainable)
    before = before.detach()
    merge_conclusion_lora(model)
    with torch.no_grad():
        after = model(input_ids).logits
    torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-6)
