#!/usr/bin/env python3
"""Minimal Conclusion-Expert LoRA warm-up for the upcycled Qwen model."""

import argparse
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import load_upcycled_qwen, parameter_counts, replace_qwen_mlp_with_moe
from src.training import inject_conclusion_lora, merge_conclusion_lora


class TextChunks(Dataset):
    def __init__(self, path: str, tokenizer, sequence_length: int) -> None:
        # 当前先把整个 txt 当作连续语言模型文本切块；正式 SFT 时应单独屏蔽 prompt labels。
        text = Path(path).read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Training file is empty: {path}")
        token_ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        self.pad_id = tokenizer.pad_token_id
        self.chunks = [
            token_ids[start : start + sequence_length]
            for start in range(0, len(token_ids), sequence_length)
        ]
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int):
        chunk = self.chunks[index]
        padding = self.sequence_length - len(chunk)
        input_ids = torch.tensor(chunk + [self.pad_id] * padding, dtype=torch.long)
        attention_mask = torch.tensor([1] * len(chunk) + [0] * padding, dtype=torch.long)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 Minimal DeepSeekMoE 的 Conclusion Expert LoRA")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data", default=str(PROJECT_ROOT.parent / "attention_linearized.txt"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "conclusion-lora"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_model(source: str, dtype: torch.dtype, local_files_only: bool):
    config = AutoConfig.from_pretrained(source, local_files_only=local_files_only)
    if getattr(config, "minimal_deepseek_moe", None):
        return load_upcycled_qwen(source, torch_dtype=dtype, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        source,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    return replace_qwen_mlp_with_moe(model)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA GPU；本训练脚本仅支持 NVIDIA CUDA。")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"设备: {torch.cuda.get_device_name(0)}，模型精度: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(args.model, dtype, args.local_files_only)
    model.config.use_cache = False
    # 梯度检查点用额外计算换取更低的显存占用，单卡训练建议开启。
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # 冻结原模型，只给 6 个 Conclusion Experts 注入低秩可训练参数。
    inject_conclusion_lora(model, rank=args.rank, alpha=args.alpha)
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    model.train()

    dataset = TextChunks(args.data, tokenizer, args.sequence_length)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    updates_per_epoch = math.ceil(len(loader) / args.gradient_accumulation)
    print({**parameter_counts(model), "chunks": len(dataset), "updates_per_epoch": updates_per_epoch})

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    for epoch in range(args.epochs):
        for micro_step, batch in enumerate(loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            # CUDA autocast 自动选择稳定的 BF16/FP16 混合精度算子。
            with torch.autocast(device_type="cuda", dtype=dtype):
                loss = model(**batch).loss / args.gradient_accumulation
            loss.backward()
            if micro_step % args.gradient_accumulation == 0 or micro_step == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                print(f"epoch={epoch + 1} step={global_step} loss={loss.item() * args.gradient_accumulation:.4f}")

    # 将 LoRA 增量合并回 Conclusion Expert，保存后无需额外适配器代码。
    merge_conclusion_lora(model)
    model.eval()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(output)
    print(f"Saved merged MoE checkpoint to {output}")


if __name__ == "__main__":
    main()
