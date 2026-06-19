from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from ske.data.bow_vocab import BowVocabulary
from ske.data.dataset import BIO_LABELS, FeatureConfig, KeyphraseDataset, collate_features
from ske.modeling import ModelConfig, ScientificKeyphraseExtractor, move_batch_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SEG-style scientific keyphrase extractor.")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--dev_jsonl", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--init_checkpoint", default=None, help="Load a previous ScientificKeyphraseExtractor checkpoint for staged fine-tuning.")
    parser.add_argument("--bow_csv", default=None)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_sentences", type=int, default=48)
    parser.add_argument("--max_candidate_sentences", type=int, default=160)
    parser.add_argument("--disable_evidence_packing", action="store_true")
    parser.add_argument("--lead_sentences", type=int, default=4)
    parser.add_argument("--neighbor_sentences", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_train_records", type=int, default=None)
    parser.add_argument("--max_dev_records", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--selector_loss_weight", type=float, default=0.4)
    parser.add_argument("--boundary_loss_weight", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)
    bow_vocab = BowVocabulary.load(args.bow_csv)
    tokenizer_source = args.init_checkpoint or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    feature_config = FeatureConfig(
        max_seq_length=args.max_seq_length,
        max_sentences=args.max_sentences,
        max_candidate_sentences=args.max_candidate_sentences,
        use_evidence_packing=not args.disable_evidence_packing,
        lead_sentences=args.lead_sentences,
        neighbor_sentences=args.neighbor_sentences,
    )
    train_dataset = KeyphraseDataset(
        args.train_jsonl,
        tokenizer,
        feature_config,
        bow_vocab=bow_vocab,
        max_records=args.max_train_records,
    )
    dev_dataset = (
        KeyphraseDataset(
            args.dev_jsonl,
            tokenizer,
            feature_config,
            bow_vocab=bow_vocab,
            max_records=args.max_dev_records,
        )
        if args.dev_jsonl
        else None
    )
    collate = partial(collate_features, pad_token_id=tokenizer.pad_token_id or 0)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate) if dev_dataset else None

    if args.init_checkpoint:
        model = ScientificKeyphraseExtractor.load(args.init_checkpoint, map_location=device)
        model.config.selector_loss_weight = args.selector_loss_weight
        model.config.boundary_loss_weight = args.boundary_loss_weight
    else:
        model = ScientificKeyphraseExtractor(
            ModelConfig(
                model_name=args.model_name,
                selector_loss_weight=args.selector_loss_weight,
                boundary_loss_weight=args.boundary_loss_weight,
            )
        )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = max(len(train_loader) * args.epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, args.amp)
        row = {"epoch": epoch, **{f"train_{key}": value for key, value in train_metrics.items()}}
        if dev_loader is not None:
            dev_metrics = evaluate(model, dev_loader, device)
            row.update({f"dev_{key}": value for key, value in dev_metrics.items()})
            if dev_metrics["boundary_f1"] > best_score:
                best_score = dev_metrics["boundary_f1"]
                save_checkpoint(model, tokenizer, feature_config, output_dir)
        else:
            save_checkpoint(model, tokenizer, feature_config, output_dir)
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
    (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "train_args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")


def save_checkpoint(model: ScientificKeyphraseExtractor, tokenizer: AutoTokenizer, feature_config: FeatureConfig, output_dir: Path) -> None:
    model.save(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "feature_config.json").write_text(json.dumps(feature_config.__dict__, indent=2), encoding="utf-8")


def train_one_epoch(
    model: ScientificKeyphraseExtractor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: bool,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    steps = 0
    progress = tqdm(loader, desc="train")
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                cls_positions=batch["cls_positions"],
                sentence_mask=batch["sentence_mask"],
                sentence_labels=batch["sentence_labels"],
                bio_labels=batch["bio_labels"],
            )
            loss = outputs["loss"]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.item())
        steps += 1
        progress.set_postfix(loss=total_loss / max(steps, 1))
    return {"loss": total_loss / max(steps, 1)}


@torch.no_grad()
def evaluate(model: ScientificKeyphraseExtractor, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    steps = 0
    boundary_tp = boundary_fp = boundary_fn = 0
    selector_tp = selector_fp = selector_fn = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            cls_positions=batch["cls_positions"],
            sentence_mask=batch["sentence_mask"],
            sentence_labels=batch["sentence_labels"],
            bio_labels=batch["bio_labels"],
        )
        total_loss += float(outputs["loss"].item())
        steps += 1
        preds = outputs["boundary_logits"].argmax(dim=-1)
        gold = batch["bio_labels"]
        valid = gold != -100
        pred_pos = (preds != BIO_LABELS["O"]) & valid
        gold_pos = (gold != BIO_LABELS["O"]) & valid
        boundary_tp += int((pred_pos & gold_pos).sum().item())
        boundary_fp += int((pred_pos & ~gold_pos).sum().item())
        boundary_fn += int((~pred_pos & gold_pos).sum().item())
        selector_pred = (outputs["sentence_probs"] >= 0.5) & batch["sentence_mask"]
        selector_gold = (batch["sentence_labels"] >= 0.5) & batch["sentence_mask"]
        selector_tp += int((selector_pred & selector_gold).sum().item())
        selector_fp += int((selector_pred & ~selector_gold).sum().item())
        selector_fn += int((~selector_pred & selector_gold).sum().item())
    return {
        "loss": total_loss / max(steps, 1),
        "boundary_precision": _precision(boundary_tp, boundary_fp),
        "boundary_recall": _recall(boundary_tp, boundary_fn),
        "boundary_f1": _f1(boundary_tp, boundary_fp, boundary_fn),
        "selector_f1": _f1(selector_tp, selector_fp, selector_fn),
    }


def _precision(tp: int, fp: int) -> float:
    return tp / max(tp + fp, 1)


def _recall(tp: int, fn: int) -> float:
    return tp / max(tp + fn, 1)


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = _precision(tp, fp)
    recall = _recall(tp, fn)
    return 2 * precision * recall / max(precision + recall, 1e-12)


if __name__ == "__main__":
    main()
