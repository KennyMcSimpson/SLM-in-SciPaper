from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ske.structure.data import (
    StructureFeatureConfig,
    StructuredSentenceDataset,
    collate_structure_features,
)
from ske.structure.modeling import StructuredModelConfig, StructuredPaperModel, move_structure_batch_to_device
from ske.structure.schema import ROLE_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the structure-aware paper understanding heads.")
    parser.add_argument("--train_jsonl", nargs="+", required=True)
    parser.add_argument("--dev_jsonl", nargs="*", default=[])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--init_encoder_checkpoint", default=None)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_sentences", type=int, default=48)
    parser.add_argument("--max_candidate_sentences", type=int, default=180)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_train_records", type=int, default=None)
    parser.add_argument("--max_dev_records", type=int, default=None)
    parser.add_argument("--target_train_records_per_dataset", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--loss_weighting", choices=["fixed", "uncertainty"], default="uncertainty")
    parser.add_argument("--role_loss_weight", type=float, default=1.0)
    parser.add_argument("--evidence_loss_weight", type=float, default=0.7)
    parser.add_argument("--evidence_pos_weight", type=float, default=0.0, help="Positive-class weight for evidence BCE. Use 0 for auto.")
    parser.add_argument("--max_auto_evidence_pos_weight", type=float, default=24.0)
    parser.add_argument("--importance_loss_weight", type=float, default=0.35)
    parser.add_argument("--importance_pos_weight", type=float, default=0.0, help="Positive-class weight for importance BCE. Use 0 for auto.")
    parser.add_argument("--max_auto_importance_pos_weight", type=float, default=24.0)
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

    tokenizer_source = args.init_encoder_checkpoint or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    feature_config = StructureFeatureConfig(
        max_seq_length=args.max_seq_length,
        max_sentences=args.max_sentences,
        max_candidate_sentences=args.max_candidate_sentences,
    )
    train_dataset = concat_datasets(
        args.train_jsonl,
        tokenizer,
        feature_config,
        args.max_train_records,
        target_records_per_dataset=args.target_train_records_per_dataset,
    )
    dev_dataset = concat_datasets(args.dev_jsonl, tokenizer, feature_config, args.max_dev_records) if args.dev_jsonl else None
    collate = partial(collate_structure_features, pad_token_id=tokenizer.pad_token_id or 0)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate) if dev_dataset else None
    evidence_stats = evidence_counts(train_dataset)
    evidence_pos_weight = args.evidence_pos_weight
    if evidence_pos_weight <= 0:
        evidence_pos_weight = auto_evidence_pos_weight(evidence_stats, args.max_auto_evidence_pos_weight)
    importance_stats = importance_counts(train_dataset)
    importance_pos_weight = args.importance_pos_weight
    if importance_pos_weight <= 0:
        importance_pos_weight = auto_evidence_pos_weight(importance_stats, args.max_auto_importance_pos_weight)
    print(
        json.dumps(
            {
                "event": "label_weights",
                "loss_weighting": args.loss_weighting,
                "evidence": {
                    "positive": evidence_stats["positive"],
                    "negative": evidence_stats["negative"],
                    "pos_weight": evidence_pos_weight,
                },
                "importance": {
                    "positive": importance_stats["positive"],
                    "negative": importance_stats["negative"],
                    "pos_weight": importance_pos_weight,
                },
            },
            ensure_ascii=False,
        )
    )

    model = StructuredPaperModel(
        StructuredModelConfig(
            model_name=args.model_name,
            loss_weighting=args.loss_weighting,
            role_loss_weight=args.role_loss_weight,
            evidence_loss_weight=args.evidence_loss_weight,
            evidence_pos_weight=evidence_pos_weight,
            importance_loss_weight=args.importance_loss_weight,
            importance_pos_weight=importance_pos_weight,
        )
    )
    init_report: dict[str, Any] | None = None
    if args.init_encoder_checkpoint:
        init_report = model.init_encoder_from_keyphrase_checkpoint(args.init_encoder_checkpoint, map_location="cpu")
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
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, args.amp)
        row: dict[str, Any] = {"epoch": epoch, **{f"train_{key}": value for key, value in train_metrics.items()}}
        if dev_loader is not None:
            dev_metrics = evaluate(model, dev_loader, device)
            row.update({f"dev_{key}": value for key, value in dev_metrics.items()})
            score = dev_metrics.get("role_f1", 0.0) + dev_metrics.get("evidence_best_f1", 0.0) + dev_metrics.get("importance_best_f1", 0.0)
            if score > best_score:
                best_score = score
                save_checkpoint(model, tokenizer, feature_config, output_dir)
        else:
            save_checkpoint(model, tokenizer, feature_config, output_dir)
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

    (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    run_args = vars(args).copy()
    run_args["init_report"] = init_report
    run_args["evidence_stats"] = evidence_stats
    run_args["importance_stats"] = importance_stats
    run_args["effective_evidence_pos_weight"] = evidence_pos_weight
    run_args["effective_importance_pos_weight"] = importance_pos_weight
    (output_dir / "train_args.json").write_text(json.dumps(run_args, ensure_ascii=False, indent=2), encoding="utf-8")


def concat_datasets(
    paths: list[str],
    tokenizer: Any,
    feature_config: StructureFeatureConfig,
    max_records: int | None,
    target_records_per_dataset: int | None = None,
) -> ConcatDataset:
    datasets: list[Dataset] = []
    for path in paths:
        if not path:
            continue
        dataset = StructuredSentenceDataset(path, tokenizer, feature_config, max_records=max_records)
        if target_records_per_dataset is not None and target_records_per_dataset > 0:
            dataset = SizedDataset(dataset, target_records_per_dataset)
        datasets.append(dataset)
    if not datasets:
        raise ValueError("No datasets were provided.")
    return ConcatDataset(datasets)


class SizedDataset(Dataset):
    """Caps or repeats one source so multi-task batches are not dominated by a large corpus."""

    def __init__(self, dataset: StructuredSentenceDataset, target_size: int) -> None:
        self.dataset = dataset
        self.target_size = max(1, target_size)

    def __len__(self) -> int:
        return self.target_size

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.dataset[idx % len(self.dataset)]

    def evidence_counts(self) -> tuple[int, int]:
        return repeated_label_counts(self.dataset, self.target_size, "evidence_label")

    def importance_counts(self) -> tuple[int, int]:
        return repeated_label_counts(self.dataset, self.target_size, "importance_label")


def evidence_counts(dataset: Dataset) -> dict[str, int]:
    positive = 0
    negative = 0
    if isinstance(dataset, ConcatDataset):
        children = dataset.datasets
    else:
        children = [dataset]
    for child in children:
        if hasattr(child, "evidence_counts"):
            child_positive, child_negative = child.evidence_counts()
            positive += int(child_positive)
            negative += int(child_negative)
    return {"positive": positive, "negative": negative}


def importance_counts(dataset: Dataset) -> dict[str, int]:
    positive = 0
    negative = 0
    if isinstance(dataset, ConcatDataset):
        children = dataset.datasets
    else:
        children = [dataset]
    for child in children:
        if hasattr(child, "importance_counts"):
            child_positive, child_negative = child.importance_counts()
            positive += int(child_positive)
            negative += int(child_negative)
    return {"positive": positive, "negative": negative}


def repeated_label_counts(dataset: StructuredSentenceDataset, target_size: int, label_key: str) -> tuple[int, int]:
    positive = 0
    negative = 0
    if len(dataset) <= 0:
        return positive, negative
    for idx in range(target_size):
        record = dataset.records[idx % len(dataset)]
        for sentence in record.get("sentences", []):
            if not isinstance(sentence, dict) or sentence.get(label_key) is None:
                continue
            if float(sentence.get(label_key) or 0.0) >= 0.5:
                positive += 1
            else:
                negative += 1
    return positive, negative


def auto_evidence_pos_weight(counts: dict[str, int], max_weight: float) -> float:
    positive = counts.get("positive", 0)
    negative = counts.get("negative", 0)
    if positive <= 0 or negative <= 0:
        return 1.0
    return max(1.0, min(float(negative) / float(positive), max_weight))


def save_checkpoint(model: StructuredPaperModel, tokenizer: Any, feature_config: StructureFeatureConfig, output_dir: Path) -> None:
    model.save(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "feature_config.json").write_text(json.dumps(feature_config.__dict__, indent=2), encoding="utf-8")


def train_one_epoch(
    model: StructuredPaperModel,
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
    progress = tqdm(loader, desc="structure-train")
    for batch in progress:
        batch = move_structure_batch_to_device(batch, device)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                section_token_ids=batch["section_token_ids"],
                cls_positions=batch["cls_positions"],
                sentence_mask=batch["sentence_mask"],
                role_labels=batch["role_labels"],
                role_candidate_masks=batch["role_candidate_masks"],
                evidence_labels=batch["evidence_labels"],
                importance_labels=batch["importance_labels"],
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
def evaluate(model: StructuredPaperModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    steps = 0
    role_tp = role_fp = role_fn = 0
    role_candidate_total = role_candidate_hit = 0
    role_candidate_prob_sum = 0.0
    evidence_tp = evidence_fp = evidence_fn = 0
    evidence_total = evidence_pred_positive = evidence_gold_positive = 0
    evidence_pos_prob_sum = evidence_pos_n = 0
    evidence_neg_prob_sum = evidence_neg_n = 0
    evidence_probs_for_thresholds: list[float] = []
    evidence_gold_for_thresholds: list[int] = []
    importance_abs = 0.0
    importance_n = 0
    importance_tp = importance_fp = importance_fn = 0
    importance_total = importance_pred_positive = importance_gold_positive = 0
    importance_pos_prob_sum = importance_pos_n = 0
    importance_neg_prob_sum = importance_neg_n = 0
    importance_probs_for_thresholds: list[float] = []
    importance_gold_for_thresholds: list[int] = []
    for batch in loader:
        batch = move_structure_batch_to_device(batch, device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            section_token_ids=batch["section_token_ids"],
            cls_positions=batch["cls_positions"],
            sentence_mask=batch["sentence_mask"],
            role_labels=batch["role_labels"],
            role_candidate_masks=batch["role_candidate_masks"],
            evidence_labels=batch["evidence_labels"],
            importance_labels=batch["importance_labels"],
        )
        total_loss += float(outputs["loss"].item())
        steps += 1
        role_pred = outputs["role_logits"].argmax(dim=-1)
        role_gold = batch["role_labels"]
        role_valid = (role_gold >= 0) & batch["sentence_mask"]
        role_tp += int(((role_pred == role_gold) & role_valid).sum().item())
        role_fp += int(((role_pred != role_gold) & role_valid).sum().item())
        role_fn += int(((role_pred != role_gold) & role_valid).sum().item())
        role_candidate_valid = (batch["role_candidate_masks"].sum(dim=-1) > 0) & batch["sentence_mask"]
        if role_candidate_valid.any():
            role_candidate_for_pred = batch["role_candidate_masks"].gather(-1, role_pred.unsqueeze(-1)).squeeze(-1)
            role_candidate_hit += int(((role_candidate_for_pred > 0) & role_candidate_valid).sum().item())
            role_candidate_total += int(role_candidate_valid.sum().item())
            role_probs = torch.softmax(outputs["role_logits"], dim=-1)
            role_candidate_mass = (role_probs * batch["role_candidate_masks"]).sum(dim=-1)
            role_candidate_prob_sum += float(role_candidate_mass[role_candidate_valid].sum().item())

        evidence_valid = (batch["evidence_labels"] >= 0) & batch["sentence_mask"]
        evidence_pred = outputs["evidence_probs"] >= 0.5
        evidence_gold = batch["evidence_labels"] >= 0.5
        evidence_tp += int((evidence_pred & evidence_gold & evidence_valid).sum().item())
        evidence_fp += int((evidence_pred & ~evidence_gold & evidence_valid).sum().item())
        evidence_fn += int((~evidence_pred & evidence_gold & evidence_valid).sum().item())
        evidence_total += int(evidence_valid.sum().item())
        evidence_pred_positive += int((evidence_pred & evidence_valid).sum().item())
        evidence_gold_positive += int((evidence_gold & evidence_valid).sum().item())
        if evidence_valid.any():
            valid_probs = outputs["evidence_probs"][evidence_valid].detach().cpu()
            valid_gold = evidence_gold[evidence_valid].detach().cpu()
            if valid_gold.any():
                evidence_pos_prob_sum += float(valid_probs[valid_gold].sum().item())
                evidence_pos_n += int(valid_gold.sum().item())
            neg_gold = ~valid_gold
            if neg_gold.any():
                evidence_neg_prob_sum += float(valid_probs[neg_gold].sum().item())
                evidence_neg_n += int(neg_gold.sum().item())
            evidence_probs_for_thresholds.extend(float(value) for value in valid_probs.tolist())
            evidence_gold_for_thresholds.extend(int(value) for value in valid_gold.tolist())

        importance_valid = (batch["importance_labels"] >= 0) & batch["sentence_mask"]
        if importance_valid.any():
            importance_abs += float((outputs["importance_scores"] - batch["importance_labels"]).abs()[importance_valid].sum().item())
            importance_n += int(importance_valid.sum().item())
            importance_pred = outputs["importance_scores"] >= 0.5
            importance_gold = batch["importance_labels"] >= 0.5
            importance_tp += int((importance_pred & importance_gold & importance_valid).sum().item())
            importance_fp += int((importance_pred & ~importance_gold & importance_valid).sum().item())
            importance_fn += int((~importance_pred & importance_gold & importance_valid).sum().item())
            importance_total += int(importance_valid.sum().item())
            importance_pred_positive += int((importance_pred & importance_valid).sum().item())
            importance_gold_positive += int((importance_gold & importance_valid).sum().item())
            valid_probs = outputs["importance_scores"][importance_valid].detach().cpu()
            valid_gold = importance_gold[importance_valid].detach().cpu()
            if valid_gold.any():
                importance_pos_prob_sum += float(valid_probs[valid_gold].sum().item())
                importance_pos_n += int(valid_gold.sum().item())
            neg_gold = ~valid_gold
            if neg_gold.any():
                importance_neg_prob_sum += float(valid_probs[neg_gold].sum().item())
                importance_neg_n += int(neg_gold.sum().item())
            importance_probs_for_thresholds.extend(float(value) for value in valid_probs.tolist())
            importance_gold_for_thresholds.extend(int(value) for value in valid_gold.tolist())

    evidence_best = best_threshold_f1(evidence_probs_for_thresholds, evidence_gold_for_thresholds)
    importance_best = best_threshold_f1(importance_probs_for_thresholds, importance_gold_for_thresholds)
    return {
        "loss": total_loss / max(steps, 1),
        "role_f1": _f1(role_tp, role_fp, role_fn),
        "role_candidate_accuracy": role_candidate_hit / max(role_candidate_total, 1),
        "role_candidate_prob_mass": role_candidate_prob_sum / max(role_candidate_total, 1),
        "evidence_precision": _precision(evidence_tp, evidence_fp),
        "evidence_recall": _recall(evidence_tp, evidence_fn),
        "evidence_f1": _f1(evidence_tp, evidence_fp, evidence_fn),
        "evidence_best_f1": evidence_best["f1"],
        "evidence_best_threshold": evidence_best["threshold"],
        "evidence_pred_positive_rate": evidence_pred_positive / max(evidence_total, 1),
        "evidence_gold_positive_rate": evidence_gold_positive / max(evidence_total, 1),
        "evidence_pos_prob_mean": evidence_pos_prob_sum / max(evidence_pos_n, 1),
        "evidence_neg_prob_mean": evidence_neg_prob_sum / max(evidence_neg_n, 1),
        "importance_mae": importance_abs / max(importance_n, 1),
        "importance_precision": _precision(importance_tp, importance_fp),
        "importance_recall": _recall(importance_tp, importance_fn),
        "importance_f1": _f1(importance_tp, importance_fp, importance_fn),
        "importance_best_f1": importance_best["f1"],
        "importance_best_threshold": importance_best["threshold"],
        "importance_pred_positive_rate": importance_pred_positive / max(importance_total, 1),
        "importance_gold_positive_rate": importance_gold_positive / max(importance_total, 1),
        "importance_pos_prob_mean": importance_pos_prob_sum / max(importance_pos_n, 1),
        "importance_neg_prob_mean": importance_neg_prob_sum / max(importance_neg_n, 1),
    }


def _precision(tp: int, fp: int) -> float:
    return tp / max(tp + fp, 1)


def _recall(tp: int, fn: int) -> float:
    return tp / max(tp + fn, 1)


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = _precision(tp, fp)
    recall = _recall(tp, fn)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def best_threshold_f1(probs: list[float], gold: list[int]) -> dict[str, float]:
    if not probs or not gold:
        return {"threshold": 0.5, "f1": 0.0}
    best = {"threshold": 0.5, "f1": 0.0}
    for threshold in [idx / 100 for idx in range(5, 96, 5)]:
        tp = fp = fn = 0
        for prob, label in zip(probs, gold):
            pred = prob >= threshold
            actual = label >= 1
            tp += int(pred and actual)
            fp += int(pred and not actual)
            fn += int((not pred) and actual)
        f1 = _f1(tp, fp, fn)
        if f1 > best["f1"]:
            best = {"threshold": threshold, "f1": f1}
    return best


if __name__ == "__main__":
    main()
