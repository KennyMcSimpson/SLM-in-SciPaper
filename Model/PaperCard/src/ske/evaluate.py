from __future__ import annotations

from .data.text_utils import normalize_text


def exact_match_scores(predictions: list[list[str]], golds: list[list[str]], top_k: int = 10) -> dict[str, float]:
    total_tp = total_pred = total_gold = 0
    for pred, gold in zip(predictions, golds):
        pred_set = {normalize_text(item) for item in pred[:top_k] if normalize_text(item)}
        gold_set = {normalize_text(item) for item in gold if normalize_text(item)}
        total_tp += len(pred_set & gold_set)
        total_pred += len(pred_set)
        total_gold += len(gold_set)
    precision = total_tp / max(total_pred, 1)
    recall = total_tp / max(total_gold, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}

