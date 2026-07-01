from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from ske.data.bow_vocab import BowVocabulary
from ske.data.dataset import BIO_LABELS
from ske.data.text_utils import normalize_text, split_sentences, token_jaccard
from ske.modeling import ScientificKeyphraseExtractor


@dataclass
class TokenMeta:
    sentence_idx: int
    start_char: int
    end_char: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract scientific keyphrases from a txt file.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_txt", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--bow_csv", default=None)
    parser.add_argument("--top_k_sentences", type=int, default=12)
    parser.add_argument("--top_k_keyphrases", type=int, default=12)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_sentences", type=int, default=64)
    parser.add_argument("--coverage_lambda", type=float, default=0.75)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)
    checkpoint = Path(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    model = ScientificKeyphraseExtractor.load(checkpoint, map_location=device).to(device)
    model.eval()
    bow_vocab = BowVocabulary.load(args.bow_csv)
    text = Path(args.input_txt).read_text(encoding="utf-8", errors="ignore")
    sentences = split_sentences(text)[: args.max_sentences]
    windows = build_inference_windows(sentences, tokenizer, args.max_seq_length)
    sentence_scores: dict[int, float] = {}
    raw_candidates: list[dict[str, Any]] = []
    for features in windows:
        batch = {key: value.to(device) for key, value in features["tensors"].items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            cls_positions=batch["cls_positions"],
            sentence_mask=batch["sentence_mask"],
        )
        sentence_probs = outputs["sentence_probs"][0].detach().cpu().tolist()
        for local_idx, global_idx in enumerate(features["sentence_indices"]):
            sentence_scores[global_idx] = max(sentence_scores.get(global_idx, 0.0), float(sentence_probs[local_idx]))
        boundary_probs = outputs["boundary_probs"][0].detach().cpu()
        boundary_preds = outputs["boundary_logits"][0].argmax(dim=-1).detach().cpu().tolist()
        window_candidates = extract_candidates_from_bio(
            sentences=features["sentences"],
            token_meta=features["token_meta"],
            boundary_probs=boundary_probs,
            boundary_preds=boundary_preds,
            sentence_probs=sentence_probs,
            bow_vocab=bow_vocab,
        )
        for item in window_candidates:
            item["sentence_index"] = features["sentence_indices"][item["sentence_index"]]
            raw_candidates.append(item)
    ranked = coverage_rerank(raw_candidates, args.top_k_keyphrases, args.coverage_lambda)
    key_sentences = [
        {
            "rank": rank + 1,
            "sentence_index": idx,
            "score": float(score),
            "sentence": sentences[idx],
        }
        for rank, (idx, score) in enumerate(sorted(sentence_scores.items(), key=lambda item: item[1], reverse=True)[: args.top_k_sentences])
    ]
    result = {
        "input": str(Path(args.input_txt).resolve()),
        "key_sentences": key_sentences,
        "keyphrases": [
            {
                "rank": idx + 1,
                "keyphrase": item["canonical"],
                "surface": item["surface"],
                "score": item["score"],
                "s_boundary": item.get("s_boundary", item.get("boundary_score", 0.0)),
                "s_selector": item.get("s_selector", item.get("selector_score", 0.0)),
                "s_bow": item.get("s_bow", item.get("bow_confidence", 0.0)),
                "s_candidate": item.get("s_candidate", item.get("score", 0.0)),
                "s_coverage": item.get("s_coverage", 0.0),
                "s_rerank": item.get("s_rerank", item.get("score", 0.0)),
                "sentence_index": item["sentence_index"],
                "selector_score": item["selector_score"],
                "boundary_score": item["boundary_score"],
                "bow_confidence": item["bow_confidence"],
                "evidence_sentence": sentences[item["sentence_index"]],
            }
            for idx, item in enumerate(ranked)
        ],
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(rendered, encoding="utf-8")
    print(rendered)


def build_inference_windows(sentences: list[str], tokenizer: Any, max_seq_length: int) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    input_ids: list[int] = []
    attention_mask: list[int] = []
    token_type_ids: list[int] = []
    cls_positions: list[int] = []
    token_meta: list[TokenMeta | None] = []
    kept_sentences: list[str] = []
    sentence_indices: list[int] = []

    def flush_window() -> None:
        nonlocal input_ids, attention_mask, token_type_ids, cls_positions, token_meta, kept_sentences, sentence_indices
        if not input_ids:
            return
        windows.append(
            {
                "tensors": {
                    "input_ids": torch.tensor([input_ids], dtype=torch.long),
                    "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
                    "token_type_ids": torch.tensor([token_type_ids], dtype=torch.long),
                    "cls_positions": torch.tensor([cls_positions], dtype=torch.long),
                    "sentence_mask": torch.ones((1, len(cls_positions)), dtype=torch.bool),
                },
                "token_meta": token_meta,
                "sentences": kept_sentences,
                "sentence_indices": sentence_indices,
            }
        )
        input_ids = []
        attention_mask = []
        token_type_ids = []
        cls_positions = []
        token_meta = []
        kept_sentences = []
        sentence_indices = []

    for global_idx, sentence in enumerate(sentences):
        encoded = tokenizer(sentence, add_special_tokens=False, return_offsets_mapping=True, truncation=True, max_length=max_seq_length - 2)
        sent_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        if not sent_ids:
            continue
        if input_ids and len(input_ids) + len(sent_ids) + 2 > max_seq_length:
            flush_window()
        local_idx = len(kept_sentences)
        segment_id = local_idx % 2
        cls_positions.append(len(input_ids))
        input_ids.append(tokenizer.cls_token_id)
        attention_mask.append(1)
        token_type_ids.append(segment_id)
        token_meta.append(None)
        input_ids.extend(sent_ids)
        attention_mask.extend([1] * len(sent_ids))
        token_type_ids.extend([segment_id] * len(sent_ids))
        token_meta.extend(TokenMeta(local_idx, start, end) for start, end in offsets)
        input_ids.append(tokenizer.sep_token_id)
        attention_mask.append(1)
        token_type_ids.append(segment_id)
        token_meta.append(None)
        kept_sentences.append(sentence)
        sentence_indices.append(global_idx)
    flush_window()
    return windows


def extract_candidates_from_bio(
    sentences: list[str],
    token_meta: list[TokenMeta | None],
    boundary_probs: torch.Tensor,
    boundary_preds: list[int],
    sentence_probs: list[float],
    bow_vocab: BowVocabulary | None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    idx = 0
    while idx < len(boundary_preds):
        if boundary_preds[idx] != BIO_LABELS["B"] or token_meta[idx] is None:
            idx += 1
            continue
        start_idx = idx
        end_idx = idx + 1
        while end_idx < len(boundary_preds) and boundary_preds[end_idx] == BIO_LABELS["I"] and token_meta[end_idx] is not None:
            if token_meta[end_idx].sentence_idx != token_meta[start_idx].sentence_idx:
                break
            end_idx += 1
        start_meta = token_meta[start_idx]
        end_meta = token_meta[end_idx - 1]
        assert start_meta is not None and end_meta is not None
        surface = sentences[start_meta.sentence_idx][start_meta.start_char : end_meta.end_char].strip(" ,.;:()[]{}")
        if len(normalize_text(surface)) >= 3:
            boundary_score = float(boundary_probs[start_idx:end_idx, 1:].max(dim=-1).values.mean().item())
            canonical, bow_confidence = bow_vocab.canonicalize(surface) if bow_vocab else (surface, 0.0)
            selector_score = float(sentence_probs[start_meta.sentence_idx])
            candidate_score = 0.65 * boundary_score + 0.25 * selector_score + 0.10 * bow_confidence
            candidates.append(
                {
                    "surface": surface,
                    "canonical": canonical,
                    "s_boundary": boundary_score,
                    "s_selector": selector_score,
                    "s_bow": bow_confidence,
                    "s_candidate": candidate_score,
                    "score": candidate_score,
                    "boundary_score": boundary_score,
                    "selector_score": selector_score,
                    "bow_confidence": bow_confidence,
                    "sentence_index": start_meta.sentence_idx,
                }
            )
        idx = max(end_idx, idx + 1)
    return deduplicate_by_canonical(candidates)


def deduplicate_by_canonical(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for item in candidates:
        key = normalize_text(item["canonical"])
        if key not in grouped or item["s_candidate"] > grouped[key]["s_candidate"]:
            grouped[key] = item
    return sorted(grouped.values(), key=lambda item: item["s_candidate"], reverse=True)


def coverage_rerank(candidates: list[dict[str, Any]], top_k: int, lambda_weight: float) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    pool = list(candidates)
    while pool and len(selected) < top_k:
        best_idx = 0
        best_score = -1e9
        for idx, item in enumerate(pool):
            redundancy = max((token_jaccard(item["canonical"], chosen["canonical"]) for chosen in selected), default=0.0)
            coverage = 1.0 - redundancy
            score = lambda_weight * item["s_candidate"] + (1.0 - lambda_weight) * coverage
            if score > best_score:
                best_idx = idx
                best_score = score
        chosen = pool.pop(best_idx)
        chosen["s_coverage"] = 1.0 - max((token_jaccard(chosen["canonical"], item["canonical"]) for item in selected), default=0.0)
        chosen["s_rerank"] = lambda_weight * chosen["s_candidate"] + (1.0 - lambda_weight) * chosen["s_coverage"]
        chosen["score"] = chosen["s_rerank"]
        selected.append(chosen)
    return selected


if __name__ == "__main__":
    main()
