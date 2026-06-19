from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .bow_vocab import BowVocabulary
from .text_utils import find_normalized_phrase_spans, normalize_text, split_sentences

BIO_LABELS = {"O": 0, "B": 1, "I": 2}
ID_TO_BIO = {value: key for key, value in BIO_LABELS.items()}


@dataclass
class FeatureConfig:
    max_seq_length: int = 512
    max_sentences: int = 48
    max_candidate_sentences: int = 160
    use_evidence_packing: bool = True
    lead_sentences: int = 4
    neighbor_sentences: int = 1
    use_bow_selector: bool = True
    bow_min_confidence: float = 0.7


class KeyphraseDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: Any,
        config: FeatureConfig,
        bow_vocab: BowVocabulary | None = None,
        max_records: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.bow_vocab = bow_vocab
        self.records = _read_records(Path(jsonl_path), max_records=max_records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return build_features(self.records[idx], self.tokenizer, self.config, self.bow_vocab)


def build_features(
    record: dict[str, Any],
    tokenizer: Any,
    config: FeatureConfig,
    bow_vocab: BowVocabulary | None = None,
) -> dict[str, Any]:
    text = "\n".join(
        str(record.get(key) or "")
        for key in ("title", "abstract", "full_text")
        if str(record.get(key) or "").strip()
    )
    sentences = split_sentences(text)[: config.max_candidate_sentences]
    keyphrases = [phrase for phrase in record.get("keyphrases", []) if isinstance(phrase, str)]
    normalized_keyphrases = sorted({normalize_text(phrase) for phrase in keyphrases if normalize_text(phrase)}, key=len, reverse=True)

    encoded_sentences: list[dict[str, Any]] = []
    for sentence_idx, sentence in enumerate(sentences):
        encoded = tokenizer(
            sentence,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=True,
            max_length=config.max_seq_length - 2,
        )
        sentence_token_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        if not sentence_token_ids:
            continue
        labels = _bio_labels_for_sentence(sentence, offsets, normalized_keyphrases)
        has_gold = any(label != BIO_LABELS["O"] for label in labels)
        has_bow = bool(bow_vocab and config.use_bow_selector and bow_vocab.sentence_hits(sentence, config.bow_min_confidence))
        encoded_sentences.append(
            {
                "sentence_idx": sentence_idx,
                "sentence": sentence,
                "input_ids": sentence_token_ids,
                "offsets": offsets,
                "labels": labels,
                "has_gold": has_gold,
                "has_bow": has_bow,
            }
        )

    selected = select_sentences_for_budget(encoded_sentences, config)

    input_ids: list[int] = []
    token_type_ids: list[int] = []
    attention_mask: list[int] = []
    bio_labels: list[int] = []
    cls_positions: list[int] = []
    sentence_labels: list[float] = []
    kept_sentences: list[str] = []
    original_sentence_indices: list[int] = []

    for local_idx, item in enumerate(selected):
        sentence_token_ids = item["input_ids"]
        labels = item["labels"]
        if len(input_ids) + len(sentence_token_ids) + 2 > config.max_seq_length:
            continue
        segment_id = local_idx % 2
        cls_positions.append(len(input_ids))
        input_ids.append(tokenizer.cls_token_id)
        token_type_ids.append(segment_id)
        attention_mask.append(1)
        bio_labels.append(-100)
        input_ids.extend(sentence_token_ids)
        token_type_ids.extend([segment_id] * len(sentence_token_ids))
        attention_mask.extend([1] * len(sentence_token_ids))
        bio_labels.extend(labels)
        input_ids.append(tokenizer.sep_token_id)
        token_type_ids.append(segment_id)
        attention_mask.append(1)
        bio_labels.append(-100)
        sentence_labels.append(1.0 if item["has_gold"] or item["has_bow"] else 0.0)
        kept_sentences.append(item["sentence"])
        original_sentence_indices.append(int(item["sentence_idx"]))

    if not input_ids:
        raise ValueError("No valid text after tokenization.")
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "bio_labels": bio_labels,
        "cls_positions": cls_positions,
        "sentence_labels": sentence_labels,
        "sentences": kept_sentences,
        "sentence_indices": original_sentence_indices,
        "doc_id": str(record.get("doc_id") or ""),
        "keyphrases": keyphrases,
    }


def collate_features(features: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    batch_size = len(features)
    max_len = max(len(feature["input_ids"]) for feature in features)
    max_sents = max(len(feature["cls_positions"]) for feature in features)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    token_type_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    bio_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    cls_positions = torch.zeros((batch_size, max_sents), dtype=torch.long)
    sentence_mask = torch.zeros((batch_size, max_sents), dtype=torch.bool)
    sentence_labels = torch.zeros((batch_size, max_sents), dtype=torch.float)

    for batch_idx, feature in enumerate(features):
        length = len(feature["input_ids"])
        input_ids[batch_idx, :length] = torch.tensor(feature["input_ids"], dtype=torch.long)
        attention_mask[batch_idx, :length] = torch.tensor(feature["attention_mask"], dtype=torch.long)
        token_type_ids[batch_idx, :length] = torch.tensor(feature["token_type_ids"], dtype=torch.long)
        bio_labels[batch_idx, :length] = torch.tensor(feature["bio_labels"], dtype=torch.long)
        sent_count = len(feature["cls_positions"])
        cls_positions[batch_idx, :sent_count] = torch.tensor(feature["cls_positions"], dtype=torch.long)
        sentence_mask[batch_idx, :sent_count] = True
        sentence_labels[batch_idx, :sent_count] = torch.tensor(feature["sentence_labels"], dtype=torch.float)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "bio_labels": bio_labels,
        "cls_positions": cls_positions,
        "sentence_mask": sentence_mask,
        "sentence_labels": sentence_labels,
        "sentences": [feature["sentences"] for feature in features],
        "sentence_indices": [feature.get("sentence_indices", []) for feature in features],
        "doc_ids": [feature["doc_id"] for feature in features],
        "keyphrases": [feature["keyphrases"] for feature in features],
    }


def select_sentences_for_budget(encoded_sentences: list[dict[str, Any]], config: FeatureConfig) -> list[dict[str, Any]]:
    if not encoded_sentences:
        return []
    if not config.use_evidence_packing:
        return _first_sentences_that_fit(encoded_sentences, config)

    total_if_all = sum(len(item["input_ids"]) + 2 for item in encoded_sentences[: config.max_sentences])
    if total_if_all <= config.max_seq_length:
        return encoded_sentences[: config.max_sentences]

    positive_indices = [idx for idx, item in enumerate(encoded_sentences) if item["has_gold"] or item["has_bow"]]
    if not positive_indices:
        return _first_sentences_that_fit(encoded_sentences, config)

    priority: list[int] = []
    priority.extend(positive_indices)
    for pos_idx in positive_indices:
        for delta in range(1, config.neighbor_sentences + 1):
            priority.extend([pos_idx - delta, pos_idx + delta])
    priority.extend(range(min(config.lead_sentences, len(encoded_sentences))))
    if len(encoded_sentences) > config.lead_sentences:
        stride = max(len(encoded_sentences) // max(config.max_sentences, 1), 1)
        priority.extend(range(config.lead_sentences, len(encoded_sentences), stride))

    selected_indices: list[int] = []
    selected_set: set[int] = set()
    used_tokens = 0
    for idx in priority:
        if idx < 0 or idx >= len(encoded_sentences) or idx in selected_set:
            continue
        item = encoded_sentences[idx]
        token_cost = len(item["input_ids"]) + 2
        if used_tokens + token_cost > config.max_seq_length and selected_indices:
            continue
        selected_indices.append(idx)
        selected_set.add(idx)
        used_tokens += token_cost
        if len(selected_indices) >= config.max_sentences:
            break
    return [encoded_sentences[idx] for idx in sorted(selected_indices)]


def _first_sentences_that_fit(encoded_sentences: list[dict[str, Any]], config: FeatureConfig) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_tokens = 0
    for item in encoded_sentences[: config.max_candidate_sentences]:
        token_cost = len(item["input_ids"]) + 2
        if used_tokens + token_cost > config.max_seq_length and selected:
            break
        selected.append(item)
        used_tokens += token_cost
        if len(selected) >= config.max_sentences:
            break
    return selected


def _bio_labels_for_sentence(sentence: str, offsets: list[tuple[int, int]], normalized_keyphrases: list[str]) -> list[int]:
    labels = [BIO_LABELS["O"]] * len(offsets)
    for phrase in normalized_keyphrases:
        for start_char, end_char in find_normalized_phrase_spans(sentence, phrase):
            token_indices = [
                idx
                for idx, (start, end) in enumerate(offsets)
                if end > start_char and start < end_char and end > start
            ]
            if not token_indices:
                continue
            first = token_indices[0]
            if labels[first] != BIO_LABELS["O"]:
                continue
            labels[first] = BIO_LABELS["B"]
            for idx in token_indices[1:]:
                labels[idx] = BIO_LABELS["I"]
    return labels


def _read_records(path: Path, max_records: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
                if max_records is not None and len(records) >= max_records:
                    break
    return records
