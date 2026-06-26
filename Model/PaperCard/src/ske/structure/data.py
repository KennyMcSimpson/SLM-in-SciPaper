from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .roles import best_summary_facet_role
from .schema import ROLE_LABELS, ROLE_TO_ID, SECTION_DEFAULT_ROLES, SECTION_TO_ID


IGNORE_INDEX = -100


@dataclass
class StructureFeatureConfig:
    max_seq_length: int = 512
    max_sentences: int = 48
    max_candidate_sentences: int = 180
    lead_sentences: int = 5
    neighbor_sentences: int = 1


class StructuredSentenceDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: Any,
        config: StructureFeatureConfig,
        max_records: int | None = None,
    ) -> None:
        self.path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.config = config
        records = [record for record in read_jsonl(self.path) if has_sentences(record)]
        self.skipped_empty_records = count_jsonl_records(self.path) - len(records)
        self.records = records[:max_records] if max_records is not None else records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return build_structure_features(self.records[idx], self.tokenizer, self.config)

    def evidence_counts(self) -> tuple[int, int]:
        positive = 0
        negative = 0
        for record in self.records:
            for sentence in record.get("sentences", []):
                if not isinstance(sentence, dict) or sentence.get("evidence_label") is None:
                    continue
                if float(sentence.get("evidence_label") or 0.0) >= 0.5:
                    positive += 1
                else:
                    negative += 1
        return positive, negative

    def importance_counts(self) -> tuple[int, int]:
        positive = 0
        negative = 0
        for record in self.records:
            for sentence in record.get("sentences", []):
                if not isinstance(sentence, dict) or sentence.get("importance_label") is None:
                    continue
                if float(sentence.get("importance_label") or 0.0) >= 0.5:
                    positive += 1
                else:
                    negative += 1
        return positive, negative


def build_structure_features(record: dict[str, Any], tokenizer: Any, config: StructureFeatureConfig) -> dict[str, Any]:
    sentences = [item for item in record.get("sentences", []) if str(item.get("text", "")).strip()]
    if not sentences:
        raise ValueError("Structured record has no sentences.")
    selected = select_sentences(sentences[: config.max_candidate_sentences], config)

    input_ids: list[int] = []
    token_type_ids: list[int] = []
    attention_mask: list[int] = []
    section_token_ids: list[int] = []
    cls_positions: list[int] = []
    sentence_section_ids: list[int] = []
    role_labels: list[int] = []
    role_candidate_masks: list[list[float]] = []
    evidence_labels: list[float] = []
    importance_labels: list[float] = []
    kept_sentences: list[str] = []
    original_sentence_indices: list[int] = []
    record_source = str(record.get("source") or "")
    summaries = record.get("summaries") or {}

    for local_idx, sentence in enumerate(selected):
        text = str(sentence.get("text", "")).strip()
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=config.max_seq_length - 2,
        )
        sent_ids = encoded["input_ids"]
        if not sent_ids:
            continue
        if len(input_ids) + len(sent_ids) + 2 > config.max_seq_length and cls_positions:
            continue
        section_id = SECTION_TO_ID.get(str(sentence.get("section") or "intro"), 0)
        role_label, role_candidate_mask = role_supervision(sentence, record_source, summaries)
        segment_id = local_idx % 2
        cls_positions.append(len(input_ids))
        input_ids.append(tokenizer.cls_token_id)
        attention_mask.append(1)
        token_type_ids.append(segment_id)
        section_token_ids.append(section_id)
        input_ids.extend(sent_ids)
        attention_mask.extend([1] * len(sent_ids))
        token_type_ids.extend([segment_id] * len(sent_ids))
        section_token_ids.extend([section_id] * len(sent_ids))
        input_ids.append(tokenizer.sep_token_id)
        attention_mask.append(1)
        token_type_ids.append(segment_id)
        section_token_ids.append(section_id)
        sentence_section_ids.append(section_id)
        role_labels.append(role_label)
        role_candidate_masks.append(role_candidate_mask)
        evidence_labels.append(float(sentence["evidence_label"]) if sentence.get("evidence_label") is not None else -1.0)
        importance_labels.append(float(sentence["importance_label"]) if sentence.get("importance_label") is not None else -1.0)
        kept_sentences.append(text)
        original_sentence_indices.append(int(sentence.get("sentence_index", len(original_sentence_indices))))

    if not input_ids:
        raise ValueError("No valid sentences after tokenization.")
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "section_token_ids": section_token_ids,
        "cls_positions": cls_positions,
        "sentence_section_ids": sentence_section_ids,
        "role_labels": role_labels,
        "role_candidate_masks": role_candidate_masks,
        "evidence_labels": evidence_labels,
        "importance_labels": importance_labels,
        "sentences": kept_sentences,
        "sentence_indices": original_sentence_indices,
        "doc_id": str(record.get("doc_id") or ""),
        "title": str(record.get("title") or ""),
        "source": str(record.get("source") or ""),
    }


def select_sentences(sentences: list[dict[str, Any]], config: StructureFeatureConfig) -> list[dict[str, Any]]:
    if len(sentences) <= config.max_sentences:
        return sentences
    priority: list[int] = []
    positives = [
        idx
        for idx, sentence in enumerate(sentences)
        if sentence.get("evidence_label") or (sentence.get("importance_label") is not None and float(sentence.get("importance_label") or 0.0) >= 0.75)
    ]
    priority.extend(positives)
    for idx in positives:
        for delta in range(1, config.neighbor_sentences + 1):
            priority.extend([idx - delta, idx + delta])
    priority.extend(range(min(config.lead_sentences, len(sentences))))
    stride = max(len(sentences) // max(config.max_sentences, 1), 1)
    priority.extend(range(0, len(sentences), stride))

    selected: list[int] = []
    seen: set[int] = set()
    for idx in priority:
        if idx < 0 or idx >= len(sentences) or idx in seen:
            continue
        seen.add(idx)
        selected.append(idx)
        if len(selected) >= config.max_sentences:
            break
    return [sentences[idx] for idx in sorted(selected)]


def collate_structure_features(features: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    batch_size = len(features)
    max_len = max(len(feature["input_ids"]) for feature in features)
    max_sents = max(len(feature["cls_positions"]) for feature in features)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    token_type_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    section_token_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    cls_positions = torch.zeros((batch_size, max_sents), dtype=torch.long)
    sentence_mask = torch.zeros((batch_size, max_sents), dtype=torch.bool)
    sentence_section_ids = torch.zeros((batch_size, max_sents), dtype=torch.long)
    role_labels = torch.full((batch_size, max_sents), IGNORE_INDEX, dtype=torch.long)
    role_candidate_masks = torch.zeros((batch_size, max_sents, len(ROLE_LABELS)), dtype=torch.float)
    evidence_labels = torch.full((batch_size, max_sents), -1.0, dtype=torch.float)
    importance_labels = torch.full((batch_size, max_sents), -1.0, dtype=torch.float)

    for batch_idx, feature in enumerate(features):
        length = len(feature["input_ids"])
        input_ids[batch_idx, :length] = torch.tensor(feature["input_ids"], dtype=torch.long)
        attention_mask[batch_idx, :length] = torch.tensor(feature["attention_mask"], dtype=torch.long)
        token_type_ids[batch_idx, :length] = torch.tensor(feature["token_type_ids"], dtype=torch.long)
        section_token_ids[batch_idx, :length] = torch.tensor(feature["section_token_ids"], dtype=torch.long)
        sent_count = len(feature["cls_positions"])
        cls_positions[batch_idx, :sent_count] = torch.tensor(feature["cls_positions"], dtype=torch.long)
        sentence_mask[batch_idx, :sent_count] = True
        sentence_section_ids[batch_idx, :sent_count] = torch.tensor(feature["sentence_section_ids"], dtype=torch.long)
        role_labels[batch_idx, :sent_count] = torch.tensor(feature["role_labels"], dtype=torch.long)
        role_candidate_masks[batch_idx, :sent_count, :] = torch.tensor(feature["role_candidate_masks"], dtype=torch.float)
        evidence_labels[batch_idx, :sent_count] = torch.tensor(feature["evidence_labels"], dtype=torch.float)
        importance_labels[batch_idx, :sent_count] = torch.tensor(feature["importance_labels"], dtype=torch.float)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "section_token_ids": section_token_ids,
        "cls_positions": cls_positions,
        "sentence_mask": sentence_mask,
        "sentence_section_ids": sentence_section_ids,
        "role_labels": role_labels,
        "role_candidate_masks": role_candidate_masks,
        "evidence_labels": evidence_labels,
        "importance_labels": importance_labels,
        "sentences": [feature["sentences"] for feature in features],
        "sentence_indices": [feature["sentence_indices"] for feature in features],
        "doc_ids": [feature["doc_id"] for feature in features],
        "titles": [feature["title"] for feature in features],
        "sources": [feature["source"] for feature in features],
    }


def role_to_label(role: Any) -> int:
    if role is None:
        return IGNORE_INDEX
    return ROLE_TO_ID.get(str(role), IGNORE_INDEX)


def role_supervision(sentence: dict[str, Any], record_source: str, summaries: dict[str, str]) -> tuple[int, list[float]]:
    source = str(sentence.get("source") or record_source or "")
    section = str(sentence.get("section") or "intro")
    if source == "qasper":
        return IGNORE_INDEX, section_candidate_mask(section)
    if source == "aclsum":
        facet_role, confidence = best_summary_facet_role(str(sentence.get("text") or ""), summaries)
        if facet_role is not None and confidence >= 0.08:
            return role_to_label(facet_role), zero_role_candidate_mask()
        return IGNORE_INDEX, section_candidate_mask(section)
    return role_to_label(sentence.get("role")), zero_role_candidate_mask()


def section_candidate_mask(section: str) -> list[float]:
    mask = [0.0] * len(ROLE_LABELS)
    for role in SECTION_DEFAULT_ROLES.get(section, ["background"]):
        role_id = ROLE_TO_ID.get(role)
        if role_id is not None:
            mask[role_id] = 1.0
    return mask


def zero_role_candidate_mask() -> list[float]:
    return [0.0] * len(ROLE_LABELS)


def read_jsonl(path: Path, max_records: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if max_records is not None and len(records) >= max_records:
                break
    return records


def has_sentences(record: dict[str, Any]) -> bool:
    return any(str(sentence.get("text", "")).strip() for sentence in record.get("sentences", []) if isinstance(sentence, dict))


def count_jsonl_records(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
